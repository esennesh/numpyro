# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Per-primitive change-propagation rules.

A rule has the signature ``rule(ctx, eqn, in_deltas) -> list[Δ]`` and must be a local
identity: whatever Δ-values it returns, materializing them must equal evaluating the
equation on the materialized inputs. The default rule materializes every input and
rebinds the primitive, so unknown primitives are always handled soundly (as ``Dense``).
"""

import jax
from jax import lax
from jax.extend.core import Literal, primitives as P
import jax.numpy as jnp
import jax.scipy.special

from numpyro.contrib.incremental.delta import (
    SAME,
    Dense,
    Same,
    Sliced,
    gather_rect,
    sliced_shape,
)

RULES = {}


def _is_same(d):
    return isinstance(d, Same)


def bind(eqn, *args):
    """Re-bind ``eqn``'s primitive on new arguments; returns a list of outputs."""
    bind_params = eqn.primitive.get_bind_params(eqn.params)
    with eqn.ctx.manager:
        out = eqn.primitive.bind(*args, **bind_params)
    return list(out) if eqn.primitive.multiple_results else [out]


def materialize(ctx, atom, delta):
    """The full new value of ``atom`` given its Δ-value."""
    if isinstance(delta, Dense):
        return delta.value
    if isinstance(delta, Same):
        return ctx.full(atom)
    return ctx.materialize_sliced(atom, delta)


def default_rule(ctx, eqn, in_deltas):
    args = [materialize(ctx, a, d) for a, d in zip(eqn.invars, in_deltas)]
    outs = bind(eqn, *args)
    ctx.refresh_body(eqn, args)
    return [Dense(o) for o in outs]


# ----------------------------------------------------------------- elementwise

_ELEMENTWISE_NAMES = """
abs acos acosh add and asin asinh atan atan2 atanh bessel_i0e bessel_i1e cbrt ceil
clamp clz complex conj convert_element_type copy cos cosh digamma div eq erf erf_inv
erfc exp exp2 expm1 floor ge gt igamma igamma_grad_a igammac imag integer_pow is_finite
le lgamma log log1p logistic lt max min mul ne neg nextafter not or polygamma
population_count pow real reduce_precision regularized_incomplete_beta rem round rsqrt
select_n shift_left shift_right_arithmetic shift_right_logical sign sin sinh sqrt square
stop_gradient sub tan tanh xor zeta
""".split()

ELEMENTWISE = {
    getattr(P, name + "_p") for name in _ELEMENTWISE_NAMES if hasattr(P, name + "_p")
}


def elementwise_rule(ctx, eqn, in_deltas):
    if any(isinstance(d, Dense) for d in in_deltas):
        return default_rule(ctx, eqn, in_deltas)
    sliced = [d for d in in_deltas if isinstance(d, Sliced)]
    ref = sliced[0]
    out_ndim = len(eqn.outvars[0].aval.shape)
    for d in sliced[1:]:
        if not d.same_block(ref):
            return default_rule(ctx, eqn, in_deltas)
    args = []
    for atom, d in zip(eqn.invars, in_deltas):
        if isinstance(d, Sliced):
            if len(atom.aval.shape) != out_ndim:
                return default_rule(ctx, eqn, in_deltas)
            args.append(d.value)
        elif len(atom.aval.shape) == 0:
            args.append(ctx.full(atom))
        elif len(atom.aval.shape) != out_ndim:
            return default_rule(ctx, eqn, in_deltas)
        else:
            args.append(ctx.slice(atom, ref.axes, ref.idxs))
    outs = bind(eqn, *args)
    return [Sliced(ref.axes, ref.idxs, o, ref.token) for o in outs]


for _p in ELEMENTWISE:
    RULES[_p] = elementwise_rule


# ----------------------------------------------------------------- shape ops


def broadcast_in_dim_rule(ctx, eqn, in_deltas):
    d = in_deltas[0]
    if not isinstance(d, Sliced) or len(eqn.invars) != 1:
        return default_rule(ctx, eqn, in_deltas)
    in_shape = eqn.invars[0].aval.shape
    shape = eqn.params["shape"]
    bd = eqn.params["broadcast_dimensions"]
    axes, idxs, dropped = [], [], []
    for a, i in zip(d.axes, d.idxs):
        o = bd[a]
        if in_shape[a] == 1 and shape[o] != 1:
            dropped.append(a)
        else:
            axes.append(o)
            idxs.append(i)
    ks = [int(i.shape[0]) for i in idxs]
    out = lax.broadcast_in_dim(d.value, sliced_shape(shape, axes, ks), bd)
    token = d.token if not dropped else (d.token, "bcast", tuple(dropped))
    return [Sliced(tuple(axes), tuple(idxs), out, token)]


RULES[P.broadcast_in_dim_p] = broadcast_in_dim_rule


def reshape_rule(ctx, eqn, in_deltas):
    d = in_deltas[0]
    if (
        not isinstance(d, Sliced)
        or len(eqn.invars) != 1
        or eqn.params.get("dimensions") is not None
    ):
        return default_rule(ctx, eqn, in_deltas)
    in_shape = tuple(eqn.invars[0].aval.shape)
    new_sizes = tuple(eqn.params["new_sizes"])
    hi = max(d.axes) + 1
    if in_shape[:hi] == new_sizes[:hi]:
        axes = d.axes
    else:
        lo = min(d.axes)
        n_tail = len(in_shape) - lo
        if n_tail <= len(new_sizes) and in_shape[lo:] == new_sizes[-n_tail:]:
            shift = len(new_sizes) - len(in_shape)
            axes = tuple(a + shift for a in d.axes)
        else:
            return default_rule(ctx, eqn, in_deltas)
    out = jnp.reshape(d.value, sliced_shape(new_sizes, axes, d.ks))
    return [Sliced(tuple(axes), d.idxs, out, d.token)]


RULES[P.reshape_p] = reshape_rule


def squeeze_rule(ctx, eqn, in_deltas):
    d = in_deltas[0]
    if not isinstance(d, Sliced):
        return default_rule(ctx, eqn, in_deltas)
    ndim = len(eqn.invars[0].aval.shape)
    dims = tuple(x % ndim for x in eqn.params["dimensions"])
    axes, idxs, dropped = [], [], []
    for a, i in zip(d.axes, d.idxs):
        if a in dims:
            dropped.append(a)
        else:
            axes.append(a - sum(1 for s in dims if s < a))
            idxs.append(i)
    out = lax.squeeze(d.value, dims)
    token = d.token if not dropped else (d.token, "squeeze", tuple(dropped))
    return [Sliced(tuple(axes), tuple(idxs), out, token)]


RULES[P.squeeze_p] = squeeze_rule


def transpose_rule(ctx, eqn, in_deltas):
    d = in_deltas[0]
    if not isinstance(d, Sliced):
        return default_rule(ctx, eqn, in_deltas)
    perm = tuple(eqn.params["permutation"])
    axes = tuple(perm.index(a) for a in d.axes)
    out = lax.transpose(d.value, perm)
    return [Sliced(axes, d.idxs, out, d.token)]


RULES[P.transpose_p] = transpose_rule


# ----------------------------------------------------------------- reductions


def reduce_rule(ctx, eqn, in_deltas):
    """Reductions whose axes avoid the sliced axes stay sliced."""
    d = in_deltas[0]
    if not isinstance(d, Sliced) or len(eqn.invars) != 1:
        return default_rule(ctx, eqn, in_deltas)
    red = tuple(eqn.params["axes"])
    if set(red) & set(d.axes):
        rule = OVERLAP_REDUCE_RULES.get(eqn.primitive)
        if rule is None:
            return default_rule(ctx, eqn, in_deltas)
        return rule(ctx, eqn, in_deltas)
    axes = tuple(a - sum(1 for r in red if r < a) for a in d.axes)
    outs = bind(eqn, d.value)
    return [Sliced(axes, d.idxs, o, d.token) for o in outs]


OVERLAP_REDUCE_RULES = {}


def reduce_sum_overlap_rule(ctx, eqn, in_deltas):
    """
    The paper's incremental accumulator: a sum over a sliced axis is updated as
    ``old_sum - sum(old_slice) + sum(new_slice)``. Axes that are sliced but not reduced
    stay sliced. The result is guarded: if the formula is not finite (e.g. an ``-inf``
    in the cached sum) the reduction is recomputed from the materialized input, so a
    non-finite value can never poison the cache.
    """
    (d,) = in_deltas
    atom = eqn.invars[0]
    red = tuple(eqn.params["axes"])
    keep = [(a, i) for a, i in zip(d.axes, d.idxs) if a not in red]
    out_axes = tuple(a - sum(1 for r in red if r < a) for a, _ in keep)
    out_idxs = tuple(i for _, i in keep)
    old_out = ctx.slice(eqn.outvars[0], out_axes, out_idxs)
    old_slice = ctx.slice(atom, d.axes, d.idxs)
    formula = old_out + bind(eqn, d.value)[0] - bind(eqn, old_slice)[0]

    def recompute():
        full = bind(eqn, ctx.materialize_sliced(atom, d))[0]
        return full if not out_axes else gather_rect(full, out_axes, out_idxs)

    out = lax.cond(jnp.all(jnp.isfinite(formula)), lambda: formula, recompute)
    if out_axes:
        token = (d.token, "reduce", red)
        return [Sliced(out_axes, out_idxs, out, token)]
    return [Dense(out)]


OVERLAP_REDUCE_RULES[P.reduce_sum_p] = reduce_sum_overlap_rule

for _name in (
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    "reduce_and",
    "reduce_or",
    "reduce_xor",
    "argmax",
    "argmin",
):
    if hasattr(P, _name + "_p"):
        RULES[getattr(P, _name + "_p")] = reduce_rule


# ----------------------------------------------------------------- gather


def bind_with(eqn, params, *args):
    """Re-bind ``eqn``'s primitive with modified ``params``."""
    bind_params = eqn.primitive.get_bind_params(params)
    with eqn.ctx.manager:
        out = eqn.primitive.bind(*args, **bind_params)
    return list(out) if eqn.primitive.multiple_results else [out]


def _gather_geometry(eqn):
    dn = eqn.params["dimension_numbers"]
    op_shape = eqn.invars[0].aval.shape
    ix_ndim = len(eqn.invars[1].aval.shape)
    out_ndim = len(eqn.outvars[0].aval.shape)
    batch_dims_out = [i for i in range(out_ndim) if i not in dn.offset_dims]
    window_dims = [
        d
        for d in range(len(op_shape))
        if d not in dn.collapsed_slice_dims and d not in dn.operand_batching_dims
    ]
    pairs = dict(zip(dn.start_indices_batching_dims, dn.operand_batching_dims))
    ok = len(batch_dims_out) == ix_ndim - 1 and len(window_dims) == len(dn.offset_dims)
    return dn, op_shape, ix_ndim, batch_dims_out, window_dims, pairs, ok


def _full_window_dim(dn, op_shape, slice_sizes, a):
    """Operand dim ``a`` is gathered whole: slicing it slices the matching output dim."""
    return (
        a not in dn.collapsed_slice_dims
        and a not in dn.start_index_map
        and a not in dn.operand_batching_dims
        and slice_sizes[a] == op_shape[a]
    )


def gather_rule(ctx, eqn, in_deltas):
    """
    ``gather`` stays sliced when (a) the indices are sliced along an index batch dim
    (the matching output batch dim is sliced; an operand batching dim paired with it
    is sliced alike), and/or (b) the operand is sliced along a dim that is gathered
    whole (the matching output offset dim is sliced, e.g. the batch dim of a batched
    ``diagonal``). A sliced operand along an indexed dim is recomputed densely.
    """
    operand_atom, indices_atom = eqn.invars[:2]
    d_op, d_ix = in_deltas[:2]
    if (
        len(eqn.invars) > 2
        or isinstance(d_op, Dense)
        or isinstance(d_ix, Dense)
        or (
            isinstance(d_op, Sliced)
            and isinstance(d_ix, Sliced)
            and d_op.token != d_ix.token
        )
    ):
        return default_rule(ctx, eqn, in_deltas)
    dn, op_shape, ix_ndim, batch_dims_out, window_dims, pairs, ok = _gather_geometry(
        eqn
    )
    if not ok:
        return default_rule(ctx, eqn, in_deltas)
    slice_sizes = list(eqn.params["slice_sizes"])
    out_axes, out_idxs, token = [], [], None
    needed = {}  # operand batching dims that must be sliced like the indices
    if isinstance(d_ix, Sliced):
        if any(a == ix_ndim - 1 for a in d_ix.axes):
            return default_rule(ctx, eqn, in_deltas)
        token = d_ix.token
        for a, i in zip(d_ix.axes, d_ix.idxs):
            out_axes.append(batch_dims_out[a])
            out_idxs.append(i)
            if a in pairs:
                needed[pairs[a]] = i
        indices = d_ix.value
    else:
        indices = ctx.full(indices_atom)
    if isinstance(d_op, Sliced):
        token = d_op.token if token is None else token
        if any(a not in d_op.axes for a in needed):
            return default_rule(ctx, eqn, in_deltas)
        for a, i in zip(d_op.axes, d_op.idxs):
            if a in needed:
                continue
            if not _full_window_dim(dn, op_shape, slice_sizes, a):
                return default_rule(ctx, eqn, in_deltas)
            out_axes.append(dn.offset_dims[window_dims.index(a)])
            out_idxs.append(i)
            slice_sizes[a] = int(i.shape[0])
        operand = d_op.value
    elif needed:
        operand = ctx.slice(operand_atom, tuple(needed), tuple(needed.values()))
    else:
        operand = ctx.full(operand_atom)
    params = {**eqn.params, "slice_sizes": tuple(slice_sizes)}
    out = bind_with(eqn, params, operand, indices)[0]
    return [Sliced(tuple(out_axes), tuple(out_idxs), out, token)]


RULES[P.gather_p] = gather_rule


# ----------------------------------------------------------------- batched ops


def _batched_rule(core_dims):
    """
    Primitives that act independently over leading batch dims (``core_dims`` trailing
    dims per input are the operation's own): slices along batch dims pass through.
    """

    def rule(ctx, eqn, in_deltas):
        if any(isinstance(d, Dense) for d in in_deltas):
            return default_rule(ctx, eqn, in_deltas)
        sliced = [d for d in in_deltas if isinstance(d, Sliced)]
        ref = sliced[0]
        for d in sliced[1:]:
            if not d.same_block(ref):
                return default_rule(ctx, eqn, in_deltas)
        args = []
        for atom, d, core in zip(eqn.invars, in_deltas, core_dims):
            nbatch = len(atom.aval.shape) - core
            if any(a >= nbatch for a in ref.axes):
                return default_rule(ctx, eqn, in_deltas)
            if isinstance(d, Sliced):
                args.append(d.value)
            else:
                args.append(ctx.slice(atom, ref.axes, ref.idxs))
        outs = bind(eqn, *args)
        return [Sliced(ref.axes, ref.idxs, o, ref.token) for o in outs]

    return rule


RULES[P.cholesky_p] = _batched_rule((2,))
RULES[P.triangular_solve_p] = _batched_rule((2, 2))


def _dot_layout(eqn):
    (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
    l_ndim = len(eqn.invars[0].aval.shape)
    r_ndim = len(eqn.invars[1].aval.shape)
    l_free = [d for d in range(l_ndim) if d not in lc and d not in lb]
    r_free = [d for d in range(r_ndim) if d not in rc and d not in rb]
    return tuple(lb), tuple(rb), l_free, r_free


def dot_general_rule(ctx, eqn, in_deltas):
    """Slices along batch dims (both operands alike) or along one operand's free dims."""
    d_l, d_r = in_deltas[:2]
    if (
        len(eqn.invars) != 2
        or any(isinstance(d, Dense) for d in in_deltas)
        or (
            isinstance(d_l, Sliced)
            and isinstance(d_r, Sliced)
            and d_l.token != d_r.token
        )
    ):
        return default_rule(ctx, eqn, in_deltas)
    lb, rb, l_free, r_free = _dot_layout(eqn)
    nb = len(lb)
    out_axes, out_idxs = [], []
    l_axes, r_axes = {}, {}  # operand axis -> idx for slicing Same operands
    token = None
    for d, this_b, other_b, free, offset, this, other in (
        (d_l, lb, rb, l_free, nb, l_axes, r_axes),
        (d_r, rb, lb, r_free, nb + len(l_free), r_axes, l_axes),
    ):
        if not isinstance(d, Sliced):
            continue
        token = d.token
        for a, i in zip(d.axes, d.idxs):
            if a in this_b:
                j = this_b.index(a)
                if j not in [this_b.index(x) for x in this if x in this_b]:
                    out_axes.append(j)
                    out_idxs.append(i)
                this[a] = i
                other[other_b[j]] = i
            elif a in free:
                out_axes.append(offset + free.index(a))
                out_idxs.append(i)
                this[a] = i
            else:  # contracting
                return default_rule(ctx, eqn, in_deltas)
    args = []
    for atom, d, need in ((eqn.invars[0], d_l, l_axes), (eqn.invars[1], d_r, r_axes)):
        if isinstance(d, Sliced):
            if set(d.axes) != set(need):
                return default_rule(ctx, eqn, in_deltas)
            args.append(d.value)
        elif need:
            args.append(ctx.slice(atom, tuple(need), tuple(need.values())))
        else:
            args.append(ctx.full(atom))
    out = bind(eqn, *args)[0]
    return [Sliced(tuple(out_axes), tuple(out_idxs), out, token)]


RULES[P.dot_general_p] = dot_general_rule


# ----------------------------------------------------------------- calls


def _call_rule(param_name):
    def rule(ctx, eqn, in_deltas):
        closed = eqn.params[param_name]
        return ctx.interpret(closed.jaxpr, closed.consts, in_deltas, eqn)

    return rule


RULES[P.jit_p] = _call_rule("jaxpr")
RULES[P.closed_call_p] = _call_rule("call_jaxpr")


# Public elementwise functions behind common ``custom_jvp_call`` equations, keyed by
# the traced function's name. On the sliced path they are re-called on the slices so
# that ``jax.grad`` through the emitted program still sees the custom JVP rule.
CUSTOM_JVP_REGISTRY = {
    "xlogy": jax.scipy.special.xlogy,
    "xlog1py": jax.scipy.special.xlog1py,
    "logaddexp": jnp.logaddexp,
    "logaddexp2": jnp.logaddexp2,
}


def _broadcast_elementwise(ctx, eqn, in_deltas, fn):
    """Apply an elementwise ``fn`` on slices, honouring size-1 broadcast dims."""
    sliced = [d for d in in_deltas if isinstance(d, Sliced)]
    ref = sliced[0]
    out_shape = eqn.outvars[0].aval.shape
    for d in sliced[1:]:
        if not d.same_block(ref):
            return None
    args = []
    for atom, d in zip(eqn.invars, in_deltas):
        shape = atom.aval.shape
        if isinstance(d, Sliced):
            if len(shape) != len(out_shape):
                return None
            args.append(d.value)
        elif len(shape) == 0:
            args.append(ctx.full(atom))
        elif len(shape) != len(out_shape):
            return None
        else:
            axes = [(a, i) for a, i in zip(ref.axes, ref.idxs) if shape[a] != 1]
            args.append(
                ctx.slice(atom, tuple(a for a, _ in axes), tuple(i for _, i in axes))
            )
    out = fn(*args)
    return [Sliced(ref.axes, ref.idxs, out, ref.token)]


def custom_jvp_rule(ctx, eqn, in_deltas):
    # Dense path: rebind whole so autodiff keeps the custom rule.
    if any(isinstance(d, Dense) for d in in_deltas):
        return default_rule(ctx, eqn, in_deltas)
    closed = eqn.params["call_jaxpr"]
    name = getattr(getattr(closed.jaxpr, "debug_info", None), "func_name", None)
    fn = CUSTOM_JVP_REGISTRY.get(name)
    if fn is not None:
        out = _broadcast_elementwise(ctx, eqn, in_deltas, fn)
        if out is not None:
            return out
    # Sliced path for an unknown function: interpret the primal body with the Δ-rules
    # (shape-specialised inner equations cannot be rebound on slices). Gradients
    # through this path differentiate the primal rather than the custom rule.
    return ctx.interpret(closed.jaxpr, closed.consts, in_deltas, eqn)


RULES[P.custom_jvp_call_p] = custom_jvp_rule
if hasattr(P, "custom_vjp_call_p"):
    RULES[P.custom_vjp_call_p] = custom_jvp_rule


def get_rule(primitive):
    return RULES.get(primitive, default_rule)


# ----------------------------------------------------------------- pullback
#
# ``PULLBACK[primitive](ctx, eqn, out_axes, out_idxs)`` returns the OLD value of the
# equation's output on the block (``out_axes``, ``out_idxs``) by slicing the inputs
# and re-running the primitive on the slices, or ``None`` to fall back to reading the
# cached full value. Each is the inverse of the forward rule's axis bookkeeping.

PULLBACK = {}


def _ks(idxs):
    return [int(i.shape[0]) for i in idxs]


def pb_elementwise(ctx, eqn, axes, idxs):
    out_ndim = len(eqn.outvars[0].aval.shape)
    args = []
    for atom in eqn.invars:
        ndim_in = len(atom.aval.shape)
        if ndim_in == 0:
            args.append(ctx.full(atom))
        elif ndim_in != out_ndim:
            return None
        else:
            args.append(ctx.slice(atom, axes, idxs))
    return bind(eqn, *args)[0]


for _p in ELEMENTWISE:
    PULLBACK[_p] = pb_elementwise


def pb_broadcast_in_dim(ctx, eqn, axes, idxs):
    if len(eqn.invars) != 1:
        return None
    (atom,) = eqn.invars
    in_shape = atom.aval.shape
    shape = eqn.params["shape"]
    bd = tuple(eqn.params["broadcast_dimensions"])
    in_axes, in_idxs = [], []
    for o, i in zip(axes, idxs):
        if o in bd:
            a = bd.index(o)
            if in_shape[a] != 1:
                in_axes.append(a)
                in_idxs.append(i)
    sl = ctx.slice(atom, tuple(in_axes), tuple(in_idxs))
    return lax.broadcast_in_dim(sl, sliced_shape(shape, axes, _ks(idxs)), bd)


PULLBACK[P.broadcast_in_dim_p] = pb_broadcast_in_dim


def pb_iota(ctx, eqn, axes, idxs):
    shape = sliced_shape(eqn.params["shape"], axes, _ks(idxs))
    dim = eqn.params["dimension"]
    dtype = eqn.params["dtype"]
    out = lax.broadcasted_iota(dtype, shape, dim)
    if dim in axes:
        idx = idxs[axes.index(dim)].astype(dtype)
        view = [1] * len(shape)
        view[dim] = shape[dim]
        out = jnp.broadcast_to(jnp.reshape(idx, view), shape)
    return out


PULLBACK[P.iota_p] = pb_iota


def pb_squeeze(ctx, eqn, axes, idxs):
    (atom,) = eqn.invars
    ndim = len(atom.aval.shape)
    dims = tuple(x % ndim for x in eqn.params["dimensions"])
    kept = [a for a in range(ndim) if a not in dims]
    in_axes = tuple(kept[o] for o in axes)
    return lax.squeeze(ctx.slice(atom, in_axes, idxs), dims)


PULLBACK[P.squeeze_p] = pb_squeeze


def pb_transpose(ctx, eqn, axes, idxs):
    (atom,) = eqn.invars
    perm = tuple(eqn.params["permutation"])
    in_axes = tuple(perm[o] for o in axes)
    return lax.transpose(ctx.slice(atom, in_axes, idxs), perm)


PULLBACK[P.transpose_p] = pb_transpose


def pb_reshape(ctx, eqn, axes, idxs):
    if len(eqn.invars) != 1 or eqn.params.get("dimensions") is not None:
        return None
    (atom,) = eqn.invars
    in_shape = tuple(atom.aval.shape)
    new_sizes = tuple(eqn.params["new_sizes"])
    hi = max(axes) + 1
    if new_sizes[:hi] == in_shape[:hi]:
        in_axes = axes
    else:
        lo = min(axes)
        n_tail = len(new_sizes) - lo
        if n_tail <= len(in_shape) and new_sizes[lo:] == in_shape[-n_tail:]:
            shift = len(in_shape) - len(new_sizes)
            in_axes = tuple(a + shift for a in axes)
        else:
            return None
    sl = ctx.slice(atom, in_axes, idxs)
    return jnp.reshape(sl, sliced_shape(new_sizes, axes, _ks(idxs)))


PULLBACK[P.reshape_p] = pb_reshape


def pb_reduce(ctx, eqn, axes, idxs):
    if len(eqn.invars) != 1:
        return None
    (atom,) = eqn.invars
    ndim = len(atom.aval.shape)
    red = tuple(eqn.params["axes"])
    kept = [a for a in range(ndim) if a not in red]
    in_axes = tuple(kept[o] for o in axes)
    return bind(eqn, ctx.slice(atom, in_axes, idxs))[0]


for _name in (
    "reduce_sum",
    "reduce_max",
    "reduce_min",
    "reduce_prod",
    "reduce_and",
    "reduce_or",
):
    if hasattr(P, _name + "_p"):
        PULLBACK[getattr(P, _name + "_p")] = pb_reduce


def pb_gather(ctx, eqn, axes, idxs):
    if len(eqn.invars) != 2:
        return None
    operand_atom, indices_atom = eqn.invars
    dn, op_shape, ix_ndim, batch_dims_out, window_dims, pairs, ok = _gather_geometry(
        eqn
    )
    if not ok:
        return None
    slice_sizes = list(eqn.params["slice_sizes"])
    op_axes, ix_axes = {}, {}
    for o, i in zip(axes, idxs):
        if o in dn.offset_dims:
            a = window_dims[dn.offset_dims.index(o)]
            if not _full_window_dim(dn, op_shape, slice_sizes, a):
                return None
            op_axes[a] = i
            slice_sizes[a] = int(i.shape[0])
        else:
            j = batch_dims_out.index(o)
            ix_axes[j] = i
            if j in pairs:
                op_axes[pairs[j]] = i
    operand = (
        ctx.slice(operand_atom, tuple(op_axes), tuple(op_axes.values()))
        if op_axes
        else ctx.full(operand_atom)
    )
    indices = (
        ctx.slice(indices_atom, tuple(ix_axes), tuple(ix_axes.values()))
        if ix_axes
        else ctx.full(indices_atom)
    )
    params = {**eqn.params, "slice_sizes": tuple(slice_sizes)}
    return bind_with(eqn, params, operand, indices)[0]


PULLBACK[P.gather_p] = pb_gather


def _pb_batched(core_dims):
    def pb(ctx, eqn, axes, idxs):
        args = []
        for atom, core in zip(eqn.invars, core_dims):
            if any(a >= len(atom.aval.shape) - core for a in axes):
                return None
            args.append(ctx.slice(atom, axes, idxs))
        return bind(eqn, *args)[0]

    return pb


PULLBACK[P.cholesky_p] = _pb_batched((2,))
PULLBACK[P.triangular_solve_p] = _pb_batched((2, 2))


def pb_dot_general(ctx, eqn, axes, idxs):
    lb, rb, l_free, r_free = _dot_layout(eqn)
    nb = len(lb)
    l_axes, r_axes = {}, {}
    for o, i in zip(axes, idxs):
        if o < nb:
            l_axes[lb[o]] = i
            r_axes[rb[o]] = i
        elif o < nb + len(l_free):
            l_axes[l_free[o - nb]] = i
        else:
            r_axes[r_free[o - nb - len(l_free)]] = i
    args = []
    for atom, need in ((eqn.invars[0], l_axes), (eqn.invars[1], r_axes)):
        if need:
            args.append(ctx.slice(atom, tuple(need), tuple(need.values())))
        else:
            args.append(ctx.full(atom))
    return bind(eqn, *args)[0]


PULLBACK[P.dot_general_p] = pb_dot_general


def is_literal(atom):
    return isinstance(atom, Literal)


__all__ = [
    "RULES",
    "PULLBACK",
    "ELEMENTWISE",
    "OVERLAP_REDUCE_RULES",
    "bind",
    "materialize",
    "default_rule",
    "get_rule",
    "SAME",
]
