# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Per-primitive change-propagation rules.

A rule has the signature ``rule(ctx, eqn, in_deltas) -> list[Δ]`` and must be a local
identity: whatever Δ-values it returns, materializing them must equal evaluating the
equation on the materialized inputs. The default rule materializes every input and
rebinds the primitive, so unknown primitives are always handled soundly (as ``Dense``).
"""

from jax import lax
from jax.extend.core import Literal, primitives as P
import jax.numpy as jnp

from numpyro.contrib.incremental.delta import SAME, Dense, Same, Sliced, sliced_shape

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
    return [Dense(o) for o in bind(eqn, *args)]


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


def gather_rule(ctx, eqn, in_deltas):
    operand_atom, indices_atom = eqn.invars[:2]
    d_op, d_ix = in_deltas[:2]
    if not isinstance(d_ix, Sliced) or isinstance(d_op, Dense) or len(eqn.invars) > 2:
        return default_rule(ctx, eqn, in_deltas)
    dn = eqn.params["dimension_numbers"]
    ix_ndim = len(indices_atom.aval.shape)
    if any(a == ix_ndim - 1 for a in d_ix.axes):
        return default_rule(ctx, eqn, in_deltas)
    out_ndim = len(eqn.outvars[0].aval.shape)
    batch_dims_out = [i for i in range(out_ndim) if i not in dn.offset_dims]
    if len(batch_dims_out) != ix_ndim - 1:
        return default_rule(ctx, eqn, in_deltas)
    out_axes = tuple(batch_dims_out[a] for a in d_ix.axes)
    # Operand batching dims paired with sliced index dims must be sliced alike.
    pairs = dict(zip(dn.start_indices_batching_dims, dn.operand_batching_dims))
    op_axes = tuple(pairs[a] for a in d_ix.axes if a in pairs)
    op_idxs = tuple(i for a, i in zip(d_ix.axes, d_ix.idxs) if a in pairs)
    if isinstance(d_op, Same):
        operand = (
            ctx.slice(operand_atom, op_axes, op_idxs)
            if op_axes
            else ctx.full(operand_atom)
        )
    else:  # Sliced operand: only acceptable when it is sliced exactly like the pairs
        if d_op.token != d_ix.token or tuple(d_op.axes) != op_axes:
            return default_rule(ctx, eqn, in_deltas)
        operand = d_op.value
    out = bind(eqn, operand, d_ix.value)[0]
    return [Sliced(out_axes, d_ix.idxs, out, d_ix.token)]


RULES[P.gather_p] = gather_rule


# ----------------------------------------------------------------- calls


def _call_rule(param_name):
    def rule(ctx, eqn, in_deltas):
        closed = eqn.params[param_name]
        return ctx.interpret(closed.jaxpr, closed.consts, in_deltas)

    return rule


RULES[P.jit_p] = _call_rule("jaxpr")
RULES[P.closed_call_p] = _call_rule("call_jaxpr")


def custom_jvp_rule(ctx, eqn, in_deltas):
    # Dense/Same path: rebind whole so autodiff keeps the custom rule. The sliced path
    # is refined in a later step; for now it materializes.
    return default_rule(ctx, eqn, in_deltas)


RULES[P.custom_jvp_call_p] = custom_jvp_rule
if hasattr(P, "custom_vjp_call_p"):
    RULES[P.custom_vjp_call_p] = custom_jvp_rule


def get_rule(primitive):
    return RULES.get(primitive, default_rule)


def is_literal(atom):
    return isinstance(atom, Literal)


__all__ = [
    "RULES",
    "ELEMENTWISE",
    "OVERLAP_REDUCE_RULES",
    "bind",
    "materialize",
    "default_rule",
    "get_rule",
    "SAME",
]
