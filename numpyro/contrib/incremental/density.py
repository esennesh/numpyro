# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Stage 1: compile a NumPyro model to a deterministic density program.

The density program is a :class:`jax.extend.core.ClosedJaxpr` whose inputs are the
latent site values (continuous sites in unconstrained space, discrete sites raw) followed
by the array-valued model arguments, and whose outputs are the per-factor *unsummed*
``log_prob`` arrays. Jacobian pseudo-factors ``_{name}_log_det`` emitted by
:func:`numpyro.infer.util._unconstrain_reparam` are attributed to their owner site.
"""

from collections import namedtuple
from functools import partial

import numpy as np

import jax
from jax import random
from jax.extend.core import ClosedJaxpr, jaxpr_as_fun
import jax.numpy as jnp

from numpyro import distributions as dist
from numpyro.distributions.distribution import validation_enabled
from numpyro.distributions.transforms import biject_to
from numpyro.handlers import seed, substitute, trace
from numpyro.infer.initialization import init_to_sample
from numpyro.infer.util import (
    _substitute_default_key,
    _unconstrain_reparam,
    compute_log_probs,
    helpful_support_errors,
)

SiteMeta = namedtuple(
    "SiteMeta",
    [
        "name",
        "shape",  # shape of the stored value (unconstrained for continuous sites)
        "dtype",
        "plates",  # tuple of (plate_name, axis) pairs, axis >= 0 into ``shape``
        "is_discrete",
        "support",
        "transform",  # biject_to(support) for continuous sites, else None
        "support_size",  # static enumerate_support size for discrete sites, else None
    ],
)
SiteMeta.__doc__ = "Static metadata of a latent sample site in a density program."

FactorMeta = namedtuple(
    "FactorMeta",
    [
        "name",
        "shape",  # shape of the unsummed log_prob array
        "dtype",
        "plates",  # tuple of (plate_name, axis) pairs, axis >= 0 into ``shape``
        "owner",  # the latent site a ``_x_log_det`` factor belongs to, else ``name``
        "is_observed",
    ],
)
FactorMeta.__doc__ = "Static metadata of one factor (log density output) of a program."


def _is_array_leaf(leaf):
    return isinstance(leaf, (jax.Array, np.ndarray)) or (
        isinstance(leaf, np.generic) and not isinstance(leaf, (np.bool_, np.integer))
    )


def _plate_axes(site, ndim_batch):
    """Map each enclosing plate of ``site`` to a non-negative axis."""
    return tuple(
        sorted(
            (frame.name, ndim_batch + frame.dim) for frame in site["cond_indep_stack"]
        ),
    )


def _check_supported(prototype_trace):
    for name, site in prototype_trace.items():
        if site["type"] == "plate":
            size, subsample_size = site["args"]
            if subsample_size is not None and subsample_size != size:
                raise NotImplementedError(
                    f"Plate '{name}' is subsampled; incremental densities need "
                    "full-size plates (slice at the jaxpr level instead)."
                )
        if site["type"] != "sample" or site["is_observed"]:
            continue
        if site["infer"].get("enumerate") == "parallel":
            raise NotImplementedError(
                f"Site '{name}' is marked for parallel enumeration; incremental "
                "densities keep discrete sites as explicit latent values."
            )
        if isinstance(site["fn"], dist.Delta):
            raise ValueError(
                f"Sample site '{name}' has a delta distribution; use "
                "`numpyro.deterministic` to add this value to the trace instead."
            )


class DensityProgram:
    """
    A compiled density program.

    Use :func:`trace_density` to build one. The attributes most users need are
    ``sites`` (name -> :class:`SiteMeta`), ``factors`` (name -> :class:`FactorMeta`, in
    output order) and :meth:`eval`.
    """

    def __init__(
        self,
        jaxpr: ClosedJaxpr,
        site_names,
        sites,
        factor_names,
        factors,
        args_tree,
        args_leaves_static,
        num_eqns,
        model=None,
        deterministic_sites=(),
    ):
        self.jaxpr = jaxpr
        # The model with param/mutable sites frozen, for replays (postprocessing).
        self.model = model
        self.deterministic_sites = tuple(deterministic_sites)
        self.site_names = tuple(site_names)
        self.sites = sites
        self.factor_names = tuple(factor_names)
        self.factors = factors
        self._args_tree = args_tree
        # One entry per leaf of (model_args, model_kwargs): the leaf itself when it is
        # static (closed over), or ``None`` when it is a traced program input.
        self._args_leaves_static = args_leaves_static
        self.num_eqns = num_eqns

    # ------------------------------------------------------------------ inputs
    @property
    def num_site_inputs(self):
        return len(self.site_names)

    @property
    def num_arg_inputs(self):
        return sum(leaf is None for leaf in self._args_leaves_static)

    def flatten_args(self, model_args=(), model_kwargs=None):
        """Return the traced leaves of ``(model_args, model_kwargs)`` in input order."""
        model_kwargs = {} if model_kwargs is None else model_kwargs
        leaves, tree = jax.tree.flatten((tuple(model_args), model_kwargs))
        if tree != self._args_tree or len(leaves) != len(self._args_leaves_static):
            raise ValueError(
                "Model arguments do not match the structure the density program was "
                f"traced with: got {tree}, expected {self._args_tree}."
            )
        return [
            leaf
            for leaf, static in zip(leaves, self._args_leaves_static)
            if static is None
        ]

    def flatten_values(self, values):
        """Return the site leaves of ``values`` in input order."""
        missing = [name for name in self.site_names if name not in values]
        if missing:
            raise ValueError(f"Missing values for sites {missing}.")
        return [jnp.asarray(values[name]) for name in self.site_names]

    def flatten_inputs(self, values, model_args=(), model_kwargs=None):
        return self.flatten_values(values) + self.flatten_args(model_args, model_kwargs)

    # ------------------------------------------------------------------ eval
    def eval(self, values, model_args=(), model_kwargs=None):
        """Evaluate every factor from scratch; returns ``{factor: log_prob array}``."""
        outs = jaxpr_as_fun(self.jaxpr)(
            *self.flatten_inputs(values, model_args, model_kwargs)
        )
        return dict(zip(self.factor_names, outs))

    def log_density(self, values, model_args=(), model_kwargs=None):
        """Joint log density (sum over all factors) from scratch."""
        factors = self.eval(values, model_args, model_kwargs)
        return sum((jnp.sum(v) for v in factors.values()), start=jnp.zeros(()))

    # ------------------------------------------------------------------ info
    def summary(self):
        """A short human-readable description of sites, factors and plates."""
        lines = [f"DensityProgram: {self.num_eqns} equations"]
        lines.append("sites:")
        for name in self.site_names:
            s = self.sites[name]
            kind = "discrete" if s.is_discrete else "continuous"
            plates = ", ".join(f"{p}@{a}" for p, a in s.plates) or "-"
            lines.append(f"  {name}: {kind} {s.shape} plates[{plates}]")
        lines.append("factors:")
        for name in self.factor_names:
            f = self.factors[name]
            plates = ", ".join(f"{p}@{a}" for p, a in f.plates) or "-"
            owner = f" (owner {f.owner})" if f.owner != name else ""
            obs = " observed" if f.is_observed else ""
            lines.append(f"  {name}: {f.shape} plates[{plates}]{owner}{obs}")
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()


def trace_density(
    model,
    model_args=(),
    model_kwargs=None,
    *,
    init_strategy=init_to_sample,
    validate=False,
    rng_key=None,
):
    """
    Compile ``model`` to a :class:`DensityProgram`.

    :param model: a NumPyro model.
    :param tuple model_args: model arguments. Array-valued leaves become traced inputs
        of the program; every other leaf (Python scalars, strings, ...) is static.
    :param dict model_kwargs: model keyword arguments, treated like ``model_args``.
    :param init_strategy: initialization strategy used for the prototype trace; its
        values are also returned as the initial latent values.
    :param bool validate: whether distribution argument/sample validation is active
        while tracing. Off by default: validation only adds masking equations.
    :param rng_key: key for the prototype trace (defaults to ``random.key(0)``).
    :return: ``(program, init_values)`` where ``init_values`` maps every latent site to
        its initial stored value (unconstrained for continuous sites).
    """
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    model_args = tuple(model_args)
    rng_key = random.key(0) if rng_key is None else rng_key

    with validation_enabled(validate):
        prototype_trace = trace(
            substitute(seed(model, rng_key), substitute_fn=init_strategy)
        ).get_trace(*model_args, **model_kwargs)
    _check_supported(prototype_trace)

    frozen = {
        k: site["value"]
        for k, site in prototype_trace.items()
        if site["type"] in ("param", "mutable")
    }
    frozen_model = _substitute_default_key(substitute(model, data=frozen))

    continuous, discrete, init_values, transforms = [], [], {}, {}
    for name, site in prototype_trace.items():
        if site["type"] != "sample" or site["is_observed"]:
            continue
        if site["fn"].support.is_discrete:
            discrete.append(name)
            init_values[name] = site["value"]
        else:
            continuous.append(name)
            with helpful_support_errors(site, raise_warnings=True):
                t = biject_to(site["fn"].support)
            transforms[name] = t
            init_values[name] = t.inv(site["value"])
    site_names = continuous + discrete
    if not site_names:
        raise ValueError("The model has no latent sample sites.")

    # Partition model arguments into traced (array) and static leaves.
    arg_leaves, args_tree = jax.tree.flatten((model_args, model_kwargs))
    args_leaves_static = [None if _is_array_leaf(leaf) else leaf for leaf in arg_leaves]
    traced_arg_leaves = [leaf for leaf in arg_leaves if _is_array_leaf(leaf)]

    def rebuild_args(traced_leaves):
        it = iter(traced_leaves)
        leaves = [next(it) if s is None else s for s in args_leaves_static]
        return jax.tree.unflatten(args_tree, leaves)

    captured = {}

    def density(site_leaves, traced_leaves):
        values = dict(zip(site_names, site_leaves))
        cont_values = {k: values[k] for k in continuous}
        disc_values = {k: values[k] for k in discrete}
        args, kwargs = rebuild_args(traced_leaves)
        substituted = substitute(
            frozen_model, substitute_fn=partial(_unconstrain_reparam, cont_values)
        )
        log_probs, model_trace = compute_log_probs(
            substituted, args, kwargs, disc_values, sum_log_prob=False
        )
        # Keep only Python-level metadata; the trace itself holds tracers.
        captured["factor_meta"] = {
            name: (
                site["is_observed"],
                _plate_axes(site, jnp.ndim(log_probs[name])),
            )
            for name, site in model_trace.items()
            if site["type"] == "sample"
        }
        return log_probs

    with validation_enabled(validate):
        closed_jaxpr, out_shape = jax.make_jaxpr(density, return_shape=True)(
            [jnp.asarray(init_values[n]) for n in site_names], traced_arg_leaves
        )
    factor_names = list(out_shape.keys())
    out_leaves, out_tree = jax.tree.flatten(out_shape)
    # Dict outputs flatten in sorted-key order; make the factor order explicit.
    factor_names = sorted(factor_names)
    assert len(out_leaves) == len(factor_names) == len(closed_jaxpr.jaxpr.outvars)

    sites = {}
    for name in site_names:
        site = prototype_trace[name]
        value = init_values[name]
        is_discrete = name in discrete
        support_size = None
        if is_discrete and site["fn"].has_enumerate_support:
            support_size = int(site["fn"].enumerate_support(expand=False).shape[0])
        sites[name] = SiteMeta(
            name=name,
            shape=tuple(jnp.shape(value)),
            dtype=jnp.result_type(value),
            plates=_plate_axes(site, len(site["fn"].batch_shape)),
            is_discrete=is_discrete,
            support=site["fn"].support,
            transform=transforms.get(name),
            support_size=support_size,
        )

    factors = {}
    for name, leaf in zip(factor_names, out_leaves):
        is_observed, plates = captured["factor_meta"][name]
        owner = name
        if name.startswith("_") and name.endswith("_log_det"):
            candidate = name[1 : -len("_log_det")]
            if candidate in sites:
                owner = candidate
        factors[name] = FactorMeta(
            name=name,
            shape=tuple(leaf.shape),
            dtype=leaf.dtype,
            plates=plates,
            owner=owner,
            is_observed=is_observed,
        )

    program = DensityProgram(
        jaxpr=closed_jaxpr,
        site_names=site_names,
        sites=sites,
        factor_names=factor_names,
        factors=factors,
        args_tree=args_tree,
        args_leaves_static=args_leaves_static,
        num_eqns=_count_eqns(closed_jaxpr.jaxpr),
        model=frozen_model,
        deterministic_sites=[
            k for k, s in prototype_trace.items() if s["type"] == "deterministic"
        ],
    )
    return program, init_values


def _count_eqns(jaxpr):
    n = 0
    for eqn in jaxpr.eqns:
        n += 1
        for param in eqn.params.values():
            inner = getattr(param, "jaxpr", None)
            if inner is not None and hasattr(inner, "eqns"):
                n += _count_eqns(inner)
            elif hasattr(param, "eqns"):
                n += _count_eqns(param)
    return n
