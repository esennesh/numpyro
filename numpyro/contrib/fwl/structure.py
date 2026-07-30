# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Static analysis of a model for the Find/Weigh/Learn procedure.

This module answers the questions the procedure needs answered before any
numerical work happens: which sites are observed, which unobserved sites are
enumerable-discrete (and therefore eliminated exactly by max-product message
passing), which are continuous (and therefore handed to a continuous
optimizer), and how the continuous sites pack into a single unconstrained
vector.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from typing import Any, Optional

from jax import Array
import jax.numpy as jnp

from numpyro import handlers
from numpyro.distributions.transforms import biject_to
from numpyro.infer.initialization import init_to_sample
from numpyro.infer.inspect import get_dependencies
from numpyro.infer.util import _guess_max_plate_nesting, helpful_support_errors


def _is_factor_site(site: dict) -> bool:
    """Whether a sample site is a ``numpyro.factor`` statement."""
    return type(site["fn"]).__name__ == "Unit"


@dataclass(frozen=True)
class LatentPacking:
    """
    A bijection between a dict of per-site unconstrained values and a single
    flat vector, in the spirit of :class:`~numpyro.infer.autoguide.AutoContinuous`.

    The flat vector is what the continuous optimizer in :mod:`~numpyro.contrib.fwl.find`
    searches over and what the mixture proposal in :mod:`~numpyro.contrib.fwl.guide`
    is defined on. Leading batch dimensions are preserved, so both ``pack`` and
    ``unpack`` compose with :func:`jax.vmap`.

    :param names: unconstrained site names, in model order.
    :param shapes: unconstrained shape of each site.
    :param dim: total dimension of the flat vector.
    """

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    dim: int

    @property
    def sizes(self) -> tuple[int, ...]:
        return tuple(math.prod(shape) for shape in self.shapes)

    def unpack(self, flat: Array) -> dict[str, Array]:
        """Split a ``(..., dim)`` array into a dict of ``(..., *shape)`` arrays."""
        batch_shape = jnp.shape(flat)[:-1]
        out = {}
        offset = 0
        for name, shape, size in zip(self.names, self.shapes, self.sizes):
            chunk = flat[..., offset : offset + size]
            out[name] = jnp.reshape(chunk, batch_shape + shape)
            offset += size
        return out

    def pack(self, values: dict[str, Array]) -> Array:
        """Concatenate a dict of ``(..., *shape)`` arrays into a ``(..., dim)`` array."""
        if not self.names:
            # A model with no continuous latents packs to an empty vector; there
            # are no values to read a batch shape from.
            return jnp.zeros((0,))
        chunks = []
        for name, shape in zip(self.names, self.shapes):
            value = values[name]
            batch_ndim = jnp.ndim(value) - len(shape)
            batch_shape = jnp.shape(value)[:batch_ndim]
            chunks.append(jnp.reshape(value, batch_shape + (-1,)))
        return jnp.concatenate(chunks, axis=-1)


@dataclass(frozen=True)
class ModelStructure:
    """
    The result of analyzing a model for Find/Weigh/Learn.

    :param continuous: unobserved sites with continuous support, in model order.
    :param discrete: unobserved sites with finite, enumerable support, in model order.
    :param observed: observed sample sites (excluding ``numpyro.factor`` statements).
    :param deterministic: ``numpyro.deterministic`` sites.
    :param max_plate_nesting: plate nesting depth, used to allocate enumeration dims.
    :param packing: the flat-vector bijection for the continuous sites.
    :param prototype_trace: a trace of the model with latents at their init values.
    :param dependencies: the moralized dependency structure from
        :func:`~numpyro.infer.inspect.get_dependencies`. Retained for
        introspection; the elimination order itself is chosen by funsor.
    """

    continuous: tuple[str, ...]
    discrete: tuple[str, ...]
    observed: tuple[str, ...]
    deterministic: tuple[str, ...]
    max_plate_nesting: int
    packing: LatentPacking
    prototype_trace: dict[str, Any]
    dependencies: dict[str, Any]

    @property
    def has_discrete(self) -> bool:
        return len(self.discrete) > 0

    @property
    def enumeration_only(self) -> bool:
        """
        Whether there is nothing for a continuous proposal to do, because the
        model has no continuous latent sites. Enumeration then evaluates
        ``log Z(theta)`` exactly and every importance weight equals it.
        """
        return self.packing.dim == 0

    @property
    def first_available_dim(self) -> int:
        """The first tensor dim available to parallel enumeration."""
        return -self.max_plate_nesting - 1

    def summary(self) -> str:
        """A short human-readable description of the structure found."""
        lines = [
            f"continuous latents: {list(self.continuous)} (packed dim {self.packing.dim})"
            + (" -- enumeration only, log Z is exact" if self.enumeration_only else ""),
            f"discrete latents:   {list(self.discrete)} (eliminated by max-product)",
            f"observed:           {list(self.observed)}",
            f"max_plate_nesting:  {self.max_plate_nesting}",
        ]
        posterior = self.dependencies["posterior_dependencies"]
        for name in self.continuous + self.discrete:
            blanket = posterior.get(name, {})
            coupling = {k: sorted(v) for k, v in blanket.items() if v}
            lines.append(
                f"  {name}: markov blanket {sorted(blanket)}"
                + (f", plates inducing full coupling {coupling}" if coupling else "")
            )
        return "\n".join(lines)


def analyze(
    model: Callable,
    rng_key: Array,
    model_args: tuple = (),
    model_kwargs: Optional[dict] = None,
    init_strategy: Callable = init_to_sample,
    allow_enumeration_only: bool = False,
) -> ModelStructure:
    """
    Classify the sites of ``model`` and build the continuous latent packing.

    :param allow_enumeration_only: whether to accept a model with no continuous
        latent sites, for which the procedure degenerates to exact enumeration.

    :raises NotImplementedError: if the model has latent sites this procedure
        cannot handle: discrete sites with countably infinite support (the
        ``Countable``/WGF row of Table 1 in the design document, not yet
        implemented) or plates with subsampling.
    :raises ValueError: if the model has no continuous latent sites and
        ``allow_enumeration_only`` is false.
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs

    # ``init_to_sample`` (or another init strategy) keeps this working for
    # improper priors, which have no ``sample`` method.
    seeded = handlers.substitute(
        handlers.seed(model, rng_key), substitute_fn=init_strategy
    )
    trace = handlers.trace(seeded).get_trace(*model_args, **model_kwargs)

    for name, site in trace.items():
        if site["type"] != "plate":
            continue
        size, subsample_size = site["args"]
        if subsample_size is not None and subsample_size != size:
            raise NotImplementedError(
                f"Plate '{name}' uses subsampling (size={size}, "
                f"subsample_size={subsample_size}). Find/Weigh/Learn locates modes of "
                "the full joint density, which a subsampled plate does not evaluate."
            )

    continuous, discrete, observed, deterministic, countable = [], [], [], [], []
    shapes = []
    for name, site in trace.items():
        if site["type"] == "deterministic":
            deterministic.append(name)
            continue
        if site["type"] != "sample" or _is_factor_site(site):
            continue
        if site["is_observed"]:
            observed.append(name)
            continue
        support = site["fn"].support
        if support.is_discrete:
            if site["fn"].has_enumerate_support:
                discrete.append(name)
            else:
                countable.append(name)
            continue
        with helpful_support_errors(site):
            transform = biject_to(support)
        unconstrained = transform.inv(site["value"])
        continuous.append(name)
        shapes.append(jnp.shape(unconstrained))

    if countable:
        raise NotImplementedError(
            f"Sites {countable} have discrete support that is not finitely enumerable. "
            "Mode-finding over countably infinite discrete supports needs the discrete "
            "Wasserstein-gradient-flow subroutine (the 'Countable' rows of Table 1), "
            "which is not implemented."
        )
    if not continuous and not allow_enumeration_only:
        raise ValueError(
            "Model has no continuous latent sites. With every discrete latent site "
            "enumerated, log Z(theta) is available exactly from "
            "numpyro.contrib.funsor.log_density, so there is nothing for an "
            "importance-sampling proposal to do. Pass allow_enumeration_only=True "
            "to return that exact result instead of raising: the guide is then "
            "empty and every importance weight equals log Z(theta)."
        )

    packing = LatentPacking(
        names=tuple(continuous),
        shapes=tuple(shapes),
        dim=sum(math.prod(shape) for shape in shapes),
    )
    return ModelStructure(
        continuous=tuple(continuous),
        discrete=tuple(discrete),
        observed=tuple(observed),
        deterministic=tuple(deterministic),
        max_plate_nesting=_guess_max_plate_nesting(trace),
        packing=packing,
        prototype_trace=trace,
        dependencies=get_dependencies(model, model_args, model_kwargs),
    )


def initial_values(
    model: Callable,
    rng_key: Array,
    model_args: tuple,
    model_kwargs: dict,
    structure: ModelStructure,
    init_strategy: Callable,
) -> tuple[Array, dict[str, Array]]:
    """
    Draw one starting point for Find: a flat unconstrained continuous vector
    and constrained values for the discrete sites.
    """
    seeded = handlers.substitute(
        handlers.seed(model, rng_key), substitute_fn=init_strategy
    )
    trace = handlers.trace(seeded).get_trace(*model_args, **model_kwargs)
    unconstrained = {}
    for name in structure.continuous:
        site = trace[name]
        with helpful_support_errors(site):
            transform = biject_to(site["fn"].support)
        unconstrained[name] = transform.inv(site["value"])
    discrete = {name: trace[name]["value"] for name in structure.discrete}
    return structure.packing.pack(unconstrained), discrete
