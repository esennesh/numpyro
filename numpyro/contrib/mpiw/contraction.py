# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Core contraction for massively parallel importance weighting (MPIW).

MPIW (Bowyer et al., 2024; Aitchison, 2019) draws ``K`` samples for each of ``n``
latent variables and reweights over all ``K**n`` combinations of samples, exploiting
conditional independence in the model/guide graph so that the cost is
``O(K**(1 + max_i |parents(i)|))`` rather than ``O(K**n)``.

This module contains the graph-agnostic numerical core: given a set of log-probability
*factors*, each labelled with the named sample-index ("K") dimensions and plate
dimensions it ranges over, it performs the log-space sum-product contraction that yields
an estimate of the log marginal likelihood ``log P_MP(x)``, and -- via the source-term
trick -- the self-normalized importance weights of every latent site.

The factors themselves are produced elsewhere (from a traced model/guide); keeping the
contraction separate lets it be tested directly against models with analytic marginal
likelihoods and posterior moments.

References:

1. *Using autodiff to estimate posterior moments, marginals and samples*,
   Sam Bowyer, Thomas Heap, Laurence Aitchison. UAI 2024.
2. *Tensor Monte Carlo: particle methods for the GPU era*, Laurence Aitchison. NeurIPS
   2019.
"""

from collections import OrderedDict
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.special

import funsor
import funsor.optimizer

funsor.set_backend("jax")


def _slice_tensor(tensor, dim, index):
    """Slice a funsor ``Tensor`` at ``dim == index``, dropping ``dim`` from its inputs.

    ``index`` may be a traced scalar (e.g. a ``lax.scan`` carry), so a dynamic slice is
    used. Tensors that do not mention ``dim`` are returned unchanged.
    """
    if dim not in tensor.inputs:
        return tensor
    axis = list(tensor.inputs).index(dim)
    data = jax.lax.dynamic_index_in_dim(tensor.data, index, axis=axis, keepdims=False)
    new_inputs = OrderedDict(
        (name, domain) for name, domain in tensor.inputs.items() if name != dim
    )
    return funsor.Tensor(data, new_inputs, tensor.dtype)


class NamedFactor(NamedTuple):
    """A single log-probability factor in an MPIW contraction.

    :param data: a real array whose axes correspond, in order, to ``dims``.
    :param dims: the names of the array's axes. Each name is either a sample-index
        ("K") dimension of some latent site or a plate dimension. All axes of ``data``
        must be named; scalar factors have empty ``dims``.
    """

    data: jax.Array
    dims: tuple[str, ...]

    def to_funsor(self) -> funsor.Funsor:
        inputs = OrderedDict(
            (name, funsor.Bint[self.data.shape[axis]])
            for axis, name in enumerate(self.dims)
        )
        return funsor.Tensor(self.data, inputs)


def contract_log_marginal(
    factors: list[NamedFactor],
    eliminate: frozenset[str],
    plates: frozenset[str],
    serial_dims: frozenset[str] = frozenset(),
) -> jax.Array:
    """Contract MPIW log-factors into a scalar ``log P_MP(x)``.

    Performs ``logsumexp``-``add`` sum-product over the sample-index dimensions,
    treating ``plates`` as product (independent) dimensions so that per-plate-element
    contractions are combined multiplicatively (additively in log space) rather than
    being folded into the combinatorial sum. This is what keeps the cost polynomial
    rather than exponential in the number of plated latents.

    :param factors: the log-probability factors. For each latent site ``i`` this
        should include the importance-weight log-ratio
        ``log p(z_i | parents) - log q(z_i | parents) - log K`` (the ``- log K``
        implementing the uniform ``1 / K**n`` average over sample combinations), and
        for each observed site the model log density ``log p(x | parents)``. Each item
        is either a :class:`NamedFactor` or, for factors produced directly from a traced
        model, a :class:`funsor.Funsor` already carrying named inputs.
    :param eliminate: names of the sample-index ("K") dimensions to sum out.
    :param plates: names of plate dimensions (a subset of the dimensions appearing in
        ``factors``); these are reduced as products, not sums.
    :param serial_dims: names of eliminated ("K") dimensions to sum over *serially*,
        via a :func:`jax.lax.scan` loop that slices those dimensions out one index at a
        time and densely contracts the remainder. This trades compute for memory: the
        combined intermediate over the remaining dimensions is never materialized with
        the serial dimensions present. Must be a subset of ``eliminate`` and must name
        only non-plated (global) dimensions -- a dimension eliminated independently per
        plate element cannot be summed with a single shared serial index. An empty set
        (default) uses the fully dense contraction. The result is identical to the dense
        path (up to floating point) and remains differentiable, so the source-term trick
        still works.
    :returns: the scalar ``log P_MP(x)``.
    """
    tensors = [f.to_funsor() if isinstance(f, NamedFactor) else f for f in factors]

    if not serial_dims:
        with funsor.interpretations.lazy:
            lazy = funsor.sum_product.sum_product(
                funsor.ops.logaddexp,
                funsor.ops.add,
                tensors,
                eliminate=eliminate | plates,
                plates=plates,
            )
        result = funsor.optimizer.apply_optimizer(lazy)
        if result.inputs:
            raise ValueError(
                "Expected log P_MP to contract to a scalar, but the result still has "
                f"free dimensions {tuple(result.inputs)}. Check that `eliminate` names "
                "every sample-index dimension and that plate dimensions are declared."
            )
        return result.data

    if not serial_dims <= eliminate:
        raise ValueError(
            f"serial_dims must be a subset of eliminate; got {tuple(serial_dims)} "
            f"not in {tuple(eliminate)}."
        )
    # A serial dimension must be eliminated by a single global logsumexp, not one that
    # is nested inside a plate (whose elimination is independent per plate element). A
    # dimension is "global" iff some factor mentions it without any plate dimension;
    # slicing a plated dimension with one shared index would silently give wrong results.
    for dim in serial_dims:
        if not any(dim in t.inputs and plates.isdisjoint(t.inputs) for t in tensors):
            raise ValueError(
                f"serial dimension {dim!r} occurs only inside plates; serial "
                "contraction is only supported for non-plated (global) dimensions."
            )
    # Peel off one serial dimension and sum over it with a scan; recurse on the rest.
    dim = sorted(serial_dims)[0]
    rest = serial_dims - {dim}
    size = next(t.inputs[dim].size for t in tensors if dim in t.inputs)

    def body(carry, index):
        sliced = [_slice_tensor(t, dim, index) for t in tensors]
        partial = contract_log_marginal(sliced, eliminate - {dim}, plates, rest)
        return carry, partial

    _, partials = jax.lax.scan(body, 0.0, jnp.arange(size))
    return jax.scipy.special.logsumexp(partials, axis=0)


def contract_with_source_terms(
    build_factors: Callable[
        [dict[str, jax.Array]],
        tuple[list[NamedFactor], frozenset[str], frozenset[str]],
    ],
    source_shapes: dict[str, tuple[int, ...]],
    serial_dims: frozenset[str] = frozenset(),
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Estimate ``log P_MP(x)`` and per-site importance weights via the source-term trick.

    A "source term" ``J_i`` is a real perturbation added into latent site ``i``'s
    factor, one entry per configuration of that site's own dimensions (its K dimension,
    plus any plates it lives in). Because the contraction is a log-sum-exp, the gradient
    of ``log P_MP`` with respect to ``J_i`` evaluated at ``J = 0`` is exactly the
    self-normalized marginal importance weight of each of site ``i``'s samples -- the
    weights sum to one over the K dimension (within each plate element). Posterior
    moments of any statistic ``m`` follow as ``sum_k w_i(k) m(z_i^k)``, computed by the
    caller, so this core stays statistics-agnostic.

    :param build_factors: given a dict of source terms (one array per site name, matching
        ``source_shapes``), returns ``(factors, eliminate, plates)`` as accepted by
        :func:`contract_log_marginal`. The source term for a site must be added into that
        site's factor.
    :param source_shapes: for each latent site name, the shape of its source-term array
        (the shape of that site's own dimensions, in the same axis order used in its
        factor).
    :param serial_dims: eliminated dimensions to contract serially (see
        :func:`contract_log_marginal`); the gradient (weights) is taken through the
        serial ``lax.scan``, so this bounds memory for the moment computation too.
    :returns: ``(log_marginal, weights)`` where ``weights[name]`` has shape
        ``source_shapes[name]`` and, summed over the site's K axis, is one per plate
        element.
    """

    def logp(source_terms: dict[str, jax.Array]) -> jax.Array:
        factors, eliminate, plates = build_factors(source_terms)
        return contract_log_marginal(factors, eliminate, plates, serial_dims)

    zeros = {name: jnp.zeros(shape) for name, shape in source_shapes.items()}
    log_marginal, weights = jax.value_and_grad(logp)(zeros)
    return log_marginal, weights
