# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Weigh: turning found modes into importance-sampling proposals (Section 3).

Each mode becomes one Gaussian component whose covariance is a damped local
empirical Fisher, ``(F(z*) + lambda I)^-1``. ``F`` is the sum of per-factor
gradient outer products,

.. math::

    \\hat F(z^*) = \\sum_v [\\nabla_z -\\log f_v(z^*)][\\nabla_z -\\log f_v(z^*)]^\\top,

which is the reading of Section 3 that stays nonzero at a joint mode: the
gradient of the *total* energy vanishes there, but the individual factors' do
not. The factors are the model's sample sites together with the log-Jacobian
terms of the support transforms, since the proposal lives in unconstrained
space.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional, Union

import jax
from jax import Array
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

from numpyro.contrib.fwl.find import FindState, index_map, replay
from numpyro.contrib.fwl.junction import CliqueTree
from numpyro.contrib.fwl.options import FWLOptions
from numpyro.contrib.fwl.structure import ModelStructure
from numpyro.distributions.util import is_identically_one


def factor_energies_fn(
    model: Callable,
    model_args: tuple,
    model_kwargs: dict,
    structure: ModelStructure,
    discrete: dict[str, Array],
    seed_key: Array,
    granularity: str = "site",
    assigned: Optional[frozenset[str]] = None,
    owners: Optional[dict[str, str]] = None,
) -> Callable[[Array], Array]:
    """
    Build ``u -> (M,)``, the vector of per-factor energies at unconstrained
    continuous position ``u``.

    With ``granularity="site"`` there is one entry per sample site (its summed
    negative log-density). With ``granularity="element"`` each site contributes
    one entry per element of its log-density array, which is the finer-grained
    empirical Fisher at the cost of a Jacobian row per observation.

    :param assigned: restrict to these factor names, for a clique-local Fisher.
    :param owners: maps a derived factor's name to the site it belongs with, used
        to keep a support transform's Jacobian factor with its own site.
    """
    owners = {} if owners is None else owners

    def energies(latent: Array) -> Array:
        trace = replay(
            model, model_args, model_kwargs, structure, latent, discrete, seed_key
        )
        terms = []
        for name, site in trace.items():
            if site["type"] != "sample":
                continue
            if assigned is not None and owners.get(name, name) not in assigned:
                continue
            value = site["value"]
            intermediates = site["intermediates"]
            if intermediates:
                log_prob = site["fn"].log_prob(value, intermediates)
            else:
                log_prob = site["fn"].log_prob(value)
            scale = site["scale"]
            if (scale is not None) and (not is_identically_one(scale)):
                log_prob = scale * log_prob
            log_prob = jnp.asarray(log_prob, dtype=latent.dtype)
            if granularity == "element":
                terms.append(-jnp.ravel(log_prob))
            else:
                terms.append(-jnp.reshape(jnp.sum(log_prob), (1,)))
        if not terms:
            return jnp.zeros((0,), dtype=latent.dtype)
        return jnp.concatenate(terms)

    return energies


def empirical_fisher(
    energies: Callable[[Array], Array], latent: Array, diagonal: bool
) -> Array:
    """
    ``sum_v g_v g_v^T`` at ``latent``, as a ``(D,)`` diagonal or a ``(D, D)`` matrix.

    The Jacobian is taken in whichever mode is cheaper: forward mode costs one
    pass per input dimension, reverse mode one per factor.
    """
    num_factors = jax.eval_shape(energies, latent).shape[0]
    jacobian = jax.jacfwd if latent.shape[-1] <= num_factors else jax.jacrev
    grads = jacobian(energies)(latent)  # (M, D)
    if diagonal:
        return jnp.sum(jnp.square(grads), axis=0)
    return grads.T @ grads


def proposal_scale(fisher: Array, damping: float, diagonal: bool) -> Array:
    """
    The Gaussian scale of ``N(z*, (F + lambda I)^-1)``: a ``(D,)`` standard
    deviation when diagonal, else a ``(D, D)`` lower Cholesky factor.
    """
    if diagonal:
        return jnp.sqrt(jnp.reciprocal(fisher + damping))
    dim = fisher.shape[-1]
    damped = fisher + damping * jnp.eye(dim, dtype=fisher.dtype)
    covariance = cho_solve(
        cho_factor(damped, lower=True), jnp.eye(dim, dtype=fisher.dtype)
    )
    # Symmetrize to clean up the asymmetry left by the two triangular solves.
    covariance = 0.5 * (covariance + covariance.T)
    return jnp.linalg.cholesky(covariance)


def clique_precision_blocks(
    model: Callable,
    model_args: tuple,
    model_kwargs: dict,
    structure: ModelStructure,
    tree: CliqueTree,
    find_state: FindState,
    options: FWLOptions,
    seed_key: Array,
) -> tuple[Array, ...]:
    """
    Decompose the damped empirical Fisher into one positive-definite block per
    clique, and return each block's lower Cholesky factor, stacked over modes.

    Because ``g_v`` vanishes outside factor ``v``'s scope, and every factor is
    assigned to a clique containing its scope, ``g_v g_v^T`` is supported inside
    one clique's block. Summing the assigned factors' outer products there, plus
    each coordinate's share of the damping, reconstructs ``F-hat + lambda I``
    exactly while making every block separately positive definite -- so the
    Cholesky factors can be learned without losing either definiteness or the
    graph's sparsity pattern.

    :return: one ``(K, |c|, |c|)`` array per clique, in clique order.
    """
    index = index_map(structure.packing)
    scopes = [
        jnp.asarray(
            sorted(i for name in sorted(clique) for i in index[name]), dtype=jnp.int32
        )
        for clique in tree.cliques
    ]
    # Split the damping so that every coordinate receives exactly lambda in total.
    membership = [0] * structure.packing.dim
    for scope in scopes:
        for i in scope.tolist():
            membership[i] += 1
    shares = [
        jnp.asarray([options.damping / membership[i] for i in scope.tolist()])
        for scope in scopes
    ]

    def per_mode(latent: Array, discrete: dict[str, Array]) -> tuple[Array, ...]:
        blocks = []
        for clique, scope, share in zip(range(len(tree.cliques)), scopes, shares):
            energies = factor_energies_fn(
                model,
                model_args,
                model_kwargs,
                structure,
                discrete,
                seed_key,
                options.fisher_granularity,
                assigned=frozenset(tree.factors[clique]),
                owners={f"_{name}_log_det": name for name in structure.continuous},
            )
            grads = jax.jacfwd(energies)(latent)[:, scope]  # (M, |c|)
            block = grads.T @ grads + jnp.diag(share)
            blocks.append(jnp.linalg.cholesky(block))
        return tuple(blocks)

    return jax.vmap(per_mode)(find_state.latent, find_state.discrete)


def weigh(
    model: Callable,
    model_args: tuple,
    model_kwargs: dict,
    structure: ModelStructure,
    find_state: FindState,
    options: FWLOptions,
    seed_key: Array,
    tree: Optional[CliqueTree] = None,
) -> Union[Array, tuple[Array, ...]]:
    """
    Compute each mode's proposal shape.

    :return: ``(K, D)`` standard deviations for ``covariance="diagonal"``,
        ``(K, D, D)`` covariance Cholesky factors for ``"full"``, or one
        ``(K, |c|, |c|)`` precision-block Cholesky factor per clique for
        ``"graph"``.
    """
    diagonal = options.covariance == "diagonal"
    if options.covariance == "graph":
        if tree is None:
            raise ValueError("covariance='graph' requires a junction tree.")
        if structure.enumeration_only:
            return ()
        return clique_precision_blocks(
            model,
            model_args,
            model_kwargs,
            structure,
            tree,
            find_state,
            options,
            seed_key,
        )
    if structure.enumeration_only:
        # No continuous coordinates to place a Gaussian on.
        shape = (options.num_modes, 0) if diagonal else (options.num_modes, 0, 0)
        return jnp.zeros(shape)

    def per_mode(latent: Array, discrete: dict[str, Array]) -> Array:
        energies = factor_energies_fn(
            model,
            model_args,
            model_kwargs,
            structure,
            discrete,
            seed_key,
            options.fisher_granularity,
        )
        fisher = empirical_fisher(energies, latent, diagonal)
        return proposal_scale(fisher, options.damping, diagonal)

    return jax.vmap(per_mode)(find_state.latent, find_state.discrete)
