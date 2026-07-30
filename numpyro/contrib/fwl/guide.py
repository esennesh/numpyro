# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
The mixture-of-Gaussians proposal built around the found modes (Section 3).

The guide follows the :class:`~numpyro.infer.autoguide.AutoContinuous` layout: a
single auxiliary site carries the whole unconstrained latent vector, drawn from

.. math::

    q_{z^*_{1:K}}(u) = \\sum_{k=1}^K \\frac{1}{K}
        \\mathcal{N}\\big(u; z^*_k, (\\hat F(z^*_k) + \\lambda I)^{-1}\\big),

and the individual model sites are then emitted as ``Delta`` sites carrying the
support transform's log-Jacobian, so the guide can stand in for any continuous
guide (including under :class:`~numpyro.infer.SVI`). Discrete latent sites are
absent from the guide by design: they are marginalized exactly by enumeration,
so a downstream :class:`~numpyro.infer.SVI` use needs
:class:`~numpyro.infer.elbo.TraceEnum_ELBO` and a ``config_enumerate``-d model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional, Union

from jax import Array, lax
import jax.numpy as jnp
from jax.typing import ArrayLike

import numpyro
from numpyro.contrib.fwl.find import index_map
from numpyro.contrib.fwl.junction import CliqueTree
from numpyro.contrib.fwl.options import FWLOptions
from numpyro.contrib.fwl.structure import ModelStructure
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to
from numpyro.distributions.util import sum_rightmost
from numpyro.infer.util import helpful_support_errors


def clique_param_name(prefix: str, clique: int) -> str:
    """Param name holding one clique's precision-block Cholesky factor."""
    return f"{prefix}_clique_chol_{clique}"


def guide_param_constraints(
    options: FWLOptions, num_cliques: int = 0
) -> dict[str, Any]:
    """The constraint attached to each of the guide's own param sites."""
    prefix = options.prefix
    result: dict[str, Any] = {f"{prefix}_loc": constraints.real}
    if options.covariance == "diagonal":
        result[f"{prefix}_scale"] = constraints.positive
    elif options.covariance == "full":
        result[f"{prefix}_scale_tril"] = constraints.lower_cholesky
    else:
        for clique in range(num_cliques):
            result[clique_param_name(prefix, clique)] = constraints.lower_cholesky
    if options.learn_mixing_weights:
        result[f"{prefix}_mixing_logits"] = constraints.real
    return result


def assemble_precision(
    blocks: Sequence[ArrayLike], scopes: Sequence[Array], dim: int
) -> Array:
    """
    Sum clique-supported positive-definite blocks into a ``(K, D, D)`` precision.

    Each block enters as ``L_c L_c^T`` scattered onto its clique's coordinates, so
    the result is positive definite whenever the blocks cover every coordinate,
    and its zero pattern is exactly the union of the clique blocks -- the moral
    graph. The conditional independencies of the proposal are therefore the
    model's own, by construction rather than by penalty.
    """
    first = jnp.asarray(blocks[0])
    precision = jnp.zeros((first.shape[0], dim, dim), dtype=first.dtype)
    for raw, scope in zip(blocks, scopes):
        block = jnp.asarray(raw)
        contribution = block @ jnp.swapaxes(block, -1, -2)
        precision = precision.at[:, scope[:, None], scope[None, :]].add(contribution)
    return precision


def make_guide(
    structure: ModelStructure,
    locs: Array,
    scales: Union[Array, tuple[Array, ...]],
    options: FWLOptions,
    tree: Optional[CliqueTree] = None,
) -> tuple[Callable, dict[str, Array]]:
    """
    Build the proposal guide and the initial values of its parameters.

    :param locs: ``(K, D)`` unconstrained modes from Find.
    :param scales: whatever :func:`~numpyro.contrib.fwl.weigh.weigh` returned --
        ``(K, D)`` standard deviations, ``(K, D, D)`` covariance Cholesky factors,
        or one precision-block Cholesky factor per clique.
    :param tree: the junction tree, required when ``options.covariance == "graph"``.
    :return: a ``(guide, init_params)`` pair. ``init_params`` is empty when
        ``options.learnable`` is false, since the proposal is then frozen, and
        when the model has no continuous latents, since the guide is then empty.
    """
    prefix = options.prefix
    latent_name = f"{prefix}_latent"
    num_modes = locs.shape[0]
    diagonal = options.covariance == "diagonal"
    graph = options.covariance == "graph"
    scale_name = f"{prefix}_scale" if diagonal else f"{prefix}_scale_tril"

    if structure.enumeration_only:
        # Every latent site is enumerated in the model, so the guide covers
        # nothing. This is the standard empty-guide idiom for a fully enumerated
        # model, and is what TraceEnum_ELBO expects in that case.
        def empty_guide(*args: Any, **kwargs: Any) -> dict[str, ArrayLike]:
            return {}

        return empty_guide, {}

    # Narrow what Weigh returned to the shape this covariance mode expects, so the
    # pairing between the two is checked once here rather than assumed throughout.
    scopes: tuple[Array, ...] = ()
    blocks: tuple[ArrayLike, ...] = ()
    dense: Array = jnp.zeros(())
    if graph:
        if tree is None:
            raise ValueError("covariance='graph' requires a junction tree.")
        if not isinstance(scales, tuple):
            raise TypeError(
                "covariance='graph' expects one precision block per clique from "
                f"weigh(), got a single array of shape {jnp.shape(scales)}."
            )
        blocks = tuple(scales)
        index = index_map(structure.packing)
        scopes = tuple(
            jnp.asarray(
                sorted(i for name in sorted(clique) for i in index[name]),
                dtype=jnp.int32,
            )
            for clique in tree.cliques
        )
    else:
        if isinstance(scales, tuple):
            raise TypeError(
                f"covariance={options.covariance!r} expects a single array of "
                "scales from weigh(), got per-clique blocks."
            )
        dense = scales

    init_params: dict[str, Array] = {}
    if options.learnable:
        init_params[f"{prefix}_loc"] = locs
        if graph:
            for clique, block in enumerate(blocks):
                init_params[clique_param_name(prefix, clique)] = block
        else:
            init_params[scale_name] = dense
        if options.learn_mixing_weights:
            init_params[f"{prefix}_mixing_logits"] = jnp.zeros(
                num_modes, dtype=locs.dtype
            )

    def guide(*args: Any, **kwargs: Any) -> dict[str, ArrayLike]:
        if options.learnable:
            loc = numpyro.param(f"{prefix}_loc", locs)
            current_blocks = tuple(
                jnp.asarray(
                    numpyro.param(
                        clique_param_name(prefix, clique),
                        block,
                        constraint=constraints.lower_cholesky,
                    )
                )
                for clique, block in enumerate(blocks)
            )
            current_dense = (
                dense
                if graph
                else numpyro.param(
                    scale_name,
                    dense,
                    constraint=(
                        constraints.positive if diagonal else constraints.lower_cholesky
                    ),
                )
            )
        else:
            loc = lax.stop_gradient(locs)
            current_blocks = tuple(
                jnp.asarray(lax.stop_gradient(block)) for block in blocks
            )
            current_dense = lax.stop_gradient(dense)
        if options.learn_mixing_weights:
            logits = numpyro.param(
                f"{prefix}_mixing_logits", jnp.zeros(num_modes, dtype=locs.dtype)
            )
        else:
            logits = jnp.zeros(num_modes, dtype=locs.dtype)

        if graph:
            component = dist.MultivariateNormal(
                loc,
                precision_matrix=assemble_precision(
                    current_blocks, scopes, structure.packing.dim
                ),
            )
        elif diagonal:
            component = dist.Normal(loc, current_dense).to_event(1)
        else:
            component = dist.MultivariateNormal(loc, scale_tril=current_dense)
        latent = numpyro.sample(
            latent_name,
            dist.MixtureSameFamily(dist.Categorical(logits=logits), component),
            infer={"is_auxiliary": True},
        )
        return unpack_to_sites(latent, structure)

    return guide, init_params


def unpack_to_sites(
    latent: ArrayLike, structure: ModelStructure
) -> dict[str, ArrayLike]:
    """
    Emit each continuous model site as a ``Delta`` site holding the constrained
    value, with the support transform's log-Jacobian as its log-density.
    """
    result: dict[str, ArrayLike] = {}
    for name, unconstrained in structure.packing.unpack(jnp.asarray(latent)).items():
        site = structure.prototype_trace[name]
        with helpful_support_errors(site):
            transform = biject_to(site["fn"].support)
        value = transform(unconstrained)
        event_ndim = site["fn"].event_dim
        if numpyro.get_mask() is False:
            log_density = 0.0
        else:
            log_density = -transform.log_abs_det_jacobian(unconstrained, value)
            log_density = sum_rightmost(
                log_density, jnp.ndim(log_density) - jnp.ndim(value) + event_ndim
            )
        result[name] = numpyro.sample(
            name, dist.Delta(value, log_density=log_density, event_dim=event_ndim)
        )
    return result
