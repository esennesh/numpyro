# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Learn: log importance weights and the bounds built from them (Section 4).

Everything here is a single expression in the model parameters ``theta`` and the
proposal parameters, so ``jax.grad`` of :func:`elbo` or :func:`iwae` with respect
to the parameter dict *is* the learning step; no :class:`~numpyro.infer.SVI` loop
is involved.

The weights are computed entirely in unconstrained space. Writing ``z = T(u)``
for the support transforms,

.. math::

    \\log \\frac{\\gamma_\\theta(z)}{q(z)}
    = \\log \\gamma_\\theta(T(u)) + \\log |J_T(u)| - \\log q_u(u)
    = -\\mathrm{PE}(u) - \\log q_u(u),

where ``PE`` is :func:`numpyro.infer.util.potential_energy`, which already
carries the log-Jacobian. When the model has discrete latent sites they are
marginalized out of ``gamma_theta`` by enumeration, so the weights are weights
for the discrete-marginal density that the proposal is defined on.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import jax
from jax import Array, random
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from numpyro import handlers
from numpyro.contrib.fwl.options import FWLOptions
from numpyro.contrib.fwl.structure import ModelStructure
from numpyro.infer.util import potential_energy


def make_estimators(
    model: Callable,
    structure: ModelStructure,
    guide: Callable,
    options: FWLOptions,
    model_args: tuple,
    model_kwargs: dict,
    seed_key: Array,
) -> tuple[Callable, Callable, Callable]:
    """
    Build ``(log_weights, elbo, iwae)``.

    Each takes ``(params, rng_key, *args, **kwargs)``. ``params`` holds
    constrained values for the model's own :func:`numpyro.param` sites and for
    the guide's proposal parameters, matching what
    :class:`~numpyro.handlers.substitute` expects. Model args default to the ones
    ``find_weigh_learn`` was called with.
    """
    latent_name = f"{options.prefix}_latent"
    model_param_names = frozenset(
        name
        for name, site in structure.prototype_trace.items()
        if site["type"] == "param"
    )
    use_enum = structure.has_discrete

    def prepared_model(params: dict[str, Array]) -> Callable:
        substituted = handlers.substitute(
            handlers.seed(model, seed_key),
            data={k: v for k, v in params.items() if k in model_param_names},
        )
        if not use_enum:
            return substituted
        from numpyro.contrib.funsor import config_enumerate, enum

        return enum(
            config_enumerate(substituted),
            first_available_dim=structure.first_available_dim,
        )

    def log_weights(
        params: dict[str, Array],
        rng_key: Array,
        *args: Any,
        num_particles: Optional[int] = None,
        **kwargs: Any,
    ) -> Array:
        """``(P,)`` log importance weights ``log gamma_theta(z_p) - log q(z_p)``."""
        if not args and not kwargs:
            args, kwargs = model_args, model_kwargs
        particles = options.num_particles if num_particles is None else num_particles
        conditioned = prepared_model(params)
        guide_with_params = handlers.substitute(guide, data=params)

        if structure.enumeration_only:
            # The proposal is empty, so log q = 0 and enumeration marginalizes
            # every latent site: each weight is exactly log Z(theta). The
            # particles are kept so the returned shape does not depend on the
            # model, but they are identical and the "bounds" below are tight.
            exact = -potential_energy(conditioned, args, kwargs, {}, enum=use_enum)
            return jnp.broadcast_to(exact, (particles,))

        def single(key: Array) -> Array:
            guide_trace = handlers.trace(
                handlers.seed(guide_with_params, key)
            ).get_trace(*args, **kwargs)
            site = guide_trace[latent_name]
            latent = site["value"]
            log_q = site["fn"].log_prob(latent)
            energy = potential_energy(
                conditioned,
                args,
                kwargs,
                structure.packing.unpack(latent),
                enum=use_enum,
            )
            return -energy - log_q

        return jax.vmap(single)(random.split(rng_key, particles))

    def elbo(
        params: dict[str, Array], rng_key: Array, *args: Any, **kwargs: Any
    ) -> Array:
        """The Jensen bound ``E_q[log gamma_theta(z) - log q(z)] <= log Z(theta)``."""
        return jnp.mean(log_weights(params, rng_key, *args, **kwargs))

    def iwae(
        params: dict[str, Array], rng_key: Array, *args: Any, **kwargs: Any
    ) -> Array:
        """The tighter importance-weighted bound ``log (1/P) sum_p w_p``."""
        weights = log_weights(params, rng_key, *args, **kwargs)
        return logsumexp(weights) - jnp.log(weights.shape[0])

    return log_weights, elbo, iwae
