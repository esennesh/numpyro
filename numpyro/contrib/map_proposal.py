# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""A self-fitting MAP-centered proposal distribution."""

from contextlib import ExitStack
from functools import partial
from itertools import product
import math
from typing import Any, NamedTuple

from jax import lax, random
import jax.numpy as jnp

import numpyro
from numpyro import handlers
from numpyro.contrib.diag_sgd import (
    SmoothedCount,
    SmoothedDiscrete,
    SmoothICDFTransform,
    dsgd,
)
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import (
    ComposeTransform,
    biject_to,
)
from numpyro.infer import Predictive
from numpyro.infer.autoguide import AutoGuide
from numpyro.infer.initialization import init_to_uniform
from numpyro.infer.util import helpful_support_errors, log_density
from numpyro.optim import Adam, Minimize, _NumPyroOptim, optax_to_numpyro

__all__ = ["AutoMAPProposal"]


def _as_iterative_optimizer(optimizer):
    if isinstance(optimizer, Minimize):
        raise TypeError("AutoMAPProposal requires an iterative optimizer.")
    if isinstance(optimizer, _NumPyroOptim):
        return optimizer
    try:
        import optax
    except ImportError as error:
        raise ImportError(
            "An iterative optimizer must be a NumPyro optimizer, or Optax must be "
            "installed to use an optax.GradientTransformation."
        ) from error
    if not isinstance(optimizer, optax.GradientTransformation):
        raise TypeError(
            "Expected an iterative numpyro.optim._NumPyroOptim or "
            f"optax.GradientTransformation, but got {type(optimizer)}."
        )
    return optax_to_numpyro(optimizer)


class _OptimizationResult(NamedTuple):
    converged: bool
    losses: jnp.ndarray
    num_steps: int
    params: Any


class _ShiftedCategorical(dist.Distribution):
    """An integer-valued Categorical distribution with a batched lower bound."""

    arg_constraints = {}
    has_enumerate_support = True
    pytree_data_fields = ("base_dist", "low")

    def __init__(self, logits, low, *, validate_args=None):
        base_dist = dist.Categorical(logits=logits)
        batch_shape = jnp.broadcast_shapes(base_dist.batch_shape, jnp.shape(low))
        self.base_dist = base_dist.expand(batch_shape)
        self.low = jnp.broadcast_to(jnp.asarray(low, dtype=int), batch_shape)
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        high = self.low + self.base_dist.probs.shape[-1] - 1
        return constraints.integer_interval(self.low, high)

    def enumerate_support(self, expand=True):
        values = self.base_dist.enumerate_support(expand=expand)
        homogeneous = bool(jnp.all(self.low == self.low.reshape(-1)[0]))
        if not expand and not homogeneous:
            raise NotImplementedError(
                "Inhomogeneous `low` not supported by `enumerate_support`."
            )
        low = self.low if expand else self.low.reshape(-1)[0]
        return values + low

    def log_prob(self, value, intermediates=None):
        return self.base_dist.log_prob(value - self.low)

    def sample(self, key, sample_shape=()):
        return self.base_dist.sample(key, sample_shape) + self.low


class AutoMAPProposal(AutoGuide):
    r"""
    A self-fitting, factorized proposal centered at a joint relaxed MAP estimate.

    This guide is intended for use as a proposal in algorithms such as SMC and
    EM, rather than for optimization by :class:`~numpyro.infer.svi.SVI`. Each
    invocation fits a proposal for the supplied model ``*args, **kwargs``. It
    first computes the joint MAP estimate of the DSGD-smoothed target,

    .. math::

        \tilde z^* = \mathop{\rm argmax}_{\tilde z}
        \log \gamma_{\theta,\eta}(\tilde z; x),

    and then fits the continuous and count proposal parameters with stochastic
    gradients of

    .. math::

        \mathbb E_{q_\phi(\tilde z)}\left[
        \log q_\phi(\tilde z)
        - \log \gamma_{\theta,\eta}(\tilde z; x)\right].

    Every :math:`K` optimizer steps, convergence is checked using the average
    loss :math:`\bar{\mathcal L}_j` over that interval. Relative to the best
    previous average :math:`b_{j-1}`, an improvement is meaningful when

    .. math::

        b_{j-1} - \bar{\mathcal L}_j
        > \tau \left(1 + \lvert b_{j-1} \rvert\right).

    Optimization terminates after the configured number of consecutive checks
    without a meaningful improvement.

    For a continuous site with support :math:`S_i` and bijection
    :math:`T_{S_i}`, the proposal is

    .. math::

        u_i &\sim \mathcal N(T_{S_i}^{-1}(\tilde z_i^*), \rho_i^2 I), \\
        z_i &= T_{S_i}(u_i).

    Finite discrete sites use an independently fitted Categorical probability
    vector, initialized with its largest mass at the rounded relaxed MAP. Count
    sites use an exact :class:`~numpyro.distributions.discrete.GammaCount`
    proposal with independently fitted concentration and rate. The GammaCount
    proposal is passed through
    :func:`~numpyro.contrib.diag_sgd.SmoothedDiscrete` during fitting. Finite
    Categorical probabilities instead use their exact mean-field coordinate
    update under :attr:`model`, avoiding a support mismatch between two finite
    DSGD transforms with different CDF grids. Calls to the fitted guide return
    genuinely discrete samples for evaluation by :attr:`model`.

    Sites with equal support constraints reuse the same proposal family but do
    not share parameters. Both optimization phases use independent instances
    of :class:`~numpyro.optim.Adam` by default and require iterative NumPyro or
    Optax optimizers. Each proposal-optimization step draws fresh Monte Carlo
    particles. Optimization stops when the average loss fails to improve by
    more than the configured tolerance for the configured number of checks;
    the maximum step counts are safety caps rather than fixed iteration counts.

    This experimental guide supports continuous latent variables with
    bijectable supports and the discrete families supported by
    :func:`~numpyro.contrib.diag_sgd.dsgd`. It does not support data
    subsampling. It refits on every call and should not be used inside a batched
    or repeatedly evaluated SVI objective.

    :param callable model: A NumPyro model.
    :param float discrete_temperature: Smoothing temperature :math:`\eta` used
        for discrete latent sites.
    :param dict dsgd_kwargs: Optional keyword arguments forwarded to
        :func:`~numpyro.contrib.diag_sgd.dsgd` and to relaxed count proposals.
    :param float init_dispersion: Initial value for every continuous-site
        dispersion.
    :param callable init_loc_fn: A per-site initialization function.
    :param int map_max_steps: Maximum number of relaxed-MAP optimization steps.
    :param map_optimizer: Iterative NumPyro or Optax optimizer for the relaxed
        MAP objective. Defaults to Adam with step size ``0.01``.
    :param float map_tolerance: Relative loss-improvement tolerance for MAP.
    :param int num_dispersion_particles: Number of Monte Carlo particles in
        each stochastic estimate of the proposal objective.
    :param str prefix: Prefix used for internal proposal sample sites.
    :param int proposal_max_steps: Maximum number of stochastic proposal-fitting
        steps.
    :param proposal_optimizer: Iterative NumPyro or Optax optimizer for the
        proposal objective. Defaults to Adam with step size ``0.01``.
    :param float proposal_tolerance: Relative loss-improvement tolerance for
        proposal fitting.
    :param int termination_check_interval: Number of optimizer steps averaged
        for each convergence check.
    :param int termination_patience: Number of consecutive checks without a
        meaningful loss improvement required for convergence.
    """

    def __init__(
        self,
        model,
        *,
        discrete_temperature=0.1,
        dsgd_kwargs=None,
        init_dispersion=0.1,
        init_loc_fn=init_to_uniform,
        map_max_steps=1000,
        map_optimizer=None,
        map_tolerance=1e-5,
        num_dispersion_particles=32,
        prefix="auto",
        proposal_max_steps=1000,
        proposal_optimizer=None,
        proposal_tolerance=1e-3,
        termination_check_interval=50,
        termination_patience=5,
    ):
        if discrete_temperature <= 0:
            raise ValueError("discrete_temperature must be positive.")
        if init_dispersion <= 0:
            raise ValueError("init_dispersion must be positive.")
        if map_max_steps < 1:
            raise ValueError("map_max_steps must be positive.")
        if map_tolerance < 0:
            raise ValueError("map_tolerance must be nonnegative.")
        if num_dispersion_particles < 1:
            raise ValueError("num_dispersion_particles must be positive.")
        if proposal_max_steps < 1:
            raise ValueError("proposal_max_steps must be positive.")
        if proposal_tolerance < 0:
            raise ValueError("proposal_tolerance must be nonnegative.")
        if termination_check_interval < 1:
            raise ValueError("termination_check_interval must be positive.")
        if termination_patience < 1:
            raise ValueError("termination_patience must be positive.")

        dsgd_kwargs = {} if dsgd_kwargs is None else dsgd_kwargs.copy()
        if "smoothed_distributions" in dsgd_kwargs:
            raise ValueError(
                "dsgd_kwargs cannot override smoothed_distributions; "
                "AutoMAPProposal requires continuous relaxed densities."
            )

        self._base_init_loc_fn = init_loc_fn
        self._count_relaxations = {}
        self._dispersions = {}
        self._discrete_temperature = discrete_temperature
        self._dsgd_model = dsgd(model, smoothed_distributions=True, **dsgd_kwargs)
        self._event_dims = {}
        self._finite_distributions = {}
        self._init_dispersion = init_dispersion
        self._map_locs = {}
        self._map_max_steps = map_max_steps
        self._map_optimizer = _as_iterative_optimizer(
            Adam(step_size=0.01) if map_optimizer is None else map_optimizer
        )
        self._map_tolerance = map_tolerance
        self._num_dispersion_particles = num_dispersion_particles
        self._proposal_max_steps = proposal_max_steps
        self._proposal_optimizer = _as_iterative_optimizer(
            Adam(step_size=0.01) if proposal_optimizer is None else proposal_optimizer
        )
        self._proposal_params = {}
        self._proposal_tolerance = proposal_tolerance
        self._smooth_transforms = {}
        self._support_ids = {}
        self._supports = []
        self._termination_check_interval = termination_check_interval
        self._termination_patience = termination_patience
        self._transforms = {}
        self.dispersion_result = None
        self.map_result = None
        super().__init__(model, init_loc_fn=self._relaxed_init_loc_fn, prefix=prefix)

    def __call__(self, *args, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    stack.enter_context(plates[frame.name])
                result[name] = numpyro.sample(
                    name, self._get_proposal(name, self._proposal_params[name])
                )
        return result

    def _find_map(self, model_args, model_kwargs, rng_key):
        def objective(unconstrained, key):
            constrained = {
                name: self._transforms[name](value)
                for name, value in unconstrained.items()
            }
            seeded_model = handlers.seed(self.relaxed_model, rng_seed=key)
            log_target, _ = log_density(
                seeded_model, model_args, model_kwargs, constrained
            )
            return -log_target

        self.map_result = self._optimize(
            self._init_locs,
            self._map_max_steps,
            objective,
            self._map_optimizer,
            rng_key,
            self._map_tolerance,
        )
        return self.map_result.params

    def _fit_finite_parameters(
        self, model_args, model_kwargs, proposal_keys, parameters
    ):
        if not self._finite_distributions:
            return parameters
        constrained = {
            name: self._get_proposal(name, parameters[name]).sample(
                proposal_keys[name], (self._num_dispersion_particles,)
            )
            for name in self._map_locs
        }
        parameters = parameters.copy()
        for name in self._finite_distributions:
            categories, low, map_value, valid = self._get_finite_metadata(name)
            logits = jnp.full(jnp.shape(map_value) + (len(categories),), -jnp.inf)
            indices = product(*(range(size) for size in jnp.shape(map_value)))
            for index in indices:
                low_value = low[index]
                for category in range(len(categories)):
                    if not bool(valid[index + (category,)]):
                        continue
                    candidate_values = constrained.copy()
                    candidate_values[name] = (
                        constrained[name]
                        .at[(slice(None),) + index]
                        .set(low_value + category)
                    )

                    def particle_log_target(particle_values):
                        seeded_model = handlers.seed(self.model, rng_seed=random.key(0))
                        log_target, _ = log_density(
                            seeded_model, model_args, model_kwargs, particle_values
                        )
                        return log_target

                    expected_log_target = jnp.mean(
                        lax.map(particle_log_target, candidate_values)
                    )
                    logits = logits.at[index + (category,)].set(expected_log_target)
            parameters[name] = {"logits": logits}
            proposal = self._get_proposal(name, parameters[name])
            constrained[name] = proposal.sample(
                proposal_keys[name], (self._num_dispersion_particles,)
            )
        return parameters

    def _get_count_relaxation(self, distribution):
        if isinstance(distribution, SmoothedCount):
            return distribution
        base_dist = getattr(distribution, "base_dist", None)
        if base_dist is None:
            return None
        return self._get_count_relaxation(base_dist)

    def _get_finite_metadata(self, name):
        base_dist = self._finite_distributions[name]
        map_value = self._transforms[name](self._map_locs[name])
        num_categories = self._smooth_transforms[name]._M
        if isinstance(base_dist, dist.DiscreteUniform):
            high = jnp.broadcast_to(base_dist.high, jnp.shape(map_value))
            low = jnp.broadcast_to(base_dist.low, jnp.shape(map_value))
        else:
            low = jnp.zeros_like(map_value, dtype=int)
            if isinstance(base_dist, (dist.BinomialLogits, dist.BinomialProbs)):
                high = jnp.broadcast_to(base_dist.total_count, jnp.shape(map_value))
            else:
                high = low + num_categories - 1
        categories = jnp.arange(num_categories)
        valid = categories <= (high - low)[..., None]
        return categories, low, map_value, valid

    def _get_proposal(self, name, parameters, *, relaxed=False):
        event_dim = self._event_dims[name]
        if name in self._count_relaxations:
            base = dist.GammaCount(
                concentration=jnp.exp(parameters["log_concentration"]),
                rate=jnp.exp(parameters["log_rate"]),
            )
            if relaxed:
                target = self._count_relaxations[name]
                base = SmoothedDiscrete(
                    base,
                    self._discrete_temperature,
                    anchor=target.anchor,
                    max_count=target.max_count,
                    width=target.width,
                )
            return base.to_event(event_dim)
        if name in self._finite_distributions:
            if relaxed:
                raise ValueError(
                    "Finite proposals use exact Categorical coordinate updates."
                )
            _, low, _, valid = self._get_finite_metadata(name)
            logits = jnp.where(valid, parameters["logits"], -jnp.inf)
            base = dist.Categorical(logits=logits)
            if isinstance(self._finite_distributions[name], dist.DiscreteUniform):
                base = _ShiftedCategorical(logits, low)
            return base.to_event(event_dim)
        scale = jnp.exp(parameters["log_scale"])
        base = dist.Normal(self._map_locs[name], scale).to_event(event_dim)
        return dist.TransformedDistribution(base, self._transforms[name])

    def _get_smooth_transform(self, distribution):
        if isinstance(distribution, dist.TransformedDistribution):
            for transform in distribution.transforms:
                if isinstance(transform, SmoothICDFTransform):
                    return transform
        base_dist = getattr(distribution, "base_dist", None)
        if base_dist is None:
            return None
        return self._get_smooth_transform(base_dist)

    def _initial_proposal_params(self):
        parameters = {}
        for name, map_loc in self._map_locs.items():
            if name in self._count_relaxations:
                map_value = self._transforms[name](map_loc)
                dtype = jnp.result_type(map_value, float)
                minimum_rate = jnp.asarray(1e-3, dtype=dtype)
                parameters[name] = {
                    "log_concentration": jnp.zeros((), dtype=dtype),
                    "log_rate": jnp.log(jnp.maximum(map_value, minimum_rate)),
                }
            elif name in self._finite_distributions:
                categories, low, map_value, valid = self._get_finite_metadata(name)
                map_category = jnp.clip(jnp.round(map_value) - low, 0, categories[-1])
                logits = -jnp.square(categories - map_category[..., None])
                parameters[name] = {"logits": jnp.where(valid, logits, 0.0)}
            else:
                parameters[name] = {
                    "log_scale": jnp.asarray(jnp.log(self._init_dispersion))
                }
        return parameters

    def _optimize(
        self,
        initial_parameters,
        max_steps,
        objective,
        optimizer,
        rng_key,
        tolerance,
    ):
        best_loss = math.inf
        converged = False
        loss_chunks = []
        num_steps = 0
        optimizer_state = optimizer.init(initial_parameters)
        stalled_checks = 0
        step_keys = random.split(rng_key, max_steps)

        def step(state, key):
            def loss_fn(parameters):
                return objective(parameters, key), None

            (loss, _), state = optimizer.eval_and_stable_update(loss_fn, state)
            return state, loss

        while num_steps < max_steps:
            chunk_size = min(
                self._termination_check_interval,
                max_steps - num_steps,
            )
            optimizer_state, losses = lax.scan(
                step,
                optimizer_state,
                step_keys[num_steps : num_steps + chunk_size],
            )
            loss_chunks.append(losses)
            num_steps += chunk_size

            current_loss = float(jnp.nanmean(losses))
            loss_scale = tolerance * (1.0 + abs(best_loss))
            if math.isfinite(current_loss) and (
                not math.isfinite(best_loss) or current_loss < best_loss - loss_scale
            ):
                best_loss = current_loss
                stalled_checks = 0
            else:
                stalled_checks += 1
            if stalled_checks >= self._termination_patience:
                converged = math.isfinite(current_loss)
                break

        return _OptimizationResult(
            converged=converged,
            losses=jnp.concatenate(loss_chunks),
            num_steps=num_steps,
            params=optimizer.get_params(optimizer_state),
        )

    def _proposal_objective(self, model_args, model_kwargs, parameters, rng_key):
        constrained = {}
        log_q = jnp.zeros(self._num_dispersion_particles)
        model_key, proposal_key = random.split(rng_key)
        proposal_keys = {
            name: key
            for name, key in zip(
                self._map_locs,
                random.split(proposal_key, len(self._map_locs)),
            )
        }
        for name in self._map_locs:
            if name in self._finite_distributions:
                map_value = self._transforms[name](self._map_locs[name])
                constrained[name] = jnp.broadcast_to(
                    map_value,
                    (self._num_dispersion_particles,) + jnp.shape(map_value),
                )
                continue
            proposal = self._get_proposal(name, parameters[name], relaxed=True)
            constrained[name] = proposal.sample(
                proposal_keys[name], (self._num_dispersion_particles,)
            )
            site_log_q = proposal.log_prob(constrained[name])
            log_q = log_q + jnp.reshape(
                site_log_q, (self._num_dispersion_particles, -1)
            ).sum(-1)

        def particle_objective(particle):
            particle_values, particle_log_q = particle
            seeded_model = handlers.seed(self.relaxed_model, rng_seed=model_key)
            log_target, _ = log_density(
                seeded_model, model_args, model_kwargs, particle_values
            )
            return particle_log_q - log_target

        return jnp.mean(lax.map(particle_objective, (constrained, log_q)))

    def _relaxed_init_loc_fn(self, site=None):
        if site is None:
            return partial(self._relaxed_init_loc_fn)
        if site["type"] == "sample" and not site["is_observed"]:
            smooth_transform = self._get_smooth_transform(site["fn"])
            if smooth_transform is not None:
                sample_shape = site["kwargs"].get("sample_shape") or ()
                unit_value = jnp.full(sample_shape + site["fn"].shape(), 0.5)
                return smooth_transform(unit_value)
        return self._base_init_loc_fn(site)

    def _setup_prototype(self, *args, **kwargs):
        original_model = self.model
        self.model = self.relaxed_model
        try:
            super()._setup_prototype(*args, **kwargs)
        finally:
            self.model = original_model

        self._count_relaxations = {}
        self._event_dims = {}
        self._finite_distributions = {}
        self._smooth_transforms = {}
        self._support_ids = {}
        self._supports = []
        self._transforms = {}
        prototype_trace = self.prototype_trace
        assert prototype_trace is not None
        for name, site in prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue
            for frame in site["cond_indep_stack"]:
                if frame.size != self._prototype_frame_full_sizes[frame.name]:
                    raise NotImplementedError(
                        "AutoMAPProposal does not support data subsampling."
                    )

            count_relaxation = self._get_count_relaxation(site["fn"])
            smooth_transform = self._get_smooth_transform(site["fn"])
            if smooth_transform is None:
                with helpful_support_errors(site):
                    transform = biject_to(site["fn"].support)
            else:
                transform = ComposeTransform(
                    [biject_to(constraints.unit_interval), smooth_transform]
                )
                self._finite_distributions[name] = smooth_transform.base_dist
                self._init_locs[name] = transform.inv(self._init_locs[name])
                self._smooth_transforms[name] = smooth_transform
            if count_relaxation is not None:
                self._count_relaxations[name] = count_relaxation
            event_dim = (
                site["fn"].event_dim
                + jnp.ndim(self._init_locs[name])
                - jnp.ndim(site["value"])
            )
            support_id = next(
                (
                    index
                    for index, support in enumerate(self._supports)
                    if site["fn"].support.eq(support, static=True)
                ),
                len(self._supports),
            )
            if support_id == len(self._supports):
                self._supports.append(site["fn"].support)

            self._event_dims[name] = event_dim
            self._support_ids[name] = support_id
            self._transforms[name] = transform

        if not self._event_dims:
            raise RuntimeError("AutoMAPProposal found no latent variables.")

        optimizer_key = numpyro.prng_key()
        assert optimizer_key is not None
        finite_key, map_key, proposal_key = random.split(optimizer_key, 3)
        with handlers.block():
            self._map_locs = self._find_map(args, kwargs, map_key)
            proposal_keys = {
                name: key
                for name, key in zip(
                    self._map_locs,
                    random.split(finite_key, len(self._map_locs)),
                )
            }
            initial_parameters = self._initial_proposal_params()
            optimizable_parameters = {
                name: parameters
                for name, parameters in initial_parameters.items()
                if name not in self._finite_distributions
            }
            if optimizable_parameters:

                def objective(parameters, key):
                    return self._proposal_objective(args, kwargs, parameters, key)

                self.dispersion_result = self._optimize(
                    optimizable_parameters,
                    self._proposal_max_steps,
                    objective,
                    self._proposal_optimizer,
                    proposal_key,
                    self._proposal_tolerance,
                )
                optimized_parameters = self.dispersion_result.params
            else:
                self.dispersion_result = None
                optimized_parameters = {}
            self._proposal_params = initial_parameters.copy()
            self._proposal_params.update(optimized_parameters)
            self._proposal_params = self._fit_finite_parameters(
                args, kwargs, proposal_keys, self._proposal_params
            )
            self._dispersions = {
                name: jnp.exp(
                    parameters[
                        "log_concentration"
                        if name in self._count_relaxations
                        else "log_scale"
                    ]
                )
                for name, parameters in self._proposal_params.items()
                if name not in self._finite_distributions
            }

    def find_map(self, rng_key, *args, **kwargs):
        """Fit the proposal and return its constrained relaxed-MAP centers."""
        with handlers.block(), handlers.seed(rng_seed=rng_key):
            self._setup_prototype(*args, **kwargs)
        return {
            name: self._transforms[name](value)
            for name, value in self._map_locs.items()
        }

    def relaxed_model(self, *args, **kwargs):
        """Evaluate the DSGD-relaxed target at the configured temperature."""
        return self._dsgd_model(self._discrete_temperature, *args, **kwargs)

    def sample_posterior(self, rng_key, params=None, *args, sample_shape=(), **kwargs):
        """Fit once for the supplied inputs, then draw exact proposal samples."""
        setup_key, sample_key = random.split(rng_key)
        with handlers.block(), handlers.seed(rng_seed=setup_key):
            self._setup_prototype(*args, **kwargs)

        names = tuple(self._map_locs)
        sample_keys = random.split(sample_key, len(names))
        samples = {
            name: self._get_proposal(name, self._proposal_params[name]).sample(
                key, sample_shape
            )
            for name, key in zip(names, sample_keys)
        }
        prototype_trace = self.prototype_trace
        assert prototype_trace is not None
        deterministic_sites = [
            name
            for name, site in prototype_trace.items()
            if site["type"] == "deterministic"
        ]
        if deterministic_sites:
            predictive = Predictive(
                model=self.model,
                posterior_samples=samples,
                return_sites=deterministic_sites,
                batch_ndims=len(sample_shape),
            )
            samples.update(predictive(sample_key, *args, **kwargs))
        return samples
