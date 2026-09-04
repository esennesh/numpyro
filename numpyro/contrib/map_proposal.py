# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""A self-fitting MAP-centered proposal distribution."""

from contextlib import ExitStack
from functools import partial
from itertools import product
import math
from typing import Any, NamedTuple

from jax import jit, lax, random
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

__all__ = ["AutoMAPProposal", "MAPProposalResult"]


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


class _MinimizeOptions(NamedTuple):
    check_interval: int
    max_steps: int
    patience: int
    tolerance: float


class _OptimizationResult(NamedTuple):
    converged: bool
    losses: jnp.ndarray
    num_steps: int
    params: Any


class MAPProposalResult(NamedTuple):
    """Fitted state and diagnostics returned by :meth:`AutoMAPProposal.fit`.

    ``map_estimate`` contains constrained relaxed-MAP values, while ``map_locs``
    contains their unconstrained representations. ``proposal_params`` and
    ``map_locs`` are sufficient to reconstruct the fitted proposal. The two
    optimization results contain convergence flags, loss histories, and step
    counts; ``proposal_result`` is ``None`` for finite-discrete-only models.
    """

    map_estimate: dict
    map_locs: dict
    map_result: _OptimizationResult
    proposal_params: dict
    proposal_result: _OptimizationResult | None


def _minimize(args, initial_parameters, objective, optimizer, options, rng_key):
    best_loss = math.inf
    converged = False
    loss_chunks = []
    num_steps = 0
    optimizer_state = optimizer.init(initial_parameters)
    stalled_checks = 0
    step_keys = random.split(rng_key, options.max_steps)

    while num_steps < options.max_steps:
        chunk_size = min(
            options.check_interval,
            options.max_steps - num_steps,
        )
        optimizer_state, losses = _run_optimization_chunk(
            args,
            objective,
            optimizer,
            optimizer_state,
            step_keys[num_steps : num_steps + chunk_size],
        )
        loss_chunks.append(losses)
        num_steps += chunk_size

        current_loss = float(jnp.nanmean(losses))
        loss_scale = (
            options.tolerance * (1.0 + abs(best_loss))
            if math.isfinite(best_loss)
            else 0.0
        )
        if math.isfinite(current_loss) and (
            not math.isfinite(best_loss) or current_loss < best_loss - loss_scale
        ):
            best_loss = current_loss
            stalled_checks = 0
        else:
            stalled_checks += 1
        if stalled_checks >= options.patience:
            converged = math.isfinite(current_loss)
            break

    return _OptimizationResult(
        converged=converged,
        losses=jnp.concatenate(loss_chunks),
        num_steps=num_steps,
        params=optimizer.get_params(optimizer_state),
    )


@partial(jit, static_argnums=(1, 2))
def _run_optimization_chunk(
    args,
    objective,
    optimizer,
    optimizer_state,
    step_keys,
):
    def step(state, key):
        def loss_fn(parameters):
            return objective(parameters, args, key), None

        (loss, _), state = optimizer.eval_and_stable_update(loss_fn, state)
        return state, loss

    return lax.scan(step, optimizer_state, step_keys)


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
    EM, rather than for optimization by :class:`~numpyro.infer.svi.SVI`. Call
    :meth:`fit` to fit a proposal for particular model ``*args, **kwargs``.
    Subsequent calls and calls to :meth:`sample_posterior` only sample from that
    fitted proposal; they do not optimize it again. Fitting first computes the
    joint MAP estimate of the DSGD-smoothed target,

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
        self._fit_result = None
        self._init_dispersion = init_dispersion
        self._map_locs = {}
        self._map_optimizer = _as_iterative_optimizer(
            Adam(step_size=0.01) if map_optimizer is None else map_optimizer
        )
        self._map_options = _MinimizeOptions(
            check_interval=termination_check_interval,
            max_steps=map_max_steps,
            patience=termination_patience,
            tolerance=map_tolerance,
        )
        self._num_dispersion_particles = num_dispersion_particles
        self._proposal_optimizer = _as_iterative_optimizer(
            Adam(step_size=0.01) if proposal_optimizer is None else proposal_optimizer
        )
        self._proposal_options = _MinimizeOptions(
            check_interval=termination_check_interval,
            max_steps=proposal_max_steps,
            patience=termination_patience,
            tolerance=proposal_tolerance,
        )
        self._proposal_params = {}
        self._smooth_transforms = {}
        self._support_ids = {}
        self._supports = []
        self._transforms = {}
        self.dispersion_result = None
        self.map_result = None
        super().__init__(model, init_loc_fn=self._relaxed_init_loc_fn, prefix=prefix)

    def __call__(self, *args, **kwargs):
        if self._fit_result is None:
            raise RuntimeError("Call AutoMAPProposal.fit() before sampling.")

        plates = self._create_plates(*args, **kwargs)
        result = {}
        for name, site in self.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    stack.enter_context(plates[frame.name])
                result[name] = numpyro.sample(
                    name,
                    self._get_proposal(
                        name,
                        self._fit_result.proposal_params[name],
                        map_locs=self._fit_result.map_locs,
                    ),
                )
        return result

    def _fit_finite_parameters(
        self, map_locs, model_args, model_kwargs, parameters, proposal_keys
    ):
        if not self._finite_distributions:
            return parameters
        constrained = {
            name: self._get_proposal(
                name,
                parameters[name],
                map_locs=map_locs,
            ).sample(proposal_keys[name], (self._num_dispersion_particles,))
            for name in map_locs
        }
        parameters = parameters.copy()
        for name in self._finite_distributions:
            categories, low, map_value, valid = self._get_finite_metadata(
                name, map_locs
            )
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
            proposal = self._get_proposal(
                name,
                parameters[name],
                map_locs=map_locs,
            )
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

    def _get_finite_metadata(self, name, map_locs=None):
        map_locs = self._map_locs if map_locs is None else map_locs
        base_dist = self._finite_distributions[name]
        map_value = self._transforms[name](map_locs[name])
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

    def _get_proposal(self, name, parameters, *, map_locs=None, relaxed=False):
        map_locs = self._map_locs if map_locs is None else map_locs
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
            _, low, _, valid = self._get_finite_metadata(name, map_locs)
            logits = jnp.where(valid, parameters["logits"], -jnp.inf)
            base = dist.Categorical(logits=logits)
            if isinstance(self._finite_distributions[name], dist.DiscreteUniform):
                base = _ShiftedCategorical(logits, low)
            return base.to_event(event_dim)
        scale = jnp.exp(parameters["log_scale"])
        base = dist.Normal(map_locs[name], scale).to_event(event_dim)
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

    def _initial_proposal_params(self, map_locs):
        parameters = {}
        for name, map_loc in map_locs.items():
            if name in self._count_relaxations:
                map_value = self._transforms[name](map_loc)
                dtype = jnp.result_type(map_value, float)
                minimum_rate = jnp.asarray(1e-3, dtype=dtype)
                parameters[name] = {
                    "log_concentration": jnp.zeros((), dtype=dtype),
                    "log_rate": jnp.log(jnp.maximum(map_value, minimum_rate)),
                }
            elif name in self._finite_distributions:
                categories, low, map_value, valid = self._get_finite_metadata(
                    name, map_locs
                )
                map_category = jnp.clip(jnp.round(map_value) - low, 0, categories[-1])
                logits = -jnp.square(categories - map_category[..., None])
                parameters[name] = {"logits": jnp.where(valid, logits, 0.0)}
            else:
                parameters[name] = {
                    "log_scale": jnp.asarray(jnp.log(self._init_dispersion))
                }
        return parameters

    def _map_objective(self, unconstrained, objective_args, rng_key):
        model_args, model_kwargs = objective_args
        constrained = {
            name: self._transforms[name](value) for name, value in unconstrained.items()
        }
        seeded_model = handlers.seed(self.relaxed_model, rng_seed=rng_key)
        log_target, _ = log_density(
            seeded_model,
            model_args,
            model_kwargs,
            constrained,
        )
        return -log_target

    def _proposal_objective(self, parameters, objective_args, rng_key):
        map_locs, model_args, model_kwargs = objective_args
        constrained = {}
        log_q = jnp.zeros(self._num_dispersion_particles)
        model_key, proposal_key = random.split(rng_key)
        proposal_keys = {
            name: key
            for name, key in zip(
                map_locs,
                random.split(proposal_key, len(map_locs)),
            )
        }
        for name in map_locs:
            if name in self._finite_distributions:
                map_value = self._transforms[name](map_locs[name])
                constrained[name] = jnp.broadcast_to(
                    map_value,
                    (self._num_dispersion_particles,) + jnp.shape(map_value),
                )
                continue
            proposal = self._get_proposal(
                name,
                parameters[name],
                map_locs=map_locs,
                relaxed=True,
            )
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

        self._dispersions = {}
        self.dispersion_result = None
        self._fit_result = None
        self._map_locs = {}
        self.map_result = None
        self._proposal_params = {}

    def fit(self, rng_key, *args, **kwargs):
        """Fit and store a MAP-centered proposal for the supplied model inputs.

        Prototype discovery runs at most once. The numerical model inputs are
        then passed dynamically to JIT-compiled optimizer chunks. Calling this
        method again reuses the prototype but refits both optimization phases.

        :return: The reusable fitted proposal state and optimization diagnostics.
        :rtype: MAPProposalResult
        """
        optimizer_key, prototype_key = random.split(rng_key)
        if self.prototype_trace is None:
            with handlers.block(), handlers.seed(rng_seed=prototype_key):
                self._setup_prototype(*args, **kwargs)

        self.dispersion_result = None
        self._fit_result = None
        self.map_result = None
        finite_key, map_key, proposal_key = random.split(optimizer_key, 3)
        with handlers.block():
            map_result = _minimize(
                (args, kwargs),
                self._init_locs,
                self._map_objective,
                self._map_optimizer,
                self._map_options,
                map_key,
            )
            map_locs = map_result.params
            proposal_keys = {
                name: key
                for name, key in zip(
                    map_locs,
                    random.split(finite_key, len(map_locs)),
                )
            }
            initial_parameters = self._initial_proposal_params(map_locs)
            optimizable_parameters = {
                name: parameters
                for name, parameters in initial_parameters.items()
                if name not in self._finite_distributions
            }
            if optimizable_parameters:
                proposal_result = _minimize(
                    (map_locs, args, kwargs),
                    optimizable_parameters,
                    self._proposal_objective,
                    self._proposal_optimizer,
                    self._proposal_options,
                    proposal_key,
                )
                optimized_parameters = proposal_result.params
            else:
                optimized_parameters = {}
                proposal_result = None
            proposal_params = initial_parameters.copy()
            proposal_params.update(optimized_parameters)
            proposal_params = self._fit_finite_parameters(
                map_locs,
                args,
                kwargs,
                proposal_params,
                proposal_keys,
            )
            dispersions = {
                name: jnp.exp(
                    parameters[
                        "log_concentration"
                        if name in self._count_relaxations
                        else "log_scale"
                    ]
                )
                for name, parameters in proposal_params.items()
                if name not in self._finite_distributions
            }

        map_estimate = {
            name: self._transforms[name](value) for name, value in map_locs.items()
        }
        fit_result = MAPProposalResult(
            map_estimate=map_estimate,
            map_locs=map_locs,
            map_result=map_result,
            proposal_params=proposal_params,
            proposal_result=proposal_result,
        )
        self._dispersions = dispersions
        self.dispersion_result = proposal_result
        self._fit_result = fit_result
        self._map_locs = map_locs
        self.map_result = map_result
        self._proposal_params = proposal_params
        return fit_result

    def relaxed_model(self, *args, **kwargs):
        """Evaluate the DSGD-relaxed target at the configured temperature."""
        return self._dsgd_model(self._discrete_temperature, *args, **kwargs)

    def sample_posterior(
        self, rng_key, fit_result=None, *args, sample_shape=(), **kwargs
    ):
        """Draw exact samples from a proposal previously fitted by :meth:`fit`.

        Pass the returned :class:`MAPProposalResult` as ``fit_result``, or omit
        it to use the most recent fit. ``sample_shape`` controls the arbitrary
        batch of proposal draws. No optimization occurs here.
        """
        if fit_result is None:
            fit_result = self._fit_result
        if fit_result is None:
            raise RuntimeError("Call AutoMAPProposal.fit() before sampling.")
        if not isinstance(fit_result, MAPProposalResult):
            raise TypeError("fit_result must be the result returned by fit().")

        names = tuple(fit_result.map_locs)
        predictive_key, sample_key = random.split(rng_key)
        sample_keys = random.split(sample_key, len(names))
        samples = {
            name: self._get_proposal(
                name,
                fit_result.proposal_params[name],
                map_locs=fit_result.map_locs,
            ).sample(key, sample_shape)
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
            samples.update(predictive(predictive_key, *args, **kwargs))
        return samples
