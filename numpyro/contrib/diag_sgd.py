# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Diagonalisation SGD (DSGD) program transformation for NumPyro models.

Based on: Wagner, Khajwal & Ong, "Diagonalisation SGD: Fast & Convergent SGD
for Non-Differentiable Models via Reparameterisation and Smoothing",
AISTATS 2024.

The transformation replaces discrete sample sites with smoothed reparameterisations
using smooth inverse-CDF transforms parameterised by a smoothing temperature η.
As η → 0 the smooth objective converges to the original objective. Theorem 5.6 of
the paper gives a concrete η-schedule under which the DSGD gradient estimator
converges almost surely to stationary points.

Usage example::

    import numpyro
    import numpyro.distributions as dist
    from numpyro.contrib.diag_sgd import dsgd, count_layers, eta_schedule

    def model(data):
        p = numpyro.sample("p", dist.Beta(1., 1.))
        z = numpyro.sample("z", dist.Bernoulli(p))   # discrete site
        numpyro.sample("obs", dist.Normal(z, 1.), obs=data)

    smoothed_model = dsgd(model)

    ell = count_layers(model, data)
    schedule = eta_schedule(K=1000, ell=ell, eta_final=0.01)

    # Inside the training loop at step k:
    eta = schedule[k]
    # Use smoothed_model(eta, data) with your SVI / ELBO routine.
"""

import copy
import functools
import math
import threading
from typing import Optional

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc, betaln, gammainc, gammaincc, logit, ndtri

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.conjugate import GammaPoisson
from numpyro.distributions.discrete import (
    BernoulliLogits,
    BernoulliProbs,
    BinomialLogits,
    BinomialProbs,
    CategoricalLogits,
    CategoricalProbs,
    DiscreteUniform,
    GammaCount,
    GeometricLogits,
    GeometricProbs,
    Poisson,
    ZeroInflatedProbs,
)
from numpyro.distributions.distribution import (
    ExpandedDistribution,
    Independent,
    TransformedDistribution,
)
from numpyro.distributions.transforms import Transform
from numpyro.primitives import Messenger

__all__ = [
    "count_layers",
    "dsgd",
    "eta_schedule",
    "smooth_cond",
    "smooth_icdf",
    "smooth_switch",
    "adaptive_relaxed_count",
    "anchored_relaxed_count",
    "count_anchor_saturates",
    "SmoothedCount",
    "SmoothedDiscrete",
    "SmoothICDFTransform",
    "StraightThroughSmoothed",
]

# ---------------------------------------------------------------------------
# Thread-local depth counter for smooth_cond / smooth_switch nesting.
# Tracks Python-trace-time nesting depth; reset by count_layers().
# ---------------------------------------------------------------------------

_depth_state = threading.local()


def _current_depth() -> int:
    return getattr(_depth_state, "depth", 0)


def _update_max_depth(d: int) -> None:
    if d > getattr(_depth_state, "max_depth", 0):
        _depth_state.max_depth = d


# ---------------------------------------------------------------------------
# Core smoothing primitive
# ---------------------------------------------------------------------------


def smooth_heaviside(x, eta):
    """Sigmoid-smoothed Heaviside: σ_η(x) = sigmoid(x / η)."""
    return jax.nn.sigmoid(x / eta)


# ---------------------------------------------------------------------------
# Smooth iCDF: Q^η_D(u) = Σ_{k=0}^{K-2} σ_η(u − F(k))
# ---------------------------------------------------------------------------


def _resolve_max_support(d, user_max_support: Optional[int]) -> int:
    """Return the number of distinct support values to include.

    The grid inverse-CDF only represents finite-support families exactly;
    unbounded families use the anchored index-space relaxation instead.
    """
    if user_max_support is not None:
        return int(user_max_support)

    if isinstance(d, (BernoulliProbs, BernoulliLogits)):
        return 2

    if isinstance(d, (CategoricalProbs, CategoricalLogits)):
        return int(d.probs.shape[-1])

    if isinstance(d, (BinomialProbs, BinomialLogits)):
        tc = d.total_count
        if hasattr(tc, "shape") and tc.shape != ():
            return int(jnp.max(tc)) + 1
        return int(tc) + 1

    if isinstance(d, DiscreteUniform):
        hi = d.high
        lo = d.low
        if hasattr(hi, "shape") and hi.shape != ():
            return int(jnp.max(hi - lo)) + 1
        return int(hi) - int(lo) + 1

    raise ValueError(
        f"numpyro.contrib.diag_sgd: the grid inverse-CDF supports only "
        f"finite-support families; {type(d).__name__} is unbounded — it uses the "
        f"anchored index-space relaxation (omit max_support)."
    )


def _support_lower_bound(d):
    """Lower bound of the support (0 for almost all discrete distributions)."""
    if isinstance(d, DiscreteUniform):
        return jnp.asarray(d.low, dtype=jnp.float32)
    return 0.0


def _compute_cdf_grid(d, M: int):
    """
    Return F(low), F(low+1), …, F(low+M−2) as an array of shape (*batch, M−1).

    F(k) = P(X ≤ k).  This is the CDF grid consumed by smooth_icdf.
    """
    ks = jnp.arange(M - 1, dtype=jnp.float32)  # (M-1,)

    # ---- fast paths --------------------------------------------------------

    if isinstance(d, (BernoulliProbs, BernoulliLogits)):
        # K = 2; one boundary F(0) = 1 − p
        return (1.0 - jnp.asarray(d.probs))[..., None]  # (*B, 1)

    if isinstance(d, (CategoricalProbs, CategoricalLogits)):
        # F(k) = cumsum(probs)[k] for k = 0 … K−2
        return jnp.cumsum(d.probs, axis=-1)[..., :-1]  # (*B, K−1)

    if isinstance(d, (BinomialProbs, BinomialLogits)):
        # F(k) = I_{1−p}(n−k, k+1)  (regularised incomplete beta)
        p = d.probs[..., None]  # (*B, 1)
        n = jnp.asarray(d.total_count, dtype=jnp.float32)[..., None]
        return betainc(n - ks, ks + 1.0, 1.0 - p)  # (*B, M−1)

    if isinstance(d, DiscreteUniform):
        # F(low + j) = (j + 1) / K;  K = high − low + 1
        n = jnp.asarray(d.high - d.low + 1, dtype=jnp.float32)
        # Expand for batch; ks broadcasts as (M−1,)
        return (ks + 1.0) / n[..., None]  # (*B, M−1)

    # Unbounded families (Poisson, Geometric, GammaCount, GammaPoisson /
    # NegativeBinomial, ZeroInflated) are intentionally not handled here: the
    # truncated grid has an O(M) saturated-tail bias for unbounded support. They
    # use the grid-free anchored index-space relaxed count
    # (anchored_relaxed_count) instead.
    raise ValueError(
        f"numpyro.contrib.diag_sgd: the grid inverse-CDF supports only "
        f"finite-support families; {type(d).__name__} is unbounded — it uses the "
        f"anchored index-space relaxation (omit max_support)."
    )


def smooth_icdf(d, u, eta, max_support: Optional[int] = None):
    """
    Smooth inverse-CDF approximation :math:`Q_{\\eta, D}(u)`.

    For a discrete distribution D on support {low, …, low+K−1}, returns

        :math:`Q_{\\eta}(u) = \\mathrm{low} + \\sum_{k=0}^{K-2} \\sigma_{\\eta}(u - Q(\\mathrm{low}+k))`

    where :math:`\\sigma_{\\eta}(x) = \\mathrm{sigmoid}(\\frac{x}{\\eta})` and
    Q is the CDF of D.  Supports only finite-support families; unbounded families
    use :func:`adaptive_relaxed_count` (index-space smoothing).

    :param d: a NumPyro finite-support discrete Distribution.
    :param u: uniform random variate(s) in (0, 1).
    :param float eta: smoothing temperature :math:`\\eta \\gt 0`.
    :param max_support: optional grid size override.
    :return: smooth sample(s), same shape as u.
    """
    M = _resolve_max_support(d, max_support)
    cdf_grid = _compute_cdf_grid(d, M)  # (*batch, M−1)
    low = _support_lower_bound(d)
    u_exp = u[..., None]  # (..., 1)
    return low + jnp.sum(smooth_heaviside(u_exp - cdf_grid, eta), axis=-1)


# ---------------------------------------------------------------------------
# Newton's method for inverse smooth iCDF
# ---------------------------------------------------------------------------


def _newton_inverse_icdf(cdf_grid, z_shifted, eta, M: int, n_steps: int = 15):
    """Solve Q^η(u) − low = z_shifted for u ∈ (0, 1) via Newton iterations."""
    u0 = jnp.clip(z_shifted / jnp.maximum(M - 1, 1), 1e-6, 1.0 - 1e-6)

    def step(u, _):
        s = jax.nn.sigmoid((u[..., None] - cdf_grid) / eta)  # (..., M−1)
        q = jnp.sum(s, axis=-1)
        dq = jnp.sum(s * (1.0 - s), axis=-1) / eta
        u_new = u - (q - z_shifted) / (dq + 1e-20)
        return jnp.clip(u_new, 1e-8, 1.0 - 1e-8), None

    u_final, _ = jax.lax.scan(step, u0, None, length=n_steps)
    return u_final


# ---------------------------------------------------------------------------
# SmoothICDFTransform
# ---------------------------------------------------------------------------


class SmoothICDFTransform(Transform):
    """
    A bijective Transform :math:`u \\mapsto Q_{\\eta, D}(u)` mapping
    Uniform(0,1) samples to smooth approximations of D-distributed samples.

    When wrapped in :class:`TransformedDistribution`, the log-prob method
    automatically returns the pushforward density :math:`-\\log |\\frac{\\partial Q_{\\eta}}{\\partial u}|`,
    which is the correct DSGD objective density.

    Intended for finite-support families; unbounded families use the grid-free
    :func:`adaptive_relaxed_count`.

    :param base_dist: a NumPyro finite-support discrete Distribution.
    :param float eta: smoothing temperature :math:`\\eta \\gt 0`.
    :param max_support: optional grid size override.
    """

    domain = constraints.unit_interval
    codomain = constraints.real
    sign = 1

    def __init__(self, base_dist, eta, max_support: Optional[int] = None):
        self.base_dist = base_dist
        self.eta = eta
        self._max_support_override = max_support

    @property
    def _M(self) -> int:
        return _resolve_max_support(self.base_dist, self._max_support_override)

    def _cdf_grid(self):
        return _compute_cdf_grid(self.base_dist, self._M)

    def __call__(self, u):
        cdf_grid = self._cdf_grid()  # (*batch, M−1)
        low = _support_lower_bound(self.base_dist)
        return low + jnp.sum(
            smooth_heaviside(u[..., None] - cdf_grid, self.eta), axis=-1
        )

    def _inverse(self, z):
        low = _support_lower_bound(self.base_dist)
        z_shifted = z - low

        # Closed form for Bernoulli (faster, no iteration)
        if isinstance(self.base_dist, (BernoulliProbs, BernoulliLogits)):
            boundary = 1.0 - self.base_dist.probs
            return self.eta * logit(jnp.clip(z_shifted, 1e-6, 1.0 - 1e-6)) + boundary

        return _newton_inverse_icdf(self._cdf_grid(), z_shifted, self.eta, self._M)

    def log_abs_det_jacobian(self, u, z, intermediates=None):
        r""":math:`\log\left|\frac{dQ_\eta}{du}\right|
        = \log\left(\frac1\eta \sum_k \sigma_\eta(u - Q(k))
        (1 - \sigma_\eta(u - Q(k)))\right)`."""
        cdf_grid = self._cdf_grid()
        s = jax.nn.sigmoid((u[..., None] - cdf_grid) / self.eta)  # (..., M−1)
        deriv = jnp.sum(s * (1.0 - s), axis=-1) / self.eta
        return jnp.log(jnp.maximum(deriv, 1e-45))

    def tree_flatten(self):
        return (self.base_dist, self.eta), (
            ("base_dist", "eta"),
            {"_max_support_override": self._max_support_override},
        )


# ---------------------------------------------------------------------------
# Adaptive index-space relaxed count for unbounded discrete distributions
#
# The grid SmoothICDFTransform smooths the quantile function in u-space,
#   Q^eta(u) = low + sum_{k<M} sigma_eta(u - F(k)),
# but needs a static truncation M and has an O(M) saturated-tail bias for
# unbounded support (each tail grid point adds ~sigma_eta(u - 1) > 0, so the sum
# does not converge in M).
#
# We instead use the *index-space* relaxation of Wagner et al., with a
# runtime-adaptive horizon (no truncation).  Writing a_k = sigma_eta(u - F(k)),
# the soft one-hot weights w_k = a_{k-1} - a_k give the normalised relaxed count
#   z(u) = sum_k k w_k / sum_k w_k,
# a genuine convex combination of the integer outcomes (see adaptive_relaxed_count).
# Its gradient is a bounded weighted sum of dF(k)/dtheta with no 1/density
# factor, so it stays low-variance across eta and keeps full signal as eta -> 0.
# The data-dependent horizon needs jax.lax.while_loop, which JAX will not
# reverse-differentiate, so the accumulation is wrapped in a custom_vjp whose
# backward takes the parameter gradient with forward-mode AD (jacfwd) -- cheap
# because the distribution parameters are low-dimensional and memory is O(1) in
# the horizon.
#
# Only the forward relaxed count is used (as the reparameterised sample in
# SmoothedCount / StraightThroughSmoothed); the density term is the analytic-
# continuation log-pmf (see SmoothedCount), so no inverse or log|dz/du| of this
# map is needed.
# ---------------------------------------------------------------------------


class _IdxSpec:
    """Closed-form, parameter-explicit pieces of an unbounded discrete family
    used by the adaptive index-space relaxation: ``log_pmf(k, params)`` (the
    log-pmf, analytically continued in ``k`` via ``gammaln`` and differentiable
    in ``params``) and the parameter pytree ``params`` (explicit so the
    accumulation can be forward-differentiated w.r.t. it).

    ``log_pmf`` rebuilds the distribution from ``params`` and defers to its
    ``log_prob``; ``params`` stays an explicit pytree so the accumulation's
    custom VJP keeps the tangent structure it expects."""

    __slots__ = ("log_pmf", "params")

    def __init__(self, log_pmf, params):
        self.log_pmf = log_pmf
        self.params = params


def _idx_spec(d) -> Optional["_IdxSpec"]:
    """Return an :class:`_IdxSpec` for supported unbounded families (Poisson,
    Geometric, GammaCount, GammaPoisson / NegativeBinomial and their
    zero-inflated wrappers), else ``None``. ``log_pmf(k, params)`` is the
    analytic continuation of the discrete log-pmf (valid for real ``k >= 0``):
    it is consumed with integer ``k`` inside the accumulation and with the
    continuous relaxed count as the :class:`SmoothedCount` density term."""
    ftype = jnp.result_type(float)

    if isinstance(d, Poisson):
        params = {"rate": jnp.asarray(d.rate, dtype=ftype)}

        def log_pmf(k, p):
            return _unvalidated_log_prob(Poisson(p["rate"]), k)

        return _IdxSpec(log_pmf, params)

    if isinstance(d, (GeometricProbs, GeometricLogits)):
        params = {"probs": jnp.asarray(d.probs, dtype=ftype)}

        def log_pmf(k, p):
            return _unvalidated_log_prob(GeometricProbs(p["probs"]), k)

        return _IdxSpec(log_pmf, params)

    if isinstance(d, GammaCount):
        params = {
            "concentration": jnp.asarray(d.concentration, dtype=ftype),
            "rate": jnp.asarray(d.rate, dtype=ftype),
        }

        def log_pmf(k, p):
            return _unvalidated_log_prob(GammaCount(p["concentration"], p["rate"]), k)

        return _IdxSpec(log_pmf, params)

    if isinstance(d, GammaPoisson):
        params = {
            "concentration": jnp.asarray(d.concentration, dtype=ftype),
            "rate": jnp.asarray(d.rate, dtype=ftype),
        }

        def log_pmf(k, p):
            return _unvalidated_log_prob(GammaPoisson(p["concentration"], p["rate"]), k)

        return _IdxSpec(log_pmf, params)

    if isinstance(d, ZeroInflatedProbs):
        # Zero-inflation adds a point mass ``gate`` at 0 to the base family.
        # Covers ZeroInflatedLogits / ZeroInflatedPoisson (subclasses).
        base = _idx_spec(d.base_dist)
        if base is None:
            return None
        params = {
            "gate": jnp.asarray(d.gate, dtype=ftype),
            "base": base.params,
        }

        def log_pmf(k, p):
            # Mixture form log[(1 - gate) p_base(k) + gate * 1[k == 0]].  ``bump``
            # is a smooth surrogate for the indicator 1[k == 0]: exact on the
            # integers (1 at 0, 0 at every k >= 1), so the accumulation and the
            # eta -> 0 limit are unchanged, but continuous in the relaxed count so
            # the density term is well-defined at non-integer z.
            bump = jnp.clip(1.0 - k, 0.0, 1.0)
            base_p = jnp.exp(base.log_pmf(k, p["base"]))
            mix = (1.0 - p["gate"]) * base_p + p["gate"] * bump
            return jnp.log(jnp.clip(mix, 1e-45, None))

        return _IdxSpec(log_pmf, params)

    return None


# Fixed-window controls for the anchored relaxed-count implementation.  The
# window is deliberately static so vmapped lanes do not inherit the runtime of
# the largest count in the batch.  These defaults can be overridden through the
# public wrappers; a future adaptive-width policy can build on the same API.
_COUNT_WINDOW = 256
_COUNT_MAX = 100_000
_COUNT_ANCHORS = ("binary", "cornish-fisher")


def _validate_count_anchor(anchor):
    if anchor not in _COUNT_ANCHORS:
        choices = ", ".join(repr(value) for value in _COUNT_ANCHORS)
        raise ValueError(f"anchor must be one of {choices}; got {anchor!r}")


@jax.custom_jvp
def _betainc_in_concentration(concentration, count, probability):
    """``betainc`` with the GammaPoisson concentration derivative."""
    return betainc(concentration, count, probability)


@_betainc_in_concentration.defjvp
def _betainc_in_concentration_jvp(primals, tangents):
    concentration, count, probability = primals
    concentration_dot, _count_dot, probability_dot = tangents
    dtype = jnp.result_type(concentration, probability)
    step = jnp.finfo(dtype).eps ** (1.0 / 3.0) * jnp.maximum(concentration, 1.0)
    lower = jnp.maximum(concentration - step, jnp.finfo(dtype).tiny)
    upper = concentration + step
    value = betainc(concentration, count, probability)
    concentration_derivative = (
        betainc(upper, count, probability) - betainc(lower, count, probability)
    ) / (upper - lower)
    probability_derivative = jnp.exp(
        (concentration - 1.0) * jnp.log(probability)
        + (count - 1.0) * jnp.log1p(-probability)
        - betaln(concentration, count)
    )
    tangent = (
        concentration_derivative * concentration_dot
        + probability_derivative * probability_dot
    )
    return value, tangent


@jax.custom_jvp
def _betainc_survival(count, concentration, probability):
    """``betainc`` in the survival orientation, with the concentration derivative.

    ``I_x(a, b) = 1 - I_{1-x}(b, a)``, so the GammaPoisson survival function is
    a ``betainc`` with the concentration in the *second* shape slot.  This is
    the mirror of :func:`_betainc_in_concentration`, which differentiates the
    first slot for the CDF orientation.
    """
    return betainc(count, concentration, probability)


@_betainc_survival.defjvp
def _betainc_survival_jvp(primals, tangents):
    count, concentration, probability = primals
    _count_dot, concentration_dot, probability_dot = tangents
    dtype = jnp.result_type(concentration, probability)
    step = jnp.finfo(dtype).eps ** (1.0 / 3.0) * jnp.maximum(concentration, 1.0)
    lower = jnp.maximum(concentration - step, jnp.finfo(dtype).tiny)
    upper = concentration + step
    value = betainc(count, concentration, probability)
    concentration_derivative = (
        betainc(count, upper, probability) - betainc(count, lower, probability)
    ) / (upper - lower)
    probability_derivative = jnp.exp(
        (count - 1.0) * jnp.log(probability)
        + (concentration - 1.0) * jnp.log1p(-probability)
        - betaln(count, concentration)
    )
    tangent = (
        concentration_derivative * concentration_dot
        + probability_derivative * probability_dot
    )
    return value, tangent


def _expand_count_parameter(value, target, batch_shape, *, trailing_ndims=0):
    """Align a parameter with leading sample and trailing recurrence axes."""
    value = jnp.broadcast_to(jnp.asarray(value), batch_shape)
    sample_ndims = jnp.ndim(target) - len(batch_shape) - trailing_ndims
    shape = (1,) * sample_ndims + tuple(batch_shape) + (1,) * trailing_ndims
    return jnp.reshape(value, shape)


def _count_cdf(base_dist, value):
    """CDF for every unbounded family supported by :func:`_idx_spec`."""
    value = jnp.floor(value)
    if isinstance(base_dist, Poisson):
        rate = _expand_count_parameter(base_dist.rate, value, base_dist.batch_shape)
        cdf = gammaincc(value + 1.0, rate)
    elif isinstance(base_dist, (GeometricProbs, GeometricLogits)):
        probs = _expand_count_parameter(base_dist.probs, value, base_dist.batch_shape)
        cdf = -jnp.expm1((value + 1.0) * jnp.log1p(-probs))
    elif isinstance(base_dist, GammaCount):
        concentration = _expand_count_parameter(
            base_dist.concentration, value, base_dist.batch_shape
        )
        rate = _expand_count_parameter(base_dist.rate, value, base_dist.batch_shape)
        cdf = gammaincc(concentration * (value + 1.0), concentration * rate)
    elif isinstance(base_dist, GammaPoisson):
        concentration = _expand_count_parameter(
            base_dist.concentration, value, base_dist.batch_shape
        )
        rate = _expand_count_parameter(base_dist.rate, value, base_dist.batch_shape)
        cdf = _betainc_in_concentration(concentration, value + 1.0, rate / (rate + 1.0))
    elif isinstance(base_dist, ZeroInflatedProbs):
        gate = _expand_count_parameter(base_dist.gate, value, base_dist.batch_shape)
        cdf = gate + (1.0 - gate) * _count_cdf(base_dist.base_dist, value)
    else:
        raise ValueError(
            f"anchored_relaxed_count: unsupported distribution "
            f"{type(base_dist).__name__}."
        )
    return jnp.where(value < 0, 0.0, cdf)


def _without_validation(value):
    """Shallow copy of a distribution tree with support validation switched off.

    ``_validate_args`` is per-instance, and a wrapper such as
    :class:`ZeroInflatedProbs` calls ``log_prob`` on the distribution it wraps,
    so switching it off on the outermost object alone is not enough.  Copy
    rather than mutate: the caller's distribution is left untouched, and
    nothing global is toggled, which matters under ``jit``/``vmap`` and across
    threads.
    """
    if not isinstance(value, dist.Distribution):
        return value
    result = copy.copy(value)
    result._validate_args = False
    nested = [
        (name, attribute)
        for name, attribute in vars(result).items()
        if isinstance(attribute, dist.Distribution)
    ]
    for name, attribute in nested:
        setattr(result, name, _without_validation(attribute))
    return result


def _unvalidated_log_prob(base_dist, value):
    """Call ``base_dist.log_prob`` with support validation bypassed.

    Every supported family already writes ``log_prob`` as the analytic
    continuation of its pmf (``gammaln``/``betaln`` in the count, not a table
    lookup), which is exactly what the relaxation needs.  What is in the way is
    the ``validate_sample`` decorator: it masks non-integer arguments to
    ``-inf``, and the relaxed count is real-valued by construction.
    """
    return _without_validation(base_dist).log_prob(value)


def _count_sf(base_dist, value):
    r"""Survival function :math:`\Pr(X > \lfloor value \rfloor)`.

    Evaluated directly rather than as ``1 - cdf``.  Once the CDF reaches 1.0 it
    is not merely imprecise, it is *constant*: every parameter derivative
    through it is zero, and the relaxation quietly loses that much of its
    gradient.  The survival function stays accurate to full relative precision
    in exactly that regime, which is why the smoothing step picks whichever of
    the two tails is the smaller one.
    """
    value = jnp.floor(value)
    if isinstance(base_dist, Poisson):
        rate = _expand_count_parameter(base_dist.rate, value, base_dist.batch_shape)
        sf = gammainc(value + 1.0, rate)
    elif isinstance(base_dist, (GeometricProbs, GeometricLogits)):
        probs = _expand_count_parameter(base_dist.probs, value, base_dist.batch_shape)
        sf = jnp.exp((value + 1.0) * jnp.log1p(-probs))
    elif isinstance(base_dist, GammaCount):
        concentration = _expand_count_parameter(
            base_dist.concentration, value, base_dist.batch_shape
        )
        rate = _expand_count_parameter(base_dist.rate, value, base_dist.batch_shape)
        # GammaCount.cdf is the upper regularised tail, so the survival
        # function is the lower one at the same arguments -- no subtraction.
        sf = gammainc(concentration * (value + 1.0), concentration * rate)
    elif isinstance(base_dist, GammaPoisson):
        concentration = _expand_count_parameter(
            base_dist.concentration, value, base_dist.batch_shape
        )
        rate = _expand_count_parameter(base_dist.rate, value, base_dist.batch_shape)
        sf = _betainc_survival(value + 1.0, concentration, 1.0 / (rate + 1.0))
    elif isinstance(base_dist, ZeroInflatedProbs):
        gate = _expand_count_parameter(base_dist.gate, value, base_dist.batch_shape)
        sf = (1.0 - gate) * _count_sf(base_dist.base_dist, value)
    else:
        raise ValueError(
            f"anchored_relaxed_count: unsupported distribution "
            f"{type(base_dist).__name__}."
        )
    return jnp.where(value < 0, 1.0, sf)


def _count_log_pmf(base_dist, value, trailing_ndims=0):
    """Log pmf of a supported count family, broadcast over recurrence axes.

    Delegates to the distribution's own ``log_prob`` rather than restating each
    formula here.  ``log_prob`` broadcasts its parameters against the *trailing*
    axes of ``value``, so the recurrence axes are rotated to the front for the
    call and rotated back afterwards; that leaves the parameters aligned with
    the batch axes without materialising an expanded copy of the distribution.
    """
    if _idx_spec(base_dist) is None:
        raise ValueError(
            f"anchored_relaxed_count: unsupported distribution "
            f"{type(base_dist).__name__}."
        )
    if not trailing_ndims:
        return _unvalidated_log_prob(base_dist, value)
    source = tuple(range(-trailing_ndims, 0))
    destination = tuple(range(trailing_ndims))
    rotated = _unvalidated_log_prob(base_dist, jnp.moveaxis(value, source, destination))
    return jnp.moveaxis(rotated, destination, source)


def _pmf_ratio_down(base_dist, value):
    r"""Return :math:`p(k-1) / p(k)` at integer ``value`` :math:`k`."""
    expand = functools.partial(
        _expand_count_parameter,
        target=value,
        batch_shape=base_dist.batch_shape,
        trailing_ndims=1,
    )
    if isinstance(base_dist, Poisson):
        return value / expand(base_dist.rate)
    if isinstance(base_dist, (GeometricProbs, GeometricLogits)):
        return jnp.ones_like(value) / (1.0 - expand(base_dist.probs))
    if isinstance(base_dist, GammaCount):
        previous_log_pmf = _count_log_pmf(base_dist, value - 1.0, trailing_ndims=1)
        current_log_pmf = _count_log_pmf(base_dist, value, trailing_ndims=1)
        # ``GammaCount.log_prob`` floors an underflowed tail rather than
        # returning ``-inf``, so this difference is finite everywhere and needs
        # no ``isnan`` guard of its own.
        return jnp.exp(previous_log_pmf - current_log_pmf)
    if isinstance(base_dist, GammaPoisson):
        concentration = expand(base_dist.concentration)
        rate = expand(base_dist.rate)
        return value * (rate + 1.0) / (value + concentration - 1.0)
    if isinstance(base_dist, ZeroInflatedProbs):
        base_ratio = _pmf_ratio_down(base_dist.base_dist, value)
        log_zero_ratio = _count_log_pmf(
            base_dist, jnp.zeros_like(value), trailing_ndims=1
        ) - _count_log_pmf(base_dist, jnp.ones_like(value), trailing_ndims=1)
        # ``where`` evaluates and differentiates inactive branches, so cap the
        # exceptional 1 -> 0 ratio before exponentiating it.
        max_log = jnp.log(jnp.finfo(log_zero_ratio.dtype).max) - 2.0
        zero_ratio = jnp.exp(jnp.minimum(log_zero_ratio, max_log))
        return jnp.where(value == 1, zero_ratio, base_ratio)
    raise ValueError(
        f"anchored_relaxed_count: unsupported distribution {type(base_dist).__name__}."
    )


def _count_skewness(base_dist):
    """Skewness used by the first-order Cornish--Fisher anchor."""
    if isinstance(base_dist, Poisson):
        return 1.0 / jnp.sqrt(base_dist.rate)
    if isinstance(base_dist, (GeometricProbs, GeometricLogits)):
        return (2.0 - base_dist.probs) / jnp.sqrt(1.0 - base_dist.probs)
    if isinstance(base_dist, GammaCount):
        counts = jnp.arange(1, base_dist.max_terms + 1).reshape(
            (-1,) + (1,) * len(base_dist.batch_shape)
        )
        tail_probabilities = gammainc(
            base_dist.concentration * counts,
            base_dist.concentration * base_dist.rate,
        )
        mean = jnp.sum(tail_probabilities, axis=0)
        second_moment = jnp.sum((2.0 * counts - 1.0) * tail_probabilities, axis=0)
        third_moment = jnp.sum(
            (3.0 * counts**2 - 3.0 * counts + 1.0) * tail_probabilities,
            axis=0,
        )
        third_central = third_moment - 3.0 * mean * second_moment + 2.0 * mean**3
        variance = jnp.maximum(second_moment - mean**2, 0.0)
        return jnp.where(variance > 0.0, third_central / variance**1.5, 0.0)
    if isinstance(base_dist, GammaPoisson):
        return (base_dist.rate + 2.0) / jnp.sqrt(
            base_dist.concentration * (base_dist.rate + 1.0)
        )
    if isinstance(base_dist, ZeroInflatedProbs):
        keep = 1.0 - base_dist.gate
        mean = base_dist.base_dist.mean
        variance = base_dist.base_dist.variance
        third_central = _count_skewness(base_dist.base_dist) * variance**1.5
        raw_second = variance + mean**2
        raw_third = third_central + 3.0 * mean * variance + mean**3
        mixed_mean = keep * mean
        mixed_second = keep * raw_second
        mixed_third = keep * raw_third
        mixed_central = (
            mixed_third - 3.0 * mixed_mean * mixed_second + 2.0 * mixed_mean**3
        )
        return mixed_central / base_dist.variance**1.5
    raise ValueError(
        f"anchored_relaxed_count: unsupported distribution {type(base_dist).__name__}."
    )


def _count_anchor_binary(base_dist, u, max_count):
    shape = jnp.broadcast_shapes(base_dist.batch_shape, jnp.shape(u))
    lower = jnp.zeros(shape, dtype=int)
    upper = jnp.full(shape, max_count, dtype=int)
    for _ in range(math.ceil(math.log2(max_count + 1))):
        middle = (lower + upper) // 2
        below = _count_cdf(base_dist, middle) < u
        lower = jnp.where(below, middle + 1, lower)
        upper = jnp.where(below, upper, middle)
    return jax.lax.stop_gradient(lower)


def count_anchor_saturates(base_dist, u, max_count=_COUNT_MAX):
    r"""Whether ``max_count`` truncates the count drawn at ``u``.

    :func:`_count_anchor_binary` searches :math:`[0, max\_count]`, so a draw
    whose quantile lies above that bound cannot be anchored.  The relaxation
    then returns a wrong value with a finite gradient and no warning, which is
    why this predicate exists: the condition cannot be raised from inside a
    traced computation, but it can be checked.

    Use it in a smoke run over the ``u`` you expect, or over
    ``jnp.asarray([1 - 1e-6])`` for a worst-case draw::

        assert not count_anchor_saturates(d, jnp.asarray([1 - 1e-6]), max_count)

    :param base_dist: supported unbounded discrete distribution.
    :param u: uniform variate(s) in ``(0, 1)``.
    :param int max_count: the bound to test.
    :return: boolean array, ``True`` where the draw would be truncated.
    """
    if max_count < 1:
        raise ValueError(f"max_count must be positive; got {max_count}")
    shape = jnp.broadcast_shapes(base_dist.batch_shape, jnp.shape(u))
    bound = jnp.broadcast_to(
        jnp.asarray(max_count, dtype=jnp.result_type(base_dist.mean, u)), shape
    )
    return _count_cdf(base_dist, bound) < u


def _count_anchor_cornish_fisher(base_dist, u):
    dtype = jnp.result_type(base_dist.mean, u)
    epsilon = jnp.finfo(dtype).eps
    normal_quantile = ndtri(jnp.clip(u, epsilon, 1.0 - epsilon))
    scale = jnp.sqrt(base_dist.variance)
    quantile = base_dist.mean + scale * normal_quantile
    quantile += (
        _count_skewness(base_dist) * scale * (jnp.square(normal_quantile) - 1.0) / 6.0
    )
    quantile = jnp.where(scale > 0.0, quantile, base_dist.mean)
    return jax.lax.stop_gradient(jnp.maximum(jnp.floor(quantile), 0).astype(int))


def anchored_relaxed_count(
    base_dist,
    u,
    eta,
    anchor="binary",
    width=_COUNT_WINDOW,
    max_count=_COUNT_MAX,
):
    r"""Fixed-window relaxed count centered on a detached quantile anchor.

    This avoids the cross-lane straggler cost of the runtime-adaptive sampler.
    ``anchor="binary"`` uses a fixed-depth CDF search and is the robust default;
    ``anchor="cornish-fisher"`` uses a cheaper moment approximation.  Both
    choices work across Poisson, Geometric, GammaCount, GammaPoisson /
    NegativeBinomial, and their zero-inflated wrappers.

    :param base_dist: supported unbounded discrete distribution.
    :param u: uniform variate(s) in ``(0, 1)``.
    :param float eta: smoothing temperature.
    :param str anchor: ``"binary"`` or ``"cornish-fisher"``.
    :param int width: number of recurrence terms retained on each side.
    :param int max_count: inclusive upper bound of the binary anchor's search.
        This is a *correctness* bound, not a speed knob: a draw whose quantile
        exceeds it is silently truncated, and the relaxation then returns a
        wrong value with a finite gradient.  It must sit above a high quantile
        of the count distribution -- above ``max_count + width`` the returned
        count pins.  Lowering it buys very little, since the search costs
        ``ceil(log2(max_count + 1))`` CDF evaluations: 17 at the default,
        5 at ``max_count=16``.  Use :func:`count_anchor_saturates` to check a
        bound before relying on it.
    """
    if _idx_spec(base_dist) is None:
        raise ValueError(
            f"anchored_relaxed_count: unsupported distribution "
            f"{type(base_dist).__name__}."
        )
    _validate_count_anchor(anchor)
    if width < 1:
        raise ValueError(f"width must be positive; got {width}")
    if max_count < 1:
        raise ValueError(f"max_count must be positive; got {max_count}")

    if anchor == "binary":
        center = _count_anchor_binary(base_dist, u, max_count)
    else:
        center = _count_anchor_cornish_fisher(base_dist, u)

    dtype = jnp.result_type(base_dist.mean, u)
    center_float = center.astype(dtype)
    center_cdf = _count_cdf(base_dist, center_float)
    center_sf = _count_sf(base_dist, center_float)
    center_pmf = jnp.exp(_count_log_pmf(base_dist, center_float))
    steps = jnp.arange(1, width + 1, dtype=dtype)

    left_values = center_float[..., None] - steps + 1.0
    left_valid = steps <= center_float[..., None]
    # Mask invalid negative indices before evaluating the ratio: inactive
    # ``where`` branches are differentiated and GammaPoisson has poles there.
    safe_left_values = jnp.where(left_valid, left_values, 1.0)
    left_factors = jnp.where(
        left_valid, _pmf_ratio_down(base_dist, safe_left_values), 1.0
    )
    left_pmf = center_pmf[..., None] * jnp.cumprod(left_factors, axis=-1)
    left_pmf = jnp.where(left_valid, left_pmf, 0.0)
    # The window is walked outwards from the anchor by the same increments in
    # both directions; the CDF subtracts them going left, the survival function
    # adds them.  Carrying both costs one cumsum and buys a gradient that
    # survives the saturated regime (see the selection below).
    backward = jnp.cumsum(
        jnp.concatenate([center_pmf[..., None], left_pmf[..., :-1]], axis=-1),
        axis=-1,
    )
    left_cdf = jnp.flip(center_cdf[..., None] - backward, axis=-1)
    left_sf = jnp.flip(center_sf[..., None] + backward, axis=-1)

    right_values = center_float[..., None] + steps
    right_factors = 1.0 / _pmf_ratio_down(base_dist, right_values)
    right_pmf = center_pmf[..., None] * jnp.cumprod(right_factors, axis=-1)
    forward = jnp.cumsum(right_pmf[..., :-1], axis=-1)
    right_cdf = center_cdf[..., None] + forward
    right_sf = center_sf[..., None] - forward

    cdf = jnp.concatenate([left_cdf, center_cdf[..., None], right_cdf], axis=-1)
    cdf = jnp.clip(cdf, 0.0, 1.0)
    sf = jnp.concatenate([left_sf, center_sf[..., None], right_sf], axis=-1)
    sf = jnp.clip(sf, 0.0, 1.0)

    offsets = jnp.arange(-width, width)
    indices = center[..., None] + offsets
    valid = indices >= 0
    # ``u - F(k)`` two ways.  They agree in exact arithmetic, but a CDF at 1.0
    # is a constant with a zero derivative, and a survival function at 0.0 is
    # the same trap mirrored, so take whichever tail is the smaller one.  Both
    # branches are finite everywhere, so no masking is needed before the
    # ``where``.
    difference = jnp.where(cdf > 0.5, sf - (1.0 - u[..., None]), u[..., None] - cdf)
    a = jax.nn.sigmoid(difference / eta)
    a_infinity = jax.nn.sigmoid((u - 1.0) / eta)
    a_minus_one = jax.nn.sigmoid(u / eta)
    corrections = jnp.where(
        indices < center[..., None],
        a - a_minus_one[..., None],
        a - a_infinity[..., None],
    )
    corrections = jnp.where(valid, corrections, 0.0)
    return center + jnp.sum(corrections, axis=-1) / (a_minus_one - a_infinity)


# Runtime horizon controls for the adaptive accumulation.  ``_ADAPT_TOL`` is the
# *relative* pmf-decay threshold: the loop stops once the incremental mass p(k)
# has fallen past its peak to ``_ADAPT_TOL`` times that peak (i.e. we are in the
# negligible-mass tail beyond the mode).  ``_ADAPT_MAX_ITERS`` is a safety cap
# (the loop exits early via the tolerance for any real rate).
_ADAPT_TOL = 1e-7
_ADAPT_MAX_ITERS = 100_000


def _accumulate(log_pmf, params, u, eta, tol, max_iters):
    """Scalar accumulation of the relaxed-count numerator and denominator.

    With ``a_k = sigma_eta(u - F(k))`` and soft one-hot weight
    ``w_k = a_{k-1} - a_k`` (``>= 0``, since ``F`` is non-decreasing so ``a_k`` is
    non-increasing), the loop returns

        S0 = sum_k w_k,      S1 = sum_k w_k * k,

    and the relaxed count is ``S1 / S0`` -- a convex combination of the integer
    outcomes.

    ``F(k) = sum_{j<=k} p(j)`` is built incrementally from ``log_pmf``; the
    incremental mass ``p(k)`` rises to a peak at the mode then decays, and the
    loop stops once ``p(k)`` has fallen past that peak to ``tol`` times it (the
    negligible-mass tail, where the remaining ``w_k`` add nothing).

    We key termination off ``p(k)`` on purpose: it's a clean exponential, rather
    than a difference of sigmoids, and it always decays past the mode.  While
    ``p`` is still rising ``p_last == p_max`` so the test cannot fire early in
    the left tail.

    .. warning:: **Cost under ``vmap`` is set by the largest rate in the batch,
       not the typical one.** One coordinate at ``rate=1e4`` among 4095 others
       at low rates imposes a ~1500x slowdown, and the reverse pass suffers it
       again once per parameter. A caller that cannot keep its rates within a
       comparable range should bound them itself."""

    a_init = jax.nn.sigmoid(u / eta)  # a_{-1}: F(-1) = 0

    def cond(c):
        k, _f_prev, _S0, _S1, _a_prev, p_last, p_max = c
        decayed = p_last < tol * p_max  # past the mode, incremental mass negligible
        return jnp.logical_and(k < max_iters, jnp.logical_not(decayed))

    def body(c):
        k, f_prev, S0, S1, a_prev, _p_last, p_max = c
        p = jnp.exp(log_pmf(k, params))  # incremental mass p(k)
        f_k = f_prev + p
        a = jax.nn.sigmoid((u - f_k) / eta)
        w = a_prev - a  # soft one-hot weight w_k (>= 0, a is non-increasing in k)
        return (k + 1.0, f_k, S0 + w, S1 + w * k, a, p, jnp.maximum(p_max, p))

    zero = jnp.zeros_like(u)
    init = (zero, zero, zero, zero, a_init, zero, zero)
    _, _, S0, S1, _, _, _ = jax.lax.while_loop(cond, body, init)
    return S0, S1


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 4, 5))
def _acc(log_pmf, params, u, eta, tol, max_iters):
    return _accumulate(log_pmf, params, u, eta, tol, max_iters)


def _acc_fwd(log_pmf, params, u, eta, tol, max_iters):
    out = _accumulate(log_pmf, params, u, eta, tol, max_iters)
    return out, (params, u, eta)


def _acc_bwd(log_pmf, tol, max_iters, res, ct):
    params, u, eta = res
    # Forward-mode Jacobian of the while_loop accumulation w.r.t. params: JAX
    # will not reverse-differentiate a while_loop but does push jvp through it.
    jac = jax.jacfwd(lambda p: _accumulate(log_pmf, p, u, eta, tol, max_iters))(params)
    # jac is a 2-tuple (one per output S0, S1) of pytrees matching ``params``.
    g_params = jax.tree_util.tree_map(
        lambda *js: sum(c * j for c, j in zip(ct, js)), *jac
    )
    return (g_params, None, None)  # cotangents for (params, u, eta)


_acc.defvjp(_acc_fwd, _acc_bwd)


def _core(spec, u, eta, tol, max_iters):
    """Vectorised ``(S0, S1)`` over an array ``u`` (batched ``params``)."""
    ushape = jnp.shape(u)
    u_flat = jnp.reshape(u, (-1,))
    params_flat = jax.tree_util.tree_map(
        lambda a: jnp.reshape(jnp.broadcast_to(a, ushape), (-1,)), spec.params
    )

    def one(p, uu):
        return _acc(spec.log_pmf, p, uu, eta, tol, max_iters)

    outs = jax.vmap(one)(params_flat, u_flat)
    return tuple(jnp.reshape(o, ushape) for o in outs)


def adaptive_relaxed_count(
    base_dist, u, eta, tol=_ADAPT_TOL, max_iters=_ADAPT_MAX_ITERS
):
    r"""Adaptive index-space relaxed count :math:`Q_\eta(u)` of an *unbounded*
    discrete distribution, for :math:`u \in (0, 1)`.

    With :math:`a_k = \sigma_\eta(u - F(k))` and soft one-hot weights
    :math:`w_k = a_{k-1} - a_k`,

        :math:`Q_\eta(u) = \frac{\sum_k k\, w_k}{\sum_k w_k}`,

    a convex combination of the integer outcomes -- a smoothed, mean-unbiased
    reparameterised sample.  The reparameterisation gradient
    :math:`\partial_\theta Q_\eta` is a bounded weighted sum of
    :math:`\partial_\theta F(k)` (no :math:`1/\mathrm{density}` factor), so it
    stays low-variance across :math:`\eta` and keeps signal as
    :math:`\eta \to 0`.

    The horizon is discovered at runtime with a ``jax.lax.while_loop`` (no static
    truncation), so ``eta`` may be traced (``jit`` / ``scan`` over ``eta``);
    gradients w.r.t. the distribution parameters come from a ``custom_vjp``
    (forward-mode ``jacfwd`` through the loop), since JAX will not
    reverse-differentiate a ``while_loop``.

    This is the reparameterised sample used by :class:`SmoothedCount` and
    :class:`StraightThroughSmoothed`.  The DSGD density term is the analytic-
    continuation log-pmf (see :class:`SmoothedCount`), so this map is never
    inverted and its ``log|dz/du|`` is never needed.

    :param base_dist: an unbounded discrete Distribution supported by
        :func:`_idx_spec` (Poisson, Geometric, GammaCount, GammaPoisson /
        NegativeBinomial, or a zero-inflated wrapper).
    :param u: uniform variate(s) in (0, 1); the return has the same shape.
    :param float eta: smoothing temperature :math:`\eta > 0`.
    :param float tol: relative pmf-decay stopping threshold.
    :param int max_iters: safety cap on the loop trip count.
    """
    spec = _idx_spec(base_dist)
    if spec is None:
        raise ValueError(
            f"adaptive_relaxed_count: unsupported distribution "
            f"{type(base_dist).__name__}."
        )
    S0, S1 = _core(spec, u, eta, tol, max_iters)
    return S1 / S0


# ---------------------------------------------------------------------------
# SmoothedCount: DSGD relaxation of an unbounded discrete distribution
# ---------------------------------------------------------------------------


class SmoothedCount(dist.Distribution):
    r"""DSGD relaxation of an *unbounded* discrete distribution.

    In the reparameterisation view of DSGD, a discrete site :math:`z = Q_D(u)`,
    :math:`u \sim \mathrm{Uniform}(0,1)`, is smoothed by replacing the
    discontinuous quantile :math:`Q_D` with the fixed-window anchored
    index-space relaxed count :math:`Q_\eta`
    (:func:`anchored_relaxed_count`): a low-variance reparameterised sample
    approximating the full mean-unbiased identity. The *density* contribution
    to the objective is the **analytic continuation of the discrete log-pmf**
    through gamma functions (for example, ``k! -> Gamma(z+1)``) evaluated at
    the continuous relaxed count -- for Poisson,

        :math:`\log \tilde p(z) = z\,\log\lambda - \lambda - \log\Gamma(z+1)`

    (other families via :func:`_idx_spec`) -- *not* the change-of-variables
    pushforward density :math:`-\log|dz/du|` that :func:`SmoothedDiscrete` uses
    for finite support.

    Why not the pushforward density: it is unbounded above as the transform
    flattens (:math:`dz/du \to 0` for a near-deterministic, low-rate
    distribution), so the smoothed ELBO's :math:`\log p - \log q` term is not a
    valid :math:`-\mathrm{KL}` and diverges. The analytic-continuation log-pmf
    is bounded. For exponential-family count laws whose log-pmf is affine in
    ``z``, its :math:`\log p - \log q` term is a genuine
    :math:`-\mathrm{KL}` surrogate, exact whenever
    :math:`\mathbb{E}[z] = \mathrm{mean}`, which the mean-unbiased relaxed count
    satisfies. GammaCount's continuation is nonlinear in ``z`` and therefore
    has finite-temperature smoothing bias. As :math:`\eta \to 0`, every relaxed
    count converges to the integer quantile and :math:`\log\tilde p` to the
    exact log-pmf, so the smoothed objective converges to the true discrete one
    (Theorem 5.6); nonlinear terms contribute O(:math:`\eta`) bias.

    ``z`` is continuous, so the site support is continuous and ``log_prob``
    needs no transform inversion; it is evaluated directly with gamma and
    incomplete-gamma functions.

    :param base_dist: an unbounded discrete Distribution supported by
        :func:`_idx_spec` (Poisson, Geometric, GammaCount, GammaPoisson /
        NegativeBinomial, or a zero-inflated wrapper).
    :param float eta: smoothing temperature :math:`\eta > 0`.
    :param str anchor: count anchor method, ``"binary"`` (default) or
        ``"cornish-fisher"``.
    :param int width: recurrence terms retained on each side of the anchor.
    :param int max_count: inclusive upper bound of the binary anchor's search;
        a correctness bound rather than a speed knob, see
        :func:`anchored_relaxed_count` and :func:`count_anchor_saturates`.
    """

    arg_constraints = {}
    pytree_data_fields = ("base_dist", "eta")
    pytree_aux_fields = ("anchor", "width", "max_count")
    support = constraints.positive
    has_rsample = True

    def __init__(
        self,
        base_dist,
        eta,
        *,
        anchor="binary",
        width=_COUNT_WINDOW,
        max_count=_COUNT_MAX,
        validate_args=None,
    ):
        _validate_count_anchor(anchor)
        self.base_dist = base_dist
        self.eta = eta
        self.anchor = anchor
        self.width = width
        self.max_count = max_count
        super().__init__(batch_shape=base_dist.batch_shape, validate_args=validate_args)

    def _spec(self) -> "_IdxSpec":
        spec = _idx_spec(self.base_dist)
        if spec is None:
            raise ValueError(
                f"SmoothedCount: unsupported distribution "
                f"{type(self.base_dist).__name__}."
            )
        return spec

    def rsample(self, key, sample_shape=()):
        u = jax.random.uniform(key, shape=sample_shape + self.batch_shape)
        return anchored_relaxed_count(
            self.base_dist,
            u,
            self.eta,
            anchor=self.anchor,
            width=self.width,
            max_count=self.max_count,
        )

    def sample(self, key, sample_shape=()):
        return self.rsample(key, sample_shape)

    def log_prob(self, value):
        spec = self._spec()
        return spec.log_pmf(value, spec.params)


# ---------------------------------------------------------------------------
# SmoothedDiscrete factory
# ---------------------------------------------------------------------------


def _use_count_relaxation(base_dist, max_support: Optional[int]) -> bool:
    """Whether to use the anchored count relaxation for ``base_dist``.

    Unbounded families always use it (the truncated grid is biased for them);
    ``max_support`` is meaningful only for finite-support families and is
    rejected for unbounded ones.
    """
    if _idx_spec(base_dist) is not None:
        if max_support is not None:
            raise ValueError(
                f"numpyro.contrib.diag_sgd: max_support is not supported for the "
                f"unbounded family {type(base_dist).__name__}; it uses the grid-free "
                f"anchored count relaxation. Omit max_support."
            )
        return True
    return False


def _smooth_wrapped(fn, build_base):
    """Peel ``Independent`` (``to_event``) and ``ExpandedDistribution``
    (``plate`` / ``.expand``) wrappers off ``fn``, apply ``build_base`` to the
    concrete base distribution, then re-apply the wrappers so event
    reinterpretation and batch expansion are preserved."""
    wrappers = []  # outer-first: ("event", n) or ("expand", batch_shape)
    d = fn
    while True:
        if isinstance(d, ExpandedDistribution):
            wrappers.append(("expand", d.batch_shape))
            d = d.base_dist
        elif isinstance(d, Independent):
            wrappers.append(("event", d.reinterpreted_batch_ndims))
            d = d.base_dist
        else:
            break
    out = build_base(d)
    for kind, val in reversed(wrappers):  # re-apply inner-to-outer
        out = out.to_event(val) if kind == "event" else out.expand(val)
    return out


def SmoothedDiscrete(
    base_dist,
    eta,
    max_support: Optional[int] = None,
    *,
    anchor="binary",
    width=_COUNT_WINDOW,
    max_count=_COUNT_MAX,
):
    """
    Wrap a discrete distribution so that samples are drawn via the smooth iCDF
    and log-probs return the correct pushforward density.

    Two different (both consistent) constructions are used:

    * **Finite-support families** (Bernoulli, Categorical, Binomial,
      DiscreteUniform) become a :class:`TransformedDistribution` through the grid
      :class:`SmoothICDFTransform`.  That transform is a strict diffeomorphism
      onto the fixed codomain ``[0, K-1]`` shared by every distribution of the
      family, so its pushforward density is proper and ``log p - log q`` is a
      valid ``-KL``.
    * **Unbounded families** (Poisson, Geometric, GammaCount, GammaPoisson /
      NegativeBinomial and their zero-inflated wrappers) become a
      :class:`SmoothedCount`: the anchored index-space relaxed count as the
      reparameterised sample, with the analytic-continuation log-pmf as the
      density.  The pushforward density is *not* used here -- it is unbounded
      above for near-deterministic (low-rate) distributions and makes
      ``log p - log q`` diverge; see :class:`SmoothedCount`.

    ``max_support`` applies only to finite-support families — it is rejected for
    unbounded ones.  ``Independent`` (``to_event``) and ``ExpandedDistribution``
    (``plate`` / ``.expand``) wrappers are peeled off, the base smoothed
    elementwise, and the wrappers re-applied.

    :param base_dist: original discrete Distribution.
    :param float eta: smoothing temperature :math:`\\eta \\gt 0`.
    :param max_support: optional grid truncation for finite-support families.
    :param str anchor: anchor method for unbounded count families: ``"binary"``
        (default) or ``"cornish-fisher"``.
    :param int width: recurrence terms retained on each side of the count anchor.
    :param int max_count: upper search bound used by the binary anchor.
    :return: a smoothed distribution approximating base_dist.
    """

    _validate_count_anchor(anchor)

    def build(d):
        if _use_count_relaxation(d, max_support):
            return SmoothedCount(
                d, eta, anchor=anchor, width=width, max_count=max_count
            )
        transform = SmoothICDFTransform(d, eta, max_support)
        uniform = dist.Uniform(0.0, 1.0).expand(d.batch_shape)
        return TransformedDistribution(uniform, transform)

    return _smooth_wrapped(base_dist, build)


# ---------------------------------------------------------------------------
# StraightThroughSmoothed
# ---------------------------------------------------------------------------


class StraightThroughSmoothed(dist.Distribution):
    """Straight-through reparameterisation of a discrete distribution.

    ``sample`` draws a smooth reparameterised value ``z`` (via the anchored
    index-space relaxation for unbounded families, else the grid iCDF) and
    returns ``round(z) + (z − stop_gradient(z))``: the forward value is the
    integer ``round(z)`` but gradients flow through the smooth ``z``.
    ``log_prob`` is the base distribution's log-prob at the rounded value.  This
    is the biased straight-through estimator; for the consistent DSGD objective
    use :func:`SmoothedDiscrete`.

    Replacing a sample site's ``fn`` with this (rather than overwriting its
    value) is what lets the straight-through path compose correctly with the
    ``seed`` and ``trace`` handlers.

    :param base_dist: original discrete Distribution.
    :param float eta: smoothing temperature :math:`\\eta \\gt 0`.
    :param max_support: optional grid size for finite-support families (rejected
        for unbounded families).
    :param str anchor: anchor method for unbounded count families: ``"binary"``
        (default) or ``"cornish-fisher"``.
    :param int width: recurrence terms retained on each side of the count anchor.
    :param int max_count: upper search bound used by the binary anchor.
    """

    arg_constraints = {}
    pytree_data_fields = ("base_dist", "eta")
    pytree_aux_fields = ("max_support", "anchor", "width", "max_count")

    def __init__(
        self,
        base_dist,
        eta,
        max_support=None,
        *,
        anchor="binary",
        width=_COUNT_WINDOW,
        max_count=_COUNT_MAX,
        validate_args=None,
    ):
        _validate_count_anchor(anchor)
        self.base_dist = base_dist
        self.eta = eta
        self.max_support = max_support
        self.anchor = anchor
        self.width = width
        self.max_count = max_count
        super().__init__(batch_shape=base_dist.batch_shape, validate_args=validate_args)

    @constraints.dependent_property(is_discrete=True, event_dim=0)
    def support(self):
        return self.base_dist.support

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        u = jax.random.uniform(key, shape=shape)
        if _use_count_relaxation(self.base_dist, self.max_support):
            z = anchored_relaxed_count(
                self.base_dist,
                u,
                self.eta,
                anchor=self.anchor,
                width=self.width,
                max_count=self.max_count,
            )
        else:
            z = smooth_icdf(self.base_dist, u, self.eta, self.max_support)
        return jnp.round(z) + (z - jax.lax.stop_gradient(z))

    def log_prob(self, value):
        return self.base_dist.log_prob(jnp.round(value))


# ---------------------------------------------------------------------------
# Smooth control-flow wrappers (track nesting depth)
# ---------------------------------------------------------------------------


def smooth_cond(smooth_pred, true_fn, false_fn, *args, **kwargs):
    """
    Smooth replacement for ``jax.lax.cond`` for use inside DSGD models.

    Both branches are evaluated and blended::

        result = smooth_pred * true_fn(...) + (1 - smooth_pred) * false_fn(...)

    ``smooth_pred`` should be a differentiable scalar in [0, 1] (e.g., the
    output of :func:`smooth_heaviside`).  This call increments the nesting-depth
    counter used by :func:`count_layers`.

    :param smooth_pred: differentiable weight for the true branch.
    :param true_fn: callable for the true branch.
    :param false_fn: callable for the false branch.
    :param args: positional arguments forwarded to both branches.
    :param kwargs: keyword arguments forwarded to both branches.
    """
    old_depth = _current_depth()
    new_depth = old_depth + 1
    _depth_state.depth = new_depth
    _update_max_depth(new_depth)
    try:
        true_out = true_fn(*args, **kwargs)
        false_out = false_fn(*args, **kwargs)
    finally:
        _depth_state.depth = old_depth

    return jax.tree.map(
        lambda t, f: smooth_pred * t + (1.0 - smooth_pred) * f,
        true_out,
        false_out,
    )


def smooth_switch(smooth_weights, branch_fns, *args, **kwargs):
    """
    Smooth replacement for ``jax.lax.switch`` for use inside DSGD models.

    All branches are evaluated and blended by ``smooth_weights``::

        result = sum_i smooth_weights[i] * branch_fns[i](...)

    ``smooth_weights`` should be a differentiable array of non-negative values
    that sum to 1 (e.g., softmax outputs).  This call increments the
    nesting-depth counter used by :func:`count_layers`.

    :param smooth_weights: array-like of K weights.
    :param branch_fns: sequence of K callables.
    :param args: positional arguments forwarded to all branches.
    :param kwargs: keyword arguments forwarded to all branches.
    """
    old_depth = _current_depth()
    new_depth = old_depth + 1
    _depth_state.depth = new_depth
    _update_max_depth(new_depth)
    try:
        outputs = [fn(*args, **kwargs) for fn in branch_fns]
    finally:
        _depth_state.depth = old_depth

    weights = list(smooth_weights)
    return jax.tree.map(
        lambda *leaves: sum(w * leaf for w, leaf in zip(weights, leaves)),
        *outputs,
    )


# ---------------------------------------------------------------------------
# Helper: detect discrete distributions
# ---------------------------------------------------------------------------


def _is_discrete(d) -> bool:
    """Return True if d is a discrete NumPyro distribution."""
    try:
        return bool(d.is_discrete)
    except (AttributeError, NotImplementedError):
        return False


# ---------------------------------------------------------------------------
# DSGDMessenger
# ---------------------------------------------------------------------------


class DSGDMessenger(Messenger):
    """
    Effect handler that applies DSGD smoothing to all discrete sample sites.

    When ``smoothed_distributions=True`` (default), each discrete site's
    distribution is replaced with :func:`SmoothedDiscrete`, so the sample
    is drawn from the smooth pushforward and the log-prob is the pushforward
    density. This gives the correct DSGD gradient estimator.

    When ``smoothed_distributions=False``, a straight-through estimator is
    used instead: a smooth sample is drawn (via the anchored index-space
    relaxation for unbounded families, else :func:`smooth_icdf`),
    rounded to the nearest integer (for the forward pass), but gradients flow
    through the smooth sample. This estimator is biased w.r.t. the DSGD
    objective but keeps discrete values in the computation graph.

    :param float eta: smoothing temperature :math:`\\eta \\gt 0`.
    :param bool smoothed_distributions: select estimator variant.
    :param max_support: optional grid size for finite-support families (rejected
        for unbounded families).
    :param str anchor: anchor method for unbounded count families: ``"binary"``
        (default) or ``"cornish-fisher"``.
    :param int width: recurrence terms retained on each side of the count anchor.
    :param int max_count: upper search bound used by the binary anchor.
    """

    def __init__(
        self,
        eta,
        smoothed_distributions: bool = True,
        max_support=None,
        *,
        anchor="binary",
        width=_COUNT_WINDOW,
        max_count=_COUNT_MAX,
    ):
        _validate_count_anchor(anchor)
        self.eta = eta
        self.smoothed_distributions = smoothed_distributions
        self.max_support = max_support
        self.anchor = anchor
        self.width = width
        self.max_count = max_count
        super().__init__()

    def process_message(self, msg):
        if msg["type"] != "sample" or msg.get("is_observed"):
            return
        d = msg["fn"]
        if not _is_discrete(d):
            return

        # Peel off Independent (``to_event``) and ExpandedDistribution
        # (``plate`` / ``.expand``) wrappers, smooth the base elementwise, then
        # re-apply the wrappers so event dims and batch expansion are preserved.
        def build(base):
            if self.smoothed_distributions:
                return SmoothedDiscrete(
                    base,
                    self.eta,
                    self.max_support,
                    anchor=self.anchor,
                    width=self.width,
                    max_count=self.max_count,
                )
            # Straight-through: replace fn with a distribution whose sample is
            # round(z) + (z − stop_gradient(z)).  Replacing fn (rather than
            # setting the value here) lets the seed handler supply the rng_key
            # and the trace handler record the straight-through value.
            return StraightThroughSmoothed(
                base,
                self.eta,
                self.max_support,
                anchor=self.anchor,
                width=self.width,
                max_count=self.max_count,
            )

        msg["fn"] = _smooth_wrapped(d, build)


# ---------------------------------------------------------------------------
# count_layers: Python-trace-time layer-count computation
# ---------------------------------------------------------------------------


def count_layers(model, *args, rng_key=None, **kwargs) -> int:
    """
    Count the number of smoothing layers ℓ in the model, used to set the exponent
    in :func:`eta_schedule`.

    Each smoothed discrete sample site is one layer (its smooth iCDF is a
    smoothed discontinuity), and each :func:`smooth_cond` / :func:`smooth_switch`
    nesting adds further layers of composition.  Concretely

        ℓ = (number of discrete sample sites) + (max smooth_cond/switch depth),

    a conservative upper bound on the composition depth of discontinuities
    (independent discrete sites are counted as if composed, which only slows the
    annealing — the safe direction for Theorem 5.6).  A model with at least one
    discrete site therefore has :math:`\\ell \\geq 1`, so :func:`eta_schedule`
    actually anneals :math:`\\eta \\to 0`; :math:`\\ell = 0` only for models with
    no discrete sites and no smooth control flow.

    Runs the model once (with a dummy rng_key if not provided) to trace its sites
    and Python-level control flow.

    :param model: NumPyro model function.
    :param args: positional arguments to pass to model.
    :param rng_key: JAX PRNGKey used for seeding (default: PRNGKey(0)).
    :param kwargs: keyword arguments to pass to model.
    :return: integer layer count :math:`\\ell`.
    """
    if rng_key is None:
        rng_key = jax.random.key(0)
    _depth_state.depth = 0
    _depth_state.max_depth = 0
    seeded = numpyro.handlers.seed(model, rng_key)
    trace = numpyro.handlers.trace(seeded).get_trace(*args, **kwargs)
    n_discrete = sum(
        1
        for site in trace.values()
        if site["type"] == "sample"
        and not site.get("is_observed")
        and _is_discrete(site["fn"])
    )
    return int(_depth_state.max_depth) + n_discrete


# ---------------------------------------------------------------------------
# dsgd: main transformation entry point
# ---------------------------------------------------------------------------


def dsgd(
    model,
    smoothed_distributions: bool = True,
    max_support=None,
    *,
    anchor="binary",
    width=_COUNT_WINDOW,
    max_count=_COUNT_MAX,
):
    """
    Transform a NumPyro model to use DSGD smoothing at all discrete sites.

    Returns a *smoothed model* ``smoothed_fn(eta, *args, **kwargs)`` that
    behaves identically to ``model(*args, **kwargs)`` except that discrete
    sample sites are replaced with their smooth iCDF reparameterisations
    controlled by the temperature parameter ``eta``.

    Typical usage::

        smoothed_model = dsgd(model)
        ell = count_layers(model, *example_args, **example_kwargs)
        schedule = eta_schedule(K=num_steps, ell=ell, eta_final=0.01)

        for k in range(num_steps):
            eta = schedule[k]
            # Pass smoothed_model(eta, ...) to SVI / ELBO.

    Unbounded count families use a fixed recurrence window around a detached
    quantile anchor, avoiding cross-lane runtime stragglers.  Binary-search
    anchors are the robust default; Cornish--Fisher anchors trade accuracy for
    cheaper setup.  ``eta`` may be concrete or traced under ``jit`` / ``scan``.

    :param model: NumPyro model function.
    :param bool smoothed_distributions: if True (default), replace discrete
        distributions with :func:`SmoothedDiscrete` (correct DSGD estimator);
        if False, use the straight-through estimator (biased but keeps integer
        values for downstream code that expects integers).
    :param max_support: optional grid size for finite-support families (rejected
        for unbounded families).
    :param str anchor: anchor method for unbounded count families: ``"binary"``
        (default) or ``"cornish-fisher"``.
    :param int width: recurrence terms retained on each side of the count anchor.
    :param int max_count: upper search bound used by the binary anchor.
    :return: ``smoothed_fn(eta, *args, **kwargs)`` — a callable that wraps
        the model under :class:`DSGDMessenger`.
    """

    def smoothed_fn(eta, *args, **kwargs):
        with DSGDMessenger(
            eta,
            smoothed_distributions=smoothed_distributions,
            max_support=max_support,
            anchor=anchor,
            width=width,
            max_count=max_count,
        ):
            return model(*args, **kwargs)

    return smoothed_fn


# ---------------------------------------------------------------------------
# eta_schedule
# ---------------------------------------------------------------------------


def eta_schedule(K: int, ell: int, eta_final: float, eps: float = 0.01):
    r"""
    Compute the full η-schedule for DSGD as per Theorem 5.6.

    The schedule is

        :math:`\eta_k = \eta_{\mathrm{final}} \cdot (k/K)^{-(1/\ell + \varepsilon)}`
        for  k = 1, …, K

    so that :math:`\eta_K = \eta_{\mathrm{final}}` and
    :math:`\eta_1 = \eta_{\mathrm{final}} \cdot K^{1/\ell + \varepsilon}` (the
    largest value).  For :math:`\ell = 0` (no discrete sites and no smooth_cond
    calls) the schedule is constant at :math:`\eta_{\mathrm{final}}`.

    The anchored relaxation accepts a traced η, so a schedule entry may be
    indexed inside a jitted/scanned step without recompiling per step.

    :param int K: total number of optimisation steps planned.
    :param int ell: layer count from :func:`count_layers`.
    :param float eta_final: smoothing temperature at the final step.
    :param float eps: small offset to the exponent (default 0.01).
    :return: a shape-(K,) JAX array of η values for steps 1 … K.
    """
    ks = jnp.arange(1, K + 1, dtype=jnp.float32)
    if ell == 0:
        return jnp.full(K, float(eta_final))
    exponent = 1.0 / ell + eps
    return float(eta_final) * jnp.power(float(K) / ks, exponent)
