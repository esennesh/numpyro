# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
import pytest

import jax
from jax import grad, jit, value_and_grad
import jax.numpy as jnp

import numpyro
from numpyro import handlers
from numpyro.contrib.diag_sgd import (
    SmoothedCount,
    SmoothedDiscrete,
    SmoothICDFTransform,
    StraightThroughSmoothed,
    _count_cdf,
    _count_log_pmf,
    _count_sf,
    _idx_spec,
    _unvalidated_log_prob,
    adaptive_relaxed_count,
    anchored_relaxed_count,
    count_anchor_saturates,
    count_layers,
    dsgd,
    eta_schedule,
    smooth_cond,
    smooth_heaviside,
    smooth_icdf,
    smooth_switch,
)
import numpyro.distributions as dist

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


ETA = 0.1  # smoothing temperature for most tests


def _sample_and_grad(d, eta, n_samples=200, seed=0):
    """Draw samples from SmoothedDiscrete and compute gradient of their mean."""
    smoothed = SmoothedDiscrete(d, eta)

    def mean_sample(rng_key):
        return smoothed.sample(rng_key).mean()

    keys = jax.random.split(jax.random.key(seed), n_samples)
    return jax.vmap(lambda k: value_and_grad(mean_sample)(k))(keys)


# ---------------------------------------------------------------------------
# smooth_heaviside
# ---------------------------------------------------------------------------


def test_smooth_heaviside_values():
    x = jnp.array([-1.0, 0.0, 1.0])
    y = smooth_heaviside(x, eta=1.0)
    expected = jax.nn.sigmoid(x)
    assert_allclose(y, expected, rtol=1e-5)


def test_smooth_heaviside_limits():
    # As eta → ∞, sigma → 0.5 everywhere; as eta → 0+, sigma → Heaviside
    x = jnp.array(2.0)
    assert_allclose(smooth_heaviside(x, eta=1e6), 0.5, atol=1e-3)
    assert float(smooth_heaviside(x, eta=1e-6)) > 0.999


# ---------------------------------------------------------------------------
# smooth_icdf – per-distribution family
# ---------------------------------------------------------------------------


class TestSmoothICDFBernoulli:
    @property
    def d(self):
        return dist.BernoulliProbs(jnp.array(0.7))

    def test_monotone(self):
        us = jnp.linspace(0.01, 0.99, 50)
        zs = smooth_icdf(self.d, us, ETA)
        assert jnp.all(jnp.diff(zs) >= 0)

    def test_range(self):
        us = jnp.linspace(0.01, 0.99, 50)
        zs = smooth_icdf(self.d, us, ETA)
        assert jnp.all(zs >= 0.0)
        assert jnp.all(zs <= 1.0)

    def test_gradient_flows(self):
        u = jnp.array(0.5)
        g = grad(lambda u_: smooth_icdf(self.d, u_, ETA))(u)
        assert jnp.isfinite(g)
        assert g > 0  # monotone increasing

    def test_small_eta_approximates_icdf(self):
        # With small η, Q^η(u) ≈ round(icdf(u))
        us = jnp.linspace(0.05, 0.95, 20)
        zs = smooth_icdf(self.d, us, eta=1e-3)
        # For Bernoulli(0.7): threshold at F(0) = 0.3
        expected = jnp.where(us > 0.3, 1.0, 0.0)
        assert_allclose(zs, expected, atol=0.01)


class TestSmoothICDFCategorical:
    @property
    def d(self):
        return dist.CategoricalProbs(jnp.array([0.1, 0.3, 0.4, 0.2]))

    def test_monotone(self):
        us = jnp.linspace(0.01, 0.99, 50)
        zs = smooth_icdf(self.d, us, ETA)
        assert jnp.all(jnp.diff(zs) >= 0)

    def test_range(self):
        us = jnp.linspace(0.01, 0.99, 50)
        zs = smooth_icdf(self.d, us, ETA)
        assert jnp.all(zs >= 0.0)
        assert jnp.all(zs <= 3.0)

    def test_gradient_flows(self):
        u = jnp.array(0.5)
        g = grad(lambda u_: smooth_icdf(self.d, u_, ETA))(u)
        assert jnp.isfinite(g) and g > 0

    def test_small_eta_approximates_icdf(self):
        us = jnp.array([0.05, 0.25, 0.55, 0.85])
        zs = smooth_icdf(self.d, us, eta=1e-3)
        # cumprobs = [0.1, 0.4, 0.8, 1.0]; icdf at each u:
        expected = jnp.array([0.0, 1.0, 2.0, 3.0])
        assert_allclose(jnp.round(zs), expected, atol=0.5)


@pytest.mark.parametrize(
    "make",
    [
        lambda: dist.Poisson(jnp.array(3.0)),
        lambda: dist.GeometricProbs(jnp.array(0.5)),
        lambda: dist.NegativeBinomialProbs(total_count=5, probs=jnp.array(0.4)),
    ],
)
def test_smooth_icdf_rejects_unbounded(make):
    # The grid inverse-CDF is biased for unbounded support and no longer handles
    # it; unbounded families must use the adaptive index-space sampler.
    with pytest.raises(ValueError, match="unbounded"):
        smooth_icdf(make(), jnp.linspace(0.1, 0.9, 5), ETA)


class TestSmoothICDFDiscreteUniform:
    @property
    def d(self):
        return dist.DiscreteUniform(low=2, high=7)

    def test_monotone(self):
        us = jnp.linspace(0.01, 0.99, 50)
        zs = smooth_icdf(self.d, us, ETA)
        assert jnp.all(jnp.diff(zs) >= 0)

    def test_range(self):
        us = jnp.linspace(0.01, 0.99, 50)
        zs = smooth_icdf(self.d, us, ETA)
        # Should stay near [2, 7]
        assert jnp.all(zs >= 1.9)
        assert jnp.all(zs <= 7.1)

    def test_gradient_flows(self):
        u = jnp.array(0.5)
        g = grad(lambda u_: smooth_icdf(self.d, u_, ETA))(u)
        assert jnp.isfinite(g) and g > 0


# ---------------------------------------------------------------------------
# SmoothICDFTransform
# ---------------------------------------------------------------------------


class TestSmoothICDFTransform:
    def _bernoulli(self):
        return SmoothICDFTransform(dist.BernoulliProbs(jnp.array(0.6)), ETA)

    def _categorical(self):
        return SmoothICDFTransform(
            dist.CategoricalProbs(jnp.array([0.2, 0.5, 0.3])), ETA
        )

    def test_forward_inverse_bernoulli(self):
        T = self._bernoulli()
        us = jnp.linspace(0.1, 0.9, 10)
        zs = T(us)
        us_back = T.inv(zs)
        assert_allclose(us_back, us, atol=1e-4)

    def test_forward_inverse_categorical(self):
        T = self._categorical()
        us = jnp.linspace(0.1, 0.9, 10)
        zs = T(us)
        us_back = T.inv(zs)
        assert_allclose(us_back, us, atol=1e-3)

    def test_log_abs_det_jacobian_bernoulli(self):
        T = self._bernoulli()
        us = jnp.linspace(0.1, 0.9, 5)
        zs = T(us)
        ladj = T.log_abs_det_jacobian(us, zs)
        # Should equal log of gradient via autodiff
        g = jax.vmap(grad(lambda u_: T(u_)))(us)
        expected = jnp.log(jnp.abs(g))
        assert_allclose(ladj, expected, rtol=1e-4)

    def test_log_abs_det_jacobian_categorical(self):
        T = self._categorical()
        us = jnp.linspace(0.1, 0.9, 5)
        zs = T(us)
        ladj = T.log_abs_det_jacobian(us, zs)
        g = jax.vmap(grad(lambda u_: T(u_)))(us)
        expected = jnp.log(jnp.abs(g))
        assert_allclose(ladj, expected, rtol=1e-3)

    def test_pytree_roundtrip(self):
        T = self._bernoulli()
        leaves, treedef = jax.tree.flatten(T)
        T2 = jax.tree.unflatten(treedef, leaves)
        us = jnp.linspace(0.1, 0.9, 5)
        assert_allclose(T(us), T2(us), rtol=1e-6)

    def test_jit_compatible(self):
        T = self._bernoulli()
        f = jit(lambda u: T(u))
        us = jnp.linspace(0.1, 0.9, 5)
        assert_allclose(f(us), T(us), rtol=1e-6)


# ---------------------------------------------------------------------------
# SmoothedDiscrete
# ---------------------------------------------------------------------------


class TestSmoothedDiscrete:
    def test_sample_bernoulli(self):
        d = dist.BernoulliProbs(jnp.array(0.7))
        sd = SmoothedDiscrete(d, ETA)
        samples = sd.sample(jax.random.key(0), (1000,))
        # Samples should be in [0, 1]
        assert jnp.all(samples >= 0.0) and jnp.all(samples <= 1.0)
        # Mean should be close to p=0.7 (allowing for smooth approximation)
        assert_allclose(samples.mean(), 0.7, atol=0.1)

    def test_log_prob_shape_bernoulli(self):
        d = dist.BernoulliProbs(jnp.array(0.7))
        sd = SmoothedDiscrete(d, ETA)
        zs = sd.sample(jax.random.key(1), (5,))
        lp = sd.log_prob(zs)
        assert lp.shape == (5,)
        assert jnp.all(jnp.isfinite(lp))

    def test_log_prob_shape_categorical(self):
        d = dist.CategoricalProbs(jnp.array([0.2, 0.5, 0.3]))
        sd = SmoothedDiscrete(d, ETA)
        zs = sd.sample(jax.random.key(2), (8,))
        lp = sd.log_prob(zs)
        assert lp.shape == (8,)
        assert jnp.all(jnp.isfinite(lp))

    def test_gradient_through_sample(self):
        def loss(p):
            d = dist.BernoulliProbs(p)
            sd = SmoothedDiscrete(d, ETA)
            z = sd.sample(jax.random.key(3))
            return z

        g = grad(loss)(jnp.array(0.5))
        assert jnp.isfinite(g)

    def test_gradient_through_log_prob(self):
        def loss(p):
            d = dist.BernoulliProbs(p)
            sd = SmoothedDiscrete(d, ETA)
            z = sd.sample(jax.random.key(4))
            return sd.log_prob(z)

        g = grad(loss)(jnp.array(0.5))
        assert jnp.isfinite(g)

    # Sampling from unbounded families (Geometric/Poisson/NegBin) is covered by
    # TestAdaptiveSampling via the grid-free adaptive sampler; the old
    # max_support-on-unbounded variants were removed with the biased grid path.


# ---------------------------------------------------------------------------
# smooth_cond / smooth_switch
# ---------------------------------------------------------------------------


def test_smooth_cond_blends_outputs():
    # At smooth_pred=0.5, result should be midpoint
    result = smooth_cond(
        0.5,
        lambda: jnp.array(2.0),
        lambda: jnp.array(4.0),
    )
    assert_allclose(result, 3.0, rtol=1e-6)


def test_smooth_cond_gradient():
    def f(pred):
        return smooth_cond(pred, lambda: jnp.array(2.0), lambda: jnp.array(4.0))

    g = grad(f)(jnp.array(0.5))
    # d/d_pred [pred*2 + (1-pred)*4] = 2 - 4 = -2
    assert_allclose(g, -2.0, rtol=1e-6)


def test_smooth_switch_blends_outputs():
    w = jnp.array([0.2, 0.3, 0.5])
    fns = [lambda: jnp.array(1.0), lambda: jnp.array(2.0), lambda: jnp.array(3.0)]
    result = smooth_switch(w, fns)
    expected = 0.2 * 1.0 + 0.3 * 2.0 + 0.5 * 3.0
    assert_allclose(result, expected, rtol=1e-6)


def test_smooth_cond_depth_tracking():
    from numpyro.contrib.diag_sgd import _depth_state

    _depth_state.depth = 0
    _depth_state.max_depth = 0

    def inner():
        smooth_cond(jnp.array(0.5), lambda: 0.0, lambda: 1.0)
        return 0.0

    smooth_cond(jnp.array(0.7), inner, lambda: 0.0)

    assert _depth_state.max_depth == 2


# ---------------------------------------------------------------------------
# count_layers
# ---------------------------------------------------------------------------


def test_count_layers_no_discrete_no_cond():
    # No discrete sites and no smooth control flow -> ell == 0.
    def model():
        numpyro.sample("p", dist.Beta(1.0, 1.0))

    assert count_layers(model) == 0


def test_count_layers_one_discrete():
    # A single discrete site is one smoothing layer.
    def model():
        numpyro.sample("x", dist.BernoulliProbs(jnp.array(0.5)))

    assert count_layers(model) == 1


def test_count_layers_multiple_discrete():
    # Each discrete site increments ell.
    def model():
        numpyro.sample("x", dist.BernoulliProbs(jnp.array(0.5)))
        numpyro.sample("y", dist.Poisson(jnp.array(3.0)))
        numpyro.sample("z", dist.GeometricProbs(jnp.array(0.4)))

    assert count_layers(model) == 3


def test_count_layers_one_cond():
    # one discrete site (1) + one smooth_cond level (1) = 2
    def model():
        x = numpyro.sample("x", dist.BernoulliProbs(jnp.array(0.5)))
        smooth_x = jax.nn.sigmoid(x.astype(float) / ETA)
        smooth_cond(smooth_x, lambda: None, lambda: None)

    assert count_layers(model) == 2


def test_count_layers_nested_cond():
    # one discrete site (1) + two nested smooth_cond levels (2) = 3
    def model():
        x = numpyro.sample("x", dist.BernoulliProbs(jnp.array(0.5)))
        sx = jax.nn.sigmoid(x.astype(float) / ETA)

        def inner():
            smooth_cond(sx, lambda: None, lambda: None)

        smooth_cond(sx, inner, lambda: None)

    assert count_layers(model) == 3


# ---------------------------------------------------------------------------
# DSGDMessenger via dsgd()
# ---------------------------------------------------------------------------


class TestDSGD:
    def _simple_model(self, data):
        p = numpyro.sample("p", dist.Beta(1.0, 1.0))
        z = numpyro.sample("z", dist.BernoulliProbs(p))
        numpyro.sample("obs", dist.Normal(z.astype(float), 1.0), obs=data)

    def test_smoothed_distributions_replaces_fn(self):
        """With smoothed_distributions=True, the discrete site fn changes."""
        smoothed = dsgd(self._simple_model)
        data = jnp.array(0.8)

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(lambda eta: smoothed(eta, data)).get_trace(ETA)

        # 'z' site should now have a TransformedDistribution (SmoothedDiscrete)
        z_fn = trace["z"]["fn"]
        assert isinstance(z_fn, dist.TransformedDistribution)

    def test_observed_sites_untouched(self):
        """Observed sites must not be smoothed."""
        smoothed = dsgd(self._simple_model)
        data = jnp.array(0.8)

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(lambda eta: smoothed(eta, data)).get_trace(ETA)

        # 'obs' is Normal (continuous), untouched
        assert isinstance(trace["obs"]["fn"], dist.Normal)

    def test_gradient_flows_through_discrete_site(self):
        def loss(eta):
            smoothed_model = dsgd(self._simple_model)
            with handlers.seed(rng_seed=1):
                trace = handlers.trace(
                    lambda: smoothed_model(eta, jnp.array(0.5))
                ).get_trace()
            return trace["z"]["value"]

        g = grad(loss)(jnp.array(ETA))
        assert jnp.isfinite(g)

    def test_straight_through_mode(self):
        """With smoothed_distributions=False, the site fn is the straight-through
        wrapper and the sampled value is integer-valued."""
        smoothed = dsgd(self._simple_model, smoothed_distributions=False)
        data = jnp.array(0.8)

        with handlers.seed(rng_seed=0):
            trace = handlers.trace(lambda eta: smoothed(eta, data)).get_trace(ETA)

        z_fn = trace["z"]["fn"]
        assert isinstance(z_fn, StraightThroughSmoothed)
        assert isinstance(z_fn.base_dist, dist.BernoulliProbs)
        z = trace["z"]["value"]
        assert_allclose(z, jnp.round(z), atol=1e-5)

    def test_categorical_model(self):
        def model(obs):
            probs = numpyro.sample("probs", dist.Dirichlet(jnp.ones(3)))
            z = numpyro.sample("z", dist.CategoricalProbs(probs))
            numpyro.sample("obs", dist.Normal(z.astype(float), 1.0), obs=obs)

        smoothed = dsgd(model)
        with handlers.seed(rng_seed=2):
            trace = handlers.trace(lambda eta: smoothed(eta, jnp.array(1.0))).get_trace(
                ETA
            )
        z_fn = trace["z"]["fn"]
        assert isinstance(z_fn, dist.TransformedDistribution)


# ---------------------------------------------------------------------------
# eta_schedule
# ---------------------------------------------------------------------------


def test_eta_schedule_ell_zero():
    sched = eta_schedule(K=100, ell=0, eta_final=0.01)
    assert sched.shape == (100,)
    assert_allclose(sched, 0.01 * jnp.ones(100), rtol=1e-5)


def test_eta_schedule_ell_one():
    K = 100
    sched = eta_schedule(K=K, ell=1, eta_final=0.01, eps=0.0)
    # eta_k = 0.01 * (K / k)^1
    ks = jnp.arange(1, K + 1, dtype=float)
    expected = 0.01 * (K / ks)
    assert_allclose(sched, expected, rtol=1e-4)
    # Final step equals eta_final
    assert_allclose(float(sched[-1]), 0.01, rtol=1e-5)
    # Schedule is decreasing
    assert jnp.all(jnp.diff(sched) <= 0)


def test_eta_schedule_ell_two():
    sched = eta_schedule(K=200, ell=2, eta_final=0.05, eps=0.01)
    assert float(sched[-1]) == pytest.approx(0.05, rel=1e-4)
    assert jnp.all(jnp.diff(sched) <= 0)


def test_eta_schedule_traced_eta_no_window_needed():
    # The adaptive sampler needs no static window: a schedule entry (traced eta)
    # drives a jitted draw directly, including the largest (first) eta.
    sched = eta_schedule(K=100, ell=1, eta_final=0.05)
    d = dist.Poisson(jnp.array(4.0))
    val = jax.jit(
        lambda eta: SmoothedDiscrete(d, eta).sample(jax.random.key(0), (16,)).mean()
    )(sched[0])
    assert jnp.isfinite(val)


# ---------------------------------------------------------------------------
# Integration: model with Geometric and Poisson latents
# ---------------------------------------------------------------------------


def test_geometric_latent_model():
    def model(obs):
        k = numpyro.sample("k", dist.GeometricProbs(jnp.array(0.3)))
        numpyro.sample("obs", dist.Normal(k.astype(float), 1.0), obs=obs)

    smoothed = dsgd(model)
    with handlers.seed(rng_seed=5):
        trace = handlers.trace(lambda eta: smoothed(eta, jnp.array(2.0))).get_trace(ETA)
    k_fn = trace["k"]["fn"]
    assert isinstance(k_fn, SmoothedCount)


def test_poisson_latent_model():
    def model(obs):
        k = numpyro.sample("k", dist.Poisson(jnp.array(5.0)))
        numpyro.sample("obs", dist.Normal(k.astype(float), 1.0), obs=obs)

    smoothed = dsgd(model)
    with handlers.seed(rng_seed=6):
        trace = handlers.trace(lambda eta: smoothed(eta, jnp.array(4.0))).get_trace(ETA)
    k_fn = trace["k"]["fn"]
    assert isinstance(k_fn, SmoothedCount)


def test_negbin_latent_model():
    def model(obs):
        k = numpyro.sample(
            "k", dist.NegativeBinomialProbs(total_count=3, probs=jnp.array(0.5))
        )
        numpyro.sample("obs", dist.Normal(k.astype(float), 1.0), obs=obs)

    smoothed = dsgd(model)
    with handlers.seed(rng_seed=7):
        trace = handlers.trace(lambda eta: smoothed(eta, jnp.array(3.0))).get_trace(ETA)
    k_fn = trace["k"]["fn"]
    assert isinstance(k_fn, SmoothedCount)


def test_jit_compatible_smoothed_model():
    def model(obs):
        z = numpyro.sample("z", dist.BernoulliProbs(jnp.array(0.5)))
        numpyro.sample("obs", dist.Normal(z.astype(float), 1.0), obs=obs)

    smoothed = dsgd(model)

    @jit
    def run(eta):
        with handlers.seed(rng_seed=0):
            handlers.trace(lambda: smoothed(eta, jnp.array(0.0))).get_trace()

    run(jnp.array(ETA))  # should not error


# ---------------------------------------------------------------------------
# Adaptive index-space smooth iCDF (grid-free path for unbounded families)
# ---------------------------------------------------------------------------


# (name, factory, kmax).  Factories build distributions lazily so no JAX arrays
# are created at import time (keeps test/conftest.py's live-array check happy).
_UNBOUNDED_CASES = [
    ("poisson", lambda: dist.Poisson(jnp.array(3.0)), 200),
    ("poisson_high", lambda: dist.Poisson(jnp.array(90.0)), 700),
    ("geometric", lambda: dist.GeometricProbs(jnp.array(0.5)), 200),
    ("geometric_logits", lambda: dist.GeometricLogits(jnp.array(0.3)), 200),
    ("negbin", lambda: dist.NegativeBinomialProbs(5, jnp.array(0.4)), 300),
    ("gammapoisson", lambda: dist.GammaPoisson(jnp.array(5.0), jnp.array(1.5)), 300),
    ("zip", lambda: dist.ZeroInflatedPoisson(jnp.array(0.3), jnp.array(4.0)), 200),
    (
        "zinb",
        lambda: dist.ZeroInflatedDistribution(
            dist.NegativeBinomialProbs(4, jnp.array(0.5)), gate=jnp.array(0.4)
        ),
        300,
    ),
]


class TestAdaptiveTransform:
    @pytest.mark.parametrize("name, make, kmax", _UNBOUNDED_CASES)
    @pytest.mark.parametrize("eta", [0.3, 0.05])
    def test_forward_monotone(self, name, make, kmax, eta):
        # Q_eta(u) is monotone non-decreasing in u (a valid quantile map).
        u = jnp.linspace(0.02, 0.98, 60)
        z = adaptive_relaxed_count(make(), u, eta)
        assert jnp.all(jnp.diff(z) >= -1e-5)
        assert jnp.all(z >= -1e-4)


class TestAnchoredTransform:
    @pytest.mark.parametrize(
        "make, value",
        [
            (lambda x: dist.Poisson(x), 4.0),
            (lambda x: dist.GeometricProbs(x), 0.4),
            (lambda x: dist.GammaPoisson(3.0, x), 1.5),
            (lambda x: dist.ZeroInflatedPoisson(0.3, x), 4.0),
            (
                lambda x: dist.ZeroInflatedDistribution(
                    dist.GeometricProbs(x), gate=0.3
                ),
                0.4,
            ),
            (
                lambda x: dist.ZeroInflatedDistribution(
                    dist.GammaPoisson(3.0, x), gate=0.3
                ),
                1.5,
            ),
        ],
    )
    @pytest.mark.parametrize("anchor", ["binary", "cornish-fisher"])
    def test_values_and_parameter_gradients_are_finite(self, make, value, anchor):
        u = jnp.linspace(0.05, 0.95, 17)

        def total(parameter):
            return anchored_relaxed_count(make(parameter), u, ETA, anchor=anchor).sum()

        parameter = jnp.array(value)
        z = anchored_relaxed_count(make(parameter), u, ETA, anchor=anchor)
        assert z.shape == u.shape
        assert jnp.all(jnp.isfinite(z))
        assert jnp.isfinite(grad(total)(parameter))

    def test_binary_is_default(self):
        d = dist.Poisson(jnp.array(4.0))
        u = jnp.linspace(0.05, 0.95, 17)
        assert_allclose(
            anchored_relaxed_count(d, u, ETA),
            anchored_relaxed_count(d, u, ETA, anchor="binary"),
        )
        assert SmoothedDiscrete(d, ETA).anchor == "binary"

    def test_invalid_anchor_rejected(self):
        d = dist.Poisson(jnp.array(4.0))
        with pytest.raises(ValueError, match="anchor must be one of"):
            SmoothedDiscrete(d, ETA, anchor="normal")

    def test_dsgd_threads_anchor_to_count_sites(self):
        def model():
            numpyro.sample("z", dist.Poisson(jnp.array(4.0)))

        smoothed = dsgd(model, anchor="cornish-fisher")
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(lambda: smoothed(ETA)).get_trace()
        assert trace["z"]["fn"].anchor == "cornish-fisher"


class TestSmoothedDiscreteRouting:
    def test_unbounded_uses_smoothed_count(self):
        # Unbounded families use SmoothedCount (adaptive relaxed-count sample +
        # analytic-continuation log-pmf density), not the grid pushforward.
        for d in (
            dist.Poisson(jnp.array(4.0)),
            dist.GeometricProbs(jnp.array(0.4)),
            dist.NegativeBinomialProbs(total_count=3, probs=jnp.array(0.5)),
            dist.ZeroInflatedPoisson(jnp.array(0.3), jnp.array(4.0)),
        ):
            sd = SmoothedDiscrete(d, ETA)
            assert isinstance(sd, SmoothedCount)

    def test_max_support_rejected_for_unbounded(self):
        # max_support has no meaning for unbounded families and is rejected.
        d = dist.Poisson(jnp.array(4.0))
        with pytest.raises(ValueError, match="max_support is not supported"):
            SmoothedDiscrete(d, ETA, max_support=30)

    def test_bounded_uses_grid(self):
        d = dist.BernoulliProbs(jnp.array(0.6))
        sd = SmoothedDiscrete(d, ETA)
        assert isinstance(sd.transforms[0], SmoothICDFTransform)

    def test_bounded_accepts_max_support(self):
        d = dist.BinomialProbs(total_count=10, probs=jnp.array(0.5))
        sd = SmoothedDiscrete(d, ETA, max_support=11)
        assert isinstance(sd.transforms[0], SmoothICDFTransform)


class TestToEventSmoothing:
    """`to_event(n)` wraps a discrete site in Independent and `plate` / `.expand`
    wraps it in ExpandedDistribution; DSGD must peel these off, smooth the base
    elementwise, and re-apply them (preserving event dims and batch expansion)."""

    def test_smootheddiscrete_accepts_expanded_to_event(self):
        # Poisson(3,) -> to_event(1) [event (3,)] -> expand((4,)) [batch (4,)]
        d = dist.Poisson(jnp.array([2.0, 5.0, 8.0])).to_event(1).expand((4,))
        sd = SmoothedDiscrete(d, 0.2)
        assert sd.batch_shape == (4,)
        assert sd.event_shape == (3,)
        z = sd.sample(jax.random.key(0))
        assert z.shape == (4, 3)
        assert sd.log_prob(z).shape == (4,)
        assert jnp.all(jnp.isfinite(sd.log_prob(z)))

    def test_expanded_to_event_model_smoothed(self):
        # Mirrors the PVAE model: base rate shape (z_dim,), to_event(1), then a
        # plate expands the batch dim -> ExpandedDistribution(Independent(...)).
        def model():
            with numpyro.plate("batch", 5):
                numpyro.sample("z", dist.Poisson(jnp.full(3, 2.0)).to_event(1))

        smoothed = dsgd(model)
        with handlers.seed(rng_seed=0):
            tr = handlers.trace(lambda e: smoothed(e)).get_trace(0.2)
        zfn = tr["z"]["fn"]
        assert tr["z"]["value"].shape == (5, 3)
        lp = zfn.log_prob(tr["z"]["value"])
        assert lp.shape == (5,)  # event dim summed, plate/batch dim kept
        assert jnp.all(jnp.isfinite(lp))

    def test_smootheddiscrete_accepts_independent(self):
        d = dist.Poisson(jnp.array([2.0, 5.0, 8.0])).to_event(1)
        sd = SmoothedDiscrete(d, 0.2)
        assert isinstance(sd, dist.Independent)
        assert sd.event_shape == (3,)
        z = sd.sample(jax.random.key(0))
        assert z.shape == (3,)
        assert sd.log_prob(z).shape == ()  # event dim summed out
        assert jnp.isfinite(sd.log_prob(z))

    def test_to_event_model_smoothed(self):
        def model():
            with numpyro.plate("batch", 4):
                numpyro.sample("z", dist.Poisson(jnp.full((4, 3), 2.0)).to_event(1))

        smoothed = dsgd(model)
        with handlers.seed(rng_seed=0):
            tr = handlers.trace(lambda e: smoothed(e)).get_trace(0.2)
        zfn = tr["z"]["fn"]
        assert isinstance(zfn, dist.Independent)
        assert zfn.event_shape == (3,)
        assert tr["z"]["value"].shape == (4, 3)

    def test_to_event_model_straight_through(self):
        def model():
            with numpyro.plate("batch", 4):
                numpyro.sample("z", dist.Poisson(jnp.full((4, 3), 2.0)).to_event(1))

        smoothed = dsgd(model, smoothed_distributions=False)
        with handlers.seed(rng_seed=1):
            tr = handlers.trace(lambda e: smoothed(e)).get_trace(0.2)
        zfn = tr["z"]["fn"]
        assert isinstance(zfn, dist.Independent)
        z = tr["z"]["value"]
        assert z.shape == (4, 3)
        assert_allclose(z, jnp.round(z), atol=1e-5)  # straight-through integers

    def test_to_event_reparam_gradient(self):
        # gradient of an objective w.r.t. the rate flows through the to_event
        # smoothed sample.
        def mean_z(rate):
            d = dist.Poisson(rate * jnp.ones(3)).to_event(1)
            z = SmoothedDiscrete(d, 0.2).sample(jax.random.key(0))
            return z.sum()

        g = grad(mean_z)(jnp.array(4.0))
        assert jnp.isfinite(g) and g > 0


class TestAdaptiveSampling:
    @pytest.mark.parametrize(
        "make",
        [
            lambda: dist.Poisson(jnp.array(4.0)),
            lambda: dist.GeometricProbs(jnp.array(0.5)),
            lambda: dist.NegativeBinomialProbs(5, jnp.array(0.4)),
            lambda: dist.ZeroInflatedPoisson(jnp.array(0.3), jnp.array(4.0)),
        ],
    )
    def test_sample_mean_matches_true_mean(self, make):
        d = make()
        sd = SmoothedDiscrete(d, 0.1)
        samples = sd.sample(jax.random.key(0), (4000,))
        assert jnp.all(samples >= -1.0)
        assert_allclose(float(samples.mean()), float(d.mean), atol=0.25)

    def test_eta_to_zero_converges(self):
        # Smoothed mean → true mean as η shrinks (bias vanishes).
        d = dist.Poisson(jnp.array(4.0))
        key = jax.random.key(1)
        errs = []
        for eta in [0.5, 0.2, 0.1]:
            z = SmoothedDiscrete(d, eta).sample(key, (4000,))
            errs.append(abs(float(z.mean()) - 4.0))
        assert errs[-1] < 0.15

    def test_log_prob_is_analytic_continuation_pmf(self):
        # Unbounded families now use SmoothedCount: log_prob is the analytic
        # continuation of the discrete log-pmf (k! -> Gamma(z+1)) evaluated at the
        # continuous relaxed count -- NOT the -log|dz/du| pushforward density,
        # which is unbounded above for near-deterministic distributions and makes
        # log p - log q diverge.
        from jax.scipy.special import gammaln

        rate = jnp.array(4.0)
        sd = SmoothedDiscrete(dist.Poisson(rate), 0.2)
        assert isinstance(sd, SmoothedCount)
        z = sd.sample(jax.random.key(2), (50,))
        lp = sd.log_prob(z)
        expected = z * jnp.log(rate) - rate - gammaln(z + 1.0)
        assert jnp.all(jnp.isfinite(lp))
        assert_allclose(lp, expected, atol=1e-5)

    def test_kl_term_is_proper_and_bounded(self):
        # Core property of the analytic-continuation density: the ELBO term
        # E_q[log p(z) - log q(z)] is a valid -KL (<= 0) and bounded, matching the
        # exact discrete -KL -- even for a near-deterministic (low-rate) model,
        # where the old -log|dz/du| pushforward density diverged to +inf and made
        # the smoothed ELBO improper.
        key = jax.random.key(7)
        lam_q = 1.0
        for lam_p in [1.0, 0.5, 0.1, 0.008]:
            q = SmoothedDiscrete(dist.Poisson(jnp.array(lam_q)), 0.3)
            p = SmoothedDiscrete(dist.Poisson(jnp.array(lam_p)), 0.3)
            z = q.sample(key, (20000,))
            kl_term = float((p.log_prob(z) - q.log_prob(z)).mean())
            true_neg_kl = -(lam_q * (jnp.log(lam_q) - jnp.log(lam_p)) - lam_q + lam_p)
            assert kl_term <= 1e-2  # proper: -KL <= 0 (small tol for MC noise)
            assert_allclose(kl_term, float(true_neg_kl), atol=0.15)


class TestAdaptiveGradients:
    def test_reparam_grad_of_mean_is_one(self):
        # E_smooth[z] = rate exactly for Poisson, so d/d(rate) E[z] = 1 for any η.
        def mean_obj(rate, key):
            sd = SmoothedDiscrete(dist.Poisson(rate), 0.2)
            return sd.sample(key, (8000,)).mean()

        g = grad(mean_obj)(jnp.array(4.0), jax.random.key(3))
        assert_allclose(float(g), 1.0, atol=0.1)

    def test_reparam_grad_low_variance_small_eta(self):
        # The index-space estimator keeps the reparam gradient unbiased and
        # bounded-variance even at small η (unlike the old value-space graft,
        # whose dz/drate variance exploded and whose mean collapsed to ~0).
        rate = jnp.array(3.0)
        keys = jax.random.split(jax.random.key(5), 2000)

        def z_of(key, r):
            u = jax.random.uniform(key)
            return adaptive_relaxed_count(dist.Poisson(r), u, 0.05)

        g = jax.vmap(lambda k: grad(lambda r: z_of(k, r))(rate))(keys)
        assert_allclose(float(g.mean()), 1.0, atol=0.1)  # unbiased
        assert float(g.std()) < 1.0  # bounded variance

    def test_param_gradient_finite(self):
        for d_fn, theta in [
            (dist.Poisson, jnp.array(4.0)),
            (lambda p: dist.GeometricProbs(p), jnp.array(0.4)),
        ]:

            def loss(th):
                sd = SmoothedDiscrete(d_fn(th), 0.2)
                z = sd.sample(jax.random.key(4), (256,))
                return jnp.mean(jnp.exp(-0.5 * z))

            g = grad(loss)(theta)
            assert jnp.isfinite(g)


def test_adaptive_jit_compatible():
    d = dist.Poisson(jnp.array(5.0))

    @jit
    def draw(key):
        return SmoothedDiscrete(d, 0.2).sample(key, (100,)).mean()

    val = draw(jax.random.key(0))
    assert jnp.isfinite(val)


class TestAdaptiveStraightThrough:
    """smoothed_distributions=False on an unbounded latent uses the grid-free
    adaptive index-space sampler: integer forward values, smooth gradients."""

    def test_forward_is_integer(self):
        def model(obs):
            k = numpyro.sample("k", dist.Poisson(jnp.array(4.0)))
            numpyro.sample("obs", dist.Normal(k, 1.0), obs=obs)

        smoothed = dsgd(model, smoothed_distributions=False)
        with handlers.seed(rng_seed=0):
            trace = handlers.trace(lambda e: smoothed(e, jnp.array(3.0))).get_trace(0.2)
        k = trace["k"]["value"]
        # straight-through forward pass is rounded to integers
        assert_allclose(k, jnp.round(k), atol=1e-5)
        # site fn is the straight-through wrapper over the original Poisson
        assert isinstance(trace["k"]["fn"], StraightThroughSmoothed)
        assert isinstance(trace["k"]["fn"].base_dist, dist.Poisson)

    def test_gradient_flows_through_st(self):
        # Gradient w.r.t. the rate flows through the straight-through sample. The
        # forward value is rounded, but the gradient is that of the smooth
        # quantile, dz/d(rate) > 0 (higher rate => larger quantile).
        def sampled_k(theta):
            def model():
                numpyro.sample("k", dist.Poisson(theta))

            smoothed = dsgd(model, smoothed_distributions=False)
            with handlers.seed(rng_seed=1):
                tr = handlers.trace(lambda e: smoothed(e)).get_trace(0.2)
            return tr["k"]["value"].sum()

        g = grad(sampled_k)(jnp.array(3.0))
        assert jnp.isfinite(g)
        assert g > 0.0  # straight-through keeps the rate in the graph


class TestAdaptiveTracedEta:
    """The adaptive sampler discovers its horizon with a while_loop, so a traced
    eta works directly (jit / scan over eta) with no static window required."""

    def test_traced_eta_transform_works(self):
        d = dist.Poisson(jnp.array(4.0))

        def f(eta):
            return adaptive_relaxed_count(d, jnp.array([0.3, 0.6]), eta)

        out = jax.jit(f)(jnp.array(0.2))
        assert jnp.all(jnp.isfinite(out))

    def test_traced_eta_jit_sampling(self):
        d = dist.Poisson(jnp.array(4.0))

        def f(eta):
            return SmoothedDiscrete(d, eta).sample(jax.random.key(0), (64,)).mean()

        val = jax.jit(f)(jnp.array(0.2))
        assert jnp.isfinite(val)

    def test_scan_over_eta_schedule(self):
        # A jitted step indexing an eta-schedule (traced eta) must not retrace.
        d = dist.Poisson(jnp.array(4.0))
        schedule = eta_schedule(K=6, ell=1, eta_final=0.05)

        @jit
        def step(i, key):
            return SmoothedDiscrete(d, schedule[i]).sample(key, (32,)).mean()

        for i in range(6):
            assert jnp.isfinite(step(i, jax.random.key(i)))


def test_dsgd_end_to_end_minimal():
    # Minimal end-to-end smoke test of the full pipeline (dsgd + count_layers +
    # eta_schedule + seed/trace + adaptive reparam): the pathwise gradient of
    # E[k] w.r.t. the Poisson rate is 1, since E_smooth[k] = rate exactly.
    def model(rate):
        numpyro.sample("k", dist.Poisson(rate))

    smoothed = dsgd(model)
    ell = count_layers(model, jnp.array(4.0))
    assert ell == 1  # one discrete latent
    schedule = eta_schedule(K=10, ell=ell, eta_final=0.1)
    eta = float(schedule[-1])  # concrete temperature from the schedule

    def mean_k(rate, key):
        with handlers.seed(rng_seed=key):
            tr = handlers.trace(lambda: smoothed(eta, rate)).get_trace()
        return tr["k"]["value"]

    keys = jax.random.split(jax.random.key(0), 4000)
    g = jnp.mean(jax.vmap(lambda key: grad(mean_k)(jnp.array(4.0), key))(keys))
    assert_allclose(float(g), 1.0, atol=0.1)


@pytest.mark.parametrize("rate", [0.0, 1e-45, 1.7e-5])
def test_count_log_pmf_poisson_vanishing_rate(rate):
    # ``value * log(rate)`` is 0 * -inf = NaN at k = 0 once the rate underflows
    # to zero, which happens in float32 for the sparse count fields this is
    # used on.  ``xlogy`` takes the correct 0 limit.
    d = dist.Poisson(jnp.asarray(rate))
    counts = jnp.asarray([0.0, 1.0, 2.0])

    assert jnp.isfinite(_count_log_pmf(d, counts)[0])
    assert_allclose(_count_log_pmf(d, counts), d.log_prob(counts), atol=1e-6)

    spec = _idx_spec(d)
    assert jnp.isfinite(spec.log_pmf(counts, spec.params)[0])


COUNT_FAMILIES = {
    "Poisson": lambda: dist.Poisson(jnp.asarray(3.5)),
    "Geometric": lambda: dist.GeometricProbs(jnp.asarray(0.3)),
    "GammaPoisson": lambda: dist.GammaPoisson(jnp.asarray(2.0), jnp.asarray(1.5)),
    "ZeroInflatedPoisson": lambda: dist.ZeroInflatedPoisson(
        jnp.asarray(0.2), jnp.asarray(3.5)
    ),
}


@pytest.mark.parametrize("family", sorted(COUNT_FAMILIES))
def test_count_log_pmf_matches_distribution_log_prob(family):
    # _count_log_pmf delegates rather than restating each formula; pin that so
    # the two cannot drift apart again.
    base_dist = COUNT_FAMILIES[family]()
    counts = jnp.arange(12.0)
    assert_allclose(
        _count_log_pmf(base_dist, counts), base_dist.log_prob(counts), atol=1e-5
    )


@pytest.mark.parametrize("family", ["Poisson", "GammaPoisson"])
def test_count_log_pmf_broadcasts_over_recurrence_axis(family):
    # Shape the recurrence actually uses: sample axes, then batch, then window.
    batched = {
        "Poisson": lambda: dist.Poisson(jnp.asarray([2.0, 3.0, 4.0])),
        "GammaPoisson": lambda: dist.GammaPoisson(
            jnp.asarray([2.0, 3.0, 4.0]), jnp.asarray([1.5, 1.0, 0.5])
        ),
    }[family]()
    counts = jnp.arange(5 * 3 * 7, dtype=jnp.result_type(float)).reshape(5, 3, 7) % 9.0
    actual = _count_log_pmf(batched, counts, trailing_ndims=1)
    assert actual.shape == counts.shape
    # Each window slot is the plain log_prob at that count, with the parameters
    # still aligned to the batch axis rather than to the window.
    for slot in range(counts.shape[-1]):
        assert_allclose(
            actual[..., slot], batched.log_prob(counts[..., slot]), atol=1e-5
        )


def test_idx_spec_zero_inflated_smooths_the_indicator():
    # The zero-inflated _idx_spec continuation is deliberately NOT log_prob: it
    # replaces 1[k == 0] with a smooth bump so the density term is defined at
    # non-integer relaxed counts.  It must still agree on the integers.
    base_dist = dist.ZeroInflatedPoisson(jnp.asarray(0.3), jnp.asarray(2.0))
    spec = _idx_spec(base_dist)

    integers = jnp.arange(6.0)
    assert_allclose(
        spec.log_pmf(integers, spec.params), base_dist.log_prob(integers), atol=1e-5
    )

    between = jnp.asarray([0.4, 0.8])
    continued = _unvalidated_log_prob(base_dist, between)
    assert jnp.all(jnp.abs(spec.log_pmf(between, spec.params) - continued) > 0.1)


@pytest.mark.parametrize("family", sorted(COUNT_FAMILIES))
def test_count_sf_matches_one_minus_cdf(family):
    # Where both are representable the two must agree; the survival function
    # earns its keep only past that point.
    base_dist = COUNT_FAMILIES[family]()
    counts = jnp.arange(10.0)
    assert_allclose(
        _count_sf(base_dist, counts), 1.0 - _count_cdf(base_dist, counts), atol=1e-6
    )


def test_count_sf_gamma_poisson_shape_gradients_are_finite():
    # The survival orientation puts the concentration in betainc's second shape
    # slot, which needs its own custom JVP.
    counts = jnp.arange(5.0)

    def by_concentration(concentration):
        return jnp.sum(
            _count_sf(dist.GammaPoisson(concentration, jnp.asarray(1.5)), counts)
        )

    def by_rate(rate):
        return jnp.sum(_count_sf(dist.GammaPoisson(jnp.asarray(2.0), rate), counts))

    assert jnp.isfinite(grad(by_concentration)(jnp.asarray(2.0)))
    assert jnp.isfinite(grad(by_rate)(jnp.asarray(1.5)))


def test_anchored_relaxed_count_gradient_survives_cdf_saturation():
    # Below rate ~1e-7 the float32 Poisson CDF at k = 0 rounds to exactly 1.0.
    # It is then a constant, and the relaxation used to lose half its gradient
    # to it -- silently, with a finite value and a finite derivative.
    uniforms = jax.random.uniform(jax.random.key(0), (4096,))

    def mean_relaxed_count(log_rate):
        base_dist = dist.Poisson(jnp.exp(log_rate))
        return jnp.mean(anchored_relaxed_count(base_dist, uniforms, 0.1, width=8))

    # d E[k] / d log(rate) is rate, up to the fixed smoothing bias of eta, so
    # the normalised gradient must not move across the saturation boundary.
    def normalised(rate):
        argument = jnp.log(jnp.asarray(rate))
        return float(grad(mean_relaxed_count)(argument)) / rate

    assert _count_cdf(dist.Poisson(jnp.asarray(1e-8)), jnp.asarray(0.0)) == 1.0
    unsaturated = normalised(1e-6)
    for rate in (1e-8, 1e-10):
        assert_allclose(normalised(rate), unsaturated, rtol=1e-3)


@pytest.mark.parametrize(
    "rate, max_count, truncates",
    [(8.0, 256, False), (30.0, 16, True), (64.0, 16, True)],
)
def test_count_anchor_saturates_tracks_truncation(rate, max_count, truncates):
    # The binary anchor searches [0, max_count], so a draw above that bound
    # cannot be anchored and the relaxed count pins near max_count + width.
    # The predicate has to agree with what actually happens.
    base_dist = dist.Poisson(jnp.asarray(rate))
    uniforms = jax.random.uniform(jax.random.key(0), (4000,))

    saturating = jnp.mean(count_anchor_saturates(base_dist, uniforms, max_count))
    relaxed = jnp.mean(
        anchored_relaxed_count(base_dist, uniforms, 0.1, width=8, max_count=max_count)
    )
    if truncates:
        assert saturating > 0.5
        assert relaxed < 0.9 * rate  # pinned well below the true mean
    else:
        assert saturating == 0.0
        assert_allclose(relaxed, rate, rtol=0.02)


def test_count_anchor_saturates_worst_case_draw():
    # The documented idiom: probe the bound with a near-one draw rather than
    # hoping the sampled u never reaches it.
    worst_case = jnp.asarray([1.0 - 1e-6])
    assert bool(count_anchor_saturates(dist.Poisson(jnp.asarray(64.0)), worst_case, 16))
    assert not bool(
        count_anchor_saturates(dist.Poisson(jnp.asarray(64.0)), worst_case, 256)
    )


def test_count_anchor_saturates_rejects_nonpositive_bound():
    with pytest.raises(ValueError, match="max_count must be positive"):
        count_anchor_saturates(dist.Poisson(jnp.asarray(1.0)), jnp.asarray([0.5]), 0)
