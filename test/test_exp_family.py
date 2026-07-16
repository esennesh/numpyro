# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Exponential-family interface tests: Monte Carlo consistency of ``mean_params``
with ``sufficient_statistics``, moment-matching round trips, canonical-params
rebuilds, wrapper unwrapping, and exact conjugate posterior recovery via
importance-weighted moment matching (a one-site QEM E/M step).
"""

import numpy as np
import pytest

import jax
from jax import random
import jax.numpy as jnp

import numpyro.distributions as dist
from numpyro.distributions.exp_family import (
    base_distribution,
    canonical_params,
    from_mean_params,
    is_exp_family,
    mean_params,
    sufficient_statistics,
)

jax.config.update("jax_enable_x64", True)


def _mvn():
    cov = jnp.array([[2.0, 0.6], [0.6, 1.0]])
    return dist.MultivariateNormal(loc=jnp.array([0.5, -1.0]), covariance_matrix=cov)


# factories, not instances: the test suite requires no live jax arrays at collection
EXAMPLE_DISTS = [
    lambda: dist.Normal(1.5, 2.0),
    lambda: dist.Normal(jnp.array([0.0, 1.0]), jnp.array([1.0, 2.0])),
    _mvn,
    lambda: dist.BernoulliProbs(0.3),
    lambda: dist.BernoulliLogits(jnp.array([-0.5, 0.7])),
    lambda: dist.CategoricalProbs(jnp.array([0.2, 0.3, 0.5])),
    lambda: dist.CategoricalLogits(jnp.array([0.1, -0.4, 1.2])),
    lambda: dist.Poisson(3.5),
    lambda: dist.Exponential(2.0),
    lambda: dist.LogNormal(1.5, 0.8),
    lambda: dist.Normal(jnp.zeros(3), jnp.ones(3)).to_event(1),
    lambda: dist.Normal(0.5, 1.5).expand((4, 2)),
    lambda: dist.Exponential(jnp.array([1.0, 3.0])).expand((5, 2)).to_event(1),
]

EXAMPLE_IDS = [
    "Normal",
    "NormalBatched",
    "MultivariateNormal",
    "BernoulliProbs",
    "BernoulliLogits",
    "CategoricalProbs",
    "CategoricalLogits",
    "Poisson",
    "Exponential",
    "LogNormal",
    "IndependentNormal",
    "ExpandedNormal",
    "ExpandedIndependentExponential",
]


@pytest.mark.parametrize("make_dist", EXAMPLE_DISTS, ids=EXAMPLE_IDS)
def test_mean_params_match_monte_carlo(make_dist):
    d = make_dist()
    num_samples = 200_000
    samples = d.sample(random.PRNGKey(0), (num_samples,))
    stats = sufficient_statistics(d, samples)
    means = mean_params(d)
    assert set(stats) == set(means)
    for name, m in means.items():
        mc = jnp.mean(stats[name], axis=0)
        assert m.shape == d.batch_shape + stats[name].shape[1 + len(d.batch_shape) :]
        np.testing.assert_allclose(mc, jnp.broadcast_to(m, mc.shape), atol=0.05)


@pytest.mark.parametrize("make_dist", EXAMPLE_DISTS, ids=EXAMPLE_IDS)
def test_from_mean_params_round_trip(make_dist):
    d = make_dist()
    rebuilt = from_mean_params(d, mean_params(d))
    assert type(base_distribution(rebuilt)) is type(base_distribution(d))
    assert rebuilt.event_dim == d.event_dim
    round_trip = mean_params(rebuilt)
    for name, m in mean_params(d).items():
        np.testing.assert_allclose(
            round_trip[name],
            jnp.broadcast_to(m, round_trip[name].shape),
            rtol=1e-6,
            atol=1e-8,
        )
    # log_prob agreement on a probe point
    probe = d.sample(random.PRNGKey(1))
    np.testing.assert_allclose(
        rebuilt.log_prob(probe), d.log_prob(probe), rtol=1e-6, atol=1e-8
    )


@pytest.mark.parametrize("make_dist", EXAMPLE_DISTS, ids=EXAMPLE_IDS)
def test_canonical_params_rebuild(make_dist):
    d = make_dist()
    base = base_distribution(d)
    rebuilt = type(base)(**canonical_params(d))
    probe = base.sample(random.PRNGKey(2))
    np.testing.assert_allclose(rebuilt.log_prob(probe), base.log_prob(probe), rtol=1e-6)


def test_is_exp_family():
    assert is_exp_family(dist.Normal(0.0, 1.0))
    assert is_exp_family(dist.Normal(jnp.zeros(2), 1.0).to_event(1).expand((3, 2)))
    assert not is_exp_family(dist.StudentT(3.0))
    with pytest.raises(NotImplementedError, match="exponential-family"):
        mean_params(dist.StudentT(3.0))
    with pytest.raises(NotImplementedError, match="exponential-family"):
        sufficient_statistics(dist.StudentT(3.0), jnp.zeros(()))
    with pytest.raises(NotImplementedError, match="exponential-family"):
        from_mean_params(dist.StudentT(3.0), {"x": jnp.zeros(())})


def test_conjugate_normal_normal_recovery():
    """Importance-weighted moment matching recovers the exact conjugate posterior.

    z ~ N(mu0, tau0); x | z ~ N(z, sigma). With K prior samples weighted by the
    likelihood, the weighted sufficient statistics converge to the analytic
    posterior's mean parameters, so ``from_mean_params`` recovers the posterior.
    """
    mu0, tau0, sigma, x_obs = 0.5, 1.2, 0.8, 2.0
    post_var = 1.0 / (1.0 / tau0**2 + 1.0 / sigma**2)
    post_mean = post_var * (mu0 / tau0**2 + x_obs / sigma**2)

    prior = dist.Normal(mu0, tau0)
    K = 400_000
    z = prior.sample(random.PRNGKey(3), (K,))
    log_w = dist.Normal(z, sigma).log_prob(x_obs)
    w = jax.nn.softmax(log_w)

    stats = sufficient_statistics(prior, z)
    m_hat = {name: jnp.sum(w * t, axis=0) for name, t in stats.items()}
    posterior = from_mean_params(prior, m_hat)

    assert posterior.loc == pytest.approx(post_mean, abs=5e-3)
    assert posterior.scale == pytest.approx(np.sqrt(post_var), abs=5e-3)


def test_conjugate_mvn_recovery():
    """Same conjugate check for a 2D Gaussian with an identity-observation likelihood."""
    prior = _mvn()
    obs_cov = 0.5 * jnp.eye(2)
    x_obs = jnp.array([1.0, 0.0])
    prec = jnp.linalg.inv(prior.covariance_matrix) + jnp.linalg.inv(obs_cov)
    post_cov = jnp.linalg.inv(prec)
    post_mean = post_cov @ (
        jnp.linalg.inv(prior.covariance_matrix) @ prior.loc
        + jnp.linalg.inv(obs_cov) @ x_obs
    )

    K = 400_000
    z = prior.sample(random.PRNGKey(4), (K,))
    log_w = dist.MultivariateNormal(z, obs_cov).log_prob(x_obs)
    w = jax.nn.softmax(log_w)

    stats = sufficient_statistics(prior, z)
    m_hat = {
        name: jnp.sum(w.reshape((K,) + (1,) * (t.ndim - 1)) * t, axis=0)
        for name, t in stats.items()
    }
    posterior = from_mean_params(prior, m_hat)

    np.testing.assert_allclose(posterior.loc, post_mean, atol=2e-2)
    np.testing.assert_allclose(posterior.covariance_matrix, post_cov, atol=2e-2)
