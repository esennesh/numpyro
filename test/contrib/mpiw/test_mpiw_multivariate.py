# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
MPIW with a multivariate (event_dim > 0) guide site. Exercises the event-aware
squeezing of filler axes: a model with both a scalar latent and a MultivariateNormal
latent, so the MVN site's samples carry a size-1 filler axis (from the scalar's K
dimension) *and* a trailing event axis that must be preserved.

The two latents are independent given the data, so each posterior is a closed-form
Gaussian.
"""

import numpy as np
import pytest

import jax
from jax import random
import jax.numpy as jnp

pytest.importorskip("funsor")

import numpyro  # noqa: E402
from numpyro.contrib.mpiw import MPIW  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

jax.config.update("jax_enable_x64", True)

D = 2
YS_OBS = 0.4  # scalar observation for s
X_OBS = np.array([1.0, -0.5])  # vector observation for mu
S_LIK, MU_LIK = 0.5, 0.5  # likelihood variances


def _model():
    s = numpyro.sample("s", dist.Normal(0.0, 1.0))
    numpyro.sample("ys", dist.Normal(s, jnp.sqrt(S_LIK)), obs=jnp.array(YS_OBS))
    mu = numpyro.sample("mu", dist.MultivariateNormal(jnp.zeros(D), jnp.eye(D)))
    numpyro.sample(
        "x", dist.MultivariateNormal(mu, MU_LIK * jnp.eye(D)), obs=jnp.asarray(X_OBS)
    )


def _guide():
    numpyro.sample("s", dist.Normal(0.2, 1.0))
    numpyro.sample(
        "mu", dist.MultivariateNormal(jnp.array([0.3, 0.0]), 1.5 * jnp.eye(D))
    )


def _analytic():
    # s: prior N(0,1), lik N(s, S_LIK)
    s_var = 1.0 / (1.0 + 1.0 / S_LIK)
    s_mean = s_var * (YS_OBS / S_LIK)
    # mu: prior N(0, I), lik N(mu, MU_LIK I)
    post_cov = np.linalg.inv(np.eye(D) + np.linalg.inv(MU_LIK * np.eye(D)))
    mu_mean = post_cov @ np.linalg.inv(MU_LIK * np.eye(D)) @ X_OBS
    return s_mean, mu_mean


def test_multivariate_moments_and_shapes():
    s_mean, mu_mean = _analytic()
    mpiw = MPIW(_model, _guide, num_samples=6000)

    # the MVN site's value keeps its event axis; the filler axis is squeezed away
    values, weights = mpiw.site_weights(random.PRNGKey(0))["mu"]
    assert values.shape == (6000, D)
    assert weights.shape == (6000,)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-6)

    m = mpiw.moments(random.PRNGKey(0), {"s": lambda v: v, "mu": lambda v: v})
    assert float(np.array(m["s"])) == pytest.approx(s_mean, abs=0.03)
    assert np.allclose(np.array(m["mu"]), mu_mean, atol=0.05)


def test_lone_multivariate_site():
    # a single global MVN latent (no filler) should also give the right posterior mean
    def model():
        mu = numpyro.sample("mu", dist.MultivariateNormal(jnp.zeros(D), jnp.eye(D)))
        numpyro.sample(
            "x",
            dist.MultivariateNormal(mu, MU_LIK * jnp.eye(D)),
            obs=jnp.asarray(X_OBS),
        )

    def guide():
        numpyro.sample(
            "mu", dist.MultivariateNormal(jnp.array([0.3, 0.0]), 1.5 * jnp.eye(D))
        )

    _, mu_mean = _analytic()
    mpiw = MPIW(model, guide, num_samples=6000)
    m = mpiw.moments(random.PRNGKey(1), {"mu": lambda v: v})
    assert np.allclose(np.array(m["mu"]), mu_mean, atol=0.05)
