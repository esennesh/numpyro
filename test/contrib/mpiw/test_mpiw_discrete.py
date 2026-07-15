# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end MPIW on a model with discrete latent variables: a single-visit
occupancy-style model with binary presence latents and a false-positive floor (so
detection probabilities stay away from 0/1). The latents are independent, so the
exact marginal likelihood and posterior are available in closed form by enumeration.
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

PSI, P1, P0 = 0.6, 0.7, 0.05  # occupancy, true-detection, false-positive probs
Y = np.array([1.0, 0.0, 0.0, 1.0, 0.0])
N = len(Y)


def _model():
    with numpyro.plate("sites", N):
        z = numpyro.sample("z", dist.Bernoulli(PSI))
        det = z * P1 + (1 - z) * P0
        numpyro.sample("y", dist.Bernoulli(det), obs=jnp.asarray(Y))


def _guide():
    with numpyro.plate("sites", N):
        numpyro.sample("z", dist.Bernoulli(0.5))


def _exact():
    y = np.array(Y)
    u1 = PSI * np.where(y == 1, P1, 1 - P1)  # z=1 unnormalized
    u0 = (1 - PSI) * np.where(y == 1, P0, 1 - P0)  # z=0 unnormalized
    marg = u1 + u0
    return float(np.log(marg).sum()), u1 / marg  # log P(Y), P(z_i=1 | y_i)


def test_discrete_log_marginal():
    log_pY, _ = _exact()
    mpiw = MPIW(_model, _guide, num_samples=2000)
    keys = random.split(random.PRNGKey(0), 200)
    px = np.array([float(jnp.exp(mpiw.log_marginal(k))) for k in keys])
    est, se = px.mean(), px.std() / np.sqrt(len(px))
    assert abs(est - np.exp(log_pY)) < 4 * se


def test_discrete_posterior_moments():
    _, pz1 = _exact()
    mpiw = MPIW(_model, _guide, num_samples=4000)
    m = mpiw.moments(random.PRNGKey(1), {"z": lambda v: v})
    # E[z_i | y] is exactly the posterior presence probability
    assert np.allclose(np.array(m["z"]), pz1, atol=0.03)


def test_discrete_site_weights_per_element_normalized():
    mpiw = MPIW(_model, _guide, num_samples=200)
    values, weights = mpiw.site_weights(random.PRNGKey(2))["z"]
    assert values.shape == (200, N)
    assert weights.shape == (200, N)
    # weights normalize per site (plate element)
    assert np.allclose(np.array(weights.sum(axis=0)), 1.0, atol=1e-6)
    # sampled values are binary
    assert set(np.unique(np.array(values)).tolist()) <= {0, 1}
