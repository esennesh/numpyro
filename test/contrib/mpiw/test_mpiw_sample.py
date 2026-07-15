# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Joint posterior sampling via MPIW backward sampling, checked against closed-form
linear-Gaussian posteriors. The key property beyond marginal correctness is that the
joint covariance between latents is recovered (backward sampling respects the model's
coupling, not just per-site marginals).
"""

from collections import namedtuple

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

TAU_A, TAU_B, SIGMA, X_OBS = 1.0, 0.7, 0.5, 1.3
ChainPost = namedtuple("ChainPost", "mean cov")


def _chain_posterior():
    TA2, TB2, S2 = TAU_A**2, TAU_B**2, SIGMA**2
    Sig0 = np.array([[TA2, TA2], [TA2, TA2 + TB2]])
    H = np.array([[0.0, 1.0]])
    R = np.array([[S2]])
    Kg = Sig0 @ H.T @ np.linalg.inv(H @ Sig0 @ H.T + R)
    mean = (Kg @ np.array([X_OBS])).ravel()
    cov = Sig0 - Kg @ H @ Sig0
    return ChainPost(mean, cov)


def _chain_model():
    a = numpyro.sample("a", dist.Normal(0.0, TAU_A))
    b = numpyro.sample("b", dist.Normal(a, TAU_B))
    numpyro.sample("x", dist.Normal(b, SIGMA), obs=jnp.array(X_OBS))


def _chain_guide():
    numpyro.sample("a", dist.Normal(0.3, 1.2))
    numpyro.sample("b", dist.Normal(0.6, 1.0))


def test_sample_posterior_shapes_and_mean():
    post = _chain_posterior()
    mpiw = MPIW(_chain_model, _chain_guide, num_samples=100)
    draws = mpiw.sample_posterior(random.PRNGKey(0), 800)
    assert draws["a"].shape == (800,)
    assert draws["b"].shape == (800,)
    sample_mean = np.array([draws["a"].mean(), draws["b"].mean()])
    assert np.allclose(sample_mean, post.mean, atol=0.1)


def test_sample_posterior_recovers_joint_covariance():
    post = _chain_posterior()
    mpiw = MPIW(_chain_model, _chain_guide, num_samples=100)
    draws = mpiw.sample_posterior(random.PRNGKey(1), 1500)
    sample = np.stack([np.array(draws["a"]), np.array(draws["b"])], axis=1)
    cov = np.cov(sample.T)
    # off-diagonal (a,b) posterior correlation is positive and captured, not just marginals
    assert cov[0, 1] > 0
    assert np.allclose(cov, post.cov, atol=0.1)


# plated hierarchical model: mu -> {z_i} -> {x_i}
S_MU, TAU, SIGMA_P = 1.0, 0.6, 0.4
X_PLATE = np.array([0.8, -0.3, 1.5, 0.1, -1.1])
N_PLATE = len(X_PLATE)
Z_LOC = np.array([0.5, -0.2, 1.0, 0.0, -0.8])


def _plate_posterior_mean():
    dim = N_PLATE + 1
    Sig0 = np.full((dim, dim), S_MU**2)
    for i in range(1, dim):
        Sig0[i, i] += TAU**2
    H = np.zeros((N_PLATE, dim))
    H[np.arange(N_PLATE), np.arange(1, dim)] = 1.0
    Sx = H @ Sig0 @ H.T + (SIGMA_P**2) * np.eye(N_PLATE)
    return Sig0 @ H.T @ np.linalg.inv(Sx) @ X_PLATE  # [mu, z_1..z_N]


def _plate_model():
    mu = numpyro.sample("mu", dist.Normal(0.0, S_MU))
    with numpyro.plate("data", N_PLATE):
        z = numpyro.sample("z", dist.Normal(mu, TAU))
        numpyro.sample("x", dist.Normal(z, SIGMA_P), obs=jnp.asarray(X_PLATE))


def _plate_guide():
    numpyro.sample("mu", dist.Normal(0.2, 1.1))
    with numpyro.plate("data", N_PLATE):
        numpyro.sample("z", dist.Normal(jnp.asarray(Z_LOC), 0.9))


def test_sample_posterior_plated():
    post = _plate_posterior_mean()
    mpiw = MPIW(_plate_model, _plate_guide, num_samples=100)
    draws = mpiw.sample_posterior(random.PRNGKey(2), 800)
    assert draws["mu"].shape == (800,)
    assert draws["z"].shape == (800, N_PLATE)
    assert draws["mu"].mean() == pytest.approx(post[0], abs=0.12)
    assert np.allclose(np.array(draws["z"]).mean(axis=0), post[1:], atol=0.1)
