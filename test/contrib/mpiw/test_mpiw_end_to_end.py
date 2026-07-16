# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end MPIW tests: run real NumPyro models + mean-field guides through the
sampled-enumeration messenger and contraction, and check log P_MP and posterior
moments against closed-form linear-Gaussian answers.
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


# ---------------------------------------------------------------------------
# Model 1: scalar conjugate Gaussian chain a -> b -> x
# ---------------------------------------------------------------------------
TAU_A, TAU_B, SIGMA, X_OBS = 1.0, 0.7, 0.5, 1.3
QA_LOC, QA_SCALE, QB_LOC, QB_SCALE = 0.3, 1.2, 0.6, 1.0
ChainTruth = namedtuple("ChainTruth", "log_px Ea Eb")


def _chain_truth():
    TA2, TB2, S2 = TAU_A**2, TAU_B**2, SIGMA**2
    mu0 = np.array([0.0, 0.0])
    Sig0 = np.array([[TA2, TA2], [TA2, TA2 + TB2]])
    H = np.array([[0.0, 1.0]])
    R = np.array([[S2]])
    m_x = (H @ mu0)[0]
    S_x = (H @ Sig0 @ H.T + R)[0, 0]
    log_px = -0.5 * (X_OBS - m_x) ** 2 / S_x - 0.5 * np.log(2 * np.pi * S_x)
    Kg = Sig0 @ H.T @ np.linalg.inv(H @ Sig0 @ H.T + R)
    mu_post = mu0 + (Kg @ np.array([X_OBS - m_x]))
    return ChainTruth(float(log_px), float(mu_post[0]), float(mu_post[1]))


def _chain_model():
    a = numpyro.sample("a", dist.Normal(0.0, TAU_A))
    b = numpyro.sample("b", dist.Normal(a, TAU_B))
    numpyro.sample("x", dist.Normal(b, SIGMA), obs=jnp.array(X_OBS))


def _chain_guide():
    numpyro.sample("a", dist.Normal(QA_LOC, QA_SCALE))
    numpyro.sample("b", dist.Normal(QB_LOC, QB_SCALE))


def test_chain_log_marginal():
    truth = _chain_truth()
    mpiw = MPIW(_chain_model, _chain_guide, num_samples=30)
    keys = random.split(random.PRNGKey(0), 400)
    px = np.array([float(jnp.exp(mpiw.log_marginal(k))) for k in keys])
    est, se = px.mean(), px.std() / np.sqrt(len(px))
    assert abs(est - np.exp(truth.log_px)) < 4 * se


def test_chain_moments():
    truth = _chain_truth()
    mpiw = MPIW(_chain_model, _chain_guide, num_samples=4000)
    m = mpiw.moments(random.PRNGKey(1), {"a": lambda v: v, "b": lambda v: v})
    assert float(m["a"]) == pytest.approx(truth.Ea, abs=0.03)
    assert float(m["b"]) == pytest.approx(truth.Eb, abs=0.03)


def test_chain_site_weights_normalized():
    mpiw = MPIW(_chain_model, _chain_guide, num_samples=100)
    w = mpiw.site_weights(random.PRNGKey(2))
    for name in ("a", "b"):
        values, weights = w[name]
        assert values.shape == (100,)
        assert weights.shape == (100,)
        assert float(weights.sum()) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Model 2: plated hierarchical Gaussian mu -> {z_i} -> {x_i}
# ---------------------------------------------------------------------------
S_MU, TAU, SIGMA_P = 1.0, 0.6, 0.4
X_PLATE = np.array([0.8, -0.3, 1.5, 0.1, -1.1])
N_PLATE = len(X_PLATE)
Z_LOC = np.array([0.5, -0.2, 1.0, 0.0, -0.8])
MU_LOC, MU_SCALE, Z_SCALE = 0.2, 1.1, 0.9
PlateTruth = namedtuple("PlateTruth", "log_px Emu Ez")


def _plate_truth():
    dim = N_PLATE + 1
    mu0 = np.zeros(dim)
    Sig0 = np.zeros((dim, dim))
    Sig0[0, 0] = S_MU**2
    for i in range(1, dim):
        Sig0[0, i] = Sig0[i, 0] = S_MU**2
        for j in range(1, dim):
            Sig0[i, j] = S_MU**2 + (TAU**2 if i == j else 0.0)
    H = np.zeros((N_PLATE, dim))
    for i in range(N_PLATE):
        H[i, i + 1] = 1.0
    R = (SIGMA_P**2) * np.eye(N_PLATE)
    S_x = H @ Sig0 @ H.T + R
    log_px = -0.5 * (X_PLATE) @ np.linalg.solve(S_x, X_PLATE)
    log_px -= 0.5 * np.log(np.linalg.det(2 * np.pi * S_x))
    Kg = Sig0 @ H.T @ np.linalg.inv(S_x)
    mu_post = mu0 + Kg @ X_PLATE
    return PlateTruth(float(log_px), float(mu_post[0]), mu_post[1:].copy())


def _plate_model():
    mu = numpyro.sample("mu", dist.Normal(0.0, S_MU))
    with numpyro.plate("data", N_PLATE):
        z = numpyro.sample("z", dist.Normal(mu, TAU))
        numpyro.sample("x", dist.Normal(z, SIGMA_P), obs=jnp.asarray(X_PLATE))


def _plate_guide():
    numpyro.sample("mu", dist.Normal(MU_LOC, MU_SCALE))
    with numpyro.plate("data", N_PLATE):
        numpyro.sample("z", dist.Normal(jnp.asarray(Z_LOC), Z_SCALE))


def test_plate_log_marginal():
    truth = _plate_truth()
    mpiw = MPIW(_plate_model, _plate_guide, num_samples=100)
    keys = random.split(random.PRNGKey(5), 300)
    px = np.array([float(jnp.exp(mpiw.log_marginal(k))) for k in keys])
    est, se = px.mean(), px.std() / np.sqrt(len(px))
    assert abs(est - np.exp(truth.log_px)) < 4 * se


def test_plate_moments_and_weights():
    truth = _plate_truth()
    mpiw = MPIW(_plate_model, _plate_guide, num_samples=4000)
    w = mpiw.site_weights(random.PRNGKey(6))
    # mu weights normalize globally; z weights normalize per plate element
    _, wmu = w["mu"]
    zval, wz = w["z"]
    assert float(wmu.sum()) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(np.array(wz.sum(axis=0)), 1.0, atol=1e-6)
    assert zval.shape == (4000, N_PLATE)

    m = mpiw.moments(random.PRNGKey(6), {"mu": lambda v: v, "z": lambda v: v})
    assert float(m["mu"]) == pytest.approx(truth.Emu, abs=0.05)
    assert np.allclose(np.array(m["z"]), truth.Ez, atol=0.05)


def test_max_plate_nesting_inferred():
    # smoke: omitting max_plate_nesting should infer it and still run
    mpiw = MPIW(_plate_model, _plate_guide, num_samples=10)
    assert np.isfinite(float(mpiw.log_marginal(random.PRNGKey(7))))
