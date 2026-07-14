# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the MPIW contraction core against models with analytic marginal likelihoods
and posterior moments (linear-Gaussian, so everything is closed form).
"""

from collections import namedtuple

import numpy as np
import pytest

import jax
from jax import random
import jax.numpy as jnp

pytest.importorskip("funsor")

from numpyro.contrib.mpiw import (  # noqa: E402
    NamedFactor,
    contract_log_marginal,
    contract_with_source_terms,
)

jax.config.update("jax_enable_x64", True)


def _normal_logpdf(x, loc, scale):
    return -0.5 * ((x - loc) / scale) ** 2 - jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi)


# ---------------------------------------------------------------------------
# Model 1: scalar conjugate Gaussian chain   a -> b -> x
# ---------------------------------------------------------------------------
ChainTruth = namedtuple("ChainTruth", "log_px Ea Eb Va Vb")


def _chain_truth(tau_a, tau_b, sigma, x_obs):
    TA2, TB2, S2 = tau_a**2, tau_b**2, sigma**2
    mu0 = np.array([0.0, 0.0])
    Sig0 = np.array([[TA2, TA2], [TA2, TA2 + TB2]])
    H = np.array([[0.0, 1.0]])
    R = np.array([[S2]])
    m_x = (H @ mu0)[0]
    S_x = (H @ Sig0 @ H.T + R)[0, 0]
    log_px = -0.5 * (x_obs - m_x) ** 2 / S_x - 0.5 * np.log(2 * np.pi * S_x)
    Kg = Sig0 @ H.T @ np.linalg.inv(H @ Sig0 @ H.T + R)
    mu_post = mu0 + (Kg @ np.array([x_obs - m_x]))
    Sig_post = Sig0 - Kg @ H @ Sig0
    return ChainTruth(
        float(log_px),
        float(mu_post[0]),
        float(mu_post[1]),
        float(Sig_post[0, 0]),
        float(Sig_post[1, 1]),
    )


TAU_A, TAU_B, SIGMA, X_OBS = 1.0, 0.7, 0.5, 1.3
GUIDE = dict(qa_loc=0.3, qa_scale=1.2, qb_loc=0.6, qb_scale=1.0)


def _chain_samples(rng, K):
    ka, kb = random.split(rng)
    a = GUIDE["qa_loc"] + GUIDE["qa_scale"] * random.normal(ka, (K,))
    b = GUIDE["qb_loc"] + GUIDE["qb_scale"] * random.normal(kb, (K,))
    return a, b


def _chain_build(a, b, source_terms=None):
    K = a.shape[0]
    logK = jnp.log(K)
    Ja = source_terms["a"] if source_terms else jnp.zeros((K,))
    Jb = source_terms["b"] if source_terms else jnp.zeros((K,))

    fa = (
        _normal_logpdf(a, 0.0, TAU_A)
        - _normal_logpdf(a, GUIDE["qa_loc"], GUIDE["qa_scale"])
        - logK
        + Ja
    )
    fb = (
        _normal_logpdf(b[None, :], a[:, None], TAU_B)
        - _normal_logpdf(b, GUIDE["qb_loc"], GUIDE["qb_scale"])[None, :]
        - logK
        + Jb[None, :]
    )
    fx = _normal_logpdf(X_OBS, b, SIGMA)

    factors = [
        NamedFactor(fa, ("ka",)),
        NamedFactor(fb, ("ka", "kb")),
        NamedFactor(fx, ("kb",)),
    ]
    return factors, frozenset({"ka", "kb"}), frozenset()


def test_chain_log_marginal_unbiased():
    truth = _chain_truth(TAU_A, TAU_B, SIGMA, X_OBS)
    K, n_rep = 30, 400
    keys = random.split(random.PRNGKey(0), n_rep)
    px = []
    for key in keys:
        a, b = _chain_samples(key, K)
        factors, elim, plates = _chain_build(a, b)
        px.append(float(jnp.exp(contract_log_marginal(factors, elim, plates))))
    est = np.mean(px)
    se = np.std(px) / np.sqrt(n_rep)
    true_px = np.exp(truth.log_px)
    # unbiased estimator of P(x): mean within a few SE of the truth
    assert abs(est - true_px) < 4 * se


def test_chain_moments_via_source_terms():
    truth = _chain_truth(TAU_A, TAU_B, SIGMA, X_OBS)
    K = 4000
    a, b = _chain_samples(random.PRNGKey(1), K)

    _, weights = contract_with_source_terms(
        lambda s: _chain_build(a, b, s), {"a": (K,), "b": (K,)}
    )
    wa, wb = np.array(weights["a"]), np.array(weights["b"])
    a_np, b_np = np.array(a), np.array(b)

    assert wa.sum() == pytest.approx(1.0, abs=1e-6)
    assert wb.sum() == pytest.approx(1.0, abs=1e-6)

    Ea, Eb = (wa * a_np).sum(), (wb * b_np).sum()
    Va = (wa * a_np**2).sum() - Ea**2
    Vb = (wb * b_np**2).sum() - Eb**2
    assert Ea == pytest.approx(truth.Ea, abs=0.03)
    assert Eb == pytest.approx(truth.Eb, abs=0.03)
    assert Va == pytest.approx(truth.Va, abs=0.03)
    assert Vb == pytest.approx(truth.Vb, abs=0.03)


def test_chain_jit():
    K = 64
    a, b = _chain_samples(random.PRNGKey(2), K)

    def logp(a, b):
        factors, elim, plates = _chain_build(a, b)
        return contract_log_marginal(factors, elim, plates)

    val = jax.jit(logp)(a, b)
    grads = jax.jit(jax.grad(lambda a, b: logp(a, b)))(a, b)
    assert np.isfinite(float(val))
    assert np.all(np.isfinite(np.array(grads)))


# ---------------------------------------------------------------------------
# Model 2: plated hierarchical Gaussian   mu -> {z_i} -> {x_i}
# ---------------------------------------------------------------------------
PlateTruth = namedtuple("PlateTruth", "log_px Emu Ez Vmu Vz")

S_MU, TAU, SIGMA_P = 1.0, 0.6, 0.4
X_PLATE = np.array([0.8, -0.3, 1.5, 0.1, -1.1])
N_PLATE = len(X_PLATE)
PGUIDE = dict(
    mu_loc=0.2,
    mu_scale=1.1,
    z_loc=np.array([0.5, -0.2, 1.0, 0.0, -0.8]),
    z_scale=0.9,
)


def _plate_truth():
    N = N_PLATE
    dim = N + 1
    mu0 = np.zeros(dim)
    Sig0 = np.zeros((dim, dim))
    Sig0[0, 0] = S_MU**2
    for i in range(1, dim):
        Sig0[0, i] = Sig0[i, 0] = S_MU**2
        for j in range(1, dim):
            Sig0[i, j] = S_MU**2 + (TAU**2 if i == j else 0.0)
    H = np.zeros((N, dim))
    for i in range(N):
        H[i, i + 1] = 1.0
    R = (SIGMA_P**2) * np.eye(N)
    m_x = H @ mu0
    S_x = H @ Sig0 @ H.T + R
    log_px = -0.5 * (X_PLATE - m_x) @ np.linalg.solve(S_x, X_PLATE - m_x)
    log_px -= 0.5 * np.log(np.linalg.det(2 * np.pi * S_x))
    Kg = Sig0 @ H.T @ np.linalg.inv(S_x)
    mu_post = mu0 + Kg @ (X_PLATE - m_x)
    Sig_post = Sig0 - Kg @ H @ Sig0
    return PlateTruth(
        float(log_px),
        float(mu_post[0]),
        mu_post[1:].copy(),
        float(Sig_post[0, 0]),
        np.diag(Sig_post)[1:].copy(),
    )


def _plate_samples(rng, K):
    km, kz = random.split(rng)
    mu = PGUIDE["mu_loc"] + PGUIDE["mu_scale"] * random.normal(km, (K,))
    z = PGUIDE["z_loc"][None, :] + PGUIDE["z_scale"] * random.normal(kz, (K, N_PLATE))
    return mu, z


def _plate_build(mu, z, source_terms=None):
    K = mu.shape[0]
    logK = jnp.log(K)
    Jmu = source_terms["mu"] if source_terms else jnp.zeros((K,))
    Jz = source_terms["z"] if source_terms else jnp.zeros((K, N_PLATE))

    fmu = (
        _normal_logpdf(mu, 0.0, S_MU)
        - _normal_logpdf(mu, PGUIDE["mu_loc"], PGUIDE["mu_scale"])
        - logK
        + Jmu
    )
    fz = (
        _normal_logpdf(z[None, :, :], mu[:, None, None], TAU)
        - _normal_logpdf(z, PGUIDE["z_loc"][None, :], PGUIDE["z_scale"])[None, :, :]
        - logK
        + Jz[None, :, :]
    )
    fx = _normal_logpdf(X_PLATE[None, :], z, SIGMA_P)

    factors = [
        NamedFactor(fmu, ("k_mu",)),
        NamedFactor(fz, ("k_mu", "k_z", "i")),
        NamedFactor(fx, ("k_z", "i")),
    ]
    return factors, frozenset({"k_mu", "k_z"}), frozenset({"i"})


def test_plate_log_marginal_converges():
    truth = _plate_truth()
    K, n_rep = 100, 300
    keys = random.split(random.PRNGKey(5), n_rep)
    px = []
    for key in keys:
        mu, z = _plate_samples(key, K)
        factors, elim, plates = _plate_build(mu, z)
        px.append(float(jnp.exp(contract_log_marginal(factors, elim, plates))))
    est = np.mean(px)
    se = np.std(px) / np.sqrt(n_rep)
    true_px = np.exp(truth.log_px)
    assert abs(est - true_px) < 4 * se


def test_plate_moments_and_per_element_weights():
    truth = _plate_truth()
    K = 4000
    mu, z = _plate_samples(random.PRNGKey(6), K)
    _, weights = contract_with_source_terms(
        lambda s: _plate_build(mu, z, s), {"mu": (K,), "z": (K, N_PLATE)}
    )
    wmu, wz = np.array(weights["mu"]), np.array(weights["z"])
    mu_np, z_np = np.array(mu), np.array(z)

    # the plate mechanic: weights normalize per plate element, not globally
    assert wmu.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(wz.sum(axis=0), 1.0, atol=1e-6)

    Emu = (wmu * mu_np).sum()
    Ez = (wz * z_np).sum(axis=0)
    assert Emu == pytest.approx(truth.Emu, abs=0.05)
    assert np.allclose(Ez, truth.Ez, atol=0.05)


def test_contract_rejects_uneliminated_dim():
    K = 8
    a, b = _chain_samples(random.PRNGKey(9), K)
    factors, _, _ = _chain_build(a, b)
    # forget to eliminate "kb"
    with pytest.raises(ValueError, match="free dimensions"):
        contract_log_marginal(factors, frozenset({"ka"}), frozenset())
