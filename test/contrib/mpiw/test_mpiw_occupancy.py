# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Integration test on a mixed continuous-discrete hierarchical model: a single-species
site-occupancy model (Bowyer et al. 2024 lineage). Global continuous occupancy/detection
logits, a binary presence latent per site, and Binomial detection counts.

This is the branch-3 capstone: it exercises the full MPIW pipeline on a real mixed model
(continuous globals + discrete plated latents + observations) and checks it against
independent references -- an exact 2D grid over the (marginalized) continuous posterior,
and NUTS on the discrete-marginalized model (the config_enumerate + NUTS gold standard).
It also exercises the memory-frugal serial-contraction path by serializing the global
logits.
"""

from math import comb

import numpy as np
import pytest

import jax
from jax import random
import jax.numpy as jnp
from jax.scipy.special import expit

pytest.importorskip("funsor")

import numpyro  # noqa: E402
from numpyro.contrib.mpiw import MPIW  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402

jax.config.update("jax_enable_x64", True)

S, V, P0 = 8, 5, 0.05  # sites, visits, false-positive detection prob
PRIOR_SCALE = 1.5
# synthetic detection counts (fixed)
_rng = np.random.default_rng(0)
_z_true = _rng.random(S) < 0.6
Y = _rng.binomial(V, np.where(_z_true, 0.6, P0))


# ---- MPIW model: full model with the discrete presence latent ----
def _model():
    lpsi = numpyro.sample("lpsi", dist.Normal(0.0, PRIOR_SCALE))
    lp = numpyro.sample("lp", dist.Normal(0.0, PRIOR_SCALE))
    with numpyro.plate("sites", S):
        z = numpyro.sample("z", dist.Bernoulli(expit(lpsi)))
        det = z * expit(lp) + (1 - z) * P0
        numpyro.sample("y", dist.Binomial(V, det), obs=jnp.asarray(Y))


# mean-field guide, warmed toward the posterior (as QEM would learn it)
def _guide():
    numpyro.sample("lpsi", dist.Normal(-1.0, 1.0))
    numpyro.sample("lp", dist.Normal(1.0, 1.0))
    with numpyro.plate("sites", S):
        numpyro.sample("z", dist.Bernoulli(0.5))


# ---- discrete-marginalized model (for NUTS): the config_enumerate + NUTS baseline ----
def _marginalized_model():
    lpsi = numpyro.sample("lpsi", dist.Normal(0.0, PRIOR_SCALE))
    lp = numpyro.sample("lp", dist.Normal(0.0, PRIOR_SCALE))
    psi, p = expit(lpsi), expit(lp)
    like1 = jnp.exp(dist.Binomial(V, p).log_prob(jnp.asarray(Y)))
    like0 = jnp.exp(dist.Binomial(V, P0).log_prob(jnp.asarray(Y)))
    numpyro.factor("obs", jnp.sum(jnp.log(psi * like1 + (1 - psi) * like0)))


def _grid_reference():
    """Exact log P(y) and posterior means of the logits by 2D numerical integration."""
    g = np.linspace(-7, 7, 141)
    lp, lpsi = np.meshgrid(g, g)
    psi, p = 1 / (1 + np.exp(-lpsi)), 1 / (1 + np.exp(-lp))
    loglik = np.zeros_like(lpsi)
    for i in range(S):
        c = comb(V, int(Y[i]))
        like1 = c * p ** Y[i] * (1 - p) ** (V - Y[i])
        like0 = c * P0 ** Y[i] * (1 - P0) ** (V - Y[i])
        loglik += np.log(psi * like1 + (1 - psi) * like0)
    logprior = (
        -0.5 * (lpsi / PRIOR_SCALE) ** 2
        - 0.5 * (lp / PRIOR_SCALE) ** 2
        - 2 * np.log(PRIOR_SCALE * np.sqrt(2 * np.pi))
    )
    logpost = loglik + logprior
    cell = (g[1] - g[0]) ** 2
    log_evidence = np.log(np.sum(np.exp(logpost)) * cell)
    w = np.exp(logpost - logpost.max())
    w /= w.sum()
    return log_evidence, np.sum(w * lpsi), np.sum(w * lp)


def test_occupancy_log_marginal_matches_exact():
    log_evidence, _, _ = _grid_reference()
    mpiw = MPIW(_model, _guide, num_samples=60)
    keys = random.split(random.PRNGKey(0), 80)
    px = np.array([float(jnp.exp(mpiw.log_marginal(k))) for k in keys])
    est, se = px.mean(), px.std() / np.sqrt(len(px))
    assert abs(est - np.exp(log_evidence)) < 4 * se


def test_occupancy_moments_vs_nuts():
    _, grid_lpsi, grid_lp = _grid_reference()

    mcmc = MCMC(
        NUTS(_marginalized_model),
        num_warmup=500,
        num_samples=2000,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(0))
    samples = mcmc.get_samples()
    nuts_lpsi = float(samples["lpsi"].mean())
    nuts_lp = float(samples["lp"].mean())
    # sanity: NUTS agrees with the exact grid
    assert abs(nuts_lpsi - grid_lpsi) < 0.1
    assert abs(nuts_lp - grid_lp) < 0.1

    mpiw = MPIW(_model, _guide, num_samples=200)
    m = mpiw.moments(random.PRNGKey(1), {"lpsi": lambda v: v, "lp": lambda v: v})
    # MPIW posterior means match the NUTS gold standard
    assert abs(float(m["lpsi"]) - nuts_lpsi) < 0.15
    assert abs(float(m["lp"]) - nuts_lp) < 0.15


def test_occupancy_serial_matches_dense():
    mpiw = MPIW(_model, _guide, num_samples=50)
    key = random.PRNGKey(3)
    # serialize the global continuous logits (non-plated) -> memory-frugal path
    dense_lm = float(mpiw.log_marginal(key))
    serial_lm = float(mpiw.log_marginal(key, serial_sites=("lpsi", "lp")))
    assert serial_lm == pytest.approx(dense_lm, abs=1e-6)

    stats = {"lpsi": lambda v: v, "lp": lambda v: v}
    dense_m = mpiw.moments(key, stats)
    serial_m = mpiw.moments(key, stats, serial_sites=("lpsi", "lp"))
    assert float(serial_m["lpsi"]) == pytest.approx(float(dense_m["lpsi"]), abs=1e-6)
    assert float(serial_m["lp"]) == pytest.approx(float(dense_m["lp"]), abs=1e-6)
