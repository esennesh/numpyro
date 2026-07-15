# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Massively parallel importance weighting on an occupancy model
======================================================================

This example turns the ``numpyro.contrib.mpiw`` integration test into a picture. It
runs massively parallel importance weighting (MPIW; Bowyer et al., 2024) on a
single-species site-occupancy model -- a mixed continuous/discrete hierarchical model
with global continuous occupancy/detection logits, a binary presence latent per site,
and Binomial detection counts -- and plots how well it recovers the posterior as the
number of samples per latent, ``K``, grows.

Three references/comparisons are shown, exactly as asserted in the test suite:

* the exact log marginal likelihood and posterior means, from a 2D grid over the
  discrete-marginalized continuous posterior;
* NUTS on the discrete-marginalized model (the ``config_enumerate`` + NUTS gold
  standard for the posterior means);
* the memory-frugal *serial* contraction path vs. the *dense* path -- identical
  estimates, different time/memory profile.

The resulting figure has three panels:

1. MPIW ``log P_MP(x)`` (mean +/- standard error over seeds) converging to the exact
   log marginal likelihood as ``K`` grows.
2. Absolute error of the MPIW posterior means of the two logits vs. the exact grid,
   with the (small) NUTS error shown for reference.
3. Wall-clock time per estimate for the dense vs. serial contraction, vs. ``K``.

**References:**

    1. Sam Bowyer, Thomas Heap, Laurence Aitchison (2024), "Using autodiff to estimate
       posterior moments, marginals and samples", UAI 2024.
    2. Thomas Heap, Sam Bowyer, Laurence Aitchison (2025), "Massively Parallel
       Expectation Maximization For Approximate Posteriors", AABI 2025.

.. image:: ../_static/img/examples/mpiw_occupancy.png
    :align: center
"""

import argparse
from math import comb
import os
import time

import matplotlib.pyplot as plt
import numpy as np

import jax
from jax import random
import jax.numpy as jnp
from jax.scipy.special import expit

import numpyro
from numpyro.contrib.mpiw import MPIW
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

jax.config.update("jax_enable_x64", True)

S, V, P0 = 8, 5, 0.05  # sites, visits, false-positive detection probability
PRIOR_SCALE = 1.5


def get_data(seed=0):
    rng = np.random.default_rng(seed)
    z_true = rng.random(S) < 0.6
    return rng.binomial(V, np.where(z_true, 0.6, P0))


def model(Y):
    lpsi = numpyro.sample("lpsi", dist.Normal(0.0, PRIOR_SCALE))
    lp = numpyro.sample("lp", dist.Normal(0.0, PRIOR_SCALE))
    with numpyro.plate("sites", S):
        z = numpyro.sample("z", dist.Bernoulli(expit(lpsi)))
        det = z * expit(lp) + (1 - z) * P0
        numpyro.sample("y", dist.Binomial(V, det), obs=jnp.asarray(Y))


def guide(Y):
    # mean-field guide, warmed toward the posterior (as QEM would learn it)
    numpyro.sample("lpsi", dist.Normal(-1.0, 1.0))
    numpyro.sample("lp", dist.Normal(1.0, 1.0))
    with numpyro.plate("sites", S):
        numpyro.sample("z", dist.Bernoulli(0.5))


def marginalized_model(Y):
    # discrete presence latent summed out analytically; used for the NUTS baseline
    lpsi = numpyro.sample("lpsi", dist.Normal(0.0, PRIOR_SCALE))
    lp = numpyro.sample("lp", dist.Normal(0.0, PRIOR_SCALE))
    psi, p = expit(lpsi), expit(lp)
    like1 = jnp.exp(dist.Binomial(V, p).log_prob(jnp.asarray(Y)))
    like0 = jnp.exp(dist.Binomial(V, P0).log_prob(jnp.asarray(Y)))
    numpyro.factor("obs", jnp.sum(jnp.log(psi * like1 + (1 - psi) * like0)))


def grid_reference(Y):
    """Exact log P(y) and posterior means of the logits by 2D numerical integration."""
    g = np.linspace(-7, 7, 161)
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
    log_evidence = float(np.log(np.sum(np.exp(logpost)) * cell))
    w = np.exp(logpost - logpost.max())
    w /= w.sum()
    return log_evidence, float(np.sum(w * lpsi)), float(np.sum(w * lp))


def run_nuts(Y, rng_key):
    mcmc = MCMC(
        NUTS(marginalized_model),
        num_warmup=1000,
        num_samples=4000,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, Y)
    s = mcmc.get_samples()
    return float(s["lpsi"].mean()), float(s["lp"].mean())


def _timed(fn):
    """Wall-clock time (seconds) of a call whose result is a JAX array."""
    fn().block_until_ready()  # compile first
    start = time.time()
    for _ in range(3):
        fn().block_until_ready()
    return (time.time() - start) / 3


def main(args):
    Y = get_data()
    print("detection counts:", Y)

    log_evidence, exact_lpsi, exact_lp = grid_reference(Y)
    print(
        f"exact: log P(y)={log_evidence:.3f}  E[lpsi]={exact_lpsi:.3f}  E[lp]={exact_lp:.3f}"
    )
    nuts_lpsi, nuts_lp = run_nuts(Y, random.key(args.seed))
    print(f"NUTS:  E[lpsi]={nuts_lpsi:.3f}  E[lp]={nuts_lp:.3f}")

    Ks = args.num_samples_grid
    lm_mean, lm_se = [], []
    err_lpsi, err_lp = [], []
    t_dense, t_serial = [], []
    for K in Ks:
        mpiw = MPIW(model, guide, num_samples=K)

        # log marginal likelihood: mean +/- standard error over seeds
        keys = random.split(random.key(args.seed), args.num_seeds)
        lms = np.array([float(mpiw.log_marginal(k, Y)) for k in keys])
        lm_mean.append(lms.mean())
        lm_se.append(lms.std() / np.sqrt(len(lms)))

        # posterior means of the logits (absolute error vs the exact grid)
        m = mpiw.moments(
            random.key(args.seed), {"lpsi": lambda v: v, "lp": lambda v: v}, Y
        )
        err_lpsi.append(abs(float(m["lpsi"]) - exact_lpsi))
        err_lp.append(abs(float(m["lp"]) - exact_lp))

        # dense vs serial contraction timing (identical values)
        key = random.key(args.seed)
        t_dense.append(_timed(lambda: mpiw.log_marginal(key, Y)))
        t_serial.append(
            _timed(lambda: mpiw.log_marginal(key, Y, serial_sites=("lpsi", "lp")))
        )
        print(f"K={K:4d}: logP_MP={lm_mean[-1]:.3f}  err(lpsi)={err_lpsi[-1]:.3f}")

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(15, 4.5), constrained_layout=True
    )

    ax1.axhline(log_evidence, color="black", ls="--", label="exact")
    ax1.errorbar(
        Ks, lm_mean, yerr=lm_se, marker="o", color="C0", capsize=3, label="MPIW"
    )
    ax1.set(
        xscale="log",
        xlabel="K (samples per latent)",
        ylabel="log P(y)",
        title="Marginal likelihood estimate",
    )
    ax1.legend()

    ax2.plot(Ks, err_lpsi, marker="o", color="C0", label="MPIW |E[lpsi]-exact|")
    ax2.plot(Ks, err_lp, marker="s", color="C1", label="MPIW |E[lp]-exact|")
    ax2.axhline(
        abs(nuts_lpsi - exact_lpsi), color="C0", ls=":", label="NUTS |lpsi err|"
    )
    ax2.axhline(abs(nuts_lp - exact_lp), color="C1", ls=":", label="NUTS |lp err|")
    ax2.set(
        xscale="log",
        yscale="log",
        xlabel="K (samples per latent)",
        ylabel="absolute error",
        title="Posterior-mean accuracy",
    )
    ax2.legend()

    ax3.plot(Ks, np.array(t_dense) * 1e3, marker="o", color="C2", label="dense")
    ax3.plot(Ks, np.array(t_serial) * 1e3, marker="s", color="C3", label="serial")
    ax3.set(
        xscale="log",
        yscale="log",
        xlabel="K (samples per latent)",
        ylabel="time per estimate (ms)",
        title="Dense vs serial contraction",
    )
    ax3.legend()

    fig.suptitle("MPIW on a site-occupancy model: accuracy and cost vs K")
    plt.savefig("mpiw_occupancy.png", dpi=120)
    print("saved mpiw_occupancy.png")


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.21.0")
    parser = argparse.ArgumentParser(description="MPIW occupancy-model demo")
    parser.add_argument(
        "--num-samples-grid",
        nargs="+",
        type=int,
        default=[3, 10, 30, 100, 300],
        help="values of K (samples per latent) to sweep",
    )
    parser.add_argument(
        "--num-seeds",
        default=30,
        type=int,
        help="seeds averaged for the log-marginal estimate",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    main(args)
