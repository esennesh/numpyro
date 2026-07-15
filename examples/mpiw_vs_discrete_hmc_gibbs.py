# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: MPIW vs DiscreteHMCGibbs at matched sample count
=========================================================

Massively parallel importance weighting (MPIW; Bowyer et al., 2024) earns its keep on
models where a latent variable is not differentiable -- it never needs gradients through
the discrete latents, and it does not require them to be analytically marginalized. This
example puts that to the test on the single-species site-occupancy model (global
continuous occupancy/detection logits, a binary presence latent per site, Binomial
detection counts) by comparing MPIW against :class:`~numpyro.infer.DiscreteHMCGibbs` --
NUTS on the continuous logits interleaved with Gibbs updates on the binary presence,
so *both* methods handle the discrete latent directly, with no marginalization.

For each nominal sample count ``M`` we form several independent posterior-mean estimates
from each method (MPIW with ``K = M`` samples per latent; DiscreteHMCGibbs with ``M``
post-burn-in draws) and report the RMSE of those estimates to the exact posterior mean
(from a 2D grid over the discrete-marginalized continuous posterior). Warmup is fixed
and separate, so only the post-burn-in sample count varies.

What the figure shows, and the honest caveats:

* In the **low-sample regime** MPIW has lower RMSE and is cheaper per estimate: its
  combinatorial reweighting extracts more from few samples than Gibbs-on-discrete does.
* DiscreteHMCGibbs is **asymptotically exact**, so given enough samples its RMSE keeps
  falling while MPIW plateaus at a bias floor set by the mean-field guide (and the
  finite-K self-normalized-importance bias). The curves cross.
* This is an easy, low-dimensional posterior whose discrete latents are conditionally
  independent; "matched sample count" also structurally favors MPIW (``K`` per site,
  reweighted over combinations). The takeaway is "competitive and cheap in its niche",
  not "beats MCMC in general".

**References:**

    1. Sam Bowyer, Thomas Heap, Laurence Aitchison (2024), "Using autodiff to estimate
       posterior moments, marginals and samples", UAI 2024.

.. image:: ../_static/img/examples/mpiw_vs_discrete_hmc_gibbs.png
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
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs

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


def grid_reference(Y):
    """Exact posterior means of the logits by 2D numerical integration (z summed out)."""
    g = np.linspace(-7, 7, 161)
    lp, lpsi = np.meshgrid(g, g)
    psi, p = expit(lpsi), expit(lp)
    loglik = np.zeros_like(lpsi)
    for i in range(S):
        c = comb(V, int(Y[i]))
        like1 = c * p ** Y[i] * (1 - p) ** (V - Y[i])
        like0 = c * P0 ** Y[i] * (1 - P0) ** (V - Y[i])
        loglik += np.log(psi * like1 + (1 - psi) * like0)
    logprior = -0.5 * (lpsi / PRIOR_SCALE) ** 2 - 0.5 * (lp / PRIOR_SCALE) ** 2
    w = np.exp((loglik + logprior) - (loglik + logprior).max())
    w /= w.sum()
    return float(np.sum(w * lpsi)), float(np.sum(w * lp))


def mpiw_estimates(Y, M, R, seed):
    """R posterior-mean estimates from MPIW with K = M samples per latent."""
    mpiw = MPIW(model, guide, num_samples=M)
    stats = {"lpsi": lambda v: v, "lp": lambda v: v}
    keys = random.split(random.key(seed), R)
    t0 = time.time()
    est = [mpiw.moments(k, stats, Y) for k in keys]
    per_estimate_time = (time.time() - t0) / R
    lpsi = np.array([float(e["lpsi"]) for e in est])
    lp = np.array([float(e["lp"]) for e in est])
    return lpsi, lp, per_estimate_time


def dhg_estimates(Y, M, R, warmup, seed):
    """R posterior-mean estimates from DiscreteHMCGibbs with M post-burn-in draws each.

    DiscreteHMCGibbs does not support vectorized chains (its Gibbs init cannot take a
    batched key), so the R chains run sequentially.
    """
    mcmc = MCMC(
        DiscreteHMCGibbs(NUTS(model)),
        num_warmup=warmup,
        num_samples=M,
        num_chains=R,
        chain_method="sequential",
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    t0 = time.time()
    mcmc.run(random.key(seed), Y)
    jax.block_until_ready(mcmc.get_samples())
    per_estimate_time = (time.time() - t0) / R
    s = mcmc.get_samples(group_by_chain=True)
    return (
        np.array(s["lpsi"].mean(axis=1)),
        np.array(s["lp"].mean(axis=1)),
        per_estimate_time,
    )


def _rmse(lpsi, lp, exact_lpsi, exact_lp):
    return float(np.sqrt(np.mean((lpsi - exact_lpsi) ** 2 + (lp - exact_lp) ** 2)))


def main(args):
    Y = get_data()
    print("detection counts:", Y)
    exact_lpsi, exact_lp = grid_reference(Y)
    print(f"exact: E[lpsi]={exact_lpsi:.3f}  E[lp]={exact_lp:.3f}")

    Ms = args.num_samples_grid
    mpiw_rmse, dhg_rmse, mpiw_t, dhg_t = [], [], [], []
    for M in Ms:
        mp_lpsi, mp_lp, mp_time = mpiw_estimates(Y, M, args.num_repeats, args.seed)
        dhg_lpsi, dhg_lp, dhg_time = dhg_estimates(
            Y, M, args.num_repeats, args.num_warmup, args.seed
        )
        mpiw_rmse.append(_rmse(mp_lpsi, mp_lp, exact_lpsi, exact_lp))
        dhg_rmse.append(_rmse(dhg_lpsi, dhg_lp, exact_lpsi, exact_lp))
        mpiw_t.append(mp_time)
        dhg_t.append(dhg_time)
        print(
            f"M={M:5d}: MPIW rmse={mpiw_rmse[-1]:.3f} ({mpiw_t[-1] * 1e3:.0f}ms)  "
            f"DHG rmse={dhg_rmse[-1]:.3f} ({dhg_t[-1] * 1e3:.0f}ms)"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    ax1.plot(Ms, mpiw_rmse, "o-", color="C0", label="MPIW (K=M)")
    ax1.plot(Ms, dhg_rmse, "s-", color="C1", label="DiscreteHMCGibbs (M draws)")
    ax1.set(
        xscale="log",
        yscale="log",
        xlabel="M (samples)",
        ylabel="RMSE of posterior mean",
        title="Accuracy at matched sample count",
    )
    ax1.legend()

    ax2.plot(Ms, np.array(mpiw_t) * 1e3, "o-", color="C0", label="MPIW")
    ax2.plot(
        Ms,
        np.array(dhg_t) * 1e3,
        "s-",
        color="C1",
        label="DiscreteHMCGibbs (excl. warmup)",
    )
    ax2.set(
        xscale="log",
        yscale="log",
        xlabel="M (samples)",
        ylabel="wall-clock per estimate (ms)",
        title="Cost at matched sample count",
    )
    ax2.legend()

    fig.suptitle(
        "MPIW vs DiscreteHMCGibbs on the occupancy model (matched sample count)"
    )
    plt.savefig("mpiw_vs_discrete_hmc_gibbs.png", dpi=120)
    print("saved mpiw_vs_discrete_hmc_gibbs.png")


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.21.0")
    parser = argparse.ArgumentParser(description="MPIW vs DiscreteHMCGibbs demo")
    parser.add_argument(
        "--num-samples-grid",
        nargs="+",
        type=int,
        default=[3, 10, 30, 100, 300, 1000],
        help="values of M (post-burn-in samples / MPIW K) to sweep",
    )
    parser.add_argument(
        "--num-repeats",
        default=10,
        type=int,
        help="independent estimates per method per M (for the RMSE)",
    )
    parser.add_argument("--num-warmup", default=500, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    main(args)
