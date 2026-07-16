# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: QEM vs mean-field VI, same guide family
================================================

The clean comparison for QEM (Heap, Bowyer & Aitchison, AABI 2025) is not against
MCMC -- a mean-field guide carries a proposal-family bias that asymptotically exact
samplers do not -- but against gradient-based VI *with the guide family held fixed*.
This example fits the same :class:`~numpyro.infer.autoguide.AutoExponentialFamily`
guide two ways and compares the trajectories:

* **QEM arm**: gradient-free EM on massively parallel importance weights
  (``numpyro.contrib.qem``), whose per-iteration metric ``log P_MP(y)`` estimates the
  evidence;
* **VI arm**: :class:`~numpyro.infer.svi.SVI` with Adam maximizing the ELBO
  (``TraceEnum_ELBO`` where discrete sites must be marginalized), with periodic
  same-yardstick MPIW evidence evaluations of the partially-trained guide.

Two models, mirroring the plan's benchmark spec (``docs/design/qem_mpiw_plan.md``):

1. a purely continuous hierarchical Gaussian -- eight schools with a log-normal
   scale prior so that every site is exponential-family -- with a long-NUTS oracle
   for the posterior moments and an alpha-scaling stress test (paper section 5.2):
   scaling the ``theta`` latents by alpha leaves QEM trajectories exactly invariant
   (Thm 2) while fixed-learning-rate VI degrades;

   The scale prior matters here in an instructive way: with ``tau ~ Exponential``
   (also exponential-family) the *VI arm's objective is ill-posed* -- the
   mean-field ELBO contains ``-E_q[1/tau^2] * E_q[(theta-mu)^2] / 2`` and
   ``E[1/tau^2]`` diverges under an Exponential guide, so ELBO estimates are
   heavy-tailed with divergent mean and SVI cannot fit this guide family at any
   learning rate (measured: fitted-guide evidence ~5 nats worse than QEM's, ELBO
   estimates in the -10^3..-10^5 range). QEM has no such pathology (it never
   evaluates ``E_q[log p]``; tiny-``tau`` proposals just receive tiny importance
   weights). A LogNormal scale prior keeps both arms well-posed.
2. the single-species site-occupancy model (mixed continuous/discrete) from the MPIW
   example, where the VI arm needs ``config_enumerate`` + ``TraceEnum_ELBO`` for the
   binary presence latents while QEM simply samples them; an exact 2D grid supplies
   the true evidence and posterior means.

.. note:: Wall-clock panels come with a caveat: the VI arm's update is a single jitted
    ``lax.scan`` step, while the QEM arm's update currently re-traces the model and
    guide through funsor every iteration (no jit). Per-iteration wall-clock therefore
    reflects implementation maturity as much as algorithmic cost; the per-iteration
    convergence curves are the like-for-like comparison.

**References:**

    1. Thomas Heap, Sam Bowyer, Laurence Aitchison (2025), "Massively Parallel
       Expectation Maximization For Approximate Posteriors", AABI 2025.
    2. Sam Bowyer, Thomas Heap, Laurence Aitchison (2024), "Using autodiff to estimate
       posterior moments, marginals and samples", UAI 2024.

.. image:: ../_static/img/examples/qem_vs_svi.png
    :align: center
"""

import argparse
from math import comb
import time

import matplotlib.pyplot as plt
import numpy as np

import jax
from jax import random
import jax.numpy as jnp
from jax.scipy.special import expit

import numpyro
from numpyro import handlers
from numpyro.contrib.funsor import config_enumerate
from numpyro.contrib.mpiw import MPIW
from numpyro.contrib.qem import QEM
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, TraceEnum_ELBO
from numpyro.infer.autoguide import AutoExponentialFamily
from numpyro.optim import Adam

jax.config.update("jax_enable_x64", True)

# --- Model 1: hierarchical Gaussian (eight schools, exponential-family priors) ---

Y_SCHOOLS = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
SIGMA_SCHOOLS = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])


def make_schools_model(alpha=1.0):
    """Eight schools; ``alpha`` rescales the theta latents (paper section 5.2)."""

    def model(y, sigma):
        mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
        tau = numpyro.sample("tau", dist.LogNormal(1.5, 0.8))
        with numpyro.plate("schools", len(y)):
            theta = numpyro.sample("theta", dist.Normal(alpha * mu, alpha * tau))
            numpyro.sample("y", dist.Normal(theta / alpha, sigma), obs=y)

    return model


# --- Model 2: site-occupancy (mixed continuous/discrete), as in mpiw_occupancy ---

S_OCC, V_OCC, P0_OCC = 8, 5, 0.05
PRIOR_SCALE_OCC = 1.5


def get_occupancy_data(seed=0):
    rng = np.random.default_rng(seed)
    z_true = rng.random(S_OCC) < 0.6
    return rng.binomial(V_OCC, np.where(z_true, 0.6, P0_OCC))


def occupancy_model(Y):
    lpsi = numpyro.sample("lpsi", dist.Normal(0.0, PRIOR_SCALE_OCC))
    lp = numpyro.sample("lp", dist.Normal(0.0, PRIOR_SCALE_OCC))
    with numpyro.plate("sites", S_OCC):
        z = numpyro.sample("z", dist.Bernoulli(expit(lpsi)))
        det = z * expit(lp) + (1 - z) * P0_OCC
        numpyro.sample("y", dist.Binomial(V_OCC, det), obs=jnp.asarray(Y))


def occupancy_marginalized(Y):
    lpsi = numpyro.sample("lpsi", dist.Normal(0.0, PRIOR_SCALE_OCC))
    lp = numpyro.sample("lp", dist.Normal(0.0, PRIOR_SCALE_OCC))
    psi, p = expit(lpsi), expit(lp)
    like1 = jnp.exp(dist.Binomial(V_OCC, p).log_prob(jnp.asarray(Y)))
    like0 = jnp.exp(dist.Binomial(V_OCC, P0_OCC).log_prob(jnp.asarray(Y)))
    numpyro.factor("obs", jnp.sum(jnp.log(psi * like1 + (1 - psi) * like0)))


def occupancy_grid_reference(Y):
    """Exact log P(y) and posterior means of the logits by 2D numerical integration."""
    g = np.linspace(-7, 7, 161)
    lp, lpsi = np.meshgrid(g, g)
    psi, p = 1 / (1 + np.exp(-lpsi)), 1 / (1 + np.exp(-lp))
    loglik = np.zeros_like(lpsi)
    for i in range(S_OCC):
        c = comb(V_OCC, int(Y[i]))
        like1 = c * p ** Y[i] * (1 - p) ** (V_OCC - Y[i])
        like0 = c * P0_OCC ** Y[i] * (1 - P0_OCC) ** (V_OCC - Y[i])
        loglik += np.log(psi * like1 + (1 - psi) * like0)
    logprior = (
        -0.5 * (lpsi / PRIOR_SCALE_OCC) ** 2
        - 0.5 * (lp / PRIOR_SCALE_OCC) ** 2
        - 2 * np.log(PRIOR_SCALE_OCC * np.sqrt(2 * np.pi))
    )
    logpost = loglik + logprior
    cell = (g[1] - g[0]) ** 2
    log_evidence = float(np.log(np.sum(np.exp(logpost)) * cell))
    w = np.exp(logpost - logpost.max())
    w /= w.sum()
    return log_evidence, float(np.sum(w * lpsi)), float(np.sum(w * lp))


# --- Benchmark arms ---


def run_qem_arm(model, guide, num_samples, num_steps, seed, *model_args):
    """QEM fit; returns (per-iteration log P_MP, cumulative seconds, final qem/state)."""
    qem = QEM(model, guide, num_samples=num_samples)
    state = qem.init(random.PRNGKey(seed), *model_args)
    log_pmps, times = [], []
    start = time.time()
    for _ in range(num_steps):
        state, log_pmp = qem.update(state, *model_args)
        log_pmps.append(float(log_pmp))
        times.append(time.time() - start)
    return np.array(log_pmps), np.array(times), qem, state


def run_svi_arm(
    model,
    guide,
    elbo,
    learning_rate,
    num_steps,
    num_checkpoints,
    seed,
    *model_args,
):
    """SVI fit in chunks; returns per-step ELBO, per-step seconds (steady state),
    checkpoint (step, params) pairs, and the final params."""
    svi = SVI(model, guide, Adam(learning_rate), elbo)
    chunk = max(1, num_steps // num_checkpoints)
    losses, checkpoints, chunk_times = [], [], []
    state = None
    for start_step in range(0, num_steps, chunk):
        t0 = time.time()
        result = svi.run(
            random.PRNGKey(seed + start_step),
            min(chunk, num_steps - start_step),
            *model_args,
            progress_bar=False,
            init_state=state,
        )
        chunk_times.append(time.time() - t0)
        state, params = result.state, result.params
        losses.append(np.asarray(result.losses))
        checkpoints.append((start_step + len(result.losses), params))
    # steady-state per-step time: first chunk pays jit compilation, so drop it
    steady = chunk_times[1:] if len(chunk_times) > 1 else chunk_times
    per_step = float(np.median(steady)) / chunk
    return np.concatenate(losses), per_step, checkpoints, params


def mpiw_evidence(
    model, guide, params, num_samples, seed, *model_args, num_seeds=3, serial_sites=()
):
    """Same-yardstick evidence: log P_MP of the guide at ``params``.

    ``serial_sites`` matters for memory: in the schools model the ``theta``
    factor couples three K-dims at once (``K_theta * K_mu * K_tau * 8``
    entries if contracted densely -- 69 GB at K=1024), so high-K reference
    evaluations must sum at least one parent serially.
    """
    mpiw = MPIW(model, handlers.substitute(guide, data=params), num_samples)
    keys = random.split(random.PRNGKey(seed), num_seeds)
    return float(
        np.mean(
            [
                float(mpiw.log_marginal(k, *model_args, serial_sites=serial_sites))
                for k in keys
            ]
        )
    )


def posterior_means(guide, params, seed, num_draws=4000):
    samples = guide.sample_posterior(
        random.PRNGKey(seed), params, sample_shape=(num_draws,)
    )
    return {name: np.asarray(s.mean(axis=0)) for name, s in samples.items()}


def run_nuts(model, seed, *model_args, **sample_sites):
    mcmc = MCMC(
        NUTS(model),
        num_warmup=1000,
        num_samples=4000,
        progress_bar=False,
    )
    mcmc.run(random.PRNGKey(seed), *model_args)
    return {name: np.asarray(s.mean(axis=0)) for name, s in mcmc.get_samples().items()}


# --- Model 1 benchmark ---


def benchmark_schools(args, axes):
    y, sigma = jnp.asarray(Y_SCHOOLS), jnp.asarray(SIGMA_SCHOOLS)
    model = make_schools_model()
    guide = AutoExponentialFamily(model)  # shared: identical init for both arms

    print("== eight schools ==", flush=True)
    qem_lms, qem_t, qem, qem_state = run_qem_arm(
        model, guide, args.num_samples, args.qem_steps, args.seed, y, sigma
    )
    print(f"QEM:  final log P_MP = {qem_lms[-1]:.3f}  ({qem_t[-1]:.1f}s)", flush=True)

    elbos_neg, svi_per_step, ckpts, svi_params = run_svi_arm(
        model,
        guide,
        Trace_ELBO(num_particles=4),
        args.learning_rate,
        args.svi_steps,
        12,
        args.seed,
        y,
        sigma,
    )
    elbos = -elbos_neg
    ckpt_steps = np.array([s for s, _ in ckpts])
    ckpt_evidence = np.array(
        [
            mpiw_evidence(model, guide, p, args.num_samples, args.seed, y, sigma)
            for _, p in ckpts
        ]
    )
    print(
        f"SVI:  final ELBO = {elbos[-1]:.3f}, MPIW evidence = {ckpt_evidence[-1]:.3f}",
        flush=True,
    )

    # high-K reference evidence at the QEM-fitted guide; mu is summed serially so
    # the resident block is K_tau x K_theta x 8 per scan step instead of K^3 x 8
    ref = mpiw_evidence(
        model,
        guide,
        qem.get_params(qem_state),
        args.ref_num_samples,
        args.seed,
        y,
        sigma,
        num_seeds=10,
        serial_sites=("mu",),
    )
    print(f"reference log P(y) (MPIW, K={args.ref_num_samples}): {ref:.3f}", flush=True)

    # oracle moments and per-arm moments
    nuts = run_nuts(model, args.seed, y, sigma)
    qem_means = posterior_means(guide, qem.get_params(qem_state), args.seed)
    svi_means = posterior_means(guide, svi_params, args.seed)

    # diagnostics
    print(
        "QEM site k-hats:",
        {k: f"{v:.2f}" for k, v in qem.site_khats(qem_state, y, sigma).items()},
        flush=True,
    )

    # (a) evidence vs iteration
    ax = axes[0]
    ax.axhline(ref, color="black", ls="--", lw=1, label="log P(y) (high-K MPIW)")
    ax.plot(np.arange(1, len(qem_lms) + 1), qem_lms, color="C0", label="QEM log P_MP")
    svi_iters = np.arange(1, len(elbos) + 1)
    ax.plot(svi_iters, elbos, color="C1", alpha=0.35, lw=0.8, label="SVI ELBO")
    ax.plot(
        ckpt_steps, ckpt_evidence, "s-", color="C1", label="SVI guide, MPIW evidence"
    )
    ax.set(
        xscale="log",
        xlabel="iteration",
        ylabel="log-evidence estimate",
        title="Eight schools: convergence per iteration",
    )
    ax.set_ylim(ref - 6, ref + 1)
    ax.legend(fontsize=8)

    # (b) evidence vs wall-clock
    ax = axes[1]
    ax.axhline(ref, color="black", ls="--", lw=1)
    ax.plot(qem_t, qem_lms, color="C0", label="QEM log P_MP")
    ax.plot(
        svi_iters * svi_per_step,
        elbos,
        color="C1",
        alpha=0.35,
        lw=0.8,
        label="SVI ELBO",
    )
    ax.plot(
        ckpt_steps * svi_per_step,
        ckpt_evidence,
        "s-",
        color="C1",
        label="SVI guide, MPIW evidence",
    )
    ax.set(
        xscale="log",
        xlabel="wall-clock (s, fitting only)",
        ylabel="log-evidence estimate",
        title="Eight schools: convergence per second",
    )
    ax.set_ylim(ref - 6, ref + 1)
    ax.legend(fontsize=8)

    # (c) posterior means vs NUTS oracle
    ax = axes[2]
    for label, means, marker, color in [
        ("QEM", qem_means, "o", "C0"),
        ("SVI", svi_means, "s", "C1"),
    ]:
        oracle = np.concatenate([[nuts["mu"]], [nuts["tau"]], nuts["theta"]])
        est = np.concatenate([[means["mu"]], [means["tau"]], means["theta"]])
        ax.plot(oracle, est, marker, ms=6, ls="", color=color, alpha=0.8, label=label)
    lims = ax.get_xlim()
    span = np.linspace(min(lims[0], -3), max(lims[1], 12), 2)
    ax.plot(span, span, color="0.7", lw=1, zorder=0)
    ax.set(
        xlabel="NUTS posterior mean",
        ylabel="guide posterior mean",
        title="Posterior means vs NUTS\n(mu, tau, theta[0..7])",
    )
    ax.legend(fontsize=8)

    # (d) alpha-scaling stress test
    ax = axes[3]
    for i, alpha in enumerate([1.0, 1e-2, 1e-4]):
        m = make_schools_model(alpha)
        g = AutoExponentialFamily(m)
        lms, _, _, _ = run_qem_arm(
            m, g, args.num_samples, args.stress_steps, args.seed, y, sigma
        )
        losses, _, _, _ = run_svi_arm(
            m,
            g,
            Trace_ELBO(num_particles=4),
            args.learning_rate,
            args.stress_steps * 20,
            4,
            args.seed,
            y,
            sigma,
        )
        ax.plot(
            np.arange(1, len(lms) + 1),
            lms,
            color="C0",
            alpha=1 - 0.35 * i,
            label=f"QEM alpha={alpha:g}",
        )
        it = np.arange(1, len(losses) + 1) / 20.0  # 20 SVI steps per QEM iter shown
        ax.plot(
            it,
            -losses,
            color="C1",
            alpha=1 - 0.35 * i,
            lw=0.8,
            label=f"SVI alpha={alpha:g}",
        )
    ax.axhline(ref, color="black", ls="--", lw=1)
    ax.set(
        xscale="log",
        xlabel="QEM iteration (20 SVI steps per)",
        ylabel="log-evidence estimate",
        title="alpha-scaling stress:\nQEM invariant, VI degrades",
    )
    ax.set_ylim(ref - 12, ref + 1.5)
    ax.legend(fontsize=7, ncol=2)


# --- Model 2 benchmark ---


def benchmark_occupancy(args, axes):
    Y = get_occupancy_data()
    log_ev, exact_lpsi, exact_lp = occupancy_grid_reference(Y)
    print("== occupancy ==", flush=True)
    print(
        f"exact: log P(y)={log_ev:.3f}  E[lpsi]={exact_lpsi:.3f}  E[lp]={exact_lp:.3f}",
        flush=True,
    )

    # QEM arm: full guide, discrete presence latents sampled
    guide_full = AutoExponentialFamily(occupancy_model)
    qem_lms, qem_t, qem, qem_state = run_qem_arm(
        occupancy_model, guide_full, args.num_samples, args.qem_steps, args.seed, Y
    )
    print(f"QEM:  final log P_MP = {qem_lms[-1]:.3f}  ({qem_t[-1]:.1f}s)", flush=True)

    # VI arm: discrete latents hidden from the guide, marginalized by TraceEnum_ELBO
    guide_cont = AutoExponentialFamily(handlers.block(occupancy_model, hide=["z"]))
    elbos_neg, svi_per_step, ckpts, svi_params = run_svi_arm(
        config_enumerate(occupancy_model),
        guide_cont,
        TraceEnum_ELBO(max_plate_nesting=1),
        args.learning_rate,
        args.svi_steps,
        12,
        args.seed,
        Y,
    )
    elbos = -elbos_neg
    print(f"SVI:  final enum-ELBO = {elbos[-1]:.3f}", flush=True)

    nuts = run_nuts(occupancy_marginalized, args.seed, Y)
    qem_means = posterior_means(guide_full, qem.get_params(qem_state), args.seed)
    svi_means = posterior_means(guide_cont, svi_params, args.seed)

    # (a) evidence vs iteration
    ax = axes[0]
    ax.axhline(log_ev, color="black", ls="--", lw=1, label="exact log P(y)")
    ax.plot(np.arange(1, len(qem_lms) + 1), qem_lms, color="C0", label="QEM log P_MP")
    svi_iters = np.arange(1, len(elbos) + 1)
    ax.plot(svi_iters, elbos, color="C1", alpha=0.6, lw=0.8, label="SVI enum-ELBO")
    ax.set(
        xscale="log",
        xlabel="iteration",
        ylabel="log-evidence estimate",
        title="Occupancy: convergence per iteration",
    )
    ax.set_ylim(log_ev - 6, log_ev + 1)
    ax.legend(fontsize=8)

    # (b) evidence vs wall-clock
    ax = axes[1]
    ax.axhline(log_ev, color="black", ls="--", lw=1)
    ax.plot(qem_t, qem_lms, color="C0", label="QEM log P_MP")
    ax.plot(
        svi_iters * svi_per_step,
        elbos,
        color="C1",
        alpha=0.6,
        lw=0.8,
        label="SVI enum-ELBO",
    )
    ax.set(
        xscale="log",
        xlabel="wall-clock (s, fitting only)",
        ylabel="log-evidence estimate",
        title="Occupancy: convergence per second",
    )
    ax.set_ylim(log_ev - 6, log_ev + 1)
    ax.legend(fontsize=8)

    # (c) posterior means of the logits
    ax = axes[2]
    names = ["lpsi", "lp"]
    x = np.arange(len(names))
    width = 0.2
    ax.bar(x - 1.5 * width, [exact_lpsi, exact_lp], width, color="0.4", label="exact")
    ax.bar(x - 0.5 * width, [nuts[n] for n in names], width, color="C2", label="NUTS")
    ax.bar(
        x + 0.5 * width,
        [float(qem_means[n]) for n in names],
        width,
        color="C0",
        label="QEM",
    )
    ax.bar(
        x + 1.5 * width,
        [float(svi_means[n]) for n in names],
        width,
        color="C1",
        label="SVI",
    )
    ax.set(
        xticks=x,
        xticklabels=names,
        ylabel="posterior mean",
        title="Occupancy: posterior means",
    )
    ax.legend(fontsize=8)


def main(args):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    benchmark_schools(args, axes[0])
    benchmark_occupancy(args, axes[1][:3])
    axes[1][3].axis("off")
    axes[1][3].text(
        0.02,
        0.5,
        "Same mean-field guide family in every arm\n"
        "(AutoExponentialFamily; prior-matched init).\n\n"
        "QEM: gradient-free EM on MPIW moments.\n"
        "SVI: Adam on the (enum-)ELBO.\n\n"
        "Wall-clock caveat: SVI updates are one jitted\n"
        "scan step; QEM updates currently re-trace\n"
        "through funsor every iteration (no jit).",
        fontsize=9,
        va="center",
    )
    fig.suptitle("QEM vs mean-field VI (guide family held fixed)")
    plt.savefig("qem_vs_svi.png", dpi=120)
    print("saved qem_vs_svi.png")


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.21.0")
    parser = argparse.ArgumentParser(description="QEM vs mean-field VI benchmark")
    parser.add_argument(
        "--num-samples",
        default=64,
        type=int,
        help="K (proposals per latent site) for QEM/MPIW; NOTE the "
        "schools model contracts a K^3-coupled factor densely, so "
        "memory grows as K^3 * 8 * 8 bytes (~16 MB at K=64, ~1 GB "
        "at K=256)",
    )
    parser.add_argument(
        "--ref-num-samples",
        default=256,
        type=int,
        help="K for the high-K reference evidence. NOTE serial "
        "contraction bounds intermediates but the theta input "
        "factor is still materialized densely at K^3 * 8 entries "
        "(~1 GB at K=256, ~8.6 GB at K=512)",
    )
    parser.add_argument("--qem-steps", default=100, type=int)
    parser.add_argument("--svi-steps", default=5000, type=int)
    parser.add_argument(
        "--stress-steps",
        default=50,
        type=int,
        help="QEM iterations per alpha in the stress test",
    )
    parser.add_argument("--learning-rate", default=0.05, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    main(args)
