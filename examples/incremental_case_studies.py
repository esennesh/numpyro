# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Incremental Densities, Case Studies 1 and 2
====================================================

Reproduces the first two case studies of [1] with NumPyro's incremental densities,
using the models, priors, kernels and data protocol of the paper's artifact
(Zenodo 10.5281/zenodo.19080766, ``examples/robust_regression`` and
``examples/gmms/binary_gmm``):

1. **Robust regression** (Box and Tiao), N = 1000 points on a grid from -5 to 5:
   ``m, b ~ Normal(0, 1)``; each point is an outlier with probability 0.1, in which case
   ``y ~ Normal(0, 100)``, else ``y ~ Normal(m x + b, 0.5)``. Kernels: a joint
   Gaussian drift of width 0.5 on ``(m, b)`` (Metropolis) and a Metropolis flip of each
   outlier indicator.
2. **Binary 2-D Gaussian mixture** (Diebolt and Robert), N = 500: mixing weight
   ``~ Beta(0.5, 0.5)``; per component a covariance ``~ InverseWishart(3.2, 1.6 I)``
   and a mean ``~ Normal(mu0_k, Sigma_k / 0.4)`` with ``mu0 = (-2, -2), (2, 2)``;
   assignments ``~ Bernoulli(weight)`` and observations ``~ Normal(mu_z, Sigma_z)``.
   Kernels: the exact conditional of one assignment (Gibbs) and a conjugate
   Normal-Inverse-Wishart / Beta resample of all parameters given the assignments.

As in the artifact, the data are simulated from the model's prior, inference starts
from a fresh prior draw, and the median wall time per kernel application is reported
for the incremental density and for full recomputation of the joint density. Here the
per-application times are measured inside one compiled sweep so that JAX dispatch
overhead (about 10 µs per call, larger than either quantity) does not dominate. The
paper's Julia prototype reports 0.017 ms (incremental) vs 3.06 ms (full) for an
outlier flip, 0.028 ms vs 1.68 ms for a cluster assignment, and the parameter kernels
at about half the speed of full recomputation; XLA's vectorised full evaluation is
far cheaper than Julia's, so the absolute numbers differ while the scaling (constant
vs linear in N) is the same.

**References:**

    1. *Incremental Computation for Efficient Programmable Inference in Probabilistic
       Programs*, Fabian Zaiser et al., PACMPL 10 (PLDI), 2026.
"""

import argparse
import time

import numpy as np

import jax
from jax import lax, random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.incremental import (
    Change,
    DiscreteGibbs,
    GibbsFn,
    IncrementalDensity,
    IncrementalMCMC,
    RandomWalk,
)
import numpyro.distributions as dist
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC

# ----------------------------------------------------------------- models

# examples/robust_regression/consts.jl
OUTLIER_PROB = 0.1
INLIER_NOISE = 0.5
OUTLIER_NOISE = 100.0
DRIFT_WIDTH = 0.5

# examples/gmms/binary_gmm/consts.jl
DIM = 2
COMPONENT_PRIOR_MEANS = jnp.array([[-2.0, -2.0], [2.0, 2.0]])
KAPPA0 = 0.4
NU0 = DIM + 1.2
PSI0 = 1.6 * jnp.eye(DIM)
ALPHA_MIX = 0.5


def robust_regression(xs, ys=None):
    m = numpyro.sample("m", dist.Normal(0.0, 1.0))
    b = numpyro.sample("b", dist.Normal(0.0, 1.0))
    with numpyro.plate("data", xs.shape[0]):
        o = numpyro.sample("o", dist.Bernoulli(OUTLIER_PROB))
        loc = jnp.where(o == 1, 0.0, m * xs + b)
        scale = jnp.where(o == 1, OUTLIER_NOISE, INLIER_NOISE)
        numpyro.sample("y", dist.Normal(loc, scale), obs=ys)


def binary_gmm(num_points, observations=None):
    weight = numpyro.sample("weight", dist.Beta(ALPHA_MIX, ALPHA_MIX))
    with numpyro.plate("component", 2):
        cov = numpyro.sample("cov", dist.InverseWishart(NU0, PSI0))
        mu = numpyro.sample(
            "mu",
            dist.MultivariateNormal(
                COMPONENT_PRIOR_MEANS, covariance_matrix=cov / KAPPA0
            ),
        )
    with numpyro.plate("data", num_points):
        z = numpyro.sample("assignment", dist.Bernoulli(weight))
        numpyro.sample(
            "observation",
            dist.MultivariateNormal(mu[z], covariance_matrix=cov[z]),
            obs=observations,
        )


def simulate(model, rng_key, *args, **kwargs):
    """A ground-truth trace from the prior, as ``IncrementalPPL.simulate`` does."""
    tr = trace(seed(model, rng_key)).get_trace(*args, **kwargs)
    return {k: v["value"] for k, v in tr.items() if v["type"] == "sample"}


def simulate_gmm(seed_value, num_points):
    """
    A prior draw of the mixture whose Beta(0.5, 0.5) mixing weight is not degenerate
    (the artifact seeds at random; a weight near 0 or 1 leaves one component unseen).
    """
    for k in range(100):
        truth = simulate(binary_gmm, random.key(seed_value + 1000 * k), num_points)
        if 0.25 <= float(truth["weight"]) <= 0.75:
            return truth
    raise RuntimeError("no non-degenerate prior draw found")


# ----------------------------------------------------------------- kernels


def conjugate_parameter_update(observations):
    """
    ``resample_parameters!`` of the artifact: a Beta draw for the mixing weight and,
    per component, an Inverse-Wishart / Normal draw from the Normal-Inverse-Wishart
    posterior given the current assignments (``niw_posterior`` in ``utils.jl``).
    """

    def gibbs_fn(rng_key, gibbs_sites, others):
        z = others["assignment"]
        key_w, key_c = random.split(rng_key)
        n1 = jnp.sum(z == 1)
        n0 = z.shape[0] - n1
        weight = dist.Beta(ALPHA_MIX + n1, ALPHA_MIX + n0).sample(key_w)

        def component(k, key):
            mask = (z == k)[:, None]
            n = jnp.sum(mask)
            mean_x = jnp.sum(observations * mask, 0) / jnp.maximum(n, 1)
            diff = (observations - mean_x) * mask
            scatter = diff.T @ diff
            mu0 = COMPONENT_PRIOR_MEANS[k]
            kappa_n = KAPPA0 + n
            nu_n = NU0 + n
            d = mean_x - mu0
            psi_n = PSI0 + scatter + (KAPPA0 * n / kappa_n) * jnp.outer(d, d)
            mu_n = (KAPPA0 * mu0 + n * mean_x) / kappa_n
            # n == 0: the posterior is the prior.
            psi_n = jnp.where(n > 0, psi_n, PSI0)
            mu_n = jnp.where(n > 0, mu_n, mu0)
            key_cov, key_mu = random.split(key)
            cov = dist.InverseWishart(nu_n, psi_n).sample(key_cov)
            mu = dist.MultivariateNormal(mu_n, covariance_matrix=cov / kappa_n).sample(
                key_mu
            )
            return mu, cov

        mus, covs = jax.vmap(component)(jnp.arange(2), random.split(key_c, 2))
        return {"weight": weight, "mu": mus, "cov": covs}

    return gibbs_fn


# ----------------------------------------------------------------- timing


def median_seconds(fn, *args, repeats=5):
    """Median wall time of a jitted ``fn`` over ``repeats`` calls (after compiling)."""
    fn = jax.jit(fn)
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def time_single_site_sweep(inc, state, site, plate, model_args, model_kwargs):
    """
    Microseconds per single-site update, incremental vs full recomputation, measured
    over one compiled sweep that flips every element of ``site`` in turn (the
    incremental sweep also writes the cache, as a kernel would).
    """
    n = inc.sites[site].shape[dict(inc.sites[site].plates)[plate]]
    probe = Change.sliced({site: jnp.zeros(1, inc.sites[site].dtype)}, {plate: 0})
    inc.plan(probe)
    assert site not in inc.input_full_reads(probe)
    state = inc.resync(state, model_args, model_kwargs)

    def incremental_sweep(state):
        snapshot = state.z  # reads come from the pre-sweep values (exact: one
        # read per element at its own position); the loop carry is write-only.

        def body(i, s):
            new = (1 - snapshot[site][i])[None]
            change = Change.sliced({site: new}, {plate: i})
            return inc.apply(s, change, model_args, model_kwargs, read_values=snapshot)[
                1
            ]

        out = lax.fori_loop(0, n, body, state)
        return out.total + out.z[site].sum()

    def full_sweep(z):
        def body(i, carry):
            z, acc = carry
            z = {**z, site: z[site].at[i].set(1 - z[site][i])}
            return z, acc + inc.program.log_density(z, model_args, model_kwargs)

        return lax.fori_loop(0, n, body, (z, jnp.zeros(())))[1]

    t_inc = median_seconds(incremental_sweep, state) * 1e6 / n
    t_full = median_seconds(full_sweep, state.z) * 1e6 / n
    return t_inc, t_full


def time_parameter_moves(inc, state, params, model_args, model_kwargs, num_moves=100):
    """Microseconds per dense change of all parameters (what a parameter kernel costs)."""
    change = Change.dense({k: state.z[k] for k in params})
    inc.plan(change)
    state = inc.resync(state, model_args, model_kwargs)

    def incremental_loop(state):
        def body(i, s):
            prop = {k: s.z[k] + 1e-3 for k in params}
            return inc.apply(s, Change.dense(prop), model_args, model_kwargs)[1]

        return lax.fori_loop(0, num_moves, body, state).total

    def full_loop(z):
        def body(i, carry):
            z, acc = carry
            z = {**z, **{k: z[k] + 1e-3 for k in params}}
            return z, acc + inc.program.log_density(z, model_args, model_kwargs)

        return lax.fori_loop(0, num_moves, body, (z, jnp.zeros(())))[1]

    t_inc = median_seconds(incremental_loop, state) * 1e6 / num_moves
    t_full = median_seconds(full_loop, state.z) * 1e6 / num_moves
    return t_inc, t_full


def report(name, rows):
    print(f"\n{name}: median µs per kernel application")
    print(f"  {'kernel':<22} {'N':>7} {'incremental':>12} {'full':>10} {'speedup':>8}")
    for kernel, n, t_inc, t_full in rows:
        print(
            f"  {kernel:<22} {n:>7d} {t_inc:>12.2f} {t_full:>10.2f} {t_full / t_inc:>7.1f}x"
        )


# ----------------------------------------------------------------- runs


def run_regression(args):
    n = args.num_data
    xs = jnp.linspace(-5.0, 5.0, n)  # xs_of_length(N) in the artifact
    truth = simulate(robust_regression, random.key(args.seed), xs)
    model_args, model_kwargs = (xs,), {"ys": truth["y"]}
    kernel = IncrementalMCMC(
        robust_regression,
        [
            (None, DiscreteGibbs(random_walk=True, modified=True)),  # MH flips
            (("m", "b"), RandomWalk(scale=DRIFT_WIDTH)),  # drift_line!
        ],
    )
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples)
    start = time.time()
    mcmc.run(random.key(args.seed + 1), *model_args, **model_kwargs)
    s = mcmc.get_samples()
    acc = ((s["o"].mean(0) > 0.5) == (truth["o"] == 1)).mean()
    print(
        f"robust regression (N = {n}, {time.time() - start:.1f}s): m {s['m'].mean():.3f} "
        f"(truth {truth['m']:.3f}), b {s['b'].mean():.3f} (truth {truth['b']:.3f}), "
        f"outlier rate {s['o'].mean():.3f} (truth {truth['o'].mean():.3f}), indicator "
        f"accuracy {acc:.3f}"
    )
    rows = []
    for size in (n, 10 * n):
        xs_n = jnp.linspace(-5.0, 5.0, size)
        ys_n = simulate(robust_regression, random.key(args.seed), xs_n)["y"]
        inc = IncrementalDensity(robust_regression, (xs_n,), {"ys": ys_n}, cache="plan")
        state = inc.init(model_args=(xs_n,), model_kwargs={"ys": ys_n})
        t_inc, t_full = time_single_site_sweep(
            inc, state, "o", "data", (xs_n,), {"ys": ys_n}
        )
        rows.append(("outlier status", size, t_inc, t_full))
        t_inc, t_full = time_parameter_moves(
            inc, state, ("m", "b"), (xs_n,), {"ys": ys_n}
        )
        rows.append(("parameters", size, t_inc, t_full))
    report("robust regression", rows)


def run_gmm(args):
    n = args.num_points
    truth = simulate_gmm(args.seed, n)
    obs = truth["observation"]
    model_args, model_kwargs = (n,), {"observations": obs}
    kernel = IncrementalMCMC(
        binary_gmm,
        [
            (None, DiscreteGibbs()),  # resample_assignment!
            (("weight", "mu", "cov"), GibbsFn(conjugate_parameter_update(obs))),
        ],
    )
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples)
    start = time.time()
    mcmc.run(random.key(args.seed + 1), *model_args, **model_kwargs)
    s = mcmc.get_samples()
    same = ((s["assignment"].mean(0) > 0.5) == (truth["assignment"] == 1)).mean()
    accuracy = max(same, 1.0 - same)  # up to label switching
    print(
        f"binary GMM (N = {n}, {time.time() - start:.1f}s): means "
        f"{np.round(np.asarray(s['mu'].mean(0)), 2).tolist()} (truth "
        f"{np.round(np.asarray(truth['mu']), 2).tolist()}), weight {s['weight'].mean():.2f} "
        f"(truth {truth['weight']:.2f}), assignment accuracy {accuracy:.3f}"
    )
    rows = []
    for size in (n, 10 * n):
        obs_n = simulate_gmm(args.seed, size)["observation"]
        inc = IncrementalDensity(
            binary_gmm, (size,), {"observations": obs_n}, cache="plan"
        )
        state = inc.init(model_args=(size,), model_kwargs={"observations": obs_n})
        t_inc, t_full = time_single_site_sweep(
            inc, state, "assignment", "data", (size,), {"observations": obs_n}
        )
        rows.append(("cluster assignment", size, t_inc, t_full))
        t_inc, t_full = time_parameter_moves(
            inc, state, ("weight", "mu", "cov"), (size,), {"observations": obs_n}
        )
        rows.append(("parameters", size, t_inc, t_full))
    report("binary GMM", rows)


def main(args):
    run_regression(args)
    run_gmm(args)


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.21.0")
    parser = argparse.ArgumentParser(description="Incremental densities: case studies")
    parser.add_argument("--num-data", default=1000, type=int, help="regression N")
    parser.add_argument("--num-points", default=500, type=int, help="GMM N")
    parser.add_argument("--num-warmup", default=200, type=int)
    parser.add_argument("--num-samples", default=200, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    main(args)
