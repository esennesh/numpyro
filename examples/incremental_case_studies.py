# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Incremental Densities, Case Studies 1 and 2
====================================================

Reproduces the first two case studies of [1] with NumPyro's incremental densities:

1. **Robust regression** (Box and Tiao's scale-inflated outlier model) with N = 1000:
   Gaussian-drift Metropolis moves on the parameters and Metropolis flips of one
   outlier indicator at a time.
2. **Binary 2-D Gaussian mixture** with N = 500: Gaussian-drift moves on the
   component parameters and mixing weight, and single cluster-assignment updates
   (exact conditional).

The paper does not publish its priors in the text, so the models below use standard
choices. The kernels follow the paper. The timing protocol reports the median wall
time per kernel application for the incremental density and for full recomputation of
the joint density, measured inside one compiled sweep so that JAX dispatch overhead
(about 10 µs per call, larger than either quantity at these sizes) does not dominate.
The paper's Table 1 reports, on its Julia prototype, 0.017 ms (incremental) vs 3.06 ms
(full) for an outlier flip and 0.028 ms vs 1.68 ms for a cluster assignment, and the
incremental parameter moves at about half the speed of full recomputation; XLA's
vectorised full evaluation is far cheaper than Julia's, so the absolute numbers differ
while the scaling (constant vs linear in N) is the same.

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
    IncrementalDensity,
    IncrementalMCMC,
    RandomWalk,
)
import numpyro.distributions as dist
from numpyro.infer import MCMC
from numpyro.infer.initialization import init_to_median

# ----------------------------------------------------------------- models


def robust_regression(x, y):
    """Box & Tiao: an outlier's noise scale is inflated by a factor of 10."""
    slope = numpyro.sample("slope", dist.Normal(0.0, 10.0))
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 10.0))
    sigma = numpyro.sample("sigma", dist.LogNormal(0.0, 1.0))
    p_outlier = numpyro.sample("p_outlier", dist.Beta(1.0, 9.0))
    with numpyro.plate("data", x.shape[0]):
        is_outlier = numpyro.sample("is_outlier", dist.Bernoulli(p_outlier))
        scale = jnp.where(is_outlier == 1, 10.0 * sigma, sigma)
        numpyro.sample("y", dist.Normal(slope * x + intercept, scale), obs=y)


def binary_gmm(points):
    """Two isotropic Gaussians in the plane with an unknown mixing weight."""
    weight = numpyro.sample("weight", dist.Beta(1.0, 1.0))
    with numpyro.plate("component", 2):
        mu = numpyro.sample("mu", dist.Normal(0.0, 10.0).expand([2]).to_event(1))
        sigma = numpyro.sample("sigma", dist.LogNormal(0.0, 1.0))
    with numpyro.plate("data", points.shape[0]):
        z = numpyro.sample("z", dist.Bernoulli(weight))
        numpyro.sample(
            "obs", dist.Normal(mu[z], sigma[z][:, None]).to_event(1), obs=points
        )


REGRESSION_PARAMS = ("slope", "intercept", "sigma", "p_outlier")
REGRESSION_SCALES = {"slope": 0.01, "intercept": 0.02, "sigma": 0.05, "p_outlier": 0.15}
GMM_PARAMS = ("mu", "sigma", "weight")
GMM_SCALES = {"mu": 0.05, "sigma": 0.05, "weight": 0.15}


def regression_data(n, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-3.0, 3.0, size=n).astype(np.float32)
    outlier = rng.rand(n) < 0.1
    noise = rng.randn(n) * np.where(outlier, 5.0, 0.5)
    y = (2.0 * x - 1.0 + noise).astype(np.float32)
    return x, y, outlier


def gmm_data(n, seed=0):
    rng = np.random.RandomState(seed)
    z = rng.rand(n) < 0.3
    centers = np.where(z[:, None], np.array([3.0, 3.0]), np.array([-3.0, -2.0]))
    scales = np.where(z, 1.0, 0.6)[:, None]
    return (centers + scales * rng.randn(n, 2)).astype(np.float32), z


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


def time_single_site_sweep(inc, state, site, plate, model_args, support_size=2):
    """
    Microseconds per single-site update, incremental vs full recomputation, measured
    over one compiled sweep that flips every element of ``site`` in turn (the
    incremental sweep also writes the cache, as a kernel would).
    """
    n = inc.sites[site].shape[dict(inc.sites[site].plates)[plate]]
    probe = Change.sliced({site: jnp.zeros(1, inc.sites[site].dtype)}, {plate: 0})
    inc.plan(probe)
    state = inc.resync(state, model_args)

    def incremental_sweep(state):
        snapshot = state.z  # reads come from the pre-sweep values (exact: one
        # read per element at its own position); the loop carry is write-only.

        def body(i, s):
            new = ((snapshot[site][i] + 1) % support_size)[None]
            change = Change.sliced({site: new}, {plate: i})
            return inc.apply(s, change, model_args, read_values=snapshot)[1]

        out = lax.fori_loop(0, n, body, state)
        return out.total + out.z[site].sum()

    def full_sweep(z):
        def body(i, carry):
            z, acc = carry
            z = {**z, site: z[site].at[i].set((z[site][i] + 1) % support_size)}
            return z, acc + inc.program.log_density(z, model_args)

        return lax.fori_loop(0, n, body, (z, jnp.zeros(())))[1]

    t_inc = median_seconds(incremental_sweep, state) * 1e6 / n
    t_full = median_seconds(full_sweep, state.z) * 1e6 / n
    return t_inc, t_full


def time_parameter_moves(inc, state, params, model_args, num_moves=100):
    """Microseconds per dense parameter change (a Gaussian-drift proposal)."""
    change = Change.dense({k: state.z[k] for k in params})
    inc.plan(change)
    state = inc.resync(state, model_args)

    def incremental_loop(state):
        def body(i, s):
            prop = {k: s.z[k] + 1e-3 for k in params}
            return inc.apply(s, Change.dense(prop), model_args)[1]

        return lax.fori_loop(0, num_moves, body, state).total

    def full_loop(z):
        def body(i, carry):
            z, acc = carry
            z = {**z, **{k: z[k] + 1e-3 for k in params}}
            return z, acc + inc.program.log_density(z, model_args)

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
    x, y, truth = regression_data(args.num_data)
    kernel = IncrementalMCMC(
        robust_regression,
        [
            (None, DiscreteGibbs(random_walk=True, modified=True)),  # MH flips
            (REGRESSION_PARAMS, RandomWalk(scale=REGRESSION_SCALES)),
        ],
        init_strategy=init_to_median,
    )
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples)
    start = time.time()
    mcmc.run(random.key(0), x, y)
    s = mcmc.get_samples()
    print(
        f"robust regression (N = {args.num_data}, {time.time() - start:.1f}s): slope "
        f"{s['slope'].mean():.2f} (truth 2), intercept {s['intercept'].mean():.2f} "
        f"(truth -1), sigma {s['sigma'].mean():.2f} (truth 0.5), outlier rate "
        f"{s['is_outlier'].mean():.3f} (truth {truth.mean():.3f}), indicator accuracy "
        f"{((s['is_outlier'].mean(0) > 0.5) == truth).mean():.3f}"
    )
    rows = []
    for n in (args.num_data, 10 * args.num_data):
        xn, yn, _ = regression_data(n)
        inc = IncrementalDensity(robust_regression, (xn, yn), cache="plan")
        state = inc.init(model_args=(xn, yn))
        t_inc, t_full = time_single_site_sweep(
            inc, state, "is_outlier", "data", (xn, yn)
        )
        rows.append(("outlier status", n, t_inc, t_full))
        t_inc, t_full = time_parameter_moves(inc, state, REGRESSION_PARAMS, (xn, yn))
        rows.append(("parameters", n, t_inc, t_full))
    report("robust regression", rows)


def run_gmm(args):
    points, truth = gmm_data(args.num_points)
    kernel = IncrementalMCMC(
        binary_gmm,
        [
            (None, DiscreteGibbs()),  # exact conditional of one assignment
            (GMM_PARAMS, RandomWalk(scale=GMM_SCALES)),
        ],
        init_strategy=init_to_median,
    )
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples)
    start = time.time()
    mcmc.run(random.key(0), points)
    s = mcmc.get_samples()
    assignment = (s["z"].mean(0) > 0.5) == truth
    accuracy = max(assignment.mean(), 1.0 - assignment.mean())  # up to label switching
    print(
        f"binary GMM (N = {args.num_points}, {time.time() - start:.1f}s): means "
        f"{np.round(np.asarray(s['mu'].mean(0)), 2).tolist()} (truth [-3, -2] and "
        f"[3, 3]), scales {np.round(np.asarray(s['sigma'].mean(0)), 2).tolist()} (truth "
        f"0.6 and 1.0), weight {s['weight'].mean():.2f} (truth 0.3 or 0.7), assignment "
        f"accuracy {accuracy:.3f}"
    )
    rows = []
    for n in (args.num_points, 10 * args.num_points):
        pts, _ = gmm_data(n)
        inc = IncrementalDensity(binary_gmm, (pts,), cache="plan")
        state = inc.init(model_args=(pts,))
        t_inc, t_full = time_single_site_sweep(inc, state, "z", "data", (pts,))
        rows.append(("cluster assignment", n, t_inc, t_full))
        t_inc, t_full = time_parameter_moves(inc, state, GMM_PARAMS, (pts,))
        rows.append(("parameters", n, t_inc, t_full))
    report("binary GMM", rows)


def main(args):
    run_regression(args)
    run_gmm(args)


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.21.0")
    parser = argparse.ArgumentParser(description="Incremental densities: case studies")
    parser.add_argument("--num-data", default=1000, type=int, help="regression N")
    parser.add_argument("--num-points", default=500, type=int, help="GMM N")
    parser.add_argument("--num-warmup", default=2000, type=int)
    parser.add_argument("--num-samples", default=1000, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    main(args)
