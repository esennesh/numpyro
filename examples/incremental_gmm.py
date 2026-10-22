# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Incremental Gibbs for a Gaussian Mixture
=================================================

A Gaussian mixture with per-datum assignments ``z`` is sampled two ways:

1. ``DiscreteHMCGibbs(NUTS(model))`` as is: every single-assignment Gibbs update
   evaluates the full model potential once per candidate component, so a sweep over
   ``N`` assignments costs ``O(N^2 K)``.
2. The same kernel adapted by ``IncrementalMCMC.from_kernel``: the model is compiled to
   an incremental density program [1], and each assignment update re-executes only the
   equations that depend on that datum, so a sweep costs ``O(N K)``.

The script prints the wall time of a sampling run with each kernel, the posterior means
they agree on, and the compiled flop count of one assignment update at two data sizes,
which stays the same.

**References:**

    1. *Incremental Computation for Efficient Programmable Inference in Probabilistic
       Programs*, Fabian Zaiser et al., PACMPL 10 (PLDI), 2026.
"""

import argparse
import time

import numpy as np

import jax
from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.incremental import Change, IncrementalDensity, IncrementalMCMC
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, DiscreteHMCGibbs


def model(y, num_components=3):
    w = numpyro.sample("w", dist.Dirichlet(jnp.ones(num_components)))
    s = numpyro.sample("s", dist.LogNormal(0.0, 0.5))
    with numpyro.plate("comp", num_components):
        mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
    with numpyro.plate("data", y.shape[0]):
        z = numpyro.sample("z", dist.Categorical(w))
        numpyro.sample("obs", dist.Normal(mu[z], s), obs=y)


def make_data(num_data, num_components, seed=0):
    rng = np.random.RandomState(seed)
    centers = np.linspace(-4.0, 4.0, num_components)
    z = rng.randint(num_components, size=num_data)
    return (centers[z] + 0.7 * rng.randn(num_data)).astype(np.float32)


def run(kernel, y, args, name):
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        progress_bar=False,
    )
    # First run includes compilation; the second reuses the compiled step.
    mcmc.run(random.key(args.seed), y, num_components=args.num_components)
    start = time.time()
    mcmc.run(random.key(args.seed + 1), y, num_components=args.num_components)
    elapsed = time.time() - start
    samples = mcmc.get_samples()
    mu = np.sort(np.asarray(samples["mu"].mean(0)))
    print(
        f"{name:>12}: {elapsed:7.2f}s for {args.num_warmup + args.num_samples} "
        f"steps; posterior mean mu = {np.round(mu, 2)}, s = {samples['s'].mean():.2f}"
    )
    return mu


def flops_of_one_assignment_update(y, num_components):
    inc = IncrementalDensity(model, (y,), {"num_components": num_components})
    change = Change.sliced({"z": jnp.array([1])}, {"data": jnp.array([len(y) // 2])})
    inc.plan(change)
    state = inc.init(model_args=(y,), model_kwargs={"num_components": num_components})

    def delta(state, change, y):
        return inc.apply_delta(state, change, (y,), {"num_components": num_components})[
            0
        ]

    compiled = jax.jit(delta).lower(state, change, y).compile()
    cost = compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    return int(cost["flops"]), inc.num_reexecuted(change), inc.program.num_eqns


def main(args):
    y = make_data(args.num_data, args.num_components, args.seed)
    print(f"N = {args.num_data} data points, K = {args.num_components} components")

    reference = DiscreteHMCGibbs(NUTS(model))
    adapted = IncrementalMCMC.from_kernel(DiscreteHMCGibbs(NUTS(model)))
    mu_ref = run(reference, y, args, "DiscreteHMCGibbs")
    mu_inc = run(adapted, y, args, "incremental")
    print(
        f"max |difference of posterior means| = {np.max(np.abs(mu_ref - mu_inc)):.3f}"
    )

    for n in (args.num_data, 10 * args.num_data):
        flops, eqns, total_eqns = flops_of_one_assignment_update(
            make_data(n, args.num_components, args.seed), args.num_components
        )
        print(
            f"one assignment update at N = {n:6d}: {flops} flops, {eqns} of "
            f"{total_eqns} equations re-executed"
        )


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.21.0")
    parser = argparse.ArgumentParser(description="Incremental Gibbs for a GMM")
    parser.add_argument("--num-data", default=400, type=int)
    parser.add_argument("--num-components", default=3, type=int)
    parser.add_argument("--num-warmup", default=300, type=int)
    parser.add_argument("--num-samples", default=300, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    main(args)
