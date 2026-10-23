# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.testing import assert_allclose
import pytest

from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.incremental import DiscreteGibbs, IncrementalMCMC, RandomWalk
import numpyro.distributions as dist
from numpyro.infer import HMC, MCMC, NUTS, SA, DiscreteHMCGibbs, HMCGibbs


def gmm_1d(probs, locs):
    c = numpyro.sample("c", dist.Categorical(probs))
    numpyro.sample("x", dist.Normal(locs[c], 0.5))


@pytest.mark.parametrize("modified", [False, True])
def test_from_discrete_hmc_gibbs_matches_analytic_moments(modified):
    probs = jnp.array([0.15, 0.3, 0.3, 0.25])
    locs = jnp.array([-2.0, 0.0, 2.0, 4.0])
    kernel = IncrementalMCMC.from_kernel(
        DiscreteHMCGibbs(NUTS(gmm_1d, trajectory_length=1.2), modified=modified)
    )
    mcmc = MCMC(kernel, num_warmup=500, num_samples=200000, progress_bar=False)
    mcmc.run(random.key(0), probs, locs)
    samples = mcmc.get_samples()
    # The chain hops between well-separated modes, so the moments of x carry a
    # Monte Carlo error of a few hundredths even at this length (the reference
    # DiscreteHMCGibbs shows the same spread). The marginal of c is exact.
    freq = jnp.bincount(samples["c"], length=4) / samples["c"].shape[0]
    assert_allclose(freq, probs, atol=0.02)
    assert_allclose(jnp.mean(samples["x"]), 1.3, atol=0.15)
    assert_allclose(jnp.var(samples["x"]), 4.36, atol=0.5)
    assert_allclose(jnp.mean(samples["c"]), 1.65, atol=0.1)
    assert_allclose(jnp.var(samples["c"]), 1.03, atol=0.1)


def plated_gmm(y):
    K = 2
    with numpyro.plate("comp", K):
        mu = numpyro.sample("mu", dist.Normal(jnp.array([-1.0, 1.0]), 3.0))
    s = numpyro.sample("s", dist.LogNormal(0.0, 0.5))
    with numpyro.plate("data", y.shape[0]):
        z = numpyro.sample("z", dist.Categorical(jnp.ones(K) / K))
        numpyro.deterministic("loc", mu[z])
        numpyro.sample("obs", dist.Normal(mu[z], s), obs=y)


def _gmm_data(n=60):
    rng = np.random.RandomState(0)
    z = rng.rand(n) < 0.5
    return np.where(z, 3.0, -3.0).astype(np.float32) + rng.randn(n).astype(np.float32)


def test_plated_gmm_agrees_with_discrete_hmc_gibbs():
    y = _gmm_data()
    ref = MCMC(
        DiscreteHMCGibbs(NUTS(plated_gmm)),
        num_warmup=300,
        num_samples=600,
        progress_bar=False,
    )
    ref.run(random.key(1), y)
    kernel = IncrementalMCMC.from_kernel(DiscreteHMCGibbs(NUTS(plated_gmm)))
    mcmc = MCMC(kernel, num_warmup=300, num_samples=600, progress_bar=False)
    mcmc.run(random.key(2), y)
    a, b = ref.get_samples(), mcmc.get_samples()
    assert set(a) == set(b)  # incl. the deterministic site
    assert b["loc"].shape == (600, y.shape[0])
    mu_ref = jnp.sort(a["mu"].mean(0))
    mu_inc = jnp.sort(b["mu"].mean(0))
    assert_allclose(mu_inc, mu_ref, atol=0.3)
    assert_allclose(mu_inc, jnp.array([-3.0, 3.0]), atol=0.5)
    assert_allclose(b["s"].mean(), a["s"].mean(), atol=0.3)
    # z is constrained-space (raw) and obs assignments are mostly right.
    frac = (b["z"] == b["z"][:, :1]).mean()
    assert 0.1 < frac < 0.9


def test_explicit_steps_with_sa_and_vectorized_chains():
    y = _gmm_data(40)
    kernel = IncrementalMCMC(
        plated_gmm,
        [(None, DiscreteGibbs(order="systematic")), (("mu", "s"), SA(plated_gmm))],
        resync_every=25,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=200,
        num_samples=200,
        num_chains=2,
        chain_method="vectorized",
        progress_bar=False,
    )
    mcmc.run(random.key(3), y)
    samples = mcmc.get_samples(group_by_chain=True)
    assert samples["mu"].shape == (2, 200, 2)
    assert samples["z"].shape == (2, 200, y.shape[0])
    assert jnp.all(jnp.isfinite(samples["s"]))
    # The incremental total stays in sync with a from-scratch evaluation.
    state = mcmc.last_state
    inc = kernel.incremental_density
    for chain in range(2):
        z = {k: v[chain] for k, v in state.z.items()}
        total = inc.program.log_density(z, (y,))
        assert_allclose(state.inc.total[chain], total, rtol=1e-4, atol=1e-2)


def test_from_hmc_gibbs_with_user_gibbs_fn():
    def model():
        x = numpyro.sample("x", dist.Normal(0.0, 2.0))
        y = numpyro.sample("y", dist.Normal(0.0, 2.0))
        numpyro.sample("obs", dist.Normal(x + y, 1.0), obs=jnp.array(1.0))

    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        # exact conditional of x given y: prior N(0, 4), likelihood N(1 - y, 1)
        y = hmc_sites["y"]
        var = 1.0 / (1.0 / 4.0 + 1.0)
        mean = var * (1.0 - y)
        return {"x": mean + jnp.sqrt(var) * random.normal(rng_key)}

    hmc_gibbs = HMCGibbs(NUTS(model), gibbs_fn=gibbs_fn, gibbs_sites=["x"])
    kernel = IncrementalMCMC.from_kernel(hmc_gibbs)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=20000, progress_bar=False)
    mcmc.run(random.key(1))
    samples = mcmc.get_samples()
    # Exact posterior: x and y are exchangeable with mean 4/9 and variance 4 - 16/9.
    for name in ("x", "y"):
        assert_allclose(samples[name].mean(), 4.0 / 9.0, atol=0.1)
        assert_allclose(samples[name].std(), np.sqrt(4.0 - 16.0 / 9.0), atol=0.1)


def test_plain_hmc_adapts():
    def model(y):
        mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
        numpyro.sample("obs", dist.Normal(mu, 1.0), obs=y)

    y = jnp.array([1.0, 2.0, 3.0])
    kernel = IncrementalMCMC.from_kernel(HMC(model, trajectory_length=1.0))
    mcmc = MCMC(kernel, num_warmup=300, num_samples=1000, progress_bar=False)
    mcmc.run(random.key(0), y)
    mu = mcmc.get_samples()["mu"]
    post_var = 1.0 / (3.0 + 1.0 / 100.0)
    assert_allclose(mu.mean(), post_var * y.sum(), atol=0.15)
    assert_allclose(mu.var(), post_var, atol=0.1)
    assert "acc. prob" in kernel.get_diagnostics_str(mcmc.last_state)


def test_random_walk_block_targets_posterior():
    def model(y):
        mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
        s = numpyro.sample("s", dist.LogNormal(0.0, 1.0))
        numpyro.sample("obs", dist.Normal(mu, s), obs=y)

    y = jnp.array([1.0, 2.0, 3.0, 4.0])
    ref = MCMC(NUTS(model), num_warmup=500, num_samples=4000, progress_bar=False)
    ref.run(random.key(0), y)
    kernel = IncrementalMCMC(
        model, [("mu", RandomWalk(scale=0.8)), ("s", RandomWalk(scale=0.5))]
    )
    mcmc = MCMC(kernel, num_warmup=500, num_samples=40000, progress_bar=False)
    mcmc.run(random.key(1), y)
    a, b = ref.get_samples(), mcmc.get_samples()
    for name in ("mu", "s"):
        assert_allclose(b[name].mean(), a[name].mean(), atol=0.15)
        assert_allclose(b[name].std(), a[name].std(), atol=0.15)
    assert "rw acc. prob" in kernel.get_diagnostics_str(mcmc.last_state)
    with pytest.raises(ValueError, match="continuous"):
        IncrementalMCMC(
            plated_gmm, [("z", RandomWalk()), (None, NUTS(plated_gmm))]
        ).init(random.key(0), 10, None, (_gmm_data(10),), {})


def test_step_validation():
    y = _gmm_data(10)
    with pytest.raises(ValueError, match="not updated"):
        IncrementalMCMC(plated_gmm, [(None, DiscreteGibbs())]).init(
            random.key(0), 10, None, (y,), {}
        )
    with pytest.raises(ValueError, match="enumerable"):
        IncrementalMCMC(
            plated_gmm, [("mu", DiscreteGibbs()), (None, NUTS(plated_gmm))]
        ).init(random.key(0), 10, None, (y,), {})
