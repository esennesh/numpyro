# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""A zoo of small models exercising the structures the incremental layer must handle."""

from collections import namedtuple

import numpy as np

import jax.numpy as jnp

import numpyro
from numpyro.contrib.control_flow import scan
import numpyro.distributions as dist
from numpyro.handlers import scale as scale_handler

Entry = namedtuple("Entry", ["model", "args", "kwargs"])


def _rng():
    return np.random.RandomState(0)


def gmm(y):
    K = 3
    w = numpyro.sample("w", dist.Dirichlet(jnp.ones(K)))
    s = numpyro.sample("s", dist.LogNormal(0.0, 1.0))
    with numpyro.plate("comp", K):
        mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
    with numpyro.plate("data", y.shape[0]):
        z = numpyro.sample("z", dist.Categorical(w))
        numpyro.sample("obs", dist.Normal(mu[z], s), obs=y)


def nested(y):
    M, N = y.shape
    with numpyro.plate("M", M, dim=-2):
        a = numpyro.sample("a", dist.Normal(0.0, 1.0))
        with numpyro.plate("N", N, dim=-1):
            b = numpyro.sample("b", dist.Normal(a, 1.0))
            numpyro.sample("obs", dist.Normal(b, 1.0), obs=y)


def reduction(y, tot):
    with numpyro.plate("N", y.shape[0]):
        x = numpyro.sample("x", dist.Normal(0.0, 1.0))
        numpyro.sample("obs", dist.Normal(x, 1.0), obs=y)
    numpyro.sample("tot", dist.Normal(x.sum(), 1.0), obs=tot)


def reparam(y):
    with numpyro.plate("N", y.shape[0]):
        s = numpyro.sample("s", dist.LogNormal(0.0, 1.0))
        numpyro.sample("obs", dist.Normal(0.0, s), obs=y)


def discrete(counts):
    rate = numpyro.sample("rate", dist.Gamma(2.0, 1.0))
    p = numpyro.sample("p", dist.Beta(2.0, 2.0))
    with numpyro.plate("N", counts.shape[0]):
        k = numpyro.sample("k", dist.Binomial(5, p))
        b = numpyro.sample("b", dist.Bernoulli(p))
        numpyro.sample("obs", dist.Poisson(rate * (1.0 + k) + b), obs=counts)


def determ(y):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    with numpyro.plate("N", y.shape[0]):
        z = numpyro.sample("z", dist.Bernoulli(0.3))
        loc = numpyro.deterministic("loc", mu * z)
        numpyro.sample("obs", dist.Normal(loc, 1.0), obs=y)


def prebatched(y, offset):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    with numpyro.plate("N", y.shape[0]):
        z = numpyro.sample("z", dist.Normal(0.0, 1.0))
        numpyro.sample("obs", dist.Normal(mu + z + offset, 1.0), obs=y)


def subsampled(y):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    with numpyro.plate("N", y.shape[0]):
        yb = numpyro.subsample(y, event_dim=0)
        z = numpyro.sample("z", dist.Normal(mu, 1.0))
        numpyro.sample("obs", dist.Normal(z, 1.0), obs=yb)


def hmm(y):
    x0 = numpyro.sample("x0", dist.Normal(0.0, 1.0))

    def step(x, yt):
        x_new = numpyro.sample("x", dist.Normal(x, 1.0))
        numpyro.sample("y", dist.Normal(x_new, 1.0), obs=yt)
        return x_new, None

    scan(step, x0, y)


def scaled(y):
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    with numpyro.plate("N", y.shape[0]):
        with scale_handler(scale=0.5):
            numpyro.sample("obs", dist.Normal(mu, 1.0), obs=y)
        numpyro.factor("pen", -0.1 * jnp.square(mu) * jnp.ones_like(y))


def static_arg(y, num_groups):
    with numpyro.plate("G", num_groups):
        mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
    with numpyro.plate("N", y.shape[0]):
        g = numpyro.sample("g", dist.Categorical(jnp.ones(num_groups) / num_groups))
        numpyro.sample("obs", dist.Normal(mu[g], 1.0), obs=y)


def mvn_mixture(points):
    """Batched cholesky / triangular_solve / diagonal gather on the sliced path."""
    with numpyro.plate("component", 2):
        cov = numpyro.sample("cov", dist.InverseWishart(3.2, 1.6 * jnp.eye(2)))
        mu = numpyro.sample(
            "mu",
            dist.MultivariateNormal(jnp.zeros(2), covariance_matrix=cov / 0.4),
        )
    with numpyro.plate("data", points.shape[0]):
        z = numpyro.sample("z", dist.Bernoulli(0.4))
        numpyro.sample(
            "obs", dist.MultivariateNormal(mu[z], covariance_matrix=cov[z]), obs=points
        )


def matrix_product(y, w):
    """dot_general with a sliced free dim (rows of a latent matrix)."""
    with numpyro.plate("data", y.shape[0]):
        h = numpyro.sample("h", dist.Normal(0.0, 1.0).expand([3]).to_event(1))
        numpyro.sample("obs", dist.Normal(h @ w, 1.0), obs=y)


def build_zoo(n=12):
    # NumPy data only: the repository's conftest forbids live JAX arrays at import.
    rng = _rng()
    y = rng.randn(n).astype(np.float32)
    y2 = rng.randn(3, n).astype(np.float32)
    counts = rng.poisson(2.0, size=n).astype(np.int32)
    offset = rng.randn(n).astype(np.float32)
    return {
        "gmm": Entry(gmm, (y,), {}),
        "nested": Entry(nested, (y2,), {}),
        "reduction": Entry(reduction, (y, np.float32(0.5)), {}),
        "reparam": Entry(reparam, (np.abs(y),), {}),
        "discrete": Entry(discrete, (counts,), {}),
        "determ": Entry(determ, (y,), {}),
        "prebatched": Entry(prebatched, (y, offset), {}),
        "subsampled": Entry(subsampled, (y,), {}),
        "hmm": Entry(hmm, (y,), {}),
        "scaled": Entry(scaled, (y,), {}),
        "static_arg": Entry(static_arg, (y,), {"num_groups": 4}),
        "mvn_mixture": Entry(mvn_mixture, (rng.randn(n, 2).astype(np.float32),), {}),
        "matrix_product": Entry(
            matrix_product, (y, rng.randn(3).astype(np.float32)), {}
        ),
    }


ZOO = build_zoo()
