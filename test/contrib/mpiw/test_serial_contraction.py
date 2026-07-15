# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
The serial (memory-frugal) contraction path must give the same log P_MP and posterior
moments as the dense path -- it only changes *how* the sum over a chosen K dimension is
evaluated (a lax.scan loop that slices that dimension out), not the result.
"""

import numpy as np
import pytest

import jax
from jax import random
import jax.numpy as jnp

pytest.importorskip("funsor")

import numpyro  # noqa: E402
from numpyro.contrib.mpiw import MPIW, NamedFactor, contract_log_marginal  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

jax.config.update("jax_enable_x64", True)


def test_serial_matches_dense_on_named_factors():
    # a small hand-built factor graph: global mu couples to plated z
    rng = np.random.default_rng(0)
    K, N = 6, 4
    fmu = jnp.asarray(rng.normal(size=(K,)))
    fz = jnp.asarray(rng.normal(size=(K, K, N)))
    fx = jnp.asarray(rng.normal(size=(K, N)))
    factors = [
        NamedFactor(fmu, ("mu",)),
        NamedFactor(fz, ("mu", "z", "i")),
        NamedFactor(fx, ("z", "i")),
    ]
    eliminate = frozenset({"mu", "z"})
    plates = frozenset({"i"})
    dense = contract_log_marginal(factors, eliminate, plates)
    # only the global dim "mu" may be serialized; "z" is plated (dim "i")
    serial_val = contract_log_marginal(factors, eliminate, plates, frozenset({"mu"}))
    assert float(serial_val) == pytest.approx(float(dense), abs=1e-6)


def test_serial_rejects_non_eliminated_dim():
    f = NamedFactor(jnp.zeros((3,)), ("z",))
    with pytest.raises(ValueError, match="subset of eliminate"):
        contract_log_marginal([f], frozenset({"z"}), frozenset(), frozenset({"w"}))


def test_serial_rejects_plated_dim():
    # "z" occurs only alongside plate "i": serializing it is not supported
    factors = [
        NamedFactor(jnp.zeros((5, 4)), ("z", "i")),
        NamedFactor(jnp.zeros((5, 4)), ("z", "i")),
    ]
    with pytest.raises(ValueError, match="only inside plates"):
        contract_log_marginal(
            factors, frozenset({"z"}), frozenset({"i"}), frozenset({"z"})
        )


# end-to-end: MPIW serial_sites must match the dense driver
TAU_A, TAU_B, SIGMA, X_OBS = 1.0, 0.7, 0.5, 1.3


def _chain_model():
    a = numpyro.sample("a", dist.Normal(0.0, TAU_A))
    b = numpyro.sample("b", dist.Normal(a, TAU_B))
    numpyro.sample("x", dist.Normal(b, SIGMA), obs=jnp.array(X_OBS))


def _chain_guide():
    numpyro.sample("a", dist.Normal(0.3, 1.2))
    numpyro.sample("b", dist.Normal(0.6, 1.0))


def test_mpiw_serial_log_marginal_matches_dense():
    mpiw = MPIW(_chain_model, _chain_guide, num_samples=40)
    key = random.PRNGKey(0)
    dense = float(mpiw.log_marginal(key))
    serial = float(mpiw.log_marginal(key, serial_sites=("a",)))
    assert serial == pytest.approx(dense, abs=1e-6)


def test_mpiw_serial_moments_match_dense():
    mpiw = MPIW(_chain_model, _chain_guide, num_samples=40)
    key = random.PRNGKey(1)
    stats = {"a": lambda v: v, "b": lambda v: v}
    dense = mpiw.moments(key, stats)
    serial = mpiw.moments(key, stats, serial_sites=("a",))
    assert float(serial["a"]) == pytest.approx(float(dense["a"]), abs=1e-6)
    assert float(serial["b"]) == pytest.approx(float(dense["b"]), abs=1e-6)


S_MU, TAU, SIGMA_P = 1.0, 0.6, 0.4
X_PLATE = np.array([0.8, -0.3, 1.5, 0.1, -1.1])
N_PLATE = len(X_PLATE)


def _plate_model():
    mu = numpyro.sample("mu", dist.Normal(0.0, S_MU))
    with numpyro.plate("data", N_PLATE):
        z = numpyro.sample("z", dist.Normal(mu, TAU))
        numpyro.sample("x", dist.Normal(z, SIGMA_P), obs=jnp.asarray(X_PLATE))


def _plate_guide():
    numpyro.sample("mu", dist.Normal(0.2, 1.1))
    with numpyro.plate("data", N_PLATE):
        numpyro.sample("z", dist.Normal(0.0, 0.9))


def test_mpiw_serial_plated_matches_dense():
    # loop the global mu serially (the dimension that couples all plate elements)
    mpiw = MPIW(_plate_model, _plate_guide, num_samples=30)
    key = random.PRNGKey(2)
    dense = float(mpiw.log_marginal(key))
    serial = float(mpiw.log_marginal(key, serial_sites=("mu",)))
    assert serial == pytest.approx(dense, abs=1e-6)
