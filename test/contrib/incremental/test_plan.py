# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""cache='plan': slice pullback and a minimal cache."""

import numpy as np
from numpy.testing import assert_allclose
import pytest
from zoo import ZOO, gmm

import jax
import jax.numpy as jnp

from numpyro.contrib.incremental import Change, IncrementalDensity, check_consistency

TOL = dict(rtol=1e-4, atol=1e-4)


def test_plan_mode_matches_all_mode_on_zoo(zoo_entry):
    name, entry = zoo_entry
    inc = IncrementalDensity(entry.model, entry.args, entry.kwargs, cache="plan")
    inc.plan_site_changes(ks=(1, 2, 3))
    check_consistency(inc, entry.args, entry.kwargs, num_changes=15, seed=3)


def test_plan_mode_caches_little_for_gmm():
    n = 200
    y = np.random.RandomState(0).randn(n).astype(np.float32)
    full = IncrementalDensity(gmm, (y,), cache="all")
    lean = IncrementalDensity(gmm, (y,), cache="plan")
    change = Change.sliced({"z": jnp.array([1])}, {"data": jnp.array([3])})
    lean.plan(change)
    s_full = full.init(model_args=(y,))
    s_lean = lean.init(model_args=(y,))
    # Only factor outputs are cut points for this change: no N x K intermediate.
    assert all(v.ndim < 2 for v in s_lean.cache.values())
    assert lean.cache_size(s_lean) < full.cache_size(s_full) / 2
    # Same deltas and consistent state.
    d_full, _ = full.apply(s_full, change, (y,))
    d_lean, s_lean2 = lean.apply(s_lean, change, (y,))
    assert_allclose(d_full, d_lean, **TOL)
    fresh = lean.program.eval(s_lean2.z, (y,))
    for k, v in lean.factor_values(s_lean2).items():
        assert_allclose(v, fresh[k], err_msg=k, **TOL)


def test_unplanned_signature_raises_and_plan_fixes_it():
    entry = ZOO["gmm"]
    inc = IncrementalDensity(entry.model, entry.args, entry.kwargs, cache="plan")
    inc.plan(Change.sliced({"z": jnp.array([1])}, {"data": 0}))
    state = inc.init(model_args=entry.args)
    dense_mu = Change.dense(mu=jnp.zeros(3))
    with pytest.raises(ValueError, match="not cached"):
        inc.apply(state, dense_mu, entry.args)
    inc.plan(dense_mu)
    state = inc.resync(state, entry.args)
    _, new_state = inc.apply(state, dense_mu, entry.args)
    fresh = inc.program.eval(new_state.z, entry.args)
    assert_allclose(inc.factor_values(new_state)["obs"], fresh["obs"], **TOL)


def test_plan_mode_gradients_and_vmap():
    entry = ZOO["gmm"]
    inc = IncrementalDensity(entry.model, entry.args, entry.kwargs, cache="plan")
    inc.plan_site_changes()
    state = inc.init(model_args=entry.args)
    mu0 = state.z["mu"]

    def incremental(mu):
        return inc.apply_delta(state, Change.dense(mu=mu), entry.args)[0]

    def reference(mu):
        return inc.program.log_density({**state.z, "mu": mu}, entry.args)

    assert_allclose(
        jax.grad(incremental)(mu0 + 0.1), jax.grad(reference)(mu0 + 0.1), **TOL
    )
    i = jnp.array([2])

    def delta_of(zi):
        return inc.apply_delta(
            state, Change.sliced({"z": zi}, {"data": i}), entry.args
        )[0]

    deltas = jax.jit(jax.vmap(delta_of))(jnp.arange(3)[:, None])
    expected = jnp.stack([delta_of(jnp.array([k])) for k in range(3)])
    assert_allclose(deltas, expected, **TOL)
