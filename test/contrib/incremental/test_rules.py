# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.testing import assert_allclose
import pytest
from zoo import ZOO, reduction

import jax
from jax.extend.core import primitives as P
import jax.numpy as jnp

from numpyro.contrib.incremental import Change, IncrementalDensity, check_consistency

TOL = dict(rtol=1e-4, atol=1e-4)


def test_check_consistency_on_zoo(zoo_entry):
    name, entry = zoo_entry
    inc = IncrementalDensity(entry.model, entry.args, entry.kwargs)
    check_consistency(inc, entry.args, entry.kwargs, num_changes=15, seed=2)


@pytest.mark.parametrize(
    "model_name, site",
    [("gmm", "mu"), ("gmm", "w"), ("discrete", "p"), ("nested", "a")],
)
def test_dense_path_gradient_matches_log_density(model_name, site):
    entry = ZOO[model_name]
    inc = IncrementalDensity(entry.model, entry.args, entry.kwargs)
    state = inc.init(model_args=entry.args, model_kwargs=entry.kwargs)
    x0 = state.z[site]

    def incremental(x):
        return inc.apply_delta(
            state, Change.dense({site: x}), entry.args, entry.kwargs
        )[0]

    def reference(x):
        return inc.program.log_density({**state.z, site: x}, entry.args, entry.kwargs)

    x = x0 + 0.3
    # The reference subtracts two O(100) float32 totals and loses ~1e-3; the
    # incremental delta is formed from per-factor differences and is more accurate.
    assert_allclose(incremental(x), reference(x) - reference(x0), rtol=1e-4, atol=3e-3)
    assert_allclose(jax.grad(incremental)(x), jax.grad(reference)(x), **TOL)
    # and under jit
    assert_allclose(jax.jit(jax.grad(incremental))(x), jax.grad(reference)(x), **TOL)


def test_reduce_sum_delta_rule_is_local():
    n = 50
    y = np.random.RandomState(0).randn(n).astype(np.float32)
    args = (y, np.float32(0.5))
    inc = IncrementalDensity(reduction, args)
    state = inc.init(model_args=args)
    change = Change.sliced({"x": jnp.array([0.7])}, {"N": 7})
    tags = inc.tags(change)
    assert tags == {"x": "Sliced", "obs": "Sliced", "tot": "Dense"}
    delta, new_state = inc.apply(state, change, args)
    fresh = inc.program.eval(new_state.z, args)
    assert_allclose(inc.factor_values(new_state)["tot"], fresh["tot"], **TOL)
    # Only a handful of equations run: not the N-sized sum.
    assert inc.num_reexecuted(change) < 20


def test_reduce_sum_guard_recovers_from_poisoned_cache():
    n = 20
    y = np.random.RandomState(0).randn(n).astype(np.float32)
    args = (y, np.float32(0.5))
    inc = IncrementalDensity(reduction, args)
    state = inc.init(model_args=args)
    # Poison the cached value of the reduce_sum over the plate axis.
    eqns = [
        e
        for e in inc.program.jaxpr.jaxpr.eqns
        if e.primitive is P.reduce_sum_p and e.invars[0].aval.shape == (n,)
    ]
    assert eqns, "expected a reduce_sum over the plate axis"
    vid = inc.var_ids[eqns[0].outvars[0]]
    cache = dict(state.cache)
    cache[vid] = jnp.full_like(cache[vid], jnp.nan)
    poisoned = state._replace(cache=cache)
    change = Change.sliced({"x": jnp.array([0.7])}, {"N": 3})
    _, new_state = inc.apply(poisoned, change, args)
    fresh = inc.program.eval(new_state.z, args)
    assert_allclose(new_state.cache[vid], jnp.sum(new_state.z["x"]), **TOL)
    assert_allclose(inc.factor_values(new_state)["tot"], fresh["tot"], **TOL)


def test_custom_jvp_sliced_path_values_and_registry():
    entry = ZOO["discrete"]
    inc = IncrementalDensity(entry.model, entry.args, entry.kwargs)
    state = inc.init(model_args=entry.args)
    change = Change.sliced({"k": jnp.array([3]), "b": jnp.array([1])}, {"N": 5})
    delta, new_state = inc.apply(state, change, entry.args)
    fresh = inc.program.eval(new_state.z, entry.args)
    for name in ("k", "b", "obs"):
        assert_allclose(inc.factor_values(new_state)[name], fresh[name], **TOL)
    assert inc.tags(change)["k"] == "Sliced"
    assert inc.tags(change)["b"] == "Sliced"
