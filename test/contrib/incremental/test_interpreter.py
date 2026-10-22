# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.testing import assert_allclose
import pytest
from zoo import ZOO

import jax
import jax.numpy as jnp

from numpyro.contrib.incremental import Change, IncrementalDensity

TOL = dict(rtol=1e-4, atol=1e-4)


def _random_changes(inc, rng, n):
    """A sequence of random single-element sliced and dense changes."""
    changes = []
    sites = list(inc.sites)
    for _ in range(n):
        name = sites[rng.randint(len(sites))]
        meta = inc.sites[name]
        k = meta.support_size or 5

        def draw(shape, discrete=meta.is_discrete, k=k):
            if discrete:
                return np.asarray(rng.randint(k, size=shape), dtype=np.int32)
            return np.asarray(rng.standard_normal(shape), dtype=np.float32)

        if meta.plates and rng.rand() < 0.7:
            # slice along a random non-empty subset of plates
            plates = [p for p, _ in meta.plates if rng.rand() < 0.7] or [
                meta.plates[0][0]
            ]
            axes = dict(meta.plates)
            index = {}
            shape = list(meta.shape)
            for p in plates:
                size = meta.shape[axes[p]]
                kk = rng.randint(1, min(size, 3) + 1)
                index[p] = rng.choice(size, size=kk, replace=False).astype(np.int32)
                shape[axes[p]] = kk
            changes.append(Change.sliced({name: draw(tuple(shape))}, index))
        else:
            changes.append(Change.dense({name: draw(meta.shape)}))
    return changes


def _check_state(inc, state, entry):
    fresh = inc.program.eval(state.z, entry.args, entry.kwargs)
    cached = inc.factor_values(state)
    for name in inc.program.factor_names:
        assert_allclose(cached[name], fresh[name], err_msg=name, **TOL)
    expected = sum(jnp.sum(v) for v in fresh.values())
    assert_allclose(state.total, expected, **TOL)


def test_random_change_sequences(zoo_entry):
    name, entry = zoo_entry
    inc = IncrementalDensity(entry.model, entry.args, entry.kwargs)
    state = inc.init(model_args=entry.args, model_kwargs=entry.kwargs)
    _check_state(inc, state, entry)
    rng = np.random.RandomState(1)
    for i, change in enumerate(_random_changes(inc, rng, 12)):
        old_total = state.total
        delta, new_state = inc.apply(state, change, entry.args, entry.kwargs)
        _check_state(inc, new_state, entry)
        assert_allclose(new_state.total - old_total, delta, **TOL)
        # apply_delta agrees with apply
        delta2, _ = inc.apply_delta(state, change, entry.args, entry.kwargs)
        assert_allclose(delta2, delta, **TOL)
        # Reject every third change: keep the old state.
        state = state if i % 3 == 2 else new_state


def test_apply_is_jittable_and_vmappable():
    entry = ZOO["gmm"]
    inc = IncrementalDensity(entry.model, entry.args, entry.kwargs)
    state = inc.init(model_args=entry.args)
    i = jnp.array([4])

    def delta_of(zi):
        return inc.apply_delta(
            state, Change.sliced({"z": zi}, {"data": i}), entry.args
        )[0]

    deltas = jax.jit(jax.vmap(delta_of))(jnp.arange(3)[:, None])
    expected = jnp.stack([delta_of(jnp.array([k])) for k in range(3)])
    assert_allclose(deltas, expected, **TOL)

    # jit over the whole apply with the change as a pytree argument
    f = jax.jit(lambda s, c: inc.apply(s, c, entry.args))
    delta, new_state = f(state, Change.sliced({"z": jnp.array([1])}, {"data": i}))
    _check_state(inc, new_state, entry)


EXPECTED_TAGS = {
    ("gmm", "z"): {
        "z": "Sliced",
        "obs": "Sliced",
        "mu": "Same",
        "w": "Same",
        "s": "Same",
        "_s_log_det": "Same",
        "_w_log_det": "Same",
    },
    # mu is plated, so the test slices one component: its own factor is sliced while
    # obs (gathered through z) must be recomputed densely.
    ("gmm", "mu"): {"mu": "Sliced", "obs": "Dense", "z": "Same", "w": "Same"},
    ("gmm", "w"): {"w": "Dense", "_w_log_det": "Dense", "z": "Dense", "obs": "Same"},
    ("nested", "b"): {"b": "Sliced", "obs": "Sliced", "a": "Same"},
    ("reparam", "s"): {"s": "Sliced", "_s_log_det": "Sliced", "obs": "Sliced"},
    # Bernoulli.log_prob goes through custom_jvp (xlogy) on the sliced path.
    ("determ", "z"): {"obs": "Sliced", "z": "Sliced", "mu": "Same"},
    ("discrete", "k"): {"k": "Sliced", "obs": "Sliced", "b": "Same", "p": "Same"},
    ("discrete", "b"): {"b": "Sliced", "obs": "Sliced", "k": "Same"},
    # x.sum() feeds tot: the reduce_sum delta rule keeps it O(1) but Dense.
    ("reduction", "x"): {"x": "Sliced", "obs": "Sliced", "tot": "Dense"},
    ("prebatched", "z"): {"obs": "Sliced", "z": "Sliced", "mu": "Same"},
    ("subsampled", "z"): {"obs": "Sliced", "z": "Sliced", "mu": "Same"},
    ("scaled", "mu"): {"obs": "Dense", "pen": "Dense", "mu": "Dense"},
    ("static_arg", "g"): {"g": "Sliced", "obs": "Sliced", "mu": "Same"},
}


@pytest.mark.parametrize("model_name, site", sorted(EXPECTED_TAGS))
def test_expected_tags(model_name, site):
    entry = ZOO[model_name]
    inc = IncrementalDensity(entry.model, entry.args, entry.kwargs)
    meta = inc.sites[site]
    if meta.plates:
        index = {p: 0 for p, _ in meta.plates}
        shape = list(meta.shape)
        for _, a in meta.plates:
            shape[a] = 1
        value = jnp.zeros(shape, dtype=meta.dtype)
        change = Change.sliced({site: value}, index)
    else:
        change = Change.dense({site: jnp.zeros(meta.shape, dtype=meta.dtype)})
    tags = inc.tags(change)
    for factor, kind in EXPECTED_TAGS[(model_name, site)].items():
        assert tags[factor] == kind, (factor, tags)


def test_hmm_falls_back_to_dense():
    entry = ZOO["hmm"]
    inc = IncrementalDensity(entry.model, entry.args, entry.kwargs)
    change = Change.sliced({"x": jnp.zeros(1)}, {0: 3})
    tags = inc.tags(change)
    assert tags["x"] == "Dense" and tags["y"] == "Dense"
    state = inc.init(model_args=entry.args)
    _, new_state = inc.apply(state, change, entry.args)
    _check_state(inc, new_state, entry)
