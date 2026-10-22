# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from numpy.testing import assert_allclose
import pytest

import jax
from jax import lax
import jax.numpy as jnp

from numpyro.contrib.incremental import Change
from numpyro.contrib.incremental.delta import gather_rect, scatter_rect


@pytest.mark.parametrize(
    "axes, ks",
    [((0,), (2,)), ((1,), (3,)), ((0, 1), (2, 2)), ((0, 2), (2, 3)), ((2, 0), (3, 2))],
)
def test_gather_scatter_rect_match_numpy(axes, ks):
    rng = np.random.RandomState(0)
    x = rng.randn(4, 5, 6).astype(np.float32)
    idxs = tuple(
        rng.choice(x.shape[a], size=k, replace=False).astype(np.int32)
        for a, k in zip(axes, ks)
    )
    got = gather_rect(x, axes, idxs)
    ix = np.ix_(*[np.arange(n) for n in x.shape])
    ix = list(ix)
    for a, i in zip(axes, idxs):
        ix[a] = i.reshape([-1 if j == a else 1 for j in range(x.ndim)])
    expected = x[tuple(ix)]
    assert_allclose(got, expected)

    value = rng.randn(*got.shape).astype(np.float32)
    written = scatter_rect(x, axes, idxs, value)
    expected_w = x.copy()
    expected_w[tuple(ix)] = value
    assert_allclose(written, expected_w)
    # The written block reads back unchanged.
    assert_allclose(gather_rect(written, axes, idxs), value)


def test_scatter_rect_under_vmap_and_fori_loop():
    x = jnp.zeros((3, 5))
    idx = jnp.array([[1], [2], [0]])
    val = jnp.ones((3, 1, 5))
    out = jax.vmap(lambda i, v: scatter_rect(x, (0,), (i,), v))(idx, val)
    assert out.shape == (3, 3, 5)
    assert_allclose(out[1, 2], 1.0)
    assert_allclose(out.sum(), 15.0)

    def body(i, carry):
        return scatter_rect(carry, (0,), (jnp.array([i]),), jnp.full((1, 5), i + 1.0))

    out = lax.fori_loop(0, 3, body, x)
    assert_allclose(out[:, 0], jnp.array([1.0, 2.0, 3.0]))


def test_change_pytree_roundtrip_and_signature():
    c = Change.sliced({"z": jnp.array([2]), "w": jnp.ones((1, 2))}, {"data": 4})
    c = c | Change.dense(mu=jnp.zeros(3))
    leaves, treedef = jax.tree.flatten(c)
    assert len(leaves) == 4  # idx, z, w, mu
    back = jax.tree.unflatten(treedef, leaves)
    assert back.signature == c.signature
    assert back.sites == ("z", "w", "mu")
    assert "sliced(z, w; data[1])" in repr(back)


def test_change_retraces_only_per_signature():
    traces = []

    @jax.jit
    def f(c):
        traces.append(1)
        return sum(jnp.sum(leaf) for leaf in jax.tree.leaves(c))

    f(Change.sliced({"z": jnp.array([1])}, {"data": 0}))
    f(Change.sliced({"z": jnp.array([5])}, {"data": 3}))  # same signature
    f(Change.dense(z=jnp.array([1, 2])))  # new signature
    assert len(traces) == 2


def test_change_validation():
    with pytest.raises(ValueError, match="1-D integer"):
        Change.sliced({"z": jnp.array([1])}, {"data": jnp.array(0.5)})
