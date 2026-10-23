# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""The emitted program for a single-element change must not scale with the data."""

import numpy as np
import pytest
from zoo import gmm

import jax
import jax.numpy as jnp

from numpyro.contrib.incremental import Change, IncrementalDensity


def _flops(n):
    y = np.random.RandomState(0).randn(n).astype(np.float32)
    inc = IncrementalDensity(gmm, (y,))
    state = inc.init(model_args=(y,))
    change = Change.sliced({"z": jnp.array([1])}, {"data": jnp.array([n // 2])})
    f = jax.jit(lambda s, c, y: inc.apply_delta(s, c, (y,))[0])
    compiled = f.lower(state, change, y).compile()
    cost = compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    return float(cost["flops"]), inc.num_reexecuted(change)


def _sweep_bytes_per_element(n):
    """Bytes accessed per element by a compiled Gibbs-style sweep over all z."""
    y = np.random.RandomState(0).randn(n).astype(np.float32)
    inc = IncrementalDensity(gmm, (y,), cache="plan")
    probe = Change.sliced({"z": jnp.zeros(1, jnp.int32)}, {"data": 0})
    inc.plan(probe)
    assert "z" not in inc.input_full_reads(probe)
    state = inc.init(model_args=(y,))

    def sweep(state, y):
        snapshot = state.z

        def body(i, s):
            new = ((snapshot["z"][i] + 1) % 3)[None]
            change = Change.sliced({"z": new}, {"data": i})
            return inc.apply(s, change, (y,), read_values=snapshot)[1]

        out = jax.lax.fori_loop(0, n, body, state)
        return out.total + out.z["z"].sum()

    cost = jax.jit(sweep).lower(state, y).compile().cost_analysis()
    cost = cost[0] if isinstance(cost, list) else cost
    return float(cost["bytes accessed"]) / n


def test_sweep_with_snapshot_reads_is_linear():
    # Per-element memory traffic of a whole sweep must not grow with the data size:
    # XLA keeps the carried arrays in place when the loop body only writes them.
    small, large = _sweep_bytes_per_element(200), _sweep_bytes_per_element(2000)
    assert large < 2 * small, (small, large)


@pytest.mark.parametrize("sizes", [(100, 1000, 10000)])
def test_single_element_update_cost_is_constant(sizes):
    flops, eqns = zip(*[_flops(n) for n in sizes])
    assert len(set(eqns)) == 1, eqns
    # Compiled flop counts can differ by small constant amounts across shapes
    # (index arithmetic), but must not grow with the data size.
    assert max(flops) - min(flops) < 64, flops
