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


@pytest.mark.parametrize("sizes", [(100, 1000, 10000)])
def test_single_element_update_cost_is_constant(sizes):
    flops, eqns = zip(*[_flops(n) for n in sizes])
    assert len(set(eqns)) == 1, eqns
    # Compiled flop counts can differ by small constant amounts across shapes
    # (index arithmetic), but must not grow with the data size.
    assert max(flops) - min(flops) < 64, flops
