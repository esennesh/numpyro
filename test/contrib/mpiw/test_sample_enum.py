# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for sampled (``num_samples``) parallel enumeration in the funsor ``enum``
messenger -- the sampling primitive underlying massively parallel importance weighting.
"""

import numpy as np
import pytest

from jax import random
import jax.numpy as jnp

pytest.importorskip("funsor")

import numpyro  # noqa: E402
from numpyro.contrib.funsor import (  # noqa: E402
    enum,
    plate_to_enum_plate,
    trace as packed_trace,  # noqa: E402
)
import numpyro.distributions as dist  # noqa: E402
from numpyro.handlers import seed  # noqa: E402


def _trace(model, dim=-2, rng=0):
    # first_available_dim = -(max_plate_nesting + 1); these models have one plate.
    m = seed(model, random.PRNGKey(rng))
    with plate_to_enum_plate(), enum(first_available_dim=dim):
        return packed_trace(m).get_trace()


def test_num_samples_single_plated_site():
    K = 5

    def model():
        with numpyro.plate("data", 4):
            numpyro.sample(
                "z",
                dist.Normal(0.0, 1.0),
                infer={"enumerate": "parallel", "num_samples": K},
            )

    tr = _trace(model, dim=-2)
    z = tr["z"]
    assert z["value"].shape == (K, 4)
    assert z["infer"]["dim_to_name"] == {-2: "z", -1: "data"}
    # distinct samples per plate element (not a broadcast singleton)
    assert not jnp.allclose(z["value"], z["value"][..., :1])
    assert z["fn"].log_prob(z["value"]).shape == (K, 4)


def test_num_samples_parent_coupling():
    """A child of a K-sampled parent gets a factor over both K dimensions."""
    K = 5

    def model():
        mu = numpyro.sample(
            "mu",
            dist.Normal(0.0, 1.0),
            infer={"enumerate": "parallel", "num_samples": K},
        )
        with numpyro.plate("data", 4):
            z = numpyro.sample(
                "z",
                dist.Normal(mu, 1.0),
                infer={"enumerate": "parallel", "num_samples": K},
            )
            numpyro.sample("x", dist.Normal(z, 0.5), obs=jnp.zeros(4))

    tr = _trace(model, dim=-2)
    # mu's K dim sits left of plate territory, leaving -1 free
    assert tr["mu"]["value"].shape == (K, 1)
    assert tr["mu"]["infer"]["dim_to_name"] == {-2: "mu"}
    # z ranges over its own K dim, mu's K dim, and the plate
    assert tr["z"]["value"].shape == (K, K, 4)
    assert tr["z"]["infer"]["dim_to_name"] == {-3: "z", -2: "mu", -1: "data"}
    # observed x's log density broadcasts over both K dims and the plate
    assert tr["x"]["fn"].log_prob(tr["x"]["value"]).shape == (K, K, 4)


def test_num_samples_discrete_site():
    K = 5

    def model():
        with numpyro.plate("data", 3):
            numpyro.sample(
                "b",
                dist.Bernoulli(0.3),
                infer={"enumerate": "parallel", "num_samples": K},
            )

    tr = _trace(model, dim=-2, rng=1)
    b = tr["b"]
    assert b["value"].shape == (K, 3)
    assert b["infer"]["dim_to_name"] == {-2: "b", -1: "data"}
    # values are actual Bernoulli draws in {0, 1}
    assert set(np.unique(np.array(b["value"])).tolist()) <= {0, 1}


def test_num_samples_does_not_disturb_plain_enumeration():
    """Exhaustive enumeration (no num_samples) is unchanged by the new code path."""

    def model():
        p = numpyro.sample("p", dist.Dirichlet(jnp.ones(3)))
        with numpyro.plate("data", 4):
            numpyro.sample("z", dist.Categorical(p), infer={"enumerate": "parallel"})

    tr = _trace(model, dim=-2)
    z = tr["z"]
    # Categorical(3) enumerated: support values 0..2 on a fresh dim, plate broadcast
    assert z["value"].shape == (3, 1)
    assert z["infer"]["dim_to_name"] == {-2: "z", -1: "data"}
