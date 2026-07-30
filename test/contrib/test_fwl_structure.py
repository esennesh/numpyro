# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Static analysis for Find/Weigh/Learn: site classification, the latent packing,
the junction tree, and option validation.

These exercise the parts of the procedure that run before any numerical work, so
they need neither optimistix nor a full pass of ``find_weigh_learn``.
"""

from numpy.testing import assert_allclose
import pytest

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist

pytest.importorskip("funsor")

from numpyro.contrib.fwl import FWLOptions, analyze  # noqa: E402


def gaussian_chain(y, length=4):
    z = numpyro.sample("z0", dist.Normal(0.0, 1.0))
    for i in range(1, length):
        z = numpyro.sample(f"z{i}", dist.Normal(z, 1.0))
    numpyro.sample("y", dist.Normal(z, 1.0), obs=y)


def gmm(data, weights=(0.4, 0.6), obs_scale=0.5):
    locs = numpyro.sample("locs", dist.Normal(0.0, 3.0).expand([2]).to_event(1))
    with numpyro.plate("N", data.shape[0]):
        k = numpyro.sample("k", dist.Categorical(jnp.asarray(weights)))
        numpyro.sample("obs", dist.Normal(locs[k], obs_scale), obs=data)


def mixed_chain(data):
    """A continuous chain plus a plated discrete site, for factor assignment."""
    a = numpyro.sample("a", dist.Normal(0.0, 1.0))
    b = numpyro.sample("b", dist.Normal(a, 1.0))
    c = numpyro.sample("c", dist.Normal(b, 1.0))
    with numpyro.plate("N", data.shape[0]):
        k = numpyro.sample("k", dist.Categorical(jnp.array([0.5, 0.5])))
        numpyro.sample("obs", dist.Normal(c + 2.0 * k, 0.6), obs=data)


def test_structure_classification():
    structure = analyze(gmm, random.PRNGKey(0), (jnp.zeros(6),))
    assert structure.continuous == ("locs",)
    assert structure.discrete == ("k",)
    assert structure.observed == ("obs",)
    assert structure.has_discrete
    assert not structure.enumeration_only
    assert structure.packing.dim == 2
    assert structure.max_plate_nesting == 1
    assert structure.first_available_dim == -2
    assert "locs" in structure.summary()


def test_latent_packing_roundtrip():
    def model():
        numpyro.sample("a", dist.Normal(0.0, 1.0))
        numpyro.sample("b", dist.Normal(jnp.zeros((2, 3)), 1.0).to_event(2))
        numpyro.sample("c", dist.LogNormal(jnp.zeros(4), 1.0).to_event(1))

    packing = analyze(model, random.PRNGKey(0)).packing
    assert packing.dim == 1 + 6 + 4
    flat = jnp.arange(packing.dim, dtype=jnp.float32)
    unpacked = packing.unpack(flat)
    assert unpacked["a"].shape == ()
    assert unpacked["b"].shape == (2, 3)
    assert unpacked["c"].shape == (4,)
    assert_allclose(packing.pack(unpacked), flat)

    # batch dimensions survive both directions
    batched = jnp.stack([flat, flat + 1.0])
    assert packing.unpack(batched)["b"].shape == (2, 2, 3)
    assert_allclose(packing.pack(packing.unpack(batched)), batched)


def test_countable_support_is_rejected():
    def model():
        rate = numpyro.sample("rate", dist.LogNormal(0.0, 1.0))
        numpyro.sample("n", dist.Poisson(rate))

    with pytest.raises(NotImplementedError, match="Wasserstein"):
        analyze(model, random.PRNGKey(0))


def test_purely_discrete_model_is_rejected_by_default():
    def model():
        numpyro.sample("a", dist.Bernoulli(0.3))

    with pytest.raises(ValueError, match="allow_enumeration_only"):
        analyze(model, random.PRNGKey(0))

    structure = analyze(model, random.PRNGKey(0), allow_enumeration_only=True)
    assert structure.enumeration_only
    assert structure.continuous == ()
    assert "enumeration only" in structure.summary()


def test_subsampled_plate_is_rejected():
    def model(data):
        loc = numpyro.sample("loc", dist.Normal(0.0, 1.0))
        with numpyro.plate("N", data.shape[0], subsample_size=3):
            batch = numpyro.subsample(data, event_dim=0)
            numpyro.sample("obs", dist.Normal(loc, 1.0), obs=batch)

    with pytest.raises(NotImplementedError, match="subsampling"):
        analyze(model, random.PRNGKey(0), (jnp.arange(10.0),))


def test_option_validation():
    with pytest.raises(ValueError, match="num_modes"):
        FWLOptions(num_modes=0)
    with pytest.raises(ValueError, match="num_particles"):
        FWLOptions(num_particles=0)
    with pytest.raises(ValueError, match="damping"):
        FWLOptions(damping=0.0)
    with pytest.raises(ValueError, match="mode_source"):
        FWLOptions(mode_source="gumbel")
    with pytest.raises(ValueError, match="covariance"):
        FWLOptions(covariance="banded")
    with pytest.raises(ValueError, match="elimination"):
        FWLOptions(elimination="sequential")
    with pytest.raises(ValueError, match="max_nesting_depth"):
        FWLOptions(max_nesting_depth=-1)


def test_nested_requires_the_joint_objective():
    with pytest.raises(ValueError, match="does not factorize over cliques"):
        FWLOptions(elimination="nested", continuous_objective="marginal")


def test_clique_tree_structures():
    from numpyro.contrib.fwl.junction import build_clique_tree

    def star(y):
        hub = numpyro.sample("hub", dist.Normal(0.0, 1.0))
        for i in range(4):
            numpyro.sample(f"leaf{i}", dist.Normal(hub, 1.0))
        numpyro.sample("y", dist.Normal(hub, 1.0), obs=y)

    def collider(y):
        a = numpyro.sample("a", dist.Normal(0.0, 1.0))
        b = numpyro.sample("b", dist.Normal(0.0, 1.0))
        c = numpyro.sample("c", dist.Normal(a + b, 1.0))
        numpyro.sample("y", dist.Normal(c, 1.0), obs=y)

    chain_tree = build_clique_tree(analyze(gaussian_chain, random.PRNGKey(0), (1.0,)))
    assert set(chain_tree.cliques) == {
        frozenset({"z0", "z1"}),
        frozenset({"z1", "z2"}),
        frozenset({"z2", "z3"}),
    }
    # rooting at the center keeps a 3-clique path at height 1, not 2
    assert chain_tree.height == 1

    star_tree = build_clique_tree(analyze(star, random.PRNGKey(0), (1.0,)))
    assert len(star_tree.cliques) == 4
    assert all(len(clique) == 2 for clique in star_tree.cliques)
    # every clique shares the hub, so a bushy tree exists and must be found
    assert star_tree.height == 1

    # moralization joins the two parents of a collider into one clique
    collider_tree = build_clique_tree(analyze(collider, random.PRNGKey(0), (1.0,)))
    assert collider_tree.cliques == (frozenset({"a", "b", "c"}),)
    assert collider_tree.height == 0


def test_factor_assignment_is_a_partition():
    from numpyro.contrib.fwl.junction import build_clique_tree, factor_scopes

    structure = analyze(mixed_chain, random.PRNGKey(0), (jnp.zeros(6),))
    tree = build_clique_tree(structure)
    scopes = factor_scopes(structure)
    assigned = [name for names in tree.factors for name in names]
    # every factor lands in exactly one clique, so clique energies sum to the total
    assert sorted(assigned) == sorted(scopes)
    assert len(assigned) == len(set(assigned))
    # and each is assigned to a clique that contains its whole scope
    for clique, names in enumerate(tree.factors):
        for name in names:
            assert scopes[name] <= tree.cliques[clique] or not scopes[name]
