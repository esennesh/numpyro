# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
import pytest

from jax import nn, random
import jax.numpy as jnp
import optax

import numpyro
from numpyro import handlers
from numpyro.contrib.diag_sgd import SmoothedCount
from numpyro.contrib.map_proposal import AutoMAPProposal
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoGuide
from numpyro.optim import Adam, Minimize


def model(observation, outcomes):
    location = numpyro.sample("location", dist.Normal(0, 1))
    probability = numpyro.sample("probability", dist.Beta(2, 2))
    numpyro.deterministic("odds", probability / (1 - probability))
    numpyro.sample("location_obs", dist.Normal(location, 1), obs=observation)
    with numpyro.plate("outcomes", len(outcomes)):
        numpyro.sample("probability_obs", dist.Bernoulli(probability), obs=outcomes)


def test_accepts_optax_optimizers():
    def normal_model(observation):
        location = numpyro.sample("location", dist.Normal(0, 1))
        numpyro.sample("obs", dist.Normal(location, 1), obs=observation)

    guide = AutoMAPProposal(
        normal_model,
        map_max_steps=300,
        map_optimizer=optax.adam(0.05),
        num_dispersion_particles=128,
        proposal_max_steps=500,
        proposal_optimizer=optax.adam(0.01),
    )
    map_estimate = guide.find_map(random.key(10), 2.0)

    assert_allclose(map_estimate["location"], 1.0, atol=1e-4)
    assert_allclose(guide._dispersions["location"], 2**-0.5, rtol=0.2)
    assert guide.dispersion_result.losses.shape == (guide.dispersion_result.num_steps,)
    assert guide.dispersion_result.num_steps <= 500
    assert guide.map_result.losses.shape == (guide.map_result.num_steps,)
    assert guide.map_result.num_steps <= 300


def test_call_exposes_only_latent_sample_sites():
    guide = AutoMAPProposal(model, num_dispersion_particles=8)
    seeded_guide = handlers.seed(guide, random.key(3))
    proposal_trace = handlers.trace(seeded_guide).get_trace(
        2.0, jnp.array([1.0, 1.0, 0.0])
    )

    sample_sites = {
        name for name, site in proposal_trace.items() if site["type"] == "sample"
    }
    assert sample_sites == {"location", "probability"}


def test_defaults_to_iterative_optimizers():
    def empty_model():
        pass

    guide = AutoMAPProposal(empty_model)

    assert isinstance(guide._map_optimizer, Adam)
    assert isinstance(guide._proposal_optimizer, Adam)


@pytest.mark.parametrize(
    "make_distribution, observation, upper_bound",
    [
        (lambda: dist.Bernoulli(jnp.array(0.7)), 1.0, 1.0),
        (lambda: dist.Binomial(3, probs=jnp.array(0.5)), 2.0, 3.0),
        (lambda: dist.Categorical(jnp.array([0.1, 0.2, 0.7])), 2.0, 2.0),
    ],
)
def test_finite_discrete_sites_use_exact_categorical_proposal(
    make_distribution, observation, upper_bound
):
    distribution = make_distribution()

    def discrete_model(observation):
        value = numpyro.sample("value", distribution)
        numpyro.sample("obs", dist.Normal(value * 1.0, 0.5), obs=observation)

    guide = AutoMAPProposal(
        discrete_model,
        discrete_temperature=0.2,
        map_max_steps=50,
        num_dispersion_particles=8,
    )
    samples = guide.sample_posterior(random.key(6), {}, observation, sample_shape=(20,))
    proposal = guide._get_proposal("value", guide._proposal_params["value"])
    categories = jnp.arange(int(upper_bound) + 1)
    expected_logits = distribution.log_prob(categories) + dist.Normal(
        categories, 0.5
    ).log_prob(observation)

    assert isinstance(proposal, dist.CategoricalLogits)
    assert_allclose(proposal.probs, nn.softmax(expected_logits), rtol=1e-5)
    assert jnp.all((0 <= samples["value"]) & (samples["value"] <= upper_bound))
    assert jnp.all(samples["value"] == jnp.round(samples["value"]))
    assert guide.dispersion_result is None
    assert "value" not in guide._dispersions
    assert set(guide._proposal_params["value"]) == {"logits"}
    assert "value" in guide._smooth_transforms


def test_finite_discrete_uniform_proposal_preserves_offset_support():
    def discrete_uniform_model(observation):
        value = numpyro.sample("value", dist.DiscreteUniform(2, 4))
        numpyro.sample("obs", dist.Normal(value * 1.0, 0.5), obs=observation)

    guide = AutoMAPProposal(
        discrete_uniform_model,
        map_max_steps=20,
        num_dispersion_particles=4,
    )
    samples = guide.sample_posterior(random.key(9), {}, 4.0, sample_shape=(20,))
    proposal = guide._get_proposal("value", guide._proposal_params["value"])

    assert proposal.is_discrete
    assert jnp.all((2 <= samples["value"]) & (samples["value"] <= 4))
    assert jnp.all(samples["value"] == jnp.round(samples["value"]))
    assert jnp.all(jnp.isfinite(proposal.log_prob(samples["value"])))


def test_inner_optimizer_refreshes_randomness():
    def empty_model():
        pass

    guide = AutoMAPProposal(empty_model, proposal_max_steps=4)
    rng_key = random.key(11)
    result = guide._optimize(
        jnp.array(0.0),
        4,
        lambda _, key: random.uniform(key),
        guide._proposal_optimizer,
        rng_key,
        guide._proposal_tolerance,
    )
    expected_losses = jnp.stack(
        [random.uniform(key) for key in random.split(rng_key, 4)]
    )

    assert_allclose(result.losses, expected_losses)


def test_inner_optimizer_stops_at_convergence():
    def empty_model():
        pass

    guide = AutoMAPProposal(
        empty_model,
        proposal_max_steps=10,
        termination_check_interval=1,
        termination_patience=2,
    )
    result = guide._optimize(
        jnp.array(0.0),
        10,
        lambda _, __: jnp.array(0.0),
        guide._proposal_optimizer,
        random.key(12),
        guide._proposal_tolerance,
    )

    assert result.converged
    assert result.num_steps == 3


def test_map_depends_on_model_arguments():
    guide = AutoMAPProposal(model, num_dispersion_particles=8)
    first_map = guide.find_map(random.key(0), 2.0, jnp.array([1.0, 1.0, 0.0]))
    second_map = guide.find_map(
        random.key(1), -2.0, outcomes=jnp.array([0.0, 0.0, 0.0])
    )

    assert isinstance(guide, AutoGuide)
    assert_allclose(first_map["location"], 1.0, atol=1e-4)
    assert_allclose(first_map["probability"], 0.6, atol=1e-4)
    assert_allclose(second_map["location"], -1.0, atol=1e-4)
    assert_allclose(second_map["probability"], 0.2, atol=1e-4)


def test_optimizes_dispersion_internally():
    def normal_model(observation):
        location = numpyro.sample("location", dist.Normal(0, 1))
        numpyro.sample("obs", dist.Normal(location, 1), obs=observation)

    guide = AutoMAPProposal(
        normal_model, init_dispersion=0.05, num_dispersion_particles=128
    )
    guide.find_map(random.key(5), 2.0)

    assert_allclose(guide._dispersions["location"], 2**-0.5, rtol=0.2)


def test_proposal_supports_and_sample_shapes():
    guide = AutoMAPProposal(model, num_dispersion_particles=8)
    samples = guide.sample_posterior(
        random.key(2),
        {},
        2.0,
        jnp.array([1.0, 1.0, 0.0]),
        sample_shape=(10,),
    )

    assert samples["location"].shape == (10,)
    assert samples["odds"].shape == (10,)
    assert samples["probability"].shape == (10,)
    assert jnp.all((0 < samples["probability"]) & (samples["probability"] < 1))
    assert len(guide._dispersions) == 2
    assert all(
        jnp.isfinite(scale) & (scale > 0) for scale in guide._dispersions.values()
    )


@pytest.mark.parametrize("argument", ["map_optimizer", "proposal_optimizer"])
def test_rejects_noniterative_optimizers(argument):
    def empty_model():
        pass

    with pytest.raises(TypeError, match="requires an iterative optimizer"):
        AutoMAPProposal(empty_model, **{argument: Minimize()})


def test_sites_with_equal_support_have_distinct_dispersions():
    def two_real_sites_model():
        numpyro.sample("a", dist.Normal(0, 0.1))
        numpyro.sample("b", dist.Normal(0, 10))

    guide = AutoMAPProposal(two_real_sites_model, num_dispersion_particles=128)
    guide.find_map(random.key(4))

    assert guide._support_ids == {"a": 0, "b": 0}
    assert set(guide._dispersions) == {"a", "b"}
    assert guide._dispersions["a"] < guide._dispersions["b"]


def test_unbounded_discrete_site_uses_gamma_count_proposal():
    def poisson_model(observation):
        value = numpyro.sample("value", dist.Poisson(jnp.array(3.0)))
        numpyro.sample("obs", dist.Normal(value * 1.0, 1.0), obs=observation)

    guide = AutoMAPProposal(
        poisson_model,
        dsgd_kwargs={"max_count": 100, "width": 16},
        map_max_steps=5,
        num_dispersion_particles=2,
        proposal_max_steps=5,
    )
    samples = guide.sample_posterior(random.key(7), {}, 4.0, sample_shape=(10,))
    proposal = guide._get_proposal("value", guide._proposal_params["value"])
    relaxed_proposal = guide._get_proposal(
        "value", guide._proposal_params["value"], relaxed=True
    )

    assert guide.model is poisson_model
    prototype_trace = guide.prototype_trace
    relaxed_trace = handlers.trace(
        handlers.seed(guide.relaxed_model, random.key(8))
    ).get_trace(4.0)
    assert prototype_trace is not None
    assert isinstance(proposal, dist.GammaCount)
    assert isinstance(relaxed_proposal, SmoothedCount)
    assert_allclose(guide._dispersions["value"], proposal.concentration)
    assert isinstance(relaxed_trace["value"]["fn"], SmoothedCount)
    assert isinstance(prototype_trace["value"]["fn"], SmoothedCount)
    assert jnp.all(samples["value"] >= 0)
    assert jnp.all(samples["value"] == jnp.round(samples["value"]))
    assert set(guide._proposal_params["value"]) == {
        "log_concentration",
        "log_rate",
    }
    assert jnp.isfinite(proposal.concentration) & (proposal.concentration > 0)
    assert jnp.isfinite(proposal.rate) & (proposal.rate > 0)
