# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import itertools

import numpy as np
from numpy.testing import assert_allclose
import pytest

import jax
from jax import random
import jax.numpy as jnp

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer.initialization import init_to_sample, init_to_value
from numpyro.infer.util import potential_energy

pytest.importorskip("optimistix")
pytest.importorskip("funsor")

from numpyro.contrib.fwl import FWLOptions, find_weigh_learn  # noqa: E402


def conjugate_model(y, obs_scale=0.5):
    """z ~ N(0, 1), y | z ~ N(z, obs_scale). Posterior and evidence in closed form."""
    z = numpyro.sample("z", dist.Normal(0.0, 1.0))
    numpyro.sample("y", dist.Normal(z, obs_scale), obs=y)


def conjugate_posterior(y, obs_scale=0.5):
    """(mean, std, log_evidence) of ``conjugate_model``."""
    precision = 1.0 + 1.0 / obs_scale**2
    var = 1.0 / precision
    mean = (y / obs_scale**2) * var
    log_evidence = dist.Normal(0.0, np.sqrt(1.0 + obs_scale**2)).log_prob(y)
    return mean, np.sqrt(var), log_evidence


def lognormal_model(y, obs_scale=0.5):
    """
    Same posterior as ``conjugate_model`` but in the *unconstrained* coordinate:
    ``u = log z`` is exactly Gaussian a posteriori, so this exercises the support
    transform's log-Jacobian in the importance weights.
    """
    z = numpyro.sample("z", dist.LogNormal(0.0, 1.0))
    numpyro.sample("y", dist.Normal(jnp.log(z), obs_scale), obs=y)


def gmm(data, weights=(0.4, 0.6), obs_scale=0.5):
    locs = numpyro.sample("locs", dist.Normal(0.0, 3.0).expand([2]).to_event(1))
    with numpyro.plate("N", data.shape[0]):
        k = numpyro.sample("k", dist.Categorical(jnp.asarray(weights)))
        numpyro.sample("obs", dist.Normal(locs[k], obs_scale), obs=data)


def chain_model(data, transition_stay=0.8, obs_scale=0.7):
    """A short discrete chain: max-product must respect the coupling between steps."""
    loc = numpyro.sample("loc", dist.Normal(0.0, 2.0))
    transition = jnp.array(
        [[transition_stay, 1 - transition_stay], [1 - transition_stay, transition_stay]]
    )
    state = 0
    for t in range(data.shape[0]):
        probs = transition[state] if t > 0 else jnp.array([0.5, 0.5])
        state = numpyro.sample(f"s_{t}", dist.Categorical(probs))
        numpyro.sample(
            f"obs_{t}",
            dist.Normal(jnp.where(state == 0, -loc, loc), obs_scale),
            obs=data[t],
        )


# NumPy rather than JAX arrays: the test suite asserts no live device arrays
# exist before the first test runs.
GMM_DATA = np.array([-2.1, -1.9, -2.3, 2.0, 2.2, 1.8], dtype=np.float32)
CHAIN_DATA = np.array([1.4, 1.6, -1.5, -1.3, 1.5], dtype=np.float32)


def gaussian_chain(y, length=4):
    """A linear-Gaussian chain: the MAP is z_i = i * y / (length + 1)."""
    z = numpyro.sample("z0", dist.Normal(0.0, 1.0))
    for i in range(1, length):
        z = numpyro.sample(f"z{i}", dist.Normal(z, 1.0))
    numpyro.sample("y", dist.Normal(z, 1.0), obs=y)


def curved_chain(y):
    """A chain with non-quadratic factors, where quadratic messages would be approximate."""
    a = numpyro.sample("a", dist.Normal(0.0, 1.0))
    b = numpyro.sample("b", dist.LogNormal(a, 1.0))
    c = numpyro.sample("c", dist.Normal(jnp.log(b), 0.8))
    numpyro.sample("y", dist.Normal(jnp.tanh(c), 0.5), obs=y)


def mixed_chain(data):
    """
    Continuous chain plus a plated discrete site, so both halves of the
    block-coordinate sweep have work. The observation touches only ``c``: were it
    to touch ``a`` as well, moralization would fuse the whole chain into one
    clique, correctly but uninformatively for this test.
    """
    a = numpyro.sample("a", dist.Normal(0.0, 1.0))
    b = numpyro.sample("b", dist.Normal(a, 1.0))
    c = numpyro.sample("c", dist.Normal(b, 1.0))
    with numpyro.plate("N", data.shape[0]):
        k = numpyro.sample("k", dist.Categorical(jnp.array([0.5, 0.5])))
        numpyro.sample("obs", dist.Normal(c + 2.0 * k, 0.6), obs=data)


@pytest.mark.parametrize("length", [2, 4, 6])
def test_nested_elimination_finds_the_analytic_chain_mode(length):
    y = 2.0
    expected = np.array([i * y / (length + 1) for i in range(1, length + 1)])
    result = find_weigh_learn(
        gaussian_chain,
        random.PRNGKey(0),
        (y, length),
        num_modes=1,
        num_particles=4,
        elimination="nested",
        max_nesting_depth=6,
    )
    found = np.array([float(result.modes[f"z{i}"][0]) for i in range(length)])
    assert_allclose(found, expected, atol=1e-3)


@pytest.mark.parametrize(
    "model_fn,args", [(gaussian_chain, (2.0,)), (curved_chain, (0.4,))]
)
def test_nested_and_joint_elimination_agree(model_fn, args):
    """
    Nested elimination gets its message derivatives from the envelope theorem
    rather than by differentiating the inner solves. That is exact, so it must
    land on the same optimum as a single joint solve -- including for
    ``curved_chain``, whose factors are not quadratic.
    """
    kwargs = dict(num_modes=1, num_particles=4, max_nesting_depth=6)
    joint = find_weigh_learn(
        model_fn, random.PRNGKey(0), args, elimination="joint", **kwargs
    )
    nested = find_weigh_learn(
        model_fn, random.PRNGKey(0), args, elimination="nested", **kwargs
    )
    assert_allclose(nested.find_state.latent, joint.find_state.latent, atol=1e-3)
    assert_allclose(nested.log_joint, joint.log_joint, atol=1e-4)


def test_nested_elimination_distributes_the_perturbation_once():
    """
    Perturb-and-MAP tilts the energy by eps . u. Nested elimination applies the
    tilt clique by clique, on each clique's interior, which is only equivalent to
    the joint tilt because the interiors partition the coordinates. Same tilt and
    same energy must therefore give the same mode.
    """
    kwargs = dict(
        num_modes=2, num_particles=4, mode_source="perturb", max_nesting_depth=6
    )
    joint = find_weigh_learn(
        gaussian_chain, random.PRNGKey(4), (2.0,), elimination="joint", **kwargs
    )
    nested = find_weigh_learn(
        gaussian_chain, random.PRNGKey(4), (2.0,), elimination="nested", **kwargs
    )
    assert_allclose(nested.find_state.latent, joint.find_state.latent, atol=1e-3)
    # and the tilt actually moved the modes away from the unperturbed optimum
    plain = find_weigh_learn(
        gaussian_chain, random.PRNGKey(4), (2.0,), num_modes=2, num_particles=4
    )
    assert not np.allclose(
        np.asarray(nested.find_state.latent),
        np.asarray(plain.find_state.latent),
        atol=1e-3,
    )


def test_nested_elimination_with_discrete_sites():
    result = find_weigh_learn(
        mixed_chain,
        random.PRNGKey(0),
        (GMM_DATA,),
        num_modes=2,
        num_particles=4,
        elimination="nested",
        max_nesting_depth=6,
    )
    assert result.tree.height >= 1  # the continuous chain really is factorized
    assert np.all(np.isfinite(result.log_joint))
    assert set(result.modes) >= {"a", "b", "c", "k"}


def test_nested_depth_guard():
    with pytest.raises(ValueError, match="max_nesting_depth"):
        find_weigh_learn(
            gaussian_chain,
            random.PRNGKey(0),
            (2.0, 12),
            num_modes=1,
            elimination="nested",
            max_nesting_depth=2,
        )


def test_graph_precision_reconstructs_the_damped_fisher():
    """
    The clique blocks are a decomposition, not an approximation: summing
    ``L_c L_c^T`` over cliques must reproduce ``F-hat + lambda I`` exactly, and
    the result must carry the moral graph's zeros.
    """
    from numpyro.contrib.fwl.find import index_map
    from numpyro.contrib.fwl.guide import assemble_precision

    kwargs = dict(num_modes=1, num_particles=4)
    full = find_weigh_learn(
        gaussian_chain, random.PRNGKey(0), (2.0,), covariance="full", **kwargs
    )
    graph = find_weigh_learn(
        gaussian_chain, random.PRNGKey(0), (2.0,), covariance="graph", **kwargs
    )
    cholesky = np.asarray(full.params["_fwl_scale_tril"][0])
    from_full = np.linalg.inv(cholesky @ cholesky.T)

    index = index_map(graph.structure.packing)
    scopes = tuple(
        jnp.asarray(sorted(i for name in sorted(clique) for i in index[name]))
        for clique in graph.tree.cliques
    )
    blocks = tuple(
        graph.params[f"_fwl_clique_chol_{i}"] for i in range(len(graph.tree.cliques))
    )
    from_blocks = np.asarray(
        assemble_precision(blocks, scopes, graph.structure.packing.dim)[0]
    )
    assert_allclose(from_blocks, from_full, atol=1e-5)

    # the chain's moral graph is tridiagonal
    distance = np.abs(np.subtract.outer(np.arange(4), np.arange(4)))
    assert_allclose(from_blocks[distance > 1], 0.0, atol=1e-7)
    assert np.all(np.abs(np.diag(from_blocks, 1)) > 1e-3)


def test_graph_covariance_weights_are_exact_for_the_true_posterior():
    """The graph path's log q must be right, not just finite."""
    y, obs_scale = 1.3, 0.5
    mean, std, log_evidence = conjugate_posterior(y, obs_scale)
    result = find_weigh_learn(
        conjugate_model,
        random.PRNGKey(0),
        (y, obs_scale),
        num_modes=2,
        num_particles=16,
        covariance="graph",
    )
    assert len(result.tree.cliques) == 1
    params = dict(result.params)
    params["_fwl_loc"] = jnp.full_like(params["_fwl_loc"], mean)
    # one coordinate, so the block is the precision's square root
    params["_fwl_clique_chol_0"] = jnp.full_like(
        params["_fwl_clique_chol_0"], 1.0 / std
    )
    weights = result.log_weights(params, random.PRNGKey(1))
    assert_allclose(weights, np.full(weights.shape, log_evidence), rtol=1e-5)


def test_graph_covariance_learns():
    result = find_weigh_learn(
        gaussian_chain,
        random.PRNGKey(0),
        (2.0,),
        num_modes=2,
        num_particles=64,
        covariance="graph",
    )
    key = random.PRNGKey(1)
    grads = jax.grad(lambda p, k: -result.iwae(p, k))(result.params, key)
    assert set(grads) == set(result.params)
    assert all(np.all(np.isfinite(g)) for g in grads.values())

    stepped = jax.tree.map(lambda p, g: p - 0.01 * g, result.params, grads)
    assert result.iwae(stepped, key) > result.iwae(result.params, key)


def test_finds_conjugate_mode():
    y = 1.3
    mean, std, _ = conjugate_posterior(y)
    result = find_weigh_learn(conjugate_model, random.PRNGKey(0), (y,), num_modes=3)
    assert result.modes["z"].shape == (3,)
    assert_allclose(result.modes["z"], np.full(3, mean), atol=1e-4)
    assert np.all(result.converged)
    # a single Gaussian factor plus the prior: the empirical Fisher of the two
    # factors is finite and the damped scale must be positive and finite
    scale = result.params["_fwl_scale"]
    assert np.all(np.isfinite(scale)) and np.all(scale > 0)
    assert result.modes["y"].shape == (3,)  # observed sites are tiled over modes


@pytest.mark.parametrize("model_fn", [conjugate_model, lognormal_model])
def test_weights_are_exact_for_the_true_posterior(model_fn):
    """
    With the proposal set to the exact posterior in unconstrained space, every
    log importance weight must equal log Z(theta) exactly. This pins down the
    Jacobian bookkeeping: ``lognormal_model`` has a non-identity transform.
    """
    y, obs_scale = 1.3, 0.5
    mean, std, log_evidence = conjugate_posterior(y, obs_scale)
    result = find_weigh_learn(
        model_fn,
        random.PRNGKey(0),
        (y, obs_scale),
        num_modes=2,
        num_particles=32,
    )
    params = dict(result.params)
    params["_fwl_loc"] = jnp.full_like(params["_fwl_loc"], mean)
    params["_fwl_scale"] = jnp.full_like(params["_fwl_scale"], std)

    weights = result.log_weights(params, random.PRNGKey(1))
    assert_allclose(weights, np.full(weights.shape, log_evidence), rtol=1e-5)
    assert_allclose(result.elbo(params, random.PRNGKey(1)), log_evidence, rtol=1e-5)
    assert_allclose(result.iwae(params, random.PRNGKey(1)), log_evidence, rtol=1e-5)


def test_bounds_order_and_evidence():
    y = 1.3
    _, _, log_evidence = conjugate_posterior(y)
    result = find_weigh_learn(
        conjugate_model, random.PRNGKey(0), (y,), num_modes=4, num_particles=2048
    )
    key = random.PRNGKey(2)
    elbo = result.elbo(result.params, key)
    iwae = result.iwae(result.params, key)
    assert elbo <= iwae + 1e-5  # Jensen
    assert iwae <= log_evidence + 0.05  # both bound the evidence from below


def test_discrete_configuration_is_the_exact_argmax():
    """
    Brute-force check that the max-product pass returns the true joint argmax
    over the discrete sites, given the continuous latent it converged to.
    """
    result = find_weigh_learn(
        chain_model, random.PRNGKey(0), (CHAIN_DATA,), num_modes=1, num_particles=4
    )
    names = [f"s_{t}" for t in range(CHAIN_DATA.shape[0])]
    found = {name: result.modes[name][0] for name in names}
    latent = result.find_state.latent[0]

    def energy(discrete):
        substituted = handlers.substitute(
            handlers.seed(chain_model, random.PRNGKey(0)), data=discrete
        )
        return potential_energy(substituted, (CHAIN_DATA,), {}, {"loc": latent[0]})

    best = min(
        itertools.product([0, 1], repeat=len(names)),
        key=lambda config: float(
            energy({name: jnp.array(v) for name, v in zip(names, config)})
        ),
    )
    assert tuple(int(found[name]) for name in names) == best


def test_gmm_recovers_cluster_structure():
    result = find_weigh_learn(
        gmm, random.PRNGKey(0), (GMM_DATA,), num_modes=6, num_particles=8
    )
    best = int(jnp.argmax(result.log_joint))
    locs = np.sort(np.asarray(result.modes["locs"][best]))
    assert_allclose(locs, np.array([-2.1, 2.0]), atol=0.2)
    assignment = np.asarray(result.modes["k"][best])
    assert (assignment[:3] == assignment[0]).all()
    assert (assignment[3:] == assignment[3]).all()
    assert assignment[0] != assignment[3]


@pytest.mark.parametrize("mode_source", ["restart", "perturb", "both"])
@pytest.mark.parametrize("covariance", ["diagonal", "full"])
def test_option_combinations(mode_source, covariance):
    result = find_weigh_learn(
        gmm,
        random.PRNGKey(1),
        (GMM_DATA,),
        num_modes=2,
        num_particles=4,
        mode_source=mode_source,
        covariance=covariance,
    )
    scale_name = "_fwl_scale" if covariance == "diagonal" else "_fwl_scale_tril"
    expected = (2, 2) if covariance == "diagonal" else (2, 2, 2)
    assert result.params[scale_name].shape == expected
    weights = result.log_weights(result.params, random.PRNGKey(0))
    assert weights.shape == (4,)
    assert np.all(np.isfinite(weights))


def test_perturbation_diversifies_modes():
    """
    Plain restarts on this symmetric mixture collapse onto the same optimum, so
    the components duplicate. Perturb-and-MAP trades a little joint density for
    genuinely distinct components, and ``"both"`` keeps some of each.
    """

    def distinct_locs(result):
        return len(
            {tuple(row) for row in np.round(np.asarray(result.modes["locs"]), 2)}
        )

    def run(mode_source):
        return find_weigh_learn(
            gmm,
            random.PRNGKey(7),
            (GMM_DATA,),
            num_modes=4,
            num_particles=4,
            mode_source=mode_source,
        )

    restart, perturb, both = run("restart"), run("perturb"), run("both")
    assert distinct_locs(restart) < distinct_locs(perturb)
    assert distinct_locs(both) > distinct_locs(restart)
    # the unperturbed runs find at least as good a joint density as perturbed ones
    assert restart.log_joint.max() >= perturb.log_joint.max() - 1e-4
    # "both" starts with the unperturbed runs
    assert_allclose(both.log_joint[0], restart.log_joint[0], atol=1e-4)


@pytest.mark.parametrize("granularity", ["site", "element"])
@pytest.mark.parametrize("objective", ["joint", "marginal"])
def test_fisher_and_objective_variants(granularity, objective):
    result = find_weigh_learn(
        gmm,
        random.PRNGKey(2),
        (GMM_DATA,),
        num_modes=2,
        num_particles=4,
        fisher_granularity=granularity,
        continuous_objective=objective,
    )
    assert np.all(np.isfinite(result.log_joint))
    assert np.all(result.params["_fwl_scale"] > 0)


def test_element_granularity_gives_a_larger_fisher():
    """
    At a joint mode the *summed* per-site gradients very nearly cancel, since
    that is what being a mode means, so the site-granularity Fisher collapses
    toward zero and the damped scale toward ``1/sqrt(lambda)``. Resolving the
    plate into per-observation outer products leaves a Fisher that does not
    cancel, giving a strictly tighter proposal.
    """
    kwargs = dict(
        num_modes=1,
        num_particles=2,
        init_strategy=init_to_value(values={"locs": jnp.array([-2.0, 2.0])}),
    )
    coarse = find_weigh_learn(
        gmm, random.PRNGKey(3), (GMM_DATA,), fisher_granularity="site", **kwargs
    )
    fine = find_weigh_learn(
        gmm, random.PRNGKey(3), (GMM_DATA,), fisher_granularity="element", **kwargs
    )
    assert np.all(
        np.asarray(fine.params["_fwl_scale"])
        <= np.asarray(coarse.params["_fwl_scale"]) + 1e-6
    )


def test_frozen_proposal_has_no_params_and_no_gradient():
    result = find_weigh_learn(
        conjugate_model,
        random.PRNGKey(0),
        (1.3,),
        num_modes=2,
        num_particles=4,
        learnable=False,
    )
    assert result.params == {}
    # the guide still runs and the weights are finite
    weights = result.log_weights({}, random.PRNGKey(0))
    assert weights.shape == (4,)
    assert np.all(np.isfinite(weights))


def test_learning_gradient_flows_to_model_params():
    """The Section 4 learning step: grad of the IWAE bound w.r.t. theta."""

    def model(y):
        theta = numpyro.param("theta", 0.5)
        z = numpyro.sample("z", dist.Normal(theta, 1.0))
        numpyro.sample("y", dist.Normal(z, 0.5), obs=y)

    result = find_weigh_learn(model, random.PRNGKey(0), (2.0,), num_modes=2)
    assert "theta" in result.params

    def loss(params, key):
        return -result.iwae(params, key)

    grads = jax.grad(loss)(result.params, random.PRNGKey(1))
    assert jnp.isfinite(grads["theta"])
    # the data pulls theta up from 0.5 toward 2.0, so the loss decreases with theta
    assert grads["theta"] < 0

    # one gradient step must improve the bound
    key = random.PRNGKey(3)
    stepped = dict(result.params)
    stepped["theta"] = stepped["theta"] - 0.1 * grads["theta"]
    assert result.iwae(stepped, key) > result.iwae(result.params, key)


def test_guide_is_usable_with_svi():
    from numpyro.contrib.funsor import config_enumerate
    from numpyro.infer import SVI, TraceEnum_ELBO

    result = find_weigh_learn(
        gmm, random.PRNGKey(0), (GMM_DATA,), num_modes=3, num_particles=4
    )
    # Enough particles that the comparison is not dominated by which mixture
    # component each draw lands in.
    svi = SVI(
        config_enumerate(gmm),
        result.guide,
        numpyro.optim.Adam(1e-2),
        TraceEnum_ELBO(num_particles=256),
    )
    state = svi.init(random.PRNGKey(1), GMM_DATA)
    first = svi.evaluate(state, GMM_DATA)
    for _ in range(50):
        state, _ = svi.stable_update(state, GMM_DATA)
    assert svi.evaluate(state, GMM_DATA) < first


def test_result_is_a_pytree():
    result = find_weigh_learn(
        conjugate_model, random.PRNGKey(0), (1.3,), num_modes=2, num_particles=4
    )
    leaves = jax.tree.leaves(result)
    assert all(isinstance(leaf, jnp.ndarray) for leaf in leaves)
    doubled = jax.tree.map(lambda x: x * 2, result)
    assert_allclose(doubled.log_joint, result.log_joint * 2)
    # survives a round trip through jit as both argument and return value
    assert_allclose(
        jax.jit(lambda r: r.log_joint.sum())(result), result.log_joint.sum()
    )
    assert_allclose(jax.jit(lambda r: r)(result).modes["z"], result.modes["z"])


def test_modes_dict_is_vmappable():
    result = find_weigh_learn(
        gmm, random.PRNGKey(0), (GMM_DATA,), num_modes=4, num_particles=4
    )
    assert set(result.modes) == {"locs", "k", "obs"}
    assert all(v.shape[0] == 4 for v in result.modes.values())

    def per_mode(mode):
        return jnp.sum(mode["locs"]) + jnp.sum(mode["k"])

    assert jax.vmap(per_mode)(result.modes).shape == (4,)


def test_estimators_accept_fresh_data():
    result = find_weigh_learn(
        conjugate_model, random.PRNGKey(0), (1.3,), num_modes=2, num_particles=4
    )
    default = result.log_weights(result.params, random.PRNGKey(0))
    same = result.log_weights(result.params, random.PRNGKey(0), 1.3)
    other = result.log_weights(result.params, random.PRNGKey(0), 4.0)
    assert_allclose(default, same)
    assert not np.allclose(default, other)


def test_num_particles_override():
    result = find_weigh_learn(
        conjugate_model, random.PRNGKey(0), (1.3,), num_modes=2, num_particles=4
    )
    assert result.log_weights(result.params, random.PRNGKey(0)).shape == (4,)
    assert result.log_weights(
        result.params, random.PRNGKey(0), num_particles=16
    ).shape == (16,)


def discrete_only_model(y, obs_scale=0.6):
    """Two coupled binary latents and one observation: log Z is a sum over four terms."""
    a = numpyro.sample("a", dist.Bernoulli(0.3))
    b = numpyro.sample("b", dist.Bernoulli(jnp.where(a == 1, 0.8, 0.2)))
    numpyro.sample("obs", dist.Normal(a + 2.0 * b, obs_scale), obs=y)


def discrete_only_reference(y, obs_scale=0.6):
    """Brute-force ``(log Z, argmax configuration)`` for ``discrete_only_model``."""
    terms = {}
    for a, b in itertools.product([0, 1], repeat=2):
        terms[(a, b)] = float(
            dist.Bernoulli(0.3).log_prob(a)
            + dist.Bernoulli(0.8 if a == 1 else 0.2).log_prob(b)
            + dist.Normal(a + 2.0 * b, obs_scale).log_prob(y)
        )
    log_z = float(jax.scipy.special.logsumexp(jnp.array(list(terms.values()))))
    return log_z, max(terms, key=terms.get)


def test_enumeration_only_returns_the_exact_result():
    y = 1.8
    log_z, best = discrete_only_reference(y)
    result = find_weigh_learn(
        discrete_only_model,
        random.PRNGKey(0),
        (y,),
        num_modes=3,
        num_particles=5,
        allow_enumeration_only=True,
    )
    assert result.structure.enumeration_only
    assert result.structure.continuous == ()
    assert "enumeration only" in result.structure.summary()

    # Find reduces to one exact max-product pass, so every run agrees on the argmax
    assert result.modes["a"].shape == (3,)
    for name, value in zip(("a", "b"), best):
        assert np.all(np.asarray(result.modes[name]) == value)
    assert np.all(result.converged)
    assert np.all(np.asarray(result.sweeps) == 0)

    # the guide is empty and carries no parameters
    assert result.params == {}
    assert result.find_state.latent.shape == (3, 0)
    guide_trace = handlers.trace(
        handlers.seed(result.guide, random.PRNGKey(1))
    ).get_trace(y)
    assert guide_trace == {}

    # every weight is exactly log Z, so both bounds are tight
    weights = result.log_weights({}, random.PRNGKey(2))
    assert weights.shape == (5,)
    assert_allclose(weights, np.full(5, log_z), rtol=1e-5)
    assert_allclose(result.elbo({}, random.PRNGKey(2)), log_z, rtol=1e-5)
    assert_allclose(result.iwae({}, random.PRNGKey(2)), log_z, rtol=1e-5)


def test_enumeration_only_learns_model_params():
    """The Section 4 gradient is still available, and is now exact."""

    def model(y):
        logit = numpyro.param("logit", 0.0)
        a = numpyro.sample("a", dist.BernoulliLogits(logit))
        numpyro.sample("obs", dist.Normal(a, 0.5), obs=y)

    result = find_weigh_learn(
        model, random.PRNGKey(0), (1.0,), num_modes=1, allow_enumeration_only=True
    )
    assert set(result.params) == {"logit"}
    grads = jax.grad(lambda p, k: -result.iwae(p, k))(result.params, random.PRNGKey(1))
    # the observation sits at a=1, so the exact evidence rises with the logit
    assert grads["logit"] < 0

    stepped = {"logit": result.params["logit"] - 0.5 * grads["logit"]}
    assert result.iwae(stepped, random.PRNGKey(1)) > result.iwae(
        result.params, random.PRNGKey(1)
    )


def test_enumeration_only_perturb_samples_rather_than_maximizes():
    """
    With no continuous half to tilt, ``mode_source="perturb"`` reduces to exact
    forward-filter-backward-sample over the discrete sites, so the runs must
    disagree where the posterior is genuinely uncertain. Every observation here
    sits exactly between the two component means.
    """

    def ambiguous(data):
        with numpyro.plate("N", data.shape[0]):
            k = numpyro.sample("k", dist.Categorical(jnp.array([0.5, 0.5])))
            numpyro.sample("obs", dist.Normal(k * 2.0, 0.7), obs=data)

    data = np.full(4, 1.0, dtype=np.float32)

    def run(mode_source):
        result = find_weigh_learn(
            ambiguous,
            random.PRNGKey(0),
            (data,),
            num_modes=8,
            num_particles=2,
            mode_source=mode_source,
            allow_enumeration_only=True,
        )
        return {tuple(row) for row in np.asarray(result.modes["k"])}

    # the argmax is a single configuration, so restarts all agree
    assert len(run("restart")) == 1
    # sampling cannot agree eight times over four fair coin flips
    assert len(run("perturb")) > 1


@pytest.mark.parametrize("covariance", ["diagonal", "full"])
def test_enumeration_only_ignores_covariance_choice(covariance):
    result = find_weigh_learn(
        discrete_only_model,
        random.PRNGKey(0),
        (1.8,),
        num_modes=2,
        num_particles=2,
        covariance=covariance,
        allow_enumeration_only=True,
    )
    log_z, _ = discrete_only_reference(1.8)
    assert_allclose(result.elbo({}, random.PRNGKey(0)), log_z, rtol=1e-5)


def test_options_object_and_overrides():
    options = FWLOptions(num_modes=2, num_particles=3, init_strategy=init_to_sample)
    result = find_weigh_learn(
        conjugate_model, random.PRNGKey(0), (1.3,), options=options, num_particles=5
    )
    assert result.options.num_modes == 2
    assert result.options.num_particles == 5
    assert result.log_weights(result.params, random.PRNGKey(0)).shape == (5,)
