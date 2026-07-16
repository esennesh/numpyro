# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end QEM: conjugate posterior recovery (scalar, plated hierarchical,
multivariate, discrete), the log P_MP convergence trace, EMA/forgetting
semantics, and the paper's reparameterization-invariance property (Thm 2).
"""

import numpy as np
import pytest

import jax
from jax import random
import jax.numpy as jnp

pytest.importorskip("funsor")

import numpyro  # noqa: E402
from numpyro.contrib.qem import QEM  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer.autoguide import AutoExponentialFamily  # noqa: E402

jax.config.update("jax_enable_x64", True)

MU0, TAU0, SIGMA, X_OBS = 0.5, 1.2, 0.8, 2.0


def scalar_model():
    z = numpyro.sample("z", dist.Normal(MU0, TAU0))
    numpyro.sample("x", dist.Normal(z, SIGMA), obs=X_OBS)


def scalar_posterior():
    post_var = 1.0 / (1.0 / TAU0**2 + 1.0 / SIGMA**2)
    post_mean = post_var * (MU0 / TAU0**2 + X_OBS / SIGMA**2)
    return post_mean, post_var


def test_conjugate_normal_recovery():
    guide = AutoExponentialFamily(scalar_model)
    qem = QEM(scalar_model, guide, num_samples=128)
    result = qem.run(random.PRNGKey(0), 60, progress_bar=False)

    post_mean, post_var = scalar_posterior()
    m = result.state.mean_params["z"]
    assert m["x"] == pytest.approx(post_mean, abs=0.03)
    assert m["xx"] - m["x"] ** 2 == pytest.approx(post_var, abs=0.05)

    # the log-marginal trace converges to the analytic evidence
    log_evidence = dist.Normal(MU0, jnp.sqrt(TAU0**2 + SIGMA**2)).log_prob(X_OBS)
    np.testing.assert_allclose(
        jnp.mean(result.log_marginals[-20:]), log_evidence, atol=0.05
    )

    # M-step output round-trips through the guide's params
    assert result.params["z_auto_loc"] == pytest.approx(m["x"])
    samples = guide.sample_posterior(
        random.PRNGKey(1), result.params, sample_shape=(1000,)
    )
    assert samples["z"].shape == (1000,)
    assert jnp.mean(samples["z"]) == pytest.approx(post_mean, abs=0.1)


def test_plated_hierarchical_recovery():
    a, b, c = 1.0, 0.7, 0.5
    x_obs = jnp.array([0.3, -0.6, 1.1, 0.4, -0.2])
    n = x_obs.shape[0]

    def model():
        mu = numpyro.sample("mu", dist.Normal(0.0, a))
        with numpyro.plate("data", n):
            z = numpyro.sample("z", dist.Normal(mu, b))
            numpyro.sample("x", dist.Normal(z, c), obs=x_obs)

    # analytic posterior
    mu_prec = 1.0 / a**2 + n / (b**2 + c**2)
    mu_mean = (jnp.sum(x_obs) / (b**2 + c**2)) / mu_prec
    v = 1.0 / (1.0 / b**2 + 1.0 / c**2)
    z_mean = v * (mu_mean / b**2 + x_obs / c**2)
    z_var = v + (v / b**2) ** 2 / mu_prec

    guide = AutoExponentialFamily(model)
    qem = QEM(model, guide, num_samples=64)
    result = qem.run(random.PRNGKey(0), 80, progress_bar=False)

    m_mu = result.state.mean_params["mu"]
    m_z = result.state.mean_params["z"]
    assert m_mu["x"] == pytest.approx(mu_mean, abs=0.05)
    assert m_mu["xx"] - m_mu["x"] ** 2 == pytest.approx(1.0 / mu_prec, abs=0.05)
    assert m_z["x"].shape == (n,)
    np.testing.assert_allclose(m_z["x"], z_mean, atol=0.05)
    np.testing.assert_allclose(m_z["xx"] - m_z["x"] ** 2, z_var, atol=0.05)


def test_multivariate_site_recovery():
    prior_loc = jnp.array([0.5, -1.0])
    prior_cov = jnp.array([[2.0, 0.6], [0.6, 1.0]])
    obs_cov = 0.5 * jnp.eye(2)
    x_obs = jnp.array([1.0, 0.0])

    def model():
        z = numpyro.sample(
            "z", dist.MultivariateNormal(prior_loc, covariance_matrix=prior_cov)
        )
        numpyro.sample(
            "x", dist.MultivariateNormal(z, covariance_matrix=obs_cov), obs=x_obs
        )

    prec = jnp.linalg.inv(prior_cov) + jnp.linalg.inv(obs_cov)
    post_cov = jnp.linalg.inv(prec)
    post_mean = post_cov @ (
        jnp.linalg.inv(prior_cov) @ prior_loc + jnp.linalg.inv(obs_cov) @ x_obs
    )

    guide = AutoExponentialFamily(model)
    qem = QEM(model, guide, num_samples=128)
    result = qem.run(random.PRNGKey(0), 80, progress_bar=False)

    m = result.state.mean_params["z"]
    np.testing.assert_allclose(m["x"], post_mean, atol=0.05)
    cov = m["xx"] - m["x"][:, None] * m["x"][None, :]
    np.testing.assert_allclose(cov, post_cov, atol=0.08)


def test_discrete_latent_recovery():
    p_prior, sigma, x_obs = 0.3, 0.7, 0.9

    def model():
        z = numpyro.sample("z", dist.Bernoulli(probs=p_prior))
        numpyro.sample("x", dist.Normal(z, sigma), obs=x_obs)

    w1 = p_prior * jnp.exp(dist.Normal(1.0, sigma).log_prob(x_obs))
    w0 = (1.0 - p_prior) * jnp.exp(dist.Normal(0.0, sigma).log_prob(x_obs))
    post_p = w1 / (w0 + w1)

    guide = AutoExponentialFamily(model)
    qem = QEM(model, guide, num_samples=128)
    result = qem.run(random.PRNGKey(0), 60, progress_bar=False)

    assert result.state.mean_params["z"]["x"] == pytest.approx(post_p, abs=0.03)


def test_decorrelated_normalizer():
    """Flag on: same conjugate recovery; weights actually get rescaled.

    With the same rng key, the main proposal batch (and hence the raw
    source-term weights) is identical in both variants, so any difference in
    the first update comes exactly from the fresh-batch normalizer.
    """
    guide = AutoExponentialFamily(scalar_model)
    qem = QEM(scalar_model, guide, num_samples=128, decorrelated_normalizer=True)
    result = qem.run(random.PRNGKey(0), 60, progress_bar=False)

    post_mean, post_var = scalar_posterior()
    m = result.state.mean_params["z"]
    assert m["x"] == pytest.approx(post_mean, abs=0.03)
    assert m["xx"] - m["x"] ** 2 == pytest.approx(post_var, abs=0.05)

    plain = QEM(scalar_model, guide, num_samples=32)
    decorr = QEM(scalar_model, guide, num_samples=32, decorrelated_normalizer=True)
    state = plain.init(random.PRNGKey(3))
    plain_state, plain_lm = plain.update(state)
    decorr_state, decorr_lm = decorr.update(state)
    # same main batch: the reported log P_MP is identical...
    np.testing.assert_allclose(plain_lm, decorr_lm)
    # ...but the moment estimate is rescaled by P_MP(z) / P_MP(z')
    assert not np.allclose(
        plain_state.mean_params["z"]["x"], decorr_state.mean_params["z"]["x"]
    )


def test_forget_semantics():
    guide = AutoExponentialFamily(scalar_model)

    # schedule: lambda(1) = 0, so the first update takes the raw estimate in full
    qem = QEM(scalar_model, guide, num_samples=32, schedule_power=1.0)
    state = qem.init(random.PRNGKey(0))
    assert state.step == 0 and set(state.mean_params) == {"z"}
    # init is prior-moment-matched
    assert state.mean_params["z"]["x"] == pytest.approx(MU0)
    assert state.mean_params["z"]["xx"] == pytest.approx(MU0**2 + TAU0**2)

    state1, log_marginal = qem.update(state)
    assert state1.step == 1
    assert jnp.isfinite(log_marginal)

    # forget=1-eps keeps the state (nearly) frozen
    frozen = QEM(scalar_model, guide, num_samples=32, forget=0.999999)
    fstate, _ = frozen.update(state)
    assert fstate.mean_params["z"]["x"] == pytest.approx(
        state.mean_params["z"]["x"], abs=1e-4
    )

    # callable forget is honored: lambda == 0 reproduces the raw first estimate
    raw = QEM(scalar_model, guide, num_samples=32, forget=lambda t: 0.0)
    rstate, _ = raw.update(state)
    np.testing.assert_allclose(
        rstate.mean_params["z"]["x"], state1.mean_params["z"]["x"]
    )

    with pytest.raises(ValueError, match="schedule_power"):
        QEM(scalar_model, guide, num_samples=32, schedule_power=0.3)
    with pytest.raises(ValueError, match="forget"):
        QEM(scalar_model, guide, num_samples=32, forget=1.5)


def test_non_exp_family_site_rejected():
    def model():
        z = numpyro.sample("z", dist.StudentT(3.0))
        numpyro.sample("x", dist.Normal(z, 1.0), obs=0.5)

    guide = AutoExponentialFamily(model)
    qem = QEM(model, guide, num_samples=8)
    with pytest.raises(ValueError, match="exponential-family"):
        qem.init(random.PRNGKey(0))


def test_reparameterization_invariance():
    """Paper Thm 2: QEM trajectories are invariant to rescaling a latent.

    Model B is model A with z scaled by alpha (prior and likelihood adjusted so
    the models are equivalent). With identical rng keys, every iteration's mean
    parameters must match the scaled ones of model A to float tolerance.
    """
    alpha = 1e-2

    def model_a():
        z = numpyro.sample("z", dist.Normal(MU0, TAU0))
        numpyro.sample("x", dist.Normal(z, SIGMA), obs=X_OBS)

    def model_b():
        z = numpyro.sample("z", dist.Normal(alpha * MU0, alpha * TAU0))
        numpyro.sample("x", dist.Normal(z / alpha, SIGMA), obs=X_OBS)

    num_steps = 10

    def trajectory(model):
        guide = AutoExponentialFamily(model)
        qem = QEM(model, guide, num_samples=32)
        state = qem.init(random.PRNGKey(7))
        locs, log_marginals = [], []
        for _ in range(num_steps):
            state, log_marginal = qem.update(state)
            locs.append(state.mean_params["z"]["x"])
            log_marginals.append(log_marginal)
        return jnp.stack(locs), jnp.stack(log_marginals)

    locs_a, lm_a = trajectory(model_a)
    locs_b, lm_b = trajectory(model_b)
    np.testing.assert_allclose(locs_b, alpha * locs_a, rtol=1e-8)
    np.testing.assert_allclose(lm_b, lm_a, rtol=1e-8)


def test_guide_works_with_svi():
    """AutoExponentialFamily remains a valid SVI guide (param-store compatible)."""
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.optim import Adam

    guide = AutoExponentialFamily(scalar_model)
    svi = SVI(scalar_model, guide, Adam(0.05), Trace_ELBO(num_particles=16))
    result = svi.run(random.PRNGKey(0), 2000, progress_bar=False)
    post_mean, post_var = scalar_posterior()
    assert result.params["z_auto_loc"] == pytest.approx(post_mean, abs=0.15)
    assert jnp.exp(result.params["z_auto_scale"]) == pytest.approx(
        jnp.sqrt(post_var), abs=0.15
    )
