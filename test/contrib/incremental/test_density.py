# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from numpy.testing import assert_allclose
import pytest
from zoo import ZOO

import jax
import jax.numpy as jnp

import numpyro
from numpyro.contrib.incremental import trace_density
import numpyro.distributions as dist
from numpyro.infer.util import log_density


def reference_log_density(entry, program, values):
    """Joint log density of the stored values computed without the program."""
    constrained = {}
    log_det = 0.0
    for name in program.site_names:
        meta = program.sites[name]
        if meta.is_discrete:
            constrained[name] = values[name]
        else:
            u = values[name]
            x = meta.transform(u)
            constrained[name] = x
            log_det = log_det + jnp.sum(meta.transform.log_abs_det_jacobian(u, x))
    lp, _ = log_density(entry.model, entry.args, entry.kwargs, constrained)
    return lp + log_det


def test_eval_matches_reference(zoo_entry):
    name, entry = zoo_entry
    program, init_values = trace_density(entry.model, entry.args, entry.kwargs)
    total = program.log_density(init_values, entry.args, entry.kwargs)
    expected = reference_log_density(entry, program, init_values)
    assert_allclose(total, expected, rtol=1e-5, atol=1e-5)


def test_eval_is_jittable(zoo_entry):
    name, entry = zoo_entry
    program, init_values = trace_density(entry.model, entry.args, entry.kwargs)
    f = jax.jit(lambda v, a: program.log_density(v, a, entry.kwargs))
    assert_allclose(
        f(init_values, entry.args),
        program.log_density(init_values, entry.args, entry.kwargs),
        rtol=1e-6,
    )


def test_gmm_metadata():
    entry = ZOO["gmm"]
    program, init_values = trace_density(entry.model, entry.args, entry.kwargs)
    n = entry.args[0].shape[0]
    assert set(program.site_names) == {"w", "s", "mu", "z"}
    assert program.sites["z"].plates == (("data", 0),)
    assert program.sites["z"].is_discrete
    assert program.sites["z"].support_size == 3
    assert program.sites["mu"].plates == (("comp", 0),)
    assert program.sites["w"].plates == ()
    assert program.sites["w"].shape == (2,)  # unconstrained simplex
    assert program.factors["obs"].plates == (("data", 0),)
    assert program.factors["obs"].shape == (n,)
    assert program.factors["obs"].is_observed
    assert program.factors["_w_log_det"].owner == "w"
    assert program.factors["_s_log_det"].owner == "s"
    assert program.factors["z"].shape == (n,)
    assert program.factors["mu"].shape == (3,)
    # Factor outputs are plain arrays in factor order.
    out = program.eval(init_values, entry.args, entry.kwargs)
    assert list(out) == list(program.factor_names)


def test_nested_metadata():
    entry = ZOO["nested"]
    program, _ = trace_density(entry.model, entry.args, entry.kwargs)
    assert program.sites["a"].plates == (("M", 0),)
    assert program.sites["a"].shape == (3, 1)
    assert program.sites["b"].plates == (("M", 0), ("N", 1))
    assert program.factors["obs"].plates == (("M", 0), ("N", 1))


def test_static_arg_is_closed_over():
    entry = ZOO["static_arg"]
    program, _ = trace_density(entry.model, entry.args, entry.kwargs)
    assert program.num_arg_inputs == 1  # only ``y`` is traced
    with pytest.raises(ValueError, match="do not match"):
        program.flatten_args(entry.args, {})


def test_validation_off_shrinks_program():
    entry = ZOO["gmm"]
    p_off, _ = trace_density(entry.model, entry.args, entry.kwargs, validate=False)
    p_on, _ = trace_density(entry.model, entry.args, entry.kwargs, validate=True)
    assert p_off.num_eqns < p_on.num_eqns


def test_unsupported_sites_raise():
    def enum_model(y):
        x = numpyro.sample("x", dist.Bernoulli(0.5), infer={"enumerate": "parallel"})
        numpyro.sample("obs", dist.Normal(x, 1.0), obs=y)

    with pytest.raises(NotImplementedError, match="enumeration"):
        trace_density(enum_model, (jnp.zeros(3),))

    def subsample_model(y):
        mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
        with numpyro.plate("N", y.shape[0], subsample_size=2):
            numpyro.sample("obs", dist.Normal(mu, 1.0), obs=numpyro.subsample(y, 0))

    with pytest.raises(NotImplementedError, match="subsampled"):
        trace_density(subsample_model, (jnp.zeros(3),))


def test_params_are_frozen():
    def model(y):
        loc = numpyro.param("loc", jnp.array(0.5))
        x = numpyro.sample("x", dist.Normal(loc, 1.0))
        numpyro.sample("obs", dist.Normal(x, 1.0), obs=y)

    program, init_values = trace_density(model, (jnp.zeros(2),))
    assert program.site_names == ("x",)
    lp = program.log_density(init_values, (jnp.zeros(2),))
    assert jnp.isfinite(lp)
