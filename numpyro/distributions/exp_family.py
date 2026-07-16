# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Exponential-family interface: sufficient statistics, mean parameters, and moment
matching for NumPyro distributions.

Mean parameters are plain pytrees (dicts of arrays) — one entry per sufficient
statistic, each leaf shaped ``batch_shape + statistic_shape``. Three registries,
keyed by distribution type via :func:`functools.singledispatch`, expose everything
an expectation-maximization loop needs:

- :func:`sufficient_statistics` — per-observation ``T(x)``,
- :func:`mean_params` — ``E[T(z)]`` under the distribution,
- :func:`from_mean_params` — moment matching: rebuild a distribution of the same
  family from a mean-parameter pytree.

:func:`canonical_params` additionally names the constructor arguments used to
rebuild each family (needed where ``arg_constraints`` over-specifies, e.g.
:class:`~numpyro.distributions.MultivariateNormal`).

Closed-form implementations are provided for Normal, MultivariateNormal,
Bernoulli, Categorical, Poisson and Exponential; :class:`Independent` and
:class:`ExpandedDistribution` wrappers are unwrapped transparently. Additional
families (including iterative NP↔EP conversions) can be registered externally —
see ``docs/design/qem_mpiw_plan.md``.
"""

from functools import singledispatch

from jax.nn import one_hot
import jax.numpy as jnp
from jax.scipy.special import logit

from numpyro.distributions.continuous import (
    Exponential,
    MultivariateNormal,
    Normal,
)
from numpyro.distributions.discrete import (
    BernoulliLogits,
    BernoulliProbs,
    CategoricalLogits,
    CategoricalProbs,
    Poisson,
)
from numpyro.distributions.distribution import ExpandedDistribution, Independent

__all__ = [
    "base_distribution",
    "canonical_params",
    "from_mean_params",
    "is_exp_family",
    "mean_params",
    "sufficient_statistics",
]


def _not_registered(d):
    return NotImplementedError(
        f"{type(d).__name__} has no exponential-family registration. Supported "
        "out of the box: Normal, MultivariateNormal, Bernoulli, Categorical, "
        "Poisson, Exponential (optionally wrapped in Independent or "
        "ExpandedDistribution). Register implementations of sufficient_statistics, "
        "mean_params and from_mean_params to add a family."
    )


def base_distribution(d):
    """Unwrap :class:`Independent` / :class:`ExpandedDistribution` wrappers."""
    while isinstance(d, (Independent, ExpandedDistribution)):
        d = d.base_dist
    return d


def is_exp_family(d) -> bool:
    """Whether ``d`` (after unwrapping) has an exponential-family registration."""
    cls = type(base_distribution(d))
    return mean_params.dispatch(cls) is not mean_params.dispatch(object)


@singledispatch
def sufficient_statistics(d, value):
    """Per-observation sufficient statistics ``T(x)`` of ``d``'s family.

    :param d: a registered :class:`~numpyro.distributions.Distribution`.
    :param value: a value in ``d``'s support, of shape
        ``sample_shape + batch_shape + event_shape``.
    :return: dict of arrays, each of shape
        ``sample_shape + batch_shape + statistic_shape``.
    """
    raise _not_registered(d)


@singledispatch
def mean_params(d):
    """Mean parameters ``E[T(z)]`` of ``d``, as a dict of arrays.

    Each leaf has shape ``batch_shape + statistic_shape``; averaging
    :func:`sufficient_statistics` of samples from ``d`` converges to this value.
    """
    raise _not_registered(d)


@singledispatch
def from_mean_params(d, params):
    """Moment matching: a distribution of the same family as prototype ``d``
    whose mean parameters equal ``params``.

    Batch shape follows the leaves of ``params`` (not the prototype); wrapper
    structure (``Independent``) is preserved. Parameterization follows the
    prototype's class (e.g. a ``BernoulliLogits`` prototype yields
    ``BernoulliLogits``), so mean parameters must lie in the interior of the
    mean domain where the inverse mapping requires it.
    """
    raise _not_registered(d)


@singledispatch
def canonical_params(d):
    """Constructor arguments ``{name: value}`` that rebuild ``d``'s family.

    Defaults to ``arg_constraints``; overridden where that over-specifies
    (e.g. MultivariateNormal exposes only ``loc`` and ``scale_tril``).
    """
    return {name: getattr(d, name) for name in d.arg_constraints}


################################################################################
# Wrappers
################################################################################


@sufficient_statistics.register(Independent)
@sufficient_statistics.register(ExpandedDistribution)
def _(d, value):
    return sufficient_statistics(d.base_dist, value)


@mean_params.register(Independent)
def _(d):
    return mean_params(d.base_dist)


@mean_params.register(ExpandedDistribution)
def _(d):
    params = mean_params(d.base_dist)
    base_batch_ndim = len(d.base_dist.batch_shape)
    return {
        k: jnp.broadcast_to(v, d.batch_shape + jnp.shape(v)[base_batch_ndim:])
        for k, v in params.items()
    }


@from_mean_params.register(Independent)
def _(d, params):
    return from_mean_params(d.base_dist, params).to_event(d.reinterpreted_batch_ndims)


@from_mean_params.register(ExpandedDistribution)
def _(d, params):
    # the leaves of ``params`` already carry the (expanded) batch shape
    return from_mean_params(d.base_dist, params)


@canonical_params.register(Independent)
@canonical_params.register(ExpandedDistribution)
def _(d):
    return canonical_params(d.base_dist)


################################################################################
# Closed-form families
################################################################################


def _float(value):
    return jnp.asarray(value, jnp.result_type(float))


@sufficient_statistics.register(Normal)
def _(d, value):
    value = _float(value)
    return {"x": value, "xx": jnp.square(value)}


@mean_params.register(Normal)
def _(d):
    return {
        "x": jnp.broadcast_to(d.loc, d.batch_shape),
        "xx": jnp.broadcast_to(jnp.square(d.loc) + jnp.square(d.scale), d.batch_shape),
    }


@from_mean_params.register(Normal)
def _(d, params):
    m1, m2 = params["x"], params["xx"]
    var = m2 - jnp.square(m1)
    tiny = jnp.finfo(jnp.result_type(var)).tiny
    return Normal(m1, jnp.sqrt(jnp.maximum(var, tiny)))


def _outer(x):
    return x[..., :, None] * x[..., None, :]


@sufficient_statistics.register(MultivariateNormal)
def _(d, value):
    value = _float(value)
    return {"x": value, "xx": _outer(value)}


@mean_params.register(MultivariateNormal)
def _(d):
    loc = jnp.broadcast_to(d.loc, d.batch_shape + d.event_shape)
    cov = jnp.broadcast_to(d.covariance_matrix, d.batch_shape + d.event_shape * 2)
    return {"x": loc, "xx": cov + _outer(loc)}


@from_mean_params.register(MultivariateNormal)
def _(d, params):
    m1, m2 = params["x"], params["xx"]
    cov = m2 - _outer(m1)
    cov = (cov + jnp.swapaxes(cov, -1, -2)) / 2  # enforce exact symmetry
    return MultivariateNormal(loc=m1, covariance_matrix=cov)


@canonical_params.register(MultivariateNormal)
def _(d):
    return {"loc": d.loc, "scale_tril": d.scale_tril}


@sufficient_statistics.register(BernoulliProbs)
@sufficient_statistics.register(BernoulliLogits)
def _(d, value):
    return {"x": _float(value)}


@mean_params.register(BernoulliProbs)
@mean_params.register(BernoulliLogits)
def _(d):
    return {"x": jnp.broadcast_to(d.probs, d.batch_shape)}


@from_mean_params.register(BernoulliProbs)
def _(d, params):
    return BernoulliProbs(params["x"])


@from_mean_params.register(BernoulliLogits)
def _(d, params):
    return BernoulliLogits(logit(params["x"]))


def _num_categories(d):
    return (d.probs if isinstance(d, CategoricalProbs) else d.logits).shape[-1]


@sufficient_statistics.register(CategoricalProbs)
@sufficient_statistics.register(CategoricalLogits)
def _(d, value):
    return {"onehot": one_hot(value, _num_categories(d), dtype=jnp.result_type(float))}


@mean_params.register(CategoricalProbs)
@mean_params.register(CategoricalLogits)
def _(d):
    return {"onehot": jnp.broadcast_to(d.probs, d.batch_shape + (_num_categories(d),))}


@from_mean_params.register(CategoricalProbs)
def _(d, params):
    probs = params["onehot"]
    return CategoricalProbs(probs / probs.sum(-1, keepdims=True))


@from_mean_params.register(CategoricalLogits)
def _(d, params):
    return CategoricalLogits(jnp.log(params["onehot"]))


@sufficient_statistics.register(Poisson)
def _(d, value):
    return {"x": _float(value)}


@mean_params.register(Poisson)
def _(d):
    return {"x": jnp.broadcast_to(d.rate, d.batch_shape)}


@from_mean_params.register(Poisson)
def _(d, params):
    return Poisson(params["x"])


@sufficient_statistics.register(Exponential)
def _(d, value):
    return {"x": _float(value)}


@mean_params.register(Exponential)
def _(d):
    return {"x": jnp.broadcast_to(1.0 / d.rate, d.batch_shape)}


@from_mean_params.register(Exponential)
def _(d, params):
    return Exponential(1.0 / params["x"])
