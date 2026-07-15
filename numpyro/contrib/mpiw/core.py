# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end massively parallel importance weighting (MPIW) on NumPyro models.

Ties together the sampled-enumeration messenger (which draws ``K`` samples per latent
site along named dimensions) and the contraction core (which reweights over all ``K**n``
combinations). Given a ``model`` and a ``guide`` whose latent sites are drawn ``K`` times
each, :class:`MPIW` estimates the log marginal likelihood ``log P_MP(x)`` and, via the
source-term trick, the self-normalized posterior importance weights of every latent site
-- from which posterior moments of any statistic follow.

The guide is assumed mean-field for now (each latent proposed independently of the
others); the model may have arbitrary static structure. See ``docs/design/qem_mpiw_plan.md``.
"""

from collections import OrderedDict
from typing import Any, Callable, NamedTuple

import jax
from jax import random
import jax.numpy as jnp

import funsor
import funsor.adjoint
from numpyro.contrib.funsor import enum, plate_to_enum_plate, trace as packed_trace
from numpyro.contrib.funsor.discrete import _get_support_value
from numpyro.contrib.mpiw.contraction import (
    contract_log_marginal,
    contract_with_source_terms,
)
from numpyro.handlers import infer_config, replay, seed, trace as orig_trace

funsor.set_backend("jax")


def _squeeze_fillers(arr, dim_to_name, event_dim=0):
    """Drop size-1 axes that are neither the site's K dimension nor a plate.

    The global dim allocator leaves a site's array with size-1 filler axes at the
    positions of *other* sites' K dimensions. These carry no information; squeezing
    them yields the natural ``(K, *plates[, *event])`` shape.

    ``dim_to_name`` positions index the *batch* region (event dims excluded, as in the
    trace), so with ``event_dim`` trailing event axes they map to batch axes
    ``batch_ndim + d``. Event axes are always preserved.
    """
    ndim = jnp.ndim(arr)
    batch_ndim = ndim - event_dim
    keep = {batch_ndim + d for d in dim_to_name}  # batch position -> axis index
    drop = tuple(j for j in range(batch_ndim) if j not in keep and arr.shape[j] == 1)
    return jnp.squeeze(arr, axis=drop) if drop else arr


def _num_samples_config(num_samples: int) -> Callable[[dict], dict]:
    """infer_config helper: mark every latent sample site for K-sample enumeration."""

    def config_fn(site: dict) -> dict:
        if site["type"] == "sample" and not site["is_observed"]:
            return {"enumerate": "parallel", "num_samples": num_samples}
        return {}

    return config_fn


class _SiteInfo(NamedTuple):
    """Precomputed per-site quantities, so source-term gradients need no re-tracing."""

    is_observed: bool
    model_log_prob: jax.Array
    model_dim_to_name: OrderedDict
    # latent sites only:
    guide_value: Any
    guide_log_prob: Any
    guide_dim_to_name: Any
    guide_event_dim: int


class _Prepared(NamedTuple):
    sites: "OrderedDict[str, _SiteInfo]"
    eliminate: frozenset  # latent site names == their K dimensions
    plates: frozenset
    log_num_samples: float


def _guess_max_plate_nesting(guide, rng_key, args, kwargs) -> int:
    """Largest plate nesting depth, from a plain (non-enumerated) trace of the guide.

    Uses the ordinary trace, not the funsor packed trace, so it does not touch the
    global DimStack (which would corrupt the subsequent enumerated traces).
    """
    tr = orig_trace(seed(guide, rng_key)).get_trace(*args, **kwargs)
    depth = 0
    for site in tr.values():
        if site["type"] == "sample":
            for frame in site["cond_indep_stack"]:
                if frame.dim is not None:
                    depth = max(depth, -frame.dim)
    return depth


class MPIW:
    """Massively parallel importance weighting for a NumPyro ``model``/``guide`` pair.

    :param model: a NumPyro model.
    :param guide: a mean-field guide over the model's latent sites (each site's
        distribution should be one whose ``K`` samples we reweight).
    :param int num_samples: number of samples ``K`` drawn per latent site.
    :param int max_plate_nesting: optional; inferred from the guide if omitted.
    """

    def __init__(self, model, guide, num_samples: int, max_plate_nesting=None):
        self.model = model
        self.guide = guide
        self.num_samples = num_samples
        self.max_plate_nesting = max_plate_nesting

    def _prepare(self, rng_key, *args, **kwargs) -> _Prepared:
        K = self.num_samples
        mpn = self.max_plate_nesting
        if mpn is None:
            rng_key, mpn_key = random.split(rng_key)
            mpn = _guess_max_plate_nesting(self.guide, mpn_key, args, kwargs)

        guide_key, model_key = random.split(rng_key)
        guide = infer_config(seed(self.guide, guide_key), _num_samples_config(K))
        first_dim = -mpn - 1
        with plate_to_enum_plate(), enum(first_available_dim=first_dim):
            guide_trace = packed_trace(guide).get_trace(*args, **kwargs)
            model = replay(seed(self.model, model_key), guide_trace)
            model_trace = packed_trace(model).get_trace(*args, **kwargs)

        sites: OrderedDict[str, _SiteInfo] = OrderedDict()
        eliminate = set()
        all_names = set()
        for name, msite in model_trace.items():
            if msite["type"] != "sample":
                continue
            m_d2n = msite["infer"]["dim_to_name"]
            m_lp = msite["fn"].log_prob(msite["value"])
            all_names.update(m_d2n.values())
            if msite["is_observed"] or name not in guide_trace:
                sites[name] = _SiteInfo(True, m_lp, m_d2n, None, None, None, 0)
            else:
                gsite = guide_trace[name]
                g_d2n = gsite["infer"]["dim_to_name"]
                g_lp = gsite["fn"].log_prob(gsite["value"])
                all_names.update(g_d2n.values())
                eliminate.add(name)
                sites[name] = _SiteInfo(
                    False,
                    m_lp,
                    m_d2n,
                    gsite["value"],
                    g_lp,
                    g_d2n,
                    gsite["fn"].event_dim,
                )

        plates = frozenset(all_names) - frozenset(eliminate)
        return _Prepared(sites, frozenset(eliminate), plates, float(jnp.log(K)))

    @staticmethod
    def _build_factors(prep: _Prepared, source_terms=None):
        """Assemble importance-weight funsor factors (optionally with source terms)."""
        factors = []
        for name, s in prep.sites.items():
            model_factor = funsor.to_funsor(
                s.model_log_prob, output=funsor.Real, dim_to_name=s.model_dim_to_name
            )
            if s.is_observed:
                factors.append(model_factor)
                continue
            # latent site: log p(z|parents) - log q(z) - log K  [ + source term ]
            guide_arr = s.guide_log_prob
            if source_terms is not None and name in source_terms:
                guide_arr = (
                    guide_arr - source_terms[name]
                )  # subtracted, so + into factor
            guide_factor = funsor.to_funsor(
                guide_arr, output=funsor.Real, dim_to_name=s.guide_dim_to_name
            )
            factors.append(model_factor - guide_factor - prep.log_num_samples)
        return factors, prep.eliminate, prep.plates

    def log_marginal(self, rng_key, *args, **kwargs) -> jax.Array:
        """Estimate ``log P_MP(x)``, an (unbiased-for-``P(x)``) marginal likelihood."""
        prep = self._prepare(rng_key, *args, **kwargs)
        factors, eliminate, plates = self._build_factors(prep)
        return contract_log_marginal(factors, eliminate, plates)

    def site_weights(self, rng_key, *args, **kwargs):
        """Return ``{site: (values, weights)}`` for each latent site.

        ``values`` are the guide's ``K`` samples (shape ``(K, *plates)``); ``weights``
        are the matching self-normalized posterior importance weights (same shape),
        summing to one over the ``K`` axis within each plate element.
        """
        prep = self._prepare(rng_key, *args, **kwargs)
        source_shapes = {
            name: jnp.shape(s.guide_log_prob)
            for name, s in prep.sites.items()
            if not s.is_observed
        }
        _, weights = contract_with_source_terms(
            lambda st: self._build_factors(prep, st), source_shapes
        )
        result = {}
        for name in source_shapes:
            s = prep.sites[name]
            d2n = s.guide_dim_to_name
            # the value carries trailing event axes; the source-term weight is batch-only
            value = _squeeze_fillers(s.guide_value, d2n, s.guide_event_dim)
            weight = _squeeze_fillers(weights[name], d2n, 0)
            result[name] = (value, weight)
        return result

    def moments(self, rng_key, statistics, *args, **kwargs):
        """Posterior moments of per-site ``statistics``.

        :param statistics: ``{site: fn}`` where ``fn(values)`` maps the site's ``K``
            samples to a statistic; the weighted sum over the ``K`` axis is returned.
        :returns: ``{site: weighted_statistic}`` for each requested site.
        """
        weights = self.site_weights(rng_key, *args, **kwargs)
        out = {}
        for name, fn in statistics.items():
            values, w = weights[name]
            stat = fn(values)
            # weight over the K axis (axis 0), broadcasting over any statistic shape
            w_b = w.reshape(w.shape + (1,) * (jnp.ndim(stat) - jnp.ndim(w)))
            out[name] = jnp.sum(w_b * stat, axis=0)
        return out

    def sample_posterior(self, rng_key, num_samples, *args, **kwargs):
        """Draw joint posterior samples of the latents.

        Uses forward-filter backward-sample over the contracted importance-weight
        factor graph (funsor adjoint under a Monte Carlo interpretation): each draw
        picks, for every latent site, one of its ``K`` samples with probability
        proportional to the massively parallel importance weights, respecting the
        graphical coupling between sites (so the joint -- not merely the marginals -- is
        sampled correctly).

        All draws come from a single set of ``K`` guide samples, so the sample resolves
        the true posterior only as well as that grid does (improving with ``K``).

        .. note:: Draws are currently produced by a Python loop of ``num_samples``
            sequential backward-sampling passes; drawing many samples is therefore
            slow. Vectorizing over draws is a future improvement.

        :param int num_samples: number of joint draws to return.
        :returns: ``{site: draws}`` where ``draws`` has shape
            ``(num_samples, *plates[, *event])``.
        """
        grid_key, draw_key = random.split(rng_key)
        prep = self._prepare(grid_key, *args, **kwargs)

        # build factors once; keep a handle to each latent's measure factor and a funsor
        # view of its K samples (K dimension + plates in, event as output) for gathering.
        factors = []
        measures = {}
        value_funsors = {}
        plate_name_to_dim = {}
        for name, s in prep.sites.items():
            model_factor = funsor.to_funsor(
                s.model_log_prob, output=funsor.Real, dim_to_name=s.model_dim_to_name
            )
            if s.is_observed:
                factors.append(model_factor)
                continue
            guide_factor = funsor.to_funsor(
                s.guide_log_prob, output=funsor.Real, dim_to_name=s.guide_dim_to_name
            )
            factor = model_factor - guide_factor - prep.log_num_samples
            factors.append(factor)
            measures[name] = factor
            event_shape = jnp.shape(s.guide_value)[
                jnp.ndim(s.guide_value) - s.guide_event_dim :
            ]
            output = funsor.Reals[event_shape] if s.guide_event_dim else funsor.Real
            value_funsors[name] = funsor.to_funsor(
                s.guide_value, output=output, dim_to_name=s.guide_dim_to_name
            )
            plate_name_to_dim[name] = OrderedDict(
                (nm, dm) for dm, nm in s.guide_dim_to_name.items() if nm != name
            )

        with funsor.interpretations.lazy:
            log_marginal = funsor.sum_product.sum_product(
                funsor.ops.logaddexp,
                funsor.ops.add,
                factors,
                eliminate=prep.eliminate | prep.plates,
                plates=prep.plates,
            )

        draws = {name: [] for name in measures}
        for key in random.split(draw_key, num_samples):
            with funsor.montecarlo.MonteCarlo(rng_key=key):
                adjoint_factors = funsor.adjoint.adjoint(
                    funsor.ops.logaddexp, funsor.ops.add, log_marginal
                )
            for name in measures:
                index = _get_support_value(adjoint_factors[measures[name]], name)
                sampled = value_funsors[name](**{name: index})
                draws[name].append(
                    funsor.to_data(sampled, name_to_dim=plate_name_to_dim[name])
                )
        return {name: jnp.stack(d) for name, d in draws.items()}
