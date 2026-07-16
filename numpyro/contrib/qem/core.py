# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
QEM: expectation maximization for exponential-family approximate posteriors, with
the E-step computed by massively parallel importance weighting (Heap, Bowyer &
Aitchison, AABI 2025; Bowyer et al. 2024).

Each iteration: (M) rebuild the guide's site distributions from the current
per-site *mean parameters* by moment matching; (E) draw ``K`` proposals per site,
contract the importance weights over all ``K**n`` combinations
(:class:`~numpyro.contrib.mpiw.MPIW`), and estimate each site's posterior mean
parameters as the weighted average of its sufficient statistics; then blend the
estimates into the state with an exponential moving average. Mean parameters are
the *only* state QEM carries (the EMA must be over mean parameters -- paper
App. B), so convergence is invariant to how each family is parameterized
(paper Thm 2). No gradients with respect to guide parameters are ever taken.

See ``docs/design/qem_mpiw_plan.md``.
"""

from collections import namedtuple

import tqdm

from jax import random
import jax.numpy as jnp

from numpyro import handlers
from numpyro.contrib.mpiw import MPIW
from numpyro.distributions.exp_family import (
    base_distribution,
    is_exp_family,
    mean_params,
    sufficient_statistics,
)

QEMState = namedtuple("QEMState", ["mean_params", "step", "rng_key"])
"""
State of the :class:`QEM` driver.

- **mean_params** - ``{site: mean-parameter pytree}`` for each latent site.
- **step** - number of updates taken.
- **rng_key** - random key carried across updates.
"""

QEMRunResult = namedtuple("QEMRunResult", ["params", "state", "log_marginals"])
"""
Result of :meth:`QEM.run`.

- **params** - guide parameter values (as :meth:`QEM.get_params`), usable with
  :class:`~numpyro.infer.Predictive` or the guide's ``sample_posterior``.
- **state** - the final :class:`QEMState`.
- **log_marginals** - per-iteration ``log P_MP(x)`` estimates.
"""


class QEM:
    """Gradient-free variational fitting by moment-matched EM on MPIW weights.

    Mirrors the :class:`~numpyro.infer.svi.SVI` API surface
    (``init`` / ``update`` / ``run`` / ``get_params`` / ``evaluate``) but carries no
    optimizer: the state is each latent site's mean-parameter pytree.

    The guide must expose per-site base distributions recognized by the
    exponential-family registry (:mod:`numpyro.distributions.exp_family`) and a
    ``params_from_mean`` method mapping mean parameters to guide parameter values
    -- :class:`~numpyro.infer.autoguide.AutoExponentialFamily` provides both and
    is the intended default.

    :param model: a NumPyro model with static control flow.
    :param guide: an :class:`~numpyro.infer.autoguide.AutoExponentialFamily` (or
        compatible) mean-field guide over the model's latent sites.
    :param int num_samples: number of proposals ``K`` drawn per latent site.
    :param forget: EMA forgetting factor ``lambda`` in
        ``m_t = lambda * m_{t-1} + (1 - lambda) * m_hat_t``: a float for a fixed
        factor, or a callable ``t -> lambda_t``. Defaults to the paper's Theorem 1
        schedule ``lambda(t) = 1 - t**(-schedule_power)`` (so the first update
        takes the raw estimate in full).
    :param float schedule_power: the power ``p`` of the default schedule; the
        paper's convergence guarantee needs ``0.5 < p <= 1``. Ignored when
        ``forget`` is given.
    :param bool decorrelated_normalizer: normalize the E-step weights by
        ``P_MP`` of a second, *independent* batch of guide samples (paper section
        4) instead of the same-batch ``P_MP``. This removes the
        numerator/denominator covariance term of the self-normalized moment
        estimator's finite-``K`` ratio bias (the ``Var[P_MP]``-driven term
        survives, by Jensen), at the cost of one extra guide trace + normalizer
        contraction per step and of weights no longer summing to exactly one --
        so per-step mean-parameter estimates gain variance and can momentarily
        leave the mean domain (relevant for discrete families); the EMA damps
        this. Defaults to False (self-normalized).
    :param serial_sites: latent sites whose ``K`` dimensions are contracted
        serially (memory-frugal path); see :class:`~numpyro.contrib.mpiw.MPIW`.
    :param int max_plate_nesting: optional; inferred from the guide if omitted.
    """

    def __init__(
        self,
        model,
        guide,
        num_samples,
        forget=None,
        schedule_power=1.0,
        decorrelated_normalizer=False,
        serial_sites=(),
        max_plate_nesting=None,
    ):
        self.model = model
        self.guide = guide
        self.num_samples = num_samples
        if forget is None:
            if not 0.5 < schedule_power <= 1.0:
                raise ValueError(
                    f"schedule_power must be in (0.5, 1], got {schedule_power}"
                )
            self._forget = lambda t: 1.0 - float(t) ** -schedule_power
        elif callable(forget):
            self._forget = forget
        else:
            if not 0.0 <= forget < 1.0:
                raise ValueError(f"forget must be in [0, 1), got {forget}")
            self._forget = lambda t: forget
        self.decorrelated_normalizer = decorrelated_normalizer
        self.serial_sites = tuple(serial_sites)
        self.max_plate_nesting = max_plate_nesting

    def init(self, rng_key, *args, **kwargs):
        """Set up the guide and return the initial :class:`QEMState`.

        Mean parameters are initialized by moment-matching each site's prior as
        recorded in the guide's prototype trace (i.e. with parent sites at their
        ``init_loc_fn`` values).
        """
        rng_key, guide_key = random.split(rng_key)
        # trigger the guide's prototype setup (a no-op if already set up)
        handlers.trace(handlers.seed(self.guide, guide_key)).get_trace(*args, **kwargs)

        init_mean = {}
        for name, site in self.guide.prototype_trace.items():
            if site["type"] != "sample" or site["is_observed"]:
                continue
            if not is_exp_family(site["fn"]):
                raise ValueError(
                    f"QEM requires exponential-family priors/proposals; site "
                    f"'{name}' has {type(base_distribution(site['fn'])).__name__}, "
                    "which has no registration in numpyro.distributions.exp_family."
                )
            init_mean[name] = mean_params(site["fn"])
        return QEMState(init_mean, 0, rng_key)

    def get_params(self, state):
        """Guide parameter values for ``state``'s mean parameters (M-step output)."""
        return self.guide.params_from_mean(state.mean_params)

    def _mpiw(self, state):
        guide = handlers.substitute(self.guide, data=self.get_params(state))
        return MPIW(self.model, guide, self.num_samples, self.max_plate_nesting)

    def update(self, state, *args, **kwargs):
        """Take a single QEM step (M-step, E-step, EMA blend).

        :return: ``(new_state, log_marginal)`` where ``log_marginal`` is this
            iteration's ``log P_MP(x)`` estimate (higher is better; its trace is
            the convergence diagnostic).
        """
        rng_key, step_key = random.split(state.rng_key)
        mpiw = self._mpiw(state)
        log_marginal, site_weights = mpiw.log_marginal_and_site_weights(
            step_key, *args, serial_sites=self.serial_sites, **kwargs
        )
        if self.decorrelated_normalizer:
            # source-term weights are normalized by the same-batch P_MP; put it
            # back and divide out the P_MP of an independent batch of proposals
            rng_key, normalizer_key = random.split(rng_key)
            log_normalizer = mpiw.log_marginal(
                normalizer_key, *args, serial_sites=self.serial_sites, **kwargs
            )
            rescale = jnp.exp(log_marginal - log_normalizer)
            site_weights = {
                name: (values, weights * rescale)
                for name, (values, weights) in site_weights.items()
            }

        t = state.step + 1
        lam = self._forget(t)
        new_mean = {}
        for name, m in state.mean_params.items():
            values, weights = site_weights[name]
            base = base_distribution(self.guide.prototype_trace[name]["fn"])
            stats = sufficient_statistics(base, values)
            m_hat = {
                k: jnp.sum(
                    weights.reshape(
                        jnp.shape(weights) + (1,) * (jnp.ndim(s) - jnp.ndim(weights))
                    )
                    * s,
                    axis=0,
                )
                for k, s in stats.items()
            }
            new_mean[name] = {k: lam * m[k] + (1.0 - lam) * m_hat[k] for k in m}
        return QEMState(new_mean, t, rng_key), log_marginal

    def evaluate(self, state, *args, **kwargs):
        """Estimate ``log P_MP(x)`` at ``state`` without updating it.

        Uses (and does not advance) ``state.rng_key``.
        """
        _, eval_key = random.split(state.rng_key)
        return self._mpiw(state).log_marginal(
            eval_key, *args, serial_sites=self.serial_sites, **kwargs
        )

    def run(self, rng_key, num_steps, *args, progress_bar=True, **kwargs):
        """Run ``num_steps`` QEM iterations from a fresh init.

        :return: a :class:`QEMRunResult` with the final guide params, state, and
            the per-iteration ``log P_MP(x)`` trace.
        """
        state = self.init(rng_key, *args, **kwargs)
        log_marginals = []
        with tqdm.trange(1, num_steps + 1, disable=not progress_bar) as steps:
            for _ in steps:
                state, log_marginal = self.update(state, *args, **kwargs)
                log_marginals.append(log_marginal)
                if progress_bar:
                    steps.set_description(
                        f"log P_MP = {float(log_marginal):.4f}", refresh=False
                    )
        return QEMRunResult(self.get_params(state), state, jnp.stack(log_marginals))
