# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the Find/Weigh/Learn procedure."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

from numpyro.infer.initialization import init_to_sample

MODE_SOURCES = ("restart", "perturb", "both")
CONTINUOUS_OBJECTIVES = ("joint", "marginal")
COVARIANCES = ("diagonal", "full", "graph")
FISHER_GRANULARITIES = ("site", "element")
ELIMINATIONS = ("joint", "nested")


@dataclass(frozen=True)
class FWLOptions:
    """
    Options for :func:`~numpyro.contrib.fwl.find_weigh_learn`.

    **Find** (Section 2 of the design document)

    :param num_modes: ``K``, the number of modes to locate, and hence the number
        of mixture components in the proposal.
    :param mode_source: how the ``K`` mode-finding runs are made to differ.
        ``"restart"`` gives each run a different starting point from
        ``init_strategy`` and takes the exact discrete argmax (max-product at
        temperature 0). ``"perturb"`` is perturb-and-MAP: the discrete sites are
        drawn by exact forward-filter-backward-sample (temperature 1) rather
        than maximized, and the continuous energy is tilted by a random linear
        term ``eps . u`` with ``eps ~ N(0, perturb_scale^2 I)``, so each run
        returns an approximate posterior sample rather than a local mode. (For a
        Gaussian target, tilting a quadratic energy displaces its minimizer by
        ``Sigma eps``, which is exactly a posterior draw when the tilt precision
        matches the Hessian.) ``"both"`` uses unperturbed runs for the first
        ``ceil(K/2)`` components and perturbed runs for the rest.
    :param perturb_scale: standard deviation of the continuous energy tilt used
        by ``mode_source in ("perturb", "both")``.
    :param init_strategy: per-site init function supplying the starting points.
        See :ref:`init_strategy`.
    :param solver: an :mod:`optimistix` minimiser for the continuous inner
        optimization. Defaults to ``optimistix.BFGS`` at the tolerances below.
    :param rtol: relative tolerance of the default solver.
    :param atol: absolute tolerance of the default solver.
    :param max_solver_steps: step cap for each continuous solve.
    :param max_sweeps: cap on block-coordinate sweeps between the discrete
        max-product pass and the continuous solve. Ignored when the model has no
        discrete latent sites, where one solve suffices.
    :param elimination: how the continuous latents are eliminated.
        ``"joint"`` runs one solve over the whole latent vector, treating it as a
        single clique. ``"nested"`` eliminates clique by clique over the junction
        tree, so each factor-to-separator message is an inner solve, as in
        Section 2. Nested elimination gets its message derivatives from the
        envelope theorem rather than by differentiating the inner solves, which is
        exact and much cheaper, but it cannot escape the cost of nesting itself:
        work grows like (solver steps) ** height of the tree. Only ancestor chains
        nest, so the tree is rooted at its center to minimize height. Requires
        ``continuous_objective="joint"``, since the discrete-marginalized energy
        does not factorize over cliques.
    :param max_nesting_depth: tree height above which ``elimination="nested"``
        refuses to run rather than hanging.
    :param continuous_objective: what the continuous solver minimizes.
        ``"joint"`` minimizes ``-log gamma_theta(z_discrete, z_continuous)`` with the
        discrete sites fixed at the current max-product configuration, i.e. the
        joint mode of Equation 3. ``"marginal"`` minimizes the
        discrete-marginalized energy ``-log sum_d gamma_theta(d, z_continuous)``,
        which is the mode of the density the proposal is actually defined on.

    **Weigh** (Section 3)

    :param covariance: the shape of each mixture component.
        ``"diagonal"`` and ``"full"`` use a damped empirical Fisher as the
        covariance. ``"graph"`` instead builds the component's *precision* as a
        sum of clique-supported blocks, one per clique of the junction tree, each
        a small dense Cholesky factor. Since ``g_v`` is supported on factor ``v``'s
        scope, ``sum_v g_v g_v^T`` already has exactly the moral graph's sparsity,
        so the blocks reproduce the damped empirical Fisher precisely at
        initialization while remaining positive definite and graph-structured
        under learning. ``"diagonal"`` and ``"full"`` are the two extremes of it.
    :param fisher_granularity: whether the outer products summed into the
        empirical Fisher are one per sample site (``"site"``, cheap: one
        Jacobian row per site) or one per element of each site's log-density
        (``"element"``, the finer and more standard empirical Fisher, costing a
        Jacobian row per observation).
    :param damping: ``lambda``, the damping added to the Fisher before inversion.

    **Learn** (Section 4)

    :param num_particles: ``P``, the number of importance samples drawn per call
        to the returned estimators.
    :param learnable: whether the proposal's locations and scales are
        :func:`numpyro.param` sites initialized at the Find/Weigh values (so a
        downstream optimizer can refine them), or frozen constants under
        :func:`jax.lax.stop_gradient`.
    :param learn_mixing_weights: whether the mixture weights are learnable too.
        Off by default: the design document specifies uniform ``1/K`` weights,
        and the reparameterized gradient of the returned estimators is unbiased
        for the component parameters but not for the weights.
    :param prefix: prefix for the guide's site names.
    :param allow_enumeration_only: whether to accept a model with no continuous
        latent sites instead of raising. Such a model needs no proposal:
        enumeration evaluates ``log Z(theta)`` exactly, so Find reduces to a
        single max-product pass, the guide is empty, and every importance weight
        equals ``log Z(theta)``. Both bounds are then tight rather than bounds.
    """

    num_modes: int = 4
    mode_source: str = "restart"
    perturb_scale: float = 1.0
    init_strategy: Callable = init_to_sample
    solver: Optional[Any] = None
    rtol: float = 1e-6
    atol: float = 1e-6
    max_solver_steps: int = 256
    max_sweeps: int = 8
    elimination: str = "joint"
    max_nesting_depth: int = 4
    continuous_objective: str = "joint"
    covariance: str = "diagonal"
    fisher_granularity: str = "site"
    damping: float = 1.0
    num_particles: int = 8
    learnable: bool = True
    learn_mixing_weights: bool = False
    prefix: str = "_fwl"
    allow_enumeration_only: bool = False

    def __post_init__(self) -> None:
        if self.num_modes < 1:
            raise ValueError(f"num_modes must be at least 1, got {self.num_modes}.")
        if self.num_particles < 1:
            raise ValueError(
                f"num_particles must be at least 1, got {self.num_particles}."
            )
        if self.damping <= 0:
            raise ValueError(
                "damping must be positive: it is what keeps the Fisher invertible, "
                f"got {self.damping}."
            )
        for field_name, allowed in (
            ("mode_source", MODE_SOURCES),
            ("continuous_objective", CONTINUOUS_OBJECTIVES),
            ("covariance", COVARIANCES),
            ("fisher_granularity", FISHER_GRANULARITIES),
            ("elimination", ELIMINATIONS),
        ):
            value = getattr(self, field_name)
            if value not in allowed:
                raise ValueError(
                    f"{field_name} must be one of {allowed}, got {value!r}."
                )
        if self.elimination == "nested" and self.continuous_objective != "joint":
            raise ValueError(
                "elimination='nested' requires continuous_objective='joint': the "
                "discrete-marginalized energy does not factorize over cliques, so "
                "there are no clique-wise messages to pass."
            )
        if self.max_nesting_depth < 0:
            raise ValueError(
                f"max_nesting_depth must be non-negative, got {self.max_nesting_depth}."
            )

    def minimiser(self) -> Any:
        """The optimistix minimiser to use, built on demand."""
        if self.solver is not None:
            return self.solver
        import optimistix as optx

        return optx.BFGS(rtol=self.rtol, atol=self.atol)
