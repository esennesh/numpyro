# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""The user-facing entry point of the Find/Weigh/Learn procedure."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional

import jax
from jax import Array

from numpyro.contrib.fwl.find import FindState, find_modes, replay, site_values
from numpyro.contrib.fwl.guide import guide_param_constraints, make_guide
from numpyro.contrib.fwl.junction import CliqueTree, build_clique_tree
from numpyro.contrib.fwl.learn import make_estimators
from numpyro.contrib.fwl.options import FWLOptions
from numpyro.contrib.fwl.structure import ModelStructure, analyze
from numpyro.contrib.fwl.weigh import weigh
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to


class _Static:
    """
    Holder for the non-array part of :class:`FWLResult`.

    Hashed by identity so that the pytree definition stays hashable, which is
    what lets an :class:`FWLResult` be passed through :func:`jax.jit`.
    """

    __slots__ = ("payload",)

    def __init__(self, payload: tuple):
        self.payload = payload

    __hash__ = object.__hash__

    def __eq__(self, other: object) -> bool:
        return self is other


@dataclass(frozen=True)
class FWLResult:
    """
    What :func:`find_weigh_learn` returns.

    Registered as a pytree whose leaves are ``modes``, ``log_joint``,
    ``converged``, ``sweeps`` and ``params``; the callables and the structural
    metadata ride along as static data.

    :param modes: the found configuration, as a dict of ``(K, ...)`` arrays with
        one entry per sample and deterministic site of the model: observed values
        at observed sites, found modes at unobserved ones, all in constrained
        space. A plain dict of arrays, so it passes through
        :func:`jax.vmap`, :func:`jax.jit` and :func:`jax.tree.map` unchanged.
    :param log_joint: ``(K,)`` unnormalized joint log-densities
        ``log gamma_theta(z*_k)``, in unconstrained space.
    :param converged: ``(K,)`` booleans, one per mode-finding run.
    :param sweeps: ``(K,)`` block-coordinate sweeps used per run.
    :param params: initial parameter values for the estimators below, holding the
        model's own :func:`numpyro.param` sites and the proposal's learnable
        locations and scales, in constrained space.
    :param guide: a NumPyro callable taking the model's args and kwargs,
        implementing the mixture proposal of Section 3.
    :param log_weights: ``(params, rng_key, *args, **kwargs) -> (P,)`` array of
        log importance weights.
    :param elbo: ``(params, rng_key, *args, **kwargs) -> ()``, the Jensen bound.
    :param iwae: ``(params, rng_key, *args, **kwargs) -> ()``, the tighter
        importance-weighted bound. Differentiate either with respect to
        ``params`` to get the Section 4 learning gradient.
    :param structure: the :class:`~numpyro.contrib.fwl.structure.ModelStructure`
        that was inferred; ``structure.summary()`` describes it.
    :param options: the :class:`~numpyro.contrib.fwl.options.FWLOptions` used.
    :param find_state: the raw unconstrained output of Find.
    :param tree: the junction tree over the continuous latents;
        ``tree.summary()`` describes the cliques, separators and factor
        assignment, and ``tree.height`` is the nesting depth that
        ``elimination="nested"`` pays for.
    """

    modes: dict[str, Array]
    log_joint: Array
    converged: Array
    sweeps: Array
    params: dict[str, Array]
    guide: Callable = field(compare=False)
    log_weights: Callable = field(compare=False)
    elbo: Callable = field(compare=False)
    iwae: Callable = field(compare=False)
    structure: ModelStructure = field(compare=False)
    options: FWLOptions = field(compare=False)
    find_state: FindState = field(compare=False)
    tree: CliqueTree = field(compare=False)
    param_constraints: dict[str, Any] = field(compare=False)

    @property
    def num_modes(self) -> int:
        return int(self.log_joint.shape[0])

    def param_transforms(self) -> dict[str, Any]:
        """Bijections from unconstrained space to each parameter's support."""
        return {
            name: biject_to(constraint)
            for name, constraint in self.param_constraints.items()
        }

    def unconstrain(
        self, params: Optional[dict[str, Array]] = None
    ) -> dict[str, Array]:
        """
        Map a constrained parameter dict to unconstrained space, for handing to
        an unconstrained optimizer such as :mod:`optax`.
        """
        params = self.params if params is None else params
        transforms = self.param_transforms()
        return {name: transforms[name].inv(value) for name, value in params.items()}

    def constrain(self, unconstrained: dict[str, Array]) -> dict[str, Array]:
        """Inverse of :meth:`unconstrain`; the estimators take constrained params."""
        transforms = self.param_transforms()
        return {name: transforms[name](value) for name, value in unconstrained.items()}


def _flatten(result: FWLResult):
    children = (
        result.modes,
        result.log_joint,
        result.converged,
        result.sweeps,
        result.params,
    )
    static = _Static(
        (
            result.guide,
            result.log_weights,
            result.elbo,
            result.iwae,
            result.structure,
            result.options,
            result.find_state,
            result.tree,
            result.param_constraints,
        )
    )
    return children, static


def _unflatten(static: _Static, children: tuple) -> FWLResult:
    modes, log_joint, converged, sweeps, params = children
    (
        guide,
        log_weights,
        elbo,
        iwae,
        structure,
        options,
        find_state,
        tree,
        param_constraints,
    ) = static.payload
    return FWLResult(
        modes=modes,
        log_joint=log_joint,
        converged=converged,
        sweeps=sweeps,
        params=params,
        guide=guide,
        log_weights=log_weights,
        elbo=elbo,
        iwae=iwae,
        structure=structure,
        options=options,
        find_state=find_state,
        tree=tree,
        param_constraints=param_constraints,
    )


jax.tree_util.register_pytree_node(FWLResult, _flatten, _unflatten)


def find_weigh_learn(
    model: Callable,
    rng_key: Array,
    model_args: tuple = (),
    model_kwargs: Optional[dict] = None,
    *,
    options: Optional[FWLOptions] = None,
    **option_overrides: Any,
) -> FWLResult:
    """
    Find joint modes of a model's posterior, build a mixture proposal around
    them, and return the machinery needed to learn the model's parameters.

    This is the procedure of *Find, Weigh, Learn: Fast MAP Estimation in
    Graphical Models*:

    1. **Find** (Section 2): locate ``K`` modes of ``gamma_theta``, eliminating
       the enumerable discrete latent sites exactly by max-product message
       passing and the continuous ones by :mod:`optimistix`, alternating between
       the two until the discrete configuration stops changing.
    2. **Weigh** (Section 3): give each mode a Gaussian whose covariance is the
       damped local empirical Fisher, and mix them uniformly.
    3. **Learn** (Section 4): expose the log importance weights under that
       proposal, and the ELBO and IWAE bounds on ``log Z(theta)`` built from
       them, as differentiable functions of the parameters.

    Example::

        result = find_weigh_learn(model, rng_key, (data,), num_modes=8)
        print(result.structure.summary())
        print(result.modes["z"].shape)         # (8, ...) found modes

        params = result.unconstrain()          # for an unconstrained optimizer
        loss = lambda p, key: -result.iwae(result.constrain(p), key)
        grads = jax.grad(loss)(params, rng_key)

    Find itself runs eagerly on the host, since funsor's adjoint pass and the
    sweep termination test are Python-level; everything it returns is traceable.

    A model with no continuous latent sites needs no proposal at all, and raises
    by default. Passing ``allow_enumeration_only=True`` returns the degenerate
    result instead: Find is a single max-product pass, the guide is empty, and
    ``log_weights`` returns ``log Z(theta)`` exactly, so ``elbo`` and ``iwae``
    coincide with it rather than bounding it.

    :param model: a NumPyro model.
    :param rng_key: PRNG key for the initializations, perturbations and any
        temperature-1 discrete draws.
    :param model_args: args for the model.
    :param model_kwargs: kwargs for the model.
    :param options: an :class:`~numpyro.contrib.fwl.options.FWLOptions`. Any
        further keyword arguments override its fields, so
        ``find_weigh_learn(model, key, num_modes=8)`` works directly.
    :return: an :class:`FWLResult`.
    """
    model_kwargs = {} if model_kwargs is None else model_kwargs
    options = FWLOptions() if options is None else options
    if option_overrides:
        options = dataclasses.replace(options, **option_overrides)

    structure_key, find_key, weigh_key, estimator_key = jax.random.split(rng_key, 4)
    structure = analyze(
        model,
        structure_key,
        model_args,
        model_kwargs,
        options.init_strategy,
        options.allow_enumeration_only,
    )
    tree = build_clique_tree(structure)
    find_state = find_modes(
        model, find_key, model_args, model_kwargs, structure, options, tree
    )
    scales = weigh(
        model,
        model_args,
        model_kwargs,
        structure,
        find_state,
        options,
        weigh_key,
        tree,
    )

    guide, guide_params = make_guide(
        structure, find_state.latent, scales, options, tree
    )
    log_weights, elbo, iwae = make_estimators(
        model, structure, guide, options, model_args, model_kwargs, estimator_key
    )

    def modes_of(latent: Array, discrete: dict[str, Array]) -> dict[str, Array]:
        return site_values(
            replay(
                model,
                model_args,
                model_kwargs,
                structure,
                latent,
                discrete,
                estimator_key,
            )
        )

    modes = jax.vmap(modes_of)(find_state.latent, find_state.discrete)

    model_params = {
        name: site["value"]
        for name, site in structure.prototype_trace.items()
        if site["type"] == "param"
    }
    param_constraints = {
        name: structure.prototype_trace[name]["kwargs"].get(
            "constraint", constraints.real
        )
        for name in model_params
    }
    param_constraints.update(guide_param_constraints(options, len(tree.cliques)))
    params = {**model_params, **guide_params}
    param_constraints = {
        name: constraint
        for name, constraint in param_constraints.items()
        if name in params
    }

    return FWLResult(
        modes=modes,
        log_joint=find_state.log_joint,
        converged=find_state.converged,
        sweeps=find_state.sweeps,
        params=params,
        guide=guide,
        log_weights=log_weights,
        elbo=elbo,
        iwae=iwae,
        structure=structure,
        options=options,
        find_state=find_state,
        tree=tree,
        param_constraints=param_constraints,
    )
