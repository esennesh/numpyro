# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Find: locating joint modes of the posterior (Section 2 of the design document).

The discrete latent sites are eliminated exactly by max-product message passing
over the model's implicit junction tree, which funsor already provides:
:func:`numpyro.contrib.funsor.discrete._sample_posterior` at ``temperature=0``
runs ``(max, add)`` variable elimination and recovers the argmax configuration by
funsor's adjoint pass, with plate dimensions vectorized and the elimination order
chosen by ``funsor.optimizer``. At ``temperature=1`` the same machinery performs
exact forward-filter-backward-sample, which is what the perturb-and-MAP mode
source uses in place of the discrete argmax.

Evaluating those messages needs values for the continuous variables appearing in
the same factors, and solving for the continuous variables needs a discrete
configuration. Running a continuous solve per discrete configuration would
reinstate the exponential cost the junction tree exists to avoid, so Find is a
block-coordinate descent: each sweep takes an exact discrete argmax with the
continuous variables held fixed, then a continuous solve with the discrete
configuration held fixed. Both half-steps decrease the same energy, and the
sweep terminates when the discrete configuration stops changing.

The continuous half-step comes in two forms, chosen by ``options.elimination``.
``"joint"`` treats the whole latent vector as one clique and runs a single solve.
``"nested"`` eliminates clique by clique over the junction tree, so each
factor-to-separator message is itself an inner solve, as Section 2 describes; see
:func:`_nested_solve` for why the envelope theorem makes that affordable and what
it still costs.

Find runs eagerly on the host: the funsor adjoint pass and the sweep termination
test are Python-level. The arrays it returns, and everything built from them in
:mod:`~numpyro.contrib.fwl.guide` and :mod:`~numpyro.contrib.fwl.learn`, are
fully traceable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

from jax import Array, lax, random
import jax.numpy as jnp

from numpyro import handlers
from numpyro.contrib.fwl.junction import CliqueTree
from numpyro.contrib.fwl.options import FWLOptions
from numpyro.contrib.fwl.structure import (
    LatentPacking,
    ModelStructure,
    _is_factor_site,
    initial_values,
)
from numpyro.distributions.util import is_identically_one
from numpyro.infer.util import _unconstrain_reparam, potential_energy


@dataclass(frozen=True)
class FindState:
    """
    The modes located by :func:`find_modes`.

    :param latent: ``(K, D)`` unconstrained continuous modes.
    :param discrete: dict of ``(K, ...)`` discrete configurations, one entry per
        discrete latent site.
    :param log_joint: ``(K,)`` values of ``log gamma_theta(z*_k)``, the
        unnormalized joint log-density at each mode. Computed in unconstrained
        space, so it includes the transform's log-Jacobian.
    :param converged: ``(K,)`` booleans; ``True`` when the final continuous solve
        reported success and the discrete configuration stopped changing.
    :param sweeps: ``(K,)`` number of block-coordinate sweeps used.
    """

    latent: Array
    discrete: dict[str, Array]
    log_joint: Array
    converged: Array
    sweeps: Array


def replay(
    model: Callable,
    model_args: tuple,
    model_kwargs: dict,
    structure: ModelStructure,
    latent: Array,
    discrete: dict[str, Array],
    seed_key: Array,
) -> dict[str, Any]:
    """
    Trace the model with the continuous latents set from an unconstrained vector
    and the discrete latents set to given values.

    The constrained values are produced by the transforms *inside* the model
    execution, so supports that depend on other latent values are handled
    correctly.
    """
    unconstrained = structure.packing.unpack(latent)
    substituted = handlers.substitute(
        handlers.substitute(handlers.seed(model, seed_key), data=discrete),
        substitute_fn=partial(_unconstrain_reparam, unconstrained),
    )
    return handlers.trace(substituted).get_trace(*model_args, **model_kwargs)


def site_values(trace: dict[str, Any]) -> dict[str, Array]:
    """Constrained values of every sample and deterministic site in a trace."""
    return {
        name: site["value"]
        for name, site in trace.items()
        if (site["type"] == "sample" and not _is_factor_site(site))
        or site["type"] == "deterministic"
    }


def _joint_energy_fn(
    model: Callable,
    model_args: tuple,
    model_kwargs: dict,
    structure: ModelStructure,
    seed_key: Array,
) -> Callable:
    """``-log gamma_theta(d, u)`` in unconstrained continuous space, ``d`` fixed."""

    def energy(latent: Array, discrete: dict[str, Array]) -> Array:
        substituted = handlers.substitute(handlers.seed(model, seed_key), data=discrete)
        return potential_energy(
            substituted, model_args, model_kwargs, structure.packing.unpack(latent)
        )

    return energy


def _marginal_energy_fn(
    model: Callable,
    model_args: tuple,
    model_kwargs: dict,
    structure: ModelStructure,
    seed_key: Array,
) -> Callable:
    """``-log sum_d gamma_theta(d, u)``, the discrete-marginalized energy."""
    if not structure.has_discrete:
        return _joint_energy_fn(model, model_args, model_kwargs, structure, seed_key)

    from numpyro.contrib.funsor import config_enumerate, enum

    enum_model = enum(
        config_enumerate(handlers.seed(model, seed_key)),
        first_available_dim=structure.first_available_dim,
    )

    def energy(latent: Array, discrete: Optional[dict[str, Array]] = None) -> Array:
        del discrete  # marginalized out rather than conditioned on
        return potential_energy(
            enum_model,
            model_args,
            model_kwargs,
            structure.packing.unpack(latent),
            enum=True,
        )

    return energy


def _discrete_argmax(
    model: Callable,
    model_args: tuple,
    model_kwargs: dict,
    structure: ModelStructure,
    continuous: dict[str, Array],
    temperature: int,
    rng_key: Array,
    seed_key: Array,
) -> dict[str, Array]:
    """
    One exact max-product pass (``temperature=0``) or forward-filter
    backward-sample pass (``temperature=1``) over the discrete latent sites,
    conditioned on constrained continuous values.
    """
    from numpyro.contrib.funsor import config_enumerate
    from numpyro.contrib.funsor.discrete import _sample_posterior

    substituted = handlers.substitute(
        config_enumerate(handlers.seed(model, seed_key)), data=continuous
    )
    values = _sample_posterior(
        substituted,
        structure.first_available_dim,
        temperature,
        rng_key,
        *model_args,
        **model_kwargs,
    )
    return {name: values[name] for name in structure.discrete}


def index_map(packing: LatentPacking) -> dict[str, tuple[int, ...]]:
    """Flat-vector indices occupied by each continuous site."""
    result, offset = {}, 0
    for name, size in zip(packing.names, packing.sizes):
        result[name] = tuple(range(offset, offset + size))
        offset += size
    return result


def _indices_of(names: frozenset[str], index: dict[str, tuple[int, ...]]) -> Array:
    return jnp.asarray(
        sorted(i for name in names for i in index[name]), dtype=jnp.int32
    )


def _clique_energy_fn(
    model: Callable,
    model_args: tuple,
    model_kwargs: dict,
    structure: ModelStructure,
    tree: CliqueTree,
    clique: int,
    discrete: dict[str, Array],
    seed_key: Array,
) -> Callable[[Array], Array]:
    """
    The energy of the factors assigned to one clique, as a function of the whole
    flat latent vector.

    Running the model evaluates every factor, but the assigned factors' scopes lie
    inside the clique by construction, so the value depends only on the clique's
    coordinates and XLA eliminates the rest as dead code. That is what makes the
    runtime cost of a clique-local solve genuinely clique-local, even though the
    Python-level trace walks the whole program.
    """
    assigned = frozenset(tree.factors[clique])
    # The Jacobian factors that ``_unconstrain_reparam`` emits belong with the
    # site whose support transform produced them.
    owner = {f"_{name}_log_det": name for name in structure.continuous}

    def energy(latent: Array) -> Array:
        trace = replay(
            model, model_args, model_kwargs, structure, latent, discrete, seed_key
        )
        total = jnp.zeros((), dtype=latent.dtype)
        for name, site in trace.items():
            if site["type"] != "sample":
                continue
            if owner.get(name, name) not in assigned:
                continue
            value = site["value"]
            intermediates = site["intermediates"]
            if intermediates:
                log_prob = site["fn"].log_prob(value, intermediates)
            else:
                log_prob = site["fn"].log_prob(value)
            scale = site["scale"]
            if (scale is not None) and (not is_identically_one(scale)):
                log_prob = scale * log_prob
            total = total - jnp.sum(jnp.asarray(log_prob, dtype=latent.dtype))
        return total

    return energy


def _nested_solve(
    model: Callable,
    model_args: tuple,
    model_kwargs: dict,
    structure: ModelStructure,
    tree: CliqueTree,
    discrete: dict[str, Array],
    latent: Array,
    tilt: Array,
    options: FWLOptions,
    seed_key: Array,
) -> tuple[Array, Array]:
    """
    Minimize the energy by nested clique-wise elimination: min-sum over the
    junction tree, with each factor-to-separator message evaluated by an inner
    :mod:`optimistix` solve.

    The inner solutions are wrapped in :func:`jax.lax.stop_gradient`, so the
    derivative of a message with respect to its separator is the *partial*
    derivative of the clique objective at the inner optimum. That is exact by the
    envelope theorem -- at the inner optimum the derivative through the inner
    argmin is multiplied by a vanishing gradient -- and it avoids differentiating
    the inner solve at all. Implicit differentiation would give the same numbers
    at sharply higher cost: measured compile time grows about 8x per level of
    nesting with ``ImplicitAdjoint`` against about 2.5x here.

    What this does *not* avoid is the cost of nesting itself. Evaluating a message
    at depth ``d`` triggers an inner solve per outer step, so work grows like
    (solver steps) ** height. Only ancestor chains nest -- sibling subtrees are
    additive -- so the exponent is the rooted tree's height, which is why the tree
    is rooted at its center. ``options.max_nesting_depth`` is the guard.

    .. note:: The envelope identity holds at an *exact* inner optimum, so the
        inner solves' residual error propagates outward and accumulates with
        depth. In float32 this is roundoff-limited rather than tolerance-limited:
        on a linear-Gaussian chain, tightening ``rtol``/``atol`` from 1e-6 to
        1e-10 does not move the error, which reaches about 2e-3 at height 4.
        Under ``jax_enable_x64`` the same problem lands within 5e-12 -- better
        than the single joint solve's 6e-9, since each clique-local solve is
        small and better conditioned. Prefer x64 for deep trees.
    """
    import optimistix as optx

    if tree.height > options.max_nesting_depth:
        raise ValueError(
            f"The junction tree has height {tree.height}, above max_nesting_depth="
            f"{options.max_nesting_depth}. Nested elimination costs "
            "(solver steps) ** height, so this would likely not finish. Use "
            "elimination='joint' for a single solve over all continuous latents, "
            "or raise max_nesting_depth if you know the solves are cheap."
        )

    index = index_map(structure.packing)
    interior = [_indices_of(names, index) for names in tree.interior]
    separator = [_indices_of(names, index) for names in tree.separator]
    base = lax.stop_gradient(latent)
    solver = options.minimiser()
    values: dict[int, Callable] = {}
    argmins: dict[int, Callable] = {}

    def build(clique: int) -> None:
        for child in tree.children[clique]:
            build(child)
        clique_energy = _clique_energy_fn(
            model,
            model_args,
            model_kwargs,
            structure,
            tree,
            clique,
            discrete,
            seed_key,
        )
        own_interior, own_separator = interior[clique], separator[clique]
        children = tree.children[clique]

        def objective(interior_value: Array, separator_value: Array) -> Array:
            full = base.at[own_separator].set(separator_value)
            full = full.at[own_interior].set(interior_value)
            total = clique_energy(full)
            # Interiors partition the latents, so tilting each clique on its own
            # interior tilts every coordinate exactly once.
            total = total + jnp.sum(tilt[own_interior] * interior_value)
            for child in children:
                total = total + values[child](full[separator[child]])
            return total

        def argmin(separator_value: Array) -> tuple[Array, Array]:
            solution = optx.minimise(
                lambda y, args: objective(y, args),
                solver,
                base[own_interior],
                args=lax.stop_gradient(separator_value),
                max_steps=options.max_solver_steps,
                throw=False,
            )
            return solution.value, solution.result == optx.RESULTS.successful

        def value(separator_value: Array) -> Array:
            frozen = lax.stop_gradient(argmin(lax.stop_gradient(separator_value))[0])
            return objective(frozen, separator_value)

        values[clique], argmins[clique] = value, argmin

    build(tree.root)

    empty = jnp.zeros((0,), dtype=latent.dtype)
    result = base
    root_interior, converged = argmins[tree.root](empty)
    result = result.at[interior[tree.root]].set(root_interior)
    stack = list(tree.children[tree.root])
    while stack:  # descend, recovering each clique's interior at its separator
        clique = stack.pop()
        clique_interior, ok = argmins[clique](result[separator[clique]])
        result = result.at[interior[clique]].set(clique_interior)
        converged = jnp.logical_and(converged, ok)
        stack.extend(tree.children[clique])
    return result, converged


def _solve_continuous(
    energy: Callable,
    latent: Array,
    tilt: Array,
    options: FWLOptions,
) -> tuple[Array, Array]:
    """Minimize a (possibly tilted) energy over the flat unconstrained vector."""
    import optimistix as optx

    def objective(y: Array, args: Any) -> Array:
        del args
        return energy(y) + jnp.sum(tilt * y)

    solution = optx.minimise(
        objective,
        options.minimiser(),
        latent,
        max_steps=options.max_solver_steps,
        throw=False,
    )
    return solution.value, solution.result == optx.RESULTS.successful


def _perturbation_flags(num_modes: int, mode_source: str) -> tuple[bool, ...]:
    if mode_source == "restart":
        return (False,) * num_modes
    if mode_source == "perturb":
        return (True,) * num_modes
    num_plain = (num_modes + 1) // 2  # "both": modes first, then samples
    return (False,) * num_plain + (True,) * (num_modes - num_plain)


def find_modes(
    model: Callable,
    rng_key: Array,
    model_args: tuple,
    model_kwargs: dict,
    structure: ModelStructure,
    options: FWLOptions,
    tree: Optional[CliqueTree] = None,
) -> FindState:
    """
    Locate ``options.num_modes`` modes of the unnormalized joint density.

    Each run is independent: it starts from its own draw from
    ``options.init_strategy`` and, when perturbed, carries its own frozen
    perturbation. Freezing the perturbation for the whole run (rather than
    redrawing it each sweep) is what makes a perturbed run a deterministic
    fixed-point iteration that can terminate.

    :param tree: the junction tree, required when
        ``options.elimination == "nested"``.
    """
    packing = structure.packing
    seed_key, run_key = random.split(rng_key)
    joint_energy = _joint_energy_fn(
        model, model_args, model_kwargs, structure, seed_key
    )
    objective_energy = (
        joint_energy
        if options.continuous_objective == "joint"
        else _marginal_energy_fn(model, model_args, model_kwargs, structure, seed_key)
    )
    perturbed_run = _perturbation_flags(options.num_modes, options.mode_source)
    nested = options.elimination == "nested"
    if nested and tree is None:
        raise ValueError("elimination='nested' requires a junction tree.")

    def solve(
        latent: Array, discrete: dict[str, Array], tilt: Array
    ) -> tuple[Array, Array]:
        if nested:
            assert tree is not None
            return _nested_solve(
                model,
                model_args,
                model_kwargs,
                structure,
                tree,
                discrete,
                latent,
                tilt,
                options,
                seed_key,
            )
        return _solve_continuous(
            partial(objective_energy, discrete=discrete), latent, tilt, options
        )

    latents, discretes, log_joints, converged, sweeps = [], [], [], [], []
    for k, key in enumerate(random.split(run_key, options.num_modes)):
        init_key, tilt_key, discrete_key = random.split(key, 3)
        latent, discrete = initial_values(
            model, init_key, model_args, model_kwargs, structure, options.init_strategy
        )
        perturbed = perturbed_run[k]
        tilt = (
            options.perturb_scale * random.normal(tilt_key, (packing.dim,))
            if perturbed
            else jnp.zeros((packing.dim,))
        )
        temperature = 1 if perturbed else 0

        if structure.enumeration_only:
            # Nothing to optimize, so one exact max-product pass is the whole of
            # Find; there is no continuous half for the sweeps to alternate with.
            discrete = _discrete_argmax(
                model,
                model_args,
                model_kwargs,
                structure,
                {},
                temperature,
                discrete_key,
                seed_key,
            )
            solver_ok, discrete_stable, num_sweeps = jnp.bool_(True), True, 0
        elif not structure.has_discrete:
            latent, solver_ok = solve(latent, discrete, tilt)
            num_sweeps, discrete_stable = 1, True
        else:
            solver_ok, discrete_stable, num_sweeps = jnp.bool_(False), False, 0
            for sweep in range(options.max_sweeps):
                trace = replay(
                    model,
                    model_args,
                    model_kwargs,
                    structure,
                    latent,
                    discrete,
                    seed_key,
                )
                continuous = {
                    name: trace[name]["value"] for name in structure.continuous
                }
                proposal = _discrete_argmax(
                    model,
                    model_args,
                    model_kwargs,
                    structure,
                    continuous,
                    temperature,
                    discrete_key,
                    seed_key,
                )
                if sweep > 0 and all(
                    bool(jnp.array_equal(proposal[name], discrete[name]))
                    for name in structure.discrete
                ):
                    discrete_stable = True
                    break
                discrete = proposal
                latent, solver_ok = _solve_continuous(
                    partial(objective_energy, discrete=discrete), latent, tilt, options
                )
                num_sweeps = sweep + 1

        latents.append(latent)
        discretes.append(discrete)
        log_joints.append(-joint_energy(latent, discrete))
        converged.append(jnp.logical_and(solver_ok, discrete_stable))
        sweeps.append(num_sweeps)

    return FindState(
        latent=jnp.stack(latents),
        discrete={
            name: jnp.stack([d[name] for d in discretes]) for name in structure.discrete
        },
        log_joint=jnp.stack(log_joints),
        converged=jnp.stack(converged),
        sweeps=jnp.asarray(sweeps),
    )
