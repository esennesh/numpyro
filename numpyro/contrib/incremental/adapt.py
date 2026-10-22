# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Run existing NumPyro MCMC kernels as block steps on an incremental density.

:class:`IncrementalMCMC` is an :class:`~numpyro.infer.mcmc.MCMCKernel` whose state
carries an :class:`~numpyro.contrib.incremental.IncrementalState`. Each *step* updates a
block of sites with an existing kernel:

- :class:`~numpyro.infer.hmc.HMC` / :class:`~numpyro.infer.hmc.NUTS` /
  :class:`~numpyro.infer.sa.SA` instances are rebound to a potential
  ``-(total + delta)`` whose gradient flows only through the factors the block
  touches (Δ-propagation makes exactly those dense);
- :class:`DiscreteGibbs` drives the proposal functions of
  :class:`~numpyro.infer.hmc_gibbs.DiscreteHMCGibbs` with a *sliced* incremental
  potential, so a single-element update costs its Markov blanket, not the model;
- :class:`GibbsFn` wraps a user ``gibbs_fn`` as in :class:`~numpyro.infer.hmc_gibbs.HMCGibbs`.

:meth:`IncrementalMCMC.from_kernel` adapts a ``DiscreteHMCGibbs``, ``HMCGibbs``, ``HMC``,
``NUTS`` or ``SA`` instance directly.
"""

from collections import namedtuple
import copy
from functools import partial
import math

import jax
from jax import lax, random, vmap
import jax.numpy as jnp

from numpyro.contrib.incremental.delta import Change
from numpyro.contrib.incremental.incremental import IncrementalDensity
from numpyro.handlers import seed, substitute, trace
from numpyro.infer.hmc import HMC, hmc
from numpyro.infer.hmc_gibbs import (
    DiscreteHMCGibbs,
    HMCGibbs,
    _discrete_gibbs_proposal,
    _discrete_modified_gibbs_proposal,
    _discrete_modified_rw_proposal,
    _discrete_rw_proposal,
    _wrap_model,
)
from numpyro.infer.initialization import init_to_sample
from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.sa import SA, _sa
from numpyro.util import identity, is_prng_key

IncrementalMCMCState = namedtuple(
    "IncrementalMCMCState", ["z", "inc", "blocks", "rng_key", "i"]
)
IncrementalMCMCState.__doc__ = """
State of :class:`IncrementalMCMC`: ``z`` the latent values (continuous sites
unconstrained), ``inc`` the :class:`IncrementalState`, ``blocks`` a tuple with one
sub-state per step, ``rng_key`` and the iteration counter ``i``.
"""


def _unset_potential(z):
    raise RuntimeError("This kernel is driven by IncrementalMCMC; call init first.")


class Block:
    """Base class of a step: updates ``sites`` given the incremental density."""

    sites = ()

    def bind(self, inc, sites):
        self.sites = tuple(sites)

    def plan(self, inc):
        """Changes (one per signature) this step applies."""
        return []

    def init(self, inc, state, rng_key, num_warmup, model_args, model_kwargs):
        return None

    def sample(self, inc, state, block_state, rng_key, model_args, model_kwargs):
        raise NotImplementedError

    def diagnostics(self, block_state):
        return ""


class PotentialBlock(Block):
    """An HMC/NUTS/SA kernel run on a block with the incremental potential."""

    def __init__(self, kernel):
        if isinstance(kernel, HMC):
            self.kind = "hmc"
        elif isinstance(kernel, SA):
            self.kind = "sa"
        else:
            raise TypeError(
                "PotentialBlock needs an HMC, NUTS or SA instance, got "
                f"{type(kernel).__name__}."
            )
        kernel = copy.copy(kernel)
        kernel._model = None
        kernel._potential_fn = _unset_potential
        kernel._init_fn = None
        kernel._sample_fn = None
        kernel._postprocess_fn = None
        self.kernel = kernel

    def bind(self, inc, sites):
        super().bind(inc, sites)

        def gen(*args, _inc=None, **kwargs):
            def potential_fn(z_block):
                delta, _ = inc.apply_delta(_inc, Change.dense(z_block), args, kwargs)
                return -(_inc.total + delta)

            return potential_fn

        self.kernel._potential_fn_gen = gen
        if self.kind == "hmc":
            self.kernel._init_fn, self.kernel._sample_fn = hmc(
                potential_fn_gen=gen,
                kinetic_fn=self.kernel._kinetic_fn,
                algo=self.kernel._algo,
            )
        else:
            self.kernel._init_fn, self.kernel._sample_fn = _sa(potential_fn_gen=gen)

    def plan(self, inc):
        return [
            Change.dense(
                {
                    s: jnp.zeros(inc.sites[s].shape, inc.sites[s].dtype)
                    for s in self.sites
                }
            )
        ]

    def _block_values(self, state):
        return {s: state.z[s] for s in self.sites}

    def init(self, inc, state, rng_key, num_warmup, model_args, model_kwargs):
        kwargs = {**model_kwargs, "_inc": state}
        return self.kernel.init(
            rng_key, num_warmup, self._block_values(state), model_args, kwargs
        )

    def sample(self, inc, state, block_state, rng_key, model_args, model_kwargs):
        z_b = self._block_values(state)
        kwargs = {**model_kwargs, "_inc": state}
        potential_fn = self.kernel._potential_fn_gen(*model_args, **kwargs)
        # Other blocks moved since the last step: refresh energy (and gradient).
        if self.kind == "hmc":
            if self.kernel._forward_mode_differentiation:
                pe, grad = potential_fn(z_b), jax.jacfwd(potential_fn)(z_b)
            else:
                pe, grad = jax.value_and_grad(potential_fn)(z_b)
            block_state = block_state._replace(z=z_b, potential_energy=pe, z_grad=grad)
        else:
            block_state = block_state._replace(
                z=z_b, potential_energy=potential_fn(z_b)
            )
        block_state = self.kernel.sample(block_state, model_args, kwargs)
        change = Change.dense({s: block_state.z[s] for s in self.sites})
        _, state = inc.apply(state, change, model_args, model_kwargs)
        return state, block_state

    def diagnostics(self, block_state):
        if self.kind == "hmc":
            return "{} steps of size {:.2e}. acc. prob={:.2f}".format(
                block_state.num_steps,
                block_state.adapt_state.step_size,
                block_state.mean_accept_prob,
            )
        return "acc. prob={:.2f}".format(block_state.mean_accept_prob)


class DiscreteGibbs(Block):
    """
    Single-site Gibbs (or Metropolised Gibbs / random-walk) updates of discrete sites
    with enumerable support, one plate element at a time, using the proposal
    functions of :class:`~numpyro.infer.hmc_gibbs.DiscreteHMCGibbs` driven by the
    incremental potential.

    :param bool random_walk: propose uniformly instead of from the conditional.
    :param bool modified: use the modified (current value excluded) proposals with
        their Metropolis correction.
    :param str order: ``"random"`` (a fresh permutation per sweep) or ``"systematic"``.
    """

    def __init__(self, *, random_walk=False, modified=False, order="random"):
        if order not in ("random", "systematic"):
            raise ValueError("order must be 'random' or 'systematic'.")
        self.random_walk = random_walk
        self.modified = modified
        self.order = order
        if random_walk:
            self.proposal_fn = (
                partial(_discrete_modified_rw_proposal, stay_prob=0.0)
                if modified
                else _discrete_rw_proposal
            )
        else:
            self.proposal_fn = (
                partial(_discrete_modified_gibbs_proposal, stay_prob=0.0)
                if modified
                else _discrete_gibbs_proposal
            )

    def bind(self, inc, sites):
        for s in sites:
            meta = inc.sites[s]
            if not meta.is_discrete or meta.support_size is None:
                raise ValueError(
                    f"DiscreteGibbs needs discrete sites with enumerable support; "
                    f"'{s}' is not one."
                )
            if len(meta.shape) != len(meta.plates):
                raise NotImplementedError(
                    f"Discrete site '{s}' has batch dimensions that are not plates."
                )
        super().bind(inc, sites)

    @staticmethod
    def _plate_info(meta):
        names = [p for p, _ in meta.plates]
        axes = [a for _, a in meta.plates]
        shape = tuple(meta.shape[a] for a in axes)
        return names, axes, shape

    def plan(self, inc):
        changes = []
        for s in self.sites:
            meta = inc.sites[s]
            names, axes, shape = self._plate_info(meta)
            if names:
                value = jnp.zeros([1] * len(axes), meta.dtype)
                changes.append(Change.sliced({s: value}, {p: 0 for p in names}))
            else:
                changes.append(Change.dense({s: jnp.zeros(meta.shape, meta.dtype)}))
        return changes

    def sample(self, inc, state, block_state, rng_key, model_args, model_kwargs):
        for site in self.sites:
            meta = inc.sites[site]
            names, axes, plate_shape = self._plate_info(meta)
            n = int(math.prod(plate_shape)) if names else 1
            support_size = meta.support_size

            def change_for(value, e, names=names, axes=axes, shape=plate_shape):
                if not names:
                    return Change.dense({site: value})
                multi = jnp.unravel_index(e, shape)
                for a, m in zip(axes, multi):
                    value = lax.dynamic_index_in_dim(value, m, axis=a, keepdims=True)
                return Change.sliced({site: value}, dict(zip(names, multi)))

            rng_key, rng_order = random.split(rng_key)
            if self.order == "random":
                order = random.permutation(rng_order, n)
            else:
                order = jnp.arange(n)

            def body(j, carry, site=site, change_for=change_for):
                rng_key, state = carry
                e = order[j]

                def potential_fn(z_new):
                    delta, _ = inc.apply_delta(
                        state, change_for(z_new[site], e), model_args, model_kwargs
                    )
                    return -(state.total + delta)

                rng_key, z_new, _, log_accept_ratio = self.proposal_fn(
                    rng_key,
                    {site: state.z[site]},
                    -state.total,
                    potential_fn=potential_fn,
                    idx=e,
                    support_size=support_size,
                )
                rng_key, rng_accept = random.split(rng_key)
                accept = random.exponential(rng_accept) > -log_accept_ratio

                def do_accept(s):
                    return inc.apply(
                        s, change_for(z_new[site], e), model_args, model_kwargs
                    )[1]

                state = lax.cond(accept, do_accept, identity, state)
                return rng_key, state

            rng_key, state = lax.fori_loop(0, n, body, (rng_key, state))
        return state, block_state


class GibbsFn(Block):
    """
    A user Gibbs update as in :class:`~numpyro.infer.hmc_gibbs.HMCGibbs`:
    ``gibbs_fn(rng_key, gibbs_sites, other_sites) -> new gibbs_sites`` with all values
    in constrained space.
    """

    def __init__(self, gibbs_fn):
        self.gibbs_fn = gibbs_fn

    def plan(self, inc):
        return [
            Change.dense(
                {
                    s: jnp.zeros(inc.sites[s].shape, inc.sites[s].dtype)
                    for s in self.sites
                }
            )
        ]

    def sample(self, inc, state, block_state, rng_key, model_args, model_kwargs):
        constrained = _constrain(inc, state.z)
        gibbs_sites = {s: constrained[s] for s in self.sites}
        others = {s: v for s, v in constrained.items() if s not in self.sites}
        new = self.gibbs_fn(rng_key, gibbs_sites, others)
        values = {}
        for s in self.sites:
            t = inc.sites[s].transform
            values[s] = new[s] if t is None else t.inv(new[s])
        _, state = inc.apply(state, Change.dense(values), model_args, model_kwargs)
        return state, block_state


def _constrain(inc, z):
    out = {}
    for name, meta in inc.sites.items():
        out[name] = z[name] if meta.transform is None else meta.transform(z[name])
    return out


class IncrementalMCMC(MCMCKernel):
    """
    Run existing kernels as block steps on an incremental density.

    :param model: a NumPyro model.
    :param steps: a list of ``(sites, kernel)`` pairs applied in order every
        iteration. ``sites`` is a site name, a sequence of names, ``None`` (the step's
        default: all discrete sites for :class:`DiscreteGibbs`, every site not
        claimed by another step otherwise). ``kernel`` is an ``HMC``/``NUTS``/``SA``
        instance (its model, if any, is ignored), a :class:`DiscreteGibbs`, a
        :class:`GibbsFn`, or any :class:`Block`.
    :param str cache: cache mode of :class:`IncrementalDensity` (``"plan"`` by
        default: only cut points are cached).
    :param int resync_every: recompute the cache and total from scratch every this
        many iterations (0 disables). Deltas are exact per step; the running total
        accumulates floating-point error only.
    :param bool validate: distribution validation while tracing the density.
    :param init_strategy: initialization strategy for chains without ``init_params``.

    Example::

        kernel = IncrementalMCMC(model, [(None, DiscreteGibbs()), (None, NUTS(model))])
        mcmc = MCMC(kernel, num_warmup=500, num_samples=1000)
        mcmc.run(random.key(0), data)
    """

    sample_field = "z"

    def __init__(
        self,
        model,
        steps,
        *,
        cache="plan",
        resync_every=0,
        validate=False,
        init_strategy=init_to_sample,
    ):
        self._model = model
        self._steps = [self._as_block(sites, kernel) for sites, kernel in steps]
        self._cache = cache
        self._resync_every = int(resync_every)
        self._validate = validate
        self._init_strategy = init_strategy
        self._inc = None
        self._blocks = None
        self._sample_fn = None
        self._model_args = ()
        self._model_kwargs = {}

    @staticmethod
    def _as_block(sites, kernel):
        if isinstance(sites, str):
            sites = (sites,)
        elif sites is not None:
            sites = tuple(sites)
        if isinstance(kernel, Block):
            return sites, kernel
        if isinstance(kernel, (HMC, SA)):
            return sites, PotentialBlock(kernel)
        if callable(kernel):
            return sites, GibbsFn(kernel)
        raise TypeError(f"Unsupported step kernel {type(kernel).__name__}.")

    @classmethod
    def from_kernel(cls, kernel, **kwargs):
        """
        Adapt an existing composite or simple kernel.

        - ``DiscreteHMCGibbs(inner)`` becomes ``[(discrete sites, DiscreteGibbs),
          (the rest, inner)]`` with the same ``random_walk``/``modified`` flags;
        - ``HMCGibbs(inner, gibbs_fn, gibbs_sites)`` becomes ``[(gibbs_sites,
          GibbsFn(gibbs_fn)), (the rest, inner)]``;
        - ``HMC``/``NUTS``/``SA`` run on every site (no locality to exploit, but the
          same machinery).
        """
        if isinstance(kernel, DiscreteHMCGibbs):
            model = _unwrap_model(kernel.inner_kernel)
            steps = [
                (
                    None,
                    DiscreteGibbs(
                        random_walk=kernel._random_walk, modified=kernel._modified
                    ),
                ),
                (None, kernel.inner_kernel),
            ]
        elif isinstance(kernel, HMCGibbs):
            model = _unwrap_model(kernel.inner_kernel)
            steps = [
                (tuple(kernel._gibbs_sites), GibbsFn(kernel._gibbs_fn)),
                (None, kernel.inner_kernel),
            ]
        elif isinstance(kernel, (HMC, SA)):
            model = kernel.model
            if model is None:
                raise ValueError("from_kernel needs a kernel built from a model.")
            steps = [(None, kernel)]
        else:
            raise TypeError(f"Cannot adapt {type(kernel).__name__}.")
        return cls(model, steps, **kwargs)

    # ------------------------------------------------------------------ setup
    @property
    def model(self):
        return self._model

    @property
    def incremental_density(self):
        return self._inc

    def _resolve_blocks(self):
        inc = self._inc
        claimed = set()
        resolved = []
        for sites, block in self._steps:
            if sites is not None:
                claimed |= set(sites)
        for sites, block in self._steps:
            if sites is None:
                if isinstance(block, DiscreteGibbs):
                    sites = tuple(
                        s
                        for s, m in inc.sites.items()
                        if m.is_discrete
                        and m.support_size is not None
                        and s not in claimed
                    )
                else:
                    sites = tuple(s for s in inc.program.site_names if s not in claimed)
                claimed |= set(sites)
            missing = [s for s in sites if s not in inc.sites]
            if missing:
                raise ValueError(f"Unknown latent sites {missing}.")
            if not sites:
                raise ValueError(f"Step {type(block).__name__} has no sites.")
            block.bind(inc, sites)
            resolved.append(block)
        unclaimed = [s for s in inc.program.site_names if s not in claimed]
        if unclaimed:
            raise ValueError(f"Sites {unclaimed} are not updated by any step.")
        return resolved

    def _setup(self, model_args, model_kwargs):
        self._model_args = model_args
        self._model_kwargs = model_kwargs
        self._inc = IncrementalDensity(
            self._model,
            model_args,
            model_kwargs,
            cache=self._cache,
            validate=self._validate,
            init_strategy=self._init_strategy,
        )
        self._blocks = self._resolve_blocks()
        changes = [c for b in self._blocks for c in b.plan(self._inc)]
        self._inc.plan(*changes)

    def _draw_init(self, rng_key, model_args, model_kwargs):
        program = self._inc.program
        tr = trace(
            substitute(seed(program.model, rng_key), substitute_fn=self._init_strategy)
        ).get_trace(*model_args, **model_kwargs)
        values = {}
        for name, meta in self._inc.sites.items():
            v = tr[name]["value"]
            values[name] = v if meta.transform is None else meta.transform.inv(v)
        return values

    # ------------------------------------------------------------------ kernel
    def init(self, rng_key, num_warmup, init_params, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
        self._setup(model_args, model_kwargs)
        self._num_warmup = num_warmup
        if is_prng_key(rng_key):
            self._sample_fn = self._sample_one
            return self._init_one(rng_key, init_params, model_args, model_kwargs)
        init_fn = partial(
            self._init_one, model_args=model_args, model_kwargs=model_kwargs
        )
        self._sample_fn = vmap(self._sample_one, in_axes=(0, None, None))
        return vmap(init_fn)(rng_key, init_params)

    def _init_one(self, rng_key, init_params, model_args, model_kwargs):
        inc = self._inc
        keys = random.split(rng_key, 2 + len(self._blocks))
        rng_key, key_init, block_keys = keys[0], keys[1], keys[2:]
        values = self._draw_init(key_init, model_args, model_kwargs)
        if init_params is not None:
            values = {**values, **init_params}
        state = inc.init(values, model_args, model_kwargs)
        blocks = tuple(
            b.init(inc, state, k, self._num_warmup, model_args, model_kwargs)
            for b, k in zip(self._blocks, block_keys)
        )
        return IncrementalMCMCState(state.z, state, blocks, rng_key, jnp.zeros((), int))

    def _sample_one(self, state, model_args, model_kwargs):
        inc = self._inc
        keys = random.split(state.rng_key, 1 + len(self._blocks))
        rng_key, block_keys = keys[0], keys[1:]
        inc_state = state.inc
        blocks = list(state.blocks)
        for j, (b, k) in enumerate(zip(self._blocks, block_keys)):
            inc_state, blocks[j] = b.sample(
                inc, inc_state, blocks[j], k, model_args, model_kwargs
            )
        i = state.i + 1
        if self._resync_every:
            inc_state = lax.cond(
                i % self._resync_every == 0,
                lambda s: inc.resync(s, model_args, model_kwargs),
                identity,
                inc_state,
            )
        return IncrementalMCMCState(inc_state.z, inc_state, tuple(blocks), rng_key, i)

    def sample(self, state, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        return self._sample_fn(state, model_args, model_kwargs)

    def postprocess_fn(self, model_args, model_kwargs):
        model_kwargs = {} if model_kwargs is None else model_kwargs
        inc = self._inc
        program = inc.program

        def fn(z):
            constrained = _constrain(inc, z)
            if program.deterministic_sites:
                tr = trace(substitute(program.model, data=constrained)).get_trace(
                    *model_args, **model_kwargs
                )
                for name in program.deterministic_sites:
                    constrained[name] = tr[name]["value"]
            return constrained

        return fn

    def get_diagnostics_str(self, state):
        parts = []
        for b, bs in zip(self._blocks, state.blocks):
            d = b.diagnostics(bs)
            if d:
                parts.append(d)
        return " | ".join(parts)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_sample_fn"] = None
        return state


def _unwrap_model(inner_kernel):
    model = inner_kernel.model
    if isinstance(model, partial) and model.func is _wrap_model:
        return model.args[0]
    return model


__all__ = [
    "Block",
    "DiscreteGibbs",
    "GibbsFn",
    "IncrementalMCMC",
    "IncrementalMCMCState",
    "PotentialBlock",
]
