# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""User-facing wrapper: :class:`IncrementalDensity` and :class:`IncrementalState`."""

from collections import namedtuple

import numpy as np

from jax.extend.core import Literal
import jax.numpy as jnp

from numpyro.contrib.incremental.delta import (
    Change,
    apply_change_to_values,
    input_deltas,
)
from numpyro.contrib.incremental.density import trace_density
from numpyro.contrib.incremental.interpreter import (
    incrementalize,
    number_vars,
    plan,
    record_cache,
    resolve_key,
)
from numpyro.infer.initialization import init_to_sample

IncrementalState = namedtuple("IncrementalState", ["z", "cache", "total"])
IncrementalState.__doc__ = """
State of an incremental density: ``z`` the current latent values (continuous sites
unconstrained), ``cache`` the cached intermediate values keyed by variable id, and
``total`` the (diagnostic) running joint log density.
"""


class IncrementalDensity:
    """
    Incrementally maintained joint log density of a NumPyro model.

    Usage::

        inc = IncrementalDensity(model, (data,))
        state = inc.init()
        delta, state = inc.apply(state, Change.sliced({"z": z_i}, {"data": i}), (data,))

    :param model: a NumPyro model.
    :param tuple model_args: model arguments (array leaves become program inputs and
        must be passed again to :meth:`init` and :meth:`apply`).
    :param dict model_kwargs: model keyword arguments, likewise.
    :param str cache: ``"all"`` caches every intermediate value of the density program;
        ``"plan"`` caches only the values the planned changes read in full (factor
        outputs, reductions over sliced axes, control flow and unknown primitives) and
        slices everything else through its producer. With ``"plan"``, call
        :meth:`plan` (or :meth:`plan_site_changes`) for every change signature before
        :meth:`init`; applying an unplanned signature raises.
    :param bool validate: distribution validation while tracing (default off).
    :param init_strategy: initialization strategy for the default initial values.
    :param rng_key: key for the prototype trace.
    """

    def __init__(
        self,
        model,
        model_args=(),
        model_kwargs=None,
        *,
        cache="all",
        validate=False,
        init_strategy=init_to_sample,
        rng_key=None,
    ):
        if cache not in ("all", "plan"):
            raise ValueError("cache must be 'all' or 'plan'.")
        self.cache_mode = cache
        self.model = model
        self.program, self.init_values = trace_density(
            model,
            model_args,
            model_kwargs,
            init_strategy=init_strategy,
            validate=validate,
            rng_key=rng_key,
        )
        self.table = number_vars(self.program.jaxpr.jaxpr)
        self.var_ids = self.table.ids
        jaxpr = self.program.jaxpr.jaxpr
        input_ids = {self.var_ids[((), v)] for v in jaxpr.invars}
        self._all_keys = frozenset(set(self.var_ids.values()) - input_ids)
        self._factor_keys = frozenset(
            self.var_ids[self._resolve(v)]
            for v in jaxpr.outvars
            if not isinstance(self._resolve(v), Literal)
        )
        if cache == "all":
            self.cache_keys = self._all_keys
        else:
            self.cache_keys = self._factor_keys
        self._compiled = {}
        self._plans = {}

    # ------------------------------------------------------------------ helpers
    @property
    def sites(self):
        return self.program.sites

    @property
    def factors(self):
        return self.program.factors

    @property
    def pullback(self):
        return self.cache_mode == "plan"

    def _resolve(self, atom):
        """Resolved key (or Literal) of a top-level atom."""
        if isinstance(atom, Literal):
            return atom
        return resolve_key(self.table, ((), atom))

    def var_id(self, var, path=()):
        """The cache id of a jaxpr variable (``path``: enclosing call-eqn ids)."""
        key = resolve_key(self.table, (path, var))
        return self.var_ids[key]

    def _plan(self, change):
        key = change.signature
        if key not in self._plans:
            reads, info = plan(self.program, self.table, change, self.pullback)
            self._plans[key] = (frozenset(reads), dict(info))
        return self._plans[key]

    def _compile(self, change):
        key = change.signature
        prog = self._compiled.get(key)
        if prog is None:
            if self.cache_mode == "plan":
                reads, _ = self._plan(change)
                missing = reads - self.cache_keys
                if missing:
                    raise ValueError(
                        f"{change} reads {len(missing)} variable(s) that are not "
                        "cached. Call `plan(change)` for every signature before "
                        "`init()` (and `resync` any existing state)."
                    )
            prog = incrementalize(
                self.program, self.table, change, self.cache_keys, self.pullback
            )
            self._compiled[key] = prog
        return prog

    def plan(self, *changes):
        """Register change signatures; their cut points are added to the cache plan."""
        for change in changes:
            reads, _ = self._plan(change)
            if not reads <= self.cache_keys:
                self.cache_keys = self.cache_keys | reads
                self._compiled.clear()
        return self

    def plan_site_changes(self, sites=None, ks=(1,), dense=True):
        """
        Plan the usual single-site changes: a dense change per site and, for plated
        sites, a sliced change along every non-empty subset of its plates with each
        slice length in ``ks``.
        """
        from itertools import combinations

        changes = []
        for name in sites or self.program.site_names:
            meta = self.sites[name]
            if dense:
                changes.append(Change.dense({name: jnp.zeros(meta.shape, meta.dtype)}))
            axes = dict(meta.plates)
            names = [p for p, _ in meta.plates]
            for r in range(1, len(names) + 1):
                for subset in combinations(names, r):
                    for k in ks:
                        shape = list(meta.shape)
                        index = {}
                        for p in subset:
                            shape[axes[p]] = k
                            index[p] = jnp.arange(k)
                        changes.append(
                            Change.sliced({name: jnp.zeros(shape, meta.dtype)}, index)
                        )
        return self.plan(*changes)

    def cache_size(self, state):
        """Number of cached scalars in ``state``."""
        return int(sum(np.prod(v.shape) for v in state.cache.values()))

    # ------------------------------------------------------------------ API
    def init(self, values=None, model_args=(), model_kwargs=None):
        """Evaluate the program from scratch and build the state."""
        values = dict(self.init_values if values is None else values)
        inputs = self.program.flatten_inputs(values, model_args, model_kwargs)
        outs, cache = record_cache(self.program, self.table, inputs, self.cache_keys)
        total = sum((jnp.sum(o) for o in outs), start=jnp.zeros(()))
        return IncrementalState(values, cache, total)

    def apply(self, state, change, model_args=(), model_kwargs=None):
        """Apply ``change``; returns ``(delta, new_state)``."""
        prog = self._compile(change)
        arg_leaves = self.program.flatten_args(model_args, model_kwargs)
        delta, _, updates = prog.apply(state.z, arg_leaves, state.cache, change)
        new_cache = dict(state.cache)
        new_cache.update(updates)
        new_z = apply_change_to_values(state.z, input_deltas(change, self.program))
        return delta, IncrementalState(new_z, new_cache, state.total + delta)

    def apply_delta(self, state, change, model_args=(), model_kwargs=None):
        """Only the change in joint log density (no cache update)."""
        prog = self._compile(change)
        arg_leaves = self.program.flatten_args(model_args, model_kwargs)
        delta, factor_deltas = prog.apply_delta(
            state.z, arg_leaves, state.cache, change
        )
        return delta, factor_deltas

    def resync(self, state, model_args=(), model_kwargs=None):
        """Recompute the cache and total from scratch at the current values."""
        return self.init(state.z, model_args, model_kwargs)

    def factor_values(self, state):
        """The cached per-factor log density arrays."""
        jaxpr = self.program.jaxpr.jaxpr
        out = {}
        for name, v in zip(self.program.factor_names, jaxpr.outvars):
            out[name] = self._cached_value(state, v)
        return out

    def _cached_value(self, state, atom):
        key = self._resolve(atom)
        if isinstance(key, Literal):
            return jnp.asarray(key.val)
        i = self.var_ids[key]
        if i in state.cache:
            return state.cache[i]
        raise KeyError(f"Factor output {atom} is not cached.")

    def explain(self, change):
        """Static report: per-factor Δ-kind and number of re-executed equations."""
        _, info = self._plan(change)
        tags = info["tags"]
        lines = [
            f"{change}: {info['num_eqns']} equations re-executed, "
            f"{len(info['reads'])} cached values read"
        ]
        for name in self.program.factor_names:
            lines.append(f"  {name}: {tags[name]}")
        return "\n".join(lines)

    def tags(self, change):
        """``{factor: 'Same' | 'Sliced' | 'Dense'}`` for ``change``."""
        return dict(self._plan(change)[1]["tags"])

    def num_reexecuted(self, change):
        return self._plan(change)[1]["num_eqns"]


# ----------------------------------------------------------------- consistency


def random_change(inc, rng, max_block=3, p_sliced=0.7):
    """
    A random single-site change for testing: a sliced change along a random non-empty
    subset of the site's plates (block of at most ``max_block`` indices per axis) with
    probability ``p_sliced`` when the site is plated, else a dense change.
    """
    sites = list(inc.sites)
    name = sites[rng.randint(len(sites))]
    meta = inc.sites[name]
    k = meta.support_size or 5

    def draw(shape):
        if meta.is_discrete:
            return np.asarray(rng.randint(k, size=shape), dtype=np.int32)
        return np.asarray(rng.standard_normal(shape), dtype=np.float32)

    if meta.plates and rng.rand() < p_sliced:
        plates = [p for p, _ in meta.plates if rng.rand() < 0.7] or [meta.plates[0][0]]
        axes = dict(meta.plates)
        index, shape = {}, list(meta.shape)
        for p in plates:
            size = meta.shape[axes[p]]
            kk = rng.randint(1, min(size, max_block) + 1)
            index[p] = rng.choice(size, size=kk, replace=False).astype(np.int32)
            shape[axes[p]] = kk
        return Change.sliced({name: draw(tuple(shape))}, index)
    return Change.dense({name: draw(meta.shape)})


def check_consistency(
    inc,
    model_args=(),
    model_kwargs=None,
    *,
    num_changes=20,
    seed=0,
    rtol=1e-4,
    atol=1e-4,
    reject_every=3,
):
    """
    Apply ``num_changes`` random changes (rejecting every ``reject_every``-th one, i.e.
    reusing the old state) and compare every cached factor and the running total with
    from-scratch evaluation after each step. Raises ``AssertionError`` on a mismatch
    and returns the largest absolute error seen.
    """
    rng = np.random.RandomState(seed)
    state = inc.init(model_args=model_args, model_kwargs=model_kwargs)
    worst = 0.0
    for step in range(num_changes):
        change = random_change(inc, rng)
        delta, new_state = inc.apply(state, change, model_args, model_kwargs)
        fresh = inc.program.eval(new_state.z, model_args, model_kwargs)
        cached = inc.factor_values(new_state)
        for name in inc.program.factor_names:
            err = np.max(np.abs(np.asarray(cached[name]) - np.asarray(fresh[name])))
            worst = max(worst, float(err))
            np.testing.assert_allclose(
                cached[name],
                fresh[name],
                rtol=rtol,
                atol=atol,
                err_msg=f"{change} {name}",
            )
        expected = sum(float(jnp.sum(v)) for v in fresh.values())
        np.testing.assert_allclose(
            new_state.total, expected, rtol=rtol, atol=atol, err_msg=f"{change} total"
        )
        if reject_every and step % reject_every == reject_every - 1:
            continue
        state = new_state
    return worst


__all__ = [
    "IncrementalDensity",
    "IncrementalState",
    "Change",
    "check_consistency",
    "random_change",
]
