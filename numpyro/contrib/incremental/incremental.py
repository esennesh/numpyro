# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""User-facing wrapper: :class:`IncrementalDensity` and :class:`IncrementalState`."""

from collections import namedtuple

import numpy as np

import jax
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
    record_cache,
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
    :param str cache: ``"all"`` caches every intermediate value of the density program.
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
        if cache != "all":
            raise NotImplementedError("Only cache='all' is implemented so far.")
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
        input_ids = {self.var_ids[v] for v in jaxpr.invars}
        self.cache_keys = frozenset(set(self.var_ids.values()) - input_ids)
        self._compiled = {}

    # ------------------------------------------------------------------ helpers
    @property
    def sites(self):
        return self.program.sites

    @property
    def factors(self):
        return self.program.factors

    def _compile(self, change):
        key = change.signature
        prog = self._compiled.get(key)
        if prog is None:
            prog = incrementalize(self.program, self.table, change, self.cache_keys)
            self._compiled[key] = prog
        return prog

    # ------------------------------------------------------------------ API
    def init(self, values=None, model_args=(), model_kwargs=None):
        """Evaluate the program from scratch and build the state."""
        values = dict(self.init_values if values is None else values)
        inputs = self.program.flatten_inputs(values, model_args, model_kwargs)
        outs, cache = record_cache(self.program, self.table, inputs)
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
        alias = self.table.alias
        while not isinstance(atom, Literal) and atom in alias:
            atom = alias[atom]
        if isinstance(atom, Literal):
            return jnp.asarray(atom.val)
        i = self.var_ids[atom]
        if i in state.cache:
            return state.cache[i]
        raise KeyError(f"Factor output {atom} is not cached.")

    def explain(self, change):
        """Static report: per-factor Δ-kind and number of re-executed equations."""
        prog = self._compile(change)
        z_abs = {
            n: jax.ShapeDtypeStruct(m.shape, m.dtype) for n, m in self.sites.items()
        }
        jaxpr = self.program.jaxpr.jaxpr
        args_abs = [
            jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype)
            for v in jaxpr.invars[self.program.num_site_inputs :]
        ]
        cache_abs = {
            i: jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype)
            for v, i in self.var_ids.items()
            if i in self.cache_keys
        }
        jax.eval_shape(prog.apply_delta, z_abs, args_abs, cache_abs, change)
        tags = dict(prog.info["tags"])
        lines = [f"{change}: {prog.info['num_eqns']} equations re-executed"]
        for name in self.program.factor_names:
            lines.append(f"  {name}: {tags[name]}")
        return "\n".join(lines)

    def tags(self, change):
        """``{factor: 'Same' | 'Sliced' | 'Dense'}`` for ``change``."""
        self.explain(change)
        return dict(self._compile(change).info["tags"])

    def num_reexecuted(self, change):
        self.explain(change)
        return self._compile(change).info["num_eqns"]


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
