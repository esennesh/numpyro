# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""User-facing wrapper: :class:`IncrementalDensity` and :class:`IncrementalState`."""

from collections import namedtuple

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


__all__ = ["IncrementalDensity", "IncrementalState", "Change"]
