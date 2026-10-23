# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Stage 2: the Δ-interpreter.

:func:`incrementalize` turns a :class:`~numpyro.contrib.incremental.density.DensityProgram`
and a change signature into an ordinary JAX function
``apply(site_values, arg_leaves, cache, change) -> (delta, factor_deltas, updates)``
that re-evaluates only the equations reached by the change, reading unchanged
intermediate values from ``cache`` (a ``{var_id: array}`` dict recorded by
:func:`record_cache`). Because the interpreter runs under tracing, ``apply`` composes
with ``jax.jit``, ``jax.vmap`` and ``jax.grad``.

Variables are identified by *keys* ``(path, var)`` where ``path`` is the tuple of
``id(eqn)`` of the enclosing call equations: JAX shares one ``Jaxpr`` object (and its
``Var`` objects) between call sites of the same function, so a bare ``Var`` does not
identify a value.
"""

from collections import namedtuple

import jax
from jax.extend.core import DropVar, Literal, primitives as P
import jax.numpy as jnp

from numpyro.contrib.incremental.delta import (
    SAME,
    Dense,
    Same,
    gather_rect,
    input_deltas,
    scatter_rect,
)
from numpyro.contrib.incremental.rules import PULLBACK, bind, get_rule

_RECURSE_PARAMS = {
    P.jit_p: "jaxpr",
    P.closed_call_p: "call_jaxpr",
    P.custom_jvp_call_p: "call_jaxpr",
}


def _sub_jaxpr(eqn):
    name = _RECURSE_PARAMS.get(eqn.primitive)
    return eqn.params[name] if name is not None else None


VarTable = namedtuple("VarTable", ["ids", "alias", "producers"])
VarTable.__doc__ = """
``ids`` maps every variable key of a jaxpr and its call bodies to a stable integer id;
``alias`` maps call-body input keys to the outer atoms they are bound to and outer
call-output keys to the inner keys that produce them (a value crossing a call boundary
has exactly one id and one cache entry); ``producers`` maps every non-call equation
output key to ``(path, eqn)`` for slice pullback. Alias targets are keys or
:class:`Literal` objects.
"""


def resolve_key(table, key):
    """Follow aliases from ``key``; returns a key or a :class:`Literal`."""
    alias = table.alias
    while isinstance(key, tuple) and key in alias:
        key = alias[key]
    return key


def number_vars(jaxpr, table=None, path=()):
    """Build the :class:`VarTable` of ``jaxpr`` (recursing into call bodies)."""
    table = VarTable({}, {}, {}) if table is None else table
    ids, alias, producers = table

    def visit(v):
        if isinstance(v, (Literal, DropVar)):
            return
        key = (path, v)
        if key not in ids:
            ids[key] = len(ids)

    for v in jaxpr.constvars:
        visit(v)
    for v in jaxpr.invars:
        if (path, v) not in alias:
            visit(v)
    for eqn in jaxpr.eqns:
        sub = _sub_jaxpr(eqn)
        if sub is None:
            for v in eqn.outvars:
                visit(v)
                if not isinstance(v, DropVar):
                    producers[(path, v)] = (path, eqn)
            continue
        sub_path = path + (id(eqn),)
        for inner, outer in zip(sub.jaxpr.invars, eqn.invars):
            alias[(sub_path, inner)] = (
                outer
                if isinstance(outer, Literal)
                else (
                    path,
                    outer,
                )
            )
        number_vars(sub.jaxpr, table, sub_path)
        for outer, inner in zip(eqn.outvars, sub.jaxpr.outvars):
            if isinstance(outer, DropVar):
                continue
            src = (
                inner
                if isinstance(inner, Literal)
                else resolve_key(table, (sub_path, inner))
            )
            alias[(path, outer)] = src
            if not isinstance(src, Literal):
                ids[(path, outer)] = ids[src]
    return table


# --------------------------------------------------------------------- recording


def record_cache(program, table, inputs, keys=None):
    """
    Evaluate the program from scratch, returning ``(outputs, cache)`` where ``cache``
    holds the value of every equation output whose id is in ``keys`` (all if None).
    """
    cache = {}
    ids = table.ids

    def run(jaxpr, consts, args, path):
        env = {}

        def read(a):
            return a.val if isinstance(a, Literal) else env[a]

        def write(v, val):
            if isinstance(v, DropVar):
                return
            env[v] = val
            i = ids.get((path, v))
            if i is not None and (keys is None or i in keys):
                cache[i] = val

        for v, c in zip(jaxpr.constvars, consts):
            env[v] = c
        for v, a in zip(jaxpr.invars, args):
            env[v] = a
        for eqn in jaxpr.eqns:
            ins = [read(a) for a in eqn.invars]
            sub = _sub_jaxpr(eqn)
            if sub is not None:
                outs = run(sub.jaxpr, sub.consts, ins, path + (id(eqn),))
            else:
                outs = bind(eqn, *ins)
            for v, o in zip(eqn.outvars, outs):
                write(v, o)
        return [read(v) for v in jaxpr.outvars]

    outs = run(program.jaxpr.jaxpr, program.jaxpr.consts, list(inputs), ())
    return outs, cache


# --------------------------------------------------------------------- context


class Context:
    """Per-apply state of the Δ-interpreter."""

    def __init__(
        self, table, cache, cache_keys, inputs, write_updates=True, pullback=False
    ):
        self.table = table
        self.var_ids = table.ids
        self.alias = table.alias
        self.producers = table.producers
        self.pullback = pullback  # slice unchanged values through their producers
        self.cache = cache  # old values, {var_id: array}
        self.cache_keys = cache_keys  # ids whose new values must be produced
        self.inputs = inputs  # {((), invar): old value}
        self.consts = {}  # {(path, constvar): value}
        self.updates = {}  # {var_id: new full value}
        self.pending = []  # [(var_id, key, Δ)] cache writes deferred to the end
        self.reads = set()  # ids read in full (for cache planning)
        self.input_full_reads = set()  # input keys read in full (not just sliced)
        self.write_updates = write_updates
        self.num_eqns = 0
        self.path = ()

    # --- resolution
    def resolve(self, atom):
        """Resolved key (or Literal) of ``atom`` in the current call path."""
        if isinstance(atom, Literal):
            return atom
        return resolve_key(self.table, (self.path, atom))

    def full_key(self, key):
        """The OLD full value behind a resolved key or Literal."""
        if isinstance(key, Literal):
            return key.val
        if key in self.consts:
            return self.consts[key]
        if key in self.inputs:
            self.input_full_reads.add(key)
            return self.inputs[key]
        i = self.var_ids[key]
        self.reads.add(i)
        if i not in self.cache:
            raise KeyError(
                f"Variable {key[1]} (id {i}) is needed but not cached; plan this "
                "change first or use cache='all'."
            )
        return self.cache[i]

    def full(self, atom):
        """The OLD full value of ``atom`` (in the current call path)."""
        return self.full_key(self.resolve(atom))

    def slice(self, atom, axes, idxs):
        """
        The OLD value of ``atom`` on the block (``axes``, ``idxs``).

        With ``pullback`` on, an equation output is sliced by slicing its producer's
        inputs and re-running the producer on the slices (see ``rules.PULLBACK``), so
        that only cut points (reductions over sliced axes, control flow, unknown
        primitives) need their full value cached.
        """
        if not axes:
            return self.full(atom)
        key = self.resolve(atom)
        if not isinstance(key, Literal) and key in self.inputs:
            # A sliced read of an input is not a full read (see input_full_reads).
            return gather_rect(self.inputs[key], axes, idxs)
        if self.pullback and not isinstance(key, Literal):
            prod = self.producers.get(key)
            if prod is not None:
                ppath, eqn = prod
                rule = PULLBACK.get(eqn.primitive)
                if rule is not None and len(eqn.outvars) == 1:
                    saved, self.path = self.path, ppath
                    try:
                        out = rule(self, eqn, tuple(axes), tuple(idxs))
                    finally:
                        self.path = saved
                    if out is not None:
                        return out
        return gather_rect(self.full_key(key), axes, idxs)

    def materialize_sliced(self, atom, d):
        return scatter_rect(self.full(atom), d.axes, d.idxs, d.value)

    # --- bodies rebound whole
    def refresh_body(self, eqn, args):
        """
        After rebinding a call-like ``eqn`` whole (e.g. ``custom_jvp_call`` on the
        dense path), recompute any cached values inside its body from the new
        ``args`` so they do not go stale. Cut points inside bodies arise when slice
        pullback enters the body on another signature.
        """
        sub = _sub_jaxpr(eqn)
        if sub is None or not self.write_updates:
            return
        path = self.path + (id(eqn),)
        if not self._body_has_cached(sub.jaxpr, path):
            return
        self._eval_recording(sub.jaxpr, sub.consts, args, path)

    def _body_has_cached(self, jaxpr, path):
        for eqn in jaxpr.eqns:
            for v in eqn.outvars:
                key = (path, v)
                if key not in self.alias and self.var_ids.get(key) in self.cache_keys:
                    return True
            sub = _sub_jaxpr(eqn)
            if sub is not None and self._body_has_cached(sub.jaxpr, path + (id(eqn),)):
                return True
        return False

    def _eval_recording(self, jaxpr, consts, args, path):
        env = {}

        def read(a):
            return a.val if isinstance(a, Literal) else env[a]

        for v, c in zip(jaxpr.constvars, consts):
            env[v] = c
        for v, a in zip(jaxpr.invars, args):
            env[v] = a
        for eqn in jaxpr.eqns:
            ins = [read(a) for a in eqn.invars]
            sub = _sub_jaxpr(eqn)
            if sub is not None:
                outs = self._eval_recording(
                    sub.jaxpr, sub.consts, ins, path + (id(eqn),)
                )
            else:
                outs = bind(eqn, *ins)
            for v, o in zip(eqn.outvars, outs):
                if isinstance(v, DropVar):
                    continue
                env[v] = o
                key = (path, v)
                if key not in self.alias and self.var_ids.get(key) in self.cache_keys:
                    self.updates[self.var_ids[key]] = o
        return [read(v) for v in jaxpr.outvars]

    # --- interpretation
    def interpret(self, jaxpr, consts, in_deltas, call_eqn=None):
        """Interpret ``jaxpr`` (the body of ``call_eqn`` when given) on Δ-inputs."""
        saved = self.path
        if call_eqn is not None:
            self.path = self.path + (id(call_eqn),)
        try:
            return self._interpret(jaxpr, consts, in_deltas)
        finally:
            self.path = saved

    def _interpret(self, jaxpr, consts, in_deltas):
        env = {}
        path = self.path
        for v, c in zip(jaxpr.constvars, consts):
            self.consts[(path, v)] = c
            env[v] = SAME
        for v, d in zip(jaxpr.invars, in_deltas):
            env[v] = d

        def read(a):
            return SAME if isinstance(a, Literal) else env[a]

        for eqn in jaxpr.eqns:
            ins = [read(a) for a in eqn.invars]
            if all(isinstance(d, Same) for d in ins):
                outs = [SAME] * len(eqn.outvars)
            else:
                self.num_eqns += 1
                outs = get_rule(eqn.primitive)(self, eqn, ins)
            for v, d in zip(eqn.outvars, outs):
                if isinstance(v, DropVar):
                    continue
                env[v] = d
                key = (path, v)
                if (
                    self.write_updates
                    and not isinstance(d, Same)
                    and key not in self.alias  # call outputs are updated by the body
                    and self.var_ids.get(key) in self.cache_keys
                ):
                    # Deferred: in-place writes must come after every read of the
                    # old values, see ``flush_updates``.
                    self.pending.append((self.var_ids[key], key, d))
        return [read(v) for v in jaxpr.outvars]

    def flush_updates(self, values):
        """
        Materialize the deferred cache writes with (barriered) new ``values``: a
        dynamic-update-slice for sliced variables, a replacement for dense ones.
        """
        for (i, key, d), value in zip(self.pending, values):
            if isinstance(d, Dense):
                self.updates[i] = value
            else:
                self.updates[i] = scatter_rect(
                    self.full_key(key), d.axes, d.idxs, value
                )
        self.pending = []
        return self.updates


# --------------------------------------------------------------------- program

IncrementalProgram = namedtuple(
    "IncrementalProgram", ["signature", "apply", "apply_delta", "info"]
)


def _factor_delta(ctx, atom, d):
    """``(delta_sum, payload)`` for one factor output with Δ-value ``d``."""
    if isinstance(d, Same):
        return jnp.zeros(()), None
    if isinstance(d, Dense):
        old = ctx.full(atom)
        return jnp.sum(d.value) - jnp.sum(old), d
    old = ctx.slice(atom, d.axes, d.idxs)
    return jnp.sum(d.value) - jnp.sum(old), d


def incrementalize(program, table, signature_change, cache_keys, pullback=False):
    """
    Build the incremental program for the signature of ``signature_change``.

    :param program: a :class:`DensityProgram`.
    :param table: the :class:`VarTable` from :func:`number_vars`.
    :param signature_change: any :class:`Change` with the wanted signature.
    :param cache_keys: ids of the cached variables (whose new values ``apply`` must
        produce when they change).
    :param bool pullback: slice unchanged values through their producers instead of
        reading them from the cache.
    :return: an :class:`IncrementalProgram` with ``apply`` and ``apply_delta``.
    """
    jaxpr = program.jaxpr.jaxpr
    consts = program.jaxpr.consts
    site_invars = jaxpr.invars[: program.num_site_inputs]
    info = {}

    def run(site_values, arg_leaves, cache, change, write_updates, read_values=None):
        deltas = input_deltas(change, program)
        inputs = {}
        in_deltas = []
        read_values = site_values if read_values is None else read_values
        for name, v in zip(program.site_names, site_invars):
            inputs[((), v)] = read_values[name]
            in_deltas.append(deltas.get(name, SAME))
        for v, a in zip(jaxpr.invars[program.num_site_inputs :], arg_leaves):
            inputs[((), v)] = a
            in_deltas.append(SAME)
        ctx = Context(table, cache, cache_keys, inputs, write_updates, pullback)
        outs = ctx.interpret(jaxpr, consts, in_deltas)
        delta = jnp.zeros(())
        factor_deltas = {}
        tags = {}
        for name, atom, d in zip(program.factor_names, jaxpr.outvars, outs):
            ds, payload = _factor_delta(ctx, atom, d)
            delta = delta + ds
            tags[name] = type(d).__name__
            if payload is not None:
                factor_deltas[name] = payload
        info["tags"] = tags
        info["num_eqns"] = ctx.num_eqns
        info["reads"] = frozenset(ctx.reads)
        site_of = dict(zip(site_invars, program.site_names))
        info["input_full_reads"] = frozenset(
            site_of[key[1]] for key in ctx.input_full_reads if key[1] in site_of
        )
        if not write_updates:
            return delta, factor_deltas, {}, change
        # Cache writes are deferred until every read of an old value is done.
        pending_values = [d.value for _, _, d in ctx.pending]
        updates = ctx.flush_updates(pending_values)
        return delta, factor_deltas, updates, change

    def apply(site_values, arg_leaves, cache, change, read_values=None):
        """
        Returns ``(delta, factor_deltas, cache_updates, change)``.

        ``read_values`` (default ``site_values``) are the values the program reads;
        a sweep over distinct elements of one site may pass a snapshot taken before
        the sweep so that the carried site array is only written inside the loop
        (XLA CPU copies a loop-carried buffer that is both read and updated in
        place, which would make each element update O(N)). This is exact when the
        changed sites are not read in full (``info["input_full_reads"]``).
        """
        return run(site_values, arg_leaves, cache, change, True, read_values)

    def apply_delta(site_values, arg_leaves, cache, change, read_values=None):
        delta, factor_deltas, _, _ = run(
            site_values, arg_leaves, cache, change, False, read_values
        )
        return delta, factor_deltas

    return IncrementalProgram(signature_change.signature, apply, apply_delta, info)


def abstract_inputs(program, table):
    """Abstract site values, argument leaves and a full cache, for planning traces."""
    jaxpr = program.jaxpr.jaxpr
    z_abs = {
        n: jax.ShapeDtypeStruct(m.shape, m.dtype) for n, m in program.sites.items()
    }
    args_abs = [
        jax.ShapeDtypeStruct(v.aval.shape, v.aval.dtype)
        for v in jaxpr.invars[program.num_site_inputs :]
    ]
    cache_abs = {
        i: jax.ShapeDtypeStruct(key[1].aval.shape, key[1].aval.dtype)
        for key, i in table.ids.items()
    }
    return z_abs, args_abs, cache_abs


def plan(program, table, change, pullback):
    """
    Trace ``apply_delta`` for ``change`` abstractly (against a cache holding every
    variable, writing nothing) and return the ids it reads in full. These are the cut
    points that must be cached for this signature; the factor outputs are added by the
    caller. Also returns the program's ``info``.
    """
    prog = incrementalize(program, table, change, frozenset(), pullback)
    z_abs, args_abs, cache_abs = abstract_inputs(program, table)
    jax.eval_shape(prog.apply, z_abs, args_abs, cache_abs, change)
    return prog.info["reads"], prog.info
