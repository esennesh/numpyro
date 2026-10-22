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
"""

from collections import namedtuple

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
from numpyro.contrib.incremental.rules import bind, get_rule

_RECURSE_PARAMS = {
    P.jit_p: "jaxpr",
    P.closed_call_p: "call_jaxpr",
    P.custom_jvp_call_p: "call_jaxpr",
}


def _sub_jaxpr(eqn):
    name = _RECURSE_PARAMS.get(eqn.primitive)
    return eqn.params[name] if name is not None else None


VarTable = namedtuple("VarTable", ["ids", "alias"])
VarTable.__doc__ = """
``ids`` maps every variable of a jaxpr and its call bodies to a stable integer id;
``alias`` maps call-body input variables to the outer atoms they are bound to and
outer call outputs to the inner atoms that produce them, so that a value crossing a
call boundary has exactly one id (and one cache entry).
"""


def number_vars(jaxpr, table=None):
    """Build the :class:`VarTable` of ``jaxpr`` (recursing into call bodies)."""
    table = VarTable({}, {}) if table is None else table
    ids, alias = table

    def resolve(atom):
        while not isinstance(atom, Literal) and atom in alias:
            atom = alias[atom]
        return atom

    def visit(v):
        if isinstance(v, (Literal, DropVar)):
            return
        if v not in ids:
            ids[v] = len(ids)

    for v in jaxpr.constvars:
        visit(v)
    for v in jaxpr.invars:
        if v not in alias:
            visit(v)
    for eqn in jaxpr.eqns:
        sub = _sub_jaxpr(eqn)
        if sub is None:
            for v in eqn.outvars:
                visit(v)
            continue
        for inner, outer in zip(sub.jaxpr.invars, eqn.invars):
            alias[inner] = outer
        number_vars(sub.jaxpr, table)
        for outer, inner in zip(eqn.outvars, sub.jaxpr.outvars):
            if isinstance(outer, DropVar):
                continue
            src = resolve(inner)
            if isinstance(src, Literal):
                alias[outer] = src
            else:
                alias[outer] = src
                ids[outer] = ids[src]
    return table


# --------------------------------------------------------------------- recording


def record_cache(program, table, inputs, keys=None):
    """
    Evaluate the program from scratch, returning ``(outputs, cache)`` where ``cache``
    holds the value of every equation output whose id is in ``keys`` (all if None).
    """
    cache = {}
    var_ids = table.ids

    def run(jaxpr, consts, args):
        env = {}

        def read(a):
            return a.val if isinstance(a, Literal) else env[a]

        def write(v, val):
            if isinstance(v, DropVar):
                return
            env[v] = val
            i = var_ids.get(v)
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
                outs = run(sub.jaxpr, sub.consts, ins)
            else:
                outs = bind(eqn, *ins)
            for v, o in zip(eqn.outvars, outs):
                write(v, o)
        return [read(v) for v in jaxpr.outvars]

    outs = run(program.jaxpr.jaxpr, program.jaxpr.consts, list(inputs))
    return outs, cache


# --------------------------------------------------------------------- context


class Context:
    """Per-apply state of the Δ-interpreter."""

    def __init__(self, table, cache, cache_keys, inputs, write_updates=True):
        self.var_ids = table.ids
        self.alias = table.alias  # static call-boundary aliases
        self.cache = cache  # old values, {var_id: array}
        self.cache_keys = cache_keys  # ids whose new values must be produced
        self.inputs = inputs  # {invar: old value}
        self.consts = {}  # {constvar: value}
        self.updates = {}  # {var_id: new full value}
        self.reads = set()  # ids read in full or sliced (for cache planning)
        self.deltas = {}  # {var: Δ} for every var visited
        self.write_updates = write_updates
        self.num_eqns = 0

    # --- resolution
    def resolve(self, atom):
        while not isinstance(atom, Literal) and atom in self.alias:
            atom = self.alias[atom]
        return atom

    def full(self, atom):
        """The OLD full value of ``atom``."""
        atom = self.resolve(atom)
        if isinstance(atom, Literal):
            return atom.val
        if atom in self.consts:
            return self.consts[atom]
        if atom in self.inputs:
            return self.inputs[atom]
        i = self.var_ids[atom]
        self.reads.add(i)
        if i not in self.cache:
            raise KeyError(
                f"Variable {atom} (id {i}) is needed but not cached; use cache='all'."
            )
        return self.cache[i]

    def slice(self, atom, axes, idxs):
        """The OLD value of ``atom`` on the block (``axes``, ``idxs``)."""
        if not axes:
            return self.full(atom)
        return gather_rect(self.full(atom), axes, idxs)

    def materialize_sliced(self, atom, d):
        return scatter_rect(self.full(atom), d.axes, d.idxs, d.value)

    # --- interpretation
    def interpret(self, jaxpr, consts, in_deltas):
        env = {}
        for v, c in zip(jaxpr.constvars, consts):
            self.consts[v] = c
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
                self.deltas[v] = d
                if (
                    self.write_updates
                    and not isinstance(d, Same)
                    and v not in self.alias  # call outputs are updated by the body
                    and self.var_ids.get(v) in self.cache_keys
                ):
                    self.updates[self.var_ids[v]] = (
                        d.value
                        if isinstance(d, Dense)
                        else self.materialize_sliced(v, d)
                    )
        return [read(v) for v in jaxpr.outvars]


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


def incrementalize(program, table, signature_change, cache_keys):
    """
    Build the incremental program for the signature of ``signature_change``.

    :param program: a :class:`DensityProgram`.
    :param table: the :class:`VarTable` from :func:`number_vars`.
    :param signature_change: any :class:`Change` with the wanted signature.
    :param cache_keys: ids of the cached variables (whose new values ``apply`` must
        produce when they change).
    :return: an :class:`IncrementalProgram` with ``apply`` and ``apply_delta``.
    """
    jaxpr = program.jaxpr.jaxpr
    consts = program.jaxpr.consts
    site_invars = jaxpr.invars[: program.num_site_inputs]
    info = {}

    def run(site_values, arg_leaves, cache, change, write_updates):
        deltas = input_deltas(change, program)
        inputs = {}
        in_deltas = []
        for name, v in zip(program.site_names, site_invars):
            inputs[v] = site_values[name]
            in_deltas.append(deltas.get(name, SAME))
        for v, a in zip(jaxpr.invars[program.num_site_inputs :], arg_leaves):
            inputs[v] = a
            in_deltas.append(SAME)
        ctx = Context(table, cache, cache_keys, inputs, write_updates)
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
        return delta, factor_deltas, ctx.updates

    def apply(site_values, arg_leaves, cache, change):
        return run(site_values, arg_leaves, cache, change, True)

    def apply_delta(site_values, arg_leaves, cache, change):
        delta, factor_deltas, _ = run(site_values, arg_leaves, cache, change, False)
        return delta, factor_deltas

    return IncrementalProgram(signature_change.signature, apply, apply_delta, info)
