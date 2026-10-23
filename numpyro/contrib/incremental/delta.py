# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Change descriptions (the paper's Δ-types, restricted to arrays).

Inside the interpreter every jaxpr variable carries a Δ-value:

- :data:`SAME` — the value did not change. Its full old value is read from the cache
  only when a rule needs it.
- :class:`Sliced` — the value changed only on a rectangular sub-block: the Cartesian
  product of index arrays ``idxs`` along static ``axes``. ``value`` holds the new
  sub-block, laid out like the array with each sliced axis replaced by its slice length.
  ``token`` identifies the index arrays so that two Sliced values with equal tokens are
  known to address the same positions.
- :class:`Dense` — the value changed arbitrarily; ``value`` is the full new array.

Users describe changes with :class:`Change`, a pytree whose static structure (site
names, kinds, plate names, slice lengths) is the *signature* that selects a compiled
incremental program, and whose leaves (indices, new values) are traced.
"""

from dataclasses import dataclass
from typing import Any, Hashable

import jax
from jax import lax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


class Same:
    """Marker for an unchanged value (singleton :data:`SAME`)."""

    __slots__ = ()

    def __repr__(self):
        return "Same"


SAME = Same()


@dataclass(frozen=True)
class Sliced:
    axes: tuple[int, ...]
    idxs: tuple[Any, ...]
    value: Any
    token: Hashable

    def __post_init__(self):
        assert len(self.axes) == len(self.idxs)
        assert len(set(self.axes)) == len(self.axes)

    @property
    def ks(self):
        return tuple(int(i.shape[0]) for i in self.idxs)

    def same_block(self, other):
        return self.token == other.token and self.axes == other.axes


@dataclass(frozen=True)
class Dense:
    value: Any


jax.tree_util.register_dataclass(
    Sliced, data_fields=["idxs", "value"], meta_fields=["axes", "token"]
)
jax.tree_util.register_dataclass(Dense, data_fields=["value"], meta_fields=[])


def sliced_shape(shape, axes, ks):
    """``shape`` with each axis in ``axes`` replaced by the matching slice length."""
    out = list(shape)
    for a, k in zip(axes, ks):
        out[a] = k
    return tuple(out)


def _index_tuple(ndim, axes, idxs):
    ix = [slice(None)] * ndim
    for j, (a, i) in enumerate(zip(axes, idxs)):
        shape = [1] * len(axes)
        shape[j] = -1
        ix[a] = jnp.reshape(i, shape)
    return tuple(ix)


def _contiguous(axes):
    axes = sorted(axes)
    return all(b == a + 1 for a, b in zip(axes, axes[1:]))


def gather_rect(x, axes, idxs):
    """Read the rectangular block of ``x`` at ``idxs`` along ``axes`` (layout kept)."""
    x = jnp.asarray(x)
    for a, i in zip(axes, idxs):
        if i.shape[0] == 1:
            # dynamic_slice lowers to an in-place-friendly op; a gather does not.
            x = lax.dynamic_index_in_dim(x, i[0], axis=a, keepdims=True)
        else:
            ix = (slice(None),) * a + (i,)
            x = x.at[ix].get(mode="promise_in_bounds", unique_indices=True)
    return x


def scatter_rect(x, axes, idxs, value):
    """Write ``value`` into the rectangular block of ``x`` at ``idxs`` along ``axes``."""
    x = jnp.asarray(x)
    if not axes:
        return jnp.asarray(value)
    value = jnp.asarray(value, x.dtype)
    if all(i.shape[0] == 1 for i in idxs):
        # Single element per sliced axis: a dynamic_update_slice, which XLA updates
        # in place inside loops (a general scatter may copy the operand).
        start = [jnp.zeros((), jnp.int32)] * x.ndim
        for a, i in zip(axes, idxs):
            start[a] = i[0].astype(jnp.int32)
        return lax.dynamic_update_slice(x, value, start)
    ix = _index_tuple(x.ndim, axes, idxs)
    if not _contiguous(axes):
        # NumPy advanced indexing moves non-contiguous advanced dims to the front.
        order = sorted(range(len(axes)), key=lambda j: axes[j])
        src = [axes[j] for j in order]
        value = jnp.moveaxis(value, src, list(range(len(axes))))
        if order != list(range(len(axes))):
            # idxs are in axes order; after sorting, the broadcast dims must follow.
            ix = _index_tuple(x.ndim, src, [idxs[j] for j in order])
    return x.at[ix].set(value, mode="promise_in_bounds", unique_indices=True)


# --------------------------------------------------------------------------- Change


@register_pytree_node_class
class Change:
    """
    A change to the latent values of a density program.

    Build one with :meth:`dense` or :meth:`sliced` and combine with ``|``. Site names,
    change kinds, plate names and slice lengths are static (pytree aux data); index
    arrays and new values are leaves, so a jitted function retraces only when the
    :attr:`signature` changes.
    """

    def __init__(self, entries=()):
        self.entries = tuple(entries)

    # --- constructors
    @classmethod
    def dense(cls, values=None, **kwargs):
        """Replace whole site values: ``Change.dense({"mu": new_mu})``."""
        values = {**(values or {}), **kwargs}
        return cls(
            [
                ("dense", (name,), (), (), (jnp.asarray(values[name]),))
                for name in values
            ]
        )

    @classmethod
    def sliced(cls, values, index):
        """
        Replace the sub-block of one or more sites addressed by ``index``.

        :param dict values: site name -> new sub-block, laid out like the site's value
            with each indexed axis replaced by the number of indices.
        :param dict index: plate name (or integer axis) -> index array (scalar or 1-D).
        """
        plates = tuple(index.keys())
        idxs = tuple(jnp.atleast_1d(jnp.asarray(index[p])) for p in plates)
        for i in idxs:
            if i.ndim != 1 or not jnp.issubdtype(i.dtype, jnp.integer):
                raise ValueError("Indices must be scalar or 1-D integer arrays.")
        names = tuple(values.keys())
        vals = tuple(jnp.asarray(values[n]) for n in names)
        ks = tuple(int(i.shape[0]) for i in idxs)
        return cls([("sliced", names, plates, ks, idxs + vals)])

    def __or__(self, other):
        return Change(self.entries + other.entries)

    @property
    def sites(self):
        return tuple(n for e in self.entries for n in e[1])

    # --- pytree
    def tree_flatten(self):
        leaves = [leaf for e in self.entries for leaf in e[4]]
        aux = tuple((e[0], e[1], e[2], e[3]) for e in self.entries)
        return leaves, aux

    @classmethod
    def tree_unflatten(cls, aux, leaves):
        leaves = list(leaves)
        entries = []
        for kind, names, plates, ks in aux:
            n = len(names) + len(plates)
            entries.append((kind, names, plates, ks, tuple(leaves[:n])))
            leaves = leaves[n:]
        return cls(entries)

    @property
    def signature(self):
        return self.tree_flatten()[1]

    def __repr__(self):
        parts = []
        for kind, names, plates, ks, _ in self.entries:
            if kind == "dense":
                parts.append(f"dense({names[0]})")
            else:
                idx = ", ".join(f"{p}[{k}]" for p, k in zip(plates, ks))
                parts.append(f"sliced({', '.join(names)}; {idx})")
        return "Change(" + "; ".join(parts) + ")"


def input_deltas(change, program):
    """Map ``change`` onto the program's site inputs; returns ``{site: Δ}``."""
    out = {}
    for g, (kind, names, plates, ks, leaves) in enumerate(change.entries):
        if kind == "dense":
            (name,) = names
            (value,) = leaves
            meta = program.sites[name]
            if tuple(value.shape) != tuple(meta.shape):
                raise ValueError(
                    f"Dense change for '{name}' has shape {value.shape}, expected "
                    f"{meta.shape}."
                )
            out[name] = Dense(value)
            continue
        idxs = leaves[: len(plates)]
        values = leaves[len(plates) :]
        token = ("change", g)
        for name, value in zip(names, values):
            meta = program.sites[name]
            plate_axes = dict(meta.plates)
            axes = []
            for p in plates:
                if isinstance(p, int):
                    axes.append(p % len(meta.shape))
                elif p in plate_axes:
                    axes.append(plate_axes[p])
                else:
                    raise ValueError(
                        f"Site '{name}' is not inside plate '{p}' "
                        f"(its plates are {[q for q, _ in meta.plates]})."
                    )
            expected = sliced_shape(meta.shape, axes, ks)
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"Sliced change for '{name}' has shape {value.shape}, expected "
                    f"{expected}."
                )
            out[name] = Sliced(tuple(axes), tuple(idxs), value, token)
    return out


def apply_change_to_values(values, deltas):
    """Return the new site values after applying ``deltas`` (``{site: Δ}``)."""
    new = dict(values)
    for name, d in deltas.items():
        if isinstance(d, Dense):
            new[name] = d.value
        elif isinstance(d, Sliced):
            new[name] = scatter_rect(values[name], d.axes, d.idxs, d.value)
    return new


def vmap_change(fn, change, in_axes=0):
    """Helper: ``jax.vmap`` a function of a :class:`Change` over batched new values."""
    return jax.vmap(fn, in_axes=in_axes)(change)


__all__ = [
    "SAME",
    "Same",
    "Sliced",
    "Dense",
    "Change",
    "gather_rect",
    "scatter_rect",
    "sliced_shape",
    "input_deltas",
    "apply_change_to_values",
    "lax",
]
