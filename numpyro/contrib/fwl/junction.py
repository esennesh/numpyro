# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Junction-tree construction over the continuous latent sites.

Used by two parts of the procedure. :mod:`~numpyro.contrib.fwl.find` eliminates
cliques by nested minimization, and :mod:`~numpyro.contrib.fwl.weigh` builds a
proposal precision as a sum of clique-supported blocks.

The moral graph is built directly from factor scopes -- for each model site, its
own continuous latent (if any) together with its continuous latent parents form a
clique -- which is what guarantees the invariant the whole construction rests on:
every factor's scope is contained in at least one clique, so every factor can be
assigned to exactly one clique and evaluated there.

Everything here is static Python operating on site names, computed once from the
model's dependency structure. No arrays and no tracing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from numpyro.contrib.fwl.structure import ModelStructure


def factor_scopes(structure: ModelStructure) -> dict[str, frozenset[str]]:
    """
    Map each model site to its continuous latent scope: itself, if it is a
    continuous latent, together with the continuous latents it depends on
    directly. ``get_dependencies`` substitutes every latent value before tracking
    provenance, so its dependencies are the Markov parents rather than the
    transitive ancestors -- exactly the scope of the site's factor.
    """
    continuous = frozenset(structure.continuous)
    scopes = {}
    for name, upstream in structure.dependencies["prior_dependencies"].items():
        scopes[name] = frozenset({name} | set(upstream)) & continuous
    return scopes


def moral_graph(
    scopes: dict[str, frozenset[str]], continuous: tuple[str, ...]
) -> dict[str, set[str]]:
    """Adjacency over the continuous latents: a clique per factor scope."""
    adjacency: dict[str, set[str]] = {name: set() for name in continuous}
    for scope in scopes.values():
        for a in scope:
            adjacency[a] |= scope - {a}
    return adjacency


def triangulate(
    adjacency: dict[str, set[str]],
) -> tuple[tuple[str, ...], tuple[frozenset[str], ...]]:
    """
    Greedy min-fill triangulation.

    :return: a perfect elimination ordering and the maximal cliques it induces.
        Ties are broken by name so the result is reproducible.
    """
    remaining = {name: set(neighbours) for name, neighbours in adjacency.items()}
    order: list[str] = []
    cliques: list[frozenset[str]] = []

    def fill_count(name: str) -> int:
        live = remaining[name] & remaining.keys()
        return sum(1 for a in live for b in live if a < b and b not in remaining[a])

    while remaining:
        chosen = min(remaining, key=lambda name: (fill_count(name), name))
        live = remaining[chosen] & remaining.keys()
        cliques.append(frozenset({chosen} | live))
        for a in live:  # add the fill-in edges
            remaining[a] |= live - {a}
        order.append(chosen)
        del remaining[chosen]
        for neighbours in remaining.values():
            neighbours.discard(chosen)

    maximal: list[frozenset[str]] = []
    for clique in sorted(cliques, key=lambda c: (-len(c), sorted(c))):
        if not any(clique <= kept for kept in maximal):
            maximal.append(clique)
    return tuple(order), tuple(maximal)


def _spanning_tree(cliques: tuple[frozenset[str], ...]) -> list[tuple[int, int]]:
    """
    Maximum-weight spanning tree with weight ``|Ci ∩ Cj|``, which is what gives
    the running intersection property. Zero-weight edges are allowed, so a model
    whose latents split into independent groups still yields one tree, joined by
    empty separators.

    Ties are broken toward the shallowest attachment point. Many models admit
    several maximum-weight trees -- a star of cliques all sharing one variable
    admits every tree on them -- and since nested elimination costs
    (solver steps) ** height, preferring a bushy tree over a path is worth the
    two extra lines.
    """
    if len(cliques) == 1:
        return []
    connected, edges = {0}, []
    depth = {0: 0}
    while len(connected) < len(cliques):
        weight, _, source, target = max(
            (
                (len(cliques[i] & cliques[j]), -depth[i], i, j)
                for i in connected
                for j in range(len(cliques))
                if j not in connected
            ),
        )
        del weight
        edges.append((source, target))
        connected.add(target)
        depth[target] = depth[source] + 1
    return edges


def _tree_center(num_cliques: int, edges: list[tuple[int, int]]) -> int:
    """
    The center of the tree, rooting at which minimizes height. Nested
    elimination costs (solver steps) ** height, so this is not cosmetic.
    """
    if num_cliques == 1:
        return 0
    neighbours: dict[int, set[int]] = {i: set() for i in range(num_cliques)}
    for i, j in edges:
        neighbours[i].add(j)
        neighbours[j].add(i)
    remaining = set(range(num_cliques))
    while len(remaining) > 2:
        leaves = [i for i in remaining if len(neighbours[i] & remaining) <= 1]
        remaining -= set(leaves)
    return min(remaining)


@dataclass(frozen=True)
class CliqueTree:
    """
    A rooted junction tree over the continuous latent sites.

    :param cliques: the maximal cliques, as frozensets of site names.
    :param parent: parent clique index, or ``None`` for the root.
    :param children: child clique indices.
    :param separator: ``C ∩ parent(C)``, empty for the root. The message a clique
        sends upward is a function of these variables only.
    :param interior: ``C \\ separator(C)``. These partition the latent sites: each
        site lies in the interior of exactly one clique, namely the highest clique
        containing it, which is why per-clique terms sum to the whole energy
        without double counting.
    :param factors: model site names assigned to each clique. Every site appears
        exactly once, so summing clique energies gives the total energy.
    :param post_order: clique indices, children before parents.
    :param elimination_order: the perfect elimination ordering from triangulation.
    :param height: longest root-to-leaf path, i.e. the nesting depth.
    """

    cliques: tuple[frozenset[str], ...]
    parent: tuple[Optional[int], ...]
    children: tuple[tuple[int, ...], ...]
    separator: tuple[frozenset[str], ...]
    interior: tuple[frozenset[str], ...]
    factors: tuple[tuple[str, ...], ...]
    post_order: tuple[int, ...]
    elimination_order: tuple[str, ...]
    height: int
    root: int

    def summary(self) -> str:
        """A human-readable description of the tree."""
        lines = [
            f"{len(self.cliques)} cliques, height {self.height}, root {self.root}",
        ]
        for i in self.post_order:
            lines.append(
                f"  clique {i}: {sorted(self.cliques[i])}"
                f" | separator {sorted(self.separator[i])}"
                f" | interior {sorted(self.interior[i])}"
                f" | factors {list(self.factors[i])}"
            )
        return "\n".join(lines)


def build_clique_tree(structure: ModelStructure) -> CliqueTree:
    """
    Build the rooted junction tree over ``structure``'s continuous latent sites,
    rooted to minimize height.
    """
    scopes = factor_scopes(structure)
    if not structure.continuous:
        # Nothing to eliminate; see ModelStructure.enumeration_only.
        return CliqueTree(
            cliques=(),
            parent=(),
            children=(),
            separator=(),
            interior=(),
            factors=(),
            post_order=(),
            elimination_order=(),
            height=0,
            root=-1,
        )
    adjacency = moral_graph(scopes, structure.continuous)
    elimination_order, cliques = triangulate(adjacency)
    edges = _spanning_tree(cliques)
    root = _tree_center(len(cliques), edges)

    neighbours: dict[int, set[int]] = {i: set() for i in range(len(cliques))}
    for i, j in edges:
        neighbours[i].add(j)
        neighbours[j].add(i)

    parent: list[Optional[int]] = [None] * len(cliques)
    children: list[list[int]] = [[] for _ in cliques]
    depth = [0] * len(cliques)
    post_order: list[int] = []
    visited = {root}
    stack = [root]
    while stack:  # pre-order walk, recording parents and depths
        current = stack.pop()
        post_order.append(current)
        for neighbour in sorted(neighbours[current]):
            if neighbour in visited:
                continue
            visited.add(neighbour)
            parent[neighbour] = current
            children[current].append(neighbour)
            depth[neighbour] = depth[current] + 1
            stack.append(neighbour)
    post_order.reverse()  # children before parents

    separators: list[frozenset[str]] = []
    for i in range(len(cliques)):
        above = parent[i]
        separators.append(frozenset() if above is None else cliques[i] & cliques[above])
    separator = tuple(separators)
    interior = tuple(cliques[i] - separator[i] for i in range(len(cliques)))

    # Assign each factor to the smallest clique containing its whole scope. A
    # factor with no continuous scope contributes a constant, and goes to the
    # root so that the root's objective is the entire energy.
    assigned: list[list[str]] = [[] for _ in cliques]
    for name in sorted(scopes):
        scope = scopes[name]
        if not scope:
            assigned[root].append(name)
            continue
        candidates = [i for i, clique in enumerate(cliques) if scope <= clique]
        if not candidates:  # pragma: no cover - excluded by the moral construction
            raise RuntimeError(
                f"Factor '{name}' has scope {sorted(scope)}, which no clique "
                "contains. This means the moral graph and the cliques disagree."
            )
        assigned[min(candidates, key=lambda i: (len(cliques[i]), i))].append(name)

    return CliqueTree(
        cliques=cliques,
        parent=tuple(parent),
        children=tuple(tuple(c) for c in children),
        separator=separator,
        interior=interior,
        factors=tuple(tuple(names) for names in assigned),
        post_order=tuple(post_order),
        elimination_order=elimination_order,
        height=max(depth) if depth else 0,
        root=root,
    )
