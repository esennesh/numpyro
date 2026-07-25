# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""Reproduces the counterexamples in "Fast, Soft MAP Estimation in Loopy
Graphical Models", in the min-sum section and the countable-sites subsection.

Five results, in the order they appear in the paper:

1. DAG-factorized models -- the setting of equation (1) -- of increasing
   cyclomatic number, including the parallel-factor pattern before and after
   fusion. This is the primary evidence, because it lives in the class the
   document is actually about.
2. A failure-rate table over pairwise MRFs of increasing cyclomatic number.
   These are minimal illustrations of the three failure modes rather than
   instances of equation (1); min-sum is exact on trees and on single-loop
   graphs, and failures begin at two independent cycles.
3. The frustrated triangle: a tied fixed point whose local-marginal-polytope LP
   relaxation is loose (bound 3 against a true optimum of 2, with half-integral
   edge marginals), so no fixed point could have decoded correctly.
4. An explicit K4 instance on which max-product converges to a confident but
   wrong assignment.
5. Discrete log-concavity, which every standard count exponential family enjoys
   per site, is not closed under max-marginalization, though it is closed under
   (max, +) convolution.

Run with ``python3 soft_map_counterexample.py``. Requires numpy and scipy only.
Numbers quoted in the paper are copied from this script's output.
"""

import itertools

import numpy as np
from scipy.optimize import linprog

DAMPING = 0.5
TOL = 1e-12
MAX_SWEEPS = 3000


def brute_force(n, edges, theta_edge, theta_node=None):
    """Exhaustive MAP over binary assignments. Returns (value, argmaxes, values)."""
    if theta_node is None:
        theta_node = [np.zeros(2)] * n
    values = {}
    for x in itertools.product([0, 1], repeat=n):
        values[x] = sum(theta_node[i][x[i]] for i in range(n)) + sum(
            theta_edge[k][x[i], x[j]] for k, (i, j) in enumerate(edges)
        )
    best = max(values.values())
    return best, [x for x, v in values.items() if v > best - 1e-9], values


def max_product(n, edges, theta_edge, theta_node=None, damping=DAMPING):
    """Damped parallel max-product in log space.

    Messages are indexed by ``(edge index, source node)`` and stored as arrays
    over the *target* node's states, max-normalized each sweep. Returns the
    node beliefs, the final message residual, and the sweep count.
    """
    if theta_node is None:
        theta_node = [np.zeros(2)] * n
    messages = {}
    for k, (i, j) in enumerate(edges):
        messages[(k, i)] = np.zeros(2)
        messages[(k, j)] = np.zeros(2)

    residual = np.inf
    for sweep in range(MAX_SWEEPS):
        updated = {}
        for k, (i, j) in enumerate(edges):
            for source in (i, j):
                # Sum of incoming messages from every edge except this one.
                incoming = theta_node[source].copy()
                for k2, (a, b) in enumerate(edges):
                    if k2 == k or source not in (a, b):
                        continue
                    incoming = incoming + messages[(k2, b if a == source else a)]
                potential = theta_edge[k] if source == i else theta_edge[k].T
                message = np.max(potential + incoming[:, None], axis=0)
                updated[(k, source)] = message - message.max()
        residual = max(np.abs(updated[key] - messages[key]).max() for key in messages)
        messages = {
            key: damping * messages[key] + (1 - damping) * updated[key]
            for key in messages
        }
        if residual < TOL:
            break

    beliefs = []
    for v in range(n):
        belief = theta_node[v].copy()
        for k, (a, b) in enumerate(edges):
            if v in (a, b):
                belief = belief + messages[(k, b if a == v else a)]
        beliefs.append(belief - belief.max())
    return np.array(beliefs), residual, sweep


def assignment_value(x, edges, theta_edge, theta_node=None):
    n = len(x)
    if theta_node is None:
        theta_node = [np.zeros(2)] * n
    return sum(theta_node[i][x[i]] for i in range(n)) + sum(
        theta_edge[k][x[i], x[j]] for k, (i, j) in enumerate(edges)
    )


def failure_rates(name, n, edges, n_trials=3000, seed=11):
    """Fraction of random instances on which max-product converges yet misdecodes."""
    rng = np.random.default_rng(seed)
    converged = tied = wrong = 0
    for _ in range(n_trials):
        theta_edge = [rng.normal(size=(2, 2)) for _ in edges]
        beliefs, residual, _ = max_product(n, edges, theta_edge)
        if residual > 1e-9:
            continue
        converged += 1
        if min(abs(b[0] - b[1]) for b in beliefs) < 1e-6:
            tied += 1
            continue
        decoded = tuple(int(np.argmax(b)) for b in beliefs)
        _, argmaxes, _ = brute_force(n, edges, theta_edge)
        if decoded not in argmaxes:
            wrong += 1
    print(
        f"  {name:30s} trials={n_trials}  converged={converged}  "
        f"tied={tied}  WRONG={wrong}"
    )


def frustrated_triangle_lp():
    """LP over the local marginal polytope L(G) for the frustrated triangle.

    Variables are the three node pseudomarginals followed by the three edge
    pseudomarginals; constraints are normalization plus edge-to-node marginal
    consistency. This is the relaxation whose vertices max-product's fixed
    points correspond to.
    """
    edges = [(0, 1), (1, 2), (0, 2)]
    theta_edge = [np.array([[0.0, 1.0], [1.0, 0.0]])] * 3
    n_node_vars, n_edge_vars = 6, 12
    n_vars = n_node_vars + n_edge_vars

    objective = np.zeros(n_vars)
    for k in range(3):  # linprog minimizes, so negate to maximize
        objective[n_node_vars + 4 * k : n_node_vars + 4 * k + 4] = -theta_edge[
            k
        ].ravel()

    rows, rhs = [], []
    for i in range(3):  # normalization: sum_a mu_i(a) = 1
        row = np.zeros(n_vars)
        row[2 * i] = row[2 * i + 1] = 1
        rows.append(row)
        rhs.append(1.0)
    for k, (i, j) in enumerate(edges):  # marginal consistency, both directions
        base = n_node_vars + 4 * k
        for a in range(2):
            row = np.zeros(n_vars)
            row[base + 2 * a] = row[base + 2 * a + 1] = 1
            row[2 * i + a] = -1
            rows.append(row)
            rhs.append(0.0)
        for b in range(2):
            row = np.zeros(n_vars)
            row[base + b] = row[base + 2 + b] = 1
            row[2 * j + b] = -1
            rows.append(row)
            rhs.append(0.0)

    result = linprog(
        objective, A_eq=np.array(rows), b_eq=np.array(rhs), bounds=[(0, 1)] * n_vars
    )
    beliefs, _, _ = max_product(3, edges, theta_edge)
    decoded = tuple(int(np.argmax(b)) for b in beliefs)
    true_best, _, _ = brute_force(3, edges, theta_edge)

    print("  potentials: theta_ij(x_i, x_j) = 1 if x_i != x_j else 0")
    print(f"  max-product beliefs: {np.round(beliefs, 6).tolist()} (exactly uniform)")
    print(
        f"  greedy decode {decoded} -> value "
        f"{assignment_value(decoded, edges, theta_edge):.1f}   "
        f"true MAP value {true_best:.1f}"
    )
    print(
        f"  local-polytope LP bound = {-result.fun:.3f}  vs  true MAP = "
        f"{true_best:.1f}  (relaxation is loose)"
    )
    print(
        f"  LP edge marginals for edge (0,1): "
        f"{np.round(result.x[6:10], 3).tolist()}  (half-integral)"
    )


# The instance quoted in the paper, found by random search over K4 potentials
# and reproduced here verbatim so the result does not depend on RNG details.
K4_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
K4_POTENTIALS = [
    np.array([[-0.9, -1.2], [1.6, -0.2]]),  # theta_01
    np.array([[1.5, -0.8], [-2.5, -0.8]]),  # theta_02
    np.array([[-0.2, 1.0], [0.1, -0.8]]),  # theta_03
    np.array([[-2.1, 2.0], [-0.8, -0.3]]),  # theta_12
    np.array([[0.0, -1.6], [-0.9, -0.8]]),  # theta_13
    np.array([[-0.6, 2.6], [-0.1, -2.5]]),  # theta_23
]


def k4_confident_but_wrong():
    """Max-product converges to a confident, incorrect assignment on K4."""
    beliefs, residual, sweeps = max_product(4, K4_EDGES, K4_POTENTIALS)
    decoded = tuple(int(np.argmax(b)) for b in beliefs)
    decoded_value = assignment_value(decoded, K4_EDGES, K4_POTENTIALS)
    true_best, argmaxes, _ = brute_force(4, K4_EDGES, K4_POTENTIALS)
    gaps = [round(float(abs(b[0] - b[1])), 1) for b in beliefs]

    for k, (i, j) in enumerate(K4_EDGES):
        print(f"  theta_{i}{j} = {K4_POTENTIALS[k].tolist()}")
    print(f"  converged in {sweeps} sweeps (residual {residual:.1e})")
    print(f"  belief log-odds gaps: {gaps} nats (confident)")
    print(f"  max-product decode {decoded} -> value {decoded_value:.2f}")
    print(f"  true MAP           {argmaxes[0]} -> value {true_best:.2f}")


# ---------------------------------------------------------------------------
# DAG-factorized models: general-arity factor graphs built from conditionals.
# This is the setting of equation (1); the pairwise routines above are not.
# ---------------------------------------------------------------------------


def log_cpt(rng, arity, k=2):
    """A random ``log p(child | parents)``, normalized over the child axis."""
    table = rng.normal(size=(k,) * arity) * 1.5
    return table - np.log(np.exp(table).sum(axis=0, keepdims=True))


def factor_graph_cyclomatic(n_vars, factors):
    """Cyclomatic number of the bipartite factor graph."""
    nodes = n_vars + len(factors)
    edges = sum(len(variables) for variables, _ in factors)
    return edges - nodes + 1


def max_product_factor_graph(n_vars, factors, k=2, damping=DAMPING):
    """Damped parallel max-product over a general-arity factor graph.

    ``factors`` is a list of ``(variables, log_table)`` pairs whose table has
    shape ``(k,) * len(variables)``. Returns the variable beliefs and the final
    message residual.
    """
    to_factor = {}
    to_variable = {}
    for a, (variables, _) in enumerate(factors):
        for v in variables:
            to_factor[(v, a)] = np.zeros(k)
            to_variable[(a, v)] = np.zeros(k)
    neighbours = {
        v: [a for a, (variables, _) in enumerate(factors) if v in variables]
        for v in range(n_vars)
    }

    residual = np.inf
    for _ in range(MAX_SWEEPS):
        new_to_factor = {}
        for v in range(n_vars):
            for a in neighbours[v]:
                total = np.zeros(k)
                for b in neighbours[v]:
                    if b != a:
                        total = total + to_variable[(b, v)]
                new_to_factor[(v, a)] = total - total.max()
        new_to_variable = {}
        for a, (variables, table) in enumerate(factors):
            for v in variables:
                extended = table.copy()
                for axis, u in enumerate(variables):
                    if u == v:
                        continue
                    shape = [1] * len(variables)
                    shape[axis] = k
                    extended = extended + new_to_factor[(u, a)].reshape(shape)
                others = tuple(axis for axis, u in enumerate(variables) if u != v)
                message = extended.max(axis=others) if others else extended
                new_to_variable[(a, v)] = message - message.max()
        residual = max(
            max(np.abs(new_to_factor[key] - to_factor[key]).max() for key in to_factor),
            max(
                np.abs(new_to_variable[key] - to_variable[key]).max()
                for key in to_variable
            ),
        )
        to_factor = {
            key: damping * to_factor[key] + (1 - damping) * new_to_factor[key]
            for key in to_factor
        }
        to_variable = {
            key: damping * to_variable[key] + (1 - damping) * new_to_variable[key]
            for key in to_variable
        }
        if residual < TOL:
            break

    beliefs = []
    for v in range(n_vars):
        belief = np.zeros(k)
        for a in neighbours[v]:
            belief = belief + to_variable[(a, v)]
        beliefs.append(belief - belief.max())
    return np.array(beliefs), residual


def factor_graph_score(x, factors):
    return sum(table[tuple(x[u] for u in variables)] for variables, table in factors)


def factor_graph_optimum(n_vars, factors, k=2):
    return max(
        factor_graph_score(x, factors)
        for x in itertools.product(range(k), repeat=n_vars)
    )


def _parallel_factors(rng, fuse):
    """Three latent children sharing the parent set {0, 1}, each with a likelihood.

    Unfused, the three conditionals give the factor graph two independent cycles.
    Fused into one factor over all five variables -- which is what parallel-factor
    fusion does in the string-diagram representation -- the graph becomes a tree.
    """
    priors = [((0,), log_cpt(rng, 1)), ((1,), log_cpt(rng, 1))]
    conditionals = [log_cpt(rng, 3) for _ in range(3)]
    likelihoods = [rng.normal(size=(2,)) for _ in range(3)]
    if not fuse:
        factors = list(priors)
        for offset, child in enumerate((2, 3, 4)):
            factors.append(((child, 0, 1), conditionals[offset]))
            factors.append(((child,), likelihoods[offset]))
        return 5, factors
    fused = np.zeros((2,) * 5)
    for index in itertools.product(range(2), repeat=5):
        i0, i1 = index[0], index[1]
        fused[index] = sum(
            conditionals[offset][index[2 + offset], i0, i1]
            + likelihoods[offset][index[2 + offset]]
            for offset in range(3)
        )
    return 5, priors + [((0, 1, 2, 3, 4), fused)]


def _lattice_dag(rng, side=3):
    """A ``side``-by-``side`` lattice DAG whose parents are the cells above and left.

    No two conditionals share a parent set, so parallel-factor fusion has nothing
    to merge, and the factor graph's cycles survive it untouched.
    """
    index = {(i, j): i * side + j for i in range(side) for j in range(side)}
    factors = []
    for i in range(side):
        for j in range(side):
            parents = []
            if i > 0:
                parents.append(index[(i - 1, j)])
            if j > 0:
                parents.append(index[(i, j - 1)])
            factors.append(
                (tuple([index[(i, j)]] + parents), log_cpt(rng, 1 + len(parents)))
            )
    return side * side, factors


def dag_failure_rates(name, build, n_trials=1500, seed=0):
    """Max-product failure rate on a family of DAG-factorized models."""
    rng = np.random.default_rng(seed)
    converged = tied = wrong = 0
    cyclomatic = None
    for _ in range(n_trials):
        n_vars, factors = build(rng)
        if cyclomatic is None:
            cyclomatic = factor_graph_cyclomatic(n_vars, factors)
        beliefs, residual = max_product_factor_graph(n_vars, factors)
        if residual > 1e-9:
            continue
        converged += 1
        if min(abs(b[0] - b[1]) for b in beliefs) < 1e-6:
            tied += 1
            continue
        decoded = tuple(int(np.argmax(b)) for b in beliefs)
        if (
            factor_graph_score(decoded, factors)
            < factor_graph_optimum(n_vars, factors) - 1e-9
        ):
            wrong += 1
    print(
        f"  {name:44s} cyclomatic={cyclomatic}  converged={converged}  "
        f"tied={tied}  WRONG={wrong}"
    )


def _concave_sequence(seq, tol=1e-9):
    """Whether a real sequence has nonincreasing first differences."""
    s = np.asarray(seq, float)
    s = s[np.isfinite(s)]
    return len(s) < 3 or bool(np.all(np.diff(s, 2) <= tol))


def log_concavity_under_max_marginalization(n=60):
    """Per-site log-concavity does not survive max-marginalization.

    Each factor is ``log p(z | z')`` for a Poisson child whose natural parameter
    is affine in the parent, ``eta = a * z' + b``. This is a genuine exponential
    family and the factor is log-concave in each argument separately, yet the
    max-marginal over the parent is frequently not log-concave: the bilinear
    coupling ``a * z * z'`` makes partial maximization a concave conjugation,
    which contributes a convex term in ``z``.
    """
    from scipy.special import gammaln

    z = np.arange(n)
    parent = np.arange(n)
    configs = [
        (-0.15, 1.5, 0.0, -0.02),
        (-0.08, 2.0, 0.3, -0.01),
        (0.05, 0.5, 0.0, -0.05),
        (-0.30, 3.0, 0.0, -0.001),
    ]
    broken = 0
    for a, b, slope, curv in configs:
        eta = a * parent + b
        incoming = slope * parent + curv * parent**2  # concave in the parent
        joint = (
            np.outer(z, eta) - np.exp(eta)[None, :] - gammaln(z + 1)[:, None]
        ) + incoming[None, :]
        in_z = all(_concave_sequence(joint[:, j]) for j in range(joint.shape[1]))
        in_parent = all(_concave_sequence(joint[i, :]) for i in range(joint.shape[0]))
        marginal = joint.max(axis=1)
        ok = _concave_sequence(marginal)
        note = ""
        if not ok:
            broken += 1
            d2 = np.diff(marginal[np.isfinite(marginal)], 2)
            note = f"  worst 2nd difference {d2.max():+.4f} at z={int(np.argmax(d2))}"
        print(
            f"  a={a:+.2f} b={b:+.2f}: factor log-concave in z={in_z}, "
            f"in parent={in_parent} -> max-marginal log-concave={ok}{note}"
        )
    print(f"  {broken} of {len(configs)} configurations lose log-concavity")

    # Convolution-structured coupling: z = parent + Poisson increment.
    print("\n  convolution-structured coupling (z = z' + increment):")
    for mu, slope, curv in [(3.0, 0.2, -0.01), (8.0, 0.0, -0.05), (1.0, 0.5, -0.002)]:
        increment = z * np.log(mu) - mu - gammaln(z + 1)
        incoming = slope * parent + curv * parent**2
        marginal = np.array(
            [
                max(
                    (
                        incoming[j] + increment[t - parent[j]]
                        for j in range(n)
                        if 0 <= t - parent[j] < n
                    ),
                    default=-np.inf,
                )
                for t in z
            ]
        )
        print(
            f"    mu={mu}: incoming concave={_concave_sequence(incoming)} -> "
            f"(max,+) convolution concave={_concave_sequence(marginal)}"
        )


if __name__ == "__main__":
    print("\n1. DAG-factorized models: the setting of equation (1)")
    print(f"   (damping {DAMPING}, tol {TOL:.0e}, random normalized conditionals)\n")
    print("   parallel factors over an identical parent set -- the fusion target:")
    dag_failure_rates(
        "three children sharing parents, UNFUSED",
        lambda rng: _parallel_factors(rng, fuse=False),
        n_trials=12000,
    )
    dag_failure_rates(
        "the same model, parallel factors FUSED",
        lambda rng: _parallel_factors(rng, fuse=True),
        n_trials=12000,
    )
    print("\n   overlapping but distinct parent sets -- nothing to fuse:")
    dag_failure_rates("3x3 lattice DAG (parents above and left)", _lattice_dag, 1200)

    print("\n2. Pairwise MRFs: failure rate by cyclomatic number")
    print("   (minimal illustrations, not instances of equation (1))\n")
    failure_rates("path 0-1-2-3 (0 cycles)", 4, [(0, 1), (1, 2), (2, 3)])
    failure_rates("4-cycle (1 cycle)", 4, [(0, 1), (1, 2), (2, 3), (0, 3)])
    failure_rates(
        "K4 minus an edge (2 cycles)", 4, [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)]
    )
    failure_rates("K4 (3 cycles)", 4, K4_EDGES)

    print("\n3. Frustrated triangle: tied fixed point, loose LP relaxation\n")
    frustrated_triangle_lp()

    print("\n4. K4: converged, confident, and wrong\n")
    k4_confident_but_wrong()

    print("\n5. Log-concavity is not closed under max-marginalization\n")
    log_concavity_under_max_marginalization()
    print()
