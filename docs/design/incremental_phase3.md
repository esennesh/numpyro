# Incremental densities — phase 3 brief

A self-contained task brief for a fresh session (any machine). Start from branch
`feature/incremental`; read `docs/design/incremental.md` first, it is the living plan
with the architecture, decisions log and benchmarks.

## Background

`numpyro/contrib/incremental/` implements incremental density computation after
Zaiser et al., *Incremental Computation for Efficient Programmable Inference in
Probabilistic Programs* (PLDI 2026, doi:10.1145/3808316):

- `density.py` compiles a model to a density jaxpr (inputs: latent site values,
  continuous ones unconstrained, plus array-valued model arguments; outputs: per-factor
  unsummed log densities).
- `rules.py` / `interpreter.py` propagate typed changes (`Same` / `Sliced` / `Dense`)
  through that jaxpr with per-primitive rules and pullbacks; `cache="plan"` caches only
  cut points. `apply` is an ordinary JAX function (`jit`/`vmap`/`grad` compose).
- `adapt.py` runs existing kernels as block steps on the incremental density
  (`IncrementalMCMC`: HMC/NUTS/SA via `potential_fn_gen`, `DiscreteGibbs`, `GibbsFn`,
  `RandomWalk`; `IncrementalMCMC.from_kernel` adapts `DiscreteHMCGibbs`/`HMCGibbs`).
- Tests: `test/contrib/incremental/` (117 passing; run with
  `JAX_PLATFORM_NAME=cpu .venv/bin/python -m pytest test/contrib/incremental -q`).
  Pre-commit hooks need `.venv/bin` on `PATH`; codespell rejects two-letter variable
  names that look like typos of common words, so spell identifiers out.
- Examples: `examples/incremental_gmm.py`, `examples/incremental_case_studies.py`
  (the paper's case studies 1–2 with the artifact's exact models).

Conventions: one commit per step, subjects prefixed `numpyro.contrib.incremental:`;
keep `docs/design/incremental.md` current (progress lines, decisions log, benchmarks).
Build on JAX program transformations where possible (the user's standing direction).

Known constraints discovered so far (details in the decisions log):

- JAX shares one `Jaxpr` object between call sites, so variables are keyed by
  `(call path, var)`.
- XLA CPU copies a loop-carried buffer on every iteration when the body both updates
  it in place and reads a value from it that escapes into another carried element.
  Sweeps therefore read from a pre-loop snapshot (`apply(..., read_values=...)`), exact
  when the swept site is only read at the swept position (`input_full_reads`).
- `scan`/`while`/`cond` and unknown primitives fall back to dense recomputation
  (exact, O(T)); a `gather` whose *operand* changes along an indexed dim is dense.

## The three items

1. **A `scan_p` rule** (`rules.py`). For the HMM zoo model
   (`test/contrib/incremental/zoo.py::hmm`, built with `numpyro.contrib.control_flow.scan`)
   a change to `x[t]` should re-execute only the scan steps whose inputs changed.
   Analyse the body jaxpr statically: when the carry at step t+1 is a function of the
   substituted/sliced `xs[t+1]` only (the carry is overwritten by substituted site
   values, which is how `scan` with substituted sites works), steps t and t+1 are the
   only affected ones; otherwise keep the dense fallback. Verify with
   `check_consistency`, and extend `test_cost.py` with the HMM: the compiled flop count
   of a single-time-step change must not grow with T.
2. **Data-dependent `gather` sparsity** (the paper's `using` clause). When the operand
   is `Sliced` at index k and the indices are `Same`, the affected output positions
   are those whose index equals k. Emit a masked update (`jnp.where(indices == k,
   new, old)` on the gathered rows) instead of a dense recompute, exact under
   `check_consistency`. This is the "update component mean `mu_k`" change of the GMM
   (`zoo.py::gmm`, `Change.sliced({"mu": ...}, {"comp": k})`).
3. **A `memo(name, fn, *args)` primitive (or handler)** for expensive latent-independent
   intermediates: record the intermediate as a program input computed once at
   `IncrementalDensity.init`, so dense changes never recompute it; test with a model
   that computes `K = kernel_matrix(X)` from data only.

## Done means

All three have unit tests in `test/contrib/incremental/`, the whole suite passes,
`docs/design/incremental.md` records progress and decisions, and the branch is ready
for review (no PR unless asked).

## Later still

`MixedHMC` and `BarkerMH` adaptation; masked fixed-capacity data growth (SMC-style);
a `cache="plan"` memory bound; upstreaming to pyro-ppl/numpyro.
