# Incremental density computation in NumPyro — project plan

Target: Zaiser et al., *Incremental Computation for
Efficient Programmable Inference in Probabilistic Programs*, PACMPL 10 (PLDI) 238, 2026,
doi:10.1145/3808316 (arXiv 2606.05348). The paper (1) compiles a probabilistic program
to a deterministic *density program* and (2) *incrementalizes* that program against
typed change descriptions, so an *updater* with caches recomputes the density after a
small trace change in time proportional to the change rather than to the dataset.
Metropolis–Hastings does one `apply` per proposal and keeps the old updater on
rejection; Gibbs applies once per candidate value; loops re-run only touched iterations.
Isolating incrementalization in a deterministic language is what makes soundness
modular.

Our reading in JAX: stage 1 is what NumPyro's effect handlers plus `jax.make_jaxpr`
already produce. Stage 2 is a jaxpr-to-jaxpr transformation with per-primitive
*change-propagation rules*, built like `numpyro.ops.provenance.eval_provenance` and like
`jax.vmap`'s batching rules. Plates are the paper's loops; the plate axis of a site's
array is the loop index. A change to one element of a plated site propagates as a
*sliced* change through elementwise ops, becomes a cheap *delta* at `reduce_sum`, and
degrades to *dense* only where the jaxpr genuinely mixes elements (reductions feeding
back, `scan`, unknown primitives). Because the sliced program is emitted by us, locality
never depends on XLA dead-code elimination and the per-step cost class is statically
known and reportable.

Home: branch `feature/incremental` off `develop`, package `numpyro/contrib/incremental/`,
one commit per step below, subjects prefixed `numpyro.contrib.incremental:`.

## Why

Every MCMC kernel in NumPyro gets its density from one closure that replays the whole
model and sums every site (`numpyro/infer/util.py:log_density`). `DiscreteHMCGibbs`
updates one discrete coordinate by calling the full potential once per support value,
so a Gibbs sweep over N mixture assignments does O(N²·K) work. Nothing recovers
sub-site (plate-element) locality.

## What the codebase already gives us

- **Density program.** `compute_log_probs(..., sum_log_prob=False)` returns per-site
  *unsummed* `log_prob` arrays: the paper's per-iteration cache values.
  `_unconstrain_reparam` emits a `_{name}_log_det` factor per continuous latent: just
  another output, attributed to its owner site.
- **Jaxpr interpretation pattern.** `numpyro/ops/provenance.py`: env walk over
  `jaxpr.eqns`, rule registry, recursion into call primitives, `jax.extend.core` only.
  Rebinding recipe (what `jax.core.eval_jaxpr` does): `bp =
  eqn.primitive.get_bind_params(eqn.params)`; `with eqn.ctx.manager:
  eqn.primitive.bind(*ins, **bp)`.
- **Plate axes.** Each site message's `cond_indep_stack` frames `(name, dim, size)` map
  a plate to an array axis: `len(fn.batch_shape) + dim` for values and log-probs.
- **Static support sizes.** `fn.enumerate_support(expand=False)` has a static leading
  size, so Gibbs candidates can be `vmap`ped; the `fori_loop` over support in
  `_discrete_gibbs_proposal` only exists because all discrete sites are flattened into
  one vector.
- **Kernel rebinding.** `HMC._init_state` leaves a preset `_init_fn`/`_sample_fn` alone
  when built from a potential; `hmc(potential_fn_gen=...)` re-evaluates the generator in
  `init_kernel` and every `sample_kernel`; `HMCGibbs` passes state through
  `model_kwargs["_gibbs_sites"]`. `SA._init_state` needs a one-line fix to honour a
  preset `_init_fn`. `BarkerMH` binds its potential once at init: out of scope.
- **Not needed:** `numpyro.subsample`/`plate(subsample_size)`; HMCECS proxies; funsor;
  `get_dependencies` (kept only for `summary()`: the Δ-interpreter derives exact
  per-array dependence itself).

## Architecture

1. `density.py` — `trace_density(model, args, kwargs)` builds the density program:
   invars = latent site values (continuous in unconstrained space, discrete raw) plus
   array-valued model args; outvars = per-factor unsummed `log_prob` arrays
   (`_x_log_det` factors attributed to `x`). Metadata per site/factor: shape, plate
   axes, support, transform, support size. Validation off by default. Non-array model
   args are static.
2. `delta.py` — Δ-values `Same | Sliced(axes, idxs, value, token) | Dense(value)`;
   rectangular `gather_rect`/`scatter_rect`; the `Change` pytree (site names, kinds,
   plate names and slice lengths are static aux data; indices and values are leaves).
3. `rules.py` + `interpreter.py` — the Δ-transformation: an interpreter run under
   tracing, each var tagged with a Δ-value. Elementwise, shape ops, `reduce_sum`
   (delta rule over a sliced axis), `gather` (sliced indices, batching dims), call
   primitives recurse with the Δ-rules (never `eval_jaxpr`-inlined: inner shapes are
   specialised), `custom_jvp_call` rebinds whole on the dense path and uses a registry on
   the sliced path, control flow and unknown primitives always materialise and rebind.
   `apply(z, args, cache, change) -> (delta, factor_slices, cache_updates)` is an
   ordinary JAX function: `jit`/`vmap`/`grad` apply.
4. `adapt.py` — `IncrementalMCMC(model, steps)` runs existing kernels as block steps:
   HMC/NUTS/SA rebound to a cache-dependent `potential_fn_gen` (blanket-local gradient
   falls out of Δ-propagation), and the `DiscreteHMCGibbs` proposal functions driven by
   a sliced incremental potential. `IncrementalMCMC.from_kernel(DiscreteHMCGibbs(NUTS(m)))`
   adapts a composite kernel directly.

Invariants: MH ratios come from per-factor slice differences, never from a running
total; the total is diagnostic and periodically resynced. Rejection keeps the old cache
pytree. Gibbs computes K deltas under `vmap` and writes the chosen candidate's factor
slices once.

## Steps and progress

1. ✅ Design doc, package skeleton, `density.py`, model zoo, `test_density.py`.
2. ✅ `delta.py`, `rules.py` (elementwise, shape ops, `iota`, `gather`, call recursion,
   default materialise), `interpreter.py` with `cache="all"`; cost-constancy test
   (2026-10-22: one GMM assignment update compiles to the same flop count at
   N = 100, 1 000 and 10 000). Discovered: a value crossing a call boundary must have
   one cache id, or a skipped call reads a stale outer copy; `number_vars` aliases
   statically.
3. ✅ `reduce_sum` delta rule (sliced axes that are not reduced stay sliced; a
   non-finite result falls back to recomputation under `lax.cond`, which becomes a
   `select` under `vmap`), `check_consistency`, gradient equality on the dense path.
4. ✅ `custom_jvp`: registry (`xlogy`, `xlog1py`, `logaddexp`) re-called on slices,
   recursion into the primal body otherwise, whole rebinding on the dense path.
5. ✅ Slice pullback and `cache="plan"` (`plan`, `plan_site_changes`; an unplanned
   signature raises). Discovered: JAX shares one `Jaxpr` object between call sites of
   the same function, so variables are keyed by `(call path, var)`; and a cut point
   inside a body that is rebound whole on the dense path must be refreshed
   (`Context.refresh_body`), or it goes stale.
6. ✅ `adapt.py` (`IncrementalMCMC`, `PotentialBlock` for HMC/NUTS/SA, `DiscreteGibbs`
   reusing the `hmc_gibbs` proposal functions, `GibbsFn`, `from_kernel`) and the
   `sa.py` one-liner. Posterior checks: 1-D mixture analytic moments at the
   tolerances of `test_discrete_gibbs_gmm_1d`; plated GMM agrees with
   `DiscreteHMCGibbs`; exact conjugate posterior for a `gibbs_fn` block; vectorized
   chains; total in sync with from-scratch evaluation after a run with `resync_every`.
7. ✅ Docs (`contrib.rst`), `examples/incremental_gmm.py`, benchmark below.
8. ✅ Paper case studies 1 and 2 (`examples/incremental_case_studies.py`), the
   `RandomWalk` block, and the sweep snapshot fix that makes single-site updates O(1)
   in practice, not only in the emitted program.

## Benchmarks (2026-10-22, CPU)

**`examples/incremental_gmm.py`, K = 3.** `DiscreteHMCGibbs(NUTS(model))` as is versus
the same kernel through `IncrementalMCMC.from_kernel`; wall time of a compiled run of
600 steps (300 warmup); posteriors agree to 0.005 at N = 400 and 0.001 at N = 8 000
(at N = 2 000 the reference's short run sits in a wide-component local mode):

| N | DiscreteHMCGibbs | incremental |
|---|---|---|
| 400 | 1.13 s | 1.28 s |
| 2 000 | 8.17 s | 5.28 s |
| 8 000 | 114.7 s | 19.3 s |

One assignment update compiles to 23 flops with 18 of 124 equations re-executed at
N = 400, 2 000, 4 000 and 20 000. The discrete sweep is linear in N for the incremental
kernel and quadratic for the reference; the NUTS block is linear in both, so the
crossover sits around N ≈ 1 000 on CPU and the gap widens from there. Constant factors
dominate at small N, as the paper also reports for Gen.

**Paper case studies 1 and 2, `examples/incremental_case_studies.py`.** Robust
regression (Box & Tiao outliers, N = 1 000) and the binary 2-D Gaussian mixture
(N = 500), with the paper's kernels: Gaussian-drift Metropolis on the parameters
(`RandomWalk`) and single-site updates of the indicators / assignments
(`DiscreteGibbs`). Both recover the truth (regression: slope 1.99, intercept −1.01,
σ 0.49, 97.7 % of indicators; mixture: both means and scales, the weight, every
assignment). Median µs per kernel application inside one compiled sweep (incremental
`apply` with cache write vs full recomputation of the joint density); the paper's
Julia prototype reports 17 µs vs 3 060 µs for an outlier flip, 28 µs vs 1 680 µs for an
assignment, and parameter moves at 0.5× of full recomputation:

| kernel | N | incremental | full | speedup |
|---|---|---|---|---|
| outlier status | 1 000 | 0.03 µs | 1.20 µs | 45× |
| outlier status | 10 000 | 0.01 µs | 12.35 µs | 1 264× |
| regression parameters | 1 000 | 4.10 µs | 1.69 µs | 0.4× |
| regression parameters | 10 000 | 16.6 µs | 14.5 µs | 0.9× |
| cluster assignment | 500 | 0.02 µs | 1.23 µs | 51× |
| cluster assignment | 5 000 | 0.02 µs | 91.1 µs | 5 866× |
| mixture parameters | 500 | 4.11 µs | 1.84 µs | 0.4× |
| mixture parameters | 5 000 | 23.3 µs | 90.4 µs | 3.9× |

XLA's vectorised full evaluation is three orders of magnitude cheaper than the Julia
baseline, so absolute numbers differ, but the structure matches the paper: single-site
updates are constant in N and the dense parameter moves pay the incremental overhead.
The priors are standard choices; the paper's exact priors live in its artifact
(Zenodo 10.5281/zenodo.19080766), not in the text.

Later: `scan` rule for Markov structure; data-dependent `gather` sparsity (the paper's
`using`); `memo` primitive for latent-independent intermediates; masked fixed-capacity
data growth (SMC-style); `MixedHMC`; `BarkerMH`.

## Decisions log

- 2026-10-22: Incrementalizer is a jaxpr interpreter under tracing (like
  `ops/provenance.py`), not a custom `Trace`/`Tracer` (avoids `jax._src`) and not a
  handler-level plate restriction (fails for `obs=` full arrays and pre-batched `fn`;
  would rely on XLA DCE).
- 2026-10-22: First branch = transform + tests + adapter for existing kernels. No new
  kernel library.
- 2026-10-22: Model args are jaxpr invars, not closed-over constants; validation off by
  default when tracing the density.
- 2026-10-22: `DiscreteGibbs` reproduces the proposal maths of `hmc_gibbs` instead of
  calling its functions: they carry the full site array through the candidate loop
  (`z_flat.at[idx].set` per candidate) and would make every element update O(N).
  Candidate deltas are `vmap`-ped over scalar values and the chosen value is written
  back with one sliced apply, with no `lax.cond` on the state.
- 2026-10-22: `test/infer/test_mcmc.py::test_beta_bernoulli_x64[SA]` fails on
  `develop` before the `sa.py` one-liner as well; not caused by this branch.
- 2026-10-22: XLA CPU copies a loop-carried buffer on every iteration when the body
  both updates it in place and reads a value from it that escapes into another carried
  element (the running total), which made a single-element update O(N) inside a
  sweep even though the Δ-program itself is O(1). Neither `optimization_barrier`,
  gather-based reads nor `dynamic_update_slice` help. Sweeps therefore read from a
  snapshot taken before the loop (`apply(..., read_values=snapshot)`), which is exact
  when every read of the swept site is at the swept position; the planner reports the
  sites a signature reads in full (`input_full_reads`) and `DiscreteGibbs` falls back
  to carried reads for them. Cache writes are deferred to the end of an apply. Single-
  element updates cost 10–30 ns at any N after this.
- 2026-10-22: `IncrementalMCMCState.z` is a view of `inc.z`, not a second field: two
  aliases of one carried array make XLA copy one of them per in-place update.
- 2026-10-22: `RandomWalk` block (Gaussian drift in unconstrained space) added for the
  paper's parameter kernels; `IncrementalMCMC` initialises with `init_to_uniform`
  like HMC (prior samples left the mixture examples in poor modes).
