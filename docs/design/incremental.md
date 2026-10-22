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
3. ⏳ `reduce_sum` delta rule, dense propagation, reparam, finiteness guard, resync;
   gradient equality on the dense path.
4. ⏳ `custom_jvp` registry/recursion.
5. ⏳ Slice pullback and `cache="plan"`.
6. ⏳ `adapt.py` and `sa.py` fix; posterior agreement with `DiscreteHMCGibbs`.
7. ⏳ Docs (`contrib.rst`), `examples/incremental_gmm.py`, benchmark table here.

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
