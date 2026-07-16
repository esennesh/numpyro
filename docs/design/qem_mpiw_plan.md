# QEM / MPIW-EM in NumPyro — project plan

Target: implement QEM (Heap, Bowyer & Aitchison, AABI 2025) — expectation maximization
for approximate posteriors, with the E-step computed by massively parallel importance
weighting (MPIW, Bowyer et al. 2024) — for generic models and guides with static control
flow and exponential-family guide distributions. Exponential-family machinery is backed
by [efax](https://github.com/NeilGirdhar/efax), but the efax-invoking code is kept to an
absolutely minimal, separable leaf (see Branch 2) so everything else can be upstreamed
without it.

Home: this fork first; later maybe upstream to pyro-ppl/numpyro via feature-branch PRs.

## What the codebase already gives us

- **Tensor-variable-elimination exists.** `contrib.funsor._enum_log_density`
  (`numpyro/contrib/funsor/infer_util.py:200`) builds per-site funsor log-prob factors
  with named dims and contracts them with
  `funsor.sum_product.sum_product(logaddexp, add, factors, eliminate=..., plates=...)`
  (+ `apply_optimizer`). `TraceEnum_ELBO` (`numpyro/infer/elbo.py:1086`) does the same
  per connected component (`_partition`, `elbo.py:1023`). MPIW's sum over Kⁿ index
  combinations is this contraction with K-sample dims in place of enumeration dims.
- **The exact hook point is marked TODO.** `contrib/funsor/enum_messenger.py:589`:
  `if msg["infer"].get("num_samples") is not None: raise NotImplementedError("TODO
  implement multiple sampling")`. Pyro-torch implemented this (`enumerate_site` +
  `_tmc_diagonal_sample`/`_tmc_mixture_sample` in `pyro/poutine/enum_messenger.py`,
  consumed by `TraceTMC_ELBO`); numpyro never did. Branch 3 fills this hole.
- **Graph discovery exists** (`numpyro/infer/inspect.py: get_dependencies`,
  `numpyro/ops/provenance.py`) but is mostly *not needed* for the contraction itself:
  factor dimensions fall out of broadcasting (a site's log_prob array automatically
  carries the named K-dims of every K-sampled ancestor its parameters depend on), and
  funsor's optimizer picks the elimination order. Dependency info is for validation and
  diagnostics only.
- **No exponential-family machinery exists** anywhere in `numpyro/distributions/` (no
  natural params, sufficient statistics, log-normalizer, moment matching;
  `conjugate.py` is compound distributions, not an EF abstraction). The KL registry
  (`kl.py`, `multipledispatch`) is the in-repo precedent for type-keyed registries.
- **efax 2.4.0 verified working** (jit/grad/vmap-safe, pytree dataclasses,
  numpyro-like batch semantics): `sufficient_statistics(x)` → EP objects, NP→EP
  closed-form everywhere, EP→NP closed-form for Normal family/Bernoulli/Categorical/
  Poisson/etc., optimistix root-finding for Gamma/Beta/Dirichlet-type families.
  Caveats: Python ≥3.12,<3.15 (numpyro supports 3.11+), near-latest jax pinned via
  tjax, heavy dependency tail, 2.0 API break in 2026-03. Coverage gaps: no Binomial,
  Multinomial, univariate von Mises.

## Branch structure

```
(1) exp-family interface (efax-free) ──┬──► (4) QEM driver
        (2) efax bridge (leaf) ────────┤        ▲
                                       │        │ (2) optional at runtime
(3) MPIW core (independent) ───────────┘
```

Branches 1 and 3 are independent and can proceed in parallel. Branch 2 is a leaf off
branch 1 — nothing depends on it except test coverage for iterative families. Branch 4
integrates 1 + 3 and merely *benefits* from 2.

Side branch — `feature/auto-mean-field`: port `AutoMeanFieldProposal` from
numpyro_template into this fork's `numpyro/infer/autoguide.py` as a proper autoguide
(prior-family-mirroring mean field), cleaning up the vestigial `sample_posterior`
(`self._mixture`, key/dict iteration bug) and `num_particles` along the way, with a
real `sample_posterior` implementation and tests. Independent of everything above;
branch 4's `AutoExponentialFamily` then derives from the ported class.

### Branch 1 — `feature/exp-family-interface`: EF protocol, moment matching, EMA (efax-free)

Everything QEM needs from "exponential family" expressed as an efax-agnostic protocol,
plus the M-step machinery, written against plain pytrees.

**Progress: complete** (branch `feature/exp-family-interface`, all green + ruff clean):
`numpyro/distributions/exp_family.py` — `sufficient_statistics` / `mean_params` /
`from_mean_params` registries via `singledispatch`, plus `canonical_params` (names the
constructor args to rebuild each family — needed where `arg_constraints`
over-specifies, e.g. MVN) and `base_distribution` / `is_exp_family` helpers.
Closed-form families: Normal, MultivariateNormal, Bernoulli (probs & logits),
Categorical (probs & logits), Poisson, Exponential; Independent/ExpandedDistribution
unwrapped transparently. Tests in `test/test_exp_family.py` (MC consistency,
round-trips, conjugate Normal-Normal and MVN recovery via weighted moment matching).

- **Mean-parameter format**: plain pytrees (dicts of arrays) per site — never efax
  objects. This is the only mutable state QEM will carry (the paper requires the EMA to
  be over *mean* params, Appendix B; making them the only state enforces that).
- **Registry** keyed by numpyro distribution type (pattern: `kl.py` multipledispatch or
  a plain dict + `singledispatch`), each entry providing:
  - `sufficient_statistics(dist, value) -> pytree` — per-observation T(x),
  - `mean_params(dist) -> pytree` — E[T(z)] under the distribution,
  - `from_mean_params(prototype_dist, pytree) -> Distribution` — moment matching /
    `set_exp_family_params` from Algorithm 1,
  - wrapper unwrapping for `Independent` / `ExpandedDistribution` (and clear rejection
    otherwise).
- **Hand-written reference implementations** for the closed-form families we can do in
  a few lines each — Normal (loc/scale and MVN), Bernoulli, Categorical, Poisson,
  Exponential — so the interface is testable and QEM is usable with zero optional deps.
  (These are also the upstreamable core; iterative families come from branch 2.)
- Tests: T(x)/mean-param consistency (E[T] via sampling vs `mean_params`), round-trip
  `from_mean_params(mean_params(d)) == d`, exact conjugate posterior recovery
  (Normal-Normal, Beta-Bernoulli via branch-2 or deferred, Gamma-Poisson deferred).

(EMA machinery lives in branch 4 — nothing else uses it.)

### Branch 2 — `feature/efax-bridge`: minimal efax adapter (leaf)

The *only* code that imports efax. Target size: one module + tests.

- `numpyro/contrib/efax_bridge/` (name TBD) with the guarded-import pattern of
  `contrib/tfp/__init__.py`; new optional dependency group pinning `efax>=2.4,<3`;
  import-time gate on Python ≥3.12.
- Contents: for each supported (numpyro dist type ↔ efax NP/EP class) pair, register a
  branch-1 implementation that delegates: conventional params → efax NP →
  `to_exp()`/`sufficient_statistics()` → plain pytree, and pytree → EP → `to_nat()`
  (closed-form or optimistix) → conventional params. Boundary conversions
  pytree ↔ efax dataclass live here and nowhere else.
- Adds the iterative families: Gamma, Beta, Dirichlet, InverseGamma, plus anything else
  cheap (LogNormal via efax directly).
- Upstreaming story: numpyro core never sees efax; if upstream wants Gamma/Beta/
  Dirichlet without the dependency, we port a small Newton/digamma-inversion into
  branch-1-style registrations later. Binomial/Multinomial likewise (absent from efax).

### Branch 3 — `feature/mpiw`: massively parallel importance weighting (independent)

The E-step engine; independently useful for marginal-likelihood estimation, posterior
moments/marginals/samples from a fixed guide. Design informed by a close read of the
enum/funsor machinery — see "How the funsor path works" below.

**Written for upstream from day one.** This branch should be a plausible PR to
pyro-ppl/numpyro on its own: it fills the marked
`NotImplementedError("TODO implement multiple sampling")` in the *existing*
`enum` messenger (`contrib/funsor/enum_messenger.py:589`) rather than forking it,
mirrors pyro-torch's `infer={"enumerate": "parallel", "num_samples": K, "tmc":
strategy}` site annotations for API parity, and keeps zero dependencies beyond what
`contrib.funsor` already requires. No QEM- or exponential-family-specific code
anywhere in this branch.

**Progress: branch 3 core complete** (branch `feature/mpiw`, all green + ruff clean):

- ✅ **Contraction core** — `numpyro/contrib/mpiw/contraction.py`. `NamedFactor`,
  `contract_log_marginal` (log-space sum-product → `log P_MP`, plates as product dims),
  `contract_with_source_terms` (source-term trick via `jax.value_and_grad`).
- ✅ **Sampled enumeration messenger** — filled the `num_samples` TODO in
  `contrib/funsor/enum_messenger.py`; draws K samples per site along a fresh named dim,
  with parent coupling (K^(1+#parents)); works for continuous and discrete sites.
- ✅ **End-to-end `MPIW` driver** — `numpyro/contrib/mpiw/core.py`: `log_marginal`,
  `site_weights`, `moments`, `sample_posterior` (FFBS via funsor adjoint; recovers joint
  covariance). Handles scalar + multivariate (event-dim) guide sites, plated/unplated,
  continuous + discrete latents.
- ✅ **Serial (memory-frugal) contraction** — `serial_dims`/`serial_sites` option:
  `lax.scan` over a chosen global dim, slicing it out per iteration; differentiable
  (source-term moments stay memory-bounded); non-plated dims only (validated). Identical
  results to the dense path.
- ✅ **Validation** — analytic linear-Gaussian (conjugate chain + plated hierarchical),
  discrete latents, multivariate sites, and the bird-occupancy integration test vs an
  exact 2D grid and NUTS on the discrete-marginalized model (config_enumerate+NUTS gold
  standard), exercising the serial path. Tests in `test/contrib/mpiw/`.
- ⏳ Remaining niceties: vectorize `sample_posterior` over draws (currently a sequential
  loop); event-dim-aware behavior for statistics beyond the mean; the fully hand-rolled
  einsum fallback (only if funsor becomes a problem).

**Components:**

1. **Sampled enumeration** — fill the `NotImplementedError` at
   `enum_messenger.py:589`: for non-observed guide sites marked
   `infer={"enumerate": "parallel", "num_samples": K}`, draw K samples (any
   distribution, continuous or discrete — no `has_enumerate_support` requirement),
   allocate a fresh named dim for the site (same `DimStack` mechanics as enumeration),
   and place samples along it. Parent-index schemes as in pyro-torch's
   `_tmc_diagonal_sample` / `_tmc_mixture_sample`: "diagonal" (child sample k pairs
   with parent sample k), "mixture" (uniform random ancestor index per child sample),
   plus the paper's default "permutation" (random permutation of parent indices —
   generalizes diagonal; Heap et al. 2023 App. A). Mean-field guides need no scheme.
2. **Weight contraction** — an MPIW analogue of `_enum_log_density`: run the K-sampling
   guide under `plate_to_enum_plate` + packed `trace`, `replay` the model against it,
   build per-site funsor factors
   `log p_site − log q_site − log K` (model term via replay; guide term and the
   −log K uniform-mixture weight attached to the site that owns the K-dim), then one
   `sum_product(logaddexp, add, ...)` contraction over all K-dims (eliminate) with
   plates as prod-vars → scalar `log P_MP(x)`, an unbiased marginal-likelihood
   estimator. Factor dims are automatic: replayed model log-probs broadcast over the
   K-dims of whichever sampled ancestors they touch, so guide-trivial (mean-field)
   cases pick up the *model's* graph structure with no extra code, and structured
   guides contribute their own K-dim couplings through the guide factors.
3. **Source-term trick** (Bowyer et al. 2024) — wrap the contraction in a function of
   per-site perturbations and use `jax.grad` (the contraction is ordinary jnp ops under
   funsor's jax backend):
   - Inject a per-site vector J_i over the site's K-dim into its factor;
     `∂ log P_MP / ∂ J_i |_{J=0}` = the site's *normalized marginal importance
     weights* w_i(k). One backward pass yields all sites' weights.
   - Per-site moments are then dot products `Σ_k w_i(k) m(z_i^k)` computed outside the
     engine — so the engine is statistics-agnostic; branch 4 supplies
     `sufficient_statistics` as m. (Equivalently J·m(z) injection; the weights form is
     strictly more general per-site.)
   - Posterior joint samples: backward/adjoint sampling over the contracted factors
     (template: `contrib/funsor/discrete.py:_sample_posterior`); marginals are the
     (value, weight) pairs directly.
4. **Public API** (configurable outputs as required):
   ```python
   mpiw = MPIW(model, guide, K=30, scheme="permutation")
   mpiw.log_marginal(rng, *args)                   # scalar log P_MP(x)
   mpiw.site_weights(rng, *args)                   # site -> (K samples, K weights)
   mpiw.moments(rng, statistics, *args)            # site -> weighted moments
   mpiw.sample_posterior(rng, num_samples, *args)  # joint posterior samples
   ```
   Optionally later: a `TraceMPIW_ELBO` loss (true MP-VI) — *this* would need the DiCE
   machinery; explicitly out of scope for QEM (see below).

**Why we can skip TraceEnum_ELBO's hard parts (DiCE etc.):**

`TraceEnum_ELBO.loss` is complicated because it must produce correct, low-variance
*gradients of the ELBO w.r.t. guide parameters* when some sites are sampled rather than
enumerated. That is what the "dice factors" are: at each non-enumerated guide site,
`log_measure = log q − detach(log q)` (`elbo.py:1013-1015`) — a term that is
identically zero in the forward pass but whose gradient reinserts the score function
∇log q; each cost term is multiplied by `exp(Σ dice factors of its non-reparam
ancestors)` (`elbo.py:1277-1296`), with provenance tracking deciding which ancestors
(DiCE, Foerster et al. 2018; Storchastic). QEM needs none of this: it never
differentiates w.r.t. guide parameters — the only autodiff is ∂/∂J at J=0 with samples
held fixed. So MPIW follows the *simple* `_enum_log_density` path (build factors, one
contraction, scalar out), not the `TraceEnum_ELBO.loss` path (partitioned cost terms,
dice factors, provenance). The cost-partitioning (`_partition`) is an optimization we
can adopt later if a single global contraction proves slow.

**How the funsor path works (reference for implementation):**

- `plate_to_enum_plate` (`infer_util.py:24`) swaps `numpyro.plate` for the funsor-aware
  plate, which registers the plate dim as a named VISIBLE dim in the global `DimStack`.
- `enum.process_message` (`enum_messenger.py:575`) intercepts eligible sample sites,
  builds `funsor.Tensor(arange(size), {name: Bint[size]})`, and `to_data` materializes
  it as an array whose support lies along a freshly allocated negative batch dim; the
  site's value therefore *carries* its named dim implicitly by position.
- The packed `trace` (`enum_messenger.py:605`) records `dim_to_name` per site after
  execution, so any site's log_prob array can be lifted `to_funsor` with named inputs.
- `_enum_log_density` (`infer_util.py:200`) lifts each site's log_prob, collects
  `sum_vars` (latent site names = dims to logsumexp out) and `prod_vars` (plates), and
  contracts. MPIW step 2 is this function with: guide+replay instead of a single model
  trace, importance-ratio factors instead of joint factors, K-dims instead of support
  dims, and a −log K per site.

**De-risking spike — STATUS: PASSED (2026-07-14).** Scratchpad
`mpiw_spike.py` validated the funsor route end-to-end on two analytic models (scalar
3-node conjugate Gaussian chain, and a plated hierarchical Gaussian μ → {zᵢ} → {xᵢ}):
- `log P_MP` is unbiased for P(x) and converges to the analytic evidence as K grows
  (both models);
- source-term trick works: `jax.grad` of per-site injected vectors J at J=0 yields
  normalized marginal weights (sum to exactly 1.0, including per-plate-element in the
  plated model) whose weighted moments match the analytic posterior to 2–3 decimals;
- `plates={i}` in `sum_product` delivers the O(K^(1+#parents)) per-plate-element
  contraction (NOT O(Kᴺ)) — the critical memory mechanic — verified by per-element
  weights each summing to 1;
- both forward and `jax.grad` `jit`-compile.

Key design finding: the spike used the **full K×K combinatorial** factor coupling,
which is correct for a **mean-field guide** (QEM's default). The diagonal / permutation
/ mixture parent-selection schemes are only needed for *structured* guides where
q(child|parent); a mean-field-first messenger needs no ancestor-reindexing at all.
⇒ v1 of the messenger targets mean-field guides; schemes are a later extension.

Original spike checklist (all met):
1. defines a 3-site conjugate Gaussian chain (a → b → x, one plate) with analytic
   log P(x) and posterior moments;
2. hand-rolls MPIW *without any messenger*: draw K samples per site from a mean-field
   guide, call `funsor.to_funsor` directly on manually-arranged log-prob arrays with
   named K-dims, add −log K, contract with `sum_product`;
3. checks `log P_MP` is unbiased and → log P(x) as K grows;
4. adds per-site J vectors and `jax.grad`; checks recovered moments against the
   analytic posterior;
5. verifies grad-through-`apply_optimizer` works and is efficient under jit.

Success criteria: numbers match analytics; grad compiles. Only then wire the messenger
(`num_samples` in `enum`), traces, and API. Fallback if funsor fights us (it is
git-pinned and lightly maintained): the spike's hand-rolled contraction generalizes to
a small log-space einsum engine with elimination order from `get_dependencies` — more
code, fewer moving parts, and the paper's own implementation works this way.

**Tests:** conjugate Gaussian chains/trees (analytic evidence + moments); agreement
with global IS for tiny n·K; unbiasedness of P_MP across seeds; a discrete-latent
model (no gradients through samples are ever needed, so discrete sites are supported
naturally); K-dim/plate interaction; memory scaling sanity (factor size
O(K^(1+#parents)) × plates, paper §6).

**Alternate low-memory computation path (to provide, not just "later").** The default
contraction materializes each factor densely (up to K^(1+#parents) × plates) and the
sum-product's peak intermediate is ~K^(treewidth+1) — this is what bounds usable K.
We should also offer a memory-frugal path that materializes as little as possible and
instead spends serial compute: loop over the index-tuples k of the summed K-dims
(and/or plate elements) with `jax.lax.scan`/`fori_loop`, accumulating the log-marginal
via running logsumexp and the source-term gradients per index, so that at no point is a
full K^(1+#parents) (let alone K^treewidth) array resident — only O(per-iteration
slice) memory, at the cost of many sequential steps. Design points to settle:
- Granularity: loop over the single largest K-dim (or the treewidth-inducing set) while
  keeping smaller dims vectorized — a tunable time/memory trade, not all-or-nothing.
- Keep the source-term trick working: either differentiate the scanned accumulator, or
  accumulate per-index contributions to the weights directly (the running-logsumexp
  normalizer is known at the end, so weights can be finalized in a second pass or via a
  carried normalizer).
- Same public results as the dense path (identical `log_marginal`/`moments` to
  numerical tolerance) — make it a `backend`/`memory_budget` switch on the contraction
  API, with the dense funsor path as default and this as the fallback for large K or
  high treewidth. The hand-rolled log-space einsum fallback (above) and this frugal
  path share machinery.
- Complements the paper's own remedies (Aitchison-style plate factorization we already
  get, plus "grouping" and "chunking", §6); this is the chunking taken to its serial
  limit.

**Bird-occupancy integration test** (small scale): a mixed continuous-discrete
hierarchical model from the paper's lineage (Bowyer 2024) — binary per-site presence
latents + hierarchical continuous occupancy/detection params. Because the binary
latents are per-site enumerable, `config_enumerate` + NUTS gives a gold-standard
reference; assert MPIW `log_marginal` and per-site `moments` (from source terms) agree
with the long-NUTS reference within tolerance. This exercises the full engine on a real
mixed model with plates + discrete + continuous sites, and doubles as the correctness
oracle the branch-4 benchmark reuses. Keep it small enough to run in CI (a handful of
sites/species); the scaling story belongs to branch 4.

### Branch 4 — `feature/qem`: the QEM driver (depends on 1 + 3)

**Progress: core complete** (branch `feature/qem` = merge of branches 1 + 3, all
green + ruff clean):

- ✅ **`QEM` driver** — `numpyro/contrib/qem/core.py`: `init`/`update`/`run`/
  `get_params`/`evaluate` mirroring SVI; state = per-site mean-param pytrees +
  step + rng key; M-step via `params_from_mean` + `handlers.substitute`
  (SVI-compatible param flow); E-step via a new
  `MPIW.log_marginal_and_site_weights` (weights + log P_MP from one contraction);
  EMA on mean params with fixed λ, callable λ(t), or the default Thm-1 schedule
  λ(t) = 1 − t^(−p).
- ✅ **`AutoExponentialFamily`** — `numpyro/infer/autoguide.py`, written directly
  against `AutoGuide` (the `AutoMeanFieldProposal` port became unnecessary; its
  vestigial bugs are simply absent). Mirrors each prior's base family with
  unconstrained `{site}_{prefix}_{arg}` params (canonical args, prior-valued
  init), supports discrete latent sites (overrides `_setup_prototype` to skip the
  base class's discrete-support rejection), working `sample_posterior`, rejects
  subsampled plates. Usable with plain SVI too (tested).
- ✅ **Tests** — `test/contrib/qem/test_qem.py`: conjugate recovery (scalar
  Normal, plated hierarchical, MVN site, discrete Bernoulli), log P_MP trace →
  analytic evidence, prior-matched init, forget/schedule semantics + validation,
  non-EF rejection, reparameterization invariance (Thm 2, α = 1e-2 trajectories
  match to rtol 1e-8), SVI compatibility.
- ⏳ Remaining: decorrelated-normalizer variant (§4 flag), PSIS k-hat diagnostics,
  the QEM-vs-VI benchmark below, λ(t) reference against the paper's exact
  constants once re-checked.

- `QEM` class mirroring the `SVI` API surface (`init`/`update`/`run`), state = per-site
  mean-param pytrees + RNG key. No optax, no param store:
  1. **M-step**: rebuild guide site distributions via branch-1 `from_mean_params`.
  2. **E-step**: `mpiw.site_weights` + branch-1 `sufficient_statistics` → one-iter
     mean-param estimates.
  3. **EMA** on mean params (Eq. 8) as pytree ops — fixed λ or the λ(t) = 1 − t^(−p)
     schedule of Theorem 1. Lives here (not branch 1): nothing else uses it.
- Guide contract: any guide whose sample sites the branch-1 registry recognizes.
- Default guide: `AutoExponentialFamily`, adapted from `AutoMeanFieldProposal`
  (`numpyro_template/src/inference/auto_mean_field.py`). That guide already has the
  right skeleton for QEM: per-site, it unwraps the prior to its base distribution,
  mirrors the *same distribution class* with learnable constructor args
  (unconstrained via `biject_to`, broadcast over the site's plate/batch shape), and
  re-applies `to_event` — so guide sites are automatically same-family as the prior,
  i.e. registry-recognizable whenever the prior is EF. Adaptation choices:
  - Parameter flow: keep the guide's `numpyro.param` sites and have the M-step emit a
    params dict (conventional params derived from mean params) that is substituted in
    SVI-style — this preserves compatibility with `SVI`/`Predictive` tooling — rather
    than bypassing the param store. (`from_mean_params` returns a Distribution; the
    guide's naming convention `{site}_{prefix}_{arg}` gives the mapping back to param
    names.)
  - Init: prior-moment-matched (`mean_params(prior site dist)`) instead of
    `init_to_sample`.
  - Known issues in the source to fix while adapting: `sample_posterior` references a
    nonexistent `self._mixture` and iterates `for site in self.prototype_trace:`
    using `site` as both key and dict; `num_particles` reads an attribute that is
    never set. Both look vestigial from an earlier class.
- Option: decorrelated-normalizer variant (fresh z′ ~ Q_MP for the P_MP normalizer,
  paper §4). Note: log P_MP is unbiased for log-evidence purposes, but the *moment*
  estimator (Eq. 7a) is self-normalized IS — a ratio estimator with O(1/K-style)
  finite-K bias. The paper's remedy removes the numerator/denominator covariance term
  of the ratio bias; the Var[P_MP]-driven term survives (E[1/P_MP(z′)] ≥ 1/P(x) by
  Jensen), so strictly it reduces rather than eliminates bias, despite the paper
  calling it unbiased. Implement as a flag; verify empirically which variant has lower
  bias/variance at practical K (extra cost: one more guide trace + one normalizer
  contraction, no source terms).
- Diagnostics: log P_MP trace; PSIS k-hat on flattened weights (reuse
  `numpyro/infer/importance.py`).
- **Benchmark: QEM vs mean-field VI** (see next section).

## Benchmark: QEM vs mean-field VI (branch 4 deliverable)

**Why VI, not MCMC.** QEM and MCMC are not a fair single-axis comparison: a mean-field
QEM guide carries a fundamental proposal-family bias (no posterior correlations) that
asymptotically-exact MCMC does not, so any "accuracy" contest is apples-to-oranges. The
clean comparison holds the *guide family fixed* and varies only the *fitting method*:
gradient-free EM on MPIW moments (QEM) vs gradient-based ELBO maximization
(mean-field VI). This isolates exactly the paper's claim — that moment-based EM updates
converge faster and more robustly than gradient ascent on the same approximate
posterior — and mirrors the paper's own Fig 1/2 methodology (QEM vs MP VI/RWS).

**Setup.**
- Same mean-field guide for both arms: the ported `AutoMeanFieldProposal` /
  `AutoExponentialFamily` (branch-4 default). VI arm = `SVI` with `Trace_ELBO`
  (optionally `RenyiELBO`/`TraceEnum_ELBO` where discrete sites need marginalizing)
  and Adam; QEM arm = the branch-4 driver. Identical model, data, init, and K where
  applicable.
- Models: a paper-style continuous hierarchical model (Radon or Bus Breakdown) plus the
  bird-occupancy model (mixed continuous-discrete) so the comparison spans both a purely
  continuous case and one where VI needs enumeration for the discrete sites.

**Metrics (both vs wall-clock and vs iterations):**
- **Estimated log-marginal likelihood over the course of fitting** — the headline
  curve. For QEM this is MPIW `log P_MP` at each iteration; for VI the (importance-
  weighted) ELBO. Show the trajectory, not just the endpoint: the paper's result is
  faster *convergence*, so the shape and wall-clock-to-plateau are the point.
- **Wall-clock per iteration and to a target log-marginal** — QEM's gradient-free
  E/M step vs VI's full-joint gradient (paper Figs 2–4 show QEM's per-iteration
  advantage).
- Optional: held-out predictive log-likelihood over fitting (paper's second metric).

**Reference oracle (not a competitor):** where feasible, a long `config_enumerate`+NUTS
run supplies "true" posterior moments to sanity-check *both* arms converge to the same
(guide-limited) neighborhood — used for validation, not as a benchmark rival, precisely
because of the proposal-bias asymmetry above.

**Optional geometry stress test** (paper §5.2): rerun under the α-scaling
reparameterization; expect VI's gradient-based trajectory to degrade/destabilize while
QEM is invariant — a qualitative robustness result on top of the speed curves.

Discrete-capable MCMC methods available in this fork if ever needed for the oracle:
`NUTS` + `config_enumerate` (contrib.funsor), plus `DiscreteHMCGibbs`, `MixedHMC`,
`HMCGibbs`, `HMCECS`, `BarkerMH`, `SA`.

## Paper-fidelity checklist

- EMA over **mean** parameters only (App. B) — enforced by state format.
- λ(t) = 1 − t^(−p) schedule available (Thm 1).
- Permutation parent-selection default (App. A; pyro-torch precedent has
  diagonal/mixture — we add permutation).
- Unbiased P_MP option via a second sample (§4).
- Reparameterization invariance (Thm 2) as an integration test: scale a latent by
  α ∈ {1e-2, 1e-4}, check QEM trajectories match to float tolerance.

## Decisions log

- Fork-first; upstream later via feature-branch PRs. Hence: efax code confined to a
  leaf contrib (branch 2); core protocol + hand-written closed-form families are the
  upstreamable subset.
- Guide contract: bridge-recognized sites required; `AutoExponentialFamily` (from
  `AutoMeanFieldProposal` in numpyro_template) as default. Arbitrary structured
  hand-written guides: later.
- Contraction engine: funsor-first behind the spike gate; hand-rolled log-einsum as
  documented fallback.
- Branch 3 is written generically for upstream: fills the existing enum-messenger
  TODO in place, pyro-torch-compatible site annotations, no QEM/EF coupling.
- EMA + λ-schedule machinery lives in branch 4 only.
- Branches (1→2) and (3) proceed in parallel; converge in branch 4.
