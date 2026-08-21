# 06 — Plan review

Status: **proposed changes, none applied.** Written 2026-08-21 against the plan
as of that date. Three lenses: statistician, software engineer, working data
scientist. Each item says whether it is a **simplify** (remove or shrink) or a
**reinforce** (a gap that will bite), what it changes, and where.

The short version: the plan is unusually good at saying *why*. Its weak spots
are (1) three internal contradictions in the layering and the interpreter
story that will surface in the first week of Phase 1, (2) statistical
acceptance criteria written as point tolerances where they need acceptance
regions with a stated N, and (3) a 1.0 scope that carries ~3,000 LOC of
product features no statistician will reach for in the first year. Fixing
(1) costs a day of doc work now; not fixing it costs a refactor in Phase 4.

---

## A. Contradictions to resolve before Phase 1 (reinforce)

### A1. `EstimandResult.identification: Verdict` violates the layering rule

`01-architecture.md` says `estimands` may import `core` only, and `estimands`
and `identify` are peers. `02-porting-ledger.md` and Phase 4 say
`EstimandResult` gains an `identification: Verdict` field. `Verdict` lives in
`identify`. One of these is wrong.

**Change:** add `core/verdict.py` holding the status vocabulary
(`identified | downgraded | blocked | unsupported | unverified`), the
`Verdict` spec, and the `Assumption` base spec. `identify`, `calibrate`,
`design/methods`, and the ledger all consume the *same* `Assumption` type. The
plan currently invents it three times (transport licensing assumptions, design
method assumptions such as parallel trends, ledger lines). One type, one home,
and `test_ledger_completeness` becomes trivially checkable.

### A2. "One `forward()`" is actually two interpreters, and the likelihood is in the wrong layer

`forward()` is `interpret.value` (numpy). The NumPyro likelihood in
`surface/model.py` cannot call a numpy function; it needs `interpret.jax`. So
the invariant the parent broke — likelihood and design math drift apart — is
*not* enforced by "both call `forward()`"; it is enforced by the two
interpreters agreeing on the same tree. The plan never states this and the
gate (`test_structural_uses_forward`) tests the wrong thing.

Worse: if `surface/model.py` contains a NumPyro model function, then a
domain-layer module imports a sampler, and the promise that heavy deps live in
`infer` is broken on day one of Phase 3. The same applies to `meta/pool.py`.

**Change:**
- `Backend.sample` takes a **model spec** — `(Expr, Priors, Likelihood)` — not
  a Python callable. The backend compiles the tree. This is the fifth
  interpreter (`interpret.numpyro`, or `interpret.jax` plus a generic
  likelihood wrapper). It is also the only way a fitted model serializes
  without pickle, so the plan already needs it.
- Add a Phase 3 exit criterion: `interpret.value(tree) == interpret.jax(tree)`
  to 1e-10 over 200 random trees and random inputs. This replaces gate 9's AST
  reachability check, which is heuristic and will produce false passes.
- `surface/model.py` and `meta/pool.py` contain no sampler imports, lazy or
  otherwise. `test_layering` asserts it.

### A3. The `Pow` dimension rule contradicts the reason `Fraction` exists

`01-architecture.md`: "`Pow(base, q)`: base must be dimensionless unless `q`
is an integer." Two paragraphs earlier: "`Fraction`, not `int`, because a
standard deviation is the square root of a variance." A square root of a
dimensioned quantity is a non-integer power of a dimensioned base.

**Change:** any rational constant `q` is allowed and the result is
`dim(base) ** q`. Only a *non-constant* exponent requires a dimensionless base.

### A4. The layering exceptions are redundant with the diagram

The text says the pillars "do not import each other except `design ← surface`
and `calibrate ← estimands`". But `surface` and `estimands` are in the domain
layer below the pillars, so every pillar may import them. The exception clause
is confusing, not wrong. Also, `meta/priors.py` "hands off into `calibrate` and
`surface`" — `meta` cannot import either.

**Change:** state the rule as "a module imports only from strictly lower
layers; same-layer imports are forbidden". Make the `meta → calibrate` handoff
data-only: `meta.priors` returns a `core` Prior spec that `calibrate` and
`surface` accept.

### A5. `design/geo.py` fails gate 3

`geo` is on the banned list. So is `geometric` via substring (`geometric`
adstock lands in Phase 3), `median` via `media`, and `android`/`roid` via
`roi`.

**Change:** rename to `design/cluster.py`. Gate 3 matches identifier *tokens*
(split on `_` and camelCase) with an allowlist, not substrings.

---

## B. Statistical content (reinforce)

### B1. The facet table's `conditioning` row states the wrong assumption

"Effect homogeneity across the collapsed strata" is sufficient, not necessary,
and is rarely true. Moving from a conditional to a marginal contrast needs
**standardization over known strata weights in the target population** — which
is a `population` question, not a homogeneity one. For non-collapsible
quantities (ratios, odds-type effects, elasticities) the marginal is not a
weighted average of the conditionals at all.

**Change:** split the row. Contrast / marginal-derivative / area: licensing
assumption is "target strata weights known"; challenged by the strata
distribution's provenance. Ratio / elasticity: transfer across a
`conditioning` difference is `blocked` unless the quantity is re-expressed as a
contrast first. This interacts with `quantity`, and the facet table should say
so.

### B2. Two things the eight facets do not pin down

- **Treatment version** (consistency / SUTVA-1). "Set dose to $X" on one
  creative, one formulation, one delivery schedule is a different intervention
  from the same dose on another. `intervention` fixes "what, to what value,
  over what support" but not *how delivered*.
- **Interference** (SUTVA-2). `level` fixes the unit of analysis but not
  whether units interfere. A cluster-randomized geo test with spillover and an
  individual-level test with none are different estimands at the same `level`.

Neither needs a ninth facet. **Change:** `intervention` gains a `version`
sub-field (free identifier, defaulting to `"unspecified"`), and the
`intervention` row's licensing assumption adds "same treatment version or
version-irrelevance, asserted". `level` gains an `interference` enum
(`none | within_cluster | declared`), and a differing value is `blocked`
without a declared interference model. Gate 11 parametrizes over the
sub-fields too.

### B3. "Transport is d-separation on a selection diagram" is true for one facet

Only the `population` row is a graph problem. `window`, `intervention`,
`level`, `conditioning` are functional or structural, and `outcome` is
asserted. The charter says the answer is "derived from a graph, not asserted
in prose". **Change:** "derived from a graph where it can be (population),
from the response surface where it must be (intervention, window, level),
and asserted with a ledger line where it cannot be (outcome)."

### B4. Static DAGs do not cover the panel setting the package is built for

The canonical user has a panel where `dose_t` responds to `outcome_{t-1}`
(budgets follow sales; doses follow symptoms). That is time-varying
confounding with treatment–outcome feedback, and back-door adjustment on a
static summary graph is not valid there; g-methods or an unrolled graph are
needed. Nothing in `identify` addresses it and the carryover machinery makes
it worse, because a lagged effect is exactly the path that feedback confounds.

**Change:** `CausalGraph` either (a) is declared over time-indexed nodes and
unrolled to a stated lag depth before d-separation, or (b) carries a
`feedback: bool` assertion; when `True` and any treatment has carryover, the
verdict is `downgraded` with the reason "time-varying confounding; static
adjustment insufficient". (b) is a day of work and honest; (a) is 1.1. Add a
`sim` world with feedback to the Phase 2 negative controls.

### B5. Acceptance criteria are point tolerances where they need acceptance regions

Every calibration-style criterion is currently flaky by construction:

| Criterion | As written | Binomial SE at N | False-fail risk |
|---|---|---|---|
| Phase 3 #2: nominal ±3 pts, N=200 | 87–93% | 2.1% | ~16% per run |
| Phase 5 #2: A/A FPR in [3%, 7%], N=500, six methods | ±2% | 1.0% | ~22% that one of six fails |
| Phase 8 #2: coverage 88–92%, N unstated | — | — | undefined |
| Phase 7 #1: "within published bias" | — | — | not a test |

**Change:** every rate criterion states N, the seed policy, and a
Clopper–Pearson (or exact binomial) acceptance region at α=0.001. In CI the
seed is fixed (so it is a regression test and says so); nightly rotates the
seed (so it is a calibration test). Phase 7 #1: state the simulated `tau`
grid and the tolerance per estimator. For SBC, name the test: ECDF-difference
with simultaneous confidence bands (Säilynoja, Bürkner & Vehtari 2022), not a
chi-square on binned ranks, and state L (ranks per simulation) and N.

### B6. Golden fixtures for sampled quantities cannot be `rtol`

"`fit_meta_model` posterior summaries at a fixed seed" will not reproduce
across PyMC → NumPyro, nor across BLAS builds for SLSQP. **Change:** split the
manifest into `exact` (closed-form; `rtol ≤ 1e-10`) and `statistical`
(sampled or optimizer-dependent; compared as a z-score against the parent's
recorded MC standard error, `|z| < 4`). Gate `test_manifest` asserts every
entry declares which.

### B7. D9: the data-derived reference removes the failure mode entirely

The plan's worry about option (b) — `y_ref = 1 [unit]` makes the intercept
unit-dependent — is correct. But `log(y / ȳ_geom)` is invariant under a unit
change because the reference scales with the data. **Change:** option (d):
the builder synthesizes `y_ref` as the geometric mean (or a user-declared
reference level) of the outcome, freezes its *numeric value and unit* into
the spec, and an intercept-type estimand differing in `y_ref` is a
`population`-facet difference handled by `transfer_to`. This needs no
warning, serializes, and keeps the checker strict. Ship it in Phase 1 and
close D9.

### B8. Meta-analysis: hand off the predictive distribution, and add HKSJ

`meta/priors.py` "pooled posterior → prior handoff". The prior for a *new*
study is the predictive distribution `N(μ, τ² + σ²_μ)`, not the posterior of
`μ`. Using the latter is the most common meta-to-prior error and the parent
should be checked for it on port. **Change:** `meta.priors` returns the
predictive by default and the gate asserts the returned SD exceeds `τ`. Add
Hartung–Knapp–Sidik–Jonkman to `meta/classical.py` (it is the current default
recommendation for the random-effects CI), and have `influence.egger` refuse
with a reason at k < 10.

### B9. Scale parameters do not pool, and the plan needs a normalization convention

"Shape parameters are the Pi groups that pool" is right. But the
half-saturation `k` is the parameter people most want to pool, and it is a
scale. The standard move is to pool `k / d_ref` with `d_ref` a per-study
reference dose (mean dose, median dose). The *choice* of `d_ref` is an
assumption. **Change:** `meta/schema` carries `reference_dose` per study;
scales enter the pool only as `scale / reference_dose`, with a ledger line
naming the reference convention. Phase 7 exit #5 should test this explicitly.

### B10. Cinelli–Hazlett exit criterion tests the wrong quantity

"Prices it within 15% of its true bias contribution." C–H benchmarking bounds
bias on a contour; it does not estimate it. **Change:** on the planted world,
assert that the confounder's true `(R²_{Y~U|X,D}, R²_{D~U|X})` lies on the
contour where the computed bias equals the true bias, to 1e-6. That is the
identity the implementation must satisfy, and it is exact.

### B11. D-optimality on a nonlinear surface is local

Phase 3 #4 "a D-optimal design beats equal spacing on the D-criterion" — at
which `θ`? **Change:** state that `surface/design.py` ships *Bayesian*
D-optimality (criterion averaged over prior draws via `linearize` at each
draw) with local-at-a-point as a special case, and the test uses a fixed
prior.

### B12. Overlap, LOO, and proxies

- `identify` names "overlap" as the challenge to S-admissibility; nothing
  computes it. Add `diagnose/overlap.py` (propensity/dose overlap and
  positivity). Small.
- No LOO/WAIC anywhere. A Bayesian toolkit without model comparison will be
  asked for it in week one. `SupportsPosterior` gains an optional
  `log_likelihood()` capability; `diagnose/compare.py` does PSIS-LOO
  (~150 LOC, numpy).
- The `proxy` role has no estimator behind it. Keep it as a verdict-only role
  and say so, or drop it.

### B13. Laplace in unconstrained space

Phase 3 #3 should add: the mode-finding and Hessian are taken in the
unconstrained parametrization and draws are mapped back with the transform;
Hessian positive-definiteness is checked and reported alongside
`nonfinite_draw_frac`. The parent's bug was the covariance source; the next
bug is the parametrization.

---

## C. Scope simplifications for 1.0 (simplify)

Each removes work from the critical path without touching a charter success
criterion. Together they are roughly 3,000 LOC, one CI job, one extra, and
~10 days.

| # | Cut or shrink | LOC | Why |
|---|---|---|---|
| C1 | **`ODESystem` → dimension-check-only in 1.0.** No `value`/`jax` interpreter for it. | ~400 + a solver dependency | Nothing ported uses an ODE; carryover is `Convolve`. A jax ODE interpreter means diffrax or `jax.experimental.ode` — dependency creep in the domain layer. Keep the node so the tree is not redesigned later. |
| C2 | **`Deriv` is not a general node.** Kernels ship closed-form `d/d dose`; `MarginalDose` uses it. | ~200 | A symbolic differentiator over 15 node types is a CAS. The jax interpreter gives gradients for free where jax is installed; core should not depend on that. |
| C3 | **`infer/pymc_backend.py` → 1.1.** Drop gate `test_backend_equivalence`. | 350 + CI toolchain | The weight claim is "either backend optional", and one optional backend proves it. The equivalence gate needs a C++ toolchain in CI and a slow KS test, for a backend the plan itself (D2) says to revisit. |
| C4 | **`meta/{privacy,publish}` → 1.1 or out.** | ~700 | DP releases and k-anonymity serve a cross-client benchmark *service*. No single-team user of a stats library needs them, and `[privacy]` names no dependency. |
| C5 | **`identify/narrative.py` → drop.** | 371 | Prose templates. The adapter can own them if anyone asks. |
| C6 | **`surface/ascent.py` → ~150 LOC.** | −250 | On a fitted surface, steepest ascent is `jax.grad` in `optimize.py`. Canonical analysis is `linearize` to second order at a point plus an eigendecomposition. Do not build a parallel quadratic-model pipeline; ridge analysis and lack-of-fit are 1.1. |
| C7 | **`viz` returns tidy frames; no plotly.** | — | Plotly is 51 MB and the wrong default for a library. Every figure helper returns a long-format `DataFrame` with documented columns; an optional matplotlib renderer is fine. |
| C8 | **Gates 6, 8, 9: replace AST heuristics with constructive checks.** | — | Gate 6: make `Interval` a pydantic type whose `definition` and `mass` are required — it is then impossible to construct one without provenance; keep the registry runtime pass. Gate 8: assert no `# noqa: BLE001` in `src/` and let ruff do the rest. Gate 9: the `value == jax` numerical test from A2. Eleven gates stay eleven; three of them stop being fragile. |
| C9 | **Gate 1 threshold.** | — | 400 ms cold including pandas and pydantic is ~1.5× a laptop's `import pandas`; CI runners are slower. Make the threshold relative (`< import pandas + 150 ms`, measured in the same subprocess) and keep the `sys.modules` assertion as the real invariant. |

---

## D. What a user will hit first (reinforce)

### D1. There is no happy path written down

*Partly addressed 2026-08-21:* every subpackage now ships a notebook series as
a phase deliverable, enforced by gate 12 — see `04-contracts-and-testing.md`
"Notebook series". What remains is the part below: the target API written
*before* the code, so the notebooks are a spec the code is built to rather
than a record of whatever API emerged.

The plan specifies representations in depth and never shows the twenty lines
a data scientist types. The five Phase 9 notebooks are the API spec, and they
are being written last. **Change:** add `07-happy-path.md` *before* Phase 1:
one code block per notebook showing the target call sequence — `Panel` →
graph → estimand → fit → result → design → calibrate → pool. If the surface
notebook needs eight spec constructors, the builders are not optional
ergonomics; they are the API. Consequently:

- Minimal builders land with the spec they build, not in Phase 8.
- **Resolve D7 now: yes.** `core/analysis.py` — a frozen container over
  `(graph, surface, posterior, evidence, ledger)`, no math — is what every
  notebook holds and what `io.serialize` writes. The "decide if Phase 3 feels
  clumsy" position defers a decision everyone already knows the answer to.

### D2. Where is the observational panel model?

`surface/model.py` as ported is the designed-experiment surface from
`continuous_learning` (main effects + interaction `gamma` on a small set of
arms). The model 90% of users fit first — hierarchical across units, carryover
plus saturation per treatment, covariates, seasonality, fit to an
observational `Panel` — is the thing `BayesianMMM` was, and the plan drops
`model/base.py` without saying where its *general* core lands. The charter's
"fit a response surface" promises it.

**Change:** name it. Either `surface/model.py` covers both (one tree, one
likelihood, units as a hierarchy level) or there is a `surface/panel_model.py`
with its own recovery test and its own row in the ledger. Either way, Phase 3
exit criteria need a panel world, not just an arms world.

### D3. Units should cost the user one line

D6's "undimensioned user code warns once and is treated as dimensionless"
will fire immediately when a dimensionless dose meets `f(x / k)` with `k` in
dose units, and the user will not understand the error. **Change:** the rule
is "a column adopts the dimension of its role". `Panel(..., roles={"dose":
Dose(unit="USD")})` once; nothing else in the notebook mentions units. The
warn-once path applies only to raw arrays entering `forward()` directly.

### D4. Compute budgets for the trust machinery

500 A/A sims × six methods, SBC with 200+ refits, coverage over N
replications — with NUTS this is hours per gate. **Change:** `diagnose` loops
default to Laplace/SVI through the backend and NUTS is opt-in per call; the
plan states a target wall-clock for each slow gate (e.g., SBC < 20 min on
4 cores).

### D5. Missing-data and panel completeness

`data/` is ~500 LOC and says nothing about unbalanced panels, gaps, or
alignment across units. **Change:** `Panel` validates and *reports*
completeness (a typed summary) and never imputes; every estimator declares
whether it accepts an unbalanced panel.

---

## E. Engineering hygiene (reinforce, small)

- **Schema migration.** `SCHEMA_VERSION` exists; no migration rule does.
  `Spec.from_json` on an older version either applies a registered migration
  or raises naming both versions. Never silently coerces. Put it in
  `core/spec.py` from Phase 1, or `analysis.axiom` files rot at the first
  field rename.
- **Typed failure types have no home.** `Unsupported`, `Blocked`, `Unverified`
  are referenced in every document. Define them once in `core/result.py`
  alongside A1's verdict vocabulary.
- **Seed contract.** Specify: public APIs take `seed: int`, internals take
  `numpy.random.Generator`; the jax interpreter derives a PRNG key from the
  same seed; every sampled result records the seed in its provenance.
- **Content hashing.** Say that specs hold no large arrays (data is referenced
  by the `Panel`'s hash), that `Fraction` and numpy scalars have explicit
  serializers, and that float canonicalization is `repr`.
- **`[netcdf]` extra.** Name the library. Plain `netCDF4` without xarray is
  awkward; `h5netcdf` + arviz's `InferenceData` is the path of least
  resistance but pulls arviz. Recommend: npz is the core format, netCDF is
  produced only by `infer` through arviz, and the manifest records which.
- **Hypothesis** is implied by gate 4 and property-testing `Dimension`'s
  group laws; add it to the dev group explicitly.
- **Single-file artifact.** An `analysis.axiom` *directory* is hard to email
  and easy to half-copy. Same layout inside a zip; `.axiom` is the file.

---

## F. Effort

68 days for ~28k LOC of typed, `mypy --strict`, tested code with eleven gates,
five baked notebooks, and ~15% new mathematics is ~400 LOC/day sustained.
That is optimistic by 1.5× even with an agent carrying the mechanical ports,
and the underestimate is concentrated in two places:

- **Phase 1 (9 days)** contains a small CAS (15 node types × 3 interpreters)
  plus dimensions, specs, protocols, io, and the estimand facet logic. With
  C1 and C2 applied, 12 days is honest; without them, 16.
- **Phase 8 (9 days)** bundles 4,000 LOC of `diagnose`, four builders, the
  adapter, a second backend, and `viz`. With C3 and C7 applied and builders
  moved earlier per D1, 9 days is plausible.

Suggested: Phase 1 → 1a (dimensions, spec, protocols, intervals, io, verdict
and result types, 5 days) and 1b (expression tree and interpreters, estimand
spec, 7 days). Total with the C-section cuts: ~72 days, with the estimate
now believable rather than shorter.

---

## G. Summary of proposed edits by file

| File | Edit |
|---|---|
| `00-charter.md` | B1 (split `conditioning` row), B2 (`version`, `interference` sub-fields), B3 (reword "derived from a graph"), D2 (name the panel model in scope) |
| `01-architecture.md` | A1 (`core/verdict.py`, one `Assumption`), A2 (backend takes a model spec; fifth interpreter), A3 (`Pow` rule), A4 (restate layering), C1/C2 (`ODESystem`, `Deriv` status), C7 (`viz` contract), E (failure types, seed, migration, hashing, artifact) |
| `02-porting-ledger.md` | A5 (`design/cluster.py`), C3–C6 rows moved to 1.1, D2 (a row for the panel model's general core) |
| `03-roadmap.md` | A2 exit criterion, B4 negative control, B5 every rate criterion restated with N and region, B10/B11/B13 exit wording, D1 (builders and `Analysis` earlier), F (Phase 1a/1b, revised totals) |
| `04-contracts-and-testing.md` | A5 token matching, B5 SBC test named, B6 exact/statistical manifest split, C8 gates 6/8/9, C9 gate 1 threshold |
| `05-open-decisions.md` | D7 → resolved yes; D9 → resolved (d); D2 → pymc moved to 1.1, the numpy-NUTS option stays open; new D10: static vs unrolled graphs (B4); new D11: reference-dose convention for pooling scales (B9) |
| *new* `07-happy-path.md` | D1 |
