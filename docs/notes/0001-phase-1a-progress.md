# 0001 — Implementation progress log

Branch: `feature/phase-1a-foundation` (cut from `develop` 2026-08-21).
Scope is the Phase 1a split proposed in `docs/plan/06-review.md` §F:
dimensions, spec, verdict and result types, protocols, intervals, stats,
entities, `data/`, `io/`, `infer/posterior.py`, the gates that cover them,
and the notebook series for `core`, `data`, `io`, `infer` (the parts that
exist). The expression tree, its interpreters, and `estimands/spec.py` are
Phase 1b and follow on a separate branch.

Decisions made along the way are in [0002](0002-foundation-decisions.md).

## Log

### 2026-08-21 — opened

- Read the plan and the review. Applying from the review before writing code:
  A1 (`core/verdict.py`, one `Assumption` type), A3 (`Pow` rule — matters for
  1b, recorded now), C8 (gate 6 constructive: `Interval` cannot be built
  without `definition` and `mass`), C9 (gate 1 relative threshold), E (schema
  migration rule, typed failure types in `core/result.py`, seed contract,
  hashing rules), D1 (`Analysis` container now, not Phase 4).
- No golden fixture touches Phase 1a, and `../mmm-framework` is not checked
  out in this environment; everything here is written fresh against the plan.
  `core/intervals.py`, `core/stats.py`, `data/*` are marked PORT in the ledger
  — they are ported *by specification* (the ledger row's description), not by
  copying. When the parent is available, diff behaviour and add golden cases
  for `intervals` (ETI/HDI on a fixed draw set).

### 2026-08-21 — Phase 1a code complete on the branch

Landed (all of it `mypy --strict`, black, ruff clean; 112 tests pass; all 9
notebooks execute under nbmake):

- `core/spec.py` — envelope, blake2b content hash, `diff`, subclass
  discovery, migration registry, `SchemaVersionError` / `UnknownSpecError`.
- `core/dimensions.py` — `Dimension` over `Fraction`, declarable `BASES`
  (`D` alias), `UnitSystem` with transitive conversions returning a
  `LedgerLine`.
- `core/verdict.py` — `Status`, `Assumption`, `Verdict` (invariants enforced
  at construction), `LedgerLine`. `core/result.py` — `Unsupported`,
  `Blocked`, `Unverified`, `is_failure`.
- `core/intervals.py` — `Interval` (definition + mass required), `eti`,
  `hdi`, `summarize`. `core/stats.py` — Clopper–Pearson acceptance region,
  ESS (Geyer IPS), MCSE, z-score.
- `core/entities.py` — the vocabulary, `Population` with strata weights,
  `TimeWindow` with basis, `Intervention` with `version` (review B2).
- `core/protocols.py` — `SupportsPosterior`, `SupportsIntervention`,
  `Capability`, `PredictiveDraws`, `missing_capabilities`.
- `core/posterior.py` — `Posterior` with npz round-trip (re-exported from
  `infer`; see 0002.11).
- `data/` — `RoleMap`, `Panel` (validates, sorts, reports `Completeness`,
  canonical CSV hash), `ScalingParameters` / `fit_scaling`.
- `io/` — `Analysis` container, `save_analysis` / `load_analysis`
  (`analysis.axiom` directory, manifest records bases and units),
  `Provenance`, `ArtifactRegistry`.
- Gates 1 (relative threshold), 2 (layer table), 3 (token matching), 4
  (factory table + cross-process hash), 5, 8, 12. Gate 6 is constructive via
  `Interval`; 7, 9, 10, 11 wait for the expression tree and estimands.
- Notebooks: `nbs/core/01,02,04,05,06`, `nbs/data/01,02`, `nbs/io/01`,
  `nbs/infer/01`. `nbs/core/03-expression-tree` is reserved for 1b.

Things found while building, each recorded in 0002: pydantic converts a
`ValueError` raised inside a validator into `ValidationError`, so axiom's
typed errors subclass `Exception`, not `ValueError` (0002.12); pandas
`read_csv` needs `float_precision="round_trip"` to reproduce 17-digit floats
(0002.9); `Posterior` had to move below `io` (0002.11).

## Status

| Module | State | Notes |
|---|---|---|
| `core/spec.py` | done | |
| `core/dimensions.py` | done | |
| `core/verdict.py`, `core/result.py` | done | review A1 + E |
| `core/intervals.py` | done | golden case vs parent still to add when parent is available |
| `core/stats.py` | done | |
| `core/entities.py` | done | |
| `core/protocols.py` | partial | `SupportsForward` / `SupportsEstimands` wait for 1b (`Expr`, `Estimand`) |
| `core/posterior.py` (+ `infer` re-export) | done | |
| `data/{roles,frame,scale}.py` | done | |
| `io/{serialize,provenance,registry,analysis}.py` | done | zip single-file artifact (review E) not yet; directory only |
| gates 1, 2, 3, 4, 5, 8, 12 | done | 6 constructive; 7, 9, 10, 11 need 1b+ |
| `nbs/core`, `nbs/data`, `nbs/io`, `nbs/infer` | done for 1a | |

### 2026-08-21 — Phase 1b complete on `feature/phase-1b-expression-tree`

- `core/expr.py` — `Const`, `Data`, `Param` (+`Prior`), `Add`, `Mul`, `Div`,
  `Pow`, `Apply`, `Convolve`, `Link`, `Opaque`, `Equation`, `System`,
  `ODESystem`; `Expr` is a discriminated union (no node base class);
  `children` / `walk` / `params` / `data_names` / `node_path`.
- `core/interpret/{dimension,value,latex}.py` — dispatch-table interpreters.
  `dimension` raises naming the node path; `value` is `forward()`; `latex`
  degrades to `Unsupported` on `Opaque`.
- `core/protocols.py` — `SupportsForward` (`expr` + `forward`; `linearize`
  joins in Phase 3).
- `estimands/spec.py` — `Quantity`, `Level`, `Estimand` (eight facets, all
  required, derived dimension asserted), `FacetDiff`, `TransferPlan`,
  `transfer_to` with a rule per facet (`_RULES`, asserted complete against
  `FACETS`).
- Gates 10 and 11 (11 parametrized over facets *and* the B2 sub-fields);
  unit tests; `nbs/core/03-expression-tree`, `nbs/estimands/01,02`.
- 196 tests pass; 12 notebooks execute; gate 12 green for `core`, `data`,
  `io`, `infer`, `estimands`.

All seven Phase 1 exit criteria hold (criterion 3's "pickle absent" is gate
5; criterion 5 is `test_checker_and_evaluator_agree_on_200_random_trees`).

## Deferred from Phase 1 (tracked)

- Golden case for `intervals` against the parent — parent not available.
- Single-file `.axiom` zip artifact (review E) — directory format only.
- `SupportsEstimands`, `EstimandResult`, realization — Phase 4 by design.

### 2026-08-21 — Phase 2 (`identify`) complete on `feature/phase-2-identify`

Built with a four-agent parallel workflow (backdoor · front-door/IV ·
transport · sim+estimators), each followed by an independent skeptical
review against the literature and random-graph oracles, then a fix round
verified by fresh agents. Landed:

- `identify/graph.py` — `CausalGraph` (Spec; bidirected edges as hidden
  common causes; `unmeasured`, `selection`, `feedback`), Bayes-ball
  d-separation, `G_{x̲}` / `G_{x̄}` surgery, `from_edges` parser.
- `identify/backdoor.py` — back-door criterion, enumeration, canonical set
  and polynomial existence test (van der Zander et al. 2014), roles.
- `identify/frontdoor.py` — front-door criterion, (conditional) instruments.
- `identify/transport.py` — selection diagrams, direct / S-admissible /
  trivial transportability (B&P 2014 Defs 6–8, Thm 2), `TransportVerdict`.
- `identify/verdict.py` — `identify()` with route preference, alternatives,
  typed assumptions per route, B4 feedback downgrade.
- `identify/estimators.py` — OLS, 2SLS, linear front-door (multivariate
  first stage for the delta-method SE); `identify/endogeneity.py` — DWH and
  Hausman, degenerate branches return `Unverified`.
- `sim/scm.py`, `sim/worlds.py` — `LinearSCM` composing a `CausalGraph`,
  seven named worlds with exact truths and interventional means.
- Recovery tests (`tests/recovery/test_identify_recovery.py`,
  `test_transport_recovery.py`) with negative controls; 507 tests total.
- Notebooks `nbs/identify/01–03`, `nbs/sim/01`.

Reviewer-found defects fixed before merge: transport `blocked` overclaimed
(B&P's own Examples 4/8 are transportable) → trivial route + `unsupported`
fall-through; status convention unified (0002.16); multi-mediator front-door
SE understated by ~12% → multivariate first stage; outcome allowed on the
right-hand side → refused; minimal-set search capped on the wrong candidate
pool → canonical-set enumeration.

**Deferred from Phase 2:** the ~25-DAG golden verdict corpus (parent not
available — `_pending` in the fixture); `identify/narrative.py` dropped per
review C5; sID recursion (Thm 3) not implemented — the verdict says so.
Exit criterion 1 (golden verdicts) is therefore open until the parent is
captured; criteria 2–5 hold.

### 2026-08-21 — Phase 3 (`surface` + `infer`) complete on `feature/phase-3-surface-infer`

Staged build. Foundation (done, by hand): vector `Const`, `Reduce`, `Gather`,
hierarchical `Prior`, `Param.shape`; `core/model.py` (`ModelSpec`,
unconstraining transforms, numpy `log_density`); `core/interpret/jax.py`
(`compile_jax`, `compile_log_density`); `DesignMatrix` + `linearize` on
`SupportsForward`; gate 9 rewritten as numerical `value == jax` agreement
(200 random trees + every shipped `ModelSpec`); `nbs/core/07`. Decision
0002.17.

Round B (agents, parallel, each with a reviewer): `surface/{kernels,
carryover,nuisance}.py`; `infer/{backend,laplace,numpyro_backend,
diagnostics,_arviz}.py`; `surface/{design,ascent}.py`.
Round C: `surface/{model,forward,linearize,optimize,frontier}.py`,
`sim/{surface_world,panel}.py`.

Each round was followed by an independent adversarial review, a fix round,
and a verification round; the decisions those forced are 0002.17–0002.19.
Headline defects caught before merge: carryover weights mis-normalized for
draw-shaped parameters (a `Reduce` semantics gap in core); NaN jax gradients
in `k` at zero dose for `s < 1`; Laplace trusting BFGS-free but
absolutely-scaled tolerances (wrong mode, wrong SD, fabricated "verified"
covariance on non-unit problems); a carryover surface convolving allocation
rows as time; design criteria rejecting well-posed designs in raw dose
units; ascent finite differences with an absolute floor. Every one of these
was a *wrong number*, not a crash — which is the failure class the charter
is written against.

Recovery: `tests/recovery/test_surface_recovery.py` (Laplace recovery,
negative control with the wrong kernel family, unit invariance of a fit,
NUTS coverage over N=60 worlds as a `slow` test with a Clopper–Pearson
region). Notebooks: `nbs/surface/01–05`, `nbs/infer/01–03`, `nbs/sim/02`,
`nbs/core/07`.

**Phase 3 exit criteria:** 1 (linearize invariant, 200 random surfaces,
< 1e-12) ✓; 2 (NUTS recovery at nominal rate) — written as the `slow`
test `test_nuts_interval_coverage` (N=60, Clopper–Pearson α=0.001), not
executed in this session; Laplace recovery ✓; 3 (Laplace zero non-finite
draws on the hierarchical world, `nonfinite_draw_frac` always reported) ✓;
4 (D-optimal beats equal spacing, margin recorded in the test) ✓; 5
(steepest ascent reaches the optimum; canonical analysis classifies
max/saddle/min) ✓; 6 (every kernel dimension-checks; shapes invariant,
scales scale by the conversion factor, fit in two units) ✓; 7 (gate 12
green for `surface`, `infer`, `sim`) ✓. Gate 9 is the numerical
`value == jax` agreement (A2/C8). Totals: 968 tests, 25 notebooks.

**Deferred from Phase 3:** `infer/pymc_backend.py` (review C3 → 1.1);
`surface/arms.py`, `surface/acquire.py`, `surface/sequential.py` (the wave
loop and acquisition — not needed by Phases 4–6; revisit in Phase 8 or
1.1); ridge analysis and lack-of-fit (C6, 1.1); D9's log-reference
convention is still open (the nuisance-conventions mechanism is the hook
for it).

### 2026-08-21 — Phase 4 (`estimands` realization) in progress on `feature/phase-4-estimands`

`SupportsEstimands` added to `core.protocols` (a fitted producer = posterior
+ counterfactuals + declared estimands + the units it was fit in). Two
agents in parallel: (A) `FitResult` becomes the producer
(`predict_under`, `marginal_under`, `capabilities`, `restrict`) and
`estimands/evaluate.py` (`EstimandResult`, `realize`, `evaluate`); (B)
`estimands/registry.py` and `estimands/graph.py` (the estimand as an
expression, for gate 10). Gates 6 and 7 drafted against the spec.

Landed: `FitResult` is the `SupportsEstimands` producer (`predict_under`
with set/scale/shift on a support window, `marginal_under` as the derivative
of the windowed outcome, `capabilities`/`restrict`, units it was fit in);
`estimands/evaluate.py` (`EstimandResult`, `realize`, `evaluate`) with
capability, identification, dimension and unit checks — every refusal typed,
every unit crossing a ledger line; `estimands/registry.py` (the domain-general
standard library); `estimands/graph.py` (the estimand as an expression tree
with `substitute`, refusing where a per-row tree cannot express the
aggregated functional); gates 6 and 7; `nbs/estimands/03`.

Reviewer-caught before merge: intervention levels were passed to the
producer in the estimand's unit while outputs were converted (a wrong number
with a plausible ledger line); negative realized doses under shift/scale;
a windowed marginal that was not the derivative of the windowed outcome
under carryover (the per-period-vs-cumulative class again); two different
`ratio` definitions under one name → ratified as 0002.20.

**Phase 4 exit criteria:** 1 — hand-built dict and Laplace agree exactly;
the NUTS leg is a `slow` test; 2 ✓ (constructive + runtime gate 6); 3 ✓
(gate 7 over every capability); 4 ✓ (dimension mismatch raises; unit
conversion with ledger line); 5 — golden ROI/contribution is an
`adapters.marketing` deliverable (Phase 8); 6 ✓.

**Deferred:** conditional (stratified) estimands realize in Phase 6 with the
transfer machinery (`Unsupported` until then, with that reason).

### 2026-08-21 — Phase 5 (`design`) complete on `feature/phase-5-design`

The first build attempt died on a session limit, and the uv-managed CPython
3.13.1 was later found SIGKILLed on launch (reinstalled with
`uv python install --reinstall 3.13.1`); the phase was rebuilt in three
parallel builders plus a calibration fix round.

Delivered: `eig`, `evoi`, `precision` (cost per outcome, Fieller), `anchor`,
`power`, `cluster` (the parent's geo design, renamed), `schedule`,
`structural` (Fisher information by differentiating `forward`; jax or
central differences, agreeing to 1e-6), `methods/*` (registry with typed
method assumptions + DiD, synthetic control, time-based regression,
cluster-based regression, switchback, ghost), `simulate` (A/A, A/B,
`calibrate_registry`, leaderboard), `economics`, `portfolio`, `optimizer`,
`sensitivity`; 127 exported symbols; `nbs/design/01–06`; decisions
0002.21–0002.24.

**Phase 5 exit criteria:** 1 ✓ all 14 `planning.*` golden cases at 1e-12
(`tests/golden/test_design.py`); `methods/` estimators have no parent
fixture — recovery tests against seeded panels instead; 2 ✓ A/A gate on
500 simulations, 40×24 panel: DiD 23, SC 29, TBR 29, CBR 25, switchback 29,
ghost 31 false positives (region [11, 42], all in [3 %, 7 %]); all six
`stable`; 3 ✓ DiD realized power 0.353/0.686/0.946 vs predicted
0.363/0.707/0.947; 4 — EIG vs refit ranking is shown in `nbs/design/02`
(MC converges to the closed form) rather than a refit-based test; 5 ✓
(gate 9 is numerical); 6 ✓.

Fix round (A/A): every method used a normal critical value on a small-df
variance → Student-t at the recorded df (0002.21); switchback lacked the
unit-fixed-effect dof factor; synthetic control's pseudo-treated subsets
fit each donor on a single remaining donor (SE 6× too large) → per-donor
placebo noise with the `‖w̄‖²` donor-overlap correction (measured
sd/SE ratio 1.009). TBR's forecast variance assumes white residuals; the
DGP's AR(1) shock leaves ~8 % excess sd, documented in its docstring.

**Deferred:** block-bootstrap SE for switchback (HAC only); a refit-based
EIG ranking test (needs the Phase 6 transfer machinery for a fair
comparison); parent ports are "by specification" — `../mmm-framework` is
not on this machine, so the ledger's PORT rows for `planning/*` were
implemented from the plan and the golden fixture.

### 2026-08-21 — Phase 6 (`calibrate`) complete on `feature/phase-6-calibrate`

Delivered: `evidence` (`Measurement`, `combine_inverse_variance`), `prior`
(`design_factor`, `mean_sd_to_gamma`, `lognormal_from_moments`,
`amplitude_prior`, `combine_measurements`, `derive_prior` → `CalibratedSpec`),
`likelihood` (`lognormal_sigma_from_moments`, `constraint_for`, `attach`,
`fit_calibrated`), `transfer` (five correction operators + `resolve` →
`ResolvedTransfer`), `ledger` (`Ledger`, `check_complete`), `check`
(`agreement`); `core.Constraint` + `ModelSpec.constraints` (numpy and jax
densities, gate 9 covers it); `amplitude_prior` on every surface kernel;
`nbs/calibrate/01–04` and a `Constraint` section in `nbs/core/07`;
decisions 0002.25–0002.27. 31 exported symbols.

**Phase 6 exit criteria:** 1 ✓ with a nuance — on a confounded panel
(6×40, bias z = +4.5 on the amplitude) both routes move the amplitude
toward truth and the likelihood route reaches `agrees` (z = −0.78); the
roadmap's "the prior route does not move the shape" is true of what the
route *adds to the density* (asserted exactly: the prior change is
invariant to k/s shifts at 1e-12) but not of the posterior, where the
tightened amplitude prior makes the likelihood re-explain the confounding
through k/s. The test asserts the true statement and documents it. 2 ✓
chord·factor = closed-form derivative at 1e-10; 3 ✓
`tests/contracts/test_ledger_completeness.py` with a negative control;
4 ✓ every `calibration.*` fixture at 1e-12 (`design_factor` is the mean of
per-draw ratios — the only candidate that reproduces the fixture); 5 ✓.

**Deferred:** a single-treatment-at-a-time `derive_prior` (multi-treatment
design factors need the transfer machinery for cross-treatment scope);
`marginal` constraints under carryover (typed `Unsupported`); per-unit
dose grids in `constraint_for`; `FitResult` does not carry the constrained
model (recoverable from `provenance["constrained_model_hash"]` or `attach`).
Roadmap wording for criterion 1 should say "at the density level".

### 2026-08-21 — Phase 7 (`meta`) complete on `feature/phase-7-meta`

Delivered: `schema` (`StudyRecord`, `Corpus`, the poolable-quantity
catalogue), `ingest` (`normalize` refusing dimensioned records without a
licensed `TransferPlan`), `contribute`, `store` (`CorpusStore` over
`io.ArtifactRegistry`), `classical` (FE, DL/PM/REML, Knapp–Hartung,
prediction intervals; core-only), `pool` (marginal hierarchical model through
`infer`, exact-conditional `theta` draws), `moderators`, `bias`
(`delta_identification`), `priors` (`prior_from_pool` → `core.Prior` +
`LedgerLine`), `influence` (LOO, Egger, funnel/forest/Baujat data),
`privacy` + `publish` (k-anonymity, dominance, epsilon ledger, Laplace /
analytic-Gaussian releases, churn-gated carry-forward); `core.Likelihood.
scale_expr`, `likelihood_scale`, `ModelSpec.data_columns`; 70 exported
symbols; `nbs/meta/01–05`; CI core job runs the classical tests
explicitly; decisions 0002.28–0002.31.

**Phase 7 exit criteria:** 1 ✓ 200 replications: tau² relative bias DL
+0.7 %, PM −1.3 %, REML −1.0 % (bounds 15/10/10 % with references); KH
coverage 191/200 inside the exact region; shrinkage vs `se²/(se²+tau²)`
within 0.77 %; 2 ✓ delta 0.276 ± 0.077 for truth 0.4 with 30 % dual-read
contributors; not identified → posterior sd = 1.005 × prior sd and the
verdict is `blocked`; 3 ✓ `classical`/`influence` import core only; the
core CI job runs them; 4 ✓ ledger conserves budget to 1e-12, k−1 cell is
`Blocked`; 5 ✓ two studies in different units refused by `normalize`,
admitted with licensed plans; 6 ✓.

**Deferred:** a centered parametrization with free tau is refused
(`Unverified`) rather than sampled — a NUTS path for it is a 1.1 item; the
`[privacy]` extra is nominal (scipy is core) and can be dropped in Phase 9.

### 2026-08-21 — Phase 8 (`diagnose`, `build`, `adapters`, `viz`) complete on `feature/phase-8-diagnose-build`

Delivered: all eleven `diagnose` modules (79 symbols) with the SBC and
coverage gates (fast tier by default, `slow` tier at the roadmap's N);
`build/` (Fields/Builder composition; Prior, Variable, Surface, Study, Meta,
Graph builders); `adapters/marketing.py` (the only marketing-named module;
ROAS/ROI/contribution/marginal ROAS as `EstimandResult`s); `viz/` (seven
plotly figures behind a lazy import, `Unsupported` without plotly);
`tests/contracts/test_backend_equivalence.py` (slow, Laplace vs NumPyro);
`nbs/diagnose/01–05`, `nbs/build/01`, `nbs/adapters/01`, `nbs/viz/01`;
decisions 0002.32–0002.34. `.gitignore`'s `build/` became `/build/` so the
package is tracked. A Phase 3 defect surfaced and was fixed: `FitResult.
marginal_under` on a `LinearKernel` surface dropped the unit axis
(`surface/model.py::_select`), with a regression test.

**Phase 8 exit criteria:** 1 ✓ SBC rank uniformity at N=200 for the
linear surface under Laplace (p 0.23–0.92), the pool model (mu p=0.81, tau
p=0.37), and the Hill surface under NumPyro (all p ≥ 0.19); negative
controls fail at p ≈ 1e-36; 2 ✓ 90 % intervals inside the exact region
[168, 190] at n=200; the carryover-mis-specified world fails (alpha 1/30);
the roadmap's "88–92 %" is asserted as the exact Clopper–Pearson region;
3 ✓ planted confounder priced within 2.1 % (bound 15 %); Darfur reproduced;
4 ✓ Laplace vs NUTS means within 0.25 sd, sds within 25 % on every
parameter; KS > 0.01 on the structural parameters (half-normal scales are
the documented normal-approximation caveat); 5 ✓ every builder round-trips;
6 ✓ gate 12 green for all fifteen subpackages.

**Finding worth a 1.1 ticket:** Laplace is measurably miscalibrated on
the Hill surface at SBC resolution (k ranks pile at the top bin when the
true k is large; small-panel skew on half-normal amplitude priors). The
SBC gate detects it, which is the point; the Hill tier of the gate runs
under NumPyro. Users fitting Hill surfaces with Laplace should run
`diagnose.sbc_surface` on their panel shape.

**Deferred:** PyMC backend (1.1, D2); block bootstrap for switchback;
Säilynoja-style simultaneous ECDF bands (DKW used, conservative).

### 2026-08-21 — Phase 9 (documentation, end-to-end notebooks, 1.0) complete

Delivered: Sphinx reference (`docs/conf.py`, `docs/api/*` — one page per
subpackage linking its notebook series; plan and notes included; `make
docs`, CI `docs` job, `-W` clean with a documented suppression for eight
Markdown-table docstrings); `nbs/end-to-end/01–05` (identify, design,
surface, calibrate, meta), each a worked decision crossing several
subpackages; `tests/contracts/test_analysis_roundtrip.py` (charter
criterion 8: graph, estimand + result, surface fit, design, calibration +
ledger, meta corpus + pooled estimate saved and reloaded with bit-identical
numbers and no pickle); `docs/notes/0003-deviations-from-parent.md` (22
deliberate divergences, plus the charter numbers measured); README and
CLAUDE.md status; version `1.0.0`.

Gaps the end-to-end notebooks exposed and what was done: `meta.pool`
refused a scaled record `normalize` had admitted under a licensed plan —
`pool` now honours the admission stamped in `detail` (test added);
`central_composite` needs `inscribed=True` on a dose box (shown in
`nbs/end-to-end/03`); `LinearEstimate` has `n` but no `df` (derived as
`n − 2` for `robustness_value`); `canonical_analysis` reports a near-flat
stationary point as a maximum rather than a ridge — recorded for 1.1.

**Charter success criteria at 1.0:** 1 — `import axiom.core` adds ~60 ms
over its four dependencies; no sampler in `sys.modules` (gate 1); 2 —
38.5 k lines with docstrings (32.8 k non-blank) against a 30 k target:
recorded as an overrun, not trimmed; no marketing noun outside `adapters`
(gate 3); 3 ✓ every captured fixture at tolerance, `_pending` documented;
4 ✓ gate 7/11; 5 ✓ gate 10; 6 ✓ recovery suites for identify, surface,
design (A/A, A/B), calibrate, meta, sensitivity; 7 ✓ SBC and coverage
gates for the surface and pool models; 8 ✓ `test_analysis_roundtrip`;
9 ✓ `make gates`; 10 ✓ 53 notebooks, gate 12 green for all fifteen
subpackages.

Final state: 1687 fast tests (+ slow tiers run once at each phase),
53 notebooks, ruff/black/mypy --strict clean, docs `-W` clean.

## Post-1.0 — the HYPER-3 case study and `design.sequential`

Written after 1.0 was tagged, on `feature/case-study-hypertension`, in
answer to "what does using all of this on one real problem look like".
`nbs/case-studies/hypertension/` is a sequential dose-finding trial in
hypertension: three doses against the standard of care, three age strata,
weekly blood-pressure monitoring, and a boundary that stops a dose arm if
it is harming people. Six notebooks, one shared synthetic world in
`hyper3.py`, 37 figures, crossing `core` → `data` → `sim` → `identify` →
`estimands` → `surface` → `design` → `meta` → `viz`.

The case study needed one thing the library did not have, so
`src/axiom/design/sequential.py` landed with it: group-sequential
boundaries (Pocock, O'Brien–Fleming, Lan–DeMets spending, and a
posterior-probability harm rule stated the way a monitoring committee
states it), exact first-crossing probabilities by numerical integration of
the canonical joint distribution rather than by simulation, and `monitor`
for walking a realized path against a rule. 22 public symbols, 45 unit
tests pinning the published boundary tables and the recursion's
second-order convergence, gate-4 factories, and `nbs/design/07-sequential`
for gate 12. Design decisions and the three first-draft claims the numbers
refused are in note 0004.

Version is left at 1.0.0 on `develop`; the new API is a 1.1.0 item.

## Next (1.1)

PyMC backend (D2); Laplace calibration on Hill surfaces (SBC finding,
Phase 8); block-bootstrap switchback SE; ridge verdict in
`canonical_analysis`; `df` on `LinearEstimate`; drop the nominal
`[privacy]` extra; NUTS path for centered pools; conditional estimands
beyond one stratifying covariate; stage-wise-ordered estimates for a
sequential trial stopped at a boundary (note 0004).
