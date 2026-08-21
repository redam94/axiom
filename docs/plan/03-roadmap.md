# 03 — Roadmap

Nine phases. Each names its deliverables, its exit criteria, and the gate that
keeps it from regressing. A phase is not done until its gate is in CI.

Effort estimates assume one focused developer working with an agent, and are
given in working days.

---

## Phase 0 — Bootstrap and golden capture · 2 days

**Deliverables**

- Repo scaffold, `pyproject.toml`, `Makefile`, CI workflow, this plan. *(done)*
- `scripts/capture_golden.py` — run inside `mmm-framework`'s environment, walk a
  fixture corpus, and write `tests/golden/parent_values.json`: every number
  `axiom` intends to reproduce, keyed by `module::function::case`.

**Why this is first.** The parent repo is live and will keep moving. The moment
`axiom` starts porting, "does this match the parent" stops being answerable by
running the parent. Capture the answers while the parent is pinned at `f1813bc`.

Capture at minimum: `design_factor`, `mean_sd_to_gamma`,
`combine_inverse_variance`, `derive_channel_prior`, `eig_gaussian`,
`eig_monte_carlo`, `sigma_exp_for_design`, `cpa_interval`, `cpa_power`,
`preposterior_sd_ratio`, `compute_evpi`, all six `planning/methods` estimators on
a fixed panel, `optimize_budget` on a fixed curve set, `adstock_weights` ×
`apply_adstock`, all saturation kernels, `build_design_matrix` (assert the
1e-12 invariant), the Cinelli–Hazlett partial-R² path, `fit_meta_model` posterior
summaries at a fixed seed, and every `identification.py` verdict on a corpus of
~25 DAGs.

**Exit** — `pytest tests/golden -q` collects and skips cleanly (nothing
implemented yet), and the JSON has ≥150 recorded values.

**Gate** — `tests/golden/test_manifest.py`: the fixture file parses, is
non-empty, and records the parent commit SHA it came from.

---

## Phase 1 — Foundation: `core`, `data`, `io` · 5 days

**Deliverables**

- `core/entities.py` — `Treatment`, `Dose`, `Unit`, `Outcome`, `Covariate`,
  `Population`, with units and numeraire.
- `core/spec.py` — the frozen `Spec` base: schema version, `content_hash`,
  `diff`, JSON round-trip.
- `core/protocols.py` — `SupportsPosterior`, `SupportsIntervention`,
  `SupportsEstimands`, `SupportsForward`, `Capability`.
- `core/intervals.py` — ETI/HDI with the definition carried on the value.
- `core/stats.py`.
- `data/{frame,roles,scale}.py` — the role-tagged `Panel` and `ScalingParameters`.
- `io/{serialize,provenance,registry}.py` — the `analysis.axiom` directory format.
- `infer/posterior.py` — a plain `Posterior` implementing `SupportsPosterior`
  over a dict of arrays, with an npz round-trip. No sampler yet.

**Exit criteria**

1. `import axiom` puts no sampler, no plotly, no arviz in `sys.modules`, and
   completes in under 400 ms cold.
2. Every `Spec` subclass round-trips: `Spec.from_json(s.to_json()) == s` and
   hashes stably across processes (`PYTHONHASHSEED` varied).
3. An `analysis.axiom` directory containing three specs and a synthetic
   posterior saves, loads, and compares equal — with `pickle` absent from the
   module's imports, asserted by AST inspection.

**Gate** — `tests/contracts/test_import_weight.py`,
`test_spec_roundtrip.py`, `test_no_pickle.py`.

---

## Phase 2 — `identify` · 6 days

**Deliverables**

- `identify/graph.py` — `CausalGraph` as a `Spec`; acyclicity, ancestry,
  d-separation. Own adjacency implementation; `networkx` is not a dependency.
- `identify/backdoor.py` — minimal and all adjustment sets, role assignment.
- `identify/frontdoor.py` — front-door and IV admissibility.
- `identify/verdict.py` — `Verdict` with route, adjustment set, unmeasured
  downgrades, and an honest `status ∈ {identified, downgraded, blocked}`.
- `identify/estimators.py` — OLS, 2SLS, linear front-door with standard errors.
- `identify/endogeneity.py`, `identify/narrative.py`.

**Exit criteria**

1. Reproduces the parent's verdict on all ~25 DAGs in the golden corpus.
2. On `sim` worlds with a known confounded effect: the naive estimate is biased,
   the back-door-adjusted estimate recovers truth within Monte-Carlo error, and
   the IV and front-door estimators recover truth on worlds built for them.
3. A DAG whose adjustment set includes an unmeasured variable produces
   `status="downgraded"` and never a point estimate without an explicit
   `assume_identified=True` that writes a ledger line.

**Gate** — `tests/recovery/test_identify_recovery.py` (marked `recovery`),
`tests/golden/test_verdicts.py`.

---

## Phase 3 — `surface` deterministic half + `infer` · 8 days

**Deliverables**

- `surface/kernels.py`, `surface/carryover.py`, `surface/nuisance.py`.
- `surface/forward.py` — the one `forward()` every other layer calls.
- `surface/linearize.py` — the design matrix and its 1e-12 invariant.
- `surface/design.py` — CCD, Box–Behnken, factorial screening, D/A/E-optimal
  exchange, Latin hypercube.
- `surface/ascent.py` — **new**: steepest ascent, canonical analysis, stationary
  point classification, ridge analysis, lack-of-fit test.
- `infer/backend.py`, `infer/numpyro_backend.py`, `infer/laplace.py`,
  `infer/diagnostics.py`, `infer/_arviz.py`.
- `surface/model.py` — the Bayesian surface: main effects + interaction `gamma`,
  fit through the backend.
- `sim/{dgp,panel,surface_world}.py` — enough to test the above.

**Exit criteria**

1. `‖X @ theta − forward(dose, theta)‖∞ < 1e-12` at any fixed transform point,
   over 200 random configurations.
2. Parameter recovery: on `sim` worlds, NUTS recovers `beta`, `alpha`, and the
   saturation shape inside their 90% intervals at the nominal rate ±3 points
   over 200 replications.
3. `infer.laplace` produces zero non-finite draws on the hierarchical world that
   broke the parent's BFGS path, and `nonfinite_draw_frac` is reported on every
   Laplace result.
4. A D-optimal design beats a naive equal-spacing design on the same budget, on
   the D-criterion, by a margin recorded in the test.
5. Steepest ascent on a known quadratic surface reaches the true optimum within
   tolerance; canonical analysis correctly classifies max / saddle / minimax on
   three constructed surfaces.

**Gate** — `tests/contracts/test_linearize_invariant.py`,
`tests/recovery/test_surface_recovery.py`, `tests/unit/test_laplace_finite.py`.

---

## Phase 4 — `estimands` · 4 days

**Deliverables**

- `estimands/spec.py`, `evaluate.py`, `registry.py`, `graph.py`.
- `EstimandResult` gains `identification: Verdict` (the Phase 2 tie-in).

**Exit criteria**

1. Every estimand in the registry realizes against a `SupportsPosterior` built
   three ways — NUTS, Laplace, and a hand-built dict — with matching values.
2. Every result carries `interval_definition` and `hdi_prob`; a test asserts no
   code path reports an interval without them.
3. An estimand requiring a capability the surface lacks returns
   `status="unsupported"` with a reason, never a wrong number.
4. Golden: the ROI/contribution estimands under `adapters.marketing` reproduce
   the parent's values.

**Gate** — `tests/contracts/test_interval_provenance.py`,
`tests/contracts/test_capability_degradation.py`.

---

## Phase 5 — `design` · 10 days

**Deliverables**

All of `design/`: `power`, `precision`, `eig`, `evoi`, `anchor`, `geo`,
`methods/*`, `simulate`, `optimizer`, `sensitivity`, `structural`, `economics`,
`portfolio`, `schedule`.

**Exit criteria**

1. Golden match on `eig_gaussian`, `eig_monte_carlo`, `cpa_*`, `evoi`, `evpi`,
   and every `methods/` estimator.
2. **A/A calibration**: across all six methods, on 500 A/A simulations, the
   false-positive rate at nominal 5% is within [3%, 7%]. A method that fails is
   marked `experimental` in its registry entry, not quietly shipped.
3. **A/B power**: realized power matches the design's predicted power within 5
   points across three effect sizes.
4. EIG ranking of a small candidate set matches brute-force posterior-variance-
   reduction ranking computed by refitting.
5. `design/structural.py` calls `surface.forward()` — asserted by AST inspection,
   because the parent's version drifted from the model it mirrored.

**Gate** — `tests/recovery/test_aa_calibration.py` (slow),
`tests/contracts/test_structural_uses_forward.py`.

---

## Phase 6 — `calibrate` · 7 days

**Deliverables**

- `calibrate/evidence.py` — `Measurement` records with estimand, scope, design
  factor, and provenance.
- `calibrate/prior.py` — the two-stage prior route.
- `calibrate/likelihood.py` — the in-graph route.
- `calibrate/transfer.py` — **the promoted notebook**: scope transfer,
  chord-vs-marginal, carryover-corrected design factors, dose-path adstocking,
  variance reweighting.
- `calibrate/ledger.py` — **new**: the assumption ledger artifact.
- `calibrate/check.py`.

**Exit criteria**

1. **Known-truth calibration**: build a `sim` world where the observational fit
   is biased by a known amount and a simulated randomized experiment measures
   the truth. Both routes move the posterior toward truth; the likelihood route
   also moves the curve shape, the prior route does not. Both are asserted.
2. Chord-versus-marginal: on a known Hill curve, the naive chord reading of an
   experiment is biased by the amount the analysis predicts, and the corrected
   reading is not.
3. Every call that transfers evidence across a scope boundary appends a ledger
   line. A test runs the full calibration path and asserts the ledger is
   non-empty and every line names an assumption and a counterfactual value.
4. Golden match on `design_factor`, `mean_sd_to_gamma`,
   `combine_inverse_variance`, `lognormal_sigma_from_moments`.

**Gate** — `tests/recovery/test_calibration_recovers_truth.py` (slow),
`tests/contracts/test_ledger_completeness.py`.

---

## Phase 7 — `meta` · 7 days

**Deliverables**

- `meta/{schema,ingest,contribute,store}.py`.
- `meta/classical.py` — **new**: fixed/random effects with DerSimonian–Laird,
  Paule–Mandel, and REML `tau`. Sampler-free, so meta-analysis works in core.
- `meta/pool.py` — the Bayesian hierarchical model through the backend.
- `meta/moderators.py`, `meta/bias.py`, `meta/priors.py`.
- `meta/influence.py` — **new**: leave-one-out, Egger's test, funnel/forest data,
  prediction intervals.
- `meta/{privacy,publish}.py` behind `[privacy]`.

**Exit criteria**

1. On simulated hierarchies with known `tau` and known study means: the
   classical estimators recover `tau` within their published bias, and the
   Bayesian pool's shrinkage matches the analytic normal-normal result.
2. The provenance bias term `delta` is recovered on a simulated corpus where a
   known fraction of contributors report both a model read and an experimental
   read, and is *not* identified (posterior ≈ prior, and the code says so) when
   no contributor reports both.
3. `meta.classical` runs with only the four core dependencies installed —
   asserted in a separate CI job that installs core only.
4. Privacy: the epsilon ledger conserves budget across a release sequence, and a
   cell below the k-anonymity threshold is refused, not clipped.

**Gate** — `tests/recovery/test_meta_recovery.py`,
`.github/workflows/ci.yml` core-only job.

---

## Phase 8 — `diagnose`, `build`, `adapters` · 9 days

**Deliverables**

- All of `diagnose/`: `sbc`, `coverage`, `weak_id`, `sensitivity`, `learning`,
  `spec_curve`, `refute`, `backtest`, `ppc`, `residuals`, `surface_prior`.
- `build/` — the four fluent builders.
- `adapters/marketing.py`.
- `infer/pymc_backend.py`.
- `viz/` — a thin figure layer behind `[viz]`.

**Exit criteria**

1. SBC rank-uniformity passes for the surface model and the meta model.
2. Nominal 90% intervals cover truth 88–92% of the time on `sim` worlds; a
   deliberately mis-specified world fails the same check, proving the check bites.
3. Sensitivity: on a world with a planted confounder of known strength, the
   Cinelli–Hazlett benchmark prices it within 15% of its true bias contribution.
4. Both backends produce statistically indistinguishable posteriors on the same
   model and data (KS test on marginals, per parameter).
5. Every builder produces a `Spec` that round-trips.

**Gate** — `tests/contracts/test_sbc_gate.py`, `tests/contracts/test_coverage_gate.py`,
`tests/contracts/test_backend_equivalence.py` (slow).

---

## Phase 9 — Documentation, notebooks, 1.0 · 6 days

**Deliverables**

- Sphinx API reference.
- Five worked notebooks, each end-to-end and each *baked* in CI:
  1. **Identify** — a confounded world; naive vs adjusted vs IV, with verdicts.
  2. **Design** — from "we should test X" to a powered, costed, method-selected
     design with its EIG and net value.
  3. **Surface** — CCD → fit → canonical analysis → constrained allocation →
     next probe.
  4. **Calibrate** — an experiment lands; both routes; the ledger it produced.
  5. **Meta** — twelve studies in, one pooled prior out, back into the surface.
- `docs/notes/` — one note per place `axiom` deliberately differs from the parent.
- Tag `1.0.0`.

**Exit** — every success criterion in `00-charter.md` holds.

---

## Sequencing notes

- Phases 2, 3, and 7 are independent after Phase 1 and can run in parallel if
  more than one person is on it. Phase 4 needs 3. Phase 5 needs 3 and 4. Phase 6
  needs 4 and 5.
- **Do Phase 0 before touching anything else.** Golden capture against a moving
  parent gets harder every day.
- Resist starting `adapters/marketing.py` early. It is the pressure valve that
  would let marketing vocabulary leak back into the core, and it is cheap once
  the core is right.

**Total: ~64 working days** to 1.0, with roughly a third of that mechanical
porting that an agent can carry.
