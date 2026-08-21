# 02 — Porting ledger

Every source file in `mmm-framework/src/mmm_framework/` gets a verdict. LOC
figures were measured on 2026-08-20 at commit `f1813bc`.

**Verdicts**

- `PORT` — the mathematics is right and domain-general. Move it, rename the
  vocabulary, keep the numbers bit-identical (golden test).
- `PORT+` — port the mathematics, then extend it. The extension is named.
- `REWRITE` — the idea survives, the code does not. Usually because it is
  coupled to `BayesianMMM` or PyTensor.
- `EXTRACT` — a large file with a small good part. Take the part.
- `DROP` — out of scope.

`SRC` is the parent path relative to `src/mmm_framework/`.

---

## identify/ — causal identification (~1,600 LOC target)

| SRC | LOC | → axiom | Verdict | Notes |
|---|---|---|---|---|
| `estimators/causal.py` | 183 | `identify/estimators.py` | **PORT** | Pure numpy OLS / 2SLS / linear front-door. Cleanest file in the parent. Rename `channel`→`treatment`. |
| `dag_model_builder/identification.py` | 721 | `identify/{graph,backdoor,frontdoor,verdict}.py` | **PORT** | The d-separation, adjustment-set search, and role assignment. Split by concern. Drop the MMM `NodeType` enum for open role tags. |
| `dag_model_builder/narrative.py` | 371 | `identify/narrative.py` | **PORT** | Human-readable DAG reading. Degeneralize the marketing phrasing into templates the adapter fills. |
| `dag_model_builder/validation.py` | 372 | `identify/graph.py` | **EXTRACT** | Only acyclicity + connectivity + reachability survive (~120 LOC). The rest validates MMM node typing. |
| `diagnostics/endogeneity.py` | 193 | `identify/endogeneity.py` | **PORT** | Endogeneity tests that decide whether the IV route is needed. |
| `dag_model_builder/builder.py` | 916 | — | **DROP** | Builds a `BayesianMMM`. Replaced by `build/graph.py` + `build/surface.py`. |
| `dag_model_builder/config_translator.py` | 805 | — | **DROP** | DAG → MMM config. Not a concept here. |
| `dag_model_builder/model_type_resolver.py` | 233 | — | **DROP** | Picks between MMM extension classes. No extension classes. |
| `dag_model_builder/frontend_adapter.py` | 424 | — | **DROP** | React Flow serialization. |
| `dag_model_builder/{dag_spec,node_configs}.py` | 502 | `identify/graph.py` | **REWRITE** | Node/edge specs become `Spec` subclasses; ~180 LOC. |

**Extension (PORT+):** the parent reports front-door/IV *identifiability* and,
since `estimators/causal.py`, estimates it. What it does not do is carry the
identification verdict onto the estimand result. In `axiom`, an `EstimandResult`
gets an `identification: Verdict` field, and `evaluate` refuses to report a
causal quantity whose route the graph says is blocked unless the caller passes
`assume_identified=True` — which writes a ledger line.

---

## estimands/ — declarative counterfactuals (~1,400 LOC target)

| SRC | LOC | → axiom | Verdict | Notes |
|---|---|---|---|---|
| `estimands/spec.py` | 423 | `estimands/spec.py` | **PORT** | The single best asset in the parent. Frozen Pydantic, schema version, interval-definition provenance on every result. Rename `ALL_CHANNELS`→`ALL_TREATMENTS`, `MarginalSpend`→`MarginalDose`. |
| `estimands/evaluate.py` | 693 | `estimands/evaluate.py` | **REWRITE** | ~60% of the math survives; the `BayesianMMM` coupling is replaced by `SupportsEstimands`. |
| `estimands/registry.py` | 313 | `estimands/registry.py` | **PORT** | Named estimand library. Marketing presets move to the adapter. |
| `estimands/capabilities.py` | 88 | `core/protocols.py` | **REWRITE** | Becomes `frozenset[Capability]` on the protocol. |
| `estimands/interventions.py` | 40 | `estimands/spec.py` | **PORT** | Fold in. |
| `estimands/graph.py` | 78 | `estimands/graph.py` | **REWRITE** | In-graph estimand tensors, written against `infer.Backend` rather than PyTensor directly. |

---

## surface/ — response-surface methodology (~4,200 LOC target)

| SRC | LOC | → axiom | Verdict | Notes |
|---|---|---|---|---|
| `continuous_learning/surface.py` | 321 | `surface/model.py` | **PORT** | The canonical JAX surface: Hill activation + upper-triangular interaction `gamma`. Its stated invariant — likelihood, DGP, and planner all call the *same* function — is the design principle for this whole package. |
| `transforms/saturation.py` | 118 | `surface/kernels.py` | **PORT+** | Add exponential, power, and a monotone I-spline family. |
| `transforms/adstock.py` | 277 | `surface/carryover.py` | **PORT** | Geometric / delayed / Weibull, normalized FIR. |
| `transforms/carryover.py` | 383 | `surface/carryover.py` | **PORT** | Merge with the above; one module. |
| `transforms/adstock_pt.py` | 192 | — | **DROP** | PyTensor-specific. Replaced by per-backend compilation of one kernel spec. |
| `frequentist/design.py` | 496 | `surface/linearize.py` | **PORT** | Builds the linear design whose invariant is `X @ theta == mu` to 1e-12 at a fixed transform point. Load-bearing for sensitivity benchmarking and structural design. |
| `continuous_learning/design.py` | 193 | `surface/design.py` | **PORT+** | Central-composite (center / axial / off-axis pairs / shutoff cells). Add Box–Behnken, D/A/E-optimal exchange, Latin hypercube, and the classical two-level screening designs. This is where RSM as a discipline lands. |
| `continuous_learning/planner.py` | 899 | `surface/optimize.py` | **PORT** | `jax.grad` allocator over the fitted surface. |
| `continuous_learning/acquisition.py` | 745 | `surface/acquire.py` | **PORT** | Knowledge-gradient / EI acquisition. Shares math with `design/eig.py`; deduplicate on port. |
| `continuous_learning/model.py` | 1244 | `surface/model.py` | **EXTRACT** | The NumPyro likelihood and priors (~500). The service scaffolding is dropped. |
| `continuous_learning/loop.py` | 937 | `surface/sequential.py` | **EXTRACT** | The wave loop trimmed to ~300; no persistence, no service. |
| `continuous_learning/stationarity.py` | 327 | `surface/sequential.py` | **PORT** | Trust-region / drift checks between waves. |
| `continuous_learning/{arms,evidence,scaling,preprocess}.py` | 902 | `surface/`, `data/scale.py` | **EXTRACT** | ~350 survives. |
| `continuous_learning/{service,serialize}.py` | 1564 | — | **DROP** | Platform glue; `io/serialize.py` replaces the latter. |
| `planning/budget.py` | 1202 | `surface/optimize.py` | **PORT** | SLSQP allocator, warm start, constraints, per-unit curves. Strip `GEO_ARM_SEP` string-keying — units are first-class here. Keep the SLSQP silent-failure fix (parent issue #290). |
| `planning/frontier.py` | 417 | `surface/frontier.py` | **PORT** | Budget frontier + goal seek → "effort frontier". |
| `planning/decision_arms.py` | 556 | `surface/arms.py` | **PORT** | Discrete levers alongside continuous dose. Generalizes cleanly (price/promo → any discrete arm). |
| `diagnostics/saturation.py` | 255 | `diagnose/surface_prior.py` | **PORT** | Prior-predictive check on the curve shape before fitting. |
| `transforms/{seasonality,trend,events,event_shapes}.py` | 903 | `surface/nuisance.py` | **EXTRACT** | ~300 LOC. Needed only to *profile out* baseline in design and identification math. Drop the `holidays` dependency — take a user-supplied event frame. |
| `model/components/trend.py` | 352 | `surface/nuisance.py` | **EXTRACT** | Flexible-trend basis, ~120 LOC. |
| `model/base.py` | 5375 | — | **DROP** | The thing being replaced. Two things are extracted from it by hand: the Laplace `trust-ncg` covariance fix (→ `infer/laplace.py`) and `_get_time_mask` semantics (→ `estimands/spec.py`). |

**Extension (PORT+):** the parent has no *classical* RSM. There is no steepest
ascent, no canonical analysis of the stationary point, no ridge analysis, no
lack-of-fit test. `surface/ascent.py` is new code (~400 LOC): first-order path,
second-order canonical form, eigen-classification of the stationary point
(maximum / minimax / saddle), and constrained ridge analysis. This is the single
largest genuinely-new module in Phase 3.

---

## design/ — experimental design and planning (~5,500 LOC target)

| SRC | LOC | → axiom | Verdict | Notes |
|---|---|---|---|---|
| `planning/eig.py` | 226 | `design/eig.py` | **PORT** | Gaussian and Monte-Carlo EIG, information half-life, re-experiment timing. Pure numpy. |
| `planning/evoi.py` | 379 | `design/evoi.py` | **PORT** | EVOI / EVPI, preposterior SD ratio, surrogate. |
| `planning/design.py` | 939 | `design/{power,geo}.py` | **PORT** | Split: general power/MDE vs geo-specific design. |
| `planning/cpa.py` | 212 | `design/precision.py` | **PORT** | Generalize cost-per-acquisition → cost-per-outcome-unit. |
| `planning/design_anchor.py` | 216 | `design/anchor.py` | **PORT** | Model-anchored effect size; "powered to detect what the model already believes". |
| `planning/methods/*` | 1263 | `design/methods/*` | **PORT** | Registry + synthetic control, TBR, GBR, ghost ads, switchback, DiD/MMT. Highest-value single block after `estimands/spec.py`. Move wholesale. |
| `planning/simulation.py` | 923 | `design/simulate.py` | **PORT** | A/A and A/B simulation, methodology leaderboard. The A/A false-positive check becomes a suite gate. |
| `planning/experiment_optimizer.py` | 1231 | `design/optimizer.py` | **PORT** | Pareto front over candidate designs, cooldown. |
| `planning/experiment_sensitivity.py` | 631 | `design/sensitivity.py` | **PORT** | How the design recommendation moves with its assumptions. |
| `planning/identification.py` | 595 | `design/structural.py` | **REWRITE** | Fisher/Laplace design for structural parameters (`beta`/`alpha`/`psi` on the equifinality ridge). The math is excellent; the implementation byte-mirrors `BayesianMMM`'s forward op in numpy. Rewrite it to call `surface.forward()` — that is the whole point of `SupportsForward`. |
| `planning/opportunity_cost.py` | 554 | `design/economics.py` | **PORT** | Signed opportunity cost of running the test. |
| `planning/experiment_value.py` | 273 | `design/economics.py` | **PORT** | Net value = information value − opportunity cost. |
| `planning/discount.py` | 104 | `design/economics.py` | **PORT** | Discount weights, mid-horizon factor. |
| `planning/priority.py` | 292 | `design/portfolio.py` | **PORT** | Ranking channels/treatments by what is worth learning. |
| `planning/experiments.py` | 210 | `design/portfolio.py` | **PORT** | Recommendation entry point. |
| `planning/history.py` | 233 | `design/portfolio.py` | **EXTRACT** | Past-study summaries feeding priors, ~100 LOC. |
| `planning/flighting.py` | 170 | `design/schedule.py` | **PORT** | Temporal-contrast patterns. Feeds `design/structural.py`: pulsing is what identifies carryover. |
| `finance/{closure,lines,valuation,evidence}.py` | 1562 | `design/economics.py` | **EXTRACT** | ~90 LOC. Value-of-information needs one scalar, `value_per_outcome_unit`, plus its provenance. The rest is finance reporting. |
| `planning/{pacing,calendar,forecast,variance,payback}.py` | 3277 | — | **DROP** | Marketing-operations reporting. Out of scope per the charter. |

---

## calibrate/ — folding experiments into models (~2,000 LOC target)

| SRC | LOC | → axiom | Verdict | Notes |
|---|---|---|---|---|
| `calibration/experiment.py` | 690 | `calibrate/{evidence,prior}.py` | **PORT** | `design_factor`, `mean_sd_to_gamma`, `combine_inverse_variance`, `derive_channel_prior`. Crown jewels. |
| `calibration/likelihood.py` | 864 | `calibrate/likelihood.py` | **REWRITE** | Keep `lognormal_sigma_from_moments` and the estimand-expression construction verbatim; rewrite graph attachment against `infer.Backend`. |
| `validation/calibration.py` | 202 | `calibrate/check.py` | **PORT** | Did the calibration take? Posterior-vs-evidence agreement. |
| *(not in `src/`)* `nbs/demos/calibrate_external_curve.ipynb` | — | `calibrate/transfer.py` | **REWRITE** | **This is a real gap in the parent.** Scope transfer, chord-versus-marginal correction, carryover-corrected design factors, spend-path adstocking, and variance reweighting exist only in a 75-cell notebook and a standalone HTML explainer. Promote them into library code with tests. ~600 LOC. |
| *(new)* | — | `calibrate/ledger.py` | **NEW** | Every transfer, assumption, and override appends a typed line: what was assumed, why, and what the number would be without it. The parent's "assumption ledger" is prose in a notebook; here it is a serialized artifact. ~250 LOC. |

---

## meta/ — meta-analysis (~2,200 LOC target)

| SRC | LOC | → axiom | Verdict | Notes |
|---|---|---|---|---|
| `benchmarks/schema.py` | 238 | `meta/schema.py` | **PORT** | `ContributionRecord`, `LiftStudyRecord`, `StudyForm`, `Provenance`. |
| `benchmarks/meta_model.py` | 261 | `meta/pool.py` | **PORT** | Bayesian random-effects pooling: per-family `tau`, moderators, provenance bias `delta_m` identified by dual-read contributors. Rewrite the PyMC block against `infer.Backend`. |
| `benchmarks/estimands.py` | 121 | `meta/schema.py` | **PORT** | The catalog of poolable quantities; wire to `axiom.estimands`. |
| `benchmarks/ingest.py` | 188 | `meta/ingest.py` | **PORT** | Study normalization. |
| `benchmarks/priors.py` | 87 | `meta/priors.py` | **PORT** | Pooled posterior → prior handoff into `calibrate` and `surface`. Closes the loop. |
| `benchmarks/store.py` | 244 | `meta/store.py` | **REWRITE** | Its own sqlite DB in the parent; here a plain content-addressed directory via `io/registry.py`. |
| `benchmarks/privacy.py` | 187 | `meta/privacy.py` | **PORT** | k-anonymity, dominance cells, epsilon ledger. `[privacy]` extra. |
| `benchmarks/publish.py` | 508 | `meta/publish.py` | **PORT** | Clipped-statistic DP releases, orthogonalized epsilon, churn-gated carry-forward. `[privacy]` extra. |
| `benchmarks/contribute.py` | 227 | `meta/contribute.py` | **PORT** | Project-side record emission. |

**Extension (PORT+):** the parent's meta-analysis exists to serve a
cross-client benchmark product, so it only pools *contribution-shaped* records.
Generalize to a standard meta-analysis surface: fixed vs random effects,
`tau` estimators for the closed-form path (DerSimonian–Laird, Paule–Mandel,
REML) so pooling works in core with no sampler, forest and funnel data
structures, Egger's test, leave-one-out influence, and prediction intervals.
~700 LOC of new code in `meta/classical.py` and `meta/influence.py`.

---

## infer/ — the sampler seam (~1,600 LOC target)

| SRC | LOC | → axiom | Verdict | Notes |
|---|---|---|---|---|
| *(new)* | — | `infer/backend.py` | **NEW** | The protocol. ~150 LOC. |
| *(new)* | — | `infer/numpyro_backend.py` | **NEW** | NUTS / SVI / Laplace. ~450 LOC. |
| *(new)* | — | `infer/pymc_backend.py` | **NEW** | Alternate. ~350 LOC. Phase 8. |
| `model/base.py` (Laplace path) | — | `infer/laplace.py` | **PORT** | The `trust-ncg` real-curvature fix from parent commits `5f0aef6` / `f1813bc`, plus the `nonfinite_draw_frac` diagnostic that exposed the BFGS `hess_inv` bug. Ship in the first commit of this module. |
| `diagnostics/convergence.py` | 269 | `infer/diagnostics.py` | **PORT** | R-hat, ESS, divergences. |
| `utils/arviz_compat.py` | 358 | `infer/_arviz.py` | **PORT** | ArviZ 1.x DataTree drift shims. Battle-tested; do not re-derive. |
| `diagnostics/identification.py` | 981 | `diagnose/weak_id.py` | **EXTRACT** | The box-constrained-optimizer guardrail (bounded params saturating in float64 under logit/log transforms → `ZeroDivisionError` inside the compiled gradient). Concept and bounds math port; the PyTensor specifics are rewritten per backend. ~400 LOC. |

---

## diagnose/ — trust machinery (~4,000 LOC target)

| SRC | LOC | → axiom | Verdict | Notes |
|---|---|---|---|---|
| `diagnostics/sbc.py` | 695 | `diagnose/sbc.py` | **PORT** | Talts-2018. Rewire the refit loop to `infer.Backend`. |
| `diagnostics/coverage.py` | 725 | `diagnose/coverage.py` | **PORT** | Fixed-truth recovery coverage + SBC coverage. |
| `diagnostics/bias_sensitivity.py` | 1249 | `diagnose/sensitivity.py` | **PORT** | Bias-parameter engine; closed-form Gaussian mixture over existing draws, no refit. |
| `validation/confounding_sensitivity.py` | 1576 | `diagnose/sensitivity.py` | **PORT** | Cinelli–Hazlett benchmarking + decision-scale tipping points. |
| `validation/sensitivity_unobserved.py` | 269 | `diagnose/sensitivity.py` | **PORT** | Parameter-scale partial-R² version. |
| ↑ the three above | 3094 | one module | **consolidate → ~1,200** | They are three views of one engine, split across two packages in the parent. |
| `diagnostics/learning.py` | 515 | `diagnose/learning.py` | **PORT** | Prior→posterior contraction / overlap / shift. Flags prior-dominated posteriors. |
| `validation/spec_curve.py` | 711 | `diagnose/spec_curve.py` | **PORT** | Specification curve / multiverse. Fits the researcher-degrees-of-freedom ethos exactly. |
| `validation/validator.py` | 2340 | `diagnose/refute.py` | **EXTRACT** | The fit-based refutation checks only — placebo treatment, permutation, random subset, added noise. ~450 LOC. The rest is report scaffolding. |
| `validation/backtest.py` | 1434 | `diagnose/backtest.py` | **EXTRACT** | Rolling-origin harness, ~500 LOC. Keep the graph-faithful numpy saturation fix (parent issue #202) — the naive version silently forecast root-saturation treatments unsaturated. |
| `validation/posterior_predictive.py` | 496 | `diagnose/ppc.py` | **PORT** | |
| `validation/residual_diagnostics.py` | 319 | `diagnose/residuals.py` | **PORT** | |
| `validation/frozen_predictor.py` | 767 | `diagnose/backtest.py` | **EXTRACT** | ~200 LOC of the freeze-and-replay contract. |
| `validation/helpers/statistical_tests.py` | 427 | `core/stats.py` | **PORT** | |
| `validation/{results,builders,config,charts}` | 4143 | — | **DROP** | Report-shaped. `axiom` returns typed results. |
| `validation/{channel_diagnostics,geo_identification,protocols}` | 744 | `diagnose/` | **EXTRACT** | ~250 LOC. |
| `diagnostics/{provenance,snapshot}.py` | 312 | `io/provenance.py` | **PORT** | |

---

## core/ data/ build/ io/ sim/ (~4,500 LOC target)

| SRC | LOC | → axiom | Verdict | Notes |
|---|---|---|---|---|
| `utils/intervals.py` | 298 | `core/intervals.py` | **PORT** | ETI vs HDI with the definition carried on the number. Directly serves design commitment #4. |
| `utils/statistics.py` | 85 | `core/stats.py` | **PORT** | |
| `utils/standardization.py` | 217 | `data/scale.py` | **PORT** | |
| `data_preparation.py` | ~430 | `data/scale.py` | **PORT** | `ScalingParameters` and its inverse. |
| `dataset.py` | ~300 | `data/frame.py` | **PORT** | Already the general role-tagged container. The parent wrote it to escape `PanelDataset`; here it is the only container. |
| `config/dataset.py` + `config/roles.py` | 266 | `data/roles.py` | **PORT** | |
| `data_loader.py` | ~1400 | `adapters/marketing.py` | **DROP/EXTRACT** | `MFFLoader` is an MFF-format reader. ~150 LOC of it becomes the adapter's `panel_from_mff`. |
| `config/spec_diff.py` | 119 | `core/spec.py` | **PORT** | Generalized onto the `Spec` base. |
| `config/{model,transforms,priors,enums,variables,likelihood}.py` | ~1400 | `core/spec.py` + per-domain specs | **REWRITE** | ~40% survives as `Spec` subclasses. |
| `config/{events,interactions,levers,mff,reach_frequency,factories,inference_methods}` | ~700 | — | **DROP** | Fold what is needed into the domain specs. |
| `builders/base.py` | 174 | `build/base.py` | **PORT** | The fluent-builder pattern the user wants kept. |
| `builders/prior.py` | 303 | `build/prior.py` | **PORT** | |
| `builders/variable.py` | 429 | `build/variable.py` | **PORT** | |
| `builders/model.py` | 1021 | `build/{surface,study,meta,graph}.py` | **REWRITE** | `BayesianMMM`-specific. The *pattern* ports, the code does not. ~700 LOC across four builders. |
| `builders/mff.py` | 307 | — | **DROP** | |
| `serialization.py` | ~1100 | `io/serialize.py` | **REWRITE** | Keep the format-version discipline and the netCDF trace handling. Replace cloudpickle with spec-replay. ~500 LOC. |
| `lineage.py` | ~70 | `io/provenance.py` | **PORT** | |
| `garden/contract.py` | 379 | `core/protocols.py` | **REWRITE** | Its philosophy is already correct; re-express as typed Protocols + `Capability`. |
| `garden/compat.py` | 423 | `tests/contracts/` | **PORT** | An executable contract suite. Belongs in tests. |
| `garden/{base,loader}.py` | 222 | — | **DROP** | |
| `synth/dgp.py` | 2451 | `sim/dgp.py` | **EXTRACT** | ~1,100 LOC of confounded dose–response worlds with causal ground truth. Strip MMM column naming. The structural-violation worlds (the parent's "5 silent-failure modes") are the most valuable part — they are what the recovery suite tests against. |
| `synth/dgp_geo.py` | 558 | `sim/panel.py` | **PORT** | |
| `continuous_learning/dgp.py` | 550 | `sim/surface_world.py` | **PORT** | Shares `surface.forward()` with the model, by construction. |
| `synth/{mff,dgp_clv}.py` | 507 | — | **DROP** | |

---

## Dropped wholesale (~93,000 LOC)

`agents/` (30,835) · `reporting/` (37,688) · `platform/` (8,097) ·
`mmm_extensions/` (9,825) · `auth/` (2,732) · `eda/` (2,640) ·
`excel_config/` (2,263) · `data_studio/` (1,431) · `integrations/` (1,133) ·
`ltv/` (404) · `datasets/` (211) · `security/` (208) · `storage/` (195) ·
`jobs.py` · `analysis.py` · `server/` · `frontend/`

Plus `frequentist/{ridge,constrained,bootstrap,search,_transforms}` (2,368) —
the parent's own verdict on that subsystem, recorded in its notes, is that the
Bayesian path is the one to use and the ridge path's value was constraints and
speed. `axiom` gets constraints from `[convex]` in `surface/optimize.py` and
speed from Laplace.

## Accounting

| Layer | Target LOC |
|---|---|
| core, data, build, io, sim | 4,500 |
| identify | 1,600 |
| estimands | 1,400 |
| surface | 4,200 |
| design | 5,500 |
| calibrate | 2,000 |
| meta | 2,200 |
| infer | 1,600 |
| diagnose | 4,000 |
| adapters, viz | 700 |
| **total** | **~27,700** |

Roughly 16% of the parent, of which perhaps 55% is a mechanical port, 30% a
rewrite against the new seam, and 15% genuinely new (classical RSM, classical
meta-analysis, scope transfer, the ledger).
