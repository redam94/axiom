# 0045 — Three programmes: the multi-party layer, used

*Kind: progress + decision. Opened 2026-09-03. Status: landed on
`feature/three-programmes`.*

Three case studies exist and each runs **one** experiment. HYPER-3
([0004](0004-case-study-hypertension.md)) is a safety story, TUTOR-60
([0009](0009-case-study-tutoring.md)) an allocation story, GEIGER-1911
([0018](0018-case-study-rutherford.md)) a design story. Notes
[0027](0027-scope-and-the-experiment-lifecycle.md) through
[0042](0042-the-thing-that-files-it.md) then built the layer a house running
*many* experiments for *many* parties needs — scopes, the lifecycle, assignment
that re-derives, delivery checks, collisions, a stopped estimate that is
corrected rather than flagged, a party level in the pool, a book of decisions
corrected as a book, a definition registry, and the builder that files all of
it — and nothing outside `nbs/build/01` and `nbs/io/02` has ever used that
layer on a problem that needed it.

Three more case studies, one per field, each a **programme** rather than an
experiment. They share one shape: several parties, several experiments on one
estimand family, every experiment planned and committed before it is read,
deviations filed as they happen, readouts corrected as a quarter, and a pool
with a party level that says what the family of experiments knows rather than
what the average of them claims.

| | code | field | parties | what goes wrong without the layer |
|---|---|---|---|---|
| `nbs/case-studies/lightspeed/` | **LUMEN-299** | physics | four instrument families, sixteen experiments, 1849–1972 | a shared systematic error pooled as sixteen independent draws: the 1930s consensus excludes the truth |
| `nbs/case-studies/income-guarantee/` | **NIT-4** | economics | four sites, 1968–1982 | a reference that changed mid-experiment, differential attrition, a guardrail finding that does not survive the book's correction |
| `nbs/case-studies/media-measurement/` | **HOUSE-3** | media measurement | three advertisers, one house, one year | two experiments in the same markets, a dose that was not delivered, a metric whose definition drifted, eight goes a quarter that are noise |

## D45.1 — the same five stages in all three, on purpose

Each study is five notebooks and they land on the same five stages the tutorial
and TUTOR-60 use: the question and its estimand; the design; the field, with
the lifecycle filed as it goes; the book, corrected; the pool and what it hands
forward. The fields are as different as the repository could make them so
that the stages are recognisably the library's and not the domain's. A reader
who knows one of the three fields should be able to read the other two by
shape.

What is shared in machinery, and is the reason these exist:

* `io.Catalog` with one `Program` per party and an `ExperimentBuilder` per
  program; every plan goes through `plan → commit → start → read` and every
  departure through `deviate`, so `conformance()` is a checked claim in each
  readout table rather than a footnote.
* `design.collisions` over the calendar before anything runs, and
  `design.factorial` afterwards on the pair that ran concurrently.
* `diagnose.check_delivery`, `diagnose.attrition`, `identify.compliance` /
  `response_to_dose` / `lee_bounds` between the field and the readout.
* `design.confidence_sequence` where somebody watched, `design.monitor` +
  `ExperimentBuilder.read_stopped` where a rule stopped it.
* `build.readouts_across` → `design.program_decisions` (and `online_decisions`
  where the readouts arrive one at a time), with the uncorrected count printed
  beside the corrected one.
* `meta.pool(effect_key="nested")` or `meta.three_level`, `meta.commensurable`
  with estimands and with the definition registry, and `prior_from_pool` with
  the party-aware targets.

## D45.2 — what each one found

*(filled in from the executed notebooks; every number below is computed in the
notebook it is attributed to.)*

### LUMEN-299

The estimand is the first decision and it is not a contrast: a constant of
nature is the response to nothing. It is declared as a `ratio` — distance
covered per second of transit, which is what every instrument actually does —
with the medium the light crossed as the intervention's `version` and the
radiation as the `population`. Reading an in-air number as in-vacuo is then a
`downgraded` transfer under `version_irrelevance`, and the assumption is false
by 88 km/s, larger than every standard error after 1926. The
`DefinitionRegistry` per laboratory is what knows that; the type cannot.

The rotating mirror is a hand-built tree, `8π ω D / c − 4π / n`, read by the
Fisher information, by the sixteen hundred simulated readings, and by the
reduction of a reading back to a speed. `c` and `D` enter only as `D/c`, so
`identifiability_ridge` is singular along both together and the baseline
survey is a second `Observation` on `D` alone; `expected_posterior_sd` prices
its grade in km/s on `c` — a steel tape would have capped the 1926 season at
3.1 km/s whatever the readings did, triangulation at 1.0. Per-reading standard
errors: 612 km/s at Annapolis (1879), 38 at Mount Wilson (1926); the geometry
promised 58×, the dimming of the returned image delivers 16×.

The 1926 season runs through `ExperimentBuilder` with four roles (the
`ModelSpec`, a `BaselineSurvey` spec defined in the world module, a
`TimeWindow`, an `ArmAssignment` over mirror configurations), is watched night
by night with `confidence_sequence` (final width 1.55× Wald), has its
re-survey filed as a `Deviation` (`downgraded`), and reads out at
299,790.1 ± 3.3. The 1935 vacuum-tube run has its plan changed with nothing
filed and is `blocked`; `readouts_across` leaves it out.

The corpus pools `relative_change` in parts per million so that `StudyRecord`
takes it without a licensed transfer. The five records of 1928–41 pass
Cochran's test (I² = 0) and pool to 299,776.3 [299,765.5, 299,787.1] — 2.9
standard errors from the truth, which is Birge's 1941 number. Add 1926 and the
two-level pool still excludes the truth by about a kilometre a second, but
`three_level` reads the disagreement between the rotating mirrors and the
Kerr cells as a party-level component of ~32 ppm and the interval covers.
Without 1926 both families share the expectation and the three-level pool
excludes it too: **a party level sees a shared systematic only when parties
disagree**, and no hierarchy over parties can see one every party shares.
`prior_from_pool` from the 1926–41 book hands Essen's 1950 cavity resonator
±8 km/s under `target="mu"` (truth 1.3 sd out) and ±17 under `new_party`
(0.6 sd out); the mean's interval is the narrowest and the wrongest.

Each experiment judged against the consensus before it is one decision in a
stream of fifteen: `none` declares three, Holm and LORD two (1926 and 1951),
e-BH on calibrated e-values one. The two are the two history remembers.

### NIT-4

A plan is two dimensionless doses, a guarantee `G` and a tax rate `t`, and the
decision is a threshold (150 hours a year) on one contrast at a plan Congress
will enact and no site ran. The survey slope on the tax rate is 3.7× the truth
(2.8× adjusted) and `identify` returns `downgraded` on `backdoor_unmeasured`
with no admissible set; `evpi_gaussian` prices the decision at 270 M USD a
year and an experiment that pins the contrast to ±20 hours captures 97 % of
it.

The allocation problem is Conlisk and Watts's, at a fixed budget: one 9 M USD
buys 3,552 families in two arms, 2,094 spread over the lattice, or 1,444 at the
D-optimal corners, and the sd of the national contrast is 10.0, 12.5 and 14.9
hours respectively — but the two-arm design's Fisher matrix is singular and
`estimable_combinations` has rank 1 of 3, so it cannot answer the question at
the plan on the table. The four sites as one design give a contrast sd of 7
hours and an MDE of 18.

Six runs are filed across four scopes. New Jersey's control counterfactual
moves when the state adopts AFDC-UP in 1969 — filed as `deviate("reference",
state="violated")` and read against a re-declared estimand, `blocked`. Gary's
definition of income moves from v1 to v2 through the registry and is filed as
a deviation, `downgraded`. Seattle–Denver's twenty-year cohort is `abandon`ed.
Delivery is `identified` everywhere; attrition is `blocked` at three of four
sites (controls observed at 64–69 % against 77–83 % under a plan) and the Lee
bounds that survive it all exclude zero. Gary's `compliance` reports perfect
compliance and `response_to_dose` reports a first-stage shift of 0.36 — 72 % of
the assigned tax rate, because 28 % of earnings went unreported — which is the
difference between a switch and a dose. The training × plan factorial at
Seattle–Denver is `unverified` (interaction −5 [−44, 34]; could have seen ±40).
`readouts_across` returns four licensed runs of six.

Thirty-three `relative_change` records, I² = 93 %: the two-level pool says
−10.1 % [−11.8, −8.4]; `three_level` says −8.4 % [−12.6, −4.2] with a party
spread of 4.1 %, and the nested Bayesian pool with `tau_party` fixed agrees
and carries `G` and `t` as moderators (−0.049, −0.130). `commensurable` is
`downgraded` across the intervention and window facets, `blocked` on a
complier-population record, and with `definitions=` blocked on Gary's four
records through a definition the estimands never saw. The guardrail: the
world plants no dissolution effect and the draw was seeded to land at
`p ≈ 0.02` at Seattle–Denver, which the notebook says; uncorrected and under
Benjamini–Hochberg it is a go, under Holm and e-BH it is not, and the four
hours contrasts are. `prior_from_pool` for a fifth site: 4.9 % sd for a new
party against 1.9 % for another Seattle–Denver cohort.

The national answer: a linear surface on the three licensed sites, `realize`
at (0.75, 0.5) gives −126 hours [−141, −111] at 90 % HDI against a truth of
−136 — inside the threshold; a 70 % tax rate fails it at every guarantee;
the marginal is −1.86 hours per point of tax rate. The transfer to the
national population is `downgraded` on `s_admissibility` and the
recommendation flips only at 2.7 USD an hour, 1.5× the memo's price.

### HOUSE-3

Every party's KPI column is the same `Outcome`, `conversions`, and what each
party *means* by it is a versioned entry in the `DefinitionRegistry`: a paid
order, a trial start until week twelve and a paid subscription after, a case
sold. `consensus` moves from `unverified` (globex silent) to `blocked` (acme
means a trial start) to `identified` after acme's week-twelve change, and the
registry is the only reason one party's search lift can be compared with
another's at all.

The model is not the lift. Each party's media-mix model overstates every one
of seven channels by 1.17× to 1.97× because demand drives both spend and
conversions and demand is in no column; `identify` says `downgraded` on
`backdoor_unmeasured` before the fit and the world confirms it. `roas` on a
count KPI is a typed `Unsupported`; on northwind's revenue it is 1.01 with a
marginal of 1.05.

The calendar has three kinds of overlap. Acme's spring social and search
tests are `concurrent` — unbiased in expectation, a 1.29× standard error, and
an interaction assumption `factorial` later finds `unverified`. Northwind's
two spring search tests share the lever and `recommend` will not schedule the
second beside the first at any net value. Acme's autumn search test starts
while an unrandomized rollout lands on its largest markets and is moved.
Weeks within a market have an ICC of about 0.3 and a design effect of 3.3;
NW-01's holdout trade-off flattens past 25 %, where its net value is 217 kUSD
(EVSI 512, opportunity cost 254, invoice 40).

Nine runs, four answers. NW-01, watched every Monday, clears zero on a
fixed-sample interval on all eight Mondays and never on the confidence
sequence (final drift 2.01 [−1.0, 5.0]). GX-02 stops at the third of four
O'Brien–Fleming looks and `read_stopped` files 34.3 against a naive 34.8,
`downgraded`. NW-02's platform delivered 53–92 % of a doubled budget:
`check_delivery` is `blocked` and `response_to_dose` reports a derivative
of 5.5 per kUSD. GX-03 reached seven markets in ten: ITT 23.0, complier
effect 32.8. AC-04's control markets went dark — the ones whose demand fell —
and `attrition` is `blocked`, the naive 8.5 ± 1.8 against a truth of 4.3, Lee
bounds [3.7, 13.2]. NW-03 was read on an assignment re-drawn after commit
with nothing filed: `blocked`, and `readouts_across` never hands it over.

Twenty-four readouts, eight primary and sixteen guardrails; a null book
expects 1.2 false goes. `none` takes six, one of them acme's churn guardrail,
which no channel touches — the false go arriving on schedule. Holm takes two,
Benjamini–Hochberg six, e-BH one. **The e-BH survivor is AC-04**, the
attrition-biased readout: error control corrects the count of false goes on a
null book and cannot see that a number is biased, because its inputs are
p-values and a biased estimate has a very good p-value. That fact is on the
run, `downgraded` with an unverified population assumption, and the
multiplicity cannot see it. LORD in arrival order rejects three, including
the churn guardrail, which arrived directly after GX-01 replenished the
budget.

The book: four search records from three parties plus a complier record.
`commensurable` blocks the complier record, downgrades AC-04's population and
the window differences, and with the registry passes today and blocks acme's
records as of March. The `study` pool's interval is 0.056 wide and the
`nested` one 0.067; `prior_from_pool` hands a fourth party a `new_party` sd of
0.044 against 0.020 for the mean, transferred into its scope with a ledger
line. Calibrating northwind's model by NW-01: the model realizes 20.3 where
the experiment says 8.7 ± 4.4 (`disagrees`, z = −2.5); the prior route barely
moves the contribution (1.38× → 1.36× truth), the likelihood route moves it to
1.27× and it still `disagrees`. The house can now say to each party what its
model claimed, what its experiment measured, and which of the two the year's
correction lets it repeat.

## D45.3 — where the layer pushed back

*(the API friction each study met, and what was worked around rather than
changed — the input to the next backlog.)*

From LUMEN-299:

* **`design.anytime.evalue` overflows** (`src/axiom/design/anytime.py`, the
  `math.exp` in `evalue`) once the monitoring statistic is more than about
  `37·sqrt(1 + rho)` standard errors from its null. A confidence sequence run
  against Newcomb's 1882 value (z ≈ −50) cannot be computed; notebook 3
  monitors against the 1924 preliminary value instead. The fix is a
  log-domain e-value; `program_decisions` and `online_decisions` would then
  need to accept it. This is the one item here that is a defect rather than a
  gap.
* **No `QuantityKind` for a level.** A constant is measured, not contrasted,
  and the nearest honest declaration is a `ratio` with the medium as the
  intervention's `version`. `Level` likewise has no word for it; `aggregate`
  is used and the notebook says why. Whether a `level` kind belongs in
  `estimands` is a charter question — the charter's estimands are causal — and
  is left open.
* **`StudyRecord` pools dimensionless quantities only**, so the corpus carries
  parts per million from the defined value rather than km/s. Correct by the
  ingest rule ([0029](0029-the-party-a-study-came-from.md)), and it means a
  corpus of a dimensioned constant has to pick a reference before it can pool.
* **`DefinitionRegistry.register` versions by registration order, not by
  `at`.** Registering 1926 then 1879 gives 1926 version 1. Not a bug — the
  registry records what a party said when it said it — but a history entered
  after the fact has to be entered in order, and the docstring should say so.
* **`ThreeLevel.shares` takes one sampling variance** and a heteroscedastic
  corpus (Fizeau's 1849 error dominates) makes the mean se² meaningless; the
  median is used.
* A custom `Spec` subclass defined in the world module (`BaselineSurvey`)
  files and reloads through the catalog because `load_spec` resolves the type
  by import path — worth knowing, and worth a test if it is relied on.

From NIT-4:

* **A `violated` deviation still licenses the readout.** `Deviation.state`
  accepts `"violated"`, `conformance()` returns `downgraded` whatever the
  state, `Verdict.licensed` is `status in (identified, downgraded)`, and
  `readouts_across` therefore includes a run whose operator has said the
  readout is *not* the planned quantity. `identify` treats a violated
  assumption the same way, so this is consistent — and it is the wrong
  default for a lifecycle, where "violated" is the one word an operator
  files to keep a number out of the book. Worth a decision: either
  `conformance()` returns `blocked` when any deviation is `violated`, or
  `readouts()` filters on assumption state. Notebook 3 works around it by
  reading New Jersey against a re-declared estimand.
* **`optimal_exchange` is quadratic in `n`**; a bisection over `n` to match a
  budget did not finish, so notebook 2 runs it at 60 rows and scales the
  shares, which is exact for a linear surface and had to be argued in prose.
* **`estimable_combinations`** defaults to `scaling="log"`, which is
  `Unsupported` for a negative parameter; `scaling="absolute"` with explicit
  `parameters` works but names no `estimable` combination for a non-integer
  one — only `null_basis` says what is estimable. `expected_posterior_sd`
  does not report the sd of a linear combination, so the contrast sd is
  assembled from `fisher_information` and `linearize` in the notebook.
* **`surface.fit` wants a balanced `(unit, time)` panel**; a cross-section of
  families has to be laid out in blocks to fit at all, and a hierarchical
  intercept over four sites fails Laplace's positive-definite check
  (`per_unit` and `shared` fit). A quadratic-in-`t` surface returns
  `Unverified` on the same check; the notebook says so and fits the linear one.
* `realize` downgrades under `units_assumed_equal` whenever an estimand sets
  two treatments, because it names a unit for one only — unavoidable for a
  two-dose estimand and worth a sentence in its docstring.
* `TransportVerdict` exposes `s_admissible_set` where `IdentificationVerdict`
  exposes `adjustment_set`; a reader moving between the two has to know.
* `jupyter nbconvert` is not in the dev group; `nbclient` under `uv run` is
  how a notebook is executed for its outputs outside pytest.


From HOUSE-3:

* **`laplace` refuses the media-mix surface with a hierarchical intercept
  over forty markets** (Hessian not positive-definite) and, with `per_unit`
  intercepts, refuses `GeometricCarryover(max_lag=4)` on two parties of three
  — a near-flat direction with the smallest eigenvalue four orders below the
  spectrum. `max_lag=2` certifies all three. The world's analyst spec records
  the choice; a reader should know that carryover length is a fit decision
  under Laplace and not only a modelling one.
* **`adapters.marketing_spec` exposes no `intercept_scale`, `noise_scale` or
  Fourier coefficient scale**, so the prior-scale trap of
  [0009](0009-case-study-tutoring.md) §0009.3 has to be escaped by
  `model_copy` on the built spec. Worth exposing on the preset.
* **`calibrate.fit_calibrated` refuses a `set`-mode intervention on a
  carryover surface** ("a constant there collapses the time axis"); a holdout
  estimand has to use `mode="scale"` for both arms with explicit `doses=`,
  and `derive_prior` needs an estimand that increases in the amplitude (an
  off-versus-on contrast raises "design factor is not positive"). Documented,
  and expensive to discover.
* **`commensurable` blocks records whose treatments or `Outcome` entities
  differ**, which forced every party onto one `conversions` outcome and made
  a family pool across channels inexpressible. Probably the right lesson;
  the docs should state it.
* **`design.cluster_mde` on a DiD design understates the realized se by about
  35 %** (3.17 against 4.35) because AR(1) demand persists beyond the block
  ICC, and `cluster_power` on raw KPI levels gives an ICC of 0.98 because
  market size dominates. A DiD design wants residual-scale inputs and nothing
  in `design` computes them; the case study two-way demeans by hand.
* **`diagnose.attrition` at the default `SRM_ALPHA = 0.001` cannot flag
  differential attrition on forty units**; the notebook uses 0.01 and says
  why. The default is right for a check that runs weekly on a hundred
  thousand users and wrong for a geo test, and nothing chooses between them.
* `ExperimentRun.deviate` with a role not in `roles` adds it, so a
  `"stopped_early"` or `"dose"` deviation shows up as a role in `summary()`.
  `meta.three_level` on four records sits on the boundary (both taus zero),
  so the nested Bayesian pool's fixed `tau_party` is an argument the data
  cannot check. `ExperimentBuilder.readouts()` returns `p_value = 1.0`
  placeholders and attaching evidence is a `model_copy` per readout; a helper
  taking a mapping would help.

## D45.4 — what this does not do

* The worlds are synthetic and calibrated to the historical record; the
  historical numbers appear beside the simulated readouts as anchors and are
  never fitted. A benchmark against the real corpora (Birge's 1941 table; the
  Keeley–Robins–Burtless pooled NIT estimates) would be a `benchmarks/` entry
  and is not here.
* Marketing vocabulary appears in HOUSE-3 and nowhere else, and only through
  `axiom.adapters.marketing`. Gate 3 scans `src/`; the case study holds to the
  rule by discipline, and its README says so.
* The site does not carry these three. `site/` has pages for HYPER-3 and
  GEIGER-1911 generated by running the library; adding three more is a
  `site/_gen` job and a separate decision.
