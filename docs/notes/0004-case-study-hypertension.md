# 0004 — HYPER-3: a sequential-trial case study, and the module it exposed

**Kind**: progress + decision. **Opened**: 2026-08-21. **Status**: landed on
`develop`; the new public API is destined for 1.1.0.

## Why

Nothing in `axiom` was wrong, but nothing in `axiom` showed a reader what the
pieces are *for* when they are all pointed at one decision. The notebook series
under `nbs/<subpackage>/` demonstrate every public symbol (gate 12) and the
`nbs/end-to-end/` series walks each pillar, but neither answers "what does using
this look like on a real problem, start to finish, with the awkward parts left
in".

`nbs/case-studies/hypertension/` is that. Six notebooks, one synthetic world,
one question: a phase-II dose-finding trial with weekly monitoring that must
stop an arm early if it is harming people. It reaches into `core`, `data`,
`sim`, `identify`, `estimands`, `surface`, `design` (power, cluster, precision,
structural, eig, evoi, **sequential**), `meta` and `viz`.

## The trial

400 adults, 2:1:1:1 to standard of care / 10 / 20 / 40 mg, stratified by age
band (25–35, 36–50, 51+), weekly seated systolic pressure for 24 weeks,
enrollment staggered over 16 calendar weeks. Twelve harm contrasts — each dose
against control overall and within each age band — reviewed six times.

The generating process has an Emax benefit that rises with dose and a **pressor
effect cubic in the dose actually taken**, concentrated where clearance is
slowest. At 40 mg the two nearly cancel over the trial population: the pooled
arm reads as a mild benefit at every review while the oldest band is being
harmed by 5.8 mmHg. That is the whole case: a monitoring plan that watches only
arms cannot see it, and one that pre-specifies strata stops it at the first
review, on three eighths of the information, with a third of the arm's units
never randomized to it.

## 0004.1 — Sequential monitoring is design-layer math, not notebook scaffolding

The case study needed a stopping boundary, its exact error rate, and a way to
walk a realized path against it. All three are used in more than one notebook,
and a boundary recursion is exactly the kind of thing rule 3 says should not be
reimplemented per caller. So `src/axiom/design/sequential.py` was written rather
than a helper cell:

- `LookSchedule` / `Boundary` / `StoppingRule` — Specs, composed, not a
  hierarchy. A rule is a schedule plus a tuple of boundaries; an efficacy
  boundary, a harm boundary and a futility boundary are three values in that
  tuple, and the rule is what makes them one continuation region per look.
- `pocock`, `obrien_fleming` — one constant solved so the total null crossing
  probability is exactly `alpha`. Reproduce the published tables (Jennison &
  Turnbull 2.1/2.3) to four decimals for K = 1…5.
- `alpha_spending` — Lan–DeMets, each threshold solved in turn against the
  budget already spent, so a look that moves changes only the thresholds after
  it. Reproduces the standard LD-OBF five-look table to three decimals.
- `harm_boundary` — the protocol sentence *"stop when P(worse than control by
  more than `margin`) ≥ p"* turned into Z thresholds. Under a flat prior this is
  `Z_k ≤ −margin/se_k − z_p`, so it is a boundary like any other and its error
  rate is computable rather than assumed.
- `crossing_probabilities` / `operating_characteristics` — numerical integration
  of the canonical joint distribution (Armitage–McPherson–Rowe), not simulation.
- `monitor` → `MonitoringPath`, which emits a `LedgerLine` carrying the
  `stopped_estimate_bias` assumption.

**Numerics.** Two choices make the recursion second-order rather than first:
the grid at each look is laid out *on* that look's continuation region, so the
carried sub-density is smooth across it; and every crossing probability is the
exact normal tail given each grid point, never a sum of grid cells beyond the
threshold. Quadrupling the grid quarters the mass defect, which
`test_design_sequential.py` pins. The default 601-point grid conserves mass to
about 3 × 10⁻⁶ and the recursion agrees with a 40 000-draw Monte Carlo of the
same rule to within Monte Carlo error at three drifts.

**Sign convention.** `Z = effect / se` signed so **positive is better**;
`Boundary.z` holds *signed* thresholds and `side` says what crossing means.
Storing magnitudes would have been tidier for two-sided boundaries and
impossible for a futility boundary whose threshold is positive.

**`binding` governs the arithmetic, not the committee.** `monitor` acts on every
boundary; `crossing_probabilities(..., binding_only=True)` ignores non-binding
ones, which is the convention that makes a non-binding futility boundary
conservative. `binding_only=False` gives the rule as run, so the gap is a number
rather than a footnote.

## 0004.2 — A posterior-probability rule states a posterior, not an error rate

The protocol sentence a monitoring committee writes is Bayesian and the
regulator's question is frequentist. `harm_boundary` implements the first and
`crossing_probabilities` prices it in the second, which is the only honest way
to have both. In HYPER-3 the 2 mmHg / 95 % rule spends between 0.1 % and 1.5 %
per contrast depending on that contrast's standard error — a wider standard
error pays more, because the fixed margin is a larger share of it.

Across twelve monitored contrasts the family-wise false-stop rate is about
0.085, computed by simulating the correlated canonical distribution directly
(three doses in a stratum share a control arm, correlating their Z at 1/3) and
checked against `core.clopper_pearson` for the marginal rate. **No multiplicity
correction is applied**, deliberately: a false harm stop costs one arm of a
phase II and a missed harm costs units their pressure control for months. The
asymmetry and the number both go in the ledger.

## 0004.3 — What the case study found that the library could not represent

Every response kernel `axiom.surface` ships is monotone and saturating. The
oldest stratum's true dose–response **turns over**. The Hill fit therefore
fails, and it fails informatively: the arm-mean residual at 40 mg is −8 mmHg and
systematic in dose, against ≤1.5 mmHg in the two strata where the family fits.

It was left as a finding at first, on the argument that a flexible curve would
interpolate anything and the honest conclusion was that the 40 mg point is
outside the model family. That argument did not survive contact with the obvious
counter: a dose that helps and then harms is the ordinary reason a phase-II trial
exists, so a dose-response library that cannot express it is missing a family
rather than an extra. Three **basis** families — `PolynomialKernel`,
`SplineKernel` (natural cubic) and `PiecewiseLinearKernel` — landed alongside
this case study; see note [0005](0005-basis-response-families.md) for the design
and §3b of notebook 4 for the refit, which cuts the worst arm-mean residual in
the oldest band from 7.95 mmHg to 3.1.

## 0004.4 — Three claims the first draft made that the numbers refused

Recorded because they are the reason the notebooks are worth executing:

1. *"The trial is badly underpowered inside its strata."* It is not. At closeout
   a stratum-level contrast detects 3.8–4.8 mmHg against a pooled 2.4, and the
   harm is 5.8. Nor, on the corrected schedule, is a single interim test hopeless:
   at the first review it has 69 % power. The statement that survives, and the one
   that motivates the sequential design, is about the *family*: one look is a coin
   flip, six looks tested at 5 % apiece spend 0.065 rather than 0.05 in one
   contrast and would trip something in about half of null trials across twelve, and
   the boundary buys that back for a tenth of the sensitivity. Notebook 5 prints
   both rules side by side.
2. *"With four dose levels, amplitude and half-maximal dose are badly
   confounded."* The condition number is 60, not thousands, and the badly
   determined pair is amplitude against *shape* (correlation −0.75). The shape
   parameter's expected posterior sd is barely below its prior under **any**
   design at 60 units: dose placement is not the binding constraint, sample size
   is, and `expected_posterior_sd` says so.
3. *"A completers analysis is biased toward zero."* It is biased, but not in a
   common direction — on this trial it pulls the 10 mg estimate toward the truth
   and pushes the 20 and 40 mg estimates away. There is no sign to correct for,
   which is exactly why `identify` downgrades the verdict rather than offering a
   correction.

## 0004.5 — Notebook scaffolding lives beside the notebooks, not in `src`

`nbs/case-studies/hypertension/hyper3.py` holds the data-generating process,
the enrollment plan, the look-frame helpers and a small plotting vocabulary. It
is imported by the notebooks (nbmake runs each notebook with its own directory
as the working directory, so a plain `import hyper3` resolves) and by nothing in
`axiom`. Keeping it out of `src` keeps it out of the dependency budget, the
layering gate and gate 12, none of which it should be subject to.

The endpoints are **windowed means** — weeks 5–8 for safety, 9–12 for the
primary — rather than single visits. That is what makes the "monitored weekly"
detail load-bearing: averaging four autocorrelated readings roughly halves the
variance of a unit's endpoint, worth about 80 units of enrollment, and it is the
difference between a stratum-level safety boundary that can fire and one that
cannot. It also gives the sequential design the thing it needs and a
latest-visit endpoint cannot supply: an information scale that is **monotone by
construction**, because a unit's window is either complete or it is not.

## What landed

| | |
|---|---|
| `src/axiom/design/sequential.py` | 22 public symbols, exported from `axiom.design` |
| `tests/unit/test_design_sequential.py` | 45 tests: published tables, Monte-Carlo agreement, convergence order, the posterior-rule round trip, validation |
| `tests/contracts/_factories.py` | gate-4 entries for `LookSchedule`, `Boundary`, `StoppingRule`, `CrossingProbabilities`, `OperatingCharacteristics` |
| `nbs/design/07-sequential.ipynb` | gate-12 coverage for the new symbols |
| `nbs/case-studies/hypertension/` | `README.md`, `hyper3.py`, six executed notebooks, 37 figures |
| `docs/api/design.md` | `axiom.design.sequential` added to the reference |

## Left open

- **Stage-wise-ordered estimates.** The ledger line names the bias of an
  estimate reported at the stopping look and the notebook quantifies it by
  simulation, but nothing computes a median-unbiased or stage-wise-ordered point
  estimate and interval. That is the obvious next function in the module.
- **Information-based monitoring with a nuisance variance.** The information
  fraction here is the share of completed endpoints, which is right for a
  balanced comparison of means with a common variance. A trial whose variance is
  itself being estimated wants `(se_planned_final / se_k)²`, and the notebooks
  show the two agreeing to three decimals rather than the module offering both.
- **Group-sequential *estimation* of the surface**, as opposed to of a contrast.
  The case study monitors twelve scalar contrasts; monitoring a fitted surface
  would need the canonical joint distribution of a vector, which this module
  does not do.
- **A non-monotone response kernel**, per 0004.3, if a second use case asks.
