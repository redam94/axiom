# 0009 — TUTOR-60: one decision end to end, and the two io bugs it found

**Kind**: progress + deviation. **Opened**: 2026-08-22. **Status**: landed on `develop`.

A second case study, `nbs/case-studies/tutoring/`, carrying a single decision through the
five stages the toolkit is organized around — problem setup, experiment planning,
measurement, reporting, recommendation — in five notebooks over one shared synthetic world.

It is deliberately a different shape from [0004](0004-case-study-hypertension.md). HYPER-3
is a safety story: a sequential design, a harm boundary, a stratum-level effect the pooled
analysis hides. TUTOR-60 is an **allocation** story: two dosed treatments on one budget, an
objective that is a cohort total rather than an effect size, and a recommendation that has
to survive a board asking what would change it.

## 0009.1 — Why both treatments are dosed in dollars

`tutoring` and `messaging` are declared with `dimension=D.currency` and unit
`USD/student-year`, not in weekly minutes and weekly messages. Two reasons, and the second
is the one that made the case study work:

1. The state's decision variable *is* the per-student allocation.
2. `surface.allocate` and `surface.frontier` take a budget in dose units summed across
   treatments. With minutes and messages on their own scales the budget constraint cannot
   be written down, and the notebook would have had to reimplement the optimizer with its
   own cost arithmetic — which is the drift rule 3 exists to forbid. On one scale, the
   existing functions answer the question that was asked.

The operational reading (minutes, messages) is one multiplication and appears on every
axis.

## 0009.2 — A per-school intercept is collinear with a school-level dose

Every school holds one allocation for the whole year. With `intercept="hierarchical"` the
unit intercepts and the response are then nearly unidentified: the first fits produced a
response spanning 15 points against data spanning 4, with intercepts compensating, and the
Laplace mode search failed to converge in four different optimizers.

`intercept="shared"` is correct here and is what `tutoring.spec` sets. The between-school
variance goes into the residual — which is exactly the price notebook 2 charges cluster
assignment, so the modelling constraint and the design cost are the same fact seen twice.

## 0009.3 — The prior-scale trap, again

The first fits were also wrong for a duller reason: `SurfaceSpec.intercept_scale` and
`noise_scale` default to 1.0, and the outcome had mean 24. A prior 13 standard deviations
from the data does not fail loudly; it produces a fit that reproduces nothing and
compensates with an inflated amplitude. Worth stating because nothing in the API says the
outcome should be O(1), and the failure looks like a modelling problem rather than a prior
problem.

## 0009.4 — Two `axiom.io` bugs, neither of which changed a number

Both surfaced from `save_analysis` → `load_analysis` on the case study's panel, and both
made a stored panel fail its own hash check on reload.

**Declaration order.** `RoleMap` is a `Spec`, so its JSON sorts its mappings: two role maps
differing only in the order their treatments were declared are equal and hash identically,
and the declaration order does not come back. `Panel.to_csv` was emitting columns in
`RoleMap.columns` order — which is insertion order — so the canonical text, and therefore
`Panel.content_hash()`, depended on something that is explicitly not part of identity. The
case study declares `tutoring` before `messaging`; alphabetical order would have hidden it.
`to_csv` now orders unit, time, outcome, then treatments and covariates **sorted by name**.
`RoleMap.columns` is untouched, so no user's DataFrame changes shape.

**Whole-valued floats.** `%.17g` writes `1200.0` as `1200`, and `load_analysis` let pandas
infer the column type, getting `int64` back for a `float64` column. The dtype was recorded
in the manifest the whole time; `_pandas_dtypes` pinned only the string columns. It now
pins every recorded dtype. Doses in whole dollars are what exposed it.

Both are pinned by a new contract test,
`test_a_stored_panel_survives_declaration_order_and_whole_valued_floats`, which builds the
failing case directly rather than relying on the case study.

## 0009.5 — Findings the notebooks establish

These are conclusions about *method*, computed in the notebooks; the case study README
states them for a reader.

- **Value of information picks the question, not just the size.** Resolving go/no-go is
  worth 131,000 USD against this prior, because the prior sits 2.2 sds clear of the
  threshold. Resolving the dose is worth 2.3 million a year. Sizing a trial before deciding
  which question it answers gets the wrong trial.
- **The outcome column was worth 336 schools.** Differencing against each school's own
  prior year takes the trial from 485 schools to 149 — the difference between infeasible
  and feasible — at no cost.
- **D-optimality answers "best for this model".** The five-arm factorial beats every
  alternative for the pre-registered spline and is singular for a spline with one more
  bend. `optimal_exchange` against a deliberately more flexible model gives up a tenth of
  the precision to keep the check.
- **Repeated measures on a unit that holds one dose are one observation.**
  `diagnose.posterior_predictive` flagged `unit_sd`; `residuals` flagged Ljung–Box;
  collapsing to school means widened every interval by 70 % and moved no point estimate.
  The planning notebook's own MDE, computed on school means, had said so.
- **The family choice is a bet on a region you are not spending in.** At the state's actual
  budget the monotone and spline fits recommend the same allocation and differ by a sixth
  on the value; they part company only at three times the money.
- **`realize` refused the estimand the state declared.** The population was stratified on
  grade band and whole schools were randomized, so no unit has a stratum and the weights
  have nothing to attach to. The report says so, in a `LedgerLine` of kind `limitation`.

## Consequences

- `nbs/case-studies/tutoring/` — five notebooks, a 700-line shared world, a README.
- `src/axiom/data/frame.py`, `src/axiom/io/serialize.py` — the two fixes above.
- `tests/contracts/test_analysis_roundtrip.py` — the regression test.
- No new public symbols, so gates 4 and 12 are unaffected.
