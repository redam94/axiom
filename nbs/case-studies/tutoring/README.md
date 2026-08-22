# TUTOR-60 — one decision, carried from "what are we asking" to "here is what to fund"

A worked case study that runs the whole of `axiom` against a single decision, end to end:
**problem setup → experiment planning → measurement → reporting → recommendation.** Every
number in the five notebooks is computed; the synthetic world they share lives in
[`tutoring.py`](tutoring.py) beside them.

## The decision

> *A state has nine million dollars for one school year and twelve thousand students
> reading two or more grade levels behind. It can buy them structured small-group tutoring,
> sold by the weekly minute, and caregiver messaging, sold by the weekly message. How much
> of each — and therefore how many of the twelve thousand can it reach?*

A pilot district ran 90 weekly minutes and liked it. The state cannot afford 90 minutes for
everyone and needs to know whether it should try.

Both treatments are dosed in **dollars per eligible student per year**, because the
decision variable is the per-student allocation and putting both on one scale is what makes
the budget an actual constraint. Twenty dollars buys one weekly tutoring minute; nine
dollars buys one weekly caregiver message, capped at five.

## The answer

**Thirty-five weekly minutes and five messages a week, for the whole cohort** — about 745
dollars a student. Not 90 minutes for the 4,900 students the same money reaches. The
difference is 13,800 cohort reading points, 19 million dollars at the department's own
valuation, with posterior probability 0.997.

The reason is not that 90 minutes fails. It is that the response curve is steep at the
bottom, the budget is fixed, and every extra dollar per student is a student not served.

## The notebooks

| | | Reaches into |
|---|---|---|
| [1 — The decision](01-the-decision.ipynb) | the objective as arithmetic; what 240 schools of observational data cannot answer; the estimand, declared and hashed; which uncertainty is the expensive one | `design.DecisionSpec`, `identify`, `estimands`, `design.evpi_gaussian` |
| [2 — Planning the experiment](02-planning-the-experiment.ipynb) | schools or students; which outcome column; which allocations and how many distinct ones; how many schools, priced | `design.ClusterDesign`, `surface.optimal_exchange`, `surface.design_matrix`, `design.evoi_gaussian` |
| [3 — Measurement](03-measurement.ipynb) | balance; the PyMC fit and its sampler; the posterior predictive check that changes the answer; the curve with its band; what the response family is deciding | `infer.PymcBackend`, `diagnose`, `surface.response_band`, `estimands.realize` |
| [4 — Reporting](04-the-report.ipynb) | a template with no data in it; `missing` as a checklist; HTML, slides and PDF from one description; the same template against a different fit | `report.ReportBuilder`, `report.resolve`, `report.write`, `report.Theme` |
| [5 — The recommendation](05-the-recommendation.ipynb) | the objective through the posterior; buy the cheap thing first; the paired comparison; what would change it; the memo and a saved analysis | `surface.allocate`, `surface.frontier`, `diagnose.tipping_point`, `io.save_analysis` |

## Reading them

They run top to bottom with no arguments and no network:

```bash
uv run pytest --nbmake nbs/case-studies/tutoring/ -q
```

Notebooks 3–5 each rebuild the same trial from `tutoring.run(tutoring.chosen_design())`, so
any one of them can be read on its own.

## The six conclusions

1. **The expensive uncertainty was not the one anybody was arguing about.** Resolving
   "should we run a tutoring program at all" is worth 131,000 dollars, because the prior
   already sits two standard deviations clear of the threshold. Resolving *how much* is
   worth 2.3 million a year. A trial designed for the first question would have been a
   waste of money.
2. **One column of arithmetic was worth 336 schools.** Differencing each school's gain
   against its own prior year removes the persistent between-school variance and takes the
   trial from 485 schools — more than exist — to 149. Nothing about the design, the sample
   or the budget changed.
3. **A design optimal for the model you hope is right cannot tell you whether it is.** The
   five-arm factorial is the most precise design in the table for the pre-registered spline
   and is *singular* for a spline with one more bend. `optimal_exchange` against the
   flexible model gives up a tenth of the precision and keeps the check — and notebook 3 is
   where that check earns its keep.
4. **Six benchmark readings from one school are one observation.** The posterior predictive
   check caught it on `unit_sd`, Ljung–Box confirmed it, and averaging first widened every
   interval by 70 % without moving a single point estimate. Notebook 2's own power
   calculation had said so in advance.
5. **The response family is a prior commitment, and the data cannot arbitrate it.** Three
   families fit this trial equally well on every posterior predictive statistic; the
   monotone one puts the peak of the curve at the most expensive dose the state can buy,
   because a Hill curve cannot have a peak anywhere else. At this budget it recommends the
   same allocation and a sixth more value; at three times the budget it recommends
   something different.
6. **The budget line does more work than the estimate.** The recommendation is flat within
   2 % between 700 and 750 dollars a student, it does not depend on what a reading point is
   worth, and it *does* depend on how many students are eligible — 72 weekly minutes if
   6,000 are, 16 if 24,000 are. That is the number to re-check before the money moves.

## What this case study changed in axiom

Two round-trip bugs in `axiom.io`, both found by trying to save the recommendation, and
neither of which changed a number:

- A `RoleMap` is a `Spec`, so its JSON sorts its mappings and the order in which treatments
  were *declared* does not survive a reload. `Panel.to_csv` was ordering its canonical text
  by that order, so a stored panel failed its own hash check on reload. It now sorts.
- `%.17g` writes a float column whose values are all integral — a dose in whole dollars — as
  `1200`, and `load_analysis` inferred it back as `int64`. The dtype was in the manifest
  all along; it is now used.

Both are pinned by
`tests/contracts/test_analysis_roundtrip.py::test_a_stored_panel_survives_declaration_order_and_whole_valued_floats`.
See [`docs/notes/0009`](../../../docs/notes/0009-case-study-tutoring.md).
