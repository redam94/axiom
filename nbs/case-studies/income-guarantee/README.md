# NIT-4 — four income-maintenance experiments, one labour-supply surface

A worked case study that uses `axiom` to **run a programme of experiments rather than
one**: four parties, fourteen years, one estimand family, and a national decision that
none of the four was run to answer on its own. Every number in the five notebooks is
computed; the world they share lives in [`guarantee.py`](guarantee.py) beside them.

## The question

> *Congress is considering a negative income tax: a guarantee of three quarters of the
> poverty line for a family that earns nothing, withdrawn at fifty cents on every dollar
> earned. The Treasury says it is affordable if a family head under it works fewer than
> 150 hours a year less than he otherwise would. Nobody knows whether he does. Four
> contractors run four randomized experiments in four sets of cities between 1968 and
> 1982 — what does the programme, taken together, say?*

A **plan** is two numbers, a guarantee $G$ and a tax rate $t$, and the dose is
two-dimensional by declaration: both are dimensionless, the transfer schedule
$\max(GP - ty, 0)$ is arithmetic, and the whole case study is written over $(G, t)$.
The decision is a threshold on one contrast — hours under the plan against hours under
no plan — and the experiments have to measure a **surface**, because the plan Congress
will enact is not the plan on the table.

## The answer

**About 125 hours a year per family head**, with a 90 % interval inside the Treasury's
threshold and the synthetic world's truth inside the interval — from a surface fitted
to the three experiments whose readouts answer the estimand their plans committed to.
A 70 % tax rate does not clear the threshold at any guarantee. Each further point of
tax rate costs nearly two hours.

**And the number is not the point.** The point is the table in notebook 3 that says,
for each of six runs, whether its readout is the analysis its committed plan described
— and the three refusals in notebook 4 that follow from it.

## The finding

Read as four reports, the programme has four numbers and a memo asking for one. Read
as a programme, it has:

| | what the library said | where |
|---|---|---|
| New Jersey's control group gained a welfare alternative a year in | a deviation on the `reference` role, `violated`; the readout answers a different estimand and the run's conformance is **`blocked`** | notebook 3 |
| Gary changed what it meant by "hours worked" in 1973 | a versioned definition with a field-level diff; a `downgraded` run; and four records the pool **may not average** with the reference, on the outcome facet, through a definition the estimands never carried | notebooks 3, 4 |
| Gary's families were taxed on two thirds of what they earned | as a switch, perfect compliance; as a dose, a derivative over a moved population — a different estimand | notebook 3 |
| Seattle-Denver's twenty-year cohort was cancelled | `abandoned`, with the reason, never read | notebook 3 |
| Seattle-Denver reported a rise in marital dissolution | a go at 5 %; not a go under Holm or e-BH over the programme's eight decisions; the synthetic world's truth is no effect | notebook 4 |
| control families dropped out more, everywhere | attrition `blocked` at three of four sites; the number that survives is a Lee bound, and every bound excludes zero | notebook 3 |
| thirty-three plan arms from four contractors | a two-level pool three points wide, a three-level pool eight points wide; the between-party spread is the world's site scale | notebook 4 |

The programme's answer is the surface. The programme's *evidence* is the record of
what each experiment departed from, and that record is what makes the surface
something other than an average of four reports.

## The notebooks

| | | reaches into |
|---|---|---|
| [1](01-the-question.ipynb) | the plan as arithmetic, the decision as an object, why the survey cannot answer it, the estimand, and what resolving it is worth | `design.DecisionSpec`, `identify.identify`, `identify.ols`, `estimands.Estimand`, `design.prior_from_history`, `design.evpi_gaussian`, `design.evoi_gaussian` |
| [2](02-allocating-families.ipynb) | allocating families to plans on a budget: the spread, the D-optimal corners, two arms — what each buys and cannot see; the randomization rule; the calendar | `surface.optimal_exchange`, `design.fisher_information`, `design.expected_posterior_sd`, `design.identifiability_ridge`, `design.estimable_combinations`, `design.assign`, `design.collisions`, `design.variance_inflation` |
| [3](03-four-experiments.ipynb) | four parties in one catalog; plans committed, departures filed as they happen, delivery and attrition checked, a dose that is not a switch, the neighbour or the effect, and the conformance table | `io.Catalog`, `build.ExperimentBuilder`, `io.ExperimentRun`, `io.DefinitionRegistry`, `diagnose.check_delivery`, `diagnose.attrition`, `identify.lee_bounds`, `identify.compliance`, `identify.response_to_dose`, `design.factorial`, `build.readouts_across` |
| [4](04-pooling-the-sites.ipynb) | thirty-three records or four contractors; are these the same quantity; does every party mean the same thing; the decisions taken together; the prior for a fifth site | `meta.three_level`, `meta.pool` (`effect_key="nested"`), `meta.commensurable`, `io.DefinitionRegistry.consensus`, `design.program_decisions`, `meta.prior_from_pool` |
| [5](05-the-national-plan.ipynb) | the surface the data refuses and the one it allows, the contrast at every plan Congress discussed, the marginal cost of a point of tax rate, the transfer to the national population, and what would change the recommendation | `surface.fit`, `estimands.realize`, `surface.marginal_band`, `Estimand.transfer_to`, `identify.transport_verdict`, `design.evpi_gaussian`, `design.time_to_re_experiment` |

## Reading them

They run top to bottom with no arguments and no network, in about a minute in total:

```bash
uv run pytest --nbmake nbs/case-studies/income-guarantee/ -q
```

Notebooks 3–5 each rebuild the same programme from `guarantee.run_programme()`, so any
one of them can be read on its own.

## Notes on the modelling

**The doses are dimensionless.** A guarantee is a fraction of a poverty line and a tax
rate is a fraction of a dollar; declaring them so is what lets one `SurfaceSpec` be
written over the pair without inventing a unit, and the operational readings — the
transfer at zero earnings, the break-even income — are one multiplication away on
every axis.

**The response is negative, so the kernels are polynomials.** Every saturating family
in `axiom.surface` carries a positive amplitude; `PolynomialKernel`'s coefficients are
signed, and with `reference_dose` at the top of each lattice its default prior means
the same thing for both doses. The interaction is the product of the two scaled doses.

**Allocation is a budget problem, and `optimal_exchange` takes an `n`.** The optimal
*shape* of a linear surface's design does not change with `n`, so the exchange runs at
sixty rows and the shares are scaled to whatever the budget buys at each allocation's
cost per family. The D-optimal corners cost two and a half times the plan on the table
and buy 1,440 families where two arms buy 3,550.

**A cross-section laid out as a panel.** `surface.fit` wants `(unit, time)` and a
balanced panel; the families the licensed experiments enrolled are shuffled into blocks
of a hundred, labelled by site, so that every family enters and the panel balances.
The only structure the surface uses is the intercept.

**The quadratic the data refuses.** A quadratic in the tax rate comes back
`Unverified` from the Laplace backend — the Hessian is not positive definite — because
the tax rates the three sites ran do not pin a parabola apart from its slope under a
prior this wide. Notebook 2 predicted it from the information matrix before any family
was enrolled; notebook 5 shows the refusal, typed, and fits the surface the data
supports.

**One seed for the dissolution draw.** `guarantee.DISSOLUTION_SEED` is chosen so that
the null draw at Seattle-Denver lands where history did — a difference an uncorrected
readout calls a finding and a corrected programme does not. Every other site's draw is
unremarkable, and the truth is no effect everywhere.

**What the world does not do.** Wives' hours, which the real experiments also measured;
a nonlinear guarantee response; and the twenty-year cohort's outcomes, which were never
read. The observational record of notebook 1 is a stylized cross-section, not a
reconstruction of any survey.
