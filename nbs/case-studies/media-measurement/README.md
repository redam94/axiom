# HOUSE-3 — one measurement house, three advertisers, a year of experiments

A worked case study that uses `axiom` to **run a programme of experiments for several
parties and keep the answers comparable**: which channels moved which KPI, for whom, by how
much, and which of those claims survive being made all at once. Every number in the five
notebooks is computed; the world they share lives in [`house.py`](house.py) beside them, and
the year as the house planned, ran and filed it is replayed by [`year.py`](year.py) so that
each notebook stands alone.

## The question

> *A measurement house runs incrementality work for three advertisers. Each has a weekly
> panel — forty markets, fifty-two weeks, spend by channel, a KPI that counts conversions —
> and a media-mix model that says what every channel contributed last year. Each has quoted
> that contribution as the lift. Nine geo experiments are proposed for the coming year. Which
> of them may run, on how many markets, when; what is each readout evidence of; how many of
> the year's go-decisions are wrong; and what can the house say to a fourth advertiser it has
> never worked for?*

Three parties, one vocabulary. Northwind (retail; a conversion is a paid order, and it also
reports revenue), acme (subscription; a conversion is a trial start until week twelve and a
paid subscription after), globex (consumer goods; a conversion is a case sold). Every party's
KPI column is `conversions` — the same `Outcome` — and what each *means* by it is a versioned
entry in the `DefinitionRegistry`, which is the only reason a search lift measured for one can
be compared with a search lift measured for another at all.

## The answer

**What the model says is not the lift.** Every one of the seven channels is overstated by the
party's own model, by 1.2x to 2.0x, because each advertiser buys more media when demand is up
and demand is in no column. `identify` says so before the fit — `downgraded`, route
`backdoor_unmeasured` — and the world, which knows the truth, confirms it.

**The calendar has three kinds of overlap, not two.** Acme's spring social and search tests
share markets and weeks and move different levers: *concurrent*, unbiased in expectation,
a variance cost with a number (a 1.29x standard error), and an interaction assumption that
`factorial` later checks and finds `unverified`. Northwind's two spring search tests share
the lever: *confounded*, and `recommend` will not schedule the second beside the first at any
net value. Acme's autumn search test starts while an unrandomized onboarding rollout is
landing on its largest markets: confounded for a different reason, and moved.

**Nine runs, four answers.** Every experiment is planned, committed and read through
`ExperimentBuilder` in its party's scope, and `ExperimentRun.conformance()` says whether the
readout is the analysis the plan committed to:

| | | conformance |
|---|---|---|
| NW-01 | search holdout, watched on a dashboard every Monday | `identified` |
| NW-02 | video budget doubled; the platform delivered 53–92% of it | `downgraded` — dose, asserted |
| NW-03 | search holdout read on an assignment re-drawn after commit, nothing filed | `blocked` |
| AC-01, AC-02 | the concurrent pair, factorial-checked | `identified` |
| AC-04 | search holdout whose control markets went dark — the ones whose demand fell | `downgraded` — population, unverified |
| GX-01 | tv holdout that ran ten weeks instead of eight | `downgraded` — occupancy and window, asserted |
| GX-02 | display holdout on four scheduled looks, stopped at the third | `downgraded` — stopped early, corrected |
| GX-03 | search holdout that reached seven markets in ten | `identified`, complier effect on the ledger |

**The dashboard was wrong every Monday.** NW-01's fixed-sample interval cleared zero on all
eight Mondays, narrowly; the confidence sequence never did. **The quarter, corrected.** Six of
twenty-four readouts are *go* at 5% party by party — one of them a guardrail no channel
touches, the one-and-a-fifth false goes a null book expects. Under e-BH, one survives: AC-04,
the readout notebook 3 shows sits two standard errors above the truth. Error control corrects
the count of false goes; it cannot see that a number is biased, and the warning about AC-04
is on its run, not in the multiplicity.

**Four records from three parties are not four draws.** The search pool refuses the complier
record, records the population difference as an assumption, checks each party's definition
of a conversion against the reference's — and would have refused acme's records in March. The
nested pool widens the family mean's interval by a fifth against the flat one; a fourth
advertiser's prior is the widest of four, and reaches its scope only through a
`Catalog.transfer` with a ledger line. Northwind's model, calibrated by its own experiment
through the likelihood, moves a quarter of the way to the truth, and `agreement` says so.

## The finding

The house started the year able to say "search drives seven percent of conversions". It ends
the year able to say what that sentence is: a family mean over three parties with a
within-party spread and a between-party spread, made from four readouts whose plans, departures
and populations are on record — and the record is the product. The number did not get better.
What got better is that every way it could be wrong is written down next to it.

## The notebooks

| | | reaches into |
|---|---|---|
| [1](01-three-parties-one-house.ipynb) | three parties, one house, and the number that was never a lift | `adapters.marketing` (`Channel`, `Geo`, `KPI`, `MarketingRoles`, `panel_from_marketing_frame`, `role_map`, `contribution`, `roas`, `marginal_roas`), `identify`, `io.Catalog`, `io.Program`, `io.DefinitionRegistry`, `build.ExperimentBuilder` |
| [2](02-the-calendar.ipynb) | the calendar: noise, arms, worth, collisions, filing | `design.ClusterDesign`, `cluster_power`, `cluster_mde`, `holdout_tradeoff`, `match_clusters`, `assign`, `ValuePerOutcome`, `DecisionSpec`, `opportunity_cost`, `evaluate_candidate`, `pareto_front`, `Occupancy`, `collisions`, `variance_inflation`, `exclusions`, `recommend`, `schedule_with_cooldown`, `ExperimentBuilder.plan` / `commit` |
| [3](03-in-the-field.ipynb) | the year, read and filed | `design.confidence_sequence`, `monitor`, `obrien_fleming`, `ExperimentBuilder.read_stopped`, `diagnose.check_delivery`, `attrition`, `identify.response_to_dose`, `compliance`, `lee_bounds`, `design.factorial`, `estimate`, `calibrate.Measurement`, `ExperimentRun.deviate` / `conformance` |
| [4](04-the-quarter.ipynb) | the quarter, decided together and on arrival | `build.readouts_across`, `design.Readout`, `from_p_value`, `stagewise_tail`, `expected_false_go`, `program_decisions`, `online_decisions`, `io.Catalog.find`, `CrossScopeError` |
| [5](05-the-book.ipynb) | the book: pooling across parties, and the model calibrated | `meta.record_from_summary`, `commensurable`, `pool` (`effect_key="nested"`), `three_level`, `prior_from_pool`, `io.Catalog.transfer` / `crossings`, `calibrate.agreement`, `derive_prior`, `fit_calibrated`, `surface.response_band` |

They run top to bottom with no arguments and no network, in about two minutes altogether:

```bash
uv run pytest --nbmake nbs/case-studies/media-measurement/ -q
```

Notebooks 3, 4 and 5 rebuild the year from `year.py` (`open_house`, `plan_year`,
`field_year`, `read_year`), and notebook 3 checks that the replay reproduces its own filings
hash for hash, so any one of them can be read on its own.

## Notes on the modelling

**This is the adapter earning its keep.** Marketing words — channel, spend, KPI, market,
ROAS — appear in this case study and nowhere else under `nbs/`, and they appear only through
`axiom.adapters.marketing`. `Channel is Treatment`, `Geo is Unit`, `KPI is Outcome`; the role
map translates with nothing left over, and every number is produced by `design`, `identify`,
`diagnose`, `calibrate`, `meta` and `io` with the adapter supplying names at the edge. Rule 2
says the core never learns these words, and the case study is the demonstration that it never
needed to. `roas` on a count KPI is a typed `Unsupported`; northwind's revenue reading of the
same year is where the ratio is a ROAS.

**One `forward()`.** The world's KPI is `axiom.core.value` on the `ModelSpec` that
`axiom.surface` builds from each party's `SurfaceSpec` — the same Hill saturation, geometric
carryover and Fourier seasonality the analyst fits — plus one term the model does not have:
an unobserved, market-level demand shock that raises conversions and raises the week's spend.
There is no second copy of the transform chain in `house.py`; the confounding is a term added
to the model's mean, not a re-implementation of it.

**Two choices the analyst made and the world did not.** The media-mix model is fitted with a
per-market intercept rather than a hierarchical one, because the Laplace backend does not
certify the hierarchical fit on a forty-market panel and fifty-two weeks per market leave the
pooling nothing to buy. Its carryover is two weeks rather than the world's four: with four lags
and forty intercepts the posterior has a near-flat direction the Laplace Hessian will not
certify, and two lags carry most of a decay of one half. Both are in `house.analyst_spec` with
their reasons.

**The design math ran optimistic.** `cluster_mde`'s standard error for NW-01 was 3.2 against a
realized 4.4: the ICC on eight-week blocks does not capture how long a market's demand
persists. The notebooks quote both, because a design number that the field then contradicts
is worth more on the page than one that is quietly revised.

**Every departure is on the run, in the vocabulary identification uses.** A stopped study
files its median-unbiased estimate and a `stopped_early` deviation in one call; a delivered
dose short of the assigned one is a `dose` deviation with the delivery report's ledger lines;
control markets that went dark are a `population` deviation left `unverified`, with Lee
bounds beside the naive number. The one change nobody filed is `blocked`, and the block
travels: `readouts_across` never hands it to the programme.
