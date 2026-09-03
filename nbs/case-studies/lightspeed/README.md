# LUMEN-299 — one constant, five instruments, a century of experiments

A worked case study that uses `axiom` to run a **measurement programme** rather than an
experiment: sixteen measurements of the speed of light, from Fizeau's toothed wheel in
1849 to Evenson's laser in 1972, rebuilt as a synthetic world calibrated to the published
record and carried through the multi-party layer — every experiment filed in a scope,
every plan committed before it ran, every readout pooled with the party it came from, and
every disagreement decided as one of a programme's decisions rather than on its own.
Every number in the five notebooks is computed; the world they share lives in
[`lightspeed.py`](lightspeed.py) beside them.

## The question

> *One constant, measured by five kinds of instrument over a hundred and twenty years.
> How far should the next experiment believe the ones before it — and how does a
> programme tell a finding from a fluctuation when every instrument carries a systematic
> error its own error bar cannot see?*

The truth is 299,792.458 km/s, exactly, since 1983. In 1941 the best pool of the best
experiments said **299,776 ± 4**, four standard errors low, and every number in it was
carefully reduced and honestly reported.

## The answer

**What the quantity is.** Not a counterfactual — a constant of nature is the response to
nothing — so the estimand is declared for what every instrument actually does: set a
transit time, read a distance. A `ratio`, with the medium the light crossed as the
intervention's `version` and the radiation as the `population`. Reading an in-air number
as in-vacuo is a `downgraded` transfer under `version_irrelevance`, and the assumption is
false by 88 km/s — larger than every standard error after 1926. A `DefinitionRegistry`
per laboratory is what knows that; the type cannot.

**Whose it is.** Party = instrument family, programme = laboratory. Sixteen runs planned,
committed, started and read in their own scopes; one survey crossed from Michelson's
scope to the vacuum tube's under a ledger line; the re-survey of Mount Wilson filed as a
deviation (`downgraded`), the vacuum tube's unfiled change caught as `blocked` and left
out of every decision.

**What a reading is worth.** A property of the baseline and its survey, not of patience.
The dimming of the returned image takes the geometry's $1/D$ back to about $1/\sqrt{D}$;
`c` and `D` are one number to the mirror (`identifiability_ridge` is singular along
*both together*), and a steel tape would have capped the 1926 season at 3 km/s whatever
the readings did. Readings could not have settled Annapolis against Washington because
the disagreement was not in them.

**What the pool sees.** A systematic shared within a family is a party-level variance
component and `three_level` finds it — when the families disagree. The five records of
1928–41 pass Cochran's test and pool to an interval three standard errors from the truth;
adding 1926 makes the families disagree and the party level covers it. Without 1926,
both families share the expectation and no hierarchy can tell. The honest number for the
next experimenter is the prediction interval or the `new_party` prior; the interval on
the mean is the narrowest and the wrongest.

**What the programme decided.** Each experiment against the consensus before it: fifteen
decisions at five per cent, three declared, two surviving Holm and LORD, one surviving
e-BH on calibrated e-values. The two are the two history remembers.

## The finding

Agreement is not accuracy, and no pool can see what every party shares. The 1941 number
was precise, agreed, and low; the 1972 number was right to a part in a billion, and the
difference between them is not a better pool. It is a new kind of instrument — filed
where it could be found, and valued for being from outside every family in the book.

## The notebooks

| | | reaches into |
|---|---|---|
| [1](01-one-constant.ipynb) | the unit, the estimand, whose measurement it is, and what each laboratory meant by *c* | `core.UNITS`, `estimands.Estimand`, `Estimand.transfer_to`, `io.Catalog`, `io.Program`, `io.DefinitionRegistry` |
| [2](02-designing-the-mirror.ipynb) | the rotating mirror as one tree; what a reading is worth by baseline; the ridge, and the survey that breaks it | `core.ModelSpec`, `design.fisher_information`, `design.expected_posterior_sd`, `design.identifiability_ridge`, `design.estimable_combinations`, `design.power_from_se`, `design.eig_gaussian` |
| [3](03-mount-wilson.ipynb) | the 1926 season as a programme step: plan, assign, read every night, deviate, file | `build.ExperimentBuilder`, `design.assign`, `design.confidence_sequence`, `io.ExperimentRun`, `calibrate.Measurement`, `build.readouts_across` |
| [4](04-the-book-of-c.ipynb) | the corpus pooled every way, and which way sees the failure | `meta.fixed_effect`, `meta.random_effects`, `meta.heterogeneity`, `meta.three_level`, `meta.leave_one_out`, `meta.egger`, `meta.funnel_data`, `meta.pool`, `meta.prior_from_pool` |
| [5](05-what-the-programme-decided.ipynb) | the century filed, one artifact crossing a scope, records refused by definition, and the decisions corrected as a stream | `io.Catalog.transfer`, `meta.commensurable`, `design.program_decisions`, `design.online_decisions`, `design.from_p_value` |

They run top to bottom with no arguments and no network, in about ten seconds together:

```bash
uv run pytest --nbmake nbs/case-studies/lightspeed/ -q
```

Every notebook rebuilds the world from `lightspeed.simulate()` with a fixed seed, so any
one of them can be read on its own.

## Notes on the modelling

**The world.** Each *generation* of instrument — Foucault's twenty-metre mirror, the
Annapolis apparatus Newcomb copied, the Mount Wilson octagon, the vacuum tube, the Kerr
cells — carries one systematic error, drawn once; each experiment adds its own and
reports the standard error its authors reported. The two generations active in 1928–45
draw their systematics around the same centre, which is the one fact about the world no
hierarchy over families can see and only a new kind of instrument reveals. Michelson's
1926 season is not drawn but *run*, reading by reading, so the number the corpus holds
for 1926 is the number notebook 3 filed.

**One `forward()`.** The rotating mirror is a hand-built expression tree —
$8\pi\,\omega D/c - 4\pi/n$ — and everything reads it: the Fisher information in
notebook 2, the sixteen hundred readings in notebook 3, and the reduction of a reading
back to a speed, which is the tree read backwards and checked against `forward`.

**A survey is an observation.** `c` and `D` enter only as $D/c$, so the mirror alone is
rank-deficient. The baseline survey is a second `Observation` on $D$ alone; its grade —
paced, taped, triangulated — is the design decision, and `expected_posterior_sd` prices
it in km/s on `c`.

**Parts per million.** The corpus pools `relative_change` from the defined value, which
makes every record dimensionless so that `StudyRecord` accepts it without a licensed
transfer. The pool is identical in km/s; the notebooks convert back for every table.

**What did not fit.** `design.anytime.evalue` overflows for a monitoring statistic more
than about forty standard errors from its null, so notebook 3 reads the running mean
against the 1924 preliminary value rather than against Newcomb's 1882 one (which it
excludes from the first night by the interval alone). A `Level` has no honest word for
"a constant", so `aggregate` is used and the notebook says why. `Occupancy` is not used:
a century of experiments in different laboratories does not collide.
