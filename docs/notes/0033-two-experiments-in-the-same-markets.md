# 0033 — Colliding is not automatically wrong, and the calendar knows which

*Kind: decision. Opened 2026-08-27. Status: implemented on
`feature/experiment-collision`. Closes item 2 of the 1.3 backlog in
[0027](0027-scope-and-the-experiment-lifecycle.md) §D27.6.*

`design/methods/registry.py` names `no_interference` — *"a unit's outcome
depends only on its own treatment"* — as an assumption of five of its six
methods. Nothing has ever checked it, and between units it is genuinely hard to.
Between *experiments* it is arithmetic, and the schedule has the information
before either one runs.

`design.portfolio.recommend` had the opposite problem: it ranked candidates by
what learning about each is worth and picked the ones that fit the budget,
having never asked whether they can run at the same time. For a house running
many experiments per party they usually cannot.

## D33.1 — three answers, not two

The tempting design is a boolean: do these two overlap. That collapses the case
that matters.

| kind | when | verdict |
|---|---|---|
| `disjoint` | no shared unit, or no shared period | `identified` |
| `concurrent` | shared units and periods, **different** levers, both independently randomized | `downgraded` |
| `confounded` | shared units and periods and a **shared lever**, or a neighbour that was not independently randomized | `blocked` |

`concurrent` is the interesting one and it is *not a bug*. Two independent
randomizations are orthogonal in expectation, so neither estimate is biased by
the other. What the overlap costs is variance — the neighbour's effect sits in
your residual — plus a real risk to `no_interference`, because the two
treatments may interact. `CONCURRENT_EXPERIMENTS` carries both, and its
`challenged_by` names the only thing that settles the second: *a factorial
analysis of the two assignments together*.

`confounded` covers two situations that feel different and are not. Both designs
moving `price` on London in weeks 6 and 7 means neither price effect is
separately identified. A neighbour that was **not** independently randomized —
a staged rollout, a hand-picked pilot — confounds everything it touches for the
same reason one level up: its assignment may correlate with anything, so the
orthogonality that rescues the concurrent case is gone. `Occupancy.randomized`
is how a schedule says so, and it defaults to `True` because most things on a
calendar are experiments; a rollout has to declare itself.

## D33.2 — the unit labels are the calendar's, not the analysis's

`Occupancy.units` are the labels a *program schedules on* — markets, clinics,
cohorts, cells. Passing four hundred thousand individual ids would work and
would miss the point: a program does not schedule individuals, and two
experiments that share a market share it whether or not they happened to draw
the same people out of it. This also keeps the spec small, which matters because
an `Occupancy` is a `Spec` and specs hold no large arrays.

`treatments` are the levers a design *moves*, not the outcomes it reads. Two
experiments measuring the same outcome without moving the same lever do not
collide on that account, and the ladder above depends on the distinction.

## D33.3 — the cost has a number

`variance_inflation(effect, share, sd)` is `1 + share·(1 − share)·effect²/sd²`.
A neighbour that moves the outcome by `effect` on a `share` of your units adds a
Bernoulli component to your residual, so the factor converts straight into the
`n` the same design now needs.

It is worth showing because the shape is not the intuitive one. The ends are
free: if nobody or everybody is in the neighbouring experiment it is a constant,
not noise. The worst case is a fifty-fifty neighbour, which is the usual one,
and a neighbour whose effect equals your outcome's own standard deviation costs
a quarter more units at that split.

The number is exact when the neighbour's effect is known and an upper bound on
the cost when it is not, because an independent assignment contributes variance
and never bias.

## D33.4 — an exclusion is not bought off by net value

`exclusions(collisions)` turns the confounded pairs into a symmetric map, and
`recommend(candidates, exclusions=...)` skips a candidate whose exclusion set
already holds a selected treatment — however good its net value is. That is the
whole point: two designs about to move the same lever on the same units in the
same weeks do not become compatible by being worth a lot, and a greedy knapsack
that only knows about money will happily schedule both.

The default `kinds=("confounded",)` puts only the blocking pairs in the map.
A concurrent pair is a *cost*, and a cost belongs in the value arithmetic
(`variance_inflation` → a larger `experiment_se` → a smaller EVSI), not in a
constraint. A program that will not accept the variance either passes
`kinds=("confounded", "concurrent")` and says so.

Skipped candidates are named in `detail["skipped_excluded"]` beside the existing
`skipped_over_budget`, because "we did not run it" and "we could not run it" are
different facts about a quarter.

## D33.5 — what this does not do

**No time-resolved overlap.** `shared_periods` intersects two windows and counts
the periods. It does not know that one experiment's units enter in week 3 and
leave in week 6, which a staggered rollout would need, and it treats a one-week
overlap of an eight-week study the same as a total one except for the count.

**No automatic scheduling.** `exclusions` feeds the existing greedy knapsack.
There is no solver that shifts an experiment's window to remove a collision,
which is the thing a program actually wants and is a scheduling problem rather
than a statistical one. `design.schedule_with_cooldown` is the nearest existing
piece and knows nothing about collisions.

**No factorial analysis.** `CONCURRENT_EXPERIMENTS.challenged_by` names the
check that would settle whether two concurrent treatments interact, and that
check does not exist. Building it means fitting both assignments and their
interaction in one model, which `identify.ols` can already do given the columns;
what is missing is the piece that assembles them and reports the interaction
against the assumption. That is the obvious next item in this area.

**Nothing files an `Occupancy`.** Like `ArmAssignment` before it, an `Occupancy`
is a `Spec` that could be a role on an `io.ExperimentRun` and a row in a
`Catalog`, and no code puts it there. The `build` entry point deferred in 0027
now has a fourth specific job.
