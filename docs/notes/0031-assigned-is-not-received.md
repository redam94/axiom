# 0031 — Assigned is not received, and the readout has to say which

*Kind: decision. Opened 2026-08-27. Status: implemented on
`feature/compliance-and-itt`. Closes item 7 of the 1.3 backlog in
[0027](0027-scope-and-the-experiment-lifecycle.md) §D27.6.*

Note [0030](0030-the-experiment-that-was-actually-run.md) ended by saying what
it did not do: `diagnose.delivery` reports that one arm was reached at 98 % and
the other at 60 %, and stops. The obvious next move looks like a correction —
adjust the estimate for non-compliance — and it is the wrong move. There is
nothing to correct, because there is no longer one quantity:

    intention to treat   the effect of being *assigned*, over everybody assigned
    complier effect      the effect of being *exposed*, over the units whose
                         exposure the assignment moved

Both are causal. Both are identified by the randomization. They are not two
estimators of one number, and `identify.compliance` reports them side by side
so that nobody has to decide which one a bare "the effect" meant.

## D31.1 — this is a facet difference, one layer below where the facets can see it

The charter's argument is that two quantities are the same quantity only if all
eight facets match. ITT and the complier effect differ on **two at once**:

| facet | ITT | complier effect |
|---|---|---|
| `intervention` | assigned to treatment | received treatment |
| `population` | everybody assigned | the compliers |

That is the multi-party failure in miniature. A house running experiments for a
party whose operations deliver at 95 % and another at 60 % will produce an ITT
from one and a complier effect from the other without anybody choosing to — the
first because non-compliance looked negligible, the second because it did not —
and `meta.pool` will pool them, because a `StudyRecord` carries a number, an se,
and a quantity name, and nothing in it distinguishes these two.

The module cannot fix that on its own. What it can do is make the distinction
impossible to lose at the point where it is created: `compliance` never returns
one estimate. `ComplianceReport.summary()` prints the population under each
number, and `ledger()` writes *"two estimands, not two estimates of one"* into
the readout.

## D31.2 — 2SLS, not a second Wald ratio

The complier effect is `two_stage_least_squares(y, x=exposure,
instruments=assignment)`. It is exactly the Wald ratio with no covariates — the
notebook checks `itt / share == complier` to six digits — and writing it as
2SLS rather than as a ratio buys three things for free: covariates, the
first-stage F that function already computes, and `weak_instrument_check`, whose
`Assumption` goes straight onto the report. Rule 3's reason applies: two
implementations of one piece of mathematics drift apart.

The relevance assumption is therefore the one of the three the data speak to
directly, and it is the one that bites. In the notebook a complier share of
0.010 gives a standard error of 10.0 against 0.051 at 61 % compliance, an
interval of `[-25, 14]`, and `instrument_strength` at `violated`.

## D31.3 — `downgraded`, never `identified` — with one exception

`ComplianceReport.verdict()` returns the identification vocabulary:

| | |
|---|---|
| `identified` | **only** under perfect compliance, where being assigned and being exposed are the same event and the randomization licenses the effect alone; `route="randomization"` |
| `downgraded` | the ordinary case, under `exclusion_restriction`, `monotonicity` and `instrument_strength` |
| `blocked` | no instrumental variation, or monotonicity's checkable implication failed |

There is no path to `identified` with any non-compliance, and there never will
be: the exclusion restriction is not checkable in the data. Monotonicity is not
checkable either, but it has one implication that is — the complier share, being
a difference of two rates, cannot be negative — so a control arm exposed more
than the treated arm moves `MONOTONICITY` to `violated` and the verdict to
`blocked` rather than downgraded. That is the difference between an assumption
nobody has tested and one the data contradict, and the ledger keeps them apart.

## D31.4 — perfect compliance is a collapse, not a failure

When the exposure column *is* the assignment column there is no instrumental
variation and 2SLS is undefined. Reporting that as an error would have been
wrong twice over: nothing failed, and the ITT is already the effect of exposure
over everybody.

So `complier_effect` returns `Unverified` with that as the reason,
`ComplianceReport.perfect_compliance` is `True`, and the verdict is
`identified` via `randomization`. The summary says *"compliance is perfect: the
two rows are the same quantity"*. A caller who never has non-compliance sees the
machinery say so and go quiet.

## D31.5 — a bug worth recording: `Unverified` is falsy

`core.Unverified.__bool__` returns `False` — deliberately, so `if not result:`
reads well at a call site. The first draft of `ComplianceReport` wrote
`self.unverified.reason if self.unverified else "no complier effect"` in three
places, and every one of them silently threw away a perfectly good reason and
printed the fallback. It looked correct in review and was caught by the output
of a smoke run.

All three now test `is not None`, and
`test_an_unverified_reason_is_not_lost_to_its_falsiness` pins it. Anything in
this repo holding an optional `Failure` should do the same: the falsy result
types make `if x` and `if x is not None` different questions.

## D31.6 — what this does not do

**No attrition model.** Units that dropped out are not units that were
unexposed; missing outcomes are a selection problem, not a compliance one, and
neither `compliance` nor `diagnose.delivery` addresses it. The delivery check
will happily compare assigned counts to *observed* counts if given them, which
detects differential attrition and says nothing about how to handle it.

**Binary exposure only.** `compliance_table` requires 0/1 columns. A
partially-delivered dose — half the sessions attended, two of five weeks
exposed — is a continuous first stage, where "complier" stops being a type and
becomes a local average derivative. That is a real generalization and it is not
here.

**No estimand objects.** The two quantities are described in prose and by the
report's structure, not as two `estimands.Estimand` values differing on
`intervention` and `population`. Building them would let `transfer_to` refuse
the pooling this note is about, which is the right eventual answer and a larger
change than this one: `Estimand` has no vocabulary for "the compliers" as a
population, because that population cannot be described by a covariate
distribution — only sized.

That last one is the real remaining gap, and it is now the most valuable thing
in this area: **a `population` facet that can hold a latent subpopulation**,
after which `meta.pool` would refuse the mixed pool by construction instead of
relying on whoever filled in the `StudyRecord`.
