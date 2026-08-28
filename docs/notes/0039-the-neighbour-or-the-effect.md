# 0039 — Was the neighbour the background, or the effect?

*Kind: decision. Opened 2026-08-28. Status: implemented on
`feature/factorial-check`. Closes the follow-on
[0033](0033-two-experiments-in-the-same-markets.md) §D33.5 named: the factorial
interaction check `CONCURRENT_EXPERIMENTS.challenged_by` asks for.*

`design.collision` classifies an overlap as *concurrent* when two designs share
units and periods but move different levers, and says the estimates are
unbiased because independent randomizations are orthogonal in expectation. That
is true of the main effects, and the assumption it rests on has two halves:

> each estimate is the effect of its own treatment averaged over whatever the
> other one was doing, **and the two treatments do not interact**

The first half is arithmetic. The second is the assumption, and until now
nothing checked it.

## D39.1 — the check is a regression, over the estimator that already exists

`design.factorial` fits `y ~ left + right + left·right` through
`identify.ols` — the same estimator, not a second one — and reports the
product's coefficient beside both main effects and the four cell means.

If the interaction is real, **neither main effect is the effect of its own
treatment.** Each is an average over the other experiment's assignment, and the
average changes when the other experiment ends. A price test that lifts revenue
3 % while a banner test is running and 1 % after it stops was never one number,
and the collision was not a variance cost — it was the estimand.

`design` importing `identify` is new here and is what the layering already
allowed: `design` sits above `surface estimands identify`, and
`tests/contracts/test_layering.py` agrees.

## D39.2 — a null interaction is `unverified`, never `identified`

This is the decision that matters, and it is the same one
[0030](0030-the-experiment-that-was-actually-run.md) §D30.6 made for the
delivery checks: failing to refute is not confirming.

An interaction is roughly **twice as hard to detect** as a main effect of the
same size, so an experiment powered for main effects needs about four times its
units to rule out an interaction of that size. Reporting "no interaction, so the
assumption holds" from such a study would be reporting the study's lack of power
as a finding.

So `Factorial.verdict()` is `blocked` on a real interaction and `unverified`
otherwise, and `power_note` says what the study *could* have detected using its
own standard errors — in the notebook, an interaction of about 0.25 or more
against main effects of 2.0 and 1.0.

## D39.3 — an empty cell is refused

A design where one combination never occurred cannot separate the interaction
from the main effects. `factorial` raises rather than fitting, because the
number it would return is a main effect wearing an interaction's name, and that
is exactly the failure the module exists to catch.

## D39.4 — what this does not do

**Two experiments, binary each.** Three concurrent experiments have three
pairwise interactions and a three-way one, and the module takes two 0/1
columns. The pairwise checks can be run by hand; the three-way cannot.

**No power calculation up front.** `power_note` prices the check *after* the
fact. What a program director actually wants is "how many units do I need to
rule out an interaction of size γ before I schedule these two together", which
is `design.power` with the interaction's variance and is not wired up.

**Nothing routes it.** `design.collisions` finds the concurrent pairs and
`factorial` checks one; no code takes the first and feeds the second. That is a
small function and a decision about where a schedule's outcome data lives, which
the `build` entry point deferred since [0027](0027-scope-and-the-experiment-lifecycle.md)
is the natural home for.
