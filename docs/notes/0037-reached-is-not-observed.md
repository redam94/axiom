# 0037 — Reached is not observed, and a bound is not a downgraded point

*Kind: decision. Opened 2026-08-28. Status: implemented on
`feature/attrition-and-bounds`. Closes the attrition half of what
[0031](0031-assigned-is-not-received.md) §D31.6 left open inside item 7 of the
1.3 backlog.*

`diagnose.delivery` checks that the treatment reached the units.
`identify.compliance` reports the two estimands that partial delivery produces.
Neither touches the other failure, and 0031 said so: *"units that dropped out
are not units that were unexposed; missing outcomes are a selection problem,
not a compliance one."*

## D37.1 — two failures that look alike and are not

| | what is missing | what survives |
|---|---|---|
| non-compliance | the *treatment* for some units | two estimands, both estimable (`identify.compliance`) |
| attrition | the *outcome* for some units | a bound, over a population that is not everybody |

A unit assigned to treatment and never exposed has an outcome, and the outcome
is real. A unit whose outcome is missing has no number, and the units without
one are not a random subset of the units with one — they churned, dropped out or
died, and whatever made them do that is plausibly related to what the number
would have been.

## D37.2 — overall attrition is a power problem; differential attrition is not

`diagnose.attrition` reports both and flags only the second. A study that loses
30 % of both arms has lost power. A study that loses 31 % of one arm and 11 % of
the other has lost the *comparison*, because the survivors are no longer the
same population, and the size of that gap is a lower bound on how wrong the
naive contrast can be.

`Attrition.verdict()` is `identified` or **`blocked`** and never `downgraded`.
No assumption over observables licenses a contrast between two arms whose
survivors are different populations. What licenses a number there is a bound,
and a bound is a different estimand rather than a weakened version of this one —
so the verdict names `identify.lee_bounds` instead of pretending an assumption
would do.

## D37.3 — Lee bounds, and the population they are over

`identify.lee_bounds` implements the trimming bounds (Lee 2009). Under
monotonicity in selection, the reporting units of the arm with the higher
response rate are a known mixture of always-responders and marginal ones;
trimming the marginal share off the top of that arm's outcome distribution gives
the lowest the always-responder effect can be, and off the bottom gives the
highest.

The notebook's world is built so the answer is checkable: the always-responder
effect is exactly 1.0, the marginal responders are low-outcome units, and

* the naive contrast over reporting units is **0.71** — wrong by almost 30 % of
  the effect, in the knowable direction;
* the bounds are **[0.37, 1.01]**, and contain the truth.

`LeeBounds.population` is a `core.Population` carrying
`LatentSelection(kind="always_responder", ...)`. That is the facet
[0032](0032-a-population-you-can-size-and-cannot-list.md) added, used by a
second construction that needed it: always-responders are sized by two response
rates and can never be listed, so an estimand built on this population inherits
the pooling refusal without anything new being written.

## D37.4 — `downgraded` even when the bounds collapse

Equal response rates trim nothing and the bounds meet at a point. The verdict is
still `downgraded`, because the quantity is still the always-responder effect
and selection monotonicity is still not testable — the units that would falsify
it are exactly the ones never observed in one arm. A collapsed bound is a
narrower answer to the same question, not a stronger licence.

## D37.5 — what this does not do

**No sampling uncertainty.** `lee_bounds` returns the *identified set*: where
the parameter can be, given infinite data. A confidence interval for a partially
identified parameter is a different and more delicate object (Imbens and Manski
2004), and returning the identified set labelled as one is better than returning
something that reads like an interval and is not. This is the most valuable
missing piece here.

**No covariates.** The bounds are unconditional. Trimming within covariate
strata and aggregating is tighter, is standard, and is not implemented — and it
is also the only route to *challenging* monotonicity, since the check is a
stratum in which the response rates run the other way.

**Binary assignment only.** Multi-arm attrition needs a bound per contrast, and
the module takes one 0/1 assignment column.
