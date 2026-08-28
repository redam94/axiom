# 0038 — A dose is not a switch

*Kind: decision. Opened 2026-08-28. Status: implemented on
`feature/continuous-exposure`. Closes the continuous-exposure half of what
[0031](0031-assigned-is-not-received.md) §D31.6 left open, and with
[0037](0037-reached-is-not-observed.md) finishes item 7 of the 1.3 backlog.*

`identify.compliance` assumes exposure is 0/1: a unit attended or did not, and
the three response types — complier, always-taker, never-taker — partition the
population. Half the sessions attended, two of five weeks exposed, sixty per
cent of the budget delivered are the ordinary case in the field, and there is no
complier in any of them, because there is no single event a unit did or did not
respond to.

## D38.1 — a third estimand, not a generalization of the second

The instrumented estimate with a continuous exposure is a **local average
derivative**: an effect per unit of exposure, over units weighted by how far the
assignment moved each one (Angrist, Graddy and Imbens 2000). It is not an
average treatment effect, not a complier effect, and not a rescaling of either.
Three things differ from the binary case at once — the *units* (per unit of
exposure, not per exposure), the *weights* (how far each unit moved, not whether
it moved), and the *population* (weighted, not a nameable set).

So it gets its own report. `DerivativeReport` mirrors `ComplianceReport`
deliberately — same two-estimands-side-by-side shape, same refusal to return one
alone — and does not reuse it, because `ComplianceTable`'s three shares are
meaningless here and forcing them into one type would have made the wrong thing
easy.

The bridge arithmetic survives: `itt = derivative × shift`, with the complier
share replaced by the first-stage shift in mean exposure. The notebook checks it
to six digits.

## D38.2 — the standardized shift is the number to read

`FirstStage.shift` is the difference in mean exposure between the arms — the
ratio's denominator, and what makes the estimate explode when it is near zero.
`standardized_shift` divides it by the exposure's own standard deviation, which
is the number that means something: a shift of 0.02 is weak or strong depending
entirely on units nobody reading a table will remember.

The notebook makes the failure concrete. Shrinking the first stage to a
hundredth of itself takes the shift from 1.02 (0.46 sd) to 0.008 (0.004 sd) and
the derivative from 0.78 to **+102**, and the report is `blocked` on
`instrument_strength` rather than printing it as a finding.

## D38.3 — `UNIFORM_FIRST_STAGE`, monotonicity's continuous cousin

Monotonicity says no unit is exposed *because* it was assigned to control. Its
continuous analogue says assignment moves every unit's exposure in the same
direction, so the derivative's weights are non-negative rather than a mixture of
effects with opposing signs. It is exactly as untestable, and its
`challenged_by` names the only implication the data can speak to: a covariate
stratum whose mean exposure moves the other way.

The verdict is `downgraded` for the same reason as the complier effect's, and
`identified` is not reachable — there is no continuous analogue of perfect
compliance that collapses the two estimands, because the derivative is on a
different scale from the intention-to-treat effect however complete the
delivery.

## D38.4 — what this does not do

**No nonlinearity.** The derivative is one number: a weighted average slope. An
exposure with a saturating response has a different derivative at every dose,
and averaging them is what 2SLS does rather than what anybody wants.
`axiom.surface` is the machinery for a response *curve*, and nothing connects
the two — a designed dose-response experiment should be fitting a surface, and
this module is for an experiment that got a dose it did not design.

**Binary assignment still.** The instrument is one 0/1 column. A multi-armed
encouragement with three intensities is over-identified and would give a
different weighting, which is a real thing to want and is not here.
