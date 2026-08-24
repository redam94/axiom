# 0020 — The tutorial: one decision across every phase, ending in a document

*Kind: progress + decision. Opened 2026-08-23. Status: implemented; `nbs/tutorial/01-the-whole-loop.ipynb`.*

The notebook tree had a series per subpackage (gate 12), five end-to-end
notebooks that each cross a few subpackages, and three case studies. What it did
not have was a *front door*: one notebook that carries a single decision from
"what are we asking" to a written report, so that a reader meets the phases in
the order the work actually happens rather than in the order the code is
organized. `nbs/tutorial/01-the-whole-loop.ipynb` is that notebook.

The worked question is a logistics operator's standing maintenance schedule:
40 depots in one region, 500 nationally, an on-time-delivery index with
carryover and saturation, and a decision about going from nothing to 60 hours a
week per depot. The world is `axiom.sim.surface_world`, so every estimate can be
checked against a truth the story's analyst does not have.

## D20.1 — The tutorial ends in `axiom-dossier`, so the dev group installs it

The last phase of any real analysis is the document, and the charter puts
rendering outside `axiom` (note [0015](0015-report-narration-addon.md)). A
tutorial that stops before the report stops one phase early, so this one imports
`axiom_dossier` in its final section.

That makes the add-on a **dev-time dependency of the root project**:
`axiom-dossier[render]` is now in the `dev` group with a
`[tool.uv.sources]` workspace entry. The direction of the dependency is
unchanged — `axiom` still imports nothing from it, and the twelve gates still
cannot see it — but `make notebooks` can now execute the one notebook that
demonstrates it. `pip install axiom` is unaffected: the dev group is not part of
any published extra.

## D20.2 — The numbers had to reverse the decision, and the parameters were chosen so they could

A tutorial whose experiment confirms the model teaches nothing. The world's
parameters were tuned until three things were simultaneously true, without any
of them being stated as an assertion in the prose:

1. the observational fit is **well identified but confounded** — with the
   confounder removed the same fit recovers `beta_a` to within a posterior sd,
   so the bias in the notebook is attributable to the unmeasured node and not to
   a weakly identified Hill;
2. the confounded read of the decision estimand sits **above** the break-even
   line and the calibrated read sits **below** it, so the experiment reverses
   the decision rather than sharpening it;
3. the break-even is arithmetic on prices (hours × cost ÷ value per point), not
   a threshold chosen to make the story work.

The third is what keeps the second honest. The break-even is `60 × 45 / 400 =
6.75` index points per depot-week; the truth is 5.9; the confounded model says
7.2. Nothing in the notebook picks the threshold after seeing the estimate.

## D20.3 — Two things the tutorial says that no other notebook says

**EVSI cannot see an unidentified prior.** The value of information is computed
*against* the prior, so a confidently wrong prior makes an experiment look less
valuable than it is. The notebook states this where the EVSI is computed: the
reason to run this experiment is the identification verdict; EVSI is how to
choose *between* designs, not whether to believe the model.

**Power for the effect is not the binding question.** At the chosen design's
standard error, power at the anchored effect is 1.000 and says nothing. What the
design has to do is separate "pays for itself" from "does not", so
`simulated_power` is checked at the *distance from the prior to the threshold*
(0.69 points, 73 % predicted power) rather than at the effect size.

The closing phase follows the same discipline: repeating the experiment would
gain 0.26 nats and four dollars, no dose on the grid clears its own bill, and the
experiment that would overturn that needs 1,377 depots against the 500 the
network has. The honest follow-up plan is to stop, and the notebook says so.

## D20.4 — `quantity_from` now reads a `Summary`

Writing the report exposed a gap in the add-on. Its collector claimed to accept
"anything carrying `.value` and `.interval` — `estimands.EstimandResult`", but an
`EstimandResult` carries neither: it holds a `core.Summary`, which states its
point as `mean`. The result was a `TypeError` on exactly the object a report is
most likely to be written from, or — for a bare `Summary` — silently reporting
the midpoint of an interval that need not be symmetric about the mean.

`quantity_from` now unwraps `.summary` and reads `.mean`, with the wrapper's type
name kept as the provenance string. Two tests in
`packages/axiom-dossier/tests/test_evidence.py` pin both halves.

## D20.5 — A metric block was printing an interval's midpoint

The same section exposed a second, quieter one. `Evidence.context()` handed each
quantity to the renderer *as its `Interval`*, and `axiom.report`'s `_metric_of`
resolves a bare interval by taking its midpoint — which is not the estimate when
the interval is an HDI. The tutorial's calibrated reading of the decision
estimand is 239.57 with a 90 % HDI of [224.55, 255.09]; the page printed 239.82.

A quarter of a point is not a wrong decision, and that is exactly why it is the
kind of drift worth closing: nothing would ever have flagged it. `context` now
hands over the whole `Quantity`, which grows a `mean` property — the one name
`_metric_of` reads a summary's point by — so the page prints the number the
record holds.
