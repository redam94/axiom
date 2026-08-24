# examples/ — the same library, twelve different fields

axiom's core has no domain in it. Its nouns are `Treatment`, `Dose`, `Unit`,
`Outcome`, `Covariate`, and a contract test fails the build if marketing
vocabulary escapes `axiom.adapters`. These examples are the argument for why that
was worth doing: a trial, a wheat field, a reactor, a nature reserve and a
checkout page are the same mathematics wearing different words.

Each one is a **narrated walkthrough**, not a listing. Every step says why it is
the necessary step and what a reasonable person would have reached for instead;
every one ends with what the run actually showed, which is not always what the
setup pointed at. Every one runs against a synthetic world whose truth is known,
so the numbers can be checked rather than admired.

```bash
python examples/run_all.py            # all twelve, about a minute
python examples/run_all.py 07 12      # just those
python examples/07_clinical_rwe_calibration.py
```

Only the core install is needed. No sampler, no extras.

## The examples

| # | Field | Question | Pillar |
|---|---|---|---|
| 01 | Agronomy | Where does yield peak in nitrogen × irrigation, and how should a rationed budget be split? | surface |
| 02 | Labour economics | Does a training programme raise earnings, when enrolment is self-selected? | identify (IV) |
| 03 | Education | How strong would an unmeasured confounder have to be to erase the tutoring effect? | identify, diagnose |
| 04 | Epidemiology | An exposure you cannot deconfound — recovered through a measured mediator. | identify (front-door) |
| 05 | Conservation ecology | Does a result from one reserve apply to another, and what must you measure there first? | identify (transport) |
| 06 | Public health | Sizing a cluster-randomized trial, where the intra-cluster correlation eats the sample. | design |
| 07 | Clinical research | Using a small clean trial to correct a large confounded registry. | calibrate |
| 08 | Energy | Which quasi-experimental method actually holds its size on *this* panel? | design |
| 09 | Process engineering | Climbing to a yield optimum, Box–Wilson style. | surface |
| 10 | Marketing | Contribution and marginal return — and proof the vocabulary is a thin adapter. | adapters, surface |
| 11 | Behavioural science | Pooling a literature that disagrees with itself. | meta |
| 12 | Product analytics | Is this A/B test worth running at all? | design (value of information) |

## Results worth skipping to

Several of these end somewhere other than where the setup points, which is the
main reason to read them rather than trust the summary:

- **06** — the same 1,600 children give 55% power or 99% power depending only on
  how they are grouped into villages. Then field cost is added and the answer
  moves again, to an interior optimum neither extreme is near.
- **12** — the *worst*-powered design wins on net value, and the one with power
  1.00 is net-negative once the withheld upside is counted.
- **09** — the unconstrained optimum lies outside the plant's operating window,
  so the answer is a corner of the window and a question for engineering.
- **07** — calibration removes most of the confounding bias and *none* of the
  three fits' intervals contains the truth. One trial of 280 patients bounds a
  registry's confounding; it does not undo it.
- **10** — the headline average and marginal returns agree to two decimal places
  while the marginal curve behind them varies by a factor of thirteen.
- **02** — a chart of estimates against sample size, from 250 to 32,000 workers,
  showing the biased estimator converging confidently on the wrong number.

## How an example is written

Every script narrates itself through `_walkthrough.py`, the one shared module
here. It is stdlib-only and imports nothing from `axiom`, so it can never be the
reason an example works or fails.

```python
from _walkthrough import Walkthrough

w = Walkthrough(field="Agronomy", title="...", question="...")

w.step(
    "Spend the plots where the curvature is",
    why="A saturating response is a shape, and a shape is estimated from runs "
        "placed where it bends.",
    instead="A 5x5 grid measures the same shape with twice the plots and still "
            "puts nothing at the centre.",
)
w.out(f"design: {design.kind}, {design.n} plots")          # printed and recorded
w.table(["dose", "yield"], rows, caption="...")            # aligned and recorded
w.figure("band", kind="band", data={...}, opt={"key": "@"}, title="...")
w.finding("What the numbers actually showed.")
```

Everything both prints and records. Run the script and you get the walkthrough at
a terminal; run it with `AXIOM_WALKTHROUGH_JSON=<path>` and the same run also
writes the structured record the [site](../site/) turns into a page with the
charts drawn. Nothing is printed that is not captured and nothing is captured
that a terminal reader does not see, which is what keeps the page honest.

Chart `opt` values beginning with `@` are references into that figure's own
`data` — `"@"` is the payload, `"@rows"` is `data["rows"]`. The site rewrites them
at build time and fails the build if one does not resolve. Available chart kinds:
`band`, `bars`, `dumbbell`, `funnel`, `heatmap`, `hist`, `intervals`, `lines`,
`scatter`.

## Writing another one

Keep the shape: a docstring stating the field's question in that field's own
terms, a synthetic world with a known truth, then one `w.step(...)` per move.

Two things are worth being strict about.

The `instead` is the part a reader gets nowhere else, so it has to be the
alternative a competent person would genuinely have reached for — not a straw
man. If a step has no honest alternative, leave it out; a dishonest one is worse
than none.

The `finding` is written **after** looking at the output, and it says what the run
showed rather than what the setup implied. Five of the twelve disagree with their
own setup, and that is the most useful property this directory has.

Then add it to the table above and to the `META` map in
`site/_gen/generate.py`, which is where the field, the one-line question and the
pillars live.

The tour on the [site](../site/) covers the same five pillars in more depth, and
`nbs/case-studies/hypertension/` works one trial end to end across all of them.
