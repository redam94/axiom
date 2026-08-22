# examples/ — the same library, twelve different fields

axiom's core has no domain in it. Its nouns are `Treatment`, `Dose`, `Unit`,
`Outcome`, `Covariate`, and a contract test fails the build if marketing
vocabulary escapes `axiom.adapters`. These examples are the argument for why that
was worth doing: a trial, a wheat field, a reactor, a nature reserve and a
checkout page are the same mathematics wearing different words.

Every script is standalone — nothing imports anything else here — and every one
runs against a synthetic world whose truth is known, so the numbers can be
checked rather than admired.

```bash
python examples/run_all.py            # all twelve, about 30 seconds
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
  how they are grouped into villages.
- **12** — the *worst*-powered design wins on net value, and the one with power
  1.00 is net-negative once the withheld upside is counted.
- **09** — the unconstrained optimum lies outside the plant's operating window,
  so the answer is a corner of the window and a question for engineering.
- **07** — calibration removes most of the confounding bias and the resulting
  model is *still* overconfident, which is exactly what a recovery test against a
  known truth exists to catch.
- **10** — a Hill curve is S-shaped, not merely concave, so the return on the next
  unit of spend can exceed the average. A single headline ROAS does not say which
  side of the bend you are standing on.

## Writing another one

Keep the shape: a docstring that states the field's question in that field's own
terms, a synthetic world with a known truth, the axiom calls, and a closing note
that says what the numbers actually showed — including when that is not what the
setup implied. Then add it to `run_all.py`'s directory and to the table above.

The tour on the [site](../site/) covers the same five pillars in more depth, and
`nbs/case-studies/hypertension/` works one trial end to end across all of them.
