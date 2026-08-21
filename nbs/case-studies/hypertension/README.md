# HYPER-3 — a sequential dose-finding trial in hypertension

A worked case study that runs the whole of `axiom` against one question, end to end.
Every number in the six notebooks is computed; the synthetic world they share lives in
[`hyper3.py`](hyper3.py) beside them.

## The question

> *We have a new antihypertensive. We want to know which of three daily doses to take
> into phase III, in adults who already have high blood pressure or are on their way to
> it. We are going to watch these people every week for six months. If one of the doses
> is making people worse than the care they would have got anyway, we want to stop that
> arm before we find out the hard way.*

**HYPER-3.** 400 adults with elevated or stage-1 hypertension, randomized 2:1:1:1 to the
standard of care or to 10, 20 or 40 mg daily, stratified by age band (25–35, 36–50, 51+).
Seated systolic pressure measured weekly for 24 weeks, enrollment staggered over 16
calendar weeks. Twelve harm contrasts — each dose against control overall and inside each
age band — monitored at six safety reviews against a posterior-probability boundary.

## What happens

The 40 mg arm raises pressure in the oldest band and lowers it in the two younger ones,
and the two very nearly cancel. Pooled over the trial population the top dose looks like
an unremarkable, mildly beneficial arm at every single review. In the 51+ band it crosses
the harm boundary at the **first** scheduled review, on three eighths of the information,
and a third of the units planned for that arm are never randomized to it.

Nothing about the estimator found that. Pre-specifying the age strata as monitored
contrasts did.

## The notebooks

| | | Reaches into |
|---|---|---|
| [1 — The trial](01-the-trial.ipynb) | the protocol, the synthetic world, randomization checks, trajectories, accrual, retention | `core`, `data`, `sim` |
| [2 — What it identifies](02-what-it-identifies.ipynb) | the DAG; why adherence must not be adjusted for; why completers must not be analysed alone; transporting to an older population | `identify`, `estimands` |
| [3 — Sizing it](03-sizing-it.ipynb) | power, the square-root allocation rule, stratum-level MDE, what an interim can and cannot see, the clinic-randomized counterfactual, cost per mmHg | `design.power`, `design.cluster`, `design.precision` |
| [4 — The dose–response](04-the-dose-response.ipynb) | a Hill surface per stratum, the informative misfit where the curve reverses, identifiability, which doses phase III should test | `surface`, `design.structural`, `viz` |
| [5 — The sequential design](05-the-sequential-design.ipynb) | the harm boundary as the protocol sentence, what it spends, what it buys in unit-weeks of exposure, efficacy boundaries and spending functions, the value of an interim | `design.sequential`, `design.eig`, `design.evoi` |
| [6 — Running the trial](06-running-the-trial.ipynb) | six reviews, twelve contrasts, the stop, the bias of the stopped estimate, the readout, heterogeneity, the ledger | `design.sequential`, `identify`, `meta`, `viz` |

## Reading them

They run top to bottom with no arguments and no network:

```bash
uv run pytest --nbmake nbs/case-studies/hypertension/ -q
```

or open them in order. Notebooks 2–6 each rebuild the same trial from
`hyper3.trial(seed=20260821)`, so any one of them can be read on its own.

## The five conclusions

1. **Randomization identifies one quantity**, the intent-to-treat contrast, and only for
   analyses that condition on nothing downstream of the assignment. Adherence is a
   mediator; dropout is a descendant of the outcome with an unmeasured common cause.
2. **A trial that can see a stratum-level harm at closeout cannot see it at an interim,
   and testing repeatedly is not a fix.** HYPER-3 detects 2.4 mmHg pooled and 3.8–4.8
   inside a band at closeout; at the first safety review the oldest band has 69 % power
   against the harm that is actually happening. Testing at 5 % on each of six reviews
   would be more sensitive and would spend 0.065 rather than 0.05 per contrast — half of
   all null trials would trip something across twelve. The boundary gives up a tenth of
   the sensitivity for a sixth of the error, and the notebook prints both.
3. **The margin is the whole trade, and it is priced.** No margin saves 80 % of the
   exposure and trips one null contrast in eight; a 4 mmHg margin trips one in 2500 and
   saves 35 %. 2 mmHg is a clinical judgement and the table is what it costs.
4. **An estimate reported at the look that stopped a study is biased away from the null**
   — here by about a seventh of the effect. The ledger line says so, and the reported
   estimate comes from every unit that reached the primary window.
5. **The pooled 40 mg number should never be reported alone.** `I²` puts 96 % of the
   spread across age bands down to real heterogeneity.

## What this case study added to axiom

Two things, both because a notebook walked into the gap.

`axiom.design.sequential` — group-sequential boundaries (Pocock, O'Brien–Fleming,
Lan–DeMets spending, posterior-probability harm rules), exact crossing probabilities by
numerical integration of the canonical joint distribution, and `monitor` for walking a
realized path against a rule. See
[`docs/notes/0004`](../../../docs/notes/0004-case-study-hypertension.md).

The **basis response families** in `axiom.surface` — `PolynomialKernel`, `SplineKernel`
(natural cubic, linear outside its boundary knots) and `PiecewiseLinearKernel` — because
every kernel that existed was monotone and this trial's oldest age band has a response
that turns over. Notebook 4 §3b is the before and after. See
[`docs/notes/0005`](../../../docs/notes/0005-basis-response-families.md).
