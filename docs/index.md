# axiom

**Bayesian causal decision science.** Declare what you want to know, find out
whether the data can tell you, design the experiment that would, fold the
experiment's answer back into the model, map the response surface it implies,
and pool the evidence across every study you have run.

Five pillars and the programme around them, one vocabulary:

| Pillar | Question it answers | Package |
|---|---|---|
| **Causal modeling** | Is this effect identified, and from what? What would a hidden confounder have to look like to overturn it? | `axiom.identify`, `axiom.estimands`, `axiom.diagnose` |
| **Experimental design & planning** | What is worth measuring next, at what size, by which method, at what cost? | `axiom.design` |
| **Experimental calibration** | How does a randomized result update an observational model, and what did that transfer assume? | `axiom.calibrate` |
| **Response-surface methodology** | What does the dose–response surface look like, where is its optimum, and where should I probe next? | `axiom.surface` |
| **Meta-analysis** | What does the whole body of evidence say, and how heterogeneous is it? | `axiom.meta` |
| **Running a programme** | Whose experiment is this, was it delivered as assigned, may these two readouts be compared, and how many of this quarter's decisions are wrong? | `axiom.io`, `axiom.design`, `axiom.diagnose`, `axiom.build` |

## Start here

Every public symbol is demonstrated, executed, in a notebook. The notebook
series live under `nbs/`, one directory per subpackage, numbered
in reading order:

- Foundation: `nbs/core/`, `nbs/data/`, `nbs/io/`, `nbs/infer/`, `nbs/dynamics/`
- Domain: `nbs/identify/`, `nbs/estimands/`, `nbs/surface/`, `nbs/sim/`
- Pillars: `nbs/design/`, `nbs/discover/`, `nbs/calibrate/`, `nbs/meta/`
- Composition: `nbs/diagnose/`, `nbs/build/`, `nbs/adapters/`, `nbs/viz/`, `nbs/report/`

Each [API page](api/index.md) links its own series.

For what using all of it at once looks like, read the **case studies** under
`nbs/case-studies/`. There are six, deliberately different in shape: three
carry one experiment each, and three carry a *programme* of several.

`hypertension/` — **HYPER-3**, a sequential dose-finding trial that must stop a
dose arm early if it is harming people. Six notebooks, one synthetic world, from
the causal graph to the boundary crossing that stops an arm at the first safety
review. Design notes: [0004](notes/0004-case-study-hypertension.md).

`tutoring/` — **TUTOR-60**, one decision carried from problem setup through
experiment planning, measurement and reporting to a recommendation a board can
act on: how much tutoring to fund, for how many students, on a fixed budget.
Five notebooks, two dosed treatments and a budget line. Design notes:
[0009](notes/0009-case-study-tutoring.md).

`rutherford/` — **GEIGER-1911**, an experiment designed before it exists: which
scattering angles, what shape of aperture, how long, and when to stop. Five
notebooks; nothing is analysed until the last. Design notes:
[0018](notes/0018-case-study-rutherford.md).

The next three are the multi-party layer (`docs/notes/0027`–`0042`) in use:
several parties, several experiments on one estimand, every one filed under
its scope with a committed plan behind it, read out, corrected as a book, and
pooled with a party level. Design notes for all three:
[0045](notes/0045-three-programmes.md).

`lightspeed/` — **LUMEN-299**, one constant and five instruments across a
century: laboratories as parties, the rotating mirror designed with the
information calculus, a corpus of readouts whose 1930s pool excludes the truth
because a shared systematic error is a party-level variance component.

`income-guarantee/` — **NIT-4**, four income-maintenance experiments over a
two-dimensional dose (a guarantee and a tax rate): the allocation problem, the
deviations history actually filed, attrition that blocks and the bound that
survives it, a guardrail finding that does not survive the quarter's
correction, and the surface that answers the national question.

`media-measurement/` — **HOUSE-3**, one measurement house running incrementality
experiments for three advertisers over a year: the calendar and its collisions,
peeking priced, delivery and compliance checked, a quarter of go/no-go decisions
corrected as a quarter, a party-level pool, and each party's observational model
calibrated by its own experiment — through `axiom.adapters.marketing`, which is
the one place that vocabulary is allowed.

## Layering

Imports point down only; `tests/contracts/test_layering.py` enforces it.

```text
  report                                   (leaf; templates over viz)
   |
  viz    adapters
   |        |
  build   diagnose                         (compose everything below)
   |        |
  meta calibrate design discover           (the pillars; peers, no cross-imports
   |        |        |                      except design <- surface, calibrate <- estimands)
   +--------+--------+
            |
        surface   estimands   identify     (domain layer)
            |         |          |
            +---------+----------+
                      |
             infer     dynamics          (sampler seam, and the system compiler;
                 |        |                 peers, neither imports the other)
                 +--------+
                      |
                  core    data    io       (foundation; numpy/scipy/pandas/pydantic)
```

```{toctree}
:maxdepth: 1
:caption: Reference

api/index
```

```{toctree}
:maxdepth: 1
:caption: Plan
:glob:

plan/README
plan/0*
```

```{toctree}
:maxdepth: 1
:caption: Notes
:glob:

notes/README
notes/0*
```
