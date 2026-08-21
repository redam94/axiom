# axiom

**Bayesian causal decision science.** Declare what you want to know, find out
whether the data can tell you, design the experiment that would, fold the
experiment's answer back into the model, map the response surface it implies,
and pool the evidence across every study you have run.

Four pillars, one vocabulary:

| Pillar | Question it answers | Package |
|---|---|---|
| **Causal modeling** | Is this effect identified, and from what? What would a hidden confounder have to look like to overturn it? | `axiom.identify`, `axiom.estimands`, `axiom.diagnose` |
| **Experimental design & planning** | What is worth measuring next, at what size, by which method, at what cost? | `axiom.design` |
| **Experimental calibration** | How does a randomized result update an observational model, and what did that transfer assume? | `axiom.calibrate` |
| **Response-surface methodology** | What does the dose–response surface look like, where is its optimum, and where should I probe next? | `axiom.surface` |
| **Meta-analysis** | What does the whole body of evidence say, and how heterogeneous is it? | `axiom.meta` |

## Start here

Every public symbol is demonstrated, executed, in a notebook. The notebook
series live under `nbs/`, one directory per subpackage, numbered
in reading order:

- Foundation: `nbs/core/`, `nbs/data/`, `nbs/io/`, `nbs/infer/`
- Domain: `nbs/identify/`, `nbs/estimands/`, `nbs/surface/`, `nbs/sim/`
- Pillars: `nbs/design/`, `nbs/calibrate/`, `nbs/meta/`
- Composition: `nbs/diagnose/`, `nbs/build/`, `nbs/adapters/`, `nbs/viz/`

Each [API page](api/index.md) links its own series.

For what using all of it at once looks like, read the **case study** under
`nbs/case-studies/hypertension/`: HYPER-3, a sequential dose-finding trial that
must stop a dose arm early if it is harming people. Six notebooks, one synthetic
world, from the causal graph to the boundary crossing that stops an arm at the
first safety review. Its design notes are
[note 0004](notes/0004-case-study-hypertension.md).

## Layering

Imports point down only; `tests/contracts/test_layering.py` enforces it.

```text
  viz    adapters                          (leaves; nothing imports them)
   |        |
  build   diagnose                         (compose everything below)
   |        |
  meta  calibrate  design                  (the four pillars; peers, no cross-imports
   |        |        |                      except design <- surface, calibrate <- estimands)
   +--------+--------+
            |
        surface   estimands   identify     (domain layer)
            |         |          |
            +---------+----------+
                      |
                    infer                  (sampler seam; optional deps live here)
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
