# viz — optional figures

viz: optional plotly figures behind the `[viz]` extra.

Importing this package never imports plotly; each figure function returns
`Unsupported(reason="plotly not installed", missing=("viz",))` when it is
absent. See `nbs/viz/`.

`causal_graph` draws the graph itself — a causal package whose central
object had no picture was asking a reader to hold a DAG in their head while
reading a verdict about it. Unmeasured nodes are hollow and bidirected edges
dashed, because those two are what separate a graph you can identify from one
you cannot. `stability` draws what a resample settled and what it did not.

Every subpackage that returns a result now has a figure of it: `convergence`
for `infer`, `boundary` for `design`, `corrections` for `calibrate`,
`panel_coverage` for `data`, `intervals` for `core`, `recovery` for `sim`,
`unrolled` for `dynamics` and `transfer` for `estimands`. The subpackages
without one — `io`, `build`, `adapters`, `report`, `display` — return
plumbing rather than findings, and a chart of a file format is decoration.

## Notebooks

The executed series under `nbs/viz/`:

- [01-figures.ipynb](../../nbs/viz/01-figures.ipynb)

## Package

```{eval-rst}
.. automodule:: axiom.viz
   :no-members:
```

## Modules

```{eval-rst}
.. automodule:: axiom.viz.figures
   :members:
```
