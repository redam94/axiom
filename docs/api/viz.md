# viz — optional figures

viz: optional plotly figures behind the `[viz]` extra.

Importing this package never imports plotly; each figure function returns
`Unsupported(reason="plotly not installed", missing=("viz",))` when it is
absent. See `nbs/viz/`.

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
