# estimands — declarative counterfactual quantities

estimands: declarative, content-hashed counterfactual quantities.

The declaration half (facets, derived dimension, `transfer_to`) and the
realization half (`realize` against any `SupportsEstimands` producer,
the standard registry, and the estimand as an expression tree). Imports
`core` only.

## Notebooks

The executed series under `nbs/estimands/`:

- [01-declaring-an-estimand.ipynb](../../nbs/estimands/01-declaring-an-estimand.ipynb)
- [02-transfer-plans.ipynb](../../nbs/estimands/02-transfer-plans.ipynb)
- [03-realization.ipynb](../../nbs/estimands/03-realization.ipynb)

## Package

```{eval-rst}
.. automodule:: axiom.estimands
   :no-members:
```

## Modules

```{eval-rst}
.. automodule:: axiom.estimands.evaluate
   :members:

.. automodule:: axiom.estimands.graph
   :members:

.. automodule:: axiom.estimands.registry
   :members:

.. automodule:: axiom.estimands.spec
   :members:
```
