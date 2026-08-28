# io — serialization, provenance, registry, scopes, experiment runs

io: the `analysis.axiom` format, provenance, a content-addressed registry, the
scopes artifacts belong to, and the life of one experiment.

No pickle, cloudpickle, or dill anywhere in this package; gate 5 asserts it.

## Notebooks

The executed series under `nbs/io/`:

- [01-save-load-an-analysis.ipynb](../../nbs/io/01-save-load-an-analysis.ipynb)
- [02-many-parties-one-catalog.ipynb](../../nbs/io/02-many-parties-one-catalog.ipynb)

## Package

```{eval-rst}
.. automodule:: axiom.io
   :no-members:
```

## Modules

```{eval-rst}
.. automodule:: axiom.io.analysis
   :members:

.. automodule:: axiom.io.catalog
   :members:

.. automodule:: axiom.io.definitions
   :members:

.. automodule:: axiom.io.experiment
   :members:

.. automodule:: axiom.io.provenance
   :members:

.. automodule:: axiom.io.registry
   :members:

.. automodule:: axiom.io.serialize
   :members:
```
