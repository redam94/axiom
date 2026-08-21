# io — serialization, provenance, registry

io: the `analysis.axiom` format, provenance, and a content-addressed registry.

No pickle, cloudpickle, or dill anywhere in this package; gate 5 asserts it.

## Notebooks

The executed series under `nbs/io/`:

- [01-save-load-an-analysis.ipynb](../../nbs/io/01-save-load-an-analysis.ipynb)

## Package

```{eval-rst}
.. automodule:: axiom.io
   :no-members:
```

## Modules

```{eval-rst}
.. automodule:: axiom.io.analysis
   :members:

.. automodule:: axiom.io.provenance
   :members:

.. automodule:: axiom.io.registry
   :members:

.. automodule:: axiom.io.serialize
   :members:
```
