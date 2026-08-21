# build — fluent builders

build: fluent, immutable builders over every `Spec`.

A builder is a frozen dataclass holding a `Fields` mapping; every fluent
method returns a new builder and `build()` returns a `Spec` that
round-trips through `load_spec`. See `nbs/build/`.

## Notebooks

The executed series under `nbs/build/`:

- [01-builders.ipynb](../../nbs/build/01-builders.ipynb)

## Package

```{eval-rst}
.. automodule:: axiom.build
   :no-members:
```

## Modules

```{eval-rst}
.. automodule:: axiom.build.base
   :members:

.. automodule:: axiom.build.graph
   :members:

.. automodule:: axiom.build.meta
   :members:

.. automodule:: axiom.build.prior
   :members:

.. automodule:: axiom.build.study
   :members:

.. automodule:: axiom.build.surface
   :members:

.. automodule:: axiom.build.variable
   :members:
```
