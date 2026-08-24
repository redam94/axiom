# core — vocabulary, dimensions, specs, protocols, verdicts, intervals

core: vocabulary, dimensions, specs, protocols, verdicts, and intervals.

The foundation layer. Imports nothing else from axiom; depends only on
numpy, scipy, and pydantic. Everything here is a `Spec` or a protocol.

## Notebooks

The executed series under `nbs/core/`:

- [01-dimensions.ipynb](../../nbs/core/01-dimensions.ipynb)
- [02-specs-and-hashing.ipynb](../../nbs/core/02-specs-and-hashing.ipynb)
- [03-expression-tree.ipynb](../../nbs/core/03-expression-tree.ipynb)
- [04-protocols-and-posterior.ipynb](../../nbs/core/04-protocols-and-posterior.ipynb)
- [05-intervals-verdicts-and-failures.ipynb](../../nbs/core/05-intervals-verdicts-and-failures.ipynb)
- [06-entities.ipynb](../../nbs/core/06-entities.ipynb)
- [07-model-spec-and-jax.ipynb](../../nbs/core/07-model-spec-and-jax.ipynb)

## Package

```{eval-rst}
.. automodule:: axiom.core
   :no-members:
```

## Modules

```{eval-rst}
.. automodule:: axiom.core.dimensions
   :members:

.. automodule:: axiom.core.entities
   :members:

.. automodule:: axiom.core.expr
   :members:

.. automodule:: axiom.core.interpret
   :no-members:

.. automodule:: axiom.core.interpret.dimension
   :members:

.. automodule:: axiom.core.interpret.jax
   :members:

.. automodule:: axiom.core.interpret.latex
   :members:

.. automodule:: axiom.core.interpret.pytensor
   :members:

.. automodule:: axiom.core.interpret.value
   :members:

.. automodule:: axiom.core.intervals
   :members:

.. automodule:: axiom.core.model
   :members:

.. automodule:: axiom.core.posterior
   :members:

.. automodule:: axiom.core.protocols
   :members:

.. automodule:: axiom.core.result
   :members:

.. automodule:: axiom.core.rounding
   :members:

.. automodule:: axiom.core.spec
   :members:

.. automodule:: axiom.core.stats
   :members:

.. automodule:: axiom.core.verdict
   :members:
```
