# dynamics — systems that are not DAGs

dynamics: simultaneous systems, time-structured systems, and the compiler that
turns both into ordinary `axiom.core.expr` trees.

A causal DAG is a *solved* model. Write down what you actually believe about a
market, a dose regimen, or an organism and you often get something else: two
quantities determined together within a period, and yesterday's value of a
quantity entering today's equation. Neither is ill-posed; both are cyclic as a
graph over variable names.

`axiom.dynamics` takes such a system as a declarative `Spec` and **compiles**
it — into expression trees, and into a time-indexed graph that is acyclic and
can be handed straight to `axiom.identify`. Nothing is added to the evaluator,
so `value`, the jax and PyTensor interpreters, `dimension`, `latex`,
`linearize` and the design math all work on a compiled system with no special
case: there is still exactly one `forward()`.

Design notes: [0010](../notes/0010-dynamic-systems.md).

## Notebooks

The executed series under `nbs/dynamics/`:

- [01-systems-and-blocks.ipynb](../../nbs/dynamics/01-systems-and-blocks.ipynb)
- [02-unrolling-and-fitting.ipynb](../../nbs/dynamics/02-unrolling-and-fitting.ipynb)

Identification on the unrolled graph is
[nbs/identify/04-dynamic-systems.ipynb](../../nbs/identify/04-dynamic-systems.ipynb).

## Package

```{eval-rst}
.. automodule:: axiom.dynamics
   :no-members:
```

## Modules

```{eval-rst}
.. automodule:: axiom.dynamics.spec
   :members:

.. automodule:: axiom.dynamics.parse
   :members:

.. automodule:: axiom.dynamics.blocks
   :members:

.. automodule:: axiom.dynamics.solve
   :members:

.. automodule:: axiom.dynamics.unroll
   :members:

.. automodule:: axiom.dynamics.algebra
   :members:
```
