# sim — worlds with known ground truth

sim: data-generating processes with known causal ground truth.

Every recovery test in the suite is written against these worlds. Phase 2
ships the linear-Gaussian SCM; surface and panel worlds arrive in Phase 3.

## Notebooks

The executed series under `nbs/sim/`:

- [01-linear-scm-worlds.ipynb](../../nbs/sim/01-linear-scm-worlds.ipynb)
- [02-surface-worlds.ipynb](../../nbs/sim/02-surface-worlds.ipynb)

## Package

```{eval-rst}
.. automodule:: axiom.sim
   :no-members:
```

## Modules

```{eval-rst}
.. automodule:: axiom.sim.panel
   :members:

.. automodule:: axiom.sim.scm
   :members:

.. automodule:: axiom.sim.surface_world
   :members:

.. automodule:: axiom.sim.worlds
   :members:
```
