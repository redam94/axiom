# surface — dose-response kernels, designs, optimization

surface: dose-response kernels, carryover, nuisance terms, designs, ascent.

Every kernel is built from `axiom.core` expression nodes and evaluated by
the interpreters — there is one `forward()`. No sampler is imported here;
fitting goes through `axiom.infer`.

## Notebooks

The executed series under `nbs/surface/`:

- [01-kernels-and-carryover.ipynb](../../nbs/surface/01-kernels-and-carryover.ipynb)
- [02-forward-and-linearize.ipynb](../../nbs/surface/02-forward-and-linearize.ipynb)
- [03-designs.ipynb](../../nbs/surface/03-designs.ipynb)
- [04-fit.ipynb](../../nbs/surface/04-fit.ipynb)
- [05-ascent-and-optimize.ipynb](../../nbs/surface/05-ascent-and-optimize.ipynb)

## Package

```{eval-rst}
.. automodule:: axiom.surface
   :no-members:
```

## Modules

```{eval-rst}
.. automodule:: axiom.surface.ascent
   :members:

.. automodule:: axiom.surface.carryover
   :members:

.. automodule:: axiom.surface.design
   :members:

.. automodule:: axiom.surface.forward
   :members:

.. automodule:: axiom.surface.frontier
   :members:

.. automodule:: axiom.surface.kernels
   :members:

.. automodule:: axiom.surface.linearize
   :members:

.. automodule:: axiom.surface.model
   :members:

.. automodule:: axiom.surface.nuisance
   :members:

.. automodule:: axiom.surface.optimize
   :members:
```
