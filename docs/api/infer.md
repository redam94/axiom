# infer — the sampler seam

infer: the sampler seam — Backend protocol, Laplace, NumPyro NUTS, PyMC NUTS
(over its own sampler, nutpie, numpyro or blackjax), and diagnostics.

Importing this package never pulls jax, numpyro, pymc, pytensor or arviz into
`sys.modules`; backends import them lazily and `get_backend` returns a
typed `Unsupported` when an extra is missing.

## Notebooks

The executed series under `nbs/infer/`:

- [01-backends.ipynb](../../nbs/infer/01-backends.ipynb)
- [02-laplace.ipynb](../../nbs/infer/02-laplace.ipynb)
- [03-convergence.ipynb](../../nbs/infer/03-convergence.ipynb)

## Package

```{eval-rst}
.. automodule:: axiom.infer
   :no-members:
```

## Modules

```{eval-rst}
.. automodule:: axiom.infer.backend
   :members:

.. automodule:: axiom.infer.diagnostics
   :members:

.. automodule:: axiom.infer.laplace
   :members:

.. automodule:: axiom.infer.numpyro_backend
   :members:

.. automodule:: axiom.infer.pymc_backend
   :members:
```
