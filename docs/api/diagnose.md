# diagnose — trust machinery

Trust machinery: SBC, coverage, posterior predictive and residual checks, sensitivity to
unobserved confounding, weak identification, prior-to-posterior learning, specification curves,
refutation, and rolling-origin backtests.

`structure` refutes the *graph* rather than the estimate: a DAG entails
conditional independencies, each one is a testable claim, and a claim the data
contradicts is a claim the graph got wrong. Surviving is not passing — a test
that fails to reject with 200 rows is weak evidence, and the module reports the
distinction rather than hiding it.

See `docs/plan/03-roadmap.md` (Phase 8), `nbs/diagnose/`, and design note
[0014](../notes/0014-refutation-stability-and-latents.md).

## Notebooks

The executed series under `nbs/diagnose/`:

- [01-sbc-and-coverage.ipynb](../../nbs/diagnose/01-sbc-and-coverage.ipynb)
- [02-sensitivity.ipynb](../../nbs/diagnose/02-sensitivity.ipynb)
- [03-learning-and-spec-curve.ipynb](../../nbs/diagnose/03-learning-and-spec-curve.ipynb)
- [04-refute-and-backtest.ipynb](../../nbs/diagnose/04-refute-and-backtest.ipynb)
- [05-ppc-and-residuals.ipynb](../../nbs/diagnose/05-ppc-and-residuals.ipynb)
- [06-refuting-the-graph.ipynb](../../nbs/diagnose/06-refuting-the-graph.ipynb)

## Package

```{eval-rst}
.. automodule:: axiom.diagnose
   :no-members:
```

## Modules

```{eval-rst}
.. automodule:: axiom.diagnose.backtest
   :members:

.. automodule:: axiom.diagnose.coverage
   :members:

.. automodule:: axiom.diagnose.learning
   :members:

.. automodule:: axiom.diagnose.ppc
   :members:

.. automodule:: axiom.diagnose.refute
   :members:

.. automodule:: axiom.diagnose.residuals
   :members:

.. automodule:: axiom.diagnose.sbc
   :members:

.. automodule:: axiom.diagnose.sensitivity
   :members:

.. automodule:: axiom.diagnose.spec_curve
   :members:

.. automodule:: axiom.diagnose.structure
   :members:

.. automodule:: axiom.diagnose.surface_prior
   :members:

.. automodule:: axiom.diagnose.weak_id
   :members:
```
