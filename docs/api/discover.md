# discover — what the data can orient

discover: essential graphs, greedy equivalence search, and the price of an
experiment in edges.

Observational data cannot identify a DAG — Markov-equivalent DAGs entail the
same independencies, so nothing in the data separates them. What is identified
is the *class*, and an essential graph is it: directed where every member
agrees, undirected where they do not. Interventions break the remaining ties,
and `orientation_gain` says which ones a proposed experiment would break before
it is run.

Three modules qualify that answer. `independence` is the conditional-independence
test the rest of the subpackage tests *with*. `stability` resamples the rows and
re-runs the search, so each edge carries how often it came back rather than only
whether it is in the one graph that was returned. `fci` drops the assumption the
others make — that every common cause was measured — and returns a PAG, whose
bidirected edge is the statement "something you did not measure drives both",
which no CPDAG can express.

Design notes: [0013](../notes/0013-discovery-and-granularity.md),
[0014](../notes/0014-refutation-stability-and-latents.md).

## Notebooks

The executed series under `nbs/discover/`:

- [01-what-the-data-can-orient.ipynb](../../nbs/discover/01-what-the-data-can-orient.ipynb)
- [02-stability-and-hidden-causes.ipynb](../../nbs/discover/02-stability-and-hidden-causes.ipynb)

## Package

```{eval-rst}
.. automodule:: axiom.discover
   :no-members:
```

## Modules

```{eval-rst}
.. automodule:: axiom.discover.essential
   :members:

.. automodule:: axiom.discover.fci
   :members:

.. automodule:: axiom.discover.independence
   :members:

.. automodule:: axiom.discover.score
   :members:

.. automodule:: axiom.discover.search
   :members:

.. automodule:: axiom.discover.stability
   :members:
```
