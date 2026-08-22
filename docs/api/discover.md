# discover — what the data can orient

discover: essential graphs, greedy equivalence search, and the price of an
experiment in edges.

Observational data cannot identify a DAG — Markov-equivalent DAGs entail the
same independencies, so nothing in the data separates them. What is identified
is the *class*, and an essential graph is it: directed where every member
agrees, undirected where they do not. Interventions break the remaining ties,
and `orientation_gain` says which ones a proposed experiment would break before
it is run.

Design notes: [0013](../notes/0013-discovery-and-granularity.md).

## Notebooks

The executed series under `nbs/discover/`:

- [01-what-the-data-can-orient.ipynb](../../nbs/discover/01-what-the-data-can-orient.ipynb)

## Package

```{eval-rst}
.. automodule:: axiom.discover
   :no-members:
```

## Modules

```{eval-rst}
.. automodule:: axiom.discover.essential
   :members:

.. automodule:: axiom.discover.score
   :members:

.. automodule:: axiom.discover.search
   :members:
```
