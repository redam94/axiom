# display — showing a result

display: cards for every result axiom returns, in a terminal or a notebook.

Sixty-one public types in axiom report an outcome, and until this subpackage
existed every one of them printed as a pydantic repr: three hundred characters
of field names on one line, with the number a reader wanted somewhere in the
middle. A result that is hard to read is a result that gets skimmed, and a
skimmed interval is a point estimate.

`show` prints a card. `render` gives the same content as plain text, which is
what the tests assert on and what a log file wants. `enable()` registers a
formatter with IPython so a notebook renders every result without `show` in
front of it.

**rich is an extra, not a dependency.** With it a card is a panel coloured by
its status; without it the same card is aligned plain text with an ASCII mark.
`axiom` still imports with four dependencies, which is what the budget exists
to protect — `rich` costs 0.018 s to import against pandas' 0.407 s, but the
rule is the rule and the fallback is real rather than nominal.

A renderer is registered by type rather than written as a method, because a
`Verdict` lives in `core` and this lives at the top of the stack. The generic
fallback matters more than the specific renderers: it is why adding this layer
improved sixty-one types rather than the dozen with a renderer of their own.

The fallback has two blind spots, and both are worth knowing before relying on
it for a new type. It builds a card from a **`Spec`'s own fields**, so a
hand-written container that is not a `Spec` — `Posterior` and `Panel` were both
— renders as a title and nothing else. And it shortens collections by joining
their elements, so a field of bare tuples loses its structure: `CausalGraph`'s
edges rendered as `X, Y, Z, X` with every arrow gone. All three now have
renderers of their own; a new type shaped like any of them will want one too.

Design notes: [0016](../notes/0016-display-and-plots.md).

## Notebooks

The executed series under `nbs/display/`:

- [01-showing-results.ipynb](../../nbs/display/01-showing-results.ipynb)

## Package

```{eval-rst}
.. automodule:: axiom.display
   :no-members:
```

## Modules

```{eval-rst}
.. automodule:: axiom.display.card
   :members:

.. automodule:: axiom.display.renderers
   :members:
```
