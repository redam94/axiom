# 0016 — Showing a result, and drawing the structure

*Kind: decision. Opened 2026-08-23. Status: implemented on `feature/display`; part of the 1.1.0 surface.*

Two gaps, both about the same thing: axiom computed carefully and communicated
badly. Sixty-one public types report an outcome and not one of them rendered;
`viz` could draw a response curve and a forest plot but nothing about the
structure the package is *for*.

## D16.1 — `axiom.display`: a result is read, so it has to be readable

Before this, every result printed as a pydantic repr — three hundred characters
of field names on one line with the number a reader wanted somewhere in the
middle. That is not a cosmetic complaint. **A result that is hard to read is a
result that gets skimmed, and a skimmed interval is a point estimate**, which is
the exact failure the rest of this codebase is built to prevent.

**D16.1a — the dependency budget held, and it cost nothing to hold it.** The
obvious implementation makes `rich` a dependency. Measured against the four-package
core: `rich` imports in 0.018 s where pandas takes 0.407 s, and adds 5.9 MB to a
~160 MB core. On the merits it is nothing like the samplers the budget exists to
exclude.

It is still an extra. `axiom.display` renders a `Card` — a title, a status, some
rows, a note — and two back-ends draw it: plain text always, a rich panel when
rich is importable. The content is identical, which is what makes the layer
testable (tests assert on the card and on plain text, never on escape codes),
and `import axiom` still needs four packages. The fallback is real rather than
nominal: the plain form is the one the tests use.

**D16.1b — the generic renderer is the feature.** There are sixty-one result
types and a dozen worth hand-writing. Hand-writing sixty-one is a way of
shipping thirty, so every `Spec` gets a card built from its own fields and the
dozen a reader actually looks at get one that knows what the fields mean. That
is why this improved sixty-one types in one commit rather than twelve.

**D16.1c — registration, not methods.** A `Verdict` lives in `core` and display
lives at layer 7; the arrow has to point this way. `@renders(Type)` registers a
renderer, `card_for` walks the MRO, and a result type gains a good rendering
without gaining a dependency. Anything downstream can register one for its own
types.

**D16.1d — notebooks get it without a `show`.** `enable()` registers a formatter
with IPython rather than patching `_repr_html_` onto the types, for the same
layering reason: nothing in `core` learns that a display layer exists.

## D16.2 — Two plots about structure

`viz` had eight figures, all about an estimate or a diagnostic, and nothing for
`identify`, `discover` or `design`.

**`causal_graph`** draws the graph. A causal package whose central object had no
picture was asking a reader to hold a DAG in their head while reading a verdict
about it. Layered rather than force-directed — depth is the longest path from a
root, so every arrow points forwards and none doubles back, which is what makes
the picture readable without studying arrowheads one at a time. Unmeasured nodes
are hollow and bidirected edges dashed, because those two are exactly what
separate a graph you can identify from one you cannot.

**`stability`** draws what a resample settled and what it did not, split three
ways. The split is the finding: a bar that is nearly all *undirected* is a stable
edge whose direction observation cannot settle — more rows will not move it and
only an intervention will — while a short bar is an edge the data is unsure about
at all. One number would lose the distinction that decides what to do next.

## What is still open

* **Nine subpackages still have no plot.** `design` has boundaries and power
  curves, `meta` has more than a forest, `surface` has more than one curve. The
  two added here are the two that were most conspicuously missing, not a
  complete set.
* **The card has no table row.** A `StabilityReport` with forty edges renders
  forty rows; a real table type would render it as a table and truncate.
* **`enable()` registers `text/plain` only.** A notebook could have an HTML
  rendering with the same content, and a `text/html` formatter is where that
  would go.
* **Fifty of the sixty-one types still use the generic card.** It is decent and
  it is not what a reader of a `Boundary` or an `EVOIResult` deserves.
