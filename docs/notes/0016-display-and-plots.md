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

## D16.3 — One figure per subpackage that returns a result

The two above left nine subpackages without a picture, which was the wrong
place to stop. Eight more, each drawing what its subpackage actually returns:

| subpackage | figure | what it answers |
|---|---|---|
| `infer` | `convergence` | may I use this sampler's output? |
| `design` | `boundary` | when did we promise to stop? |
| `calibrate` | `corrections` | which correction moved the number? |
| `data` | `panel_coverage` | where are the gaps? |
| `core` | `intervals` | do these overlap? |
| `sim` | `recovery` | did the estimator find what was there? |
| `dynamics` | `unrolled` | does feedback really become a DAG? |
| `estimands` | `transfer` | what does moving this result cost? |

`io`, `build`, `adapters`, `report` and `display` get none, and that is the
honest answer rather than a gap: they return plumbing, and a chart of a file
format is decoration.

Two decisions worth keeping. **`boundary` puts the alpha spent in the hover
rather than on a second axis** — two scales on one plot is the chart mistake
this package refuses everywhere else, and a stopping rule is one scale. And
**`corrections` is a waterfall**, because the useful question about a
calibrated estimate is never "what is the answer" but "which correction moved
it", which the corrected number alone cannot answer.

`causal_graph` and `unrolled` share one layout function. That was not tidiness:
an unrolled system's *depth is time*, so the same layered algorithm puts the
periods across the page with no parsing of column names, and a cycle that
survived the unroll shows as a node pushed to the far right rather than hidden.

**D16.3a — a test caught the figure drawing half the graph.** `Unrolled.columns`
holds the exogenous data columns only; the solved variables are the keys of
`expressions`. Drawing `columns` alone left every endogenous node out and every
remaining node a root, so the whole system rendered at depth zero — a picture of
time with no time in it, which looked plausible enough to ship.

## What is still open

* **`meta` and `surface` have one figure each and deserve more.** `meta` has
  heterogeneity and shrinkage; `surface` has more than one curve.
* **The card has no table row.** A `StabilityReport` with forty edges renders
  forty rows; a real table type would render it as a table and truncate.
* **`enable()` registers `text/plain` only.** A notebook could have an HTML
  rendering with the same content, and a `text/html` formatter is where that
  would go.
* **Fifty of the sixty-one types still use the generic card.** It is decent and
  it is not what a reader of a `Boundary` or an `EVOIResult` deserves.

## D16.4 — The notebooks, and why this was not a `print` → `show` sweep

Rules 6 and 12 say every public symbol appears in an executed notebook, so the
notebooks are where a reader meets this. The obvious move — rewrite every
`print` as a `show` — is wrong, and the survey is why.

Across 77 notebooks there are **1735 `print` calls, of which 31 have the shape
`print(<a single name>)`.** The other 1704 are f-strings: `print(f"rhat: {row.rhat:.3f}")`
and its kin. Those are not lazy reprs waiting to be upgraded; they are a chosen
sentence about one number, and a card is a worse rendering of a sentence. Of the
31, most bind a string, a content hash, a `DataFrame`, or an arviz object, and
`show("...")` renders a card titled `str` — strictly worse than the print it
replaced. **Sixteen prints, in twelve notebooks, were converting a result that
axiom has a renderer for. Those are the ones that changed.**

The actual gap the survey exposed was not formatting at all: **72 of 77
notebooks drew no figure**. A package that had just added a figure per
subpackage was demonstrating almost none of them. So each subpackage series
gained a closing section that shows a result as a card and draws that
subpackage's figure — `convergence` for `infer`, `boundary` for `design`,
`causal_graph` for `identify`, `stability` for `discover`, `panel_coverage` for
`data`, `intervals` for `core`, `recovery` for `sim`, `unrolled` for `dynamics`,
`corrections` for `calibrate`, `transfer` for `estimands`.

Each section is **self-contained** — it imports and builds what it draws. That
is deliberate: a section that reached back for a variable defined eleven cells
earlier would break the next time someone edited the middle of the notebook, and
the point of these is to be the part a reader can lift out and run.

`io`, `build`, `adapters` and `report` get no figure section, for the reason
D16.3 already gives: they return plumbing.

**A run caught one.** `ConvergenceReport` has no `.verdict` — the fields are
`rows`, `converged`, `thresholds` and the rest — so the `infer` section raised
on execution. The notebook now shows the report itself, which is the better
card anyway. That is the argument for rule 6 in one line: nothing else in the
suite would have found it, because nothing else calls the API the way a reader
does.

## D16.5 — Three types rendered *worse* than `print`, and only looking found it

Converting sixteen prints to cards should have been mechanical. Rendering the
result of each conversion and reading it beside the `print` it replaced found
three types where the card was worse than what it replaced:

| type | `print` said | the card said |
|---|---|---|
| `Posterior` | `names=['mu'], chains=4, draws=500` | `Posterior` |
| `Panel` | `units=2, periods=2, rows=3, balanced=False` | `Panel` |
| `CausalGraph` | `X -> Y, Z -> X, Z -> Y` | `nodes … edges  X, Y, Z, X, Z, Y` |

Two causes, and both were predictable from D16.1b if anyone had checked.
`Posterior` and `Panel` **are not `Spec` subclasses** — they are hand-written
containers — so the generic renderer, which builds a card from a model's own
fields, had no fields to walk and produced a card with a title and nothing else.
`CausalGraph` *is* a `Spec`, but its `edges` are bare 2-tuples, so the generic
collection shortener flattened `X -> Y, Z -> X` into `X, Y, Z, X`. **Every arrow
gone**, and a causal graph rendered as an unordered node list — the one thing a
causal graph must never be.

The third is the worst of them, because it is not empty. An empty card announces
itself; a card confidently listing six node names *looks* like a rendering that
worked.

All three now have renderers, and the tests that cover them assert the thing that
was actually wrong — that the card has more than one line, that `->` survives,
that the bidirected edge which decides identifiability is present. `Posterior`
sorts `names()` before rendering, for the reason note 0015 gives about the
agriculture example: it is a `frozenset`, and an unsorted one reorders per run.

**Nothing else in the suite would have caught these.** Every display test passed
throughout; the generic renderer did exactly what it was written to do. The check
that found it was rendering the output and reading it, which is the same check
that found the overlapping PDF tables and the duplicated section headings.

One vocabulary note. Both new cards wanted a "warning" status and there isn't
one: `Status` is `good | assumed | bad | neutral`, meaning *it holds*, *it is
taken on trust*, *it failed*. An unbalanced panel is none of those — it is a
fact with consequences — and graph feedback is not a failure at all, since
`dynamics` exists to handle it. Rather than stretch `assumed` to mean "careful",
both stay `neutral` and the note carries the fact. A vocabulary that means
something is worth more than a mark on every card.

## D16.6 — Two more, caught by executing all seventy-seven

The unit tests were green and the ten edited series passed individually. The
full `make notebooks` run failed twice, and both were mine.

**`roles.treatments` is a `dict`.** The `Panel` renderer read it as pairs —
`for name, _ in obj.roles.treatments` — which iterates the *keys* and tries to
unpack a string. It never fired in the unit test because that panel had no
treatments, so the loop body never ran. A test that exercises the empty case
and calls it covered is the shape of this mistake, and there is now a second
test with a treatment in it.

**An import that arrived after its use.** The `print` → `show` conversion
inserted `from axiom.display import show` into the first cell it changed, unless
the notebook imported from `axiom.display` *somewhere* already. In
`nbs/sim/02-surface-worlds.ipynb` it did — in the closing section appended to
the end, eleven cells *after* the `show` that needed it. The guard asked "does
this notebook import it" when the question is "does it import it **before
here**". Cells run in order and a whole-file grep does not know that.

Both are the same lesson as D16.5 from a different angle: the check that finds
these is running the thing end to end, in order, the way a reader will.

While fixing the second I replaced a `show({"recovered": len(truth)})` that had
crept into the `sim` section — a dict of my own making, rendered to prove the
renderer runs. That is a demonstration of nothing. It now shows the panel
`arms_world` actually returns, which exercises the treatments line that had just
been wrong.

## D16.7 — One notebook had never run, because of its directory's name

Reconciling "77 notebooks on disk" against "76 collected" turned up a gap that
predates all of this work: **`nbs/build/01-builders.ipynb` had never been
executed by `make notebooks`.**

pytest's default `norecursedirs` contains `build`. The project set `testpaths`
but never `norecursedirs`, so the default applied, and the `build` subpackage's
notebook series was skipped every run since it was written. Nothing reported it.
A skipped directory and a directory with nothing in it produce identical output,
which is why this survived: the suite was green the whole time, and green was
the wrong signal.

Gate 12 did not catch it either, and could not have. It checks *statically* that
every exported symbol appears in a notebook — and it does appear. Whether that
notebook ever ran is a different question, and nothing was asking it.

`norecursedirs` is now set explicitly, all 77 collect, and the builders notebook
passes. The rule worth extracting: **a subpackage's name must not be able to
decide whether its deliverable is tested.** `axiom` has a subpackage called
`build`; it is a coincidence that this collided with a pytest default, and a
coincidence should not be load-bearing.

Worth a follow-up: gate 12 asserts a symbol is *mentioned* in a notebook. A
gate that also asserted the notebook's series is *collected* by the runner would
have caught this on the day it appeared.
