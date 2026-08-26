# 0025 — A notebook is not a paler medium than a terminal

*Kind: decision + progress. Opened 2026-08-26. Status: implemented on `feature/notebook-display`.*

`axiom.display` was written so a result would stop printing as a pydantic repr
(note 0016). Its docstring promised "a panel with a colour for its status" and
its notebook entry point, `enable()`, registered exactly one formatter:
`text/plain`, backed by `render` — the form that exists *because* it is
deliberately unstyled and testable. So every notebook that called `enable()` got
aligned monospace and never a rich panel, `print(latex(hill))` printed TeX
source, and the eighty-four notebooks between them printed two thousand lines
through `print`.

## D25.1 — three formatters, in the order a front-end prefers them

`enable()` registers `text/latex` for the model tree, `text/html` for every
other result, and `text/plain` under both. A front-end offered several takes the
richest, so the ordering is the whole design:

- an expression, `Equation`, `System`, `ODESystem` or `ModelSpec` is
  mathematics, and is *set* as mathematics;
- everything else is a card, drawn by rich and exported to self-contained HTML;
- `text/plain` remains for a diff, a log, an nbconvert to text, and a front-end
  with neither — and it is still the string the tests assert on.

`plain=True` — a keyword that until now was accepted and ignored — registers
only the last. Without rich installed the HTML formatter is skipped and the
same thing happens, which is the property the dependency budget is protecting.

The HTML formatter returns `None` for the math types rather than skipping their
registration, because the catch-all is registered on `Spec` and an expression
node is a `Spec`: without the `None` the front-end would take the HTML card and
the typeset form would never be reached.

One trap worth recording. rich's `Console` detects a notebook by itself, and a
recording console built to export HTML *publishes the panel to the notebook*
instead of recording it — the buffer comes back empty and the reader gets a
second, unstyled copy. `force_jupyter=False` on that console is load-bearing;
a test pins it.

## D25.2 — what the tree gets, and what it refuses

Renderers for every `Expr` node, `Equation`, `System`, `ODESystem` and
`ModelSpec`: the algebra, its dimension, its parameters, what it reads. A tree
that fails its own dimension check reports that on the card rather than raising
out of a formatter, which a notebook would render as a traceback over the cell
that merely *displayed* it.

`show_math` sets an expression on demand and prints TeX outside a notebook. A
tree with an `Opaque` node cannot be written down faithfully, so it shows the
typed refusal as a card: a rendering that quietly dropped the part it could not
read would be worse than no rendering.

`_MATH_TYPES` is read off the `Expr` union, so a fourteenth node cannot silently
lose its typeset rendering.

## D25.3 — `table` / `render_table`, in the same relationship as `show` / `render`

Two hundred and twelve cells printed one formatted line per row of something and
left the reader to line the columns up. Most of them printed a hand-built header
above the loop — a table admitting what it is. `table` draws a rich table where
rich is installed and aligned text with numeric columns right-justified where it
is not; `render_table` gives that text directly.

It lives in `display` rather than in each notebook importing `rich`, for the
reason the rest of the subpackage exists: `make notebooks` must not require an
extra, and the content must be identical either way so a test can assert on it.

## D25.4 — what stayed a `print`

Eleven loops. Their output is hierarchical, not tabular: a transfer plan's
entries nested under its target, a stopping rule's drift scenarios each with two
lines of crossing probabilities, a resolved report's blocks under their section.
Flattening those into rows would lose the shape the reader is being shown. Where
a loop's *inner* list was tabular it became a table under the printed group
label, which is the honest decomposition.

Ten DataFrames printed through `.to_string()` now go through IPython's `display`,
which is Jupyter's own styled table and a better rendering of a frame than
anything this layer would invent.

## What it cost elsewhere

Four notebooks bound a local named `table` — three DataFrames and one tuple
unpack — which shadowed the import from that cell onward. Renamed to what they
hold. mypy grew one narrow allowance, `untyped_calls_exclude = ["IPython"]`, for
the two untyped display helpers `show_math` calls.
