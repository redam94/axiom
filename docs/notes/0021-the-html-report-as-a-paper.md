# 0021 — The HTML report is laid out as a paper, not as a page of defaults

*Kind: decision. Opened 2026-08-23. Status: implemented; `axiom/report/render/html.py`.*

The HTML renderer produced correct documents that nobody wanted to read: one
column as wide as the window, every table a grid of boxes, metrics as a bare
flex row, and prose set at whatever measure the viewport happened to be. The
content was already a journal article — `axiom_dossier.journal` gives it the
abstract, the numbered sections and the serif — and the stylesheet was not.
This note records what the stylesheet now does and why, because it is the one
part of the report system whose output is judged by eye.

The reference is the LaTeXML rendering arXiv serves for HTML papers: a fixed
measure, a contents rail in the margin, captions where a journal puts them,
and horizontal rules on tables.

## D21.1 — Prose is set on a measure; figures and tables break out past it

Each section is a three-column grid — a named `text` column of `34em` between
two flexible gutters, and a named `wide` column spanning all three. Paragraphs
and headings sit in `text`, which holds a line to roughly seventy-five
characters at any window width. Figures, tables and the metric panel take
`wide`, so a six-column table is not squeezed into the width that suits a
sentence.

The alternative — one column sized to the content — makes either the prose
unreadable or the tables cramped, and which one it makes depends on the window.

## D21.2 — The contents rail is out of the flow, so it cannot move the article

Above `68em` the rail is `position: fixed` in the left margin rather than a
grid column. In a grid it would push the article off centre by half its width,
so a reader who never looks at it would still pay for it. Below `68em` it
becomes a two-column box above the title. It is emitted only for reports of
three sections or more.

## D21.3 — Captions go above tables and below figures

A table is read from its caption down; a figure is read up to its caption. The
`Table 2.` / `Figure 1.` label is wrapped in a span and set in the text colour
so the number can be found while scanning, and the sentence after it stays
quiet.

## D21.4 — Tables get three rules and a wash, not a grid

Rules above and below the header and one under the last row, no vertical rules,
no per-row borders — booktabs, essentially. Rows alternate a 3 % wash of the
text colour, which is the one addition to that convention: the design table in
a full dossier runs to thirty rows and a reader tracking one across four
columns needs the help.

## D21.5 — Print flows; it no longer starts each section on a fresh sheet

The old stylesheet set `break-before: page` on every `h2`, which for a
ten-section dossier prints ten sheets, most of them a third full. An article
flows. What print now keeps is `break-inside: avoid` on figures, tables, the
metric panel and the ledger, and orphan/widow control on paragraphs.

Paged output as such is `render/pdf.py`'s job, and it is unchanged.

## D21.6 — The theme still owns every value; the stylesheet owns the layout

The colours, fonts and sizes are emitted as custom properties in a `:root`
block generated from the `Theme`, and the rest of the stylesheet is a static
string written against `var(--ink)`, `var(--measure)` and the rest. A theme
change touches no rule, and a layout change touches no theme. Two derived
values that were not on the theme — a tint of the accent and a wash of the ink
— are computed from it rather than added to it, because a theme that has to
name every alpha it might want is a theme nobody can write by hand.

## D21.7 — Plotly is told the font family, and to find its own margins

`Theme.font` is a PostScript name (`Times-Roman`) for the PDF renderer's
benefit; passing it to plotly alone left every figure lettered in the browser's
default sans inside a serif document. The figure now gets
`font.family = f"{font}, {font_fallback}"`, the same chain the CSS uses.

Estimand labels are long enough to run off the left of plotly's fixed 55 px
margin, so both axes are set to `automargin`: the margin grows to fit the
labels instead of the labels being drawn over the panel.

## What this did not fix

`axiom_dossier.figures.diagnostics_bar` draws z-statistics and counts on one
axis; when the values are mostly positive the negative bars collapse against
the left edge and their labels collide with the category names. The
`SPREAD_LIMIT` guard in that function exists for exactly this and does not
catch it, because the ratio is inside the limit. That is a figure bug in the
add-on, not a stylesheet one, and it is open.
