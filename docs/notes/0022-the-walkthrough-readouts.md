# 0022 — A readout carries the run's own tables and charts, not a summary of them

*Kind: decision + deviation. Opened 2026-08-24. Status: implemented; `axiom_dossier.charts`, `axiom_dossier.walkthrough`.*

`packages/axiom-dossier/examples/readouts.py` builds one journal-styled report
per example under `examples/`, from the structured record each script writes as
it runs. Twelve reports, and every one of them was missing the run it described:
no figures at all against forty-three the records held, two tables against
nineteen, and a design table whose rows were paragraphs of prose and one line of
raw `dict` repr with its braces doubled.

The bridge carried the narrative and dropped everything else. This note records
what it carries now and why each piece lands where it does.

## D22.1 — Four kinds of recorded thing, four places in the report

A walkthrough step records four different things, and the old mapping flattened
three of them into one `detail` dictionary — which is also the source of the
design table, which is how a table of protocol settings came to hold
`readout: design : central_composite, 12 plots; detail : {'alpha': 1.41…}` in a
cell. `MethodStep` now keeps them apart:

| recorded as | lands in | why not `detail` |
|---|---|---|
| `say` blocks | the step's prose | — |
| `instead` | `MethodStep.instead`, its own run-in paragraph | it is an argument, not a setting |
| `out` lines | `MethodStep.readout`, a monospaced block | printed output is not a parameter |
| tables and charts | `MethodStep.exhibits`, beside the prose | data, which lives in the context |

`detail` keeps its original meaning — short keys, short values, the design table
— and for a walkthrough record it is now empty, so the design table is absent
rather than wrong.

## D22.2 — The readout is kept as lines, and rendered with its columns intact

The old bridge joined the first four printed lines with `"; "`. Example output is
column-aligned: `nitrogen   : 0 to 200 kg/ha` next to `irrigation : 0 to 120 mm`
is a small table, and running it together destroys the only thing making it
readable. Lines are kept as lines, capped at twelve with a count of what was
dropped, and rendered as one paragraph of inline-code lines.

That last part is a compromise. `axiom.report` has no preformatted block, and
adding one to carry six lines of terminal output would be a change to the core
spec and all three renderers for a feature only this bridge wants. The inline
code mark gives a monospaced face in HTML and in PDF, and the HTML stylesheet
now sets `white-space: pre-wrap` on `code`, without which the browser collapses
exactly the columns the readout is being kept for.

## D22.3 — An exhibit is named by the step and checked before it is drawn

A step names the exhibits it produced; the data arrives separately through
`build(extra_exhibits=…)`, and `build` drops any figure or table block whose key
has nothing behind it. That ordering matters: naming an exhibit that cannot be
drawn — plotly absent, a payload missing a field — would leave `missing()`
non-empty and the whole report unrenderable, so a chart that cannot be drawn
costs its own figure and nothing else.

## D22.4 — Eight chart kinds, translated rather than described

`axiom_dossier.charts` draws the eight kinds a walkthrough can record — `band`,
`bars`, `dumbbell`, `funnel`, `heatmap`, `intervals`, `lines`, `scatter` —
in plotly, from the same payload and `opt` mapping the site's SVG charts read.
`opt` values beginning with `@` are paths into the figure's own payload, which
is the convention `examples/_walkthrough.py` records.

It is a translator, not a chart library: it invents no number, and a kind it does
not know or a payload missing a field it needs is a typed `Unsupported` naming
what was wrong. Half a chart is worse than no chart.

Two rules from `figures.py` are kept here, because they are about reading rather
than about taste: a bar's label sits outside the bar so a short bar does not
print its number over the row name, and a chart with bars on both sides of zero
is drawn against a zero line. A heat map's marks carry their labels in boxes,
because a mark has to be legible at both ends of a sequential scale.

## D22.5 — The evidence holds raw text; the escape happens at the paragraph

`literal()` doubles braces so an analyst's `{}` is not read as a template
placeholder. The old bridge applied it on the way *in*, to text that also becomes
a table cell — and a cell is not a template, so the escape leaked to the page as
`{{'alpha': 1.41}}`. The rule is now the other way round: an `Evidence` holds the
text as it was recorded, and every builder escapes what it puts in a `Paragraph`.
That also fixed two latent cases nobody had hit — a note on a quantity, and the
notebook prose the agent pipeline puts in a step.

## What this does not fix

The results section of every readout still says no findings were recorded, and
it is right. The examples record their findings as **prose** — "2SLS bought
accuracy and cost width" — not as quantities with intervals. The `intervals`
charts do hold structured estimates, but those are mostly parameter-recovery
errors rather than the study's finding, and promoting them would be the report
guessing at which number its own author meant. The fix is one line per example
in `examples/`: record the headline quantity. Until then the readouts carry the
argument, the reasoning and the pictures, and say plainly that they carry no
metric.
