# Generated documents

Committed on purpose. These are what `axiom-dossier` produces, and a reader
should be able to open one without installing an extra or holding an API key.
Every file here is overwritten in place by re-running the example that made it.

## `hyper3-apa.{pdf,html,pptx}` — an APA manuscript

```bash
python examples/hypertension.py --apa --full --narrate
```

The HYPER-3 trial in APA manuscript form, following the UCSD Psychology guide
*How to Write APA Style Research Papers* and the empirical-paper example beside
it: title page with an author note and running head, abstract on its own page,
body from the introduction through the discussion, then the tables and the
figures **each on its own page after the text**, numbered and captioned
`*Table 1*.` / `*Figure 1*.`

Figure 1 is the one worth opening. Five contrasts on a common axis with the
decision threshold marked: four green, and one red on the other side of zero.
The finding is a thing you see rather than a thing you work out from a column
of numbers.

## `hyper3-full.{pdf,html,pptx}` — the same trial, journal style

```bash
python examples/hypertension.py --journal --full --narrate
```

Numbered sections, exhibits embedded beside the prose that discusses them
rather than gathered at the end.

**Both are narrated.** Every section's prose was rewritten by
`gemini-3.7-flash` (`gemini-3.5-flash-lite` for the abstract) and every numeral
and claim in the result was traced back to the evidence record before it was
kept. The provenance section lists which model wrote which section. Re-running
changes the wording and no number: the numbers come from the trial.

## `hyper3-agent.{pdf,html}` — the whole case study, read by the agent

```bash
python examples/hypertension_agent.py --model --execute
```

Not written from one script that knows the trial: **read** from the six
notebooks under `nbs/case-studies/hypertension/`, which store none of their
outputs. The pipeline executes all ninety-two code cells in one namespace,
harvests what they leave behind, labels each object from the prose that
introduced it, notices which figures the prose describes that the run never
drew, and writes the code that draws them.

The planning and outcome notes (`docs/notes/0004`, `0005`, and the series
README) are consolidated into the report's remarks.

Figures here come from three places and the report does not pretend otherwise:
the ones the notebooks drew, the ones this package generates from the evidence,
and the one or two the agent wrote code to add.

## `readouts/*.{pdf,html}` — one per axiom example

```bash
python examples/readouts.py
```

Built from the structured record each script writes as it runs. Each carries the
question, every step with its reasoning **and the alternative that example
rejected**, the readouts, and the analyst's closing remarks verbatim.

**Not narrated**, and they need no key: this is what the package produces with
no language model involved at all, which is most of it.

They have no metric blocks, no figures and no threshold readings. That is not a
gap in the report — the twelve examples record their findings as prose rather
than as quantities, so there is nothing structured to draw or tabulate. An
example that records quantities gets all of it with no change to this package.

## About the HTML

The charts are **live plotly**, not pictures: hover a point and it tells you the
quantity and its value. The PDF and PPTX rasterise the same figures through
kaleido.

The committed HTML *links* the plotting library rather than bundling it, so the
charts need a network connection to draw and each file stays around 30 kB
instead of 4.8 MB. That is a choice made for the repository, not the default:
`Dossier.write` bundles by default, because a report you email should not need
a CDN to draw its own figures. `--standalone` writes that version.

## Regenerating everything

```bash
cd packages/axiom-dossier
python examples/hypertension.py --apa --full --narrate    # needs GEMINI_API_KEY
python examples/hypertension.py --journal --full          # no key, generated prose
python examples/readouts.py
```
