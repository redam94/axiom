# Generated documents

Committed on purpose. These are what `axiom-dossier` produces, and a reader
should be able to open one without installing an extra or holding an API key.
Every file here is overwritten in place by re-running the example that made it.

## `hyper3-apa.{pdf,html,pptx}` — an APA manuscript

```bash
python examples/hypertension.py --apa --narrate
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
python examples/hypertension.py --narrate
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

## `geiger-{full,apa}.{pdf,html,pptx}` — an experiment designed, run, and read

```bash
python examples/rutherford.py --narrate
python examples/rutherford.py --apa --narrate
```

GEIGER-1911 from `nbs/case-studies/rutherford/`: alpha particles on a gold foil,
to decide whether the positive charge of an atom is concentrated or spread. The
station plan, the hours and the apertures come from `scattering.py`; the counts
are one Poisson draw at each station; the bound is
`design.profile_likelihood` against those counts. It comes back **R < 33 fm**,
against a floor of 29.6 fm that no amount of counting can beat because the beam
energy sets it. Rutherford published 34 fm in 1911.

Worth opening for two things this shape has and a trial does not. **The methods
section is most of the document**, because in a designed experiment the argument
*is* the design — an angle chosen for the wrong reason cannot be fixed by the
analysis, and each step carries the alternative it was chosen over. And **the
finding is a one-sided bound**: there is no lower limit on the radius here, and
the report states that rather than printing a point estimate that reads like one.

## `tutor60-{full,apa}.{pdf,html,pptx}` — one decision, from the budget to the memo

```bash
python examples/tutor60.py --narrate
python examples/tutor60.py --apa --narrate
```

TUTOR-60 from `nbs/case-studies/tutoring/`: nine million dollars, twelve thousand
students reading behind, and two treatments sold by the weekly minute and the
weekly message. It fits the trial through PyMC, so it takes a minute or two.

The recommendation is thirty-five weekly minutes and five messages for the whole
cohort, not the pilot's ninety minutes for the fraction the same money reaches —
ahead by **13,838 cohort reading points**, about 19 million dollars at the
department's own valuation, in 99.7% of the posterior.

What it shows that the other two do not: **the finding is a comparison between
two things you could buy**, paired inside every posterior draw so the uncertainty
common to both cancels. That is what a decision report has instead of a headline
effect, and it is why the question is written as the one the comparison settles
rather than as "how much should we buy" — a how-much question has no threshold,
and a conclusions section cannot answer it.

The example is `tutor60.py` rather than `tutoring.py` because the case study's own
world module is `tutoring.py` and the two must not shadow each other.

## `readouts/*.{pdf,html}` — one per axiom example

```bash
python examples/readouts.py
```

Built from the structured record each script writes as it runs. Each carries the
question, every step with its reasoning **and the alternative that example
rejected**, the printed output as it was printed, every table the run recorded,
every chart it recorded — redrawn in plotly from the numbers behind it — and the
analyst's closing remarks verbatim. Nineteen tables and forty-three figures
across the twelve.

**Not narrated**, and they need no key: this is what the package produces with
no language model involved at all, which is most of it.

They have no metric blocks and no threshold readings. That is not a gap in the
report — the twelve examples record their *findings* as prose rather than as
quantities, so there is nothing structured to read one from, and inventing them
from the sentences would be guessing. An example that records quantities gets
those sections with no change to this package.

## About the HTML

The charts are **live plotly**, not pictures: hover a point and it tells you the
quantity and its value. The PDF and PPTX rasterise the same figures through
kaleido.

The committed HTML *links* the plotting library rather than bundling it, so the
charts need a network connection to draw and each file stays around 30 kB
instead of 4.8 MB. That is a choice made for the repository, not the default:
`Dossier.write` bundles by default, because a report you email should not need
a CDN to draw its own figures. `--standalone` writes that version.

## The provenance appendix

Every file here ends with one, and until recently none of them did. `build`
decided which context keys a section could name an exhibit from by looking only
at the figures and tables it had drawn, so the three that come from the evidence
record itself — the standing assumptions, the quantities and their sources, and
the run — were dropped from every report. A package whose fourth rule is that
every number carries its provenance was rendering that section as a lone
paragraph. See `docs/notes/0023`.

## Regenerating everything

```bash
cd packages/axiom-dossier
python examples/hypertension.py --apa --narrate            # needs GEMINI_API_KEY
python examples/hypertension.py --narrate                  # needs GEMINI_API_KEY
python examples/hypertension_agent.py --model --execute    # needs GEMINI_API_KEY
python examples/rutherford.py --narrate                    # needs GEMINI_API_KEY
python examples/rutherford.py --apa --narrate              # needs GEMINI_API_KEY
python examples/tutor60.py --narrate                       # needs GEMINI_API_KEY, fits a model
python examples/tutor60.py --apa --narrate                 # needs GEMINI_API_KEY, fits a model
python examples/readouts.py                                # no key, generated prose
```

All but the last are narrated and the wording changes on every run; the numbers
do not. `readouts.py` needs no key at all, which is the point of it: it is what
the package produces with no language model involved.
