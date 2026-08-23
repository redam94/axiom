# axiom-dossier

Shareable research reports from [axiom](../../README.md) results — in HTML, PPTX
or PDF.

axiom's charter is explicit that this is not its job:

> **A report generator.** `axiom` returns typed results; rendering is someone
> else's job.

This is that someone else. It depends on `axiom`; `axiom` does not know it
exists, and none of axiom's twelve contract gates can see it.

## What it adds

`axiom.report` already renders a template to three formats. What it still needed
a person for was the writing. This package removes that, in two steps that are
kept deliberately separate.

### 1. The sections write themselves — no model, no key, no network

A methods section is generated from the `Verdict` and the `Assumption`s that
licensed the analysis, so it **cannot describe an identification route the
analysis did not take**. A limitations section is generated from the assumptions
still unresolved, so it **cannot be shorter than the truth**. Results are
emitted as metric blocks, so an interval cannot be dropped on the way to a page.

```python
from axiom_dossier import EvidenceBuilder, build

evidence = (
    EvidenceBuilder("HYPER-3", "Does 40 mg lower systolic pressure?")
    .verdict(verdict)                                     # from axiom.identify
    .finding(
        "contrast", result, label="40 mg vs control", unit="mmHg",
        threshold=-5.0, beneficial="lower",               # what the decision turns on
    )
    .diagnostic("coverage", 0.94, label="Interval coverage")
    .build()
)

built = build(evidence)          # no language model involved
built.write("hyper3.pdf")
```

### Journal style, and how long the report is

```python
built = build(evidence, style="journal", verbosity="full")
```

`style="journal"` changes both the shape and the type. The shape is the order a
reader navigates by habit — abstract, introduction, methods, results, model
checking, **discussion**, **conclusions**, limitations, provenance — with
numbered headings and numbered `Table N.` captions. The type is serif, tight and
almost monochrome. `style="plain"` is the readout: no abstract, no discussion,
no numbering.

`verbosity` is `brief` / `standard` / `full`, and it changes **what the draft
contains**, not just how long the model is told to write. Asking a model to
"write more" about a three-line draft is asking it to pad; `full` adds per-step
detail, per-finding headings, the ledger and the assumption recaps, and the
narration inherits the richer draft.

### Figures and data tables

`figures.py` draws what a causal report needs and `tables.py` builds what it
tabulates — both from the evidence, so neither can disagree with the metric
blocks beside them.

The figure that carries the weight is `findings_plot`: every finding on one
axis with its interval, the decision threshold marked, and each row coloured by
which side it settled on. Three colours, not two — a row whose interval *spans*
the threshold is grey, because unsettled is a third state and drawing it as
either of the others is the lie this package exists to avoid.

**HTML charts are interactive.** `axiom.report` renders a plotly figure live in
HTML and rasterises it through kaleido for PDF and PPTX, so the same report is
hoverable on screen and printable on paper.

Two refusals worth knowing about. A figure is never given a title, because the
caption carries it and a figure with both says everything twice. And
`diagnostics_plot` returns `Unsupported` when the diagnostics span more than two
orders of magnitude: a retention share of 0.797 drawn beside a count of 372 is a
bar of no width next to a bar of full width, which tells a reader the share is
nothing. The table reports them properly instead.

```python
built = build(evidence, exhibits="embedded")   # each with the prose that discusses it
built = build(evidence, exhibits="gathered")   # collected after the text, APA-style
built = build(evidence, exhibits="none")
```

### APA manuscript style

```python
built = build(evidence, style="apa", authors=("A. Author",), affiliation="Somewhere")
```

Follows the UCSD Psychology guide *How to Write APA Style Research Papers* and
the empirical-paper example beside it: title page with author note and running
head, abstract on its own page, body continuous from introduction to discussion,
then tables and figures each on their own page after the text, captioned
`*Table 1*.` and `*Figure 1*.` Twelve-point Times, one-inch margins, no display
type.

Two departures, both deliberate. A causal report has to say what licensed its
estimate, so **Limitations** and **Provenance** stay as their own sections; APA
folds the first into the discussion and has no equivalent of the second.

### Interpretation, without the overreach

A conclusions section is where a report is most likely to lie, so interpretation
here is mechanical. Give a finding a `threshold` — a null, a minimum worthwhile
difference, a budget — and the discussion reads the interval against it:

- interval wholly on one side → *the question is settled in that direction*;
- interval containing it → *unsettled*, and explicitly **not** a finding of no
  effect, which is the distinction applied work destroys most often;
- no threshold → the estimate is stated and the report stops short of judging it.

The reading also inherits the identification verdict: an unidentified effect is
described as *the observed difference*, never as what the treatment did, and the
qualifier travels with each sentence rather than sitting only in a preamble.

### 2. A model may improve the prose — but never the numbers

```python
from axiom_dossier import Narrator, build

built = build(evidence, narrator=Narrator())    # Gemini rewrites the drafts
built.rejected()                                 # narrations that failed the check
```

The model is never asked to analyse anything. It is given the evidence and a
draft that is already correct, and asked to make the draft read better. Then the
result is checked, mechanically, against **two** gates:

- **numbers** — every numeral in the returned text must trace to a recorded
  quantity;
- **claims** — every claim word must already appear in the record. "Robust",
  "significant", "proves", "definitive", "always": a causal analysis does not
  get these for free, and they carry no numeral for the first gate to catch.
  This one exists because a conclusions section is precisely where a model
  reaches for them.

Either gate failing rejects the narration and keeps the generated draft, and the
document says so in its own provenance section.

That check is the reason this is safe to use. Rounding is allowed in the
direction a writer actually rounds — a value of `12.43` licenses "12.4" and
"12" — but not the other way, because "12.4321" claims a precision the analysis
never made.

## Two models, on purpose

| role | default | what it does |
|---|---|---|
| `prose` | `gemini-3.7-flash` | rewrites methods / results / limitations |
| `light` | `gemini-3.5-flash-lite` | abstract, captions, the cheap mechanical passes |

Rewriting a methods section is the quality-sensitive call. Producing a one-line
abstract is not, and paying flagship rates for it is waste.

```python
from axiom_dossier import Gemini, Narrator

narrator = Narrator(prose=Gemini("gemini-3.7-flash"), light=Gemini("gemini-3.5-flash-lite"))
```

## Install

```bash
pip install axiom-dossier              # sections + HTML
pip install "axiom-dossier[render]"    # + PPTX and PDF
pip install "axiom-dossier[gemini]"    # + narration
pip install "axiom-dossier[all]"
```

The key is read from `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) **at the moment of
the call** and is never stored on the object — `Gemini` holds a model id and
nothing else, so it stays safe to serialize into a report's provenance.

Everything except the narration works with the `gemini` extra absent. A missing
extra or a missing key is a typed `Unsupported` naming what to install, not an
`ImportError` at the top of the file.

## Layout

```
src/axiom_dossier/
  evidence.py    the record a report is written from — and the only place its numbers come from
  sections.py    methods / results / diagnostics / limitations, plus the verbosity table
  interpret.py   discussion and conclusions, read mechanically against the threshold
  journal.py     the paper shape: abstract, introduction, numbering, the serif theme
  numbers.py     gate one — which numerals in this prose are invented?
  claims.py      gate two — which claims does the record not actually make?
  language.py    the model seam — a Protocol, a Gemini implementation, an Offline stand-in
  narrate.py     rewrite a draft, verify it, keep the draft when it drifts
  dossier.py     assemble the whole document and write it out
```

## The agent: reading a whole case study

```bash
pip install "axiom-dossier[agent]"
python examples/hypertension_agent.py --model --execute
```

The HYPER-3 case study is six notebooks, ninety-two code cells and **no stored
outputs** — the repository strips them, so every number in it exists only while
the code is running. A report on it cannot be assembled by parsing anything. So
`axiom_dossier.agent` executes the series in one namespace and reports on what
it leaves behind, using LangGraph to make the order a declared thing and the
repair loop an edge rather than a `while`:

```
read ─▶ run ─▶ label ─▶ gaps ─▶ fill ─▶ assemble ─▶ verify ─┐
                         ▲                                  │
                         └──────────── unresolved ──────────┘
```

| stage | model? | what it does |
|---|---|---|
| `read` | no | parse the notebook series and the planning notes |
| `run` | no | execute every notebook in one namespace; harvest per book |
| `label` | yes | name each harvested object from the prose that introduced it |
| `gaps` | yes | find figures the prose describes that the run never drew |
| `fill` | yes | write and run Python that draws them |
| `assemble` | no | build the `Evidence` and the document |
| `verify` | no | check the run; loop back if a gap is still open |

**The three model stages name and notice; they never compute.** A variable
called `itt` is not a label — the sentence above it in the notebook says what it
is, and that is what the model reads. It is shown the variable, its type and its
prose, never its value, because a model given the number puts the number in the
label and the gates would then be checking prose against a model's own
arithmetic. Every number still comes from executing the analysis, and everything
reaching prose still passes the numeric and claim gates.

On the real series it runs 6/6 notebooks, harvests 9 quantities and 8 figures,
keeps the 5–6 that are findings rather than scaffolding, and closes the one or
two gaps it finds by writing code — typically the sequential-monitoring
trajectories against the harm boundary, which the notebooks compute and never
plot as a standalone exhibit.

`--execute` is a separate flag from `--model` because the fill stage runs
model-written Python **in this process, with this process's permissions**. That
is right for a developer tool pointed at a repository you already trust and
wrong for anything else; `run_python` refuses without `allow_execution=True`.

## Two worked examples

```bash
python examples/hypertension.py --narrate      # one trial, end to end
python examples/readouts.py                    # all twelve axiom examples
```

**The output of both is committed**, under
[`examples/out/`](examples/out/) — one narrated trial report in PDF, HTML and
PPTX, and twenty-four generated readouts. Open one rather than taking this
README's word for it; `examples/out/README.md` records exactly how each was
produced. Re-running an example overwrites its files in place.

**`hypertension.py`** turns the HYPER-3 case study — the sequential dose-finding
trial under `nbs/case-studies/hypertension/` — into the document such a trial has
to produce: protocol and design up front, then identification, findings and
limitations. Nothing in it is typed in. The protocol constants come from
`hyper3.py`, the arm counts from the realized randomization, and every contrast
from `identify.ols` against the analysis frame the trial actually produced.
Change the seed and the whole report changes.

It is also the example that shows why a conclusions section needs care. The
40 mg arm pools to −1.53 mmHg (90 % −2.97 to −0.10) — mildly beneficial — while
in the 51+ band it is **+6.20 mmHg (90 % 3.98 to 8.42)**, harm. `contested()`
detects that the recorded findings disagree, and the conclusion refuses to
answer with one number instead of quoting whichever finding came first.

**`readouts.py`** runs each of the twelve scripts in `examples/` with its
structured-record capture on and writes one journal-shaped report per example.
Each carries the question, every step with its reasoning *and the alternative
the example rejected*, the readouts, and the analyst's closing remarks verbatim.

Those remarks are the one part of a report this package will not touch: they are
a person's words, and a model rewriting them while they stay attributed to that
person is a misattribution, not an improvement. `remarks` is in `_NEVER_NARRATED`
beside `provenance`.

The examples record their findings as prose rather than as quantities, so their
reports carry the narrative and the reasoning but have no metric blocks and no
threshold reading — there is nothing structured to read. An example that records
quantities gets those sections with no change here.

## Testing

```bash
cd packages/axiom-dossier
uv run pytest                    # offline; no key, no network
uv run pytest -m live            # calls the real API; needs GEMINI_API_KEY

python examples/hyper3.py --journal --full --narrate
```

The live tests are deselected by `addopts` even when a key is in the
environment, so a default `pytest` never spends money.

`Offline` is a real `LanguageModel`, not a mock, so the narration path is
exercised end to end without a network. The `live` tests are deselected by
default.
