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
    .finding("contrast", result, label="40 mg vs control", unit="mmHg")
    .diagnostic("coverage", 0.94, label="Interval coverage")
    .build()
)

built = build(evidence)          # no language model involved
built.write("hyper3.pdf")
```

### 2. A model may improve the prose — but never the numbers

```python
from axiom_dossier import Narrator, build

built = build(evidence, narrator=Narrator())    # Gemini rewrites the drafts
built.rejected()                                 # narrations that failed the check
```

The model is never asked to analyse anything. It is given the evidence and a
draft that is already correct, and asked to make the draft read better. Then the
result is checked, mechanically: **every numeral in the returned text must trace
to a recorded quantity.** One that does not gets the narration rejected and the
generated draft kept, and the document says so in its own provenance section.

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
  sections.py    methods / results / diagnostics / limitations, generated from typed results
  numbers.py     the provenance check: which numerals in this prose are invented?
  language.py    the model seam — a Protocol, a Gemini implementation, an Offline stand-in
  narrate.py     rewrite a draft, verify it, keep the draft when it drifts
  dossier.py     assemble the whole document and write it out
```

## Testing

```bash
cd packages/axiom-dossier
uv run pytest                    # offline; no key, no network
uv run pytest -m live            # calls the real API; needs GEMINI_API_KEY
```

`Offline` is a real `LanguageModel`, not a mock, so the narration path is
exercised end to end without a network. The `live` tests are deselected by
default.
