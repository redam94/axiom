# Generated documents

Committed on purpose. These are what `axiom-dossier` produces, and a reader
should be able to open one without installing an extra or holding an API key.
Every file here is overwritten in place by re-running the example that made it.

## `hyper3-full.{pdf,html,pptx}`

```bash
python examples/hypertension.py --narrate
```

The HYPER-3 sequential dose-finding trial from
[`nbs/case-studies/hypertension/`](../../../../nbs/case-studies/hypertension/),
as the document such a trial has to produce: protocol, measurement, analysis and
safety monitoring, then identification, results, model checking, discussion,
conclusions and limitations. Journal style, `verbosity="full"`, all three
formats.

**Narrated.** Every section's prose was rewritten by `gemini-3.7-flash`
(`gemini-3.5-flash-lite` for the abstract) and every numeral and claim in the
result was traced back to the evidence record before it was kept. Page 8 lists
which model wrote which section and that each was verified.

Because a language model is not deterministic, re-running this changes the
wording. It does not change any number: those come from the trial.

The reason this one is worth reading is its conclusion. The 40 mg arm pools to
−1.53 mmHg (90 % −2.97 to −0.10) — mildly beneficial — while in the 51+ band it
is **+6.20 mmHg (90 % 3.98 to 8.42)**, harm. The conclusions section detects that
the recorded findings disagree and refuses to answer with one number, which is
the case study's central lesson arrived at mechanically rather than written in.

## `readouts/*.{pdf,html}`

```bash
python examples/readouts.py
```

One report per example in [`examples/`](../../../../examples/), built from the
structured record each script writes as it runs. Each carries the question,
every step with its reasoning **and the alternative that example rejected**, the
readouts, and the analyst's closing remarks verbatim.

**Not narrated**, and they need no key: this is what the package produces with
no language model involved at all, which is most of it.

They have no metric blocks and no threshold readings. That is not a gap in the
report — the twelve examples record their findings as prose rather than as
quantities, so there is nothing structured for those sections to read. An
example that records quantities gets them with no change to this package.

## Regenerating everything

```bash
cd packages/axiom-dossier
python examples/hypertension.py --narrate     # needs GEMINI_API_KEY
python examples/hypertension.py               # same document, generated prose
python examples/readouts.py
```
