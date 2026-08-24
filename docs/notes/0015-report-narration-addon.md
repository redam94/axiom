# 0015 — The report generator the charter refuses, as a package beside it

*Kind: decision. Opened 2026-08-22. Status: implemented on `feature/report-narration`; `packages/axiom-dossier` 0.1.0.*

The ask was a language model writing better-formatted reports, with methods and
analysis sections generated from axiom's own results. Two lines of the charter
are directly in the way, both under **Out — permanently, not "later"**:

> - Any LLM or agent framework.
> - A report generator. `axiom` returns typed results; rendering is someone
>   else's job.

The second line is also the resolution. This work lives in
`packages/axiom-dossier`, a separate distribution that depends on `axiom`;
`axiom` does not know it exists, imports nothing from it, and none of the twelve
gates can see it (they walk `axiom.__path__`). The charter did not need
overriding — it named the boundary and the work went on the other side of it.

Worth recording that the boundary had already been crossed once without being
recorded: `axiom.report` is a full report engine that shipped in Phase 9, and
note [0007](0007-uncertainty-and-reports.md) describes it without mentioning
that the charter forbids it. That line is now stale in one direction and
accurate in the other, and this note is the place that says so.

## D15.1 — What is generated, and what is merely rewritten

The package does two things and keeps them apart on purpose.

**The sections are generated from typed results, with no model involved.**
`methods_section` is built from the `IdentificationVerdict` and the
`Assumption`s that licensed the analysis, so it names the route and the
adjustment set and *cannot describe a route the analysis did not take*.
`limitations_section` is built from the assumptions that are not `satisfied`,
so it *cannot be shorter than the truth*. Findings are emitted as `Metric`
blocks, which is how the interval survives the trip to a page. None of this
needs a key, a network or an extra: with no model configured you still get a
complete document.

**A model may then rewrite that prose, and only that.** The model is never asked
to analyse anything — it is handed the evidence and a draft that is already
correct and asked to make the draft read better. Metrics, tables, figures and
ledgers are not passed to it at all.

## D15.2 — The numeric gate is what makes narration publishable

A model asked to narrate results will produce a fluent sentence containing a
number nobody computed. Prompting against it helps and does not settle it. So
the output is checked mechanically: every numeral in the returned text must
trace to a quantity in the `Evidence` record (`numbers.unverified`). If one does
not, **the narration is rejected and the generated draft is kept** — the
`Narration` records which literals failed, and the document's provenance section
prints it. A failure costs prose quality and never correctness.

Two deliberate weakenings, both recorded in the module rather than discovered
later:

- **Magnitude is licensed.** An estimate of `-12.4` licenses the literal "12.4",
  because a report writes that effect as "lowered by 12.4". A sign error
  therefore passes. The alternative was a check that rejects the sentences
  reports are actually made of, which is a check nobody would leave switched on.
- **Rounding is licensed downwards only.** `12.43` licenses "12.4" and "12" but
  not "12.4321", because the latter claims precision the analysis never made.

The gate catches invented numbers. It does not catch a licensed number attached
to the wrong noun, and the docstring says so.

## D15.3 — Two models, split by what the call is worth

`gemini-3.7-flash` writes the prose; `gemini-3.5-flash-lite` does the abstract
and the cheap mechanical passes. Rewriting a methods section is the
quality-sensitive call and an abstract is not, and paying flagship rates for the
latter is waste. Both ids were verified against the live API rather than
assumed — a live test asserts each default model exists and answers, because a
constant holding a model name that quietly stopped existing is exactly the
failure a unit test cannot see.

The key is read from the environment **at the moment of the call** and never
stored on the object, so a `Gemini` stays safe to serialize into a report's
provenance appendix. A missing extra or a missing key is a typed `Unsupported`
naming what to install, not an `ImportError` at the top of a module.

## D15.4 — Per-section scope, found by reading the output

The first working draft handed every section the same complete facts block. Each
section then dutifully restated the whole study: methods, results and model
checking all opened by re-describing the design and the estimate. Nothing failed
— every numeral traced, every test passed — and the document was unreadable.
`SECTION_INSTRUCTION` now tells each section what it is for and what belongs to
its neighbours. It was caught by rendering the PDF and looking at it.

**Superseded by [0023](0023-what-each-section-may-say.md).** Telling a section
what belongs to its neighbours, while still showing it the whole study, did not
hold: the same phrases came back in six sections of the narrated HYPER-3 report.
Each section is now *shown* only the facts it may state.

## D15.5 — A bug in axiom that this work found

`report/render/pdf.py` passed **bare strings** as reportlab table cells.
reportlab does not wrap a raw string: it draws it at full width and lets it run
over the next column, so any table of sentences — an assumption ledger, a
provenance appendix — rendered as overlapping, unreadable text. Cells are
`Paragraph`s now, aligned top, and
`test_pdf_table_cells_wrap_instead_of_running_over_the_next_column` pins it: a
table of long sentences must overflow onto further pages, which unwrapped rows
never would. The test was confirmed to fail against the old renderer before the
fix was kept.

This is a fix to axiom proper, not to the add-on, and it improves every table
axiom has ever rendered to PDF.

## D15.6 — Interpretation, and the second gate it required

The first version stopped at reporting. Adding a **Discussion** and a
**Conclusions** section means adding the two places a report is most likely to
lie, so two things went in with them.

**Interpretation is read off a threshold, not judged.** A `Quantity` now carries
the value the decision turns on, and the reading is derived from where its
interval sits relative to that value: wholly on one side settles the question,
containing it does not. The distinction is made explicit in the generated text
because it is the one applied work destroys most often — *an interval spanning
the threshold is an unsettled question, not a finding of no effect.* The reading
also inherits the identification verdict, and carries that qualifier in every
sentence rather than only in the section's preamble, so a line quoted out of a
discussion does not lose it.

**`claims.unlicensed` is the second gate.** The numeric check cannot see "the
effect is robust" or "this proves the treatment works" — sentences with no
numeral at all, and exactly the sentences a conclusions section invites. A claim
word is now licensed only if the evidence record already uses it, on the same
principle as a number, and either gate failing rejects the narration. This was
the open item the first version recorded, and adding an interpretation section
without it would have been indefensible.

## D15.7 — Verbosity changes the draft, not the prompt

Asking a model to "write more" about a three-line draft is asking it to pad.
`brief` / `standard` / `full` therefore change what the *generated* draft
contains — per-step detail, per-finding headings, the ledger, the recaps — and
the narration inherits the richer draft along with a longer sentence target. A
level is a row in one table, so a level that forgets a switch is a test failure
rather than a silent difference.

## D15.8 — Journal style is a shape before it is a typeface

`style="journal"` changes the section order to the one a reader navigates by
habit — abstract, introduction, methods, results, checking, discussion,
conclusions, limitations, provenance — numbers the headings and the `Table N.`
captions, and applies a serif theme. The order is the substantive half: putting
discussion before results, or dropping limitations, produces a document that
reads as marketing whatever face it is set in.

One axiom change was needed. `Theme.font` is used verbatim by all three
renderers, but `Times-Roman` is a PostScript name rather than a CSS family, so
the PDF rendered serif while the HTML silently fell back to sans. `Theme` now
carries a `font_fallback` that HTML uses, defaulting to the previous hard-coded
chain, so a serif theme renders as one in every format.

## D15.9 — Two more things reading the output caught

Rendering the paper and looking at it found what the tests did not: a per-finding
heading at `full` verbosity printed the same words as the metric block directly
beneath it, and assumption names reached the narrated abstract as
`no_unmeasured_confounding`, because the prompt's fact block passed identifiers
straight through. Both are fixed. This is the second time in this package's
short life that rendering and reading found defects nothing else did.

## What is open

* **Figures.** The dossier generates prose and tables. A `Figure` block pointing
  at a `surface.ResponseBand` already renders through `axiom.viz`; nothing yet
  *selects* which figures a report should carry.
* **Narration is per section.** A whole-document pass could remove the residual
  repetition between sections that per-section instructions only reduce.
* **The claim list is a list.** It catches the vocabulary of overreach, not
  overreach itself: "the data leave little room for doubt" passes both gates.
* **No references section.** A paper cites; this one has a provenance appendix
  instead, which is the right artefact for a generated report and not the same
  thing.
* **One provider.** `LanguageModel` is a Protocol and `Gemini` is one
  implementation; nothing else is written yet.
