# 0023 — Each section is shown only the facts it may state

*Kind: decision. Opened 2026-08-24. Status: implemented; `axiom_dossier.narrate`,
`axiom_dossier.evidence`, `axiom_dossier.figures`, `axiom_dossier.tables`,
`axiom.report.render`,
`axiom_dossier.interpret`, `axiom_dossier.sections`, `axiom_dossier.journal`,
`axiom_dossier.dossier`.*

The narrated HYPER-3 report in `packages/axiom-dossier/examples/out/hyper3-full.html`
read as seven separate memos about one trial. Counted over its prose sections:

| appears in | Abs | Intro | Meth | Res | Check | Disc | Concl | Limit |
|---|---|---|---|---|---|---|---|---|
| "dropout is ignorable given baseline" | 1 | 1 | 1 | · | 2 | 1 | 1 | 1 |
| "the pooled contrast answers the question" | 1 | 1 | 1 | · | 2 | 1 | 1 | 1 |
| each of the five estimates, with interval | ✓ | · | · | ✓ | · | ✓ | ✓ | · |
| prose naming a Figure or a Table | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Three separable faults, and only the first is about prompting. A live run
against the model then found four more (D23.6) and one bug (D23.7).

## D23.1 — A prohibition against a visible fact is a hope, not a constraint

`evidence_brief` handed every section the whole study, and `SECTION_INSTRUCTION`
tried to claw the territory back in prose: *"Do not state the estimate"*, *"Do
not restate the design"*. The module docstring conceded the design as it stated
it — *"The facts block is the whole study so that you can get the wording right;
it is not a list of things to mention."*

That does not work, and the Model-checking section is the proof. Told to cover
"what it does and does not establish", it filled the negative half of the clause
by reciting both standing assumptions twice.

So the brief is scoped instead. `narrate.Licence` is a record of what one section
may state, `narrate.LICENCE` grants each section its share, and
`evidence_brief(evidence, licence=...)` emits only that. **A fact a section
cannot see is a fact it cannot repeat**, which is a guarantee where an
instruction was a request.

| section | is shown | is not shown |
|---|---|---|
| introduction | question, the answering quantity's threshold | any estimate |
| methods | route + reason, steps, assumption names | finding and diagnostic values |
| results | every finding with its interval | verdict, assumptions, steps |
| diagnostics | diagnostics only | finding values, assumptions |
| discussion | finding labels + threshold readings, verdict | **the intervals** |
| conclusions | the pivotal finding's number, other labels | **every other number** |
| limitations | assumptions in full | findings, steps |
| abstract | everything except the estimates, plus the pivotal one | the other four estimates |

The three withholdings in bold carry the note. Exactly one section holds the
intervals and exactly one holds the assumption statements; every other names the
thing and points at the section that argues it.

The checks are deliberately *not* scoped to match. `numbers.unverified` and
`claims.unlicensed` still run against the whole record: the licence decides what
the model can repeat, the checks decide what it may publish, and narrowing the
first must not narrow the second — otherwise a sentence legitimately carrying a
number from elsewhere would be rejected for having escaped its scope.

## D23.2 — Sections are narrated in document order, and told what came before

Each section was an independent call that knew of no other, so each re-laid its
own ground. `narrate.ESTABLISHED` names what each section leaves behind, and
`build` threads the list forward as an "already established" block. It is
assembled from which sections have run — not from what the model returned — so it
costs no call and cannot drift.

## D23.3 — Exhibits are numbered before narration, not after

`_number_captions` ran at the very end, so no prompt could name "Figure 2" and no
sentence in twelve reports ever did. It now runs before narration and returns its
numbering, and each section is told which exhibits it owns and asked to name each
once, which is the sentence the UCSD guide asks for by name: *"These results are
displayed in Figure 1."* Gathered exhibits sit after the text and so are never
seen by that pass; `_gathered_citations` maps them back to the body section that
should cite them, using the same `_EXHIBITS` table that decides where an exhibit
would have been embedded.

## D23.4 — The drafts had to move too, because narration cannot remove content

Narration is bounded by *"Cover what the draft covers and nothing else"*, so a
scoped brief over an unscoped draft just hands the model the numbers it was
denied. Four changes, each of which also improves the no-model document:

- `interpret.reading_of` takes `stats`. The discussion reads every finding
  without its interval — the guide asks that section to "repeat the results
  section in simpler terms and without referring to stats", which is not a
  stylistic preference: the second copy of a table is where a reader stops.
- `interpret._agreeing` buckets findings by the reading they share, so four
  findings that agree are read in one sentence and the one that disagrees gets
  the words. Four identically shaped sentences was the discussion's real fault,
  not that it read every finding.
- `interpret.pivotal` names the finding an answer turns on — the unfavourable
  one, else the first. It is the single quantity the conclusions and the abstract
  are allowed to state, and it is the same rule `contested` already enforced.
- `sections.results_section` writes a sentence per finding rather than for the
  lead alone. The narrator sees only paragraphs, so a finding absent from the
  draft was absent from the narrated results — which is how four of five
  estimates came to be stated first in the discussion.

## D23.5 — No emphasis outside a heading

The guide's FAQ is explicit: *"Do not emphasize things with boldface (except
headings)."* The generated drafts opened paragraphs with `**Question.**`,
`**Objective.**`, `**Not with one number.**`. Those are gone from every style,
and `_fallback_abstract` is one paragraph of continuous prose answering the
guide's five questions rather than four bolded labels. A contract test asserts
no non-heading block contains `**`.

## D23.6 — Four things only a live run showed

Running it against `gemini-3.7-flash` found four faults that neither the tests
nor the deterministic drafts could have.

**The abstract needed a licence too.** It began with `FULL_LICENCE`, on the
reasoning that an abstract has nothing to repeat. Told to state the headline and
name the rest, the light model stated all five estimates with their intervals and
the abstract became the results section with its headings removed.
`ABSTRACT_LICENCE` withholds the results table and grants the pivotal magnitude
alone — the same rule as the conclusions, for the same reason.

**A directive inside the facts block gets published.** The brief said
"UNRESOLVED ASSUMPTIONS: 2. They are named in the conclusions and stated in full
in the limitations section. Say how many there are and point the reader there",
and the discussion published "Those 2 unresolved assumptions are named in the
conclusions and stated in full in the limitations section." The facts block now
carries only facts; every directive lives in the instruction.

**Verbosity cannot buy content a section does not have.** Asked for twelve
sentences, the introduction — which holds one question and one threshold — wrote
the threshold rule four ways. `_budget` caps the target at the draft's own
sentence count plus two, so `full` still buys a longer methods section and no
longer pads a short one. Padding is repetition inside a section rather than
across them, and it is the same fault wearing different clothes.

**A check the prompt never names is a tripwire, not a rule.** `claims.unlicensed`
rejected a whole section over "demonstrates that" — a phrase the model reached for
meaning nothing stronger than "shows". The system prompt now lists every phrase
in `CLAIM_WORDS`, generated from that table so it cannot drift, and says what to
write instead. Rejections across the two flagship samples went from one in nine
to zero.

## D23.7 — A bug this found: the provenance section was rendering empty

Not a prompting fault, and pre-existing. `build` computed `available` — the set
of context keys a section may name an exhibit from — as `set(exhibit_context)`
alone, while the render context is `{**context_for(evidence), **exhibit_context}`.
So the three tables `context_for` supplies were dropped from every report:
the standing assumptions in the methods, and **both tables of the provenance
appendix**. A document whose fourth rule is that every number carries its
provenance was rendering that section as a lone paragraph. `available` is now the
union, and the committed samples went from three tables to six.

## D23.8 — Three ways the numeric check refused prose the report itself wrote

Two new case-study reports (`examples/rutherford.py`, `examples/tutor60.py`) put
the checks in front of material the HYPER-3 report does not have, and each one
found the same fault in a different place: **`licensed_numbers` did not license
numerals the document had put in front of the model itself.** A narration that
faithfully repeated the draft was thrown away, and the provenance appendix
recorded it exactly as it records an invented number.

- **Fields the draft renders and the licence skipped.** `strings_of` walked
  `step.title`, `what`, `why` and `detail` — but not `instead`, not `readout`,
  not an exhibit's caption. All three are printed verbatim by `methods_section`,
  and `instead` is the sentence the package calls the most useful one in a
  methods section. GEIGER's rejected alternative reads "a round hole would have
  to be about ±13° wide"; the narration repeated the 13 and lost the section.
- **Counts of the record's own parts.** Every generated section states one —
  "conditional on 2 unresolved assumptions", "4 step(s) are recorded in full
  below" — and none is stored anywhere as a number. `licensed_numbers` now
  includes the lengths of findings, diagnostics, quantities, steps, assumptions,
  unresolved, remarks and ledger. A report may count what it holds.
- **The exhibit numbers D23.3 asks for.** The instruction says to write "shown
  in Table 4"; the check then rejected the section for the numeral 4. The
  numbers in the labels a section was handed are passed to `unverified` as
  `allow`, so the check no longer fights the instruction. This one was
  self-inflicted and arrived with the cross-referencing.

Across the six narrated sample documents the rejection count went 8 → 0.

## D23.9 — Two phrasings that only a non-trial report exposes

- `reading_of` closed every unfavourable-or-neutral reading with "The estimate is
  an increase relative to no effect" — true only when the threshold *is* no
  effect. GEIGER's threshold is the radius a diffuse atom would have, so the
  clause stated a comparison nobody made. It is now written only when the
  threshold is zero.
- The non-causal subject was "the observed difference", which says the quantity
  is a contrast between groups: true of a trial arm, false of a nuclear radius.
  It is "the estimate", which keeps the causal/not-causal distinction the verdict
  carries without asserting a contrast that does not exist.
- `sentence()` capitalises a recorded `reason` or `statement` before a section
  appends it after a full stop. These are written as clauses, and every report —
  HYPER-3 included — was printing them as sentences beginning in lower case.

## D23.10 — The graph and the equations were missing, and they are the argument

Two things a causal report cannot be read without, and neither was in the record.

**The graph.** `EvidenceBuilder.verdict` already mined an `IdentificationVerdict`
for its route, adjustment set, mediators and instrument — but not for the graph
those were read off. The report said "identified via the backdoor route" and gave
a reader nothing to disagree with except the conclusion. `GraphRecord` now holds
the nodes, the directed and bidirected edges, the unmeasured nodes, the
treatment/outcome pair and the graph's own content hash, read structurally so
this package still does not import `axiom.identify`. It renders three ways: the
edge list in the identification step's prose, a table of arrows beside a
paragraph saying what an arrow claims, and a figure — delegated to
`viz.causal_graph`, which already draws unmeasured nodes hollow and bidirected
edges dashed, because those two are what decide identifiability. The four
attributes it reads are exactly what `GraphRecord` carries.

**The equations.** A methods section that says "fitted by ANCOVA" has told the
reader the name of a thing rather than the thing. `MethodStep.equations` renders
in the monospaced form the readouts use, since `report` has no math block and
the three renderers have no LaTeX between them. That is worse than typeset
mathematics and much better than what the package had.

TUTOR-60 carries both of its graphs: the observational one where a school's
unmeasured capacity drives what it buys and how its pupils do — `downgraded`,
no adjustment recovers it — and the randomized one where those two arrows are
gone. The rejected alternative in the design step is the first graph, which is
the clearest statement of what the trial was bought for. GEIGER has no graph at
all, and the blocks correctly do not render.

## D23.11 — Two bugs the equations found

**A code span was not literal.** `report.render.html._inline` substituted the
code spans and *then* ran the bold and italic passes across the whole result, so
`beta0 + tau * treated_i + beta1 * baseline_i` lost both asterisks to an `<em>`.
The PDF renderer had the same fault. Both now apply emphasis outside code spans
only. This was shipping before the equations existed — any recorded `readout`
containing a `*` was already being mangled.

**Narration deleted them.** `_with_prose` replaced every `Paragraph` in a
section, and an equation is a paragraph of backticked lines. So every narrated
document lost its mathematics and its printed output — the exhibits survived
because they are `Table` and `Figure`, and the equations did not because the
package has no block type for them. `_is_verbatim` now recognises a monospaced
paragraph and puts it on the list of things narration may not touch, beside the
metrics and the tables. It is not prose: it is the record showing itself.

## What it cost, measured the same way

The narrated HYPER-3 journal report, live, every section verified:

| in prose | before | after |
|---|---|---|
| "dropout is ignorable given baseline" | 8 sections | 3 (abstract, conclusions, limitations) |
| "the pooled contrast answers the question" | 8 sections | 3 |
| the five estimates with intervals | abstract, results, discussion, conclusions | results; the pivotal one also in abstract and conclusions |
| sentences naming a Figure or Table | 0 | 5 |
| sections narrated and verified | 8 of 8 | 8 of 8 |

The three surviving assumption mentions each do work: the abstract and the
conclusions are read on their own, and the limitations is where they are argued.

## Open

- `plain` style numbers no captions, so a plain readout cites no exhibits. That
  is consistent — there is no number to cite — but a readout might still deserve
  "the table below".
- The example's diagnostic labels arrive as "band age 25 35" rather than
  "age 25-35", which the narration faithfully reproduces. That is in
  `examples/hypertension.py`, not here.
- `interpret`'s vocabulary is axiom's mandated one — treatment, dose, unit,
  outcome — so an unidentified GEIGER reads "Reading them as the effect of the
  treatment requires an assumption this analysis does not supply." There is no
  treatment in a scattering experiment. Rule 2 makes this the right vocabulary
  for the package's domain and GEIGER is the edge of that domain; recorded rather
  than worked around.
