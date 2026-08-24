"""Rewriting the generated sections into prose, and refusing the result when it drifts.

The model is never asked to analyse anything. It is given the evidence and a
draft that is already correct, and asked to make the draft read better. That is
a narrow job, and narrowing it is what makes the output usable: there is no
question the model has to *answer*, so there is nothing for it to guess at.

Then the result is checked. Every numeral in the returned text must trace to the
evidence record (``numbers.unverified``). If any does not, the narration is
**rejected and the deterministic draft is kept** — the ``Narration`` says so,
names the offending literals, and the document renders the version that was
right. A failure here costs prose quality and never correctness.

    narrator = Narrator(prose=Gemini(), light=Gemini(LIGHT_MODEL))
    out = narrator.section(methods_section(ev), ev)
    out.verified      # False -> out.text is the original draft
    out.rejected      # the literals the model produced that nothing licenses

**Each section is shown only the facts it may state.** That is the whole of the
anti-repetition mechanism, and it replaces an earlier design that showed every
section the entire study and then forbade most of it in prose. Prohibitions
against a visible fact fail: a model handed five estimates and told not to
mention them mentions them. So ``LICENCE`` gives the discussion the *readings*
of the findings and withholds their intervals, gives the conclusions one
number, and gives the assumption statements to the limitations section alone.
A fact a section cannot see is a fact it cannot repeat.

The sections are narrated in document order and each is told what the ones
before it established, so it can refer back in a clause rather than re-laying
the ground. That note is assembled mechanically from which sections have run;
nothing about it costs a call or depends on what the model said.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from axiom.core import NonEmptyStr, Spec, Unsupported
from axiom.report import Heading, Paragraph, Section

from axiom_dossier import claims, numbers
from axiom_dossier.evidence import Evidence, Quantity
from axiom_dossier.interpret import pivotal
from axiom_dossier.language import LIGHT_MODEL, Gemini, LanguageModel
from axiom_dossier.sections import VERBOSITY, Verbosity, standing_assumptions

__all__ = [
    "ABSTRACT_LICENCE",
    "ESTABLISHED",
    "FULL_LICENCE",
    "LICENCE",
    "SECTION_INSTRUCTION",
    "Licence",
    "Narration",
    "Narrator",
    "established_note",
    "evidence_brief",
    "exhibit_note",
    "narrate_text",
    "pivotal",
]


def _checked_words() -> str:
    """The claim vocabulary, listed for the prompt rather than left as a tripwire.

    ``claims.unlicensed`` rejects a whole rewrite over one of these, and the
    model was never told which words they are — so a discussion would write
    "demonstrates that the pooled estimate settled below the threshold", meaning
    nothing stronger than "shows", and lose the section to the check. Naming
    them turns a hidden rule into a followable one, and generating the list from
    ``CLAIM_WORDS`` keeps the prompt honest when that table changes.
    """
    return "; ".join(
        f"{kind}: {', '.join(sorted(phrases))}"
        for kind, phrases in sorted(claims.CLAIM_WORDS.items())
    )


SYSTEM = f"""You rewrite one section of a statistical research report.

You are given three things: the facts that section may state (between the
<<facts>> markers), a note on what the sections before it already established,
and a draft to rewrite. Rewrite the
draft so it reads as a section of a serious empirical paper: plain, declarative,
no marketing register, no hedging beyond what the facts state.

Absolute rules:
- Never introduce a number that is not in the facts. Not a rounded one, not a
  restated one, not an approximate one, not a percentage you computed.
- Never claim a result is significant, causal, robust, proven or confirmed
  unless the facts already say so in those words. These are checked, and a
  single one that the record does not make gets the whole rewrite thrown away.
  The checked phrases are, in full — {_checked_words()}. Avoid every one of
  them unless it already appears in the facts, including where you mean
  something innocuous by it: "demonstrates that" is rejected exactly as
  "proves" is. Write "shows", "indicates", "is", or name what happened.
- An interval that contains the decision threshold means the question is
  unsettled. It does not mean there is no effect. Never write it as one.
- Never add a citation, a p-value, a sample size, or a date.
- Keep every interval attached to its estimate.
- Return prose only. No headings, no bullet lists, no markdown fences, and no
  bold or italic emphasis -- a heading is the only emphasis this report uses.

The facts block is everything you may state and the whole of it. It is the
section's share of the study, not the study. If some fact you would like to
lean on is missing from it, that is deliberate: another section carries it.
Refer to that section rather than reconstructing the fact from what you can see.

Anything listed under "already established" has been written. Refer back to it
in a clause -- "the backdoor route identified in the methods" -- and never
restate it. Restating an established fact is the single most common way these
drafts go wrong, and it is what makes a report read as seven separate memos.
"""

#: What each generated section is for, as the move it makes rather than as a
#: list of prohibitions. Each ends in the "Big Picture" test the UCSD guide
#: gives for that section, which is the sentence to write against when the rest
#: of the instruction runs out.
SECTION_INSTRUCTION = {
    "introduction": (
        "Write the Introduction.\n"
        "The move: state the question, then state what would count as an answer. "
        "Name the quantity that answers it and the value the decision turns on, "
        "and say what an interval on one side of that value settles and what an "
        "interval spanning it does not.\n"
        "Do not state any estimate -- the study's own results are not part of "
        "its introduction. Do not speculate about why the question matters; the "
        "record does not know, and inventing a motivation is the one liberty "
        "this section invites.\n"
        "Test: what question will this report try to answer, and how?"
    ),
    "methods": (
        "Write the Methods section.\n"
        "The move: describe what was done, in the order it was done, so that "
        "someone could do it again. Where the draft records an alternative that "
        "was considered and rejected, keep it -- the rejected alternative is the "
        "most useful sentence in a methods section and the one most often cut.\n"
        "Cover the design, how the effect is identified, and how it was "
        "estimated. State the assumptions by name and by what each one licenses; "
        "the limitations section argues about them and this one only declares "
        "them.\n"
        "Where the facts give an identification graph, name the arrows the route "
        "turns on rather than only the route's name -- a reader who disagrees has "
        "to be able to find the arrow they disagree with. Where they give an "
        "equation, say what it estimates; the equation itself is set beside your "
        "prose and you do not need to transcribe it.\n"
        "Do not state any estimate, any interval, or any diagnostic.\n"
        "Test: what would a researcher need to do, and in what order, to "
        "replicate this analysis?"
    ),
    "results": (
        "Write the Results section. This is the one section that carries the "
        "numbers, so carry all of them.\n"
        "The move: report each estimated quantity with its interval, exactly as "
        "the facts give it. Every quantity in the facts gets a sentence; a "
        "finding reported in a metric block and nowhere in the prose is a "
        "finding the reader skims past.\n"
        "Open by naming the comparison the numbers answer, not by re-describing "
        "the design.\n"
        "Report what was measured, not what it means. No sentence here may "
        "begin 'This means', 'This shows', or 'This suggests', and none may say "
        "whether a result is good news. The discussion does that work, and this "
        "section is judged on staying out of it.\n"
        "Test: could a reader reconstruct every number in the study from this "
        "section alone?"
    ),
    "diagnostics": (
        "Write the Model checking section.\n"
        "The move: report what was checked about the fit, and say what a check "
        "of this kind can and cannot do -- it can lower confidence in an "
        "estimate and can never establish one.\n"
        "Say that once, in one sentence. Do not enumerate the standing "
        "assumptions to illustrate it; they are named in the methods and argued "
        "in the limitations, and listing them a third time here is what turns "
        "this section into a restatement of those two.\n"
        "Do not restate the headline effect.\n"
        "Test: what about this analysis was examined, and what is still "
        "unexamined?"
    ),
    "discussion": (
        "Write the Discussion.\n"
        "The move: say in plain English what was found, and then what it "
        "licenses.\n"
        "You have deliberately not been given the intervals. Do not approximate "
        "them and do not reconstruct them -- the results section carries them, "
        "and a discussion that repeats them is the most common way this report "
        "becomes unreadable. Refer to a quantity by its label and by which side "
        "of the threshold it settled on.\n"
        "Open on the most consequential finding rather than the first one "
        "listed. Where the findings agree with one another, say so once and "
        "move on; where they disagree, that disagreement is the discussion and "
        "is worth most of the words.\n"
        "State what the identification licenses and what it does not. Refer to "
        "the standing assumptions by count and send the reader to the "
        "limitations section; do not restate what each one claims.\n"
        "Keep every hedge in the draft: an interval spanning the threshold "
        "means the question is unsettled, never that there is no effect.\n"
        "Test: what did we learn, and what new question does it raise?"
    ),
    "conclusions": (
        "Write the Conclusions.\n"
        "The move: answer the question in the first sentence, then say what the "
        "answer is conditional on, then stop.\n"
        "You have been given one number and no others. That is the answer's "
        "magnitude; every other quantity is referred to by label alone. A "
        "conclusions section that lists the findings again has become a second "
        "results section.\n"
        "Be brief. A conclusions section that runs long has started arguing, "
        "and the argument belongs in the discussion above it.\n"
        "Test: what is the answer, and under what does it hold?"
    ),
    "limitations": (
        "Write the Limitations section. This is the only section that states "
        "what the standing assumptions claim, so state them here in full.\n"
        "The move: for each assumption, say what it claims, what it licenses, "
        "and what would challenge it. An assumption a reader cannot disagree "
        "with specifically is one they will dismiss generally.\n"
        "Where an assumption is known to fail, lead with that rather than "
        "listing it alongside the ones that merely have not been checked.\n"
        "Do not restate the design and do not repeat any estimate.\n"
        "Test: what would make a reader doubt this analysis, and what would "
        "settle the doubt?"
    ),
}

#: What each section leaves behind for the ones after it. Threaded forward as
#: the "already established" note, so section six can point at section two
#: instead of rebuilding it. Phrased as the fact rather than as the section's
#: title, because a model refers back more naturally to a thing than to a number.
ESTABLISHED = {
    "introduction": "the question, and the value the decision turns on",
    "methods": (
        "the design, how the effect is identified, and the assumptions the " "analysis rests on"
    ),
    "results": "every estimated quantity, with its interval",
    "diagnostics": "what was checked about the fit",
    "discussion": "what each finding means and what the identification licenses",
    "conclusions": "the answer to the question",
    "limitations": "what each standing assumption claims and what would challenge it",
    "remarks": "the analyst's own account of what the run showed",
}


@dataclass(frozen=True)
class Licence:
    """What one section may state — the whole of the anti-repetition mechanism.

    Every field withholds rather than adds. The default is a section that may
    say nothing, and ``LICENCE`` grants each one its share; ``FULL_LICENCE``
    grants everything and is what the abstract gets, because an abstract is a
    compression of the whole study and has nothing to repeat.

    The two that do the most work are ``finding_values`` and
    ``assumption_detail``. Exactly one section holds each: results carries the
    numbers, limitations carries the assumption statements. Every other section
    sees a label or a name and refers to the section that has the substance.
    """

    question: bool = False
    #: The identification route and the reason for it. Methods and discussion.
    verdict: bool = False
    #: The status word alone — "identified", "unidentified" — with no reason.
    verdict_status: bool = False
    steps: bool = False
    #: The identification graph and the estimating equations. The methods
    #: section, and nothing else: they are what a reader checks the analysis
    #: against, and repeating them in a discussion is repeating the whole of it.
    mechanics: bool = False
    #: Label, value, interval. The results section, and nothing else.
    finding_values: bool = False
    #: Label and where the interval sits relative to the threshold, no numbers.
    finding_readings: bool = False
    #: The one finding the conclusion turns on, with its number.
    pivotal_value: bool = False
    #: The lead finding's threshold and unit, for stating what would count as
    #: an answer before any answer is given.
    lead_threshold: bool = False
    diagnostics: bool = False
    #: Names only. The methods declares them by name, and so does a conclusions
    #: section, which is read on its own as often as the abstract is.
    assumption_names: bool = False
    #: How many are standing, and nothing else. Enough to say the reading is
    #: conditional and to send the reader somewhere; not enough to list them a
    #: fourth time.
    assumption_count: bool = False
    #: Statement, state and what would challenge it. Limitations, and nothing else.
    assumption_detail: bool = False
    remarks: bool = False


FULL_LICENCE = Licence(
    question=True,
    verdict=True,
    verdict_status=True,
    steps=True,
    mechanics=True,
    finding_values=True,
    finding_readings=True,
    pivotal_value=True,
    lead_threshold=True,
    diagnostics=True,
    assumption_names=True,
    assumption_count=True,
    assumption_detail=True,
    remarks=True,
)
"""Everything. The default for a caller who names no section."""

ABSTRACT_LICENCE = replace(
    FULL_LICENCE, finding_values=False, finding_readings=True, pivotal_value=True
)
"""Everything except the results table.

An abstract has nothing to repeat -- it is written before the report it
summarizes and read instead of it -- so it is the one section granted the
verdict, the steps, the diagnostics and the assumption statements at once. What
it must not have is every estimate: told to state the headline and name the
rest, a light model handed all five stated all five, and the abstract became the
results section with its headings removed. It sees the reading of each finding
and the magnitude of exactly one."""

#: Section key -> what it may state. Read this table beside ``ESTABLISHED``:
#: together they say what each section holds and what it hands on.
LICENCE: dict[str, Licence] = {
    "introduction": Licence(question=True, verdict_status=True, lead_threshold=True),
    "methods": Licence(
        question=True, verdict=True, steps=True, mechanics=True, assumption_names=True
    ),
    "results": Licence(finding_values=True),
    "diagnostics": Licence(diagnostics=True),
    "discussion": Licence(
        question=True,
        verdict=True,
        finding_readings=True,
        # The count, not the names. The limitations section argues them and the
        # conclusions names them; a discussion that also lists them is the
        # fourth telling, and it is the one a reader skips.
        assumption_count=True,
    ),
    "conclusions": Licence(
        question=True,
        verdict_status=True,
        finding_readings=True,
        pivotal_value=True,
        assumption_names=True,
    ),
    "limitations": Licence(verdict_status=True, assumption_detail=True),
    "abstract": ABSTRACT_LICENCE,
}


class Narration(Spec):
    """One narrated block, with whether it survived the numeric check.

    ``text`` is always safe to render: it is the model's version when that
    version checked out and the original draft when it did not.
    """

    key: NonEmptyStr
    text: str
    model: str = ""
    verified: bool = True
    narrated: bool = False
    rejected: tuple[str, ...] = ()
    overclaimed: tuple[str, ...] = ()

    def summary(self) -> str:
        if not self.narrated:
            return f"{self.key}: generated text, not narrated"
        if self.verified:
            return f"{self.key}: narrated by {self.model}, every numeral and claim traced"
        faults = []
        if self.rejected:
            faults.append(f"untraceable numeral(s): {', '.join(self.rejected)}")
        if self.overclaimed:
            faults.append(f"unlicensed claim(s): {', '.join(self.overclaimed)}")
        return (
            f"{self.key}: narration by {self.model} rejected "
            f"({'; '.join(faults)}); kept the generated text"
        )


def _reading(q: Quantity) -> str:
    """Where one interval sits, in words and with no numbers in it."""
    side = q.against_threshold()
    if side == "no threshold":
        return "no decision threshold was recorded, so it is reported and not judged"
    if side == "no interval":
        return "no interval was recorded"
    if side == "spans":
        return "the interval contains the threshold, so the question is unsettled"
    where = "below" if side == "below" else "above"
    good = q.is_beneficial()
    if good is True:
        return f"settled {where} the threshold, on the favourable side"
    if good is False:
        return f"settled {where} the threshold, on the unfavourable side"
    return f"settled {where} the threshold"


def evidence_brief(evidence: Evidence, *, licence: Licence = FULL_LICENCE) -> str:
    """The facts, flattened into the block the prompt shows the model.

    Deliberately terse and complete *within the licence*: every quantity the
    section may state, with its interval, and every assumption it may argue
    about, with its state. The model cannot be trusted to ask for something it
    was not given, so it is given everything it is allowed to say — and nothing
    else, because a fact in front of it is a fact it will find a use for.

    The default licence is everything, which is what the abstract wants and
    what a caller narrating a section of their own gets when they say nothing.
    """
    lines: list[str] = [f"TITLE: {evidence.title}"]
    if licence.question and evidence.question:
        lines.append(f"QUESTION: {evidence.question}")

    if evidence.verdict is not None:
        if licence.verdict:
            # A verdict may carry no reason, and a trailing em dash with nothing
            # after it reads to a model as a fact that got cut off.
            because = f" — {evidence.verdict.reason}" if evidence.verdict.reason else ""
            lines.append(
                f"IDENTIFICATION: {evidence.verdict.status} "
                f"via {evidence.verdict.route or 'no route'}{because}"
            )
        elif licence.verdict_status:
            lines.append(f"IDENTIFICATION: {evidence.verdict.status}")

    if licence.finding_values and evidence.findings:
        lines.append("FINDINGS:")
        lines.extend(f"  - {q.label}: {q.stated()}" for q in evidence.findings)
    elif licence.finding_readings and evidence.findings:
        # Labels and readings, no magnitudes. A section given this cannot
        # restate the results table because it has never seen one.
        lead = pivotal(evidence) if licence.pivotal_value else None
        lines.append("FINDINGS (the numbers themselves are in the results section):")
        for q in evidence.findings:
            # The pivotal finding is marked by being the only one that arrives
            # with a number, not by a label. A label gets published: the earlier
            # "THE FINDING THE ANSWER TURNS ON:" line came back as the sentence
            # "the finding the answer turns on shows that ...".
            stated = f" — {q.stated()}" if lead is not None and q.key == lead.key else ""
            lines.append(f"  - {q.label}: {_reading(q)}{stated}")

    if licence.lead_threshold and evidence.findings:
        lead = evidence.findings[0]
        if lead.threshold is None:
            lines.append(
                f"THE QUESTION IS ANSWERED BY: {lead.label}. No decision threshold "
                "was recorded for it."
            )
        else:
            unit = f" {lead.unit}" if lead.unit else ""
            lines.append(
                f"THE QUESTION IS ANSWERED BY: {lead.label}, against a decision "
                f"threshold of {lead.threshold:.{lead.precision}f}{unit}."
            )

    if licence.diagnostics and evidence.diagnostics:
        lines.append("DIAGNOSTICS:")
        lines.extend(f"  - {q.label}: {q.stated()}" for q in evidence.diagnostics)

    if licence.mechanics and evidence.graph is not None and evidence.graph.edges:
        graph = evidence.graph
        lines.append(f"IDENTIFICATION GRAPH: {graph.to_text()}")
        if graph.treatment and graph.outcome:
            lines.append(f"  the target is the effect of {graph.treatment} on {graph.outcome}")
        if graph.unmeasured:
            lines.append(f"  not measured: {', '.join(graph.unmeasured)}")

    if licence.steps and evidence.steps:
        lines.append("STEPS:")
        for step in evidence.steps:
            lines.append(f"  - {step.title}: {step.what} {step.why}".rstrip())
            if step.instead:
                lines.append(f"      considered instead: {step.instead}")
            if licence.mechanics and step.equations:
                lines.extend(f"      equation: {eq}" for eq in step.equations)

    standing = evidence.unresolved()
    if standing:
        # Written as English, not as identifiers: a model handed
        # "no_unmeasured_confounding" will faithfully print the underscores into
        # the abstract, and a paper does not contain snake_case.
        names = [name.replace("_", " ") for name in standing]
        if licence.assumption_count and not (licence.assumption_detail or licence.assumption_names):
            # A fact, not an instruction. The earlier wording carried "say how
            # many there are and point the reader there" inside the facts block,
            # and the discussion published that sentence almost verbatim.
            lines.append(
                f"UNRESOLVED ASSUMPTIONS: {len(names)}. What each one claims is "
                "stated in the limitations section."
            )
        elif licence.assumption_detail:
            lines.append("UNRESOLVED ASSUMPTIONS:")
            for a in standing_assumptions(evidence):
                challenged = f" Challenged by: {a.challenged_by}." if a.challenged_by else ""
                lines.append(
                    f"  - {a.name.replace('_', ' ')} ({a.state}): {a.statement}{challenged}"
                )
        elif licence.assumption_names:
            lines.append(
                f"UNRESOLVED ASSUMPTIONS: {len(names)}, named {', '.join(names)}. "
                "What each one claims is stated in the limitations section."
            )

    if licence.remarks and evidence.remarks:
        lines.append("WHAT THE ANALYST SAID THE RUN SHOWED:")
        lines.extend(f"  - {text}" for text in evidence.remarks)
    return "\n".join(lines)


def established_note(keys: Sequence[str]) -> str:
    """What the sections before this one put on the record, as a prompt block.

    Empty for the first section narrated, which is right: there is nothing
    behind it to refer to, and a note saying so would only invite the model to
    remark on the absence.
    """
    known = [(k, ESTABLISHED[k]) for k in keys if k in ESTABLISHED]
    if not known:
        return ""
    lines = ["<<already established, earlier in this report>>"]
    lines.extend(f"  - {fact} ({key})" for key, fact in known)
    lines.append("Refer back to these; do not restate them.")
    lines.append("<<end>>")
    return "\n".join(lines)


def _exhibit_numbers(exhibits: Sequence[tuple[str, str]]) -> tuple[float, ...]:
    """The numerals in the labels a section was asked to cite — "Table 4" -> 4.0."""
    out: list[float] = []
    for label, _caption in exhibits:
        out.extend(float(m) for m in re.findall(r"\d+", label))
    return tuple(out)


def exhibit_note(exhibits: Sequence[tuple[str, str]]) -> str:
    """The numbered exhibits this section owns, and the duty to point at them.

    Each pair is ``("Figure 2", "Estimated quantities with their intervals.")``.
    A report whose prose never names its own figures is one whose reader never
    looks at them, which is the whole reason the guide asks for the sentence.
    """
    if not exhibits:
        return ""
    lines = ["<<exhibits in this section>>"]
    lines.extend(f"  - {label}: {caption}" for label, caption in exhibits)
    lines.append(
        "Name each of these once, by number, in the sentence its content "
        "supports — 'These are shown in Figure 2.' Do not describe what the "
        "exhibit contains beyond what the facts already state."
    )
    lines.append("<<end>>")
    return "\n".join(lines)


def narrate_text(
    key: str,
    draft: str,
    evidence: Evidence,
    model: LanguageModel,
    *,
    instruction: str = "",
    licence: Licence | None = None,
    established: Sequence[str] = (),
    exhibits: Sequence[tuple[str, str]] = (),
    budget: int = 0,
    temperature: float = 0.2,
    verbosity: Verbosity = "standard",
) -> Narration | Unsupported:
    """Rewrite ``draft`` against ``evidence``; keep the draft if the result drifts.

    Two checks, and either one rejects. ``numbers.unverified`` catches a quantity
    the record does not hold; ``claims.unlicensed`` catches an assertion it does
    not make — "robust", "significant", "proves" — which is the failure mode a
    discussion or conclusions section invites and which carries no numeral for
    the first check to find.

    Note that the checks run against the *whole* record, not against the
    section's licence. The licence decides what the model is shown and so what
    it can repeat; the checks decide what it may publish. Narrowing the first
    does not narrow the second, so a sentence that legitimately carries a
    number from elsewhere is still accepted rather than rejected for having
    escaped its scope.

    ``Unsupported`` only when the model could not be reached at all: a missing
    extra, a missing key, an empty response. A model that answers and overreaches
    is not ``Unsupported``, it is a ``Narration`` with ``verified`` false,
    because the document can still be produced.
    """
    target = budget or _budget(draft, int(VERBOSITY[verbosity]["sentences"]))
    granted = licence if licence is not None else LICENCE.get(key, FULL_LICENCE)
    # The marker stays ``<<facts>>``: it is the seam ``language.Offline`` reads
    # the licensed block back out of, and renaming it turned every offline
    # narration into an echo of the whole prompt -- sentence targets included,
    # which the numeric check then rejected as invented quantities.
    parts = [
        "THE FACTS THIS SECTION MAY STATE:\n"
        f"<<facts>>\n{evidence_brief(evidence, licence=granted)}\n<<end>>"
    ]
    prior = established_note(established)
    if prior:
        parts.append(prior)
    shown = exhibit_note(exhibits)
    if shown:
        parts.append(shown)
    parts.append(instruction or "Rewrite the draft below.")
    parts.append(
        f"Aim for about {target} sentences; cover the draft's content rather than "
        "padding to length."
    )
    parts.append(f"DRAFT:\n{draft}")
    produced = model.generate("\n\n".join(parts), system=SYSTEM, temperature=temperature)
    if isinstance(produced, Unsupported):
        return produced
    text = produced.strip()
    if not text:
        return Unsupported(
            reason=f"{model.name} returned an empty rewrite", detail={"section": key}
        )
    # The exhibit numbers this section was told to cite are licensed for it.
    # Asking a section to write "shown in Table 4" and then rejecting it for the
    # numeral 4 is the check fighting the instruction, and it is what kept
    # throwing away the one section whose exhibit happened to be numbered past
    # the quantities in the record.
    bad_numbers = numbers.unverified(text, evidence, allow=_exhibit_numbers(exhibits))
    bad_claims = claims.unlicensed(text, evidence)
    if bad_numbers or bad_claims:
        return Narration(
            key=key,
            text=draft,
            model=model.name,
            verified=False,
            narrated=True,
            rejected=tuple(dict.fromkeys(lit.text for lit in bad_numbers)),
            overclaimed=tuple(dict.fromkeys(c.text.lower() for c in bad_claims)),
        )
    return Narration(key=key, text=text, model=model.name, verified=True, narrated=True)


_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def _budget(draft: str, configured: int) -> int:
    """How many sentences to ask for: the verbosity target, capped by the draft.

    Verbosity buys *content*, and a section with little content cannot spend it.
    The introduction holds one question and one threshold; asked for twelve
    sentences it wrote the threshold rule four ways -- "an interval spanning
    0.00 mmHg does not settle the question", then "an interval that contains the
    decision threshold does not mean that there is no effect", and so on. That is
    padding, and padding is repetition inside a section rather than across them.

    A little expansion is right, because prose that flows takes more words than a
    generated draft does. Three times the draft is not expansion.
    """
    present = len(_SENTENCE_END.findall(draft)) or 1
    return max(3, min(configured, present + 2))


def _is_verbatim(block: object) -> bool:
    """Whether a paragraph is a monospaced block rather than prose.

    ``readout_text`` and ``equation_text`` build a paragraph whose every line is
    wrapped in backticks, which is how the package sets fixed-pitch material in
    a renderer that has no code block. Those are not prose: they are the record
    showing itself — the estimating equation, the output as it was printed — and
    they belong with the metrics and the tables on the list of things narration
    may not touch.
    """
    if not isinstance(block, Paragraph) or not block.text:
        return False
    lines = [line for line in block.text.splitlines() if line.strip()]
    return bool(lines) and all(
        line.strip().startswith("`") and line.strip().endswith("`") for line in lines
    )


def _draft_of(section: Section) -> str:
    """The prose paragraphs of a section, joined — what there is to rewrite."""
    return "\n\n".join(
        b.text
        for b in section.blocks
        if isinstance(b, Paragraph) and b.text and not _is_verbatim(b)
    )


def _with_prose(section: Section, text: str) -> Section:
    """The section with its paragraphs replaced by one narrated block.

    Metrics, tables, figures and ledgers are left exactly where they were: those
    carry the numbers, and the model does not get to rearrange them. So are the
    monospaced blocks — the equations and the printed readouts — for the same
    reason: replacing every paragraph in the section deleted the estimating
    equation from every narrated report, which is the one thing a methods
    section exists to show.

    Headings need a rule of their own. Narration merges a section's paragraphs
    into one flowing block, so a heading that introduced nothing *but* prose has
    been left with nothing to introduce and goes with it. A heading that also
    introduces a table or a figure still has a job and stays. Without this a
    narrated methods section renders as a stack of empty subheadings.
    """
    blocks = list(section.blocks)
    orphaned: set[int] = set()
    i = 0
    while i < len(blocks):
        if not isinstance(blocks[i], Heading):
            i += 1
            continue
        j = i + 1
        run = []
        while j < len(blocks) and not isinstance(blocks[j], Heading):
            run.append(blocks[j])
            j += 1
        if run and all(isinstance(b, Paragraph) and not _is_verbatim(b) for b in run):
            orphaned.add(i)
        i = j
    kept = tuple(
        b
        for k, b in enumerate(blocks)
        if k not in orphaned and (not isinstance(b, Paragraph) or _is_verbatim(b))
    )
    narrated = (Paragraph(text=text),) if text else ()
    return Section(title=section.title, blocks=(*narrated, *kept), summary=section.summary)


@dataclass(frozen=True)
class Narrator:
    """Two models: one for prose, a cheaper one for the mechanical passes.

    The split is not decoration. Rewriting a methods section is the expensive,
    quality-sensitive call; producing a one-line abstract or a caption is not,
    and paying flagship rates for it is waste. ``light`` handles the latter.
    """

    prose: LanguageModel = field(default_factory=Gemini)
    light: LanguageModel = field(default_factory=lambda: Gemini(LIGHT_MODEL))

    def section(
        self,
        section: Section,
        evidence: Evidence,
        *,
        key: str = "",
        verbosity: Verbosity = "standard",
        established: Sequence[str] = (),
        exhibits: Sequence[tuple[str, str]] = (),
    ) -> tuple[Section, Narration]:
        """Narrate one section's prose, returning the section to render and the record.

        ``established`` names the sections already written, in document order,
        so this one can refer back instead of re-laying their ground.
        ``exhibits`` names the numbered figures and tables it should point at.

        When the model is unreachable the section comes back untouched with a
        ``Narration`` that says why, so a caller can build the whole document
        with no key at all and still get a report.
        """
        name = key or section.title.lower().replace(" ", "_")
        draft = _draft_of(section)
        if not draft:
            return section, Narration(key=name, text="", model="", narrated=False)
        out = narrate_text(
            name,
            draft,
            evidence,
            self.prose,
            instruction=SECTION_INSTRUCTION.get(name, ""),
            established=established,
            exhibits=exhibits,
            verbosity=verbosity,
        )
        if isinstance(out, Unsupported):
            return section, Narration(
                key=name, text=draft, model=self.prose.name, narrated=False, rejected=(out.reason,)
            )
        if not out.verified:
            return section, out
        return _with_prose(section, out.text), out

    def abstract(
        self, evidence: Evidence, *, verbosity: Verbosity = "standard"
    ) -> Narration | Unsupported:
        """A structured abstract from the evidence alone, on the cheap model.

        Compression, not judgement — which is exactly the job the light model is
        for, and why paying flagship rates for it would be waste.

        The one section with a full licence, because it has nothing to repeat:
        it is written before the report it summarizes and read instead of it.
        Even so it is told to carry the headline rather than the whole results
        table, which is the abstract's characteristic failure.
        """
        sentences = {"brief": 3, "standard": 5, "full": 8}[verbosity]
        lead = pivotal(evidence)
        headline = (
            f" One quantity in the facts carries a number — {lead.label}. State that "
            "one with its interval and describe the others by where they settled."
            if lead is not None and len(evidence.findings) > 1
            else ""
        )
        return narrate_text(
            "abstract",
            evidence_brief(evidence, licence=ABSTRACT_LICENCE),
            evidence,
            self.light,
            licence=ABSTRACT_LICENCE,
            instruction=(
                f"Write a structured abstract of at most {sentences} sentences for the "
                "report these facts describe. Answer five questions in this order, as "
                "continuous prose and without labelling them: what was studied, what "
                "question was asked, how it was answered, what was found, and what that "
                "tells us. State how the effect is identified and the strongest standing "
                f"assumption.{headline} An abstract that lists every quantity has become "
                "a results table and is the characteristic way this section fails."
            ),
            budget=sentences,
            verbosity=verbosity,
        )
