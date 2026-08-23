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
"""

from __future__ import annotations

from dataclasses import dataclass, field

from axiom.core import NonEmptyStr, Spec, Unsupported
from axiom.report import Heading, Paragraph, Section

from axiom_dossier import numbers
from axiom_dossier.evidence import Evidence
from axiom_dossier.language import LIGHT_MODEL, Gemini, LanguageModel

__all__ = [
    "SECTION_INSTRUCTION",
    "Narration",
    "Narrator",
    "evidence_brief",
    "narrate_text",
]

SYSTEM = """You rewrite sections of a statistical research report.

You are given a block of established facts and a draft. Rewrite the draft so it
reads as a methods or results section of a serious empirical paper: plain,
declarative, no marketing register, no hedging beyond what the facts state.

Absolute rules:
- Never introduce a number that is not in the facts. Not a rounded one, not a
  restated one, not an approximate one, not a percentage you computed.
- Never claim a result is significant, causal, robust, or confirmed unless the
  facts say so in those terms.
- Never add a citation, a p-value, a sample size, or a date.
- Keep every interval attached to its estimate.
- Cover what the draft covers and nothing else. The facts block is the whole
  study so that you can get the wording right; it is not a list of things to
  mention. Material belonging to another section of the report is already
  written there, and repeating it is the single most common way these drafts
  go wrong.
- Return prose only. No headings, no bullet lists, no markdown fences.
"""

#: What each generated section is for. Without these every section is handed the
#: same complete facts block and dutifully restates the entire study, so methods,
#: results and diagnostics all open by re-describing the design and the estimate.
SECTION_INSTRUCTION = {
    "methods": (
        "Rewrite the draft as the Methods section. Cover only what was done: the "
        "design, how the effect is identified, and how it was estimated. Do not "
        "state the estimate, the intervals, or any diagnostic — those are other "
        "sections."
    ),
    "results": (
        "Rewrite the draft as the Results section. State the estimated quantities "
        "and their intervals, and nothing else. Do not re-describe the design or "
        "the identification strategy; the reader has just read them."
    ),
    "diagnostics": (
        "Rewrite the draft as the Model checking section. Cover only what was "
        "checked about the fit and what it does and does not establish. Do not "
        "restate the design or the headline effect."
    ),
    "limitations": (
        "Rewrite the draft as the Limitations section. Cover only the assumptions "
        "still standing, what each one licenses, and what would challenge it. Do "
        "not restate the design or repeat the estimate."
    ),
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

    def summary(self) -> str:
        if not self.narrated:
            return f"{self.key}: generated text, not narrated"
        if self.verified:
            return f"{self.key}: narrated by {self.model}, every numeral traced"
        return (
            f"{self.key}: narration by {self.model} rejected "
            f"({len(self.rejected)} untraceable numeral(s): {', '.join(self.rejected)}); "
            "kept the generated text"
        )


def evidence_brief(evidence: Evidence) -> str:
    """The facts, flattened into the block the prompt shows the model.

    Deliberately terse and complete: every quantity with its interval, every
    assumption with its state. The model cannot be trusted to ask for something
    it was not given, so it is given everything it is allowed to say.
    """
    lines: list[str] = [f"TITLE: {evidence.title}"]
    if evidence.question:
        lines.append(f"QUESTION: {evidence.question}")
    if evidence.verdict is not None:
        lines.append(
            f"IDENTIFICATION: {evidence.verdict.status} "
            f"via {evidence.verdict.route or 'no route'} — {evidence.verdict.reason}"
        )
    if evidence.findings:
        lines.append("FINDINGS:")
        lines.extend(f"  - {q.label}: {q.stated()}" for q in evidence.findings)
    if evidence.diagnostics:
        lines.append("DIAGNOSTICS:")
        lines.extend(f"  - {q.label}: {q.stated()}" for q in evidence.diagnostics)
    if evidence.steps:
        lines.append("STEPS:")
        for step in evidence.steps:
            lines.append(f"  - {step.title}: {step.what} {step.why}".rstrip())
    standing = evidence.unresolved()
    if standing:
        lines.append(f"UNRESOLVED ASSUMPTIONS: {', '.join(standing)}")
    return "\n".join(lines)


def narrate_text(
    key: str,
    draft: str,
    evidence: Evidence,
    model: LanguageModel,
    *,
    instruction: str = "",
    temperature: float = 0.2,
) -> Narration | Unsupported:
    """Rewrite ``draft`` against ``evidence``; keep the draft if the result drifts.

    ``Unsupported`` only when the model could not be reached at all — a missing
    extra, a missing key, an empty response. A model that answers and invents a
    number is not an ``Unsupported``: it is a ``Narration`` with ``verified``
    false, because the document can still be produced.
    """
    prompt = (
        f"<<facts>>\n{evidence_brief(evidence)}\n<<end>>\n\n"
        f"{instruction or 'Rewrite the draft below.'}\n\n"
        f"DRAFT:\n{draft}\n"
    )
    produced = model.generate(prompt, system=SYSTEM, temperature=temperature)
    if isinstance(produced, Unsupported):
        return produced
    text = produced.strip()
    if not text:
        return Unsupported(
            reason=f"{model.name} returned an empty rewrite", detail={"section": key}
        )
    bad = numbers.unverified(text, evidence)
    if bad:
        return Narration(
            key=key,
            text=draft,
            model=model.name,
            verified=False,
            narrated=True,
            rejected=tuple(dict.fromkeys(lit.text for lit in bad)),
        )
    return Narration(key=key, text=text, model=model.name, verified=True, narrated=True)


def _draft_of(section: Section) -> str:
    """The paragraphs of a section, joined — what there is to rewrite."""
    return "\n\n".join(b.text for b in section.blocks if isinstance(b, Paragraph) and b.text)


def _with_prose(section: Section, text: str) -> Section:
    """The section with its paragraphs replaced by one narrated block.

    Metrics, tables, figures and ledgers are left exactly where they were: those
    carry the numbers, and the model does not get to rearrange them.

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
        if run and all(isinstance(b, Paragraph) for b in run):
            orphaned.add(i)
        i = j
    kept = tuple(
        b for k, b in enumerate(blocks) if k not in orphaned and not isinstance(b, Paragraph)
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
        self, section: Section, evidence: Evidence, *, key: str = ""
    ) -> tuple[Section, Narration]:
        """Narrate one section's prose, returning the section to render and the record.

        When the model is unreachable the section comes back untouched with a
        ``Narration`` that says why, so a caller can build the whole document
        with no key at all and still get a report.
        """
        name = key or section.title.lower().replace(" ", "_")
        draft = _draft_of(section)
        if not draft:
            return section, Narration(key=name, text="", model="", narrated=False)
        out = narrate_text(
            name, draft, evidence, self.prose, instruction=SECTION_INSTRUCTION.get(name, "")
        )
        if isinstance(out, Unsupported):
            return section, Narration(
                key=name, text=draft, model=self.prose.name, narrated=False, rejected=(out.reason,)
            )
        if not out.verified:
            return section, out
        return _with_prose(section, out.text), out

    def abstract(self, evidence: Evidence, *, sentences: int = 4) -> Narration | Unsupported:
        """A short abstract from the evidence alone, on the cheap model."""
        draft = evidence_brief(evidence)
        return narrate_text(
            "abstract",
            draft,
            evidence,
            self.light,
            instruction=(
                f"Write an abstract of at most {sentences} sentences for the report these "
                "facts describe. State the question, what was estimated, the headline "
                "quantity with its interval, and the strongest standing assumption."
            ),
        )
