"""Journal style: the shape and the typography of a paper rather than a readout.

Two different things, both of which "journal style" means and which are easy to
confuse:

**The shape.** A paper has an order a reader navigates by muscle memory —
abstract, introduction, methods, results, discussion, conclusions, limitations,
provenance — and numbered headings so a section can be referred to. That order
is not decoration: putting discussion before results, or omitting limitations,
makes a document that reads as marketing regardless of its typeface.

**The typography.** Serif body text, a smaller measure, quieter rules, an
accent that is nearly black rather than a brand blue. ``JOURNAL_THEME`` is a
``Theme`` like any other, so it hashes into the report's content hash and a
restyled report is distinguishable from a rewritten one.

The abstract and introduction are generated here rather than in ``sections``
because they only exist in this shape: a readout has no abstract.
"""

from __future__ import annotations

from axiom.report import Heading, Paragraph, Section, Theme

from axiom_dossier.evidence import Evidence
from axiom_dossier.sections import VERBOSITY, Verbosity, standing_assumptions

__all__ = [
    "JOURNAL_SECTIONS",
    "JOURNAL_THEME",
    "abstract_section",
    "introduction_section",
    "numbered",
]

JOURNAL_THEME = Theme(
    name="journal",
    font="Times-Roman",
    # Times-Roman is a PostScript name, not a CSS family; without this the HTML
    # renders sans-serif while the PDF renders serif and the two disagree.
    font_fallback='Georgia, "Times New Roman", Times, serif',
    mono_font="Courier",
    base_size=10.5,
    title_size=19.0,
    heading_sizes=(13.5, 11.5, 10.5),
    text_color="#111111",
    muted_color="#555555",
    accent_color="#1a1a1a",
    rule_color="#c9c9c9",
    page="a4",
    margin=64.0,
)
"""A paper, not a slide: serif, tight, and almost monochrome."""

JOURNAL_SECTIONS = (
    "abstract",
    "introduction",
    "methods",
    "results",
    "diagnostics",
    "discussion",
    "conclusions",
    "limitations",
    "provenance",
)
"""The order a reader navigates by habit. Discussion after results, always."""


def numbered(
    sections: tuple[Section, ...], *, skip: tuple[str, ...] = ("Abstract",)
) -> tuple[Section, ...]:
    """Prefix section titles with ``1.``, ``2.`` … so they can be referred to.

    The abstract is skipped by convention: it is not section 1 of a paper, it is
    what comes before section 1.
    """
    out: list[Section] = []
    index = 0
    for section in sections:
        if section.title in skip:
            out.append(section)
            continue
        index += 1
        out.append(
            Section(
                title=f"{index}. {section.title}",
                blocks=section.blocks,
                summary=section.summary,
            )
        )
    return tuple(out)


def _fallback_abstract(evidence: Evidence) -> str:
    """An abstract assembled from the record, for when no model is narrating.

    Structured the way a journal asks for one — objective, methods, results,
    conclusion — because that is also the order in which the record holds it.
    """
    parts: list[str] = []
    if evidence.question:
        parts.append(f"**Objective.** {evidence.question}")
    if evidence.verdict is not None:
        route = evidence.verdict.route or "the recorded route"
        parts.append(
            f"**Methods.** The effect is {evidence.verdict.status} via {route}. "
            f"{evidence.verdict.reason}"
        )
    elif evidence.steps:
        parts.append(f"**Methods.** {evidence.steps[0].what}")
    if evidence.findings:
        stated = "; ".join(f"{q.label} {q.stated()}" for q in evidence.findings)
        parts.append(f"**Results.** {stated}.")
    standing = evidence.unresolved()
    if standing:
        parts.append(
            "**Conclusions.** The reading above holds under "
            f"{len(standing)} unresolved assumption(s): "
            f"{', '.join(n.replace('_', ' ') for n in standing)}."
        )
    else:
        parts.append("**Conclusions.** No assumption in the record is left unresolved.")
    return " ".join(parts)


def abstract_section(
    evidence: Evidence,
    *,
    text: str = "",
    title: str = "Abstract",
    verbosity: Verbosity = "standard",
) -> Section:
    """The abstract. ``text`` overrides the generated one — that is where a
    narrated abstract arrives."""
    _ = VERBOSITY[verbosity]
    body = text.strip() or _fallback_abstract(evidence)
    blocks: list[object] = [Paragraph(text=body)]
    keywords = [q.label for q in evidence.findings][:4]
    if keywords:
        blocks.append(
            Paragraph(text=f"*Reported quantities:* {'; '.join(keywords)}", emphasis=False)
        )
    return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]


def introduction_section(
    evidence: Evidence, *, title: str = "Introduction", verbosity: Verbosity = "standard"
) -> Section:
    """What was asked and what would count as an answer.

    Generated rather than written, so it states the question, the quantity that
    answers it, and the threshold that decides it — and nothing about why the
    question matters, which is the one thing the record genuinely does not know.
    """
    detail = VERBOSITY[verbosity]
    blocks: list[object] = []
    if evidence.question:
        blocks.append(Paragraph(text=f"This report addresses one question: {evidence.question}"))
    else:
        blocks.append(
            Paragraph(text="No question was recorded for this analysis; the findings follow.")
        )

    if evidence.findings:
        lead = evidence.findings[0]
        if lead.threshold is not None:
            unit = f" {lead.unit}" if lead.unit else ""
            blocks.append(
                Paragraph(
                    text=(
                        f"The question is answered by {lead.label}, and the value it turns "
                        f"on is {lead.threshold:.{lead.precision}f}{unit}. An interval "
                        "lying wholly on one side of that value settles the question; one "
                        "spanning it does not."
                    )
                )
            )
        else:
            blocks.append(
                Paragraph(
                    text=(
                        f"The question is answered by {lead.label}. No decision threshold "
                        "was recorded, so this report states the quantity and its interval "
                        "without judging whether it is large enough to act on."
                    )
                )
            )

    if detail["include_assumption_recap"]:
        standing = standing_assumptions(evidence)
        if standing:
            blocks.append(
                Paragraph(
                    text=(
                        "The answer is conditional throughout. "
                        f"{len(standing)} assumption(s) carry it, and each is stated in "
                        "the methods and again in the limitations."
                    )
                )
            )
    if evidence.verdict is not None and evidence.verdict.status != "identified":
        blocks.append(
            Heading(text="A caution before the methods", level=3),
        )
        blocks.append(
            Paragraph(
                text=(
                    f"The effect is **{evidence.verdict.status}**, not identified. "
                    "Everything below describes what was measured; it does not establish "
                    "what the treatment does."
                )
            )
        )
    return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]
