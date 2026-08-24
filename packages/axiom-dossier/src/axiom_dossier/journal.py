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
from axiom_dossier.interpret import pivotal
from axiom_dossier.sections import (
    VERBOSITY,
    Verbosity,
    literal,
    plural,
    sentence,
    standing_assumptions,
)

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
    "remarks",
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

    Answers the five questions the UCSD guide asks of an abstract — what was
    studied, what was asked, how it was answered, what was found, and what that
    tells us — as one paragraph of continuous prose. No labels and no bold: the
    guide is explicit that an abstract is a one-paragraph summary and that
    nothing outside a heading is emphasised.

    It states **one** quantity with its interval, the one the answer turns on,
    and names the others. An abstract that lists every finding has become the
    results table, and a reader who wanted the table would have turned to it.
    """
    parts: list[str] = []
    if evidence.question:
        asked = literal(evidence.question).strip()
        parts.append(asked if asked[-1:] in ".?!" else asked + ".")

    # Each sentence is written only when there is something to put in it. A
    # structured abstract whose labels stand alone -- "Methods. Conclusions." --
    # is worse than a shorter one, and a record without findings is a real case:
    # an example that narrates its results in prose has steps but no quantities.
    if evidence.verdict is not None:
        route = evidence.verdict.route or "the recorded route"
        # A verdict may carry no reason, and appending an empty one leaves the
        # sentence trailing a space before the next.
        because = f" {sentence(evidence.verdict.reason)}" if evidence.verdict.reason else ""
        parts.append(f"The effect is {evidence.verdict.status} via {route}.{because}")
    elif evidence.steps:
        opening = (evidence.steps[0].what or evidence.steps[0].title).rstrip(". ") + "."
        recorded = plural(len(evidence.steps), "step", verb="is")
        parts.append(f"{opening} {recorded} recorded in full below.")

    if evidence.findings:
        lead = pivotal(evidence) or evidence.findings[0]
        others = [q.label for q in evidence.findings if q.key != lead.key]
        parts.append(f"{lead.label} is {lead.stated()}.")
        if others:
            # Semicolons: a finding's label may itself contain a comma
            # ("40 mg vs control, pooled"), and a comma-joined list of them
            # reads as twice as many findings as the record holds.
            counted = plural(len(others), "further quantity", verb="is")
            parts.append(f"{counted} reported: {'; '.join(others)}.")
    elif evidence.remarks:
        parts.append(literal(evidence.remarks[0]))

    standing = evidence.unresolved()
    if standing:
        parts.append(
            "The reading above holds under "
            f"{plural(len(standing), 'unresolved assumption')}: "
            f"{'; '.join(n.replace('_', ' ') for n in standing)}."
        )
    elif evidence.assumptions or evidence.verdict is not None:
        parts.append("No assumption in the record is left unresolved.")
    else:
        # Saying "none unresolved" when none were recorded would read as a
        # clean bill of health for a record that never took the examination.
        parts.append(
            "No assumptions were recorded for this run, which is not the same as "
            "none being required."
        )
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
        blocks.append(Paragraph(text=f"Reported quantities: {'; '.join(keywords)}", emphasis=True))
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
        blocks.append(
            Paragraph(text=f"This report addresses one question: {literal(evidence.question)}")
        )
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
                        f"{plural(len(standing), 'assumption', verb='carries')} it, "
                        "and each is stated in "
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
                    f"The effect is {evidence.verdict.status}, not identified. "
                    "Everything below describes what was measured; it does not establish "
                    "what the treatment does."
                )
            )
        )
    return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]
