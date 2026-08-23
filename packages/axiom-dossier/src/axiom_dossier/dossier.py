"""Assembling the whole document, and writing it out as HTML, PPTX or PDF.

``build`` is the one call most users need. It turns an ``Evidence`` into an
``axiom.report.Report`` plus the context that fills it, optionally narrating the
prose on the way, and hands back something that can be written to any of the
three formats axiom already renders.

    built = build(evidence)                                  # readout, no model
    built = build(evidence, style="journal", verbosity="full",
                  narrator=Narrator())                       # a paper
    built.write("hyper3.pdf")

Two axes, deliberately separate. ``style`` chooses the *shape*: ``plain`` is a
readout — methods, results, checking, limitations — and ``journal`` is a paper,
with an abstract, an introduction, a discussion and conclusions, numbered
headings and serif type. ``verbosity`` chooses how much each section says, and
does it by including more content rather than by asking a model to write longer.

What comes back is deliberately not a ``Spec``: it holds the data. The
*template* inside it is a Spec and hashes, which is what lets two runs be
compared — same template hash means the layout did not move, same evidence hash
means the claims did not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from axiom.core import Unsupported
from axiom.report import Figure, Report, Section, Table, Theme
from axiom.report import missing as report_missing
from axiom.report import write as report_write

from axiom_dossier.evidence import Evidence
from axiom_dossier.interpret import conclusions_section, discussion_section
from axiom_dossier.journal import (
    JOURNAL_SECTIONS,
    JOURNAL_THEME,
    abstract_section,
    introduction_section,
    numbered,
)
from axiom_dossier.narrate import Narration, Narrator
from axiom_dossier.sections import (
    Verbosity,
    assumption_rows,
    diagnostics_section,
    limitations_section,
    literal,
    methods_section,
    provenance_section,
    remarks_section,
    results_section,
    standing_assumptions,
)

__all__ = [
    "DEFAULT_SECTIONS",
    "JOURNAL_SECTIONS",
    "Dossier",
    "Style",
    "build",
    "context_for",
]

Format = Literal["html", "pptx", "pdf"]
Style = Literal["plain", "journal"]
"""``plain`` is a readout; ``journal`` is a paper. They differ in shape and type."""

DEFAULT_SECTIONS = (
    "methods",
    "results",
    "remarks",
    "diagnostics",
    "limitations",
    "provenance",
)
"""The readout order: what was done, what came out, what was checked, what it rests on."""

_BUILDERS = {
    "abstract": abstract_section,
    "introduction": introduction_section,
    "methods": methods_section,
    "results": results_section,
    "remarks": remarks_section,
    "diagnostics": diagnostics_section,
    "discussion": discussion_section,
    "conclusions": conclusions_section,
    "limitations": limitations_section,
    "provenance": provenance_section,
}

#: Sections whose prose is generated as an interpretation and must not be
#: rewritten before the numbers they interpret exist. ``provenance`` is never
#: narrated at all — it is the audit trail, and a model has no business in it.
#: ``remarks`` joins ``provenance`` here: it is a person's words, and having a
#: model rewrite them while they stay attributed to that person is not a
#: quality improvement, it is a misattribution.
_NEVER_NARRATED = ("provenance", "remarks")


def context_for(evidence: Evidence, narrations: Sequence[Narration] = ()) -> dict[str, object]:
    """Everything the generated sections name: quantities, and the three tables.

    The tables are built here rather than in the sections because a section is a
    template and holds no data — the same separation axiom's report package
    draws, kept rather than worked around.
    """
    ctx = evidence.context()
    ctx["assumption_table"] = assumption_rows(standing_assumptions(evidence))
    ctx["provenance_table"] = [
        {
            "Quantity": q.label,
            "Key": q.key,
            "Value": q.stated(),
            "Produced by": q.source or "—",
        }
        for q in evidence.quantities()
    ]
    run = [{"Field": k, "Value": v} for k, v in sorted(evidence.provenance.items())]
    run.append({"Field": "evidence hash", "Value": evidence.content_hash()})
    for n in narrations:
        if n.narrated:
            run.append(
                {
                    "Field": f"narration: {n.key}",
                    "Value": (
                        f"{n.model} — {'verified' if n.verified else 'REJECTED, draft kept'}"
                    ),
                }
            )
    ctx["run_table"] = run
    return ctx


def _number_captions(sections: tuple[Section, ...]) -> tuple[Section, ...]:
    """``Table 1.`` / ``Figure 1.`` in front of every caption, in document order.

    A paper refers to its exhibits by number; a caption that does not carry one
    cannot be referred to at all.
    """
    tables = figures = 0
    out: list[Section] = []
    for section in sections:
        blocks: list[object] = []
        for block in section.blocks:
            if isinstance(block, Table):
                tables += 1
                caption = f"Table {tables}. {block.caption}".rstrip(". ")
                blocks.append(block.model_copy(update={"caption": caption}))
            elif isinstance(block, Figure):
                figures += 1
                caption = f"Figure {figures}. {block.caption}".rstrip(". ")
                blocks.append(block.model_copy(update={"caption": caption}))
            else:
                blocks.append(block)
        out.append(
            Section(title=section.title, blocks=tuple(blocks), summary=section.summary)  # type: ignore[arg-type]
        )
    return tuple(out)


@dataclass
class Dossier:
    """A built document: the template, the data that fills it, and what was narrated."""

    report: Report
    context: dict[str, object]
    narrations: tuple[Narration, ...] = ()
    evidence: Evidence | None = field(default=None, repr=False)
    style: Style = "plain"
    verbosity: Verbosity = "standard"

    def missing(self) -> tuple[str, ...]:
        """Context keys the template still needs. Empty means it will render."""
        return tuple(report_missing(self.report, self.context))

    def rejected(self) -> tuple[Narration, ...]:
        """Narrations that failed a check and were replaced by their draft."""
        return tuple(n for n in self.narrations if n.narrated and not n.verified)

    def write(self, path: str, format: Format | None = None) -> str | Unsupported:
        """Render and write. ``Unsupported`` names a missing extra or a missing key."""
        return report_write(self.report, self.context, path, format)

    def summary(self) -> str:
        narrated = sum(1 for n in self.narrations if n.narrated)
        bad = len(self.rejected())
        parts = [f"{self.style}/{self.verbosity}", f"{len(self.report.sections)} sections"]
        if narrated:
            parts.append(f"{narrated} narrated")
        if bad:
            parts.append(f"{bad} rejected and replaced by the generated draft")
        return "; ".join(parts)


def build(
    evidence: Evidence,
    *,
    narrator: Narrator | None = None,
    style: Style = "plain",
    verbosity: Verbosity = "standard",
    sections: Sequence[str] | None = None,
    theme: Theme | None = None,
    subtitle: str = "",
    name: str = "dossier",
) -> Dossier:
    """Turn evidence into a document. Narration is optional and never load-bearing.

    With ``narrator`` absent this makes no network call and needs no extra beyond
    the renderer's. With one present each section's prose is rewritten and then
    checked; a section whose rewrite introduces an untraceable number or an
    unlicensed claim keeps its generated text, and ``Dossier.rejected()`` says
    which.

    ``style="journal"`` selects the paper shape and typography, and narrates the
    abstract on the cheap model — the one place the light model earns its place,
    because an abstract is a compression rather than a judgement.
    """
    if style not in ("plain", "journal"):
        raise ValueError(f"unknown style {style!r}; have ['journal', 'plain']")
    chosen = (
        tuple(sections)
        if sections is not None
        else (JOURNAL_SECTIONS if style == "journal" else DEFAULT_SECTIONS)
    )
    unknown = [s for s in chosen if s not in _BUILDERS]
    if unknown:
        raise ValueError(f"unknown section(s) {unknown}; have {sorted(_BUILDERS)}")

    # The abstract summarises the rest, so it is narrated from the evidence
    # rather than from its own draft, and on the light model.
    abstract_text = ""
    narrations: list[Narration] = []
    if narrator is not None and "abstract" in chosen:
        produced = narrator.abstract(evidence, verbosity=verbosity)
        if isinstance(produced, Narration):
            narrations.append(produced)
            if produced.verified:
                abstract_text = produced.text

    # A report with no recorded remarks should not carry an empty section
    # saying so; every other section is always meaningful, this one is not.
    if not evidence.remarks:
        chosen = tuple(k for k in chosen if k != "remarks")

    built: list[Section] = []
    for key in chosen:
        if key == "abstract":
            section = abstract_section(evidence, text=abstract_text, verbosity=verbosity)
        else:
            section = _BUILDERS[key](evidence, verbosity=verbosity)
        if narrator is not None and key not in _NEVER_NARRATED and key != "abstract":
            section, narration = narrator.section(section, evidence, key=key, verbosity=verbosity)
            narrations.append(narration)
        built.append(section)

    shaped = tuple(built)
    if style == "journal":
        shaped = _number_captions(shaped)
        shaped = numbered(shaped)

    report = Report(
        name=name,
        title=evidence.title,
        subtitle=literal(subtitle or evidence.question),
        theme=theme or (JOURNAL_THEME if style == "journal" else Theme()),
        sections=shaped,
    )
    return Dossier(
        report=report,
        context=context_for(evidence, narrations),
        narrations=tuple(narrations),
        evidence=evidence,
        style=style,
        verbosity=verbosity,
    )
