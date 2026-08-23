"""Assembling the whole document, and writing it out as HTML, PPTX or PDF.

``build`` is the one call most users need. It turns an ``Evidence`` into an
``axiom.report.Report`` plus the context that fills it, optionally narrating the
prose on the way, and hands back something that can be written to any of the
three formats axiom already renders.

    built = build(evidence)                      # no key, no network, still a report
    built = build(evidence, narrator=Narrator())  # better prose, same numbers
    built.write("hyper3.pdf")

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
from axiom.report import Report, Section, Theme
from axiom.report import missing as report_missing
from axiom.report import write as report_write

from axiom_dossier.evidence import Evidence
from axiom_dossier.narrate import Narration, Narrator
from axiom_dossier.sections import (
    assumption_rows,
    diagnostics_section,
    limitations_section,
    methods_section,
    provenance_section,
    results_section,
    standing_assumptions,
)

__all__ = ["Dossier", "DEFAULT_SECTIONS", "build", "context_for"]

Format = Literal["html", "pptx", "pdf"]

DEFAULT_SECTIONS = ("methods", "results", "diagnostics", "limitations", "provenance")
"""The order a reader expects: what was done, what came out, what was checked, what it rests on."""

_BUILDERS = {
    "methods": methods_section,
    "results": results_section,
    "diagnostics": diagnostics_section,
    "limitations": limitations_section,
    "provenance": provenance_section,
}


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


@dataclass
class Dossier:
    """A built document: the template, the data that fills it, and what was narrated."""

    report: Report
    context: dict[str, object]
    narrations: tuple[Narration, ...] = ()
    evidence: Evidence | None = field(default=None, repr=False)

    def missing(self) -> tuple[str, ...]:
        """Context keys the template still needs. Empty means it will render."""
        return tuple(report_missing(self.report, self.context))

    def rejected(self) -> tuple[Narration, ...]:
        """Narrations that failed the numeric check and were replaced by their draft."""
        return tuple(n for n in self.narrations if n.narrated and not n.verified)

    def write(self, path: str, format: Format | None = None) -> str | Unsupported:
        """Render and write. ``Unsupported`` names a missing extra or a missing key."""
        return report_write(self.report, self.context, path, format)

    def summary(self) -> str:
        narrated = sum(1 for n in self.narrations if n.narrated)
        bad = len(self.rejected())
        parts = [f"{len(self.report.sections)} sections"]
        if narrated:
            parts.append(f"{narrated} narrated")
        if bad:
            parts.append(f"{bad} narration(s) rejected and replaced by the generated draft")
        return "; ".join(parts)


def build(
    evidence: Evidence,
    *,
    narrator: Narrator | None = None,
    sections: Sequence[str] = DEFAULT_SECTIONS,
    theme: Theme | None = None,
    subtitle: str = "",
    name: str = "dossier",
) -> Dossier:
    """Turn evidence into a document. Narration is optional and never load-bearing.

    With ``narrator`` absent this makes no network call and needs no extra beyond
    the renderer's. With one present each section's prose is rewritten and then
    checked; a section whose rewrite introduces an untraceable number keeps its
    generated text, and ``Dossier.rejected()`` says which.
    """
    unknown = [s for s in sections if s not in _BUILDERS]
    if unknown:
        raise ValueError(f"unknown section(s) {unknown}; have {sorted(_BUILDERS)}")

    built: list[Section] = []
    narrations: list[Narration] = []
    for key in sections:
        section = _BUILDERS[key](evidence)
        if narrator is not None and key != "provenance":
            section, narration = narrator.section(section, evidence, key=key)
            narrations.append(narration)
        built.append(section)

    report = Report(
        name=name,
        title=evidence.title,
        subtitle=subtitle or evidence.question,
        theme=theme or Theme(),
        sections=tuple(built),
    )
    return Dossier(
        report=report,
        context=context_for(evidence, narrations),
        narrations=tuple(narrations),
        evidence=evidence,
    )
