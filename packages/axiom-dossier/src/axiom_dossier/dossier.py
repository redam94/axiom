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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from axiom.core import Unsupported
from axiom.report import Figure, Report, Section, Table, Theme
from axiom.report import missing as report_missing
from axiom.report import write as report_write

from axiom_dossier.apa import (
    APA_SECTIONS,
    APA_THEME,
    apa_caption,
    exhibit_sections,
    title_block,
    title_page_section,
)
from axiom_dossier.evidence import Evidence
from axiom_dossier.figures import figures_for
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
from axiom_dossier.tables import tables_for

__all__ = [
    "DEFAULT_SECTIONS",
    "Exhibits",
    "JOURNAL_SECTIONS",
    "Dossier",
    "Style",
    "build",
    "context_for",
]

Format = Literal["html", "pptx", "pdf"]
Style = Literal["plain", "journal", "apa"]
"""``plain`` is a readout, ``journal`` a paper, ``apa`` an APA manuscript."""

Exhibits = Literal["embedded", "gathered", "none"]
"""Where figures and tables sit. APA gathers them after the text; a readout embeds."""

DEFAULT_SECTIONS = (
    "methods",
    "results",
    "remarks",
    "diagnostics",
    "limitations",
    "provenance",
)
"""The readout order: what was done, what came out, what was checked, what it rests on."""

#: Section key -> builder. Heterogeneous on purpose: the title page takes
#: author details the others have no use for, so the shared contract is only
#: "takes an Evidence, returns a Section".
_BUILDERS: dict[str, Callable[..., Section]] = {
    "title_page": title_page_section,
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


#: Which exhibits belong to which section when they are embedded rather than
#: gathered. A figure goes with the prose that discusses it; a table goes under
#: the figure it tabulates. Keys absent from the context are skipped, so a
#: report without plotly loses its figures and keeps its tables.
_EXHIBITS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "methods": (("table", "design_table", "The design, as recorded parameters."),),
    "results": (
        ("figure", "findings_figure", "Estimated quantities with their intervals."),
        ("table", "findings_table", "Estimated quantities."),
    ),
    "diagnostics": (
        ("figure", "diagnostics_figure", "Recorded diagnostics."),
        ("table", "diagnostics_table", "Recorded diagnostics."),
    ),
}


#: The order exhibits appear in when gathered, and how each is described.
#: Tables before figures, matching the manuscript order the guide specifies.
_GATHERED_TABLES = (
    ("design_table", "The design, as recorded parameters."),
    ("findings_table", "Estimated quantities with their intervals and thresholds."),
    ("diagnostics_table", "Recorded diagnostics."),
)
_GATHERED_FIGURES = (
    (
        "findings_figure",
        "Estimated quantities with their intervals. The dashed rule marks the "
        "decision threshold; a row whose interval crosses it is unsettled rather "
        "than null.",
    ),
    ("diagnostics_figure", "Recorded diagnostics."),
)


def _with_exhibits(section: Section, key: str, available: set[str]) -> Section:
    """Append this section's figures and tables, skipping the ones with no data."""
    extra: list[object] = []
    for kind, source, description in _EXHIBITS.get(key, ()):
        if source not in available:
            continue
        if kind == "figure":
            extra.append(Figure(source=source, caption=description))
        else:
            extra.append(Table(source=source, caption=description, max_rows=40))
    if not extra:
        return section
    return Section(
        title=section.title,
        blocks=(*section.blocks, *extra),  # type: ignore[arg-type]
        summary=section.summary,
    )


def _without_missing_exhibits(section: Section, available: set[str]) -> Section:
    """Drop the figure and table blocks whose data is not in the context.

    A step names every exhibit it recorded; whether one can be *drawn* is
    decided later — plotly may be absent, or a payload may not hold what its
    chart kind needs. Naming one that is not there would make ``missing()``
    non-empty and the whole report unrenderable, so the block goes and the rest
    of the step stays.
    """
    kept = tuple(
        block
        for block in section.blocks
        if not (isinstance(block, Figure | Table) and block.source not in available)
    )
    if len(kept) == len(section.blocks):
        return section
    return Section(title=section.title, blocks=kept, summary=section.summary)


#: Sections whose exhibits were numbered when they were built. Running the
#: document-order numbering over them again is what produced "Figure 1. Figure
#: 1." and gave a gathered table a second number that disagreed with its title.
_SELF_NUMBERED = ("Tables", "Figures")


def _number_captions(
    sections: tuple[Section, ...], *, style: Style = "journal"
) -> tuple[Section, ...]:
    """``Table 1.`` / ``Figure 1.`` in front of every caption, in document order.

    A paper refers to its exhibits by number; a caption that does not carry one
    cannot be referred to at all. APA italicises the label and number and leaves
    the description roman, which is what ``apa_caption`` produces.
    """

    def caption_for(kind: str, n: int, description: str) -> str:
        if style == "apa":
            return apa_caption(kind, n, description)
        return f"{kind} {n}. {description}".rstrip(". ")

    tables = figures = 0
    out: list[Section] = []
    for section in sections:
        if section.title in _SELF_NUMBERED:
            out.append(section)
            continue
        blocks: list[object] = []
        for block in section.blocks:
            if isinstance(block, Table):
                tables += 1
                blocks.append(
                    block.model_copy(
                        update={"caption": caption_for("Table", tables, block.caption)}
                    )
                )
            elif isinstance(block, Figure):
                figures += 1
                blocks.append(
                    block.model_copy(
                        update={"caption": caption_for("Figure", figures, block.caption)}
                    )
                )
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

    def write(
        self,
        path: str,
        format: Format | None = None,
        *,
        inline_plotly: bool = True,
    ) -> str | Unsupported:
        """Render and write. ``Unsupported`` names a missing extra or a missing key.

        ``inline_plotly`` bundles the plotting library into the HTML, which is
        the default because a report that needs a CDN to draw its own figures is
        not a document you can email. It costs a few megabytes; pass ``False``
        for a small file that fetches the library at open time.
        """
        return report_write(self.report, self.context, path, format, inline_plotly=inline_plotly)

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
    exhibits: Exhibits | None = None,
    sections: Sequence[str] | None = None,
    theme: Theme | None = None,
    subtitle: str = "",
    name: str = "dossier",
    authors: Sequence[str] = (),
    affiliation: str = "",
    author_note: str = "",
    extra_figures: Mapping[str, tuple[Any, str]] | None = None,
    extra_exhibits: Mapping[str, object] | None = None,
) -> Dossier:
    """Turn evidence into a document. Narration is optional and never load-bearing.

    With ``narrator`` absent this makes no network call and needs no extra beyond
    the renderer's. With one present each section's prose is rewritten and then
    checked; a section whose rewrite introduces an untraceable number or an
    unlicensed claim keeps its generated text, and ``Dossier.rejected()`` says
    which.

    ``style="journal"`` selects the paper shape and typography and ``style="apa"``
    an APA manuscript — title page, abstract on its own page, and exhibits
    gathered after the text. Both narrate the abstract on the cheap model, which
    is the one place the light model earns its place: an abstract is a
    compression rather than a judgement.

    ``exhibits`` says where figures and tables go. ``embedded`` puts each with
    the prose that discusses it; ``gathered`` collects them into Tables and
    Figures sections at the end, which is what APA asks for and therefore the
    default for that style. ``none`` leaves them out.

    ``extra_exhibits`` carries figures and tables the evidence does not produce
    but a step names — a walkthrough record's own charts, say. They join the
    exhibit context before the sections are built, which is what lets a step's
    exhibit be checked against its data rather than assumed into existence.
    """
    if style not in ("plain", "journal", "apa"):
        raise ValueError(f"unknown style {style!r}; have ['apa', 'journal', 'plain']")
    if exhibits is None:
        exhibits = "gathered" if style == "apa" else "embedded"
    if exhibits not in ("embedded", "gathered", "none"):
        raise ValueError(f"unknown exhibits {exhibits!r}; have ['embedded', 'gathered', 'none']")
    default_sections = {
        "apa": APA_SECTIONS,
        "journal": JOURNAL_SECTIONS,
        "plain": DEFAULT_SECTIONS,
    }[style]
    chosen = tuple(sections) if sections is not None else default_sections
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

    chosen_theme = (
        theme
        or {
            "apa": APA_THEME,
            "journal": JOURNAL_THEME,
            "plain": Theme(),
        }[style]
    )

    # Figures and tables are built before the sections, because a section may
    # only name an exhibit that has data behind it -- naming one that does not
    # would make `missing()` non-empty and the report unrenderable.
    exhibit_context: dict[str, object] = {}
    extra_captions: list[tuple[str, str]] = []
    if exhibits != "none":
        exhibit_context.update(tables_for(evidence))
        exhibit_context.update(figures_for(evidence, theme=chosen_theme))
        exhibit_context.update(extra_exhibits or {})
        # Figures produced elsewhere -- by a notebook the agent ran, or by code
        # it wrote to close a gap -- arrive already drawn. They are exhibits like
        # any other and are gathered after the generated ones.
        for key, (figure, description) in (extra_figures or {}).items():
            exhibit_context[key] = figure
            extra_captions.append((key, description))
    available = set(exhibit_context)

    built: list[Section] = []
    for key in chosen:
        if key == "title_page":
            built.append(
                title_page_section(
                    evidence,
                    authors=tuple(authors),
                    affiliation=affiliation,
                    author_note=author_note,
                    verbosity=verbosity,
                )
            )
            continue
        if key == "abstract":
            section = abstract_section(evidence, text=abstract_text, verbosity=verbosity)
        else:
            section = _BUILDERS[key](evidence, verbosity=verbosity)
        if exhibits == "embedded":
            section = _with_exhibits(section, key, available)
        section = _without_missing_exhibits(section, available)
        if narrator is not None and key not in _NEVER_NARRATED and key != "abstract":
            section, narration = narrator.section(section, evidence, key=key, verbosity=verbosity)
            narrations.append(narration)
        built.append(section)

    if exhibits == "gathered":
        built.extend(
            exhibit_sections(
                tables=[
                    (key, description) for key, description in _GATHERED_TABLES if key in available
                ],
                figures=[
                    (key, description) for key, description in _GATHERED_FIGURES if key in available
                ],
            )
        )

    # APA puts the author block under the title; every other style puts the
    # question there, because a readout has no authors and does have a question.
    chosen_subtitle = subtitle or (
        title_block(authors, affiliation) if style == "apa" else evidence.question
    )

    shaped = tuple(built)
    if style in ("journal", "apa"):
        shaped = _number_captions(shaped, style=style)
    if style == "journal":
        shaped = numbered(shaped)

    report = Report(
        name=name,
        title=evidence.title,
        subtitle=literal(chosen_subtitle),
        theme=chosen_theme,
        sections=shaped,
    )
    return Dossier(
        report=report,
        context={**context_for(evidence, narrations), **exhibit_context},
        narrations=tuple(narrations),
        evidence=evidence,
        style=style,
        verbosity=verbosity,
    )
