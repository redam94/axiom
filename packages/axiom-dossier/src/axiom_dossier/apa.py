"""APA manuscript style: the order, the title page, and where exhibits go.

Follows the UCSD Psychology guide *How to Write APA Style Research Papers*
(Geller) and the empirical-paper example beside it, which is the traditional
(6th-edition) layout: a running head, exhibits gathered at the end, and figure
captions written ``Figure 1. …`` beneath the figure rather than above it.

What that guide specifies, and what this module implements:

* **Order** — title page (page 1), abstract on its own page (page 2), then the
  body from the introduction through the discussion continuously, then
  references on a new page, then tables, then figures, each on its own page.
* **The introduction has no heading.** Its heading is the title of the paper.
  Abstract, Methods, Results, Discussion and References are centred and bold.
* **Figures carry no title and no heading** — the caption below does that work,
  and it opens with an italic ``Figure N.``
* **Double-spaced throughout, ragged right, one-inch margins**, and no bold
  outside headings.

Two departures, both deliberate and both visible in the output rather than
hidden. A causal report has to state what licensed the estimate, so
**Limitations** and **Provenance** are kept as their own sections after the
discussion; APA folds the first into the discussion and has no equivalent of
the second. And the exhibits can be embedded instead of gathered, which the
example itself allows: ``exhibits="embedded"``.
"""

from __future__ import annotations

from collections.abc import Sequence

from axiom.report import (
    Figure,
    Heading,
    PageBreak,
    Paragraph,
    Section,
    Table,
    Theme,
)

from axiom_dossier.evidence import Evidence
from axiom_dossier.sections import Verbosity, literal

__all__ = [
    "APA_SECTIONS",
    "APA_THEME",
    "RUNNING_HEAD_LIMIT",
    "apa_caption",
    "exhibit_sections",
    "running_head",
    "title_block",
    "title_page_section",
]

APA_THEME = Theme(
    name="apa",
    font="Times-Roman",
    font_fallback='"Times New Roman", Times, Georgia, serif',
    mono_font="Courier",
    base_size=12.0,
    title_size=12.0,  # APA sets the title in the body face, not display type
    heading_sizes=(12.0, 12.0, 12.0),
    text_color="#000000",
    muted_color="#000000",
    accent_color="#000000",
    rule_color="#ffffff",  # APA tables use no vertical rules and few horizontal
    background_color="#ffffff",
    page="letter",
    margin=72.0,  # one inch
)
"""Twelve-point Times, one-inch margins, black on white, no display type."""

APA_SECTIONS = (
    "title_page",
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
"""APA order, plus the two sections a causal report cannot do without."""

RUNNING_HEAD_LIMIT = 50
"""Characters, including spaces — the guide's limit for a running head."""


def running_head(title: str) -> str:
    """The title in caps, truncated to the guide's fifty characters.

    Truncated on a word boundary where one is available, because a running head
    cut mid-word reads as a defect on every page of the manuscript.
    """
    caps = " ".join(title.upper().split())
    if len(caps) <= RUNNING_HEAD_LIMIT:
        return caps
    cut = caps[:RUNNING_HEAD_LIMIT]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip()


def apa_caption(kind: str, number: int, description: str) -> str:
    """``*Figure 1*. description`` — the guide's caption, in the renderers' markup.

    The number is italicised with the label and the description is not, which is
    the 6th-edition form the example paper uses throughout.
    """
    return f"*{kind} {number}*. {description}".rstrip()


def title_page_section(
    evidence: Evidence,
    *,
    authors: tuple[str, ...] = (),
    affiliation: str = "",
    author_note: str = "",
    title: str = "Author Note",
    verbosity: Verbosity = "standard",
) -> Section:
    """The author block and running head, which is the rest of the title page.

    The *title* itself is not repeated here. Every renderer already sets a
    report's title and subtitle at the head of the first page, so a section that
    printed them again would give the manuscript two title pages stacked on one
    another — which is exactly what the first draft did. ``build`` puts the
    authors and affiliation into the subtitle instead, and what is left for this
    section is the part APA does label: the Author Note.

    The page break at the end is what puts the abstract on page 2, as the guide
    requires.
    """
    _ = verbosity
    blocks: list[object] = []
    note = author_note or (
        "This report was generated from a recorded analysis. Every quantity in it "
        "resolves to that record, and the provenance section names what produced "
        "each one."
    )
    blocks.append(Paragraph(text=literal(note)))
    if affiliation and authors:
        blocks.append(
            Paragraph(text=literal(f"Correspondence concerning this report: {affiliation}."))
        )
    blocks.append(Paragraph(text=f"Running head: {running_head(evidence.title)}"))
    blocks.append(PageBreak())
    return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]


def title_block(authors: Sequence[str], affiliation: str) -> str:
    """Authors and affiliation as the subtitle line under the title."""
    names = ", ".join(a for a in authors if a.strip())
    parts = [p for p in (names, affiliation.strip()) if p]
    return " · ".join(parts)


def exhibit_sections(
    *,
    tables: Sequence[tuple[str, str]] = (),
    figures: Sequence[tuple[str, str]] = (),
) -> tuple[Section, ...]:
    """Tables then figures, each on its own page, after the references.

    Each argument is a sequence of ``(context key, description)``: names rather
    than data, so this stays a template like every other section. Tables get
    their number and title above them and figures get theirs below, which is the
    placement the guide specifies and the reason they are built separately here
    rather than by one loop.

    A figure carries no heading at all — ``Figure 1.`` is the first thing in its
    caption and there is nothing above it, which is what makes a gathered
    exhibit page look like the example's.
    """
    out: list[Section] = []

    if tables:
        blocks: list[object] = []
        for n, (key, description) in enumerate(tables, start=1):
            blocks.append(Heading(text=apa_caption("Table", n, description), level=3))
            blocks.append(Table(source=key, caption="", max_rows=40))
            if n < len(tables):
                blocks.append(PageBreak())
        out.append(Section(title="Tables", blocks=tuple(blocks)))  # type: ignore[arg-type]

    if figures:
        blocks = []
        for n, (key, description) in enumerate(figures, start=1):
            blocks.append(Figure(source=key, caption=apa_caption("Figure", n, description)))
            if n < len(figures):
                blocks.append(PageBreak())
        out.append(Section(title="Figures", blocks=tuple(blocks)))  # type: ignore[arg-type]

    return tuple(out)
