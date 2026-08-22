"""The template language: a report as a tree of blocks that name their data.

A ``Report`` is a ``Spec``, so it round-trips, hashes, and diffs like every
other configuration object here — and, more usefully, it contains **no data**.
Every block that shows something names a ``source``: a key that is looked up in
a context at render time. That is what makes a report a *template* rather than
a document. The same ``Report`` renders this quarter's numbers and next
quarter's, and its content hash says the layout did not change between them.

    report = Report(name="readout", title="HYPER-3", sections=(
        Section(title="Dose response", blocks=(
            Paragraph(text="The 40 mg arm reverses above {turning_point:.0f} mg."),
            Figure(source="dose_response", caption="Response with its 90 % band"),
            Metric(source="top_contrast", label="40 mg vs control"),
        )),
    ))
    missing(report, context)        # -> () when every source resolves
    render(report, context, "html")

Three block types are about uncertainty and behave accordingly. ``Figure``
resolves a ``surface.ResponseBand`` through ``viz`` so a surface arrives with
its band; ``Metric`` renders an ``Interval``, ``Summary`` or ``EstimandResult``
with the interval *and its definition and mass*, never the point alone; and
a ``LedgerBlock`` block prints the assumption trail. A report built from this vocabulary
cannot quietly drop the provenance rule 4 asks for.
"""

from __future__ import annotations

import string
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Literal

from pydantic import Field, model_validator

from axiom.core import NonEmptyStr, Spec
from axiom.report.theme import Theme

__all__ = [
    "AnyBlock",
    "Divider",
    "Figure",
    "Heading",
    "LedgerBlock",
    "Metric",
    "PageBreak",
    "Paragraph",
    "Report",
    "Section",
    "Table",
    "placeholders",
]


def placeholders(text: str) -> tuple[str, ...]:
    """The ``{name}`` fields in a template string, in order, without duplicates.

    ``{name:.2f}`` counts as ``name``; ``{{`` is a literal brace and is not a
    field. Anything the standard formatter rejects is a ``ValueError`` here
    rather than at render time, which is the whole reason a template is a spec.
    """
    out: list[str] = []
    try:
        parsed = list(string.Formatter().parse(text))
    except ValueError as e:  # pragma: no cover - malformed braces
        raise ValueError(f"bad placeholder syntax in {text!r}: {e}") from e
    for _, field, _, _ in parsed:
        if field is None:
            continue
        name = field.split(".")[0].split("[")[0]
        if not name:
            raise ValueError(f"positional placeholders are not supported, in {text!r}")
        if name not in out:
            out.append(name)
    return tuple(out)


class Heading(Spec):
    """A section heading. ``level`` 1-3, matching the theme's three heading sizes."""

    block: Literal["heading"] = "heading"
    text: NonEmptyStr
    level: int = Field(default=2, ge=1, le=3)

    def sources(self) -> tuple[str, ...]:
        return placeholders(self.text)


class Paragraph(Spec):
    """Body text. Supports ``{placeholders}`` and a little inline markup.

    ``**bold**``, ``*italic*`` and ```code``` are the whole of the markup; a
    report is prose and numbers, not a document format.
    """

    block: Literal["paragraph"] = "paragraph"
    text: str
    emphasis: bool = False

    @model_validator(mode="after")
    def _valid(self) -> Paragraph:
        placeholders(self.text)
        return self

    def sources(self) -> tuple[str, ...]:
        return placeholders(self.text)


class Figure(Spec):
    """A figure, named by the context key that holds it.

    The source may resolve to a plotly ``Figure`` or to a
    ``surface.ResponseBand``; a band is rendered through ``viz.response_curve``
    so it arrives with its uncertainty. ``height`` overrides the theme.
    """

    block: Literal["figure"] = "figure"
    source: NonEmptyStr
    caption: str = ""
    height: float | None = Field(default=None, gt=0)
    full_page: bool = False

    def sources(self) -> tuple[str, ...]:
        return (self.source, *placeholders(self.caption))


class Table(Spec):
    """A table, named by the context key that holds it.

    The source may resolve to a pandas ``DataFrame`` or to a sequence of
    mappings. ``columns`` restricts and orders them; ``max_rows`` truncates and
    the renderers say so in a footnote rather than silently dropping rows.
    """

    block: Literal["table"] = "table"
    source: NonEmptyStr
    caption: str = ""
    columns: tuple[str, ...] = ()
    max_rows: int = Field(default=20, ge=1)
    precision: int = Field(default=3, ge=0, le=12)

    def sources(self) -> tuple[str, ...]:
        return (self.source, *placeholders(self.caption))


class Metric(Spec):
    """One number, with its interval when it has one.

    The source may resolve to a float, a ``core.Interval``, a ``core.Summary``
    or an ``estimands.EstimandResult``. When there is an interval it is shown
    with its definition and mass — that is not configurable, because a number
    that had uncertainty and lost it on the way to a slide is the failure this
    package exists to prevent.
    """

    block: Literal["metric"] = "metric"
    source: NonEmptyStr
    label: NonEmptyStr
    unit: str = ""
    precision: int = Field(default=2, ge=0, le=12)

    def sources(self) -> tuple[str, ...]:
        return (self.source,)


class LedgerBlock(Spec):
    """The assumption trail: a sequence of ``core.LedgerLine``."""

    block: Literal["ledger"] = "ledger"
    source: NonEmptyStr
    caption: str = ""
    show_detail: bool = False

    def sources(self) -> tuple[str, ...]:
        return (self.source, *placeholders(self.caption))


class Divider(Spec):
    """A horizontal rule."""

    block: Literal["divider"] = "divider"

    def sources(self) -> tuple[str, ...]:
        return ()


class PageBreak(Spec):
    """Start a new page or slide here."""

    block: Literal["page_break"] = "page_break"

    def sources(self) -> tuple[str, ...]:
        return ()


AnyBlock = Annotated[
    Heading | Paragraph | Figure | Table | Metric | LedgerBlock | Divider | PageBreak,
    Field(discriminator="block"),
]
"""The block vocabulary, as a discriminated union on ``block``."""


class Section(Spec):
    """A titled run of blocks. Each section starts a new page or slide."""

    title: NonEmptyStr
    blocks: tuple[AnyBlock, ...] = ()
    summary: str = ""

    @model_validator(mode="after")
    def _valid(self) -> Section:
        placeholders(self.summary)
        return self

    def sources(self) -> tuple[str, ...]:
        out: list[str] = list(placeholders(self.summary))
        for block in self.blocks:
            for name in block.sources():
                if name not in out:
                    out.append(name)
        return tuple(out)


class Report(Spec):
    """A styled, repeatable report: sections of blocks that name their data.

    Holds no data. ``sources`` is every context key it needs, so a template can
    be checked against a context before anything is rendered, and the same
    template can be rendered against many contexts — which is the "automated
    and repeated" half of the job.
    """

    name: NonEmptyStr
    title: NonEmptyStr
    subtitle: str = ""
    theme: Theme = Theme()
    sections: tuple[Section, ...] = ()
    footer: str = ""

    @model_validator(mode="after")
    def _valid(self) -> Report:
        placeholders(self.subtitle)
        placeholders(self.footer)
        if not self.sections:
            raise ValueError("a report needs at least one section")
        seen: set[str] = set()
        for section in self.sections:
            if section.title in seen:
                raise ValueError(f"duplicate section title {section.title!r}")
            seen.add(section.title)
        return self

    def sources(self) -> tuple[str, ...]:
        out: list[str] = list(placeholders(self.subtitle)) + list(placeholders(self.footer))
        for section in self.sections:
            for name in section.sources():
                if name not in out:
                    out.append(name)
        return tuple(out)

    def blocks(self) -> Iterable[tuple[Section, AnyBlock]]:
        """Every ``(section, block)`` pair, in reading order."""
        for section in self.sections:
            for block in section.blocks:
                yield section, block


def missing(report: Report, context: Mapping[str, object]) -> tuple[str, ...]:
    """Context keys the report needs and the context does not have, in reading order.

    Empty means the template is renderable. Checking this first is how a
    scheduled report fails at build time with a list of names instead of at
    render time with a ``KeyError``.
    """
    return tuple(name for name in report.sources() if name not in context)


def sources_of(blocks: Sequence[AnyBlock]) -> tuple[str, ...]:
    """Every context key a run of blocks needs, in order, without duplicates."""
    out: list[str] = []
    for block in blocks:
        for name in block.sources():
            if name not in out:
                out.append(name)
    return tuple(out)
