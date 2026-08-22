"""Writing a report as a sequence of calls instead of a nested literal.

The same composition-over-inheritance shape as ``axiom.build``: every method
returns a *new* builder, nothing mutates, and ``.build()`` returns the ``Spec``.
Authoring a template then reads in the order the reader will read it::

    template = (
        ReportBuilder("readout", "HYPER-3")
        .subtitle("Phase II dose finding, {as_of}")
        .section("Dose response", summary="Every curve carries its 90 % band.")
        .paragraph("The 40 mg arm reverses above {turning_point:.0f} mg.")
        .figure("dose_response", caption="Response with its band")
        .metric("top_contrast", "40 mg vs control", unit="mmHg")
        .section("Assumptions")
        .ledger("ledger")
        .build()
    )

Blocks attach to the section opened most recently, so `.section()` is the only
thing that changes where the next block lands.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Self

from axiom.report.spec import (
    AnyBlock,
    Divider,
    Figure,
    Heading,
    LedgerBlock,
    Metric,
    PageBreak,
    Paragraph,
    Report,
    Section,
    Table,
)
from axiom.report.theme import Theme

__all__ = ["ReportBuilder"]


@dataclass(frozen=True)
class ReportBuilder:
    """A frozen builder for a ``Report``. Every method returns a new one."""

    name: str
    title: str
    _subtitle: str = ""
    _footer: str = ""
    _theme: Theme = Theme()
    _sections: tuple[Section, ...] = ()

    # -- report-level ----------------------------------------------------------------

    def subtitle(self, text: str) -> Self:
        return replace(self, _subtitle=text)

    def footer(self, text: str) -> Self:
        return replace(self, _footer=text)

    def theme(self, theme: Theme) -> Self:
        return replace(self, _theme=theme)

    def section(self, title: str, *, summary: str = "") -> Self:
        """Open a new section. Subsequent blocks attach to it."""
        return replace(self, _sections=(*self._sections, Section(title=title, summary=summary)))

    # -- blocks ----------------------------------------------------------------------

    def _add(self, block: AnyBlock) -> Self:
        if not self._sections:
            raise ValueError("open a section before adding blocks: .section(title) first")
        head, last = self._sections[:-1], self._sections[-1]
        return replace(
            self,
            _sections=(*head, last.model_copy(update={"blocks": (*last.blocks, block)})),
        )

    def heading(self, text: str, *, level: int = 2) -> Self:
        return self._add(Heading(text=text, level=level))

    def paragraph(self, text: str, *, emphasis: bool = False) -> Self:
        return self._add(Paragraph(text=text, emphasis=emphasis))

    def figure(
        self,
        source: str,
        *,
        caption: str = "",
        height: float | None = None,
        full_page: bool = False,
    ) -> Self:
        """A figure from the context. A ``ResponseBand`` source arrives with its band."""
        return self._add(Figure(source=source, caption=caption, height=height, full_page=full_page))

    def table(
        self,
        source: str,
        *,
        caption: str = "",
        columns: tuple[str, ...] = (),
        max_rows: int = 20,
        precision: int = 3,
    ) -> Self:
        return self._add(
            Table(
                source=source,
                caption=caption,
                columns=columns,
                max_rows=max_rows,
                precision=precision,
            )
        )

    def metric(self, source: str, label: str, *, unit: str = "", precision: int = 2) -> Self:
        """One number and, when the source carries one, its interval."""
        return self._add(Metric(source=source, label=label, unit=unit, precision=precision))

    def ledger(self, source: str, *, caption: str = "", show_detail: bool = False) -> Self:
        return self._add(LedgerBlock(source=source, caption=caption, show_detail=show_detail))

    def divider(self) -> Self:
        return self._add(Divider())

    def page_break(self) -> Self:
        return self._add(PageBreak())

    # -- finish ----------------------------------------------------------------------

    def build(self) -> Report:
        """The ``Report`` spec. Raises if no section was ever opened."""
        return Report(
            name=self.name,
            title=self.title,
            subtitle=self._subtitle,
            footer=self._footer,
            theme=self._theme,
            sections=self._sections,
        )
