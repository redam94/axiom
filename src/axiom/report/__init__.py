"""Styled, repeatable reports in HTML, PPTX and PDF from one template.

A ``Report`` is a ``Spec`` holding **no data**: every block that shows something
names a context key instead. That is the whole design. It makes a report a
template — the same one renders this quarter's numbers and next quarter's — and
it makes ``missing(report, context)`` a build-time check that returns a list of
names rather than a ``KeyError`` halfway through a render.

    from axiom.report import ReportBuilder, missing, render

    template = (
        ReportBuilder("readout", "HYPER-3")
        .section("Dose response")
        .paragraph("The 40 mg arm reverses above {turning_point:.0f} mg.")
        .figure("dose_response", caption="Response with its 90 % band")
        .metric("top_contrast", "40 mg vs control", unit="mmHg")
        .build()
    )
    assert missing(template, context) == ()
    html = render(template, context, "html")

**Three formats, one resolution pass.** ``resolve`` fills the template once and
the renderers only lay the result out, so the three cannot disagree about what a
number is. HTML keeps plotly figures **interactive** and needs nothing beyond
plotly; PPTX and PDF are static and need the ``report`` extra
(``python-pptx``, ``reportlab``, ``kaleido``), returning a typed ``Unsupported``
naming the extra when it is absent rather than failing at import.

**Uncertainty is not a rendering option.** A ``Figure`` whose source is a
``surface.ResponseBand`` is drawn through ``viz.response_curve``, which cannot
draw a surface without its band; a ``Metric`` whose source carries an interval
shows the interval with its definition and mass. There is no argument anywhere
in this package that turns either off, which is the reporting half of rule 4.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from axiom.core import Unsupported
from axiom.report.builder import ReportBuilder
from axiom.report.render.html import render_html
from axiom.report.render.images import figure_png
from axiom.report.render.pdf import render_pdf
from axiom.report.render.pptx import render_pptx
from axiom.report.resolve import (
    ResolvedBlock,
    ResolvedFigure,
    ResolvedLedger,
    ResolvedMetric,
    ResolvedReport,
    ResolvedSection,
    ResolvedTable,
    ResolvedText,
    resolve,
)
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
    missing,
    placeholders,
)
from axiom.report.theme import GEOMETRY, PageSize, Theme

__all__ = [
    "FORMATS",
    "GEOMETRY",
    "AnyBlock",
    "Divider",
    "Figure",
    "Format",
    "Heading",
    "LedgerBlock",
    "Metric",
    "PageBreak",
    "PageSize",
    "Paragraph",
    "Report",
    "ReportBuilder",
    "ResolvedBlock",
    "ResolvedFigure",
    "ResolvedLedger",
    "ResolvedMetric",
    "ResolvedReport",
    "ResolvedSection",
    "ResolvedTable",
    "ResolvedText",
    "Section",
    "Table",
    "Theme",
    "figure_png",
    "missing",
    "placeholders",
    "render",
    "render_html",
    "render_pdf",
    "render_pptx",
    "resolve",
    "write",
]

Format = Literal["html", "pptx", "pdf"]
"""What ``render`` can produce. ``html`` keeps figures interactive; the others do not."""

FORMATS: tuple[Format, ...] = ("html", "pptx", "pdf")
"""Every renderer, in the order of how little they need installed."""

#: The file extension each format writes.
_SUFFIX: dict[str, str] = {"html": ".html", "pptx": ".pptx", "pdf": ".pdf"}


def render(
    report: Report,
    context: Mapping[str, object],
    format: Format = "html",  # noqa: A002 - the domain word
    *,
    inline_plotly: bool = True,
) -> str | bytes | Unsupported:
    """Fill ``report`` from ``context`` and render it. ``str`` for HTML, ``bytes`` otherwise.

    ``Unsupported`` when a context key is missing (it names every one) or when
    the format's extra is not installed (it names the extra). A source that is
    present but of an unusable type is a ``ValueError``, because that is a
    mistake in the template rather than a gap in the data.
    """
    if format not in FORMATS:
        raise ValueError(f"unknown format {format!r}; have {list(FORMATS)}")
    resolved = resolve(report, context)
    if isinstance(resolved, Unsupported):
        return resolved
    if format == "html":
        return render_html(resolved, inline_plotly=inline_plotly)
    if format == "pptx":
        return render_pptx(resolved)
    return render_pdf(resolved)


def write(
    report: Report,
    context: Mapping[str, object],
    path: str,
    format: Format | None = None,  # noqa: A002 - the domain word
    *,
    inline_plotly: bool = True,
) -> str | Unsupported:
    """Render and write to ``path``, returning the path. Format is inferred from the suffix.

    The one convenience function: a scheduled report is a call to this, and
    everything it can refuse — a missing context key, a missing extra — comes
    back as a typed failure that names what to fix.
    """
    import pathlib

    target = pathlib.Path(path)
    chosen = format
    if chosen is None:
        suffix = target.suffix.lower()
        matched = [f for f, s in _SUFFIX.items() if s == suffix]
        if not matched:
            raise ValueError(
                f"cannot infer a format from {target.name!r}; "
                f"use one of {sorted(_SUFFIX.values())} or pass format="
            )
        chosen = matched[0]  # type: ignore[assignment]
    rendered = render(report, context, chosen, inline_plotly=inline_plotly)  # type: ignore[arg-type]
    if isinstance(rendered, Unsupported):
        return rendered
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rendered, str):
        target.write_text(rendered, encoding="utf-8")
    else:
        target.write_bytes(rendered)
    return str(target)
