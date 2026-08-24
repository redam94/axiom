"""PDF, from the same template, through reportlab's flowables.

Paged output is the easy one: blocks become flowables and the library handles
pagination, so the only layout decisions here are the theme's. ``reportlab`` is
pure Python and pulls no system libraries, which is why it is the PDF backend
rather than an HTML-to-PDF engine — those want a browser or a cairo/pango
stack, and the dependency budget in ``pyproject.toml`` says no.

Figures arrive as PNG through ``render.images.figure_png``, so a PDF and a
slide of the same report show the same picture, band included.
"""

from __future__ import annotations

import importlib.util
import io
from typing import Any

from axiom.core import Unsupported
from axiom.report.render.images import figure_png
from axiom.report.resolve import (
    ResolvedFigure,
    ResolvedLedger,
    ResolvedMetric,
    ResolvedReport,
    ResolvedTable,
    ResolvedText,
)
from axiom.report.theme import Theme

__all__ = ["render_pdf"]


def _styles(theme: Theme) -> dict[str, Any]:
    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import ParagraphStyle

    def style(name: str, size: float, colour: str, **kw: Any) -> ParagraphStyle:
        return ParagraphStyle(
            name,
            fontName=theme.font,
            fontSize=size,
            leading=size * 1.4,
            textColor=HexColor(colour),
            **kw,
        )

    return {
        "title": style("title", theme.title_size, theme.text_color, spaceAfter=4),
        "subtitle": style("subtitle", theme.base_size * 1.15, theme.muted_color, spaceAfter=18),
        "h1": style("h1", theme.heading_size(1), theme.text_color, spaceBefore=16, spaceAfter=6),
        "h2": style("h2", theme.heading_size(2), theme.text_color, spaceBefore=10, spaceAfter=4),
        "h3": style("h3", theme.heading_size(3), theme.text_color, spaceBefore=8, spaceAfter=3),
        "body": style("body", theme.base_size, theme.text_color, spaceAfter=6),
        "emphasis": style("emphasis", theme.base_size * 1.1, theme.accent_color, spaceAfter=6),
        "muted": style("muted", theme.base_size * 0.9, theme.muted_color, spaceAfter=4),
        "metric_label": style("metric_label", theme.base_size * 0.85, theme.muted_color),
        "metric_value": style("metric_value", theme.title_size * 0.72, theme.accent_color),
        "caption": style("caption", theme.base_size * 0.9, theme.muted_color, spaceAfter=10),
        # Table cells are Paragraphs rather than bare strings so that long text
        # wraps inside its column. reportlab does not wrap a raw string: it draws
        # it at full width and lets it run over the next column, which turns a
        # table of statements into an unreadable overlap.
        "cell": style("cell", theme.base_size * 0.85, theme.text_color),
        "cell_head": style("cell_head", theme.base_size * 0.85, theme.muted_color),
    }


def _escape(text: str) -> str:
    """reportlab paragraphs are mini-HTML; the template's marks map onto its tags."""
    import re

    out = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    out = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", out)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", out)
    return out.replace("\n", "<br/>")


def render_pdf(resolved: ResolvedReport) -> bytes | Unsupported:
    """The report as a PDF, as bytes. ``Unsupported`` without the extra."""
    # Only the *absence of the extra* is a typed refusal. Anything else that goes
    # wrong importing it is a bug and must not be reported as "install the extra".
    if importlib.util.find_spec("reportlab") is None:
        return Unsupported(
            reason="pdf output needs reportlab; install the report extra",
            missing=("report",),
        )
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        HRFlowable,
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    theme: Theme = resolved.theme
    styles = _styles(theme)
    width, height = theme.page_geometry()
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=(width, height),
        leftMargin=theme.margin,
        rightMargin=theme.margin,
        topMargin=theme.margin,
        bottomMargin=theme.margin,
        title=resolved.title,
        author="axiom.report",
        subject=resolved.subtitle,
    )
    # reportlab measures in points, which is what the theme uses; no conversion.
    frame_width = width - 2 * theme.margin

    story: list[Any] = [Paragraph(_escape(resolved.title), styles["title"])]
    if resolved.subtitle:
        story.append(Paragraph(_escape(resolved.subtitle), styles["subtitle"]))

    for index, section in enumerate(resolved.sections):
        if index:
            story.append(PageBreak())
        story.append(Paragraph(_escape(section.title), styles["h1"]))
        story.append(
            HRFlowable(width="100%", color=HexColor(theme.rule_color), spaceAfter=8, thickness=0.7)
        )
        if section.summary:
            story.append(Paragraph(_escape(section.summary), styles["muted"]))

        pending: list[ResolvedMetric] = []

        def flush(pending: list[ResolvedMetric] = pending) -> None:
            if not pending:
                return
            cells = [
                [
                    Paragraph(_escape(m.label.upper()), styles["metric_label"]),
                    Paragraph(_escape(m.text), styles["metric_value"]),
                    Paragraph(_escape(m.interval_text), styles["metric_label"]),
                ]
                for m in pending
            ]
            table = Table(
                [list(column) for column in zip(*cells, strict=True)],
                colWidths=[frame_width / len(cells)] * len(cells),
            )
            table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("TOPPADDING", (0, 0), (-1, -1), 1),
                    ]
                )
            )
            story.extend([table, Spacer(1, 12)])
            pending.clear()

        for position, block in enumerate(section.blocks):
            if position in section.breaks:
                flush()
                story.append(PageBreak())
            if isinstance(block, ResolvedMetric):
                pending.append(block)
                continue
            flush()
            match block:
                case ResolvedText() if block.is_heading:
                    story.append(
                        Paragraph(_escape(block.text), styles[f"h{min(block.level + 1, 3)}"])
                    )
                case ResolvedText():
                    story.append(
                        Paragraph(
                            _escape(block.text),
                            styles["emphasis"] if block.emphasis else styles["body"],
                        )
                    )
                case ResolvedFigure():
                    png = figure_png(
                        block.figure, theme, height=block.height or theme.figure_height
                    )
                    if isinstance(png, Unsupported):
                        return png
                    image = Image(io.BytesIO(png))
                    scale = frame_width / image.imageWidth
                    image.drawWidth = frame_width
                    image.drawHeight = image.imageHeight * scale
                    story.append(image)
                    if block.caption:
                        story.append(Paragraph(_escape(block.caption), styles["caption"]))
                case ResolvedTable():
                    if not block.columns:
                        # A source that resolved to an empty sequence has no
                        # columns to size. Saying so beats dividing by zero, and
                        # beats dropping the block silently: an empty table in a
                        # report is usually a context key that went stale.
                        story.append(
                            Paragraph(
                                _escape(f"{block.caption or 'Table'}: no rows"),
                                styles["muted"],
                            )
                        )
                        continue
                    data = [
                        [Paragraph(_escape(str(c)), styles["cell_head"]) for c in block.columns]
                    ] + [
                        [Paragraph(_escape(str(c)), styles["cell"]) for c in row]
                        for row in block.rows
                    ]
                    table = Table(
                        data, colWidths=[frame_width / len(block.columns)] * len(block.columns)
                    )
                    table.setStyle(
                        TableStyle(
                            [
                                ("FONTNAME", (0, 0), (-1, -1), theme.font),
                                ("FONTSIZE", (0, 0), (-1, -1), theme.base_size * 0.85),
                                ("TEXTCOLOR", (0, 0), (-1, -1), HexColor(theme.text_color)),
                                ("LINEBELOW", (0, 0), (-1, 0), 1.0, HexColor(theme.rule_color)),
                                ("LINEBELOW", (0, 1), (-1, -2), 0.4, HexColor(theme.rule_color)),
                                # wrapped cells read top-aligned; RIGHT would ragged
                                # the left edge of every multi-line statement
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                                ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ]
                        )
                    )
                    story.append(table)
                    if block.caption:
                        story.append(Paragraph(_escape(block.caption), styles["caption"]))
                    if block.truncated:
                        story.append(
                            Paragraph(
                                f"{block.truncated} further row(s) not shown", styles["muted"]
                            )
                        )
                case ResolvedLedger():
                    for line in block.lines:
                        story.append(
                            Paragraph(
                                f"<b>{_escape(line.kind)}</b>  {_escape(line.statement)}",
                                styles["body"],
                            )
                        )
                        if line.assumption is not None:
                            story.append(
                                Paragraph(
                                    f"assumes <i>{_escape(line.assumption.name)}</i> "
                                    f"({_escape(line.assumption.state)}): "
                                    f"{_escape(line.assumption.statement)}",
                                    styles["muted"],
                                )
                            )
                    if block.caption:
                        story.append(Paragraph(_escape(block.caption), styles["caption"]))
        flush()

    if resolved.footer:
        story.extend(
            [
                Spacer(1, 18),
                HRFlowable(width="100%", color=HexColor(theme.rule_color), thickness=0.7),
                Paragraph(_escape(resolved.footer), styles["muted"]),
            ]
        )
    document.build(story)
    return buffer.getvalue()
