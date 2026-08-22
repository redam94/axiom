"""PowerPoint, from the same template.

Slides need a packing rule, and an implicit one is how decks end up with text
running off the bottom. The rule here is stated and small:

* every section starts a slide, titled with the section title;
* a ``PageBreak`` starts another slide inside the section, continuing the title;
* a figure takes its own slide, because a figure squeezed beside a paragraph is
  not worth putting on a screen;
* metrics gather into a row of cards, up to four per slide;
* anything else flows down the body until the slide is full, then continues.

The theme drives colours and fonts, and figures come through
``render.images.figure_png`` so a slide and a PDF of the same report show the
same picture — including the uncertainty band, which is part of the figure and
not something this renderer could drop if it wanted to.
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
    ResolvedSection,
    ResolvedTable,
    ResolvedText,
)
from axiom.report.theme import Theme

__all__ = ["render_pptx"]

#: How many body blocks fit on one slide before it continues onto another.
BODY_BLOCKS_PER_SLIDE = 5
#: How many metric cards fit in one row.
METRICS_PER_ROW = 4


def _rgb(pptx: Any, colour: str) -> Any:
    return pptx.dml.color.RGBColor.from_string(colour.lstrip("#").upper())


def _pack(section: ResolvedSection) -> list[list[Any]]:
    """Split a section's blocks into slides by the rule in the module docstring."""
    slides: list[list[Any]] = [[]]
    body = 0
    for index, block in enumerate(section.blocks):
        if index in section.breaks and slides[-1]:
            slides.append([])
            body = 0
        if isinstance(block, ResolvedFigure):
            if slides[-1]:
                slides.append([])
            slides[-1].append(block)
            slides.append([])
            body = 0
            continue
        if body >= BODY_BLOCKS_PER_SLIDE:
            slides.append([])
            body = 0
        slides[-1].append(block)
        body += 1
    return [s for s in slides if s]


def _add_textbox(slide: Any, pptx: Any, box: tuple[float, float, float, float]) -> Any:
    left, top, width, height = (pptx.util.Pt(v) for v in box)
    frame = slide.shapes.add_textbox(left, top, width, height).text_frame
    frame.word_wrap = True
    return frame


def _write(
    para: Any, pptx: Any, text: str, size: float, colour: str, *, bold: bool = False
) -> None:
    run = para.add_run()
    run.text = text
    run.font.size = pptx.util.Pt(size)
    run.font.bold = bold
    run.font.color.rgb = _rgb(pptx, colour)


def render_pptx(resolved: ResolvedReport) -> bytes | Unsupported:
    """The report as a ``.pptx`` file, as bytes. ``Unsupported`` without the extra."""
    # As in the pdf renderer: only a missing extra is a typed refusal.
    if importlib.util.find_spec("pptx") is None:
        return Unsupported(
            reason="pptx output needs python-pptx; install the report extra",
            missing=("report",),
        )
    import pptx as pptx_module
    from pptx import Presentation

    theme: Theme = resolved.theme
    width, height = theme.slide_geometry()
    presentation = Presentation()
    presentation.slide_width = pptx_module.util.Pt(width)
    presentation.slide_height = pptx_module.util.Pt(height)
    blank = presentation.slide_layouts[6]
    margin = theme.margin
    body_width = width - 2 * margin

    # -- title slide ------------------------------------------------------------------
    slide = presentation.slides.add_slide(blank)
    frame = _add_textbox(slide, pptx_module, (margin, height * 0.34, body_width, height * 0.3))
    _write(
        frame.paragraphs[0],
        pptx_module,
        resolved.title,
        theme.title_size,
        theme.text_color,
        bold=True,
    )
    if resolved.subtitle:
        _write(
            frame.add_paragraph(),
            pptx_module,
            resolved.subtitle,
            theme.base_size * 1.3,
            theme.muted_color,
        )

    for section in resolved.sections:
        for page, blocks in enumerate(_pack(section)):
            slide = presentation.slides.add_slide(blank)
            head = _add_textbox(slide, pptx_module, (margin, margin * 0.5, body_width, 34.0))
            title = section.title if page == 0 else f"{section.title} (cont.)"
            _write(
                head.paragraphs[0],
                pptx_module,
                title,
                theme.heading_size(1),
                theme.text_color,
                bold=True,
            )

            top = margin * 0.5 + 40.0
            if page == 0 and section.summary:
                summary = _add_textbox(slide, pptx_module, (margin, top, body_width, 30.0))
                _write(
                    summary.paragraphs[0],
                    pptx_module,
                    section.summary,
                    theme.base_size,
                    theme.muted_color,
                )
                top += 34.0

            metrics = [b for b in blocks if isinstance(b, ResolvedMetric)]
            if metrics:
                card = body_width / min(len(metrics), METRICS_PER_ROW)
                for i, metric in enumerate(metrics[:METRICS_PER_ROW]):
                    box = _add_textbox(
                        slide, pptx_module, (margin + i * card, top, card - 8.0, 70.0)
                    )
                    _write(
                        box.paragraphs[0],
                        pptx_module,
                        metric.label.upper(),
                        theme.base_size * 0.85,
                        theme.muted_color,
                    )
                    _write(
                        box.add_paragraph(),
                        pptx_module,
                        metric.text,
                        theme.title_size * 0.8,
                        theme.accent_color,
                        bold=True,
                    )
                    if metric.interval_text:
                        _write(
                            box.add_paragraph(),
                            pptx_module,
                            metric.interval_text,
                            theme.base_size * 0.85,
                            theme.muted_color,
                        )
                top += 82.0

            for block in blocks:
                if isinstance(block, ResolvedMetric):
                    continue
                if isinstance(block, ResolvedFigure):
                    png = figure_png(
                        block.figure, theme, height=block.height or (height - top - margin) * 0.92
                    )
                    if isinstance(png, Unsupported):
                        return png
                    picture_height = min(height - top - margin, (height - top - margin))
                    slide.shapes.add_picture(
                        io.BytesIO(png),
                        pptx_module.util.Pt(margin),
                        pptx_module.util.Pt(top),
                        height=pptx_module.util.Pt(picture_height),
                    )
                    if block.caption:
                        cap = _add_textbox(
                            slide, pptx_module, (margin, height - margin * 0.9, body_width, 20.0)
                        )
                        _write(
                            cap.paragraphs[0],
                            pptx_module,
                            block.caption,
                            theme.base_size * 0.85,
                            theme.muted_color,
                        )
                    break
                if isinstance(block, ResolvedText):
                    box = _add_textbox(slide, pptx_module, (margin, top, body_width, 40.0))
                    size = theme.heading_size(block.level) if block.is_heading else theme.base_size
                    colour = theme.text_color if not block.emphasis else theme.accent_color
                    _write(
                        box.paragraphs[0],
                        pptx_module,
                        block.text,
                        size,
                        colour,
                        bold=block.is_heading,
                    )
                    top += 20.0 + 6.0 * (block.text.count("\n") + len(block.text) // 110)
                elif isinstance(block, ResolvedTable):
                    rows, cols = len(block.rows) + 1, len(block.columns)
                    table_height = min(24.0 * rows, height - top - margin)
                    shape = slide.shapes.add_table(
                        rows,
                        cols,
                        pptx_module.util.Pt(margin),
                        pptx_module.util.Pt(top),
                        pptx_module.util.Pt(body_width),
                        pptx_module.util.Pt(table_height),
                    ).table
                    for c, name in enumerate(block.columns):
                        cell = shape.cell(0, c)
                        cell.text = name
                        _style_cell(cell, pptx_module, theme, bold=True)
                    for r, row in enumerate(block.rows, start=1):
                        for c, value in enumerate(row):
                            cell = shape.cell(r, c)
                            cell.text = value
                            _style_cell(cell, pptx_module, theme)
                    top += table_height + 12.0
                elif isinstance(block, ResolvedLedger):
                    box = _add_textbox(
                        slide, pptx_module, (margin, top, body_width, height - top - margin)
                    )
                    first = True
                    for line in block.lines:
                        para = box.paragraphs[0] if first else box.add_paragraph()
                        first = False
                        _write(
                            para,
                            pptx_module,
                            f"{line.kind}  ",
                            theme.base_size * 0.85,
                            theme.accent_color,
                            bold=True,
                        )
                        _write(
                            para,
                            pptx_module,
                            line.statement,
                            theme.base_size * 0.9,
                            theme.text_color,
                        )
                    top += 18.0 * (len(block.lines) + 1)

    buffer = io.BytesIO()
    presentation.save(buffer)
    return buffer.getvalue()


def _style_cell(cell: Any, pptx: Any, theme: Theme, *, bold: bool = False) -> None:
    for paragraph in cell.text_frame.paragraphs:
        for run in paragraph.runs:
            run.font.size = pptx.util.Pt(theme.base_size * 0.85)
            run.font.bold = bold
            run.font.color.rgb = _rgb(pptx, theme.text_color)
