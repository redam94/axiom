"""Interactive HTML: one self-contained file, plotly figures still live.

The only renderer whose figures stay interactive, and the one with the lightest
dependency — plotly and nothing else. ``plotly.js`` is inlined once for the
whole document (``inline_plotly=False`` swaps it for the CDN, which is smaller
but needs the network to open), and every figure after the first reuses it.

The CSS is generated from the ``Theme`` rather than templated from a file, so
the same theme values drive all three renderers and there is no stylesheet to
drift out of sync with the PPTX and PDF output.
"""

from __future__ import annotations

import html as _html
import importlib
import re
from typing import Any

from axiom.core import Unsupported
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

__all__ = ["render_html"]

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_CODE = re.compile(r"`(.+?)`")


def _inline(text: str) -> str:
    """Escape, then apply the three inline marks the template language allows."""
    out = _html.escape(text)
    out = _CODE.sub(r"<code>\1</code>", out)
    out = _BOLD.sub(r"<strong>\1</strong>", out)
    out = _ITALIC.sub(r"<em>\1</em>", out)
    return out.replace("\n", "<br>")


def _css(theme: Theme) -> str:
    width, _ = theme.page_geometry()
    return f"""
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; padding: {theme.margin}pt;
  background: {theme.background_color}; color: {theme.text_color};
  font-family: {theme.font}, system-ui, -apple-system, sans-serif;
  font-size: {theme.base_size}pt; line-height: 1.55;
}}
main {{ max-width: {width * 1.6}pt; margin: 0 auto; }}
h1 {{ font-size: {theme.title_size}pt; margin: 0 0 .15em; letter-spacing: -.01em; }}
h2 {{ font-size: {theme.heading_size(1)}pt; margin: 1.6em 0 .4em;
      padding-bottom: .25em; border-bottom: 1px solid {theme.rule_color}; }}
h3 {{ font-size: {theme.heading_size(2)}pt; margin: 1.2em 0 .3em; }}
h4 {{ font-size: {theme.heading_size(3)}pt; margin: 1em 0 .3em; }}
p {{ margin: .5em 0; }}
p.emphasis {{ font-size: {theme.base_size * 1.12}pt; color: {theme.accent_color}; }}
.subtitle {{ color: {theme.muted_color}; font-size: {theme.base_size * 1.15}pt;
             margin: 0 0 1.4em; }}
.summary {{ color: {theme.muted_color}; margin: 0 0 .8em; }}
code {{ font-family: {theme.mono_font}, ui-monospace, monospace;
        background: rgba(0,0,0,.05); padding: .08em .3em; border-radius: 3px; }}
figure {{ margin: 1.1em 0; }}
figcaption {{ color: {theme.muted_color}; font-size: {theme.base_size * .92}pt; margin-top: .3em; }}
table {{ border-collapse: collapse; width: 100%; margin: .8em 0;
         font-size: {theme.base_size * .95}pt; }}
th, td {{ text-align: left; padding: .35em .6em; border-bottom: 1px solid {theme.rule_color}; }}
th {{ font-weight: 600; border-bottom: 2px solid {theme.rule_color}; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.metrics {{ display: flex; flex-wrap: wrap; gap: 1.4em; margin: 1em 0; }}
.metric {{ min-width: 12em; }}
.metric .label {{ color: {theme.muted_color}; font-size: {theme.base_size * .9}pt;
                  text-transform: uppercase; letter-spacing: .04em; }}
.metric .value {{ font-size: {theme.title_size * .82}pt; font-weight: 600;
                  color: {theme.accent_color}; font-variant-numeric: tabular-nums; }}
.metric .interval {{ color: {theme.muted_color}; font-size: {theme.base_size * .9}pt;
                     font-variant-numeric: tabular-nums; }}
.ledger {{ border-left: 3px solid {theme.rule_color}; padding-left: .9em; margin: 1em 0; }}
.ledger .line {{ margin: .55em 0; }}
.ledger .kind {{ color: {theme.accent_color}; font-family: {theme.mono_font}, monospace;
                 font-size: {theme.base_size * .85}pt; }}
.ledger .assumption {{ color: {theme.muted_color}; font-size: {theme.base_size * .9}pt; }}
hr {{ border: 0; border-top: 1px solid {theme.rule_color}; margin: 1.4em 0; }}
footer {{ margin-top: 2.5em; padding-top: .8em; border-top: 1px solid {theme.rule_color};
          color: {theme.muted_color}; font-size: {theme.base_size * .85}pt; }}
@media print {{ body {{ padding: 0; }} h2 {{ break-before: page; }} }}
"""


def _figure_html(block: ResolvedFigure, theme: Theme, first: bool, inline_plotly: bool) -> str:
    height = block.height or theme.figure_height
    figure = block.figure
    figure.update_layout(
        height=height,
        margin={"l": 55, "r": 25, "t": 40 if figure.layout.title.text else 12, "b": 45},
        font={"family": theme.font, "size": theme.base_size},
        paper_bgcolor=theme.background_color,
        plot_bgcolor=theme.background_color,
        colorway=list(theme.palette),
    )
    include: Any = (True if inline_plotly else "cdn") if first else False
    body = figure.to_html(include_plotlyjs=include, full_html=False, default_height=height)
    caption = f"<figcaption>{_inline(block.caption)}</figcaption>" if block.caption else ""
    return f"<figure>{body}{caption}</figure>"


def _table_html(block: ResolvedTable) -> str:
    def numeric(column: int) -> bool:
        return all(
            not row[column]
            or row[column].replace(",", "").replace("-", "").replace(".", "").isdigit()
            for row in block.rows
        )

    classes = ["num" if numeric(i) else "" for i in range(len(block.columns))]
    head = "".join(
        f'<th class="{c}">{_html.escape(name)}</th>'
        for name, c in zip(block.columns, classes, strict=True)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="{c}">{_html.escape(cell)}</td>'
            for cell, c in zip(row, classes, strict=True)
        )
        + "</tr>"
        for row in block.rows
    )
    caption = f"<figcaption>{_inline(block.caption)}</figcaption>" if block.caption else ""
    note = (
        f"<figcaption>{block.truncated} further row(s) not shown</figcaption>"
        if block.truncated
        else ""
    )
    return (
        f"<figure><table><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table>{caption}{note}</figure>"
    )


def _metrics_html(metrics: list[ResolvedMetric]) -> str:
    cards = "".join(
        f'<div class="metric"><div class="label">{_html.escape(m.label)}</div>'
        f'<div class="value">{_html.escape(m.text)}</div>'
        f'<div class="interval">{_html.escape(m.interval_text)}</div></div>'
        for m in metrics
    )
    return f'<div class="metrics">{cards}</div>'


def _ledger_html(block: ResolvedLedger) -> str:
    lines = []
    for line in block.lines:
        assumption = ""
        if line.assumption is not None:
            assumption = (
                f'<div class="assumption">assumes <em>{_html.escape(line.assumption.name)}</em>'
                f" ({_html.escape(line.assumption.state)}): "
                f"{_html.escape(line.assumption.statement)}</div>"
            )
        detail = ""
        if block.show_detail and line.detail:
            pairs = ", ".join(f"{k}={v}" for k, v in sorted(line.detail.items()))
            detail = f'<div class="assumption"><code>{_html.escape(pairs)}</code></div>'
        lines.append(
            f'<div class="line"><span class="kind">{_html.escape(line.kind)}</span> '
            f"{_inline(line.statement)}{assumption}{detail}</div>"
        )
    caption = f"<figcaption>{_inline(block.caption)}</figcaption>" if block.caption else ""
    return f'<div class="ledger">{"".join(lines)}{caption}</div>'


def _section_html(
    section: ResolvedSection, theme: Theme, state: dict[str, bool], inline: bool
) -> str:
    parts = [f"<h2>{_html.escape(section.title)}</h2>"]
    if section.summary:
        parts.append(f'<p class="summary">{_inline(section.summary)}</p>')
    pending: list[ResolvedMetric] = []

    def flush() -> None:
        if pending:
            parts.append(_metrics_html(pending))
            pending.clear()

    for block in section.blocks:
        if isinstance(block, ResolvedMetric):
            pending.append(block)
            continue
        flush()
        match block:
            case ResolvedText() if block.is_heading:
                tag = f"h{min(block.level + 2, 4)}"
                parts.append(f"<{tag}>{_inline(block.text)}</{tag}>")
            case ResolvedText():
                cls = ' class="emphasis"' if block.emphasis else ""
                parts.append(f"<p{cls}>{_inline(block.text)}</p>")
            case ResolvedFigure():
                parts.append(_figure_html(block, theme, state["first_figure"], inline))
                state["first_figure"] = False
            case ResolvedTable():
                parts.append(_table_html(block))
            case ResolvedLedger():
                parts.append(_ledger_html(block))
    flush()
    return "\n".join(parts)


def render_html(resolved: ResolvedReport, *, inline_plotly: bool = True) -> str | Unsupported:
    """A complete, self-contained HTML document. Figures stay interactive.

    ``Unsupported`` only when a figure is present and plotly is not; a report of
    text, tables, metrics and a ledger renders on the core dependencies alone.
    """
    theme = resolved.theme
    has_figures = any(isinstance(b, ResolvedFigure) for s in resolved.sections for b in s.blocks)
    if has_figures:
        # plotly ships no type information; go through importlib, as viz does.
        try:
            importlib.import_module("plotly")
        except ImportError:
            return Unsupported(reason="plotly is needed to render figures", missing=("viz",))
    state = {"first_figure": True}
    body = "\n".join(_section_html(s, theme, state, inline_plotly) for s in resolved.sections)
    subtitle = f'<p class="subtitle">{_inline(resolved.subtitle)}</p>' if resolved.subtitle else ""
    footer = f"<footer>{_inline(resolved.footer)}</footer>" if resolved.footer else ""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="axiom.report">
<meta name="axiom-report-hash" content="{resolved.source_hash}">
<title>{_html.escape(resolved.title)}</title>
<style>{_css(theme)}</style>
</head><body><main>
<h1>{_html.escape(resolved.title)}</h1>
{subtitle}
{body}
{footer}
</main></body></html>
"""
