"""Interactive HTML: one self-contained file, plotly figures still live.

The only renderer whose figures stay interactive, and the one with the lightest
dependency — plotly and nothing else. ``plotly.js`` is inlined once for the
whole document (``inline_plotly=False`` swaps it for the CDN, which is smaller
but needs the network to open), and every figure after the first reuses it.

The CSS is generated from the ``Theme`` rather than templated from a file, so
the same theme values drive all three renderers and there is no stylesheet to
drift out of sync with the PPTX and PDF output. The theme supplies the values;
the stylesheet below supplies the layout, and reads them through custom
properties so there is exactly one place either can change.

The layout is a journal article's, because that is the shape these reports
have: a measure of about seventy-five characters for prose, figures and tables
breaking out wider than the text they interrupt, horizontal rules on tables
rather than a grid of boxes, captions above tables and below figures, and a
contents rail to navigate numbered sections with. That is not decoration — a
document a reader has to fight is one whose numbers do not get read.
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
#: The ``Table 3.`` / ``Figure 1:`` a caption often opens with, so it can be set
#: apart from the sentence after it the way a journal sets it.
_CAPTION_LABEL = re.compile(r"^((?:Table|Figure|Fig\.|Exhibit)\s+[\w.]+\s*[.:]?)(\s+|$)")
#: A leading section number, so ``2. Methods`` is recognisable as the methods
#: section and the contents rail can set the number apart from the title.
_SECTION_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)*)[.)]?\s+")


def _inline(text: str) -> str:
    """Escape, then apply the three inline marks the template language allows.

    Emphasis is applied **outside code spans only**. A code span is literal —
    that is the whole of what it is for — and running the italic pass across
    the substituted output turned the two asterisks of
    ``beta0 + tau * treated_i + beta1 * baseline_i`` into an ``<em>`` and
    deleted them from the equation.
    """
    out = _html.escape(text)
    # split on a capturing group: odd indices are the contents of code spans.
    parts = _CODE.split(out)
    for i, part in enumerate(parts):
        if i % 2:
            parts[i] = f"<code>{part}</code>"
        else:
            part = _BOLD.sub(r"<strong>\1</strong>", part)
            parts[i] = _ITALIC.sub(r"<em>\1</em>", part)
    return "".join(parts).replace("\n", "<br>")


def _rgba(colour: str, alpha: float) -> str:
    """A ``#rrggbb`` theme colour as ``rgba()``, for tints and washes of it."""
    r, g, b = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _caption_html(text: str) -> str:
    """A caption with its ``Table 2.`` label set apart, or nothing at all."""
    if not text:
        return ""
    body = _CAPTION_LABEL.sub(r'<span class="label">\1</span>\2', _inline(text), count=1)
    return f"<figcaption>{body}</figcaption>"


def _split_number(title: str) -> tuple[str, str]:
    """``("2", "Methods")`` for a numbered title, ``("", title)`` for anything else."""
    match = _SECTION_NUMBER.match(title)
    if match is None:
        return "", title.strip()
    return match.group(1), title[match.end() :].strip()


def _tokens(theme: Theme) -> str:
    """The theme, as the custom properties the stylesheet is written against."""
    return f"""
:root {{
  color-scheme: light;
  --ink: {theme.text_color};
  --muted: {theme.muted_color};
  --accent: {theme.accent_color};
  --bg: {theme.background_color};
  --rule: {theme.rule_color};
  --tint: {_rgba(theme.accent_color, 0.055)};
  --wash: {_rgba(theme.text_color, 0.032)};
  --serif: {theme.font}, {theme.font_fallback};
  --mono: {theme.mono_font}, ui-monospace, SFMono-Regular, Menlo, monospace;
  --size-title: {theme.title_size}pt;
  --size-h2: {theme.heading_size(1)}pt;
  --size-h3: {theme.heading_size(2)}pt;
  --size-h4: {theme.heading_size(3)}pt;
  --measure: 34em;
  --wide: 46em;
  --rail: 12.5em;
  --gutter: clamp(1.15rem, 4vw, {theme.margin}pt);
}}
html {{ font-size: {theme.base_size}pt; }}
@page {{ margin: {theme.margin}pt; }}
"""


#: Everything that is layout rather than taste. Written against the custom
#: properties above, so a theme change never edits a rule.
_STYLESHEET = """
*, *::before, *::after { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; scroll-behavior: smooth; }
body {
  margin: 0;
  padding: var(--gutter) var(--gutter) calc(var(--gutter) * 1.6);
  background: var(--bg); color: var(--ink);
  font-family: var(--serif); font-size: 1rem; line-height: 1.62;
  font-kerning: normal; text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
}

/* --- the page: the article centred, a contents rail in the left margin --- */
.paper { display: grid; grid-template-columns: minmax(0, 1fr); gap: 2rem; }
main { width: 100%; max-width: var(--wide); margin-inline: auto; min-width: 0; }
@media (min-width: 68em) {
  /* Out of the flow, so the rail cannot pull the article off centre: a reader
     navigating by contents and one reading straight through see the same page. */
  .toc {
    position: fixed; top: var(--gutter); width: var(--rail);
    left: max(var(--gutter), calc(50% - var(--wide) / 2 - var(--rail) - 2.5em));
    max-height: calc(100vh - 2 * var(--gutter)); overflow: auto;
  }
}

.toc { font-size: .8em; line-height: 1.4; }
.toc-title {
  margin: 0 0 .7em; padding: 0; border: 0; font-size: 1em; font-weight: 700;
  text-transform: uppercase; letter-spacing: .11em; color: var(--muted);
}
.toc ol { list-style: none; margin: 0; padding: 0; }
.toc a {
  display: flex; gap: .5em; padding: .32em .7em; color: var(--muted);
  text-decoration: none; border-left: 2px solid var(--rule);
}
.toc a:hover, .toc a:focus-visible {
  color: var(--ink); border-left-color: var(--accent); background: var(--wash);
}
.toc .n { flex: 0 0 .9em; text-align: right; font-variant-numeric: tabular-nums; opacity: .75; }
@media (max-width: 67.99em) {
  .toc {
    border: 1px solid var(--rule); padding: 1em 1.2em;
    width: 100%; max-width: var(--wide); margin-inline: auto;
  }
  .toc ol { columns: 2 11em; column-gap: 2em; }
  .toc a { border-left: 0; padding: .22em 0; }
  .toc a:hover, .toc a:focus-visible { background: none; }
}

/* --- title block --- */
.titleblock {
  text-align: center; margin: 0 0 2.6em;
  padding-bottom: 1.4em; border-bottom: 1px solid var(--rule);
}
h1 {
  font-size: var(--size-title); line-height: 1.18; font-weight: 700;
  letter-spacing: -.008em; margin: 0 0 .4em; text-wrap: balance;
}
.subtitle {
  color: var(--muted); font-size: 1.06em; font-style: italic; line-height: 1.45;
  margin: 0 auto; max-width: 32em; text-align: center; text-wrap: pretty;
}

/* --- sections: prose on the measure, figures wider than it --- */
.section {
  display: grid;
  grid-template-columns:
    [wide-start] minmax(0, 1fr)
    [text-start] minmax(0, var(--measure))
    [text-end] minmax(0, 1fr) [wide-end];
}
.section > * { grid-column: text; min-width: 0; }
.section > figure, .section > .metrics { grid-column: wide; }
.section.abstract { font-size: .96em; }
.section.abstract > h2 {
  text-align: center; border: 0; padding: 0; font-size: 1em; font-weight: 700;
  text-transform: uppercase; letter-spacing: .12em; margin: 0 0 .5em;
}
.section.abstract > * { grid-column: text; }

h2, h3, h4 { break-after: avoid; scroll-margin-top: 1.2rem; text-wrap: balance; }
h2 {
  font-size: var(--size-h2); font-weight: 700; line-height: 1.25;
  letter-spacing: -.005em; margin: 2.3em 0 .55em;
  padding-bottom: .28em; border-bottom: 1px solid var(--rule);
}
.section:first-of-type h2 { margin-top: .4em; }
h3 { font-size: var(--size-h3); font-weight: 700; margin: 1.9em 0 .35em; }
h4 { font-size: var(--size-h4); font-weight: 700; margin: 1.7em 0 .3em; }
h2 .anchor, h3 .anchor {
  float: right; margin-left: .5em; color: var(--rule); text-decoration: none;
  font-size: .72em; font-weight: 400; opacity: 0; transition: opacity .12s;
}
h2:hover .anchor, h3:hover .anchor, .anchor:focus-visible { opacity: 1; color: var(--accent); }

p { margin: 0 0 .8em; text-align: justify; hyphens: auto; }
p:last-child { margin-bottom: 0; }
.summary { color: var(--muted); margin: 0 0 1.1em; }
a { color: var(--accent); text-underline-offset: .16em; text-decoration-thickness: .06em; }
strong { font-weight: 700; }
/* `white-space: pre-wrap` is not a detail: a readout is column-aligned output,
   and HTML collapses the columns away without it. */
code {
  font-family: var(--mono); font-size: .86em; background: var(--wash);
  border-radius: 2px; padding: .1em .35em; white-space: pre-wrap;
}
hr { border: 0; border-top: 1px solid var(--rule); margin: 2em 0; }

/* A pulled-out line: the parameters a step ran with, or a remark on a number. */
p.emphasis {
  margin: 1em 0 1.1em; padding: .6em .95em; text-align: left;
  border-left: 2px solid var(--accent); background: var(--tint);
  font-size: .95em; line-height: 1.5;
}

/* --- figures and tables --- */
figure { margin: 1.9em 0; padding: 0; }
figcaption {
  color: var(--muted); font-size: .85em; line-height: 1.45;
  margin-top: .75em; text-align: left;
}
figcaption .label { color: var(--ink); font-weight: 700; }
.table-figure { overflow-x: auto; }
/* A table is read from its caption down; a figure is read up to it. */
.table-figure > figcaption:first-child { margin: 0 0 .6em; }
table {
  border-collapse: collapse; width: 100%; margin: 0;
  font-size: .88em; line-height: 1.42; font-variant-numeric: tabular-nums;
}
thead th {
  text-align: left; font-weight: 700; padding: .5em .7em; vertical-align: bottom;
  border-top: 1.5px solid var(--ink); border-bottom: 1px solid var(--ink);
}
tbody td { padding: .42em .7em; vertical-align: top; border: 0; }
tbody tr:nth-child(even) { background: var(--wash); }
tbody tr:last-child td { border-bottom: 1.5px solid var(--ink); }
td.num, th.num { text-align: right; }

/* --- the numbers, as a panel of hairline-separated cells --- */
.metrics {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(12em, 1fr));
  gap: 1px; background: var(--rule); border: 1px solid var(--rule);
  margin: 1.7em 0;
}
.metric {
  background: var(--bg); padding: .8em .95em;
  display: flex; flex-direction: column;
}
.metric .label { color: var(--muted); font-size: .8em; font-weight: 600; line-height: 1.35; }
.metric .value {
  font-size: 1.22em; font-weight: 600; line-height: 1.28; margin-top: .3em;
  color: var(--accent); font-variant-numeric: tabular-nums; text-wrap: balance;
}
/* The interval sits on the floor of its cell, so a row of them reads as a row
   however differently the values above them wrap. */
.metric .interval {
  color: var(--muted); font-size: .8em; line-height: 1.35; margin-top: auto;
  padding-top: .25em; font-variant-numeric: tabular-nums;
}
.metric .interval:empty { display: none; }

/* --- the assumption trail --- */
.ledger {
  margin: 1.7em 0; padding: .9em 1.1em;
  background: var(--wash); border-left: 2px solid var(--accent);
  font-size: .93em; line-height: 1.5;
}
.ledger > figcaption:first-child {
  margin: 0 0 .7em; color: var(--ink); font-weight: 700;
  font-size: .88em; text-transform: uppercase; letter-spacing: .07em;
}
.ledger .line { margin: 0 0 .75em; }
.ledger .line:last-child { margin-bottom: 0; }
.ledger .kind {
  display: inline-block; font-family: var(--mono); font-size: .78em;
  color: var(--muted); background: var(--bg); border: 1px solid var(--rule);
  border-radius: 2px; padding: .05em .4em; margin-right: .4em;
  vertical-align: baseline;
}
.ledger .assumption { color: var(--muted); font-size: .93em; margin-top: .25em; }
.ledger .assumption code { font-size: .95em; }

footer {
  margin-top: 3em; padding-top: .9em; border-top: 1px solid var(--rule);
  color: var(--muted); font-size: .82em; line-height: 1.5;
}

@media print {
  html { scroll-behavior: auto; }
  body { padding: 0; }
  .toc, .anchor { display: none !important; }
  .paper { display: block; }
  main { max-width: none; }
  .section { display: block; }
  a { color: inherit; text-decoration: none; }
  /* An article flows: a section does not start a fresh sheet, it starts where
     the last one ended, and only the things that break badly are held together. */
  p { orphans: 3; widows: 3; }
  figure, .metrics, .ledger { break-inside: avoid; }
  thead { display: table-header-group; }
}
"""


def _css(theme: Theme) -> str:
    return _tokens(theme) + _STYLESHEET


def _figure_html(block: ResolvedFigure, theme: Theme, first: bool, inline_plotly: bool) -> str:
    height = block.height or theme.figure_height
    figure = block.figure
    figure.update_layout(
        height=height,
        margin={"l": 55, "r": 25, "t": 40 if figure.layout.title.text else 12, "b": 45},
        # The family, not the face: `Times-Roman` is a PostScript name a browser
        # cannot resolve, so a figure lettered with it alone falls back to
        # sans-serif inside a serif document.
        font={"family": f"{theme.font}, {theme.font_fallback}", "size": theme.base_size},
        paper_bgcolor=theme.background_color,
        plot_bgcolor=theme.background_color,
        colorway=list(theme.palette),
        # Estimand names are long enough to run off the left of a fixed margin;
        # automargin grows it instead of drawing the label over the panel.
        xaxis_automargin=True,
        yaxis_automargin=True,
    )
    include: Any = (True if inline_plotly else "cdn") if first else False
    body = figure.to_html(include_plotlyjs=include, full_html=False, default_height=height)
    return f"<figure>{body}{_caption_html(block.caption)}</figure>"


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
    note = (
        f"<figcaption>{block.truncated} further row(s) not shown</figcaption>"
        if block.truncated
        else ""
    )
    # A table's caption goes above the table it names, a figure's below.
    return (
        f'<figure class="table-figure">{_caption_html(block.caption)}'
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table>{note}</figure>"
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
    return f'<figure class="ledger">{_caption_html(block.caption)}{"".join(lines)}</figure>'


def _section_html(
    section: ResolvedSection, anchor: str, theme: Theme, state: dict[str, bool], inline: bool
) -> str:
    number, title = _split_number(section.title)
    classes = "section" + (" abstract" if title.lower() == "abstract" else "")
    link = f'<a class="anchor" href="#{anchor}" aria-hidden="true">§</a>'
    parts = [f"<h2>{link}{_html.escape(section.title)}</h2>"]
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
    body = "\n".join(parts)
    return f'<section id="{anchor}" class="{classes}">\n{body}\n</section>'


def _toc_html(sections: tuple[ResolvedSection, ...], anchors: list[str]) -> str:
    """A contents rail, once there are enough sections for one to be worth having."""
    if len(sections) < 3:
        return ""
    items = []
    for section, anchor in zip(sections, anchors, strict=True):
        number, title = _split_number(section.title)
        label = f'<span class="n">{_html.escape(number)}</span>' if number else ""
        items.append(f'<li><a href="#{anchor}">{label}<span>{_html.escape(title)}</span></a></li>')
    return (
        '<nav class="toc" aria-label="Contents">'
        '<div class="toc-title">Contents</div>'
        f'<ol>{"".join(items)}</ol></nav>'
    )


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
    anchors = [f"section-{i}" for i, _ in enumerate(resolved.sections, start=1)]
    body = "\n".join(
        _section_html(s, a, theme, state, inline_plotly)
        for s, a in zip(resolved.sections, anchors, strict=True)
    )
    subtitle = f'<p class="subtitle">{_inline(resolved.subtitle)}</p>' if resolved.subtitle else ""
    footer = f"<footer>{_inline(resolved.footer)}</footer>" if resolved.footer else ""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="axiom.report">
<meta name="axiom-report-hash" content="{resolved.source_hash}">
<title>{_html.escape(resolved.title)}</title>
<style>{_css(theme)}</style>
</head><body><div class="paper">
{_toc_html(resolved.sections, anchors)}
<main>
<header class="titleblock">
<h1>{_html.escape(resolved.title)}</h1>
{subtitle}
</header>
{body}
{footer}
</main>
</div></body></html>
"""
