"""Assemble the axiom site from content fragments plus generated data.

Content lives in ``site/_src/*.html`` as body fragments. This script wraps each in
the shared shell (head, workflow nav, footer) and substitutes ``{{path:spec}}``
tokens with values from ``site/assets/data/*.json`` — so a number written into the
prose is the number a real run produced, baked in at build time rather than
fetched by script.

    python site/_gen/build.py
"""

from __future__ import annotations

import html
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "site"
SRC = SITE / "_src"
DATA = SITE / "assets" / "data"

# The five pillars in the order you actually meet them in a piece of work. The nav
# renders this as a chain, because the order is real information, not decoration.
WORKFLOW = [
    ("identify", "Identify", "Can the data answer it?"),
    ("design", "Design", "What should I measure?"),
    ("calibrate", "Calibrate", "Fold the experiment in"),
    ("surface", "Surface", "Map the response"),
    ("meta", "Pool", "All the evidence"),
]
# Two worked case studies now, so they are named rather than both being "Case study":
# HYPER-3 is an analysis story and GEIGER-1911 is a design one, and the nav should say
# which is which rather than making a reader open both to find out.
EXTRA = [
    ("about", "About"),
    ("guides", "Guides"),
    ("tutorial", "Tutorial"),
    ("examples", "Examples"),
    ("benchmarks", "Benchmarks"),
    ("case-study", "HYPER-3"),
    ("rutherford", "GEIGER-1911"),
    ("api", "API"),
]

# The concept shelf. The rest of the site shows axiom doing things; these pages
# explain the ideas the doing rests on, for a reader who has not met them.
#
# Declared once, here, because three things have to agree and drift if they are
# written three times: the index, the reading order, and the previous/next feet
# on the pages themselves. A guide with no row does not appear anywhere, which
# is the failure you want — a broken link is worse than a missing card.
GUIDE_SHELVES: list[tuple[str, str, list[tuple[str, str, str]]]] = [
    (
        "Foundations",
        "What a causal question is, and why the usual tools do not answer one.",
        [
            (
                "guide-causal-question",
                "Prediction is not intervention",
                "A model that forecasts an outcome perfectly can be silent about "
                "every decision you might take. Why, and what the difference is.",
            ),
            (
                "guide-confounding",
                "Confounders, colliders and mediators",
                "The same variable, in three positions. Adjusting for the wrong one "
                "does not fail loudly — it returns a number.",
            ),
            (
                "guide-graphs",
                "Writing your beliefs down",
                "A causal graph is not a diagram of the data. It is a set of claims "
                "about mechanism, and it is testable.",
            ),
            (
                "guide-identification",
                "Identification comes before estimation",
                "Whether an effect is recoverable at all is a question about the "
                "graph, not the sample. It is answerable before you fit anything.",
            ),
        ],
    ),
    (
        "Writing the question down so it travels",
        "The part most workflows skip, and the reason a sound number ends up "
        "answering the wrong question somewhere else.",
        [
            (
                "guide-estimand",
                "The estimand, in eight facets",
                "Two quantities are the same quantity only if all eight match. "
                "Six of the eight can be bridged; two cannot.",
            ),
            (
                "guide-dimensions",
                "Dimensions, units and scope",
                "Three different kinds of mismatch that all look like a unit "
                "problem. Only one of them is.",
            ),
            (
                "guide-transport",
                "Why a model does not travel",
                "A selection diagram, four things it can say, and the formula that "
                "carries an effect from where it was measured to where it is used.",
            ),
            (
                "guide-provenance",
                "Provenance: hashes, ledgers, and the paper trail",
                "Every number carries how it got here — or it is not evidence, it "
                "is a number in a slide.",
            ),
        ],
    ),
    (
        "Doing the work",
        "Where the ideas above turn into decisions about what to measure and what "
        "to believe.",
        [
            (
                "guide-priors",
                "Priors that mean something",
                "A prior is a claim about magnitudes in units you can argue about, "
                "not a regularization knob.",
            ),
            (
                "guide-design",
                "Deciding what to measure",
                "Power is not the question. What the decision turns on is the "
                "question, and it changes what you should go and measure.",
            ),
            (
                "guide-calibration",
                "Folding an experiment in",
                "An experiment measures one thing precisely. Calibration is how "
                "that one thing constrains a model of everything else.",
            ),
            (
                "guide-critique",
                "What would have to be true to overturn this?",
                "The most useful output of an analysis is usually the size of the "
                "confounder that would erase it.",
            ),
        ],
    ),
]

GUIDE_ORDER: list[tuple[str, str, str]] = [
    guide for _, _, guides in GUIDE_SHELVES for guide in guides
]
GUIDE_TITLES: dict[str, str] = {slug: title for slug, title, _ in GUIDE_ORDER}

TOKEN = re.compile(r"\{\{\s*([a-zA-Z0-9_.\[\]-]+?)\s*(?::([^}]+))?\s*\}\}")

# {{@ value lower upper }} -> where the point estimate sits inside its own interval,
# as a percentage. The interval readout uses it to place the marker, so a skewed
# posterior is drawn skewed instead of being quietly centred.
POSITION = re.compile(r"\{\{@\s*(\S+)\s+(\S+)\s+(\S+)\s*\}\}")


def load_data() -> dict[str, Any]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(DATA.glob("*.json"))}


PART = re.compile(r"([^\[\]]*)((?:\[\d+\])*)")


def lookup(data: dict[str, Any], path: str) -> Any:
    """Walk a dotted path, with any number of chained [i] indices per segment."""
    cur: Any = data
    for part in path.split("."):
        m = PART.fullmatch(part)
        if m is None:
            raise KeyError(part)
        name, idx = m.group(1), m.group(2)
        if name:
            cur = cur[int(name)] if isinstance(cur, list) else cur[name]
        for n in re.findall(r"\[(\d+)\]", idx):
            cur = cur[int(n)]
    return cur


def render_token(data: dict[str, Any], path: str, spec: str | None) -> str:
    value = lookup(data, path)
    if spec is None:
        if isinstance(value, float):
            return f"{value:g}"
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        return html.escape(str(value))
    if spec == "raw":
        return str(value)
    if spec == "n":  # thousands separator, no decimals
        return f"{round(float(value)):,}"
    if spec == "pct":
        return f"{float(value) * 100:.0f}%"
    if spec == "pct1":
        return f"{float(value) * 100:.1f}%"
    return format(float(value), spec)


def substitute(text: str, data: dict[str, Any], where: str) -> str:
    def pos(m: re.Match[str]) -> str:
        try:
            v, lo, hi = (float(lookup(data, m.group(i))) for i in (1, 2, 3))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise SystemExit(f"{where}: cannot resolve {m.group(0)} — {exc}") from exc
        if hi == lo:
            return "50%"
        return f"{max(0.0, min(1.0, (v - lo) / (hi - lo))) * 100:.1f}%"

    text = POSITION.sub(pos, text)

    def repl(m: re.Match[str]) -> str:
        path, spec = m.group(1), m.group(2)
        try:
            return render_token(data, path, spec)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise SystemExit(f"{where}: cannot resolve {{{{{path}}}}} — {exc}") from exc

    return TOKEN.sub(repl, text)


def parse_meta(text: str, name: str) -> tuple[dict[str, Any], str]:
    if not text.lstrip().startswith("<!--meta"):
        raise SystemExit(f"{name}: fragment must open with a <!--meta {{...}} --> block")
    start = text.index("<!--meta") + len("<!--meta")
    end = text.index("-->", start)
    meta = json.loads(text[start:end])
    return meta, text[end + 3 :]


PILLAR_PAGE = {
    "identify": "identify.html",
    "design": "design.html",
    "calibrate": "calibrate.html",
    "surface": "surface.html",
    "meta": "meta.html",
}


NBS_GITHUB = "https://github.com/redam94/axiom/blob/main"

KIND_TAG = {
    "spec": "spec",
    "class": "class",
    "function": "fn",
    "callable": "callable",
    "exception": "error",
    "value": "value",
}


def symbol_html(sym: dict[str, Any]) -> str:
    """One public symbol: how to call it, and a line that really calls it.

    The snippet is not written here or anywhere else by hand — ``generate.py``
    lifts it out of the executed notebook that demonstrates the symbol. A symbol
    with no snippet says so plainly; inventing one would defeat the point.
    """
    name = html.escape(sym["name"])
    signature = html.escape(sym["signature"])
    summary = html.escape(sym["summary"])
    tag = KIND_TAG.get(sym["kind"], sym["kind"])
    search = html.escape(f"{sym['name']} {sym['summary']}".lower(), quote=True)

    head = (
        f'<summary><span class="sym-n">{name}</span>' f'<span class="sym-t">{tag}</span></summary>'
    )
    body = [f'<p class="sym-sig">{name}{signature}</p>' if signature else ""]
    if summary:
        body.append(f'<p class="sym-d">{summary}</p>')
    if sym["usage"]:
        nb = html.escape(sym["notebook"])
        body.append(
            f'<div class="code">\n'
            f'  <div class="code-h">'
            f'<span><a href="{NBS_GITHUB}/{nb}">{nb}</a></span>'
            f'<button class="copy" type="button">Copy</button></div>\n'
            f'  <pre>{html.escape(sym["usage"])}</pre>\n'
            f"</div>"
        )
    else:
        body.append(
            '<p class="sym-none">No executed statement uses this symbol — it appears '
            "in the notebooks only inside an <code>import</code>, which is all gate 12 "
            "requires. Worth a worked line.</p>"
        )
    return (
        f'<details class="sym" data-name="{html.escape(sym["name"], quote=True)}" '
        f'data-search="{search}">{head}'
        f'<div class="sym-b">{"".join(x for x in body if x)}</div></details>'
    )


def api_html(data: dict[str, Any]) -> str:
    """The searchable symbol map. Generated, because typing 800+ names is how they rot."""
    api = data["api"]
    blocks = []
    for pkg in api["packages"]:
        name = pkg["name"]
        page = PILLAR_PAGE.get(name)
        walk = f' · <a href="{page}">walkthrough</a>' if page else ""
        nb_dir = f"nbs/{name}"
        notebooks = (
            f' · <a href="{NBS_GITHUB}/{nb_dir}">{pkg["n_notebooks"]} notebooks</a>'
            if pkg["n_notebooks"]
            else ""
        )
        syms = "\n    ".join(symbol_html(s) for s in pkg["symbols"])
        blocks.append(
            f'<div class="api-pkg" id="{name}" data-pkg="{name}">\n'
            f'  <div class="api-h"><h3>axiom.{name}</h3>'
            f'<span class="tag">{pkg["layer"]}</span>'
            f'<span class="api-count">{pkg["n"]} symbols{notebooks}{walk}</span></div>\n'
            f'  <p>{html.escape(pkg["blurb"])}</p>\n'
            f'  <div class="api-syms">\n    {syms}\n  </div>\n'
            f"</div>"
        )
    return "\n".join(blocks)


PILLAR_LABEL = {
    "identify": "Identify",
    "design": "Design",
    "calibrate": "Calibrate",
    "surface": "Surface",
    "meta": "Pool",
    "diagnose": "Diagnose",
    "adapters": "Adapters",
}


# ----------------------------------------------------------------------------------
# the walkthrough pages: one per example, built from the record its own run wrote
# ----------------------------------------------------------------------------------

GITHUB = "https://github.com/redam94/axiom/blob/main/examples"


_INLINE_CODE = re.compile(r"`(.+?)`")
_INLINE_EM = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")


def inline_html(text: str) -> str:
    """Escape, then allow ``\u0060code\u0060`` and ``*emphasis*`` — and nothing else.

    Generated prose is plain text everywhere else on the site, which is right for
    text a run wrote. The tutorial steps are authored, and an author naming
    ``surviving_evidence`` wants it set as code rather than printed with its
    backticks showing.

    Emphasis is applied **outside code spans only**. A code span is literal; the
    same function in ``axiom.report`` had this exact bug, where the italic pass
    ran across the substituted output and ate the asterisks of an equation.
    """
    parts = _INLINE_CODE.split(html.escape(text))
    for i, part in enumerate(parts):
        parts[i] = f"<code>{part}</code>" if i % 2 else _INLINE_EM.sub(r"<em>\1</em>", part)
    return "".join(parts)


def para_html(text: str, cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    return f"<p{attr}>{html.escape(text)}</p>"


def framing_html(docstring: str) -> str:
    """Render an example's module docstring.

    The convention in ``examples/`` is a heading line followed by an indented
    paragraph. Keeping that structure on the page means the framing reads as the
    author wrote it rather than as one undifferentiated wall.
    """
    out = []
    for block in docstring.split("\n\n"):
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        if not lines[0].startswith(" ") and len(lines) > 1 and lines[1].startswith(" "):
            body = " ".join(ln.strip() for ln in lines[1:])
            out.append(
                f'<p class="wt-h">{html.escape(lines[0].strip())}</p>' f"<p>{html.escape(body)}</p>"
            )
        else:
            out.append(f'<p>{html.escape(" ".join(ln.strip() for ln in lines))}</p>')
    return "".join(out)


def legend_html(legend: list[list[str]]) -> str:
    """``[["accent", "posterior mean"], ["ink-2:dash", "the truth"]]`` -> the swatch row."""
    if not legend:
        return ""
    items = []
    for swatch, label in legend:
        token, _, style = swatch.partition(":")
        cls = f' class="{style}"' if style else ""
        items.append(f'<span><i{cls} style="--k:var(--{token})"></i>{html.escape(label)}</span>')
    return f'<div class="legend">{"".join(items)}</div>'


# The chart types site/assets/charts.js knows how to draw. Listed here so a typo
# in an example fails the build rather than rendering "unknown chart type" to a
# reader who has no way to tell it was ever meant to be a picture.
CHART_KINDS = {
    "band",
    "bars",
    "dumbbell",
    "funnel",
    "heatmap",
    "hist",
    "intervals",
    "lines",
    "scatter",
    "sequential",
}


def resolve_paths(opt: Any, block: dict[str, Any], data: dict[str, Any], where: str) -> Any:
    """Turn a figure's ``@`` references into real paths, and check they resolve.

    An example writes ``{"rows": "@rows"}`` because it has no business knowing where
    its data will be stored. Here is where it finds out — and where a reference to
    something the payload does not contain stops the build, rather than failing
    silently in a browser and shipping a page with a hole in it.
    """
    if isinstance(opt, str) and opt.startswith("@"):
        rest = opt[1:]
        path = f"{block['name']}.{rest}" if rest else block["name"]
        try:
            lookup(data, f"{block['file']}.{path}")
        except (KeyError, IndexError, TypeError) as exc:
            raise SystemExit(f"{where}: chart path {opt!r} does not resolve — {exc}") from exc
        return path
    if isinstance(opt, dict):
        return {k: resolve_paths(v, block, data, where) for k, v in opt.items()}
    if isinstance(opt, list):
        return [resolve_paths(v, block, data, where) for v in opt]
    return opt


def figure_html(block: dict[str, Any], stem: str, data: dict[str, Any]) -> str:
    if block["kind"] not in CHART_KINDS:
        raise SystemExit(f"{stem}/{block['name']}: no chart type {block['kind']!r}")
    opt = resolve_paths(dict(block["opt"]), block, data, f"{stem}/{block['name']}")
    if block.get("height") and "height" not in opt:
        opt["height"] = block["height"]
    # single-quoted attribute, so the apostrophe in a label has to be escaped too
    payload = html.escape(json.dumps(opt), quote=True)
    note = f'<p class="fig-s">{html.escape(block["note"])}</p>' if block.get("note") else ""
    return (
        f'<figure class="fig" id="{stem}-{block["name"]}">\n'
        f'  <div class="fig-head"><p class="fig-t">{html.escape(block["title"])}</p>{note}</div>\n'
        f'  {legend_html(block.get("legend") or [])}\n'
        f'  <div class="chart" data-chart="{block["kind"]}" data-file="{block["file"]}"\n'
        f"       data-opt='{payload}'></div>\n"
        f"</figure>"
    )


def table_html(block: dict[str, Any]) -> str:
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in block["columns"])
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
        for row in block["rows"]
    )
    caption = f"<caption>{html.escape(block['caption'])}</caption>" if block.get("caption") else ""
    return (
        f'<div class="tbl-wrap"><table class="t">{caption}'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def blocks_html(blocks: list[dict[str, Any]], stem: str, data: dict[str, Any]) -> str:
    out = []
    for block in blocks:
        kind = block["type"]
        if kind == "say":
            out.append(f'<div class="prose">{para_html(block["text"])}</div>')
        elif kind == "out":
            text = "\n".join(block["lines"]).rstrip()
            out.append(
                '<div class="out"><div class="out-h">what it printed</div>'
                f"<pre>{html.escape(text)}</pre></div>"
            )
        elif kind == "table":
            out.append(table_html(block))
        elif kind == "figure":
            out.append(figure_html(block, stem, data))
        else:  # a new block type must be rendered deliberately, not silently dropped
            raise SystemExit(f"{stem}: no renderer for walkthrough block {kind!r}")
    return "\n".join(out)


def steps_html(entry: dict[str, Any], data: dict[str, Any]) -> str:
    stem = entry["stem"]
    walk = entry["walkthrough"]
    total = len(walk["steps"])
    out = []
    for step in walk["steps"]:
        instead = ""
        if step.get("instead"):
            instead = (
                '<div class="note is-alt"><p class="note-t">The path not taken</p>'
                f'{para_html(step["instead"])}</div>'
            )
        code = (
            f'<div class="code">\n'
            f'  <div class="code-h"><span>examples/{stem}.py</span>'
            f'<button class="copy" type="button">Copy</button></div>\n'
            f'  <pre>{html.escape(step["code"])}</pre>\n'
            f"</div>"
            if step["code"].strip()
            else ""
        )
        out.append(f"""<section class="wt-step" id="step-{step['n']}">
  <div class="wrap">
    <div class="wt-head">
      <p class="wt-n">Step {step['n']} <span>of {total}</span></p>
      <h2>{html.escape(step['title'])}</h2>
      <div class="wt-why prose">{para_html(step['why'])}</div>
      {instead}
    </div>
    <div class="wt-body">
      {code}
      {blocks_html(step['blocks'], stem, data)}
    </div>
  </div>
</section>""")
    return "\n".join(out)


def walkthrough_body(entry: dict[str, Any], prev: Any, nxt: Any, data: dict[str, Any]) -> str:
    stem = entry["stem"]
    walk = entry["walkthrough"]
    pillars = "".join(
        (
            f'<a class="pkg" href="{p}.html">{PILLAR_LABEL.get(p, p)}</a>'
            if p in PILLAR_PAGE
            else f'<span class="pkg">{PILLAR_LABEL.get(p, p)}</span>'
        )
        for p in entry["pillars"]
    )
    contents = "".join(
        f'<a class="wt-toc-i" href="#step-{s["n"]}">'
        f'<span class="wt-toc-n">{s["n"]}</span>{html.escape(s["title"])}</a>'
        for s in walk["steps"]
    )
    findings = "".join(para_html(f) for f in walk["findings"])
    setup = (
        f"""<section>
  <div class="wrap">
    <div class="section-head">
      <p class="eyebrow">Before the first step</p>
      <h2>What this example imports, and why that list is short</h2>
      <p>Everything below runs on the four core dependencies. No sampler, no extras.</p>
    </div>
    <div class="code" style="max-width:var(--measure)">
      <div class="code-h"><span>examples/{stem}.py</span>
        <button class="copy" type="button">Copy</button></div>
      <pre>{html.escape(entry['setup_code'])}</pre>
    </div>
  </div>
</section>"""
        if entry.get("setup_code")
        else ""
    )
    nav = []
    if prev:
        nav.append(
            f'<a class="btn btn-2" href="{prev["slug"]}.html">&larr; '
            f'{prev["number"]} · {html.escape(prev["field"])}</a>'
        )
    nav.append('<a class="btn btn-2" href="examples.html">All twelve</a>')
    if nxt:
        nav.append(
            f'<a class="btn btn-2" href="{nxt["slug"]}.html">'
            f'{nxt["number"]} · {html.escape(nxt["field"])} &rarr;</a>'
        )

    return f"""<section class="hero">
  <div class="wrap">
    <p class="eyebrow">Example {entry['number']} · {html.escape(entry['field'])}</p>
    <h1 style="max-width:18ch">{html.escape(walk['title'])}</h1>
    <p class="lede prose mt">{html.escape(walk['question'])}</p>
    <div class="ex-meta">{pillars}
      <span class="pkg">{len(walk['steps'])} steps</span>
      <span class="pkg">{entry['n_figures']} figures</span>
      <span class="pkg">{entry['lines']} lines</span>
      <span class="pkg">runs in {entry['seconds']}s</span></div>
    <div class="hero-cta">
      <a class="btn btn-1" href="#step-1">Start the walkthrough</a>
      <a class="btn btn-2" href="{GITHUB}/{stem}.py">The whole file on GitHub &rarr;</a>
    </div>
  </div>
</section>

<section>
  <div class="wrap">
    <div class="grid-2">
      <div class="prose wt-framing">{framing_html(entry['docstring'])}</div>
      <div>
        <p class="eyebrow">The route</p>
        <nav class="wt-toc">{contents}</nav>
      </div>
    </div>
  </div>
</section>

{setup}

{steps_html(entry, data)}

<section class="wt-close">
  <div class="wrap">
    <div class="section-head">
      <p class="eyebrow">The close</p>
      <h2>What it actually showed</h2>
      <p>Written after looking at the output, which is why it does not always agree
        with the setup.</p>
    </div>
    <div class="prose wt-findings">{findings}</div>
  </div>
</section>

<section>
  <div class="wrap">
    <details class="ex-more">
      <summary>The whole file, and everything this run printed</summary>
      <div class="ex-body">
        <div class="pair">
          <div class="code">
            <div class="code-h"><span>examples/{stem}.py</span>
              <button class="copy" type="button">Copy</button></div>
            <pre>{html.escape(entry['code'])}</pre>
          </div>
          <div class="out">
            <div class="out-h">stdout, captured on this build</div>
            <pre>{html.escape(entry['output'])}</pre>
          </div>
        </div>
      </div>
    </details>
    <div class="hero-cta" style="margin-top:28px">{''.join(nav)}</div>
  </div>
</section>"""


def apparatus_svg(a: dict[str, Any]) -> str:
    """The experiment, side on: source, collimator, foil, and the counter on its arm.

    Inline SVG rather than a chart, because this is a picture of an apparatus and
    not a picture of data — there is nothing to plot. Every stroke is
    ``currentColor`` so it reads in both themes, which a hardcoded ink would not.
    """
    angles = a["angles"]
    arcs = []
    for i, theta in enumerate((angles[2], angles[5], angles[7])):
        rad = math.radians(theta)
        x, y = 300 + 170 * math.cos(rad), 150 - 170 * math.sin(rad)
        dim = "" if i == 2 else ' opacity="0.35"'
        arcs.append(
            f'<line x1="300" y1="150" x2="{x:.1f}" y2="{y:.1f}" stroke="currentColor" '
            f'stroke-width="1.2" stroke-dasharray="3 3"{dim}/>'
        )
        if i == 2:
            arcs.append(
                f'<rect x="{x - 20:.1f}" y="{y - 11:.1f}" width="40" height="22" rx="3" '
                f'fill="var(--accent-wash)" stroke="var(--accent)" stroke-width="1.5"/>'
                f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" font-size="10" '
                f'fill="var(--accent-deep)">counter</text>'
            )
        arcs.append(
            f'<text x="{300 + 196 * math.cos(rad):.1f}" y="{150 - 196 * math.sin(rad):.1f}" '
            f'text-anchor="middle" font-size="10" fill="currentColor" opacity="0.7">'
            f"{theta:g}\u00b0</text>"
        )
    return f"""<svg viewBox="0 0 620 300" role="img" class="diagram"
     aria-label="An alpha source and collimator on the left, a gold foil at the centre,
     and a counter on a rotatable arm at {angles[7]:g} degrees.">
  <g fill="none" stroke="currentColor" stroke-width="1.6">
    <rect x="18" y="132" width="54" height="36" rx="4"/>
    <rect x="112" y="140" width="26" height="20" rx="2"/>
    <rect x="164" y="140" width="26" height="20" rx="2"/>
    <line x1="72" y1="150" x2="300" y2="150"/>
    <line x1="300" y1="150" x2="470" y2="150" stroke-dasharray="5 4" opacity="0.45"/>
    <line x1="300" y1="96" x2="300" y2="204" stroke="var(--accent)" stroke-width="4"/>
  </g>
  <path d="M 360 150 A 60 60 0 0 0 {300 + 60 * math.cos(math.radians(angles[7])):.1f}
        {150 - 60 * math.sin(math.radians(angles[7])):.1f}"
        fill="none" stroke="currentColor" stroke-width="1.2" opacity="0.6"/>
  <text x="352" y="120" font-size="12" fill="currentColor" opacity="0.8">&#952;</text>
  {''.join(arcs)}
  <g font-size="10.5" fill="currentColor" text-anchor="middle">
    <text x="45" y="188">radium C&#8242;</text>
    <text x="45" y="201" opacity="0.65">{a['energy_mev']:g} MeV &#945;</text>
    <text x="151" y="188">collimator</text>
    <text x="300" y="222" fill="var(--accent-deep)">gold foil</text>
    <text x="300" y="235" opacity="0.65">{a['foil_um']:g} &#181;m</text>
    <text x="440" y="168" opacity="0.65">undeflected beam</text>
    <text x="45" y="214" opacity="0.65">{a['beam_per_s']:,.0f}/s</text>
  </g>
</svg>"""


def slit_svg(g: dict[str, Any]) -> str:
    """The aperture at one station, looking straight down the beam.

    At this angle the scattering arrives on a ring, so the two shapes are drawn
    against it: the annular slot the plan cuts, and the round hole of the same
    *solid angle* it is compared with.

    Radius on the page is the angle from the beam axis, which is **not** an
    area-preserving projection — so the two shapes do not have equal area in the
    picture even though they subtend equal solid angle, and the caption says so
    rather than letting a reader measure the drawing and catch it. The radial
    band is also floored at a legible width; the real one is thinner still.
    """
    cx, cy, ring = 150.0, 150.0, 96.0
    # degrees -> pixels, set so the ring sits at the station's angle
    scale = ring / g["theta"]
    half = max(g["half"] * scale, 1.6)
    hole = max(g["hole_radius"] * scale, 3.0)
    return f"""<svg viewBox="0 0 620 300" role="img" class="diagram"
     aria-label="Looking down the beam: the scattering arrives on a ring at
     {g['theta']:g} degrees. The annular slot is a thin band on that ring; a round
     hole of the same area is a disc {g['hole_radius']:.0f} degrees across.">
  <g transform="translate(0,0)">
    <circle cx="{cx}" cy="{cy}" r="{ring}" fill="none" stroke="currentColor"
            stroke-width="1.2" stroke-dasharray="4 4" opacity="0.5"/>
    <circle cx="{cx}" cy="{cy}" r="{ring}" fill="none" stroke="var(--accent)"
            stroke-width="{2 * half:.2f}" opacity="0.85"/>
    <circle cx="{cx}" cy="{cy}" r="2.5" fill="currentColor"/>
    <line x1="{cx}" y1="{cy}" x2="{cx + ring:.1f}" y2="{cy}" stroke="currentColor"
          stroke-width="1" opacity="0.4"/>
    <text x="{cx + ring / 2:.0f}" y="{cy - 6}" font-size="10" text-anchor="middle"
          fill="currentColor" opacity="0.7">&#952; = {g['theta']:g}&#176;</text>
    <text x="{cx}" y="272" font-size="11" text-anchor="middle" fill="var(--accent-deep)">
      annular slot</text>
    <text x="{cx}" y="286" font-size="10" text-anchor="middle" fill="currentColor"
          opacity="0.7">&#177;{g['half']:.2f}&#176; radial, {g['arc']:g}&#176; of arc</text>
  </g>
  <g transform="translate(320,0)">
    <circle cx="{cx}" cy="{cy}" r="{ring}" fill="none" stroke="currentColor"
            stroke-width="1.2" stroke-dasharray="4 4" opacity="0.5"/>
    <circle cx="{cx + ring:.1f}" cy="{cy}" r="{hole:.1f}" fill="var(--warn-wash, #f7e3e0)"
            stroke="#b5453b" stroke-width="1.6" opacity="0.9"/>
    <circle cx="{cx}" cy="{cy}" r="2.5" fill="currentColor"/>
    <text x="{cx}" y="272" font-size="11" text-anchor="middle" fill="#b5453b">
      round hole, same solid angle</text>
    <text x="{cx}" y="286" font-size="10" text-anchor="middle" fill="currentColor"
          opacity="0.7">radius {g['hole_radius']:.0f}&#176;</text>
  </g>
  <g font-size="10" fill="currentColor" opacity="0.75">
    <text x="150" y="40" text-anchor="middle">reports the angle to
      {g['slot_bias_pct']:.3f}%</text>
    <text x="470" y="40" text-anchor="middle">reports it
      {g['hole_bias_pct']:.1f}% high</text>
  </g>
</svg>"""


DIAGRAMS = {"apparatus": apparatus_svg, "slit": slit_svg}


def diagram_html(step: dict[str, Any], series: dict[str, Any]) -> str:
    """The step's diagram, if it declared one and the run captured its numbers."""
    name = step.get("diagram")
    payload = series.get("captured", {}).get(step.get("diagram_from", name or ""))
    if not name or payload is None:
        return ""
    return f"""<figure class="fig">
  {DIAGRAMS[name](payload)}
  <figcaption>{inline_html(step.get('caption', ''))}</figcaption>
</figure>"""


def math_html(items: list[dict[str, str]]) -> str:
    """Each equation twice: as mathematics, and as the axiom that expresses it.

    The site carries no LaTeX and pulls in no renderer, so the mathematics is
    marked up as HTML — which stays selectable, scales with the reader's type
    size, and costs no request. The pairing is the point: an equation on its own
    is a claim, and the code beside it is how that claim reaches a fit.
    """
    rows = []
    for eq in items:
        rows.append(f"""<div class="eqrow">
  <div class="eqmath">
    <p class="eqlabel">{inline_html(eq['label'])}</p>
    <div class="eq">{eq['math']}</div>
  </div>
  <div class="code eqcode">
    <div class="code-h"><span>in axiom</span></div>
    <pre>{html.escape(eq['code'])}</pre>
  </div>
</div>""")
    return f'<div class="eqs">{"".join(rows)}</div>'


def guides_html() -> str:
    """The guides index: one block per shelf, one card per guide, in reading order."""
    blocks = []
    for shelf, (name, blurb, guides) in enumerate(GUIDE_SHELVES):
        cards = "".join(f"""<a class="card" href="{slug}.html">
  <p class="card-n">{i:02d}</p>
  <h3>{html.escape(title)}</h3>
  <p>{html.escape(blurb_)}</p>
  <div class="card-f">
    <p class="card-go">Read it <span aria-hidden="true">&rarr;</span></p>
  </div>
</a>""" for i, (slug, title, blurb_) in enumerate(guides, start=1 + sum(
            len(g) for _, _, g in GUIDE_SHELVES[:shelf])))
        blocks.append(f"""<div class="section-head" style="margin-top:48px">
  <p class="eyebrow">Shelf {shelf + 1} · {len(guides)} guides</p>
  <h2>{html.escape(name)}</h2>
  <p>{html.escape(blurb)}</p>
</div>
<div class="cards">{cards}</div>""")
    return "".join(blocks)


def guide_nav_html(slug: str) -> str:
    """The foot of one guide: where it sits in the shelf, and what is either side.

    Built from ``GUIDE_ORDER`` rather than written into the fragments, so
    re-ordering the shelf cannot leave a page pointing at the guide that used to
    follow it. A slug with no row is a hard error for the same reason.
    """
    slugs = [s for s, _, _ in GUIDE_ORDER]
    if slug not in slugs:
        raise SystemExit(f"{slug}: no row in build.GUIDE_SHELVES; add one or rename it")
    i = slugs.index(slug)
    links = []
    if i:
        prev = slugs[i - 1]
        links.append(
            f'<a class="btn btn-2" href="{prev}.html">&larr; '
            f"{html.escape(GUIDE_TITLES[prev])}</a>"
        )
    links.append('<a class="btn btn-2" href="guides.html">All the guides</a>')
    if i + 1 < len(slugs):
        nxt = slugs[i + 1]
        links.append(
            f'<a class="btn btn-1" href="{nxt}.html">'
            f'{html.escape(GUIDE_TITLES[nxt])} &rarr;</a>'
        )
    return f"""<section>
  <div class="wrap">
    <p class="eyebrow">Guide {i + 1} of {len(slugs)}</p>
    <div class="hero-cta" style="margin-top:12px">{"".join(links)}</div>
  </div>
</section>"""


def tutorial_html(data: dict[str, Any]) -> str:
    """The tutorial index: one block per series, one card per step.

    Unlike the examples, the steps inside a series are not independent — step 6
    of the depot tutorial recalibrates the fit step 3 produced. The cards are
    numbered rather than tiled so the order reads as the instruction it is.
    """
    blocks = []
    for series in data["tutorial"]["series"]:
        cards = "".join(f"""<a class="card" href="{step['slug']}.html">
  <p class="card-n">STEP {step['n']}</p>
  <h3>{html.escape(step['title'])}</h3>
  <p>{html.escape(step['asks'])}</p>
  <div class="card-f">
    <p class="card-go">Read it <span aria-hidden="true">&rarr;</span></p>
  </div>
</a>""" for step in series["steps"])
        blocks.append(f"""<div class="section-head" style="margin-top:48px">
  <p class="eyebrow">{html.escape(series['kind'])} · {len(series['steps'])} steps</p>
  <h2>{html.escape(series['title'])}</h2>
  <p>{html.escape(series['problem'])}</p>
</div>
<div class="prose" style="margin-bottom:24px"><p>{html.escape(series['lede'])}</p></div>
<div class="cards">{cards}</div>""")
    return "".join(blocks)


def tutorial_body(step: dict[str, Any], series: dict[str, Any], data: dict[str, Any]) -> str:
    """One step: what it is for, the code, and exactly what that code printed.

    The output is captured at build time by ``site/_gen/generate.py``, so a step
    whose numbers have moved shows up as a changed page rather than as prose that
    quietly stopped being true.
    """
    steps = series["steps"]
    i = step["n"] - 1
    prev = steps[i - 1] if i else None
    nxt = steps[i + 1] if i + 1 < len(steps) else None
    contents = "".join(
        f'<a class="wt-toc-i" href="{t["slug"]}.html">'
        f'<span class="wt-toc-n">{t["n"]}</span>{html.escape(t["title"])}</a>'
        for t in steps
    )
    nav = []
    if prev:
        nav.append(
            f'<a class="btn btn-2" href="{prev["slug"]}.html">&larr; '
            f'{prev["n"]} · {html.escape(prev["title"])}</a>'
        )
    nav.append(f'<a class="btn btn-2" href="tutorial.html">All {len(steps)} steps</a>')
    if nxt:
        nav.append(
            f'<a class="btn btn-2" href="{nxt["slug"]}.html">'
            f'{nxt["n"]} · {html.escape(nxt["title"])} &rarr;</a>'
        )
    else:
        nav.append('<a class="btn" href="tutorial.html">The other tutorial &rarr;</a>')

    return f"""<section class="hero">
  <div class="wrap">
    <p class="eyebrow">{html.escape(series['title'])} · step {step['n']} of {len(steps)}</p>
    <h1 style="max-width:18ch">{html.escape(step['title'])}</h1>
    <p class="lede prose mt">{html.escape(step['asks'])}</p>
  </div>
</section>

<section>
  <div class="wrap">
    <div class="prose">
      <p>{inline_html(step['lede'])}</p>
      {"".join(f"<p><strong>{inline_html(part)}</strong></p>"
               for part in step["beat"].split("\n\n"))}
    </div>
  </div>
</section>

{diagram_html(step, series)}

{math_html(step["math"]) if step.get("math") else ""}

<section>
  <div class="wrap">
    <div class="code" style="max-width:var(--measure)">
      <div class="code-h"><span>step {step['n']}</span>
        <button class="copy" type="button">Copy</button></div>
      <pre>{html.escape(step['code'])}</pre>
    </div>
    <div class="code" style="max-width:var(--measure);margin-top:16px">
      <div class="code-h"><span>what it printed</span></div>
      <pre>{html.escape(step['output'])}</pre>
    </div>
    <p class="prose mt"><small>Every line above was captured by running this code
      at build time. The steps share one namespace, so the code on this page is
      the code you would type after the steps before it.</small></p>
  </div>
</section>

<section>
  <div class="wrap">
    <div class="wt-toc">{contents}</div>
    <div class="hero-cta" style="margin-top:28px">{''.join(nav)}</div>
  </div>
</section>"""


def examples_html(data: dict[str, Any]) -> str:
    """The index: one card per example, each linking to its walkthrough."""
    cards = []
    for e in data["examples"]["entries"]:
        walk = e["walkthrough"]
        pillars = " · ".join(PILLAR_LABEL.get(p, p) for p in e["pillars"])
        cards.append(f"""<a class="card" href="{e['slug']}.html">
  <p class="card-n">{e['number']} · {html.escape(e['field']).upper()}</p>
  <h3>{html.escape(walk['title'])}</h3>
  <p>{html.escape(e['question'])}</p>
  <div class="card-f">
    <p class="card-go">{len(walk['steps'])} steps, {e['n_figures']} figures
      <span aria-hidden="true">→</span></p>
    <p class="card-stat"><small>{html.escape(pillars)}</small></p>
  </div>
</a>""")
    return f'<div class="cards">{"".join(cards)}</div>'


def benchmarks_html(data: dict[str, Any]) -> str:
    """One block per dataset: the comparison table, the terms, the full output."""
    blocks = []
    for e in data["benchmarks"]["entries"]:
        rows = []
        for r in e["rows"]:
            fmt = r["fmt"]
            ok = r["rel"] < 1e-2
            rows.append(
                f"<tr><td>{html.escape(r['label'])}</td>"
                f"<td>{format(r['axiom'], fmt)}</td>"
                f"<td>{format(r['published'], fmt)}</td>"
                f"<td>{r['rel']:.1e}</td>"
                f"<td class=\"{'num-good' if ok else 'num-bad'}\">"
                f"{'match' if ok else 'CHECK'}</td></tr>"
            )
        tag = (
            '<span class="tag is-good">committed</span>'
            if e["availability"] == "vendored"
            else '<span class="tag is-warn">fetch to run</span>'
        )
        blocks.append(f"""<article class="ex" id="{e['name']}">
  <div class="ex-head">
    <p class="ex-n">{html.escape(e['field']).upper()} &middot; {html.escape(e['pillar'])}</p>
    <h3>{html.escape(e['title'])}</h3>
    <p class="ex-q">Checked against <strong>{html.escape(e['reference'])}</strong> &mdash;
      {html.escape(e['scale'])}.</p>
    <div class="ex-meta">{tag}
      <span class="pkg">{len(e['rows'])} published values</span>
      <span class="pkg">worst relative error {e['worst_rel']:.1e}</span></div>
  </div>
  <div class="ex-body" style="padding-top:16px">
    <div class="tbl-wrap">
      <table class="t">
        <caption>{html.escape(e['citation'])}</caption>
        <thead><tr><th>quantity</th><th>axiom</th><th>published</th>
          <th>relative error</th><th></th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    <p class="stamp"><b>licence</b> &middot; {html.escape(e['licence'])}</p>
  </div>
  <details class="ex-more">
    <summary>Read the full run</summary>
    <div class="ex-body">
      <div class="out">
        <div class="out-h">benchmarks/case_{e['name'].replace('lalonde_nsw', 'lalonde')}.py</div>
        <pre>{html.escape(e['output'])}</pre>
      </div>
    </div>
  </details>
</article>""")
    return "\n".join(blocks)


def nav_html(active: str) -> str:
    steps = []
    for i, (key, label, hint) in enumerate(WORKFLOW):
        cls = "step" + (" step-on" if key == active else "")
        steps.append(
            f'<a class="{cls}" href="{key}.html" title="{html.escape(hint)}">'
            f'<span class="step-n">{i + 1}</span><span class="step-l">{label}</span></a>'
        )
    chain = '<span class="step-link" aria-hidden="true"></span>'.join(steps)
    extra = "".join(
        f'<a class="navx{" navx-on" if k == active else ""}" href="{k}.html">{lbl}</a>'
        for k, lbl in EXTRA
    )
    return f"""<a class="brand{' brand-on' if active == 'index' else ''}" href="index.html">
      <span class="brand-mark" aria-hidden="true"></span>axiom</a>
    <nav class="chain" aria-label="The five pillars, in working order">{chain}</nav>
    <nav class="navx-wrap" aria-label="More">{extra}</nav>"""


SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta property="og:type" content="website">
<link rel="icon" href="assets/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&\
family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&\
family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/axiom.css">
<script>
// Set the theme before first paint so the page never flashes the wrong one.
(function () {{
  try {{
    var t = localStorage.getItem('axiom-theme');
    if (t === 'light' || t === 'dark') document.documentElement.setAttribute('data-theme', t);
  }} catch (e) {{ /* private mode: fall through to the system setting */ }}
}})();
</script>
</head>
<body data-page="{page}">
<a class="skip" href="#main">Skip to content</a>
<header class="topbar">
  <div class="topbar-in">
    {nav}
    <button class="theme" type="button" aria-label="Switch between light and dark">
      <span class="theme-ico" aria-hidden="true"></span>
    </button>
  </div>
</header>
<main id="main">
{body}
</main>
<footer class="foot">
  <div class="foot-in">
    <p class="foot-b"><strong>axiom</strong> — Bayesian causal decision science.
      Declare what you want to know, find out whether the data can tell you, design the
      experiment that would, fold its answer back in, map the surface it implies, and pool
      everything you have.</p>
    <p class="foot-s">Every figure and every number on this site was produced by running
      axiom. Rebuild them with <code>python site/_gen/generate.py</code>.
      Source: <a href="https://github.com/redam94/axiom">github.com/redam94/axiom</a>.</p>
  </div>
</footer>
<script src="assets/charts.js"></script>
<script src="assets/site.js"></script>
</body>
</html>
"""


def main() -> int:
    data = load_data()
    if not data:
        raise SystemExit("no data — run site/_gen/generate.py first")
    fragments = sorted(SRC.glob("*.html"))
    if not fragments:
        raise SystemExit(f"no fragments in {SRC}")
    for frag in fragments:
        meta, body = parse_meta(frag.read_text(), frag.name)
        body = body.replace("<!--API-->", api_html(data))
        body = body.replace("<!--EXAMPLES-->", examples_html(data))
        body = body.replace("<!--TUTORIAL-->", tutorial_html(data))
        body = body.replace("<!--BENCHMARKS-->", benchmarks_html(data))
        body = body.replace("<!--GUIDES-->", guides_html())
        page = frag.stem
        # Every guide gets the same foot, generated from the shelf. Doing it here
        # rather than in the fragments is what stops a re-ordered shelf leaving a
        # page pointing at whatever used to come next.
        if page.startswith("guide-"):
            body = body.replace("<!--GUIDE-NAV-->", guide_nav_html(page))
        body = substitute(body, data, frag.name)
        out = SHELL.format(
            title=html.escape(meta["title"]),
            description=html.escape(meta["description"]),
            page=page,
            nav=nav_html(meta.get("active", page)),
            body=body,
        )
        (SITE / f"{page}.html").write_text(out)
        print(f"  {page}.html  ({len(out) / 1024:.0f} kB)")

    # One page per tutorial step, in every series. Generated rather than
    # authored for the same reason the example walkthroughs are: the output on
    # the page is what the code printed when the site was built.
    for series in data["tutorial"]["series"]:
        steps = series["steps"]
        for step in steps:
            out = SHELL.format(
                title=html.escape(f"{step['n']}. {step['title']} — axiom tutorial"),
                description=html.escape(f"Step {step['n']} of {len(steps)}: {step['asks']}"),
                page="tutorial-step",
                nav=nav_html("tutorial"),
                body=tutorial_body(step, series, data),
            )
            (SITE / f"{step['slug']}.html").write_text(out)
            print(f"  {step['slug']}.html  ({len(out) / 1024:.0f} kB)")

    # One walkthrough page per example. These are generated rather than authored:
    # their content is the record the example's own run wrote, so a page cannot
    # describe a step the code no longer takes.
    entries = data["examples"]["entries"]
    for i, entry in enumerate(entries):
        prev = entries[i - 1] if i else None
        nxt = entries[i + 1] if i + 1 < len(entries) else None
        walk = entry["walkthrough"]
        title = f"{entry['number']} · {walk['title']} — {entry['field']} · axiom"
        out = SHELL.format(
            title=html.escape(title),
            description=html.escape(
                f"{entry['field']}: {entry['question']} A {len(walk['steps'])}-step "
                "walkthrough with charts, the code for every step, and the reasoning "
                "behind each one."
            ),
            page="example",
            nav=nav_html("examples"),
            body=walkthrough_body(entry, prev, nxt, data),
        )
        (SITE / f"{entry['slug']}.html").write_text(out)
        print(f"  {entry['slug']}.html  ({len(out) / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
