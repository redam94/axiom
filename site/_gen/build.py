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
EXTRA = [
    ("examples", "Examples"),
    ("benchmarks", "Benchmarks"),
    ("case-study", "Case study"),
    ("api", "API"),
]

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


def api_html(data: dict[str, Any]) -> str:
    """The searchable symbol map. Generated, because typing 700+ names is how they rot."""
    blocks = []
    for pkg in data["overview"]["packages"]:
        name = pkg["name"]
        page = PILLAR_PAGE.get(name)
        walk = f' · <a href="{page}">walkthrough</a>' if page else ""
        syms = "".join(f'<span class="sym">{html.escape(s)}</span>' for s in pkg["symbols"])
        blocks.append(
            f'<div class="api-pkg" id="{name}" data-pkg="{name}">\n'
            f'  <div class="api-h"><h3>axiom.{name}</h3>'
            f'<span class="tag">{pkg["layer"]}</span>'
            f'<span class="api-count">{pkg["n"]} symbols{walk}</span></div>\n'
            f'  <p>{html.escape(pkg["blurb"])}</p>\n'
            f'  <div class="api-syms">{syms}</div>\n'
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


def examples_html(data: dict[str, Any]) -> str:
    """One block per example: the framing, the code, and its real captured output."""
    blocks = []
    for e in data["examples"]["entries"]:
        pillars = "".join(
            (
                f'<a class="pkg" href="{p}.html">{PILLAR_LABEL.get(p, p)}</a>'
                if p in PILLAR_PAGE
                else f'<span class="pkg">{PILLAR_LABEL.get(p, p)}</span>'
            )
            for p in e["pillars"]
        )
        # the docstring's first line is the title; the rest is the framing
        lines = e["docstring"].split("\n")
        title = lines[0].strip()
        # the eyebrow already names the field, so drop a redundant "Field — " prefix
        for dash in (" — ", " -- ", " - "):
            head, sep, rest = title.partition(dash)
            if sep and head.strip().lower() == e["field"].strip().lower():
                title = rest.strip()
                break
        title = title[:1].upper() + title[1:]
        framing = "\n".join(lines[1:]).strip()
        framing_html = "".join(
            f"<p>{html.escape(para.strip())}</p>" for para in framing.split("\n\n") if para.strip()
        )
        blocks.append(f"""<article class="ex" id="{e['stem']}">
  <div class="ex-head">
    <p class="ex-n">{e['number']} &middot; {html.escape(e['field']).upper()}</p>
    <h3>{html.escape(title)}</h3>
    <p class="ex-q">{html.escape(e['question'])}</p>
    <div class="ex-meta">{pillars}
      <span class="pkg">{e['lines']} lines</span>
      <span class="pkg">runs in {e['seconds']}s</span></div>
  </div>
  <details class="ex-more">
    <summary>Read the framing, the code, and what it printed</summary>
    <div class="ex-body">
      <div class="ex-framing">{framing_html}</div>
      <div class="pair">
        <div class="code">
          <div class="code-h"><span>examples/{e['stem']}.py</span>
            <button class="copy" type="button">Copy</button></div>
          <pre>{html.escape(e['code'])}</pre>
        </div>
        <div class="out">
          <div class="out-h">what it printed</div>
          <pre>{html.escape(e['output'])}</pre>
        </div>
      </div>
    </div>
  </details>
</article>""")
    return "\n".join(blocks)


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
        body = body.replace("<!--BENCHMARKS-->", benchmarks_html(data))
        body = substitute(body, data, frag.name)
        page = frag.stem
        out = SHELL.format(
            title=html.escape(meta["title"]),
            description=html.escape(meta["description"]),
            page=page,
            nav=nav_html(meta.get("active", page)),
            body=body,
        )
        (SITE / f"{page}.html").write_text(out)
        print(f"  {page}.html  ({len(out) / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
