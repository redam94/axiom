/* Render every chart on every built page, headlessly, and fail on the things a
 * browser would show but nothing else catches.
 *
 *     node site/_gen/check.js
 *
 * The charts are drawn by `assets/charts.js` in the reader's browser, so nothing
 * in the Python build can tell whether they come out right — `build.py` checks
 * that a data path resolves, not that the picture is sane. This script supplies
 * the missing half: a DOM stub thin enough to run charts.js unmodified, and three
 * assertions over the SVG it produces.
 *
 *   - a mark painted outside the plot rectangle and not inside the clip. An SVG
 *     group paints wherever its coordinates say and `.chart svg` is
 *     `overflow: visible`, so a chart whose geometry is not derived from its own
 *     domain paints across the page. A funnel's contours are computed from the
 *     pooled estimate and the largest standard error, not from the studies, and
 *     ran 306px past the left edge of a 630px plot and over the prose beside it.
 *   - an axis with a single tick, which is what a descending domain used to
 *     produce: `ticks()` returned `[lo]` for `hi < lo`, so a funnel's standard
 *     error axis drew one meaningless label and no gridlines.
 *   - anything that throws.
 *
 * Exit status is 1 if any chart has a problem, so it can gate a deploy.
 */
"use strict";

const fs = require("fs");
const path = require("path");

const SITE = path.resolve(__dirname, "..");
const DATA = path.join(SITE, "assets", "data");

/* -- a DOM thin enough to run charts.js, and no thinner --------------------- */

class Node {
  constructor(tag) {
    this.tagName = tag;
    this.attrs = {};
    this.children = [];
    this.style = {};
    this.dataset = {};
    this.parentNode = null;
    this.textContent = "";
    this.classList = { add() {}, remove() {}, contains: () => false };
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k]; }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  addEventListener() {}
  querySelector() { return null; }
  querySelectorAll() { return []; }
  closest() { return null; }
  getTotalLength() { return 100; }
  get clientWidth() { return this._w || 700; }
  get offsetWidth() { return 100; }
  get offsetHeight() { return 40; }
  set innerHTML(_v) {}
  get innerHTML() { return ""; }
  walk(fn) { fn(this); this.children.forEach((c) => c.walk(fn)); }
}

/* Any colour will do — this checks geometry, not paint. */
const INK = "#111";
const palette = { series: [] };
for (const k of ["ink", "ink2", "ink3", "grid", "rule", "accent", "accentdeep", "boundary",
  "boundarywash", "panel", "panel2", "paper", "good", "accentwash"]) palette[k] = INK;
for (const k of ["ink-2", "ink-3", "accent-deep", "boundary-wash", "panel-2", "accent-wash"]) {
  palette[k] = INK;
}
for (let i = 1; i <= 8; i++) { palette.series.push(INK); palette["s" + i] = INK; }

global.document = {
  createElementNS: (_ns, tag) => new Node(tag),
  createElement: (tag) => new Node(tag),
  addEventListener() {},
  querySelectorAll: () => [],
  documentElement: new Node("html"),
};
global.window = {
  getComputedStyle: () => ({ getPropertyValue: () => INK }),
  matchMedia: () => ({ matches: false, addEventListener() {} }),
  requestAnimationFrame: (f) => f(),
  addEventListener() {},
  removeEventListener() {},
  console,
};
global.getComputedStyle = global.window.getComputedStyle;
global.requestAnimationFrame = global.window.requestAnimationFrame;

let registry;
{
  const src = fs.readFileSync(path.join(SITE, "assets", "charts.js"), "utf8");
  // charts.js keeps its registry private; hand it out without editing the file.
  const shim = src.replace(/var registry = \{\};/, "var registry = {}; global.__r = registry;");
  if (shim === src) {
    console.error("check.js: could not reach charts.js's registry — has it been renamed?");
    process.exit(2);
  }
  eval(shim); // eslint-disable-line no-eval
  registry = global.__r;
}

/* -- the chart embeds, read off the built pages ----------------------------- */

const unescape = (s) =>
  s.replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">").replace(/&amp;/g, "&");

function embeds() {
  const out = [];
  const tag = /<div class="chart"([\s\S]*?)><\/div>/g;
  for (const file of fs.readdirSync(SITE).filter((f) => f.endsWith(".html")).sort()) {
    const text = fs.readFileSync(path.join(SITE, file), "utf8");
    let m;
    while ((m = tag.exec(text))) {
      const blob = m[1];
      const kind = /data-chart="([^"]+)"/.exec(blob);
      const data = /data-file="([^"]+)"/.exec(blob);
      const opt = /data-opt='([\s\S]*?)'/.exec(blob);
      if (!kind || !data) continue;
      let parsed = {};
      if (opt) {
        try {
          parsed = JSON.parse(unescape(opt[1]));
        } catch (e) {
          out.push({ page: file, kind: kind[1], bad: "data-opt is not JSON: " + e.message });
          continue;
        }
      }
      out.push({ page: file, kind: kind[1], file: data[1], opt: parsed });
    }
  }
  return out;
}

const cache = {};
const load = (f) => (cache[f] ||= JSON.parse(fs.readFileSync(path.join(DATA, f + ".json"))));

/* -- the three assertions --------------------------------------------------- */

function inspect(spec) {
  if (spec.bad) return [spec.bad];
  if (!registry[spec.kind]) return ["no chart type " + spec.kind];

  const fig = new Node("div"); // attachTable walks up for a container; give it one
  const node = fig.appendChild(new Node("div"));
  node._w = 700;
  try {
    registry[spec.kind](node, load(spec.file), spec.opt, palette);
  } catch (e) {
    return ["threw: " + e.message];
  }

  let svg = null;
  node.walk((n) => { if (n.tagName === "svg" && !svg) svg = n; });
  if (!svg) return ["drew no svg"];

  // The plot rectangle: the clip when there is one, the axis baseline otherwise.
  let iw = null;
  let ih = null;
  svg.walk((n) => {
    if (n.tagName === "clipPath" && n.children.length && iw === null) {
      iw = parseFloat(n.children[0].attrs.width);
      ih = parseFloat(n.children[0].attrs.height);
    }
  });
  if (iw === null) {
    svg.walk((n) => {
      const a = n.attrs;
      if (n.tagName === "line" && a.x1 === "0" && a.y1 === a.y2 && parseFloat(a.x2) > 100) {
        if (iw === null || parseFloat(a.y1) > ih) { iw = parseFloat(a.x2); ih = parseFloat(a.y1); }
      }
    });
  }
  if (iw === null) return []; // no axes to speak of — a bare sparkline

  const clipped = (n) => {
    for (let p = n; p; p = p.parentNode) if (p.attrs && p.attrs["clip-path"]) return true;
    return false;
  };
  let bleed = 0;
  svg.walk((n) => {
    if (n.tagName !== "path" || clipped(n)) return;
    const nums = (n.attrs.d || "").match(/-?\d+(\.\d+)?/g);
    if (!nums) return;
    for (let i = 0; i + 1 < nums.length; i += 2) {
      const x = parseFloat(nums[i]);
      const y = parseFloat(nums[i + 1]);
      bleed = Math.max(bleed, -x, x - iw, -y, y - ih);
    }
  });

  let ticks = 0;
  svg.walk((n) => {
    if (n.tagName === "text" && n.attrs["text-anchor"] === "end" && n.textContent) ticks++;
  });

  const problems = [];
  if (bleed > 8) problems.push(`paints ${bleed.toFixed(0)}px outside a ${iw.toFixed(0)}px plot`);
  const suppressed = Array.isArray(spec.opt.yTicks) && spec.opt.yTicks.length === 0;
  if (ticks === 1 && !suppressed) problems.push("y axis has one tick");
  return problems;
}

/* -- run -------------------------------------------------------------------- */

const specs = embeds();
if (!specs.length) {
  console.error("check.js: no charts found — run site/_gen/build.py first");
  process.exit(2);
}
let failed = 0;
for (const spec of specs) {
  const problems = inspect(spec);
  if (!problems.length) continue;
  failed++;
  for (const p of problems) console.log(`  ${spec.kind.padEnd(11)} ${spec.page}  — ${p}`);
}
const pages = new Set(specs.map((s) => s.page)).size;
console.log(`${specs.length} charts on ${pages} pages · ${failed ? failed + " with problems" : "all clean"}`);
process.exit(failed ? 1 : 0);
