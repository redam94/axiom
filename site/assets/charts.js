/* ==========================================================================
   axiom — charts
   --------------------------------------------------------------------------
   Small hand-rolled SVG charting, so every figure is theme-aware, hoverable,
   and backed by a table view. Colours are read from CSS custom properties at
   render time, which is what makes the dark mode a real palette rather than an
   inverted one. Red (--boundary) is reserved for a crossed threshold.

   Usage from markup:
     <div class="chart" data-chart="lines" data-file="design" data-opt='{...}'></div>
   ========================================================================== */

(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var cache = {};
  var registry = {};
  var mounted = [];
  var uid = 0;

  /* -- tiny helpers -------------------------------------------------------- */

  function el(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    if (attrs) for (var k in attrs) if (attrs[k] !== null && attrs[k] !== undefined) {
      n.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(n);
    return n;
  }

  function css(node, name) {
    return getComputedStyle(node).getPropertyValue(name).trim();
  }

  function palette(node) {
    var p = { series: [] };
    ["--ink", "--ink-2", "--ink-3", "--grid", "--rule", "--accent", "--accent-deep",
     "--boundary", "--boundary-wash", "--panel", "--panel-2", "--paper", "--good",
     "--accent-wash"].forEach(function (k) {
      var hex = css(node, k);
      var bare = k.replace(/^--/, "");
      p[bare.replace(/-/g, "")] = hex;   /* "ink2"  */
      p[bare] = hex;                      /* "ink-2" — both spellings resolve */
    });
    for (var i = 1; i <= 8; i++) {
      var hex = css(node, "--s" + i);
      p.series.push(hex);
      p["s" + i] = hex;   /* so a series can name its slot explicitly */
    }
    return p;
  }

  /* Resolve a colour name from the palette. An unknown name falls back to a real
     colour rather than being passed through as an invalid CSS value, which would
     paint the mark black and look deliberate. */
  function colourOf(p, name, fallback) {
    if (!name) return fallback;
    if (name.charAt(0) === "#") return name;
    return p[name] || fallback;
  }

  function extent(arrs) {
    var lo = Infinity, hi = -Infinity;
    arrs.forEach(function (a) {
      for (var i = 0; i < a.length; i++) {
        var v = a[i];
        if (v === null || v === undefined || !isFinite(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    });
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    if (lo === hi) { lo -= 0.5; hi += 0.5; }
    return [lo, hi];
  }

  function ticks(lo, hi, n) {
    /* A descending domain is a real axis, not an empty one: a funnel plot's
       standard error grows downward, so its yDomain arrives as [max, 0]. Sort
       the ends and let the scale place the values — returning [lo] here is why
       an inverted axis used to draw one meaningless label and no gridlines. */
    if (hi < lo) { var swap = lo; lo = hi; hi = swap; }
    var span = hi - lo;
    if (span <= 0) return [lo];
    var step = Math.pow(10, Math.floor(Math.log10(span / n)));
    var err = (span / n) / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5;
    else if (err >= 1.5) step *= 2;
    var out = [], t = Math.ceil(lo / step) * step;
    for (; t <= hi + step * 1e-9; t += step) out.push(Math.abs(t) < step * 1e-9 ? 0 : t);
    return out;
  }

  function fmt(v, d) {
    if (v === null || v === undefined || !isFinite(v)) return "—";
    if (d !== undefined) return v.toFixed(d);
    var a = Math.abs(v);
    if (a >= 10000) return Math.round(v).toLocaleString();
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    if (a === 0) return "0";
    return v.toFixed(3);
  }

  function scale(domain, range) {
    var d0 = domain[0], d1 = domain[1], r0 = range[0], r1 = range[1];
    var m = (d1 - d0) === 0 ? 0 : (r1 - r0) / (d1 - d0);
    var f = function (v) { return r0 + (v - d0) * m; };
    f.invert = function (p) { return m === 0 ? d0 : d0 + (p - r0) / m; };
    f.domain = domain;
    f.range = range;
    return f;
  }

  function path(pts) {
    var d = "", started = false;
    for (var i = 0; i < pts.length; i++) {
      var p = pts[i];
      if (p === null) { started = false; continue; }
      d += (started ? "L" : "M") + p[0].toFixed(2) + " " + p[1].toFixed(2);
      started = true;
    }
    return d;
  }

  /* -- frame: margins, grid, axes ------------------------------------------ */

  function frame(root, o) {
    var w = o.width, h = o.height;
    var m = o.margin;
    var svg = el("svg", {
      viewBox: "0 0 " + w + " " + h, width: w, height: h,
      role: "img", "aria-label": o.label || ""
    }, root);
    var p = o.p;
    var iw = w - m.l - m.r, ih = h - m.t - m.b;
    var g = el("g", { transform: "translate(" + m.l + "," + m.t + ")" }, svg);

    /* Marks that can run past their own domain go in `marks`, which is clipped to
       the plot rectangle. Annotations stay in `g`, because several of them sit
       outside it on purpose — a direct series label to the right of its last
       point, an hline caption, the axis titles.

       Without this an SVG group paints wherever its coordinates say, and a chart
       whose geometry is not derived from its domain paints across the page: a
       funnel's contours are computed from the pooled estimate and the largest
       standard error, not from the studies, so they ran 340px past the left edge
       of a 700px plot and over the prose beside it. */
    var cid = "axclip" + ++uid;
    el("rect", { x: 0, y: 0, width: iw, height: ih }, el("clipPath", { id: cid }, svg));
    var marks = el("g", { "clip-path": "url(#" + cid + ")" }, g);

    var x = scale(o.xDomain, [0, iw]);
    var y = scale(o.yDomain, [ih, 0]);

    var yt = o.yTicks || ticks(o.yDomain[0], o.yDomain[1], o.yTickCount || 5);
    yt.forEach(function (t) {
      var yy = y(t);
      if (yy < -1 || yy > ih + 1) return;
      el("line", {
        x1: 0, x2: iw, y1: yy, y2: yy, stroke: p.grid, "stroke-width": 1,
        "shape-rendering": "crispEdges"
      }, g);
      el("text", {
        x: -9, y: yy + 4, "text-anchor": "end", fill: p.ink3,
        "font-family": "var(--f-mono)", "font-size": 10.5
      }, g).textContent = (o.yFmt || fmt)(t);
    });

    var xt = o.xTicks || ticks(o.xDomain[0], o.xDomain[1], o.xTickCount || 6);
    xt.forEach(function (t) {
      var xx = x(t);
      if (xx < -1 || xx > iw + 1) return;
      el("text", {
        x: xx, y: ih + 18, "text-anchor": "middle", fill: p.ink3,
        "font-family": "var(--f-mono)", "font-size": 10.5
      }, g).textContent = (o.xFmt || fmt)(t);
    });

    el("line", {
      x1: 0, x2: iw, y1: ih, y2: ih, stroke: p.rule, "stroke-width": 1,
      "shape-rendering": "crispEdges"
    }, g);

    if (o.xLabel) {
      el("text", {
        x: iw / 2, y: ih + 38, "text-anchor": "middle", fill: p.ink3,
        "font-family": "var(--f-ui)", "font-size": 11
      }, g).textContent = o.xLabel;
    }
    if (o.yLabel) {
      /* Sit the rotated title outside the widest tick label, not on top of it —
         otherwise it overprints the minus signs on a negative axis. */
      var widest = 0;
      yt.forEach(function (t) {
        widest = Math.max(widest, String((o.yFmt || fmt)(t)).length);
      });
      var titleX = -Math.min(m.l - 4, 15 + widest * 6.3);
      el("text", {
        transform: "translate(" + titleX + "," + ih / 2 + ") rotate(-90)",
        "text-anchor": "middle", fill: p.ink3,
        "font-family": "var(--f-ui)", "font-size": 11
      }, g).textContent = o.yLabel;
    }
    return { svg: svg, g: g, marks: marks, x: x, y: y, iw: iw, ih: ih, m: m };
  }

  /* -- tooltip ------------------------------------------------------------- */

  function tipFor(root) {
    var t = root.querySelector(".tip");
    if (!t) { t = document.createElement("div"); t.className = "tip"; root.appendChild(t); }
    return t;
  }

  function showTip(root, tip, px, py, html) {
    tip.innerHTML = html;
    tip.classList.add("on");
    var rw = root.clientWidth;
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    var left = px + 14;
    if (left + tw > rw) left = px - tw - 14;
    if (left < 0) left = 0;
    var top = py - th - 12;
    if (top < 0) top = py + 16;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }

  function hideTip(tip) { tip.classList.remove("on"); }

  /* -- table view ---------------------------------------------------------- */

  function attachTable(root, columns, rows) {
    if (!rows || !rows.length) return;
    var fig = root.closest(".fig") || root.parentNode;
    if (fig.querySelector(".tbl-toggle")) return;
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "tbl-toggle";
    btn.textContent = "Show data table";
    var box = document.createElement("div");
    box.className = "chart-tbl";
    var html = "<table class='t'><thead><tr>";
    columns.forEach(function (c) { html += "<th>" + c + "</th>"; });
    html += "</tr></thead><tbody>";
    rows.forEach(function (r) {
      html += "<tr>";
      r.forEach(function (c) { html += "<td>" + c + "</td>"; });
      html += "</tr>";
    });
    box.innerHTML = html + "</tbody></table>";
    btn.addEventListener("click", function () {
      var on = box.classList.toggle("on");
      btn.textContent = on ? "Hide data table" : "Show data table";
    });
    var stamp = fig.querySelector(".stamp");
    if (stamp) { fig.insertBefore(btn, stamp); fig.insertBefore(box, stamp); }
    else { fig.appendChild(btn); fig.appendChild(box); }
  }

  /* ========================================================================
     chart types
     ======================================================================== */

  /* Sequential monitoring: a harm boundary and the statistic walking into it. */
  registry.sequential = function (root, d, o, p) {
    var panels = o.panels.map(function (i) { return d.panels[i]; });
    var w = root.clientWidth;
    /* Two panels side by side need room for their own axes; below that, stack. */
    var cols = (panels.length > 1 && w >= 420) ? 2 : 1;
    var pw = (w - (cols - 1) * 18) / cols;
    var ph = o.panelHeight || 190;
    var wrap = document.createElement("div");
    wrap.style.display = "grid";
    /* minmax(0,…) so a track can shrink below the SVG's intrinsic width */
    wrap.style.gridTemplateColumns = "repeat(" + cols + ", minmax(0, 1fr))";
    wrap.style.gap = "18px";
    root.appendChild(wrap);
    var yDom = o.yDomain || [-5, 4];

    panels.forEach(function (pan) {
      var cell = document.createElement("div");
      cell.style.position = "relative";
      wrap.appendChild(cell);
      /* One tooltip per panel, anchored to the panel. It used to be a single tip on
         the chart root while the coordinates handed to showTip were relative to the
         cell, so in a grid of panels every tooltip but the top-left one appeared
         offset by that cell's position in the grid. */
      var tip = tipFor(cell);

      var ttl = document.createElement("div");
      ttl.style.cssText = "font-family:var(--f-ui);font-size:.82rem;font-weight:600;" +
        "color:var(--ink);margin-bottom:6px";
      ttl.textContent = pan.label;
      cell.appendChild(ttl);

      var span = pan.information[pan.information.length - 1] - pan.information[0];
      var xpad = span * 0.09;  /* keep a first- or last-look crossing off the edge */
      var f = frame(cell, {
        width: pw, height: ph, margin: { l: 34, r: 12, t: 8, b: 34 }, p: p,
        xDomain: [pan.information[0] - xpad,
                  pan.information[pan.information.length - 1] + xpad],
        yDomain: yDom, xTickCount: 4, yTickCount: 4,
        xLabel: o.xLabel, xFmt: function (v) { return v.toFixed(2); },
        yFmt: function (v) { return v.toFixed(0); },
        label: "Monitoring path for " + pan.label
      });

      /* the harm region: everything below the boundary */
      var poly = [];
      pan.information.forEach(function (t, i) { poly.push([f.x(t), f.y(pan.boundary[i])]); });
      for (var i = pan.information.length - 1; i >= 0; i--) {
        poly.push([f.x(pan.information[i]), f.ih]);
      }
      el("path", {
        d: path(poly) + "Z", fill: p.boundarywash, stroke: "none", opacity: 0.85
      }, f.g);

      el("path", {
        d: path(pan.information.map(function (t, i) { return [f.x(t), f.y(pan.boundary[i])]; })),
        fill: "none", stroke: p.boundary, "stroke-width": 2, "stroke-dasharray": "5 4"
      }, f.g);

      el("line", {
        x1: 0, x2: f.iw, y1: f.y(0), y2: f.y(0), stroke: p.ink3, "stroke-width": 1,
        "stroke-dasharray": "2 3", opacity: .6
      }, f.g);

      var taken = pan.looks_taken;
      var seen = [];
      for (var j = 0; j < taken; j++) seen.push([f.x(pan.information[j]), f.y(pan.z[j])]);
      var colour = o.colour === "stratum" && pan.stratum !== "all"
        ? p.series[["age_25_35", "age_36_50", "age_51_plus"].indexOf(pan.stratum) + 1] || p.accent
        : p.ink2;

      /* Reviews the trial never got to, because it stopped. Drawn faintly and
         labelled, so the reader can see the path was not a fluke of one look —
         but never confused with what was actually observed. */
      if (o.showContinuation && taken < pan.information.length) {
        var rest = [];
        for (var m = taken - 1; m < pan.information.length; m++) {
          rest.push([f.x(pan.information[m]), f.y(pan.z[m])]);
        }
        el("path", {
          d: path(rest), fill: "none", stroke: colour, "stroke-width": 1.6,
          "stroke-dasharray": "2 4", opacity: .45
        }, f.marks);
        for (var q = taken; q < pan.information.length; q++) {
          el("circle", {
            cx: f.x(pan.information[q]), cy: f.y(pan.z[q]), r: 2.6, fill: colour,
            opacity: .45
          }, f.g);
        }
      }

      var line = el("path", {
        d: path(seen), fill: "none", stroke: colour, "stroke-width": 2.4,
        "stroke-linejoin": "round", "stroke-linecap": "round"
      }, f.marks);

      /* A path with a single point has no length and would draw nothing — which is
         exactly the case when a rule fires at the very first look. */
      if (seen.length === 1) {
        el("circle", {
          cx: seen[0][0], cy: seen[0][1], r: 7, fill: "none", stroke: colour,
          "stroke-width": 2.4, opacity: .5
        }, f.g);
      }

      if (o.animate && seen.length > 1 &&
          !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        var len = line.getTotalLength();
        line.style.strokeDasharray = len;
        line.style.strokeDashoffset = len;
        line.style.transition = "stroke-dashoffset 1.1s cubic-bezier(.4,0,.2,1) .15s";
        requestAnimationFrame(function () { line.style.strokeDashoffset = "0"; });
      }

      seen.forEach(function (pt, k) {
        var c = el("circle", {
          cx: pt[0], cy: pt[1], r: 4.2, fill: p.panel, stroke: colour, "stroke-width": 2
        }, f.g);
        var hit = el("circle", {
          cx: pt[0], cy: pt[1], r: 14, fill: "transparent", style: "cursor:pointer"
        }, f.g);
        hit.addEventListener("mouseenter", function () {
          showTip(cell, tip, pt[0] + 34, pt[1] + 26,
            "<b>week " + pan.weeks[k] + "</b><br>Z " + pan.z[k].toFixed(2) +
            "<br>effect " + (pan.effects[k] >= 0 ? "+" : "") + pan.effects[k].toFixed(2) +
            " mmHg<br><em>boundary " + pan.boundary[k].toFixed(2) + "</em>");
        });
        hit.addEventListener("mouseleave", function () { hideTip(tip); });
        c.setAttribute("pointer-events", "none");
      });

      if (pan.stopped) {
        var sx = f.x(pan.stop_information), sy = f.y(pan.stop_z);
        var mark = el("g", { opacity: o.animate ? 0 : 1 }, f.g);
        el("circle", {
          cx: sx, cy: sy, r: 9, fill: "none", stroke: p.boundary, "stroke-width": 2
        }, mark);
        el("path", {
          d: "M" + (sx - 4.2) + " " + (sy - 4.2) + "L" + (sx + 4.2) + " " + (sy + 4.2) +
             "M" + (sx + 4.2) + " " + (sy - 4.2) + "L" + (sx - 4.2) + " " + (sy + 4.2),
          stroke: p.boundary, "stroke-width": 2.2, "stroke-linecap": "round"
        }, mark);
        /* Below the marker, not above: above is where the boundary line runs, and
           the space under a crossing is by definition empty. Nudge inward when the
           crossing sits against an edge. */
        var lx = Math.max(28, Math.min(f.iw - 28, sx));
        var ly = Math.min(f.ih - 5, sy + 21);
        el("text", {
          x: lx, y: ly, fill: p.boundary, "text-anchor": "middle",
          "font-family": "var(--f-mono)", "font-size": 10.5, "font-weight": 600
        }, mark).textContent = "stopped";
        if (o.animate && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
          mark.style.transition = "opacity .4s ease 1.15s";
          requestAnimationFrame(function () { mark.style.opacity = 1; });
        }
      }
    });

    var trows = [];
    panels.forEach(function (pan) {
      pan.information.forEach(function (t, i) {
        if (i >= pan.looks_taken) return;
        trows.push([pan.label, pan.weeks[i], t.toFixed(3), pan.z[i].toFixed(2),
                    pan.boundary[i].toFixed(2), pan.effects[i].toFixed(2)]);
      });
    });
    attachTable(root, ["panel", "week", "information", "Z", "boundary", "effect (mmHg)"], trows);
  };

  /* A response curve with its posterior band, optionally against the truth. */
  registry.band = function (root, d, o, p) {
    /* a dotted path, so a page can keep many bands in one data file */
    var b = lookup(d, o.key);
    var w = root.clientWidth, h = o.height || 300;
    var tip = tipFor(root);
    var ys = [b.lower, b.upper];
    if (o.truth !== false && b.truth) ys.push(b.truth);
    var yd = extent(ys);
    var pad = (yd[1] - yd[0]) * 0.08;
    var f = frame(root, {
      width: w, height: h, margin: { l: 52, r: 14, t: 10, b: 46 }, p: p,
      xDomain: [b.doses[0], b.doses[b.doses.length - 1]],
      yDomain: [yd[0] - pad, yd[1] + pad],
      xLabel: o.xLabel || ("dose of " + b.treatment), yLabel: o.yLabel || "expected outcome",
      label: o.label || "Response curve with posterior band"
    });

    var up = b.doses.map(function (x, i) { return [f.x(x), f.y(b.upper[i])]; });
    var dn = [];
    for (var i = b.doses.length - 1; i >= 0; i--) dn.push([f.x(b.doses[i]), f.y(b.lower[i])]);
    el("path", {
      d: path(up.concat(dn)) + "Z", fill: p.accent, opacity: .16, stroke: "none"
    }, f.g);

    if (o.truth !== false && b.truth) {
      el("path", {
        d: path(b.doses.map(function (x, i) { return [f.x(x), f.y(b.truth[i])]; })),
        fill: "none", stroke: p.ink2, "stroke-width": 2, "stroke-dasharray": "5 4"
      }, f.g);
    }
    el("path", {
      d: path(b.doses.map(function (x, i) { return [f.x(x), f.y(b.mean[i])]; })),
      fill: "none", stroke: p.accent, "stroke-width": 2.4, "stroke-linecap": "round"
    }, f.g);

    var cross = el("line", {
      y1: 0, y2: f.ih, stroke: p.ink3, "stroke-width": 1, "stroke-dasharray": "2 3",
      opacity: 0
    }, f.g);
    var dot = el("circle", { r: 4.5, fill: p.panel, stroke: p.accent, "stroke-width": 2,
      opacity: 0 }, f.g);

    el("rect", {
      x: 0, y: 0, width: f.iw, height: f.ih, fill: "transparent", style: "cursor:crosshair"
    }, f.g).addEventListener("mousemove", function (ev) {
      var box = f.svg.getBoundingClientRect();
      var mx = ev.clientX - box.left - f.m.l;
      var dose = f.x.invert(mx);
      var k = 0, best = Infinity;
      b.doses.forEach(function (v, j) {
        var dd = Math.abs(v - dose);
        if (dd < best) { best = dd; k = j; }
      });
      cross.setAttribute("x1", f.x(b.doses[k]));
      cross.setAttribute("x2", f.x(b.doses[k]));
      cross.setAttribute("opacity", .7);
      dot.setAttribute("cx", f.x(b.doses[k]));
      dot.setAttribute("cy", f.y(b.mean[k]));
      dot.setAttribute("opacity", 1);
      var html = "<b>dose " + fmt(b.doses[k]) + "</b><br>mean " + fmt(b.mean[k]) +
        "<br><em>" + Math.round(b.mass * 100) + "% ETI [" + fmt(b.lower[k]) + ", " +
        fmt(b.upper[k]) + "]</em>";
      if (b.truth) html += "<br>truth " + fmt(b.truth[k]);
      showTip(root, tip, f.x(b.doses[k]) + f.m.l, f.y(b.mean[k]) + 10, html);
    });
    root.addEventListener("mouseleave", function () {
      hideTip(tip);
      cross.setAttribute("opacity", 0);
      dot.setAttribute("opacity", 0);
    });

    attachTable(root, ["dose", "posterior mean", "lower", "upper", "truth"],
      b.doses.map(function (x, i) {
        return [fmt(x), fmt(b.mean[i]), fmt(b.lower[i]), fmt(b.upper[i]),
                b.truth ? fmt(b.truth[i]) : "—"];
      }));
  };

  /* Any number of x/y series on one axis. */
  registry.lines = function (root, d, o, p) {
    var series = o.series.map(function (s) {
      return {
        label: s.label,
        x: typeof s.x === "string" ? lookup(d, s.x) : s.x,
        y: typeof s.y === "string" ? lookup(d, s.y) : s.y,
        colour: s.colour
      };
    });
    var w = root.clientWidth, h = o.height || 300;
    var tip = tipFor(root);
    var xd = o.xDomain || extent(series.map(function (s) { return s.x; }));
    var yd = o.yDomain || extent(series.map(function (s) { return s.y; }));
    var f = frame(root, {
      width: w, height: h,
      margin: { l: o.yLabel ? 68 : 50, r: o.rightPad || 14, t: 10, b: 46 }, p: p,
      xDomain: xd, yDomain: yd, xLabel: o.xLabel, yLabel: o.yLabel,
      yFmt: o.yPct ? function (v) { return Math.round(v * 100) + "%"; } : fmt,
      label: o.label || ""
    });

    if (o.hline !== undefined) {
      el("line", {
        x1: 0, x2: f.iw, y1: f.y(o.hline), y2: f.y(o.hline), stroke: p.ink3,
        "stroke-width": 1.5, "stroke-dasharray": "4 4", opacity: .8
      }, f.g);
      if (o.hlineLabel) {
        el("text", {
          x: f.iw - 4, y: f.y(o.hline) - 6, "text-anchor": "end", fill: p.ink3,
          "font-family": "var(--f-mono)", "font-size": 10
        }, f.g).textContent = o.hlineLabel;
      }
    }

    series.forEach(function (s, i) {
      var colour = colourOf(p, s.colour, p.series[i % 8]);
      s._c = colour;
      el("path", {
        d: path(s.x.map(function (x, j) { return [f.x(x), f.y(s.y[j])]; })),
        fill: "none", stroke: colour, "stroke-width": 2.2, "stroke-linecap": "round",
        "stroke-linejoin": "round"
      }, f.g);
      if (o.direct !== false) {
        var last = s.x.length - 1;
        el("text", {
          x: f.x(s.x[last]) + 6, y: f.y(s.y[last]) + 4, fill: colour,
          "font-family": "var(--f-mono)", "font-size": 10.5, "font-weight": 600
        }, f.g).textContent = s.label;
      }
    });

    var cross = el("line", { y1: 0, y2: f.ih, stroke: p.ink3, "stroke-width": 1,
      "stroke-dasharray": "2 3", opacity: 0 }, f.g);
    el("rect", {
      x: 0, y: 0, width: f.iw, height: f.ih, fill: "transparent", style: "cursor:crosshair"
    }, f.g).addEventListener("mousemove", function (ev) {
      var box = f.svg.getBoundingClientRect();
      var mx = ev.clientX - box.left - f.m.l;
      var xv = f.x.invert(mx);
      var ref = series[0];
      var k = 0, best = Infinity;
      ref.x.forEach(function (v, j) {
        var dd = Math.abs(v - xv);
        if (dd < best) { best = dd; k = j; }
      });
      cross.setAttribute("x1", f.x(ref.x[k]));
      cross.setAttribute("x2", f.x(ref.x[k]));
      cross.setAttribute("opacity", .7);
      var html = "<b>" + (o.xLabel || "x") + " " + fmt(ref.x[k]) + "</b>";
      series.forEach(function (s) {
        var v = s.y[k];
        html += "<br><span style='color:" + s._c + "'>■</span> " + s.label + " " +
          (o.yPct ? (v * 100).toFixed(1) + "%" : fmt(v));
      });
      showTip(root, tip, f.x(ref.x[k]) + f.m.l, 40, html);
    });
    root.addEventListener("mouseleave", function () {
      hideTip(tip); cross.setAttribute("opacity", 0);
    });

    var cols = [o.xLabel || "x"].concat(series.map(function (s) { return s.label; }));
    attachTable(root, cols, series[0].x.map(function (x, i) {
      return [fmt(x)].concat(series.map(function (s) {
        return o.yPct ? (s.y[i] * 100).toFixed(1) + "%" : fmt(s.y[i]);
      }));
    }));
  };

  /* Estimates with intervals, stacked — the shape of a forest plot. */
  registry.intervals = function (root, d, o, p) {
    var rows = typeof o.rows === "string" ? lookup(d, o.rows) : o.rows;
    var w = root.clientWidth;
    var rowH = o.rowHeight || 30;
    var labelW = o.labelWidth || 150;
    var h = rows.length * rowH + 62 + (o.pooled ? 24 : 0);
    var tip = tipFor(root);
    var lows = rows.map(function (r) { return r[o.lower || "lower"]; });
    var his = rows.map(function (r) { return r[o.upper || "upper"]; });
    var pooled = o.pooled ? lookup(d, o.pooled) : null;
    if (pooled) { lows.push(pooled.lower); his.push(pooled.upper); }
    if (o.truth !== undefined) { lows.push(o.truth); his.push(o.truth); }
    var xd = extent([lows, his]);
    var span = xd[1] - xd[0];
    var f = frame(root, {
      width: w, height: h, margin: { l: labelW, r: 20, t: 18, b: 42 }, p: p,
      xDomain: [xd[0] - span * .08, xd[1] + span * .08],
      yDomain: [0, 1], yTicks: [], xLabel: o.xLabel,
      /* numeric labels are wide; ask for few and let the nice-ticks rule round out */
      xTickCount: o.xTickCount || Math.max(3, Math.floor((w - labelW) / 90)),
      label: o.label || "Estimates with intervals"
    });

    if (o.truth !== undefined) {
      el("line", {
        x1: f.x(o.truth), x2: f.x(o.truth), y1: -2, y2: rows.length * rowH + 4,
        stroke: p.ink2, "stroke-width": 1.5, "stroke-dasharray": "4 4"
      }, f.g);
      el("text", {
        x: f.x(o.truth), y: -2, "text-anchor": "middle", fill: p.ink2,
        "font-family": "var(--f-mono)", "font-size": 10
      }, f.g).textContent = o.truthLabel || "truth";
    }
    if (o.zero) {
      el("line", {
        x1: f.x(0), x2: f.x(0), y1: -2, y2: rows.length * rowH + 4,
        stroke: p.ink3, "stroke-width": 1, "stroke-dasharray": "2 3", opacity: .7
      }, f.g);
    }

    rows.forEach(function (r, i) {
      var yy = i * rowH + rowH / 2;
      var lo = f.x(r[o.lower || "lower"]), hi = f.x(r[o.upper || "upper"]);
      var mid = f.x(r[o.value || "estimate"]);
      var colour = r.colour
        ? colourOf(p, r.colour, p.accent)
        : (o.colourBy && r[o.colourBy] !== undefined
            ? p.series[o.colourKeys.indexOf(r[o.colourBy]) % 8] : p.accent);
      if (r.bad) colour = p.boundary;

      el("text", {
        x: -12, y: yy + 4, "text-anchor": "end", fill: p.ink,
        "font-family": "var(--f-ui)", "font-size": 12
      }, f.g).textContent = r[o.label_key || "label"];

      el("line", {
        x1: lo, x2: hi, y1: yy, y2: yy, stroke: colour, "stroke-width": 2,
        "stroke-linecap": "round", opacity: .55
      }, f.g);
      el("line", { x1: lo, x2: lo, y1: yy - 4, y2: yy + 4, stroke: colour,
        "stroke-width": 2 }, f.g);
      el("line", { x1: hi, x2: hi, y1: yy - 4, y2: yy + 4, stroke: colour,
        "stroke-width": 2 }, f.g);
      el("circle", {
        cx: mid, cy: yy, r: o.weightBy && r[o.weightBy]
          ? Math.max(4, Math.min(9, 3 + 26 * r[o.weightBy])) : 5,
        fill: colour, stroke: p.panel, "stroke-width": 2
      }, f.g);

      var hit = el("rect", {
        x: -labelW + 8, y: yy - rowH / 2, width: f.iw + labelW - 8, height: rowH,
        fill: "transparent", style: "cursor:pointer"
      }, f.g);
      hit.addEventListener("mouseenter", function () {
        showTip(root, tip, mid + labelW, yy + 8,
          "<b>" + r[o.label_key || "label"] + "</b><br>" +
          fmt(r[o.value || "estimate"]) + "<br><em>" +
          (o.mass || 95) + "% [" + fmt(r[o.lower || "lower"]) + ", " +
          fmt(r[o.upper || "upper"]) + "]</em>" +
          (r.note ? "<br>" + r.note : ""));
      });
      hit.addEventListener("mouseleave", function () { hideTip(tip); });
    });

    if (pooled) {
      var yy = rows.length * rowH + 14;
      el("line", {
        x1: -labelW + 10, x2: f.iw, y1: yy - 12, y2: yy - 12, stroke: p.rule,
        "stroke-width": 1
      }, f.g);
      el("text", {
        x: -12, y: yy + 5, "text-anchor": "end", fill: p.ink,
        "font-family": "var(--f-ui)", "font-size": 12, "font-weight": 600
      }, f.g).textContent = o.pooledLabel || "pooled";
      var lo = f.x(pooled.lower), hi = f.x(pooled.upper), mid = f.x(pooled.estimate);
      el("path", {
        d: "M" + mid + " " + (yy - 7) + "L" + hi + " " + yy + "L" + mid + " " + (yy + 7) +
           "L" + lo + " " + yy + "Z",
        fill: p.accent, stroke: p.accentdeep, "stroke-width": 1
      }, f.g);
    }

    attachTable(root, [o.label_key || "label", "estimate", "lower", "upper"],
      rows.map(function (r) {
        return [r[o.label_key || "label"], fmt(r[o.value || "estimate"]),
                fmt(r[o.lower || "lower"]), fmt(r[o.upper || "upper"])];
      }));
  };

  /* Scatter with optional funnel contours. */
  registry.funnel = function (root, d, o, p) {
    var fu = lookup(d, o.key);
    var w = root.clientWidth, h = o.height || 340;
    var tip = tipFor(root);
    var xd = extent([fu.y, [fu.pooled]]);
    var sp = (xd[1] - xd[0]) * .18;
    var f = frame(root, {
      width: w, height: h, margin: { l: 54, r: 16, t: 12, b: 46 }, p: p,
      xDomain: [xd[0] - sp, xd[1] + sp], yDomain: [fu.max_se * 1.08, 0],
      xLabel: o.xLabel || "estimate", yLabel: "standard error",
      label: "Funnel plot"
    });

    fu.contours.slice().reverse().forEach(function (c, ci) {
      var pts = c.se.map(function (s, i) { return [f.x(c.lower[i]), f.y(s)]; });
      for (var i = c.se.length - 1; i >= 0; i--) pts.push([f.x(c.upper[i]), f.y(c.se[i])]);
      el("path", {
        d: path(pts) + "Z", fill: p.accent, opacity: .06 + ci * 0.03, stroke: p.rule,
        "stroke-width": 1
      }, f.marks);
    });

    el("line", {
      x1: f.x(fu.pooled), x2: f.x(fu.pooled), y1: 0, y2: f.ih, stroke: p.accent,
      "stroke-width": 1.6, "stroke-dasharray": "4 4"
    }, f.g);

    var keys = o.colourKeys || ["experiment", "model"];
    fu.y.forEach(function (v, i) {
      var kind = fu.read ? fu.read[i] : null;
      var colour = kind ? p.series[keys.indexOf(kind) % 8] : p.accent;
      var cx = f.x(v), cy = f.y(fu.se[i]);
      el("circle", {
        cx: cx, cy: cy, r: 5.5, fill: colour, stroke: p.panel, "stroke-width": 2
      }, f.g);
      var hit = el("circle", { cx: cx, cy: cy, r: 13, fill: "transparent",
        style: "cursor:pointer" }, f.g);
      hit.addEventListener("mouseenter", function () {
        showTip(root, tip, cx + f.m.l, cy + 12,
          "<b>" + fu.labels[i] + "</b><br>" + fmt(v) + " ± " + fmt(fu.se[i]) +
          (kind ? "<br><em>" + kind + " read</em>" : ""));
      });
      hit.addEventListener("mouseleave", function () { hideTip(tip); });
    });

    attachTable(root, ["study", "estimate", "se", "read"],
      fu.y.map(function (v, i) {
        return [fu.labels[i], fmt(v), fmt(fu.se[i]), fu.read ? fu.read[i] : "—"];
      }));
  };

  /* Grouped horizontal bars. */
  registry.bars = function (root, d, o, p) {
    var rows = typeof o.rows === "string" ? lookup(d, o.rows) : o.rows;
    var w = root.clientWidth;
    var rowH = o.rowHeight || 40;
    var labelW = o.labelWidth || 140;
    var h = rows.length * rowH + 52;
    var tip = tipFor(root);
    var vals = rows.map(function (r) { return r.value; });
    var lo = Math.min(0, extent([vals])[0]);
    var hi = extent([vals])[1];
    var f = frame(root, {
      width: w, height: h, margin: { l: labelW, r: 24, t: 6, b: 40 }, p: p,
      xDomain: [lo, hi * 1.06], yDomain: [0, 1], yTicks: [], xLabel: o.xLabel,
      xFmt: o.xFmt === "money" ? function (v) { return Math.round(v).toLocaleString(); } : fmt,
      label: o.label || ""
    });
    var zero = f.x(0);

    rows.forEach(function (r, i) {
      var yy = i * rowH + rowH / 2;
      var bh = Math.min(18, rowH - 16);
      var xv = f.x(r.value);
      var colour = colourOf(p, r.colour, p.series[i % 8]);
      el("text", {
        x: -12, y: yy + 4, "text-anchor": "end", fill: p.ink,
        "font-family": "var(--f-ui)", "font-size": 12.5
      }, f.g).textContent = r.label;
      el("rect", {
        x: Math.min(zero, xv), y: yy - bh / 2, width: Math.abs(xv - zero), height: bh,
        fill: colour, rx: 3
      }, f.g);
      el("text", {
        x: xv + (xv >= zero ? 8 : -8), y: yy + 4,
        "text-anchor": xv >= zero ? "start" : "end", fill: p.ink2,
        "font-family": "var(--f-mono)", "font-size": 11
      }, f.g).textContent = r.display || fmt(r.value);
      if (r.note) {
        var hit = el("rect", {
          x: -labelW + 8, y: yy - rowH / 2, width: f.iw + labelW - 8, height: rowH,
          fill: "transparent", style: "cursor:pointer"
        }, f.g);
        hit.addEventListener("mouseenter", function () {
          showTip(root, tip, xv + labelW, yy + 6, "<b>" + r.label + "</b><br>" + r.note);
        });
        hit.addEventListener("mouseleave", function () { hideTip(tip); });
      }
    });

    attachTable(root, [o.labelHead || "row", o.xLabel || "value"],
      rows.map(function (r) { return [r.label, r.display || fmt(r.value)]; }));
  };

  /* A grid of cells — a contour of one quantity over two others. */
  registry.heatmap = function (root, d, o, p) {
    var m = lookup(d, o.key);
    var xs = m[o.x], ys = m[o.y], z = m[o.z];
    var w = root.clientWidth, h = o.height || 330;
    var tip = tipFor(root);
    var f = frame(root, {
      width: w, height: h, margin: { l: 54, r: 16, t: 10, b: 48 }, p: p,
      xDomain: [xs[0], xs[xs.length - 1]], yDomain: [ys[0], ys[ys.length - 1]],
      xLabel: o.xLabel, yLabel: o.yLabel,
      xFmt: o.pct ? function (v) { return Math.round(v * 100) + "%"; } : fmt,
      yFmt: o.pct ? function (v) { return Math.round(v * 100) + "%"; } : fmt,
      label: o.label || "Heat map"
    });

    var flat = [];
    z.forEach(function (row) { row.forEach(function (v) { flat.push(v); }); });
    var zd = o.zDomain || extent([flat]);
    var cw = f.iw / (xs.length - 1), ch = f.ih / (ys.length - 1);

    /* one hue, light to dark — sequential, never a rainbow */
    function colourOf(v) {
      var t = (v - zd[0]) / (zd[1] - zd[0]);
      t = Math.max(0, Math.min(1, o.reverse ? 1 - t : t));
      return "color-mix(in srgb, " + (o.hue ? p[o.hue] : p.accent) + " " +
        (8 + t * 84).toFixed(1) + "%, " + p.panel + ")";
    }

    z.forEach(function (row, j) {
      row.forEach(function (v, i) {
        var rect = el("rect", {
          x: f.x(xs[i]) - cw / 2, y: f.y(ys[j]) - ch / 2, width: cw + .6, height: ch + .6,
          fill: colourOf(v), style: "cursor:pointer"
        }, f.g);
        rect.addEventListener("mouseenter", function () {
          rect.setAttribute("stroke", p.ink);
          rect.setAttribute("stroke-width", 1.5);
          showTip(root, tip, f.x(xs[i]) + f.m.l, f.y(ys[j]) + 8,
            "<b>" + (o.zLabel || "value") + " " + fmt(v) + "</b><br>" +
            o.xLabel + " " + (o.pct ? (xs[i] * 100).toFixed(0) + "%" : fmt(xs[i])) + "<br>" +
            o.yLabel + " " + (o.pct ? (ys[j] * 100).toFixed(0) + "%" : fmt(ys[j])));
        });
        rect.addEventListener("mouseleave", function () {
          rect.removeAttribute("stroke");
          hideTip(tip);
        });
      });
    });

    (o.marks || []).forEach(function (mk) {
      var cx = f.x(mk.x), cy = f.y(mk.y);
      el("circle", { cx: cx, cy: cy, r: 5, fill: "none", stroke: p.boundary,
        "stroke-width": 2 }, f.g);
      el("text", {
        x: cx + 9, y: cy - 7, fill: p.boundary, "font-family": "var(--f-mono)",
        "font-size": 10.5, "font-weight": 600
      }, f.g).textContent = mk.label;
    });
  };

  /* Simple scatter with labels — Baujat and friends. */
  registry.scatter = function (root, d, o, p) {
    var s = lookup(d, o.key);
    var xs = s[o.x], ys = s[o.y], labels = s[o.labels];
    var w = root.clientWidth, h = o.height || 320;
    var tip = tipFor(root);
    var f = frame(root, {
      width: w, height: h, margin: { l: 54, r: 22, t: 12, b: 48 }, p: p,
      xDomain: [0, extent([xs])[1] * 1.1], yDomain: [0, extent([ys])[1] * 1.12],
      xLabel: o.xLabel, yLabel: o.yLabel, label: o.label || "Scatter"
    });
    xs.forEach(function (x, i) {
      var cx = f.x(x), cy = f.y(ys[i]);
      el("circle", { cx: cx, cy: cy, r: 5.5, fill: p.accent, stroke: p.panel,
        "stroke-width": 2 }, f.g);
      var big = x > extent([xs])[1] * .45 || ys[i] > extent([ys])[1] * .45;
      if (big) {
        el("text", {
          x: cx + 9, y: cy + 4, fill: p.ink2, "font-family": "var(--f-mono)",
          "font-size": 10
        }, f.g).textContent = labels[i];
      }
      var hit = el("circle", { cx: cx, cy: cy, r: 13, fill: "transparent",
        style: "cursor:pointer" }, f.g);
      hit.addEventListener("mouseenter", function () {
        showTip(root, tip, cx + f.m.l, cy + 10, "<b>" + labels[i] + "</b><br>" +
          o.xLabel + " " + fmt(x) + "<br>" + o.yLabel + " " + fmt(ys[i]));
      });
      hit.addEventListener("mouseleave", function () { hideTip(tip); });
    });
    attachTable(root, ["study", o.xLabel, o.yLabel],
      xs.map(function (x, i) { return [labels[i], fmt(x), fmt(ys[i])]; }));
  };

  /* A distribution, with the decision threshold marked. */
  registry.hist = function (root, d, o, p) {
    var counts = lookup(d, o.counts), edges = lookup(d, o.edges);
    var w = root.clientWidth, h = o.height || 220;
    var tip = tipFor(root);
    var f = frame(root, {
      width: w, height: h, margin: { l: 44, r: 14, t: 10, b: 44 }, p: p,
      xDomain: [edges[0], edges[edges.length - 1]],
      yDomain: [0, extent([counts])[1] * 1.08],
      xLabel: o.xLabel, yLabel: "draws", label: o.label || "Posterior draws"
    });
    counts.forEach(function (c, i) {
      var x0 = f.x(edges[i]), x1 = f.x(edges[i + 1]);
      var over = o.threshold !== undefined && (edges[i] + edges[i + 1]) / 2 >= o.threshold;
      var r = el("rect", {
        x: x0 + 1, y: f.y(c), width: Math.max(1, x1 - x0 - 2), height: f.ih - f.y(c),
        fill: over ? p.accent : p.ink3, opacity: over ? .85 : .38, rx: 2
      }, f.g);
      r.addEventListener("mouseenter", function () {
        showTip(root, tip, x0 + f.m.l, f.y(c) + 6,
          "<b>" + fmt(edges[i]) + " – " + fmt(edges[i + 1]) + "</b><br>" + c + " draws");
      });
      r.addEventListener("mouseleave", function () { hideTip(tip); });
    });
    if (o.threshold !== undefined) {
      el("line", {
        x1: f.x(o.threshold), x2: f.x(o.threshold), y1: 0, y2: f.ih, stroke: p.boundary,
        "stroke-width": 2, "stroke-dasharray": "4 4"
      }, f.g);
      el("text", {
        x: f.x(o.threshold) + 7, y: 12, fill: p.boundary, "font-family": "var(--f-mono)",
        "font-size": 10.5, "font-weight": 600
      }, f.g).textContent = o.thresholdLabel || "threshold";
    }
  };

  /* Two points per row joined by a rule — before and after. */
  registry.dumbbell = function (root, d, o, p) {
    var rows = typeof o.rows === "string" ? lookup(d, o.rows) : o.rows;
    var w = root.clientWidth;
    var rowH = o.rowHeight || 42;
    var labelW = o.labelWidth || 150;
    var h = rows.length * rowH + 56;
    var tip = tipFor(root);
    var xd = extent([rows.map(function (r) { return r.a; }),
                     rows.map(function (r) { return r.b; }),
                     o.truth !== undefined ? [o.truth] : []]);
    var sp = (xd[1] - xd[0]) * .12;
    var f = frame(root, {
      width: w, height: h, margin: { l: labelW, r: 26, t: 8, b: 42 }, p: p,
      xDomain: [xd[0] - sp, xd[1] + sp], yDomain: [0, 1], yTicks: [],
      xLabel: o.xLabel, label: o.label || ""
    });
    if (o.truth !== undefined) {
      el("line", {
        x1: f.x(o.truth), x2: f.x(o.truth), y1: -2, y2: rows.length * rowH,
        stroke: p.ink2, "stroke-width": 1.5, "stroke-dasharray": "4 4"
      }, f.g);
      el("text", {
        x: f.x(o.truth), y: -1, "text-anchor": "middle", fill: p.ink2,
        "font-family": "var(--f-mono)", "font-size": 10
      }, f.g).textContent = o.truthLabel || "truth";
    }
    rows.forEach(function (r, i) {
      var yy = i * rowH + rowH / 2;
      el("text", {
        x: -12, y: yy + 4, "text-anchor": "end", fill: p.ink,
        "font-family": "var(--f-ui)", "font-size": 12.5
      }, f.g).textContent = r.label;
      el("line", {
        x1: f.x(r.a), x2: f.x(r.b), y1: yy, y2: yy, stroke: p.rule, "stroke-width": 3,
        "stroke-linecap": "round"
      }, f.g);
      var ax = f.x(r.a), bx = f.x(r.b);
      var dir = bx >= ax ? 1 : -1;   /* labels point outward from the pair */
      el("circle", { cx: ax, cy: yy, r: 5.5, fill: p.ink2, stroke: p.panel,
        "stroke-width": 2 }, f.g);
      el("circle", { cx: bx, cy: yy, r: 6.5, fill: p.accent, stroke: p.panel,
        "stroke-width": 2 }, f.g);
      el("text", {
        x: ax + dir * 12, y: yy + 4, "text-anchor": dir > 0 ? "start" : "end",
        fill: p.ink3, "font-family": "var(--f-mono)", "font-size": 10.5
      }, f.g).textContent = fmt(r.a);
      el("text", {
        x: bx - dir * 12, y: yy + 4, "text-anchor": dir > 0 ? "end" : "start",
        fill: p.accent, "font-family": "var(--f-mono)", "font-size": 10.5,
        "font-weight": 600
      }, f.g).textContent = fmt(r.b);
      var hit = el("rect", {
        x: -labelW + 8, y: yy - rowH / 2, width: f.iw + labelW - 8, height: rowH,
        fill: "transparent", style: "cursor:pointer"
      }, f.g);
      hit.addEventListener("mouseenter", function () {
        showTip(root, tip, f.x(r.b) + labelW, yy + 6, "<b>" + r.label + "</b><br>" +
          (o.aLabel || "before") + " " + fmt(r.a) + "<br>" +
          (o.bLabel || "after") + " " + fmt(r.b));
      });
      hit.addEventListener("mouseleave", function () { hideTip(tip); });
    });
  };

  /* -- plumbing ------------------------------------------------------------ */

  function lookup(obj, pathStr) {
    var cur = obj;
    pathStr.split(".").forEach(function (part) {
      var m = part.match(/^(.*?)\[(\d+)\]$/);
      if (m) {
        if (m[1]) cur = cur[m[1]];
        cur = cur[parseInt(m[2], 10)];
      } else {
        cur = cur[part];
      }
    });
    return cur;
  }

  function fetchData(file) {
    if (!cache[file]) {
      cache[file] = fetch("assets/data/" + file + ".json").then(function (r) {
        if (!r.ok) throw new Error("cannot load " + file + ".json");
        return r.json();
      });
    }
    return cache[file];
  }

  function render(node) {
    var kind = node.dataset.chart;
    var file = node.dataset.file;
    var opts = node.dataset.opt ? JSON.parse(node.dataset.opt) : {};
    if (!registry[kind]) {
      node.innerHTML = "<p class='small muted'>unknown chart type: " + kind + "</p>";
      return;
    }
    fetchData(file).then(function (d) {
      node.innerHTML = "";
      if (node.clientWidth < 40) return;
      try {
        registry[kind](node, d, opts, palette(node));
      } catch (e) {
        node.innerHTML = "<p class='small muted'>chart failed: " + e.message + "</p>";
        if (window.console) console.error(kind, e);
      }
    }).catch(function (e) {
      node.innerHTML = "<p class='small muted'>" + e.message + "</p>";
    });
  }

  function mountAll() {
    mounted = Array.prototype.slice.call(document.querySelectorAll(".chart[data-chart]"));
    mounted.forEach(function (n) {
      /* draw when it first comes near the viewport, so the hero animates on load
         and the rest do not all fire at once */
      if (!("IntersectionObserver" in window)) { render(n); return; }
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting) { render(n); io.disconnect(); }
        });
      }, { rootMargin: "180px" });
      io.observe(n);
    });
  }

  var resizeTimer = null;
  function onResize() {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      mounted.forEach(function (n) {
        if (n.dataset.opt) {
          var o = JSON.parse(n.dataset.opt);
          o.animate = false;
          n.dataset.opt = JSON.stringify(o);
        }
        if (n.firstChild) render(n);
      });
    }, 180);
  }

  window.AxiomCharts = {
    mount: mountAll,
    redraw: function () { mounted.forEach(function (n) { if (n.firstChild) render(n); }); },
    render: render
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", mountAll);
  } else {
    mountAll();
  }
  window.addEventListener("resize", onResize);
})();
