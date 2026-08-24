"""Plotly figures over axiom results, behind the ``[viz]`` extra.

plotly is imported lazily inside each figure function; ``axiom.viz``
imports without it and every figure returns ``Unsupported(reason="plotly
not installed", missing=("viz",))`` when it is absent. Figures are thin:
the numbers they draw come from the result objects, which carry their own
provenance (interval definition and mass, N, acceptance regions), and the
labels use the domain-general vocabulary (treatment, dose, outcome).

Each function documents the attributes it reads, so a result written by a
concurrent module only has to satisfy those:

* ``response_curve`` — a ``surface.FitResult`` (``predict_under``,
  ``surface.spec``, ``data``, ``n_periods``).
* ``forest`` — ``meta.ForestData`` (``rows[].label/estimate/interval/weight``,
  ``pooled_estimate``, ``pooled_interval``, ``prediction_interval``,
  ``mass``).
* ``funnel`` — ``meta.FunnelData`` (``y``, ``se``, ``pooled``,
  ``contours[].mass/se/lower/upper``).
* ``sbc_ranks`` — ``diagnose.SBCResult`` (``parameters[].name/histogram/
  bins/n/n_ranks/passed/ecdf_band``).
* ``coverage_plot`` — ``diagnose.CoverageResult``: ``parameters[]`` with
  ``name``, ``rate``, ``passed``, and a count-valued ``region`` with ``n``, ``lower`` /
  ``upper`` (``core.AcceptanceRegion``); ``mass`` on the result.
* ``spec_curve_plot`` — ``diagnose.SpecCurve`` (``rows[].estimate/interval/
  labels/converged/failure``, ``mass``, ``definition``, ``estimand_name``).
* ``backtest_plot`` — ``diagnose.Backtest``: ``scores[]`` with
  ``step``, ``mae``, ``rmse``, ``crps``, ``coverage``; ``mass``.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from axiom.core import TimeWindow, Unsupported
from axiom.core.intervals import IntervalDefinition
from axiom.surface.bands import ResponseBand, marginal_band, response_band

__all__ = [
    "available",
    "backtest_plot",
    "boundary",
    "causal_graph",
    "convergence",
    "corrections",
    "coverage_plot",
    "forest",
    "funnel",
    "intervals",
    "panel_coverage",
    "recovery",
    "response_curve",
    "sbc_ranks",
    "spec_curve_plot",
    "stability",
    "transfer",
    "unrolled",
]

Array = npt.NDArray[np.float64]

_MISSING = Unsupported(
    reason="plotly not installed",
    missing=("viz",),
    detail={"install": "pip install 'axiom[viz]'"},
)


def _graph_objects() -> Any | Unsupported:
    """``plotly.graph_objects`` or the typed failure; never raises on a missing extra."""
    # plotly ships no type information; go through importlib so mypy sees ``Any``.
    try:
        go: Any = importlib.import_module("plotly.graph_objects")
    except ImportError:
        return _MISSING
    return go


def _subplots() -> Any | Unsupported:
    try:
        module: Any = importlib.import_module("plotly.subplots")
    except ImportError:
        return _MISSING
    return module.make_subplots


def available() -> bool:
    """True when plotly can be imported."""
    return not isinstance(_graph_objects(), Unsupported)


def _missing_attributes(obj: object, names: Sequence[str], what: str) -> Unsupported | None:
    absent = [n for n in names if not hasattr(obj, n)]
    if absent:
        return Unsupported(
            reason=f"{what} lacks attributes the figure reads: {absent}",
            missing=tuple(absent),
            detail={"type": type(obj).__name__},
        )
    return None


def _band(
    go: Any, x: Sequence[float], lower: Sequence[float], upper: Sequence[float], name: str
) -> Any:
    xs = list(x) + list(x)[::-1]
    ys = list(upper) + list(lower)[::-1]
    return go.Scatter(
        x=xs, y=ys, fill="toself", mode="lines", line={"width": 0}, opacity=0.25, name=name
    )


# -- surface ----------------------------------------------------------------------------------


def _band_figure(go: Any, band: Any, title: str) -> Any:
    """One band object -> one figure. Every surface figure goes through here.

    The band trace is added *first* and unconditionally: a surface curve without
    its uncertainty is not a thing this module can draw, which is the point of
    routing every one of them through a ``ResponseBand``.
    """
    x_title, y_title = band.axis_titles()
    fig = go.Figure()
    fig.add_trace(_band(go, list(band.doses), list(band.lower), list(band.upper), band.label()))
    fig.add_trace(
        go.Scatter(x=list(band.doses), y=list(band.mean), mode="lines", name="posterior mean")
    )
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    return fig


def _curve(
    result: Any,
    treatment: str,
    kind: str,
    n_grid: int,
    mass: float,
    definition: IntervalDefinition,
    window: TimeWindow | None,
    seed: int | None,
) -> Any | Unsupported:
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    if isinstance(result, ResponseBand):
        return _band_figure(go, result, _title(result))
    bad = _missing_attributes(
        result, ("predict_under", "marginal_under", "surface", "data", "n_periods"), "result"
    )
    if bad is not None:
        return bad
    build = response_band if kind == "response" else marginal_band
    band = build(
        result,
        treatment,
        n_grid=n_grid,
        mass=mass,
        definition=definition,
        window=window,
        seed=seed,
    )
    if isinstance(band, Unsupported):
        return band
    return _band_figure(go, band, _title(band))


def _title(band: Any) -> str:
    if band.kind == "marginal":
        return f"Marginal effect of {band.treatment} on {band.outcome}"
    return f"Response of {band.outcome} to {band.treatment} dose"


def response_curve(
    result: Any,
    treatment: str = "",
    *,
    n_grid: int = 25,
    mass: float = 0.9,
    definition: IntervalDefinition = "eti",
    window: TimeWindow | None = None,
    seed: int | None = None,
) -> Any | Unsupported:
    """Expected outcome against the dose of ``treatment``, **with its posterior band**.

    Takes either a fit — in which case ``surface.response_band`` evaluates the
    grid through the draws — or a ``ResponseBand`` already built, so a report
    that has computed the band once does not compute it again. The band is not
    optional and there is no argument that turns it off: a response curve is a
    posterior quantity, and a line through the posterior mean of the parameters
    is not the curve the model believes.
    """
    return _curve(result, treatment, "response", n_grid, mass, definition, window, seed)


def marginal_curve(
    result: Any,
    treatment: str = "",
    *,
    n_grid: int = 25,
    mass: float = 0.9,
    definition: IntervalDefinition = "eti",
    window: TimeWindow | None = None,
    seed: int | None = None,
) -> Any | Unsupported:
    """``d outcome / d dose`` against dose, with its posterior band.

    The figure a dose decision is read off. Where the band straddles zero the
    model does not know whether the next unit of dose helps or hurts, which a
    line through the posterior mean would hide.
    """
    return _curve(result, treatment, "marginal", n_grid, mass, definition, window, seed)


# -- meta ------------------------------------------------------------------------------------


def forest(data: Any) -> Any | Unsupported:
    """A forest plot: one row per study with its interval, the pooled row, and the
    prediction interval when the data carries one."""
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(data, ("rows", "pooled_estimate", "mass"), "forest data")
    if bad is not None:
        return bad
    labels = [r.label for r in data.rows]
    est = [r.estimate for r in data.rows]
    lo = [r.estimate - r.interval.lower for r in data.rows]
    hi = [r.interval.upper - r.estimate for r in data.rows]
    sizes = [8.0 + 24.0 * (r.weight or 0.0) for r in data.rows]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=est,
            y=labels,
            mode="markers",
            marker={"size": sizes, "symbol": "square"},
            error_x={"type": "data", "symmetric": False, "array": hi, "arrayminus": lo},
            name="studies",
        )
    )
    pooled_iv = getattr(data, "pooled_interval", None)
    err = (
        {
            "type": "data",
            "symmetric": False,
            "array": [pooled_iv.upper - data.pooled_estimate],
            "arrayminus": [data.pooled_estimate - pooled_iv.lower],
        }
        if pooled_iv is not None
        else None
    )
    fig.add_trace(
        go.Scatter(
            x=[data.pooled_estimate],
            y=["pooled"],
            mode="markers",
            marker={"size": 14, "symbol": "diamond"},
            error_x=err,
            name="pooled",
        )
    )
    pred = getattr(data, "prediction_interval", None)
    if pred is not None:
        fig.add_trace(
            go.Scatter(
                x=[pred.lower, pred.upper],
                y=["prediction", "prediction"],
                mode="lines",
                line={"width": 4},
                name=f"prediction interval ({pred.mass:.0%})",
            )
        )
    fig.add_vline(x=0.0, line={"dash": "dot"})
    fig.update_layout(
        title=f"Forest plot ({data.mass:.0%} intervals)",
        xaxis_title="estimate",
        yaxis_title="study",
    )
    return fig


def funnel(data: Any) -> Any | Unsupported:
    """A funnel plot: study estimates against their standard errors (precision upward)
    with the pseudo-confidence contours around the pooled estimate."""
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(data, ("y", "se", "pooled", "contours"), "funnel data")
    if bad is not None:
        return bad
    fig = go.Figure()
    for c in data.contours:
        fig.add_trace(
            go.Scatter(
                x=list(c.lower) + list(c.upper)[::-1],
                y=list(c.se) + list(c.se)[::-1],
                mode="lines",
                line={"dash": "dash", "width": 1},
                name=f"{c.mass:.0%} contour",
            )
        )
    fig.add_trace(go.Scatter(x=list(data.y), y=list(data.se), mode="markers", name="studies"))
    fig.add_vline(x=data.pooled, line={"dash": "dot"})
    fig.update_layout(
        title="Funnel plot",
        xaxis_title="estimate",
        yaxis_title="standard error",
        yaxis={"autorange": "reversed"},
    )
    return fig


# -- diagnose ---------------------------------------------------------------------------------


def sbc_ranks(result: Any, *, columns: int = 3) -> Any | Unsupported:
    """Rank histograms per parameter from an SBC run, with the expected count per bin and
    the binomial band around it; failed parameters are titled as such."""
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(result, ("parameters",), "SBC result")
    if bad is not None:
        return bad
    make_subplots = _subplots()
    if isinstance(make_subplots, Unsupported):
        return make_subplots
    params = list(result.parameters)
    if not params:
        return Unsupported(reason="SBC result has no ranked parameters", missing=("parameters",))
    columns = max(1, min(columns, len(params)))
    rows = int(np.ceil(len(params) / columns))
    titles = [f"{p.name} ({'pass' if p.passed else 'FAIL'})" for p in params]
    fig = make_subplots(rows=rows, cols=columns, subplot_titles=titles)
    for i, p in enumerate(params):
        r, c = divmod(i, columns)
        expected = p.n / p.bins
        # binomial band around the expected count per bin (normal approximation)
        sd = float(np.sqrt(p.n * (1.0 / p.bins) * (1.0 - 1.0 / p.bins)))
        fig.add_trace(
            go.Bar(x=list(range(p.bins)), y=list(p.histogram), name=p.name, showlegend=False),
            row=r + 1,
            col=c + 1,
        )
        for level in (expected - 2.0 * sd, expected, expected + 2.0 * sd):
            fig.add_hline(y=level, line={"dash": "dot", "width": 1}, row=r + 1, col=c + 1)
    fig.update_layout(title="SBC rank histograms", xaxis_title="rank bin", yaxis_title="count")
    return fig


def coverage_plot(result: Any) -> Any | Unsupported:
    """Per-parameter coverage against the nominal mass with each parameter's exact
    acceptance region (``region.lower``/``region.upper``)."""
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(result, ("parameters", "mass"), "coverage result")
    if bad is not None:
        return bad
    params = list(result.parameters)
    for p in params:
        bad = _missing_attributes(p, ("name", "rate", "region", "passed"), "parameter coverage")
        if bad is not None:
            return bad
    names = [p.name for p in params]
    # ``rate`` is covered / n; the acceptance region is in counts, so divide by its n.
    cov = [float(p.rate) for p in params]
    lo = [float(p.rate) - float(p.region.lower) / float(p.region.n) for p in params]
    hi = [float(p.region.upper) / float(p.region.n) - float(p.rate) for p in params]
    colors = ["#2a9d8f" if p.passed else "#e76f51" for p in params]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=names,
            y=cov,
            mode="markers",
            marker={"size": 10, "color": colors},
            error_y={"type": "data", "symmetric": False, "array": hi, "arrayminus": lo},
            name="coverage (acceptance region)",
        )
    )
    fig.add_hline(y=float(result.mass), line={"dash": "dot"})
    fig.update_layout(
        title=f"Interval coverage at nominal {float(result.mass):.0%}",
        xaxis_title="parameter",
        yaxis_title="coverage",
        yaxis={"range": [0.0, 1.0]},
    )
    return fig


def spec_curve_plot(curve: Any) -> Any | Unsupported:
    """The specification curve: estimates sorted with their intervals, coloured by whether
    the interval excludes zero; failed specifications are omitted and counted in the title."""
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(curve, ("rows", "mass", "definition"), "specification curve")
    if bad is not None:
        return bad
    fitted = [r for r in curve.rows if r.estimate is not None and r.interval is not None]
    n_failed = len(curve.rows) - len(fitted)
    if not fitted:
        return Unsupported(reason="no specification produced an estimate", missing=("rows",))
    order = sorted(range(len(fitted)), key=lambda i: fitted[i].estimate)
    est = [fitted[i].estimate for i in order]
    lo = [fitted[i].estimate - fitted[i].interval.lower for i in order]
    hi = [fitted[i].interval.upper - fitted[i].estimate for i in order]
    text = [
        "<br>".join(f"{k}: {v}" for k, v in fitted[i].labels.items())
        + ("" if fitted[i].converged else "<br>(not converged)")
        for i in order
    ]
    colors = [
        "#2a9d8f" if (fitted[i].interval.lower > 0 or fitted[i].interval.upper < 0) else "#8d99ae"
        for i in order
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(order))),
            y=est,
            mode="markers",
            marker={"color": colors, "size": 8},
            error_y={"type": "data", "symmetric": False, "array": hi, "arrayminus": lo},
            text=text,
            hoverinfo="text+y",
            name=f"{curve.definition} {float(curve.mass):.0%}",
        )
    )
    fig.add_hline(y=0.0, line={"dash": "dot"})
    name = getattr(curve, "estimand_name", "estimand")
    fig.update_layout(
        title=f"Specification curve for {name} ({len(fitted)} fitted, {n_failed} failed)",
        xaxis_title="specification (sorted by estimate)",
        yaxis_title=name,
    )
    return fig


def backtest_plot(backtest: Any) -> Any | Unsupported:
    """Per-horizon scores of a rolling-origin backtest — MAE, RMSE, CRPS — and the interval
    coverage against the nominal mass."""
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(backtest, ("scores", "mass"), "backtest")
    if bad is not None:
        return bad
    make_subplots = _subplots()
    if isinstance(make_subplots, Unsupported):
        return make_subplots
    rows = list(backtest.scores)
    for r in rows:
        bad = _missing_attributes(r, ("step", "mae", "rmse", "crps", "coverage"), "horizon score")
        if bad is not None:
            return bad
    h = [int(r.step) for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("error by horizon", "interval coverage"))
    for attr in ("mae", "rmse", "crps"):
        fig.add_trace(
            go.Scatter(
                x=h, y=[float(getattr(r, attr)) for r in rows], mode="lines+markers", name=attr
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(x=h, y=[float(r.coverage) for r in rows], mode="lines+markers", name="coverage"),
        row=1,
        col=2,
    )
    fig.add_hline(y=float(backtest.mass), line={"dash": "dot"}, row=1, col=2)
    fig.update_xaxes(title_text="horizon (periods)")
    fig.update_yaxes(title_text="outcome error", row=1, col=1)
    fig.update_yaxes(title_text="coverage", range=[0.0, 1.0], row=1, col=2)
    fig.update_layout(title="Rolling-origin backtest")
    return fig


# -- structure -----------------------------------------------------------------------


def _layered_from(
    nodes: Sequence[str], parents: dict[str, set[str]]
) -> dict[str, tuple[float, float]]:
    """Positions from an adjacency map, so a DAG and an unrolled system share it."""
    depth: dict[str, int] = {}
    remaining = list(nodes)
    for _ in range(len(remaining) + 1):
        progressed = False
        for node in list(remaining):
            known = [p for p in parents.get(node, set()) if p in depth]
            if len(known) == len(parents.get(node, set())):
                depth[node] = 1 + max((depth[p] for p in known), default=-1)
                remaining.remove(node)
                progressed = True
        if not remaining or not progressed:
            break
    for node in remaining:  # a cycle: put it after everything it does reach
        depth[node] = max(depth.values(), default=0) + 1
    by_depth: dict[int, list[str]] = {}
    for node in nodes:
        by_depth.setdefault(depth.get(node, 0), []).append(node)
    positions: dict[str, tuple[float, float]] = {}
    for level, level_nodes in by_depth.items():
        for i, node in enumerate(sorted(level_nodes)):
            positions[node] = (float(level), -(i - (len(level_nodes) - 1) / 2.0))
    return positions


def _draw_graph(
    go: Any,
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    *,
    bidirected: Sequence[tuple[str, str]] = (),
    hollow: Sequence[str] = (),
    height: float = 420.0,
    x_title: str = "",
) -> Any:
    """The shared drawing: layered nodes, arrows for edges, dashes for bidirected.

    ``go`` is passed in rather than fetched: every caller has already checked
    that plotly imports, and fetching it again here would mean handling a
    failure that cannot happen at a point with nothing useful to say about it.
    """
    parents: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in edges:
        parents.setdefault(b, set()).add(a)
        parents.setdefault(a, set())
    position = _layered_from(list(nodes), parents)

    def arrow(a: str, b: str, dashed: bool) -> dict[str, Any]:
        x0, y0 = position[a]
        x1, y1 = position[b]
        return {
            "x": x0,
            "y": y0,
            "ax": x1,
            "ay": y1,
            "xref": "x",
            "yref": "y",
            "axref": "x",
            "ayref": "y",
            "showarrow": True,
            "arrowhead": 0 if dashed else 3,
            "arrowsize": 1.1,
            "arrowwidth": 1.4,
            "arrowcolor": "#8a8f98" if dashed else "#3c4148",
            "opacity": 0.9,
            "standoff": 14,
            "startstandoff": 14,
        }

    annotations = [arrow(b, a, False) for a, b in edges if a in position and b in position]
    annotations += [arrow(b, a, True) for a, b in bidirected if a in position and b in position]
    faint = set(hollow)
    xs = [position[n][0] for n in nodes]
    ys = [position[n][1] for n in nodes]
    figure = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=list(nodes),
            textposition="middle center",
            textfont={"size": 11},
            marker={
                "size": 46,
                "color": ["#ffffff" if n in faint else "#e8edf3" for n in nodes],
                "line": {
                    "width": [2.0 if n in faint else 1.2 for n in nodes],
                    "color": ["#8a8f98" if n in faint else "#3c4148" for n in nodes],
                },
            },
            hovertext=[
                f"{n}{chr(32)+chr(40)+chr(117)+chr(41)}" if n in faint else n for n in nodes
            ],
            hoverinfo="text",
            showlegend=False,
        )
    )
    figure.update_layout(
        annotations=annotations,
        height=height,
        xaxis={
            "visible": bool(x_title),
            "title": x_title,
            "showgrid": False,
            "range": [min(xs) - 0.6, max(xs) + 0.6],
            "showticklabels": False,
        },
        yaxis={"visible": False, "range": [min(ys) - 0.8, max(ys) + 0.8]},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        margin={"l": 20, "r": 20, "t": 20, "b": 40 if x_title else 20},
        showlegend=False,
    )
    return figure


def _layered(graph: Any) -> dict[str, tuple[float, float]]:
    """Positions for a DAG: depth from the roots across, spread within a depth down.

    A layered layout rather than a force-directed one because a causal graph has
    a direction and the reader is looking for it. Depth is the longest path from
    a root, so an edge always points forwards and never doubles back — which is
    the property that makes the picture readable without arrowheads being
    studied one at a time.
    """
    depth: dict[str, int] = {}
    for node in graph.topological_order():
        parents = [p for p in graph.parents(node) if p in depth]
        depth[node] = 1 + max((depth[p] for p in parents), default=-1)
    by_depth: dict[int, list[str]] = {}
    for node in graph.nodes:
        by_depth.setdefault(depth.get(node, 0), []).append(node)
    positions: dict[str, tuple[float, float]] = {}
    for level, nodes in by_depth.items():
        for i, node in enumerate(sorted(nodes)):
            offset = i - (len(nodes) - 1) / 2.0
            positions[node] = (float(level), -offset)
    return positions


def causal_graph(graph: Any, *, height: float = 420.0) -> Any | Unsupported:
    """The graph itself: nodes, directed edges, and the bidirected ones.

    axiom is a causal package whose central object had no picture. A verdict
    says "identified via the back door adjusting for age"; this is where a
    reader sees *why* — which path the adjustment blocks, and which arrow the
    unmeasured common cause puts there.

    Unmeasured nodes are drawn hollow and bidirected edges dashed, because those
    two are what separate a graph you can identify from one you cannot.
    """
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(graph, ("nodes", "edges"), "causal graph")
    if bad is not None:
        return bad
    if not graph.nodes:
        return Unsupported(reason="the graph has no nodes to draw")
    return _draw_graph(
        go,
        list(graph.nodes),
        list(graph.edges),
        bidirected=list(getattr(graph, "bidirected", ())),
        hollow=list(getattr(graph, "unmeasured", ())),
        height=height,
    )


def stability(report: Any, *, height: float = 380.0) -> Any | Unsupported:
    """How often each edge came back, split by which way it pointed.

    The split is the finding. A bar that is nearly all "undirected" is a *stable
    edge whose direction observation cannot settle* — more rows will not move
    it, and only an intervention will. A short bar is an edge the data is unsure
    about at all. Drawing them as one number would lose exactly the distinction
    that decides what to do next.
    """
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(report, ("edges", "n_bootstrap"), "stability report")
    if bad is not None:
        return bad
    if not report.edges:
        return Unsupported(reason="the report has no edges to draw")

    labels = [f"{e.a} – {e.b}" for e in report.edges]
    figure = go.Figure()
    for name, values, colour in (
        ("→ forward", [e.forward for e in report.edges], "#2f7fd1"),
        ("← backward", [e.backward for e in report.edges], "#8a63c4"),
        ("undirected", [e.undirected for e in report.edges], "#b8bec7"),
    ):
        figure.add_trace(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                name=name,
                marker={"color": colour},
                hovertemplate="%{y}<br>" + name + " in %{x:.0%} of resamples<extra></extra>",
            )
        )
    figure.update_layout(
        barmode="stack",
        height=height,
        xaxis={"title": f"share of {report.n_bootstrap} resamples", "range": [0, 1]},
        yaxis={"autorange": "reversed"},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        legend={"orientation": "h", "y": -0.2},
    )
    return figure


# -- one per subpackage that returns something worth seeing ----------------------------


def convergence(report: Any, *, height: float = 380.0) -> Any | Unsupported:
    """R-hat per parameter against the threshold that decides the run. (``infer``)

    The question a sampler's output actually poses is "may I use this?", and the
    answer is per parameter rather than global — one bad row is enough. Drawing
    every row against the threshold puts the failing ones where they cannot be
    missed, which a table of forty numbers does not.

    Rows with no R-hat (a single chain, a constant) are drawn at zero and named,
    because "not computed" is a different state from "fine" and collapsing them
    is how an unchecked parameter passes for a checked one.
    """
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(report, ("rows",), "convergence report")
    if bad is not None:
        return bad
    rows = list(report.rows)
    if not rows:
        return Unsupported(reason="the report has no parameters")

    limit = getattr(getattr(report, "thresholds", None), "rhat_max", None)
    values = [(r.rhat if r.rhat is not None else 0.0) for r in rows]
    names = [r.name for r in rows]
    failed = [limit is not None and r.rhat is not None and r.rhat > limit for r in rows]
    figure = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker={"color": ["#b5453b" if f else "#2f7fd1" for f in failed]},
            hovertemplate="%{y}<br>R-hat %{x:.4f}<extra></extra>",
        )
    )
    if limit is not None:
        figure.add_vline(
            x=limit,
            line={"dash": "dash", "width": 1.2, "color": "#5b6472"},
            annotation_text=f"threshold {limit:g}",
            annotation_position="top",
        )
    figure.update_layout(
        height=height,
        xaxis={"title": "R-hat"},
        yaxis={"autorange": "reversed"},
        showlegend=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return figure


def boundary(rule: Any, *, height: float = 360.0) -> Any | Unsupported:
    """The stopping threshold at each look, which is the protocol sentence. (``design``)

    A group-sequential design is a promise about when you will stop, and the
    promise is these numbers. Drawn as a step, because the threshold holds
    *until* the next look rather than sliding between them — a smooth line
    would draw a rule nobody agreed to.

    The alpha spent by each look rides in the hover rather than on a second
    axis: two scales on one plot is the chart mistake this package refuses
    everywhere else.
    """
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(rule, ("z", "kind"), "boundary")
    if bad is not None:
        return bad
    z = list(rule.z)
    if not z:
        return Unsupported(reason="the boundary has no looks")
    spent = list(getattr(rule, "spent", ())) or [float("nan")] * len(z)
    looks = list(range(1, len(z) + 1))
    figure = go.Figure(
        go.Scatter(
            x=looks,
            y=z,
            mode="lines+markers",
            line={"shape": "hv", "width": 2.0, "color": "#b5453b"},
            marker={"size": 9},
            customdata=spent,
            hovertemplate=(
                "look %{x}<br>stop beyond z = %{y:.3f}"
                "<br>alpha spent by here: %{customdata:.4f}<extra></extra>"
            ),
            name=f"{rule.kind} boundary",
        )
    )
    figure.update_layout(
        height=height,
        xaxis={"title": "look", "dtick": 1},
        yaxis={"title": "monitoring statistic (z)"},
        showlegend=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return figure


def corrections(applied: Sequence[Any], *, height: float = 360.0) -> Any | Unsupported:
    """What each correction did to the number, in order. (``calibrate``)

    A calibrated estimate is the raw one with a series of operators applied, and
    the useful question is never "what is the answer" but "which correction
    moved it, and by how much". A waterfall answers that; the single corrected
    number does not, and a reader who cannot see the steps has to take the
    result on trust.
    """
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    steps = list(applied)
    if not steps:
        return Unsupported(reason="no corrections were applied")
    missing = _missing_attributes(steps[0], ("name", "counterfactual", "corrected"), "correction")
    if missing is not None:
        return missing

    labels = ["uncorrected"] + [c.name.replace("_", " ") for c in steps] + ["corrected"]
    start = float(steps[0].counterfactual)
    deltas = [float(c.corrected) - float(c.counterfactual) for c in steps]
    measures = ["absolute"] + ["relative"] * len(steps) + ["total"]
    values = [start, *deltas, 0.0]
    figure = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            connector={"line": {"color": "#b8bec7"}},
            increasing={"marker": {"color": "#2f7fd1"}},
            decreasing={"marker": {"color": "#b5453b"}},
            totals={"marker": {"color": "#3c4148"}},
            hovertemplate="%{x}<br>%{y:+.4g}<extra></extra>",
        )
    )
    figure.update_layout(
        height=height,
        yaxis={"title": "estimate"},
        showlegend=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return figure


def panel_coverage(panel: Any, *, height: float = 400.0, max_units: int = 60) -> Any | Unsupported:
    """Which unit was observed in which period. (``data``)

    Every panel method that follows depends on this shape, and a gap in it is
    the thing that quietly turns an estimator into a different estimator. It is
    also the one property of a dataset that a table cannot show and a picture
    can: unbalance has a *pattern*, and the pattern says whether units dropped
    out, arrived late, or were never there.
    """
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(panel, ("frame", "roles"), "panel")
    if bad is not None:
        return bad
    # `frame` and `roles` are properties on Panel, not methods
    frame = panel.frame
    roles = panel.roles
    unit, time = roles.unit, roles.time
    units = list(dict.fromkeys(frame[unit]))[:max_units]
    periods = sorted(dict.fromkeys(frame[time]))
    present = {(u, t) for u, t in zip(frame[unit], frame[time], strict=False)}
    grid = [[1.0 if (u, t) in present else 0.0 for t in periods] for u in units]
    figure = go.Figure(
        go.Heatmap(
            z=grid,
            x=[str(t) for t in periods],
            y=[str(u) for u in units],
            colorscale=[[0.0, "#f2f4f7"], [1.0, "#2f7fd1"]],
            showscale=False,
            hovertemplate="unit %{y}<br>period %{x}<br>%{z}<extra></extra>",
            xgap=1,
            ygap=1,
        )
    )
    dropped = len(dict.fromkeys(frame[unit])) - len(units)
    figure.update_layout(
        height=height,
        xaxis={"title": str(time)},
        yaxis={"title": f"{unit} ({dropped} more not shown)" if dropped else str(unit)},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return figure


def intervals(named: Any, *, height: float = 360.0, unit: str = "") -> Any | Unsupported:
    """Several named intervals on one axis. (``core``)

    The comparison a reader makes by hand, made once. Accepts a mapping of name
    to ``Interval`` — anything carrying ``lower`` and ``upper`` — and draws them
    on a shared scale, which is the only way overlap is visible.

    Deliberately not given a threshold argument. Where an interval sits relative
    to a decision value is an interpretation, and interpretations live where the
    quantity that carries the threshold lives, not in a drawing helper.
    """
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    items = list(named.items()) if hasattr(named, "items") else list(named)
    if not items:
        return Unsupported(reason="no intervals to draw")
    labels = [str(k) for k, _ in items]
    lows, highs, mids = [], [], []
    for _, interval in items:
        missing = _missing_attributes(interval, ("lower", "upper"), "interval")
        if missing is not None:
            return missing
        lows.append(float(interval.lower))
        highs.append(float(interval.upper))
        mids.append((float(interval.lower) + float(interval.upper)) / 2.0)

    figure = go.Figure(
        go.Scatter(
            x=mids,
            y=labels,
            mode="markers",
            marker={"size": 10, "symbol": "line-ns-open", "color": "#3c4148"},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [h - m for h, m in zip(highs, mids, strict=True)],
                "arrayminus": [m - lo for m, lo in zip(mids, lows, strict=True)],
                "color": "#3c4148",
                "thickness": 1.6,
                "width": 7,
            },
            hovertemplate="%{y}<br>%{x:.4g}<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_layout(
        height=height,
        xaxis={"title": unit},
        yaxis={"autorange": "reversed"},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return figure


def transfer(plan: Any, *, height: float = 320.0) -> Any | Unsupported:
    """Which facets differ between source and target, and what that costs. (``estimands``)

    A transfer plan's finding is a *set*: these facets differ, so these
    corrections are required and these assumptions come with them. Drawn as
    counts because the number of differing facets is what decides whether a
    result travels at all, and it is the first thing a reader wants and the last
    thing a paragraph gives them.
    """
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(plan, ("differing", "status"), "transfer plan")
    if bad is not None:
        return bad
    counts = {
        "facets that differ": len(getattr(plan, "differing", ())),
        "corrections required": len(getattr(plan, "corrections", ())),
        "assumptions added": len(getattr(plan, "assumptions", ())),
        "ledger lines": len(getattr(plan, "ledger_lines", ())),
    }
    colour = {"identified": "#3aa17e", "downgraded": "#c9a227"}.get(plan.status, "#b5453b")
    figure = go.Figure(
        go.Bar(
            x=list(counts.values()),
            y=list(counts),
            orientation="h",
            marker={"color": colour},
            text=[str(v) for v in counts.values()],
            textposition="auto",
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    figure.update_layout(
        height=height,
        xaxis={"title": f"count — transfer is {plan.status}"},
        yaxis={"autorange": "reversed"},
        showlegend=False,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return figure


def recovery(truth: Any, estimated: Any, *, height: float = 400.0) -> Any | Unsupported:
    """Estimated against true, for a world where the truth is known. (``sim``)

    The picture every recovery test is implicitly making. Points on the diagonal
    are parameters the estimator found; distance from it is bias, and it is
    visible at a glance in a way a table of differences is not.

    Both arguments are mappings from name to value, so this works for any
    simulated world rather than for one result type.
    """
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    shared = [k for k in truth if k in estimated]
    if not shared:
        return Unsupported(
            reason="truth and estimate share no parameter names",
            detail={
                "truth": ", ".join(list(truth)[:6]),
                "estimated": ", ".join(list(estimated)[:6]),
            },
        )
    xs = [float(truth[k]) for k in shared]
    ys = [float(estimated[k]) for k in shared]
    lo = min(min(xs), min(ys))
    hi = max(max(xs), max(ys))
    pad = (hi - lo) * 0.08 or 1.0
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[lo - pad, hi + pad],
            y=[lo - pad, hi + pad],
            mode="lines",
            line={"dash": "dash", "width": 1.2, "color": "#b8bec7"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text",
            text=shared,
            textposition="top center",
            textfont={"size": 10},
            marker={"size": 10, "color": "#2f7fd1"},
            hovertemplate="%{text}<br>true %{x:.4g} · estimated %{y:.4g}<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_layout(
        height=height,
        xaxis={"title": "true value", "range": [lo - pad, hi + pad]},
        yaxis={"title": "estimated", "range": [lo - pad, hi + pad]},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return figure


def unrolled(system: Any, *, height: float = 420.0) -> Any | Unsupported:
    """A dynamic system after unrolling, laid out so depth is time. (``dynamics``)

    The point of unrolling is that a system with feedback becomes a DAG once
    time is made explicit, and this is where that stops being a claim. The
    layered layout puts each node one step to the right of its parents, so the
    horizontal axis *is* the period — no parsing of column names required, and
    a cycle that survived the unroll would be visible as a node pushed to the
    far right rather than hidden.
    """
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(system, ("columns", "edges", "expressions"), "unrolled system")
    if bad is not None:
        return bad
    # `columns` is the exogenous data only; the endogenous nodes are the keys of
    # `expressions`. Drawing `columns` alone leaves every solved variable out and
    # every remaining node a root, which puts the whole system at depth zero and
    # loses the one thing this figure exists to show.
    nodes = list(dict.fromkeys([*system.columns, *system.expressions]))
    if not nodes:
        return Unsupported(reason="the unrolled system has no nodes to draw")
    return _draw_graph(
        go,
        nodes,
        list(system.edges),
        height=height,
        x_title="time →",
    )
