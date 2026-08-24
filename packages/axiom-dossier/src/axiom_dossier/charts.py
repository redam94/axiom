"""The charts an example recorded, redrawn as plotly figures.

A walkthrough record does not carry pictures. It carries what a picture is made
of: a payload of numbers, a ``kind`` naming the shape to draw them in, and an
``opt`` mapping saying which field is the x, what the axis is called and where
the reference line goes. The site draws them in SVG; this module draws the same
eight shapes in plotly, so a report built from a record shows the run's own
figures rather than describing them.

Nothing here invents a number. Every builder reads the payload the run wrote and
draws it; a kind this module does not know is a typed ``Unsupported`` naming it,
never a blank figure and never a guess at what was meant.

The eight kinds, and what each payload holds:

============  ==============================================================
``band``      ``doses``/``mean``/``lower``/``upper``, optional ``truth``
``bars``      ``rows`` of ``label``/``value``, optional ``display``/``note``
``dumbbell``  ``rows`` of ``label``/``a``/``b`` — the same quantity, twice
``funnel``    ``y``/``se``/``labels``/``pooled``/``contours``
``heatmap``   two axis vectors and a matrix, plus optional ``marks``
``intervals`` ``rows`` of ``label``/``estimate``/``lower``/``upper``
``lines``     ``series`` of ``label``/``x``/``y``
``scatter``   parallel ``x``/``y``/``labels`` vectors
============  ==============================================================

``opt`` values beginning with ``@`` are paths into the figure's own payload:
``"@"`` is the payload itself and ``"@rows"`` is ``payload["rows"]``, which is
the convention ``examples/_walkthrough.py`` records them in.
"""

from __future__ import annotations

from typing import Any

from axiom.core import Unsupported
from axiom.report import Theme

__all__ = ["CHART_KINDS", "chart_figure"]

CHART_KINDS = (
    "band",
    "bars",
    "dumbbell",
    "funnel",
    "heatmap",
    "intervals",
    "lines",
    "scatter",
)
"""Every chart kind a walkthrough can record, and this module can draw."""

#: Rows a chart marked as failing are drawn in, matching ``figures._SETTLED``:
#: a state, not a series, so it does not move when the palette does.
_BAD = "#b5453b"


def _graph_objects() -> Any | Unsupported:
    try:
        import plotly.graph_objects as go
    except ModuleNotFoundError:
        return Unsupported(
            reason="plotly is not installed; figures need it",
            detail={"install": 'pip install "axiom-dossier[render]"'},
            missing=("plotly",),
        )
    return go


def _at(payload: Any, path: Any) -> Any:
    """Resolve an ``opt`` value: an ``@path`` into the payload, or itself."""
    if not isinstance(path, str) or not path.startswith("@"):
        return path
    cur = payload
    for part in path[1:].split("."):
        if part:
            cur = cur[part]
    return cur


def _rows(payload: Any, opt: dict[str, Any], key: str = "rows") -> list[dict[str, Any]]:
    value = _at(payload, opt.get(key, f"@{key}"))
    return list(value) if value is not None else []


def _axes(figure: Any, opt: dict[str, Any], theme: Theme) -> None:
    """The two rules every chart here keeps: no title inside it, labelled axes."""
    figure.update_layout(
        title="",
        xaxis_title=opt.get("xLabel", ""),
        yaxis_title=opt.get("yLabel", ""),
        showlegend=False,
        plot_bgcolor=theme.background_color,
        paper_bgcolor=theme.background_color,
        font={"family": theme.font, "size": theme.base_size, "color": theme.text_color},
        colorway=list(theme.palette),
    )
    figure.update_xaxes(gridcolor=theme.rule_color, zeroline=False)
    figure.update_yaxes(gridcolor=theme.rule_color, zeroline=False)


def _row_height(opt: dict[str, Any], rows: int, *, per: float, base: float) -> float:
    """A row chart is as tall as its rows, not as tall as a default."""
    return float(opt.get("height") or rows * float(opt.get("rowHeight", per)) + base)


def _reference_line(figure: Any, x: float, label: str, theme: Theme) -> None:
    figure.add_vline(
        x=x,
        line={"dash": "dash", "width": 1.2, "color": theme.muted_color},
        annotation_text=label,
        annotation_position="top",
        annotation_font={"size": theme.base_size * 0.85, "color": theme.muted_color},
    )


def _band(go: Any, payload: Any, opt: dict[str, Any], theme: Theme) -> Any:
    """A response curve with the band the posterior actually has."""
    band = _at(payload, opt.get("key", "@"))
    doses = list(band["doses"])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=doses + doses[::-1],
            y=list(band["upper"]) + list(band["lower"])[::-1],
            fill="toself",
            fillcolor=_rgba(theme.colour(0), theme.band_opacity),
            line={"width": 0},
            hoverinfo="skip",
            name="interval",
        )
    )
    if band.get("truth"):
        figure.add_trace(
            go.Scatter(
                x=doses,
                y=list(band["truth"]),
                mode="lines",
                line={"color": theme.muted_color, "width": 2, "dash": "dash"},
                name="truth",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=doses,
            y=list(band["mean"]),
            mode="lines",
            line={"color": theme.colour(0), "width": 2.2},
            name="mean",
            hovertemplate="%{x:.4g}<br>%{y:.4g}<extra></extra>",
        )
    )
    _axes(figure, opt, theme)
    figure.update_layout(height=float(opt.get("height") or theme.figure_height))
    return figure


def _intervals(go: Any, payload: Any, opt: dict[str, Any], theme: Theme) -> Any:
    """A forest plot: an estimate per row, its interval, and what it is read against."""
    rows = _rows(payload, opt)
    lower_key, upper_key = opt.get("lower", "lower"), opt.get("upper", "upper")
    values = [float(r["estimate"]) for r in rows]
    figure = go.Figure(
        go.Scatter(
            x=values,
            y=[str(r["label"]) for r in rows],
            mode="markers",
            marker={
                "size": 10,
                "symbol": "square",
                "color": [_BAD if r.get("bad") else theme.colour(0) for r in rows],
            },
            error_x={
                "type": "data",
                "symmetric": False,
                "array": [float(r[upper_key]) - v for r, v in zip(rows, values, strict=True)],
                "arrayminus": [v - float(r[lower_key]) for r, v in zip(rows, values, strict=True)],
                "color": theme.muted_color,
                "thickness": 1.4,
                "width": 6,
            },
            text=[str(r.get("note", "")) for r in rows],
            hovertemplate="%{y}<br>%{x:.4g}<br>%{text}<extra></extra>",
        )
    )
    if opt.get("truth") is not None:
        _reference_line(figure, float(opt["truth"]), str(opt.get("truthLabel", "truth")), theme)
    if opt.get("zero"):
        _reference_line(figure, 0.0, "no effect", theme)
    pooled = _at(payload, opt["pooled"]) if opt.get("pooled") else None
    if pooled is not None:
        _reference_line(
            figure, float(pooled["estimate"]), str(opt.get("pooledLabel", "pooled")), theme
        )
    _axes(figure, opt, theme)
    figure.update_layout(
        height=_row_height(opt, len(rows), per=30.0, base=90.0),
        yaxis={"autorange": "reversed"},
    )
    return figure


def _lines(go: Any, payload: Any, opt: dict[str, Any], theme: Theme) -> Any:
    """One line per series, and the reference the series are read against."""
    series = list(opt.get("series", ()))
    figure = go.Figure()
    for i, spec in enumerate(series):
        figure.add_trace(
            go.Scatter(
                x=list(_at(payload, spec["x"])),
                y=list(_at(payload, spec["y"])),
                mode="lines",
                name=str(spec.get("label", f"series {i + 1}")),
                line={"color": theme.colour(i), "width": 2},
                hovertemplate="%{x:.4g}<br>%{y:.4g}<extra></extra>",
            )
        )
    if opt.get("hline") is not None:
        figure.add_hline(
            y=float(opt["hline"]),
            line={"dash": "dash", "width": 1.2, "color": theme.muted_color},
            annotation_text=str(opt.get("hlineLabel", "")),
            annotation_position="top left",
            annotation_font={"size": theme.base_size * 0.85, "color": theme.muted_color},
        )
    _axes(figure, opt, theme)
    figure.update_layout(
        height=float(opt.get("height") or theme.figure_height),
        showlegend=len(series) > 1,
        legend={"orientation": "h", "y": -0.22, "x": 0},
    )
    if opt.get("yPct"):
        figure.update_yaxes(tickformat=".0%")
    return figure


def _bars(go: Any, payload: Any, opt: dict[str, Any], theme: Theme) -> Any:
    """Horizontal bars, labelled with their own values.

    The same two rules ``figures.diagnostics_plot`` keeps: the label sits outside
    the bar so a short one does not put its number on top of the row name, and a
    chart with a bar on each side of zero is drawn against a zero line.
    """
    rows = _rows(payload, opt)
    values = [float(r["value"]) for r in rows]
    span = max(values + [0.0]) - min(values + [0.0])
    pad = max(span, 1.0) * 0.16
    figure = go.Figure(
        go.Bar(
            x=values,
            y=[str(r["label"]) for r in rows],
            orientation="h",
            marker={"color": theme.colour(0)},
            text=[str(r.get("display", f"{v:.4g}")) for r, v in zip(rows, values, strict=True)],
            textposition="outside",
            cliponaxis=False,
            hovertext=[str(r.get("note", "")) for r in rows],
            hovertemplate="%{y}<br>%{x:.4g}<br>%{hovertext}<extra></extra>",
        )
    )
    _axes(figure, opt, theme)
    figure.update_layout(
        height=_row_height(opt, len(rows), per=34.0, base=80.0),
        yaxis={"autorange": "reversed"},
    )
    figure.update_xaxes(
        range=[min(values + [0.0]) - pad, max(values + [0.0]) + pad],
        zeroline=min(values, default=0.0) < 0.0,
        zerolinecolor=theme.text_color,
        zerolinewidth=1,
    )
    return figure


def _dumbbell(go: Any, payload: Any, opt: dict[str, Any], theme: Theme) -> Any:
    """The same quantity read two ways, with the gap between them drawn."""
    rows = _rows(payload, opt)
    labels = [str(r["label"]) for r in rows]
    figure = go.Figure()
    for label, row in zip(labels, rows, strict=True):
        figure.add_trace(
            go.Scatter(
                x=[float(row["a"]), float(row["b"])],
                y=[label, label],
                mode="lines",
                line={"color": theme.rule_color, "width": 3},
                hoverinfo="skip",
                showlegend=False,
            )
        )
    for key, name, colour, size in (
        ("a", str(opt.get("aLabel", "a")), theme.muted_color, 9),
        ("b", str(opt.get("bLabel", "b")), theme.colour(0), 11),
    ):
        figure.add_trace(
            go.Scatter(
                x=[float(r[key]) for r in rows],
                y=labels,
                mode="markers",
                name=name,
                marker={"size": size, "color": colour},
                hovertemplate="%{y}<br>" + name + " %{x:.4g}<extra></extra>",
            )
        )
    if opt.get("truth") is not None:
        _reference_line(figure, float(opt["truth"]), str(opt.get("truthLabel", "truth")), theme)
    _axes(figure, opt, theme)
    figure.update_layout(
        height=_row_height(opt, len(rows), per=38.0, base=100.0),
        yaxis={"autorange": "reversed"},
        showlegend=True,
        legend={"orientation": "h", "y": -0.25, "x": 0},
    )
    return figure


def _scatter(go: Any, payload: Any, opt: dict[str, Any], theme: Theme) -> Any:
    """Points, with the ones that were named carrying their names."""
    data = _at(payload, opt.get("key", "@"))
    xs = list(data[opt.get("x", "x")])
    ys = list(data[opt.get("y", "y")])
    labels = [str(v) for v in data.get(opt.get("labels", "labels"), [""] * len(xs))]
    figure = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers+text" if any(labels) else "markers",
            marker={"size": 10, "color": theme.colour(0)},
            text=labels,
            textposition="middle right",
            textfont={"size": theme.base_size * 0.85, "color": theme.muted_color},
            hovertemplate="%{x:.4g}, %{y:.4g}<extra></extra>",
        )
    )
    _axes(figure, opt, theme)
    figure.update_layout(height=float(opt.get("height") or theme.figure_height))
    return figure


def _heatmap(go: Any, payload: Any, opt: dict[str, Any], theme: Theme) -> Any:
    """One hue, light to dark — a sequential surface, never a rainbow."""
    data = _at(payload, opt.get("key", "@"))
    xs = list(data[opt["x"]])
    ys = list(data[opt["y"]])
    figure = go.Figure(
        go.Heatmap(
            x=xs,
            y=ys,
            z=[list(row) for row in data[opt["z"]]],
            colorscale=[
                [0.0, _rgba(theme.colour(0), 0.08)],
                [1.0, theme.colour(0)],
            ],
            colorbar={
                "title": {"text": str(opt.get("zLabel", "")), "side": "right"},
                "thickness": 12,
                "outlinewidth": 0,
            },
            hovertemplate="%{x:.4g}, %{y:.4g}<br>%{z:.4g}<extra></extra>",
        )
    )
    marks = list(opt.get("marks", ()))
    if marks:
        figure.add_trace(
            go.Scatter(
                x=[float(m["x"]) for m in marks],
                y=[float(m["y"]) for m in marks],
                mode="markers",
                marker={
                    "size": 10,
                    "symbol": "x-thin",
                    "line": {"width": 2, "color": theme.background_color},
                },
                text=[str(m.get("label", "")) for m in marks],
                hovertemplate="%{text}<extra></extra>",
            )
        )
        for mark in marks:
            figure.add_annotation(
                x=float(mark["x"]),
                y=float(mark["y"]),
                text=str(mark.get("label", "")),
                showarrow=False,
                yshift=14,
                bgcolor=theme.background_color,
                bordercolor=theme.rule_color,
                borderpad=2,
                font={"size": theme.base_size * 0.85, "color": theme.text_color},
            )
    _axes(figure, opt, theme)
    figure.update_layout(height=float(opt.get("height") or theme.figure_height))
    if opt.get("pct"):
        figure.update_xaxes(tickformat=".0%")
        figure.update_yaxes(tickformat=".0%")
    return figure


def _funnel(go: Any, payload: Any, opt: dict[str, Any], theme: Theme) -> Any:
    """Estimate against its standard error, inside the contours a null implies."""
    data = _at(payload, opt.get("key", "@"))
    figure = go.Figure()
    for i, contour in enumerate(reversed(list(data.get("contours", ())))):
        ses = list(contour["se"])
        figure.add_trace(
            go.Scatter(
                x=list(contour["lower"]) + list(contour["upper"])[::-1],
                y=ses + ses[::-1],
                fill="toself",
                fillcolor=_rgba(theme.colour(0), 0.06 + 0.03 * i),
                line={"width": 1, "color": theme.rule_color},
                hoverinfo="skip",
                name=f"{float(contour.get('mass', 0)):.0%}",
            )
        )
    figure.add_vline(
        x=float(data["pooled"]),
        line={"dash": "dash", "width": 1.6, "color": theme.muted_color},
    )
    figure.add_trace(
        go.Scatter(
            x=list(data["y"]),
            y=list(data["se"]),
            mode="markers",
            marker={"size": 9, "color": theme.colour(0)},
            text=[str(v) for v in data.get("labels", ())],
            hovertemplate="%{text}<br>%{x:.4g} ± %{y:.3g}<extra></extra>",
        )
    )
    _axes(figure, opt, theme)
    figure.update_layout(
        height=float(opt.get("height") or theme.figure_height),
        yaxis={"autorange": "reversed", "title": {"text": "standard error"}},
    )
    return figure


def _rgba(colour: str, alpha: float) -> str:
    r, g, b = (int(colour[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


#: Kind -> the builder that draws it. Adding a shape is adding a row here and a
#: function above; nothing else in the package needs to know.
_BUILDERS = {
    "band": _band,
    "bars": _bars,
    "dumbbell": _dumbbell,
    "funnel": _funnel,
    "heatmap": _heatmap,
    "intervals": _intervals,
    "lines": _lines,
    "scatter": _scatter,
}


def chart_figure(
    kind: str, payload: Any, opt: dict[str, Any], *, theme: Theme | None = None
) -> Any | Unsupported:
    """One recorded chart as a plotly figure, or a typed refusal.

    ``Unsupported`` for a kind this module cannot draw, for plotly being absent,
    and for a payload that does not hold what its kind needs — the third is the
    interesting one: a chart drawn from a payload missing a field would be a
    picture of nothing, and a report is better off saying so.
    """
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    builder = _BUILDERS.get(kind)
    if builder is None:
        return Unsupported(
            reason=f"no drawing for a chart of kind {kind!r}",
            detail={"kinds": ", ".join(CHART_KINDS)},
        )
    try:
        return builder(go, payload, dict(opt), theme or Theme())
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        # Never swallowed and never guessed at: the caller gets the field that
        # was missing, and the report drops the figure rather than drawing a
        # half of it.
        return Unsupported(
            reason=f"the recorded payload does not hold what a {kind!r} chart needs",
            detail={"error": f"{type(exc).__name__}: {exc}"[:200]},
        )
