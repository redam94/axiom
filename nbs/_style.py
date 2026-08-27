"""One look for every figure in ``nbs/``, and the six shapes they keep needing.

The notebooks are the place where a reader decides whether a piece of ``axiom``
is worth their afternoon, and a printed number does not make that case: a
posterior that halved, a design that costs a third as much per nat, a curve that
saturates at the dose someone was about to double — those are *pictures*, and
until this module existed each notebook that wanted one hand-rolled twelve lines
of plotly and drifted from every other notebook that had done the same.

So: one registered template, one validated palette, and a small set of builders
for the shapes that actually recur across eighty-four notebooks.

    import sys; sys.path[:0] = ["..", "../.."]   # nbs/ is on the path either way
    from _style import curve_band, compare, intervals

    curve_band(se_grid, eig, title="What a nat costs", x_title="experiment se")

**This is presentation, not library code.** It lives under ``nbs/`` rather than
in ``axiom.viz`` on purpose: ``axiom.viz`` draws *axiom's result types* — a
``ResponseBand``, a ``SpecCurve``, a ``CausalGraph`` — and every function there
is duck-typed over one of them. These builders take arrays. Reach for
``axiom.viz`` whenever a figure exists for the object at hand; reach for this
when a notebook needs to draw the *argument* rather than the result.

Requires ``axiom[viz]`` (plotly). The notebooks are a development surface and
``make notebooks`` installs the whole dev group, so they import it unguarded.

Colours are the validated default of the data-viz reference palette: slots
blue / orange / aqua clear the colour-blindness and normal-vision separation
floors on every pair, which is why no builder here draws more than three series
without asking for a fourth by name. Every series is direct-labelled as well as
legended, because two of the slots sit below 3:1 against the surface and colour
alone is never allowed to carry identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

__all__ = [
    "AQUA",
    "AXIS",
    "BLUE",
    "CRITICAL",
    "GOOD",
    "GRID",
    "INK",
    "MUTED",
    "ORANGE",
    "SERIES",
    "SUBTLE",
    "SURFACE",
    "WARNING",
    "annotate",
    "band",
    "caption",
    "compare",
    "curve_band",
    "density",
    "dumbbell",
    "figure",
    "heat",
    "intervals",
    "lines",
    "mark_x",
    "mark_y",
    "points",
    "scatter_fit",
    "shade",
    "steps",
]

# --- the palette ----------------------------------------------------------
# Categorical slots, in fixed order. Never cycle past the third without a
# reason: three is what validates on all pairs, in both colour-vision
# simulations, against this surface.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
VIOLET = "#4a3aa7"
SERIES = (BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET)

# Status. Reserved — a status colour never stands in for "series 4", and never
# carries meaning without a word beside it.
GOOD = "#0ca30c"
WARNING = "#fab219"
CRITICAL = "#d03b3b"

# Chrome and ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SUBTLE = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

_FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'

_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": _FONT, "size": 13, "color": SUBTLE},
        title={
            "font": {"size": 16, "color": INK},
            "x": 0.0,
            "xref": "paper",
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
            "pad": {"t": 14, "b": 8},
            "automargin": True,
        },
        colorway=list(SERIES),
        margin={"l": 64, "r": 28, "t": 84, "b": 56},
        height=380,
        hoverlabel={"font": {"family": _FONT, "size": 12}, "bgcolor": SURFACE},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.01,
            "x": 0.0,
            "xanchor": "left",
            "font": {"size": 12, "color": SUBTLE},
            "bgcolor": "rgba(0,0,0,0)",
        },
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "linecolor": AXIS,
            "ticks": "outside",
            "tickcolor": AXIS,
            "ticklen": 4,
            "tickfont": {"size": 12, "color": MUTED},
            "title": {"font": {"size": 12, "color": MUTED}, "standoff": 10},
        },
        yaxis={
            "gridcolor": GRID,
            "gridwidth": 1,
            "zeroline": False,
            "linecolor": "rgba(0,0,0,0)",
            "ticks": "",
            "tickfont": {"size": 12, "color": MUTED},
            "title": {"font": {"size": 12, "color": MUTED}, "standoff": 10},
        },
    )
)
pio.templates["axiom"] = _TEMPLATE
pio.templates.default = "axiom"


def _rgba(hex_color: str, alpha: float) -> str:
    """``#2a78d6`` at an alpha, for a band under its own curve."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def figure(
    *,
    title: str = "",
    subtitle: str = "",
    x_title: str = "",
    y_title: str = "",
    height: float = 380.0,
    showlegend: bool | None = None,
) -> go.Figure:
    """An empty figure wearing the house template.

    ``subtitle`` is the line that says what the reader should take from the
    picture. Use it: a title names the axes, a subtitle makes the point.
    """
    text = title
    if subtitle:
        text = f"{title}<br><sup style='color:{MUTED}'>{subtitle}</sup>" if title else subtitle
    fig = go.Figure()
    fig.update_layout(
        title={"text": text},
        xaxis={"title": {"text": x_title}},
        yaxis={"title": {"text": y_title}},
        height=height,
        margin={"t": 100 if subtitle else 84},
    )
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    return fig


def _rows_height(fig: go.Figure, n_rows: int, *, row: float = 34.0) -> float:
    """Height for a figure with one row per item: chrome first, then the rows.

    Sizing off the row count alone squashes the plot into whatever the title and
    the caption did not take.
    """
    margin = fig.layout.margin
    chrome = float(margin.t or 96) + float(margin.b or 56)
    return max(chrome + 3 * row, chrome + row * n_rows)


def _label_end(fig: go.Figure, x: float, y: float, text: str, color: str) -> None:
    """Direct-label a series at its right-hand end, so colour is never alone.

    Two curves that end at nearly the same value would print one label over the
    other, so an end label that lands on top of an existing one is nudged clear.
    """
    spans = [
        np.ptp(np.asarray(trace.y, dtype=float))
        for trace in fig.data
        if trace.y is not None and len(trace.y) > 1
    ]
    span = max(spans) if spans else 1.0
    shift = 0
    for existing in fig.layout.annotations:
        if existing.xanchor == "left" and existing.y is not None:
            if abs(float(existing.y) - y) < 0.07 * (span or 1.0):
                shift = (existing.yshift or 0) - 15
    fig.add_annotation(
        x=x,
        y=y,
        text=f"<b>{text}</b>",
        showarrow=False,
        xanchor="left",
        yanchor="middle",
        xshift=8,
        yshift=shift,
        font={"size": 12, "color": color},
    )


def _legend_fit(fig: go.Figure) -> go.Figure:
    """One series needs no legend box — the title and the direct label name it.

    Two or more always get one (identity is never colour-alone), and the top
    margin follows, so a figure never carries a legend row it does not use.
    """
    named = sum(1 for trace in fig.data if trace.showlegend is True)
    subtitled = "<sup" in (fig.layout.title.text or "")
    if named <= 1:
        fig.update_layout(showlegend=False, margin={"t": 72 if subtitled else 56})
    else:
        fig.update_layout(showlegend=True, margin={"t": 100 if subtitled else 84})
    return fig


def band(
    fig: go.Figure,
    x: Sequence[float],
    lo: Sequence[float],
    hi: Sequence[float],
    *,
    color: str = BLUE,
    name: str = "",
    alpha: float = 0.16,
) -> go.Figure:
    """Shade an interval band on an existing figure, under whatever draws next."""
    xs = np.asarray(x, dtype=float)
    fig.add_scatter(
        x=np.concatenate([xs, xs[::-1]]),
        y=np.concatenate([np.asarray(hi, dtype=float), np.asarray(lo, dtype=float)[::-1]]),
        mode="lines",
        fill="toself",
        fillcolor=_rgba(color, alpha),
        line={"width": 0},
        hoverinfo="skip",
        showlegend=bool(name),
        name=name,
    )
    return fig


def curve_band(
    x: Sequence[float],
    y: Sequence[float],
    lo: Sequence[float] | None = None,
    hi: Sequence[float] | None = None,
    *,
    label: str = "",
    color: str = BLUE,
    fig: go.Figure | None = None,
    dash: str | None = None,
    **layout: Any,
) -> go.Figure:
    """One curve, optionally inside an interval band. The workhorse.

    Pass ``fig=`` to lay a second curve over the first — two curves is the
    shape of most before/after arguments in these notebooks.
    """
    fig = fig if fig is not None else figure(**layout)
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if lo is not None and hi is not None:
        band(fig, xs, lo, hi, color=color)
    fig.add_scatter(
        x=xs,
        y=ys,
        mode="lines",
        name=label or "value",
        showlegend=bool(label),
        line={"color": color, "width": 2, "dash": dash} if dash else {"color": color, "width": 2},
        hovertemplate="%{x:.4g} → <b>%{y:.4g}</b><extra></extra>",
    )
    if label:
        _label_end(fig, float(xs[-1]), float(ys[-1]), label, color)
        fig.update_layout(margin={"r": 104})
    return _legend_fit(fig)


def lines(
    x: Sequence[float],
    series: Mapping[str, Sequence[float]],
    *,
    colors: Sequence[str] | None = None,
    **layout: Any,
) -> go.Figure:
    """Several curves on one axis, direct-labelled at the right-hand end."""
    fig = figure(**layout)
    palette = list(colors) if colors is not None else list(SERIES)
    xs = np.asarray(x, dtype=float)
    for i, (name, ys) in enumerate(series.items()):
        values = np.asarray(ys, dtype=float)
        color = palette[i % len(palette)]
        fig.add_scatter(
            x=xs,
            y=values,
            mode="lines",
            name=name,
            line={"color": color, "width": 2},
            hovertemplate=f"{name}: <b>%{{y:.4g}}</b><extra></extra>",
        )
        if len(series) <= 4:
            _label_end(fig, float(xs[-1]), float(values[-1]), name, color)
    fig.update_layout(hovermode="x unified")
    if len(series) <= 4:
        fig.update_layout(margin={"r": 104})
    return _legend_fit(fig)


def steps(x: Sequence[float], y: Sequence[float], *, color: str = BLUE, **layout: Any) -> go.Figure:
    """A quantity that holds and then jumps — a stopping boundary, a schedule."""
    fig = figure(**layout)
    fig.add_scatter(
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        mode="lines+markers",
        line={"color": color, "width": 2, "shape": "hv"},
        marker={"size": 8, "color": color},
        showlegend=False,
        hovertemplate="%{x} → <b>%{y:.4g}</b><extra></extra>",
    )
    return fig


def compare(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    highlight: int | str | None = None,
    color: str = BLUE,
    accent: str = ORANGE,
    value_fmt: str = "{:.3g}",
    **layout: Any,
) -> go.Figure:
    """Ranked horizontal bars — the "which of these do I buy" picture.

    ``highlight`` names (or indexes) the one bar the sentence is about; it takes
    the accent hue and every bar keeps its number as a direct label, so the
    ranking survives a greyscale print.
    """
    fig = figure(**layout)
    names = list(labels)
    vals = [float(v) for v in values]
    picked = names.index(highlight) if isinstance(highlight, str) else highlight
    colors = [accent if i == picked else color for i in range(len(names))]
    fig.add_bar(
        x=vals,
        y=names,
        orientation="h",
        marker={"color": colors, "line": {"width": 0}},
        text=[value_fmt.format(v) for v in vals],
        textposition="outside",
        textfont={"size": 12, "color": SUBTLE},
        cliponaxis=False,
        showlegend=False,
        hovertemplate="%{y}: <b>%{x:.4g}</b><extra></extra>",
    )
    fig.update_layout(
        bargap=0.34,
        height=layout.get("height", _rows_height(fig, len(names))),
        margin={"l": 8 + 7 * max((len(n) for n in names), default=8), "r": 78},
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=AXIS)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def intervals(
    rows: Sequence[tuple[str, float, float, float]],
    *,
    ref: float | None = None,
    ref_label: str = "",
    color: str = BLUE,
    accent: str = ORANGE,
    highlight: str | None = None,
    **layout: Any,
) -> go.Figure:
    """A forest: ``(label, point, lo, hi)`` per row, with an optional reference.

    Every number ``axiom`` reports carries an interval, and this is the picture
    of that promise: what is estimated, and how much of the estimate is width.
    """
    fig = figure(**layout)
    for i, (name, point, lo, hi) in enumerate(rows):
        hue = accent if name == highlight else color
        fig.add_scatter(
            x=[lo, hi],
            y=[name, name],
            mode="lines",
            line={"color": hue, "width": 2},
            showlegend=False,
            hoverinfo="skip",
        )
        fig.add_scatter(
            x=[point],
            y=[name],
            mode="markers",
            marker={"size": 10, "color": hue, "line": {"width": 2, "color": SURFACE}},
            showlegend=False,
            name=name,
            hovertemplate=f"{name}: <b>%{{x:.4g}}</b> [{lo:.4g}, {hi:.4g}]<extra></extra>",
        )
        del i
    if ref is not None:
        mark_x(fig, ref, text=ref_label)
    fig.update_layout(
        height=layout.get("height", _rows_height(fig, len(rows))),
        margin={"l": 8 + 7 * max((len(str(r[0])) for r in rows), default=8), "r": 44},
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, automargin=True)
    fig.update_yaxes(autorange="reversed", showgrid=False, automargin=True)
    return fig


def dumbbell(
    labels: Sequence[str],
    before: Sequence[float],
    after: Sequence[float],
    *,
    before_label: str = "before",
    after_label: str = "after",
    **layout: Any,
) -> go.Figure:
    """What changed, per row — the shape of "here is what this bought you"."""
    fig = figure(**layout)
    names = list(labels)
    for i, (name, b, a) in enumerate(zip(names, before, after, strict=True)):
        fig.add_scatter(
            x=[b, a],
            y=[name, name],
            mode="lines",
            line={"color": AXIS, "width": 2},
            showlegend=False,
            hoverinfo="skip",
        )
        fig.add_scatter(
            x=[b],
            y=[name],
            mode="markers",
            marker={"size": 10, "color": MUTED},
            name=before_label,
            showlegend=i == 0,
            hovertemplate=f"{before_label}: <b>%{{x:.4g}}</b><extra></extra>",
        )
        fig.add_scatter(
            x=[a],
            y=[name],
            mode="markers",
            marker={"size": 10, "color": BLUE, "line": {"width": 2, "color": SURFACE}},
            name=after_label,
            showlegend=i == 0,
            hovertemplate=f"{after_label}: <b>%{{x:.4g}}</b><extra></extra>",
        )
    fig.update_layout(
        height=layout.get("height", _rows_height(fig, len(names))),
        margin={"l": 8 + 7 * max((len(n) for n in names), default=8), "r": 44},
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, automargin=True)
    fig.update_yaxes(autorange="reversed", showgrid=False, automargin=True)
    return _legend_fit(fig)


def density(
    samples: Mapping[str, Sequence[float]],
    *,
    ref: float | None = None,
    ref_label: str = "",
    colors: Sequence[str] | None = None,
    **layout: Any,
) -> go.Figure:
    """Kernel densities on one axis — a prior against the posterior it became."""
    from scipy.stats import gaussian_kde

    fig = figure(**layout)
    palette = list(colors) if colors is not None else list(SERIES)
    for i, (name, draws) in enumerate(samples.items()):
        values = np.asarray(draws, dtype=float)
        grid = np.linspace(values.min(), values.max(), 200)
        pdf = gaussian_kde(values)(grid)
        color = palette[i % len(palette)]
        fig.add_scatter(
            x=grid,
            y=pdf,
            mode="lines",
            name=name,
            line={"color": color, "width": 2},
            fill="tozeroy",
            fillcolor=_rgba(color, 0.14),
            showlegend=True,
            hovertemplate=f"{name}: %{{x:.4g}}<extra></extra>",
        )
    if ref is not None:
        mark_x(fig, ref, text=ref_label)
    fig.update_yaxes(showticklabels=False, showgrid=False, title={"text": ""})
    return _legend_fit(fig)


def points(
    groups: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    *,
    colors: Sequence[str] | None = None,
    symbols: Sequence[str] | None = None,
    size: int = 10,
    **layout: Any,
) -> go.Figure:
    """Named clouds of (x, y) — where a design put its points, where a path went."""
    fig = figure(**layout)
    palette = list(colors) if colors is not None else list(SERIES)
    marks = list(symbols) if symbols is not None else ["circle", "diamond", "square", "x"]
    for i, (name, (xs, ys)) in enumerate(groups.items()):
        fig.add_scatter(
            x=np.asarray(xs, dtype=float),
            y=np.asarray(ys, dtype=float),
            mode="markers",
            name=name,
            marker={
                "size": size,
                "color": palette[i % len(palette)],
                "symbol": marks[i % len(marks)],
                "line": {"width": 2, "color": SURFACE},
            },
            showlegend=True,
            hovertemplate=f"{name}: %{{x:.4g}}, %{{y:.4g}}<extra></extra>",
        )
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    return _legend_fit(fig)


def scatter_fit(
    truth: Sequence[float],
    estimate: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    color: str = BLUE,
    **layout: Any,
) -> go.Figure:
    """Estimate against truth, with the line every point should sit on."""
    fig = figure(**layout)
    t, e = np.asarray(truth, dtype=float), np.asarray(estimate, dtype=float)
    lo = float(min(t.min(), e.min()))
    hi = float(max(t.max(), e.max()))
    pad = 0.06 * (hi - lo or 1.0)
    fig.add_scatter(
        x=[lo - pad, hi + pad],
        y=[lo - pad, hi + pad],
        mode="lines",
        line={"color": AXIS, "width": 2, "dash": "dot"},
        name="exact recovery",
        showlegend=True,
        hoverinfo="skip",
    )
    fig.add_scatter(
        x=t,
        y=e,
        mode="markers+text" if labels is not None else "markers",
        text=list(labels) if labels is not None else None,
        textposition="top center",
        textfont={"size": 11, "color": SUBTLE},
        marker={"size": 10, "color": color, "line": {"width": 2, "color": SURFACE}},
        name="estimated",
        showlegend=True,
        hovertemplate="truth %{x:.4g} → estimate <b>%{y:.4g}</b><extra></extra>",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    return _legend_fit(fig)


def heat(
    matrix: Sequence[Sequence[float]],
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    *,
    diverging: bool = False,
    zmid: float | None = None,
    colorbar_title: str = "",
    text_fmt: str = "{:.2f}",
    text: Sequence[Sequence[str]] | None = None,
    **layout: Any,
) -> go.Figure:
    """A grid of one number — a sensitivity sweep, a design grid, a cost table.

    One hue light→dark for magnitude; ``diverging=True`` for a signed quantity,
    where the neutral midpoint has to read as "nothing" rather than as a colour.
    """
    fig = figure(**layout)
    z = np.asarray(matrix, dtype=float)
    scale = "RdBu" if diverging else [[0.0, "#e8f1fc"], [0.5, "#3987e5"], [1.0, "#0d366b"]]
    fig.add_heatmap(
        z=z,
        x=list(x_labels),
        y=list(y_labels),
        colorscale=scale,
        reversescale=diverging,
        zmid=(0.0 if diverging else None) if zmid is None else zmid,
        text=(
            [[str(v) for v in row] for row in text]
            if text is not None
            else [["" if np.isnan(v) else text_fmt.format(v) for v in row] for row in z]
        ),
        texttemplate="%{text}",
        textfont={"size": 11},
        xgap=2,
        ygap=2,
        colorbar={
            "title": {"text": colorbar_title, "font": {"size": 12, "color": MUTED}},
            "thickness": 12,
            "outlinewidth": 0,
            "tickfont": {"size": 11, "color": MUTED},
        },
        hovertemplate="%{y} · %{x}: <b>%{z:.4g}</b><extra></extra>",
    )
    fig.update_layout(height=layout.get("height", _rows_height(fig, len(y_labels), row=38.0)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def mark_x(
    fig: go.Figure,
    x: float,
    *,
    text: str = "",
    color: str = MUTED,
    dash: str = "dot",
    y: float = 0.97,
    right: bool = False,
) -> go.Figure:
    """A vertical reference — a threshold, a decision point, the truth."""
    fig.add_vline(x=x, line={"color": color, "width": 2, "dash": dash})
    if text:
        fig.add_annotation(
            x=x,
            xref="x",
            y=y,
            yref="paper",
            text=text,
            showarrow=False,
            xanchor="right" if right else "left",
            yanchor="top",
            xshift=-6 if right else 6,
            font={"size": 11, "color": color},
        )
    return fig


def mark_y(
    fig: go.Figure, y: float, *, text: str = "", color: str = MUTED, dash: str = "dot"
) -> go.Figure:
    """A horizontal reference — zero on a slope, a target, a break-even."""
    fig.add_hline(x0=0, x1=1, y=y, line={"color": color, "width": 2, "dash": dash})
    if text:
        fig.add_annotation(
            x=0.99,
            xref="paper",
            y=y,
            yref="y",
            text=text,
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font={"size": 11, "color": color},
        )
    return fig


def shade(
    fig: go.Figure, x0: float, x1: float, *, text: str = "", color: str = MUTED, alpha: float = 0.09
) -> go.Figure:
    """A region that means something — a rope, an acceptance band, a budget."""
    fig.add_vrect(
        x0=x0,
        x1=x1,
        fillcolor=_rgba(color, alpha),
        line={"width": 0},
        layer="below",
    )
    if text:
        fig.add_annotation(
            x=(x0 + x1) / 2,
            xref="x",
            y=0.03,
            yref="paper",
            text=text,
            showarrow=False,
            yanchor="bottom",
            font={"size": 11, "color": color},
        )
    return fig


def annotate(
    fig: go.Figure, x: float, y: float, text: str, *, color: str = SUBTLE, arrow: bool = True
) -> go.Figure:
    """Point at the thing the paragraph is about."""
    fig.add_annotation(
        x=x,
        y=y,
        text=text,
        showarrow=arrow,
        arrowhead=0,
        arrowwidth=1.5,
        arrowcolor=AXIS,
        ax=0,
        ay=-38,
        font={"size": 11, "color": color},
        align="center",
    )
    return fig


def caption(fig: go.Figure, text: str) -> go.Figure:
    """A line under the figure, for the sentence a reader should leave with."""
    import textwrap

    wrapped = "<br>".join(textwrap.wrap(" ".join(text.split()), width=110))
    fig.add_annotation(
        text=f"<i>{wrapped}</i>",
        xref="paper",
        yref="paper",
        x=0.0,
        y=0.0,
        yshift=-56,
        xanchor="left",
        yanchor="top",
        align="left",
        showarrow=False,
        font={"size": 11, "color": MUTED},
    )
    fig.update_layout(margin={"b": 72 + 18 * (wrapped.count("<br>") + 1)})
    return fig
