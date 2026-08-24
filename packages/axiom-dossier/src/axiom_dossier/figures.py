"""The figures a causal report actually needs, drawn from the evidence record.

``axiom.report`` already knows how to place a figure: a ``Figure`` block holding
a plotly figure renders **interactive** in HTML and is rasterised through kaleido
for PDF and PPTX. What it cannot do is decide what to draw. That is this module.

One figure carries most of the weight. A report whose findings are point
estimates with intervals wants them on a common axis with the decision
threshold marked, because that is the picture in which "this interval clears the
threshold and that one does not" is a thing you *see* rather than a thing you
work out from a table. ``findings_plot`` draws it, and colours each row by which
side of the threshold it settled on — so the HYPER-3 report shows four rows on
one side of zero and one crossing to the other, which is the entire finding.

Two rules the drawings keep, both from APA:

* **No title inside the figure.** The caption underneath carries it, and a
  figure with both says everything twice. Every function here leaves the title
  empty and the caller supplies a caption.
* **Axes are labelled with the quantity and its unit**, because an axis reading
  ``0.0 … 6.0`` with no unit is not a measurement.

Plotly is an extra (``axiom[viz]``). Every function returns a typed
``Unsupported`` naming it rather than raising, so a report without the extra
loses its figures and keeps everything else.
"""

from __future__ import annotations

from typing import Any

from axiom.core import Unsupported
from axiom.report import Theme

from axiom_dossier.evidence import Evidence, Quantity

__all__ = [
    "SPREAD_LIMIT",
    "available",
    "diagnostics_plot",
    "figures_for",
    "findings_plot",
]

SPREAD_LIMIT = 20.0
"""How much of its own axis the smallest bar may give up before it stops being a bar.

Twenty: a bar a twentieth of the axis is five per cent of the panel, which is
around where a reader stops seeing a quantity and starts seeing a tick mark.
The comparison is against the *axis*, not against the largest bar, because a
chart with a negative value is read against a span that reaches past zero on
both sides.
"""

#: Which side of the threshold a row fell on, and the colour that says so.
#: Deliberately not the theme's series palette: these are *states*, not series,
#: and cycling a categorical palette through them would make "favourable" mean
#: a different colour in a report with a different number of findings.
_SETTLED = {
    True: "#2f7d4f",  # clears the threshold on the good side
    False: "#b5453b",  # clears it on the bad side
    None: "#5b6472",  # spans it, or has no threshold: unsettled
}


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


def available() -> bool:
    """Whether figures can be drawn at all."""
    return not isinstance(_graph_objects(), Unsupported)


def _axis_title(quantities: tuple[Quantity, ...]) -> str:
    """The x-axis label: the unit when the rows share one, else nothing."""
    units = {q.unit for q in quantities if q.unit}
    return units.pop() if len(units) == 1 else ""


def findings_plot(
    evidence: Evidence,
    *,
    theme: Theme | None = None,
    height: float | None = None,
) -> Any | Unsupported:
    """Every finding on one axis, with its interval and the decision threshold.

    One row per finding, point plus interval, ordered as recorded but drawn top
    to bottom so the first finding reads first. A dashed rule marks the
    threshold when the findings share one; rows are coloured by which side they
    settled on, and a row whose interval spans the threshold is grey, because
    "unsettled" is a third state and drawing it as either of the other two is
    the lie the whole package exists to avoid.
    """
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    if not evidence.findings:
        return Unsupported(reason="there are no findings to plot", detail={"title": evidence.title})
    palette = theme or Theme()
    rows = list(evidence.findings)
    labels = [q.label for q in rows]
    values = [q.value for q in rows]
    lower = [q.value - q.interval.lower if q.interval else 0.0 for q in rows]
    upper = [q.interval.upper - q.value if q.interval else 0.0 for q in rows]
    colours = [_SETTLED[q.is_beneficial()] for q in rows]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=values,
            y=labels,
            mode="markers",
            marker={"size": 11, "symbol": "square", "color": colours},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": upper,
                "arrayminus": lower,
                "color": palette.muted_color,
                "thickness": 1.4,
                "width": 6,
            },
            hovertemplate="%{y}<br>%{x:.3g}<extra></extra>",
            showlegend=False,
        )
    )

    thresholds = {q.threshold for q in rows if q.threshold is not None}
    if len(thresholds) == 1:
        mark = thresholds.pop()
        figure.add_vline(
            x=mark,
            line={"dash": "dash", "width": 1.2, "color": palette.muted_color},
            annotation_text=f"threshold {mark:g}",
            annotation_position="top",
            annotation_font={"size": palette.base_size * 0.85},
        )

    figure.update_layout(
        title="",  # APA: the caption carries the title, never the figure
        xaxis_title=_axis_title(tuple(rows)),
        yaxis_title="",
        yaxis={"autorange": "reversed"},
        height=height,
        showlegend=False,
        plot_bgcolor=palette.background_color,
        paper_bgcolor=palette.background_color,
        font={"family": palette.font, "size": palette.base_size, "color": palette.text_color},
    )
    figure.update_xaxes(gridcolor=palette.rule_color, zeroline=False)
    figure.update_yaxes(gridcolor=palette.rule_color)
    return figure


def diagnostics_plot(
    evidence: Evidence,
    *,
    theme: Theme | None = None,
    height: float | None = None,
) -> Any | Unsupported:
    """The recorded diagnostics as bars — what was checked about the analysis.

    Diagnostics rarely share a unit (a coverage share beside a unit count), so
    the bars are labelled with their own values rather than read off a shared
    axis. It is a summary of what was checked, not a comparison.
    """
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    if not evidence.diagnostics:
        return Unsupported(
            reason="there are no diagnostics to plot", detail={"title": evidence.title}
        )
    palette = theme or Theme()
    rows = list(evidence.diagnostics)

    # One axis, or no chart. A retention share of 0.797 drawn beside a count of
    # 372 is a bar of zero width next to a bar of full width, which tells a
    # reader the share is nothing. The table says it properly; refusing the
    # figure is better than drawing a misleading one.
    #
    # The span is measured from zero, because that is where every bar starts:
    # a z of -2.26 beside a count of 40 is read against 42 units of axis, and
    # the 0.9 between them is a fiftieth of it whatever the ratio of the two
    # magnitudes says.
    values = [q.value for q in rows]
    magnitudes = [abs(v) for v in values if v]
    span = max(values + [0.0]) - min(values + [0.0])
    if magnitudes and span / min(magnitudes) > SPREAD_LIMIT:
        return Unsupported(
            reason=(
                "the diagnostics span too many orders of magnitude to share one axis; "
                "the diagnostics table reports them instead"
            ),
            detail={
                "largest": f"{max(magnitudes):g}",
                "smallest": f"{min(magnitudes):g}",
                "span": f"{span:g}",
                "limit": str(SPREAD_LIMIT),
            },
        )
    figure = go.Figure(
        go.Bar(
            x=values,
            y=[q.label for q in rows],
            orientation="h",
            marker={"color": palette.colour(0)},
            text=[f"{q.value:.{q.precision}f}" for q in rows],
            # Outside, always: `auto` puts the label of a short bar just past
            # its end, which for a negative bar is on top of the row's name.
            # `cliponaxis` is what lets a label sit in the padding below.
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x:.4g}<extra></extra>",
        )
    )
    figure.update_layout(
        title="",
        xaxis_title=_axis_title(tuple(rows)),
        yaxis_title="",
        yaxis={"autorange": "reversed"},
        height=height,
        showlegend=False,
        plot_bgcolor=palette.background_color,
        paper_bgcolor=palette.background_color,
        font={"family": palette.font, "size": palette.base_size, "color": palette.text_color},
    )
    # Room at both ends for the labels now sitting outside the bars, and a
    # baseline whenever there is a bar on each side of it: a diverging chart
    # without its zero is a chart whose bars start nowhere in particular.
    pad = max(span, 1.0) * 0.14
    figure.update_xaxes(
        gridcolor=palette.rule_color,
        range=[min(values + [0.0]) - pad, max(values + [0.0]) + pad],
        zeroline=min(values) < 0.0,
        zerolinecolor=palette.text_color,
        zerolinewidth=1,
    )
    return figure


def figures_for(evidence: Evidence, *, theme: Theme | None = None) -> dict[str, Any]:
    """Every figure this evidence can support, keyed for the render context.

    Missing ones are simply absent rather than present-and-broken: a section
    asks for ``findings_figure`` only when the evidence has findings, so a key
    that is not here is a key nothing names.
    """
    out: dict[str, Any] = {}
    for key, drawing in (
        ("findings_figure", findings_plot(evidence, theme=theme)),
        ("diagnostics_figure", diagnostics_plot(evidence, theme=theme)),
    ):
        if not isinstance(drawing, Unsupported):
            out[key] = drawing
    return out
