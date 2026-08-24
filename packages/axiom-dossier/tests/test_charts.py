"""Redrawing a walkthrough's recorded charts: every kind, and every refusal.

The payloads here are the shapes ``examples/_walkthrough.py`` actually records,
cut down to the smallest thing that still has the feature under test in it.
"""

from __future__ import annotations

import pytest
from axiom.core import Unsupported
from axiom.report import Theme

from axiom_dossier import JOURNAL_THEME
from axiom_dossier.charts import CHART_KINDS, chart_figure

needs_plotly = pytest.mark.skipif(
    isinstance(chart_figure("bars", {"rows": []}, {}), Unsupported)
    and "plotly" in str(chart_figure("bars", {"rows": []}, {})),
    reason="plotly is not installed",
)

#: One minimal payload per kind, keyed the way a record keys them.
PAYLOADS: dict[str, tuple[object, dict[str, object]]] = {
    "band": (
        {
            "treatment": "dose",
            "doses": [0.0, 1.0, 2.0],
            "mean": [0.0, 1.0, 1.4],
            "lower": [-0.2, 0.8, 1.1],
            "upper": [0.2, 1.2, 1.7],
            "truth": [0.0, 0.9, 1.5],
        },
        {"key": "@", "xLabel": "dose", "yLabel": "response"},
    ),
    "bars": (
        {"rows": [{"label": "a", "value": 2.0, "display": "+2.0"}]},
        {"rows": "@rows", "xLabel": "units"},
    ),
    "dumbbell": (
        {"rows": [{"label": "mean X", "a": 0.0, "b": 1.2}]},
        {"rows": "@rows", "aLabel": "source", "bLabel": "target", "xLabel": "mean"},
    ),
    "funnel": (
        {
            "y": [0.1, 0.3],
            "se": [0.04, 0.15],
            "labels": ["s01", "s02"],
            "pooled": 0.17,
            "max_se": 0.2,
            "contours": [
                {"mass": 0.9, "se": [0.0, 0.2], "lower": [0.1, -0.2], "upper": [0.2, 0.5]}
            ],
        },
        {"key": "@", "xLabel": "standardized effect"},
    ),
    "heatmap": (
        {"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [[1.0, 2.0], [3.0, 4.0]]},
        {"key": "@", "x": "x", "y": "y", "z": "z", "zLabel": "yield"},
    ),
    "intervals": (
        {"rows": [{"label": "beta", "estimate": 0.2, "lower": -0.1, "upper": 0.5, "bad": False}]},
        {"rows": "@rows", "truth": 0, "truthLabel": "the truth", "xLabel": "error"},
    ),
    "lines": (
        {"budgets": [1.0, 2.0], "outcome": [3.0, 4.0]},
        {
            "series": [{"label": "best achievable", "x": "@budgets", "y": "@outcome"}],
            "xLabel": "budget",
            "yLabel": "outcome",
        },
    ),
    "scatter": (
        {"x": [1.0, 2.0], "y": [3.0, 4.0], "labels": ["", "x4"]},
        {"key": "@", "x": "x", "y": "y", "labels": "labels", "xLabel": "a", "yLabel": "b"},
    ),
}


@needs_plotly
@pytest.mark.parametrize("kind", CHART_KINDS)
def test_every_recorded_kind_draws(kind: str) -> None:
    """A kind a walkthrough can record is a kind this module can draw."""
    payload, opt = PAYLOADS[kind]
    figure = chart_figure(kind, payload, opt, theme=JOURNAL_THEME)
    assert not isinstance(figure, Unsupported), figure
    assert figure.data, f"{kind} drew nothing"
    assert figure.layout.title.text in (None, ""), "the caption carries the title, not the figure"


@needs_plotly
def test_the_payload_paths_a_record_writes_are_followed() -> None:
    """``@rows`` is the payload's rows; a plain value is itself."""
    payload, opt = PAYLOADS["lines"]
    figure = chart_figure("lines", payload, opt)
    assert list(figure.data[0].x) == [1.0, 2.0]
    assert list(figure.data[0].y) == [3.0, 4.0]
    assert figure.layout.xaxis.title.text == "budget"


@needs_plotly
def test_a_bar_label_sits_outside_its_bar_and_a_negative_bar_gets_a_baseline() -> None:
    """The same two rules ``diagnostics_plot`` keeps, for the same reason."""
    figure = chart_figure(
        "bars",
        {"rows": [{"label": "z", "value": -2.26}, {"label": "n", "value": 1.0}]},
        {"rows": "@rows"},
    )
    assert not isinstance(figure, Unsupported), figure
    assert figure.data[0].textposition == "outside"
    assert figure.data[0].cliponaxis is False
    assert figure.layout.xaxis.zeroline
    low, high = figure.layout.xaxis.range
    assert low < -2.26 and high > 1.0, "no room for the labels"


@needs_plotly
def test_a_failing_row_is_coloured_as_a_state_rather_than_as_a_series() -> None:
    rows = [
        {"label": "ok", "estimate": 0.1, "lower": 0.0, "upper": 0.2, "bad": False},
        {"label": "missed", "estimate": 2.0, "lower": 1.5, "upper": 2.5, "bad": True},
    ]
    figure = chart_figure("intervals", {"rows": rows}, {"rows": "@rows"})
    assert not isinstance(figure, Unsupported), figure
    colours = list(figure.data[0].marker.color)
    assert colours[1] == "#b5453b" and colours[0] != colours[1]


@needs_plotly
def test_a_reference_the_record_asked_for_is_drawn() -> None:
    payload, opt = PAYLOADS["intervals"]
    figure = chart_figure("intervals", payload, opt)
    assert not isinstance(figure, Unsupported), figure
    assert figure.layout.shapes, "the truth line was not drawn"
    assert any("the truth" in str(a.text) for a in figure.layout.annotations)


@needs_plotly
def test_a_heat_maps_marks_are_legible_on_both_ends_of_its_scale() -> None:
    payload, opt = PAYLOADS["heatmap"]
    marks = {**opt, "marks": [{"x": 0.5, "y": 0.5, "label": "the optimum"}]}
    figure = chart_figure("heatmap", payload, marks, theme=Theme())
    assert not isinstance(figure, Unsupported), figure
    assert any("the optimum" in str(a.text) for a in figure.layout.annotations)
    assert figure.layout.annotations[0].bgcolor, "a label with no box vanishes on a dark cell"


@needs_plotly
def test_the_theme_decides_the_colour_and_the_face() -> None:
    payload, opt = PAYLOADS["band"]
    figure = chart_figure("band", payload, opt, theme=JOURNAL_THEME)
    assert not isinstance(figure, Unsupported), figure
    assert figure.layout.font.family == JOURNAL_THEME.font
    assert figure.layout.paper_bgcolor == JOURNAL_THEME.background_color
    assert figure.data[-1].line.color == JOURNAL_THEME.colour(0)


def test_an_unknown_kind_is_named_rather_than_drawn_blank() -> None:
    out = chart_figure("sankey", {}, {})
    assert isinstance(out, Unsupported)
    assert "sankey" in out.reason
    assert "band" in out.detail["kinds"]


@needs_plotly
def test_a_payload_that_does_not_hold_what_its_kind_needs_is_refused() -> None:
    """Half a chart is worse than none: the caller is told which field was missing."""
    out = chart_figure("band", {"doses": [0.0, 1.0]}, {"key": "@"})
    assert isinstance(out, Unsupported)
    assert "band" in out.reason
    assert "upper" in out.detail["error"]
