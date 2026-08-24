"""One figure per subpackage: does each draw the thing its subpackage returns?"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom.core import D, Interval, Outcome, Param, Posterior, Unsupported, dimensionless
from axiom.data import Panel, RoleMap
from axiom.design import obrien_fleming
from axiom.dynamics import Variable, parse_system, unroll
from axiom.infer import diagnose
from axiom.viz import (
    available,
    boundary,
    convergence,
    corrections,
    intervals,
    panel_coverage,
    recovery,
    unrolled,
)

pytestmark = pytest.mark.skipif(not available(), reason="plotly is not installed")


def kinds(figure: object) -> set[str]:
    return {getattr(t, "type", "?") for t in getattr(figure, "data", ())}


# -- infer -------------------------------------------------------------------------------


def test_convergence_draws_a_row_per_parameter() -> None:
    rng = np.random.default_rng(0)
    posterior = Posterior({"a": rng.normal(size=(4, 400)), "b": rng.normal(size=(4, 400))})
    figure = convergence(diagnose(posterior))
    assert not isinstance(figure, Unsupported), figure
    assert kinds(figure) == {"bar"}
    assert set(figure.data[0].y) >= {"a", "b"}


def test_convergence_marks_the_threshold_the_run_is_judged_against() -> None:
    rng = np.random.default_rng(1)
    figure = convergence(diagnose(Posterior({"a": rng.normal(size=(4, 400))})))
    assert figure.layout.shapes, "no threshold rule was drawn"
    assert any("threshold" in str(a.text) for a in figure.layout.annotations)


def test_convergence_refuses_a_report_with_no_rows() -> None:
    class Empty:
        rows: tuple = ()

    out = convergence(Empty())
    assert isinstance(out, Unsupported) and "no parameters" in out.reason


# -- design ------------------------------------------------------------------------------


def test_a_boundary_is_drawn_as_a_step_not_a_slope() -> None:
    """The threshold holds until the next look; a smooth line draws a rule
    nobody agreed to."""
    figure = boundary(obrien_fleming(0.025, [0.25, 0.5, 0.75, 1.0]))
    assert not isinstance(figure, Unsupported), figure
    assert figure.data[0].line.shape == "hv"
    assert len(figure.data[0].y) == 4


def test_the_alpha_spent_rides_in_the_hover_rather_than_a_second_axis() -> None:
    """Two scales on one plot is the chart mistake this package refuses."""
    figure = boundary(obrien_fleming(0.025, [0.5, 1.0]))
    assert figure.data[0].customdata is not None
    assert "customdata" in figure.data[0].hovertemplate
    assert len(figure.data) == 1, "a second trace would be a second scale"


# -- calibrate ---------------------------------------------------------------------------


def test_corrections_draw_each_step_that_moved_the_number() -> None:
    from axiom.calibrate import variance_reweight

    applied = [
        variance_reweight(0.40, n_source=1200, n_target=300),
        variance_reweight(0.31, n_source=300, n_target=900),
    ]
    figure = corrections(applied)
    assert not isinstance(figure, Unsupported), figure
    assert kinds(figure) == {"waterfall"}
    # start, one per correction, and the total
    assert len(figure.data[0].x) == len(applied) + 2
    assert figure.data[0].measure[-1] == "total"


def test_corrections_refuses_an_empty_sequence() -> None:
    out = corrections([])
    assert isinstance(out, Unsupported) and "no corrections" in out.reason


# -- data --------------------------------------------------------------------------------


def test_panel_coverage_shows_where_the_gap_is() -> None:
    """Unbalance has a pattern; a count of missing rows does not have one."""
    frame = pd.DataFrame(
        {
            "unit": ["a"] * 3 + ["b"] * 2,
            "period": [1, 2, 3, 1, 3],
            "y": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    panel = Panel(frame, RoleMap(unit="unit", time="period", outcome=("y", Outcome(name="y"))))
    figure = panel_coverage(panel)
    assert not isinstance(figure, Unsupported), figure
    assert kinds(figure) == {"heatmap"}
    grid = figure.data[0].z
    assert grid[0] == [1.0, 1.0, 1.0], "unit a is complete"
    assert grid[1] == [1.0, 0.0, 1.0], "unit b's gap is at period 2 and is visible"


# -- core --------------------------------------------------------------------------------


def test_intervals_puts_them_on_one_scale() -> None:
    figure = intervals(
        {
            "a": Interval(lower=-2.0, upper=-1.0, definition="eti", mass=0.9),
            "b": Interval(lower=-0.5, upper=0.5, definition="eti", mass=0.9),
        },
        unit="mmHg",
    )
    assert not isinstance(figure, Unsupported), figure
    assert list(figure.data[0].y) == ["a", "b"]
    assert figure.layout.xaxis.title.text == "mmHg"
    assert figure.data[0].x[0] == pytest.approx(-1.5), "the marker sits at the midpoint"


def test_intervals_refuses_something_that_is_not_an_interval() -> None:
    out = intervals({"a": object()})
    assert isinstance(out, Unsupported) and "lower" in str(out.detail) + out.reason


def test_intervals_refuses_an_empty_mapping() -> None:
    assert isinstance(intervals({}), Unsupported)


# -- sim ---------------------------------------------------------------------------------


def test_recovery_draws_the_diagonal_and_the_points() -> None:
    figure = recovery({"a": 1.0, "b": -2.0}, {"a": 1.1, "b": -1.9})
    assert not isinstance(figure, Unsupported), figure
    assert len(figure.data) == 2, "the diagonal and the estimates"
    assert list(figure.data[1].text) == ["a", "b"]


def test_recovery_says_so_when_the_names_do_not_line_up() -> None:
    out = recovery({"a": 1.0}, {"b": 1.0})
    assert isinstance(out, Unsupported)
    assert "share no parameter names" in out.reason


# -- dynamics ----------------------------------------------------------------------------


def test_an_unrolled_system_is_laid_out_with_time_across() -> None:
    system = parse_system(
        "stock = decay * stock[t-1] + beta * inflow",
        variables=(
            Variable(name="stock", dimension=D.outcome, initial=0.0),
            Variable(name="inflow", dimension=D.currency, role="exogenous"),
        ),
        parameters=(
            Param(name="decay", dimension=dimensionless()),
            Param(name="beta", dimension=D.outcome / D.currency),
        ),
    )
    result = unroll(system, periods=3)
    assert not isinstance(result, Unsupported), result
    figure = unrolled(result)
    assert not isinstance(figure, Unsupported), figure
    # `columns` is the exogenous data only; the solved variables are the keys of
    # `expressions`, and drawing only the former leaves every node a root
    assert len(figure.data[0].x) == len(result.columns) + len(result.expressions)
    assert set(figure.data[0].text) >= {"stock.t0", "inflow.t0"}
    assert figure.layout.xaxis.title.text == "time →"
    # depth is time: a later period sits to the right of an earlier one
    assert max(figure.data[0].x) > min(figure.data[0].x)


def test_unrolled_refuses_something_that_is_not_one() -> None:
    out = unrolled(object())
    assert isinstance(out, Unsupported) and "columns" in out.reason
