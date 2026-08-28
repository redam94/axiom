from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom.core import Unverified
from axiom.diagnose import (
    NO_DIFFERENTIAL_ATTRITION,
    Attrition,
    AttritionRow,
    attrition,
)
from axiom.identify import SELECTION_MONOTONICITY, LeeBounds, lee_bounds

_EFFECT = 1.0


def _world(
    *,
    n: int = 6000,
    marginal_floor: float = -1.2,
    always_floor: float = -0.5,
    seed: int = 0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Treatment raises response, and the units it pulls in are low-outcome ones.

    ``always_floor`` bounds the always-responders from below; between
    ``marginal_floor`` and it are the marginal responders, who report only when
    assigned to treatment. The always-responder effect is exactly ``_EFFECT``.
    """
    rng = np.random.default_rng(seed)
    assigned = (rng.random(n) < 0.5).astype(float)
    u = rng.normal(size=n)
    outcome = 10.0 + _EFFECT * assigned + u
    always = u > always_floor
    marginal = (u > marginal_floor) & (u <= always_floor)
    selected = np.where(always, 1.0, np.where(marginal, assigned, 0.0))
    frame = pd.DataFrame(
        {
            # a missing outcome is never read; the sentinel proves it
            "y": np.where(selected == 1.0, outcome, -999.0),
            "assigned": assigned,
            "reported": selected,
        }
    )
    return frame, assigned, selected


def _counts(assigned: np.ndarray, selected: np.ndarray) -> tuple[dict[str, int], dict[str, int]]:
    return (
        {"treated": int(assigned.sum()), "control": int((1 - assigned).sum())},
        {
            "treated": int(selected[assigned == 1.0].sum()),
            "control": int(selected[assigned == 0.0].sum()),
        },
    )


# -- the check ------------------------------------------------------------------------------


def test_differential_attrition_is_caught_and_blocks() -> None:
    _, assigned, selected = _world()
    result = attrition(*_counts(assigned, selected))
    assert isinstance(result, Attrition)
    assert result.differential_attrition
    assert result.differential == pytest.approx(0.204, abs=0.01)
    assert result.worst_arm == "control"
    assert result.verdict().status == "blocked"
    assert "not the same population" in result.verdict().reason
    assert "lee_bounds" in result.verdict().reason
    assert result.assumption().state == "violated"


def test_even_attrition_is_a_power_problem_not_a_comparability_one() -> None:
    """Losing 30 % of both arms costs power; the contrast survives."""
    result = attrition({"treated": 1000, "control": 1000}, {"treated": 700, "control": 705})
    assert not result.differential_attrition
    assert result.overall == pytest.approx(0.7025)
    assert result.verdict().status == "identified"
    assert result.assumption().state == "unverified"
    assert "even" in result.ledger_line().statement


def test_no_loss_and_total_loss_are_degenerate() -> None:
    everyone = attrition({"a": 50, "b": 50}, {"a": 50, "b": 50})
    nobody = attrition({"a": 50, "b": 50}, {"a": 0, "b": 0})
    for result in (everyone, nobody):
        assert not result.differential_attrition
        assert result.p_value == 1.0 and result.chi_square == 0.0


def test_the_summary_reports_per_arm_losses() -> None:
    _, assigned, selected = _world()
    summary = attrition(*_counts(assigned, selected)).summary()
    assert "reported" in summary and "lost" in summary and "blocked" in summary
    assert NO_DIFFERENTIAL_ATTRITION.name == "no_differential_attrition"
    assert "identify.lee_bounds" in NO_DIFFERENTIAL_ATTRITION.challenged_by


def test_attrition_refuses_impossible_inputs() -> None:
    with pytest.raises(ValueError, match="cannot report without being assigned"):
        AttritionRow(arm="a", assigned=5, observed=6)
    with pytest.raises(ValueError, match="same arms"):
        attrition({"a": 5, "b": 5}, {"a": 5, "c": 5})
    with pytest.raises(ValueError, match="at least two arms"):
        attrition({"a": 5}, {"a": 5})
    with pytest.raises(ValueError, match="two arms with units"):
        attrition({"a": 5, "b": 0}, {"a": 5, "b": 0})
    with pytest.raises(ValueError, match="alpha must be in"):
        attrition({"a": 5, "b": 5}, {"a": 5, "b": 5}, alpha=0.0)


def test_an_empty_arm_has_no_rate() -> None:
    result = attrition({"a": 10, "b": 0, "c": 10}, {"a": 5, "b": 0, "c": 4})
    empty = next(row for row in result.arms if row.arm == "b")
    assert np.isnan(empty.rate) and empty.lost == 0


# -- the answer that survives ---------------------------------------------------------------


def test_the_bounds_contain_the_truth_and_the_naive_contrast_does_not() -> None:
    """The point of the module, in one assertion each way."""
    frame, _, _ = _world()
    result = lee_bounds(frame, "y", "assigned", "reported")
    assert isinstance(result, LeeBounds)
    assert result.lower <= _EFFECT <= result.upper
    assert result.naive < result.upper
    assert result.naive == pytest.approx(0.71, abs=0.05)
    assert abs(result.naive - _EFFECT) > 0.2  # the naive contrast is meaningfully wrong


def test_the_trim_share_is_the_selection_gap_and_prices_the_width() -> None:
    frame, _, _ = _world()
    result = lee_bounds(frame, "y", "assigned", "reported")
    assert result.trimmed == pytest.approx(1.0 - result.rate_control / result.rate_treated)
    assert result.trimmed == pytest.approx(0.23, abs=0.02)
    assert result.width > 0.5
    # a smaller gap trims less and bounds tighter
    milder, _, _ = _world(marginal_floor=-0.7)
    tighter = lee_bounds(milder, "y", "assigned", "reported")
    assert isinstance(tighter, LeeBounds)
    assert tighter.trimmed < result.trimmed and tighter.width < result.width


def test_equal_selection_rates_point_identify() -> None:
    frame = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "assigned": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "reported": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
        }
    )
    result = lee_bounds(frame, "y", "assigned", "reported")
    assert isinstance(result, LeeBounds)
    assert result.point_identified and result.trimmed == 0.0
    assert result.lower == result.upper == pytest.approx(result.naive)
    assert result.width == 0.0


def test_the_bounded_population_is_a_latent_one() -> None:
    frame, _, _ = _world()
    result = lee_bounds(frame, "y", "assigned", "reported")
    assert isinstance(result, LeeBounds)
    population = result.population
    assert population.is_latent
    assert population.latent is not None
    assert population.latent.kind == "always_responder"
    assert population.latent.instrument == "assigned"
    assert population.latent.share == pytest.approx(1.0 - result.trimmed)


def test_the_verdict_is_downgraded_even_when_the_bounds_collapse() -> None:
    frame, _, _ = _world()
    wide = lee_bounds(frame, "y", "assigned", "reported")
    assert isinstance(wide, LeeBounds)
    verdict = wide.verdict()
    assert verdict.status == "downgraded"
    assert verdict.assumptions == (SELECTION_MONOTONICITY,)
    assert SELECTION_MONOTONICITY.state == "unverified"
    assert "not testable" in SELECTION_MONOTONICITY.challenged_by


def test_the_direction_of_selection_does_not_change_the_answer() -> None:
    """Flipping which arm over-selects flips which arm is trimmed, not the estimand."""
    frame, _, _ = _world()
    flipped = frame.assign(assigned=1.0 - frame["assigned"])
    forward = lee_bounds(frame, "y", "assigned", "reported")
    backward = lee_bounds(flipped, "y", "assigned", "reported")
    assert isinstance(forward, LeeBounds) and isinstance(backward, LeeBounds)
    assert backward.trimmed_arm.startswith("not-")
    assert backward.naive == pytest.approx(-forward.naive)
    assert backward.lower == pytest.approx(-forward.upper, abs=1e-9)
    assert backward.upper == pytest.approx(-forward.lower, abs=1e-9)


def test_excludes_zero_says_whether_the_sign_is_settled() -> None:
    frame, _, _ = _world()
    strong = lee_bounds(frame, "y", "assigned", "reported")
    assert isinstance(strong, LeeBounds) and strong.excludes_zero
    null, _, _ = _world(seed=3)
    flat = null.assign(y=null["y"] - _EFFECT * null["assigned"])
    weak = lee_bounds(flat, "y", "assigned", "reported")
    assert isinstance(weak, LeeBounds) and not weak.excludes_zero


# -- what it will not do --------------------------------------------------------------------


def test_an_arm_with_no_reports_is_unverified_not_a_number() -> None:
    frame, _, _ = _world()
    silent = frame.assign(reported=np.where(frame["assigned"] == 1.0, frame["reported"], 0.0))
    result = lee_bounds(silent, "y", "assigned", "reported")
    assert isinstance(result, Unverified)
    assert "no outcomes at all" in result.reason


def test_too_few_reports_to_trim_is_unverified() -> None:
    frame = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0],
            "assigned": [1.0, 1.0, 0.0, 0.0],
            "reported": [1.0, 1.0, 1.0, 0.0],
        }
    )
    result = lee_bounds(frame, "y", "assigned", "reported")
    assert isinstance(result, Unverified)
    assert "fewer than two reported outcomes" in result.reason


def test_one_empty_arm_is_unverified() -> None:
    frame = pd.DataFrame({"y": [1.0, 2.0], "assigned": [1.0, 1.0], "reported": [1.0, 1.0]})
    result = lee_bounds(frame, "y", "assigned", "reported")
    assert isinstance(result, Unverified)
    assert "both arms" in result.reason


def test_the_columns_must_be_three_and_binary() -> None:
    frame, _, _ = _world()
    with pytest.raises(ValueError, match="three different columns"):
        lee_bounds(frame, "y", "assigned", "assigned")
    with pytest.raises(ValueError, match="must be 0/1"):
        lee_bounds(frame, "assigned", "y", "reported")
    with pytest.raises(KeyError, match="not in frame"):
        lee_bounds(frame, "nope", "assigned", "reported")
    with pytest.raises(ValueError, match="treated_arm must be 0 or 1"):
        lee_bounds(frame, "y", "assigned", "reported", treated_arm=2.0)


def test_the_results_round_trip() -> None:
    frame, assigned, selected = _world()
    bounds = lee_bounds(frame, "y", "assigned", "reported")
    check = attrition(*_counts(assigned, selected))
    assert isinstance(bounds, LeeBounds)
    assert LeeBounds.from_json(bounds.to_json()) == bounds
    assert Attrition.from_json(check.to_json()) == check
