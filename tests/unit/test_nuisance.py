from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom.core import Add, Const, D, Spec, data_names, dimension, load_spec, params, value
from axiom.surface.nuisance import (
    EventIndicators,
    FourierSeasonality,
    LinearTrend,
    NuisanceSet,
    NuisanceTerm,
)


def _frame(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t": np.arange(n, dtype=float),
            "holiday": [1.0 if i % 5 == 0 else 0.0 for i in range(n)],
            "outage": [0.0] * (n - 1) + [1.0],
        }
    )


SEASON = FourierSeasonality(period=12.0, order=3)
TREND = LinearTrend(origin=2.0, scale=10.0)
EVENTS = EventIndicators(events=("holiday", "outage"))
SET = NuisanceSet(terms=(SEASON, TREND, EVENTS))


@pytest.mark.parametrize("term", [SEASON, TREND, EVENTS, SET], ids=lambda t: type(t).__name__)
def test_terms_satisfy_the_protocol_and_are_flat(term: Spec) -> None:
    assert isinstance(term, NuisanceTerm)
    assert type(term).__mro__[1] is Spec


def test_column_counts() -> None:
    assert SEASON.column_names() == ("sin_1", "cos_1", "sin_2", "cos_2", "sin_3", "cos_3")
    assert SEASON.column_names("annual_") == (
        "annual_sin_1",
        "annual_cos_1",
        "annual_sin_2",
        "annual_cos_2",
        "annual_sin_3",
        "annual_cos_3",
    )
    assert len(FourierSeasonality(period=7.0, order=5).column_names()) == 10
    assert TREND.column_names("p_") == ("p_trend",)
    assert EVENTS.column_names("ignored_") == ("holiday", "outage")
    assert EVENTS.parameter_names("p_") == ("p_coef_holiday", "p_coef_outage")
    assert SET.column_names() == (
        "n0_sin_1",
        "n0_cos_1",
        "n0_sin_2",
        "n0_cos_2",
        "n0_sin_3",
        "n0_cos_3",
        "n1_trend",
        "holiday",
        "outage",
    )
    assert len(SET.parameter_names("b_")) == 9
    assert SET.parameter_names("b_")[0] == "b_n0_coef_sin_1"


def test_zero_order_seasonality_is_rejected() -> None:
    with pytest.raises(ValueError):
        FourierSeasonality(period=12.0, order=0)
    with pytest.raises(ValueError):
        FourierSeasonality(period=0.0, order=1)
    with pytest.raises(ValueError):
        LinearTrend(scale=0.0)
    with pytest.raises(ValueError):
        EventIndicators(events=())
    with pytest.raises(ValueError):
        EventIndicators(events=("a", "a"))


def test_seasonality_augment_matches_numpy() -> None:
    frame = _frame()
    out = SEASON.augment(frame, "t", "s_")
    t = frame["t"].to_numpy()
    for k in (1, 2, 3):
        np.testing.assert_allclose(out[f"s_sin_{k}"], np.sin(2 * np.pi * k * t / 12.0))
        np.testing.assert_allclose(out[f"s_cos_{k}"], np.cos(2 * np.pi * k * t / 12.0))
    assert list(out.columns[:3]) == ["t", "holiday", "outage"]  # original columns kept
    assert "s_sin_1" not in frame.columns  # input frame untouched


def test_trend_augment_matches_numpy() -> None:
    frame = _frame()
    out = TREND.augment(frame, "t")
    np.testing.assert_allclose(out["trend"], (frame["t"].to_numpy() - 2.0) / 10.0)


def test_datetime_time_column_is_days_since_epoch() -> None:
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=4, freq="D")})
    days = (frame["date"] - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
    # explicit origin/scale: the datetime is days since the epoch
    out = LinearTrend(origin=0.0, scale=1.0).augment(frame, "date")
    np.testing.assert_allclose(out["trend"], days.to_numpy(dtype=float))
    weekly = FourierSeasonality(period=7.0, order=1).augment(frame, "date")
    assert weekly["sin_1"].iloc[0] == pytest.approx(
        np.sin(2 * np.pi * days.iloc[0] / 7.0), abs=1e-9
    )


def test_trend_defaults_resolve_to_the_observed_window() -> None:
    """Finding 8: origin = first observed time, scale = window length, recorded in attrs."""
    frame = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=4, freq="D")})
    out = LinearTrend().augment(frame, "date")
    np.testing.assert_allclose(out["trend"], [0.0, 1 / 3, 2 / 3, 1.0])
    first_day = float((pd.Timestamp("2024-01-01") - pd.Timestamp("1970-01-01")).days)
    assert out.attrs["trend"]["trend"] == {"origin": first_day, "scale": 3.0}
    assert "trend" not in frame.attrs  # input frame untouched
    # numeric time, unsorted: the origin is the minimum, not the first row
    numeric = pd.DataFrame({"t": [5.0, 1.0, 9.0]})
    got = LinearTrend().augment(numeric, "t", "p_")
    np.testing.assert_allclose(got["p_trend"], [0.5, 0.0, 1.0])
    assert got.attrs["trend"]["p_trend"] == {"origin": 1.0, "scale": 8.0}
    assert LinearTrend().resolve(numeric["t"].to_numpy()) == (1.0, 8.0)
    # one resolved, one explicit
    half = LinearTrend(scale=4.0).augment(numeric, "t")
    np.testing.assert_allclose(half["trend"], [1.0, 0.0, 2.0])
    assert half.attrs["trend"]["trend"] == {"origin": 1.0, "scale": 4.0}
    # two trends in one frame keep both records
    both = LinearTrend(origin=0.0, scale=1.0).augment(got, "t", "q_")
    assert set(both.attrs["trend"]) == {"p_trend", "q_trend"}


def test_trend_scale_cannot_resolve_from_a_degenerate_window() -> None:
    single = pd.DataFrame({"t": [3.0, 3.0]})
    with pytest.raises(ValueError, match="single distinct"):
        LinearTrend().augment(single, "t")
    # an explicit scale makes it fine
    out = LinearTrend(scale=2.0).augment(single, "t")
    np.testing.assert_allclose(out["trend"], [0.0, 0.0])
    with pytest.raises(ValueError, match="empty"):
        LinearTrend().resolve(np.array([]))
    assert LinearTrend(origin=1.0, scale=2.0).resolve(np.array([])) == (1.0, 2.0)
    with pytest.raises(ValueError):
        LinearTrend(scale=-1.0)
    with pytest.raises(ValueError):
        LinearTrend(origin=np.inf)


def test_non_finite_time_values_are_rejected() -> None:
    """Finding 7: NaT / nan / inf in the time column is an error, not a nan basis row."""
    dates = pd.Series(pd.date_range("2024-01-01", periods=4, freq="D"))
    dates.iloc[2] = pd.NaT
    with pytest.raises(ValueError, match="non-finite"):
        FourierSeasonality(period=7.0, order=1).augment(pd.DataFrame({"date": dates}), "date")
    with pytest.raises(ValueError, match="rows \\[2\\]"):
        LinearTrend().augment(pd.DataFrame({"date": dates}), "date")
    with pytest.raises(ValueError, match="non-finite"):
        LinearTrend().augment(pd.DataFrame({"t": [0.0, np.nan, 2.0]}), "t")
    with pytest.raises(ValueError, match="non-finite"):
        SEASON.augment(pd.DataFrame({"t": [0.0, 1.0, np.inf]}), "t")


def test_augment_refuses_to_overwrite_and_names_missing_columns() -> None:
    frame = _frame()
    once = SEASON.augment(frame, "t")
    with pytest.raises(ValueError, match="already exists"):
        SEASON.augment(once, "t")
    with pytest.raises(KeyError, match="time column"):
        SEASON.augment(frame, "week")
    with pytest.raises(TypeError, match="numeric or datetime"):
        LinearTrend().augment(frame.assign(t=frame["t"].astype(str)), "t")


def test_events_check_presence_and_binary_values() -> None:
    frame = _frame()
    assert EVENTS.augment(frame, "t") is frame
    with pytest.raises(KeyError, match="missing"):
        EventIndicators(events=("holiday", "strike")).augment(frame, "t")
    with pytest.raises(ValueError, match="0/1"):
        EVENTS.augment(frame.assign(holiday=frame["holiday"] * 2), "t")
    with pytest.raises(ValueError, match="0/1"):
        EVENTS.augment(frame.assign(outage=[np.nan] + [0.0] * 11), "t")


@pytest.mark.parametrize("term", [SEASON, TREND, EVENTS, SET], ids=lambda t: type(t).__name__)
def test_expr_dimension_is_outcome_and_params_match(term: NuisanceTerm) -> None:
    expr = term.expr("p_")
    assert dimension(expr) == D.outcome
    assert dimension(term.expr("p_", D.entity)) == D.entity
    declared = term.parameters("p_")
    assert tuple(p.name for p in declared) == term.parameter_names("p_")
    assert {p.name: p for p in params(expr)} == {p.name: p for p in declared}
    assert data_names(expr) == term.column_names("p_")
    for p in declared:
        assert p.dimension == D.outcome
        assert p.prior is not None
        assert p.prior.family == "normal"
        assert p.prior.hyper == {"mu": 0.0, "sigma": 1.0}


def test_expr_value_is_the_linear_combination() -> None:
    frame = SET.augment(_frame(), "t")
    theta = {name: float(i + 1) for i, name in enumerate(SET.parameter_names())}
    got = value(SET.expr(), data={c: frame[c].to_numpy() for c in SET.column_names()}, params=theta)
    basis = np.column_stack([frame[c].to_numpy() for c in SET.column_names()])
    expected = basis @ np.array(list(theta.values()))
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_coefficient_scale_sets_the_prior() -> None:
    (p,) = LinearTrend(coefficient_scale=3.0).parameters()
    assert p.prior is not None
    assert p.prior.hyper == {"mu": 0.0, "sigma": 3.0}


def test_nuisance_set_rejects_colliding_event_columns_and_allows_empty() -> None:
    with pytest.raises(ValueError, match="share basis columns"):
        NuisanceSet(terms=(EVENTS, EventIndicators(events=("holiday",))))
    two_seasons = NuisanceSet(
        terms=(FourierSeasonality(period=7.0, order=1), FourierSeasonality(period=365.0, order=1))
    )
    assert len(set(two_seasons.column_names())) == 4
    empty = NuisanceSet(terms=())
    assert empty.column_names() == ()
    assert empty.parameters() == ()
    assert isinstance(empty.expr(), Const)
    assert dimension(empty.expr()) == D.outcome
    assert isinstance(SET.expr(), Add)


@pytest.mark.parametrize(
    "term", [SEASON, TREND, LinearTrend(), EVENTS, SET], ids=lambda t: type(t).__name__
)
def test_json_round_trip(term: Spec) -> None:
    back = load_spec(term.to_json())
    assert back == term
    assert type(back) is type(term)
    assert back.content_hash() == term.content_hash()
    assert isinstance(back, NuisanceTerm)
    expr = back.expr("p_")
    assert load_spec(expr.to_json()) == expr
