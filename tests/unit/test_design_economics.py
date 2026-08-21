"""design.economics and design.portfolio: discounting, opportunity cost, net value, ranking."""

from __future__ import annotations

import math

import numpy as np
import pytest

from axiom.core import Unsupported
from axiom.design.economics import (
    ExperimentValue,
    OpportunityCost,
    ValuePerOutcome,
    discount_weights,
    experiment_value,
    information_value_of,
    mid_horizon_factor,
    opportunity_cost,
)
from axiom.design.eig import decayed_sd
from axiom.design.evoi import DecisionSpec, evoi_gaussian
from axiom.design.portfolio import (
    Recommendation,
    StudySummary,
    TreatmentCandidate,
    prior_from_history,
    rank_treatments,
    recommend,
)

VPO = ValuePerOutcome(value=2.0, outcome_unit="outcome", numeraire="USD", source="finance table")
DECISION = DecisionSpec(threshold=1.0, value_per_outcome_unit=1000.0, numeraire="USD")


# -- discounting --------------------------------------------------------------------------


def test_discount_weights_shape_monotone_and_sum() -> None:
    w = discount_weights(5, 0.1)
    assert w.shape == (5,) and w[0] == 1.0
    assert np.all(np.diff(w) < 0)
    np.testing.assert_allclose(w, 1.1 ** -np.arange(5), rtol=1e-12)
    assert float(w.sum()) == pytest.approx((1 - 1.1**-5) / (1 - 1 / 1.1), rel=1e-12)
    np.testing.assert_array_equal(discount_weights(4, 0.0), np.ones(4))
    assert mid_horizon_factor(4, 0.0) == 1.0
    assert mid_horizon_factor(5, 0.1) == pytest.approx(float(w.mean()), rel=1e-12)
    assert mid_horizon_factor(5, 0.2) < mid_horizon_factor(5, 0.1) < 1.0
    assert mid_horizon_factor(10, 0.1) < mid_horizon_factor(5, 0.1)
    with pytest.raises(ValueError):
        discount_weights(0, 0.1)
    with pytest.raises(ValueError):
        discount_weights(3, -0.1)


# -- opportunity cost ---------------------------------------------------------------------


def test_opportunity_cost_sign_and_scaling() -> None:
    oc = opportunity_cost(0.2, 4, 100.0, 1.5, VPO, 0.0)
    assert isinstance(oc, OpportunityCost)
    # 0.2 * 100 * 4 = 80 dose units withheld; 1.5 outcome each; 2 USD per outcome.
    assert oc.dose_withheld == pytest.approx(80.0)
    assert oc.outcome_forgone == pytest.approx(120.0)
    assert oc.value == pytest.approx(240.0)
    assert oc.numeraire == "USD" and oc.outcome_unit == "outcome"
    assert oc.ratio_sd == 0.0 and oc.n_ratio_draws == 1
    # Linear in the holdout fraction and the dose.
    assert opportunity_cost(0.4, 4, 100.0, 1.5, VPO, 0.0).value == pytest.approx(2 * oc.value)
    assert opportunity_cost(0.2, 4, 200.0, 1.5, VPO, 0.0).value == pytest.approx(2 * oc.value)
    # Discounting shrinks it by exactly the mid-horizon factor.
    disc = opportunity_cost(0.2, 4, 100.0, 1.5, VPO, 0.1)
    assert disc.value == pytest.approx(oc.value * mid_horizon_factor(4, 0.1), rel=1e-12)
    assert disc.value < oc.value
    # A net-negative treatment (value below the dose's cost) makes the cost negative.
    neg = opportunity_cost(0.2, 4, 100.0, 0.25, VPO, 0.0, dose_cost_per_unit=1.0)
    assert neg.value == pytest.approx(80.0 * (0.25 * 2.0 - 1.0))
    assert neg.value < 0.0
    harmful = opportunity_cost(0.2, 4, 100.0, -0.5, VPO, 0.0)
    assert harmful.value < 0.0 and harmful.outcome_forgone < 0.0
    # Draws: the mean is used and the sd reported.
    draws = np.asarray([1.0, 2.0, 3.0])
    mc = opportunity_cost(0.2, 4, 100.0, draws, VPO, 0.0)
    assert mc.ratio_mean == pytest.approx(2.0) and mc.ratio_sd == pytest.approx(1.0)
    assert mc.n_ratio_draws == 3
    with pytest.raises(ValueError):
        opportunity_cost(1.0, 4, 100.0, 1.5, VPO, 0.0)
    with pytest.raises(ValueError):
        opportunity_cost(0.2, 4, -1.0, 1.5, VPO, 0.0)


def test_value_per_outcome_ledger_line() -> None:
    line = VPO.ledger_line()
    assert line.kind == "value_per_outcome"
    assert "finance table" in line.statement and "USD" in line.statement
    with pytest.raises(ValueError):
        ValuePerOutcome(value=0.0, outcome_unit="o", numeraire="USD", source="x")


# -- experiment value ---------------------------------------------------------------------


def test_experiment_value_arithmetic() -> None:
    oc = opportunity_cost(0.2, 4, 100.0, 1.5, VPO, 0.0)
    ev = experiment_value(1000.0, oc, 100.0)
    assert isinstance(ev, ExperimentValue)
    assert ev.net == pytest.approx(1000.0 - 240.0 - 100.0)
    assert ev.numeraire == "USD"
    bare = experiment_value(50.0, -20.0, 10.0, numeraire="EUR")
    assert bare.net == pytest.approx(60.0)
    with pytest.raises(ValueError, match="numeraire"):
        experiment_value(50.0, -20.0, 10.0)
    with pytest.raises(ValueError, match="numeraire"):
        experiment_value(50.0, oc, 10.0, numeraire="EUR")
    with pytest.raises(ValueError):
        experiment_value(-1.0, oc, 10.0)
    with pytest.raises(ValueError):
        ExperimentValue(
            information_value=1.0, opportunity_cost=0.0, fixed_cost=0.0, net=5.0, numeraire="USD"
        )


def test_information_value_is_evsi() -> None:
    iv = information_value_of(DECISION, 1.2, 0.5, 0.3)
    assert iv == pytest.approx(evoi_gaussian(DECISION, 1.2, 0.5, 0.3).evsi)
    assert 0.0 < iv
    assert (
        information_value_of(DECISION, 1.2, 0.5, 0.01)
        > iv
        > information_value_of(DECISION, 1.2, 0.5, 5.0)
    )


# -- portfolio ----------------------------------------------------------------------------


def test_prior_from_history_inverse_variance_with_decay() -> None:
    fresh = StudySummary(treatment="a", estimate=1.0, se=0.2, periods_ago=0.0, definition="wald")
    old = StudySummary(treatment="a", estimate=2.0, se=0.2, periods_ago=8.0, definition="wald")
    mean, sd = prior_from_history((fresh, old), half_life_periods=4.0)
    aged = decayed_sd(0.2, 8.0, 4.0)  # 0.2 * 2 ** (8 / 8) = 0.4
    assert aged == pytest.approx(0.4)
    p1, p2 = 1 / 0.2**2, 1 / aged**2
    assert mean == pytest.approx((p1 * 1.0 + p2 * 2.0) / (p1 + p2))
    assert sd == pytest.approx(1 / math.sqrt(p1 + p2))
    # The fresh study dominates: the mean is nearer 1 than 2.
    assert mean < 1.5
    # With no decay (age 0) the same two studies combine symmetrically.
    m0, s0 = prior_from_history((fresh, old.model_copy(update={"periods_ago": 0.0})), 4.0)
    assert m0 == pytest.approx(1.5) and s0 == pytest.approx(0.2 / math.sqrt(2))
    with pytest.raises(ValueError):
        prior_from_history((), 4.0)
    with pytest.raises(ValueError, match="several treatments"):
        prior_from_history((fresh, old.model_copy(update={"treatment": "b"})), 4.0)


def _cand(name: str, **kw: float) -> TreatmentCandidate:
    base = dict(
        prior_mean=1.2, prior_sd=0.5, experiment_se=0.3, opportunity_cost=0.0, fixed_cost=0.0
    )
    base.update(kw)
    return TreatmentCandidate(name=name, decision=DECISION, **base)


def test_rank_treatments_by_net_then_eig_deterministic() -> None:
    a = _cand("a", experiment_se=0.3)
    b = _cand("b", experiment_se=0.1)  # more precise: more EIG and more EVSI
    c = _cand("c", experiment_se=0.3, fixed_cost=1e6)  # hugely negative net
    ranked = rank_treatments((c, a, b))
    assert [p.treatment for p in ranked] == ["b", "a", "c"]
    assert [p.rank for p in ranked] == [1, 2, 3]
    assert ranked[0].eig > ranked[1].eig
    assert ranked[2].net_value < 0.0 < ranked[1].net_value
    assert ranked[0].numeraire == "USD"
    # Same net value: the EIG breaks the tie, then the name.
    d = _cand("d")
    e = _cand("e")
    tie = rank_treatments((e, d))
    assert [p.treatment for p in tie] == ["d", "e"]
    assert rank_treatments((d, e)) == tie
    with pytest.raises(ValueError, match="distinct"):
        rank_treatments((d, d))


def test_recommend_greedy_knapsack_with_budget() -> None:
    cheap = _cand("cheap", experiment_se=0.3, fixed_cost=10.0)
    precise = _cand("precise", experiment_se=0.05, fixed_cost=100.0)
    losing = _cand("losing", experiment_se=0.3, fixed_cost=1e6)
    net_cheap = rank_treatments((cheap,))[0].net_value
    net_precise = rank_treatments((precise,))[0].net_value
    assert net_cheap / 10.0 > net_precise / 100.0  # cheap wins per unit cost
    unbounded = recommend((losing, precise, cheap), budget=None)
    assert isinstance(unbounded, Recommendation)
    assert set(unbounded.selected) == {"cheap", "precise"}
    assert unbounded.selected[0] == "cheap"
    assert unbounded.total_cost == pytest.approx(110.0)
    assert unbounded.total_net_value == pytest.approx(net_cheap + net_precise)
    tight = recommend((losing, precise, cheap), budget=50.0)
    assert isinstance(tight, Recommendation)
    assert tight.selected == ("cheap",)
    assert "precise" in tight.detail["skipped_over_budget"]
    nothing = recommend((losing,), budget=None)
    assert isinstance(nothing, Unsupported)
    assert "positive_net_value" in nothing.missing
    with pytest.raises(ValueError):
        recommend((cheap,), budget=-1.0)
