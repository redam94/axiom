"""meta.influence: leave-one-out, Egger's test, funnel contours, forest and Baujat data."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy import stats as st

from axiom.core import Interval
from axiom.meta.influence import (
    EggerTest,
    ForestData,
    FunnelData,
    LeaveOneOut,
    baujat,
    egger,
    forest_data,
    funnel_data,
    leave_one_out,
)

Y3 = np.array([0.2, 0.5, 0.8])
SE3 = np.array([0.1, 0.2, 0.4])


def _classical() -> Any:
    return pytest.importorskip("axiom.meta.classical")


# -- leave-one-out -----------------------------------------------------------------------


def test_leave_one_out_fixed_effect_hand_case() -> None:
    _classical()
    loo = leave_one_out(Y3, SE3, method="fe", mass=0.95)
    assert isinstance(loo, LeaveOneOut)
    w = 1.0 / SE3**2
    full = float((w * Y3).sum() / w.sum())
    full_se = math.sqrt(1.0 / w.sum())
    assert loo.k == 3
    assert loo.full_estimate == pytest.approx(full, rel=1e-12)
    assert loo.full_se == pytest.approx(full_se, rel=1e-12)
    for i in range(3):
        keep = np.arange(3) != i
        e = float((w[keep] * Y3[keep]).sum() / w[keep].sum())
        s = math.sqrt(1.0 / w[keep].sum())
        assert loo.estimates[i] == pytest.approx(e, rel=1e-12)
        assert loo.ses[i] == pytest.approx(s, rel=1e-12)
        assert loo.influence[i] == pytest.approx((full - e) / full_se, rel=1e-12)
        assert loo.intervals[i].definition == "wald" and loo.intervals[i].mass == 0.95
        assert loo.intervals[i].lower == pytest.approx(e - 1.959963984540054 * s, rel=1e-9)
    assert all(t == 0.0 for t in loo.tau2s)


def test_leave_one_out_random_effects_matches_direct_repool() -> None:
    cl = _classical()
    y = np.array([0.1, 0.4, 0.9, 0.3, 0.6])
    se = np.array([0.1, 0.15, 0.2, 0.12, 0.3])
    loo = leave_one_out(y, se, method="dl")
    assert loo.method == "dl"
    for i in range(5):
        keep = np.arange(5) != i
        p = cl.random_effects(y[keep], se[keep], tau_method="dl", mass=0.95)
        assert loo.estimates[i] == pytest.approx(float(p.estimate), rel=1e-12)
        assert loo.tau2s[i] == pytest.approx(float(p.tau2), abs=1e-12)


def test_leave_one_out_needs_three_studies() -> None:
    _classical()
    with pytest.raises(ValueError):
        leave_one_out([0.1, 0.2], [0.1, 0.1])


# -- Egger ---------------------------------------------------------------------------------


def test_egger_matches_hand_weighted_regression() -> None:
    y = np.array([0.1, 0.3, 0.6, 0.2, 0.9, 0.4])
    se = np.array([0.05, 0.1, 0.3, 0.08, 0.5, 0.15])
    res = egger(y, se)
    assert isinstance(res, EggerTest)
    # Weighted (1/se^2) regression of y on se: intercept of the z ~ precision form
    # equals the slope on se here, and vice versa.
    w = 1.0 / se**2
    x = np.column_stack([np.ones(6), se])
    wx = x * w[:, None]
    beta = np.linalg.solve(x.T @ wx, wx.T @ y)
    resid = y - x @ beta
    s2 = float((w * resid**2).sum()) / 4
    cov = s2 * np.linalg.inv(x.T @ wx)
    assert res.intercept == pytest.approx(beta[1], rel=1e-10)
    assert res.slope == pytest.approx(beta[0], rel=1e-10)
    assert res.se == pytest.approx(math.sqrt(cov[1, 1]), rel=1e-10)
    t = beta[1] / math.sqrt(cov[1, 1])
    assert res.t == pytest.approx(t, rel=1e-10)
    assert res.p == pytest.approx(2 * st.t.sf(abs(t), 4), rel=1e-10)
    assert res.df == 4 and res.k == 6
    crit = st.t.ppf(0.975, 4)
    assert res.interval.lower == pytest.approx(res.intercept - crit * res.se, rel=1e-10)


def test_egger_symmetric_funnel_large_p_asymmetric_small_p() -> None:
    rng = np.random.default_rng(7)
    k = 60
    se = np.exp(rng.uniform(math.log(0.05), math.log(0.5), k))
    # symmetric: y ~ N(mu, se^2)
    y_sym = 0.3 + se * rng.standard_normal(k)
    assert egger(y_sym, se).p > 0.05
    # asymmetric: small studies inflated in proportion to their se
    y_asym = 0.3 + 3.0 * se + se * rng.standard_normal(k)
    assert egger(y_asym, se).p < 1e-6


# -- funnel -------------------------------------------------------------------------------


def test_funnel_contours_closed_form() -> None:
    fd = funnel_data(Y3, SE3, 0.5, masses=(0.9, 0.95, 0.99), n_grid=11)
    assert isinstance(fd, FunnelData)
    assert fd.k == 3 and fd.pooled == 0.5
    assert fd.precision == pytest.approx(tuple(1.0 / SE3))
    assert len(fd.contours) == 3
    for c in fd.contours:
        z = st.norm.ppf((1 + c.mass) / 2)
        assert c.se[0] == 0.0 and c.se[-1] == pytest.approx(0.4)
        for s, lo, hi in zip(c.se, c.lower, c.upper, strict=True):
            assert lo == pytest.approx(0.5 - z * s, abs=1e-12)
            assert hi == pytest.approx(0.5 + z * s, abs=1e-12)
    assert fd.contours[2].lower[-1] < fd.contours[0].lower[-1]


def test_funnel_accepts_pooled_object() -> None:
    class P:
        estimate = 0.25
        se = 0.1
        tau2 = 0.0

    fd = funnel_data(Y3, SE3, P())
    assert fd.pooled == 0.25


# -- forest -------------------------------------------------------------------------------


def test_forest_data_shapes_from_arrays_and_float() -> None:
    fd = forest_data(Y3, SE3, 0.5, labels=["a", "b", "c"], mass=0.9)
    assert isinstance(fd, ForestData)
    assert [r.label for r in fd.rows] == ["a", "b", "c"]
    iv = fd.rows[1].interval
    assert isinstance(iv, Interval) and iv.definition == "wald" and iv.mass == 0.9
    assert iv.lower == pytest.approx(0.5 - 1.6448536269514722 * 0.2, rel=1e-12)
    assert iv.upper == pytest.approx(0.5 + 1.6448536269514722 * 0.2, rel=1e-12)
    assert fd.pooled_estimate == 0.5
    assert fd.pooled_se is None and fd.pooled_interval is None and fd.prediction_interval is None
    assert all(r.weight is None for r in fd.rows)


def test_forest_data_with_pooled_estimate_has_prediction_interval() -> None:
    cl = _classical()
    y = np.array([0.1, 0.4, 0.9, 0.3, 0.6])
    se = np.array([0.1, 0.15, 0.2, 0.12, 0.3])
    pooled = cl.random_effects(y, se, tau_method="dl", mass=0.95)
    fd = forest_data(y, se, pooled)
    assert len(fd.rows) == 5
    assert fd.pooled_interval is not None and fd.pooled_interval.definition == "wald"
    assert fd.prediction_interval is not None
    assert fd.prediction_interval.width >= fd.pooled_interval.width
    assert fd.tau2 == pytest.approx(float(pooled.tau2))
    weights = [r.weight for r in fd.rows]
    assert all(w is not None for w in weights)
    assert sum(w for w in weights if w is not None) == pytest.approx(1.0)


def test_forest_data_from_corpus_like() -> None:
    class R:
        def __init__(self, s: str, e: float, se: float) -> None:
            self.study, self.estimate, self.se = s, e, se

    class C:
        records = (R("s1", 0.2, 0.1), R("s2", 0.5, 0.2), R("s3", 0.8, 0.4))

    fd = forest_data(C(), None, 0.5)
    assert [r.label for r in fd.rows] == ["s1", "s2", "s3"]


# -- Baujat -------------------------------------------------------------------------------


def test_baujat_hand_case() -> None:
    _classical()
    b = baujat(Y3, SE3)
    w = 1.0 / SE3**2
    mu = float((w * Y3).sum() / w.sum())
    assert b.q_contribution == pytest.approx(tuple(w * (Y3 - mu) ** 2), rel=1e-12)
    for i in range(3):
        keep = np.arange(3) != i
        e = float((w[keep] * Y3[keep]).sum() / w[keep].sum())
        v = 1.0 / w[keep].sum()
        assert b.influence[i] == pytest.approx((mu - e) ** 2 / v, rel=1e-12)


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        egger([0.1, 0.2, 0.3], [0.1, -0.1, 0.1])
    with pytest.raises(ValueError):
        funnel_data([0.1], [0.1, 0.2], 0.0)
    with pytest.raises(ValueError):
        egger(Y3, SE3, mass=1.5)
