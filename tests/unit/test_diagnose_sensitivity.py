"""Cinelli–Hazlett benchmarking, bias-shifted posteriors, and tipping points."""

from __future__ import annotations

import numpy as np
import pytest

from axiom.core import Posterior, Unsupported
from axiom.diagnose.sensitivity import (
    Benchmark,
    BiasBounds,
    RobustnessValue,
    TippingPoint,
    benchmark,
    bias_bounds,
    partial_r2,
    robustness_value,
    shift_posterior,
    tipping_point,
)

# Cinelli & Hazlett (2020), Table 1: Darfur, "peacefactor ~ directlyharmed + ...".
DARFUR = dict(estimate=0.0973, se=0.0232, df=783)


def test_darfur_robustness_values() -> None:
    rv = robustness_value(**DARFUR, q=1.0, alpha=0.05)
    assert isinstance(rv, RobustnessValue)
    assert rv.rv == pytest.approx(0.139, abs=1e-3)
    assert rv.rv_alpha == pytest.approx(0.076, abs=1e-3)
    assert rv.r2_yd_x == pytest.approx(0.022, abs=1e-3)
    assert rv.rv_alpha <= rv.rv
    assert RobustnessValue.from_json(rv.to_json()) == rv


def test_darfur_published_bound_one_times_female() -> None:
    # Cinelli & Hazlett (2020), Table 2 / sensemakr ovb_bounds: the "1x female" bound is
    # r2_dz_x = 0.0092, r2_yz_dx = 0.1246 -> adjusted 0.0752 (se 0.0218, t 3.44).
    b = bias_bounds(**DARFUR, r2_yz_dx=0.1246, r2_dz_x=0.0092)
    assert b.adjusted_estimate == pytest.approx(0.0752, abs=2e-4)
    assert b.adjusted_se == pytest.approx(0.0218, abs=2e-4)
    assert b.adjusted_t == pytest.approx(3.44, abs=0.02)


def test_benchmark_formulas() -> None:
    x, y = 0.00916, 0.11
    b = benchmark(**DARFUR, covariate="female", r2_dxj_x=x, r2_yxj_dx=y, k_d=1.0, k_y=1.0)
    assert isinstance(b, Benchmark)
    assert b.r2_dz_x == pytest.approx(x / (1 - x), rel=1e-12)
    r2zxj = x**2 / ((1 - x) * (1 - x))
    eta = (1.0 + np.sqrt(r2zxj)) / np.sqrt(1 - r2zxj)
    assert b.r2_yz_dx == pytest.approx(eta**2 * y / (1 - y), rel=1e-12)
    assert b.bounds == bias_bounds(**DARFUR, r2_yz_dx=b.r2_yz_dx, r2_dz_x=b.r2_dz_x)
    stronger = benchmark(**DARFUR, covariate="female", r2_dxj_x=x, r2_yxj_dx=y, k_d=2.0, k_y=2.0)
    assert isinstance(stronger, Benchmark)
    assert stronger.r2_dz_x > b.r2_dz_x and stronger.r2_yz_dx > b.r2_yz_dx
    assert stronger.bounds.bias > b.bounds.bias
    assert Benchmark.from_json(b.to_json()) == b


def test_bias_formula_and_interval() -> None:
    b = bias_bounds(**DARFUR, r2_yz_dx=0.1, r2_dz_x=0.05, mass=0.95)
    assert isinstance(b, BiasBounds)
    expected = 0.0232 * np.sqrt(783) * np.sqrt(0.1 * 0.05 / 0.95)
    assert b.bias == pytest.approx(expected, rel=1e-12)
    assert b.adjusted_estimate == pytest.approx(0.0973 - expected, rel=1e-12)
    assert b.adjusted_interval.definition == "wald"
    assert b.adjusted_interval.mass == 0.95
    away = bias_bounds(**DARFUR, r2_yz_dx=0.1, r2_dz_x=0.05, reduce=False)
    assert away.adjusted_estimate == pytest.approx(0.0973 + expected, rel=1e-12)
    assert bias_bounds(**DARFUR, r2_yz_dx=0.0, r2_dz_x=0.3).bias == 0.0


def test_rv_solves_its_defining_equation() -> None:
    rv = robustness_value(**DARFUR)
    f2 = (rv.t / np.sqrt(783)) ** 2
    assert rv.rv**2 / (1.0 - rv.rv) == pytest.approx(f2, rel=1e-10)
    # at strength RV the confounder exactly removes the estimate
    b = bias_bounds(**DARFUR, r2_yz_dx=rv.rv, r2_dz_x=rv.rv)
    assert b.adjusted_estimate == pytest.approx(0.0, abs=1e-10)


def test_rv_alpha_clips_at_zero_for_insignificant_estimate() -> None:
    rv = robustness_value(0.01, 0.02, 100)
    assert rv.rv_alpha == 0.0
    assert rv.rv > 0.0


def test_partial_r2_and_validation() -> None:
    assert partial_r2(2.0, 96) == pytest.approx(4 / 100)
    with pytest.raises(ValueError):
        robustness_value(1.0, 0.0, 10)
    with pytest.raises(ValueError):
        bias_bounds(1.0, 0.1, 10, r2_yz_dx=1.0, r2_dz_x=0.1)
    with pytest.raises(ValueError):
        benchmark(1.0, 0.1, 10, covariate="x", r2_dxj_x=0.1, r2_yxj_dx=0.1, k_d=0.0)


def test_benchmark_returns_unsupported_when_bound_leaves_unit_interval() -> None:
    out = benchmark(**DARFUR, covariate="x", r2_dxj_x=0.4, r2_yxj_dx=0.1, k_d=3.0)
    assert isinstance(out, Unsupported)
    assert "r2_dz_x" in out.reason


def test_shift_posterior_gaussian_convolution() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(2.0, 0.5, size=20_000)
    s = shift_posterior(x, bias_mean=0.5, bias_sd=0.5, mass=0.9, seed=3)
    assert s.shifted.mean == pytest.approx(1.5, abs=0.02)
    assert s.shifted.sd == pytest.approx(np.sqrt(0.5**2 + 0.5**2), rel=0.03)
    assert s.interval.definition == "eti" and s.interval.mass == 0.9
    assert s.names() == {"estimate", "bias", "shifted"}
    assert s.posterior.provenance["seed"] == 3
    np.testing.assert_allclose(s.draws("shifted"), s.draws("estimate") - s.draws("bias"))


def test_shift_posterior_constant_and_draws() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    s = shift_posterior(x, bias_mean=1.0)
    np.testing.assert_allclose(np.sort(s.draws("shifted")[0]), x - 1.0)
    post = Posterior({"tau": x[None, :]})
    s2 = shift_posterior(post, name="tau", bias_draws=[0.5, 1.5], seed=0)
    assert s2.bias.mean in (0.5, 1.0, 1.5) or 0.5 <= s2.bias.mean <= 1.5
    with pytest.raises(ValueError):
        shift_posterior(post)
    with pytest.raises(ValueError):
        shift_posterior(x, bias_sd=-1.0)


def test_tipping_point_flips_at_smallest_bias() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(1.0, 0.1, size=4000)
    tp = tipping_point(x, 0.5, np.linspace(-1.0, 1.0, 41), certainty=0.5)
    assert isinstance(tp, TippingPoint)
    assert tp.decision_at_zero is True
    assert tp.flipped and tp.bias is not None
    assert tp.bias == pytest.approx(0.5, abs=0.05 + 1e-9)
    assert tp.interval is not None and tp.interval.mass == 0.9
    assert tp.probability is not None and tp.probability < 0.5
    assert tp.bias_grid == tuple(sorted(tp.bias_grid, key=abs))
    assert TippingPoint.from_json(tp.to_json()) == tp


def test_tipping_point_no_flip() -> None:
    x = np.full(100, 5.0) + np.arange(100) * 1e-3
    tp = tipping_point(x, 0.0, [0.1, 0.2])
    assert not tp.flipped and tp.bias is None and tp.interval is None
    with pytest.raises(ValueError):
        tipping_point(x, 0.0, [])
    with pytest.raises(ValueError):
        tipping_point(x, 0.0, [0.1], certainty=1.0)
