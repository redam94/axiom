"""Unit tests for meta.classical against hand computations and closed forms."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from axiom.core import Spec
from axiom.meta.classical import (
    PooledEstimate,
    fixed_effect,
    heterogeneity,
    prediction_interval,
    random_effects,
    reml_log_likelihood,
    tau_dersimonian_laird,
    tau_paule_mandel,
    tau_reml,
)

Y = np.array([0.10, 0.30, 0.55, -0.05, 0.40, 0.20])
SE = np.array([0.10, 0.15, 0.12, 0.20, 0.10, 0.25])


def test_fixed_effect_hand_computation() -> None:
    w = 1 / SE**2
    mu = np.sum(w * Y) / np.sum(w)
    se = np.sqrt(1 / np.sum(w))
    fe = fixed_effect(Y, SE, mass=0.95)
    assert fe.estimate == pytest.approx(mu, rel=1e-12)
    assert fe.se == pytest.approx(se, rel=1e-12)
    assert fe.interval.definition == "wald" and fe.interval.mass == 0.95
    assert fe.interval.upper == pytest.approx(mu + 1.959963984540054 * se, rel=1e-9)
    assert sum(fe.weights) == pytest.approx(1.0) and fe.model == "fixed"
    assert Spec.from_json(fe.to_json()) == fe
    assert fixed_effect([0.2], [0.1]).heterogeneity.df == 0


def test_heterogeneity_closed_forms() -> None:
    w = 1 / SE**2
    mu = np.sum(w * Y) / np.sum(w)
    q = float(np.sum(w * (Y - mu) ** 2))
    het = heterogeneity(Y, SE)
    assert het.q == pytest.approx(q, rel=1e-12) and het.df == 5
    assert het.p_value == pytest.approx(stats.chi2.sf(q, 5), rel=1e-12)
    assert het.i2 == pytest.approx(max(0, (q - 5) / q), rel=1e-12)
    assert het.h2 == pytest.approx(q / 5, rel=1e-12)


def test_dersimonian_laird_hand_computation() -> None:
    w = 1 / SE**2
    mu = np.sum(w * Y) / np.sum(w)
    q = np.sum(w * (Y - mu) ** 2)
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    expected = max(0.0, (q - 5) / c)
    tau = tau_dersimonian_laird(Y, SE)
    assert expected > 0 and tau.tau2 == pytest.approx(expected, rel=1e-12)
    assert tau.iterations == 0 and tau.converged and not tau.truncated
    # a homogeneous corpus truncates to zero
    hom = tau_dersimonian_laird([0.2, 0.21, 0.19], [0.3, 0.3, 0.3])
    assert hom.tau2 == 0.0 and hom.truncated


def test_pm_and_reml_zero_on_homogeneous_corpus() -> None:
    y = [0.2, 0.21, 0.19, 0.2]
    se = [0.3, 0.3, 0.3, 0.3]
    for f in (tau_paule_mandel, tau_reml, tau_dersimonian_laird):
        t = f(y, se)
        assert t.tau2 == 0.0 and t.truncated and t.converged


def test_pm_and_reml_on_heterogeneous_corpus() -> None:
    pm = tau_paule_mandel(Y, SE)
    reml = tau_reml(Y, SE)
    dl = tau_dersimonian_laird(Y, SE)
    assert pm.converged and reml.converged and pm.tau2 > 0 and reml.tau2 > 0
    # PM solves Q(tau²) = k − 1 exactly
    w = 1 / (SE**2 + pm.tau2)
    mu = np.sum(w * Y) / np.sum(w)
    assert np.sum(w * (Y - mu) ** 2) == pytest.approx(5.0, abs=1e-8)
    # REML is a maximum of the restricted log-likelihood
    ll = reml_log_likelihood(Y, SE, reml.tau2)
    for h in (1e-3, 1e-2):
        assert ll >= reml_log_likelihood(Y, SE, reml.tau2 + h)
        assert ll >= reml_log_likelihood(Y, SE, max(reml.tau2 - h, 0.0))
    # the three agree to the order expected at k = 6 and share the same sign of the story
    assert pm.tau2 == pytest.approx(reml.tau2, rel=0.5)
    assert dl.tau2 == pytest.approx(pm.tau2, rel=0.5)
    assert pm.iterations > 0 and reml.iterations > 0
    assert Spec.from_json(pm.to_json()) == pm


def test_random_effects_weights_and_interval() -> None:
    for method in ("dl", "pm", "reml"):
        re = random_effects(Y, SE, tau_method=method)  # type: ignore[arg-type]
        w = 1 / (SE**2 + re.tau2)
        assert re.estimate == pytest.approx(np.sum(w * Y) / np.sum(w), rel=1e-12)
        assert re.se == pytest.approx(np.sqrt(1 / np.sum(w)), rel=1e-12)
        assert np.allclose(re.weights, w / np.sum(w))
        assert re.tau_method == method and re.model == "random"
        assert re.tau_estimate is not None and re.tau_estimate.tau2 == re.tau2
        assert re.interval.definition == "wald" and re.k == 6
        assert Spec.from_json(re.to_json()) == re
    with pytest.raises(ValueError, match="tau_method"):
        random_effects(Y, SE, tau_method="ml")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 2"):
        random_effects([0.1], [0.1])


def test_knapp_hartung_widens() -> None:
    # k = 5: t_4 = 2.776 against z = 1.96 widens the interval even when the scale is
    # just below one (DL leaves it near one; PM pins it to exactly one)
    y = [0.1, 0.9, -0.3, 0.6, 0.2]
    se = [0.3, 0.1, 0.3, 0.2, 0.1]
    plain = random_effects(y, se, tau_method="dl")
    kh = random_effects(y, se, tau_method="dl", knapp_hartung=True)
    assert kh.estimate == plain.estimate and kh.tau2 == plain.tau2
    scale = float(kh.detail["knapp_hartung_scale"])
    assert 0.5 < scale < 1.5
    pm = random_effects(y, se, tau_method="pm", knapp_hartung=True)
    assert float(pm.detail["knapp_hartung_scale"]) == pytest.approx(1.0, abs=1e-8)
    assert kh.se == pytest.approx(plain.se * np.sqrt(scale), rel=1e-12)
    t = stats.t.ppf(0.975, 4)
    assert kh.interval.width == pytest.approx(2 * t * kh.se, rel=1e-12)
    assert kh.interval.width > plain.interval.width
    assert kh.detail["quantile"] == "t_4" and kh.knapp_hartung


def test_prediction_interval_uses_t_with_k_minus_2_df() -> None:
    re = random_effects(Y, SE, tau_method="reml", mass=0.95)
    pi = prediction_interval(re)
    t = stats.t.ppf(0.975, 4)
    half = t * np.sqrt(re.tau2 + re.se**2)
    assert pi.lower == pytest.approx(re.estimate - half, rel=1e-12)
    assert pi.upper == pytest.approx(re.estimate + half, rel=1e-12)
    assert pi.mass == 0.95 and pi.width > re.interval.width
    pi90 = prediction_interval(re, 0.9)
    assert pi90.mass == 0.9 and pi90.width < pi.width
    with pytest.raises(ValueError, match="k ≥ 3"):
        prediction_interval(random_effects(Y[:2], SE[:2]))


def test_pooled_estimate_invariants() -> None:
    fe = fixed_effect(Y, SE)
    with pytest.raises(ValueError, match="tau_method='none'"):
        PooledEstimate(**{**fe.to_dict(), "tau2": 0.1})
    with pytest.raises(ValueError, match="weights"):
        PooledEstimate(**{**fe.to_dict(), "weights": (1.0,)})
