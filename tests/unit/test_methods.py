"""Recovery, null, Unsupported, and A/A calibration for the six experiment methods."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from axiom.core import Interval, Unsupported, clopper_pearson
from axiom.design.methods import (
    METHODS,
    MethodEstimate,
    PanelArrays,
    bartlett_bandwidth,
    estimate,
    estimate_cluster_based_regression,
    estimate_difference_in_differences,
    estimate_ghost,
    estimate_switchback,
    estimate_synthetic_control,
    estimate_time_based_regression,
    panel_arrays,
    simplex_weights,
    t_critical,
    wald_t,
)

N_UNITS, N_PRE, N_POST = 12, 10, 6
TREATED = (0, 1, 2, 3)
PRE, POST = slice(0, N_PRE), slice(N_PRE, N_PRE + N_POST)
PANEL_METHODS = (
    "difference_in_differences",
    "synthetic_control",
    "time_based_regression",
    "cluster_based_regression",
)


def make_panel(
    seed: int, effect: float, *, n_units: int = N_UNITS, noise: float = 0.5
) -> PanelArrays:
    """Units with their own levels, a shared time trend, white noise, and an additive effect."""
    rng = np.random.default_rng(seed)
    level = rng.normal(10.0, 1.0, size=(n_units, 1))
    trend = np.linspace(0.0, 2.0, N_PRE + N_POST)[None, :]
    y = level + trend + rng.normal(0.0, noise, size=(n_units, N_PRE + N_POST))
    y[list(TREATED), POST] += effect
    return panel_arrays(y, TREATED, PRE, POST)


def make_switchback(
    seed: int, effect: float, *, n_units: int = 6, n_periods: int = 40
) -> PanelArrays:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, size=(n_units, n_periods)).astype(float)
    level = rng.normal(5.0, 1.0, size=(n_units, 1))
    y = level + effect * a + rng.normal(0.0, 0.5, size=(n_units, n_periods))
    return panel_arrays(y, (), slice(0, 0), slice(0, n_periods), assignment=a)


def make_ghost(seed: int, effect: float, *, n_units: int = 40) -> PanelArrays:
    rng = np.random.default_rng(seed)
    treated = tuple(range(n_units // 2))
    exposed = rng.integers(0, 2, size=n_units).astype(float)
    y = rng.normal(3.0, 1.0, size=(n_units, N_PRE + N_POST))
    for i in treated:
        if exposed[i] == 1.0:
            y[i, POST] += effect
    return panel_arrays(y, treated, PRE, POST, exposed=exposed)


def run(method: str, arrays: PanelArrays) -> MethodEstimate:
    out = estimate(method, arrays)
    assert isinstance(out, MethodEstimate), out
    return out


# -- recovery and shape ----------------------------------------------------------------


@pytest.mark.parametrize("method", PANEL_METHODS)
def test_panel_methods_recover_additive_effect(method: str) -> None:
    est = run(method, make_panel(11, 2.0))
    assert abs(est.effect - 2.0) < 3.0 * est.se
    assert est.method == method
    assert isinstance(est.interval, Interval) and est.interval.definition == "wald"
    assert est.interval.mass == 0.95
    assert est.interval.contains(est.effect)
    assert est.n_treated == len(TREATED) and est.n_control == N_UNITS - len(TREATED)
    assert est.n_pre == N_PRE and est.n_post == N_POST
    assert est.detail["df"] > 0


@pytest.mark.parametrize("method", PANEL_METHODS)
def test_panel_methods_null_is_near_zero(method: str) -> None:
    est = run(method, make_panel(12, 0.0))
    assert abs(est.effect) < 3.0 * est.se


def test_switchback_recovers_and_records_bandwidth() -> None:
    est = run("switchback", make_switchback(3, 1.0))
    assert abs(est.effect - 1.0) < 3.0 * est.se
    assert est.interval.definition == "wald"
    assert est.detail["bandwidth"] == bartlett_bandwidth(40)
    assert est.se_method == "hac_bartlett_within_unit"
    assert est.detail["df"] > 0
    null = run("switchback", make_switchback(4, 0.0))
    assert abs(null.effect) < 3.0 * null.se


def test_ghost_recovers_and_records_cells() -> None:
    est = run("ghost", make_ghost(5, 1.5))
    assert abs(est.effect - 1.5) < 3.0 * est.se
    assert est.se_method == "two_sample_welch"
    assert est.detail["n_treated_exposed"] >= 2 and est.detail["n_control_exposed"] >= 2
    assert est.detail["df"] > 0
    null = run("ghost", make_ghost(6, 0.0))
    assert abs(null.effect) < 3.0 * null.se


def test_mass_is_honoured_and_interval_widens() -> None:
    arrays = make_panel(7, 1.0)
    narrow = estimate("difference_in_differences", arrays, mass=0.5)
    wide = estimate("difference_in_differences", arrays, mass=0.99)
    assert isinstance(narrow, MethodEstimate) and isinstance(wide, MethodEstimate)
    assert narrow.interval.mass == 0.5 and wide.interval.mass == 0.99
    assert wide.interval.width > narrow.interval.width


def test_keyword_conveniences_match_dispatch() -> None:
    arrays = make_panel(8, 1.0)
    y = arrays.outcome
    pairs: list[tuple[str, MethodEstimate | Unsupported]] = [
        ("difference_in_differences", estimate_difference_in_differences(y, TREATED, PRE, POST)),
        ("synthetic_control", estimate_synthetic_control(y, TREATED, PRE, POST)),
        ("time_based_regression", estimate_time_based_regression(y, TREATED, PRE, POST)),
        ("cluster_based_regression", estimate_cluster_based_regression(y, TREATED, PRE, POST)),
    ]
    for name, direct in pairs:
        via = estimate(name, arrays)
        assert isinstance(direct, MethodEstimate) and isinstance(via, MethodEstimate)
        assert direct.effect == pytest.approx(via.effect) and direct.se == pytest.approx(via.se)
    sb = make_switchback(1, 0.5)
    assert sb.assignment is not None
    direct_sb = estimate_switchback(sb.outcome, sb.assignment)
    assert isinstance(direct_sb, MethodEstimate)
    assert direct_sb.effect == pytest.approx(run("switchback", sb).effect)
    gh = make_ghost(2, 0.5)
    assert gh.exposed is not None
    direct_gh = estimate_ghost(gh.outcome, gh.treated, POST, gh.exposed, pre=PRE)
    assert isinstance(direct_gh, MethodEstimate)
    assert direct_gh.effect == pytest.approx(run("ghost", gh).effect)
    with pytest.raises(ValueError):
        estimate_difference_in_differences(y, TREATED, PRE, POST, mass=1.0)


# -- method-specific numbers ----------------------------------------------------------------


def test_student_t_critical_value_is_used_by_every_method() -> None:
    # Every method's SE is an estimated variance with finite df: the interval half-width
    # is t_{df} * se, wider than z * se, and the critical value is on the record.
    from scipy import stats

    for m in PANEL_METHODS:
        est = run(m, make_panel(41, 1.0))
        df = est.detail["df"]
        c = est.detail["critical_value"]
        assert est.critical == "student_t"
        assert c == pytest.approx(stats.t.ppf(0.975, df)) and c > stats.norm.ppf(0.975)
        assert est.interval.width == pytest.approx(2.0 * c * est.se)
    for est in (run("switchback", make_switchback(42, 1.0)), run("ghost", make_ghost(43, 1.0))):
        assert est.critical == "student_t"
        assert est.interval.width == pytest.approx(2.0 * est.detail["critical_value"] * est.se)
    assert t_critical(0.95, 10) == pytest.approx(2.2281388519649385)
    assert t_critical(0.95, 1e9) == pytest.approx(1.959964, rel=1e-5)
    iv = wald_t(1.0, 0.5, 0.9, 4)
    assert iv.definition == "wald" and iv.mass == 0.9
    assert iv.lower == pytest.approx(1.0 - 0.5 * t_critical(0.9, 4))
    with pytest.raises(ValueError):
        t_critical(0.95, 0)
    with pytest.raises(ValueError):
        t_critical(1.0, 4)
    with pytest.raises(ValueError):
        wald_t(0.0, -1.0, 0.9, 4)


def test_switchback_dof_factor() -> None:
    est = run("switchback", make_switchback(44, 1.0, n_units=6, n_periods=40))
    n, t = 6, 40
    assert est.detail["df"] == n * t - n - 1
    assert est.detail["dof_factor"] == pytest.approx(n * t / (n * t - n - 1))


def test_did_matches_hand_computation() -> None:
    arrays = make_panel(21, 1.0)
    y = arrays.outcome
    change = y[:, POST].mean(axis=1) - y[:, PRE].mean(axis=1)
    t, c = change[list(TREATED)], change[list(arrays.control)]
    est = run("difference_in_differences", arrays)
    assert est.effect == pytest.approx(t.mean() - c.mean())
    assert est.se == pytest.approx(np.sqrt(t.var(ddof=1) / t.size + c.var(ddof=1) / c.size))
    assert est.detail["df"] == N_UNITS - 2 and est.detail["pooled"] == 0.0


def test_did_single_treated_unit_pools_control_variance() -> None:
    y = make_panel(22, 1.0).outcome
    est = estimate("difference_in_differences", panel_arrays(y, (0,), PRE, POST))
    assert isinstance(est, MethodEstimate)
    assert est.detail["pooled"] == 1.0 and est.n_treated == 1


def test_cbr_matches_lstsq() -> None:
    arrays = make_panel(23, 1.0)
    y = arrays.outcome
    d = np.zeros(N_UNITS)
    d[list(TREATED)] = 1.0
    x = np.column_stack([np.ones(N_UNITS), d, y[:, PRE].mean(axis=1)])
    beta = np.linalg.lstsq(x, y[:, POST].mean(axis=1), rcond=None)[0]
    est = run("cluster_based_regression", arrays)
    assert est.effect == pytest.approx(beta[1])
    assert est.detail["df"] == N_UNITS - 3 and est.se_method == "ols_classical"


def test_tbr_cumulative_in_detail_and_exact_relationship() -> None:
    est = run("time_based_regression", make_panel(24, 1.0))
    assert est.detail["cumulative_per_unit"] == pytest.approx(est.effect * N_POST)
    assert est.detail["cumulative_total"] == pytest.approx(est.effect * N_POST * len(TREATED))
    assert est.detail["df"] == N_PRE - 2
    # Control aggregate exactly predicts the treated aggregate: zero residual variance.
    rng = np.random.default_rng(0)
    x = rng.normal(size=N_PRE + N_POST)
    y = np.vstack([2.0 + 3.0 * x, 2.0 + 3.0 * x, x, x])
    out = estimate("time_based_regression", panel_arrays(y, (0, 1), PRE, POST))
    assert isinstance(out, Unsupported)


def test_synthetic_control_weights_and_detail() -> None:
    rng = np.random.default_rng(25)
    donors = rng.normal(size=(4, 8))
    w_true = np.array([0.5, 0.3, 0.2, 0.0])
    w = simplex_weights(donors.T @ w_true, donors)
    assert w.min() >= 0 and w.sum() == pytest.approx(1.0)
    assert np.allclose(w, w_true, atol=1e-4)
    assert np.array_equal(simplex_weights(np.ones(3), np.ones((1, 3))), np.ones(1))
    est = run("synthetic_control", make_panel(26, 1.0))
    assert est.se_method == "placebo_in_space_weight_corrected"
    assert est.detail["n_placebos"] == N_UNITS - len(TREATED)
    n_donors = N_UNITS - len(TREATED)
    # The averaged simplex weights have squared norm between 1/n_donors (uniform) and 1.
    assert 1.0 / n_donors - 1e-9 <= est.detail["weight_overlap"] <= 1.0 + 1e-9
    assert est.se == pytest.approx(
        np.sqrt(
            est.detail["unit_noise_variance"] * (1 / len(TREATED) + est.detail["weight_overlap"])
        )
    )
    assert 0.0 <= est.detail["placebo_rank"] <= 1.0
    assert est.detail["pre_rmspe"] >= 0 and np.isinf(est.detail["rmspe_ratio_max"])
    filtered = estimate_synthetic_control(
        make_panel(26, 1.0).outcome, TREATED, PRE, POST, rmspe_ratio_max=50.0
    )
    assert isinstance(filtered, MethodEstimate)
    assert filtered.detail["rmspe_ratio_max"] == 50.0
    assert filtered.effect == pytest.approx(est.effect)
    with pytest.raises(ValueError):
        estimate_synthetic_control(
            make_panel(26, 1.0).outcome, TREATED, PRE, POST, rmspe_ratio_max=0
        )


def test_switchback_effect_is_within_contrast() -> None:
    arrays = make_switchback(27, 1.0)
    assert arrays.assignment is not None
    y, a = arrays.outcome, arrays.assignment
    a_dm = a - a.mean(axis=1, keepdims=True)
    y_dm = y - y.mean(axis=1, keepdims=True)
    est = run("switchback", arrays)
    assert est.effect == pytest.approx((a_dm * y_dm).sum() / (a_dm**2).sum())
    assert bartlett_bandwidth(100) == 4 and bartlett_bandwidth(2) == 1
    with pytest.raises(ValueError):
        bartlett_bandwidth(0)


def test_ghost_matches_welch() -> None:
    arrays = make_ghost(28, 1.0)
    assert arrays.exposed is not None
    post_mean = arrays.outcome[:, POST].mean(axis=1)
    tmask = np.zeros(arrays.n_units, dtype=bool)
    tmask[list(arrays.treated)] = True
    ex = arrays.exposed > 0.5
    t, c = post_mean[tmask & ex], post_mean[~tmask & ex]
    est = run("ghost", arrays)
    assert est.effect == pytest.approx(t.mean() - c.mean())
    assert est.se == pytest.approx(np.sqrt(t.var(ddof=1) / t.size + c.var(ddof=1) / c.size))


# -- Unsupported paths -----------------------------------------------------------------


def test_unsupported_missing_inputs() -> None:
    plain = make_panel(31, 1.0)
    sb = estimate("switchback", plain)
    assert isinstance(sb, Unsupported) and "assignment" in sb.missing
    gh = estimate("ghost", plain)
    assert isinstance(gh, Unsupported) and "exposed" in gh.missing
    # Registry-level requirement checks.
    no_pre = panel_arrays(plain.outcome, TREATED, slice(0, 0), POST)
    for m in PANEL_METHODS:
        out = estimate(m, no_pre)
        assert isinstance(out, Unsupported) and "pre_period" in out.missing
    all_treated = panel_arrays(plain.outcome, tuple(range(N_UNITS)), PRE, POST)
    out = estimate("difference_in_differences", all_treated)
    assert isinstance(out, Unsupported) and "controls" in out.missing


def test_unsupported_degenerate_data() -> None:
    y = np.ones((N_UNITS, N_PRE + N_POST))
    constant = panel_arrays(y, TREATED, PRE, POST)
    for m in PANEL_METHODS:
        assert isinstance(estimate(m, constant), Unsupported), m
    # Synthetic control: too few donors / pre periods.
    good = make_panel(32, 1.0).outcome
    assert isinstance(
        estimate("synthetic_control", panel_arrays(good[:5], TREATED, PRE, POST)), Unsupported
    )
    assert isinstance(
        estimate("synthetic_control", panel_arrays(good, TREATED, slice(0, 1), POST)), Unsupported
    )
    # DiD: one control unit has no variance.
    assert isinstance(
        estimate("difference_in_differences", panel_arrays(good[:5], TREATED, PRE, POST)),
        Unsupported,
    )
    # CBR: fewer than four units.
    assert isinstance(
        estimate("cluster_based_regression", panel_arrays(good[:3], (0,), PRE, POST)), Unsupported
    )
    # TBR: fewer than three pre periods.
    assert isinstance(
        estimate("time_based_regression", panel_arrays(good, TREATED, slice(0, 2), POST)),
        Unsupported,
    )
    # Switchback: a schedule that never switches.
    sb = make_switchback(33, 1.0)
    never = panel_arrays(
        sb.outcome, (), slice(0, 0), slice(0, 40), assignment=np.ones_like(sb.outcome)
    )
    out = estimate("switchback", never)
    assert isinstance(out, Unsupported) and "switch" in out.reason
    # Ghost: an empty cell.
    gh = make_ghost(34, 1.0)
    none_exposed = panel_arrays(gh.outcome, gh.treated, PRE, POST, exposed=np.zeros(gh.n_units))
    out = estimate("ghost", none_exposed)
    assert isinstance(out, Unsupported) and "exposed_treated" in out.missing


# -- A/A calibration -------------------------------------------------------------------

Maker = Callable[[int], PanelArrays]
N_SIMS = 200
REGION = clopper_pearson(N_SIMS, 0.05, alpha=1e-3)


def _aa_count(method: str, make: Maker) -> int:
    false_positives = 0
    for seed in range(N_SIMS):
        est = estimate(method, make(1000 + seed))
        assert isinstance(est, MethodEstimate), est
        false_positives += int(est.excludes_zero())
    return false_positives


@pytest.mark.parametrize(
    "method", ("difference_in_differences", "time_based_regression", "cluster_based_regression")
)
def test_aa_false_positive_rate_panel_methods(method: str) -> None:
    # Twenty units so the normal approximation to the unit-level variances is fair.
    k = _aa_count(method, lambda s: make_panel(s, 0.0, n_units=20))
    assert REGION.accepts(k), (method, k, REGION)


@pytest.mark.slow
def test_aa_false_positive_rate_synthetic_control() -> None:
    # Four treated units fit on one pool: the shared donor noise (the |w_bar|^2 term) must be
    # in the SE, or this rate runs well above 5%.
    k = _aa_count("synthetic_control", lambda s: make_panel(s, 0.0, n_units=20))
    assert REGION.accepts(k), (k, REGION)


def test_aa_false_positive_rate_switchback() -> None:
    k = _aa_count("switchback", lambda s: make_switchback(s, 0.0, n_units=8, n_periods=60))
    assert REGION.accepts(k), (k, REGION)


def test_aa_false_positive_rate_ghost() -> None:
    k = _aa_count("ghost", lambda s: make_ghost(s, 0.0, n_units=60))
    assert REGION.accepts(k), (k, REGION)


def test_registry_names_match_estimators() -> None:
    assert set(METHODS) == set(PANEL_METHODS) | {"switchback", "ghost"}
