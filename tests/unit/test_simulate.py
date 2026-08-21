"""``design.simulate``: the panel, the truth it applies, A/A calibration, and the leaderboard.

Fake methods are injected by monkeypatching ``simulate.estimate`` (the
registry dispatches by the six fixed names, so a custom registry entry has
nowhere else to go). Tests that need the real estimators skip until the
method modules are importable.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping
from types import MappingProxyType

import numpy as np
import pytest

from axiom.core import Unsupported, wald
from axiom.design import simulate
from axiom.design.methods.registry import (
    ASSUMPTIONS,
    METHODS,
    MethodEstimate,
    MethodSpec,
    PanelArrays,
)
from axiom.design.simulate import (
    CalibrationResult,
    Leaderboard,
    SimulatedPower,
    SimulationSpec,
    calibrate_method,
    calibrate_registry,
    design_for_method,
    difference_in_differences_se,
    leaderboard,
    simulate_panel,
    simulated_power,
)

Estimator = Callable[[PanelArrays, float], MethodEstimate | Unsupported]


def _has_real_methods() -> bool:
    try:
        importlib.import_module("axiom.design.methods.difference_in_differences")
    except ImportError:
        return False
    return True


def _fake_spec(name: str) -> MethodSpec:
    return MethodSpec(
        name=name,
        assumptions=(ASSUMPTIONS["parallel_trends"],),
        requires_pre_period=True,
        requires_controls=True,
    )


def _did(arrays: PanelArrays, mass: float, *, se_scale: float = 1.0) -> MethodEstimate:
    """Stand-in DiD: treated minus control mean pre-to-post change, two-sample SE."""
    y = arrays.outcome
    change = y[:, arrays.post].mean(axis=1) - y[:, arrays.pre].mean(axis=1)
    t = np.asarray(arrays.treated)
    c = np.asarray(arrays.control)
    effect = float(change[t].mean() - change[c].mean())
    se = se_scale * math.sqrt(change[t].var(ddof=1) / t.size + change[c].var(ddof=1) / c.size)
    return MethodEstimate(
        method="stand_in",
        effect=effect,
        se=se,
        interval=wald(effect, se, mass),
        n_treated=int(t.size),
        n_control=int(c.size),
        n_pre=arrays.pre.stop - arrays.pre.start,
        n_post=arrays.post.stop - arrays.post.start,
        se_method="two_sample",
    )


def _tiny(arrays: PanelArrays, mass: float) -> MethodEstimate:
    """A broken method: right point estimate, an interval a thousand times too narrow."""
    return _did(arrays, mass, se_scale=1e-3)


def _wide(arrays: PanelArrays, mass: float) -> MethodEstimate:
    return _did(arrays, mass, se_scale=3.0)


def _install(
    monkeypatch: pytest.MonkeyPatch, table: Mapping[str, Estimator]
) -> Mapping[str, MethodSpec]:
    def fake_estimate(
        method: str | MethodSpec,
        arrays: PanelArrays,
        *,
        mass: float,
        registry: Mapping[str, MethodSpec],
    ) -> MethodEstimate | Unsupported:
        name = method if isinstance(method, str) else method.name
        return table[name](arrays, mass)

    monkeypatch.setattr(simulate, "estimate", fake_estimate)
    return MappingProxyType({name: _fake_spec(name) for name in table})


# -- the panel -------------------------------------------------------------------------


def test_panel_shapes_windows_and_attached_arrays() -> None:
    spec = SimulationSpec(n_units=12, n_periods=10, n_pre=4, n_treated=5, seed=3)
    p = simulate_panel(spec, np.random.default_rng(spec.seed))
    assert p.outcome.shape == (12, 10)
    assert p.pre == slice(0, 4) and p.post == slice(4, 10)
    assert len(p.treated) == 5 and len(p.control) == 7
    assert p.assignment is not None and p.assignment.shape == (12, 10)
    assert p.exposed is not None and p.exposed.shape == (12,)
    assert set(np.unique(p.assignment)) <= {0.0, 1.0}


def test_panel_is_reproducible_from_seed() -> None:
    spec = SimulationSpec(seed=11)
    a = simulate_panel(spec, np.random.default_rng(11))
    b = simulate_panel(spec, np.random.default_rng(11))
    np.testing.assert_array_equal(a.outcome, b.outcome)
    assert a.treated == b.treated


def _noiseless(effect: float) -> SimulationSpec:
    return SimulationSpec(
        n_units=10,
        n_periods=8,
        n_pre=3,
        n_treated=4,
        unit_sd=0.0,
        period_sd=0.0,
        noise_sd=1e-12,
        effect=effect,
        seed=5,
    )


def test_holdout_truth_is_effect_on_treated_post_only() -> None:
    spec = _noiseless(2.0)
    p = simulate_panel(spec, np.random.default_rng(0), design="holdout")
    expected = np.zeros((10, 8))
    expected[list(p.treated), 3:] = 2.0
    np.testing.assert_allclose(p.outcome, expected, atol=1e-9)


def test_switchback_truth_follows_assignment_everywhere() -> None:
    spec = _noiseless(2.0)
    p = simulate_panel(spec, np.random.default_rng(0), design="switchback")
    assert p.assignment is not None
    np.testing.assert_allclose(p.outcome, 2.0 * p.assignment, atol=1e-9)
    assert p.assignment[list(p.control)].sum() > 0, "controls are switched too"


def test_ghost_truth_is_effect_on_exposed_treated_post_only() -> None:
    spec = _noiseless(2.0).model_copy(update={"exposure_rate": 0.5})
    p = simulate_panel(spec, np.random.default_rng(0), design="ghost")
    assert p.exposed is not None
    expected = np.zeros((10, 8))
    expected[list(p.treated), 3:] = 2.0
    expected *= p.exposed[:, None]
    np.testing.assert_allclose(p.outcome, expected, atol=1e-9)


def test_base_draw_is_shared_across_designs() -> None:
    spec = SimulationSpec(seed=2, effect=0.0)
    designs = ("holdout", "switchback", "ghost")
    outs = [simulate_panel(spec, np.random.default_rng(2), design=d).outcome for d in designs]
    for o in outs[1:]:
        np.testing.assert_array_equal(outs[0], o)


def test_common_shock_has_stated_marginal_sd() -> None:
    spec = SimulationSpec(
        n_units=2,
        n_periods=4000,
        n_pre=1,
        n_treated=1,
        unit_sd=0.0,
        noise_sd=1e-9,
        period_sd=0.7,
        rho=0.8,
        seed=9,
    )
    p = simulate_panel(spec, np.random.default_rng(9))
    assert p.outcome[0].std() == pytest.approx(0.7, rel=0.1)


def test_design_for_method_reads_data_requirements() -> None:
    assert design_for_method("difference_in_differences") == "holdout"
    assert design_for_method("synthetic_control") == "holdout"
    assert design_for_method("time_based_regression") == "holdout"
    assert design_for_method("cluster_based_regression") == "holdout"
    assert design_for_method("switchback") == "switchback"
    assert design_for_method("ghost") == "ghost"


def test_difference_in_differences_se_formula() -> None:
    spec = SimulationSpec(n_units=30, n_treated=10, n_periods=12, n_pre=4, noise_sd=2.0)
    expected = 2.0 * math.sqrt(1 / 4 + 1 / 8) * math.sqrt(1 / 10 + 1 / 20)
    assert difference_in_differences_se(spec) == pytest.approx(expected)
    # unit and period variance components do not enter
    assert difference_in_differences_se(
        spec.model_copy(update={"unit_sd": 9.0, "period_sd": 9.0, "rho": 0.9})
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "bad",
    [
        {"n_pre": 16},
        {"n_treated": 20},
        {"n_treated": 0},
        {"rho": 1.0},
        {"noise_sd": 0.0},
        {"mass": 1.0},
        {"on_fraction": 0.0},
        {"exposure_rate": 0.0},
        {"n_simulations": 0},
        {"effect": float("nan")},
    ],
)
def test_spec_rejects_invalid_arguments(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SimulationSpec(**bad)


# -- calibration -----------------------------------------------------------------------


def test_stand_in_did_calibrates(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _install(monkeypatch, {"stand_in": _did})
    spec = SimulationSpec(n_simulations=200, seed=1)
    r = calibrate_method("stand_in", spec, alpha=0.05, registry=reg)
    assert isinstance(r, CalibrationResult)
    assert r.passed, r.reason
    assert r.region.n == 200 and r.region.p == 0.05 and r.region.alpha == 1e-3
    assert r.region.accepts(r.false_positive_count)
    assert r.n_unsupported == 0
    assert r.false_positive_rate == r.false_positive_count / 200


def test_calibration_forces_zero_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _install(monkeypatch, {"stand_in": _did})
    spec = SimulationSpec(n_simulations=100, seed=1, effect=50.0)
    r = calibrate_method("stand_in", spec, registry=reg)
    assert r.passed, r.reason


def test_broken_method_fails_and_is_marked_experimental(monkeypatch: pytest.MonkeyPatch) -> None:
    reg = _install(monkeypatch, {"stand_in": _did, "tiny": _tiny})
    spec = SimulationSpec(n_simulations=100, seed=4)
    new_reg, results = calibrate_registry(spec, reg, alpha=0.05)
    by_name = {r.method: r for r in results}
    assert not by_name["tiny"].passed
    assert "outside" in by_name["tiny"].reason
    assert by_name["tiny"].false_positive_rate > 0.5
    assert by_name["stand_in"].passed
    assert new_reg["tiny"].status == "experimental"
    assert new_reg["stand_in"].status == "stable"
    # the input registry is untouched and the output is read-only
    assert reg["tiny"].status == "stable"
    with pytest.raises(TypeError):
        new_reg["tiny"] = reg["tiny"]  # type: ignore[index]
    assert METHODS["difference_in_differences"].status == "stable"


def test_unsupported_returns_are_counted_not_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def sometimes(arrays: PanelArrays, mass: float) -> MethodEstimate | Unsupported:
        calls["n"] += 1
        if calls["n"] % 20 == 0:
            return Unsupported(reason="declined this panel")
        return _did(arrays, mass)

    reg = _install(
        monkeypatch, {"sometimes": sometimes, "never": lambda a, m: Unsupported(reason="no")}
    )
    spec = SimulationSpec(n_simulations=100, seed=6)
    r = calibrate_method("sometimes", spec, registry=reg)
    assert r.n_unsupported == 5 and r.n_evaluated == 95
    assert r.region.n == 95
    assert r.passed, r.reason
    never = calibrate_method("never", spec, registry=reg)
    assert never.n_unsupported == 100 and not never.passed
    assert "Unsupported" in never.reason and math.isnan(never.false_positive_rate)
    with pytest.raises(ValueError, match="Unsupported on all"):
        simulated_power("never", spec, registry=reg)


def test_estimator_exceptions_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(arrays: PanelArrays, mass: float) -> MethodEstimate:
        raise RuntimeError("estimator bug")

    reg = _install(monkeypatch, {"boom": boom})
    with pytest.raises(RuntimeError, match="estimator bug"):
        calibrate_method("boom", SimulationSpec(n_simulations=3), registry=reg)


def test_calibrate_rejects_bad_alpha() -> None:
    with pytest.raises(ValueError):
        calibrate_method("difference_in_differences", SimulationSpec(), alpha=0.0)


# -- power and the leaderboard ---------------------------------------------------------


def test_simulated_power_scores_bias_coverage_and_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg = _install(monkeypatch, {"stand_in": _did})
    spec = SimulationSpec(n_units=40, n_treated=20, n_simulations=200, seed=7, effect=0.4)
    from axiom.design.power import power_from_se

    predicted = power_from_se(0.4, difference_in_differences_se(spec), alpha=0.05).power
    r = simulated_power("stand_in", spec, registry=reg, predicted_power=predicted)
    assert isinstance(r, SimulatedPower)
    assert r.truth == 0.4 and r.design == "holdout"
    assert abs(r.bias) < 0.1
    assert 0.85 <= r.coverage <= 1.0
    assert r.region is not None and r.region.p == predicted
    assert r.within_prediction is True
    assert r.rmse >= abs(r.bias)
    assert abs(r.power - predicted) < 0.1
    plain = simulated_power("stand_in", spec, registry=reg)
    assert plain.region is None and plain.within_prediction is None


def test_leaderboard_orders_by_power_with_uncalibrated_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reg = _install(monkeypatch, {"wide": _wide, "tiny": _tiny, "stand_in": _did})
    spec = SimulationSpec(n_units=40, n_treated=20, n_simulations=60, seed=8)
    board = leaderboard(spec, [0.3, 0.6], reg, alpha=0.05)
    assert isinstance(board, Leaderboard)
    assert [r.method for r in board.rows] == ["stand_in", "wide", "tiny"]
    assert [r.rank for r in board.rows] == [1, 2, 3]
    assert board.row("stand_in").calibrated and board.row("wide").calibrated
    assert not board.row("tiny").calibrated and board.row("tiny").status == "experimental"
    assert board.row("stand_in").mean_power > board.row("wide").mean_power
    assert board.effects == (0.3, 0.6)
    for row in board.rows:
        assert len(row.powers) == len(row.biases) == len(row.coverages) == 2
    assert board.powers[0][1].effect == 0.6
    assert board.calibrations[2].method == "tiny"


def test_leaderboard_rejects_empty_or_zero_effects() -> None:
    with pytest.raises(ValueError):
        leaderboard(SimulationSpec(), [], METHODS)
    with pytest.raises(ValueError):
        leaderboard(SimulationSpec(), [0.0], METHODS)


# -- the real estimators ---------------------------------------------------------------


@pytest.mark.skipif(not _has_real_methods(), reason="method modules not yet importable")
def test_real_difference_in_differences_calibrates() -> None:
    spec = SimulationSpec(n_simulations=200, seed=21)
    r = calibrate_method("difference_in_differences", spec, alpha=0.05)
    assert r.passed, r.reason
    assert r.design == "holdout"


@pytest.mark.skipif(not _has_real_methods(), reason="method modules not yet importable")
def test_real_methods_all_run_on_the_shared_panel() -> None:
    spec = SimulationSpec(n_simulations=5, seed=22, effect=1.0)
    for name in METHODS:
        r = simulated_power(name, spec)
        assert r.n_evaluated >= 1, name
