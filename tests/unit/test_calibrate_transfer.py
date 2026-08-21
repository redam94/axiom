"""``calibrate.transfer``: correction operators and ``resolve``."""

from __future__ import annotations

import numpy as np
import pytest
from _factories import _estimand

from axiom.calibrate.ledger import UNCORRECTED, Ledger
from axiom.calibrate.transfer import (
    Correction,
    ResolvedTransfer,
    aggregation_level,
    carryover_window_factor,
    chord_to_marginal,
    dose_path_accumulation,
    resolve,
    variance_reweight,
)
from axiom.core import (
    Assumption,
    D,
    Intervention,
    Outcome,
    Spec,
    TimeWindow,
    Treatment,
    Unsupported,
    Verdict,
    causal_convolve,
)
from axiom.estimands import Level, Quantity
from axiom.surface import (
    GeometricCarryover,
    HillKernel,
    NoCarryover,
    Surface,
    SurfaceSpec,
)

A = Treatment(name="a", dimension=D.currency, unit="USD")
Y = Outcome(name="y", dimension=D.outcome)
K, S, BETA, ALPHA = 2.0, 1.5, 2.0, 0.3
THETA = {"alpha": ALPHA, "k_a": K, "s_a": S, "beta_a": BETA, "lam_a": 0.6}


def hill(x: float) -> float:
    u = (x / K) ** S
    return float(BETA * u / (1.0 + u))


def hill_slope(x: float) -> float:
    u = (x / K) ** S
    return float(BETA * S * (x / K) ** (S - 1.0) / (K * (1.0 + u) ** 2))


def surface(carryover: bool = False) -> Surface:
    return Surface(
        SurfaceSpec(
            name="s",
            treatments=(A,),
            outcome=Y,
            kernels={"a": HillKernel(reference_dose=K)},
            carryover={"a": GeometricCarryover(max_lag=4)} if carryover else {},
        )
    )


# -- chord vs marginal ------------------------------------------------------------------


def test_chord_reading_is_biased_by_the_predicted_amount_and_the_corrected_one_is_not() -> None:
    """Phase 6 exit criterion 2, tolerance rtol 1e-10 (two forward() evaluations)."""
    d0, d1, at = 0.0, 4.0, 4.0  # a 0-vs-4 contrast read as the marginal at dose 4
    c = chord_to_marginal(surface(), THETA, "a", d0, d1, at)
    assert isinstance(c, Correction)
    chord = (hill(d1) - hill(d0)) / (d1 - d0)
    slope = hill_slope(at)
    # the experiment's chord reading, taken as the marginal, is biased by exactly chord/slope
    np.testing.assert_allclose(c.counterfactual, chord, rtol=1e-10)
    np.testing.assert_allclose(c.corrected, slope, rtol=1e-10)
    np.testing.assert_allclose(chord / slope, 1.0 / c.value, rtol=1e-10)
    assert chord > 2.0 * slope  # the chord overstates the marginal by more than 2x here
    # the corrected reading is unbiased: chord * factor == marginal
    np.testing.assert_allclose(chord * c.value, slope, rtol=1e-10)
    assert c.kind == "multiplicative" and c.facet == "intervention"
    assert isinstance(c.ledger_line.assumption, Assumption)
    assert c.ledger_line.assumption.facet == "intervention"
    assert c.ledger_line.detail["counterfactual"] == repr(chord) or np.isclose(
        float(c.ledger_line.detail["counterfactual"]), chord, rtol=1e-10
    )
    np.testing.assert_allclose(float(c.ledger_line.detail["value"]), slope, rtol=1e-10)
    assert Spec.from_json(c.to_json()) == c


def test_chord_to_marginal_uses_the_steady_state_surface_when_carryover_is_declared() -> None:
    with_c = chord_to_marginal(surface(carryover=True), THETA, "a", 0.0, 4.0, 2.0)
    without = chord_to_marginal(surface(), THETA, "a", 0.0, 4.0, 2.0)
    assert isinstance(with_c, Correction) and isinstance(without, Correction)
    np.testing.assert_allclose(with_c.value, without.value, rtol=1e-12)
    assert with_c.detail["surface"].endswith(":steady_state")


def test_chord_to_marginal_typed_failures() -> None:
    same = chord_to_marginal(surface(), THETA, "a", 1.0, 1.0, 1.0)
    assert isinstance(same, Unsupported) and "d0 == d1" in same.reason
    # a flat chord: Hill is zero at zero dose; doses (0, 0 + tiny) give a zero-ish chord
    flat = chord_to_marginal(surface(), {**THETA, "beta_a": 1e-300}, "a", 0.0, 4.0, 2.0)
    assert isinstance(flat, Unsupported) and "zero" in flat.reason
    with pytest.raises(ValueError, match="finite"):
        chord_to_marginal(surface(), THETA, "a", 0.0, float("nan"), 2.0)


# -- carryover window --------------------------------------------------------------------


@pytest.mark.parametrize("lam", [0.3, 0.6, 0.95])
@pytest.mark.parametrize("window", [1, 2, 3, 4])
def test_geometric_window_factor_matches_closed_form(lam: float, window: int) -> None:
    max_lag = 4
    c = carryover_window_factor(
        GeometricCarryover(max_lag=max_lag), {"lam_a": lam}, window, treatment="a"
    )
    assert isinstance(c, Correction)
    share = (1.0 - lam**window) / (1.0 - lam**max_lag)
    np.testing.assert_allclose(c.counterfactual, share, rtol=1e-12)
    np.testing.assert_allclose(c.value, 1.0 / share, rtol=1e-12)
    np.testing.assert_allclose(c.corrected, 1.0, rtol=1e-12)
    assert c.facet == "window" and c.ledger_line.assumption is not None
    assert c.ledger_line.assumption.facet == "window"
    if window == max_lag:
        np.testing.assert_allclose(c.value, 1.0, rtol=1e-12)


def test_no_carryover_gives_factor_one_with_a_line() -> None:
    c = carryover_window_factor(NoCarryover(), {}, 1, treatment="a")
    assert isinstance(c, Correction)
    assert c.value == 1.0 and c.counterfactual == 1.0
    assert c.ledger_line.kind == "facet:window" and c.ledger_line.assumption is not None
    with pytest.raises(ValueError, match=">= 1"):
        carryover_window_factor(NoCarryover(), {}, 0, treatment="a")


def test_window_factor_refuses_draw_axes() -> None:
    with pytest.raises(ValueError, match="point theta"):
        carryover_window_factor(
            GeometricCarryover(max_lag=3), {"lam_a": np.array([[0.5], [0.6]])}, 2, treatment="a"
        )


# -- dose path accumulation --------------------------------------------------------------


def test_dose_path_accumulation_matches_hand_convolution() -> None:
    lam, max_lag = 0.5, 3
    w = np.array([1.0, lam, lam**2])
    w = w / w.sum()
    path = np.array([0.0, 3.0, 1.0, 0.0, 2.0])
    c = dose_path_accumulation(
        GeometricCarryover(max_lag=max_lag), {"lam_a": lam}, path, treatment="a"
    )
    assert isinstance(c, Correction)
    hand = np.zeros_like(path)
    for t in range(path.size):
        for lag in range(max_lag):
            if t - lag >= 0:
                hand[t] += w[lag] * path[t - lag]
    np.testing.assert_allclose(c.corrected, hand.sum(), rtol=1e-12)
    np.testing.assert_allclose(c.corrected, causal_convolve(path, w).sum(), rtol=1e-12)
    np.testing.assert_allclose(c.counterfactual, path.sum(), rtol=1e-12)
    np.testing.assert_allclose(c.value, hand.sum() / path.sum(), rtol=1e-12)
    assert c.value < 1.0  # the last dose spills past the path's end
    assert c.facet == "intervention"
    # a path long enough for everything to land gives factor 1
    long_path = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    full = dose_path_accumulation(
        GeometricCarryover(max_lag=max_lag), {"lam_a": lam}, long_path, treatment="a"
    )
    assert isinstance(full, Correction)
    np.testing.assert_allclose(full.value, 1.0, rtol=1e-12)


def test_dose_path_accumulation_failures() -> None:
    zero = dose_path_accumulation(NoCarryover(), {}, np.zeros(3), treatment="a")
    assert isinstance(zero, Unsupported)
    with pytest.raises(ValueError, match="non-negative"):
        dose_path_accumulation(NoCarryover(), {}, np.array([1.0, -1.0]), treatment="a")
    with pytest.raises(ValueError, match="non-empty"):
        dose_path_accumulation(NoCarryover(), {}, np.array([]), treatment="a")


# -- variance reweight / aggregation ------------------------------------------------------


def test_variance_reweight_closed_forms() -> None:
    c = variance_reweight(2.0, 100, 25)
    np.testing.assert_allclose(c.value, 2.0, rtol=1e-12)
    np.testing.assert_allclose(c.corrected, 4.0, rtol=1e-12)
    assert c.kind == "se_scale" and c.counterfactual == 2.0 and c.facet == "level"
    m, rho = 10, 0.2
    k = variance_reweight(1.0, 50, 50, icc=rho, cluster_size=m)
    np.testing.assert_allclose(k.value, np.sqrt(1.0 + (m - 1) * rho), rtol=1e-12)
    both = variance_reweight(3.0, 200, 50, icc=rho, cluster_size=m)
    np.testing.assert_allclose(both.corrected, 3.0 * 2.0 * np.sqrt(1.0 + 9 * rho), rtol=1e-12)
    assert float(both.ledger_line.detail["value"]) == both.corrected
    with pytest.raises(ValueError, match="together"):
        variance_reweight(1.0, 1, 1, icc=0.1)
    with pytest.raises(ValueError, match="icc"):
        variance_reweight(1.0, 1, 1, icc=1.5, cluster_size=2)
    with pytest.raises(ValueError, match="positive"):
        variance_reweight(0.0, 1, 1)


def test_aggregation_level_point_invariant_se_reweighted() -> None:
    ind, clu = Level(unit="individual"), Level(unit="cluster")
    m, rho = 20, 0.1
    up = aggregation_level(ind, clu, cluster_size=m, icc=rho)
    down = aggregation_level(clu, ind, cluster_size=m, icc=rho)
    assert isinstance(up, Correction) and isinstance(down, Correction)
    expected = np.sqrt((1.0 + (m - 1) * rho) / m)
    np.testing.assert_allclose(up.value, expected, rtol=1e-12)
    np.testing.assert_allclose(down.value, 1.0 / expected, rtol=1e-12)
    assert up.kind == "se_scale" and up.detail["point_factor"] == "1.0"
    assert up.ledger_line.assumption is not None
    assert up.ledger_line.assumption.name == "linear_aggregation"
    same = aggregation_level(clu, clu, cluster_size=m, icc=rho)
    assert isinstance(same, Correction) and same.value == 1.0
    blocked = aggregation_level(
        ind, Level(unit="cluster", interference="within_cluster"), cluster_size=m, icc=rho
    )
    assert isinstance(blocked, Unsupported) and "interference" in blocked.reason


# -- Correction invariants -----------------------------------------------------------------


def test_correction_refuses_lines_without_provenance() -> None:
    good = variance_reweight(1.0, 4, 1)
    line = good.ledger_line
    with pytest.raises(ValueError, match="names no assumption"):
        Correction(
            **{**good.model_dump(), "ledger_line": line.model_copy(update={"assumption": None})}
        )
    with pytest.raises(ValueError, match="counterfactual"):
        Correction(
            **{
                **good.model_dump(),
                "ledger_line": line.model_copy(
                    update={
                        "detail": {k: v for k, v in line.detail.items() if k != "counterfactual"}
                    }
                ),
            }
        )
    with pytest.raises(ValueError, match="facet line"):
        Correction(**{**good.model_dump(), "ledger_line": line.model_copy(update={"kind": "note"})})
    with pytest.raises(ValueError, match="strictly positive"):
        Correction(**{**good.model_dump(), "value": -1.0})


# -- resolve -------------------------------------------------------------------------------


def _plan():  # type: ignore[no-untyped-def]
    source = _estimand(
        window=TimeWindow(start=0, stop=2, basis="cumulative"), level=Level(unit="individual")
    )
    target = _estimand(
        window=TimeWindow(start=0, stop=8, basis="cumulative"), level=Level(unit="cluster")
    )
    return source.transfer_to(target)


def test_resolve_arithmetic_and_apply() -> None:
    plan = _plan()
    assert plan.differing == ("window", "level")
    win = carryover_window_factor(GeometricCarryover(max_lag=4), {"lam_a": 0.5}, 2, treatment="a")
    lvl = aggregation_level(
        Level(unit="individual"), Level(unit="cluster"), cluster_size=10, icc=0.2
    )
    assert isinstance(win, Correction) and isinstance(lvl, Correction)
    r = resolve(plan, corrections=[win, lvl])
    assert isinstance(r, ResolvedTransfer)
    assert r.status == "downgraded" and r.licensed
    np.testing.assert_allclose(r.factor, win.value, rtol=1e-12)
    np.testing.assert_allclose(r.se_scale, lvl.value, rtol=1e-12)
    assert r.offset == 0.0
    est, se = r.apply(1.5, 0.2)
    np.testing.assert_allclose(est, 1.5 * win.value, rtol=1e-12)
    np.testing.assert_allclose(se, 0.2 * win.value * lvl.value, rtol=1e-12)
    assert r.completeness.status == "identified"
    assert r.ledger.facets_covered() == {"level": 1, "window": 1}
    assert all(ln.detail["status"] == "corrected" for ln in r.ledger.lines)
    assert Spec.from_json(r.to_json()) == r
    with pytest.raises(ValueError, match="non-negative"):
        r.apply(1.0, -1.0)


def test_resolve_without_corrections_keeps_facets_as_explicitly_uncorrected() -> None:
    plan = _plan()
    r = resolve(plan)
    assert r.factor == 1.0 and r.se_scale == 1.0 and r.offset == 0.0
    assert r.ledger == Ledger.from_plan(plan)
    assert all(ln.detail["counterfactual"] == UNCORRECTED for ln in r.ledger.lines)
    assert r.completeness.status == "identified"
    assert r.apply(2.0, 0.5) == (2.0, 0.5)


def test_resolve_folds_several_corrections_for_one_facet_into_one_line() -> None:
    plan = _plan()
    a = variance_reweight(1.0, 4, 1)
    b = aggregation_level(Level(unit="individual"), Level(unit="cluster"), cluster_size=4, icc=0.0)
    assert isinstance(b, Correction)
    r = resolve(plan, corrections=[a, b])
    np.testing.assert_allclose(r.se_scale, a.value * b.value, rtol=1e-12)
    assert r.ledger.facets_covered() == {"level": 1, "window": 1}
    (line,) = [ln for ln in r.ledger.lines if ln.kind == "facet:level"]
    assert line.detail["correction"] == "variance_reweight,aggregation_level"
    np.testing.assert_allclose(float(line.detail["se_scale"]), a.value * b.value, rtol=1e-12)
    assert line.detail["plan_assumption"] == "linear_aggregation"
    assert r.completeness.status == "identified"


def test_resolve_refuses_a_correction_for_a_facet_that_does_not_differ() -> None:
    plan = _estimand().transfer_to(_estimand(window=TimeWindow(start=0, stop=4)))
    with pytest.raises(ValueError, match="level"):
        resolve(plan, corrections=[variance_reweight(1.0, 2, 1)])


def test_blocked_propagation() -> None:
    blocked_plan = _estimand().transfer_to(
        _estimand(
            quantity=Quantity(kind="marginal"), reference=None, dimension=D.outcome / D.currency
        )
    )
    assert blocked_plan.status == "blocked"
    r = resolve(blocked_plan)
    assert r.status == "blocked" and not r.licensed
    assert r.completeness.status == "blocked" and "blocked" in r.completeness.reason
    # a blocked transport verdict blocks an otherwise licensed plan
    plan = _estimand().transfer_to(
        _estimand(population=_estimand().population.model_copy(update={"name": "south"}))
    )
    assert plan.status == "downgraded"
    t = Verdict(status="blocked", reason="no S-admissible set", route="direct")
    r2 = resolve(plan, transport=t)
    assert r2.status == "blocked" and "transport blocked" in r2.reason
    assert r2.ledger.lines[-1].kind == "transport"
    assert r2.ledger.lines[-1].detail["counterfactual"] == "no transport verdict"
    # an identified transport verdict marks the population assumption satisfied
    ok = Verdict(status="identified", reason="same population", route="same_population")
    r3 = resolve(plan, transport=ok)
    assert r3.status == "downgraded"
    pop = next(ln for ln in r3.ledger.lines if ln.kind == "facet:population")
    assert pop.assumption is not None and pop.assumption.state == "satisfied"
    assert pop.detail["transport"] == "same_population"
    assert r3.completeness.status == "identified"


def test_intervention_facet_correction_composes_with_plan() -> None:
    source = _estimand()
    target = _estimand(intervention=Intervention(doses={"fertilizer": 50.0}, version="granular"))
    plan = source.transfer_to(target)
    assert plan.differing == ("intervention",) and "chord_to_marginal" in plan.corrections
    c = chord_to_marginal(surface(), THETA, "a", 0.0, 4.0, 2.0)
    assert isinstance(c, Correction)
    r = resolve(plan, corrections=[c])
    np.testing.assert_allclose(r.factor, c.value, rtol=1e-12)
    (line,) = r.ledger.lines
    assert line.source == plan.source and line.target == plan.target
    assert line.assumption is not None and line.assumption.name == "surface_correct_between_doses"
    assert r.completeness.status == "identified"
