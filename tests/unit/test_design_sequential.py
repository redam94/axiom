"""Group-sequential boundaries against published tables, and the recursion against
a Monte Carlo of the canonical joint distribution."""

from __future__ import annotations

import math

import numpy as np
import pytest

from axiom.core import LedgerLine
from axiom.design.sequential import (
    Boundary,
    CrossingProbabilities,
    LookOutcome,
    LookSchedule,
    MonitoringPath,
    OperatingCharacteristics,
    StoppingRule,
    alpha_spending,
    crossing_probabilities,
    harm_boundary,
    information_fractions,
    monitor,
    obrien_fleming,
    operating_characteristics,
    pocock,
    spending,
)

EQUAL_5 = (0.2, 0.4, 0.6, 0.8, 1.0)

# Jennison & Turnbull, table 2.1 / 2.3: the two-sided 5 % constants for K equally
# spaced looks. Pocock is the constant itself; O'Brien-Fleming is c in c / sqrt(t),
# which for equal spacing is the final critical value.
POCOCK_TABLE = {1: 1.960, 2: 2.178, 3: 2.289, 4: 2.361, 5: 2.413}
OBF_TABLE = {1: 1.960, 2: 1.977, 3: 2.004, 4: 2.024, 5: 2.040}
# Lan-DeMets O'Brien-Fleming spending at five equal looks, two-sided 5 %.
LAN_DEMETS_OBF = (4.3826, 3.0997, 2.5534, 2.2538, 2.0635)


def equal(k: int) -> tuple[float, ...]:
    return tuple((i + 1) / k for i in range(k))


# -- boundaries reproduce the published constants ---------------------------------------


@pytest.mark.parametrize("k", sorted(POCOCK_TABLE))
def test_pocock_matches_the_table(k: int) -> None:
    boundary = pocock(0.05, equal(k))
    assert boundary.z == pytest.approx((POCOCK_TABLE[k],) * k, abs=5e-4)
    assert boundary.spent[-1] == pytest.approx(0.05, abs=1e-6)


@pytest.mark.parametrize("k", sorted(OBF_TABLE))
def test_obrien_fleming_matches_the_table(k: int) -> None:
    boundary = obrien_fleming(0.05, equal(k))
    assert boundary.z[-1] == pytest.approx(OBF_TABLE[k], abs=5e-4)
    assert float(boundary.detail["constant"]) == pytest.approx(OBF_TABLE[k], abs=5e-4)
    # c / sqrt(t): every look before the last is harder to cross than the last.
    assert boundary.z[0] >= boundary.z[-1]
    assert (k == 1) or boundary.z[0] > boundary.z[-1]


def test_lan_demets_obrien_fleming_matches_the_table() -> None:
    boundary = alpha_spending(0.05, EQUAL_5, family="obrien_fleming")
    assert boundary.z == pytest.approx(LAN_DEMETS_OBF, abs=1e-3)
    assert boundary.spent[-1] == pytest.approx(0.05, abs=1e-6)
    # Each threshold spends exactly the increment the spending function budgeted.
    for t, cumulative in zip(EQUAL_5, boundary.spent, strict=True):
        assert cumulative == pytest.approx(spending("obrien_fleming", t, 0.05), abs=1e-6)


def test_pocock_spending_is_close_to_the_pocock_shape() -> None:
    """The spending function named after a shape reproduces it only approximately."""
    shaped = pocock(0.05, EQUAL_5)
    spent = alpha_spending(0.05, EQUAL_5, family="pocock")
    assert spent.z[-1] == pytest.approx(shaped.z[-1], abs=0.03)
    assert spent.z != shaped.z


def test_a_single_look_is_the_fixed_sample_test() -> None:
    for boundary in (
        pocock(0.05, (1.0,)),
        obrien_fleming(0.05, (1.0,)),
        alpha_spending(0.05, (1.0,)),
    ):
        assert boundary.z[0] == pytest.approx(1.959964, abs=1e-4)


def test_one_sided_boundaries_are_signed_by_side() -> None:
    upper = alpha_spending(0.025, EQUAL_5, side="upper")
    lower = alpha_spending(0.025, EQUAL_5, side="lower")
    assert lower.z == pytest.approx(tuple(-z for z in upper.z), abs=1e-9)
    assert upper.crossed(4, upper.z[4] + 0.01) and not upper.crossed(4, upper.z[4] - 0.01)
    assert lower.crossed(4, lower.z[4] - 0.01) and not lower.crossed(4, lower.z[4] + 0.01)


def test_power_spending_is_monotone_in_rho() -> None:
    early_linear = alpha_spending(0.05, EQUAL_5, family="power", rho=1.0).z[0]
    early_cubic = alpha_spending(0.05, EQUAL_5, family="power", rho=3.0).z[0]
    assert early_cubic > early_linear


def test_spending_functions_reach_alpha_at_full_information() -> None:
    for kind in ("obrien_fleming", "pocock", "power"):
        assert spending(kind, 1.0, 0.05) == pytest.approx(0.05, abs=1e-9)
        assert spending(kind, 0.0, 0.05) == 0.0


# -- the recursion ----------------------------------------------------------------------


def monte_carlo(rule: StoppingRule, drift: float, n: int, seed: int) -> dict[str, float]:
    """Crossing frequencies by simulating the canonical joint distribution directly."""
    rng = np.random.default_rng(seed)
    increments = np.asarray(rule.looks.increments)
    steps = rng.normal(
        loc=drift * increments, scale=np.sqrt(increments), size=(n, rule.looks.n_looks)
    )
    b = np.cumsum(steps, axis=1)
    z = b / np.sqrt(np.asarray(rule.looks.information))
    counts: dict[str, float] = {b.kind: 0.0 for b in rule.boundaries}
    alive = 0
    for row in z:
        for look, value in enumerate(row):
            crossings = rule.crossings(look, float(value))
            if crossings:
                counts[crossings[0].kind] += 1
                break
        else:
            alive += 1
    counts["continue"] = float(alive)
    return {k: v / n for k, v in counts.items()}


@pytest.fixture
def three_boundary_rule() -> StoppingRule:
    return StoppingRule(
        name="hyper3",
        looks=LookSchedule(
            labels=tuple(f"week_{4 * (k + 1)}" for k in range(5)), information=EQUAL_5
        ),
        boundaries=(
            alpha_spending(0.025, EQUAL_5, side="upper", kind="efficacy"),
            harm_boundary(0.95, EQUAL_5, margin=2.0, se_at_full_information=1.5),
            Boundary(kind="futility", side="lower", z=(-2.0, -1.0, -0.3, 0.2, 0.6), binding=False),
        ),
    )


@pytest.mark.parametrize("drift", [0.0, 2.8, -2.8])
def test_recursion_matches_monte_carlo(three_boundary_rule: StoppingRule, drift: float) -> None:
    exact = crossing_probabilities(three_boundary_rule, drift, binding_only=False)
    sampled = monte_carlo(three_boundary_rule, drift, n=40_000, seed=11)
    for kind, per_look in exact.per_look.items():
        # 40k draws: three binomial standard errors is at most 0.0075.
        assert sum(per_look) == pytest.approx(sampled[kind], abs=0.008)
    assert exact.continue_probability == pytest.approx(sampled["continue"], abs=0.008)


def test_first_crossing_probabilities_are_a_distribution(three_boundary_rule: StoppingRule) -> None:
    exact = crossing_probabilities(three_boundary_rule, 1.0, binding_only=False)
    total = sum(sum(v) for v in exact.per_look.values()) + exact.continue_probability
    assert total == pytest.approx(1.0, abs=1e-5)
    assert exact.by_look() == pytest.approx(
        tuple(sum(v[k] for v in exact.per_look.values()) for k in range(5)), abs=1e-12
    )


def test_the_boundary_spends_exactly_its_alpha_under_the_null() -> None:
    for boundary in (
        pocock(0.05, EQUAL_5),
        obrien_fleming(0.05, EQUAL_5),
        alpha_spending(0.05, EQUAL_5, family="pocock"),
    ):
        rule = StoppingRule(
            name="one",
            looks=LookSchedule(labels=tuple("abcde"), information=EQUAL_5),
            boundaries=(boundary,),
        )
        assert crossing_probabilities(rule, 0.0).cumulative("efficacy") == pytest.approx(
            0.05, abs=1e-5
        )


def test_a_harm_boundary_adds_to_the_error_spent(three_boundary_rule: StoppingRule) -> None:
    """A rule is not the sum of its boundaries: adding one spends more, not less."""
    efficacy_only = three_boundary_rule.model_copy(
        update={"boundaries": (three_boundary_rule.boundaries[0],)}
    )
    alone = crossing_probabilities(efficacy_only, 0.0).cumulative("efficacy")
    together = crossing_probabilities(three_boundary_rule, 0.0)
    assert alone == pytest.approx(0.025, abs=1e-5)
    assert together.cumulative("harm") > 0.0
    # The efficacy boundary loses a little of its own alpha to paths the harm
    # boundary stops first.
    assert together.cumulative("efficacy") < alone


def test_a_non_binding_boundary_is_ignored_by_the_arithmetic(
    three_boundary_rule: StoppingRule,
) -> None:
    ignored = crossing_probabilities(three_boundary_rule, 0.0, binding_only=True)
    as_run = crossing_probabilities(three_boundary_rule, 0.0, binding_only=False)
    assert "futility" not in ignored.per_look
    assert as_run.cumulative("futility") > 0.5
    # Ignoring futility can only make the efficacy error larger, never smaller.
    assert ignored.cumulative("efficacy") >= as_run.cumulative("efficacy") - 1e-9


def test_a_rule_with_no_binding_boundary_refuses_rather_than_returning_zero() -> None:
    rule = StoppingRule(
        name="advisory",
        looks=LookSchedule(labels=("a", "b"), information=(0.5, 1.0)),
        boundaries=(Boundary(kind="futility", side="lower", z=(-1.0, 0.0), binding=False),),
    )
    with pytest.raises(ValueError, match="no binding boundary"):
        crossing_probabilities(rule, 0.0)
    assert crossing_probabilities(rule, 0.0, binding_only=False).cumulative("futility") > 0.0


def test_operating_characteristics_expected_information(three_boundary_rule: StoppingRule) -> None:
    null = operating_characteristics(three_boundary_rule, 0.0, binding_only=False)
    benefit = operating_characteristics(three_boundary_rule, 3.5, binding_only=False)
    assert isinstance(null, OperatingCharacteristics)
    assert 0.2 <= benefit.expected_information <= 1.0
    # A design that stops early at a large drift enrolls less than one that never stops.
    assert benefit.expected_information < 1.0
    assert benefit.expected_looks < three_boundary_rule.looks.n_looks
    assert null.stop_probability == pytest.approx(1.0 - null.crossings.continue_probability)


def test_never_stopping_costs_full_information() -> None:
    unreachable = Boundary(kind="efficacy", side="upper", z=(11.0, 11.0))
    rule = StoppingRule(
        name="never",
        looks=LookSchedule(labels=("a", "b"), information=(0.5, 1.0)),
        boundaries=(unreachable,),
    )
    oc = operating_characteristics(rule, 0.0)
    assert oc.expected_information == pytest.approx(1.0, abs=1e-9)
    assert oc.expected_looks == pytest.approx(2.0, abs=1e-9)


def test_grid_resolution_does_not_move_the_answer(three_boundary_rule: StoppingRule) -> None:
    coarse = crossing_probabilities(three_boundary_rule, 1.5, n_grid=401).cumulative("efficacy")
    fine = crossing_probabilities(three_boundary_rule, 1.5, n_grid=2401).cumulative("efficacy")
    assert coarse == pytest.approx(fine, abs=1e-5)


def test_the_default_grid_conserves_mass(three_boundary_rule: StoppingRule) -> None:
    """Second-order convergence: quadrupling the points cuts the defect fourfold."""
    defects = []
    for n_grid in (401, 801, 1601):
        exact = crossing_probabilities(three_boundary_rule, 1.5, n_grid=n_grid, binding_only=False)
        total = sum(sum(v) for v in exact.per_look.values()) + exact.continue_probability
        defects.append(abs(total - 1.0))
    assert defects[0] < 1e-5
    assert defects[0] / defects[1] == pytest.approx(4.0, rel=0.25)
    assert defects[1] / defects[2] == pytest.approx(4.0, rel=0.25)


# -- the posterior-probability harm rule ------------------------------------------------


def test_harm_boundary_is_the_stated_posterior_rule() -> None:
    """Recover the rule from the threshold: crossing means P(effect < -margin) >= p."""
    from scipy import stats as sps

    margin, se_final, p = 3.0, 1.4, 0.95
    boundary = harm_boundary(p, EQUAL_5, margin=margin, se_at_full_information=se_final)
    for look, t in enumerate(EQUAL_5):
        se = se_final / math.sqrt(t)
        effect = boundary.z[look] * se  # an estimate sitting exactly on the boundary
        posterior = float(sps.norm.cdf((-margin - effect) / se))
        assert posterior == pytest.approx(p, abs=1e-9)


def test_a_zero_margin_harm_rule_is_a_constant_threshold() -> None:
    boundary = harm_boundary(0.975, EQUAL_5)
    assert boundary.z == pytest.approx((-1.959964,) * 5, abs=1e-5)


def test_a_margin_makes_the_early_looks_harder_to_cross() -> None:
    boundary = harm_boundary(0.95, EQUAL_5, margin=4.0, se_at_full_information=1.5)
    assert all(a > b for a, b in zip(boundary.z[:-1], boundary.z[1:], strict=True))


# -- the realized path ------------------------------------------------------------------


def test_monitor_stops_at_the_first_crossing(three_boundary_rule: StoppingRule) -> None:
    path = monitor(
        three_boundary_rule,
        [0.4, -0.8, -3.1, 9.9],
        effects=[1.0, -2.0, -6.0, 0.0],
        ses=[2.5, 2.0, 1.9, 1.0],
    )
    assert isinstance(path, MonitoringPath)
    assert path.decision == "stop_harm"
    assert path.stopped_at == 2
    assert len(path.looks) == 3, "looks after the stop are never taken"
    assert path.looks[-1].crossed == "harm"
    assert path.looks[-1].effect == -6.0
    assert path.information_used == pytest.approx(0.6)


def test_the_outer_boundary_wins_when_two_fire(three_boundary_rule: StoppingRule) -> None:
    """A statistic below the harm boundary is also below futility. It is a harm stop."""
    harm, futility = three_boundary_rule.boundaries[1], three_boundary_rule.boundaries[2]
    z = harm.z[2] - 0.5
    assert harm.crossed(2, z) and futility.crossed(2, z)
    assert three_boundary_rule.crossings(2, z)[0].kind == "harm"
    assert monitor(three_boundary_rule, [0.0, 0.0, z]).decision == "stop_harm"


def test_completing_and_still_running_are_different_decisions(
    three_boundary_rule: StoppingRule,
) -> None:
    inside = [0.0, 0.0, 0.0, 0.5, 1.0]
    assert monitor(three_boundary_rule, inside).decision == "completed"
    assert monitor(three_boundary_rule, inside[:3]).decision == "continue"
    assert monitor(three_boundary_rule, []).decision == "continue"


def test_the_ledger_line_names_the_bias_of_a_stopped_estimate(
    three_boundary_rule: StoppingRule,
) -> None:
    stopped = monitor(three_boundary_rule, [0.0, 0.0, -3.5]).ledger_line()
    assert isinstance(stopped, LedgerLine)
    assert stopped.assumption is not None
    assert stopped.assumption.name == "stopped_estimate_bias"
    assert "harm boundary crossed at look 3" in stopped.statement
    ran_on = monitor(three_boundary_rule, [0.0, 0.0]).ledger_line()
    assert ran_on.assumption is not None
    assert ran_on.assumption.name == "canonical_joint_distribution"


def test_monitor_refuses_more_statistics_than_looks(three_boundary_rule: StoppingRule) -> None:
    with pytest.raises(ValueError, match="statistics for a schedule"):
        monitor(three_boundary_rule, [0.0] * 6)
    with pytest.raises(ValueError, match="one entry per statistic"):
        monitor(three_boundary_rule, [0.0, 0.0], effects=[1.0])
    with pytest.raises(ValueError, match="not finite"):
        monitor(three_boundary_rule, [float("nan")])


def test_look_outcome_is_frozen(three_boundary_rule: StoppingRule) -> None:
    outcome = monitor(three_boundary_rule, [0.1]).looks[0]
    assert isinstance(outcome, LookOutcome)
    with pytest.raises(AttributeError):
        outcome.z = 2.0  # type: ignore[misc]


# -- schedules and validation -----------------------------------------------------------


def test_information_fractions_from_counts() -> None:
    assert information_fractions([25, 50, 100]) == pytest.approx((0.25, 0.5, 1.0))
    assert information_fractions([25, 50], total=100) == pytest.approx((0.25, 0.5))
    with pytest.raises(ValueError, match="exceeds the planned total"):
        information_fractions([25, 150], total=100)
    with pytest.raises(ValueError, match="at least one count"):
        information_fractions([])


def test_a_schedule_must_be_increasing_and_bounded() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        LookSchedule(labels=("a", "b"), information=(0.5, 0.5))
    with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
        LookSchedule(labels=("a",), information=(1.5,))
    with pytest.raises(ValueError, match="same length"):
        LookSchedule(labels=("a",), information=(0.5, 1.0))
    schedule = LookSchedule(labels=("a", "b", "c"), information=(0.25, 0.5, 1.0))
    assert schedule.increments == pytest.approx((0.25, 0.25, 0.5))
    assert schedule.n_looks == 3


def test_a_rule_whose_boundaries_cross_is_refused() -> None:
    with pytest.raises(ValueError, match="continuation region at look 1 is empty"):
        StoppingRule(
            name="impossible",
            looks=LookSchedule(labels=("a",), information=(1.0,)),
            boundaries=(
                Boundary(kind="efficacy", side="upper", z=(1.0,)),
                Boundary(kind="futility", side="lower", z=(2.0,)),
            ),
        )
    with pytest.raises(ValueError, match="thresholds but the schedule has"):
        StoppingRule(
            name="ragged",
            looks=LookSchedule(labels=("a", "b"), information=(0.5, 1.0)),
            boundaries=(Boundary(kind="efficacy", side="upper", z=(2.0,)),),
        )


def test_a_two_sided_boundary_needs_positive_thresholds() -> None:
    with pytest.raises(ValueError, match="positive thresholds"):
        Boundary(kind="efficacy", side="two_sided", z=(-2.0,))
    two_sided = Boundary(kind="efficacy", side="two_sided", z=(2.0,))
    assert two_sided.crossed(0, -2.5) and two_sided.crossed(0, 2.5)
    assert two_sided.limits(0) == (-2.0, 2.0)
    assert two_sided.nominal_alpha(0) == pytest.approx(0.0455, abs=1e-4)


def test_the_rule_finds_its_boundaries_by_kind(three_boundary_rule: StoppingRule) -> None:
    assert three_boundary_rule.of_kind("harm") is three_boundary_rule.boundaries[1]
    assert three_boundary_rule.of_kind("efficacy") is three_boundary_rule.boundaries[0]


def test_crossing_probabilities_refuse_a_non_finite_drift(
    three_boundary_rule: StoppingRule,
) -> None:
    with pytest.raises(ValueError, match="drift must be finite"):
        crossing_probabilities(three_boundary_rule, float("inf"))
    with pytest.raises(ValueError, match="n_grid must be at least"):
        crossing_probabilities(three_boundary_rule, 0.0, n_grid=10)


def test_an_impossible_alpha_is_refused_not_clipped() -> None:
    with pytest.raises(ValueError, match="alpha must be in"):
        pocock(1.5, EQUAL_5)
    with pytest.raises(ValueError, match="probability must be in"):
        harm_boundary(0.4, EQUAL_5)
    with pytest.raises(ValueError, match="margin must be non-negative"):
        harm_boundary(0.95, EQUAL_5, margin=-1.0)


def test_crossing_probabilities_round_trips(three_boundary_rule: StoppingRule) -> None:
    exact = crossing_probabilities(three_boundary_rule, 0.5, binding_only=False)
    assert CrossingProbabilities.from_dict(exact.to_dict()) == exact
