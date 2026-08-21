"""design.optimizer and design.sensitivity: scoring, Pareto front, cooldown, tipping points."""

from __future__ import annotations

import math

import pytest

from axiom.design.economics import ValuePerOutcome
from axiom.design.eig import eig_gaussian
from axiom.design.evoi import DecisionSpec, evoi_gaussian
from axiom.design.optimizer import (
    CandidateScore,
    DesignCandidate,
    EconomicInputs,
    ProgramSchedule,
    evaluate_candidate,
    pareto_front,
    schedule_with_cooldown,
)
from axiom.design.power import difference_se, power_from_se
from axiom.design.sensitivity import SensitivityTable, elasticity, perturb

VPO = ValuePerOutcome(value=2.0, outcome_unit="outcome", numeraire="USD", source="finance table")
DECISION = DecisionSpec(threshold=1.0, value_per_outcome_unit=1000.0, numeraire="USD")
ECON = EconomicInputs(value_per_outcome=VPO, dose_per_period=100.0, discount_rate=0.0)


def _cand(
    name: str,
    *,
    se: float = 0.3,
    cost: float = 10.0,
    holdout: float = 0.2,
    n_periods: int = 4,
    cooldown: int = 2,
) -> DesignCandidate:
    return DesignCandidate(
        name=name,
        method="difference_in_differences",
        n_units=50,
        n_periods=n_periods,
        holdout_fraction=holdout,
        experiment_se=se,
        cost=cost,
        cooldown_periods=cooldown,
    )


def _score(name: str, net: float, cost: float, eig: float) -> CandidateScore:
    """A hand-built score with the given objectives (evsi absorbs the arithmetic)."""
    c = _cand(name, cost=cost)
    return CandidateScore(
        candidate=c,
        eig=eig,
        evpi=1.0,
        evsi=net + cost,
        opportunity_cost=0.0,
        cost=cost,
        net_value=net,
        power=0.5,
        effect=1.0,
        alpha=0.05,
        numeraire="USD",
    )


def test_design_candidate_validation() -> None:
    with pytest.raises(ValueError, match="not registered"):
        _cand("x").model_copy(update={"method": "nope"}).model_validate(
            {**_cand("x").to_dict(), "method": "nope"}
        )
    c = _cand("x", se=difference_se(50, 1.0, 0.2))
    assert c.duration == 6
    assert c.experiment_se == pytest.approx(1.0 / math.sqrt(50 * 0.2 * 0.8))


def test_evaluate_candidate_composes_the_pieces() -> None:
    c = _cand("a", se=0.3, cost=10.0, holdout=0.2, n_periods=4)
    s = evaluate_candidate(c, DECISION, 1.2, 0.5, ECON)
    ev = evoi_gaussian(DECISION, 1.2, 0.5, 0.3)
    assert s.eig == pytest.approx(eig_gaussian(0.5, 0.3))
    assert s.evsi == pytest.approx(ev.evsi) and s.evpi == pytest.approx(ev.evpi)
    # 0.2 * 100 * 4 = 80 dose units, at the prior mean 1.2 outcome each, 2 USD each.
    assert s.opportunity_cost == pytest.approx(80.0 * 1.2 * 2.0)
    assert s.net_value == pytest.approx(s.evsi - s.opportunity_cost - 10.0)
    assert s.power == pytest.approx(power_from_se(1.2, 0.3).power)
    assert s.effect == 1.2 and s.detail["effect_source"] == "prior_mean"
    assert s.numeraire == "USD" and s.name == "a"
    # An explicit ratio and effect are used instead of the prior mean.
    s2 = evaluate_candidate(
        c, DECISION, 1.2, 0.5, ECON.model_copy(update={"marginal_value_ratio": 0.5}), effect=0.6
    )
    assert s2.opportunity_cost == pytest.approx(80.0 * 0.5 * 2.0)
    assert s2.power == pytest.approx(power_from_se(0.6, 0.3).power)
    assert s2.power < s.power
    with pytest.raises(ValueError, match="numeraire"):
        evaluate_candidate(c, DECISION.model_copy(update={"numeraire": "EUR"}), 1.2, 0.5, ECON)


def test_pareto_front_on_a_hand_built_set() -> None:
    a = _score("a", net=10.0, cost=10.0, eig=1.0)
    b = _score("b", net=8.0, cost=5.0, eig=0.5)  # cheaper than a: on the front
    c = _score("c", net=7.0, cost=6.0, eig=0.4)  # dominated by b on all three
    d = _score("d", net=9.0, cost=20.0, eig=2.0)  # most EIG: on the front
    e = _score("e", net=10.0, cost=10.0, eig=1.0)  # exact duplicate of a: kept
    front = pareto_front((c, d, b, a, e))
    assert [s.name for s in front] == ["a", "e", "d", "b"]
    # Fewer objectives: d loses (costlier and less net than a), e stays a tie.
    assert [s.name for s in pareto_front((c, d, b, a, e), ("net_value", "-cost"))] == [
        "a",
        "e",
        "b",
    ]
    assert [s.name for s in pareto_front((c, d, b, a), ("net_value",))] == ["a"]
    assert pareto_front(()) == ()
    with pytest.raises(ValueError, match="unknown objective"):
        pareto_front((a,), ("profit",))
    with pytest.raises(ValueError, match="distinct"):
        pareto_front((a, a))


def test_schedule_with_cooldown() -> None:
    a = _score("a", net=10.0, cost=1.0, eig=1.0)  # 4 periods + 2 cooldown
    b = _score("b", net=8.0, cost=1.0, eig=1.0)
    c = _score("c", net=-1.0, cost=1.0, eig=1.0)  # never scheduled
    long = CandidateScore(
        **{
            **_score("long", net=9.0, cost=1.0, eig=1.0).to_dict(),
            "candidate": _cand("long", n_periods=7, cost=1.0).to_dict(),
        }
    )
    sched = schedule_with_cooldown((c, b, a, long), horizon_periods=12)
    assert isinstance(sched, ProgramSchedule)
    # a first (0-4, free at 6); long needs 7 periods from 6 -> past 12, skipped; b at 6-10.
    assert [(s.name, s.start, s.end, s.free_at) for s in sched.slots] == [
        ("a", 0, 4, 6),
        ("b", 6, 10, 12),
    ]
    assert set(sched.skipped) == {"long", "c"}
    assert sched.total_net_value == pytest.approx(18.0)
    # A longer horizon fits the long one before b.
    wide = schedule_with_cooldown((c, b, a, long), horizon_periods=30)
    assert [s.name for s in wide.slots] == ["a", "long", "b"]
    assert wide.slots[1].start == 6 and wide.slots[2].start == 15
    # The run must end within the horizon; the cooldown may spill over.
    tight = schedule_with_cooldown((a,), horizon_periods=4)
    assert [s.name for s in tight.slots] == ["a"] and tight.slots[0].free_at == 6
    with pytest.raises(ValueError):
        schedule_with_cooldown((a,), horizon_periods=0)


def test_sensitivity_tipping_point_on_a_constructed_flip() -> None:
    # 'precise' buys more information but holds out more dose: it wins when the
    # value of an outcome is small and loses to 'cheap' as value_per_outcome grows,
    # because the opportunity cost grows linearly in it while EVSI does not.
    cheap = _cand("cheap", se=0.3, cost=0.0, holdout=0.1, n_periods=2)
    precise = _cand("precise", se=0.05, cost=0.0, holdout=0.5, n_periods=2)
    econ = EconomicInputs(value_per_outcome=VPO, dose_per_period=1.0, discount_rate=0.0)
    grid = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
    table = perturb((cheap, precise), DECISION, 1.2, 0.5, econ, "value_per_outcome", grid)
    assert isinstance(table, SensitivityTable)
    assert table.mode == "absolute" and table.base_value == 2.0
    assert table.winners[0] == "precise" and table.winners[-1] == "cheap"
    assert len(table.tipping_points) == 1
    lo, hi = table.tipping_points[0]
    assert lo in grid and hi in grid and lo < hi
    assert not table.stable
    assert table.base_net_values == tuple(
        evaluate_candidate(c, DECISION, 1.2, 0.5, econ).net_value for c in (cheap, precise)
    )
    # Net value of each candidate falls with the value per outcome: negative elasticity.
    el = elasticity(table)
    assert set(el) == {"cheap", "precise"}
    assert el["precise"] < el["cheap"] < 0.0 or (el["precise"] < 0.0 and el["cheap"] < 0.0)
    # A multiplier-mode sweep on the per-candidate input reports base 1.0.
    mult = perturb((cheap, precise), DECISION, 1.2, 0.5, econ, "experiment_se", (0.5, 1.0, 2.0))
    assert mult.mode == "multiplier" and mult.base_value == 1.0
    assert mult.base_winner == table.base_winner
    with pytest.raises(ValueError, match="increasing"):
        perturb((cheap,), DECISION, 1.2, 0.5, econ, "prior_sd", (1.0, 0.5))
    stable = perturb((cheap,), DECISION, 1.2, 0.5, econ, "discount_rate", (0.0, 0.1, 0.2))
    assert stable.stable
    with pytest.raises(ValueError, match="zero base"):
        elasticity(stable)  # the base rate is 0.0: no relative change is defined
    off_grid = perturb(
        (cheap,),
        DECISION,
        1.2,
        0.5,
        econ.model_copy(update={"discount_rate": 0.5}),
        "discount_rate",
        (0.0, 0.1, 0.2),
    )
    with pytest.raises(ValueError, match="bracket"):
        elasticity(off_grid)  # the base rate 0.5 lies outside the grid
