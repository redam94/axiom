from __future__ import annotations

import pytest

from axiom.core import TimeWindow
from axiom.design import (
    CONCURRENT_EXPERIMENTS,
    Collision,
    DecisionSpec,
    Occupancy,
    Recommendation,
    TreatmentCandidate,
    collisions,
    exclusions,
    recommend,
    variance_inflation,
)


def _occupancy(
    name: str,
    units: tuple[str, ...],
    window: tuple[int, int],
    treatments: tuple[str, ...] = ("price",),
    *,
    randomized: bool = True,
) -> Occupancy:
    return Occupancy(
        experiment=name,
        units=units,
        window=TimeWindow(start=window[0], stop=window[1]),
        treatments=treatments,
        randomized=randomized,
    )


_PLAN = [
    _occupancy("NW-14", ("london", "leeds", "bristol"), (0, 8), ("price",)),
    _occupancy("NW-15", ("leeds", "york"), (4, 12), ("banner",)),
    _occupancy("NW-16", ("london", "york"), (6, 10), ("price",)),
    _occupancy("NW-17", ("cardiff",), (0, 8), ("banner",)),
    _occupancy("ROLLOUT", ("leeds", "cardiff"), (2, 6), ("layout",), randomized=False),
]


def _by_pair(found: tuple[Collision, ...]) -> dict[tuple[str, str], Collision]:
    return {c.pair: c for c in found}


# -- the ladder -----------------------------------------------------------------------------


def test_no_shared_unit_or_no_shared_period_is_disjoint() -> None:
    elsewhere = collisions([_PLAN[0], _occupancy("X", ("dublin",), (0, 8))])
    assert elsewhere == ()
    later = collisions([_PLAN[0], _occupancy("Y", ("london",), (8, 12))])
    assert later == ()
    assert (
        len(collisions([_PLAN[0], _occupancy("Y", ("london",), (8, 12))], include_disjoint=True))
        == 1
    )


def test_different_levers_on_shared_units_are_concurrent_not_wrong() -> None:
    """Independent randomizations are orthogonal in expectation; the cost is variance."""
    collision = _by_pair(collisions(_PLAN))[("NW-14", "NW-15")]
    assert collision.kind == "concurrent"
    assert collision.units == ("leeds",) and len(collision.periods) == 4
    assert collision.treatments == ()
    verdict = collision.verdict()
    assert verdict.status == "downgraded"
    assert verdict.assumptions[0].name == "concurrent_experiments"
    assert verdict.assumptions[0].state == "unverified"
    assert "background" in verdict.reason


def test_the_same_lever_twice_is_confounded_and_blocked() -> None:
    collision = _by_pair(collisions(_PLAN))[("NW-14", "NW-16")]
    assert collision.kind == "confounded"
    assert collision.treatments == ("price",)
    assert collision.verdict().status == "blocked"
    assert "separately identified" in collision.reason
    assert collision.assumption().state == "violated"


def test_a_neighbour_that_was_not_randomized_confounds_whatever_it_touches() -> None:
    found = _by_pair(collisions(_PLAN))
    for pair in (("NW-14", "ROLLOUT"), ("NW-15", "ROLLOUT"), ("NW-17", "ROLLOUT")):
        collision = found[pair]
        assert collision.kind == "confounded"
        assert collision.treatments == ()  # not a shared lever — a shared assignment
        assert "not independently randomized" in collision.reason


def test_the_assumption_is_the_one_design_methods_already_names() -> None:
    assert CONCURRENT_EXPERIMENTS.name == "concurrent_experiments"
    assert "do not interact" in CONCURRENT_EXPERIMENTS.statement
    assert "factorial" in CONCURRENT_EXPERIMENTS.challenged_by
    collision = _by_pair(collisions(_PLAN))[("NW-14", "NW-15")]
    assert collision.assumption().detail["units"] == "1"
    assert collision.ledger_line().kind == "collision"
    assert "concurrent over 1 unit(s) and 4 period(s)" in collision.ledger_line().statement


# -- what it refuses to construct -----------------------------------------------------------


def test_an_experiment_does_not_collide_with_itself() -> None:
    with pytest.raises(ValueError, match="does not collide with itself"):
        Collision(left="a", right="a", kind="disjoint")
    with pytest.raises(ValueError, match="names must be distinct"):
        collisions([_PLAN[0], _PLAN[0]])


def test_a_collision_must_be_consistent_with_its_kind() -> None:
    with pytest.raises(ValueError, match="shares at least one unit"):
        Collision(left="a", right="b", kind="concurrent")
    with pytest.raises(ValueError, match="must say what confounds it"):
        Collision(left="a", right="b", kind="confounded", units=("u",), periods=(1,))
    with pytest.raises(ValueError, match="shares no unit or no period"):
        Collision(left="a", right="b", kind="disjoint", units=("u",), periods=(1,))


def test_an_occupancy_needs_units_and_no_repeats() -> None:
    with pytest.raises(ValueError, match="occupies no units"):
        Occupancy(experiment="x", units=(), window=TimeWindow(start=0, stop=1))
    with pytest.raises(ValueError, match="repeats a unit"):
        Occupancy(experiment="x", units=("a", "a"), window=TimeWindow(start=0, stop=1))
    with pytest.raises(ValueError, match="repeats a treatment"):
        Occupancy(
            experiment="x",
            units=("a",),
            window=TimeWindow(start=0, stop=1),
            treatments=("p", "p"),
        )


# -- the price ------------------------------------------------------------------------------


def test_variance_inflation_is_bernoulli_and_zero_at_the_ends() -> None:
    assert variance_inflation(5.0, 0.0, 5.0) == 1.0  # nobody treated
    assert variance_inflation(5.0, 1.0, 5.0) == 1.0  # everybody treated: a constant, not noise
    assert variance_inflation(5.0, 0.5, 5.0) == pytest.approx(1.25)
    assert variance_inflation(2.0, 0.5, 5.0) == pytest.approx(1.04)
    # an effect twice the sd on half the units is a 2x variance
    assert variance_inflation(10.0, 0.5, 5.0) == pytest.approx(2.0)


def test_variance_inflation_refuses_what_it_cannot_price() -> None:
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match=r"share must be in \[0, 1\]"):
            variance_inflation(1.0, bad, 1.0)
    with pytest.raises(ValueError, match="sd finite and positive"):
        variance_inflation(1.0, 0.5, 0.0)


# -- feeding the schedule -------------------------------------------------------------------


def test_exclusions_are_symmetric_and_confounded_only_by_default() -> None:
    found = collisions(_PLAN)
    strict = exclusions(found)
    assert strict["NW-14"] == ("NW-16", "ROLLOUT")
    assert "NW-14" in strict["NW-16"]  # symmetric
    assert "NW-15" not in strict["NW-14"]  # concurrent is a cost, not a conflict
    loose = exclusions(found, kinds=("confounded", "concurrent"))
    assert "NW-15" in loose["NW-14"]


def _candidate(name: str, se: float, cost: float) -> TreatmentCandidate:
    return TreatmentCandidate(
        name=name,
        prior_mean=0.0,
        prior_sd=1.0,
        experiment_se=se,
        decision=DecisionSpec(
            name="ship", threshold=0.0, value_per_outcome_unit=1000.0, numeraire="USD"
        ),
        fixed_cost=cost,
    )


def test_recommend_skips_a_candidate_that_cannot_run_beside_a_chosen_one() -> None:
    candidates = [_candidate("price", 0.2, 10.0), _candidate("banner", 0.25, 20.0)]
    free = recommend(candidates)
    assert isinstance(free, Recommendation)
    assert set(free.selected) == {"price", "banner"}

    constrained = recommend(candidates, exclusions={"price": ["banner"], "banner": ["price"]})
    assert isinstance(constrained, Recommendation)
    assert len(constrained.selected) == 1
    assert constrained.detail["skipped_excluded"]
    assert "beside" in constrained.detail["skipped_excluded"]


def test_an_exclusion_is_not_bought_off_by_net_value() -> None:
    """The second experiment is worth more and still may not run."""
    candidates = [_candidate("price", 0.4, 10.0), _candidate("banner", 0.05, 10.0)]
    ranked = recommend(candidates)
    assert isinstance(ranked, Recommendation)
    first = ranked.detail["order"].split(", ")[0]
    blocked = recommend(
        candidates, exclusions={a: [b] for a, b in (("price", "banner"), ("banner", "price"))}
    )
    assert isinstance(blocked, Recommendation)
    assert blocked.selected == (first,)


def test_recommend_without_exclusions_is_unchanged() -> None:
    candidates = [_candidate("price", 0.2, 10.0), _candidate("banner", 0.25, 20.0)]
    a = recommend(candidates)
    b = recommend(candidates, exclusions={})
    assert isinstance(a, Recommendation) and isinstance(b, Recommendation)
    assert a.selected == b.selected
    assert b.detail["skipped_excluded"] == ""
