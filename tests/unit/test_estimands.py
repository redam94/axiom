from __future__ import annotations

import pytest
from _factories import _estimand

from axiom.core import D, DimensionError, Intervention, Population, Spec, TimeWindow
from axiom.estimands import FACETS, Level, Quantity, TransferPlan, derived_dimension


def test_construction_invariants() -> None:
    e = _estimand()
    assert Spec.from_json(e.to_json()) == e
    assert e.differing_facets(e) == ()
    with pytest.raises(ValueError, match="does not set treatment"):
        _estimand(intervention=Intervention(doses={"other": 1.0}))
    with pytest.raises(ValueError, match="needs a reference"):
        _estimand(reference=None)
    with pytest.raises(DimensionError, match="derives to"):
        _estimand(dimension=D.time)
    with pytest.raises(ValueError, match="sorted"):
        _estimand(conditioning=("b", "a"))
    assert (
        _estimand(
            quantity=Quantity(kind="marginal"), reference=None, dimension=D.outcome / D.currency
        ).dimension
        == D.outcome / D.currency
    )


def test_derived_dimension_table() -> None:
    assert derived_dimension("contrast", D.outcome, D.currency) == D.outcome
    assert derived_dimension("marginal", D.outcome, D.currency) == D.outcome / D.currency
    assert derived_dimension("ratio", D.outcome, D.currency) == D.outcome / D.currency
    assert derived_dimension("elasticity", D.outcome, D.currency).is_dimensionless
    assert derived_dimension("area", D.outcome, D.currency) == D.outcome * D.currency


def test_facets_are_complete_and_comparable() -> None:
    e = _estimand()
    assert len(FACETS) == 8
    for f in FACETS:
        e.facet(f)
    assert e.differing_facets(_estimand(window=TimeWindow(start=1, stop=8))) == ("window",)


def test_multi_facet_plan_accumulates() -> None:
    e = _estimand()
    t = _estimand(
        population=Population(name="all", strata={"soil": {"clay": 0.5, "loam": 0.5}}),
        window=TimeWindow(start=0, stop=12, basis="per_period"),
        intervention=Intervention(doses={"fertilizer": 200.0}, version="liquid"),
    )
    plan: TransferPlan = e.transfer_to(t)
    assert plan.status == "downgraded" and plan.licensed
    assert plan.differing == ("intervention", "population", "window")
    names = {a.name for a in plan.assumptions}
    assert {
        "surface_correct_between_doses",
        "version_irrelevance",
        "s_admissibility",
        "stationary_dynamics",
        "carryover_contained",
    } <= names
    assert set(plan.corrections) == {"chord_to_marginal", "cumulative_to_per_period"}
    assert len(plan.ledger_lines) == 3 and all(
        line.assumption is not None for line in plan.ledger_lines
    )
    assert Spec.from_json(plan.to_json()) == plan


def test_conditioning_rules() -> None:
    e = _estimand(conditioning=("soil",))
    marginal = e.transfer_to(_estimand())  # collapse soil; target knows soil weights
    entry = marginal.entry("conditioning")
    assert entry.assumptions[0].state == "satisfied"
    unknown = e.transfer_to(_estimand(population=Population(name="north")))
    assert unknown.entry("conditioning").assumptions[0].state == "unverified"
    ratio = _estimand(
        quantity=Quantity(kind="ratio"), dimension=D.outcome / D.currency, conditioning=("soil",)
    )
    blocked = ratio.transfer_to(
        _estimand(quantity=Quantity(kind="ratio"), dimension=D.outcome / D.currency)
    )
    assert blocked.status == "blocked" and "collapse" in blocked.reason


def test_level_rules() -> None:
    e = _estimand()
    assert (
        e.transfer_to(_estimand(level=Level(unit="aggregate"))).entry("level").assumptions[0].name
        == "linear_aggregation"
    )
    assert (
        e.transfer_to(_estimand(level=Level(unit="cluster", interference="within_cluster"))).status
        == "blocked"
    )
    declared = e.transfer_to(
        _estimand(
            level=Level(unit="cluster", interference="declared", interference_model="spatial lag")
        )
    )
    assert (
        declared.status == "downgraded"
        and declared.entry("level").assumptions[0].state == "asserted"
    )
