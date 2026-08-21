"""Gate 11: every facet is populated, and every single-facet difference is licensed or blocked.

Parametrized over ``FACETS`` itself, so a ninth facet without a licensing
rule fails CI on arrival. Also parametrized over the sub-fields the review
added (``intervention.version``, ``level.interference``).
"""

from __future__ import annotations

import pytest
from _factories import _estimand

from axiom.core import D, Intervention, LedgerLine, Outcome, Population, TimeWindow, Treatment
from axiom.estimands import FACETS, Estimand, Facet, Level, Quantity

# one variant per facet that differs in exactly that facet (plus `dimension`
# where the facet forces it, which the test accounts for)
VARIANTS: dict[str, dict[str, object]] = {
    "quantity": {
        "quantity": Quantity(kind="marginal"),
        "dimension": D.outcome / D.currency,
        "reference": None,
    },
    "intervention": {"intervention": Intervention(doses={"fertilizer": 150.0}, version="granular")},
    "intervention.version": {
        "intervention": Intervention(doses={"fertilizer": 100.0}, version="liquid")
    },
    "intervention.treatment": {
        "treatment": Treatment(name="irrigation", dimension=D.currency, unit="USD"),
        "intervention": Intervention(doses={"irrigation": 100.0}, version="granular"),
        "reference": Intervention(doses={"irrigation": 0.0}, version="granular"),
    },
    "outcome": {"outcome": Outcome(name="yield_marketable", dimension=D.outcome, unit="kg")},
    "outcome.dimension": {
        "outcome": Outcome(name="revenue", dimension=D.currency),
        "dimension": D.currency,
    },
    "population": {"population": Population(name="all")},
    "window": {"window": TimeWindow(start=0, stop=12)},
    "window.basis": {"window": TimeWindow(start=0, stop=8, basis="per_period")},
    "level": {"level": Level(unit="aggregate")},
    "level.interference": {"level": Level(unit="cluster", interference="within_cluster")},
    "conditioning": {"conditioning": ("soil",)},
    "dimension": {"dimension": D.outcome},  # cannot differ alone; covered via quantity/outcome
}


def test_every_facet_is_a_required_field() -> None:
    fields = Estimand.model_fields
    for facet in FACETS:
        if facet == "intervention":
            assert fields["treatment"].is_required() and fields["intervention"].is_required()
        elif facet == "conditioning":
            assert "conditioning" in fields  # empty tuple is a legitimate (marginal) value
        else:
            assert fields[facet].is_required(), facet


def test_every_facet_has_a_variant_in_this_test() -> None:
    assert {v.split(".")[0] for v in VARIANTS} == set(FACETS)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_single_facet_difference_is_licensed_or_blocked(variant: str) -> None:
    facet: Facet = variant.split(".")[0]  # type: ignore[assignment]
    source = _estimand()
    target = _estimand(**VARIANTS[variant])
    plan = source.transfer_to(target)
    if variant == "dimension":
        assert plan.status == "identified" and not plan.differing
        return
    assert facet in plan.differing, (variant, plan.differing)
    entry = plan.entry(facet)
    if entry.blocked is not None:
        assert plan.status == "blocked"
        assert entry.blocked.reason.strip()
        assert facet in plan.reason
    else:
        assert plan.status == "downgraded"
        assert entry.assumptions and all(a.facet == facet for a in entry.assumptions)
        assert all(
            a.state != "satisfied" or a.name == "target_strata_weights_known"
            for a in entry.assumptions
        )
    # exactly one ledger line per differing facet, typed, with both hashes
    lines = [line for line in plan.ledger_lines if line.kind == f"facet:{facet}"]
    assert len(lines) == 1 and isinstance(lines[0], LedgerLine)
    assert lines[0].source == source.content_hash() and lines[0].target == target.content_hash()
    assert len(plan.ledger_lines) == len(plan.differing)


def test_identical_estimands_transfer_identified() -> None:
    e = _estimand()
    plan = e.transfer_to(e.model_copy())
    assert plan.status == "identified" and not plan.assumptions and not plan.ledger_lines


def test_no_silent_pass_in_the_rule_table() -> None:
    """A FacetDiff cannot be constructed with neither an assumption nor a block."""
    from axiom.estimands import FacetDiff

    with pytest.raises(ValueError, match="neither"):
        FacetDiff(facet="window", statement="differs")
