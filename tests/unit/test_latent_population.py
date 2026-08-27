from __future__ import annotations

import pytest

from axiom.core import (
    D,
    Intervention,
    LatentSelection,
    Outcome,
    Population,
    TimeWindow,
    Treatment,
)
from axiom.estimands import Estimand, Level, Quantity

_LETTER = LatentSelection(kind="complier", instrument="letter", exposure="attended", share=0.61)
_CALL = LatentSelection(kind="complier", instrument="phone_call", exposure="attended", share=0.4)
_ALWAYS = LatentSelection(kind="always_taker", instrument="letter", exposure="attended")


def _estimand(population: Population, name: str = "lift") -> Estimand:
    return Estimand(
        name=name,
        quantity=Quantity(kind="contrast"),
        treatment=Treatment(name="course", dimension=D.currency, unit="USD"),
        intervention=Intervention(doses={"course": 1.0}),
        reference=Intervention(doses={"course": 0.0}),
        outcome=Outcome(name="score", dimension=D.outcome, unit="pt", aggregation="mean"),
        population=population,
        window=TimeWindow(start=0, stop=8),
        level=Level(unit="individual"),
        dimension=D.outcome,
    )


_EVERYBODY = _estimand(Population(name="enrolled"))
_COMPLIERS = _estimand(Population(name="enrolled", latent=_LETTER))
_CALLERS = _estimand(Population(name="enrolled", latent=_CALL))


# -- the selection --------------------------------------------------------------------------


def test_a_selection_is_identified_by_its_response_not_by_its_size() -> None:
    """The compliers of a letter and of a phone call are different people."""
    resized = _LETTER.model_copy(update={"share": 0.2})
    assert _LETTER.same_stratum(resized)
    assert not _LETTER.same_stratum(_CALL)
    assert not _LETTER.same_stratum(_ALWAYS)


def test_a_share_is_a_share() -> None:
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match=r"share must be in \[0, 1\]"):
            LatentSelection(kind="complier", instrument="letter", exposure="attended", share=bad)
    assert LatentSelection(kind="c", instrument="i", exposure="e").share is None


def test_a_population_knows_whether_it_can_be_listed() -> None:
    assert not Population(name="enrolled").is_latent
    assert Population(name="enrolled", latent=_LETTER).is_latent
    assert "61.0%" in str(_LETTER) and "compliers of letter" in str(_LETTER)


def test_strata_and_a_latent_selection_coexist() -> None:
    """A complier population inside a stratum is a real thing to want."""
    population = Population(
        name="north", strata={"soil": {"clay": 0.3, "loam": 0.7}}, latent=_LETTER
    )
    assert population.is_latent and population.strata["soil"]["loam"] == 0.7


# -- the facet ------------------------------------------------------------------------------


def test_a_latent_population_differs_from_the_one_around_it() -> None:
    plan = _COMPLIERS.transfer_to(_EVERYBODY)
    assert plan.differing == ("population",)
    assert plan.status == "downgraded"
    (assumption,) = plan.entry("population").assumptions
    assert assumption.name == "latent_type_homogeneity"
    assert assumption.state == "unverified"
    assert assumption.detail["instrument"] == "letter"


def test_the_bridge_is_named_in_both_directions() -> None:
    outward = _COMPLIERS.transfer_to(_EVERYBODY).entry("population").assumptions[0]
    inward = _EVERYBODY.transfer_to(_COMPLIERS).entry("population").assumptions[0]
    assert "outward to the whole population" in outward.statement
    assert "inward to the subpopulation" in inward.statement
    assert "never the units themselves" in outward.challenged_by


def test_two_different_latent_strata_are_blocked() -> None:
    plan = _COMPLIERS.transfer_to(_CALLERS)
    assert plan.status == "blocked"
    assert "different sets of units" in plan.reason
    assert "no observable distinguishes" in plan.reason
    assert plan.entry("population").blocked is not None
    assert plan.entry("population").assumptions == ()


def test_the_same_stratum_at_a_different_size_is_not_blocked() -> None:
    resized = _estimand(
        Population(name="enrolled", latent=_LETTER.model_copy(update={"share": 0.55}))
    )
    plan = _COMPLIERS.transfer_to(resized)
    assert plan.status == "downgraded"
    assert [a.name for a in plan.assumptions] == ["s_admissibility"]


def test_a_latent_move_across_populations_names_both_assumptions() -> None:
    elsewhere = _estimand(Population(name="south", latent=_LETTER))
    plan = _COMPLIERS.transfer_to(elsewhere)
    assert plan.status == "downgraded"
    assert [a.name for a in plan.assumptions] == ["s_admissibility"]

    to_everyone_elsewhere = _COMPLIERS.transfer_to(_estimand(Population(name="south")))
    assert [a.name for a in to_everyone_elsewhere.assumptions] == [
        "latent_type_homogeneity",
        "s_admissibility",
    ]


def test_an_unchanged_population_still_transfers_cleanly() -> None:
    assert _COMPLIERS.transfer_to(_COMPLIERS).status == "identified"
    assert _EVERYBODY.transfer_to(_EVERYBODY).status == "identified"
    assert _COMPLIERS.transfer_to(_COMPLIERS).differing == ()


def test_the_ordinary_population_difference_is_unchanged() -> None:
    """The facet rule gained a case; it did not change the one that was there."""
    plan = _EVERYBODY.transfer_to(_estimand(Population(name="south")))
    assert plan.status == "downgraded"
    assert [a.name for a in plan.assumptions] == ["s_admissibility"]
    assert "pending selection-diagram verdict" in plan.assumptions[0].detail["verdict"]


def test_every_differing_facet_still_gets_exactly_one_ledger_line() -> None:
    plan = _COMPLIERS.transfer_to(_estimand(Population(name="south")))
    kinds = [line.kind for line in plan.ledger_lines]
    assert kinds == ["facet:population"]
    assert plan.ledger_lines[0].assumption is not None
    assert plan.ledger_lines[0].detail["status"] == "assumed"


def test_the_selection_round_trips_and_hashes() -> None:
    assert LatentSelection.from_json(_LETTER.to_json()) == _LETTER
    assert _LETTER.content_hash() != _CALL.content_hash()
    population = Population(name="enrolled", latent=_LETTER)
    assert Population.from_json(population.to_json()) == population
    assert population.content_hash() != Population(name="enrolled").content_hash()
