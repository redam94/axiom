"""The record: what goes in, what it refuses, and what it says is unresolved."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from axiom.core import Assumption, Interval, Verdict
from axiom.identify import CausalGraph, identify

from axiom_dossier import Evidence, EvidenceBuilder, Quantity, quantity_from


def test_a_float_and_an_interval_both_become_quantities() -> None:
    plain = quantity_from("n", 300, label="Participants")
    assert (plain.value, plain.interval) == (300.0, None)

    banded = quantity_from(
        "effect", Interval(lower=-16.7, upper=-8.1, definition="eti", mass=0.9), label="Effect"
    )
    assert banded.value == pytest.approx(-12.4)
    assert banded.interval is not None
    assert "no separate estimate" in banded.note


def test_a_result_object_is_read_through_value_and_interval() -> None:
    @dataclass
    class Result:
        value: float
        interval: Interval

    r = Result(2.5, Interval(lower=1.0, upper=4.0, definition="hdi", mass=0.95))
    q = quantity_from("r", r, label="Contrast", unit="mmHg")
    assert (q.value, q.unit) == (2.5, "mmHg")
    assert q.interval is not None and q.interval.definition == "hdi"
    assert q.source == "Result"


def test_something_with_no_number_in_it_is_a_typeerror_naming_the_type() -> None:
    with pytest.raises(TypeError, match="cannot read a quantity out of str"):
        quantity_from("x", "twelve", label="Twelve")
    with pytest.raises(TypeError, match="boolean is not a reportable quantity"):
        quantity_from("x", True, label="Yes")


def test_a_quantity_states_itself_with_the_interval_and_its_provenance() -> None:
    q = Quantity(
        key="c",
        label="A vs B",
        value=-12.4,
        unit="mmHg",
        interval=Interval(lower=-16.7, upper=-8.1, definition="eti", mass=0.9),
        precision=1,
    )
    stated = q.stated()
    assert "-12.4 mmHg" in stated
    assert "90%" in stated and "ETI" in stated
    assert "-16.7 to -8.1" in stated


def test_two_quantities_cannot_share_a_key() -> None:
    builder = EvidenceBuilder("T").finding("c", 1.0, label="One").diagnostic("c", 2.0, label="Two")
    with pytest.raises(ValueError, match="share a key"):
        builder.build()


def test_the_identification_verdict_is_read_for_everything_it_knows() -> None:
    """The point of generating a methods section: it names the adjustment set."""
    graph = CausalGraph.from_edges("age -> dose, age -> pressure, dose -> pressure")
    verdict = identify(graph, "dose", "pressure")
    evidence = EvidenceBuilder("T").verdict(verdict).build()

    step = evidence.steps[0]
    assert step.title == "Identification"
    assert "effect of dose on pressure" in step.what
    assert "adjusts for age" in step.what
    assert step.detail["route"] == "backdoor"
    assert step.detail["adjustment set"] == "age"


def test_a_bare_core_verdict_is_accepted_too() -> None:
    verdict = Verdict(status="identified", reason="by design", route="randomization")
    evidence = EvidenceBuilder("T").verdict(verdict).build()
    assert evidence.verdict is not None and evidence.verdict.route == "randomization"
    assert "randomization" in evidence.steps[0].what


def test_something_that_is_not_a_verdict_is_refused() -> None:
    with pytest.raises(TypeError, match="expected a core.Verdict"):
        EvidenceBuilder("T").verdict(object())


def test_unresolved_lists_every_assumption_that_is_not_identified() -> None:
    settled = Assumption(
        name="randomized", facet="design", statement="arms were randomized", state="satisfied"
    )
    standing = Assumption(
        name="no_unmeasured_confounding",
        facet="population",
        statement="x is the only common cause",
        state="unverified",
    )
    broken = Assumption(
        name="positivity",
        facet="population",
        statement="every unit could be treated",
        state="violated",
    )
    evidence = EvidenceBuilder("T").assume(settled, standing, broken).build()
    assert evidence.unresolved() == ("no_unmeasured_confounding", "positivity")


def test_the_context_carries_intervals_rather_than_bare_points() -> None:
    """Handing the renderer the interval is what keeps uncertainty attached."""
    evidence = (
        EvidenceBuilder("T")
        .finding(
            "c",
            Interval(lower=1.0, upper=3.0, definition="eti", mass=0.9),
            label="Contrast",
        )
        .build()
    )
    ctx = evidence.context()
    assert isinstance(ctx["c"], Interval)
    assert "90%" in str(ctx["c_stated"])


def test_evidence_hashes_and_round_trips_like_any_spec() -> None:
    evidence = EvidenceBuilder("T", "why?").finding("c", 1.5, label="C").build()
    again = Evidence.from_json(evidence.to_json())
    assert again == evidence
    assert again.content_hash() == evidence.content_hash()


def test_by_key_finds_a_quantity_or_says_what_there_is() -> None:
    evidence = EvidenceBuilder("T").finding("c", 1.0, label="C").build()
    assert evidence.by_key("c").label == "C"
    with pytest.raises(KeyError, match="no quantity 'nope'"):
        evidence.by_key("nope")
