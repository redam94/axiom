from __future__ import annotations

from pathlib import Path

import pytest

from axiom.core import D, Outcome, Treatment
from axiom.io import (
    Catalog,
    Change,
    Consensus,
    Definition,
    DefinitionRegistry,
    Program,
    registered,
)

_PAID = Outcome(name="conversion", dimension=D.outcome, unit="count", aggregation="sum")
_RATE = Outcome(name="conversion", dimension=D.outcome, unit="count", aggregation="mean")


def _registry(tmp_path: Path) -> tuple[DefinitionRegistry, Program, Program, Program]:
    catalog = Catalog(tmp_path / "catalog")
    parties = tuple(Program(party=p, program="growth") for p in ("northwind", "acme", "globex"))
    for program in parties:
        catalog.register(program)
    return DefinitionRegistry(catalog), *parties


# -- versions are content -------------------------------------------------------------------


def test_registering_the_same_spec_again_is_a_no_op(tmp_path: Path) -> None:
    """A pipeline that registers on every run must not manufacture versions."""
    reg, northwind, _, _ = _registry(tmp_path)
    first = reg.register(northwind, "conversion", _PAID, at="2026-01-05")
    again = reg.register(northwind, "conversion", _PAID, at="2026-01-06")
    assert first == again
    assert first.version == 1 and first.supersedes == ""
    assert len(reg.history(northwind, "conversion")) == 1


def test_a_changed_spec_is_the_next_version_and_names_what_it_replaced(tmp_path: Path) -> None:
    reg, northwind, _, _ = _registry(tmp_path)
    first = reg.register(northwind, "conversion", _PAID, at="2026-01-05")
    second = reg.register(northwind, "conversion", _RATE, at="2026-03-02", note="switched")
    assert second.version == 2
    assert second.supersedes == first.digest
    assert second.note == "switched"
    assert [d.version for d in reg.history(northwind, "conversion")] == [1, 2]
    assert second.key == "northwind/growth:conversion@2"


def test_a_definition_records_the_type_it_pins(tmp_path: Path) -> None:
    reg, northwind, _, _ = _registry(tmp_path)
    outcome = reg.register(northwind, "conversion", _PAID)
    treatment = reg.register(
        northwind, "offer", Treatment(name="offer", dimension=D.currency, unit="USD")
    )
    assert outcome.type_name.endswith("Outcome")
    assert treatment.type_name.endswith("Treatment")
    assert reg.names(northwind) == ("conversion", "offer")
    assert "Outcome" in str(outcome)


def test_version_numbering_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="version 1 supersedes nothing"):
        Definition(scope="a/b", name="x", version=1, digest="d", type_name="t", supersedes="e")
    with pytest.raises(ValueError, match="must name the digest it replaced"):
        Definition(scope="a/b", name="x", version=2, digest="d", type_name="t")


def test_a_definition_needs_a_name(tmp_path: Path) -> None:
    reg, northwind, _, _ = _registry(tmp_path)
    with pytest.raises(ValueError, match="needs a name"):
        reg.register(northwind, "  ", _PAID)


# -- drift within a party -------------------------------------------------------------------


def test_a_change_is_a_field_level_diff_not_a_flag(tmp_path: Path) -> None:
    """'Somebody redefined conversion in week nine' should come back as what changed."""
    reg, northwind, _, _ = _registry(tmp_path)
    reg.register(northwind, "conversion", _PAID, at="2026-01-05")
    reg.register(northwind, "conversion", _RATE, at="2026-03-02", note="switched to a rate")
    (change,) = reg.changes(northwind, "conversion")
    assert isinstance(change, Change)
    assert change.changed == {"aggregation": "'sum' -> 'mean'"}
    assert change.at == "2026-03-02" and change.note == "switched to a rate"
    assert change.from_version == 1 and change.to_version == 2
    assert "aggregation: 'sum' -> 'mean'" in str(change)


def test_a_term_that_never_changed_has_no_changes(tmp_path: Path) -> None:
    reg, northwind, _, _ = _registry(tmp_path)
    reg.register(northwind, "conversion", _PAID)
    assert reg.changes(northwind, "conversion") == ()
    assert reg.changes(northwind, "never-registered") == ()


def test_a_change_is_between_consecutive_versions() -> None:
    with pytest.raises(ValueError, match="consecutive versions"):
        Change(scope="a/b", name="x", from_version=1, to_version=3)


def test_history_and_get_reach_every_version(tmp_path: Path) -> None:
    reg, northwind, _, _ = _registry(tmp_path)
    reg.register(northwind, "conversion", _PAID)
    reg.register(northwind, "conversion", _RATE)
    assert reg.get(northwind, "conversion", 1) == _PAID
    assert reg.get(northwind, "conversion", 2) == _RATE
    assert reg.get(northwind, "conversion") == _RATE  # latest
    assert reg.current(northwind, "conversion").version == 2
    with pytest.raises(KeyError, match="no version 5"):
        reg.get(northwind, "conversion", 5)
    with pytest.raises(KeyError, match="no definition of"):
        reg.get(northwind, "nope")
    with pytest.raises(KeyError, match="no definition of"):
        reg.current(northwind, "nope")


# -- disagreement across parties ------------------------------------------------------------


def test_parties_that_agree_are_identified(tmp_path: Path) -> None:
    reg, northwind, acme, globex = _registry(tmp_path)
    for program in (northwind, acme, globex):
        reg.register(program, "conversion", _PAID)
    result = reg.consensus("conversion", [northwind, acme, globex])
    assert isinstance(result, Consensus)
    assert result.verdict().status == "identified"
    assert result.agreed == _PAID.content_hash()
    assert result.differences == {}


def test_parties_that_disagree_are_blocked_and_the_diff_says_how(tmp_path: Path) -> None:
    reg, northwind, acme, globex = _registry(tmp_path)
    reg.register(northwind, "conversion", _RATE)
    reg.register(acme, "conversion", _PAID)
    reg.register(globex, "conversion", _PAID)
    result = reg.consensus("conversion", [northwind, acme, globex])
    verdict = result.verdict()
    assert verdict.status == "blocked"
    assert result.agreed == ""
    assert len(result.by_digest) == 2
    assert result.by_digest[_PAID.content_hash()] == ("acme/growth", "globex/growth")
    assert result.differences["northwind/growth"] == {"aggregation": "'sum' -> 'mean'"}
    assert "aggregation" in verdict.reason


def test_silence_is_not_agreement(tmp_path: Path) -> None:
    """A party that never said what it means has not agreed with anybody."""
    reg, northwind, acme, globex = _registry(tmp_path)
    reg.register(northwind, "conversion", _PAID)
    reg.register(acme, "conversion", _PAID)
    result = reg.consensus("conversion", [northwind, acme, globex])
    assert result.missing == ("globex/growth",)
    assert result.verdict().status == "unverified"
    assert "silence is not agreement" in result.verdict().reason
    assert "(not registered) globex/growth" in result.summary()


def test_a_type_change_is_reported_as_one(tmp_path: Path) -> None:
    reg, northwind, acme, _ = _registry(tmp_path)
    reg.register(northwind, "thing", _PAID)
    reg.register(acme, "thing", Treatment(name="thing", dimension=D.currency, unit="USD"))
    result = reg.consensus("thing", [northwind, acme])
    assert result.verdict().status == "blocked"
    assert "type" in next(iter(result.differences.values()))


def test_consensus_needs_programs(tmp_path: Path) -> None:
    reg, _, _, _ = _registry(tmp_path)
    with pytest.raises(ValueError, match="at least one program"):
        reg.consensus("conversion", [])


# -- it lives in the catalog ----------------------------------------------------------------


def test_definitions_are_scoped_like_everything_else(tmp_path: Path) -> None:
    reg, northwind, acme, _ = _registry(tmp_path)
    reg.register(northwind, "conversion", _PAID)
    assert reg.names(acme) == ()
    assert reg.history(acme, "conversion") == ()
    # the spec itself is in northwind's store, and acme cannot read it
    from axiom.io import CrossScopeError

    with pytest.raises(CrossScopeError):
        reg.catalog.store(acme).get(_PAID.content_hash())


def test_the_registry_reads_back_from_a_reopened_catalog(tmp_path: Path) -> None:
    reg, northwind, _, _ = _registry(tmp_path)
    reg.register(northwind, "conversion", _PAID, at="2026-01-05")
    reg.register(northwind, "conversion", _RATE, at="2026-03-02")
    reopened = DefinitionRegistry(Catalog(reg.catalog.root))
    assert [d.version for d in reopened.history(northwind, "conversion")] == [1, 2]
    assert reopened.get(northwind, "conversion") == _RATE
    assert "DefinitionRegistry(" in repr(reopened)


def test_registered_maps_every_term_to_its_current_version(tmp_path: Path) -> None:
    reg, northwind, _, _ = _registry(tmp_path)
    reg.register(northwind, "conversion", _PAID)
    reg.register(northwind, "conversion", _RATE)
    reg.register(northwind, "offer", Treatment(name="offer", dimension=D.currency, unit="USD"))
    current = registered(reg, northwind)
    assert {name: d.version for name, d in current.items()} == {"conversion": 2, "offer": 1}


def test_the_records_round_trip(tmp_path: Path) -> None:
    reg, northwind, acme, _ = _registry(tmp_path)
    reg.register(northwind, "conversion", _PAID)
    reg.register(northwind, "conversion", _RATE)
    reg.register(acme, "conversion", _PAID)
    entry = reg.current(northwind, "conversion")
    change = reg.changes(northwind, "conversion")[0]
    result = reg.consensus("conversion", [northwind, acme])
    assert Definition.from_json(entry.to_json()) == entry
    assert Change.from_json(change.to_json()) == change
    assert Consensus.from_json(result.to_json()) == result
