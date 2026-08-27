from __future__ import annotations

from pathlib import Path

import pytest

from axiom.core import LedgerLine, TimeWindow
from axiom.io import Catalog, CatalogEntry, CrossScopeError, Program, ProgramStore

_LINE = LedgerLine(
    kind="scope_crossing",
    statement="the pooled prior is shared under the data agreement",
)


def _catalog(tmp_path: Path) -> tuple[Catalog, Program, Program]:
    catalog = Catalog(tmp_path / "catalog")
    a = Program(party="northwind", program="dose-response", started="2026-01-05")
    b = Program(party="acme", program="dose-response")
    catalog.register(a)
    catalog.register(b)
    return catalog, a, b


# -- the scope token ------------------------------------------------------------------------


def test_scope_is_party_over_program() -> None:
    p = Program(party="northwind", program="dose-response")
    assert p.scope == "northwind/dose-response" == str(p)


@pytest.mark.parametrize(
    "bad", ["North", "a/b", "..", "../escape", "", " ", "-leading", "with space", "tab\there"]
)
def test_scope_tokens_cannot_escape_a_root(bad: str) -> None:
    with pytest.raises(ValueError):
        Program(party=bad, program="ok")
    with pytest.raises(ValueError):
        Program(party="ok", program=bad)


def test_programs_are_equal_by_party_and_name_not_by_description() -> None:
    a = Program(party="p", program="q", description="one")
    b = Program(party="p", program="q", description="two")
    assert a.scope == b.scope
    assert a.content_hash() != b.content_hash()  # provenance still counts as content


# -- storage and isolation ------------------------------------------------------------------


def test_put_is_idempotent_and_indexed(tmp_path: Path) -> None:
    catalog, a, _ = _catalog(tmp_path)
    store = catalog.store(a)
    window = TimeWindow(start=0, stop=8)
    first = store.put(window, label="window", tags={"kind": "plan"})
    again = store.put(window, label="window", tags={"kind": "plan"})
    assert first == again == window.content_hash()
    rows = catalog.find(scope=a.scope, label="window")
    assert len(rows) == 1 and rows[0].tags == {"kind": "plan"}
    assert store.get(first) == window
    assert first in store


def test_artifacts_live_under_their_scope(tmp_path: Path) -> None:
    catalog, a, _ = _catalog(tmp_path)
    store = catalog.store(a)
    assert store.root == catalog.root / "scopes" / "northwind" / "dose-response"
    assert store.scope == a.scope
    assert repr(store).startswith("ProgramStore('northwind/dose-response'")


def test_another_scope_gets_a_cross_scope_error_not_the_artifact(tmp_path: Path) -> None:
    catalog, a, b = _catalog(tmp_path)
    digest = catalog.store(a).put(TimeWindow(start=0, stop=8))
    with pytest.raises(CrossScopeError) as exc:
        catalog.store(b).get(digest)
    assert exc.value.holder == a.scope and exc.value.asked_by == b.scope
    assert "Catalog.transfer" in str(exc.value)


def test_an_unknown_digest_is_a_plain_key_error(tmp_path: Path) -> None:
    catalog, _, b = _catalog(tmp_path)
    with pytest.raises(KeyError):
        catalog.store(b).get("0" * 64)
    with pytest.raises(KeyError):
        catalog.entry("0" * 64)


def test_scopes_and_programs_list_what_was_registered(tmp_path: Path) -> None:
    catalog, a, b = _catalog(tmp_path)
    assert [p.scope for p in catalog.programs()] == [b.scope, a.scope]  # sorted
    assert catalog.scopes() == (b.scope, a.scope)


# -- the boundary ---------------------------------------------------------------------------


def test_transfer_records_the_line_that_licensed_it(tmp_path: Path) -> None:
    catalog, a, b = _catalog(tmp_path)
    window = TimeWindow(start=0, stop=8)
    digest = catalog.store(a).put(window)
    line_digest = catalog.transfer(digest, source=a, target=b, line=_LINE)

    assert catalog.store(b).get(digest) == window  # now readable there
    assert catalog.scope_of(digest) == (b.scope, a.scope)

    crossings = catalog.crossings()
    assert len(crossings) == 1
    crossing = crossings[0]
    assert crossing.scope == b.scope and crossing.derived_from == line_digest
    assert catalog.store(b).get(line_digest) == _LINE


def test_transfer_to_the_same_scope_is_refused(tmp_path: Path) -> None:
    catalog, a, _ = _catalog(tmp_path)
    digest = catalog.store(a).put(TimeWindow(start=0, stop=8))
    with pytest.raises(ValueError, match="same scope"):
        catalog.transfer(digest, source=a, target=a, line=_LINE)


def test_transfer_of_an_artifact_the_source_does_not_hold_is_refused(tmp_path: Path) -> None:
    catalog, a, b = _catalog(tmp_path)
    digest = catalog.store(b).put(TimeWindow(start=0, stop=8))
    with pytest.raises(CrossScopeError):
        catalog.transfer(digest, source=a, target=b, line=_LINE)


def test_nothing_crosses_without_a_transfer(tmp_path: Path) -> None:
    catalog, a, _ = _catalog(tmp_path)
    catalog.store(a).put(TimeWindow(start=0, stop=8))
    assert catalog.crossings() == ()


# -- the index ------------------------------------------------------------------------------


def test_find_filters_on_scope_type_label_and_tags(tmp_path: Path) -> None:
    catalog, a, b = _catalog(tmp_path)
    catalog.store(a).put(TimeWindow(start=0, stop=8), label="plan", tags={"arm": "treated"})
    catalog.store(a).put(TimeWindow(start=0, stop=4), label="plan", tags={"arm": "control"})
    catalog.store(b).put(TimeWindow(start=0, stop=8), label="plan", tags={"arm": "treated"})

    assert len(catalog.find(type_name="TimeWindow")) == 3
    assert len(catalog.find(scope=a.scope, type_name="TimeWindow")) == 2
    assert len(catalog.find(label="plan", tags={"arm": "treated"})) == 2
    assert len(catalog.find(scope=a.scope, tags={"arm": "control"})) == 1
    assert catalog.find(label="nope") == ()
    assert len(catalog.store(a).find(type_name="TimeWindow")) == 2
    assert len(catalog.store(a)) == 3  # two windows and the Program itself


def test_the_index_round_trips_and_survives_a_reopen(tmp_path: Path) -> None:
    catalog, a, _ = _catalog(tmp_path)
    catalog.store(a).put(TimeWindow(start=0, stop=8), label="plan", tags={"arm": "treated"})
    reopened = Catalog(catalog.root)
    assert len(reopened) == len(catalog)
    entry = reopened.find(scope=a.scope, label="plan")[0]
    assert CatalogEntry.from_row(entry.row()) == entry
    assert entry.created  # stamped


def test_a_second_catalog_object_sees_the_first_ones_writes(tmp_path: Path) -> None:
    catalog, a, _ = _catalog(tmp_path)
    other = Catalog(catalog.root)
    assert len(other) == 2
    catalog.store(a).put(TimeWindow(start=0, stop=8))
    assert len(other) == 3  # the index is re-read when it grows


def test_tabs_and_newlines_are_refused_because_the_index_is_a_tsv(tmp_path: Path) -> None:
    catalog, a, _ = _catalog(tmp_path)
    store: ProgramStore = catalog.store(a)
    with pytest.raises(ValueError, match="tab or a newline"):
        store.put(TimeWindow(start=0, stop=8), label="a\tb")
    with pytest.raises(ValueError, match="tab or a newline"):
        store.put(TimeWindow(start=0, stop=8), tags={"note": "one\ntwo"})


def test_a_malformed_index_row_says_so() -> None:
    with pytest.raises(ValueError, match="index row has"):
        CatalogEntry.from_row("too\tfew\tfields")
