from __future__ import annotations

import pytest

from axiom.core import Assumption, Blocked, LedgerLine, Unsupported, Unverified, Verdict, is_failure


def _a(state: str = "unverified") -> Assumption:
    return Assumption(name="overlap", facet="population", statement="positivity holds", state=state)  # type: ignore[arg-type]


def test_verdict_invariants() -> None:
    with pytest.raises(ValueError, match="reason"):
        Verdict(status="blocked")
    with pytest.raises(ValueError, match="assumption"):
        Verdict(status="downgraded")
    with pytest.raises(ValueError, match="unverified"):
        Verdict(status="identified", assumptions=(_a(),))
    ok = Verdict(status="identified", assumptions=(_a("satisfied"),))
    assert ok.licensed
    down = Verdict(status="downgraded", assumptions=(_a(),))
    assert down.licensed
    assert not Verdict(status="blocked", reason="no route").licensed
    assert not Verdict(status="unsupported", reason="no capability").licensed


def test_assumption_transitions_are_copies() -> None:
    a = _a()
    assert a.asserted().state == "asserted" and a.state == "unverified"
    assert a.satisfied().state == "satisfied" and a.violated().state == "violated"
    with pytest.raises(ValueError):
        Assumption(name=" ", facet="x", statement="y")


def test_failures_are_falsy_and_need_reasons() -> None:
    for cls in (Unsupported, Blocked, Unverified):
        f = cls(reason="because")
        assert not f and is_failure(f)
        with pytest.raises(ValueError):
            cls(reason="  ")
    assert not is_failure(1.0)
    assert Unsupported(reason="x", missing=("marginal",)).status == "unsupported"


def test_ledger_line_optional_assumption() -> None:
    line = LedgerLine(kind="facet:window", statement="assumed stationary", assumption=_a())
    assert line.assumption is not None and line.assumption.facet == "population"
    with pytest.raises(ValueError):
        LedgerLine(kind="", statement="x")
