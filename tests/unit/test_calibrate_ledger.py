"""``calibrate.ledger``: immutability, ``from_plan``, ``check_complete``, reporting."""

from __future__ import annotations

import pandas as pd
import pytest
from _factories import _assumption, _estimand
from pydantic import ValidationError

from axiom.calibrate.ledger import UNCORRECTED, Ledger, facet_of
from axiom.core import D, LedgerLine, Spec, TimeWindow
from axiom.estimands import Level, Quantity, TransferPlan


def _plan() -> TransferPlan:
    return _estimand().transfer_to(
        _estimand(window=TimeWindow(start=0, stop=12), level=Level(unit="individual"))
    )


def _facet_line(facet: str, **detail: str) -> LedgerLine:
    return LedgerLine(
        kind=f"facet:{facet}",
        statement=f"{facet} corrected",
        assumption=_assumption().model_copy(update={"facet": facet}),
        detail={"counterfactual": "1.0", "value": "2.0", **detail},
    )


def test_append_returns_a_new_ledger() -> None:
    empty = Ledger()
    one = empty.append(_facet_line("window"))
    two = one.append(_facet_line("level"), LedgerLine(kind="note", statement="free text"))
    assert empty.lines == () and len(one.lines) == 1 and len(two.lines) == 3
    assert one.content_hash() != two.content_hash()
    assert two.facets_covered() == {"level": 1, "window": 1}
    assert facet_of(two.lines[2]) is None
    with pytest.raises(ValidationError):  # frozen
        one.lines = ()  # type: ignore[misc]
    assert Spec.from_json(two.to_json()) == two


def test_from_plan_stamps_the_uncorrected_counterfactual() -> None:
    plan = _plan()
    led = Ledger.from_plan(plan)
    assert len(led.lines) == len(plan.ledger_lines) == 2
    for line in led.lines:
        assert line.detail["counterfactual"] == UNCORRECTED
        assert line.detail["value"] == UNCORRECTED
        assert line.detail["correction"] == "none"
        assert line.detail["status"] == "assumed"
        assert line.assumption is not None and line.assumption.facet == facet_of(line)
    assert led.check_complete(plan).status == "identified"
    assert "window" in led.check_complete(plan).reason


def test_check_complete_names_every_problem() -> None:
    plan = _plan()
    # missing facet
    v = Ledger(lines=(_facet_line("window"),)).check_complete(plan)
    assert v.status == "blocked" and "level: missing" in v.reason and "window" not in v.reason
    # duplicated facet
    v = Ledger(
        lines=(_facet_line("window"), _facet_line("window"), _facet_line("level"))
    ).check_complete(plan)
    assert "window: duplicated (2 lines)" in v.reason
    # no assumption / no counterfactual / no value
    bare = LedgerLine(kind="facet:level", statement="x", detail={"value": "1"})
    v = Ledger(lines=(_facet_line("window"), bare)).check_complete(plan)
    assert "level: no assumption" in v.reason and "level: no counterfactual" in v.reason
    novalue = _facet_line("level").model_copy(update={"detail": {"counterfactual": "1"}})
    v = Ledger(lines=(_facet_line("window"), novalue)).check_complete(plan)
    assert "level: no value" in v.reason
    # a blocked plan line
    blocked = _estimand().transfer_to(
        _estimand(
            quantity=Quantity(kind="marginal"), reference=None, dimension=D.outcome / D.currency
        )
    )
    v = Ledger.from_plan(blocked).check_complete(blocked)
    assert v.status == "blocked" and "quantity: blocked" in v.reason
    # facets outside the plan are ignored
    extra = Ledger(lines=(_facet_line("window"), _facet_line("level"), _facet_line("outcome")))
    assert extra.check_complete(plan).status == "identified"
    # an identical estimand: nothing differs, trivially complete
    same = _estimand().transfer_to(_estimand())
    assert Ledger().check_complete(same).status == "identified"


def test_to_frame_and_summary() -> None:
    plan = _plan()
    led = Ledger.from_plan(plan).append(LedgerLine(kind="transport", statement="ok"))
    frame = led.to_frame()
    assert isinstance(frame, pd.DataFrame) and len(frame) == 3
    assert list(frame["facet"]) == ["window", "level", ""]
    assert set(frame.columns) >= {"kind", "assumption", "state", "counterfactual", "value"}
    assert frame["assumption"].iloc[0] == "stationary_dynamics"
    text = led.summary()
    assert text.startswith("ledger: 3 line(s)") and "facet:window" in text and UNCORRECTED in text
    assert Ledger().summary() == "ledger: empty"
    assert Ledger().to_frame().empty
