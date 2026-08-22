"""Gate (D6.7): every evidence transfer leaves a complete, typed ledger.

Two estimands differing in ``window`` and ``level`` are transferred, one
correction operator per facet is applied, and the resolved transfer's
ledger must: be non-empty; carry a typed ``Assumption`` and a
``counterfactual`` on every line; list every differing facet exactly once;
and pass ``Ledger.check_complete``. Removing a facet's line must make
``check_complete`` name that facet.
"""

from __future__ import annotations

from _factories import _estimand

from axiom.calibrate.ledger import Ledger
from axiom.calibrate.transfer import (
    Correction,
    aggregation_level,
    carryover_window_factor,
    resolve,
)
from axiom.core import Assumption, TimeWindow
from axiom.estimands import Level
from axiom.surface import GeometricCarryover


def _resolved():  # type: ignore[no-untyped-def]
    source = _estimand(
        name="experiment",
        window=TimeWindow(start=0, stop=2, basis="cumulative"),
        level=Level(unit="individual"),
    )
    target = _estimand(
        name="decision",
        window=TimeWindow(start=0, stop=8, basis="cumulative"),
        level=Level(unit="cluster"),
    )
    plan = source.transfer_to(target)
    assert plan.differing == ("window", "level")
    window = carryover_window_factor(
        GeometricCarryover(max_lag=6), {"lam_fertilizer": 0.7}, 2, treatment="fertilizer"
    )
    level = aggregation_level(source.level, target.level, cluster_size=25, icc=0.05)
    assert isinstance(window, Correction) and isinstance(level, Correction)
    return plan, resolve(plan, corrections=[window, level]), (window, level)


def test_resolved_transfer_ledger_is_complete_and_typed() -> None:
    plan, resolved, corrections = _resolved()
    ledger = resolved.ledger
    assert ledger.lines, "the ledger must not be empty"
    for line in ledger.lines:
        assert isinstance(line.assumption, Assumption), line
        assert "counterfactual" in line.detail and "value" in line.detail, line
    for c in corrections:
        assert isinstance(c.ledger_line.assumption, Assumption)
        assert c.ledger_line.assumption.facet == c.facet
    assert ledger.facets_covered() == {f: 1 for f in plan.differing}
    verdict = ledger.check_complete(plan)
    assert verdict.status == "identified", verdict.reason
    assert resolved.completeness == verdict
    assert resolved.status == "downgraded" and resolved.licensed
    # the numbers on the line are the numbers the correction computed
    for c in corrections:
        (line,) = [ln for ln in ledger.lines if ln.kind == f"facet:{c.facet}"]
        assert float(line.detail["counterfactual"]) == c.counterfactual
        assert float(line.detail["value"]) == c.corrected


def test_dropping_a_facet_line_is_reported_by_name() -> None:
    plan, resolved, _ = _resolved()
    for facet in plan.differing:
        pruned = Ledger(
            lines=tuple(ln for ln in resolved.ledger.lines if ln.kind != f"facet:{facet}")
        )
        verdict = pruned.check_complete(plan)
        assert verdict.status != "identified"
        assert f"{facet}: missing" in verdict.reason
        others = [f for f in plan.differing if f != facet]
        assert all(f"{o}:" not in verdict.reason for o in others)
    # a facet recorded twice is just as incomplete
    doubled = resolved.ledger.append(resolved.ledger.lines[0])
    assert "duplicated" in doubled.check_complete(plan).reason
    # a line stripped of its counterfactual is incomplete
    line = resolved.ledger.lines[0]
    stripped = line.model_copy(
        update={"detail": {k: v for k, v in line.detail.items() if k != "counterfactual"}}
    )
    bad = Ledger(lines=(stripped, *resolved.ledger.lines[1:])).check_complete(plan)
    assert "no counterfactual" in bad.reason


def test_plan_without_corrections_is_explicitly_uncorrected_not_silent() -> None:
    plan, _, _ = _resolved()
    bare = resolve(plan)
    assert bare.completeness.status == "identified"
    assert all(ln.detail["correction"] == "none" for ln in bare.ledger.lines)
    assert all("uncorrected" in ln.detail["counterfactual"] for ln in bare.ledger.lines)
