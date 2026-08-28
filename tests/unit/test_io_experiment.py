from __future__ import annotations

from pathlib import Path

import pytest

from axiom.core import LedgerLine, TimeWindow
from axiom.io import (
    Catalog,
    Deviation,
    ExperimentRun,
    LifecycleError,
    Program,
    Stage,
)

_PLANNED = TimeWindow(start=0, stop=8)
_REALIZED = TimeWindow(start=0, stop=12)
_ESTIMAND = "e" * 64
_OTHER_ESTIMAND = "f" * 64
_READOUT = "m" * 64


def _designed() -> ExperimentRun:
    return ExperimentRun.design(
        "NW-1",
        scope="northwind/dose-response",
        roles={"window": _PLANNED},
        estimand=_ESTIMAND,
        at="2026-01-06T00:00:00+00:00",
    )


def _read(run: ExperimentRun, *, estimand: str = _ESTIMAND) -> ExperimentRun:
    return run.start(at="t1").read(_READOUT, estimand=estimand, at="t2")


# -- construction ---------------------------------------------------------------------------


def test_design_hashes_the_roles_it_is_given() -> None:
    run = _designed()
    assert run.stage == "designed"
    assert run.roles == {"window": _PLANNED.content_hash()}
    assert run.plan_hash == ""
    assert [t.stage for t in run.history] == ["designed"]


def test_a_spec_or_a_hash_may_be_given_for_the_estimand() -> None:
    by_hash = ExperimentRun.design("a", estimand=_PLANNED.content_hash())
    by_spec = ExperimentRun.design("a", estimand=_PLANNED)
    assert by_hash.estimand_hash == by_spec.estimand_hash


def test_roles_change_freely_before_commit_and_not_after() -> None:
    run = _designed().with_role("schedule", _REALIZED)
    assert set(run.roles) == {"window", "schedule"}
    committed = run.commit(at="t0")
    with pytest.raises(LifecycleError):
        committed.with_role("schedule", _PLANNED)


def test_commit_without_roles_is_refused() -> None:
    with pytest.raises(LifecycleError):
        ExperimentRun.design("empty", estimand=_ESTIMAND).commit()


def test_commit_freezes_the_plan_digest() -> None:
    run = _designed().commit(at="t0")
    assert run.plan_hash == run.plan_digest() != ""
    assert run.stage == "committed"


def test_the_plan_digest_covers_the_estimand_as_well_as_the_roles() -> None:
    a = _designed()
    b = a.model_copy(update={"estimand_hash": _OTHER_ESTIMAND})
    assert a.plan_digest() != b.plan_digest()


# -- the lifecycle --------------------------------------------------------------------------


def test_the_stages_run_in_order() -> None:
    run = _designed().commit(at="t0").start(at="t1").read(_READOUT, estimand=_ESTIMAND, at="t2")
    run = run.calibrated(at="t3").pooled(at="t4")
    assert run.stage == "pooled"
    assert [t.stage for t in run.history] == [
        "designed",
        "committed",
        "running",
        "read",
        "calibrated",
        "pooled",
    ]


@pytest.mark.parametrize("stage", ["running", "read", "calibrated", "pooled"])
def test_stages_cannot_be_skipped(stage: Stage) -> None:
    run = _designed()
    with pytest.raises(LifecycleError) as exc:
        getattr(run, {"running": "start", "read": "read"}.get(stage, stage))(
            *([_READOUT] if stage == "read" else [])
        )
    assert "designed" in str(exc.value)


def test_a_pooled_run_is_terminal() -> None:
    run = _read(_designed().commit(at="t0")).calibrated(at="t3").pooled(at="t4")
    with pytest.raises(LifecycleError, match="terminal"):
        run.abandon("too late")


def test_abandoning_takes_a_reason_and_ends_the_run() -> None:
    run = _designed().commit(at="t0")
    with pytest.raises(ValueError, match="takes a reason"):
        run.abandon("  ")
    abandoned = run.abandon("the site withdrew", at="t9")
    assert abandoned.stage == "abandoned"
    assert abandoned.history[-1].note == "the site withdrew"


def test_the_ledger_is_append_only_at_any_stage() -> None:
    line = LedgerLine(kind="note", statement="the field window slipped a week")
    run = _designed().with_ledger_line(line).commit(at="t0").with_ledger_line(line)
    assert run.ledger == (line, line)


# -- deviation ------------------------------------------------------------------------------


def test_deviation_before_commit_is_refused() -> None:
    with pytest.raises(LifecycleError):
        _designed().deviate("window", _REALIZED, reason="nope")


def test_deviation_moves_the_role_and_records_what_moved() -> None:
    run = _designed().commit(at="t0").start(at="t1")
    deviated = run.deviate("window", _REALIZED, reason="the field ran four weeks long", at="t9")
    assert deviated.roles["window"] == _REALIZED.content_hash()
    (dev,) = deviated.deviations
    assert dev.role == "window"
    assert dev.planned == _PLANNED.content_hash()
    assert dev.realized == _REALIZED.content_hash()
    assert dev.stage == "running" and dev.at == "t9"
    assert run.roles["window"] == _PLANNED.content_hash()  # the original is unchanged


def test_a_deviation_is_the_assumption_it_makes() -> None:
    dev = Deviation(
        role="window",
        planned="a" * 8,
        realized="b" * 8,
        reason="the field ran long",
        state="asserted",
    )
    assumption = dev.assumption()
    assert assumption.name == "deviation:window"
    assert assumption.state == "asserted"
    assert assumption.challenged_by
    assert assumption.detail["planned"] == "a" * 8


# -- conformance ----------------------------------------------------------------------------


def test_an_unread_run_is_unverified_not_identified() -> None:
    for run in (
        _designed(),
        _designed().commit(at="t0"),
        _designed().commit(at="t0").start(at="t"),
    ):
        verdict = run.conformance()
        assert verdict.status == "unverified" and verdict.route == "conformance"


def test_a_plan_read_as_committed_is_identified() -> None:
    verdict = _read(_designed().commit(at="t0")).conformance()
    assert verdict.status == "identified"
    assert verdict.assumptions == ()


def test_a_recorded_deviation_downgrades_and_names_its_assumption() -> None:
    run = _designed().commit(at="t0").start(at="t1")
    run = run.deviate("window", _REALIZED, reason="the field ran four weeks long")
    verdict = run.read(_READOUT, estimand=_ESTIMAND, at="t2").conformance()
    assert verdict.status == "downgraded"
    assert [a.name for a in verdict.assumptions] == ["deviation:window"]
    assert verdict.licensed


def test_an_unrecorded_change_to_a_committed_plan_is_blocked() -> None:
    run = _designed().commit(at="t0")
    tampered = run.model_copy(update={"roles": {"window": _REALIZED.content_hash()}})
    verdict = _read(tampered).conformance()
    assert verdict.status == "blocked"
    assert "unrecorded" in verdict.reason


def test_a_readout_for_another_estimand_is_blocked() -> None:
    verdict = _read(_designed().commit(at="t0"), estimand=_OTHER_ESTIMAND).conformance()
    assert verdict.status == "blocked"
    assert "different quantities" in verdict.reason


def test_a_readout_with_no_committed_plan_is_blocked() -> None:
    run = _designed().model_copy(update={"stage": "running"})
    verdict = run.read(_READOUT, estimand=_ESTIMAND, at="t2").conformance()
    assert verdict.status == "blocked"
    assert "never committed" in verdict.reason


def test_an_abandoned_run_is_blocked() -> None:
    verdict = _designed().commit(at="t0").abandon("the site withdrew").conformance()
    assert verdict.status == "blocked"


def test_the_readout_defaults_to_the_committed_estimand() -> None:
    run = _designed().commit(at="t0").start(at="t1").read(_READOUT, at="t2")
    assert run.readout_estimand == _ESTIMAND
    assert run.conformance().status == "identified"


# -- the run as an artifact -----------------------------------------------------------------


def test_a_run_round_trips_and_summarizes() -> None:
    run = _read(_designed().commit(at="t0"))
    assert ExperimentRun.from_json(run.to_json()) == run
    summary = run.summary()
    assert "NW-1" in summary and "identified" in summary
    assert str(run) == "NW-1 [read] identified"


def test_a_run_is_filed_in_its_scope_and_found_by_stage(tmp_path: Path) -> None:
    catalog = Catalog(tmp_path / "catalog")
    program = Program(party="northwind", program="dose-response")
    catalog.register(program)
    store = catalog.store(program)

    run = _designed().commit(at="t0")
    store.put(run, label=run.experiment, tags={"stage": run.stage})
    done = _read(run)
    store.put(done, label=done.experiment, tags={"stage": done.stage})

    assert len(store.find(type_name="ExperimentRun")) == 2
    (row,) = store.find(type_name="ExperimentRun", tags={"stage": "read"})
    loaded = store.get(row.digest)
    assert isinstance(loaded, ExperimentRun)
    assert loaded.conformance().status == "identified"
