from __future__ import annotations

from pathlib import Path

import pytest

from axiom.build import ExperimentBuilder, readouts_across
from axiom.calibrate import Measurement
from axiom.core import D, Intervention, Outcome, Population, TimeWindow, Treatment
from axiom.design import (
    ArmAllocation,
    LookSchedule,
    Occupancy,
    StoppedEstimate,
    StoppingRule,
    assign,
    monitor,
    pocock,
)
from axiom.estimands import Estimand, Level, Quantity
from axiom.io import Catalog, ExperimentRun, Program

_UNITS = tuple(f"u{i:04d}" for i in range(200))


def _estimand(name: str = "lift") -> Estimand:
    return Estimand(
        name=name,
        quantity=Quantity(kind="contrast"),
        treatment=Treatment(name="course", dimension=D.currency, unit="USD"),
        intervention=Intervention(doses={"course": 1.0}),
        reference=Intervention(doses={"course": 0.0}),
        outcome=Outcome(name="score", dimension=D.outcome, unit="pt", aggregation="mean"),
        population=Population(name="enrolled"),
        window=TimeWindow(start=0, stop=8),
        level=Level(unit="individual"),
        dimension=D.outcome,
    )


def _builder(tmp_path: Path, party: str = "northwind") -> ExperimentBuilder:
    catalog = Catalog(tmp_path / "catalog")
    program = Program(party=party, program="growth")
    catalog.register(program)
    return ExperimentBuilder(catalog, program)


def _roles() -> dict[str, object]:
    return {
        "assignment": assign(_UNITS, ArmAllocation.equal("control", "treated"), salt="s").spec,
        "occupancy": Occupancy(
            experiment="NW-14",
            units=("london", "leeds"),
            window=TimeWindow(start=0, stop=8),
            treatments=("course",),
        ),
    }


def _path() -> object:
    looks = LookSchedule(labels=("L1", "L2", "L3", "L4"), information=(0.25, 0.5, 0.75, 1.0))
    rule = StoppingRule(name="Pocock-4", looks=looks, boundaries=(pocock(0.05, looks),))
    return monitor(rule, [0.5, 0.9, 2.6], effects=[1.0, 1.8, 4.2], ses=[2.0, 1.4, 1.1])


# -- the clean path -------------------------------------------------------------------------


def test_a_plan_files_every_role_and_records_its_hash(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    lift = _estimand()
    run = builder.plan("NW-14", estimand=lift, roles=_roles(), at="t0")  # type: ignore[arg-type]
    assert isinstance(run, ExperimentRun)
    assert set(run.roles) == {"assignment", "occupancy"}
    assert run.estimand_hash == lift.content_hash()
    assert run.scope == "northwind/growth"
    # both specs are in the store, findable by experiment
    filed = builder.store.find(tags={"experiment": "NW-14"})
    assert len(filed) == 3  # estimand, assignment, occupancy
    for digest in run.roles.values():
        assert builder.store.get(digest) is not None


def test_the_whole_life_ends_identified(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    lift = _estimand()
    run = builder.plan("NW-14", estimand=lift, roles=_roles(), at="t0")  # type: ignore[arg-type]
    run = builder.commit(run, at="t1")
    run = builder.start(run, at="t2")
    run = builder.read(
        run,
        Measurement(estimand=lift, estimate=2.4, se=0.5, source="NW-14"),
        at="t3",
    )
    assert run.stage == "read"
    assert run.conformance().status == "identified"
    assert run.deviations == ()


def test_a_plan_needs_at_least_one_role(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    with pytest.raises(ValueError, match="at least one planned role"):
        builder.plan("NW-14", estimand=_estimand(), roles={})


# -- the one place it does more than plumb --------------------------------------------------


def test_a_stopped_readout_files_the_correction_and_the_departure(tmp_path: Path) -> None:
    """Two moves that belong together, and come apart when done separately."""
    builder = _builder(tmp_path)
    lift = _estimand()
    run = builder.plan("NW-15", estimand=lift, roles=_roles(), at="t0")  # type: ignore[arg-type]
    run = builder.start(builder.commit(run, at="t1"), at="t2")
    run, correction = builder.read_stopped(run, _path(), lift, at="t3")  # type: ignore[arg-type]

    assert isinstance(correction, StoppedEstimate)
    assert correction.drift < correction.naive_drift  # corrected toward the null
    assert run.stage == "read"
    (deviation,) = run.deviations
    assert deviation.role == "stopped_early" and deviation.state == "asserted"
    assert f"{correction.naive_drift:.4g}" in deviation.reason
    assert [line.kind for line in run.ledger] == ["stopped_estimate"]


def test_a_stopped_readout_is_downgraded_not_identified(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    lift = _estimand()
    run = builder.plan("NW-15", estimand=lift, roles=_roles(), at="t0")  # type: ignore[arg-type]
    run = builder.start(builder.commit(run, at="t1"), at="t2")
    run, _ = builder.read_stopped(run, _path(), lift, at="t3")  # type: ignore[arg-type]
    verdict = run.conformance()
    assert verdict.status == "downgraded"
    assert [a.name for a in verdict.assumptions] == ["deviation:stopped_early"]


def test_the_filed_measurement_carries_the_corrected_number(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    lift = _estimand()
    run = builder.plan("NW-15", estimand=lift, roles=_roles(), at="t0")  # type: ignore[arg-type]
    run = builder.start(builder.commit(run, at="t1"), at="t2")
    run, correction = builder.read_stopped(run, _path(), lift, at="t3")  # type: ignore[arg-type]
    (row,) = builder.store.find(type_name="Measurement", tags={"experiment": "NW-15"})
    filed = builder.store.get(row.digest)
    assert isinstance(filed, Measurement)
    assert filed.estimate == pytest.approx(correction.effect)
    assert filed.estimate != pytest.approx(correction.naive_effect)


def test_a_path_with_no_standard_error_cannot_be_put_on_the_estimand_scale(
    tmp_path: Path,
) -> None:
    builder = _builder(tmp_path)
    lift = _estimand()
    looks = LookSchedule(labels=("L1", "L2", "L3"), information=(0.3, 0.6, 1.0))
    rule = StoppingRule(name="P", looks=looks, boundaries=(pocock(0.05, looks),))
    bare = monitor(rule, [0.5, 0.9, 2.6])  # no ses
    run = builder.plan("NW-16", estimand=lift, roles=_roles(), at="t0")  # type: ignore[arg-type]
    run = builder.start(builder.commit(run, at="t1"), at="t2")
    with pytest.raises(ValueError, match="carried no standard error"):
        builder.read_stopped(run, bare, lift, at="t3")


def test_an_uncorrectable_path_says_why(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    lift = _estimand()
    looks = LookSchedule(labels=("L1", "L2", "L3"), information=(0.3, 0.6, 1.0))
    rule = StoppingRule(name="P", looks=looks, boundaries=(pocock(0.05, looks),))
    running = monitor(rule, [0.1], ses=[1.0])  # ran out of statistics
    run = builder.plan("NW-17", estimand=lift, roles=_roles(), at="t0")  # type: ignore[arg-type]
    run = builder.start(builder.commit(run, at="t1"), at="t2")
    with pytest.raises(ValueError, match="cannot be corrected"):
        builder.read_stopped(run, running, lift, at="t3")


# -- filing and collecting ------------------------------------------------------------------


def test_runs_come_back_newest_first_and_can_be_filtered(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    lift = _estimand()
    run = builder.plan("NW-14", estimand=lift, roles=_roles(), at="t0")  # type: ignore[arg-type]
    run = builder.commit(run, at="t1")
    run = builder.read(
        builder.start(run, at="t2"),
        Measurement(estimand=lift, estimate=2.4, se=0.5, source="NW-14"),
        at="t3",
    )
    assert [r.stage for r in builder.runs()] == ["read", "committed"]
    assert [r.stage for r in builder.runs(stage="read")] == ["read"]
    assert "1 filed run" not in repr(builder)  # two versions of one experiment
    assert "northwind/growth" in repr(builder)


def test_readouts_skip_a_run_its_own_conformance_blocks(tmp_path: Path) -> None:
    """A readout the run calls blocked is not evidence about anything."""
    builder = _builder(tmp_path)
    lift, other = _estimand(), _estimand("something_else")
    good = builder.plan("NW-14", estimand=lift, roles=_roles(), at="t0")  # type: ignore[arg-type]
    good = builder.read(
        builder.start(builder.commit(good, at="t1"), at="t2"),
        Measurement(estimand=lift, estimate=2.4, se=0.5, source="NW-14"),
        at="t3",
    )
    bad = builder.plan("NW-99", estimand=lift, roles=_roles(), at="t0")  # type: ignore[arg-type]
    bad = builder.read(
        builder.start(builder.commit(bad, at="t1"), at="t2"),
        Measurement(estimand=other, estimate=9.9, se=0.5, source="NW-99"),
        at="t3",
    )
    assert bad.conformance().status == "blocked"
    keys = {r.experiment for r in builder.readouts()}
    assert keys == {"NW-14"}


def test_readouts_across_parties_are_the_programmes_input(tmp_path: Path) -> None:
    lift = _estimand()
    builders = []
    for party in ("northwind", "acme"):
        builder = _builder(tmp_path / party, party)
        run = builder.plan("E-1", estimand=lift, roles=_roles(), at="t0")  # type: ignore[arg-type]
        builder.read(
            builder.start(builder.commit(run, at="t1"), at="t2"),
            Measurement(estimand=lift, estimate=2.4, se=0.5, source="E-1"),
            at="t3",
        )
        builders.append(builder)
    readouts = readouts_across(builders)
    assert [r.party for r in readouts] == ["northwind", "acme"]
    assert {r.key for r in readouts} == {"northwind/E-1:primary", "acme/E-1:primary"}


# -- definitions ----------------------------------------------------------------------------


def test_the_estimands_outcome_registers_itself(tmp_path: Path) -> None:
    builder = _builder(tmp_path)
    lift = _estimand()
    entry = builder.define_outcome(lift, note="from the NW-14 analysis")
    assert entry.name == "score" and entry.version == 1
    assert builder.definitions.get(builder.program, "score") == lift.outcome
    # registering the same analysis again is a no-op, not a version
    assert builder.define_outcome(lift).version == 1
