"""One experiment, from a plan to a filed readout, in the vocabulary each layer speaks.

Everything the last dozen notes built is a `Spec` that somebody has to put
somewhere. `design.assign` returns an `ArmAssignment`; `design.collision` an
`Occupancy`; `design.stopped_estimate` a corrected pair; `io.ExperimentRun` has
a `roles` mapping and a lifecycle waiting for them; `io.DefinitionRegistry`
wants the outcome the analysis actually used. Every one of those notes ends with
some version of *"nothing files it yet"*, and this is the thing that does.

`ExperimentBuilder` is a facade over an `io.Catalog` scope. It speaks in the
typed objects each layer produces and writes the hashes an `ExperimentRun`
records, so the caller never handles a digest:

    builder = ExperimentBuilder(catalog, program)
    run = builder.plan("NW-14", estimand=lift, roles={"assignment": assigned.spec,
                                                      "occupancy": occupancy,
                                                      "schedule": schedule})
    run = builder.commit(run)
    run = builder.start(run)
    run = builder.read(run, measurement)
    builder.file(run)

**It adds no statistics.** Every number it moves was computed by the module that
owns it. What it adds is that the plan, the assignment, the occupancy, the
readout and the correction end up in one scope under one experiment id, with the
`ExperimentRun` conformance verdict standing over them —
which is the whole argument of
``docs/notes/0027-scope-and-the-experiment-lifecycle.md`` and was, until now,
something a caller had to assemble by hand.

**The one place it does more than plumb** is :meth:`read_stopped`. A study that
crossed a boundary has a biased naive estimate; ``design.stopped_estimate``
corrects it. Filing the corrected number *and* a ``"stopped_early"``
``Deviation`` naming the naive one is the pair of moves that keeps the two
facts together, and doing them separately is how they come apart.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from axiom.calibrate import Measurement
from axiom.core import Spec
from axiom.design import (
    MonitoringPath,
    Readout,
    StoppedEstimate,
    stopped_estimate,
)
from axiom.estimands import Estimand
from axiom.io import (
    Catalog,
    Definition,
    DefinitionRegistry,
    ExperimentRun,
    Program,
    ProgramStore,
)

__all__ = ["ExperimentBuilder", "readouts_across"]


@dataclass(frozen=True)
class ExperimentBuilder:
    """One party's experiments: plan them, run them, read them, file them.

    Frozen and stateless — every method takes the run it is acting on and
    returns the next one, exactly as ``io.ExperimentRun`` does. The builder
    holds only where things go.
    """

    catalog: Catalog
    program: Program

    @property
    def store(self) -> ProgramStore:
        return self.catalog.store(self.program)

    @property
    def definitions(self) -> DefinitionRegistry:
        return DefinitionRegistry(self.catalog)

    # -- planning -----------------------------------------------------------

    def plan(
        self,
        experiment: str,
        *,
        estimand: Estimand,
        roles: Mapping[str, Spec],
        at: str | None = None,
    ) -> ExperimentRun:
        """File every role spec in this scope and return a run at ``designed``.

        ``roles`` are the typed objects the plan is made of — an
        ``ArmAssignment``, an ``Occupancy``, a ``Schedule``, a
        ``DesignCandidate``. Each is stored under its role name and the run
        records its hash, so the run stays small and the specs stay findable.
        """
        if not roles:
            raise ValueError(f"experiment {experiment!r} needs at least one planned role")
        store = self.store
        store.put(estimand, label=f"{experiment}:estimand", tags={"experiment": experiment})
        for role, spec in roles.items():
            store.put(spec, label=f"{experiment}:{role}", tags={"experiment": experiment})
        return ExperimentRun.design(
            experiment,
            scope=self.program.scope,
            roles=roles,
            estimand=estimand,
            at=at,
        )

    def commit(self, run: ExperimentRun, *, at: str | None = None, note: str = "") -> ExperimentRun:
        """Freeze the plan and file the committed run."""
        committed = run.commit(at=at, note=note)
        self.file(committed)
        return committed

    def start(self, run: ExperimentRun, *, at: str | None = None) -> ExperimentRun:
        return run.start(at=at)

    # -- reading ------------------------------------------------------------

    def read(
        self,
        run: ExperimentRun,
        measurement: Measurement,
        *,
        at: str | None = None,
    ) -> ExperimentRun:
        """File the readout and attach it, with the estimand it actually answers."""
        store = self.store
        store.put(
            measurement,
            label=f"{run.experiment}:measurement",
            tags={"experiment": run.experiment},
        )
        read = run.read(measurement, estimand=measurement.estimand, at=at)
        self.file(read)
        return read

    def read_stopped(
        self,
        run: ExperimentRun,
        path: MonitoringPath,
        estimand: Estimand,
        *,
        mass: float = 0.95,
        method: str = "",
        at: str | None = None,
    ) -> tuple[ExperimentRun, StoppedEstimate]:
        """Correct a boundary-stopped readout, file the correction, and record the departure.

        Two moves that belong together: the ``Measurement`` carries the
        median-unbiased estimate rather than the naive one, and a
        ``"stopped_early"`` ``Deviation`` records what the naive number was and
        why it was replaced. ``ExperimentRun.conformance`` then reports
        ``downgraded`` rather than ``identified``, which is the honest status
        for a study that did not run its planned length.

        Raises when the path has no correctable readout or carried no standard
        error — a corrected estimate with no scale is not a measurement.
        """
        correction = stopped_estimate(path, mass=mass)
        if not isinstance(correction, StoppedEstimate):
            raise ValueError(f"this path cannot be corrected: {correction.reason}")
        if correction.effect is None or correction.se_full is None:
            raise ValueError(
                f"the stopping look of {run.experiment!r} carried no standard error, so the "
                "corrected drift cannot be put on the estimand's scale"
            )
        half = correction.effect_interval.half_width if correction.effect_interval else 0.0
        if half <= 0.0:  # pragma: no cover - a degenerate interval
            raise ValueError("the corrected interval has no width")
        measurement = Measurement(
            estimand=estimand,
            estimate=correction.effect,
            se=half / 1.959963984540054,
            definition="wald",
            mass=mass,
            method=method or path.rule.name,
            source=run.experiment,
        )
        store = self.store
        store.put(
            correction,
            label=f"{run.experiment}:stopped_estimate",
            tags={"experiment": run.experiment},
        )
        store.put(
            measurement,
            label=f"{run.experiment}:measurement",
            tags={"experiment": run.experiment},
        )
        deviated = run.deviate(
            "stopped_early",
            correction,
            reason=(
                f"{path.rule.name} stopped at look {correction.look + 1}; the naive drift "
                f"{correction.naive_drift:.4g} is replaced by the median-unbiased "
                f"{correction.drift:.4g}"
            ),
            state="asserted",
            at=at,
        )
        read = deviated.read(measurement, estimand=estimand, at=at)
        read = read.with_ledger_line(correction.ledger_line())
        self.file(read)
        return read, correction

    # -- filing -------------------------------------------------------------

    def file(self, run: ExperimentRun) -> str:
        """Store the run in this scope, tagged by experiment and stage."""
        return self.store.put(
            run,
            label=run.experiment,
            tags={"experiment": run.experiment, "stage": run.stage},
        )

    def runs(self, *, stage: str = "") -> tuple[ExperimentRun, ...]:
        """Every filed run in this scope, latest version of each experiment first.

        ``stage`` restricts to runs filed at that stage. Runs are returned
        newest-first within an experiment, because a re-filed run supersedes
        its earlier versions and the newest is the one anybody wants.
        """
        tags = {"stage": stage} if stage else None
        rows = self.store.find(type_name="ExperimentRun", tags=tags)
        out: list[ExperimentRun] = []
        for row in rows:
            spec = self.store.get(row.digest)
            if isinstance(spec, ExperimentRun):
                out.append(spec)
        return tuple(reversed(out))

    def readouts(self, *, metric: str = "primary") -> tuple[Readout, ...]:
        """The filed runs that have been read, as ``design.program`` readouts.

        One per experiment, from its latest filed version, and only for runs
        whose conformance is licensed — a readout the run itself calls
        ``blocked`` is not evidence about anything and quietly including it in
        a programme's error control would be the failure ``io.ExperimentRun``
        exists to prevent.

        The ``Readout`` carries no p-value or e-value: those come from the
        analysis, not from the lifecycle. Use
        ``dataclasses.replace``-style ``model_copy`` to attach one, or build
        the readouts directly when the numbers are to hand.
        """
        seen: set[str] = set()
        out: list[Readout] = []
        for run in self.runs():
            if run.experiment in seen or not run.readout_hash:
                continue
            if not run.conformance().licensed:
                continue
            seen.add(run.experiment)
            out.append(
                Readout(
                    experiment=run.experiment,
                    party=self.program.party,
                    metric=metric,
                    p_value=1.0,
                )
            )
        return tuple(out)

    # -- definitions --------------------------------------------------------

    def define(self, name: str, spec: Spec, *, note: str = "", at: str | None = None) -> Definition:
        """Register what this party means by ``name``, in this party's scope."""
        return self.definitions.register(self.program, name, spec, note=note, at=at)

    def define_outcome(self, estimand: Estimand, *, note: str = "") -> Definition:
        """Register the estimand's own ``Outcome`` under its name.

        The common case, and the one nothing did: an analysis that has an
        estimand already knows what outcome it used, and registering it is what
        makes ``meta.commensurable``'s definition check possible later.
        """
        return self.define(estimand.outcome.name, estimand.outcome, note=note)

    def __repr__(self) -> str:
        return f"ExperimentBuilder({self.program.scope!r}, {len(self.runs())} filed run(s))"


def readouts_across(
    builders: Sequence[ExperimentBuilder], *, metric: str = "primary"
) -> tuple[Readout, ...]:
    """Every party's licensed readouts, in the order the builders were given.

    The input ``design.program`` and ``design.online`` take, assembled from the
    lifecycle rather than from a spreadsheet.
    """
    return tuple(r for builder in builders for r in builder.readouts(metric=metric))
