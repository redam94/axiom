"""``ExperimentRun``: the object that binds a plan to the readout that answers it.

``build.StudyBuilder`` builds a plan into a ``SimulationSpec``, a
``DesignCandidate``, a ``Schedule``; after the study runs, ``build_measurement``
builds a ``calibrate.Measurement``. Nothing compares the two ends, so the
question "is this readout the analysis we said we would run?" has no answer in
the library — and across many parties that question is the premise every
comparison rests on. See ``docs/notes/0027-scope-and-the-experiment-lifecycle.md``.

A run is a ``Spec``: an id, a scope, a stage, a mapping of role to content hash,
and two append-only sequences — ``deviations`` and ``ledger``. Transitions are
methods returning a new run; an illegal one raises ``LifecycleError`` naming
both stages.

    designed → committed → running → read → calibrated → pooled

with ``abandoned`` reachable from any stage before ``pooled``.

``commit`` is the transition that matters: it freezes ``plan_hash`` over the
roles as they stand, and from then on a change to a planned role must go through
``deviate``, which requires a reason. ``conformance`` reads the result back as a
``core.Verdict`` in the vocabulary identification already uses — an unrecorded
change to a committed plan is ``blocked``, a recorded one is ``downgraded``
carrying the deviation as an ``Assumption``, and only an untouched plan read for
the estimand it was committed for is ``identified``.

The roles are ``Spec``s of any type, held by hash. That is what keeps this
module in ``io``, below every package whose objects it records; an ergonomic
front that takes the typed ``design`` objects belongs in ``build``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from collections.abc import Mapping
from typing import Literal

from axiom.core.result import NonEmptyStr
from axiom.core.spec import Spec
from axiom.core.verdict import Assumption, AssumptionState, LedgerLine, Verdict

__all__ = [
    "Deviation",
    "ExperimentRun",
    "LifecycleError",
    "Stage",
    "Transition",
]

Stage = Literal["designed", "committed", "running", "read", "calibrated", "pooled", "abandoned"]
"""Where a run is in its life. ``designed`` before the plan is frozen; ``pooled`` once its
readout has entered a corpus; ``abandoned`` when it will not be finished."""

_NEXT: Mapping[Stage, tuple[Stage, ...]] = {
    "designed": ("committed", "abandoned"),
    "committed": ("running", "abandoned"),
    "running": ("read", "abandoned"),
    "read": ("calibrated", "abandoned"),
    "calibrated": ("pooled", "abandoned"),
    "pooled": (),
    "abandoned": (),
}


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


class LifecycleError(ValueError):
    """A transition the lifecycle does not allow."""

    def __init__(self, experiment: str, *, frm: Stage, to: Stage) -> None:
        self.experiment, self.frm, self.to = experiment, frm, to
        allowed = ", ".join(_NEXT[frm]) or "nothing — it is terminal"
        super().__init__(
            f"experiment {experiment!r} is {frm!r} and cannot become {to!r}; "
            f"{frm!r} allows {allowed}"
        )


class Transition(Spec):
    """One stage change, with when it happened and why."""

    stage: Stage
    at: str
    note: str = ""


class Deviation(Spec):
    """A departure from the committed plan: what changed, from what, to what, and why.

    ``role`` is the planned role that moved (``"schedule"``, ``"candidate"``,
    ``"window"``), or a free name for a departure with no spec behind it
    (``"stopped_early"``). ``planned`` and ``realized`` are content hashes when
    the role has a spec and free text when it does not. ``state`` is how the
    deviation is carried into ``ExperimentRun.conformance``: ``asserted`` when
    an operator has said the readout is still the quantity that was planned,
    ``unverified`` (the default) when nobody has, ``violated`` when it is known
    to have changed the quantity.
    """

    role: NonEmptyStr
    planned: str
    realized: str
    reason: NonEmptyStr
    at: str = ""
    stage: Stage = "running"
    state: AssumptionState = "unverified"

    def assumption(self) -> Assumption:
        """The deviation as the assumption it silently makes."""
        return Assumption(
            name=f"deviation:{self.role}",
            facet="plan",
            statement=(
                f"{self.role} departed from the committed plan ({self.reason}); the readout "
                "is taken to answer the estimand the plan was committed for regardless"
            ),
            challenged_by=(
                "re-reading the experiment under the planned "
                f"{self.role}, or an estimand that names the realized one"
            ),
            state=self.state,
            detail={"planned": self.planned, "realized": self.realized, "stage": self.stage},
        )


class ExperimentRun(Spec):
    """One experiment's life, from a plan to a readout that can be compared to others.

    ``roles`` maps a role name to the content hash of the spec playing it —
    ``"candidate"``, ``"schedule"``, ``"simulation"``, ``"assignment"``,
    whatever the plan is made of. ``plan_hash`` is fixed at ``commit`` over
    ``roles`` and ``estimand_hash`` together; ``plan_digest`` recomputes it from
    the current roles, and the two disagreeing is the whole point of the object.
    """

    experiment: NonEmptyStr
    scope: str = ""
    stage: Stage = "designed"
    roles: dict[str, str] = {}
    estimand_hash: str = ""
    plan_hash: str = ""
    readout_hash: str = ""
    readout_estimand: str = ""
    deviations: tuple[Deviation, ...] = ()
    ledger: tuple[LedgerLine, ...] = ()
    history: tuple[Transition, ...] = ()

    # -- construction -------------------------------------------------------

    @classmethod
    def design(
        cls,
        experiment: str,
        *,
        scope: str = "",
        roles: Mapping[str, Spec] | None = None,
        estimand: Spec | str = "",
        at: str | None = None,
    ) -> ExperimentRun:
        """A run at ``designed``: the roles hashed, the plan not yet frozen.

        ``roles`` takes the specs themselves and stores their hashes;
        ``estimand`` takes either the ``Estimand`` spec or its hash.
        """
        return cls(
            experiment=experiment,
            scope=scope,
            roles={k: v.content_hash() for k, v in (roles or {}).items()},
            estimand_hash=estimand if isinstance(estimand, str) else estimand.content_hash(),
            history=(Transition(stage="designed", at=at or _now()),),
        )

    def plan_digest(self) -> str:
        """blake2b-256 over the current roles and estimand — what ``commit`` freezes."""
        payload = json.dumps(
            {"roles": dict(sorted(self.roles.items())), "estimand": self.estimand_hash},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=32).hexdigest()

    def with_role(self, role: str, spec: Spec | str) -> ExperimentRun:
        """Add or replace a planned role *before* the plan is committed.

        After ``commit`` this raises: a committed plan changes through
        ``deviate`` and nowhere else.
        """
        if self.stage != "designed":
            raise LifecycleError(self.experiment, frm=self.stage, to="designed")
        digest = spec if isinstance(spec, str) else spec.content_hash()
        return self.model_copy(update={"roles": {**self.roles, role: digest}})

    def with_ledger_line(self, line: LedgerLine) -> ExperimentRun:
        """Append a ledger line at any stage; the ledger is append-only."""
        return self.model_copy(update={"ledger": (*self.ledger, line)})

    # -- transitions --------------------------------------------------------

    def _to(
        self, stage: Stage, *, at: str | None, note: str = "", **fields: object
    ) -> ExperimentRun:
        if stage not in _NEXT[self.stage]:
            raise LifecycleError(self.experiment, frm=self.stage, to=stage)
        history = (*self.history, Transition(stage=stage, at=at or _now(), note=note))
        return self.model_copy(update={"stage": stage, "history": history, **fields})

    def commit(self, *, at: str | None = None, note: str = "") -> ExperimentRun:
        """Freeze the plan. Every later change to a planned role needs a ``Deviation``."""
        if not self.roles:
            raise LifecycleError(self.experiment, frm=self.stage, to="committed")
        return self._to("committed", at=at, note=note, plan_hash=self.plan_digest())

    def start(self, *, at: str | None = None, note: str = "") -> ExperimentRun:
        """The experiment is in the field."""
        return self._to("running", at=at, note=note)

    def read(
        self,
        measurement: Spec | str,
        *,
        estimand: Spec | str = "",
        at: str | None = None,
        note: str = "",
    ) -> ExperimentRun:
        """Attach the readout and the estimand it answers.

        A readout answering a different estimand than the plan was committed for
        is attached, not refused: the library's job is to record the mismatch —
        ``conformance`` turns ``blocked`` — not to decide it away.
        """
        digest = measurement if isinstance(measurement, str) else measurement.content_hash()
        answers = estimand if isinstance(estimand, str) else estimand.content_hash()
        return self._to(
            "read",
            at=at,
            note=note,
            readout_hash=digest,
            readout_estimand=answers or self.estimand_hash,
        )

    def calibrated(self, *, at: str | None = None, note: str = "") -> ExperimentRun:
        """The readout has been folded into a model through ``calibrate``."""
        return self._to("calibrated", at=at, note=note)

    def pooled(self, *, at: str | None = None, note: str = "") -> ExperimentRun:
        """The readout has entered a ``meta`` corpus."""
        return self._to("pooled", at=at, note=note)

    def abandon(self, reason: str, *, at: str | None = None) -> ExperimentRun:
        """The run will not be finished. Terminal, and the reason is required."""
        if not reason.strip():
            raise ValueError(f"experiment {self.experiment!r}: abandoning takes a reason")
        return self._to("abandoned", at=at, note=reason)

    # -- deviation ----------------------------------------------------------

    def deviate(
        self,
        role: str,
        realized: Spec | str,
        *,
        reason: str,
        state: AssumptionState = "unverified",
        at: str | None = None,
    ) -> ExperimentRun:
        """Record that a committed role moved, and move it.

        Refused before ``commit``: until the plan is frozen there is nothing to
        deviate from, and ``with_role`` is the way to change it.
        """
        if self.stage == "designed":
            raise LifecycleError(self.experiment, frm=self.stage, to="committed")
        digest = realized if isinstance(realized, str) else realized.content_hash()
        deviation = Deviation(
            role=role,
            planned=self.roles.get(role, ""),
            realized=digest,
            reason=reason,
            at=at or _now(),
            stage=self.stage,
            state=state,
        )
        return self.model_copy(
            update={
                "roles": {**self.roles, role: digest},
                "deviations": (*self.deviations, deviation),
            }
        )

    # -- the answer ---------------------------------------------------------

    def conformance(self) -> Verdict:
        """Whether this readout is the analysis the plan committed to.

        The four answers are identification's, and for the same reason transport
        borrows them: whether two readouts may be compared is a question about
        what licenses the comparison, and a ``downgraded`` answer must name at
        least one assumption.
        """
        if self.stage == "abandoned":
            return Verdict(
                status="blocked",
                reason="the run was abandoned and has no readout to compare",
                route="conformance",
            )
        if not self.readout_hash:
            return Verdict(
                status="unverified",
                reason=f"the run is {self.stage!r}; conformance is answerable once it is read",
                route="conformance",
            )
        if not self.plan_hash:
            return Verdict(
                status="blocked",
                reason="the plan was never committed; there is nothing to compare the readout to",
                route="conformance",
            )
        if self.readout_estimand != self.estimand_hash:
            return Verdict(
                status="blocked",
                reason=(
                    f"the readout answers estimand {self.readout_estimand[:12]}… and the plan was "
                    f"committed for {self.estimand_hash[:12]}…; these are different quantities"
                ),
                route="conformance",
            )
        drifted = self.plan_digest() != self.plan_hash
        if drifted and not self.deviations:
            return Verdict(
                status="blocked",
                reason=(
                    "the committed plan does not match the roles the run was read under, and no "
                    "deviation was filed; the change is unrecorded"
                ),
                route="conformance",
            )
        if self.deviations:
            return Verdict(
                status="downgraded",
                reason=f"{len(self.deviations)} recorded departure(s) from the committed plan",
                assumptions=tuple(d.assumption() for d in self.deviations),
                route="conformance",
            )
        return Verdict(status="identified", route="conformance")

    # -- reading ------------------------------------------------------------

    def summary(self) -> str:
        """One line per fact, for a notebook or a log."""
        verdict = self.conformance()
        lines = [
            f"experiment {self.experiment!r} in scope {self.scope or '(unscoped)'}",
            f"  stage      {self.stage}",
            f"  roles      {', '.join(sorted(self.roles)) or '(none)'}",
            f"  plan       {self.plan_hash[:16] or '(uncommitted)'}",
            f"  readout    {self.readout_hash[:16] or '(unread)'}",
            f"  deviations {len(self.deviations)}",
            f"  conformance {verdict.status}" + (f" — {verdict.reason}" if verdict.reason else ""),
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"{self.experiment} [{self.stage}] {self.conformance().status}"
