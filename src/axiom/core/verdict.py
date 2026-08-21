"""The shared status vocabulary, ``Assumption``, ``Verdict``, and ``LedgerLine``.

One home for the types that ``identify`` (identification verdicts and
transport), ``estimands`` (transfer plans), ``design/methods`` (method
assumptions such as parallel trends), and ``calibrate`` (the ledger) all
consume. The plan's review (06, A1) found the type invented three times; it
is defined once here so the ledger is checkable by type.

Status vocabulary, shared across the package:

* ``identified`` — the quantity is licensed without further assumption.
* ``downgraded`` — licensed only under named assumptions, listed.
* ``blocked`` — not licensed; ``reason`` says why.
* ``unsupported`` — the producer lacks a capability the request needs.
* ``unverified`` — an assumption the machinery cannot yet check. Reported as
  such, never as satisfied.
"""

from __future__ import annotations

from typing import Literal

from pydantic import field_validator

from axiom.core.result import NonEmptyStr
from axiom.core.spec import Spec

__all__ = [
    "Assumption",
    "AssumptionState",
    "LedgerLine",
    "Status",
    "Verdict",
]

Status = Literal["identified", "downgraded", "blocked", "unsupported", "unverified"]
AssumptionState = Literal["unverified", "satisfied", "violated", "asserted"]


class Assumption(Spec):
    """A named, falsifiable condition that licenses a step.

    ``facet`` is the estimand facet (or method) the assumption belongs to;
    ``challenged_by`` names what would falsify it. ``state`` defaults to
    ``unverified`` and is only ever moved to ``satisfied`` by code that checked
    it, or ``asserted`` by a user who said so with a ledger line.
    """

    name: NonEmptyStr
    facet: NonEmptyStr
    statement: NonEmptyStr
    challenged_by: str = ""
    state: AssumptionState = "unverified"
    detail: dict[str, str] = {}

    def asserted(self) -> Assumption:
        return self.model_copy(update={"state": "asserted"})

    def satisfied(self) -> Assumption:
        return self.model_copy(update={"state": "satisfied"})

    def violated(self) -> Assumption:
        return self.model_copy(update={"state": "violated"})


class Verdict(Spec):
    """An honest answer about whether a quantity is licensed, and under what.

    Invariants enforced at construction: a ``blocked`` or ``unsupported``
    verdict carries a non-empty ``reason``; a ``downgraded`` verdict carries at
    least one assumption; an ``identified`` verdict carries no unverified or
    violated assumption.
    """

    status: Status
    reason: str = ""
    assumptions: tuple[Assumption, ...] = ()
    route: str = ""

    @field_validator("reason")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    def model_post_init(self, __context: object) -> None:
        if self.status in ("blocked", "unsupported") and not self.reason:
            raise ValueError(f"a {self.status!r} verdict must carry a reason")
        if self.status == "downgraded" and not self.assumptions:
            raise ValueError("a 'downgraded' verdict must name at least one assumption")
        if self.status == "identified":
            bad = [a.name for a in self.assumptions if a.state in ("unverified", "violated")]
            if bad:
                raise ValueError(
                    f"an 'identified' verdict cannot carry unverified/violated assumptions: {bad}"
                )

    @property
    def licensed(self) -> bool:
        return self.status in ("identified", "downgraded")


class LedgerLine(Spec):
    """One entry in the assumption ledger.

    Every transfer of evidence across a boundary — a unit conversion, a facet
    difference bridged by an assumption, an explicit override — appends one.
    ``kind`` is a short machine token; ``statement`` is the human line;
    ``assumption`` is set when the line records a falsifiable assumption and
    unset for facts (a registered conversion factor). ``source`` and ``target``
    are content hashes of the specs on either side when there are two.
    """

    kind: NonEmptyStr
    statement: NonEmptyStr
    assumption: Assumption | None = None
    detail: dict[str, str] = {}
    source: str = ""
    target: str = ""
