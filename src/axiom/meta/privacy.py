"""Privacy gates for publishing pooled statistics: k-anonymity, dominance, and an ε ledger.

New in axiom (roadmap M7.6; no parent module). Three objects:

* ``Cell`` — the contributions behind one published number: ``(contributor,
  value)`` pairs. Built from plain pairs or, lazily, from ``StudyRecord``s via
  ``cell_from_records`` (``contributor``/``estimate`` fields only).
* ``PrivacyPolicy`` + ``check_cell`` — a cell is **refused** (a ``blocked``
  ``Verdict``), never clipped or padded, when it has fewer than ``k`` distinct
  contributors, when any single contributor holds more than ``dominance_p`` of
  the cell's absolute total, or when the largest ``dominance_top_n``
  contributors together hold more than ``dominance_top_p`` of it (the
  classical *p%* and *(n,k)* dominance rules of statistical disclosure
  control, Hundepool et al. 2012).
* ``EpsilonLedger`` + ``charge`` — an immutable differential-privacy budget.
  Charges compose sequentially (basic composition: the ε of a sequence of
  releases is the sum of their ε). A charge that would exceed the budget
  returns ``Blocked`` and leaves the ledger unchanged; nothing is ever
  partially charged.

The ``[privacy]`` extra is nominal: this module imports nothing beyond core.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Literal

from pydantic import field_validator, model_validator

from axiom.core import Blocked, NonEmptyStr, Spec, Verdict

__all__ = [
    "Cell",
    "EpsilonCharge",
    "EpsilonLedger",
    "Mechanism",
    "PrivacyPolicy",
    "cell_from_records",
    "charge",
    "check_cell",
    "contributor_totals",
]

Mechanism = Literal["laplace", "gaussian"]


class Cell(Spec):
    """The ``(contributor, value)`` pairs behind one publishable statistic."""

    records: tuple[tuple[NonEmptyStr, float], ...]
    name: str = ""

    @field_validator("records")
    @classmethod
    def _finite(cls, v: tuple[tuple[str, float], ...]) -> tuple[tuple[str, float], ...]:
        if not v:
            raise ValueError("a cell needs at least one record")
        for c, x in v:
            if not math.isfinite(x):
                raise ValueError(f"value for contributor {c!r} must be finite, got {x}")
        return v

    @property
    def contributors(self) -> frozenset[str]:
        return frozenset(c for c, _ in self.records)

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(x for _, x in self.records)

    @property
    def n(self) -> int:
        return len(self.records)


def cell_from_records(records: Iterable[Any], *, name: str = "") -> Cell:
    """Build a ``Cell`` from objects exposing ``contributor`` and ``estimate`` (``StudyRecord``)."""
    pairs = tuple((str(r.contributor), float(r.estimate)) for r in records)
    return Cell(records=pairs, name=name)


class PrivacyPolicy(Spec):
    """Disclosure thresholds for one release programme.

    ``k``: minimum distinct contributors per cell. ``dominance_p``: the largest
    share of the cell's absolute total any one contributor may hold.
    ``dominance_top_n``/``dominance_top_p``: the largest share the top ``n``
    contributors together may hold. ``epsilon_total``: the ε budget.
    """

    k: int = 3
    dominance_p: float = 0.5
    dominance_top_n: int = 2
    dominance_top_p: float = 0.8
    epsilon_total: float = 1.0

    @model_validator(mode="after")
    def _ranges(self) -> PrivacyPolicy:
        if self.k < 2:
            raise ValueError(f"k must be at least 2, got {self.k}")
        for name in ("dominance_p", "dominance_top_p"):
            v = getattr(self, name)
            if not 0.0 < v <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {v}")
        if self.dominance_top_n < 1:
            raise ValueError("dominance_top_n must be at least 1")
        if not (math.isfinite(self.epsilon_total) and self.epsilon_total > 0):
            raise ValueError("epsilon_total must be finite and positive")
        return self


def contributor_totals(cell: Cell) -> dict[str, float]:
    """Absolute contribution per contributor: the sum of ``|value|`` over its records."""
    totals: defaultdict[str, float] = defaultdict(float)
    for c, x in cell.records:
        totals[c] += abs(x)
    return dict(totals)


def check_cell(cell: Cell, policy: PrivacyPolicy) -> Verdict:
    """``identified`` when the cell clears k-anonymity and both dominance rules; else ``blocked``.

    A failing cell is refused outright: no clipped, padded, or partial
    statistic is produced from it.
    """
    totals = contributor_totals(cell)
    m = len(totals)
    if m < policy.k:
        return Verdict(
            status="blocked",
            reason=f"cell has {m} contributor(s); policy requires at least {policy.k}",
            route="k_anonymity",
        )
    grand = sum(totals.values())
    if grand > 0:
        shares = sorted((v / grand for v in totals.values()), reverse=True)
        if shares[0] > policy.dominance_p:
            return Verdict(
                status="blocked",
                reason=(
                    f"one contributor holds {shares[0]:.3f} of the cell total; "
                    f"policy allows at most {policy.dominance_p}"
                ),
                route="dominance_p",
            )
        top = sum(shares[: policy.dominance_top_n])
        if policy.dominance_top_n < m and top > policy.dominance_top_p:
            return Verdict(
                status="blocked",
                reason=(
                    f"top {policy.dominance_top_n} contributors hold {top:.3f} of the cell "
                    f"total; policy allows at most {policy.dominance_top_p}"
                ),
                route="dominance_top_n",
            )
    return Verdict(status="identified", route="privacy_policy")


class EpsilonCharge(Spec):
    """One ε charge against the ledger, named by the release that made it (timestamp-free)."""

    release_id: NonEmptyStr
    epsilon: float
    mechanism: Mechanism
    delta: float = 0.0

    @model_validator(mode="after")
    def _positive(self) -> EpsilonCharge:
        if not (math.isfinite(self.epsilon) and self.epsilon > 0):
            raise ValueError("epsilon must be finite and positive")
        if not 0.0 <= self.delta < 1.0:
            raise ValueError("delta must be in [0, 1)")
        return self


class EpsilonLedger(Spec):
    """An immutable ε budget with its sequence of charges (basic sequential composition)."""

    budget: float
    charges: tuple[EpsilonCharge, ...] = ()

    @model_validator(mode="after")
    def _within_budget(self) -> EpsilonLedger:
        if not (math.isfinite(self.budget) and self.budget > 0):
            raise ValueError("budget must be finite and positive")
        if self.used > self.budget * (1.0 + 1e-12):
            raise ValueError(f"charges total {self.used} exceed budget {self.budget}")
        ids = [c.release_id for c in self.charges]
        if len(set(ids)) != len(ids):
            raise ValueError("release ids on a ledger must be unique")
        return self

    @property
    def used(self) -> float:
        return float(math.fsum(c.epsilon for c in self.charges))

    @property
    def remaining(self) -> float:
        return self.budget - self.used

    @property
    def delta_used(self) -> float:
        return float(math.fsum(c.delta for c in self.charges))


def charge(
    ledger: EpsilonLedger,
    *,
    release_id: str,
    epsilon: float,
    mechanism: Mechanism,
    delta: float = 0.0,
) -> EpsilonLedger | Blocked:
    """A new ledger with the charge appended, or ``Blocked`` (ledger untouched) if over budget."""
    if not (math.isfinite(epsilon) and epsilon > 0):
        raise ValueError("epsilon must be finite and positive")
    if epsilon > ledger.remaining * (1.0 + 1e-12) + 0.0:
        return Blocked(
            reason=(
                f"release {release_id!r} asks for epsilon={epsilon} but only "
                f"{ledger.remaining} of {ledger.budget} remains"
            ),
            detail={
                "requested": repr(epsilon),
                "remaining": repr(ledger.remaining),
                "budget": repr(ledger.budget),
            },
        )
    if any(c.release_id == release_id for c in ledger.charges):
        return Blocked(
            reason=f"release id {release_id!r} was already charged on this ledger",
            detail={"release_id": release_id},
        )
    line = EpsilonCharge(release_id=release_id, epsilon=epsilon, mechanism=mechanism, delta=delta)
    return EpsilonLedger(budget=ledger.budget, charges=(*ledger.charges, line))
