"""Two experiments in the same markets in the same weeks, and what that costs.

`design.methods` names ``no_interference`` — "a unit's outcome depends only on
its own treatment" — as an assumption of five of its six methods, and nothing
has ever checked it. Between units it is hard to check. Between *experiments* it
is arithmetic: two designs collide when they share units and share periods, and
a program that schedules them has the information to see it before either runs
(note 0027, §D27.6.2).

For a house running many experiments per party this is not an edge case. It is
the normal state, and the interesting part is that colliding is not
automatically wrong:

* **disjoint** — no shared unit, or no shared period. Nothing to say.
* **concurrent** — shared units and periods, different treatments, both
  independently randomized. Neither estimate is biased: independent
  randomizations are orthogonal in expectation. What it costs is *variance* —
  the other experiment's effect sits in your residual — and it puts
  ``no_interference`` genuinely at risk, because the two treatments may
  interact. :func:`variance_inflation` prices the first; only a factorial
  analysis settles the second.
* **confounded** — shared units and periods and a **shared treatment**, or a
  neighbour that was not independently randomized (a staged rollout, a
  hand-picked pilot). The same lever moved twice by two designs that each think
  they own it. Neither effect is separately identified and no assumption
  rescues it; :meth:`Collision.verdict` is ``blocked``.

The unit labels here are whatever a program actually schedules — markets,
clinics, cohorts, cells. Passing four hundred thousand individual ids would work
and would miss the point: collisions are decided at the granularity a calendar
is kept in.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from typing import Literal

from pydantic import model_validator

from axiom.core import Assumption, LedgerLine, NonEmptyStr, Spec, TimeWindow, Verdict

__all__ = [
    "CONCURRENT_EXPERIMENTS",
    "Collision",
    "CollisionKind",
    "Occupancy",
    "collisions",
    "exclusions",
    "variance_inflation",
]

CollisionKind = Literal["disjoint", "concurrent", "confounded"]

CONCURRENT_EXPERIMENTS = Assumption(
    name="concurrent_experiments",
    facet="intervention",
    statement=(
        "another experiment ran on the same units over the same periods, moving a different "
        "treatment; each estimate is the effect of its own treatment averaged over whatever "
        "the other one was doing, and the two treatments do not interact"
    ),
    challenged_by=(
        "a factorial analysis of the two assignments together: an interaction term that is "
        "not zero says the two effects are not each other's background"
    ),
)


class Occupancy(Spec):
    """What one experiment occupies while it runs: units, periods, and levers.

    ``units`` are the labels the program schedules on — markets, clinics,
    cohorts — not necessarily the units of analysis. ``treatments`` are the
    levers the design *moves*; two experiments measuring the same outcome
    without moving the same lever do not collide on that account.

    ``randomized`` says whether this experiment's assignment was drawn
    independently of everything else. A staged rollout, a hand-picked pilot, or
    an assignment inherited from another design is ``False``, and a neighbour
    that is not independently randomized is confounded with rather than merely
    concurrent to whatever it overlaps.
    """

    experiment: NonEmptyStr
    units: tuple[NonEmptyStr, ...]
    window: TimeWindow
    treatments: tuple[NonEmptyStr, ...] = ()
    randomized: bool = True

    @model_validator(mode="after")
    def _distinct(self) -> Occupancy:
        if not self.units:
            raise ValueError(f"experiment {self.experiment!r} occupies no units")
        if len(set(self.units)) != len(self.units):
            raise ValueError(f"experiment {self.experiment!r} repeats a unit label")
        if len(set(self.treatments)) != len(self.treatments):
            raise ValueError(f"experiment {self.experiment!r} repeats a treatment")
        return self

    def shared_units(self, other: Occupancy) -> tuple[str, ...]:
        return tuple(sorted(set(self.units) & set(other.units)))

    def shared_periods(self, other: Occupancy) -> tuple[int, ...]:
        low = max(self.window.start, other.window.start)
        high = min(self.window.stop, other.window.stop)
        return tuple(range(low, high))

    def shared_treatments(self, other: Occupancy) -> tuple[str, ...]:
        return tuple(sorted(set(self.treatments) & set(other.treatments)))


class Collision(Spec):
    """One pair of experiments that overlap, and what the overlap does to each.

    ``kind`` is the ladder in the module docstring. ``units`` and ``periods``
    are what they share; ``treatments`` is empty for a concurrent collision and
    non-empty for one confounded by a shared lever.
    """

    left: NonEmptyStr
    right: NonEmptyStr
    kind: CollisionKind
    units: tuple[str, ...] = ()
    periods: tuple[int, ...] = ()
    treatments: tuple[str, ...] = ()
    reason: str = ""

    @model_validator(mode="after")
    def _consistent(self) -> Collision:
        if self.left == self.right:
            raise ValueError(f"an experiment does not collide with itself: {self.left!r}")
        if self.kind == "disjoint" and (self.units and self.periods):
            raise ValueError("a disjoint pair shares no unit or no period")
        if self.kind != "disjoint" and not (self.units and self.periods):
            raise ValueError(f"a {self.kind!r} pair shares at least one unit and one period")
        if self.kind == "confounded" and not self.reason:
            raise ValueError("a confounded collision must say what confounds it")
        return self

    @property
    def pair(self) -> tuple[str, str]:
        return (self.left, self.right)

    def assumption(self) -> Assumption:
        """``CONCURRENT_EXPERIMENTS``, ``violated`` when the overlap is confounded."""
        detail = {
            "left": self.left,
            "right": self.right,
            "units": str(len(self.units)),
            "periods": f"{len(self.periods)}",
            "shared_treatments": ", ".join(self.treatments),
        }
        base = CONCURRENT_EXPERIMENTS.model_copy(update={"detail": detail})
        return base.violated() if self.kind == "confounded" else base

    def verdict(self) -> Verdict:
        """``identified`` when disjoint, ``downgraded`` when concurrent, ``blocked``
        when the two designs move the same lever over the same units and periods."""
        if self.kind == "disjoint":
            return Verdict(status="identified", route="schedule")
        if self.kind == "concurrent":
            return Verdict(
                status="downgraded",
                reason=(
                    f"{self.left} and {self.right} share {len(self.units)} unit(s) over "
                    f"{len(self.periods)} period(s); each is estimated over the other as "
                    "background"
                ),
                assumptions=(self.assumption(),),
                route="schedule",
            )
        return Verdict(status="blocked", reason=self.reason, route="schedule")

    def ledger_line(self) -> LedgerLine:
        return LedgerLine(
            kind="collision",
            statement=(
                f"{self.left} and {self.right}: {self.kind} over {len(self.units)} unit(s) "
                f"and {len(self.periods)} period(s)"
                + (f" sharing {', '.join(self.treatments)}" if self.treatments else "")
                + (f" — {self.reason}" if self.reason else "")
            ),
            assumption=None if self.kind == "disjoint" else self.assumption(),
            detail={
                "left": self.left,
                "right": self.right,
                "kind": self.kind,
                "units": ", ".join(self.units[:8]),
                "periods": f"[{self.periods[0]}, {self.periods[-1]}]" if self.periods else "",
            },
        )


def _classify(a: Occupancy, b: Occupancy) -> Collision:
    units = a.shared_units(b)
    periods = a.shared_periods(b)
    if not units or not periods:
        return Collision(left=a.experiment, right=b.experiment, kind="disjoint")
    shared = a.shared_treatments(b)
    if shared:
        return Collision(
            left=a.experiment,
            right=b.experiment,
            kind="confounded",
            units=units,
            periods=periods,
            treatments=shared,
            reason=(
                f"both designs move {', '.join(shared)} on the same units over the same "
                "periods, so neither treatment's effect is separately identified"
            ),
        )
    if not (a.randomized and b.randomized):
        unrandomized = ", ".join(x.experiment for x in (a, b) if not x.randomized)
        return Collision(
            left=a.experiment,
            right=b.experiment,
            kind="confounded",
            units=units,
            periods=periods,
            reason=(
                f"{unrandomized} was not independently randomized, so its assignment may "
                "correlate with the other's and the overlap is not orthogonal in expectation"
            ),
        )
    return Collision(
        left=a.experiment,
        right=b.experiment,
        kind="concurrent",
        units=units,
        periods=periods,
    )


def collisions(
    occupancies: Sequence[Occupancy], *, include_disjoint: bool = False
) -> tuple[Collision, ...]:
    """Every pair of experiments that overlap, classified.

    Pairs are compared in the order given and reported with the earlier
    experiment as ``left``. Disjoint pairs are omitted by default — a schedule
    of thirty experiments has four hundred of them and one interesting row.
    """
    names = [o.experiment for o in occupancies]
    if len(set(names)) != len(names):
        repeated = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"experiment names must be distinct; repeated: {repeated}")
    out = [_classify(a, b) for a, b in itertools.combinations(occupancies, 2)]
    if include_disjoint:
        return tuple(out)
    return tuple(c for c in out if c.kind != "disjoint")


def exclusions(
    found: Iterable[Collision], *, kinds: Sequence[CollisionKind] = ("confounded",)
) -> dict[str, tuple[str, ...]]:
    """A symmetric map of experiment to the experiments it may not run beside.

    Feeds ``design.recommend(exclusions=...)``. The default excludes only
    ``confounded`` pairs, because a concurrent pair is a cost rather than a
    conflict; pass ``kinds=("confounded", "concurrent")`` for a program that
    will not accept the variance either.
    """
    wanted = set(kinds)
    out: dict[str, set[str]] = {}
    for collision in found:
        if collision.kind not in wanted:
            continue
        out.setdefault(collision.left, set()).add(collision.right)
        out.setdefault(collision.right, set()).add(collision.left)
    return {name: tuple(sorted(others)) for name, others in sorted(out.items())}


def variance_inflation(effect: float, share: float, sd: float) -> float:
    """What a concurrent experiment costs the one beside it, as a variance factor.

    A neighbouring experiment that moves the outcome by ``effect`` on a
    ``share`` of your units adds a Bernoulli component to your residual:
    variance rises from ``sd²`` to ``sd² + share·(1 − share)·effect²``, and the
    factor returned is the ratio. Multiply a standard error by its square root,
    or an ``n`` by the factor itself, to price the collision in the units
    ``design.power`` speaks.

    It is an upper bound on the cost when the neighbour's effect is unknown and
    exactly right when it is: the neighbour's assignment is independent, so it
    contributes variance and not bias.
    """
    if not (math.isfinite(effect) and math.isfinite(sd) and sd > 0.0):
        raise ValueError(f"effect must be finite and sd finite and positive, got {effect}, {sd}")
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"share must be in [0, 1], got {share}")
    return 1.0 + share * (1.0 - share) * effect**2 / sd**2
