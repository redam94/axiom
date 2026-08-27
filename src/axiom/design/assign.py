"""Putting units into arms: a rule that can be re-derived, not a roster somebody kept.

``design.cluster.match_clusters`` pairs clusters and flips a coin inside each
pair. That is a design calculation. What was missing is the other half — the
thing that runs on the day, decides which arm a unit is in, and can be asked
six months later to prove it decided that (note 0027, §D27.6.1).

Three methods, in the order a design reaches for them:

* ``"hash"`` — the arm is a pure function of the unit's id and a salt. No
  state, no roster, no ordering: a unit that appears for the first time in week
  three lands in the arm it would always have landed in, two systems that never
  talk agree, and re-deriving the whole assignment is running the same function
  again. The cost is that the realized split is only approximately the target
  one, by the binomial.
* ``"block"`` — permuted blocks within each stratum, so the realized split is
  *exact* at the end of every block. The cost is that it needs the roster up
  front and an ordering, both of which are recorded.
* ``"rerandomize"`` — draw block assignments until covariate imbalance is under
  a threshold (Morgan and Rubin 2012). The cost is stated on the result and
  repeated here because it is the one people forget: re-randomization narrows
  the estimator's sampling distribution, so a Wald interval computed as though
  the design were completely randomized is **conservative** rather than wrong.
  Adjusting for the balancing covariates, or testing over the accepted set,
  recovers the power. ``RERANDOMIZED`` says so on every result that used it.

**What a result is.** ``assign`` returns an ``Assigned`` — a frozen carrier
holding the per-unit arms as an array — whose ``spec`` is an ``ArmAssignment``:
the *rule* plus the counts, the balance table, and a content hash of the
roster. The rule is what travels (it is a ``Spec``, so it hashes and
round-trips); the roster does not go into it, because a spec holds no large
arrays. ``Assigned.verify()`` re-derives the arms from the rule and returns a
``Verdict`` — ``identified`` when they reproduce exactly, ``blocked`` naming
the first unit that does not.

The balance numbers are standardized differences and are *not* a proof of
exchangeability, the same caveat ``cluster.Assignment`` carries.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import model_validator

from axiom.core import Assumption, LedgerLine, NonEmptyStr, Spec, Verdict

__all__ = [
    "RERANDOMIZED",
    "ArmAllocation",
    "ArmAssignment",
    "AssignMethod",
    "Assigned",
    "BalanceRow",
    "arm_for",
    "assign",
    "bucket",
    "standardized_differences",
]

Array = npt.NDArray[np.float64]
AssignMethod = Literal["hash", "block", "rerandomize"]

_MAX_BLOCK = 200
_DENOMINATOR = 1000

RERANDOMIZED = Assumption(
    name="rerandomized",
    facet="quantity",
    statement=(
        "the assignment was drawn until covariate imbalance fell under a threshold, so its "
        "randomization distribution is not the complete-randomization one a Wald interval "
        "assumes; the interval is conservative rather than wrong"
    ),
    challenged_by=(
        "an analysis that adjusts for the covariates the design balanced on, or a "
        "randomization test over the set of assignments the threshold would have accepted"
    ),
    state="asserted",
)


def bucket(unit: str, *, salt: str = "") -> float:
    """A stable uniform draw in ``[0, 1)`` for one unit.

    blake2b over ``salt`` and the unit id, the same hash ``Spec.content_hash``
    uses. It does not depend on the process, the roster, or the order units
    arrive in, which is the whole point: it is the only part of an assignment
    that needs nothing kept.
    """
    payload = f"{salt}\x00{unit}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


class ArmAllocation(Spec):
    """The arms and the share of units each is meant to get.

    Shares are positive and sum to one. The first arm is the reference the
    balance table compares against — by convention the control arm, though
    nothing here requires one.
    """

    arms: tuple[NonEmptyStr, ...]
    shares: tuple[float, ...]

    @model_validator(mode="after")
    def _valid(self) -> ArmAllocation:
        if len(self.arms) < 2:
            raise ValueError(f"an allocation needs at least two arms, got {list(self.arms)}")
        if len(self.arms) != len(self.shares):
            raise ValueError(
                f"{len(self.arms)} arms and {len(self.shares)} shares; they must match"
            )
        if len(set(self.arms)) != len(self.arms):
            raise ValueError(f"arm names must be distinct: {list(self.arms)}")
        for arm, share in zip(self.arms, self.shares, strict=True):
            if not (math.isfinite(share) and share > 0.0):
                raise ValueError(f"share for arm {arm!r} must be finite and positive, got {share}")
        total = math.fsum(self.shares)
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError(f"shares must sum to 1, got {total!r}")
        return self

    @classmethod
    def equal(cls, *arms: str) -> ArmAllocation:
        """Equal shares over the named arms."""
        return cls(arms=tuple(arms), shares=tuple([1.0 / len(arms)] * len(arms)))

    @property
    def reference(self) -> str:
        """The first arm: what the balance table compares the others against."""
        return self.arms[0]

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    def expected_counts(self, n: int) -> tuple[float, ...]:
        """The counts the shares imply for ``n`` units — fractional, not rounded."""
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        return tuple(n * s for s in self.shares)

    def edges(self) -> tuple[float, ...]:
        """Cumulative shares: the bucket boundaries ``arm_for`` compares against."""
        out, running = [], 0.0
        for share in self.shares:
            running += share
            out.append(running)
        out[-1] = 1.0
        return tuple(out)

    def block_size(self) -> int:
        """The smallest block in which every arm gets a whole number of units.

        Raises when the shares need a block larger than 200 — that is a sign
        the shares were meant to be approximate, and ``"hash"`` is the method
        for approximate shares.
        """
        denominators = [
            Fraction(s).limit_denominator(_DENOMINATOR).denominator for s in self.shares
        ]
        size = 1
        for d in denominators:
            size = size * d // math.gcd(size, d)
        if size > _MAX_BLOCK:
            raise ValueError(
                f"shares {list(self.shares)} need a block of {size} units for whole counts, "
                f"which is over the {_MAX_BLOCK} limit; use method='hash' for shares that are "
                "only meant to hold approximately, or round the shares"
            )
        return size


def arm_for(unit: str, allocation: ArmAllocation, *, salt: str = "") -> str:
    """The arm one unit belongs to under hash assignment — stateless and total.

    This is the function an assignment service calls. It needs no roster and no
    previous call, so a unit seen for the first time long after the design was
    written lands where it always would have.
    """
    draw = bucket(unit, salt=salt)
    for arm, edge in zip(allocation.arms, allocation.edges(), strict=True):
        if draw < edge:
            return arm
    return allocation.arms[-1]


class BalanceRow(Spec):
    """One covariate's means by arm and its standardized difference.

    ``smd`` is the largest absolute standardized difference between any arm and
    the allocation's reference arm, standardized by the covariate's sd over all
    units. It is a description of one draw, not evidence that the draw was
    exchangeable.
    """

    covariate: NonEmptyStr
    means: tuple[float, ...]
    smd: float
    sd: float

    @model_validator(mode="after")
    def _finite(self) -> BalanceRow:
        if not all(math.isfinite(m) for m in self.means):
            raise ValueError(f"{self.covariate}: arm means must be finite, got {self.means}")
        if not (math.isfinite(self.smd) and self.smd >= 0.0):
            raise ValueError(f"{self.covariate}: smd must be finite and non-negative")
        return self


def standardized_differences(
    covariates: Mapping[str, npt.ArrayLike],
    arm_of: npt.ArrayLike,
    allocation: ArmAllocation,
) -> tuple[BalanceRow, ...]:
    """The balance table: per covariate, the arm means and the worst standardized gap.

    A covariate that is constant over the units has ``sd == 0`` and an ``smd``
    of ``0``: there is nothing to be imbalanced about. An arm with no units
    gets ``nan`` for its mean and is skipped by the gap, which is reported
    rather than silently treated as balanced.
    """
    index = np.asarray(arm_of, dtype=np.int64)
    rows: list[BalanceRow] = []
    for name in sorted(covariates):
        x = np.asarray(covariates[name], dtype=np.float64)
        if x.shape != index.shape:
            raise ValueError(f"covariate {name!r} has {x.shape} values for {index.shape} units")
        if not np.all(np.isfinite(x)):
            raise ValueError(f"covariate {name!r} must be finite")
        sd = float(x.std(ddof=1)) if x.size > 1 else 0.0
        means = []
        for a in range(allocation.n_arms):
            members = x[index == a]
            means.append(float(members.mean()) if members.size else float("nan"))
        if sd > 0.0:
            gaps = [
                abs(m - means[0]) / sd
                for m in means[1:]
                if math.isfinite(m) and math.isfinite(means[0])
            ]
            smd = max(gaps) if gaps else 0.0
        else:
            smd = 0.0
        rows.append(
            BalanceRow(
                covariate=name,
                means=tuple(0.0 if math.isnan(m) else m for m in means),
                smd=float(smd),
                sd=sd,
            )
        )
    return tuple(rows)


class ArmAssignment(Spec):
    """The rule an assignment followed, plus what it produced.

    Everything needed to re-derive the arms is here: the ``allocation``, the
    ``method``, the ``salt`` (hash) or ``seed`` (block, rerandomize), the
    stratum labels' own hash, and ``roster_hash`` over the unit ids in the
    order they were given. What is *not* here is the roster itself — a spec
    holds no large arrays — so ``Assigned`` carries that and ``verify`` puts
    the two back together.

    ``balance_met`` is ``False`` when re-randomization ran out of draws before
    reaching ``threshold``; the assignment returned is then the best of the
    draws and says so rather than passing for one that met the bar.
    """

    allocation: ArmAllocation
    method: AssignMethod
    n_units: int
    roster_hash: str
    counts: tuple[int, ...]
    salt: str = ""
    seed: int | None = None
    strata_hash: str = ""
    n_strata: int = 0
    block: int = 0
    balance: tuple[BalanceRow, ...] = ()
    threshold: float | None = None
    draws_used: int = 1
    balance_met: bool = True
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _valid(self) -> ArmAssignment:
        if len(self.counts) != self.allocation.n_arms:
            raise ValueError(f"{len(self.counts)} counts for {self.allocation.n_arms} arms")
        if sum(self.counts) != self.n_units:
            raise ValueError(f"counts sum to {sum(self.counts)}, not n_units {self.n_units}")
        if self.method == "hash" and self.seed is not None:
            raise ValueError("hash assignment is decided by the salt; it takes no seed")
        if self.method != "rerandomize" and not self.balance_met:
            raise ValueError("only re-randomization can fail to meet a balance threshold")
        return self

    @property
    def worst_smd(self) -> float:
        """The largest standardized difference in the balance table; ``0`` with no covariates."""
        return max((row.smd for row in self.balance), default=0.0)

    def share_of(self, arm: str) -> float:
        """The realized share of one arm, against ``allocation`` for the target."""
        if arm not in self.allocation.arms:
            raise KeyError(f"no arm {arm!r}; have {list(self.allocation.arms)}")
        if self.n_units == 0:
            return 0.0
        return self.counts[self.allocation.arms.index(arm)] / self.n_units

    def ledger_line(self) -> LedgerLine:
        """One line saying how units reached their arms, and what that costs the analysis."""
        realized = ", ".join(f"{arm} {self.share_of(arm):.3f}" for arm in self.allocation.arms)
        statement = (
            f"{self.n_units} units assigned by {self.method} over {self.allocation.n_arms} arms "
            f"(realized {realized}); worst standardized difference {self.worst_smd:.3f}"
        )
        if self.method == "rerandomize" and not self.balance_met:
            statement += (
                f" — threshold {self.threshold} not met in {self.draws_used} draws; "
                "the best draw was kept"
            )
        return LedgerLine(
            kind="assignment",
            statement=statement,
            assumption=RERANDOMIZED if self.method == "rerandomize" else None,
            detail={
                "method": self.method,
                "salt": self.salt,
                "seed": "" if self.seed is None else str(self.seed),
                "roster_hash": self.roster_hash,
                "strata": str(self.n_strata),
                "draws_used": str(self.draws_used),
                "balance_met": str(self.balance_met),
                **self.detail,
            },
        )


@dataclass(frozen=True)
class Assigned:
    """``assign``'s return: the rule as a ``Spec``, and the arms as an array beside it."""

    spec: ArmAssignment
    units: tuple[str, ...]
    arm_of: npt.NDArray[np.int64]
    strata: tuple[str, ...] = ()
    covariates: Mapping[str, Array] | None = None

    def arm(self, unit: str) -> str:
        """The arm one unit is in."""
        try:
            i = self.units.index(unit)
        except ValueError:
            raise KeyError(f"no unit {unit!r} in this assignment") from None
        return self.spec.allocation.arms[int(self.arm_of[i])]

    def units_in(self, arm: str) -> tuple[str, ...]:
        """Every unit in one arm, in roster order."""
        if arm not in self.spec.allocation.arms:
            raise KeyError(f"no arm {arm!r}; have {list(self.spec.allocation.arms)}")
        a = self.spec.allocation.arms.index(arm)
        return tuple(u for u, i in zip(self.units, self.arm_of, strict=True) if int(i) == a)

    def counts(self) -> dict[str, int]:
        """Realized units per arm."""
        return dict(zip(self.spec.allocation.arms, self.spec.counts, strict=True))

    def verify(self) -> Verdict:
        """Re-derive the arms from the rule and say whether they reproduce.

        ``identified`` when every unit lands where it landed the first time;
        ``blocked`` naming the first unit that does not, which is what an audit
        six months later needs to see rather than a boolean.
        """
        redone = assign(
            self.units,
            self.spec.allocation,
            method=self.spec.method,
            salt=self.spec.salt,
            seed=self.spec.seed,
            strata=self.strata or None,
            covariates=self.covariates,
            threshold=self.spec.threshold,
            max_draws=self.spec.draws_used if self.spec.method == "rerandomize" else 1,
        )
        mismatch = np.flatnonzero(redone.arm_of != self.arm_of)
        if mismatch.size:
            first = int(mismatch[0])
            return Verdict(
                status="blocked",
                reason=(
                    f"re-deriving the assignment moved {mismatch.size} of {len(self.units)} "
                    f"units; the first is {self.units[first]!r}, recorded in "
                    f"{self.spec.allocation.arms[int(self.arm_of[first])]!r} and re-derived "
                    f"into {self.spec.allocation.arms[int(redone.arm_of[first])]!r}"
                ),
                route="assignment",
            )
        return Verdict(status="identified", route="assignment")

    def __len__(self) -> int:
        return len(self.units)

    def __repr__(self) -> str:
        counts = ", ".join(f"{k}={v}" for k, v in self.counts().items())
        return f"Assigned({self.spec.method}, {len(self)} units, {counts})"


def _roster_hash(units: Sequence[str]) -> str:
    h = hashlib.blake2b(digest_size=32)
    for unit in units:
        h.update(unit.encode())
        h.update(b"\x00")
    return h.hexdigest()


def _hash_arms(units: Sequence[str], allocation: ArmAllocation, salt: str) -> npt.NDArray[np.int64]:
    lookup = {arm: i for i, arm in enumerate(allocation.arms)}
    return np.asarray([lookup[arm_for(u, allocation, salt=salt)] for u in units], dtype=np.int64)


def _block_arms(
    units: Sequence[str],
    allocation: ArmAllocation,
    strata: Sequence[str],
    rng: np.random.Generator,
    block: int,
) -> npt.NDArray[np.int64]:
    """Permuted blocks within each stratum, walked in the roster's own order."""
    per_block = [round(block * s) for s in allocation.shares]
    pattern = np.repeat(np.arange(allocation.n_arms, dtype=np.int64), per_block)
    out = np.empty(len(units), dtype=np.int64)
    for stratum in sorted(set(strata)):
        members = [i for i, s in enumerate(strata) if s == stratum]
        drawn: list[int] = []
        while len(drawn) < len(members):
            drawn.extend(rng.permutation(pattern).tolist())
        for position, i in enumerate(members):
            out[i] = drawn[position]
    return out


def _worst_smd(
    columns: Mapping[str, Array], arm_of: npt.NDArray[np.int64], allocation: ArmAllocation
) -> float:
    rows = standardized_differences(columns, arm_of, allocation)
    return max((row.smd for row in rows), default=0.0)


def assign(
    units: Sequence[str],
    allocation: ArmAllocation,
    *,
    method: AssignMethod = "hash",
    salt: str = "",
    seed: int | None = None,
    strata: Sequence[str] | None = None,
    covariates: Mapping[str, npt.ArrayLike] | None = None,
    threshold: float | None = None,
    max_draws: int = 1000,
) -> Assigned:
    """Put units into arms, and record the rule that did it.

    ``method="hash"`` needs only ``salt`` and is a pure function of each unit's
    id. ``"block"`` and ``"rerandomize"`` need ``seed`` and use the roster's
    order, so both are reproducible from what the ``ArmAssignment`` records.
    ``strata`` (one label per unit) blocks within each stratum;
    ``covariates`` fills the balance table and, for ``"rerandomize"``, is what
    ``threshold`` is a bound on (the largest absolute standardized difference).

    Re-randomization draws until the threshold is met or ``max_draws`` is
    exhausted; when it is exhausted the best draw is returned with
    ``balance_met=False`` rather than an exception, because the best of a
    thousand draws is a usable design and a silent pass is not.
    """
    roster = tuple(str(u) for u in units)
    if not roster:
        raise ValueError("assign needs at least one unit")
    if len(set(roster)) != len(roster):
        duplicates = sorted({u for u in roster if roster.count(u) > 1})
        raise ValueError(f"unit ids must be distinct; repeated: {duplicates[:5]}")
    labels = tuple(str(s) for s in strata) if strata is not None else ("",) * len(roster)
    if len(labels) != len(roster):
        raise ValueError(f"{len(labels)} strata labels for {len(roster)} units")
    if method != "hash" and seed is None:
        raise ValueError(f"method={method!r} needs a seed; only 'hash' is seedless")
    if method == "hash" and seed is not None:
        raise ValueError("hash assignment is decided by the salt; it takes no seed")
    if method == "rerandomize":
        if not covariates:
            raise ValueError("re-randomization needs covariates to balance on")
        if threshold is None or not (math.isfinite(threshold) and threshold > 0.0):
            raise ValueError(f"re-randomization needs a positive threshold, got {threshold}")
        if max_draws < 1:
            raise ValueError(f"max_draws must be at least 1, got {max_draws}")

    columns = {k: np.asarray(v, dtype=np.float64) for k, v in (covariates or {}).items()}
    block = 0 if method == "hash" else allocation.block_size()
    draws_used, balance_met = 1, True

    if method == "hash":
        arm_of = _hash_arms(roster, allocation, salt)
    elif method == "block":
        arm_of = _block_arms(roster, allocation, labels, np.random.default_rng(seed), block)
    else:
        rng = np.random.default_rng(seed)
        best = _block_arms(roster, allocation, labels, rng, block)
        best_smd = _worst_smd(columns, best, allocation)
        balance_met = best_smd <= float(threshold or 0.0)
        while not balance_met and draws_used < max_draws:
            candidate = _block_arms(roster, allocation, labels, rng, block)
            worst = _worst_smd(columns, candidate, allocation)
            draws_used += 1
            if worst < best_smd:
                best, best_smd = candidate, worst
            balance_met = best_smd <= float(threshold or 0.0)
        arm_of = best

    counts = tuple(int(np.count_nonzero(arm_of == a)) for a in range(allocation.n_arms))
    detail: dict[str, str] = {}
    if method == "hash":
        detail["realized_vs_target"] = (
            "hash assignment splits by the binomial; the realized shares are only "
            "approximately the target ones"
        )
    spec = ArmAssignment(
        allocation=allocation,
        method=method,
        n_units=len(roster),
        roster_hash=_roster_hash(roster),
        counts=counts,
        salt=salt,
        seed=seed,
        strata_hash=_roster_hash(labels) if strata is not None else "",
        n_strata=len(set(labels)) if strata is not None else 0,
        block=block,
        balance=standardized_differences(columns, arm_of, allocation),
        threshold=threshold if method == "rerandomize" else None,
        draws_used=draws_used,
        balance_met=balance_met,
        detail=detail,
    )
    return Assigned(
        spec=spec,
        units=roster,
        arm_of=arm_of,
        strata=labels if strata is not None else (),
        covariates=columns or None,
    )
