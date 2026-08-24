"""Group-sequential monitoring: stopping boundaries, their exact crossing probabilities,
and the decision a realized sequence of interim statistics produces.

A study that looks at its own data more than once and may stop on what it sees
is not a study whose fixed-sample error rate applies. This module supplies the
three things such a design needs, and nothing else:

1. **Boundaries** — thresholds on the interim test statistic, one per look,
   built either from a classic shape (:func:`pocock`, :func:`obrien_fleming`),
   from an alpha-spending function (:func:`alpha_spending`), or from a
   posterior-probability rule stated in words (:func:`harm_boundary`).
2. **Crossing probabilities** — the exact probability, under a stated drift,
   that each boundary is the one that stops the study, at each look
   (:func:`crossing_probabilities`, :func:`operating_characteristics`). No
   simulation: the numbers come from numerical integration of the canonical
   joint distribution, so the type I error a rule actually spends is a
   computed number rather than a claimed one.
3. **The decision** — :func:`monitor` walks a realized sequence of interim
   statistics against a rule and returns the look it stopped at, what it
   crossed, and a ledger line saying so.

**The canonical scale.** At look ``k`` the analysis produces an effect estimate
``d_k`` with standard error ``se_k``; the monitoring statistic is

    Z_k = d_k / se_k,   signed so that **positive means the treatment is better**.

Write ``t_k`` for the *information fraction* at look ``k`` — the share of the
study's final information already accumulated, which for a balanced two-arm
comparison with a common outcome variance is the share of the planned
observations. Under the canonical joint distribution (Scharfstein, Tsiatis and
Robins; the same limit that licenses a Wald interval at a single look), the
B-values ``B_k = Z_k · sqrt(t_k)`` are a Brownian motion observed at times
``t_k``:

    B_k ~ N(drift · t_k, t_k),    Cov(B_j, B_k) = min(t_j, t_k),

where ``drift`` is the Z the design expects at full information,
``drift = effect / se_at_full_information``. Under the null ``drift = 0``; a
fixed-sample design with 80 % power at two-sided 5 % has ``drift = 2.802``.
Everything below is arithmetic on that one object, which is why the same
functions serve a frequentist spending boundary and a Bayesian
posterior-probability rule: both are thresholds on ``Z``.

**Sign convention for thresholds.** ``Boundary.z`` holds *signed* thresholds on
the Z scale and ``side`` says what crossing means:

* ``"upper"`` — crossed when ``Z >= z_k``. An efficacy boundary; ``z_k > 0``.
* ``"lower"`` — crossed when ``Z <= z_k``. A harm boundary (``z_k`` strongly
  negative) or a futility boundary (``z_k`` near zero, possibly positive).
* ``"two_sided"`` — crossed when ``|Z| >= z_k``; ``z_k > 0`` is required.

A rule may carry several boundaries. When two fire at the same look the one
whose threshold is further out wins, which is what a monitoring committee does:
a statistic below the harm boundary is a harm stop even though it is also below
the futility boundary.

**Binding and non-binding.** ``Boundary.binding`` governs the *arithmetic*, not
the committee. A non-binding boundary is one the error-rate calculation
pretends does not exist — the conservative convention for futility, whose whole
point is that ignoring it can only lower the type I error. :func:`monitor`
always acts on every boundary; :func:`crossing_probabilities` takes
``binding_only`` (default ``True``), so the gap between the alpha a rule
nominally spends and the alpha it spends *as run* is a number you can print
rather than a footnote.

**The numerics.** The recursion is the standard one (Armitage, McPherson and
Rowe): carry the sub-density of ``B_k`` restricted to "no boundary crossed
yet", convolve it with the increment's normal density, and integrate the part
that lands outside the continuation region. Two details make it accurate.
The grid at each look is laid out *on* that look's continuation region, so the
integrand is smooth across it and the trapezoid rule is second order rather
than first; and every crossing probability is computed from the exact normal
tail given each grid point rather than by summing grid cells beyond the
threshold, so a threshold that falls between grid points costs nothing. The
constants reproduce the published boundaries — with five equally spaced looks
at two-sided 5 %, ``pocock`` gives 2.413 and ``obrien_fleming`` gives 2.040 at
the final look — and ``tests/unit/test_design_sequential.py`` pins them.

**What this module does not do.** It does not choose the looks, estimate the
effect, or know what the outcome is. The interim statistics come from whatever
analysis the protocol names — ``identify.ols``, ``design.estimate``, a
posterior contrast from ``estimands.realize`` — and arrive here as a sequence
of ``Z``. Nor does it adjust the final estimate for the stopping rule: a study
stopped at a boundary has a biased naive estimate, and the ledger line
:meth:`MonitoringPath.ledger_line` emits says so rather than letting the number
travel alone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import model_validator
from scipy import optimize, stats

from axiom.core import Assumption, LedgerLine, NonEmptyStr, Spec

__all__ = [
    "Boundary",
    "BoundaryKind",
    "CANONICAL",
    "CrossingProbabilities",
    "Decision",
    "LookOutcome",
    "LookSchedule",
    "MonitoringPath",
    "OperatingCharacteristics",
    "STOPPED_ESTIMATE_BIAS",
    "Side",
    "SpendingFunction",
    "StoppingRule",
    "alpha_spending",
    "crossing_probabilities",
    "harm_boundary",
    "information_fractions",
    "monitor",
    "obrien_fleming",
    "operating_characteristics",
    "pocock",
    "spending",
]

Array = npt.NDArray[np.float64]

BoundaryKind = Literal["efficacy", "harm", "futility"]
Side = Literal["upper", "lower", "two_sided"]
SpendingFunction = Literal["obrien_fleming", "pocock", "power"]
Decision = Literal["continue", "stop_efficacy", "stop_harm", "stop_futility", "completed"]

_DECISION: dict[BoundaryKind, Decision] = {
    "efficacy": "stop_efficacy",
    "harm": "stop_harm",
    "futility": "stop_futility",
}
_INF = float("inf")
_N_GRID = 601
_WIDTH = 8.0
_Z_MAX = 12.0

CANONICAL = Assumption(
    name="canonical_joint_distribution",
    facet="quantity",
    statement=(
        "the interim statistics are jointly normal with Cov(Z_j, Z_k) = sqrt(t_j / t_k) — "
        "the canonical joint distribution of a sequentially monitored estimate"
    ),
    challenged_by=(
        "an interim analysis whose standard error is not the one the information fractions "
        "were computed from, or a statistic whose normal approximation has not been checked "
        "at the earliest look"
    ),
)
STOPPED_ESTIMATE_BIAS = Assumption(
    name="stopped_estimate_bias",
    facet="quantity",
    statement=(
        "an estimate reported at the look that stopped the study is biased away from the "
        "null; its naive interval does not have the nominal coverage"
    ),
    challenged_by="a median-unbiased or stage-wise-ordered estimate computed for the same path",
)


# -- specs -----------------------------------------------------------------------------


class LookSchedule(Spec):
    """When the study looks at itself, and how much information it has when it does.

    ``information`` is the information *fraction* at each look: strictly
    increasing, in ``(0, 1]``. It is the share of the planned final
    information, which for a balanced comparison of means is the share of the
    planned observations. A schedule whose last entry is below 1 describes a
    study whose final analysis is not one of the looks.
    """

    labels: tuple[NonEmptyStr, ...]
    information: tuple[float, ...]

    @model_validator(mode="after")
    def _valid(self) -> LookSchedule:
        if len(self.labels) != len(self.information):
            raise ValueError(
                f"labels and information must have the same length, got "
                f"{len(self.labels)} and {len(self.information)}"
            )
        if not self.information:
            raise ValueError("a look schedule needs at least one look")
        previous = 0.0
        for t in self.information:
            if not 0.0 < t <= 1.0:
                raise ValueError(f"information fractions must be in (0, 1], got {t}")
            if t <= previous:
                raise ValueError(
                    f"information fractions must be strictly increasing, got {self.information}"
                )
            previous = t
        return self

    @property
    def n_looks(self) -> int:
        return len(self.information)

    @property
    def increments(self) -> tuple[float, ...]:
        """The information gained between consecutive looks."""
        previous = 0.0
        out: list[float] = []
        for t in self.information:
            out.append(t - previous)
            previous = t
        return tuple(out)


class Boundary(Spec):
    """Signed thresholds on the monitoring statistic, one per look.

    ``kind`` says what crossing means, ``side`` says which direction crossing
    is (see the module docstring for the sign convention), and ``binding`` says
    whether the error-rate arithmetic is allowed to use it. ``spent`` records
    the cumulative probability the boundary was *built* to spend under the
    null; it is empty for a boundary someone wrote down.
    """

    kind: BoundaryKind
    side: Side
    z: tuple[float, ...]
    binding: bool = True
    spent: tuple[float, ...] = ()
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _valid(self) -> Boundary:
        if not self.z:
            raise ValueError("a boundary needs at least one threshold")
        if self.spent and len(self.spent) != len(self.z):
            raise ValueError("spent, when present, has one entry per threshold")
        if self.side == "two_sided" and any(t <= 0.0 for t in self.z):
            raise ValueError(f"a two-sided boundary needs positive thresholds, got {self.z}")
        if any(not math.isfinite(t) for t in self.z):
            raise ValueError(f"boundary thresholds must be finite, got {self.z}")
        return self

    @property
    def n_looks(self) -> int:
        return len(self.z)

    def crossed(self, look: int, z: float) -> bool:
        """Does statistic ``z`` at (0-based) ``look`` cross this boundary?"""
        threshold = self.z[look]
        if self.side == "upper":
            return z >= threshold
        if self.side == "lower":
            return z <= threshold
        return abs(z) >= threshold

    def nominal_alpha(self, look: int) -> float:
        """The p-value a statistic sitting exactly on the threshold would have."""
        threshold = self.z[look]
        if self.side == "two_sided":
            return float(2.0 * stats.norm.sf(abs(threshold)))
        if self.side == "upper":
            return float(stats.norm.sf(threshold))
        return float(stats.norm.cdf(threshold))

    def limits(self, look: int) -> tuple[float, float]:
        """The ``(lower, upper)`` this boundary alone imposes on the continuation region."""
        threshold = self.z[look]
        if self.side == "upper":
            return -_INF, threshold
        if self.side == "lower":
            return threshold, _INF
        return -threshold, threshold


class StoppingRule(Spec):
    """A look schedule and the boundaries evaluated at every look.

    Composition, not a hierarchy: an efficacy boundary, a harm boundary and a
    futility boundary are three ``Boundary`` values in one tuple, and the rule
    is what says they are evaluated together. Every boundary must carry one
    threshold per look.
    """

    name: NonEmptyStr
    looks: LookSchedule
    boundaries: tuple[Boundary, ...]

    @model_validator(mode="after")
    def _valid(self) -> StoppingRule:
        if not self.boundaries:
            raise ValueError("a stopping rule needs at least one boundary")
        for boundary in self.boundaries:
            if boundary.n_looks != self.looks.n_looks:
                raise ValueError(
                    f"boundary {boundary.kind!r} has {boundary.n_looks} thresholds but the "
                    f"schedule has {self.looks.n_looks} looks"
                )
        for look in range(self.looks.n_looks):
            lower, upper = self.continuation(look)
            if lower >= upper:
                raise ValueError(
                    f"the continuation region at look {look + 1} is empty: "
                    f"({lower:.4g}, {upper:.4g}) — the boundaries cross, so the study "
                    "would stop before it started"
                )
        return self

    def continuation(self, look: int) -> tuple[float, float]:
        """The open interval of ``Z`` at ``look`` for which the study continues.

        Every boundary counts, binding or not: this is what the committee does.
        """
        return _limits(self.boundaries, look)

    def crossings(self, look: int, z: float) -> tuple[Boundary, ...]:
        """Every boundary ``z`` crosses at ``look``, outermost first.

        Precedence is by the *threshold*, not by how far past it ``z`` landed:
        a statistic below both the harm and the futility boundary is a harm
        stop. This is the same attribution :func:`crossing_probabilities` makes
        when it splits the crossing region into labelled pieces.
        """
        hit = [b for b in self.boundaries if b.crossed(look, z)]
        hit.sort(key=lambda b: _extremity(b, look), reverse=True)
        return tuple(hit)

    def of_kind(self, kind: BoundaryKind) -> Boundary | None:
        """The first boundary of ``kind``, or ``None``."""
        for boundary in self.boundaries:
            if boundary.kind == kind:
                return boundary
        return None


def _limits(boundaries: Sequence[Boundary], look: int) -> tuple[float, float]:
    lower, upper = -_INF, _INF
    for boundary in boundaries:
        low, high = boundary.limits(look)
        lower, upper = max(lower, low), min(upper, high)
    return lower, upper


def _extremity(boundary: Boundary, look: int) -> float:
    """How far out a boundary's threshold sits, so the outer of two wins a tie."""
    threshold = boundary.z[look]
    return -threshold if boundary.side == "lower" else threshold


class CrossingProbabilities(Spec):
    """Per-look probability that each boundary is the one that stops the study.

    These are *first*-crossing probabilities and therefore sum, over boundaries
    and looks, to ``1 - continue_probability``. ``binding_only`` records
    whether non-binding boundaries were ignored.
    """

    drift: float
    information: tuple[float, ...]
    per_look: dict[str, tuple[float, ...]]
    continue_probability: float
    binding_only: bool

    @model_validator(mode="after")
    def _valid(self) -> CrossingProbabilities:
        total = sum(sum(v) for v in self.per_look.values()) + self.continue_probability
        # A structural guard, not a precision claim: a mislabelled or overlapping
        # crossing region loses or double-counts whole percentage points, while the
        # trapezoid rule at the default grid loses a few parts per million (the
        # convergence is pinned in tests/unit/test_design_sequential.py).
        if not math.isclose(total, 1.0, abs_tol=1e-3):
            raise ValueError(f"crossing probabilities and continuation sum to {total!r}, not 1")
        return self

    def cumulative(self, kind: str) -> float:
        """Total probability of stopping at ``kind``, over all looks."""
        return float(sum(self.per_look.get(kind, ())))

    @property
    def stop_probability(self) -> float:
        return 1.0 - self.continue_probability

    def by_look(self) -> tuple[float, ...]:
        """Probability of stopping at each look, whatever the boundary."""
        return tuple(
            float(sum(v[k] for v in self.per_look.values())) for k in range(len(self.information))
        )


class OperatingCharacteristics(Spec):
    """What a rule does at one drift: where it stops, and what it costs to get there.

    ``expected_information`` is in information-fraction units — multiply it by
    the planned final sample size to read it as expected enrollment. ``drift``
    is the Z the design expects at full information (``effect / se``), so
    ``drift = 0`` is the null and ``crossings.cumulative("efficacy")`` there is
    the type I error the rule actually spends.
    """

    rule: NonEmptyStr
    drift: float
    crossings: CrossingProbabilities
    expected_information: float
    expected_looks: float

    @property
    def stop_probability(self) -> float:
        return self.crossings.stop_probability


@dataclass(frozen=True)
class LookOutcome:
    """One interim look: what it saw and what the rule said about it."""

    look: int
    label: str
    information: float
    z: float
    decision: Decision
    crossed: str = ""
    effect: float | None = None
    se: float | None = None


@dataclass(frozen=True)
class MonitoringPath:
    """A realized sequence of looks and the decision it produced.

    ``decision`` is ``"completed"`` when every look in the schedule was taken
    without a crossing, ``"continue"`` when the supplied statistics ran out
    before the schedule did, and a ``stop_*`` otherwise.
    """

    rule: StoppingRule
    looks: tuple[LookOutcome, ...]
    decision: Decision
    stopped_at: int | None = None

    @property
    def information_used(self) -> float:
        return self.looks[-1].information if self.looks else 0.0

    @property
    def stop(self) -> LookOutcome | None:
        """The look that stopped the study, or ``None``."""
        return None if self.stopped_at is None else self.looks[self.stopped_at]

    def threshold(self) -> float:
        """The threshold the stopping look crossed; ``nan`` if it did not stop."""
        stop = self.stop
        if stop is None:
            return float("nan")
        for boundary in self.rule.boundaries:
            if boundary.kind == stop.crossed:
                return boundary.z[stop.look]
        return float("nan")

    def ledger_line(self) -> LedgerLine:
        stop = self.stop
        if stop is None:
            statement = (
                f"{self.rule.name}: no boundary crossed in {len(self.looks)} look(s); "
                f"{self.decision} at information fraction {self.information_used:.3f}"
            )
            assumption = CANONICAL
        else:
            statement = (
                f"{self.rule.name}: {stop.crossed} boundary crossed at look "
                f"{stop.look + 1} ({stop.label}), Z = {stop.z:.3f} against "
                f"{self.threshold():.3f}, at information fraction {stop.information:.3f}"
            )
            assumption = STOPPED_ESTIMATE_BIAS
        return LedgerLine(
            kind="sequential_monitoring",
            statement=statement,
            assumption=assumption,
            detail={
                "rule": self.rule.name,
                "decision": self.decision,
                "looks_taken": str(len(self.looks)),
                "information_used": f"{self.information_used:.12g}",
            },
        )


# -- the recursion ---------------------------------------------------------------------


@dataclass(frozen=True)
class _State:
    """The sub-density of ``B_k`` on a grid laid out over the continuation region."""

    b: Array
    w: Array
    f: Array


def _weights(b: Array) -> Array:
    h = float(b[1] - b[0])
    w = np.full(b.size, h)
    w[0] = w[-1] = h / 2.0
    return w


def _grid_for(
    lower: float, upper: float, t: float, drift: float, n_grid: int, width: float
) -> Array | None:
    """A uniform grid over the continuation region, clipped to where the mass is."""
    half = width * math.sqrt(t)
    centre = drift * t
    lo = max(lower, min(0.0, centre) - half)
    hi = min(upper, max(0.0, centre) + half)
    if not hi > lo:
        return None
    return np.linspace(lo, hi, n_grid)


def _between(state: _State | None, dt: float, drift: float, lo: float, hi: float) -> float:
    """``P(lo < B_k <= hi)`` given the sub-density at the previous look.

    ``state is None`` means the previous look is time zero, where ``B`` is a
    point mass at the origin. The mass beyond a threshold is the exact normal
    probability given each grid point, never a grid sum over the tail, so a
    threshold between grid points costs nothing.
    """
    sd = math.sqrt(dt)
    if state is None:
        return float(
            stats.norm.cdf((hi - drift * dt) / sd) - stats.norm.cdf((lo - drift * dt) / sd)
        )
    centre = state.b + drift * dt
    mass = stats.norm.cdf((hi - centre) / sd) - stats.norm.cdf((lo - centre) / sd)
    return float((state.f * state.w) @ mass)


def _advance(state: _State | None, target: Array, dt: float, drift: float) -> _State:
    """The density of ``B_k`` on ``target``, before this look's boundaries are applied."""
    sd = math.sqrt(dt)
    if state is None:
        f = np.asarray(stats.norm.pdf(target, loc=drift * dt, scale=sd), dtype=np.float64)
    else:
        step = target[None, :] - state.b[:, None] - drift * dt
        f = np.asarray((state.f * state.w) @ stats.norm.pdf(step, scale=sd), dtype=np.float64)
    return _State(b=target, w=_weights(target), f=f)


def _regions(
    boundaries: Sequence[Boundary], look: int, sqrt_t: float
) -> tuple[tuple[str, float, float], ...]:
    """Disjoint crossing regions in B units at ``look``, labelled by boundary kind.

    Lower boundaries are consumed outward-in and upper boundaries inward-out, so
    a statistic crossing two boundaries is attributed to the outer one — the
    same precedence :meth:`StoppingRule.crossings` applies to a realized look.
    """
    lower: list[tuple[str, float]] = []
    upper: list[tuple[str, float]] = []
    for boundary in boundaries:
        threshold = boundary.z[look] * sqrt_t
        if boundary.side == "upper":
            upper.append((boundary.kind, threshold))
        elif boundary.side == "lower":
            lower.append((boundary.kind, threshold))
        else:
            upper.append((boundary.kind, threshold))
            lower.append((boundary.kind, -threshold))

    out: list[tuple[str, float, float]] = []
    edge = -_INF
    for kind, threshold in sorted(lower, key=lambda item: item[1]):
        if threshold > edge:
            out.append((kind, edge, threshold))
            edge = threshold
    edge = _INF
    for kind, threshold in sorted(upper, key=lambda item: item[1], reverse=True):
        if threshold < edge:
            out.append((kind, threshold, edge))
            edge = threshold
    return tuple(out)


def _propagate(
    boundaries: Sequence[Boundary],
    looks: LookSchedule,
    drift: float,
    n_grid: int,
    width: float,
) -> tuple[dict[str, list[float]], float]:
    """First-crossing probabilities per boundary kind per look, and P(never crossing)."""
    per_look: dict[str, list[float]] = {b.kind: [0.0] * looks.n_looks for b in boundaries}
    state: _State | None = None
    alive = 1.0
    for look, (dt, t) in enumerate(zip(looks.increments, looks.information, strict=True)):
        sqrt_t = math.sqrt(t)
        for kind, lo, hi in _regions(boundaries, look, sqrt_t):
            per_look[kind][look] += _between(state, dt, drift, lo, hi)
        lower, upper = _limits(boundaries, look)
        target = _grid_for(lower * sqrt_t, upper * sqrt_t, t, drift, n_grid, width)
        if target is None:
            return per_look, 0.0
        state = _advance(state, target, dt, drift)
        alive = float(state.f @ state.w)
    return per_look, alive


def _binding(rule: StoppingRule, binding_only: bool) -> tuple[Boundary, ...]:
    kept = tuple(b for b in rule.boundaries if b.binding or not binding_only)
    if not kept:
        raise ValueError(
            f"stopping rule {rule.name!r} has no binding boundary; pass binding_only=False "
            "for the probabilities of the rule as the committee runs it"
        )
    return kept


def crossing_probabilities(
    rule: StoppingRule,
    drift: float,
    *,
    binding_only: bool = True,
    n_grid: int = _N_GRID,
    width: float = _WIDTH,
) -> CrossingProbabilities:
    """Exact first-crossing probabilities for ``rule`` at ``drift``.

    ``drift`` is the Z the design expects at full information
    (``effect / se_at_full_information``); ``drift = 0`` is the null, so
    ``crossing_probabilities(rule, 0.0).cumulative("efficacy")`` is the type I
    error the rule spends. With ``binding_only`` (the default) non-binding
    boundaries are ignored, the convention that makes a non-binding futility
    boundary conservative; pass ``False`` for the rule as run.
    """
    if not math.isfinite(drift):
        raise ValueError(f"drift must be finite, got {drift}")
    if n_grid < 51:
        raise ValueError(f"n_grid must be at least 51, got {n_grid}")
    boundaries = _binding(rule, binding_only)
    per_look, alive = _propagate(boundaries, rule.looks, drift, n_grid, width)
    return CrossingProbabilities(
        drift=drift,
        information=rule.looks.information,
        per_look={k: tuple(v) for k, v in per_look.items()},
        continue_probability=alive,
        binding_only=binding_only,
    )


def operating_characteristics(
    rule: StoppingRule,
    drift: float,
    *,
    binding_only: bool = True,
    n_grid: int = _N_GRID,
    width: float = _WIDTH,
) -> OperatingCharacteristics:
    """:func:`crossing_probabilities` plus the expected information and look count.

    Expected information is in information-fraction units: a study that always
    runs to the end has ``expected_information == looks.information[-1]``.
    """
    crossings = crossing_probabilities(
        rule, drift, binding_only=binding_only, n_grid=n_grid, width=width
    )
    by_look = crossings.by_look()
    final = rule.looks.information[-1]
    expected_information = (
        sum(p * t for p, t in zip(by_look, rule.looks.information, strict=True))
        + crossings.continue_probability * final
    )
    expected_looks = (
        sum(p * (k + 1) for k, p in enumerate(by_look))
        + crossings.continue_probability * rule.looks.n_looks
    )
    return OperatingCharacteristics(
        rule=rule.name,
        drift=drift,
        crossings=crossings,
        expected_information=float(expected_information),
        expected_looks=float(expected_looks),
    )


# -- boundaries ------------------------------------------------------------------------


def information_fractions(counts: Sequence[float], total: float | None = None) -> tuple[float, ...]:
    """Information fractions from cumulative counts (enrolled, completed, events).

    ``total`` defaults to the last count, which makes the last look the final
    analysis. Pass the planned total to describe a study whose looks stop short
    of it.
    """
    values = [float(c) for c in counts]
    if not values:
        raise ValueError("information_fractions needs at least one count")
    planned = float(total) if total is not None else values[-1]
    if planned <= 0:
        raise ValueError(f"total must be positive, got {planned}")
    if values[-1] > planned:
        raise ValueError(f"the last count {values[-1]:g} exceeds the planned total {planned:g}")
    return tuple(v / planned for v in values)


def spending(kind: SpendingFunction, t: float, alpha: float, *, rho: float = 1.0) -> float:
    """Cumulative error a spending function allows to have been spent by fraction ``t``.

    * ``"obrien_fleming"`` — ``2 · (1 − Φ(z_{1−α/2} / sqrt(t)))``, the Lan–DeMets
      function whose boundary is nearly the O'Brien–Fleming shape: almost
      nothing spent early, so an early stop demands an extreme result.
    * ``"pocock"`` — ``α · log(1 + (e − 1) · t)``, which spends roughly evenly
      and buys early stopping at the price of a larger final critical value.
    * ``"power"`` — ``α · t^rho``; ``rho = 1`` is linear spending and larger
      ``rho`` is more conservative early.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"the information fraction must be in [0, 1], got {t}")
    if t == 0.0:
        return 0.0
    if kind == "obrien_fleming":
        return float(2.0 * stats.norm.sf(stats.norm.isf(alpha / 2.0) / math.sqrt(t)))
    if kind == "pocock":
        return float(alpha * math.log1p((math.e - 1.0) * t))
    if rho <= 0.0:
        raise ValueError(f"rho must be positive, got {rho}")
    return float(alpha * t**rho)


def _schedule(information: Sequence[float] | LookSchedule) -> LookSchedule:
    if isinstance(information, LookSchedule):
        return information
    fractions = tuple(float(t) for t in information)
    return LookSchedule(
        labels=tuple(f"look_{k + 1}" for k in range(len(fractions))), information=fractions
    )


def _cumulative_spent(
    looks: LookSchedule, boundary: Boundary, n_grid: int, width: float
) -> tuple[float, ...]:
    per_look, _ = _propagate((boundary,), looks, 0.0, n_grid, width)
    out: list[float] = []
    running = 0.0
    for value in per_look[boundary.kind]:
        running += value
        out.append(running)
    return tuple(out)


def _shaped(
    alpha: float,
    looks: LookSchedule,
    shape: Array,
    side: Side,
    kind: BoundaryKind,
    label: str,
    n_grid: int,
    width: float,
) -> Boundary:
    """Solve the one multiplier ``c`` for which ``c · shape`` spends exactly ``alpha``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    sign = -1.0 if side == "lower" else 1.0

    def excess(c: float) -> float:
        candidate = Boundary(kind=kind, side=side, z=tuple(float(sign * c * s) for s in shape))
        per_look, _ = _propagate((candidate,), looks, 0.0, n_grid, width)
        return float(sum(per_look[kind])) - alpha

    lo, hi = 0.05, _Z_MAX
    if excess(lo) < 0.0:
        raise ValueError(
            f"alpha={alpha:g} is larger than {looks.n_looks} looks of the {label} shape "
            "can spend"
        )
    c = float(optimize.brentq(excess, lo, hi, xtol=1e-11, rtol=1e-13))
    z = tuple(float(sign * c * s) for s in shape)
    boundary = Boundary(kind=kind, side=side, z=z)
    return boundary.model_copy(
        update={
            "spent": _cumulative_spent(looks, boundary, n_grid, width),
            "detail": {"shape": label, "alpha": f"{alpha:.12g}", "constant": f"{c:.12g}"},
        }
    )


def pocock(
    alpha: float,
    information: Sequence[float] | LookSchedule,
    *,
    side: Side = "two_sided",
    kind: BoundaryKind = "efficacy",
    n_grid: int = _N_GRID,
    width: float = _WIDTH,
) -> Boundary:
    """A constant critical value on the Z scale, solved so the total error is ``alpha``.

    Every look is judged at the same threshold, so the first look has the same
    chance of stopping as the last: the design that stops soonest, and the one
    that pays most at the final look. With five equally spaced looks the
    two-sided constant is 2.413 against the fixed-sample 1.960.
    """
    looks = _schedule(information)
    return _shaped(alpha, looks, np.ones(looks.n_looks), side, kind, "pocock", n_grid, width)


def obrien_fleming(
    alpha: float,
    information: Sequence[float] | LookSchedule,
    *,
    side: Side = "two_sided",
    kind: BoundaryKind = "efficacy",
    n_grid: int = _N_GRID,
    width: float = _WIDTH,
) -> Boundary:
    """Thresholds ``c / sqrt(t)``, solved so the total error is ``alpha``.

    The early looks are nearly unstoppable and the final critical value is
    barely above the fixed-sample one — with five equally spaced looks, 2.040
    at the last look against 1.960. This is the shape a trial uses when it
    wants the option to stop early without paying much for it.
    """
    looks = _schedule(information)
    shape = 1.0 / np.sqrt(np.asarray(looks.information, dtype=np.float64))
    return _shaped(alpha, looks, shape, side, kind, "obrien_fleming", n_grid, width)


def _crossed_mass(state: _State | None, dt: float, threshold: float, side: Side) -> float:
    """Mass beyond a symmetric-or-one-sided threshold in B units, given the live density."""
    if side == "two_sided":
        return _between(state, dt, 0.0, threshold, _INF) + _between(
            state, dt, 0.0, -_INF, -threshold
        )
    if side == "upper":
        return _between(state, dt, 0.0, threshold, _INF)
    return _between(state, dt, 0.0, -_INF, -threshold)


def alpha_spending(
    alpha: float,
    information: Sequence[float] | LookSchedule,
    *,
    family: SpendingFunction = "obrien_fleming",
    side: Side = "two_sided",
    kind: BoundaryKind = "efficacy",
    rho: float = 1.0,
    n_grid: int = _N_GRID,
    width: float = _WIDTH,
) -> Boundary:
    """A Lan–DeMets boundary: solve each threshold to spend exactly what is budgeted.

    Unlike :func:`pocock` and :func:`obrien_fleming`, this does not need the
    looks to be the ones that were planned. Each threshold is solved in turn
    against the error already spent, so a look added, moved or skipped changes
    only the thresholds after it — which is what makes the method usable by a
    committee that meets when it can rather than when the protocol said it
    would.
    """
    looks = _schedule(information)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    sign = -1.0 if side == "lower" else 1.0

    thresholds: list[float] = []
    spent: list[float] = []
    state: _State | None = None
    for look, (dt, t) in enumerate(zip(looks.increments, looks.information, strict=True)):
        budget = spending(family, t, alpha, rho=rho) - (spent[-1] if spent else 0.0)
        sqrt_t = math.sqrt(t)

        def deficit(
            c: float,
            *,
            state: _State | None = state,
            dt: float = dt,
            sqrt_t: float = sqrt_t,
            budget: float = budget,
        ) -> float:
            return _crossed_mass(state, dt, c * sqrt_t, side) - budget

        alive = deficit(0.0) + budget
        if budget <= 0.0:
            c = _Z_MAX
        elif alive < budget:
            raise ValueError(
                f"the {family} spending function asks for {budget:.3g} at look {look + 1} "
                f"but only {alive:.3g} of the path is still alive there"
            )
        else:
            c = float(optimize.brentq(deficit, 0.0, _Z_MAX, xtol=1e-11))
        thresholds.append(float(sign * c))
        spent.append((spent[-1] if spent else 0.0) + _crossed_mass(state, dt, c * sqrt_t, side))

        candidate = Boundary(kind=kind, side=side, z=tuple(thresholds))
        lower, upper = _limits((candidate,), look)
        target = _grid_for(lower * sqrt_t, upper * sqrt_t, t, 0.0, n_grid, width)
        if target is None:  # pragma: no cover - c is bounded, so the region is never empty
            raise ValueError(f"the continuation region at look {look + 1} is empty")
        state = _advance(state, target, dt, 0.0)

    return Boundary(
        kind=kind,
        side=side,
        z=tuple(thresholds),
        spent=tuple(spent),
        detail={
            "shape": f"{family}_spending",
            "alpha": f"{alpha:.12g}",
            "rho": f"{rho:.12g}" if family == "power" else "",
        },
    )


def harm_boundary(
    probability: float,
    information: Sequence[float] | LookSchedule,
    *,
    margin: float = 0.0,
    se_at_full_information: float = 1.0,
    kind: BoundaryKind = "harm",
) -> Boundary:
    """The Z threshold of a posterior-probability harm rule, look by look.

    The rule is the one a monitoring committee states in words: *stop when the
    posterior probability that the treatment is worse than the control by more
    than* ``margin`` *reaches* ``probability``. Under a flat prior the posterior
    at look ``k`` is ``N(d_k, se_k²)``, so that rule is

        P(effect < −margin | data) ≥ p   ⟺   Z_k ≤ −margin / se_k − z_p,

    and with ``se_k = se / sqrt(t_k)`` the threshold is a function of the
    information fraction alone. ``margin = 0`` gives a constant threshold — a
    Pocock-shaped rule in disguise — and a positive margin makes the rule
    *harder* to trigger early, when the standard error is large and a large
    observed harm is cheap to come by. A probability rule states a posterior,
    not an error rate: pass the result to :func:`crossing_probabilities` to find
    out what it spends.
    """
    if not 0.5 <= probability < 1.0:
        raise ValueError(f"probability must be in [0.5, 1), got {probability}")
    if margin < 0.0:
        raise ValueError(f"margin must be non-negative, got {margin}")
    if se_at_full_information <= 0.0:
        raise ValueError(f"se_at_full_information must be positive, got {se_at_full_information}")
    looks = _schedule(information)
    z_p = float(stats.norm.isf(1.0 - probability))
    z = tuple(-margin * math.sqrt(t) / se_at_full_information - z_p for t in looks.information)
    return Boundary(
        kind=kind,
        side="lower",
        z=z,
        detail={
            "shape": "posterior_probability",
            "probability": f"{probability:.12g}",
            "margin": f"{margin:.12g}",
            "se_at_full_information": f"{se_at_full_information:.12g}",
        },
    )


# -- monitoring ------------------------------------------------------------------------


def monitor(
    rule: StoppingRule,
    z: Sequence[float],
    *,
    effects: Sequence[float] | None = None,
    ses: Sequence[float] | None = None,
) -> MonitoringPath:
    """Walk realized statistics against ``rule`` and stop at the first crossing.

    ``z`` may be shorter than the schedule — a study still running has taken
    only the looks it has taken, and the decision is ``"continue"``. Every
    boundary is evaluated, binding or not: ``binding`` governs the error-rate
    arithmetic, not the committee.
    """
    if len(z) > rule.looks.n_looks:
        raise ValueError(f"{len(z)} statistics for a schedule of {rule.looks.n_looks} looks")
    if effects is not None and len(effects) != len(z):
        raise ValueError("effects, when supplied, has one entry per statistic")
    if ses is not None and len(ses) != len(z):
        raise ValueError("ses, when supplied, has one entry per statistic")

    outcomes: list[LookOutcome] = []
    stopped_at: int | None = None
    decision: Decision = "continue"
    for look, value in enumerate(z):
        statistic = float(value)
        if not math.isfinite(statistic):
            raise ValueError(f"the statistic at look {look + 1} is not finite: {statistic}")
        crossings = rule.crossings(look, statistic)
        step: Decision = _DECISION[crossings[0].kind] if crossings else "continue"
        outcomes.append(
            LookOutcome(
                look=look,
                label=rule.looks.labels[look],
                information=rule.looks.information[look],
                z=statistic,
                decision=step,
                crossed=crossings[0].kind if crossings else "",
                effect=None if effects is None else float(effects[look]),
                se=None if ses is None else float(ses[look]),
            )
        )
        if crossings:
            stopped_at, decision = look, step
            break
    else:
        decision = "completed" if len(z) == rule.looks.n_looks else "continue"

    return MonitoringPath(
        rule=rule, looks=tuple(outcomes), decision=decision, stopped_at=stopped_at
    )
