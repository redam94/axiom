"""The estimate a sequentially monitored study is allowed to report.

``design.sequential`` monitors a study and, when a boundary fires, emits
``STOPPED_ESTIMATE_BIAS`` — an assumption whose ``challenged_by`` reads *"a
median-unbiased or stage-wise-ordered estimate computed for the same path"*.
This module is that challenge. Until it existed the bias was named on every
stopped readout and corrected on none, so an early-stopped estimate travelled
into ``calibrate`` and ``meta`` carrying a ledger line that said it was wrong
and a number that was (note 0027, §D27.6.4).

**Why the naive number is biased.** A study stops early when its interim
statistic is extreme. Conditioning on that, the estimate at the stopping look
is a random walk caught against a barrier, not a draw from a symmetric
distribution around the truth — it is biased *away from the null*, and its Wald
interval does not have the nominal coverage.

**Stage-wise ordering.** The correction needs a total order on the sample
space — a rule saying which of two possible outcomes is "more extreme" — and
the estimate is then read off the tail probability that order induces. This
module uses the stage-wise (sample-space) ordering of Emerson and Fleming,
Fairbanks and Madsen; Jennison and Turnbull §8.5. An outcome ``(j, z')`` is at
least as extreme as the observed ``(k, z)`` when

* it stopped at an earlier look by crossing the **upper** edge of that look's
  continuation region (``j < k``), or
* it survived to look ``k`` and landed at ``z' >= z``.

Write ``p(θ)`` for the probability of that set at drift ``θ``. It is the same
recursion ``design.sequential`` already integrates, run to the stopping look,
so no simulation is involved and nothing here re-implements the transform.
``p`` is strictly increasing in ``θ``, which gives three numbers by inversion:

    p(drift)  = 0.5           the median-unbiased estimate
    p(lower)  = (1 - mass)/2  the stage-wise interval's lower limit
    p(upper)  = (1 + mass)/2  its upper limit

The ordering does not depend on which boundary fired, so the same construction
serves an efficacy stop, a harm stop, a futility stop, and a study that ran to
the end — the last of which is also biased, by the looks it *could* have
stopped at. On a single-look rule every number collapses to the fixed-sample
answer: the median-unbiased drift is ``z`` and the interval is ``z ± 1.96``.

**Where the correction bites, stated so it is not overclaimed.** The stage-wise
ordering conditions on *not having stopped earlier*, so at the **first** look
there is nothing to condition on and the median-unbiased estimate equals the
naive one exactly. This is a property of the ordering, not an omission: a
first-look stop is corrected by a bias-adjusted MLE and is not corrected here.
The correction grows with the look index and with how readily the rule stops
early. Stopping at ``z = 2.5`` under a four-look Pocock rule, the naive drift
falls by 0.14 at the second look, 0.19 at the third and 0.22 at the fourth.
Under O'Brien-Fleming, whose first two boundaries sit at 4.05 and 2.86, that
``z`` does not stop the study at all until the third look, where the correction
is 0.03. A rule that rarely stops early has little to correct, which is the
right answer and not a weakness of the estimator.

Both claims are pinned by simulation in
``tests/unit/test_design_stopped.py``: at drift 2 under a four-look Pocock rule
the naive estimate has median 2.12 and the corrected one 2.02, and the 95 %
stage-wise interval covers 95.0 % of the time.

**What it does not fix.** A median-unbiased estimate is not a mean-unbiased
one, and the stage-wise interval is not the shortest interval with its
coverage. It is the construction whose coverage is exact under the canonical
joint distribution and whose ordering does not depend on the unknown drift,
which is what makes it reportable. ``CANONICAL`` therefore rides on every
result alongside ``STAGEWISE_ORDERING``.

The private recursion lives in ``sequential`` and is imported rather than
copied: one integrator, as rule 3 has it for ``forward``.
"""

from __future__ import annotations

import math

from scipy import optimize

from axiom.core import (
    Assumption,
    Interval,
    LedgerLine,
    NonEmptyStr,
    Spec,
    Unsupported,
)
from axiom.design.sequential import (
    CANONICAL,
    STOPPED_ESTIMATE_BIAS,
    Decision,
    MonitoringPath,
    StoppingRule,
    _advance,
    _between,
    _grid_for,
    _limits,
)

__all__ = [
    "STAGEWISE_ORDERING",
    "StoppedEstimate",
    "stagewise_tail",
    "stopped_estimate",
]

_INF = float("inf")
_N_GRID = 301
_WIDTH = 8.0
_BRACKET = 12.0
_BRACKET_MAX = 60.0
_MASS = 0.95
"""The default confidence level, matching the two-sided 5 % the boundaries are built at."""

STAGEWISE_ORDERING = Assumption(
    name="stagewise_ordering",
    facet="quantity",
    statement=(
        "the sample space is ordered stage-wise — an earlier stop across the upper edge of "
        "the continuation region is more extreme than any later outcome — and the reported "
        "estimate is the drift whose tail probability under that order is one half"
    ),
    challenged_by=(
        "a different ordering of the sample space (likelihood-ratio or MLE ordering), which "
        "gives a different point estimate and a different interval for the same path"
    ),
    state="asserted",
)


def stagewise_tail(
    rule: StoppingRule,
    look: int,
    z: float,
    drift: float,
    *,
    n_grid: int = _N_GRID,
    width: float = _WIDTH,
) -> float:
    """``p(drift)``: the probability of an outcome at least as extreme as ``(look, z)``.

    Extreme in the stage-wise sense (module docstring): crossing the upper edge
    of the continuation region before ``look``, or surviving to ``look`` and
    landing at or above ``z``. Every boundary the rule carries counts, binding
    or not — a study that ran past a non-binding futility boundary really did
    run past it, whatever the error arithmetic is entitled to pretend, and on a
    rule with a tight futility edge that boundary does most of the conditioning.
    For the binding-only answer, pass a rule built from the binding boundaries.

    Strictly increasing in ``drift``; ``0`` and ``1`` are approached but not
    reached.
    """
    if not 0 <= look < rule.looks.n_looks:
        raise ValueError(f"look {look} is outside the schedule's {rule.looks.n_looks} looks")
    if not math.isfinite(z):
        raise ValueError(f"z must be finite, got {z}")
    if not math.isfinite(drift):
        raise ValueError(f"drift must be finite, got {drift}")
    if n_grid < 51:
        raise ValueError(f"n_grid must be at least 51, got {n_grid}")

    boundaries = rule.boundaries
    state = None
    total = 0.0
    for k in range(look + 1):
        dt = rule.looks.increments[k]
        t = rule.looks.information[k]
        sqrt_t = math.sqrt(t)
        lower, upper = _limits(boundaries, k)
        if k == look:
            total += _between(state, dt, drift, z * sqrt_t, _INF)
            break
        total += _between(state, dt, drift, upper * sqrt_t, _INF)
        target = _grid_for(lower * sqrt_t, upper * sqrt_t, t, drift, n_grid, width)
        if target is None:  # the continuation region is empty; nothing survives to look
            break
        state = _advance(state, target, dt, drift)
    return min(max(total, 0.0), 1.0)


def _invert(
    rule: StoppingRule,
    look: int,
    z: float,
    target: float,
    *,
    n_grid: int,
    width: float,
) -> float:
    """The drift whose stage-wise tail probability is ``target``."""

    def f(drift: float) -> float:
        return stagewise_tail(rule, look, z, drift, n_grid=n_grid, width=width) - target

    lo, hi = -_BRACKET, _BRACKET
    while f(lo) > 0.0 and lo > -_BRACKET_MAX:
        lo *= 2.0
    while f(hi) < 0.0 and hi < _BRACKET_MAX:
        hi *= 2.0
    if f(lo) > 0.0 or f(hi) < 0.0:
        raise ValueError(
            f"cannot bracket a drift with stage-wise tail probability {target} for "
            f"z = {z} at look {look + 1}; the tail is flat over ±{_BRACKET_MAX}"
        )
    return float(optimize.brentq(f, lo, hi, xtol=1e-8, rtol=1e-10))


class StoppedEstimate(Spec):
    """What a monitored study reports, beside what it would naively have reported.

    ``drift`` is the median-unbiased drift under the stage-wise ordering and
    ``drift_interval`` its confidence interval; both are on the canonical scale
    ``effect / se_at_full_information``. When the stopping look carried a
    standard error the same numbers appear on the effect scale as ``effect``
    and ``effect_interval``, and ``se_full`` is the standard error at full
    information the conversion used.

    ``naive_drift`` is ``z / sqrt(t)`` — the number the study would have
    reported had nobody asked — so ``bias`` is what the stopping rule was worth
    to it. The sign is informative: ``bias > 0`` for a study that stopped for
    efficacy, ``bias < 0`` for one that stopped for harm or futility.
    """

    rule: NonEmptyStr
    look: int
    information: float
    z: float
    decision: Decision
    crossed: str = ""
    mass: float = _MASS
    naive_drift: float
    drift: float
    drift_interval: Interval
    se_full: float | None = None
    naive_effect: float | None = None
    effect: float | None = None
    effect_interval: Interval | None = None

    @property
    def bias(self) -> float:
        """``naive_drift - drift``: what the stopping rule added to the naive number."""
        return self.naive_drift - self.drift

    @property
    def effect_bias(self) -> float | None:
        """``bias`` on the effect scale, when the effect scale is known."""
        if self.naive_effect is None or self.effect is None:
            return None
        return self.naive_effect - self.effect

    @property
    def stopped_early(self) -> bool:
        return self.decision.startswith("stop_")

    def ledger_line(self) -> LedgerLine:
        """The line that replaces ``STOPPED_ESTIMATE_BIAS`` on a corrected readout."""
        return LedgerLine(
            kind="stopped_estimate",
            statement=(
                f"{self.rule}: the naive drift {self.naive_drift:.4g} at look "
                f"{self.look + 1} is replaced by the median-unbiased {self.drift:.4g} "
                f"{self.drift_interval.text()}, a correction of {-self.bias:+.4g} under "
                "the stage-wise ordering"
            ),
            assumption=STAGEWISE_ORDERING,
            detail={
                "rule": self.rule,
                "decision": self.decision,
                "crossed": self.crossed,
                "look": str(self.look + 1),
                "information": f"{self.information:.12g}",
                "z": f"{self.z:.12g}",
                "naive_drift": f"{self.naive_drift:.12g}",
                "drift": f"{self.drift:.12g}",
                "corrects": STOPPED_ESTIMATE_BIAS.name,
                "canonical": CANONICAL.name,
            },
        )

    def text(self) -> str:
        """The corrected number and what it was corrected from, on the scale that exists."""
        if self.effect is not None and self.effect_interval is not None:
            naive = f"{self.naive_effect:.4g}" if self.naive_effect is not None else "—"
            return (
                f"{self.effect:.4g} {self.effect_interval.text()} "
                f"(naive {naive}, corrected by {-self.effect_bias:+.4g})"  # type: ignore[operator]
            )
        return (
            f"drift {self.drift:.4g} {self.drift_interval.text()} "
            f"(naive {self.naive_drift:.4g}, corrected by {-self.bias:+.4g})"
        )

    def __str__(self) -> str:
        return self.text()


def stopped_estimate(
    path: MonitoringPath,
    *,
    mass: float = _MASS,
    n_grid: int = _N_GRID,
    width: float = _WIDTH,
) -> StoppedEstimate | Unsupported:
    """The median-unbiased estimate and stage-wise interval for a monitored path.

    Works for any path that reached a terminal look — a stop at any boundary,
    or a study that ran to the end of its schedule, which is biased too by the
    looks it could have stopped at. A path whose statistics ran out before its
    schedule did (``decision == "continue"``) has no final analysis to correct
    and comes back ``Unsupported`` rather than as a number computed against a
    look that has not happened.

    When the terminal look carries a standard error the result also reports the
    effect scale; without one it reports the canonical drift scale alone, which
    is the honest answer to a path that never said what its units were.
    """
    if not 0.0 < mass < 1.0:
        raise ValueError(f"mass must be in (0, 1), got {mass}")
    if path.decision == "continue":
        return Unsupported(
            reason=(
                f"the path stopped short of its schedule ({len(path.looks)} of "
                f"{path.rule.looks.n_looks} looks) with no boundary crossed; there is no "
                "final analysis to correct"
            ),
            missing=("a terminal look",),
        )
    if not path.looks:
        return Unsupported(reason="the path has no looks", missing=("a terminal look",))

    terminal = path.stop if path.stop is not None else path.looks[-1]
    look, z, t = terminal.look, terminal.z, terminal.information
    sqrt_t = math.sqrt(t)

    drift = _invert(path.rule, look, z, 0.5, n_grid=n_grid, width=width)
    lower = _invert(path.rule, look, z, (1.0 - mass) / 2.0, n_grid=n_grid, width=width)
    upper = _invert(path.rule, look, z, (1.0 + mass) / 2.0, n_grid=n_grid, width=width)
    drift_interval = Interval(lower=lower, upper=upper, definition="stagewise", mass=mass)

    se_full = naive_effect = effect = None
    effect_interval = None
    if terminal.se is not None:
        se_full = terminal.se * sqrt_t
        naive_effect = (z / sqrt_t) * se_full
        effect = drift * se_full
        effect_interval = Interval(
            lower=lower * se_full, upper=upper * se_full, definition="stagewise", mass=mass
        )

    return StoppedEstimate(
        rule=path.rule.name,
        look=look,
        information=t,
        z=z,
        decision=path.decision,
        crossed=terminal.crossed,
        mass=mass,
        naive_drift=z / sqrt_t,
        drift=drift,
        drift_interval=drift_interval,
        se_full=se_full,
        naive_effect=naive_effect,
        effect=effect,
        effect_interval=effect_interval,
    )
