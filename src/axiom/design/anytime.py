"""Intervals for a study somebody will look at whenever they like.

``design.sequential`` controls the error rate of a study that looks at itself a
number of times **fixed in advance**. That is the right machinery for a trial
with a data monitoring committee and a protocol. It is not what happens to an
experiment inside a company, where the dashboard is open, the interval is
refreshed hourly, and the decision is taken the first afternoon it looks good.
A spending function cannot price a look nobody scheduled (note 0027, §D27.6.6).

A **confidence sequence** can. It is an interval at every possible time such
that the probability the *whole sequence* ever excludes the truth is at most
``alpha``. A stakeholder who peeks continuously and stops the moment the
interval clears zero is doing something a fixed-sample interval cannot survive
and a confidence sequence is built for.

**The construction.** On the canonical scale (``design.sequential``) the
B-value ``B_t = Z_t · sqrt(t)`` is a Brownian motion with drift ``θ`` and
variance ``t``. For any ``λ``, ``exp(λ(B_t − θt) − λ²t/2)`` is a martingale.
Mixing over ``λ ~ N(0, 1/ρ)`` gives a closed form (Robbins 1970; Howard,
Ramdas, McAuliffe and Sekhon 2021),

    M_t(θ) = sqrt(ρ / (t + ρ)) · exp( (B_t − θt)² / (2(t + ρ)) ),

which starts at 1 and is a non-negative martingale, so Ville's inequality
bounds ``P(∃t : M_t ≥ 1/alpha) ≤ alpha``. Inverting gives the boundary

    u(t) = sqrt( (t + ρ) · log( (t + ρ) / (ρ · alpha²) ) )

and the interval ``B_t/t ± u(t)/t`` on the drift scale, at every ``t`` at once.

**What it costs, stated first because it is the whole trade.** At a single look
the anytime interval is *wider* than the fixed-sample one — about 55 % wider at
the information fraction it is tuned for at ``alpha = 0.05``, and further out on
either side of that. There is no free anytime validity: the width is the peeking. What you get
back is that no look is illegitimate, the interval never has to be re-derived
for a schedule nobody kept, and the number a stakeholder read at 3pm on a
Tuesday is a number they were entitled to read.

**Tuning.** ``ρ`` decides where the sequence is tightest; it is a free
parameter of the mixing distribution and cannot be avoided, only chosen.
:func:`tune` picks the ``ρ`` that minimizes the interval's half-width at a
target information fraction, numerically rather than by a rule of thumb, and the
choice is recorded on the result.

**The e-value.** ``M_t(0)`` is the evidence against the null at time ``t`` and
is itself the object that makes this work: an e-value, whose reciprocal is an
anytime-valid p-value and which combines across experiments by multiplication
and averaging in ways a p-value does not. :func:`evalue` returns it, and
``design.program`` uses it for error control across a book of experiments.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import Field, model_validator
from scipy import optimize

from axiom.core import Interval, LedgerLine, NonEmptyStr, Spec
from axiom.design.sequential import CANONICAL, LookSchedule

__all__ = [
    "AnytimeLook",
    "ConfidenceSequence",
    "anytime_p",
    "confidence_sequence",
    "evalue",
    "mixture_boundary",
    "tune",
]

_MASS = 0.95


def mixture_boundary(information: float, *, alpha: float, rho: float) -> float:
    """``u(t)``: how far ``B_t`` may stray from ``θt`` at *any* time, at level ``alpha``.

    The normal-mixture boundary of the module docstring. It grows like
    ``sqrt(t log t)`` rather than ``sqrt(t)``, which is exactly the gap between
    an interval valid at one time and one valid at all of them.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (math.isfinite(rho) and rho > 0.0):
        raise ValueError(f"rho must be finite and positive, got {rho}")
    if not (math.isfinite(information) and information >= 0.0):
        raise ValueError(f"information must be finite and non-negative, got {information}")
    total = information + rho
    return math.sqrt(total * math.log(total / (rho * alpha * alpha)))


def tune(alpha: float, target: float = 1.0) -> float:
    """The ``rho`` whose interval is narrowest at information fraction ``target``.

    Minimizes ``u(target) / target`` — the half-width on the drift scale — over
    ``log rho``. A design that expects to be read at the halfway point should
    tune at ``0.5`` and accept a wider interval at the end; one that expects to
    run to completion should tune at ``1``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (math.isfinite(target) and target > 0.0):
        raise ValueError(f"target information must be finite and positive, got {target}")

    def width(log_rho: float) -> float:
        return mixture_boundary(target, alpha=alpha, rho=math.exp(log_rho)) / target

    result = optimize.minimize_scalar(width, bounds=(-20.0, 20.0), method="bounded")
    return float(math.exp(result.x))


def evalue(z: float, information: float, *, rho: float, drift: float = 0.0) -> float:
    """``M_t(drift)``: the evidence against ``drift`` accumulated by information ``t``.

    An e-value. It starts at 1, is a martingale under the null, and never needs
    to know how many times it was read — which is what makes ``1 / evalue`` an
    anytime-valid p-value.
    """
    if not (math.isfinite(rho) and rho > 0.0):
        raise ValueError(f"rho must be finite and positive, got {rho}")
    if not (math.isfinite(information) and information >= 0.0):
        raise ValueError(f"information must be finite and non-negative, got {information}")
    if not (math.isfinite(z) and math.isfinite(drift)):
        raise ValueError(f"z and drift must be finite, got {z}, {drift}")
    b = z * math.sqrt(information)
    total = information + rho
    return math.sqrt(rho / total) * math.exp((b - drift * information) ** 2 / (2.0 * total))


def anytime_p(z: float, information: float, *, rho: float) -> float:
    """``1 / evalue``, capped at 1: a p-value that is valid however often it is read.

    Unlike a fixed-sample p-value this may be quoted at every look and compared
    to ``alpha`` at every look. It is larger than the fixed-sample p-value at
    any given look, which is the same trade the interval makes.
    """
    return min(1.0, 1.0 / evalue(z, information, rho=rho))


class AnytimeLook(Spec):
    """One look at a confidence sequence: the interval then, and the evidence so far.

    ``running_evalue`` is the largest e-value seen up to and including this
    look. It is the one to quote: Ville's inequality bounds the probability
    that the running maximum ever exceeds ``1/alpha``, so a decision taken on
    the maximum is the decision the guarantee covers.
    """

    look: int = Field(ge=0)
    label: str = ""
    information: float = Field(gt=0, le=1)
    z: float
    drift: float
    interval: Interval
    evalue: float = Field(ge=0)
    running_evalue: float = Field(ge=0)
    crossed: bool

    @model_validator(mode="after")
    def _consistent(self) -> AnytimeLook:
        if self.running_evalue < self.evalue:
            raise ValueError("the running maximum cannot be below this look's e-value")
        if self.interval.definition != "anytime":
            raise ValueError(f"an anytime look carries an anytime interval, not {self.interval}")
        return self

    @property
    def anytime_p(self) -> float:
        return min(1.0, 1.0 / self.running_evalue) if self.running_evalue > 0 else 1.0


class ConfidenceSequence(Spec):
    """A sequence of intervals valid at every look at once, and what it cost to be.

    ``alpha`` is the error rate over the *whole* sequence, not per look.
    ``rho`` is the mixing parameter and ``tuned_at`` the information fraction it
    was chosen for. ``looks`` are the realized looks in order; ``crossed_at`` is
    the first look whose interval excluded ``null``, or ``None``.
    """

    name: NonEmptyStr
    alpha: float = Field(gt=0, lt=1)
    rho: float = Field(gt=0)
    tuned_at: float = Field(gt=0, le=1)
    null: float = 0.0
    mass: float = Field(default=_MASS, gt=0, lt=1)
    looks: tuple[AnytimeLook, ...] = Field(min_length=1)

    @property
    def crossed_at(self) -> int | None:
        """The first look whose interval excluded the null, or ``None``."""
        for look in self.looks:
            if look.crossed:
                return look.look
        return None

    @property
    def final(self) -> AnytimeLook:
        return self.looks[-1]

    def width_ratio(self, information: float = 1.0) -> float:
        """How much wider this sequence is than a fixed-sample interval at ``information``.

        The price of anytime validity, as a number. A fixed-sample Wald
        half-width on the drift scale at information ``t`` is
        ``z_{1-alpha/2} / sqrt(t)``; the sequence's is ``u(t) / t``.
        """
        from scipy import stats

        fixed = float(stats.norm.isf(self.alpha / 2.0)) / math.sqrt(information)
        return (mixture_boundary(information, alpha=self.alpha, rho=self.rho) / information) / fixed

    def ledger_line(self) -> LedgerLine:
        crossed = self.crossed_at
        where = (
            f"crossed at look {crossed + 1} of {len(self.looks)}"
            if crossed is not None
            else f"never crossed in {len(self.looks)} look(s)"
        )
        return LedgerLine(
            kind="confidence_sequence",
            statement=(
                f"{self.name}: anytime-valid at alpha {self.alpha:g} over every look, "
                f"{where}; final {self.final.interval.text()}, running e-value "
                f"{self.final.running_evalue:.4g} (anytime p = {self.final.anytime_p:.3g})"
            ),
            assumption=CANONICAL,
            detail={
                "alpha": f"{self.alpha:g}",
                "rho": f"{self.rho:.12g}",
                "tuned_at": f"{self.tuned_at:g}",
                "looks": str(len(self.looks)),
                "width_vs_fixed": f"{self.width_ratio(self.tuned_at):.4g}",
            },
        )

    def summary(self) -> str:
        crossed = self.crossed_at
        head = f"{self.name}: {len(self.looks)} look(s) at alpha {self.alpha:g}, anytime-valid"
        body = (
            f"  crossed at look {crossed + 1}"
            if crossed is not None
            else "  never crossed the null"
        )
        return "\n".join(
            [
                head,
                body,
                f"  final drift {self.final.drift:+.4g} {self.final.interval.text()}",
                f"  running e-value {self.final.running_evalue:.4g}"
                f"  (anytime p = {self.final.anytime_p:.3g})",
                f"  width against a fixed-sample interval at t = {self.tuned_at:g}: "
                f"{self.width_ratio(self.tuned_at):.3f}x",
            ]
        )


def confidence_sequence(
    z: Sequence[float],
    information: Sequence[float] | LookSchedule,
    *,
    alpha: float = 0.05,
    name: str = "confidence sequence",
    null: float = 0.0,
    rho: float | None = None,
    tuned_at: float | None = None,
    labels: Sequence[str] | None = None,
) -> ConfidenceSequence:
    """Walk a realized sequence of interim statistics and report the anytime interval at each.

    ``z`` and ``information`` are the monitoring statistic and the information
    fraction at each look — the same inputs ``design.monitor`` takes, and they
    need not be the looks anybody planned. ``rho`` defaults to
    :func:`tune`\\ ``(alpha, tuned_at)`` with ``tuned_at`` defaulting to the last
    information fraction supplied, so a sequence read to the end is tightest at
    the end.

    The intervals are on the drift scale (``effect / se_at_full_information``)
    and are labelled ``anytime``, not ``wald``: they are not the same object and
    gate 6 exists so they cannot be confused.
    """
    fractions = (
        tuple(information.information)
        if isinstance(information, LookSchedule)
        else tuple(float(t) for t in information)
    )
    if len(z) != len(fractions):
        raise ValueError(f"{len(z)} statistics for {len(fractions)} information fractions")
    if not fractions:
        raise ValueError("a confidence sequence needs at least one look")
    if any(not 0.0 < t <= 1.0 for t in fractions):
        raise ValueError(f"information fractions must be in (0, 1], got {fractions}")
    if any(b <= a for a, b in zip(fractions, fractions[1:], strict=False)):
        raise ValueError(f"information fractions must be strictly increasing, got {fractions}")
    if labels is not None and len(labels) != len(fractions):
        raise ValueError(f"{len(labels)} labels for {len(fractions)} looks")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    target = tuned_at if tuned_at is not None else fractions[-1]
    mixing = rho if rho is not None else tune(alpha, target)
    looks: list[AnytimeLook] = []
    running = 0.0
    for k, (statistic, t) in enumerate(zip(z, fractions, strict=True)):
        half = mixture_boundary(t, alpha=alpha, rho=mixing) / t
        centre = float(statistic) / math.sqrt(t)
        e = evalue(float(statistic), t, rho=mixing, drift=null)
        running = max(running, e)
        interval = Interval(
            lower=centre - half,
            upper=centre + half,
            definition="anytime",
            mass=1.0 - alpha,
        )
        looks.append(
            AnytimeLook(
                look=k,
                label="" if labels is None else labels[k],
                information=t,
                z=float(statistic),
                drift=centre,
                interval=interval,
                evalue=e,
                running_evalue=running,
                crossed=not interval.contains(null),
            )
        )
    return ConfidenceSequence(
        name=name,
        alpha=alpha,
        rho=mixing,
        tuned_at=target,
        null=null,
        mass=1.0 - alpha,
        looks=tuple(looks),
    )
