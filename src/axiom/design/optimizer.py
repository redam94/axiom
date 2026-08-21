"""Scoring candidate experiment designs, the Pareto front, and sequencing under cooldown.

Ported by specification from the parent's ``planning/experiments.py`` and
``planning/experiment_optimizer.py``. A ``DesignCandidate`` is one concrete
way of running the experiment — a method from ``methods.registry.METHODS``,
a size, a horizon, a holdout share, the standard error it would achieve
(from ``design.power.difference_se`` or a ``design.cluster`` design), its
out-of-pocket cost, and the cooldown the population needs afterwards.
``evaluate_candidate`` scores it on one decision:

* ``eig`` — ``design.eig.eig_gaussian(prior_sd, experiment_se)`` in nats;
* ``evpi`` / ``evsi`` — ``design.evoi.evoi_gaussian`` in the numeraire;
* ``opportunity_cost`` — ``design.economics.opportunity_cost`` for the
  holdout over the horizon (signed);
* ``net_value`` — ``evsi − opportunity_cost − cost``;
* ``power`` — ``design.power.power_from_se`` for the anchored effect.

``pareto_front`` returns the non-dominated candidates on a tuple of
objectives (a leading ``-`` means *minimize*); ``schedule_with_cooldown``
lays candidates end to end, best net value first, each occupying
``n_periods + cooldown_periods`` of the horizon. Every result carries a
``detail`` explaining its numbers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from axiom.core import NonEmptyStr, Spec
from axiom.design.economics import ValuePerOutcome, opportunity_cost
from axiom.design.eig import eig_gaussian
from axiom.design.evoi import DecisionSpec, evoi_gaussian
from axiom.design.methods.registry import METHODS
from axiom.design.power import power_from_se

__all__ = [
    "CandidateScore",
    "DesignCandidate",
    "EconomicInputs",
    "ProgramSchedule",
    "ScheduledExperiment",
    "evaluate_candidate",
    "pareto_front",
    "schedule_with_cooldown",
]

Array = npt.NDArray[np.float64]

_OBJECTIVES = ("net_value", "cost", "eig", "evsi", "evpi", "power", "opportunity_cost")


# -- specs -------------------------------------------------------------------------------


class DesignCandidate(Spec):
    """One concrete experiment design: method, size, horizon, holdout, precision, cost.

    ``method`` must be a key of ``methods.registry.METHODS``;
    ``experiment_se`` is the standard error the design would achieve on the
    decision parameter (the caller derives it — ``power.difference_se``,
    ``cluster_mde(...).se``, ``eig.experiment_se_for_design``); ``cost`` is
    the fixed, out-of-pocket cost in the numeraire; ``cooldown_periods`` is
    how long the population must rest before the next experiment.
    """

    name: NonEmptyStr
    method: NonEmptyStr
    n_units: int = Field(ge=2)
    n_periods: int = Field(ge=1)
    holdout_fraction: float = Field(gt=0, lt=1)
    experiment_se: float = Field(gt=0)
    cost: float = Field(ge=0)
    n_clusters: int | None = Field(default=None, ge=2)
    cooldown_periods: int = Field(default=0, ge=0)
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _valid(self) -> DesignCandidate:
        if self.method not in METHODS:
            raise ValueError(f"method {self.method!r} is not registered; have {sorted(METHODS)}")
        if not (math.isfinite(self.experiment_se) and math.isfinite(self.cost)):
            raise ValueError("experiment_se and cost must be finite")
        return self

    @property
    def duration(self) -> int:
        """``n_periods + cooldown_periods`` — the horizon one run occupies."""
        return self.n_periods + self.cooldown_periods


class EconomicInputs(Spec):
    """The economic inputs a candidate's opportunity cost needs, stated once.

    ``dose_per_period`` is the dose the whole population receives per
    period; ``marginal_value_ratio`` is the prior mean outcome per dose
    unit, or ``None`` to use the decision parameter's prior mean — that is
    the case where the decision *is* about the marginal value ratio;
    ``dose_cost_per_unit`` is the numeraire cost of a dose unit.
    """

    value_per_outcome: ValuePerOutcome
    dose_per_period: float = Field(ge=0)
    discount_rate: float = Field(ge=0)
    dose_unit: NonEmptyStr = "dose"
    dose_cost_per_unit: float = Field(default=0.0, ge=0)
    marginal_value_ratio: float | None = None

    @model_validator(mode="after")
    def _finite(self) -> EconomicInputs:
        for name in ("dose_per_period", "discount_rate", "dose_cost_per_unit"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.marginal_value_ratio is not None and not math.isfinite(self.marginal_value_ratio):
            raise ValueError("marginal_value_ratio must be finite")
        return self


class CandidateScore(Spec):
    """A candidate's information, value, cost, and power on one decision.

    ``cost`` is the candidate's fixed cost; ``net_value = evsi −
    opportunity_cost − cost``. ``power`` is for ``effect`` at ``alpha``
    (two-sided) with the candidate's ``experiment_se``.
    """

    candidate: DesignCandidate
    eig: float = Field(ge=0)
    evpi: float = Field(ge=0)
    evsi: float = Field(ge=0)
    opportunity_cost: float
    cost: float = Field(ge=0)
    net_value: float
    power: float = Field(ge=0, le=1)
    effect: float
    alpha: float = Field(gt=0, lt=1)
    numeraire: str = ""
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _arithmetic(self) -> CandidateScore:
        expected = self.evsi - self.opportunity_cost - self.cost
        if not math.isclose(self.net_value, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"net_value {self.net_value} is not evsi − opportunity − cost")
        if self.cost != self.candidate.cost:
            raise ValueError("cost must equal the candidate's cost")
        return self

    @property
    def name(self) -> str:
        return self.candidate.name

    def objective(self, name: str) -> float:
        """The value of a named objective (``pareto_front`` vocabulary, no sign prefix)."""
        if name not in _OBJECTIVES:
            raise ValueError(f"unknown objective {name!r}; have {_OBJECTIVES}")
        return float(getattr(self, name))


class ScheduledExperiment(Spec):
    """One slot of a program: ``[start, end)`` is the run, ``[end, free_at)`` the cooldown."""

    name: NonEmptyStr
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    free_at: int = Field(ge=1)
    net_value: float

    @model_validator(mode="after")
    def _ordered(self) -> ScheduledExperiment:
        if not self.start < self.end <= self.free_at:
            raise ValueError("need start < end <= free_at")
        return self


class ProgramSchedule(Spec):
    """A non-overlapping sequence of experiments within ``horizon_periods``."""

    horizon_periods: int = Field(ge=1)
    slots: tuple[ScheduledExperiment, ...]
    skipped: tuple[NonEmptyStr, ...]
    total_net_value: float
    numeraire: str = ""
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _non_overlapping(self) -> ProgramSchedule:
        last = 0
        for s in self.slots:
            if s.start < last:
                raise ValueError(f"slot {s.name!r} starts before the previous one is free")
            if s.end > self.horizon_periods:
                raise ValueError(f"slot {s.name!r} runs past the horizon")
            last = s.free_at
        return self


# -- scoring -----------------------------------------------------------------------------


def evaluate_candidate(
    candidate: DesignCandidate,
    decision: DecisionSpec,
    prior_mean: float,
    prior_sd: float,
    economics: EconomicInputs,
    *,
    alpha: float = 0.05,
    effect: float | None = None,
) -> CandidateScore:
    """Score ``candidate`` on ``decision`` under ``theta ~ N(prior_mean, prior_sd²)``.

    ``effect`` is the effect size the power is computed for — the anchored
    effect from ``design.anchor`` when one exists — and defaults to
    ``prior_mean``, which the ``detail`` records. The decision's numeraire,
    when stated, must match ``economics.value_per_outcome.numeraire``.
    """
    if not math.isfinite(prior_mean):
        raise ValueError(f"prior_mean must be finite, got {prior_mean}")
    vpo = economics.value_per_outcome
    if decision.numeraire and decision.numeraire != vpo.numeraire:
        raise ValueError(
            f"decision numeraire {decision.numeraire!r} disagrees with value_per_outcome's "
            f"{vpo.numeraire!r}"
        )
    eff = prior_mean if effect is None else effect
    if not math.isfinite(eff):
        raise ValueError(f"effect must be finite, got {eff}")
    ratio = economics.marginal_value_ratio
    ratio_value = prior_mean if ratio is None else ratio
    evoi = evoi_gaussian(decision, prior_mean, prior_sd, candidate.experiment_se)
    oc = opportunity_cost(
        candidate.holdout_fraction,
        candidate.n_periods,
        economics.dose_per_period,
        ratio_value,
        vpo,
        economics.discount_rate,
        dose_unit=economics.dose_unit,
        dose_cost_per_unit=economics.dose_cost_per_unit,
    )
    pw = power_from_se(eff, candidate.experiment_se, alpha=alpha, two_sided=True)
    eig = eig_gaussian(prior_sd, candidate.experiment_se)
    net = evoi.evsi - oc.value - candidate.cost
    return CandidateScore(
        candidate=candidate,
        eig=eig,
        evpi=evoi.evpi,
        evsi=evoi.evsi,
        opportunity_cost=oc.value,
        cost=candidate.cost,
        net_value=net,
        power=pw.power,
        effect=eff,
        alpha=alpha,
        numeraire=vpo.numeraire,
        detail={
            "eig": "0.5 * ln(1 + prior_sd^2 / experiment_se^2) nats",
            "evsi": "evoi_gaussian(decision, prior_mean, prior_sd, experiment_se).evsi",
            "opportunity_cost": oc.detail["formula"],
            "dose_withheld_discounted": f"{oc.dose_withheld_discounted:.12g}",
            "marginal_value_ratio": (
                "prior_mean (decision parameter)" if ratio is None else f"{ratio_value:.12g}"
            ),
            "net_value": "evsi - opportunity_cost - cost",
            "power": "power_from_se(effect, experiment_se, alpha, two_sided=True)",
            "effect_source": "prior_mean" if effect is None else "given",
            "prior_mean": f"{prior_mean:.12g}",
            "prior_sd": f"{prior_sd:.12g}",
            "experiment_se": f"{candidate.experiment_se:.12g}",
        },
    )


# -- Pareto front ------------------------------------------------------------------------


def _parse_objectives(objectives: Sequence[str]) -> tuple[tuple[str, float], ...]:
    if not objectives:
        raise ValueError("need at least one objective")
    parsed: list[tuple[str, float]] = []
    for o in objectives:
        sign = -1.0 if o.startswith("-") else 1.0
        name = o[1:] if o.startswith("-") else o
        if name not in _OBJECTIVES:
            raise ValueError(f"unknown objective {o!r}; have {_OBJECTIVES} with optional '-'")
        parsed.append((name, sign))
    if len({n for n, _ in parsed}) != len(parsed):
        raise ValueError("objectives must be distinct")
    return tuple(parsed)


def pareto_front(
    scores: Sequence[CandidateScore],
    objectives: Sequence[str] = ("net_value", "-cost", "eig"),
) -> tuple[CandidateScore, ...]:
    """The non-dominated scores: nothing is at least as good on every objective and better on one.

    Objectives are ``CandidateScore`` field names, maximized unless prefixed
    with ``-``. Exact duplicates on every objective are all kept. The front
    is returned in decreasing order of the first objective (as signed), then
    the second, and so on, then by name, so the order is reproducible.
    """
    parsed = _parse_objectives(objectives)
    if not scores:
        return ()
    names = [s.name for s in scores]
    if len(set(names)) != len(names):
        raise ValueError(f"candidate names must be distinct, got {names}")
    # Signed so that larger is always better.
    values = np.asarray(
        [[sign * s.objective(name) for name, sign in parsed] for s in scores], dtype=np.float64
    )
    keep: list[int] = []
    for i in range(len(scores)):
        others = np.delete(values, i, axis=0)
        dominated = bool(
            np.any(np.all(others >= values[i], axis=1) & np.any(others > values[i], axis=1))
        )
        if not dominated:
            keep.append(i)
    keep.sort(key=lambda i: tuple(-v for v in values[i]) + (scores[i].name,))
    return tuple(scores[i] for i in keep)


# -- sequencing --------------------------------------------------------------------------


def schedule_with_cooldown(
    scores: Sequence[CandidateScore], horizon_periods: int
) -> ProgramSchedule:
    """Greedy end-to-end sequence: best net value first, each run then its cooldown.

    Candidates with non-positive net value are skipped. The rest are
    visited in decreasing net value (ties: name) and placed at the first
    free period if the run — not the trailing cooldown — ends within the
    horizon; a candidate that does not fit is skipped and the next is
    tried. The greedy order is not guaranteed optimal for the total; the
    ``detail`` says so and lists the order tried.
    """
    if horizon_periods < 1:
        raise ValueError(f"horizon_periods must be positive, got {horizon_periods}")
    names = [s.name for s in scores]
    if len(set(names)) != len(names):
        raise ValueError(f"candidate names must be distinct, got {names}")
    numeraires = {s.numeraire for s in scores}
    if len(numeraires) > 1:
        raise ValueError(f"scores are in different numeraires: {sorted(numeraires)}")
    order = sorted(scores, key=lambda s: (-s.net_value, s.name))
    slots: list[ScheduledExperiment] = []
    skipped: list[str] = []
    free_at = 0
    total = 0.0
    for s in order:
        c = s.candidate
        if s.net_value <= 0.0 or free_at + c.n_periods > horizon_periods:
            skipped.append(s.name)
            continue
        end = free_at + c.n_periods
        slots.append(
            ScheduledExperiment(
                name=s.name,
                start=free_at,
                end=end,
                free_at=end + c.cooldown_periods,
                net_value=s.net_value,
            )
        )
        total += s.net_value
        free_at = end + c.cooldown_periods
    return ProgramSchedule(
        horizon_periods=horizon_periods,
        slots=tuple(slots),
        skipped=tuple(skipped),
        total_net_value=total,
        numeraire=next(iter(numeraires)) if numeraires else "",
        detail={
            "rule": "greedy by net_value; a run must end within the horizon; cooldown follows",
            "order": ", ".join(s.name for s in order),
            "optimality": "greedy, not guaranteed to maximize the total net value",
        },
    )
