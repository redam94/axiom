"""Which treatment to learn about next: priors from past studies, ranking, a budgeted pick.

Ported by specification from the parent's ``planning/history.py`` and
``planning/priority.py``. Three pieces:

* ``prior_from_history`` — past studies of one treatment, each an estimate
  with a standard error and an age. Each standard error is first aged by
  ``design.eig.decayed_sd`` (the posterior variance doubles every half-life,
  so a study ``t`` periods old contributes ``se · 2^(t / (2·hl))``), and the
  aged studies are combined by inverse variance:

      prec_i = 1 / se_i(t_i)²,   mean = Σ prec_i · est_i / Σ prec_i,
      sd = 1 / sqrt(Σ prec_i).

* ``rank_treatments`` — for every candidate the Gaussian EIG and EVSI of
  its experiment and the net value ``evsi − opportunity_cost −
  fixed_cost`` (``design.economics``); ranked by net value, ties by EIG,
  then name, so the order is a function of the inputs alone.
* ``recommend`` — the greedy knapsack on net value per unit cost under an
  optional budget: candidates with positive net value, taken in order of
  ``net / cost`` (a free candidate first), each added if it still fits.
  When nothing has positive net value the answer is ``Unsupported``: no
  experiment is worth running, and the function says so rather than
  returning an empty list that reads like a budget problem.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pydantic import Field, model_validator

from axiom.core import NonEmptyStr, Spec, Unsupported
from axiom.design.economics import experiment_value, information_value_of
from axiom.design.eig import decayed_sd, eig_gaussian
from axiom.design.evoi import DecisionSpec

__all__ = [
    "LearningPriority",
    "Recommendation",
    "StudySummary",
    "TreatmentCandidate",
    "prior_from_history",
    "rank_treatments",
    "recommend",
]


# -- specs -------------------------------------------------------------------------------


class StudySummary(Spec):
    """One past study of a treatment: its estimate, standard error, and age.

    ``definition`` names what the estimate is and how its interval was
    formed (``"wald"``, ``"hdi"``, an estimand expression) so that studies
    of different things are not silently pooled.
    """

    treatment: NonEmptyStr
    estimate: float
    se: float = Field(gt=0)
    periods_ago: float = Field(ge=0)
    definition: NonEmptyStr
    name: str = ""

    @model_validator(mode="after")
    def _finite(self) -> StudySummary:
        if not (math.isfinite(self.estimate) and math.isfinite(self.se)):
            raise ValueError("estimate and se must be finite")
        if not math.isfinite(self.periods_ago):
            raise ValueError("periods_ago must be finite")
        return self


class TreatmentCandidate(Spec):
    """A treatment one could run an experiment on, with the inputs its value needs.

    ``prior_mean`` / ``prior_sd`` describe the decision parameter today
    (``prior_from_history`` is one way to get them); ``experiment_se`` is
    what the planned experiment would measure it to; ``opportunity_cost``
    is the signed value of ``design.economics.opportunity_cost`` and
    ``fixed_cost`` the out-of-pocket cost, both in ``decision.numeraire``.
    """

    name: NonEmptyStr
    prior_mean: float
    prior_sd: float = Field(gt=0)
    experiment_se: float = Field(gt=0)
    decision: DecisionSpec
    opportunity_cost: float = 0.0
    fixed_cost: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _finite(self) -> TreatmentCandidate:
        for field in ("prior_mean", "prior_sd", "experiment_se", "opportunity_cost"):
            if not math.isfinite(getattr(self, field)):
                raise ValueError(f"{field} must be finite")
        return self

    @property
    def cost(self) -> float:
        """``opportunity_cost + fixed_cost`` — what the experiment costs all told (signed)."""
        return self.opportunity_cost + self.fixed_cost


class LearningPriority(Spec):
    """A candidate's EIG, EVSI, and net value, with its rank (``1`` is first)."""

    treatment: NonEmptyStr
    eig: float = Field(ge=0)
    evoi: float = Field(ge=0)
    net_value: float
    cost: float
    rank: int = Field(ge=1)
    numeraire: str = ""
    detail: dict[str, str] = {}


class Recommendation(Spec):
    """The selected treatments and why: the ranked priorities and the budget arithmetic."""

    selected: tuple[NonEmptyStr, ...]
    priorities: tuple[LearningPriority, ...] = Field(min_length=1)
    budget: float | None = None
    total_cost: float
    total_net_value: float
    numeraire: str = ""
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _consistent(self) -> Recommendation:
        names = {p.treatment for p in self.priorities}
        unknown = [s for s in self.selected if s not in names]
        if unknown:
            raise ValueError(f"selected names not among the priorities: {unknown}")
        if self.budget is not None and not (math.isfinite(self.budget) and self.budget >= 0):
            raise ValueError(f"budget must be finite and non-negative, got {self.budget}")
        return self


# -- priors from history -----------------------------------------------------------------


def prior_from_history(
    summaries: Sequence[StudySummary], half_life_periods: float
) -> tuple[float, float]:
    """``(mean, sd)`` of the inverse-variance combination of the decayed studies.

    Every summary must concern the same treatment (a ``ValueError``
    otherwise — different treatments are different parameters). A study's
    standard error is aged by ``decayed_sd(se, periods_ago,
    half_life_periods)`` before weighting, so old studies count for less and a
    study of age ``0`` counts in full.
    """
    if not summaries:
        raise ValueError("need at least one study summary")
    treatments = {s.treatment for s in summaries}
    if len(treatments) != 1:
        raise ValueError(f"summaries concern several treatments: {sorted(treatments)}")
    precision = 0.0
    weighted = 0.0
    for s in summaries:
        aged = decayed_sd(s.se, s.periods_ago, half_life_periods)
        prec = 1.0 / (aged * aged)
        precision += prec
        weighted += prec * s.estimate
    return weighted / precision, 1.0 / math.sqrt(precision)


# -- ranking -----------------------------------------------------------------------------


def _check_distinct(candidates: Sequence[TreatmentCandidate]) -> None:
    if not candidates:
        raise ValueError("need at least one candidate")
    names = [c.name for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError(f"candidate names must be distinct, got {names}")


def _numeraire(candidates: Sequence[TreatmentCandidate]) -> str:
    numeraires = {c.decision.numeraire for c in candidates}
    if len(numeraires) != 1:
        raise ValueError(f"candidates are valued in different numeraires: {sorted(numeraires)}")
    return next(iter(numeraires))


def rank_treatments(candidates: Sequence[TreatmentCandidate]) -> tuple[LearningPriority, ...]:
    """Every candidate scored and ranked by net value, then EIG, then name.

    ``net_value = evsi − opportunity_cost − fixed_cost`` with EVSI from
    ``evoi_gaussian`` against the candidate's decision; ``eig`` is the
    Gaussian EIG of the same experiment in nats. All candidates must share
    a numeraire.
    """
    _check_distinct(candidates)
    numeraire = _numeraire(candidates)
    scored: list[tuple[float, float, str, TreatmentCandidate, float, float]] = []
    for c in candidates:
        eig = eig_gaussian(c.prior_sd, c.experiment_se)
        info = information_value_of(c.decision, c.prior_mean, c.prior_sd, c.experiment_se)
        ev = experiment_value(
            info, c.opportunity_cost, c.fixed_cost, numeraire=numeraire or "unspecified"
        )
        scored.append((ev.net, eig, c.name, c, info, ev.net))
    scored.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return tuple(
        LearningPriority(
            treatment=c.name,
            eig=eig,
            evoi=info,
            net_value=net,
            cost=c.cost,
            rank=i + 1,
            numeraire=numeraire,
            detail={
                "prior_mean": f"{c.prior_mean:.12g}",
                "prior_sd": f"{c.prior_sd:.12g}",
                "experiment_se": f"{c.experiment_se:.12g}",
                "opportunity_cost": f"{c.opportunity_cost:.12g}",
                "fixed_cost": f"{c.fixed_cost:.12g}",
                "net_value": "evsi - opportunity_cost - fixed_cost",
            },
        )
        for i, (net, eig, _, c, info, _) in enumerate(scored)
    )


def _value_per_cost(p: LearningPriority) -> float:
    # A free (or paying) experiment has unbounded value per cost and goes first.
    return math.inf if p.cost <= 0.0 else p.net_value / p.cost


def recommend(
    candidates: Sequence[TreatmentCandidate],
    budget: float | None = None,
    *,
    exclusions: Mapping[str, Sequence[str]] | None = None,
) -> Recommendation | Unsupported:
    """Greedy knapsack over ``rank_treatments``: net value per cost, within ``budget``.

    Only candidates with positive net value are eligible. They are taken
    in decreasing ``net_value / cost`` (ties: higher net value, then name);
    each is added if its cost fits in what remains of the budget (a
    non-positive cost always fits). ``Unsupported`` when no candidate has
    positive net value.

    ``exclusions`` maps a treatment to the treatments it may not be run beside
    — ``design.collision.exclusions`` builds it from a schedule — and a
    candidate whose exclusion set already holds a selected treatment is skipped
    however good its net value is. Two experiments that would move the same
    lever on the same units in the same weeks do not become compatible by being
    worth a lot. The skipped ones are named in ``detail["skipped_excluded"]``
    rather than dropped silently.
    """
    if budget is not None and not (math.isfinite(budget) and budget >= 0.0):
        raise ValueError(f"budget must be finite and non-negative, got {budget}")
    priorities = rank_treatments(candidates)
    eligible = [p for p in priorities if p.net_value > 0.0]
    if not eligible:
        return Unsupported(
            reason=(
                "no candidate has positive net value: the information each experiment buys "
                "is worth less than its opportunity and fixed costs"
            ),
            missing=("positive_net_value",),
            detail={p.treatment: f"net_value={p.net_value:.6g}" for p in priorities},
        )
    order = sorted(eligible, key=lambda p: (-_value_per_cost(p), -p.net_value, p.treatment))
    forbidden = {k: set(v) for k, v in (exclusions or {}).items()}
    remaining = math.inf if budget is None else budget
    selected: list[str] = []
    total_cost = 0.0
    total_net = 0.0
    skipped: list[str] = []
    excluded: list[str] = []
    for p in order:
        clash = sorted(forbidden.get(p.treatment, set()).intersection(selected))
        if clash:
            excluded.append(f"{p.treatment} (beside {', '.join(clash)})")
        elif p.cost <= remaining:
            selected.append(p.treatment)
            total_cost += p.cost
            total_net += p.net_value
            remaining -= p.cost
        else:
            skipped.append(p.treatment)
    return Recommendation(
        selected=tuple(selected),
        priorities=priorities,
        budget=budget,
        total_cost=total_cost,
        total_net_value=total_net,
        numeraire=priorities[0].numeraire,
        detail={
            "rule": "greedy by net_value / cost among positive net value, within budget",
            "order": ", ".join(p.treatment for p in order),
            "skipped_over_budget": ", ".join(skipped),
            "skipped_excluded": ", ".join(excluded),
            "remaining_budget": "unbounded" if budget is None else f"{remaining:.12g}",
        },
    )
