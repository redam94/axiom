"""The money side of an experiment: value per outcome, discounting, opportunity cost, net value.

Ported by specification from the parent's ``planning/opportunity_cost.py``,
``planning/experiment_value.py`` and ``planning/discount.py`` (no golden
values; the parent's numbers were not captured). The vocabulary is
general: a *dose* is withheld from a *holdout* of units, forgoing
*outcome* worth something in a *numeraire*.

* ``ValuePerOutcome`` — the conversion from outcome units to the numeraire,
  with the statement of where the number came from (rule 4: the value of
  an outcome is an assumption, and it travels as a ledger line).
* ``discount_weights`` — ``w_t = (1 + rate)^(-t)`` for ``t = 0 … n − 1``;
  ``mid_horizon_factor`` is their mean, the single factor that turns an
  undiscounted total over the horizon into its discounted value (for small
  rates it is close to the factor at the horizon's midpoint, hence the name).
* ``opportunity_cost`` — withholding a share ``h`` of the dose for ``n``
  periods withholds ``D = h · dose_per_period · Σ_t w_t`` (discounted dose
  units). The prior says a unit of dose produces ``r`` outcome units
  (``marginal_value_ratio``, point or draws, the mean is used), worth
  ``v`` each, and costs ``c`` in the numeraire. The opportunity cost is

      OC = D · (E[r] · v − c),

  **signed**: when the prior's expected marginal value is below the dose's
  cost — the treatment is net-negative — withholding it *gains*, and the
  cost is negative. Nothing clips it.
* ``experiment_value`` — ``net = information_value − opportunity_cost −
  fixed_cost``; ``information_value_of`` is the EVSI of ``design.evoi`` for
  the decision the experiment informs.

Every monetary field carries its numeraire; every outcome field its unit.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from axiom.core import Assumption, LedgerLine, NonEmptyStr, Spec
from axiom.design.evoi import DecisionSpec, evoi_gaussian

__all__ = [
    "ExperimentValue",
    "OpportunityCost",
    "ValuePerOutcome",
    "discount_weights",
    "experiment_value",
    "information_value_of",
    "mid_horizon_factor",
    "opportunity_cost",
]

Array = npt.NDArray[np.float64]


def _finite(name: str, v: float) -> float:
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")
    return float(v)


def _rate(rate: float) -> float:
    if not (math.isfinite(rate) and rate >= 0.0):
        raise ValueError(f"discount rate must be finite and non-negative, got {rate}")
    return float(rate)


def _periods(n_periods: int) -> int:
    if n_periods < 1:
        raise ValueError(f"n_periods must be positive, got {n_periods}")
    return int(n_periods)


# -- specs -------------------------------------------------------------------------------


class ValuePerOutcome(Spec):
    """``value`` numeraire units per ``outcome_unit``, and where that number came from.

    ``source`` is the human statement of provenance (a contract, a
    finance figure, an analyst's assumption); ``assumption`` is set when
    the value rests on something falsifiable. ``ledger_line()`` renders
    both as the ``core.LedgerLine`` every evidence transfer appends.
    """

    value: float = Field(gt=0)
    outcome_unit: NonEmptyStr
    numeraire: NonEmptyStr
    source: NonEmptyStr
    assumption: Assumption | None = None

    @model_validator(mode="after")
    def _finite_value(self) -> ValuePerOutcome:
        if not math.isfinite(self.value):
            raise ValueError(f"value must be finite, got {self.value}")
        return self

    def ledger_line(self) -> LedgerLine:
        return LedgerLine(
            kind="value_per_outcome",
            statement=(
                f"one {self.outcome_unit} is worth {self.value:.6g} {self.numeraire} "
                f"({self.source})"
            ),
            assumption=self.assumption,
            detail={
                "value": f"{self.value:.12g}",
                "outcome_unit": self.outcome_unit,
                "numeraire": self.numeraire,
            },
        )


class OpportunityCost(Spec):
    """What a holdout forgoes, in the numeraire, with every factor that produced it.

    ``value`` is ``dose_withheld_discounted · (ratio_mean · value_per_outcome
    − dose_cost_per_unit)`` and keeps its sign. ``outcome_forgone`` is the
    expected discounted outcome (in ``outcome_unit``) the holdout does not
    produce; ``ratio_sd`` is ``0`` when the ratio was a point.
    """

    value: float
    numeraire: NonEmptyStr
    holdout_fraction: float = Field(gt=0, lt=1)
    n_periods: int = Field(ge=1)
    discount_rate: float = Field(ge=0)
    dose_per_period: float = Field(ge=0)
    dose_unit: NonEmptyStr
    dose_withheld: float = Field(ge=0)
    dose_withheld_discounted: float = Field(ge=0)
    dose_cost_per_unit: float = Field(ge=0)
    ratio_mean: float
    ratio_sd: float = Field(ge=0)
    n_ratio_draws: int = Field(ge=1)
    outcome_forgone: float
    outcome_unit: NonEmptyStr
    value_per_outcome: ValuePerOutcome
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _finite_fields(self) -> OpportunityCost:
        for name in ("value", "ratio_mean", "outcome_forgone"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.value_per_outcome.numeraire != self.numeraire:
            raise ValueError("numeraire must match value_per_outcome.numeraire")
        if self.value_per_outcome.outcome_unit != self.outcome_unit:
            raise ValueError("outcome_unit must match value_per_outcome.outcome_unit")
        return self


class ExperimentValue(Spec):
    """``net = information_value − opportunity_cost − fixed_cost``, all in ``numeraire``."""

    information_value: float = Field(ge=0)
    opportunity_cost: float
    fixed_cost: float = Field(ge=0)
    net: float
    numeraire: NonEmptyStr
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _arithmetic(self) -> ExperimentValue:
        for name in ("information_value", "opportunity_cost", "fixed_cost", "net"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        expected = self.information_value - self.opportunity_cost - self.fixed_cost
        if not math.isclose(self.net, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"net {self.net} is not information − opportunity − fixed = {expected}"
            )
        return self


# -- discounting -------------------------------------------------------------------------


def discount_weights(n_periods: int, rate: float) -> Array:
    """``(1 + rate)^(-t)`` for ``t = 0 … n_periods − 1``: period 0 is undiscounted."""
    n = _periods(n_periods)
    r = _rate(rate)
    t = np.arange(n, dtype=np.float64)
    return np.asarray((1.0 + r) ** (-t), dtype=np.float64)


def mid_horizon_factor(n_periods: int, rate: float) -> float:
    """Mean of ``discount_weights``: undiscounted total × this = discounted total.

    Equals ``1`` at a zero rate and is strictly decreasing in both the rate
    and the horizon; ``(1 + rate)^(-(n − 1) / 2)``, the factor at the
    midpoint, is its first-order approximation.
    """
    return float(np.mean(discount_weights(n_periods, rate)))


# -- opportunity cost --------------------------------------------------------------------


def _ratio_summary(marginal_value_ratio: float | npt.ArrayLike) -> tuple[float, float, int]:
    x = np.asarray(marginal_value_ratio, dtype=np.float64).ravel()
    if x.size == 0:
        raise ValueError("marginal_value_ratio needs at least one value")
    if not bool(np.all(np.isfinite(x))):
        raise ValueError("marginal_value_ratio must be finite")
    if x.size == 1:
        return float(x[0]), 0.0, 1
    return float(x.mean()), float(x.std(ddof=1)), int(x.size)


def opportunity_cost(
    holdout_fraction: float,
    n_periods: int,
    dose_per_period: float,
    marginal_value_ratio: float | npt.ArrayLike,
    value_per_outcome: ValuePerOutcome,
    discount_rate: float,
    *,
    dose_unit: str = "dose",
    dose_cost_per_unit: float = 0.0,
) -> OpportunityCost:
    """Signed value forgone by withholding ``holdout_fraction`` of the dose for ``n_periods``.

    ``dose_per_period`` is the dose the whole population receives per period
    (in ``dose_unit``); ``marginal_value_ratio`` is the prior on outcome
    units per dose unit — a point or draws, whose mean is what an expected
    cost uses (its sd is reported, not used); ``dose_cost_per_unit`` is what
    a unit of dose costs in the numeraire (``0`` when the dose is free or its
    cost is accounted elsewhere). See the module docstring for the formula;
    a negative result means the prior expects withholding to pay.
    """
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in (0, 1), got {holdout_fraction}")
    n = _periods(n_periods)
    r = _rate(discount_rate)
    dose = _finite("dose_per_period", dose_per_period)
    if dose < 0.0:
        raise ValueError(f"dose_per_period must be non-negative, got {dose}")
    c = _finite("dose_cost_per_unit", dose_cost_per_unit)
    if c < 0.0:
        raise ValueError(f"dose_cost_per_unit must be non-negative, got {c}")
    ratio_mean, ratio_sd, n_draws = _ratio_summary(marginal_value_ratio)
    factor = mid_horizon_factor(n, r)
    withheld = holdout_fraction * dose * n
    withheld_disc = withheld * factor
    outcome_forgone = withheld_disc * ratio_mean
    value = withheld_disc * (ratio_mean * value_per_outcome.value - c)
    return OpportunityCost(
        value=value,
        numeraire=value_per_outcome.numeraire,
        holdout_fraction=float(holdout_fraction),
        n_periods=n,
        discount_rate=r,
        dose_per_period=dose,
        dose_unit=dose_unit,
        dose_withheld=withheld,
        dose_withheld_discounted=withheld_disc,
        dose_cost_per_unit=c,
        ratio_mean=ratio_mean,
        ratio_sd=ratio_sd,
        n_ratio_draws=n_draws,
        outcome_forgone=outcome_forgone,
        outcome_unit=value_per_outcome.outcome_unit,
        value_per_outcome=value_per_outcome,
        detail={
            "formula": (
                "dose_withheld_discounted * (ratio_mean * value_per_outcome - cost_per_unit)"
            ),
            "mid_horizon_factor": f"{factor:.12g}",
            "ratio_source": "point" if n_draws == 1 else f"mean of {n_draws} draws",
        },
    )


# -- experiment value --------------------------------------------------------------------


def information_value_of(
    decision: DecisionSpec, prior_mean: float, prior_sd: float, experiment_se: float
) -> float:
    """EVSI of an experiment measuring the decision parameter to ``experiment_se``.

    A thin wrapper over ``design.evoi.evoi_gaussian``: the expected value,
    in the decision's numeraire, of choosing *after* seeing the estimate
    rather than on the prior mean alone.
    """
    return evoi_gaussian(decision, prior_mean, prior_sd, experiment_se).evsi


def experiment_value(
    information_value: float,
    opportunity_cost: OpportunityCost | float,
    fixed_cost: float,
    *,
    numeraire: str = "",
) -> ExperimentValue:
    """``information_value − opportunity_cost − fixed_cost`` in one numeraire.

    An ``OpportunityCost`` brings its numeraire along; a bare float needs
    ``numeraire`` stated. Both given must agree.
    """
    iv = _finite("information_value", information_value)
    if iv < 0.0:
        raise ValueError(f"information_value cannot be negative, got {iv}")
    fc = _finite("fixed_cost", fixed_cost)
    if fc < 0.0:
        raise ValueError(f"fixed_cost cannot be negative, got {fc}")
    detail: dict[str, str] = {}
    if isinstance(opportunity_cost, OpportunityCost):
        oc = opportunity_cost.value
        if numeraire and numeraire != opportunity_cost.numeraire:
            raise ValueError(
                f"numeraire {numeraire!r} disagrees with the opportunity cost's "
                f"{opportunity_cost.numeraire!r}"
            )
        numeraire = opportunity_cost.numeraire
        detail["opportunity_cost_formula"] = opportunity_cost.detail.get("formula", "")
    else:
        oc = _finite("opportunity_cost", float(opportunity_cost))
        if not numeraire:
            raise ValueError("a bare opportunity_cost needs an explicit numeraire")
    return ExperimentValue(
        information_value=iv,
        opportunity_cost=oc,
        fixed_cost=fc,
        net=iv - oc - fc,
        numeraire=numeraire,
        detail={**detail, "formula": "information_value - opportunity_cost - fixed_cost"},
    )
