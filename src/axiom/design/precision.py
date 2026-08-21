"""Cost per outcome unit: its interval, the power to bound it, and the largest detectable value.

Ported from the parent's ``planning/cpa.py`` (golden ``planning.cpa::*``,
bit-stable) with the vocabulary generalized: *cost per acquisition* becomes
cost per outcome unit, ``cost / effect`` for a known experiment cost and an
estimated effect ``effect ± effect_se``.

A ratio with a noisy denominator has no symmetric interval. Two are given:

* the **Fieller** interval, exact when the numerator is a known constant: the
  set of ratios ``r`` with ``(cost − r · effect)² ≤ z² r² effect_se²``, whose
  roots are ``cost / (effect ± z · effect_se)`` — the effect's Wald interval
  inverted. When that Wald interval reaches zero the set is unbounded above
  and the result says so (``status="unbounded"``, ``upper=None``) instead of
  reporting a finite number;
* the **naive** delta-method interval ``estimate ± z · estimate · effect_se /
  effect``, a ``core.Interval(definition="wald")``, for comparison. It is
  symmetric, too narrow on the right, and the parent shipped it as the
  answer until the Fieller bounds were added.

``cost_per_outcome_power`` is the probability that the Fieller upper bound
comes out finite (or below a threshold); ``max_detectable_cost_per_outcome``
is the cost per outcome unit at the minimum detectable effect.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import field_validator, model_validator
from scipy.special import ndtr, ndtri

from axiom.core import Interval, Spec, wald

__all__ = [
    "CostPerOutcomeInterval",
    "CostPerOutcomePower",
    "cost_per_outcome_interval",
    "cost_per_outcome_power",
    "max_detectable_cost_per_outcome",
]


def _positive(name: str, v: float) -> float:
    if not (math.isfinite(v) and v > 0):
        raise ValueError(f"{name} must be finite and positive, got {v}")
    return float(v)


def _alpha(v: float) -> float:
    if not 0.0 < v < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {v}")
    return float(v)


class CostPerOutcomeInterval(Spec):
    """``cost / effect`` with a Fieller interval and the naive delta-method interval beside it.

    ``lower``/``upper`` are the Fieller bounds at confidence ``1 − alpha``;
    ``upper`` is ``None`` when the effect's Wald interval reaches zero and
    the ratio is unbounded above (``status="unbounded"``). ``effect_interval``
    and ``naive_interval`` are Wald intervals at the same mass.
    """

    cost: float
    effect: float
    effect_se: float
    alpha: float
    estimate: float
    lower: float
    upper: float | None
    method: Literal["fieller"] = "fieller"
    status: Literal["bounded", "unbounded"]
    effect_interval: Interval
    naive_se: float
    naive_interval: Interval

    @field_validator("alpha")
    @classmethod
    def _alpha_in_unit_interval(cls, v: float) -> float:
        return _alpha(v)

    @model_validator(mode="after")
    def _consistent(self) -> CostPerOutcomeInterval:
        if (self.upper is None) != (self.status == "unbounded"):
            raise ValueError("upper is None iff status is 'unbounded'")
        if self.upper is not None and self.lower > self.upper:
            raise ValueError(f"lower {self.lower} exceeds upper {self.upper}")
        return self

    @property
    def mass(self) -> float:
        return 1.0 - self.alpha

    def contains(self, x: float) -> bool:
        return self.lower <= x and (self.upper is None or x <= self.upper)


class CostPerOutcomePower(Spec):
    """Power to bound the cost per outcome unit (below ``threshold`` if one is set).

    ``effect_required`` is the effect the estimate must exceed by ``z ·
    effect_se`` for the Fieller upper bound to fall below ``threshold``
    (``cost / threshold``; ``0`` when no threshold, i.e. for the bound to be
    finite at all). ``power`` is ``Φ((true_effect − effect_required) /
    effect_se − z_{1 − alpha/2})``.
    """

    cost: float
    true_effect: float
    effect_se: float
    alpha: float
    threshold: float | None
    effect_required: float
    power: float

    @field_validator("alpha")
    @classmethod
    def _alpha_in_unit_interval(cls, v: float) -> float:
        return _alpha(v)

    @field_validator("power")
    @classmethod
    def _probability(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"power must be in [0, 1], got {v}")
        return v


def cost_per_outcome_interval(
    cost: float, effect: float, effect_se: float, *, alpha: float = 0.05
) -> CostPerOutcomeInterval:
    """Fieller and naive intervals for ``cost / effect`` at confidence ``1 − alpha``.

    ``effect`` must be positive: a cost per outcome unit is undefined for a
    non-positive point estimate. A positive estimate whose interval reaches
    zero gives ``status="unbounded"`` with a finite lower bound only.
    """
    c = _positive("cost", cost)
    b = _positive("effect", effect)
    se = _positive("effect_se", effect_se)
    a = _alpha(alpha)
    mass = 1.0 - a
    effect_interval = wald(b, se, mass)
    estimate = c / b
    naive_se = estimate * se / b
    naive_interval = wald(estimate, naive_se, mass)
    lower = c / effect_interval.upper
    if effect_interval.lower > 0.0:
        return CostPerOutcomeInterval(
            cost=c,
            effect=b,
            effect_se=se,
            alpha=a,
            estimate=estimate,
            lower=lower,
            upper=c / effect_interval.lower,
            status="bounded",
            effect_interval=effect_interval,
            naive_se=naive_se,
            naive_interval=naive_interval,
        )
    return CostPerOutcomeInterval(
        cost=c,
        effect=b,
        effect_se=se,
        alpha=a,
        estimate=estimate,
        lower=lower,
        upper=None,
        status="unbounded",
        effect_interval=effect_interval,
        naive_se=naive_se,
        naive_interval=naive_interval,
    )


def cost_per_outcome_power(
    cost: float,
    true_effect: float,
    effect_se: float,
    *,
    alpha: float = 0.05,
    threshold: float | None = None,
) -> CostPerOutcomePower:
    """Probability the Fieller upper bound is finite (``threshold=None``) or below ``threshold``.

    The upper bound ``cost / (effect_hat − z · effect_se)`` is below
    ``threshold`` iff ``effect_hat > cost / threshold + z · effect_se``; with
    ``effect_hat ~ N(true_effect, effect_se²)`` that has probability
    ``Φ((true_effect − cost / threshold) / effect_se − z)``. With no
    threshold the requirement is that the bound exists, ``cost / threshold →
    0``, which is the power of a one-sided test of the effect against zero
    at level ``alpha / 2``.
    """
    c = _positive("cost", cost)
    if not math.isfinite(true_effect):
        raise ValueError(f"true_effect must be finite, got {true_effect}")
    se = _positive("effect_se", effect_se)
    a = _alpha(alpha)
    required = 0.0 if threshold is None else c / _positive("threshold", threshold)
    z = float(ndtri(1.0 - a / 2.0))
    power = float(ndtr((true_effect - required) / se - z))
    return CostPerOutcomePower(
        cost=c,
        true_effect=float(true_effect),
        effect_se=se,
        alpha=a,
        threshold=None if threshold is None else float(threshold),
        effect_required=required,
        power=power,
    )


def max_detectable_cost_per_outcome(cost: float, effect_mde: float) -> float:
    """``cost / effect_mde``: the largest cost per outcome unit the experiment can bound.

    An effect at the minimum detectable effect is the smallest the design is
    powered for; the ratio at that effect is the largest cost per outcome
    unit the design can establish.
    """
    return _positive("cost", cost) / _positive("effect_mde", effect_mde)
