"""How the recommended design moves when one input moves: sweeps, tipping points, elasticity.

Ported by specification from the parent's ``planning/experiment_sensitivity.py``.
``perturb`` re-scores every candidate (``design.optimizer.evaluate_candidate``)
at each value of a grid over one input and records the winner — the
candidate with the largest net value — at each point. A *tipping point* is
a pair of adjacent grid values between which the winner changes; the table
lists them all, so "the recommendation holds for value per outcome between
a and b" is read straight off.

Inputs that can be swept:

* ``prior_sd``, ``value_per_outcome``, ``discount_rate`` — global; the grid
  holds absolute values.
* ``experiment_se``, ``holdout_fraction`` — per candidate; the grid holds
  *multipliers* applied to every candidate's own value (a multiplier of
  ``1`` is the base point), because a common absolute value would erase the
  differences between candidates that the question is about.

The base inputs are explicit arguments; nothing is defaulted from the
grid. ``elasticity`` is the finite-difference elasticity of each
candidate's net value to the swept input at the base point,
``(Δnet / net) / (Δx / x)`` over the two grid values bracketing the base.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, model_validator

from axiom.core import NonEmptyStr, Spec
from axiom.design.evoi import DecisionSpec
from axiom.design.optimizer import (
    DesignCandidate,
    EconomicInputs,
    evaluate_candidate,
)

__all__ = ["Parameter", "SensitivityTable", "elasticity", "perturb"]

Parameter = Literal[
    "prior_sd", "experiment_se", "value_per_outcome", "discount_rate", "holdout_fraction"
]
Mode = Literal["absolute", "multiplier"]
_MULTIPLIER: frozenset[str] = frozenset({"experiment_se", "holdout_fraction"})


class SensitivityTable(Spec):
    """Net value of every candidate at every grid value of one input, and where the winner flips.

    ``net_values[i][j]`` is candidate ``j`` at grid value ``i``; ``winners[i]``
    the argmax there (ties to the earlier name); ``tipping_points`` the
    ``(grid[i], grid[i+1])`` pairs across which the winner changes.
    ``base_value`` / ``base_winner`` / ``base_net_values`` are at the inputs
    as given (``1.0`` for a multiplier-mode input).
    """

    parameter: Parameter
    mode: Mode
    grid: tuple[float, ...] = Field(min_length=2)
    candidates: tuple[NonEmptyStr, ...] = Field(min_length=1)
    net_values: tuple[tuple[float, ...], ...]
    winners: tuple[NonEmptyStr, ...]
    tipping_points: tuple[tuple[float, float], ...]
    base_value: float
    base_winner: NonEmptyStr
    base_net_values: tuple[float, ...]
    numeraire: str = ""
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _shapes(self) -> SensitivityTable:
        g, k = len(self.grid), len(self.candidates)
        if len(self.net_values) != g or any(len(row) != k for row in self.net_values):
            raise ValueError("net_values must be (len(grid), len(candidates))")
        if len(self.winners) != g:
            raise ValueError("one winner per grid value")
        if len(self.base_net_values) != k:
            raise ValueError("one base net value per candidate")
        if any(b <= a for a, b in zip(self.grid, self.grid[1:], strict=False)):
            raise ValueError("grid must be strictly increasing")
        return self

    @property
    def stable(self) -> bool:
        """``True`` when the same candidate wins across the whole grid."""
        return not self.tipping_points


def _winner(names: Sequence[str], nets: Sequence[float]) -> str:
    best = max(range(len(names)), key=lambda j: (nets[j], -j))
    return names[best]


def _apply(
    parameter: Parameter,
    value: float,
    candidates: Sequence[DesignCandidate],
    prior_sd: float,
    economics: EconomicInputs,
) -> tuple[tuple[DesignCandidate, ...], float, EconomicInputs]:
    """The inputs with ``parameter`` set to ``value`` (or scaled by it)."""
    if parameter == "prior_sd":
        return tuple(candidates), value, economics
    if parameter == "value_per_outcome":
        vpo = economics.value_per_outcome.model_copy(update={"value": value})
        return tuple(candidates), prior_sd, economics.model_copy(update={"value_per_outcome": vpo})
    if parameter == "discount_rate":
        return tuple(candidates), prior_sd, economics.model_copy(update={"discount_rate": value})
    if parameter == "experiment_se":
        scaled = tuple(
            c.model_copy(update={"experiment_se": c.experiment_se * value}) for c in candidates
        )
        return scaled, prior_sd, economics
    if parameter == "holdout_fraction":
        scaled = tuple(
            c.model_copy(update={"holdout_fraction": c.holdout_fraction * value})
            for c in candidates
        )
        return scaled, prior_sd, economics
    raise ValueError(f"unknown parameter {parameter!r}")


def _base_value(parameter: Parameter, prior_sd: float, economics: EconomicInputs) -> float:
    if parameter == "prior_sd":
        return prior_sd
    if parameter == "value_per_outcome":
        return economics.value_per_outcome.value
    if parameter == "discount_rate":
        return economics.discount_rate
    return 1.0


def perturb(
    candidates: Sequence[DesignCandidate],
    decision: DecisionSpec,
    prior_mean: float,
    prior_sd: float,
    economics: EconomicInputs,
    parameter: Parameter,
    grid: Sequence[float],
    *,
    alpha: float = 0.05,
    effect: float | None = None,
) -> SensitivityTable:
    """Re-score every candidate at each grid value of ``parameter``; report winners and flips.

    The grid must be strictly increasing with at least two values; values
    that make an input invalid (a multiplier pushing a holdout fraction to
    ``1``) raise the same ``ValueError`` the spec would. ``alpha`` and
    ``effect`` pass through to ``evaluate_candidate``.
    """
    if not candidates:
        raise ValueError("need at least one candidate")
    names = tuple(c.name for c in candidates)
    if len(set(names)) != len(names):
        raise ValueError(f"candidate names must be distinct, got {list(names)}")
    g = tuple(float(v) for v in grid)
    if len(g) < 2 or any(b <= a for a, b in zip(g, g[1:], strict=False)):
        raise ValueError("grid must be strictly increasing with at least two values")
    if any(not math.isfinite(v) for v in g):
        raise ValueError("grid values must be finite")

    def score_at(value: float) -> tuple[float, ...]:
        cands, sd, econ = _apply(parameter, value, candidates, prior_sd, economics)
        return tuple(
            evaluate_candidate(
                c, decision, prior_mean, sd, econ, alpha=alpha, effect=effect
            ).net_value
            for c in cands
        )

    rows = tuple(score_at(v) for v in g)
    winners = tuple(_winner(names, row) for row in rows)
    tips = tuple((g[i], g[i + 1]) for i in range(len(g) - 1) if winners[i] != winners[i + 1])
    base = _base_value(parameter, prior_sd, economics)
    base_row = score_at(base)
    mode: Mode = "multiplier" if parameter in _MULTIPLIER else "absolute"
    return SensitivityTable(
        parameter=parameter,
        mode=mode,
        grid=g,
        candidates=names,
        net_values=rows,
        winners=winners,
        tipping_points=tips,
        base_value=base,
        base_winner=_winner(names, base_row),
        base_net_values=base_row,
        numeraire=economics.value_per_outcome.numeraire,
        detail={
            "winner": "argmax net_value at each grid value; ties to the earlier candidate",
            "tipping_point": "adjacent grid values between which the winner changes",
            "mode": (
                "grid values multiply each candidate's own value"
                if mode == "multiplier"
                else "grid values replace the input"
            ),
            "prior_mean": f"{prior_mean:.12g}",
            "alpha": f"{alpha:.12g}",
        },
    )


def elasticity(table: SensitivityTable) -> dict[str, float]:
    """``(Δnet / net_base) / (Δx / x_base)`` per candidate, from the grid values around the base.

    Uses the nearest grid value below and the nearest at or above the base
    (central when the base sits inside the grid). A ``ValueError`` when the
    grid does not bracket the base or the base value is zero; a candidate
    whose base net value is zero gets ``inf`` (or ``nan`` when its change
    is also zero), which is what the ratio says.
    """
    x0 = table.base_value
    if x0 == 0.0:
        raise ValueError("elasticity is undefined at a zero base value")
    below = [i for i, v in enumerate(table.grid) if v < x0]
    above = [i for i, v in enumerate(table.grid) if v > x0]
    if not below or not above:
        raise ValueError(
            f"grid {table.grid} does not bracket the base value {x0}; add values on both sides"
        )
    lo, hi = below[-1], above[0]
    dx = table.grid[hi] - table.grid[lo]
    out: dict[str, float] = {}
    for j, name in enumerate(table.candidates):
        d_net = table.net_values[hi][j] - table.net_values[lo][j]
        base = table.base_net_values[j]
        slope = d_net / dx
        if base == 0.0:
            out[name] = math.nan if slope == 0.0 else math.copysign(math.inf, slope)
        else:
            out[name] = slope * x0 / base
    return out
