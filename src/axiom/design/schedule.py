"""Temporal-contrast patterns for a single treatment: what identifies carryover.

A carryover's weights sum to one, so a dose held constant passes through
it unchanged (``Σ_l w_l · x = x``) and the data carry *no* information on
the decay — the Fisher information of ``lam`` under a constant schedule is
exactly zero (``design.structural``). Identification comes from temporal
contrast: pulses, ramps, switchbacks. This module builds those patterns as
``Schedule`` specs and scores their contrast; ``design.structural.
fisher_information`` on a surface with carryover is the model-based
verdict the score is a heuristic for.

``contrast_score`` is the mean squared first difference divided by the mean
squared dose,

    score = [Σ_t (d_t − d_{t−1})² / (T − 1)] / [Σ_t d_t² / T],

which is scale-free (doubling every dose leaves it unchanged), ``0`` for a
constant schedule (and for an all-zero one), and at most ``4`` (a
full-amplitude alternation). It is defined for every schedule with at
least two periods.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from axiom.core import NonEmptyStr, Spec

__all__ = [
    "Pattern",
    "Schedule",
    "alternating",
    "constant",
    "contrast_score",
    "pulse",
    "ramp",
    "random_switchback",
]

Array = npt.NDArray[np.float64]
Pattern = Literal["constant", "pulse", "ramp", "random_switchback", "alternating"]


class Schedule(Spec):
    """One treatment's dose per period: ``doses[t]`` for ``t = 0 … n_periods − 1``.

    ``pattern`` names the generator and ``detail`` carries its numbers
    (levels, on/off lengths, seed). ``as_grid(n_units)`` lays the schedule
    out as the ``(n_units, n_periods)`` array a carryover surface's
    ``forward`` reads, the same doses in every unit.
    """

    pattern: Pattern
    n_periods: int = Field(ge=1)
    doses: tuple[float, ...]
    treatment: NonEmptyStr = "dose"
    detail: dict[str, float] = {}

    @model_validator(mode="after")
    def _consistent(self) -> Schedule:
        if len(self.doses) != self.n_periods:
            raise ValueError(f"doses has {len(self.doses)} entries for n_periods={self.n_periods}")
        if not all(math.isfinite(d) for d in self.doses):
            raise ValueError("doses must be finite")
        return self

    def as_array(self) -> Array:
        return np.asarray(self.doses, dtype=np.float64)

    def as_grid(self, n_units: int = 1) -> Array:
        """``(n_units, n_periods)``: the schedule repeated in every unit, time last."""
        if n_units < 1:
            raise ValueError("n_units must be positive")
        return np.tile(self.as_array()[None, :], (n_units, 1))

    @property
    def total(self) -> float:
        return float(np.sum(self.as_array()))

    @property
    def mean(self) -> float:
        return float(np.mean(self.as_array()))


def _check_periods(n_periods: int) -> None:
    if n_periods < 1:
        raise ValueError("n_periods must be positive")


def _check_level(name: str, level: float) -> None:
    if not math.isfinite(level):
        raise ValueError(f"{name} must be finite, got {level}")


def constant(n_periods: int, dose: float, *, treatment: str = "dose") -> Schedule:
    """The same dose in every period — no temporal contrast."""
    _check_periods(n_periods)
    _check_level("dose", dose)
    return Schedule(
        pattern="constant",
        n_periods=n_periods,
        doses=(float(dose),) * n_periods,
        treatment=treatment,
        detail={"dose": float(dose)},
    )


def pulse(
    n_periods: int,
    high: float,
    low: float,
    *,
    on: int,
    off: int,
    start_on: bool = True,
    treatment: str = "dose",
) -> Schedule:
    """``on`` periods at ``high`` then ``off`` at ``low``, repeating.

    ``start_on`` picks the phase: ``True`` begins with the high block.
    """
    _check_periods(n_periods)
    _check_level("high", high)
    _check_level("low", low)
    if on < 1 or off < 1:
        raise ValueError("on and off must be positive period counts")
    cycle = [high] * on + [low] * off if start_on else [low] * off + [high] * on
    doses = tuple(float(cycle[t % len(cycle)]) for t in range(n_periods))
    return Schedule(
        pattern="pulse",
        n_periods=n_periods,
        doses=doses,
        treatment=treatment,
        detail={
            "high": float(high),
            "low": float(low),
            "on": float(on),
            "off": float(off),
            "start_on": 1.0 if start_on else 0.0,
        },
    )


def alternating(n_periods: int, high: float, low: float, *, treatment: str = "dose") -> Schedule:
    """``high, low, high, low, …`` — the maximal-contrast pulse with ``on = off = 1``."""
    s = pulse(n_periods, high, low, on=1, off=1, treatment=treatment)
    return s.model_copy(
        update={"pattern": "alternating", "detail": {"high": float(high), "low": float(low)}}
    )


def ramp(n_periods: int, start: float, stop: float, *, treatment: str = "dose") -> Schedule:
    """Linear from ``start`` at period 0 to ``stop`` at the last period (``start`` alone if one)."""
    _check_periods(n_periods)
    _check_level("start", start)
    _check_level("stop", stop)
    values = np.linspace(start, stop, n_periods) if n_periods > 1 else np.asarray([start])
    return Schedule(
        pattern="ramp",
        n_periods=n_periods,
        doses=tuple(float(v) for v in values),
        treatment=treatment,
        detail={"start": float(start), "stop": float(stop)},
    )


def random_switchback(
    n_periods: int,
    high: float,
    low: float,
    *,
    seed: int | None,
    p_high: float = 0.5,
    treatment: str = "dose",
) -> Schedule:
    """Each period independently at ``high`` with probability ``p_high``, else ``low``; seeded."""
    _check_periods(n_periods)
    _check_level("high", high)
    _check_level("low", low)
    if not 0.0 < p_high < 1.0:
        raise ValueError(f"p_high must be in (0, 1), got {p_high}")
    rng = np.random.default_rng(seed)
    on = rng.random(n_periods) < p_high
    doses = tuple(float(high) if flag else float(low) for flag in on)
    detail = {"high": float(high), "low": float(low), "p_high": float(p_high)}
    if seed is not None:
        detail["seed"] = float(seed)
    return Schedule(
        pattern="random_switchback",
        n_periods=n_periods,
        doses=doses,
        treatment=treatment,
        detail=detail,
    )


def contrast_score(schedule: Schedule) -> float:
    """Mean squared first difference over mean squared dose (module docstring); ``0`` if flat.

    A one-period schedule has no first difference and scores ``0``.
    """
    d = schedule.as_array()
    if d.size < 2:
        return 0.0
    power = float(np.mean(d**2))
    if power == 0.0:
        return 0.0
    return float(np.mean(np.diff(d) ** 2) / power)
