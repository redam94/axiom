"""Credible intervals that carry their own definition.

An ``Interval`` cannot be constructed without ``definition`` (``"eti"`` or
``"hdi"``) and ``mass`` (the probability it contains). That is gate 6 made
constructive: there is no interval in axiom whose meaning is in a variable
name. The parent shipped the same quantity at two masses and two definitions
in two places before anyone noticed.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import field_validator

from axiom.core.spec import Spec

__all__ = [
    "Interval",
    "IntervalDefinition",
    "Summary",
    "eti",
    "hdi",
    "interval",
    "summarize",
    "wald",
]

IntervalDefinition = Literal["eti", "hdi", "wald"]
"""``eti``/``hdi`` are posterior credible intervals; ``wald`` is a frequentist
``estimate ± z · se`` confidence interval. The type says which."""


class Interval(Spec):
    """``[lower, upper]`` with the definition and the mass it was computed at."""

    lower: float
    upper: float
    definition: IntervalDefinition
    mass: float

    @field_validator("mass")
    @classmethod
    def _mass_in_unit_interval(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"mass must be in (0, 1), got {v}")
        return v

    def model_post_init(self, __context: object) -> None:
        if not (np.isfinite(self.lower) and np.isfinite(self.upper)):
            raise ValueError(f"interval bounds must be finite, got [{self.lower}, {self.upper}]")
        if self.lower > self.upper:
            raise ValueError(f"lower {self.lower} exceeds upper {self.upper}")

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, x: float) -> bool:
        return self.lower <= x <= self.upper

    def __str__(self) -> str:
        pct = int(round(self.mass * 100))
        return f"[{self.lower:.6g}, {self.upper:.6g}] ({pct}% {self.definition.upper()})"


class Summary(Spec):
    """Point summaries of a set of draws plus one provenance-carrying interval."""

    mean: float
    median: float
    sd: float
    interval: Interval
    n: int


def _draws_1d(draws: npt.ArrayLike) -> npt.NDArray[np.float64]:
    x = np.asarray(draws, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 2:
        raise ValueError("need at least two finite draws to form an interval")
    return np.asarray(x, dtype=np.float64)


def eti(draws: npt.ArrayLike, mass: float) -> Interval:
    """Equal-tailed interval: the ``(1-mass)/2`` and ``1-(1-mass)/2`` quantiles."""
    x = _draws_1d(draws)
    tail = (1.0 - mass) / 2.0
    lo, hi = np.quantile(x, [tail, 1.0 - tail])
    return Interval(lower=float(lo), upper=float(hi), definition="eti", mass=mass)


def hdi(draws: npt.ArrayLike, mass: float) -> Interval:
    """Highest-density interval: the narrowest window containing ``mass`` of the draws.

    Computed on the sorted sample (Chen & Shao 1999); unimodal assumption, as
    in every common implementation. Ties in width resolve to the lowest.
    """
    x = np.sort(_draws_1d(draws))
    n = x.size
    k = int(np.ceil(mass * n))
    if k >= n:
        return Interval(lower=float(x[0]), upper=float(x[-1]), definition="hdi", mass=mass)
    widths = x[k:] - x[: n - k]
    i = int(np.argmin(widths))
    return Interval(lower=float(x[i]), upper=float(x[i + k]), definition="hdi", mass=mass)


def interval(draws: npt.ArrayLike, *, definition: IntervalDefinition, mass: float) -> Interval:
    if definition == "eti":
        return eti(draws, mass)
    if definition == "hdi":
        return hdi(draws, mass)
    raise ValueError(f"{definition!r} is not a posterior interval; use wald() for a CI")


def wald(estimate: float, se: float, mass: float) -> Interval:
    """Frequentist ``estimate ± z_{(1+mass)/2} · se``, labelled as such."""
    if se < 0 or not np.isfinite(se):
        raise ValueError(f"standard error must be finite and non-negative, got {se}")
    z = float(_norm_ppf((1.0 + mass) / 2.0))
    return Interval(lower=estimate - z * se, upper=estimate + z * se, definition="wald", mass=mass)


def _norm_ppf(p: float) -> float:
    from scipy.special import ndtri

    return float(ndtri(p))


def summarize(
    draws: npt.ArrayLike, *, definition: IntervalDefinition = "hdi", mass: float = 0.9
) -> Summary:
    x = _draws_1d(draws)
    return Summary(
        mean=float(x.mean()),
        median=float(np.median(x)),
        sd=float(x.std(ddof=1)),
        interval=interval(x, definition=definition, mass=mass),
        n=int(x.size),
    )
