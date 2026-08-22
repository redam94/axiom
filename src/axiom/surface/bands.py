"""A fitted surface evaluated over a dose grid, with its uncertainty attached.

A response curve is a posterior quantity and there is no honest way to draw or
report one as a line. ``forward(surface, dose, theta_mean)`` is a line — it is
the surface at *one* point of the posterior, and a plot of it says the curve is
known when it is not. Every caller that wanted a curve was reimplementing the
same loop over draws, or skipping it.

:class:`ResponseBand` is that curve as data: the dose grid, the posterior mean
and median at every grid point, and the interval at every grid point *with the
definition and mass that produced it* (rule 4). :func:`response_band` builds
one for the expected outcome and :func:`marginal_band` for the derivative —
outcome per unit dose, the quantity a dose decision is actually about.

``viz.response_curve`` and ``viz.marginal_curve`` render exactly this object
and nothing else, so there is one implementation of "evaluate the surface over
a grid through the draws" (rule 3) and a figure cannot be drawn from a band
that does not exist. ``tests/contracts/test_surface_uncertainty.py`` holds the
other half of that: every figure of a surface carries a band trace.

**What is averaged over what.** At each grid dose the intervention sets that
treatment to the level everywhere in ``window`` and leaves the others at their
observed doses; each posterior draw is pushed through the same ``forward`` the
likelihood used, and the draw's value is the mean over units and periods. The
band is therefore the uncertainty in the *expected* outcome at that dose, not a
predictive interval for one unit — ``noise`` is not added. The distinction
matters and is recorded in ``ResponseBand.kind``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import model_validator

from axiom.core import (
    Dimension,
    Interval,
    Intervention,
    NonEmptyStr,
    Spec,
    TimeWindow,
    Unsupported,
    eti,
    hdi,
    is_failure,
)
from axiom.core.intervals import IntervalDefinition

__all__ = [
    "BandKind",
    "ResponseBand",
    "SupportsBands",
    "marginal_band",
    "response_band",
]

Array = npt.NDArray[np.float64]

BandKind = Literal["response", "marginal"]
"""``response``: expected outcome at a dose. ``marginal``: its derivative in the dose."""


@runtime_checkable
class SupportsBands(Protocol):
    """What a band needs from a fit: draws under an intervention, and the fitted grid."""

    @property
    def surface(self) -> Any: ...
    @property
    def data(self) -> Mapping[str, Any]: ...
    @property
    def n_periods(self) -> int: ...
    def predict_under(
        self, iv: Intervention, window: TimeWindow | None = None, seed: int | None = None
    ) -> Any: ...
    def marginal_under(
        self,
        iv: Intervention,
        treatment: str,
        window: TimeWindow | None = None,
        seed: int | None = None,
    ) -> Any: ...


class ResponseBand(Spec):
    """A dose grid, the curve on it, and the interval around the curve at every point.

    ``lower``/``upper`` are the ``definition`` interval at ``mass`` over the
    posterior draws, computed per grid point; :meth:`interval_at` returns any
    one of them as an ``Interval`` so the definition and mass travel with the
    number rather than with the plot. ``n_draws`` is how many draws it was
    computed from, which is the other thing a reader needs to judge it.
    """

    treatment: NonEmptyStr
    outcome: NonEmptyStr
    kind: BandKind
    doses: tuple[float, ...]
    mean: tuple[float, ...]
    median: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    definition: IntervalDefinition
    mass: float
    n_draws: int
    dimension: Dimension
    dose_unit: str | None = None
    outcome_unit: str | None = None

    @model_validator(mode="after")
    def _valid(self) -> ResponseBand:
        n = len(self.doses)
        if n < 2:
            raise ValueError(f"a band needs at least two grid points, got {n}")
        for name in ("mean", "median", "lower", "upper"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} values for {n} doses")
        if not 0.0 < self.mass < 1.0:
            raise ValueError(f"mass must be in (0, 1), got {self.mass}")
        if self.n_draws < 1:
            raise ValueError(f"n_draws must be positive, got {self.n_draws}")
        for lo, up in zip(self.lower, self.upper, strict=True):
            if lo > up:
                raise ValueError(f"lower {lo} exceeds upper {up}")
        return self

    @property
    def n_grid(self) -> int:
        return len(self.doses)

    @property
    def width(self) -> tuple[float, ...]:
        """Interval width at each grid point — how much the curve is *not* known."""
        return tuple(up - lo for lo, up in zip(self.lower, self.upper, strict=True))

    def interval_at(self, index: int) -> Interval:
        """The interval at one grid point, carrying its definition and mass."""
        return Interval(
            lower=self.lower[index],
            upper=self.upper[index],
            definition=self.definition,
            mass=self.mass,
        )

    def label(self) -> str:
        """How to describe the band in a legend or a caption."""
        return f"{self.mass:.0%} {self.definition}"

    def axis_titles(self) -> tuple[str, str]:
        """``(x, y)`` titles with units, for whatever is drawing it."""
        dose = f"{self.treatment} dose" + (f" ({self.dose_unit})" if self.dose_unit else "")
        unit = f" ({self.outcome_unit})" if self.outcome_unit else ""
        if self.kind == "marginal":
            per = f" per {self.dose_unit}" if self.dose_unit else " per dose"
            return dose, f"d {self.outcome} / d {self.treatment}{unit}{per}"
        return dose, f"expected {self.outcome}{unit}"


def _grid(
    result: SupportsBands, treatment: str, doses: Sequence[float] | None, n_grid: int
) -> Array:
    if doses is not None:
        grid = np.asarray(list(doses), dtype=np.float64)
        if grid.size < 2:
            raise ValueError("doses needs at least two levels")
        if np.any(grid < 0.0):
            raise ValueError("doses must be non-negative")
        return grid
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2")
    observed = np.asarray(result.data[treatment], dtype=np.float64)
    top = float(np.nanmax(observed)) if observed.size else 1.0
    return np.linspace(0.0, top if top > 0 else 1.0, n_grid)


def _summarise(
    values: Array, definition: IntervalDefinition, mass: float
) -> tuple[float, float, float, float]:
    per_draw = values.reshape(values.shape[0] * values.shape[1], -1).mean(axis=1)
    finite = per_draw[np.isfinite(per_draw)]
    if finite.size == 0:
        raise ValueError("every draw at this dose is non-finite; the surface cannot be summarised")
    interval = eti(finite, mass) if definition == "eti" else hdi(finite, mass)
    return (
        float(finite.mean()),
        float(np.median(finite)),
        float(interval.lower),
        float(interval.upper),
    )


def _band(
    result: SupportsBands,
    treatment: str,
    kind: BandKind,
    *,
    doses: Sequence[float] | None,
    n_grid: int,
    mass: float,
    definition: IntervalDefinition,
    window: TimeWindow | None,
    seed: int | None,
) -> ResponseBand | Unsupported:
    spec = result.surface.spec
    if treatment not in spec.treatment_names:
        raise ValueError(f"no treatment {treatment!r}; have {list(spec.treatment_names)}")
    grid = _grid(result, treatment, doses, n_grid)
    win = window if window is not None else TimeWindow(start=0, stop=int(result.n_periods))

    means: list[float] = []
    medians: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    n_draws = 0
    for level in grid:
        iv = Intervention(doses={treatment: float(level)}, mode="set")
        drawn = (
            result.predict_under(iv, window=win, seed=seed)
            if kind == "response"
            else result.marginal_under(iv, treatment, window=win, seed=seed)
        )
        if is_failure(drawn):
            return (
                drawn
                if isinstance(drawn, Unsupported)
                else Unsupported(
                    reason=f"the fit cannot be evaluated under an intervention: {drawn}",
                    missing=(),
                )
            )
        values = np.asarray(drawn.values, dtype=np.float64)
        n_draws = int(values.shape[0] * values.shape[1])
        m, med, lo, up = _summarise(values, definition, mass)
        means.append(m)
        medians.append(med)
        lower.append(lo)
        upper.append(up)

    entity = spec.treatment(treatment)
    dimension = (
        spec.outcome_dimension if kind == "response" else spec.outcome_dimension / entity.dimension
    )
    return ResponseBand(
        treatment=treatment,
        outcome=spec.outcome.name,
        kind=kind,
        doses=tuple(float(v) for v in grid),
        mean=tuple(means),
        median=tuple(medians),
        lower=tuple(lower),
        upper=tuple(upper),
        definition=definition,
        mass=mass,
        n_draws=n_draws,
        dimension=dimension,
        dose_unit=entity.unit,
        outcome_unit=spec.outcome.unit,
    )


def response_band(
    result: SupportsBands,
    treatment: str,
    *,
    doses: Sequence[float] | None = None,
    n_grid: int = 25,
    mass: float = 0.9,
    definition: IntervalDefinition = "eti",
    window: TimeWindow | None = None,
    seed: int | None = None,
) -> ResponseBand | Unsupported:
    """Expected outcome against dose, with the posterior interval at every grid point.

    ``doses`` names the grid explicitly; otherwise ``n_grid`` points run from
    zero to the largest observed dose. Every draw goes through the same
    ``forward`` the likelihood used, so this is the curve the model actually
    believes rather than the curve at the posterior mean of its parameters —
    for a nonlinear surface those are not the same line.
    """
    return _band(
        result,
        treatment,
        "response",
        doses=doses,
        n_grid=n_grid,
        mass=mass,
        definition=definition,
        window=window,
        seed=seed,
    )


def marginal_band(
    result: SupportsBands,
    treatment: str,
    *,
    doses: Sequence[float] | None = None,
    n_grid: int = 25,
    mass: float = 0.9,
    definition: IntervalDefinition = "eti",
    window: TimeWindow | None = None,
    seed: int | None = None,
) -> ResponseBand | Unsupported:
    """``d outcome / d dose`` against dose, with the posterior interval at every point.

    The quantity a titration decision is about: whether the next unit of dose
    helps. Its sign is the thing worth an interval — a marginal effect whose
    band straddles zero is a dose at which the model does not know whether to
    go up or down, and a line through the posterior mean will not say so.
    """
    return _band(
        result,
        treatment,
        "marginal",
        doses=doses,
        n_grid=n_grid,
        mass=mass,
        definition=definition,
        window=window,
        seed=seed,
    )
