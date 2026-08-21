"""Panel-simulation primitives: dose plans, long frames, and balanced panels from arrays.

These are the pieces every simulated panel world is assembled from. They
know nothing about response surfaces: a ``DosePlan`` says how doses are
drawn, ``long_frame`` turns ``(n_units, n_periods)`` arrays into the long
table a ``Panel`` holds, and ``panel_from_arrays`` attaches the ``RoleMap``
so the result is a balanced, role-tagged ``Panel`` with no step in between
where a column could lose its role.

Layout convention (shared with ``surface.model.prepare``): every wide array
is ``(n_units, n_periods)``; the long frame is unit-major, period-minor,
which is the order ``Panel`` sorts into, so ``panel.array(column)`` gives
the same array back.

Public functions take ``seed: int | None``; the ``_rng`` variants take a
``numpy.random.Generator`` and are what the worlds call so that one
generator threads through a whole simulation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, model_validator

from axiom.core import Covariate, Outcome, Spec, Treatment
from axiom.data import Panel, RoleMap

__all__ = [
    "DoseDistribution",
    "DosePlan",
    "draw_doses",
    "long_frame",
    "panel_from_arrays",
    "unit_labels",
]

Array = npt.NDArray[np.float64]
DoseDistribution = Literal["lognormal", "uniform"]
"""How simulated doses are drawn around ``DosePlan.scale``."""


class DosePlan(Spec):
    """How doses are drawn for one treatment: a family, a scale, a spread, and a zero share.

    ``lognormal``: ``dose = scale · exp(spread · z)``, ``z ~ N(0, 1)`` — the
    median dose is ``scale`` and ``spread`` is the log-scale standard
    deviation. ``uniform``: ``dose ~ U(scale · (1 − spread), scale · (1 +
    spread))`` with ``spread ≤ 1`` — the mean dose is ``scale``. In either
    family a share ``zero_fraction`` of (unit, period) cells is set to zero
    dose, which is what identifies the intercept separately from a
    saturating response when doses never otherwise approach zero.
    """

    distribution: DoseDistribution = "lognormal"
    scale: float = Field(default=1.0, gt=0)
    spread: float = Field(default=0.5, ge=0)
    zero_fraction: float = Field(default=0.0, ge=0, lt=1)

    @model_validator(mode="after")
    def _uniform_spread(self) -> DosePlan:
        if self.distribution == "uniform" and self.spread > 1.0:
            raise ValueError(
                f"a uniform dose plan needs spread <= 1 so doses stay non-negative, "
                f"got {self.spread}"
            )
        return self


def draw_doses_rng(plan: DosePlan, n_units: int, n_periods: int, rng: np.random.Generator) -> Array:
    """``(n_units, n_periods)`` doses from ``plan`` using the caller's generator."""
    if n_units < 1 or n_periods < 1:
        raise ValueError(f"need n_units >= 1 and n_periods >= 1, got {n_units}, {n_periods}")
    shape = (n_units, n_periods)
    if plan.distribution == "lognormal":
        doses = plan.scale * np.exp(plan.spread * rng.standard_normal(shape))
    else:
        doses = plan.scale * (1.0 + plan.spread * (2.0 * rng.random(shape) - 1.0))
    if plan.zero_fraction > 0:
        doses = np.where(rng.random(shape) < plan.zero_fraction, 0.0, doses)
    return np.asarray(doses, dtype=np.float64)


def draw_doses(plan: DosePlan, n_units: int, n_periods: int, *, seed: int | None = None) -> Array:
    """``(n_units, n_periods)`` doses from ``plan``; deterministic given ``seed``."""
    return draw_doses_rng(plan, n_units, n_periods, np.random.default_rng(seed))


def unit_labels(n_units: int, prefix: str = "u") -> tuple[str, ...]:
    """``("u00", "u01", …)`` — zero-padded so that string order is numeric order.

    ``Panel`` sorts units as strings; zero-padding keeps the panel's unit
    order equal to the index order the arrays were generated in.
    """
    if n_units < 1:
        raise ValueError(f"need n_units >= 1, got {n_units}")
    width = len(str(n_units - 1))
    return tuple(f"{prefix}{i:0{width}d}" for i in range(n_units))


def _as_wide(name: str, values: npt.ArrayLike, n_units: int, n_periods: int) -> Array:
    a = np.asarray(values, dtype=np.float64)
    if a.shape != (n_units, n_periods):
        raise ValueError(
            f"column {name!r} has shape {a.shape}; expected (n_units, n_periods) = "
            f"{(n_units, n_periods)}"
        )
    if not np.all(np.isfinite(a)):
        raise ValueError(f"column {name!r} has non-finite values; a simulated panel is complete")
    return a


def long_frame(
    columns: Mapping[str, npt.ArrayLike],
    *,
    units: Sequence[str],
    periods: Sequence[int] | None = None,
    unit_column: str = "unit",
    time_column: str = "t",
) -> pd.DataFrame:
    """Stack ``(n_units, n_periods)`` arrays into a unit-major, period-minor long table.

    ``periods`` defaults to ``0 … n_periods − 1``; its length is read off the
    first column. Column names may not collide with the unit or time column.
    """
    if not columns:
        raise ValueError("long_frame needs at least one column")
    first = np.asarray(next(iter(columns.values())), dtype=np.float64)
    if first.ndim != 2:
        raise ValueError(
            f"columns must be 2-D (n_units, n_periods) arrays, got shape {first.shape}"
        )
    n_units, n_periods = first.shape
    if len(units) != n_units:
        raise ValueError(f"{len(units)} unit labels for {n_units} rows of the arrays")
    if len(set(units)) != len(units):
        raise ValueError("unit labels must be distinct")
    period_values = list(range(n_periods)) if periods is None else list(periods)
    if len(period_values) != n_periods:
        raise ValueError(f"{len(period_values)} periods for {n_periods} columns of the arrays")
    if unit_column in columns or time_column in columns:
        raise ValueError(
            f"{unit_column!r} / {time_column!r} are index columns and cannot also be measured"
        )
    frame = pd.DataFrame(
        {
            unit_column: np.repeat(np.asarray(units, dtype=object), n_periods),
            time_column: np.tile(np.asarray(period_values), n_units),
        }
    )
    for name, values in columns.items():
        frame[name] = _as_wide(name, values, n_units, n_periods).reshape(-1)
    return frame


def panel_from_arrays(
    *,
    doses: Mapping[str, npt.ArrayLike],
    outcome: npt.ArrayLike,
    treatments: Sequence[Treatment],
    outcome_entity: Outcome,
    covariates: Mapping[str, tuple[Covariate, npt.ArrayLike]] | None = None,
    units: Sequence[str],
    periods: Sequence[int] | None = None,
    unit_column: str = "unit",
    time_column: str = "t",
) -> Panel:
    """A balanced ``Panel`` from wide arrays plus the entities that give the columns roles.

    Dose columns are named by ``treatment.name`` (the convention
    ``SurfaceSpec`` reads), the outcome column by ``outcome_entity.name``,
    covariate columns by the key in ``covariates``. Every treatment must
    have a dose array and vice versa.

    ``Panel`` sorts its rows by (unit, time), so the array rows and columns
    come back in sorted order. ``units`` must therefore already be in string
    order and ``periods`` in ascending order (``unit_labels`` gives such
    labels); anything else would silently permute the rows, and is rejected.
    """
    by_name = {t.name: t for t in treatments}
    if set(by_name) != set(doses):
        raise ValueError(
            f"treatments {sorted(by_name)} and dose arrays {sorted(doses)} must name the same set"
        )
    unit_strings = [str(u) for u in units]
    if unit_strings != sorted(unit_strings):
        raise ValueError(
            f"units must be in sorted (string) order because Panel sorts rows by unit label; "
            f"got {list(units)}. Use unit_labels() or sort the arrays' rows to match."
        )
    if periods is not None and list(periods) != sorted(periods):
        raise ValueError(
            f"periods must be in ascending order because Panel sorts rows by time; got "
            f"{list(periods)}"
        )
    cov = dict(covariates or {})
    columns: dict[str, npt.ArrayLike] = {outcome_entity.name: outcome}
    columns.update({name: doses[name] for name in by_name})
    columns.update({name: values for name, (_, values) in cov.items()})
    frame = long_frame(
        columns, units=units, periods=periods, unit_column=unit_column, time_column=time_column
    )
    roles = RoleMap(
        unit=unit_column,
        time=time_column,
        outcome=(outcome_entity.name, outcome_entity),
        treatments=dict(by_name),
        covariates={name: entity for name, (entity, _) in cov.items()},
    )
    return Panel(frame, roles)
