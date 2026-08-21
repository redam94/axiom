"""Baseline (nuisance) terms: seasonality, trend, and event indicators, to profile out.

A nuisance term contributes ``Σ_i coef_i · basis_i`` to the mean, where each
basis column is a dimensionless function of time (or a user-supplied 0/1
indicator) and each coefficient carries the outcome dimension. Terms know
how to *augment* a frame with their basis columns and how to *build* the
expression that reads them back, so the design matrix, the likelihood, and
the simulator share one definition of the baseline.

Every term is a flat ``Spec`` (0002.13) satisfying ``NuisanceTerm``;
``NuisanceSet`` composes several with one ``augment`` / ``expr`` /
``parameters`` over all of them, prefixing each term's names with ``n{i}_``
so two seasonalities (weekly and annual) cannot collide.

Time handling: a numeric time column is used as-is; a datetime column is
converted to fractional days since 1970-01-01. ``period`` (seasonality) and
``origin`` / ``scale`` (trend) are in those same units. A time column with
a missing value (``NaT``, ``nan``) or an infinity is rejected with
``ValueError`` rather than silently producing a ``nan`` basis row.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Annotated, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, ValidationInfo, field_validator, model_validator

from axiom.core import Add, Const, D, Data, Dimension, Expr, Mul, Param, Prior, Spec, dimensionless

__all__ = [
    "AnyNuisanceTerm",
    "EventIndicators",
    "FourierSeasonality",
    "LinearTrend",
    "NuisanceSet",
    "NuisanceTerm",
]


@runtime_checkable
class NuisanceTerm(Protocol):
    """A baseline term: basis columns in the frame, coefficients in the tree.

    ``prefix`` is prepended verbatim to generated column and parameter names
    (pass ``"annual_"`` to get ``annual_sin_1`` and ``annual_coef_sin_1``).
    ``EventIndicators`` reads user columns, so its column names ignore the
    prefix; its coefficient names do not.
    """

    @property
    def kind(self) -> str: ...
    def column_names(self, prefix: str = "") -> tuple[str, ...]: ...
    def parameter_names(self, prefix: str = "") -> tuple[str, ...]: ...
    def augment(self, frame: pd.DataFrame, time_column: str, prefix: str = "") -> pd.DataFrame: ...
    def parameters(
        self, prefix: str = "", outcome_dimension: Dimension | None = None
    ) -> tuple[Param, ...]: ...
    def expr(self, prefix: str = "", outcome_dimension: Dimension | None = None) -> Expr: ...


# -- shared builders ----------------------------------------------------------------------


def _numeric_time(frame: pd.DataFrame, time_column: str) -> npt.NDArray[np.float64]:
    """The time column as float64: numeric as-is, datetimes as days since the epoch."""
    if time_column not in frame.columns:
        raise KeyError(f"time column {time_column!r} is not in the frame")
    series = frame[time_column]
    if pd.api.types.is_datetime64_any_dtype(series):
        epoch = pd.Timestamp("1970-01-01", tz=series.dt.tz)
        days = (series - epoch) / pd.Timedelta(days=1)
        t = np.asarray(days.to_numpy(), dtype=np.float64)
    elif pd.api.types.is_numeric_dtype(series):
        t = np.asarray(series.to_numpy(), dtype=np.float64)
    else:
        raise TypeError(
            f"time column {time_column!r} must be numeric or datetime, got {series.dtype}"
        )
    bad = ~np.isfinite(t)
    if np.any(bad):
        rows = np.flatnonzero(bad)
        raise ValueError(
            f"time column {time_column!r} has {int(bad.sum())} non-finite value(s) "
            f"(NaT / nan / inf) at rows {rows[:5].tolist()}{'...' if rows.size > 5 else ''}"
        )
    return t


def _add_columns(
    frame: pd.DataFrame, columns: Iterable[tuple[str, npt.NDArray[np.float64]]]
) -> pd.DataFrame:
    """A copy of ``frame`` with the columns appended; refuses to overwrite an existing column."""
    out = frame.copy()
    for name, values in columns:
        if name in out.columns:
            raise ValueError(
                f"column {name!r} already exists; augment once, or choose another prefix"
            )
        out[name] = values
    return out


def _coefficients(
    stems: Iterable[str], prefix: str, outcome_dimension: Dimension | None, scale: float
) -> tuple[Param, ...]:
    """One ``normal(0, scale)`` coefficient per basis column, carrying the outcome dimension."""
    dim = outcome_dimension if outcome_dimension is not None else D.outcome
    return tuple(
        Param(
            name=f"{prefix}coef_{stem}",
            dimension=dim,
            prior=Prior(family="normal", hyper={"mu": 0.0, "sigma": float(scale)}),
        )
        for stem in stems
    )


def _linear_combination(columns: Iterable[str], coefficients: Iterable[Param]) -> Expr:
    """``Σ_i coef_i · Data(col_i)`` with dimensionless basis columns."""
    terms = tuple(
        Mul(factors=(coef, Data(name=col, dimension=dimensionless())))
        for col, coef in zip(columns, coefficients, strict=True)
    )
    return Add(terms=terms)


# -- terms ---------------------------------------------------------------------------------


class FourierSeasonality(Spec):
    """``sin(2πk t / period)``, ``cos(2πk t / period)`` for ``k = 1 … order``.

    ``period`` is in the time column's units (periods for an integer index,
    days for datetimes). ``order`` must be at least one — an order-zero
    seasonality is no seasonality and is rejected rather than silently
    contributing nothing.
    """

    kind: Literal["seasonality"] = "seasonality"
    period: float = Field(gt=0)
    order: int = Field(ge=1)
    coefficient_scale: float = Field(default=1.0, gt=0)

    @property
    def stems(self) -> tuple[str, ...]:
        return tuple(f"{fn}_{k}" for k in range(1, self.order + 1) for fn in ("sin", "cos"))

    def column_names(self, prefix: str = "") -> tuple[str, ...]:
        return tuple(f"{prefix}{stem}" for stem in self.stems)

    def parameter_names(self, prefix: str = "") -> tuple[str, ...]:
        return tuple(f"{prefix}coef_{stem}" for stem in self.stems)

    def augment(self, frame: pd.DataFrame, time_column: str, prefix: str = "") -> pd.DataFrame:
        t = _numeric_time(frame, time_column)
        columns: list[tuple[str, npt.NDArray[np.float64]]] = []
        for k in range(1, self.order + 1):
            angle = 2.0 * math.pi * k * t / self.period
            columns.append((f"{prefix}sin_{k}", np.sin(angle)))
            columns.append((f"{prefix}cos_{k}", np.cos(angle)))
        return _add_columns(frame, columns)

    def parameters(
        self, prefix: str = "", outcome_dimension: Dimension | None = None
    ) -> tuple[Param, ...]:
        return _coefficients(self.stems, prefix, outcome_dimension, self.coefficient_scale)

    def expr(self, prefix: str = "", outcome_dimension: Dimension | None = None) -> Expr:
        return _linear_combination(
            self.column_names(prefix), self.parameters(prefix, outcome_dimension)
        )


class LinearTrend(Spec):
    """``trend = (t − origin) / scale``: one dimensionless basis column.

    ``origin`` and ``scale`` are in the time column's units. Both default to
    ``None``, which ``augment`` resolves against the frame it is given:
    ``origin`` becomes the earliest observed time and ``scale`` the observed
    window length ``max(t) − min(t)``, so the column runs from 0 to 1 over
    the window and the coefficient is the outcome change across it — which
    is what the ``normal(0, coefficient_scale)`` prior is meant to be read
    against. That makes the defaults meaningful for datetime columns too
    (days since the epoch would otherwise give a trend column in the tens
    of thousands). A window of a single distinct time cannot resolve
    ``scale`` and is rejected. Explicit values are used as given.

    The values actually used are recorded on the returned frame as
    ``frame.attrs["trend"][column_name] == {"origin": ..., "scale": ...}`` so
    a later frame (a forecast window) can be augmented with the same
    convention by passing them explicitly.
    """

    kind: Literal["trend"] = "trend"
    origin: float | None = None
    scale: float | None = None
    coefficient_scale: float = Field(default=1.0, gt=0)

    @field_validator("origin", "scale")
    @classmethod
    def _finite(cls, v: float | None, info: ValidationInfo) -> float | None:
        if v is None:
            return v
        if not math.isfinite(v):
            raise ValueError(f"{info.field_name} must be finite or None")
        if info.field_name == "scale" and v <= 0:
            raise ValueError("scale must be positive or None")
        return v

    @property
    def stems(self) -> tuple[str, ...]:
        return ("trend",)

    def column_names(self, prefix: str = "") -> tuple[str, ...]:
        return (f"{prefix}trend",)

    def parameter_names(self, prefix: str = "") -> tuple[str, ...]:
        return (f"{prefix}coef_trend",)

    def resolve(self, t: npt.ArrayLike) -> tuple[float, float]:
        """The ``(origin, scale)`` used for the time values ``t``: explicit fields, else the
        first observed time and the window length.

        Raises ``ValueError`` when ``scale`` must be inferred from a window
        of zero length (one distinct time, or an empty frame).
        """
        times = np.asarray(t, dtype=np.float64)
        if times.size == 0:
            if self.origin is None or self.scale is None:
                raise ValueError("cannot resolve a trend origin/scale from an empty time column")
            return self.origin, self.scale
        origin = float(times.min()) if self.origin is None else self.origin
        if self.scale is None:
            width = float(times.max() - times.min())
            if not width > 0.0:
                raise ValueError(
                    "cannot resolve the trend scale: the time column has a single distinct "
                    "value; pass scale= explicitly"
                )
            scale = width
        else:
            scale = self.scale
        return origin, scale

    def augment(self, frame: pd.DataFrame, time_column: str, prefix: str = "") -> pd.DataFrame:
        t = _numeric_time(frame, time_column)
        origin, scale = self.resolve(t)
        column = f"{prefix}trend"
        out = _add_columns(frame, [(column, (t - origin) / scale)])
        recorded = dict(out.attrs.get("trend", {}))
        recorded[column] = {"origin": origin, "scale": scale}
        out.attrs = {**out.attrs, "trend": recorded}
        return out

    def parameters(
        self, prefix: str = "", outcome_dimension: Dimension | None = None
    ) -> tuple[Param, ...]:
        return _coefficients(self.stems, prefix, outcome_dimension, self.coefficient_scale)

    def expr(self, prefix: str = "", outcome_dimension: Dimension | None = None) -> Expr:
        return _linear_combination(
            self.column_names(prefix), self.parameters(prefix, outcome_dimension)
        )


class EventIndicators(Spec):
    """User-supplied 0/1 columns, one coefficient each.

    The columns are the user's, so ``column_names`` returns ``events`` as
    given (the prefix applies to coefficient names only) and ``augment``
    adds nothing — it checks that every event column is present and binary
    and returns the frame unchanged.
    """

    kind: Literal["events"] = "events"
    events: tuple[str, ...] = Field(min_length=1)
    coefficient_scale: float = Field(default=1.0, gt=0)

    @field_validator("events")
    @classmethod
    def _distinct(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if any(not e.strip() for e in v):
            raise ValueError("event column names must be non-empty")
        if len(set(v)) != len(v):
            raise ValueError(f"event column names must be distinct: {v}")
        return v

    @property
    def stems(self) -> tuple[str, ...]:
        return self.events

    def column_names(self, prefix: str = "") -> tuple[str, ...]:
        return self.events

    def parameter_names(self, prefix: str = "") -> tuple[str, ...]:
        return tuple(f"{prefix}coef_{event}" for event in self.events)

    def augment(self, frame: pd.DataFrame, time_column: str, prefix: str = "") -> pd.DataFrame:
        missing = [e for e in self.events if e not in frame.columns]
        if missing:
            raise KeyError(f"event columns missing from the frame: {missing}")
        for event in self.events:
            values = np.asarray(frame[event].to_numpy(), dtype=np.float64)
            if not np.all(np.isin(values, (0.0, 1.0))):
                raise ValueError(f"event column {event!r} must be 0/1 with no missing values")
        return frame

    def parameters(
        self, prefix: str = "", outcome_dimension: Dimension | None = None
    ) -> tuple[Param, ...]:
        return _coefficients(self.events, prefix, outcome_dimension, self.coefficient_scale)

    def expr(self, prefix: str = "", outcome_dimension: Dimension | None = None) -> Expr:
        return _linear_combination(self.events, self.parameters(prefix, outcome_dimension))


AnyNuisanceTerm = Annotated[
    FourierSeasonality | LinearTrend | EventIndicators, Field(discriminator="kind")
]
"""The shipped terms as a discriminated union on ``kind``, for embedding in other specs."""


class NuisanceSet(Spec):
    """Several nuisance terms with one ``augment`` / ``expr`` / ``parameters``.

    Term ``i`` receives the prefix ``f"{prefix}n{i}_"``, so generated names
    are unique across terms by construction. Event columns are the user's
    and are checked for collisions at construction. An empty set is the
    absence of a baseline: its ``expr`` is a zero constant in the outcome
    dimension. A set is itself a ``NuisanceTerm`` (``kind == "set"``).
    """

    kind: Literal["set"] = "set"
    terms: tuple[AnyNuisanceTerm, ...]

    @model_validator(mode="after")
    def _no_collisions(self) -> NuisanceSet:
        names = self.column_names()
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"nuisance terms share basis columns: {dupes}")
        return self

    @staticmethod
    def _term_prefix(prefix: str, i: int) -> str:
        return f"{prefix}n{i}_"

    def column_names(self, prefix: str = "") -> tuple[str, ...]:
        return tuple(
            name
            for i, term in enumerate(self.terms)
            for name in term.column_names(self._term_prefix(prefix, i))
        )

    def parameter_names(self, prefix: str = "") -> tuple[str, ...]:
        return tuple(
            name
            for i, term in enumerate(self.terms)
            for name in term.parameter_names(self._term_prefix(prefix, i))
        )

    def augment(self, frame: pd.DataFrame, time_column: str, prefix: str = "") -> pd.DataFrame:
        out = frame
        for i, term in enumerate(self.terms):
            out = term.augment(out, time_column, self._term_prefix(prefix, i))
        return out

    def parameters(
        self, prefix: str = "", outcome_dimension: Dimension | None = None
    ) -> tuple[Param, ...]:
        return tuple(
            p
            for i, term in enumerate(self.terms)
            for p in term.parameters(self._term_prefix(prefix, i), outcome_dimension)
        )

    def expr(self, prefix: str = "", outcome_dimension: Dimension | None = None) -> Expr:
        dim = outcome_dimension if outcome_dimension is not None else D.outcome
        if not self.terms:
            return Const(value=0.0, dimension=dim)
        return Add(
            terms=tuple(
                term.expr(self._term_prefix(prefix, i), dim) for i, term in enumerate(self.terms)
            )
        )
