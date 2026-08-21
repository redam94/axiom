"""Scaling, with the parameters kept so it inverts.

A scaled column is dimensionless: it is the column divided by a reference in
the same unit (its max, its mean) and optionally centred. ``ScalingParameters``
is a ``Spec`` so the reference used travels with the analysis, and
``unscale`` restores the original dimension exactly.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import model_validator

from axiom.core.spec import Spec
from axiom.data.frame import Panel

__all__ = ["ColumnScaling", "ScalingMethod", "ScalingParameters", "fit_scaling"]

ScalingMethod = Literal["none", "max", "mean", "standardize"]


class ColumnScaling(Spec):
    """``scaled = (x - loc) / scale``. ``loc`` is 0 unless standardizing."""

    method: ScalingMethod
    loc: float = 0.0
    scale: float = 1.0

    @model_validator(mode="after")
    def _positive_scale(self) -> ColumnScaling:
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(f"scale must be finite and positive, got {self.scale}")
        if not np.isfinite(self.loc):
            raise ValueError("loc must be finite")
        return self

    def apply(self, x: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return (np.asarray(x, dtype=float) - self.loc) / self.scale

    def invert(self, z: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(z, dtype=float) * self.scale + self.loc


class ScalingParameters(Spec):
    """Per-column scaling for a panel's measured columns."""

    columns: dict[str, ColumnScaling]

    def scale(self, panel: Panel) -> Panel:
        df = panel.frame
        for col, cs in self.columns.items():
            if col not in df.columns:
                raise KeyError(f"column {col!r} not in panel")
            df[col] = cs.apply(df[col].to_numpy())
        return Panel(df, panel.roles)

    def unscale(self, panel: Panel) -> Panel:
        df = panel.frame
        for col, cs in self.columns.items():
            df[col] = cs.invert(df[col].to_numpy())
        return Panel(df, panel.roles)

    def unscale_column(self, column: str, z: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return self.columns[column].invert(z)


def _fit_one(x: pd.Series, method: ScalingMethod) -> ColumnScaling:
    v = x.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError(f"column {x.name!r} has no finite values to scale by")
    if method == "none":
        return ColumnScaling(method="none")
    if method == "max":
        m = float(np.max(np.abs(v)))
        if m == 0:
            raise ValueError(f"column {x.name!r} is all zeros; cannot scale by max")
        return ColumnScaling(method="max", scale=m)
    if method == "mean":
        m = float(np.mean(v))
        if m == 0:
            raise ValueError(f"column {x.name!r} has zero mean; cannot scale by mean")
        return ColumnScaling(method="mean", scale=abs(m))
    if method == "standardize":
        sd = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
        if sd == 0:
            raise ValueError(f"column {x.name!r} is constant; cannot standardize")
        return ColumnScaling(method="standardize", loc=float(np.mean(v)), scale=sd)
    raise ValueError(f"unknown scaling method {method!r}")  # pragma: no cover


def fit_scaling(
    panel: Panel,
    *,
    treatments: ScalingMethod = "max",
    outcome: ScalingMethod = "max",
    covariates: ScalingMethod = "standardize",
) -> ScalingParameters:
    """Fit scaling per role. Indices (unit, time) are never scaled."""
    df = panel.frame
    roles = panel.roles
    cols: dict[str, ColumnScaling] = {}
    cols[roles.outcome[0]] = _fit_one(df[roles.outcome[0]], outcome)
    for c in roles.treatments:
        cols[c] = _fit_one(df[c], treatments)
    for c in roles.covariates:
        cols[c] = _fit_one(df[c], covariates)
    return ScalingParameters(columns=cols)
