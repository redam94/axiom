"""The design matrix in the linear parameters at a fixed point of the nonlinear ones.

A surface mean is ``mu = offset(theta_nl) + Σ_j X_j(theta_nl) · theta_j`` over its
*linear* parameters — intercepts, amplitudes ``beta``, interaction ``gamma``,
nuisance coefficients — once the *nonlinear* ones (scales ``k``, shapes ``s``,
carryover parameters) are held at ``theta_at``. Nothing here differentiates:
because the mean is linear in those parameters by construction, the columns
are obtained **exactly** by evaluating the same tree ``forward()`` evaluates
(rule 3) with unit vectors on the linear parameters — ``X[:, j] = mu(e_j) −
mu(0)`` and ``offset = mu(0)``. The invariant

    ‖X @ theta_lin + offset − forward(dose, theta)‖∞ < 1e-12

is the Phase 3 gate (``tests/contracts/test_linearize_invariant.py``) and
``check_linearization`` measures it.

Rows of ``X`` are the entries of the forward output in C order: one row per
run for a 1-D dose grid, ``unit``-major then ``time`` for a ``(n_units,
n_periods)`` panel layout. A vector parameter (a per-unit intercept of shape
``(n_units,)``) contributes one column per element, named ``name[i]``;
``linear_coefficients`` flattens a parameter dict into that column order.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from axiom.core import DesignMatrix, ModelSpec, Param, SupportsForward, params, value

__all__ = ["check_linearization", "column_names", "design_matrix", "linear_coefficients"]

Array = npt.NDArray[np.float64]
_INDEXED = re.compile(r"^(?P<name>.+?)\[(?P<index>[0-9,]+)\]$")


def _indices(p: Param) -> list[tuple[int, ...]]:
    return [()] if not p.shape else [tuple(int(i) for i in idx) for idx in np.ndindex(*p.shape)]


def _column(p: Param, idx: tuple[int, ...]) -> str:
    return p.name if not idx else f"{p.name}[{','.join(str(i) for i in idx)}]"


def column_names(model: ModelSpec, linear: Sequence[str]) -> tuple[str, ...]:
    """One column per scalar entry of every linear parameter, in ``linear`` order."""
    out: list[str] = []
    for name in linear:
        p = model.parameter(name)
        out.extend(_column(p, idx) for idx in _indices(p))
    return tuple(out)


def linear_coefficients(columns: Sequence[str], theta: Mapping[str, npt.ArrayLike]) -> Array:
    """The vector ``theta[columns]``: scalars by name, vector entries by ``name[i]``.

    A column named by a bare parameter name must map to exactly one value
    (a scalar or a size-one array); an array with more entries is a
    ``ValueError`` naming the parameter rather than a silent truncation to
    its first entry. A missing value is a ``KeyError`` naming the column.
    """
    out = np.empty(len(columns), dtype=np.float64)
    for j, col in enumerate(columns):
        if col in theta:
            given = np.asarray(theta[col], dtype=float)
            if given.size != 1:
                raise ValueError(
                    f"linear parameter {col!r} is one column and needs a single value; got an "
                    f"array of shape {given.shape}. A vector parameter is addressed by "
                    f"{col + '[i]'!r} columns, not by {col!r}."
                )
            out[j] = float(given.reshape(-1)[0])
            continue
        m = _INDEXED.match(col)
        if m is None or m.group("name") not in theta:
            raise KeyError(f"no value for linear parameter column {col!r}")
        idx = tuple(int(i) for i in m.group("index").split(","))
        out[j] = float(np.asarray(theta[m.group("name")], dtype=float)[idx])
    return out


def design_matrix(
    model: ModelSpec,
    linear: Sequence[str],
    dose: Mapping[str, npt.ArrayLike],
    theta_at: Mapping[str, npt.ArrayLike],
) -> DesignMatrix:
    """``mu = offset + X @ theta[columns]`` for ``model.mean`` at ``theta_at``.

    ``linear`` names the parameters the mean is linear in; the values of
    those parameters in ``theta_at`` are ignored (only the nonlinear point
    matters, and it must be supplied — a missing one raises ``KeyError``
    naming it). ``DesignMatrix.at`` records the nonlinear parameters that
    appear in the mean.
    """
    lin = [model.parameter(name) for name in linear]
    names = {p.name for p in lin}
    base = {k: v for k, v in theta_at.items() if k not in names}
    zeros = {p.name: np.zeros(p.shape, dtype=np.float64) for p in lin}
    offset = np.asarray(value(model.mean, data=dose, params={**base, **zeros}), dtype=np.float64)
    columns: list[str] = []
    raw: list[Array] = []
    for p in lin:
        for idx in _indices(p):
            unit = np.zeros(p.shape, dtype=np.float64)
            unit[idx] = 1.0
            mu = value(model.mean, data=dose, params={**base, **zeros, p.name: unit})
            columns.append(_column(p, idx))
            raw.append(np.asarray(mu, dtype=np.float64) - offset)
    shaped = np.broadcast_arrays(offset, *raw)
    off = np.asarray(shaped[0], dtype=np.float64).reshape(-1)
    X = (
        np.stack([np.asarray(c, dtype=np.float64).reshape(-1) for c in shaped[1:]], axis=1)
        if raw
        else np.zeros((off.size, 0), dtype=np.float64)
    )
    nonlinear = [p.name for p in params(model.mean) if p.name not in names]
    at = {n: np.asarray(theta_at[n], dtype=np.float64) for n in nonlinear}
    return DesignMatrix(X=X, columns=tuple(columns), offset=off, at=at)


def check_linearization(
    surface: SupportsForward,
    dose: Mapping[str, npt.ArrayLike],
    theta: Mapping[str, npt.ArrayLike],
) -> float:
    """``‖X @ theta_lin + offset − forward(dose, theta)‖∞`` — the gate's number, for one point."""
    dm = surface.linearize(dose, theta)
    predicted = dm.offset + dm.X @ linear_coefficients(dm.columns, theta)
    actual = np.asarray(surface.forward(dose, theta), dtype=np.float64).reshape(-1)
    if predicted.shape != actual.shape:
        raise ValueError(
            f"linearization has {predicted.shape[0]} rows but forward() returned "
            f"{actual.shape[0]} values; rows must be the C-order entries of the forward output"
        )
    return float(np.max(np.abs(predicted - actual))) if actual.size else 0.0
