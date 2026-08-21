"""Carryover as normalized finite-impulse-response weights built from expression nodes.

A carryover kernel turns a dose series into its lagged, decayed version:
``carried[t] = Σ_l w_l · dose[t − l]`` for ``l = 0 … max_lag − 1``. The
weights are a dimensionless vector that sums to one, so ``apply(dose)``
keeps the dose's dimension — a per-period-versus-cumulative mismatch cannot
hide inside a carryover (it shows up at the ``Convolve`` node's dimension
check). The weights are *built*, never computed here: ``Const(lags)``,
``Pow``, ``Apply``, ``Div``, ``Reduce``, ``Mul``, ``Add`` — no ``Opaque`` —
so the numpy and jax interpreters evaluate the same tree (rule 3).

Every family is a flat ``Spec`` (0002.13) satisfying ``CarryoverKernel``.
All carryover parameters are dimensionless **shapes**: they are invariant
to the unit the dose is measured in (the *time* unit is fixed by the panel's
period, which is why ``max_lag`` is an integer count of periods).

``half_life`` is a diagnostic on parameter *values*: the number of lags
after the peak at which the un-normalized weight falls to half its peak
(``log 2 / log(1/λ)`` for a geometric decay). Comparing it with the analysis
window is the ``stationary_dynamics`` assumption in ``estimands``. It
validates its inputs (``λ ∈ (0, 1]`` for the decay families, ``λ > 0`` and
``κ > 0`` for Weibull) and returns ``+inf`` — never a warning, ``nan`` or a
negative number — for a decay of exactly one.

The weights are normalized with ``Reduce(op="sum", keepdims=True)`` so that
parameter values carrying leading draw axes (``lam`` of shape ``(draws, 1)``)
give per-draw weights of shape ``(draws, max_lag)`` and ``apply`` gives
``(draws, T)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import Field

from axiom.core import (
    Add,
    Apply,
    Const,
    Convolve,
    Div,
    Expr,
    Param,
    Pow,
    Prior,
    Reduce,
    Spec,
    dimensionless,
)

__all__ = [
    "CARRYOVERS",
    "AnyCarryover",
    "CarryoverKernel",
    "CarryoverRole",
    "DelayedCarryover",
    "GeometricCarryover",
    "NoCarryover",
    "WeibullCarryover",
    "carryover_from_name",
]

CarryoverRole = Literal["shape"]
"""Every carryover parameter is a dimensionless shape."""

Values = Mapping[str, npt.ArrayLike]
"""Parameter values by full name (``"lam_tv"``); scalars or posterior draws."""


@runtime_checkable
class CarryoverKernel(Protocol):
    """A lag-weight family that builds its own expression tree.

    ``parameters`` names the dimensionless ``Param`` nodes introduced for one
    treatment (``"lam_tv"``, ``"theta_tv"``); ``weights`` is the normalized
    weight vector (sums to one through a ``Reduce``); ``apply`` convolves the
    dose with it; ``half_life`` is a numpy diagnostic on parameter values.
    """

    @property
    def name(self) -> str: ...
    @property
    def max_lag(self) -> int: ...
    @property
    def roles(self) -> Mapping[str, CarryoverRole]: ...
    def parameters(self, treatment: str) -> tuple[Param, ...]: ...
    def weights(self, treatment: str) -> Expr: ...
    def apply(self, dose: Expr, treatment: str) -> Expr: ...
    def half_life(self, params: Values, treatment: str) -> npt.NDArray[np.float64]: ...


# -- shared builders ---------------------------------------------------------------------


def _lags(max_lag: int) -> Const:
    """``(0, 1, …, max_lag − 1)`` as a dimensionless vector constant."""
    return Const(value=tuple(float(i) for i in range(max_lag)), dimension=dimensionless())


def _normalized(raw: Expr) -> Expr:
    """``raw / Σ raw`` along the lag axis — the weights sum to one by construction.

    ``keepdims=True`` keeps the reduced axis so a ``(draws, L)`` raw vector
    is divided by its own ``(draws, 1)`` row sums rather than by a ``(draws,)``
    vector broadcast along the wrong axis.
    """
    return Div(numerator=raw, denominator=Reduce(op="sum", arg=raw, keepdims=True))


def _shape(role: str, treatment: str, prior: Prior) -> Param:
    return Param(name=f"{role}_{treatment}", dimension=dimensionless(), prior=prior)


def _values(params: Values, *names: str) -> tuple[npt.NDArray[np.float64], ...]:
    missing = [n for n in names if n not in params]
    if missing:
        raise KeyError(f"carryover parameter values missing: {missing}")
    return tuple(np.asarray(params[n], dtype=float) for n in names)


def _check_decay(name: str, lam: npt.NDArray[np.float64]) -> None:
    """A decay rate lives in ``(0, 1]``; anything else is a caller error, not a number."""
    if not np.all(np.isfinite(lam)):
        raise ValueError(f"{name} must be finite")
    if np.any(lam <= 0.0) or np.any(lam > 1.0):
        raise ValueError(f"{name} must lie in (0, 1]; got values outside that range")


def _check_positive(name: str, x: npt.NDArray[np.float64]) -> None:
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be finite")
    if np.any(x <= 0.0):
        raise ValueError(f"{name} must be strictly positive")


def _log_half_over_log(lam: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """``log(1/2) / log λ`` for ``λ < 1`` and ``+inf`` at ``λ == 1``, without warnings."""
    decaying = lam < 1.0
    safe = np.where(decaying, lam, 0.5)
    return np.asarray(np.where(decaying, np.log(0.5) / np.log(safe), np.inf), dtype=np.float64)


# -- families ----------------------------------------------------------------------------


class GeometricCarryover(Spec):
    """``w_l ∝ λ^l``: exponential decay from lag zero. ``λ ~ beta(2, 2)`` on (0, 1)."""

    name: Literal["geometric"] = "geometric"
    max_lag: int = Field(ge=1)

    @property
    def roles(self) -> dict[str, CarryoverRole]:
        return {"lam": "shape"}

    def parameters(self, treatment: str) -> tuple[Param, ...]:
        return (_shape("lam", treatment, Prior(family="beta", hyper={"alpha": 2.0, "beta": 2.0})),)

    def weights(self, treatment: str) -> Expr:
        (lam,) = self.parameters(treatment)
        return _normalized(Pow(base=lam, exponent=_lags(self.max_lag)))

    def apply(self, dose: Expr, treatment: str) -> Expr:
        return Convolve(signal=dose, kernel=self.weights(treatment))

    def half_life(self, params: Values, treatment: str) -> npt.NDArray[np.float64]:
        """``log(1/2) / log λ``: lags until the weight halves; ``+inf`` at ``λ = 1``.

        Raises ``ValueError`` unless every ``λ`` lies in ``(0, 1]``.
        """
        (lam,) = _values(params, f"lam_{treatment}")
        _check_decay(f"lam_{treatment}", lam)
        return _log_half_over_log(lam)


class DelayedCarryover(Spec):
    """``w_l ∝ λ^((l − θ)^2)``: a peak at lag ``θ`` with symmetric geometric decay around it.

    ``λ ~ beta(2, 2)`` sets the decay, ``θ ~ uniform(0, max_lag − 1)`` the
    delay; both dimensionless. Needs ``max_lag ≥ 2`` so the delay has room.
    """

    name: Literal["delayed"] = "delayed"
    max_lag: int = Field(ge=2)

    @property
    def roles(self) -> dict[str, CarryoverRole]:
        return {"lam": "shape", "theta": "shape"}

    def parameters(self, treatment: str) -> tuple[Param, ...]:
        return (
            _shape("lam", treatment, Prior(family="beta", hyper={"alpha": 2.0, "beta": 2.0})),
            _shape(
                "theta",
                treatment,
                Prior(family="uniform", hyper={"low": 0.0, "high": float(self.max_lag - 1)}),
            ),
        )

    def weights(self, treatment: str) -> Expr:
        lam, theta = self.parameters(treatment)
        offset = Add(terms=(_lags(self.max_lag), Apply(fn="neg", arg=theta)))
        return _normalized(Pow(base=lam, exponent=Pow(base=offset, exponent=Fraction(2))))

    def apply(self, dose: Expr, treatment: str) -> Expr:
        return Convolve(signal=dose, kernel=self.weights(treatment))

    def half_life(self, params: Values, treatment: str) -> npt.NDArray[np.float64]:
        """``sqrt(log(1/2) / log λ)``: lags past the peak ``θ`` until the weight halves.

        ``+inf`` at ``λ = 1``; raises ``ValueError`` unless every ``λ`` lies in ``(0, 1]``.
        """
        (lam,) = _values(params, f"lam_{treatment}")
        _check_decay(f"lam_{treatment}", lam)
        return np.asarray(np.sqrt(_log_half_over_log(lam)), dtype=np.float64)


class WeibullCarryover(Spec):
    """``w_l ∝ exp(−(l / λ)^κ)``: the Weibull survival function over lags.

    ``λ`` is the lag scale (in periods, dimensionless), ``κ`` the shape:
    ``κ = 1`` is geometric decay with rate ``exp(−1/λ)``; ``κ > 1`` holds the
    weight up before it drops. Priors: ``λ ~ gamma(2, 4 / max_lag)`` (mean
    ``max_lag / 2``), ``κ ~ gamma(2, 1)`` (mean 2). The tree is written as
    ``l^κ / λ^κ`` so that the lag-zero term has a finite gradient in ``λ``
    and ``κ`` for every ``κ > 0`` (``(0 / λ)^κ`` would not, for ``κ < 1``).

    ``λ`` must be strictly positive: at ``λ = 0`` the lag-zero term is
    ``0^κ / 0^κ``, which is undefined (``nan``), and as ``λ → 0`` every
    weight beyond lag zero underflows to zero while the gradient in ``λ``
    grows without bound. The gamma prior has no mass at zero; a fixed or
    user-supplied ``λ`` must respect the same restriction, and ``half_life``
    rejects ``λ ≤ 0`` and ``κ ≤ 0`` with ``ValueError``.
    """

    name: Literal["weibull"] = "weibull"
    max_lag: int = Field(ge=1)

    @property
    def roles(self) -> dict[str, CarryoverRole]:
        return {"lam": "shape", "kappa": "shape"}

    def parameters(self, treatment: str) -> tuple[Param, ...]:
        return (
            _shape(
                "lam",
                treatment,
                Prior(family="gamma", hyper={"alpha": 2.0, "beta": 4.0 / self.max_lag}),
            ),
            _shape("kappa", treatment, Prior(family="gamma", hyper={"alpha": 2.0, "beta": 1.0})),
        )

    def weights(self, treatment: str) -> Expr:
        lam, kappa = self.parameters(treatment)
        scaled = Div(
            numerator=Pow(base=_lags(self.max_lag), exponent=kappa),
            denominator=Pow(base=lam, exponent=kappa),
        )
        return _normalized(Apply(fn="exp", arg=Apply(fn="neg", arg=scaled)))

    def apply(self, dose: Expr, treatment: str) -> Expr:
        return Convolve(signal=dose, kernel=self.weights(treatment))

    def half_life(self, params: Values, treatment: str) -> npt.NDArray[np.float64]:
        """``λ · (log 2)^(1/κ)``: the lag at which the survival weight halves.

        Raises ``ValueError`` unless every ``λ > 0`` and every ``κ > 0``.
        """
        lam, kappa = _values(params, f"lam_{treatment}", f"kappa_{treatment}")
        _check_positive(f"lam_{treatment}", lam)
        _check_positive(f"kappa_{treatment}", kappa)
        return np.asarray(lam * np.log(2.0) ** (1.0 / kappa), dtype=np.float64)


class NoCarryover(Spec):
    """The identity: all weight at lag zero. ``apply`` returns the dose expression itself."""

    name: Literal["none"] = "none"

    @property
    def max_lag(self) -> int:
        return 1

    @property
    def roles(self) -> dict[str, CarryoverRole]:
        return {}

    def parameters(self, treatment: str) -> tuple[Param, ...]:
        return ()

    def weights(self, treatment: str) -> Expr:
        return Const(value=(1.0,), dimension=dimensionless())

    def apply(self, dose: Expr, treatment: str) -> Expr:
        return dose

    def half_life(self, params: Values, treatment: str) -> npt.NDArray[np.float64]:
        return np.asarray(0.0, dtype=np.float64)


AnyCarryover = Annotated[
    GeometricCarryover | DelayedCarryover | WeibullCarryover | NoCarryover,
    Field(discriminator="name"),
]
"""The shipped families as a discriminated union on ``name``, for embedding in other specs."""

CARRYOVERS: dict[str, type[Spec]] = {
    "geometric": GeometricCarryover,
    "delayed": DelayedCarryover,
    "weibull": WeibullCarryover,
    "none": NoCarryover,
}
"""Family name to carryover class. Every entry satisfies ``CarryoverKernel``."""


def carryover_from_name(name: str, **fields: Any) -> CarryoverKernel:
    """Build a carryover by family name, e.g. ``carryover_from_name("geometric", max_lag=8)``."""
    if name not in CARRYOVERS:
        raise ValueError(f"unknown carryover family {name!r}; known: {sorted(CARRYOVERS)}")
    kernel = CARRYOVERS[name](**fields)
    if not isinstance(kernel, CarryoverKernel):  # pragma: no cover - registry invariant
        raise TypeError(f"{type(kernel).__name__} does not satisfy CarryoverKernel")
    return kernel
