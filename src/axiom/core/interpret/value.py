"""Numeric evaluation of an expression tree with numpy. This *is* ``forward()``.

``value(expr, data=..., params=..., opaque=...)`` evaluates with dimensions
erased. ``data`` maps column name to array, ``params`` maps parameter name to
array (a scalar, or draws that broadcast), ``opaque`` maps an ``Opaque``
node's name to a callable taking its evaluated inputs. Arrays broadcast by
numpy's rules; ``Convolve`` acts along the last axis.

``Equation`` evaluates to ``rhs``; ``System`` to a tuple of ``rhs`` values;
``ODESystem`` is not evaluable in 1.0 (C1) and raises ``NotImplementedError``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from fractions import Fraction
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import expit, logit

from axiom.core.expr import (
    Add,
    Apply,
    Const,
    Convolve,
    Data,
    Div,
    Equation,
    Gather,
    Link,
    Model,
    Mul,
    ODESystem,
    Opaque,
    Param,
    Pow,
    Reduce,
    System,
)

__all__ = ["Array", "OpaqueFn", "OpaqueRegistry", "causal_convolve", "value"]

Array = npt.NDArray[np.float64]
OpaqueFn = Callable[..., npt.ArrayLike]
OpaqueRegistry = Mapping[str, OpaqueFn]

_APPLY: dict[str, Callable[[Array], Array]] = {
    "exp": np.exp,
    "log": np.log,
    "log1p": np.log1p,
    "expm1": np.expm1,
    "tanh": np.tanh,
    "sigmoid": expit,
    "logit": logit,
    "softplus": lambda x: np.logaddexp(0.0, x),
    "neg": np.negative,
    "relu": lambda x: np.maximum(x, 0.0),
    "step": lambda x: np.where(x > 0.0, 1.0, 0.0),
}
_REDUCE: dict[str, Callable[[Array, bool], Array]] = {
    "sum": lambda x, k: np.sum(np.atleast_1d(x), axis=-1, keepdims=k),
    "mean": lambda x, k: np.mean(np.atleast_1d(x), axis=-1, keepdims=k),
    "max": lambda x, k: np.max(np.atleast_1d(x), axis=-1, keepdims=k),
}
_LINK: dict[str, Callable[[Array], Array]] = {
    "identity": lambda x: x,
    "log": np.log,
    "logit": logit,
}


def causal_convolve(signal: Array, weights: Array) -> Array:
    """``y[..., t] = Σ_l w[..., l] · x[..., t-l]`` with zero history before ``t=0``.

    ``weights`` is ``(..., L)``; its leading axes broadcast against the
    leading axes of ``signal`` (so a ``(draws, 1, L)`` kernel against a
    ``(units, T)`` signal gives ``(draws, units, T)``).
    """
    w = np.atleast_1d(np.asarray(weights, dtype=float))
    x = np.asarray(signal, dtype=float)
    n = x.shape[-1]
    n_lags = min(int(w.shape[-1]), n)
    lead = np.broadcast_shapes(w.shape[:-1], x.shape[:-1])
    out = np.zeros(lead + (n,), dtype=float)
    for lag in range(n_lags):
        wl = w[..., lag : lag + 1]
        if lag == 0:
            out += wl * x
        else:
            out[..., lag:] += wl * x[..., :-lag]
    return out


class _Env:
    def __init__(
        self, data: Mapping[str, Any], params: Mapping[str, Any], opaque: OpaqueRegistry
    ) -> None:
        self.data, self.params, self.opaque = data, params, opaque

    def ev(self, node: Model) -> Array:
        match node:
            case Const():
                return np.asarray(node.value, dtype=float)
            case Data():
                if node.name not in self.data:
                    raise KeyError(f"data column {node.name!r} was not supplied")
                return np.asarray(self.data[node.name], dtype=float)
            case Param():
                if node.name not in self.params:
                    raise KeyError(f"parameter {node.name!r} was not supplied")
                return np.asarray(self.params[node.name], dtype=float)
            case Add():
                out = self.ev(node.terms[0])
                for t in node.terms[1:]:
                    out = out + self.ev(t)
                return out
            case Mul():
                out = self.ev(node.factors[0])
                for f in node.factors[1:]:
                    out = out * self.ev(f)
                return out
            case Div():
                return self.ev(node.numerator) / self.ev(node.denominator)
            case Pow():
                base = self.ev(node.base)
                if isinstance(node.exponent, Fraction):
                    return np.power(base, float(node.exponent))
                return np.power(base, self.ev(node.exponent))
            case Apply():
                return _APPLY[node.fn](self.ev(node.arg))
            case Link():
                return _LINK[node.fn](self.ev(node.arg))
            case Reduce():
                return _REDUCE[node.op](self.ev(node.arg), node.keepdims)
            case Gather():
                if node.index.name not in self.data:
                    raise KeyError(f"index column {node.index.name!r} was not supplied")
                idx = np.asarray(self.data[node.index.name], dtype=int)
                return np.take(self.ev(node.source), idx, axis=-1)
            case Convolve():
                return causal_convolve(self.ev(node.signal), self.ev(node.kernel))
            case Opaque():
                if node.name not in self.opaque:
                    raise KeyError(
                        f"opaque function {node.name!r} is not registered; pass it via opaque="
                    )
                return np.asarray(
                    self.opaque[node.name](*(self.ev(i) for i in node.inputs)), dtype=float
                )
            case Equation():
                return self.ev(node.rhs)
            case System():
                return np.stack(np.broadcast_arrays(*(self.ev(eq) for eq in node.equations)))
            case ODESystem():
                raise NotImplementedError(
                    "ODESystem is dimension-check-only in 1.0; it has no value interpreter"
                )
        raise TypeError(f"not an expression node: {type(node).__name__}")


def value(
    model: Model,
    *,
    data: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
    opaque: OpaqueRegistry | None = None,
) -> Array:
    """Evaluate ``model`` numerically. Missing inputs raise ``KeyError`` naming the name."""
    return _Env(data or {}, params or {}, opaque or {}).ev(model)
