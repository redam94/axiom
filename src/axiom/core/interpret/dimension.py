"""Abstract interpretation of an expression tree over ``Dimension``.

``dimension(model)`` returns the dimension of an expression, or raises
``DimensionError`` naming the offending node by its path. For ``Equation``,
``System``, and ``ODESystem`` it checks that every equation balances and
returns the dimension of the (first) left-hand side.

This is the type checker. It runs once, at spec construction; ``forward``
runs on raw arrays with dimensions erased.
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction

from axiom.core.dimensions import Dimension, DimensionError, dimensionless
from axiom.core.expr import (
    Add,
    Apply,
    Const,
    Convolve,
    Data,
    Div,
    Equation,
    Link,
    Model,
    Mul,
    ODESystem,
    Opaque,
    Param,
    Pow,
    System,
)

__all__ = ["check", "dimension"]


def _leaf(node: Const | Data | Param | Opaque, path: str) -> Dimension:
    return node.dimension


def _add(node: Add, path: str) -> Dimension:
    first = dimension(node.terms[0], f"{path}[0]")
    for i, term in enumerate(node.terms[1:], start=1):
        d = dimension(term, f"{path}[{i}]")
        if d != first:
            raise DimensionError(
                f"{path}: terms of a sum must share a dimension; term 0 is {first}, "
                f"term {i} is {d}"
            )
    return first


def _mul(node: Mul, path: str) -> Dimension:
    out = dimensionless()
    for i, f in enumerate(node.factors):
        out = out * dimension(f, f"{path}[{i}]")
    return out


def _div(node: Div, path: str) -> Dimension:
    return dimension(node.numerator, f"{path}[0]") / dimension(node.denominator, f"{path}[1]")


def _pow(node: Pow, path: str) -> Dimension:
    base = dimension(node.base, f"{path}[0]")
    if isinstance(node.exponent, Fraction):
        return base**node.exponent
    exp = dimension(node.exponent, f"{path}[1]")
    if not exp.is_dimensionless:
        raise DimensionError(f"{path}: a variable exponent must be dimensionless, got {exp}")
    if not base.is_dimensionless:
        raise DimensionError(
            f"{path}: a base raised to a variable exponent must be dimensionless, got {base}; "
            "divide by a scale parameter first"
        )
    return dimensionless()


def _apply(node: Apply, path: str) -> Dimension:
    arg = dimension(node.arg, f"{path}[0]")
    if not arg.is_dimensionless:
        raise DimensionError(
            f"{path}: {node.fn}() requires a dimensionless argument, got {arg}; "
            "divide by a reference or scale parameter first"
        )
    return dimensionless()


def _link(node: Link, path: str) -> Dimension:
    arg = dimension(node.arg, f"{path}[0]")
    if not arg.is_dimensionless:
        raise DimensionError(
            f"{path}: link {node.fn!r} requires a dimensionless argument, got {arg}"
        )
    return dimensionless()


def _convolve(node: Convolve, path: str) -> Dimension:
    signal = dimension(node.signal, f"{path}[0]")
    kernel = dimension(node.kernel, f"{path}[1]")
    if not kernel.is_dimensionless:
        raise DimensionError(
            f"{path}: convolution weights must be dimensionless, got {kernel}; "
            "a per-period-versus-cumulative mismatch usually shows up here"
        )
    return signal * kernel


def _opaque(node: Opaque, path: str) -> Dimension:
    for i, inp in enumerate(node.inputs):
        dimension(inp, f"{path}[{i}]")  # inputs must themselves be well-formed
    return node.dimension


def _equation(node: Equation, path: str) -> Dimension:
    lhs = dimension(node.lhs, f"{path}[0]")
    rhs = dimension(node.rhs, f"{path}[1]")
    if lhs != rhs:
        label = f" {node.name!r}" if node.name else ""
        raise DimensionError(f"{path}: equation{label} does not balance: lhs {lhs} vs rhs {rhs}")
    return lhs


def _system(node: System, path: str) -> Dimension:
    dims = [dimension(eq, f"{path}[{i}]") for i, eq in enumerate(node.equations)]
    return dims[0]


def _ode(node: ODESystem, path: str) -> Dimension:
    t = node.time.dimension
    n = len(node.states)
    for i, (state, rhs) in enumerate(zip(node.states, node.rhs, strict=True)):
        expected = state.dimension / t
        got = dimension(rhs, f"{path}[{n + i}]")
        if got != expected:
            raise DimensionError(
                f"{path}: d({state.name})/d{node.time.name} has dimension {expected} "
                f"but its right-hand side is {got}"
            )
    return node.states[0].dimension


_DISPATCH: dict[type, Callable[..., Dimension]] = {
    Const: _leaf,
    Data: _leaf,
    Param: _leaf,
    Opaque: _opaque,
    Add: _add,
    Mul: _mul,
    Div: _div,
    Pow: _pow,
    Apply: _apply,
    Link: _link,
    Convolve: _convolve,
    Equation: _equation,
    System: _system,
    ODESystem: _ode,
}


def dimension(model: Model, path: str = "") -> Dimension:
    """The dimension of ``model``; raises ``DimensionError`` naming the failing node."""
    here = f"{path}.{model.node}" if path else model.node
    return _DISPATCH[type(model)](model, here)


def check(model: Model, expected: Dimension | None = None) -> Dimension:
    """``dimension`` plus an optional assertion against a declared dimension."""
    got = dimension(model)
    if expected is not None and got != expected:
        raise DimensionError(f"expression has dimension {got}, declared {expected}")
    return got
