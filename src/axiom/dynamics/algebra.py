"""Building and rewriting ``Expr`` trees: substitution, and arithmetic that keeps them small.

Unrolling and elimination both work by *rewriting* expression trees, and a
naive rewrite grows them fast: substituting one equation into another twenty
times over turns a three-node right-hand side into thousands of nodes, most
of them ``x + 0`` and ``1 * y``. Every constructor here folds those away as
it builds, which is the difference between an unrolled system that
serializes and one that does not.

The folding is *structural*, never numeric: ``mul(a, b)`` drops a factor
only when it is literally the constant one, and never reorders anything that
would change floating-point results. Two expressions built through these
helpers evaluate identically to the same expressions built by hand — gate 9
compares numpy against jax on the result, which is where a clever
"simplification" would be caught.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from fractions import Fraction

from axiom.core import (
    Add,
    Apply,
    Const,
    Convolve,
    Data,
    Dimension,
    Div,
    Expr,
    Gather,
    Link,
    Mul,
    Opaque,
    Param,
    Pow,
    Reduce,
    dimension,
    dimensionless,
    walk,
)

__all__ = [
    "add",
    "const",
    "div",
    "is_one",
    "is_zero",
    "map_data",
    "mul",
    "neg",
    "one",
    "size",
    "sub",
    "substitute",
    "zero",
]


def const(value: float, dim: Dimension | None = None) -> Const:
    """A scalar constant; dimensionless unless a dimension is given."""
    return Const(value=float(value), dimension=dim if dim is not None else dimensionless())


def zero(dim: Dimension | None = None) -> Const:
    return const(0.0, dim)


def one() -> Const:
    return const(1.0)


def constant_value(expr: Expr) -> float | None:
    """The value of an expression built only from scalar constants, or ``None``.

    Elimination produces entries like ``1 - 1``: arithmetic on constants that
    happens to cancel. Recognizing that as zero is what tells a singular
    block from a solvable one, and it has to be *exact* — so this evaluates
    the constant subtree with the same operations the interpreter would,
    rather than folding constants into the tree and hoping the order did not
    matter.
    """
    match expr:
        case Const():
            if expr.is_vector:
                return None
            assert isinstance(expr.value, float)
            return expr.value
        case Add():
            total = 0.0
            for term in expr.terms:
                v = constant_value(term)
                if v is None:
                    return None
                total += v
            return total
        case Mul():
            product = 1.0
            for factor in expr.factors:
                v = constant_value(factor)
                if v is None:
                    return None
                product *= v
            return product
        case Div():
            top, bottom = constant_value(expr.numerator), constant_value(expr.denominator)
            if top is None or bottom is None or bottom == 0.0:
                return None
            return top / bottom
        case _:
            return None


def is_zero(expr: Expr) -> bool:
    """True when the expression is constant and exactly zero."""
    return constant_value(expr) == 0.0


def is_one(expr: Expr) -> bool:
    """True when the expression is a dimensionless constant exactly equal to one."""
    return constant_value(expr) == 1.0 and dimension(expr).is_dimensionless


def size(expr: Expr) -> int:
    """Node count — what decides whether an unrolled system is worth serializing."""
    return sum(1 for _ in walk(expr))


def add(*terms: Expr) -> Expr:
    """Sum, flattening nested ``Add`` and dropping structural zeros.

    Summing nothing, or only zeros, gives a zero of the terms' dimension.
    """
    flat: list[Expr] = []
    for t in terms:
        if isinstance(t, Add):
            flat.extend(t.terms)
        else:
            flat.append(t)
    kept = [t for t in flat if not is_zero(t)]
    if not kept:
        return zero(dimension(flat[0])) if flat else zero()
    if len(kept) == 1:
        return kept[0]
    return Add(terms=tuple(kept))


def _is_signed_one(expr: Expr) -> bool:
    return (
        isinstance(expr, Const)
        and not expr.is_vector
        and expr.value == -1.0
        and expr.dimension.is_dimensionless
    )


def mul(*factors: Expr) -> Expr:
    """Product, flattening nested ``Mul``, absorbing zero, and collapsing signs.

    Dimensionless factors of exactly ``1`` are dropped and repeated ``-1``
    factors collapse to at most one. Multiplying by ``±1`` is exact in
    floating point, so this changes the tree and never the number — which
    is the line this module does not cross.
    """
    flat: list[Expr] = []
    for f in factors:
        if isinstance(f, Mul):
            flat.extend(f.factors)
        else:
            flat.append(f)
    if any(is_zero(f) for f in flat):
        dim = dimensionless()
        for f in flat:
            dim = dim * dimension(f)
        return zero(dim)
    negatives = sum(1 for f in flat if _is_signed_one(f))
    kept = [f for f in flat if not is_one(f) and not _is_signed_one(f)]
    if negatives % 2:
        kept.insert(0, const(-1.0))
    if not kept:
        return one()
    if len(kept) == 1:
        return kept[0]
    return Mul(factors=tuple(kept))


def neg(expr: Expr) -> Expr:
    """``-expr``, as a multiplication by the dimensionless constant ``-1``."""
    if is_zero(expr):
        return expr
    return mul(const(-1.0), expr)


def sub(left: Expr, right: Expr) -> Expr:
    return add(left, neg(right))


def div(numerator: Expr, denominator: Expr) -> Expr:
    """Quotient; a numerator that is structurally zero stays zero, ``/1`` disappears."""
    if is_one(denominator):
        return numerator
    if is_zero(numerator):
        return zero(dimension(numerator) / dimension(denominator))
    return Div(numerator=numerator, denominator=denominator)


def map_data(expr: Expr, fn: Callable[[Data], Expr]) -> Expr:
    """Rebuild ``expr`` with every ``Data`` leaf replaced by ``fn(leaf)``.

    A ``Gather`` index must stay a ``Data`` node — the tree has no other way
    to name an integer column — so a replacement that is not one raises
    ``TypeError`` naming the index.
    """
    match expr:
        case Const() | Param():
            return expr
        case Data():
            return fn(expr)
        case Add():
            return add(*(map_data(t, fn) for t in expr.terms))
        case Mul():
            return mul(*(map_data(f, fn) for f in expr.factors))
        case Div():
            return div(map_data(expr.numerator, fn), map_data(expr.denominator, fn))
        case Pow():
            base = map_data(expr.base, fn)
            if isinstance(expr.exponent, Fraction):
                return Pow(base=base, exponent=expr.exponent)
            return Pow(base=base, exponent=map_data(expr.exponent, fn))
        case Apply():
            return Apply(fn=expr.fn, arg=map_data(expr.arg, fn))
        case Link():
            return Link(fn=expr.fn, arg=map_data(expr.arg, fn))
        case Reduce():
            return Reduce(op=expr.op, arg=map_data(expr.arg, fn), keepdims=expr.keepdims)
        case Convolve():
            return Convolve(signal=map_data(expr.signal, fn), kernel=map_data(expr.kernel, fn))
        case Gather():
            index = fn(expr.index)
            if not isinstance(index, Data):
                raise TypeError(
                    f"a Gather index must stay a data column; {expr.index.name!r} was "
                    f"replaced by a {type(index).__name__}"
                )
            return Gather(source=map_data(expr.source, fn), index=index)
        case Opaque():
            return Opaque(
                name=expr.name,
                inputs=tuple(map_data(i, fn) for i in expr.inputs),
                dimension=expr.dimension,
            )
    raise TypeError(f"not an expression node: {type(expr).__name__}")


def substitute(expr: Expr, mapping: Mapping[str, Expr]) -> Expr:
    """Replace ``Data`` leaves by name; leaves the mapping does not name are kept."""

    def replace(leaf: Data) -> Expr:
        return mapping.get(leaf.name, leaf)

    return map_data(expr, replace)
