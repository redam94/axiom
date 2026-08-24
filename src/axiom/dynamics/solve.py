"""Compiling simultaneity away: exact for a linear block, a declared number of sweeps otherwise.

A simultaneous block is ``m`` equations in ``m`` unknowns that hold at the
same instant. Nothing in ``axiom.core.expr`` solves equations, and adding a
node that did would put a solver inside every interpreter. Instead the block
is solved *symbolically at compile time* and what comes out is an ordinary
expression tree.

Two routes, and the report always says which one was taken.

**Linear (exact).** ``affine_split`` decides, structurally, whether each
right-hand side is affine in the block's unknowns, and if so extracts the
coefficient expressions. The block is then ``(I - A) v = b`` with ``A`` and
``b`` holding expressions, and Gauss-Jordan elimination over those
expressions gives the reduced form — the econometric reduced form, built by
the same algebra a textbook does it with. The result is exact for every
parameter value and differentiable, so the design and identifiability math
downstream sees the true derivative.

**Nonlinear (approximate, and labelled).** When a right-hand side is not
affine — a saturating response of one endogenous variable to another — the
block is compiled as ``sweeps`` Gauss-Seidel passes from a declared start.
That is an approximation, so ``BlockSolution.residuals`` carries the
expression ``rhs(v) - v`` for each unknown: evaluate it on real data and
parameters and you get the error you are actually running, rather than a
promise.

One property of a Gauss-Seidel sweep to know before reading a residual: the
*last* variable updated in the sweep satisfies its own equation exactly, so
its residual is identically zero by construction. The number to read is the
largest residual over the block, not any single one.

Fixed points are not always unique, and Gauss-Seidel does not always
converge to the one you meant. This module does not pretend otherwise: it
reports, it does not certify.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Literal

from pydantic import Field, model_validator

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
    NonEmptyStr,
    Opaque,
    Param,
    Pow,
    Reduce,
    Spec,
    Unsupported,
    walk,
)
from axiom.dynamics.algebra import add, div, is_zero, mul, neg, one, size, sub, substitute, zero

__all__ = [
    "AffineForm",
    "BlockSolution",
    "SolveMethod",
    "affine_split",
    "gauss_seidel",
    "reduced_form",
    "solve_block",
]

SolveMethod = Literal["substitution", "linear", "fixed_point"]

MAX_NODES = 20_000
"""Node budget for a compiled block. Past this the solution is reported ``Unsupported``
rather than returned: a tree this size neither serializes nor reads."""


class AffineForm:
    """``expr = sum_j coefficients[j] * unknown_j + intercept``, with expression coefficients.

    Not a ``Spec``: it is the intermediate of one call, and its parts end up
    inside the ``BlockSolution`` that is one.
    """

    __slots__ = ("coefficients", "intercept")

    def __init__(self, coefficients: Mapping[str, Expr], intercept: Expr) -> None:
        self.coefficients = dict(coefficients)
        self.intercept = intercept

    def coefficient(self, name: str, dim: Dimension) -> Expr:
        """The coefficient on ``name``, or a zero of dimension ``dim``."""
        return self.coefficients.get(name, zero(dim))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"AffineForm(on={sorted(self.coefficients)})"


def _mentions(expr: Expr, unknowns: frozenset[str]) -> bool:
    return any(isinstance(n, Data) and n.name in unknowns for _, n in walk(expr))


def affine_split(expr: Expr, unknowns: Sequence[str]) -> AffineForm | None:
    """Split ``expr`` into coefficients on ``unknowns`` and a remainder, or ``None``.

    ``None`` means *not affine in those names*: they appear under a
    nonlinearity (``Apply``, ``Link``, a non-unit ``Pow``), multiplied by
    each other, in a denominator, or inside a node this analysis does not
    look through (``Convolve``, ``Reduce``, ``Gather``, ``Opaque``). It is a
    structural verdict, so it is conservative: ``x - x`` counts as affine
    with coefficient zero, but ``exp(log(x))`` does not count as affine at
    all.
    """
    names = frozenset(unknowns)

    def split(node: Expr) -> AffineForm | None:
        match node:
            case Const() | Param():
                return AffineForm({}, node)
            case Data():
                if node.name in names:
                    return AffineForm({node.name: one()}, zero(node.dimension))
                return AffineForm({}, node)
            case Add():
                coefficients: dict[str, Expr] = {}
                intercepts: list[Expr] = []
                for term in node.terms:
                    part = split(term)
                    if part is None:
                        return None
                    for k, v in part.coefficients.items():
                        coefficients[k] = add(coefficients[k], v) if k in coefficients else v
                    intercepts.append(part.intercept)
                return AffineForm(coefficients, add(*intercepts))
            case Mul():
                carrying = [f for f in node.factors if _mentions(f, names)]
                plain = [f for f in node.factors if not _mentions(f, names)]
                if len(carrying) > 1:
                    return None
                if not carrying:
                    return AffineForm({}, node)
                part = split(carrying[0])
                if part is None:
                    return None
                scale = mul(*plain) if plain else one()
                return AffineForm(
                    {k: mul(scale, v) for k, v in part.coefficients.items()},
                    mul(scale, part.intercept),
                )
            case Div():
                if _mentions(node.denominator, names):
                    return None
                part = split(node.numerator)
                if part is None:
                    return None
                return AffineForm(
                    {k: div(v, node.denominator) for k, v in part.coefficients.items()},
                    div(part.intercept, node.denominator),
                )
            case Pow():
                if not _mentions(node, names):
                    return AffineForm({}, node)
                if not isinstance(node.exponent, Fraction) or node.exponent != 1:
                    return None
                return split(node.base)
            case Apply() | Link() | Reduce() | Convolve() | Gather() | Opaque():
                if _mentions(node, names):
                    return None
                return AffineForm({}, node)
        raise TypeError(f"not an expression node: {type(node).__name__}")  # pragma: no cover

    return split(expr)


class BlockSolution(Spec):
    """How one block was compiled, and into what.

    ``expressions`` maps each unknown to a tree with no reference to any
    unknown left in it. ``method`` is ``"substitution"`` for a block of one
    with no self-reference, ``"linear"`` for an exact reduced form, and
    ``"fixed_point"`` for ``sweeps`` Gauss-Seidel passes. ``residuals`` is
    empty for the exact methods and holds ``rhs(v) - v`` per unknown for the
    approximate one — the thing to evaluate before believing the fit. The
    unknown updated last in each sweep has a residual that is identically
    zero; read the largest residual across the block, not one of them.
    """

    variables: tuple[NonEmptyStr, ...] = Field(min_length=1)
    method: SolveMethod
    expressions: dict[str, Expr]
    residuals: dict[str, Expr] = {}
    sweeps: int = Field(default=0, ge=0)
    node_count: int = Field(ge=1)
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _covers(self) -> BlockSolution:
        if set(self.expressions) != set(self.variables):
            raise ValueError("expressions must have exactly one entry per variable")
        if self.residuals and set(self.residuals) != set(self.variables):
            raise ValueError("residuals, when present, must have one entry per variable")
        if self.method == "fixed_point" and self.sweeps < 1:
            raise ValueError("a fixed-point solution needs at least one sweep")
        if self.method != "fixed_point" and self.residuals:
            raise ValueError(f"a {self.method} solution is exact and carries no residual")
        return self

    @property
    def exact(self) -> bool:
        return self.method != "fixed_point"


def reduced_form(
    equations: Mapping[str, Expr],
    dimensions: Mapping[str, Dimension],
    *,
    max_nodes: int = MAX_NODES,
) -> dict[str, Expr] | Unsupported:
    """Solve ``v = rhs_v(v, ...)`` exactly when every right-hand side is affine in the unknowns.

    Gauss-Jordan elimination on ``(I - A) v = b`` with expression entries.
    A pivot is usable when it is not *structurally* zero; a column with no
    usable pivot means the block is symbolically singular — the equations do
    not determine the variables separately, which is an identification
    statement about the model, not a numerical accident, so it comes back as
    ``Unsupported`` naming the column.
    """
    unknowns = tuple(equations)
    n = len(unknowns)
    forms: dict[str, AffineForm] = {}
    for v in unknowns:
        form = affine_split(equations[v], unknowns)
        if form is None:
            return Unsupported(
                reason=(
                    f"the equation for {v!r} is not affine in the block's unknowns "
                    f"{list(unknowns)}; an exact reduced form does not exist"
                ),
                detail={"variable": v, "block": ", ".join(unknowns)},
            )
        forms[v] = form

    # rows of (I - A) and the right-hand side b
    matrix: list[list[Expr]] = []
    rhs: list[Expr] = []
    for i, v in enumerate(unknowns):
        row: list[Expr] = []
        for j, u in enumerate(unknowns):
            coefficient = forms[v].coefficient(u, dimensions[v] / dimensions[u])
            entry = sub(one(), coefficient) if i == j else neg(coefficient)
            row.append(entry)
        matrix.append(row)
        rhs.append(forms[v].intercept)

    for col in range(n):
        pivot = next((r for r in range(col, n) if not is_zero(matrix[r][col])), None)
        if pivot is None:
            return Unsupported(
                reason=(
                    f"the block {list(unknowns)} is symbolically singular in {unknowns[col]!r}: "
                    "no equation determines it once the others are eliminated"
                ),
                detail={"variable": unknowns[col], "block": ", ".join(unknowns)},
            )
        if pivot != col:
            matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
            rhs[col], rhs[pivot] = rhs[pivot], rhs[col]
        pivot_entry = matrix[col][col]
        matrix[col] = [div(e, pivot_entry) for e in matrix[col]]
        rhs[col] = div(rhs[col], pivot_entry)
        for r in range(n):
            if r == col or is_zero(matrix[r][col]):
                continue
            factor = matrix[r][col]
            matrix[r] = [
                sub(a, mul(factor, b)) for a, b in zip(matrix[r], matrix[col], strict=True)
            ]
            rhs[r] = sub(rhs[r], mul(factor, rhs[col]))
        total = sum(size(e) for e in rhs)
        if total > max_nodes:
            return Unsupported(
                reason=(
                    f"eliminating the block {list(unknowns)} passed the {max_nodes}-node budget "
                    f"at column {col + 1} of {n}; solve fewer variables together or "
                    "supply the reduced form yourself"
                ),
                detail={"nodes": str(total), "block": ", ".join(unknowns)},
            )
    return {v: rhs[i] for i, v in enumerate(unknowns)}


def gauss_seidel(
    equations: Mapping[str, Expr],
    start: Mapping[str, Expr],
    sweeps: int,
    *,
    max_nodes: int = MAX_NODES,
) -> dict[str, Expr] | Unsupported:
    """``sweeps`` Gauss-Seidel passes: substitute the running values, in declaration order.

    Each pass updates the unknowns one at a time and uses the values already
    updated in the same pass, which converges faster than a Jacobi pass.

    **The tree grows geometrically in the number of sweeps** — each pass
    substitutes the whole running expression into every place an unknown
    appears, so the size multiplies rather than adds. Useful sweep counts
    are single digits; the budget is checked after every pass so an
    optimistic ``sweeps=25`` comes back as ``Unsupported`` in a second
    instead of exhausting memory. If a handful of sweeps does not make the
    residual small, the block is not a contraction at these parameters and
    more sweeps is the wrong instrument.
    """
    if sweeps < 1:
        raise ValueError(f"a fixed-point solution needs at least one sweep, got {sweeps}")
    current = {v: start[v] for v in equations}
    for sweep in range(1, sweeps + 1):
        for v, rhs in equations.items():
            current[v] = substitute(rhs, current)
        nodes = sum(size(e) for e in current.values())
        if nodes > max_nodes:
            return Unsupported(
                reason=(
                    f"sweep {sweep} of {sweeps} over the block {list(equations)} passed the "
                    f"{max_nodes}-node budget ({nodes} nodes); a fixed-point compilation grows "
                    "geometrically in the sweeps, so use fewer and read the residual"
                ),
                detail={"sweep": str(sweep), "nodes": str(nodes), "block": ", ".join(equations)},
            )
    return current


def solve_block(
    equations: Mapping[str, Expr],
    dimensions: Mapping[str, Dimension],
    *,
    simultaneous: bool,
    start: Mapping[str, Expr] | None = None,
    sweeps: int = 0,
    max_nodes: int = MAX_NODES,
) -> BlockSolution | Unsupported:
    """Compile one block: substitution, exact reduced form, or declared fixed-point sweeps.

    ``start`` and ``sweeps`` are only consulted when the block is
    simultaneous *and* not affine. Without a ``start`` a nonlinear block is
    ``Unsupported`` — iterating from an undeclared point is the kind of
    silent guess this repository does not make.
    """
    unknowns = tuple(equations)
    if not simultaneous:
        expressions = dict(equations)
        return BlockSolution(
            variables=unknowns,
            method="substitution",
            expressions=expressions,
            node_count=sum(size(e) for e in expressions.values()),
        )
    solved = reduced_form(equations, dimensions, max_nodes=max_nodes)
    if not isinstance(solved, Unsupported):
        return BlockSolution(
            variables=unknowns,
            method="linear",
            expressions=solved,
            node_count=sum(size(e) for e in solved.values()),
            detail={"block": ", ".join(unknowns)},
        )
    if start is None or sweeps < 1:
        return Unsupported(
            reason=(
                f"{solved.reason}. Compiling it as a fixed point needs a starting "
                "expression per unknown and sweeps >= 1"
            ),
            detail=solved.detail,
            missing=("start", "sweeps"),
        )
    swept = gauss_seidel(equations, start, sweeps, max_nodes=max_nodes)
    if isinstance(swept, Unsupported):
        return swept
    residuals = {v: sub(substitute(equations[v], swept), swept[v]) for v in unknowns}
    nodes = sum(size(e) for e in swept.values())
    return BlockSolution(
        variables=unknowns,
        method="fixed_point",
        expressions=swept,
        residuals=residuals,
        sweeps=sweeps,
        node_count=nodes,
        detail={"block": ", ".join(unknowns), "start": ", ".join(sorted(start))},
    )
