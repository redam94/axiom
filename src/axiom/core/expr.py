"""The model expression tree: a closed set of typed ``Spec`` nodes.

A regression is ``Add(Mul(Param, Data), ...)``; a structural model is a
``System``; a carryover process is a ``Convolve``; a differential equation is
an ``ODESystem``. One tree, several interpreters (``axiom.core.interpret``):
``value`` *is* ``forward()``, ``dimension`` is the type checker, ``latex`` is
the renderer. There is no second declaration of a model's units that could
disagree with the model.

Nodes are flat ``Spec`` classes tagged by a ``node`` literal so the recursive
``Expr`` union is a pydantic discriminated union and serializes. There is no
node base class; shared behaviour is in functions (``children``, ``params``,
``data_names``, the operators below) and in the interpreters' dispatch tables.

Nodes carry the arithmetic operators, so ``beta * dose / k`` builds exactly the
tree ``Div(numerator=Mul(factors=(beta, dose)), denominator=k)`` builds. The
constructors remain the canonical form and every existing call site is
unchanged; the operators are there so a model reads like the mathematics. A
bare number lifts to a *dimensionless* ``Const`` — nothing local could give it
another dimension — so a literal that carries units is still written out.

Decisions applied from the plan review: ``Pow`` with a rational constant is
allowed on a dimensioned base and gives ``dim ** q`` (A3); ``ODESystem`` is
dimension-check-only in 1.0 (C1); there is no general ``Deriv`` node (C2) —
kernels ship closed-form derivatives.
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from numbers import Real
from typing import Annotated, Any, Literal, get_args

from pydantic import Field, field_validator, model_validator

from axiom.core.dimensions import Dimension, dimensionless
from axiom.core.result import NonEmptyStr
from axiom.core.spec import Spec

__all__ = [
    "Add",
    "Apply",
    "ApplyFn",
    "Const",
    "Convolve",
    "Data",
    "Div",
    "Equation",
    "Expr",
    "Gather",
    "Link",
    "LinkFn",
    "Model",
    "Mul",
    "ODESystem",
    "Opaque",
    "Param",
    "Pow",
    "PRIOR_HYPER",
    "Prior",
    "PriorFamily",
    "Reduce",
    "System",
    "children",
    "data_names",
    "node_path",
    "params",
    "walk",
]

ApplyFn = Literal[
    "exp",
    "log",
    "log1p",
    "expm1",
    "tanh",
    "sigmoid",
    "logit",
    "softplus",
    "neg",
    "relu",
    "step",
    "sin",
    "cos",
]
LinkFn = Literal["identity", "log", "logit"]

_MAX_EXPONENT_DENOMINATOR = 10_000
"""A rational exponent is snapped to a denominator no larger than this."""


PriorFamily = Literal["normal", "halfnormal", "lognormal", "beta", "gamma", "uniform", "fixed"]
PRIOR_HYPER: dict[str, tuple[str, ...]] = {
    "normal": ("mu", "sigma"),
    "halfnormal": ("sigma",),
    "lognormal": ("mu", "sigma"),
    "beta": ("alpha", "beta"),
    "gamma": ("alpha", "beta"),
    "uniform": ("low", "high"),
    "fixed": ("value",),
}
"""Required hyperparameters per family. ``gamma`` uses shape ``alpha`` and *rate* ``beta``."""


class Prior(Spec):
    """A named prior family with hyperparameters.

    A hyperparameter is a number or the *name of another parameter* — that is
    how a hierarchy is declared (``alpha_unit ~ normal(mu="alpha_mean",
    sigma="alpha_sd")``). The support follows the family: ``normal`` on R;
    ``halfnormal``, ``lognormal``, ``gamma`` on R+; ``beta`` on (0, 1);
    ``uniform`` on (low, high); ``fixed`` is a point mass and not inferred.
    """

    family: PriorFamily
    hyper: dict[str, float | str]

    @model_validator(mode="after")
    def _complete(self) -> Prior:
        need = PRIOR_HYPER[self.family]
        missing = [h for h in need if h not in self.hyper]
        extra = [h for h in self.hyper if h not in need]
        if missing or extra:
            raise ValueError(
                f"{self.family} prior takes {need}; missing {missing}, unexpected {extra}"
            )
        for k, v in self.hyper.items():
            if isinstance(v, str) and not v.strip():
                raise ValueError(f"hyperparameter {k!r} names an empty parameter")
            if isinstance(v, bool):
                raise ValueError(f"hyperparameter {k!r} cannot be bool")
        return self

    @property
    def parents(self) -> tuple[str, ...]:
        """Names of parameters this prior's hyperparameters refer to."""
        return tuple(v for v in self.hyper.values() if isinstance(v, str))


# -- operators -------------------------------------------------------------------------

# ``beta * dose / k`` builds the same tree as ``Div(numerator=Mul(factors=(beta, dose)),
# denominator=k)``; the constructors stay the documented form and nothing in the codebase
# has to change. Every node class aliases these functions in its body — that is a method
# to Python and to mypy, and it keeps the promise in this module's docstring that there is
# no node base class and shared behaviour lives in functions.
#
# Deliberately absent: ``__eq__`` and the comparisons. Structural equality is load-bearing
# (``node_path``, the ``params`` dedup, ``content_hash``, every golden fixture) and there
# is no comparison node to build.


def _lift(v: Expr | float | int) -> Expr:
    """A bare number becomes a *dimensionless* ``Const``.

    Nothing local could give it another dimension, and a literal that silently
    borrowed its neighbour's would be exactly the wrong-number bug rule 4 exists
    to prevent: write ``Const(value=..., dimension=...)`` when it carries units.
    Any real number is accepted, numpy scalars included; ``bool`` is not, because
    a ``True`` that quietly became ``1.0`` is never what was meant.
    """
    if isinstance(v, bool):
        raise TypeError("bool is not an expression operand")
    if isinstance(v, Real):
        return Const(value=float(v), dimension=dimensionless())
    if isinstance(v, _EXPR_CLASSES):
        return v
    raise TypeError(
        f"{type(v).__name__} is not an expression operand; expected an Expr node or a number"
    )


def _is_zero(v: object) -> bool:
    """A bare literal zero, which is the additive identity in *any* dimension.

    Dropping it is what makes ``sum(terms)`` build the tree a reader expects:
    ``sum`` starts from ``0``, and a dimensionless zero left in the sum would
    fail the dimension check for every model whose terms carry units. An
    explicit ``Const(value=0.0, ...)`` node is a term the author wrote and is
    never dropped.
    """
    return isinstance(v, Real) and not isinstance(v, bool) and v == 0


def _add(self: Any, other: Expr | float | int) -> Expr:
    """``a + b``, flattened: ``Add`` is n-ary, so ``(a + b) + c`` and ``a + (b + c)`` are
    the one three-term node and hash alike. A bare ``0`` on either side is dropped."""
    if _is_zero(other):
        return _lift(self)
    if _is_zero(self):
        return _lift(other)
    left, right = _lift(self), _lift(other)
    terms = (left.terms if isinstance(left, Add) else (left,)) + (
        right.terms if isinstance(right, Add) else (right,)
    )
    return Add(terms=terms)


def _radd(self: Any, other: Expr | float | int) -> Expr:
    return _add(other, self)


def _mul(self: Any, other: Expr | float | int) -> Expr:
    """``a * b``, flattened the same way ``Add`` is."""
    left, right = _lift(self), _lift(other)
    factors = (left.factors if isinstance(left, Mul) else (left,)) + (
        right.factors if isinstance(right, Mul) else (right,)
    )
    return Mul(factors=factors)


def _rmul(self: Any, other: Expr | float | int) -> Expr:
    return _mul(other, self)


def _neg(self: Any) -> Expr:
    """``-a`` is ``(-1) · a``, not ``Apply(fn="neg")``: ``neg`` takes a dimensionless
    argument, and a negated expression keeps whatever dimension it had."""
    return _mul(-1.0, self)


def _sub(self: Any, other: Expr | float | int) -> Expr:
    if _is_zero(other):
        return _lift(self)
    return _add(self, _neg(_lift(other)))


def _rsub(self: Any, other: Expr | float | int) -> Expr:
    return _add(other, _neg(self))


def _truediv(self: Any, other: Expr | float | int) -> Expr:
    return Div(numerator=_lift(self), denominator=_lift(other))


def _rtruediv(self: Any, other: Expr | float | int) -> Expr:
    return Div(numerator=_lift(other), denominator=_lift(self))


def _pow(self: Any, other: Expr | Fraction | float | int) -> Expr:
    """``a ** q``. A number becomes the rational exponent ``Pow`` snaps it to; an
    expression exponent stays one, and then both sides must be dimensionless."""
    if isinstance(other, bool):
        raise TypeError("bool is not an exponent")
    exponent: Expr | Fraction = (
        Fraction(other).limit_denominator(_MAX_EXPONENT_DENOMINATOR)
        if isinstance(other, int | float | Real)
        else other
    )
    return Pow(base=_lift(self), exponent=exponent)


def _rpow(self: Any, other: Expr | float | int) -> Expr:
    return Pow(base=_lift(other), exponent=_lift(self))


def _getitem(self: Any, index: Data) -> Expr:
    """``alpha[unit_index]`` — the unit-level parameter meeting the panel."""
    if not isinstance(index, Data):
        raise TypeError(
            f"an expression is indexed by a Data column of integers, "
            f"not by {type(index).__name__}"
        )
    return Gather(source=_lift(self), index=index)


# -- leaves --------------------------------------------------------------------


class Const(Spec):
    """A literal with a declared dimension: a scalar, or a short vector (a lag index, knots).

    Specs hold no large arrays; a vector ``Const`` is for structural constants
    of a model (``(0, 1, ..., L-1)`` for carryover lags), not for data.
    """

    node: Literal["const"] = "const"
    value: float | tuple[float, ...]
    dimension: Dimension

    @field_validator("value")
    @classmethod
    def _finite(cls, v: float | tuple[float, ...]) -> float | tuple[float, ...]:
        vals = v if isinstance(v, tuple) else (v,)
        if not vals:
            raise ValueError("a vector Const needs at least one element")
        if len(vals) > 4096:
            raise ValueError("a Const holds at most 4096 values; data belongs in the Panel")
        import math

        if not all(math.isfinite(x) for x in vals):
            raise ValueError("Const values must be finite")
        return v

    @property
    def is_vector(self) -> bool:
        return isinstance(self.value, tuple)

    __add__, __radd__, __sub__, __rsub__ = _add, _radd, _sub, _rsub
    __mul__, __rmul__, __truediv__, __rtruediv__ = _mul, _rmul, _truediv, _rtruediv
    __pow__, __rpow__, __neg__, __getitem__ = _pow, _rpow, _neg, _getitem


class Data(Spec):
    """A column of the panel. Its dimension is the one its role declares."""

    node: Literal["data"] = "data"
    name: NonEmptyStr
    dimension: Dimension

    __add__, __radd__, __sub__, __rsub__ = _add, _radd, _sub, _rsub
    __mul__, __rmul__, __truediv__, __rtruediv__ = _mul, _rmul, _truediv, _rtruediv
    __pow__, __rpow__, __neg__, __getitem__ = _pow, _rpow, _neg, _getitem


class Param(Spec):
    """A parameter to be inferred, with a declared dimension and an optional prior.

    Scale parameters carry dose or time dimensions; shape parameters are
    dimensionless. That split is what ``meta`` pools on.
    """

    node: Literal["param"] = "param"
    name: NonEmptyStr
    dimension: Dimension
    prior: Prior | None = None
    shape: tuple[int, ...] = ()

    @field_validator("shape")
    @classmethod
    def _positive(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(n < 1 for n in v):
            raise ValueError("shape entries must be positive")
        return v

    @property
    def is_shape(self) -> bool:
        return self.dimension.is_dimensionless

    @property
    def size(self) -> int:
        out = 1
        for n in self.shape:
            out *= n
        return out

    __add__, __radd__, __sub__, __rsub__ = _add, _radd, _sub, _rsub
    __mul__, __rmul__, __truediv__, __rtruediv__ = _mul, _rmul, _truediv, _rtruediv
    __pow__, __rpow__, __neg__, __getitem__ = _pow, _rpow, _neg, _getitem


# -- combinators ------------------------------------------------------------------


class Add(Spec):
    """Sum of terms sharing one dimension."""

    node: Literal["add"] = "add"
    terms: tuple[Expr, ...] = Field(min_length=1)

    __add__, __radd__, __sub__, __rsub__ = _add, _radd, _sub, _rsub
    __mul__, __rmul__, __truediv__, __rtruediv__ = _mul, _rmul, _truediv, _rtruediv
    __pow__, __rpow__, __neg__, __getitem__ = _pow, _rpow, _neg, _getitem


class Mul(Spec):
    """Product; exponents add."""

    node: Literal["mul"] = "mul"
    factors: tuple[Expr, ...] = Field(min_length=1)

    __add__, __radd__, __sub__, __rsub__ = _add, _radd, _sub, _rsub
    __mul__, __rmul__, __truediv__, __rtruediv__ = _mul, _rmul, _truediv, _rtruediv
    __pow__, __rpow__, __neg__, __getitem__ = _pow, _rpow, _neg, _getitem


class Div(Spec):
    """Quotient; exponents subtract."""

    node: Literal["div"] = "div"
    numerator: Expr
    denominator: Expr

    __add__, __radd__, __sub__, __rsub__ = _add, _radd, _sub, _rsub
    __mul__, __rmul__, __truediv__, __rtruediv__ = _mul, _rmul, _truediv, _rtruediv
    __pow__, __rpow__, __neg__, __getitem__ = _pow, _rpow, _neg, _getitem


class Pow(Spec):
    """``base ** exponent``.

    A rational constant exponent is allowed on any base and gives
    ``dim(base) ** q``. An expression exponent requires both base and exponent
    to be dimensionless.
    """

    node: Literal["pow"] = "pow"
    base: Expr
    exponent: Annotated[Expr | Fraction, Field(union_mode="left_to_right")]

    @field_validator("exponent", mode="before")
    @classmethod
    def _coerce(cls, v: object) -> object:
        if isinstance(v, bool):
            raise ValueError("exponent cannot be bool")
        if isinstance(v, int | float | str):
            return Fraction(v).limit_denominator(_MAX_EXPONENT_DENOMINATOR)
        return v

    __add__, __radd__, __sub__, __rsub__ = _add, _radd, _sub, _rsub
    __mul__, __rmul__, __truediv__, __rtruediv__ = _mul, _rmul, _truediv, _rtruediv
    __pow__, __rpow__, __neg__, __getitem__ = _pow, _rpow, _neg, _getitem


class Apply(Spec):
    """A scalar function of one argument. Argument and result are dimensionless.

    ``relu`` (``max(x, 0)``) and ``step`` (``1`` where ``x > 0``, else ``0``)
    are the two non-smooth members, shipped for the truncated-power bases the
    spline kernels build: ``relu`` raised to a power ``p >= 2`` is
    ``C^(p-1)``, so a cubic spline's kink is invisible to a gradient, and
    ``step`` is exactly ``d relu / dx`` away from the origin. Both take the
    value ``0`` at ``x = 0``, identically under numpy and jax, so a
    piecewise-linear derivative evaluated *at* a knot reports the slope
    arriving into it rather than the one leaving.

    ``sin`` and ``cos`` are here for the Laplacian eigenfunctions of the
    Hilbert-space Gaussian-process basis (``surface.GaussianProcessKernel``):
    the basis is a sine and its derivative is a cosine.
    """

    node: Literal["apply"] = "apply"
    fn: ApplyFn
    arg: Expr

    __add__, __radd__, __sub__, __rsub__ = _add, _radd, _sub, _rsub
    __mul__, __rmul__, __truediv__, __rtruediv__ = _mul, _rmul, _truediv, _rtruediv
    __pow__, __rpow__, __neg__, __getitem__ = _pow, _rpow, _neg, _getitem


class Convolve(Spec):
    """Causal convolution of ``signal`` with ``kernel`` along the time axis.

    ``kernel`` evaluates to a weight vector ``(..., L)`` and must be
    dimensionless (discrete weights); the result has ``dim(signal) ·
    dim(kernel)``. The value interpreter computes ``y[..., t] = Σ_l w[..., l]
    · x[..., t-l]`` along the last axis of ``signal``, broadcasting any
    leading (draw) axes of ``kernel`` against those of ``signal``.
    """

    node: Literal["convolve"] = "convolve"
    signal: Expr
    kernel: Expr

    __add__, __radd__, __sub__, __rsub__ = _add, _radd, _sub, _rsub
    __mul__, __rmul__, __truediv__, __rtruediv__ = _mul, _rmul, _truediv, _rtruediv
    __pow__, __rpow__, __neg__, __getitem__ = _pow, _rpow, _neg, _getitem


class Reduce(Spec):
    """Reduce ``arg`` along its last axis: ``sum``, ``mean``, ``max``. Dimension is preserved.

    Used to normalize carryover weights (``w / sum(w)``) and for aggregate
    outcomes; the last axis is the one ``Convolve`` acts along.
    """

    node: Literal["reduce"] = "reduce"
    op: Literal["sum", "mean", "max"]
    arg: Expr
    keepdims: bool = False
    """Keep the reduced axis (length 1) so the result broadcasts against ``arg`` —
    use this to normalize a vector by its sum when parameters carry draw axes."""

    __add__, __radd__, __sub__, __rsub__ = _add, _radd, _sub, _rsub
    __mul__, __rmul__, __truediv__, __rtruediv__ = _mul, _rmul, _truediv, _rtruediv
    __pow__, __rpow__, __neg__, __getitem__ = _pow, _rpow, _neg, _getitem


class Gather(Spec):
    """``source[..., index]``: pick entries of a vector-valued expression by an integer column.

    This is how a unit-level parameter meets the panel: ``Gather(Param("alpha",
    shape=(n_units,)), Data("unit_index"))``. ``index`` is a ``Data`` column of
    zero-based integers; its dimension is ignored. Dimension is ``dim(source)``.
    """

    node: Literal["gather"] = "gather"
    source: Expr
    index: Data

    __add__, __radd__, __sub__, __rsub__ = _add, _radd, _sub, _rsub
    __mul__, __rmul__, __truediv__, __rtruediv__ = _mul, _rmul, _truediv, _rtruediv
    __pow__, __rpow__, __neg__, __getitem__ = _pow, _rpow, _neg, _getitem


class Link(Spec):
    """A GLM link applied to a dimensionless argument (after a reference divide)."""

    node: Literal["link"] = "link"
    fn: LinkFn
    arg: Expr

    __add__, __radd__, __sub__, __rsub__ = _add, _radd, _sub, _rsub
    __mul__, __rmul__, __truediv__, __rtruediv__ = _mul, _rmul, _truediv, _rtruediv
    __pow__, __rpow__, __neg__, __getitem__ = _pow, _rpow, _neg, _getitem


class Opaque(Spec):
    """A user-supplied function the tree cannot express.

    Declares its result dimension; the value interpreter looks the function
    up by ``name`` in the registry it is given. Anything needing
    introspection (LaTeX, symbolic transport) degrades to ``Unsupported``.
    """

    node: Literal["opaque"] = "opaque"
    name: NonEmptyStr
    inputs: tuple[Expr, ...] = ()
    dimension: Dimension

    __add__, __radd__, __sub__, __rsub__ = _add, _radd, _sub, _rsub
    __mul__, __rmul__, __truediv__, __rtruediv__ = _mul, _rmul, _truediv, _rtruediv
    __pow__, __rpow__, __neg__, __getitem__ = _pow, _rpow, _neg, _getitem


Expr = Annotated[
    Const
    | Data
    | Param
    | Add
    | Mul
    | Div
    | Pow
    | Apply
    | Convolve
    | Reduce
    | Gather
    | Link
    | Opaque,
    Field(discriminator="node"),
]

_EXPR_CLASSES: tuple[type[Spec], ...] = get_args(get_args(Expr)[0])
"""The union's members, for the operators' operand check. Derived so it cannot drift."""


# -- structure -----------------------------------------------------------------------


class Equation(Spec):
    """``lhs = rhs``; both sides must share a dimension."""

    node: Literal["equation"] = "equation"
    lhs: Expr
    rhs: Expr
    name: str = ""


class System(Spec):
    """A set of equations, each checked independently."""

    node: Literal["system"] = "system"
    equations: tuple[Equation, ...] = Field(min_length=1)


class ODESystem(Spec):
    """``d(state_i)/dt = rhs_i`` for each state. Dimension-check-only in 1.0 (C1).

    Each ``rhs_i`` must have dimension ``dim(state_i) / dim(time)``. States
    are ``Data`` nodes (their dimension is declared there) and may appear in
    any right-hand side.
    """

    node: Literal["ode"] = "ode"
    states: tuple[Data, ...] = Field(min_length=1)
    rhs: tuple[Expr, ...] = Field(min_length=1)
    time: Data

    @model_validator(mode="after")
    def _aligned(self) -> ODESystem:
        if len(self.states) != len(self.rhs):
            raise ValueError(
                f"ODESystem has {len(self.states)} states but {len(self.rhs)} right-hand sides"
            )
        names = [s.name for s in self.states]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate state names: {names}")
        return self


Model = Expr | Equation | System | ODESystem
"""Anything an interpreter accepts."""

for _cls in (
    Add,
    Mul,
    Div,
    Pow,
    Apply,
    Convolve,
    Reduce,
    Gather,
    Link,
    Opaque,
    Equation,
    System,
    ODESystem,
):
    _cls.model_rebuild()

# -- traversal ------------------------------------------------------------------------


def children(node: Model) -> tuple[Model, ...]:
    """Direct sub-nodes, in a stable order."""
    match node:
        case Add():
            return node.terms
        case Mul():
            return node.factors
        case Div():
            return (node.numerator, node.denominator)
        case Pow():
            return (
                (node.base, node.exponent)
                if not isinstance(node.exponent, Fraction)
                else (node.base,)
            )
        case Apply() | Link() | Reduce():
            return (node.arg,)
        case Convolve():
            return (node.signal, node.kernel)
        case Gather():
            return (node.source, node.index)
        case Opaque():
            return node.inputs
        case Equation():
            return (node.lhs, node.rhs)
        case System():
            return node.equations
        case ODESystem():
            return (*node.states, *node.rhs, node.time)
        case _:
            return ()


def walk(node: Model, path: str = "") -> Iterator[tuple[str, Model]]:
    """Pre-order traversal yielding ``(path, node)``; paths look like ``add[1].mul[0]``."""
    yield path or node.node, node
    for i, child in enumerate(children(node)):
        yield from walk(child, f"{path or node.node}[{i}].{child.node}")


def node_path(root: Model, target: Model) -> str:
    for p, n in walk(root):
        if n is target or n == target:
            return p
    raise KeyError("target node is not in the tree")


def params(node: Model) -> tuple[Param, ...]:
    """Every distinct ``Param`` in the tree, by first appearance."""
    out: dict[str, Param] = {}
    for _, n in walk(node):
        if isinstance(n, Param) and n.name not in out:
            out[n.name] = n
    return tuple(out.values())


def data_names(node: Model) -> tuple[str, ...]:
    """Every distinct ``Data`` name in the tree, by first appearance."""
    out: list[str] = []
    for _, n in walk(node):
        if isinstance(n, Data) and n.name not in out:
            out.append(n.name)
    return tuple(out)
