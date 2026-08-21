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
``data_names``) and in the interpreters' dispatch tables.

Decisions applied from the plan review: ``Pow`` with a rational constant is
allowed on a dimensioned base and gives ``dim ** q`` (A3); ``ODESystem`` is
dimension-check-only in 1.0 (C1); there is no general ``Deriv`` node (C2) —
kernels ship closed-form derivatives.
"""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from axiom.core.dimensions import Dimension
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
    "Link",
    "LinkFn",
    "Model",
    "Mul",
    "ODESystem",
    "Opaque",
    "Param",
    "Pow",
    "Prior",
    "System",
    "children",
    "data_names",
    "node_path",
    "params",
    "walk",
]

ApplyFn = Literal["exp", "log", "log1p", "expm1", "tanh", "sigmoid", "logit", "softplus", "neg"]
LinkFn = Literal["identity", "log", "logit"]


class Prior(Spec):
    """A named prior family with numeric hyperparameters. Interpreted by ``infer``."""

    family: Literal["normal", "halfnormal", "lognormal", "beta", "gamma", "uniform", "fixed"]
    hyper: dict[str, float]


# -- leaves --------------------------------------------------------------------


class Const(Spec):
    """A literal with a declared dimension."""

    node: Literal["const"] = "const"
    value: float
    dimension: Dimension


class Data(Spec):
    """A column of the panel. Its dimension is the one its role declares."""

    node: Literal["data"] = "data"
    name: NonEmptyStr
    dimension: Dimension


class Param(Spec):
    """A parameter to be inferred, with a declared dimension and an optional prior.

    Scale parameters carry dose or time dimensions; shape parameters are
    dimensionless. That split is what ``meta`` pools on.
    """

    node: Literal["param"] = "param"
    name: NonEmptyStr
    dimension: Dimension
    prior: Prior | None = None

    @property
    def is_shape(self) -> bool:
        return self.dimension.is_dimensionless


# -- combinators ------------------------------------------------------------------


class Add(Spec):
    """Sum of terms sharing one dimension."""

    node: Literal["add"] = "add"
    terms: tuple[Expr, ...] = Field(min_length=1)


class Mul(Spec):
    """Product; exponents add."""

    node: Literal["mul"] = "mul"
    factors: tuple[Expr, ...] = Field(min_length=1)


class Div(Spec):
    """Quotient; exponents subtract."""

    node: Literal["div"] = "div"
    numerator: Expr
    denominator: Expr


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
            return Fraction(v).limit_denominator(10_000)
        return v


class Apply(Spec):
    """A transcendental function. Argument and result are dimensionless."""

    node: Literal["apply"] = "apply"
    fn: ApplyFn
    arg: Expr


class Convolve(Spec):
    """Causal convolution of ``signal`` with ``kernel`` along the time axis.

    ``kernel`` evaluates to a 1-D weight vector and must be dimensionless
    (discrete weights); the result has ``dim(signal) · dim(kernel)``. The
    value interpreter computes ``y[t] = Σ_l w[l] · x[t-l]`` along the last
    axis of ``signal``.
    """

    node: Literal["convolve"] = "convolve"
    signal: Expr
    kernel: Expr


class Link(Spec):
    """A GLM link applied to a dimensionless argument (after a reference divide)."""

    node: Literal["link"] = "link"
    fn: LinkFn
    arg: Expr


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


Expr = Annotated[
    Const | Data | Param | Add | Mul | Div | Pow | Apply | Convolve | Link | Opaque,
    Field(discriminator="node"),
]

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

for _cls in (Add, Mul, Div, Pow, Apply, Convolve, Link, Opaque, Equation, System, ODESystem):
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
        case Apply() | Link():
            return (node.arg,)
        case Convolve():
            return (node.signal, node.kernel)
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
