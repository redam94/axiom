"""``DynamicSystem``: named variables and one equation each, over time-shifted references.

A DAG is a *solved* model: it presumes you can order the variables so that
every arrow points forward. Two common situations break that presumption
without being ill-posed.

* **Simultaneity.** Price and quantity are set together; a treatment
  responds to the outcome it moves within the same accounting period.
  Written honestly the equations form a cycle, and the "graph" is not
  acyclic.
* **Time structure.** Yesterday's outcome enters today's equation. The
  summary graph over ``{treatment, outcome}`` has a loop; the graph over
  ``{treatment@t, outcome@t}`` for every ``t`` does not.

Both are handled here by *compiling*, not by extending the evaluator. A
``DynamicSystem`` is declarative: variables with dimensions and roles, and
one right-hand side per endogenous variable written as an ordinary
``axiom.core.expr`` tree whose ``Data`` leaves may name *time-shifted*
references. ``axiom.dynamics.unroll`` turns that into plain ``Expr`` trees
with no cycles and no time shifts, which the existing interpreters
evaluate — rule 3 holds: there is still exactly one ``forward()``.

Reference syntax
----------------
A variable name is ``[A-Za-z_]\\w*`` — no dots — so two suffixes are
unambiguous:

* ``y.l1``, ``y.l2`` — ``y`` at ``t-1``, ``t-2``. This is what an equation
  writes; ``y`` alone means ``y`` at ``t``.
* ``y.t3`` — ``y`` at absolute period 3. This is what *unrolling* writes;
  no equation may contain one.

Leads (``y.f1``, a forward-looking expectation) parse and are rejected with
a message naming what would be needed: a rational-expectations solution,
which this module does not do (note 0010, §"out of scope").
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Literal

from pydantic import Field, field_validator, model_validator

from axiom.core import (
    Data,
    Dimension,
    DimensionError,
    Expr,
    NonEmptyStr,
    Param,
    Spec,
    SpecError,
    dimension,
    params,
    walk,
)
from axiom.core.expr import Model

__all__ = [
    "DynamicEquation",
    "DynamicSystem",
    "DynamicsError",
    "LeadNotSupportedError",
    "Role",
    "Variable",
    "lag_ref",
    "parse_ref",
    "parse_time_ref",
    "refs_in",
    "time_ref",
]

Role = Literal["endogenous", "exogenous"]

_VAR_RE = re.compile(r"^[A-Za-z_]\w*$")
_LAG_RE = re.compile(r"^([A-Za-z_]\w*)\.l(\d+)$")
_TIME_RE = re.compile(r"^([A-Za-z_]\w*)\.t(\d+)$")
_LEAD_RE = re.compile(r"^([A-Za-z_]\w*)\.f(\d+)$")


class DynamicsError(SpecError):
    """The system is malformed: an unknown reference, a missing equation, a bad name."""


class LeadNotSupportedError(DynamicsError):
    """An equation refers to a future value; solving that needs rational expectations."""


def lag_ref(var: str, lag: int) -> str:
    """``("y", 0) -> "y"``; ``("y", 2) -> "y.l2"``."""
    if lag < 0:
        raise DynamicsError(f"lag must be non-negative, got {lag} for {var!r}")
    return var if lag == 0 else f"{var}.l{lag}"


def time_ref(var: str, period: int) -> str:
    """``("y", 3) -> "y.t3"`` — the name of ``y`` at absolute period 3."""
    if period < 0:
        raise DynamicsError(f"period must be non-negative, got {period} for {var!r}")
    return f"{var}.t{period}"


def parse_ref(name: str) -> tuple[str, int]:
    """``"y.l2" -> ("y", 2)``; ``"y" -> ("y", 0)``. A lead raises."""
    if lead := _LEAD_RE.match(name):
        raise LeadNotSupportedError(
            f"reference {name!r} is a lead ({lead.group(1)} at t+{lead.group(2)}). "
            "A forward-looking equation needs a rational-expectations solution, which "
            "axiom.dynamics does not compute; write the expectation as its own "
            "exogenous variable if you have one."
        )
    if m := _LAG_RE.match(name):
        return m.group(1), int(m.group(2))
    if _TIME_RE.match(name):
        raise DynamicsError(
            f"reference {name!r} names an absolute period; an equation is written once "
            "for every period and may only use lags (y, y.l1, ...)"
        )
    if not _VAR_RE.match(name):
        raise DynamicsError(
            f"{name!r} is not a variable reference; expected 'y', 'y.l1', or 'y[t-1]'"
        )
    return name, 0


def parse_time_ref(name: str) -> tuple[str, int]:
    """``"y.t3" -> ("y", 3)``. Raises for anything that is not an absolute reference."""
    if m := _TIME_RE.match(name):
        return m.group(1), int(m.group(2))
    raise DynamicsError(f"{name!r} is not an absolute reference like 'y.t3'")


def refs_in(expr: Model) -> tuple[tuple[str, int], ...]:
    """Every ``(variable, lag)`` the expression reads, by first appearance."""
    out: list[tuple[str, int]] = []
    for _, node in walk(expr):
        if isinstance(node, Data):
            ref = parse_ref(node.name)
            if ref not in out:
                out.append(ref)
    return tuple(out)


class Variable(Spec):
    """One named quantity of the system, with its dimension and its role.

    ``role="endogenous"`` means an equation determines it; ``"exogenous"``
    means it is supplied (a treatment schedule, a covariate, a shock).
    ``observed`` says whether the panel measures it — an unobserved
    endogenous variable is a latent state, and identification has to route
    around it. ``initial`` is the value used for lags reaching before the
    first period; it is in the variable's own units.
    """

    name: NonEmptyStr
    dimension: Dimension
    role: Role = "endogenous"
    observed: bool = True
    initial: float = 0.0
    description: str = ""

    @field_validator("name")
    @classmethod
    def _plain(cls, v: str) -> str:
        if not _VAR_RE.match(v):
            raise ValueError(
                f"variable name {v!r} must be a plain identifier; '.' is reserved for "
                "time-shifted references (y.l1, y.t3)"
            )
        return v


class DynamicEquation(Spec):
    """``target[t] = rhs``, where ``rhs`` reads variables at ``t`` and earlier.

    The right-hand side is an ordinary expression tree. Its ``Data`` leaves
    are references: ``Data("x")`` is ``x`` at ``t``, ``Data("x.l1")`` is
    ``x`` at ``t-1``. Its ``Param`` leaves are the system's parameters, and
    they do not vary with ``t`` — a time-varying coefficient is a variable,
    not a parameter.
    """

    target: NonEmptyStr
    rhs: Expr
    name: str = ""

    @field_validator("target")
    @classmethod
    def _plain(cls, v: str) -> str:
        if not _VAR_RE.match(v):
            raise ValueError(f"equation target {v!r} must be a plain identifier")
        return v

    @property
    def refs(self) -> tuple[tuple[str, int], ...]:
        """Every ``(variable, lag)`` the right-hand side reads."""
        return refs_in(self.rhs)

    @property
    def label(self) -> str:
        return self.name or self.target


class DynamicSystem(Spec):
    """A set of variables and one equation per endogenous variable.

    Construction checks that the system is *well posed as a program*: every
    endogenous variable has exactly one equation, no exogenous variable has
    one, every reference resolves to a declared variable, every ``Data``
    node's dimension matches the variable it names, and each equation's two
    sides share a dimension. It does **not** check that the system is
    acyclic — a cycle is simultaneity, which is the point; see
    ``axiom.dynamics.block_order``.
    """

    name: str = ""
    variables: tuple[Variable, ...] = Field(min_length=1)
    equations: tuple[DynamicEquation, ...] = ()
    description: str = ""

    @model_validator(mode="after")
    def _closed(self) -> DynamicSystem:
        declared: dict[str, Variable] = {}
        for v in self.variables:
            if v.name in declared:
                raise ValueError(f"variable {v.name!r} is declared more than once")
            declared[v.name] = v
        by_target: dict[str, DynamicEquation] = {}
        for eq in self.equations:
            if eq.target in by_target:
                raise ValueError(f"variable {eq.target!r} has more than one equation")
            if eq.target not in declared:
                raise ValueError(f"equation for undeclared variable {eq.target!r}")
            if declared[eq.target].role != "endogenous":
                raise ValueError(
                    f"{eq.target!r} is exogenous and cannot have an equation; "
                    "declare it endogenous or drop the equation"
                )
            by_target[eq.target] = eq
        missing = [
            v.name for v in self.variables if v.role == "endogenous" and v.name not in by_target
        ]
        if missing:
            raise ValueError(f"endogenous variables with no equation: {sorted(missing)}")
        for eq in self.equations:
            for _, node in walk(eq.rhs):
                if not isinstance(node, Data):
                    continue
                var, _lag = parse_ref(node.name)
                if var not in declared:
                    raise ValueError(
                        f"equation {eq.label!r} reads undeclared variable {var!r} "
                        f"(reference {node.name!r})"
                    )
                if node.dimension != declared[var].dimension:
                    raise DimensionError(
                        f"equation {eq.label!r}: reference {node.name!r} is declared "
                        f"{node.dimension} here but {var!r} is {declared[var].dimension}"
                    )
            got = dimension(eq.rhs)
            want = declared[eq.target].dimension
            if got != want:
                raise DimensionError(
                    f"equation {eq.label!r}: right-hand side has dimension {got} but "
                    f"{eq.target!r} is {want}"
                )
        return self

    # -- lookup -------------------------------------------------------------------

    def variable(self, name: str) -> Variable:
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(f"no variable {name!r} in system {self.name!r}")

    def equation(self, target: str) -> DynamicEquation:
        for eq in self.equations:
            if eq.target == target:
                return eq
        raise KeyError(f"no equation for {target!r} in system {self.name!r}")

    @property
    def endogenous(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.variables if v.role == "endogenous")

    @property
    def exogenous(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.variables if v.role == "exogenous")

    @property
    def observed(self) -> tuple[str, ...]:
        return tuple(v.name for v in self.variables if v.observed)

    @property
    def max_lag(self) -> int:
        """The deepest lag any equation reads; ``0`` for a purely contemporaneous system."""
        return max((lag for eq in self.equations for _, lag in eq.refs), default=0)

    @property
    def parameters(self) -> tuple[Param, ...]:
        """Every distinct ``Param`` across the equations, by first appearance."""
        out: dict[str, Param] = {}
        for eq in self.equations:
            for p in params(eq.rhs):
                out.setdefault(p.name, p)
        return tuple(out.values())

    # -- structure ----------------------------------------------------------------

    def contemporaneous_edges(self) -> tuple[tuple[str, str], ...]:
        """``(u, v)`` when ``u`` at lag 0 appears in the equation for ``v``."""
        return tuple(
            (u, eq.target)
            for eq in self.equations
            for u, lag in eq.refs
            if lag == 0 and u != eq.target
        )

    def lagged_edges(self) -> tuple[tuple[str, str, int], ...]:
        """``(u, v, lag)`` for every ``lag >= 1`` reference, plus self-references at lag 0."""
        return tuple(
            (u, eq.target, lag)
            for eq in self.equations
            for u, lag in eq.refs
            if lag >= 1 or u == eq.target
        )

    def cycles_through(self) -> tuple[str, ...]:
        """Endogenous variables that lie on a contemporaneous cycle, sorted."""
        from axiom.dynamics.blocks import block_order

        return tuple(
            sorted(n for b in block_order(self).blocks if b.simultaneous for n in b.variables)
        )

    def iter_equations(self) -> Iterator[DynamicEquation]:
        yield from self.equations
