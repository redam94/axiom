"""Unrolling: from a system with cycles and time shifts to plain expression trees.

Two compilations, because there are two honest things a dynamic system can
mean, and conflating them is how dynamic models get fit wrong.

**Conditional form** (``conditional_form``) writes each endogenous variable
at ``t`` as a function of *observed* history: ``y = f(x, y.l1, ...)`` with
``y.l1`` a real column of the panel. One row per unit-period, the ordinary
dynamic regression. It is only meaningful when every lagged endogenous
variable it reads is measured — the compiler checks that, because
substituting an unobserved lag with data you do not have is the bug this
check exists to prevent.

**Marginal form** (``unroll``) writes the whole trajectory in terms of
exogenous inputs alone: ``y.t3 = f(x.t0, x.t1, x.t2, x.t3)``, obtained by
substituting the system into itself period by period. This is what a latent
state needs, what a policy simulation evaluates, and what the identification
and identifiability machinery reads, because only here does every path from
an input to an outcome appear explicitly.

The two agree numerically when the lagged columns are the model's own
fitted values and disagree otherwise; the difference is exactly the
difference between a one-step-ahead conditional mean and a marginal one.

Both routes leave ``axiom.core.expr`` untouched. What comes out is an
ordinary ``Expr``, which means ``value``, the jax and pytensor
interpreters, ``dimension``, ``latex``, ``linearize`` and the design math
all work on an unrolled dynamic system with no special case anywhere —
rule 3, kept.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import pandas as pd
from pydantic import Field, model_validator

from axiom.core import (
    Data,
    Dimension,
    Expr,
    Likelihood,
    ModelSpec,
    NonEmptyStr,
    Param,
    Prior,
    Spec,
    Unsupported,
    data_names,
    dimension,
    params,
)
from axiom.dynamics.algebra import const, map_data, size
from axiom.dynamics.blocks import BlockOrder, block_order
from axiom.dynamics.solve import MAX_NODES, BlockSolution, solve_block
from axiom.dynamics.spec import DynamicSystem, lag_ref, parse_ref, time_ref

__all__ = [
    "Form",
    "Unrolled",
    "conditional_form",
    "lagged_columns",
    "prepare_panel",
    "to_model_spec",
    "unroll",
    "unrolled_edges",
]

Form = Literal["conditional", "marginal"]


class Unrolled(Spec):
    """A compiled dynamic system: one expression per solved node, and how it got there.

    ``expressions`` is keyed by variable name in conditional form and by
    ``"y.t3"`` in marginal form. ``columns`` lists every data column the
    expressions read, which is exactly what a panel has to supply.
    ``solutions`` records one ``BlockSolution`` per block per period, so an
    approximate block cannot hide inside a large model: ``exact`` is false
    for the whole compilation if any of them is.
    """

    system: str = ""
    system_hash: str = ""
    form: Form
    periods: int = Field(ge=1)
    expressions: dict[str, Expr]
    solutions: tuple[BlockSolution, ...] = ()
    columns: tuple[str, ...] = ()
    edges: tuple[tuple[NonEmptyStr, NonEmptyStr], ...] = ()
    node_count: int = Field(ge=0)
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _consistent(self) -> Unrolled:
        if not self.expressions:
            raise ValueError("an unrolled system has at least one expression")
        if self.form == "conditional" and self.periods != 1:
            raise ValueError("the conditional form is written once, for one generic period")
        return self

    @property
    def exact(self) -> bool:
        """True when no block was compiled as a fixed-point approximation."""
        return all(s.exact for s in self.solutions)

    @property
    def approximate_blocks(self) -> tuple[BlockSolution, ...]:
        return tuple(s for s in self.solutions if not s.exact)

    def expression(self, variable: str, period: int | None = None) -> Expr:
        """The compiled expression for a variable, at a period in marginal form."""
        key = variable if self.form == "conditional" else time_ref(variable, period or 0)
        if key not in self.expressions:
            raise KeyError(f"no compiled expression for {key!r}; have {sorted(self.expressions)}")
        return self.expressions[key]


def _resolved_reference(
    system: DynamicSystem,
    node: Data,
    *,
    period: int | None,
    unknowns: frozenset[str],
    solved: Mapping[str, Expr],
) -> Expr:
    """One ``Data`` leaf of an equation, resolved against what is already known.

    ``period`` is ``None`` in conditional form (``t`` is generic) and the
    absolute period in marginal form.
    """
    var, lag = parse_ref(node.name)
    declared = system.variable(var)
    if period is None:
        if lag == 0:
            if var in unknowns:
                return Data(name=var, dimension=declared.dimension)
            if declared.role == "endogenous":
                return solved[var]
            return Data(name=var, dimension=declared.dimension)
        return Data(name=lag_ref(var, lag), dimension=declared.dimension)
    at = period - lag
    if at < 0:
        return const(declared.initial, declared.dimension)
    if lag == 0 and var in unknowns:
        return Data(name=var, dimension=declared.dimension)
    if declared.role == "exogenous":
        return Data(name=time_ref(var, at), dimension=declared.dimension)
    return solved[time_ref(var, at)]


def _compile_period(
    system: DynamicSystem,
    order: BlockOrder,
    *,
    period: int | None,
    solved: dict[str, Expr],
    sweeps: int,
    max_nodes: int,
) -> tuple[BlockSolution, ...] | Unsupported:
    """Solve every block of one period in order, writing results into ``solved``."""
    dims: dict[str, Dimension] = {v.name: v.dimension for v in system.variables}
    out: list[BlockSolution] = []
    for block in order.blocks:
        unknowns = frozenset(block.variables)

        def resolve(leaf: Data, unknowns: frozenset[str] = unknowns) -> Expr:
            return _resolved_reference(
                system, leaf, period=period, unknowns=unknowns, solved=solved
            )

        equations: dict[str, Expr] = {
            v: map_data(system.equation(v).rhs, resolve) for v in block.variables
        }
        start: dict[str, Expr] | None = None
        if block.simultaneous and sweeps > 0:
            # The declared initial value is also the fixed-point start: one declared
            # number, and no column invented behind the caller's back. A warm start
            # from the previous period would iterate less but multiply the tree by
            # its own size every period, and the residual would be no more honest.
            start = {v: const(system.variable(v).initial, dims[v]) for v in block.variables}
        solution = solve_block(
            equations,
            dims,
            simultaneous=block.simultaneous,
            start=start,
            sweeps=sweeps,
            max_nodes=max_nodes,
        )
        if isinstance(solution, Unsupported):
            where = "" if period is None else f" at period {period}"
            return Unsupported(
                reason=f"block {list(block.variables)}{where}: {solution.reason}",
                detail=solution.detail,
                missing=solution.missing,
            )
        for v in block.variables:
            key = v if period is None else time_ref(v, period)
            solved[key] = solution.expressions[v]
        out.append(solution)
    return tuple(out)


def lagged_columns(system: DynamicSystem) -> tuple[str, ...]:
    """Every lagged reference the conditional form needs as a materialized column."""
    seen: list[str] = []
    for eq in system.equations:
        for var, lag in eq.refs:
            if lag >= 1:
                name = lag_ref(var, lag)
                if name not in seen:
                    seen.append(name)
    return tuple(seen)


def conditional_form(
    system: DynamicSystem, *, sweeps: int = 0, max_nodes: int = MAX_NODES
) -> Unrolled | Unsupported:
    """Compile one generic period: ``y = f(exogenous at t, every lag as a column)``.

    Every lagged endogenous variable becomes a data column, so each must be
    observed; one that is not comes back ``Unsupported`` naming it, with the
    marginal form as the route that does not need it.
    """
    unobserved = sorted(
        {
            var
            for eq in system.equations
            for var, lag in eq.refs
            if lag >= 1
            and system.variable(var).role == "endogenous"
            and not system.variable(var).observed
        }
    )
    if unobserved:
        return Unsupported(
            reason=(
                f"the conditional form reads {unobserved} at a lag, but they are latent: "
                "there is no column to put there. Use unroll() for the marginal form, "
                "which substitutes the latent history away"
            ),
            missing=tuple(unobserved),
            detail={"latent": ", ".join(unobserved)},
        )
    order = block_order(system)
    solved: dict[str, Expr] = {}
    solutions = _compile_period(
        system, order, period=None, solved=solved, sweeps=sweeps, max_nodes=max_nodes
    )
    if isinstance(solutions, Unsupported):
        return solutions
    columns: list[str] = []
    for expr in solved.values():
        for name in data_names(expr):
            if name not in columns:
                columns.append(name)
    return Unrolled(
        system=system.name,
        system_hash=system.content_hash(),
        form="conditional",
        periods=1,
        expressions=solved,
        solutions=solutions,
        columns=tuple(columns),
        edges=unrolled_edges(system, periods=1 + system.max_lag),
        node_count=sum(size(e) for e in solved.values()),
        detail={"max_lag": str(system.max_lag), "blocks": str(len(order.blocks))},
    )


def unroll(
    system: DynamicSystem, periods: int, *, sweeps: int = 0, max_nodes: int = MAX_NODES
) -> Unrolled | Unsupported:
    """Compile every period into expressions over exogenous inputs alone.

    ``periods`` is the horizon: nodes ``var.t0 ... var.t{periods-1}`` are
    produced. References reaching before period 0 take the variable's
    declared ``initial``. The node budget is checked as the horizon grows,
    so a nonlinear system that would explode says so instead of hanging.
    """
    if periods < 1:
        raise ValueError(f"periods must be at least 1, got {periods}")
    order = block_order(system)
    solved: dict[str, Expr] = {}
    solutions: list[BlockSolution] = []
    for t in range(periods):
        got = _compile_period(
            system, order, period=t, solved=solved, sweeps=sweeps, max_nodes=max_nodes
        )
        if isinstance(got, Unsupported):
            return got
        solutions.extend(got)
        total = sum(size(e) for e in solved.values())
        if total > max_nodes:
            return Unsupported(
                reason=(
                    f"unrolling {system.name or 'the system'} to {t + 1} of {periods} periods "
                    f"passed the {max_nodes}-node budget ({total} nodes). A nonlinear system "
                    "grows with the horizon; unroll fewer periods, or fit the conditional form"
                ),
                detail={"periods_compiled": str(t + 1), "nodes": str(total)},
            )
    columns: list[str] = []
    for expr in solved.values():
        for name in data_names(expr):
            if name not in columns:
                columns.append(name)
    return Unrolled(
        system=system.name,
        system_hash=system.content_hash(),
        form="marginal",
        periods=periods,
        expressions=solved,
        solutions=tuple(solutions),
        columns=tuple(columns),
        edges=unrolled_edges(system, periods=periods),
        node_count=sum(size(e) for e in solved.values()),
        detail={"blocks": str(len(order.blocks)), "max_lag": str(system.max_lag)},
    )


def unrolled_edges(
    system: DynamicSystem, periods: int, *, reduced: bool = True
) -> tuple[tuple[str, str], ...]:
    """Edges of the time-indexed graph, ``("x.t1", "y.t2")``, sorted and deduplicated.

    With ``reduced=True`` (the default) the members of a simultaneous block
    share the block's parents and have no arrows between them: that is the
    reduced form, and it is a DAG. With ``reduced=False`` the structural
    edges are emitted as written, cycles included — useful to *show* the
    simultaneity, useless as a causal graph, since no DAG algorithm accepts
    it.
    """
    order = block_order(system)
    edges: set[tuple[str, str]] = set()
    for t in range(periods):
        for block in order.blocks:
            members = set(block.variables)
            parents: set[tuple[str, int]] = set()
            for v in block.variables:
                for u, lag in system.equation(v).refs:
                    if not reduced:
                        if t - lag >= 0:
                            edges.add((time_ref(u, t - lag), time_ref(v, t)))
                        continue
                    if lag == 0 and u in members:
                        continue
                    if t - lag >= 0:
                        parents.add((u, t - lag))
            if not reduced:
                continue
            for u, s in parents:
                for v in block.variables:
                    edges.add((time_ref(u, s), time_ref(v, t)))
    return tuple(sorted(edges))


def prepare_panel(
    system: DynamicSystem,
    frame: pd.DataFrame,
    *,
    unit: str | None = None,
    time: str,
) -> pd.DataFrame:
    """Add the lagged columns the conditional form reads, per unit, in time order.

    Rows are sorted by ``(unit, time)``; a lag reaching before a unit's
    first period takes the variable's declared ``initial``. The frame comes
    back with the original columns plus one per entry of
    ``lagged_columns(system)``; nothing is dropped, so the caller can see
    which rows the initial condition touched.
    """
    needed = lagged_columns(system)
    missing = sorted(
        {parse_ref(c)[0] for c in needed} - set(frame.columns)
        | ({time} - set(frame.columns))
        | (({unit} - set(frame.columns)) if unit else set())
    )
    if missing:
        raise KeyError(f"panel is missing columns the system reads: {missing}")
    keys = [unit, time] if unit else [time]
    out = frame.sort_values(keys).copy()
    grouped = out.groupby(unit, sort=False) if unit else None
    for column in needed:
        var, lag = parse_ref(column)
        fill = system.variable(var).initial
        shifted = grouped[var].shift(lag) if grouped is not None else out[var].shift(lag)
        out[column] = shifted.fillna(fill)
    return out


def to_model_spec(
    unrolled: Unrolled,
    outcome: str,
    *,
    likelihood: Likelihood,
    priors: Mapping[str, Prior],
    extra_parameters: Sequence[Param] = (),
    period: int | None = None,
    name: str = "",
) -> ModelSpec:
    """Wrap one compiled node as a ``ModelSpec`` that any backend fits.

    ``priors`` supplies a prior for every parameter in the chosen
    expression; ``extra_parameters`` carries the ones that appear only in
    the likelihood (a scale). The outcome column is the variable's own name
    in conditional form and ``"y.t3"`` in marginal form — the same name the
    panel has to carry.
    """
    mean = unrolled.expression(outcome, period)
    key = outcome if unrolled.form == "conditional" else time_ref(outcome, period or 0)
    declared: list[Param] = []
    for p in params(mean):
        prior = priors.get(p.name, p.prior)
        if prior is None:
            raise KeyError(f"no prior for parameter {p.name!r}; pass one in priors=")
        declared.append(p.model_copy(update={"prior": prior}))
    seen = {p.name for p in declared}
    for p in extra_parameters:
        if p.name in seen:
            continue
        if p.prior is None:
            prior = priors.get(p.name)
            if prior is None:
                raise KeyError(f"no prior for parameter {p.name!r}; pass one in priors=")
            p = p.model_copy(update={"prior": prior})
        declared.append(p)
        seen.add(p.name)
    return ModelSpec(
        name=name or f"{unrolled.system or 'system'}:{key}",
        mean=mean,
        outcome=Data(name=key, dimension=dimension(mean)),
        likelihood=likelihood,
        parameters=tuple(declared),
    )
