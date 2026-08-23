"""dynamics: systems that are simultaneous, or have time structure, or both.

A causal DAG is a solved model. Write down what you actually believe about a
market, a dose regimen, or an organism and you often get something else: two
quantities determined together within a period, and yesterday's value of a
quantity entering today's equation. Neither is ill-posed; both are cyclic as
a graph over variable *names*.

This subpackage takes such a system as a declarative ``Spec`` and *compiles*
it — into ordinary ``axiom.core.expr`` trees, and into a time-indexed graph
that is acyclic and can be handed straight to ``axiom.identify``.

    from axiom.core import D, Param, Prior
    from axiom.dynamics import Variable, parse_system, unroll

    system = parse_system(
        "stock = decay * stock[t-1] + inflow",
        variables=(
            Variable(name="stock", dimension=D.dose),
            Variable(name="inflow", dimension=D.dose, role="exogenous"),
        ),
        parameters=(Param(name="decay", dimension=D.dimensionless),),
        name="one-compartment",
    )
    compiled = unroll(system, periods=4)   # stock.t3 in terms of inflow.t0 ... inflow.t3

What comes out is an ``Expr``, so every interpreter, the design math and the
identifiability analysis work on it with no special case: there is still one
``forward()``.

Layer 1 — this subpackage imports ``axiom.core`` and nothing else from
axiom, so ``identify``, ``design`` and ``diagnose`` can all read a compiled
system.
"""

from axiom.dynamics.blocks import Block, BlockOrder, block_order, strongly_connected_components
from axiom.dynamics.parse import ParseError, parse_equations, parse_system
from axiom.dynamics.solve import (
    AffineForm,
    BlockSolution,
    SolveMethod,
    affine_split,
    reduced_form,
    solve_block,
)
from axiom.dynamics.spec import (
    DynamicEquation,
    DynamicsError,
    DynamicSystem,
    LeadNotSupportedError,
    Role,
    Variable,
    lag_ref,
    parse_ref,
    refs_in,
    time_ref,
)
from axiom.dynamics.unroll import (
    Form,
    Unrolled,
    conditional_form,
    lagged_columns,
    prepare_panel,
    to_model_spec,
    unroll,
    unrolled_edges,
)

__all__ = [
    "AffineForm",
    "Block",
    "BlockOrder",
    "BlockSolution",
    "DynamicEquation",
    "DynamicSystem",
    "DynamicsError",
    "Form",
    "LeadNotSupportedError",
    "ParseError",
    "Role",
    "SolveMethod",
    "Unrolled",
    "Variable",
    "affine_split",
    "block_order",
    "conditional_form",
    "lag_ref",
    "lagged_columns",
    "parse_equations",
    "parse_ref",
    "parse_system",
    "prepare_panel",
    "reduced_form",
    "refs_in",
    "solve_block",
    "strongly_connected_components",
    "time_ref",
    "to_model_spec",
    "unroll",
    "unrolled_edges",
]
