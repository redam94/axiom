"""Unrolled graphs and the sequential back-door criterion."""

from __future__ import annotations

import pytest

from axiom.core import D, Param, dimensionless
from axiom.dynamics import Variable, parse_system
from axiom.identify import (
    CausalGraph,
    SequentialPlan,
    identify,
    sequential_backdoor_admissible,
    sequential_plan,
    unrolled_graph,
    unrolled_mixed_graph,
)

NONE = dimensionless()


def feedback_system(persistent_confounder: bool = False, observed: bool = False):
    """A dose that responds to the last outcome, with a confounder of the outcome."""
    variables = [
        Variable(name="outcome", dimension=D.outcome),
        Variable(name="dose", dimension=D.currency),
        Variable(name="protocol", dimension=D.currency, role="exogenous"),
    ]
    parameters = [
        Param(name="beta", dimension=D.outcome / D.currency),
        Param(name="rho", dimension=NONE),
        Param(name="kappa", dimension=D.outcome),
        Param(name="phi", dimension=D.currency / D.outcome),
    ]
    if persistent_confounder:
        # frailty drives the dose *and* the outcome, and carries over: the classic
        # unmeasured time-varying confounder, which no adjustment set can close.
        variables.append(Variable(name="frailty", dimension=NONE, observed=observed))
        parameters.append(Param(name="persist", dimension=NONE))
        parameters.append(Param(name="omega", dimension=D.currency))
        text = """
    outcome = beta * dose + rho * outcome[t-1] + kappa * frailty
    dose    = phi * outcome[t-1] + protocol + omega * frailty
    frailty = persist * frailty[t-1]
    """
    else:
        variables.append(
            Variable(name="frailty", dimension=NONE, role="exogenous", observed=observed)
        )
        text = """
    outcome = beta * dose + rho * outcome[t-1] + kappa * frailty
    dose    = phi * outcome[t-1] + protocol
    """
    return parse_system(
        text, variables=tuple(variables), parameters=tuple(parameters), name="feedback"
    )


def market_system():
    return parse_system(
        """
        quantity = a - b * price
        price    = d + e * quantity + cost
        """,
        variables=(
            Variable(name="quantity", dimension=D.outcome),
            Variable(name="price", dimension=D.currency),
            Variable(name="cost", dimension=D.currency, role="exogenous"),
        ),
        parameters=(
            Param(name="a", dimension=D.outcome),
            Param(name="b", dimension=D.outcome / D.currency),
            Param(name="d", dimension=D.currency),
            Param(name="e", dimension=D.currency / D.outcome),
        ),
        name="market",
    )


# -- the graph ----------------------------------------------------------------------


def test_the_unrolled_graph_is_a_dag_with_one_node_per_variable_and_period() -> None:
    graph = unrolled_graph(feedback_system(), periods=3)
    assert len(graph.nodes) == 4 * 3
    assert ("dose.t1", "outcome.t1") in graph.edges
    assert ("outcome.t0", "dose.t1") in graph.edges
    assert graph.topological_order()  # acyclic


def test_an_unobserved_variable_is_unmeasured_at_every_period() -> None:
    graph = unrolled_graph(feedback_system(), periods=2)
    assert set(graph.unmeasured) == {"frailty.t0", "frailty.t1"}


def test_the_reduced_form_of_a_cycle_is_a_dag_and_the_structural_one_is_not() -> None:
    reduced = unrolled_graph(market_system(), periods=2)
    assert reduced.topological_order()
    assert ("cost.t0", "quantity.t0") in reduced.edges
    # no arrow between the simultaneous variables -- a bidirected edge instead,
    # because they are determined together and share the block's disturbances
    assert ("price.t0", "quantity.t0") not in reduced.edges
    assert ("price.t0", "quantity.t0") in reduced.bidirected

    structural = unrolled_mixed_graph(market_system(), periods=2)
    assert not structural.is_acyclic
    assert ("quantity.t0", "price.t0") in structural.edges


def test_zero_periods_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        unrolled_graph(feedback_system(), periods=0)


# -- the criterion ------------------------------------------------------------------


def test_the_g_formula_adjustment_is_the_history_up_to_each_stage() -> None:
    graph = unrolled_graph(feedback_system(), periods=3)
    plan = sequential_plan(graph, ["dose.t0", "dose.t1", "dose.t2"], "outcome.t2")
    assert isinstance(plan, SequentialPlan)
    assert plan.licensed
    assert plan.verdict.status == "downgraded"
    assert plan.adjustments == ((), ("outcome.t0",), ("outcome.t1",))
    assert plan.conditioning_at[2] == ("dose.t0", "dose.t1", "outcome.t0", "outcome.t1")


def test_positivity_rides_along_unverified_so_the_verdict_is_a_downgrade() -> None:
    graph = unrolled_graph(feedback_system(), periods=3)
    plan = sequential_plan(graph, ["dose.t0", "dose.t1"], "outcome.t2")
    states = {a.name: a.state for a in plan.verdict.assumptions}
    assert states["positivity"] == "unverified"
    assert states["sequential_exchangeability"] == "satisfied"


def test_time_varying_confounding_is_named_as_such() -> None:
    graph = unrolled_graph(feedback_system(), periods=3)
    plan = sequential_plan(graph, ["dose.t0", "dose.t1", "dose.t2"], "outcome.t2")
    assert plan.static_adjustment_fails


def test_a_persistent_unmeasured_confounder_blocks_the_plan() -> None:
    graph = unrolled_graph(feedback_system(persistent_confounder=True), periods=3)
    plan = sequential_plan(graph, ["dose.t0", "dose.t1", "dose.t2"], "outcome.t2")
    assert not plan.licensed
    assert plan.verdict.status == "blocked"
    assert "stage" in plan.verdict.reason


def test_measuring_the_persistent_confounder_unblocks_it() -> None:
    graph = unrolled_graph(feedback_system(persistent_confounder=True, observed=True), periods=3)
    plan = sequential_plan(graph, ["dose.t0", "dose.t1", "dose.t2"], "outcome.t2")
    assert plan.licensed
    assert any("frailty" in name for stage in plan.adjustments for name in stage)


def test_conditioning_on_a_descendant_of_a_later_treatment_is_refused() -> None:
    graph = unrolled_graph(feedback_system(), periods=3)
    ok, why = sequential_backdoor_admissible(
        graph, ["dose.t0", "dose.t1"], "outcome.t2", [("outcome.t1",), ()]
    )
    assert not ok
    assert "descendants" in why


def test_the_criterion_needs_one_set_per_stage() -> None:
    graph = unrolled_graph(feedback_system(), periods=2)
    with pytest.raises(ValueError, match="adjustment sets"):
        sequential_backdoor_admissible(graph, ["dose.t0", "dose.t1"], "outcome.t1", [()])


def test_a_plan_needs_at_least_one_stage() -> None:
    graph = unrolled_graph(feedback_system(), periods=2)
    with pytest.raises(ValueError, match="at least one treatment"):
        sequential_plan(graph, [], "outcome.t1")


def test_a_single_stage_agrees_with_the_static_back_door() -> None:
    graph = CausalGraph.from_edges("z -> x, z -> y, x -> y")
    plan = sequential_plan(graph, ["x"], "y")
    static = identify(graph, "x", "y")
    assert plan.licensed
    assert plan.adjustments == (("z",),)
    assert static.adjustment_set == ("z",)
