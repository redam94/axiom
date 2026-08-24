"""SWIGs: the potential outcome on the graph, and the theorem that it matches the back-door."""

from __future__ import annotations

import itertools
import random

import pytest

from axiom.core import D, Param, dimensionless
from axiom.dynamics import Variable, parse_system
from axiom.identify import (
    CausalGraph,
    backdoor_admissible,
    sequential_ignorability,
    sequential_plan,
    single_world_ignorability,
    swig,
    unrolled_graph,
)
from axiom.identify.graph import GraphError
from axiom.identify.swig import labels_of, potential_outcome, renamed_in

NONE = dimensionless()
NAMES = ("A", "B", "C", "D", "E")


def _random_dag(rng: random.Random) -> CausalGraph:
    order = list(NAMES)
    rng.shuffle(order)
    edges = [(u, v) for i, u in enumerate(order) for v in order[i + 1 :] if rng.random() < 0.35]
    return CausalGraph(nodes=NAMES, edges=tuple(edges))


# -- construction ----------------------------------------------------------------------


def test_the_split_puts_the_fixed_half_upstream_and_relabels_descendants() -> None:
    graph = CausalGraph.from_edges("Z -> X, Z -> Y, X -> M, M -> Y")
    split = swig(graph, ["X"])
    assert split.to_text() == "M(x) -> Y(x), Z -> X, Z -> Y(x), x -> M(x)"
    # the random half keeps its parents and loses its children
    assert split.parents("X") == frozenset({"Z"})
    assert split.children("X") == frozenset()
    assert split.children("x") == frozenset({"M(x)"})


def test_a_node_untouched_by_the_intervention_keeps_its_name() -> None:
    graph = CausalGraph.from_edges("Z -> X, X -> Y, W -> Z")
    split = swig(graph, ["X"])
    assert "W" in split.nodes and "Z" in split.nodes
    assert potential_outcome("Y", ["x"]) in split.nodes


def test_labels_are_parsed_back_out_of_a_potential_outcome() -> None:
    assert labels_of("Y(x)") == ("x",)
    assert labels_of("Y(a, b)") == ("a", "b")
    assert labels_of("Y") == ()


def test_a_lower_case_label_that_would_collide_gets_a_star() -> None:
    graph = CausalGraph.from_edges("dose -> outcome")
    split = swig(graph, ["dose"])
    assert "dose*" in split.nodes
    assert "outcome(dose*)" in split.nodes


def test_explicit_labels_are_honoured_and_collisions_refused() -> None:
    graph = CausalGraph.from_edges("X -> Y")
    assert "x0" in swig(graph, {"X": "x0"}).nodes
    with pytest.raises(GraphError, match="collide"):
        swig(graph, {"X": "Y"})
    with pytest.raises(ValueError, match="at least one intervention"):
        swig(graph, [])


def test_finding_a_node_in_a_swig_survives_relabelling() -> None:
    split = swig(CausalGraph.from_edges("X -> Y"), ["X"])
    assert renamed_in(split, "Y") == "Y(x)"
    assert renamed_in(split, "X") == "X"
    with pytest.raises(GraphError, match="no node"):
        renamed_in(split, "Q")


# -- the theorem -----------------------------------------------------------------------


def test_single_world_ignorability_is_the_back_door_criterion() -> None:
    """``Y(x) indep X | Z`` on the SWIG holds exactly when Z is back-door admissible.

    Two derivations of the same condition — one counterfactual, one purely
    graphical — held against each other over every adjustment set of every
    random graph in the sample.
    """
    rng = random.Random(1)
    checked = 0
    for _ in range(120):
        graph = _random_dag(rng)
        for x, y in itertools.permutations(NAMES, 2):
            if not graph.is_ancestor(x, y):
                continue
            rest = [n for n in NAMES if n not in (x, y)]
            for size in range(len(rest) + 1):
                for z in itertools.combinations(rest, size):
                    assert single_world_ignorability(graph, x, y, z) == backdoor_admissible(
                        graph, x, y, z
                    ), f"{graph.to_text()} | {x} {y} {z}"
                    checked += 1
    assert checked > 2000


def test_conditioning_on_a_potential_outcome_is_refused() -> None:
    """A relabelled node is a variable in the hypothetical world; data cannot condition on it."""
    graph = CausalGraph.from_edges("X -> M, M -> Y")
    assert not single_world_ignorability(graph, "X", "Y", ["M"])
    assert not backdoor_admissible(graph, "X", "Y", ["M"])


# -- sequential regimes ----------------------------------------------------------------


def feedback_system():
    return parse_system(
        """
        outcome = beta * dose + rho * outcome[t-1] + kappa * frailty
        dose    = phi * outcome[t-1] + protocol
        """,
        variables=(
            Variable(name="outcome", dimension=D.outcome),
            Variable(name="dose", dimension=D.currency),
            Variable(name="protocol", dimension=D.currency, role="exogenous"),
            Variable(name="frailty", dimension=NONE, role="exogenous", observed=False),
        ),
        parameters=(
            Param(name="beta", dimension=D.outcome / D.currency),
            Param(name="rho", dimension=NONE),
            Param(name="kappa", dimension=D.outcome),
            Param(name="phi", dimension=D.currency / D.outcome),
        ),
        name="feedback",
    )


def test_the_sequential_plan_is_sequentially_ignorable_on_the_swig() -> None:
    """The g-formula check and the potential-outcome check, independently derived, agree."""
    graph = unrolled_graph(feedback_system(), periods=3)
    stages = ["dose.t0", "dose.t1", "dose.t2"]
    plan = sequential_plan(graph, stages, "outcome.t2")
    assert plan.licensed
    ok, why = sequential_ignorability(graph, stages, "outcome.t2", plan.adjustments)
    assert ok, why


def test_conditioning_early_on_a_later_stages_potential_outcome_is_refused() -> None:
    graph = unrolled_graph(feedback_system(), periods=3)
    stages = ["dose.t0", "dose.t1", "dose.t2"]
    ok, why = sequential_ignorability(graph, stages, "outcome.t2", [("outcome.t1",), (), ()])
    assert not ok
    assert "a world the data did not follow" in why


def test_an_unadjusted_plan_is_rejected() -> None:
    graph = unrolled_graph(feedback_system(), periods=3)
    stages = ["dose.t0", "dose.t1", "dose.t2"]
    ok, why = sequential_ignorability(graph, stages, "outcome.t2", [(), (), ()])
    assert not ok
    assert "not independent" in why


def test_the_stage_count_must_match() -> None:
    graph = unrolled_graph(feedback_system(), periods=2)
    with pytest.raises(ValueError, match="adjustment sets"):
        sequential_ignorability(graph, ["dose.t0", "dose.t1"], "outcome.t1", [()])
