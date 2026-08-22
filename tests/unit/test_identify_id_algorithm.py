"""ID and IDC: the estimand when there is one, the hedge when there is not."""

from __future__ import annotations

import pytest

from axiom.identify import (
    CausalGraph,
    Density,
    Hedge,
    IdentifiedEffect,
    admissible_set_exists,
    districts,
    identify,
    identify_conditional_effect,
    identify_effect,
    latent_projection,
)
from axiom.identify.formula import Marginal, Product, Ratio, marginal, product, ratio, to_latex
from axiom.identify.formula import to_text as formula_text
from axiom.identify.formula import variables as formula_variables
from axiom.identify.graph import GraphError
from axiom.identify.id_algorithm import induced_subgraph

BACKDOOR = "Z -> X, Z -> Y, X -> Y"
FRONTDOOR = "X -> M, M -> Y, U -> X, U -> Y"
BOW = "X -> Y, U -> X, U -> Y"
NAPKIN = "W1 -> W2, W2 -> X, X -> Y, U1 -> W1, U1 -> X, U2 -> W1, U2 -> Y"


# -- graph machinery -------------------------------------------------------------------


def test_districts_are_the_bidirected_components() -> None:
    graph = CausalGraph.from_edges("A -> B, A <-> B, C -> D, D <-> E")
    assert districts(graph) == (("A", "B"), ("C",), ("D", "E"))


def test_latent_projection_turns_a_hidden_cause_into_a_bidirected_edge() -> None:
    graph = CausalGraph.from_edges(FRONTDOOR, unmeasured=["U"])
    admg = latent_projection(graph)
    assert set(admg.nodes) == {"M", "X", "Y"}
    assert ("X", "Y") in admg.bidirected
    assert ("X", "M") in admg.edges


def test_latent_projection_passes_a_hidden_mediator_through() -> None:
    graph = CausalGraph.from_edges("X -> U, U -> Y", unmeasured=["U"])
    admg = latent_projection(graph)
    assert admg.edges == (("X", "Y"),)
    assert admg.bidirected == ()


def test_a_bidirected_edge_and_an_explicit_latent_give_the_same_answer() -> None:
    explicit = identify_effect(CausalGraph.from_edges(FRONTDOOR, unmeasured=["U"]), "X", "Y")
    projected = identify_effect(CausalGraph.from_edges("X -> M, M -> Y, X <-> Y"), "X", "Y")
    assert explicit.formula == projected.formula


def test_induced_subgraph_keeps_only_interior_edges() -> None:
    graph = CausalGraph.from_edges("A -> B, B -> C, A <-> C")
    sub = induced_subgraph(graph, ["A", "B"])
    assert sub.edges == (("A", "B"),)
    assert sub.bidirected == ()
    with pytest.raises(GraphError, match="unknown nodes"):
        induced_subgraph(graph, ["Q"])


# -- identification --------------------------------------------------------------------


def test_the_back_door_case_returns_the_adjustment_formula() -> None:
    result = identify_effect(CausalGraph.from_edges(BACKDOOR), "X", "Y")
    assert isinstance(result, IdentifiedEffect)
    assert result.identified
    assert result.to_text() == "sum_{Z} [P(Y | X, Z) P(Z)]"
    assert result.verdict.route == "id_algorithm"


def test_the_front_door_case_returns_the_front_door_formula() -> None:
    result = identify_effect(CausalGraph.from_edges(FRONTDOOR, unmeasured=["U"]), "X", "Y")
    assert result.identified
    assert result.to_text() == "sum_{M} [P(M | X) [sum_{X} [P(X) P(Y | M, X)]]]"


def test_the_bow_arc_is_a_hedge_and_the_hedge_is_reported() -> None:
    result = identify_effect(CausalGraph.from_edges(BOW, unmeasured=["U"]), "X", "Y")
    assert not result.identified
    assert isinstance(result.hedge, Hedge)
    assert result.verdict.status == "blocked"
    assert "hedge" in result.to_text()
    assert set(result.hedge.root) == {"X", "Y"}


def test_an_instrument_alone_does_not_point_identify_the_effect() -> None:
    result = identify_effect(
        CausalGraph.from_edges("Z -> X, X -> Y, U -> X, U -> Y", unmeasured=["U"]), "X", "Y"
    )
    assert not result.identified


def test_the_napkin_graph_has_no_back_door_yet_is_identified() -> None:
    graph = CausalGraph.from_edges(NAPKIN, unmeasured=["U1", "U2"])
    assert not admissible_set_exists(graph, "X", "Y", measured_only=True)

    result = identify_effect(graph, "X", "Y")
    assert result.identified
    assert "W1" in result.to_text()
    # and the estimand uses only measured variables, which is the whole point
    assert set(formula_variables(result.formula)) <= set(graph.measured)


def test_id_identifies_where_the_route_menu_refuses() -> None:
    """The case that makes a *complete* algorithm worth having.

    ``identify`` searches back-door, front-door and instrument and returns
    ``blocked`` here, naming all three. That is an honest report of a failed
    search and it is not a proof: ID returns an estimand for the same graph.
    A menu cannot tell "no route I know" from "no route exists"; ID can.
    """
    graph = CausalGraph.from_edges("M -> Y, W -> M, W -> X, W -> Y, X -> M, M <-> W, X <-> Y")
    routes = identify(graph, "X", "Y")
    assert not routes.licensed
    assert "no back-door, front-door, or instrumental route" in routes.verdict.reason

    result = identify_effect(graph, "X", "Y")
    assert result.identified
    assert result.to_text() == "sum_{M, W} [P(W) P(M | W, X) [sum_{X} [P(X | W) P(Y | M, W, X)]]]"


def test_identification_is_downgraded_not_certified() -> None:
    result = identify_effect(CausalGraph.from_edges(BACKDOOR), "X", "Y")
    assert result.verdict.status == "downgraded"
    assert {a.name for a in result.verdict.assumptions} == {"graph_is_correct", "positivity"}


def test_a_multi_variable_treatment_and_outcome_work() -> None:
    graph = CausalGraph.from_edges("X1 -> Y1, X2 -> Y2, Z -> X1, Z -> Y1")
    result = identify_effect(graph, ["X1", "X2"], ["Y1", "Y2"])
    assert result.identified
    assert result.treatment == ("X1", "X2")


def test_malformed_queries_are_refused_by_name() -> None:
    graph = CausalGraph.from_edges(BACKDOOR)
    with pytest.raises(GraphError, match="both treatment and outcome"):
        identify_effect(graph, "X", "X")
    with pytest.raises(GraphError, match="unknown node"):
        identify_effect(graph, "X", "Q")
    with pytest.raises(ValueError, match="at least one treatment"):
        identify_effect(graph, [], "Y")
    hidden = CausalGraph.from_edges("U -> X, U -> Y, X -> Y", unmeasured=["U"])
    with pytest.raises(GraphError, match="must be measured"):
        identify_effect(hidden, "U", "Y")


# -- IDC -------------------------------------------------------------------------------


def test_a_conditional_effect_with_no_conditioning_is_the_unconditional_one() -> None:
    graph = CausalGraph.from_edges(BACKDOOR)
    assert identify_conditional_effect(graph, "X", "Y", []).formula == (
        identify_effect(graph, "X", "Y").formula
    )


def test_a_conditional_effect_is_identified_and_reports_the_query_asked() -> None:
    """IDC may rewrite the query; the report says what was asked, not what it became."""
    graph = CausalGraph.from_edges("Z -> X, Z -> Y, X -> Y, X -> W, W -> Y")
    result = identify_conditional_effect(graph, "X", "Y", ["W"])
    assert result.identified
    assert result.conditioned == ("W",)
    assert result.verdict.route == "idc_algorithm"
    # W behaves like an intervention here, so IDC moved it into the do
    assert result.detail["moved_into_do"] == "W"


def test_a_conditioning_variable_cannot_be_the_treatment() -> None:
    graph = CausalGraph.from_edges(BACKDOOR)
    with pytest.raises(GraphError, match="cannot also be treatment"):
        identify_conditional_effect(graph, "X", "Y", ["X"])


def test_a_conditional_effect_over_a_hedge_is_blocked() -> None:
    graph = CausalGraph.from_edges("X -> Y, X -> W, U -> X, U -> Y", unmeasured=["U"])
    result = identify_conditional_effect(graph, "X", "Y", ["W"])
    assert not result.identified
    assert result.verdict.route == "idc_algorithm"


# -- the formula algebra ---------------------------------------------------------------


def test_a_sum_over_nothing_relevant_disappears() -> None:
    density = Density(outcomes=("Y",), given=("X",))
    assert marginal(["Q"], density) == density


def test_a_sum_of_a_joint_over_its_own_variables_is_the_smaller_joint() -> None:
    assert marginal(["X"], Density(outcomes=("X", "Y"))) == Density(outcomes=("Y",))


def test_a_ratio_of_joints_is_a_conditional() -> None:
    got = ratio(Density(outcomes=("X", "Y")), Density(outcomes=("Y",)))
    assert got == Density(outcomes=("X",), given=("Y",))


def test_factors_that_do_not_mention_the_summed_variable_are_lifted_out() -> None:
    inside = Density(outcomes=("Y",), given=("Z",))
    outside = Density(outcomes=("W",))
    got = marginal(["Z"], product(inside, Density(outcomes=("Z",)), outside))
    assert isinstance(got, Product)
    assert outside in got.factors
    assert any(isinstance(f, Marginal) for f in got.factors)


def test_nested_sums_merge() -> None:
    got = marginal(["A"], marginal(["B"], Density(outcomes=("A", "B", "C"), given=("D",))))
    assert isinstance(got, Marginal)
    assert got.over == ("A", "B")


def test_a_ratio_of_identical_formulas_is_refused() -> None:
    density = Density(outcomes=("Y",))
    with pytest.raises(ValueError, match="constant one"):
        ratio(density, density)


def test_a_density_cannot_condition_on_its_own_outcome() -> None:
    with pytest.raises(ValueError, match="its own outcome"):
        Density(outcomes=("Y",), given=("Y",))


def test_formulas_render_as_text_and_latex() -> None:
    formula = ratio(Density(outcomes=("Y", "X")), Density(outcomes=("X",)))
    assert formula_text(formula) == "P(Y | X)"
    nested = marginal(
        ["Z"], product(Density(outcomes=("Y",), given=("Z",)), Density(outcomes=("Z",)))
    )
    assert "\\sum" in to_latex(nested)
    quotient = Ratio(numerator=Density(outcomes=("A",)), denominator=Density(outcomes=("B",)))
    assert "\\frac" in to_latex(quotient)
