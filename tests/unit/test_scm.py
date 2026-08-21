from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from pydantic import ValidationError

from axiom.core import Spec, load_spec
from axiom.identify.graph import CausalGraph, GraphError
from axiom.sim.scm import LinearSCM, SCMError, coefficient_key, latent_key
from axiom.sim.worlds import (
    confounded_world,
    feedback_world,
    frontdoor_world,
    hidden_confounder_world,
    iv_world,
    mediator_world,
    transport_pair,
)

WORLDS = [
    confounded_world,
    hidden_confounder_world,
    iv_world,
    frontdoor_world,
    mediator_world,
    feedback_world,
]


# -- construction and parsing -------------------------------------------------------


def test_from_text_parses_coefficients_latents_and_reversed_arrows() -> None:
    scm = LinearSCM.from_text(
        "Z -> X: 0.8, X -> Y: 2.0; Y <- Z: 1.5\nX <-> Y: 0.7", unmeasured=["Z"], name="w"
    )
    assert scm.nodes == ("X", "Y", "Z")
    assert scm.edges == (("X", "Y"), ("Z", "X"), ("Z", "Y"))
    assert scm.bidirected == (("X", "Y"),)
    assert scm.coefficients == {"X->Y": 2.0, "Z->X": 0.8, "Z->Y": 1.5}
    assert scm.latent_sd == {"X<->Y": 0.7}
    assert scm.unmeasured == ("Z",)
    assert scm.name == "w"
    assert LinearSCM.from_text(scm.to_text(), unmeasured=["Z"], name="w") == scm


def test_from_text_builds_a_causal_graph() -> None:
    scm = LinearSCM.from_text(
        "Z -> X: 0.8, X -> Y: 2.0, X <-> Y: 0.7", unmeasured=["Z"], selection=["Z"], name="w"
    )
    assert isinstance(scm.graph, CausalGraph)
    assert scm.graph == CausalGraph.from_edges(
        "Z -> X, X -> Y, X <-> Y", unmeasured=["Z"], selection=["Z"], name="w"
    )
    assert scm.graph.edges == scm.edges == (("X", "Y"), ("Z", "X"))
    assert scm.graph.bidirected == scm.bidirected == (("X", "Y"),)
    assert scm.graph.selection == scm.selection == ("Z",)
    assert scm.graph.feedback is scm.feedback is False
    assert scm.topological_order() == scm.graph.topological_order() == ("Z", "X", "Y")
    assert scm.directed_paths("Z", "Y") == scm.graph.directed_paths("Z", "Y") == (("Z", "X", "Y"),)
    assert scm.parents("Y") == scm.graph.parents("Y") == frozenset({"X"})
    assert scm.children("Z") == scm.graph.children("Z") == frozenset({"X"})
    # The graph is the identification machinery's object, not a copy of its fields.
    assert scm.graph.d_separated("Z", "Y", ["X"]) is False
    assert scm.graph.d_separated("Z", "Y", ["X"]) == CausalGraph.from_edges(
        "Z -> X, X -> Y, X <-> Y"
    ).d_separated("Z", "Y", ["X"])


def test_from_text_rejects_bad_items() -> None:
    with pytest.raises(SCMError):
        LinearSCM.from_text("Z -> X")
    with pytest.raises(SCMError):
        LinearSCM.from_text("Z -> X: many")
    with pytest.raises(SCMError):
        LinearSCM.from_text("Z => X: 1")


def test_from_text_rejects_duplicated_edges() -> None:
    with pytest.raises(SCMError, match="more than once"):
        LinearSCM.from_text("A -> B: 1.0, A -> B: 2.0")
    with pytest.raises(SCMError, match="more than once"):
        LinearSCM.from_text("A -> B: 1.0, B <- A: 2.0")
    with pytest.raises(SCMError, match="more than once"):
        LinearSCM.from_text("A -> B: 1.0, A <-> B: 1.0, B <-> A: 2.0")


def test_to_text_round_trips_at_repr_precision() -> None:
    scm = LinearSCM.from_text("A -> B: 0.1, B -> C: 2.0, A <-> C: 0.7")
    exact = LinearSCM(
        graph=scm.graph,
        coefficients={"A->B": 1.0 / 3.0, "B->C": 1e-5},
        latent_sd={"A<->C": 0.1 + 0.2},
    )
    assert (
        exact.to_text() == "A -> B: 0.3333333333333333, B -> C: 1e-05, A <-> C: 0.30000000000000004"
    )
    assert LinearSCM.from_text(exact.to_text()) == exact
    assert str(exact) == exact.to_text()


def test_keys_are_normalised_and_validated() -> None:
    scm = LinearSCM(
        graph=CausalGraph(edges=[("A", "B")], bidirected=[("B", "A")]),
        coefficients={" A -> B ": "2"},  # type: ignore[dict-item]
    )
    assert scm.coefficients == {"A->B": 2.0}
    assert coefficient_key("A", "B") == "A->B"
    assert latent_key("B", "A") == "A<->B"
    assert scm.latent_sd == {}
    ab = CausalGraph(edges=[("A", "B")])
    with pytest.raises(SCMError, match="without a coefficient"):
        LinearSCM(graph=CausalGraph(edges=[("A", "B"), ("B", "C")]), coefficients={"A->B": 1.0})
    with pytest.raises(SCMError, match="not in the graph"):
        LinearSCM(graph=ab, coefficients={"A->B": 1.0, "B->A": 1.0})
    with pytest.raises(SCMError, match="latent_sd"):
        LinearSCM(graph=ab, coefficients={"A->B": 1.0}, latent_sd={"A<->B": 1.0})
    with pytest.raises(SCMError, match="unknown nodes"):
        LinearSCM(graph=ab, coefficients={"A->B": 1.0}, noise_sd={"Q": 1.0})
    with pytest.raises(SCMError, match="non-negative"):
        LinearSCM(graph=ab, coefficients={"A->B": 1.0}, noise_sd={"A": -1.0})
    with pytest.raises(SCMError, match="must be finite"):
        LinearSCM(graph=ab, coefficients={"A->B": float("nan")})
    with pytest.raises(GraphError, match="cycle"):
        CausalGraph(edges=[("A", "B"), ("B", "A")])
    with pytest.raises(SCMError, match="cannot parse"):
        LinearSCM(graph=ab, coefficients={"A=>B": 1.0})


def test_duplicated_normalised_keys_are_rejected() -> None:
    ab = CausalGraph(edges=[("A", "B")], bidirected=[("A", "B")])
    with pytest.raises(SCMError, match="duplicates"):
        LinearSCM(graph=ab, coefficients={"A->B": 1.0, " A -> B ": 2.0})
    with pytest.raises(SCMError, match="duplicates"):
        LinearSCM(graph=ab, coefficients={"A->B": 1.0}, latent_sd={"A<->B": 1.0, "B<->A": 2.0})


def test_isolated_nodes_and_unknown_node_queries() -> None:
    scm = LinearSCM(graph=CausalGraph(nodes=["Q"], edges=[("A", "B")]), coefficients={"A->B": 1.0})
    assert scm.nodes == ("A", "B", "Q")
    assert scm.topological_order() == ("A", "B", "Q")
    with pytest.raises(SCMError, match="unknown node"):
        scm.total_effect("A", "nope")
    with pytest.raises(SCMError, match="unknown node"):
        scm.simulate(5, seed=0, intervene={"nope": 1.0})
    with pytest.raises(SCMError, match="unknown node"):
        scm.parents("nope")
    with pytest.raises(ValueError):
        scm.simulate(0, seed=0)


def test_intervention_values_must_be_real_numbers() -> None:
    scm = confounded_world()
    with pytest.raises(SCMError, match="'X'"):
        scm.simulate(5, seed=0, intervene={"X": "1"})  # type: ignore[dict-item]
    with pytest.raises(SCMError, match="'X'"):
        scm.simulate(5, seed=0, intervene={"X": None})  # type: ignore[dict-item]
    with pytest.raises(SCMError, match="'X'"):
        scm.interventional_mean("Y", {"X": True})
    with pytest.raises(ValueError, match="finite"):
        scm.simulate(5, seed=0, intervene={"X": float("inf")})
    assert scm.interventional_mean("Y", {"X": np.float64(1.0)}) == pytest.approx(2.0)
    assert scm.interventional_mean("Y", {"X": 1}) == pytest.approx(2.0)


# -- round trip -----------------------------------------------------------------------


@pytest.mark.parametrize("world", WORLDS, ids=lambda f: f.__name__)
def test_json_round_trip(world: Callable[[], LinearSCM]) -> None:
    scm = world()
    assert isinstance(scm, Spec)
    back = LinearSCM.from_json(scm.to_json())
    assert back == scm
    assert isinstance(back.graph, CausalGraph)
    assert load_spec(scm.to_json()) == scm
    assert back.content_hash() == scm.content_hash()


def test_frozen() -> None:
    scm = confounded_world()
    with pytest.raises(ValidationError):
        scm.coefficients = {}  # type: ignore[misc]
    with pytest.raises(ValidationError):
        scm.graph = CausalGraph()  # type: ignore[misc]


# -- graph agreement ----------------------------------------------------------------


@pytest.mark.parametrize("world", WORLDS, ids=lambda f: f.__name__)
def test_every_world_carries_a_causal_graph(world: Callable[[], LinearSCM]) -> None:
    scm = world()
    assert isinstance(scm.graph, CausalGraph)
    assert scm.graph.name == scm.name == world.__name__.removesuffix("_world")
    assert scm.nodes == scm.graph.nodes
    assert set(scm.coefficients) == {coefficient_key(a, b) for a, b in scm.graph.edges}
    assert set(scm.latent_sd) <= {latent_key(a, b) for a, b in scm.graph.bidirected}
    assert tuple(scm.simulate(5, seed=0).columns) == scm.graph.topological_order()


def test_feedback_world_flags_the_assertion() -> None:
    assert feedback_world().feedback is True
    assert feedback_world().graph.feedback is True
    assert confounded_world().feedback is False
    assert feedback_world().edges == confounded_world().edges


# -- simulation ----------------------------------------------------------------------


def test_simulate_shapes_columns_and_order() -> None:
    scm = iv_world()
    frame = scm.simulate(100, seed=0)
    assert frame.shape == (100, 3)
    assert tuple(frame.columns) == scm.topological_order() == ("Z", "X", "Y")
    assert all(frame[c].dtype == np.float64 for c in frame.columns)
    assert frame.equals(scm.simulate(100, seed=0))
    assert not frame.equals(scm.simulate(100, seed=1))


def test_observed_drops_unmeasured() -> None:
    scm = hidden_confounder_world()
    frame = scm.simulate(50, seed=3)
    assert "Z" in frame.columns
    obs = scm.observed(frame)
    assert tuple(obs.columns) == ("X", "Y")
    assert scm.measured == ("X", "Y")
    assert set(scm.measured) == scm.graph.measured
    assert confounded_world().observed(frame).shape == frame.shape


def test_intervention_fixes_the_node_and_shares_noise() -> None:
    scm = confounded_world()
    frame = scm.simulate(20, seed=0, intervene={"X": 1.0})
    assert (frame["X"] == 1.0).all()
    base = scm.simulate(20, seed=0)
    # Z is upstream of X and drawn from the same stream: unchanged by do(X).
    assert np.allclose(frame["Z"], base["Z"])
    # Y's noise is shared, so the difference is exactly the structural shift.
    shift = (frame["Y"] - base["Y"]).to_numpy()
    assert np.allclose(shift, 2.0 * (1.0 - base["X"].to_numpy()))


def test_latent_confounding_appears_in_the_data() -> None:
    scm = iv_world()
    frame = scm.simulate(50_000, seed=1)
    cov = np.cov(frame["X"], frame["Y"])
    # Cov(X, Y) = 2 Var(X) + Cov(X, U) = 2 * 3 + 1 = 7.
    assert abs(cov[0, 1] - 7.0) < 0.15
    assert abs(cov[0, 0] - 3.0) < 0.1


# -- ground truth ----------------------------------------------------------------------


def test_total_and_direct_effects_are_path_products() -> None:
    assert confounded_world().total_effect("X", "Y") == pytest.approx(2.0)
    assert confounded_world().total_effect("Z", "Y") == pytest.approx(1.5 + 0.8 * 2.0)
    assert confounded_world().total_effect("Y", "X") == 0.0
    assert frontdoor_world().total_effect("X", "Y") == pytest.approx(1.8)
    assert frontdoor_world().direct_effect("X", "Y") == 0.0
    assert mediator_world().total_effect("X", "Y") == pytest.approx(1.5)
    assert mediator_world().direct_effect("X", "Y") == pytest.approx(1.0)
    assert mediator_world().coefficient("M", "Y") == pytest.approx(0.5)
    assert iv_world().total_effect("Z", "Y") == pytest.approx(2.0)


@pytest.mark.parametrize(
    "world", [confounded_world, frontdoor_world, mediator_world, iv_world], ids=lambda f: f.__name__
)
def test_monte_carlo_do_matches_total_effect(world: Callable[[], LinearSCM]) -> None:
    scm = world()
    n = 200_000
    y1 = scm.simulate(n, seed=7, intervene={"X": 1.0})["Y"].mean()
    y0 = scm.simulate(n, seed=7, intervene={"X": 0.0})["Y"].mean()
    assert abs((y1 - y0) - scm.total_effect("X", "Y")) < 0.02


def test_interventional_mean_is_exact() -> None:
    source, target = transport_pair()
    assert source.interventional_mean("Y", {"X": 1.0}) == pytest.approx(2.0)
    assert target.interventional_mean("Y", {"X": 1.0}) == pytest.approx(2.0 + 2.25)
    assert target.interventional_mean("Y") == pytest.approx(1.5 * (1.5 + 0.8 * 2.0))
    frame = target.simulate(100_000, seed=11, intervene={"X": 1.0})
    assert abs(frame["Y"].mean() - 4.25) < 0.03


def test_transport_pair_differs_only_in_z() -> None:
    source, target = transport_pair()
    assert isinstance(source.graph, CausalGraph) and isinstance(target.graph, CausalGraph)
    assert source.graph == target.graph.model_copy(update={"name": "transport_source"})
    assert source.selection == target.selection == ("Z",)
    assert source.coefficients == target.coefficients
    assert source.total_effect("X", "Y") == target.total_effect("X", "Y") == 2.0
    diff = source.diff(target)
    assert set(diff.changed) == {"intercepts.Z", "graph.name"}
