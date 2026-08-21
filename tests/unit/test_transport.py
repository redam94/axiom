"""Selection diagrams and transportability: ``axiom.identify.transport``."""

from __future__ import annotations

import pytest

from axiom.core import Spec, Unsupported, load_spec
from axiom.identify.graph import CausalGraph, GraphError
from axiom.identify.transport import (
    TransportVerdict,
    directly_transportable,
    minimal_s_admissible_sets,
    s_admissible,
    s_admissible_sets,
    selection_diagram,
    transport_verdict,
    trivially_transportable,
)

CONFOUNDED = "Z -> X, Z -> Y, X -> Y"
MEDIATED = "X -> M, M -> Y"
BOW = "X -> Y, X <-> Y"
TWO_CONFOUNDERS = "A -> X, A -> Y, B -> X, B -> Y, X -> Y"


def _names(v: TransportVerdict) -> list[str]:
    return [a.name for a in v.verdict.assumptions]


# -- (a) no selection nodes ------------------------------------------------------------


def test_no_selection_is_same_population() -> None:
    g = CausalGraph.from_edges(CONFOUNDED)
    v = transport_verdict(g, "X", "Y")
    assert v.status == "identified" and v.route == "same_population"
    assert v.verdict.reason == "" and v.verdict.assumptions == ()
    assert v.formula == "P(Y|do(X))"
    assert v.selection == () and v.s_admissible_set == () and v.given is None
    assert v.graph_hash == g.content_hash()
    assert directly_transportable(g, "X", "Y") and trivially_transportable(g, "X", "Y")
    assert s_admissible(g, "X", "Y", ())
    sets = s_admissible_sets(g, "X", "Y")
    assert not isinstance(sets, Unsupported) and sets[0] == frozenset()
    assert minimal_s_admissible_sets(g, "X", "Y") == (frozenset(),)


def test_no_selection_trivial_test_is_identification() -> None:
    # with no S-nodes "trivially transportable" is just "the graph identifies the effect"
    assert not trivially_transportable(CausalGraph.from_edges(BOW), "X", "Y")
    assert trivially_transportable(CausalGraph.from_edges(MEDIATED), "X", "Y")


# -- (b) selection on the outcome: trivial (B&P 2014 Example 4) ----------------------------


def test_selection_on_outcome_is_trivial() -> None:
    g = CausalGraph.from_edges("X -> Y", selection=["Y"])
    assert not directly_transportable(g, "X", "Y")
    assert trivially_transportable(g, "X", "Y")
    assert s_admissible_sets(g, "X", "Y") == ()
    v = transport_verdict(g, "X", "Y")
    assert v.status == "identified" and v.route == "trivial" and v.licensed
    assert v.formula == "P*(Y|do(X)) = P*(Y|X)"
    assert v.s_admissible_set == () and v.mediators == ()
    assert _names(v) == ["trivial_transportability"]
    a = v.verdict.assumptions[0]
    assert a.state == "satisfied" and a.facet == "population"
    assert a.detail == {"route": "backdoor", "set": ""}
    assert "Def. 6" in a.statement


def test_selection_on_outcome_with_confounder_uses_target_backdoor() -> None:
    g = CausalGraph.from_edges(CONFOUNDED, selection=["Y"])
    assert not directly_transportable(g, "X", "Y")
    assert trivially_transportable(g, "X", "Y")
    v = transport_verdict(g, "X", "Y")
    assert v.status == "identified" and v.route == "trivial"
    assert v.s_admissible_set == ("Z",)
    assert v.formula == "P*(Y|do(X)) = Σ_{Z} P*(Y|X,Z) P*(Z)"
    assert v.verdict.assumptions[0].detail["set"] == "Z"


def test_trivial_route_through_target_frontdoor() -> None:
    g = CausalGraph.from_edges("X -> M, M -> Y, X <-> Y", selection=["M"])
    assert not directly_transportable(g, "X", "Y")
    assert s_admissible_sets(g, "X", "Y", measured_only=False) == ()
    assert trivially_transportable(g, "X", "Y")
    v = transport_verdict(g, "X", "Y")
    assert v.status == "identified" and v.route == "trivial"
    assert v.mediators == ("M",) and v.s_admissible_set == ()
    assert v.formula == "P*(Y|do(X)) = Σ_{M} P*(M|X) Σ_{X'} P*(Y|X',M) P*(X')"
    assert v.verdict.assumptions[0].detail == {"route": "frontdoor", "set": "M"}


def test_direct_but_not_trivial() -> None:
    # the old ancestor proxy said True here; the target effect is not identifiable
    g = CausalGraph.from_edges("X -> Y, X <-> Y, Y -> Q", selection=["Q"])
    assert directly_transportable(g, "X", "Y")
    assert not trivially_transportable(g, "X", "Y")
    assert transport_verdict(g, "X", "Y").route == "direct"


# -- (c) selection on a confounder: the transport formula ---------------------------------


def test_selection_on_confounder_needs_z() -> None:
    g = CausalGraph.from_edges(CONFOUNDED, selection=["Z"])
    assert not directly_transportable(g, "X", "Y")
    assert not s_admissible(g, "X", "Y", ())
    assert s_admissible(g, "X", "Y", ["Z"])
    assert s_admissible_sets(g, "X", "Y") == (frozenset({"Z"}),)
    assert minimal_s_admissible_sets(g, "X", "Y") == (frozenset({"Z"}),)
    # the target could also adjust for Z alone, but the S-admissible route is preferred
    assert trivially_transportable(g, "X", "Y")

    v = transport_verdict(g, "X", "Y")
    assert v.status == "identified" and v.route == "s_admissible_adjustment"
    assert v.licensed and v.given is None
    assert v.s_admissible_set == ("Z",)
    assert v.selection == ("Z",)
    assert v.formula == "P*(Y|do(X)) = Σ_{Z} P(Y|do(X),Z) P*(Z)"
    assert _names(v) == ["s_admissibility"]
    a = v.verdict.assumptions[0]
    assert a.state == "satisfied" and a.facet == "population"
    assert "Z=['Z']" in a.statement and a.detail == {"set": "Z"}


def test_semi_markovian_s_admissible_set() -> None:
    g = CausalGraph.from_edges(CONFOUNDED + ", X <-> Y", selection=["Z"])
    assert s_admissible_sets(g, "X", "Y") == (frozenset({"Z"}),)
    assert minimal_s_admissible_sets(g, "X", "Y") == (frozenset({"Z"}),)
    # the latent X <-> Y edge leaves the target no back-door or front-door set
    assert not trivially_transportable(g, "X", "Y")
    v = transport_verdict(g, "X", "Y")
    assert v.status == "identified" and v.route == "s_admissible_adjustment"
    assert v.s_admissible_set == ("Z",)


def test_s_admissibility_is_not_monotone() -> None:
    g = CausalGraph.from_edges("D -> X, D -> C, B -> C, B -> Y, X -> Y", selection=["D"])
    assert s_admissible(g, "X", "Y", ())
    assert not s_admissible(g, "X", "Y", ["C"])  # opens D -> C <- B
    assert s_admissible(g, "X", "Y", ["B", "C"])
    sets = s_admissible_sets(g, "X", "Y")
    assert not isinstance(sets, Unsupported)
    assert frozenset({"C"}) not in sets and frozenset({"B", "C"}) in sets
    assert minimal_s_admissible_sets(g, "X", "Y") == (frozenset(),)
    assert transport_verdict(g, "X", "Y").route == "direct"


# -- (d) selection upstream of the treatment only -------------------------------------------


def test_selection_on_cause_of_treatment_is_direct() -> None:
    g = CausalGraph.from_edges("W -> X, X -> Y", selection=["W"])
    assert directly_transportable(g, "X", "Y")
    # the target identifies P*(y|do(x)) = P*(y|x) on its own too; direct is preferred
    assert trivially_transportable(g, "X", "Y")
    v = transport_verdict(g, "X", "Y")
    assert v.status == "identified" and v.route == "direct"
    assert v.formula == "P*(Y|do(X)) = P(Y|do(X))"
    assert v.verdict.assumptions == ()


def test_selection_on_treatment_itself_is_direct() -> None:
    g = CausalGraph.from_edges(CONFOUNDED, selection=["X"])
    assert directly_transportable(g, "X", "Y")
    assert transport_verdict(g, "X", "Y").route == "direct"


def test_selection_on_sibling_of_outcome_is_direct() -> None:
    g = CausalGraph.from_edges("X -> Y, W <-> Y, W -> X", selection=["W"])
    assert directly_transportable(g, "X", "Y")
    assert transport_verdict(g, "X", "Y").route == "direct"


def test_selection_on_irrelevant_node_is_direct() -> None:
    g = CausalGraph.from_edges("X -> Y, Y -> Q", selection=["Q"])
    assert trivially_transportable(g, "X", "Y")
    assert directly_transportable(g, "X", "Y")
    assert transport_verdict(g, "X", "Y").route == "direct"


# -- (e) selection on a mediator -------------------------------------------------------------


def test_selection_on_mediator_is_trivial_when_target_identifies() -> None:
    g = CausalGraph.from_edges(MEDIATED, selection=["M"])
    assert not directly_transportable(g, "X", "Y")
    # M would separate S[M] from Y but is a descendant of X: not admissible
    assert not s_admissible(g, "X", "Y", ["M"])
    assert s_admissible_sets(g, "X", "Y") == ()
    assert s_admissible_sets(g, "X", "Y", measured_only=False) == ()
    v = transport_verdict(g, "X", "Y")
    assert v.status == "identified" and v.route == "trivial"
    assert v.formula == "P*(Y|do(X)) = P*(Y|X)"


@pytest.mark.parametrize(
    ("text", "selection", "s_node"),
    [
        (BOW, ["Y"], "S[Y]"),
        ("X -> M, M -> Y, X <-> Y, M <-> Y", ["M"], "S[M]"),
    ],
)
def test_no_sufficient_condition_is_unsupported(
    text: str, selection: list[str], s_node: str
) -> None:
    g = CausalGraph.from_edges(text, selection=selection)
    assert not directly_transportable(g, "X", "Y")
    assert not trivially_transportable(g, "X", "Y")
    assert s_admissible_sets(g, "X", "Y", measured_only=False) == ()
    v = transport_verdict(g, "X", "Y")
    assert v.status == "unsupported" and v.route == "" and not v.licensed
    assert s_node in v.verdict.reason
    assert "sID" in v.verdict.reason and "Thm 3" in v.verdict.reason
    assert v.missing == ("sid_recursion",)
    assert v.formula == "" and v.s_admissible_set == () and v.verdict.assumptions == ()


# -- (f) S-admissible only through an unmeasured node -------------------------------------------


def test_unmeasured_admissible_set_is_downgraded_unmeasured_route() -> None:
    g = CausalGraph.from_edges(CONFOUNDED, selection=["Z"], unmeasured=["Z"])
    assert s_admissible_sets(g, "X", "Y") == ()
    assert s_admissible_sets(g, "X", "Y", measured_only=False) == (frozenset({"Z"}),)
    assert not trivially_transportable(g, "X", "Y")
    v = transport_verdict(g, "X", "Y")
    assert v.status == "downgraded"
    assert v.route == "s_admissible_adjustment_unmeasured"
    assert v.s_admissible_set == ("Z",)
    assert _names(v) == ["s_admissibility", "unmeasured_s_admissible_set"]
    a = v.verdict.assumptions[1]
    assert a.state == "unverified" and a.detail["unmeasured"] == "Z"
    assert v.formula == "P*(Y|do(X)) = Σ_{Z} P(Y|do(X),Z) P*(Z)"


# -- (g) a caller-proposed set ---------------------------------------------------------------


def test_given_empty_set_is_blocked_when_not_admissible() -> None:
    g = CausalGraph.from_edges(CONFOUNDED, selection=["Z"])
    v = transport_verdict(g, "X", "Y", given=())
    assert v.status == "blocked" and v.route == "" and not v.licensed
    assert v.given == () and v.s_admissible_set == () and v.formula == ""
    assert "S[Z]" in v.verdict.reason and "Z=[]" in v.verdict.reason
    assert v.verdict.assumptions == () and v.missing == ()


def test_given_admissible_set_is_identified() -> None:
    g = CausalGraph.from_edges(CONFOUNDED, selection=["Z"])
    v = transport_verdict(g, "X", "Y", given=["Z"])
    assert v.status == "identified" and v.route == "s_admissible_adjustment"
    assert v.given == ("Z",) and v.s_admissible_set == ("Z",)
    assert v.formula == "P*(Y|do(X)) = Σ_{Z} P(Y|do(X),Z) P*(Z)"
    assert _names(v) == ["s_admissibility"]
    # a bare string is one node, not a sequence of characters
    assert transport_verdict(g, "X", "Y", given="Z") == v


def test_given_empty_set_on_directly_transportable_graph() -> None:
    g = CausalGraph.from_edges("W -> X, X -> Y", selection=["W"])
    v = transport_verdict(g, "X", "Y", given=())
    assert v.status == "identified" and v.route == "s_admissible_adjustment"
    assert v.given == () and v.formula == "P*(Y|do(X)) = P(Y|do(X))"


def test_given_descendant_of_treatment_is_blocked() -> None:
    g = CausalGraph.from_edges(MEDIATED, selection=["M"])
    v = transport_verdict(g, "X", "Y", given=["M"])
    assert v.status == "blocked" and v.given == ("M",)
    assert "descendants of the treatment X" in v.verdict.reason and "['M']" in v.verdict.reason
    # the same malformed proposal is blocked even without selection nodes
    plain = transport_verdict(CausalGraph.from_edges(MEDIATED), "X", "Y", given=["M"])
    assert plain.status == "blocked"


def test_given_blocked_reason_names_only_connected_s_nodes() -> None:
    g = CausalGraph.from_edges(TWO_CONFOUNDERS, selection=["A", "B"])
    v = transport_verdict(g, "X", "Y", given=["A"])
    assert v.status == "blocked"
    assert "S[B]" in v.verdict.reason and "S[A]" not in v.verdict.reason
    both = transport_verdict(g, "X", "Y", given=["A", "B"])
    assert both.status == "identified" and both.s_admissible_set == ("A", "B")


def test_given_without_selection_is_same_population() -> None:
    g = CausalGraph.from_edges(CONFOUNDED)
    v = transport_verdict(g, "X", "Y", given=["Z"])
    assert v.status == "identified" and v.route == "same_population"
    assert v.given == ("Z",) and v.s_admissible_set == ("Z",)
    assert v.formula == "P(Y|do(X))"


def test_given_unmeasured_set_is_downgraded() -> None:
    g = CausalGraph.from_edges(CONFOUNDED, selection=["Z"], unmeasured=["Z"])
    v = transport_verdict(g, "X", "Y", given=["Z"])
    assert v.status == "downgraded" and v.route == "s_admissible_adjustment_unmeasured"
    assert v.given == ("Z",) and "unmeasured_s_admissible_set" in _names(v)


def test_given_malformed_raises() -> None:
    g = CausalGraph.from_edges(CONFOUNDED, selection=["Z"])
    with pytest.raises(GraphError, match="proposed set"):
        transport_verdict(g, "X", "Y", given=["X"])
    with pytest.raises(GraphError, match="proposed set"):
        transport_verdict(g, "X", "Y", given=["Y", "Z"])
    with pytest.raises(GraphError, match="unknown node"):
        transport_verdict(g, "X", "Y", given=["Q"])


# -- (h) feedback graphs ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "selection", "route"),
    [
        (CONFOUNDED, [], "same_population"),
        ("W -> X, X -> Y", ["W"], "direct"),
        (CONFOUNDED, ["Z"], "s_admissible_adjustment"),
        ("X -> Y", ["Y"], "trivial"),
    ],
)
def test_feedback_downgrades_licensed_verdicts(text: str, selection: list[str], route: str) -> None:
    g = CausalGraph.from_edges(text, selection=selection, feedback=True)
    v = transport_verdict(g, "X", "Y")
    assert v.status == "downgraded" and v.route == route
    assert v.verdict.assumptions[-1].name == "no_time_varying_confounding"
    assert v.verdict.assumptions[-1].state == "unverified"
    plain = transport_verdict(g.model_copy(update={"feedback": False}), "X", "Y")
    assert plain.status == "identified"
    assert len(v.verdict.assumptions) == len(plain.verdict.assumptions) + 1


def test_feedback_leaves_unlicensed_verdicts_alone() -> None:
    blocked = transport_verdict(
        CausalGraph.from_edges(CONFOUNDED, selection=["Z"], feedback=True), "X", "Y", given=()
    )
    assert blocked.status == "blocked" and blocked.verdict.assumptions == ()
    unsupported = transport_verdict(
        CausalGraph.from_edges(BOW, selection=["Y"], feedback=True), "X", "Y"
    )
    assert unsupported.status == "unsupported" and unsupported.verdict.assumptions == ()


# -- (i) round trip -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("selection", "given"),
    [([], None), (["Z"], None), (["Y"], None), (["Z"], ()), (["Z"], ["Z"])],
)
def test_transport_verdict_round_trips(selection: list[str], given: list[str] | None) -> None:
    g = CausalGraph.from_edges(CONFOUNDED, selection=selection)
    v = transport_verdict(g, "X", "Y", given=given)
    s = v.to_json()
    back = TransportVerdict.from_json(s)
    assert back == v and back.content_hash() == v.content_hash()
    assert load_spec(s) == v
    assert isinstance(back, Spec) and back.verdict.status == v.status
    assert back.given == v.given


def test_transport_verdict_requires_names() -> None:
    from axiom.core import Verdict

    with pytest.raises(ValueError):
        TransportVerdict(treatment=" ", outcome="Y", verdict=Verdict(status="identified"))


# -- (j) the selection diagram -----------------------------------------------------------------


def test_selection_diagram_adds_unmeasured_s_nodes() -> None:
    g = CausalGraph.from_edges(CONFOUNDED, selection=["Z", "Y"], unmeasured=["Z"], name="g")
    d = selection_diagram(g)
    assert set(d.nodes) == {"X", "Y", "Z", "S[Y]", "S[Z]"}
    assert ("S[Z]", "Z") in d.edges and ("S[Y]", "Y") in d.edges
    assert d.selection == ()
    assert set(d.unmeasured) == {"Z", "S[Y]", "S[Z]"}
    assert not d.is_measured("S[Z]") and d.parents("S[Z]") == frozenset()
    assert d.name == "g" and d.feedback is g.feedback
    # an unchanged graph comes back unchanged
    plain = CausalGraph.from_edges(CONFOUNDED)
    assert selection_diagram(plain) == plain
    # d-separation on the diagram works like on any DAG
    assert d.remove_edges_into("X").d_separated("Y", "S[Z]", ["X", "Z"])
    assert not d.remove_edges_into("X").d_separated("Y", "S[Y]", ["X", "Z"])


def test_selection_diagram_keeps_bidirected_edges() -> None:
    g = CausalGraph.from_edges("X -> Y, X <-> Y", selection=["X"])
    d = selection_diagram(g)
    assert d.bidirected == (("X", "Y"),)
    # in G_bar X the bidirected edge into X goes too, so S[X] is cut off
    assert directly_transportable(g, "X", "Y")


def test_selection_diagram_rejects_name_collision() -> None:
    g = CausalGraph(nodes=("S[Z]",), edges=(("Z", "X"), ("X", "Y")), selection=("Z",))
    with pytest.raises(GraphError, match="collide"):
        selection_diagram(g)


# -- argument validation and search bounds -------------------------------------------------------


def test_bad_arguments_raise_graph_error() -> None:
    g = CausalGraph.from_edges(CONFOUNDED, selection=["Z"])
    with pytest.raises(GraphError, match="unknown node"):
        transport_verdict(g, "X", "Q")
    with pytest.raises(GraphError, match="same node"):
        directly_transportable(g, "X", "X")
    with pytest.raises(GraphError, match="unknown node"):
        s_admissible(g, "X", "Y", ["S[Z]"])
    assert not s_admissible(g, "X", "Y", ["X"])
    assert not s_admissible(g, "X", "Y", ["Y"])


def test_search_bound_returns_unsupported() -> None:
    edges = ", ".join(f"C{i} -> X, C{i} -> Y" for i in range(4))
    g = CausalGraph.from_edges(edges + ", X -> Y", selection=["C0"])
    r = s_admissible_sets(g, "X", "Y", max_candidates=3)
    assert isinstance(r, Unsupported) and "max_candidates=3" in r.reason
    assert isinstance(minimal_s_admissible_sets(g, "X", "Y", max_candidates=3), Unsupported)
    ok = s_admissible_sets(g, "X", "Y", max_candidates=4)
    assert not isinstance(ok, Unsupported)
    assert ok[0] == frozenset({"C0"})
    assert minimal_s_admissible_sets(g, "X", "Y") == (frozenset({"C0"}),)
    # sorted by size then lexicographically
    assert [sorted(s) for s in ok[:4]] == [["C0"], ["C0", "C1"], ["C0", "C2"], ["C0", "C3"]]
    # when no set exists at all the empty tuple is the complete answer, bound or not
    none = CausalGraph.from_edges(edges + ", X -> Y", selection=["Y"])
    assert s_admissible_sets(none, "X", "Y", max_candidates=3) == ()


def test_measured_search_overflow_still_names_a_set() -> None:
    edges = ", ".join(f"C{i} -> X, C{i} -> Y" for i in range(5)) + ", X -> Y"
    g = CausalGraph.from_edges(edges, selection=["C0"])
    v = transport_verdict(g, "X", "Y", max_candidates=3)
    assert v.status == "identified" and v.route == "s_admissible_adjustment"
    assert v.s_admissible_set == ("C0",)
    assert "measured S-admissible search was not attempted" in v.verdict.reason
    assert "max_candidates=3" in v.verdict.reason
    assert transport_verdict(g, "X", "Y", max_candidates=5).verdict.reason == ""


def test_unmeasured_only_overflow_is_reported_in_reason() -> None:
    hidden = ["Z", "U0", "U1", "U2", "U3"]
    g = CausalGraph.from_edges(
        CONFOUNDED + ", U0 -> Y, U1 -> Y, U2 -> Y, U3 -> Y", selection=["Z"], unmeasured=hidden
    )
    v = transport_verdict(g, "X", "Y", max_candidates=4)
    assert v.status == "downgraded" and v.route == "s_admissible_adjustment_unmeasured"
    assert v.s_admissible_set == ("Z",)
    assert "unmeasured S-admissible search was not attempted" in v.verdict.reason
    assert "max_candidates=4" in v.verdict.reason
    assert _names(v) == ["s_admissibility", "unmeasured_s_admissible_set"]
    assert transport_verdict(g, "X", "Y", max_candidates=5).verdict.reason == ""


def test_frontdoor_overflow_is_named_in_unsupported_reason() -> None:
    g = CausalGraph.from_edges("X -> M, M -> Y, X <-> Y", selection=["M"])
    assert not trivially_transportable(g, "X", "Y", max_candidates=0)
    v = transport_verdict(g, "X", "Y", max_candidates=0)
    assert v.status == "unsupported"
    assert "front-door search was not attempted" in v.verdict.reason
    assert v.missing == ("sid_recursion", "bounded_frontdoor_search")


def test_minimal_sets_drop_supersets() -> None:
    # two confounders with S on both; either alone is not enough, both together are
    g = CausalGraph.from_edges(TWO_CONFOUNDERS, selection=["A", "B"])
    assert minimal_s_admissible_sets(g, "X", "Y") == (frozenset({"A", "B"}),)
    v = transport_verdict(g, "X", "Y")
    assert v.status == "identified" and v.s_admissible_set == ("A", "B")
    assert v.formula == "P*(Y|do(X)) = Σ_{A,B} P(Y|do(X),A,B) P*(A,B)"
