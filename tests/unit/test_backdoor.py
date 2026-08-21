"""Back-door criterion, adjustment-set enumeration, and roles on the classic graphs."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from axiom.core import Unsupported
from axiom.identify import frontdoor
from axiom.identify.backdoor import (
    ROLE_PRECEDENCE,
    RoleAssignment,
    adjustment_sets,
    admissible_set_exists,
    assign_roles,
    backdoor_admissible,
    canonical_adjustment_set,
    minimal_adjustment_sets,
    requires_unmeasured,
    roles,
)
from axiom.identify.graph import CausalGraph, GraphError

F = frozenset


def sets(*members: str) -> tuple[frozenset[str], ...]:
    return tuple(F(m.split()) if m else F() for m in members)


def instrument_nodes(g: CausalGraph, x: str, y: str) -> tuple[str, ...]:
    return tuple(sorted(n for n, r in roles(g, x, y).items() if r == "instrument"))


# -- classic graphs ------------------------------------------------------------------

CONFOUNDER = CausalGraph.from_edges("Z -> X, Z -> Y, X -> Y")
MEDIATOR = CausalGraph.from_edges("X -> M, M -> Y")
M_BIAS = CausalGraph.from_edges(
    "U1 -> X, U1 -> M, U2 -> M, U2 -> Y, X -> Y", unmeasured=["U1", "U2"]
)
BUTTERFLY = CausalGraph.from_edges(
    "U1 -> X, U1 -> M, U2 -> M, U2 -> Y, M -> X, M -> Y, X -> Y", unmeasured=["U1", "U2"]
)
INSTRUMENT = CausalGraph.from_edges("Z -> X, X -> Y, X <-> Y")
# Pearl 2009, Fig. 3.4 (treatment Xi, outcome Xj)
PEARL_3_4 = CausalGraph.from_edges(
    "X1 -> X3, X1 -> X4, X2 -> X4, X2 -> X5, X3 -> Xi, X4 -> Xi, X4 -> Xj, X5 -> Xj, "
    "Xi -> X6, X6 -> Xj"
)


def test_simple_confounder() -> None:
    assert not backdoor_admissible(CONFOUNDER, "X", "Y", ())
    assert backdoor_admissible(CONFOUNDER, "X", "Y", ["Z"])
    assert adjustment_sets(CONFOUNDER, "X", "Y") == sets("Z")
    assert minimal_adjustment_sets(CONFOUNDER, "X", "Y") == sets("Z")
    assert canonical_adjustment_set(CONFOUNDER, "X", "Y") == F("Z")
    assert admissible_set_exists(CONFOUNDER, "X", "Y")
    assert not requires_unmeasured(CONFOUNDER, "X", "Y")
    assert roles(CONFOUNDER, "X", "Y") == {"X": "treatment", "Y": "outcome", "Z": "confounder"}


def test_simple_confounder_unmeasured_is_downgrade() -> None:
    g = CONFOUNDER.with_unmeasured("Z")
    assert adjustment_sets(g, "X", "Y") == ()
    assert minimal_adjustment_sets(g, "X", "Y") == ()
    assert canonical_adjustment_set(g, "X", "Y") is None
    assert adjustment_sets(g, "X", "Y", measured_only=False) == sets("Z")
    assert canonical_adjustment_set(g, "X", "Y", measured_only=False) == F("Z")
    assert not admissible_set_exists(g, "X", "Y")
    assert admissible_set_exists(g, "X", "Y", measured_only=False)
    assert requires_unmeasured(g, "X", "Y")
    # the criterion itself does not care about measurement
    assert backdoor_admissible(g, "X", "Y", ["Z"])


def test_mediator_chain() -> None:
    assert backdoor_admissible(MEDIATOR, "X", "Y", ())
    assert not backdoor_admissible(MEDIATOR, "X", "Y", ["M"])  # descendant of X
    assert adjustment_sets(MEDIATOR, "X", "Y") == sets("")
    assert minimal_adjustment_sets(MEDIATOR, "X", "Y") == sets("")
    assert canonical_adjustment_set(MEDIATOR, "X", "Y") == F()
    assert not requires_unmeasured(MEDIATOR, "X", "Y")
    assert roles(MEDIATOR, "X", "Y") == {"X": "treatment", "M": "mediator", "Y": "outcome"}


def test_m_bias() -> None:
    assert backdoor_admissible(M_BIAS, "X", "Y", ())
    assert not backdoor_admissible(M_BIAS, "X", "Y", ["M"])
    assert backdoor_admissible(M_BIAS, "X", "Y", ["M", "U1"])
    assert backdoor_admissible(M_BIAS, "X", "Y", ["M", "U2"])
    assert adjustment_sets(M_BIAS, "X", "Y") == sets("")
    assert adjustment_sets(M_BIAS, "X", "Y", measured_only=False) == sets(
        "", "U1", "U2", "M U1", "M U2", "U1 U2", "M U1 U2"
    )
    assert minimal_adjustment_sets(M_BIAS, "X", "Y", measured_only=False) == sets("")
    # the canonical set is admissible but not minimal; M is not an ancestor of {X, Y}
    assert canonical_adjustment_set(M_BIAS, "X", "Y") == F()
    assert canonical_adjustment_set(M_BIAS, "X", "Y", measured_only=False) == F({"U1", "U2"})
    assert not requires_unmeasured(M_BIAS, "X", "Y")
    assert roles(M_BIAS, "X", "Y") == {
        "X": "treatment",
        "Y": "outcome",
        "M": "collider",
        "U1": "instrument",
        "U2": "neutral",
    }


def test_butterfly_bias() -> None:
    # M is both a confounder (X <- M -> Y) and a collider (U1 -> M <- U2);
    # it must be in every admissible set and then needs U1 or U2 as well.
    assert not backdoor_admissible(BUTTERFLY, "X", "Y", ())
    assert not backdoor_admissible(BUTTERFLY, "X", "Y", ["M"])
    assert not backdoor_admissible(BUTTERFLY, "X", "Y", ["U1", "U2"])
    assert backdoor_admissible(BUTTERFLY, "X", "Y", ["M", "U1"])
    assert adjustment_sets(BUTTERFLY, "X", "Y") == ()
    assert canonical_adjustment_set(BUTTERFLY, "X", "Y") is None
    assert adjustment_sets(BUTTERFLY, "X", "Y", measured_only=False) == sets(
        "M U1", "M U2", "M U1 U2"
    )
    assert minimal_adjustment_sets(BUTTERFLY, "X", "Y", measured_only=False) == sets("M U1", "M U2")
    assert canonical_adjustment_set(BUTTERFLY, "X", "Y", measured_only=False) == F(
        {"M", "U1", "U2"}
    )
    assert requires_unmeasured(BUTTERFLY, "X", "Y")
    r = roles(BUTTERFLY, "X", "Y")
    assert r["M"] == "confounder"  # confounder outranks collider
    assert r["U1"] == "confounder" and r["U2"] == "confounder"


def test_instrument_graph_has_no_adjustment_set() -> None:
    assert not backdoor_admissible(INSTRUMENT, "X", "Y", ())
    assert not backdoor_admissible(INSTRUMENT, "X", "Y", ["Z"])
    assert adjustment_sets(INSTRUMENT, "X", "Y") == ()
    assert adjustment_sets(INSTRUMENT, "X", "Y", measured_only=False) == ()
    assert minimal_adjustment_sets(INSTRUMENT, "X", "Y") == ()
    assert canonical_adjustment_set(INSTRUMENT, "X", "Y", measured_only=False) is None
    assert not admissible_set_exists(INSTRUMENT, "X", "Y", measured_only=False)
    assert not requires_unmeasured(INSTRUMENT, "X", "Y")  # nothing to downgrade to
    assert roles(INSTRUMENT, "X", "Y") == {"X": "treatment", "Y": "outcome", "Z": "instrument"}


def test_confounded_instrument_is_not_an_instrument() -> None:
    # Z -> X with Z <-> Y: Z shares a latent cause with Y, so it is a proxy of that
    # latent confounder, not an instrument.
    g = CausalGraph.from_edges("Z -> X, X -> Y, Z <-> Y")
    assert roles(g, "X", "Y")["Z"] == "proxy"
    assert adjustment_sets(g, "X", "Y") == sets("Z")
    # Z <- W -> Y: Z is d-connected to Y in G_underline_x via the fork at W.
    g2 = CausalGraph.from_edges("Z -> X, W -> Z, W -> Y, X -> Y")
    r = roles(g2, "X", "Y")
    assert r["W"] == "confounder" and r["Z"] == "neutral"
    assert adjustment_sets(g2, "X", "Y") == sets("W", "Z", "W Z")
    assert minimal_adjustment_sets(g2, "X", "Y") == sets("W", "Z")
    assert canonical_adjustment_set(g2, "X", "Y") == F({"W", "Z"})
    # with W unmeasured Z is the proxy and still a complete measured adjustment set
    g3 = g2.with_unmeasured("W")
    assert roles(g3, "X", "Y")["Z"] == "proxy"
    assert adjustment_sets(g3, "X", "Y") == sets("Z")
    assert not requires_unmeasured(g3, "X", "Y")


def test_latent_confounded_instrument_is_an_instrument() -> None:
    # Z <-> X: Z is a proxy of a latent cause of X with no path to Y except through X.
    g = CausalGraph.from_edges("Z <-> X, X -> Y, X <-> Y")
    assert roles(g, "X", "Y") == {"X": "treatment", "Y": "outcome", "Z": "instrument"}
    assert instrument_nodes(g, "X", "Y") == frontdoor.instruments(g, "X", "Y", measured_only=False)


@pytest.mark.parametrize(
    "text",
    [
        "Z -> X, X -> Y, X <-> Y",
        "Z <-> X, X -> Y, X <-> Y",
        "Z -> X, X -> Y, Z <-> Y",
        "Z -> X, W -> Z, W -> Y, X -> Y",
        "A <-> X, A -> C, B -> C, B -> Y, X -> Y",
        "A -> N, B -> N, N -> X, B -> Y",
        "U1 -> X, U1 -> M, U2 -> M, U2 -> Y, X -> Y",
        "Z1 -> X, Z2 -> Z1, Z3 -> X, Z3 -> Y, X -> Y, X <-> Y",
        "Z -> X, X -> M, M -> Y, Z -> D, X <-> Y",
        "X -> Y",
    ],
)
def test_instrument_role_matches_frontdoor_instruments(text: str) -> None:
    g = CausalGraph.from_edges(text)
    assert instrument_nodes(g, "X", "Y") == frontdoor.instruments(g, "X", "Y", measured_only=False)


def test_proxy_of_unmeasured_confounder() -> None:
    g = CausalGraph.from_edges("U -> X, U -> Y, U -> P, X -> Y", unmeasured=["U"])
    r = roles(g, "X", "Y")
    assert r == {"U": "confounder", "P": "proxy", "X": "treatment", "Y": "outcome"}
    assert adjustment_sets(g, "X", "Y") == ()  # a proxy does not block the confounder's path
    assert not backdoor_admissible(g, "X", "Y", ["P"])
    assert requires_unmeasured(g, "X", "Y")
    # a proxy that is also a descendant of X is a collider, not a proxy
    g2 = CausalGraph.from_edges("U -> X, U -> Y, U -> P, X -> P, X -> Y", unmeasured=["U"])
    assert roles(g2, "X", "Y")["P"] == "collider"


def test_descendant_of_outcome_and_common_child() -> None:
    g = CausalGraph.from_edges("X -> Y, Y -> D, X -> C, Y -> C, X -> W")
    r = roles(g, "X", "Y")
    assert r["D"] == "descendant_of_outcome"
    assert r["C"] == "collider"  # collider outranks descendant_of_outcome
    assert r["W"] == "neutral"
    assert adjustment_sets(g, "X", "Y") == sets("")


def test_collider_sides_are_reached_by_open_walks() -> None:
    # treatment side through a bidirected edge at X
    g = CausalGraph.from_edges("A <-> X, A -> C, B -> C, B -> Y, X -> Y")
    r = roles(g, "X", "Y")
    assert r["C"] == "collider" and r["A"] == "instrument" and r["B"] == "neutral"
    assert backdoor_admissible(g, "X", "Y", ())
    assert not backdoor_admissible(g, "X", "Y", ["C"])
    # outcome side through a fork
    g2 = CausalGraph.from_edges("X -> Y, X -> C, B -> C, W -> B, W -> Y")
    assert roles(g2, "X", "Y")["C"] == "collider"
    # a common effect that is itself an ancestor of X is a non-collider on a back-door
    # path; it is not reported as a collider and conditioning on it is admissible
    g3 = CausalGraph.from_edges("A -> N, B -> N, N -> X, B -> Y, X -> Y")
    r3 = roles(g3, "X", "Y")
    assert r3["N"] == "neutral" and r3["B"] == "confounder"
    assert backdoor_admissible(g3, "X", "Y", ["N"])
    assert minimal_adjustment_sets(g3, "X", "Y") == sets("B", "N")
    # two colliders in a row: N's latent parent is not on the treatment side because its
    # other child C is not an ancestor of X (the walk X <- A -> C <- U -> N is blocked at C),
    # and C itself is an instrument (C <-> N <- B -> Y is blocked at N)
    g4 = CausalGraph.from_edges("A -> X, A -> C, C <-> N, B -> N, B -> Y, X -> Y")
    r4 = roles(g4, "X", "Y")
    assert r4["N"] == "neutral" and r4["C"] == "instrument"
    assert backdoor_admissible(g4, "X", "Y", ["A", "N"])
    assert instrument_nodes(g4, "X", "Y") == frontdoor.instruments(g4, "X", "Y")


def test_role_precedence_covers_every_role() -> None:
    assert len(ROLE_PRECEDENCE) == len(set(ROLE_PRECEDENCE)) == 9


# -- Pearl 2009 Fig. 3.4 ------------------------------------------------------------------


def test_pearl_fig_3_4_minimal_family() -> None:
    assert minimal_adjustment_sets(PEARL_3_4, "Xi", "Xj") == sets(
        "X1 X4", "X2 X4", "X3 X4", "X4 X5"
    )
    assert not backdoor_admissible(PEARL_3_4, "Xi", "Xj", ["X4"])
    assert not backdoor_admissible(PEARL_3_4, "Xi", "Xj", ())
    assert backdoor_admissible(PEARL_3_4, "Xi", "Xj", ["X3", "X4"])
    assert backdoor_admissible(PEARL_3_4, "Xi", "Xj", ["X4", "X5"])
    assert canonical_adjustment_set(PEARL_3_4, "Xi", "Xj") == F({"X1", "X2", "X3", "X4", "X5"})
    all_sets = adjustment_sets(PEARL_3_4, "Xi", "Xj")
    assert isinstance(all_sets, tuple)
    assert all("X4" in s for s in all_sets)
    assert F({"X1", "X2", "X3", "X4", "X5"}) in all_sets
    r = roles(PEARL_3_4, "Xi", "Xj")
    assert r["X6"] == "mediator"
    assert r["X1"] == r["X2"] == r["X4"] == "confounder"
    assert r["X3"] == r["X5"] == "neutral"


# -- validation ----------------------------------------------------------------------


def test_malformed_queries_raise() -> None:
    with pytest.raises(GraphError, match="unknown node"):
        backdoor_admissible(CONFOUNDER, "X", "Q", ())
    with pytest.raises(GraphError, match="must differ"):
        backdoor_admissible(CONFOUNDER, "X", "X", ())
    with pytest.raises(GraphError, match="unknown nodes"):
        backdoor_admissible(CONFOUNDER, "X", "Y", ["Q"])
    with pytest.raises(GraphError, match="may not contain"):
        backdoor_admissible(CONFOUNDER, "X", "Y", ["X"])
    with pytest.raises(GraphError, match="may not contain"):
        backdoor_admissible(CONFOUNDER, "X", "Y", ["Y"])
    with pytest.raises(GraphError):
        adjustment_sets(CONFOUNDER, "X", "Q")
    with pytest.raises(GraphError):
        canonical_adjustment_set(CONFOUNDER, "X", "Q")
    with pytest.raises(GraphError):
        roles(CONFOUNDER, "Q", "Y")
    with pytest.raises(GraphError):
        requires_unmeasured(CONFOUNDER, "Y", "Y")


def test_unmeasured_treatment_or_outcome_is_a_malformed_measured_query() -> None:
    hidden_x = CONFOUNDER.with_unmeasured("X")
    hidden_y = CONFOUNDER.with_unmeasured("Y")
    for g in (hidden_x, hidden_y):
        for fn in (
            adjustment_sets,
            minimal_adjustment_sets,
            canonical_adjustment_set,
            admissible_set_exists,
        ):
            with pytest.raises(GraphError, match="treatment/outcome is unmeasured"):
                fn(g, "X", "Y")
            # the purely graphical question is still answered
            assert fn(g, "X", "Y", measured_only=False) == fn(CONFOUNDER, "X", "Y")
        with pytest.raises(GraphError, match="treatment/outcome is unmeasured"):
            requires_unmeasured(g, "X", "Y")
        # the criterion and the roles do not ask about measurement of x or y
        assert backdoor_admissible(g, "X", "Y", ["Z"])
        assert roles(g, "X", "Y") == roles(CONFOUNDER, "X", "Y")


# -- enumeration bound -----------------------------------------------------------------


def _many_confounders(k: int) -> CausalGraph:
    text = ", ".join(f"C{i:02d} -> X, C{i:02d} -> Y" for i in range(k)) + ", X -> Y"
    return CausalGraph.from_edges(text)


def test_overflow_returns_unsupported() -> None:
    g = _many_confounders(17)
    out = adjustment_sets(g, "X", "Y")
    assert isinstance(out, Unsupported)
    assert not out and out.status == "unsupported"
    assert out.detail["n_candidates"] == "17" and out.detail["max_candidates"] == "16"
    # every confounder is an ancestor of {X, Y}, so the canonical set overflows too
    mins = minimal_adjustment_sets(g, "X", "Y")
    assert isinstance(mins, Unsupported)
    assert mins.detail["n_candidates"] == "17"
    # existence and the canonical set are still decided without enumeration
    assert admissible_set_exists(g, "X", "Y")
    assert canonical_adjustment_set(g, "X", "Y") == F(f"C{i:02d}" for i in range(17))
    assert not requires_unmeasured(g, "X", "Y")
    # a lower cap on a small graph
    small = _many_confounders(4)
    assert isinstance(adjustment_sets(small, "X", "Y", max_candidates=3), Unsupported)
    assert isinstance(minimal_adjustment_sets(small, "X", "Y", max_candidates=3), Unsupported)
    assert adjustment_sets(small, "X", "Y", max_candidates=4) == sets("C00 C01 C02 C03")
    assert minimal_adjustment_sets(small, "X", "Y", max_candidates=4) == sets("C00 C01 C02 C03")


def test_overflow_with_no_admissible_set_is_a_complete_answer() -> None:
    text = ", ".join(f"N{i:02d} -> X" for i in range(17)) + ", X -> Y, X <-> Y"
    g = CausalGraph.from_edges(text)
    assert adjustment_sets(g, "X", "Y") == ()
    assert minimal_adjustment_sets(g, "X", "Y") == ()
    assert canonical_adjustment_set(g, "X", "Y") is None


def test_isolated_nodes_do_not_count_against_minimal_sets() -> None:
    padding = [f"I{i:02d}" for i in range(20)]
    g = CausalGraph.from_edges("Z -> X, Z -> Y, X -> Y", nodes=padding)
    assert minimal_adjustment_sets(g, "X", "Y") == sets("Z")
    assert canonical_adjustment_set(g, "X", "Y") == F("Z")
    assert admissible_set_exists(g, "X", "Y")
    # full enumeration would pad {Z} with every subset of the isolated nodes
    full = adjustment_sets(g, "X", "Y")
    assert isinstance(full, Unsupported) and full.detail["n_candidates"] == "21"
    assert adjustment_sets(g, "X", "Y", max_candidates=21) is not None
    # non-ancestors of {X, Y} that are not isolated are equally irrelevant
    g2 = CausalGraph.from_edges(
        "Z -> X, Z -> Y, X -> Y, "
        + ", ".join(f"Y -> D{i:02d}" for i in range(10))
        + ", "
        + ", ".join(f"Z -> S{i:02d}" for i in range(10))
    )
    assert minimal_adjustment_sets(g2, "X", "Y") == sets("Z")
    assert minimal_adjustment_sets(g2, "X", "Y", max_candidates=1) == sets("Z")


def test_selection_and_feedback_leave_answers_unchanged() -> None:
    for base in (CONFOUNDER, M_BIAS, BUTTERFLY, PEARL_3_4):
        x, y = ("Xi", "Xj") if base is PEARL_3_4 else ("X", "Y")
        variants = [
            base.with_selection(y),
            base.with_selection(*base.nodes),
            base.model_copy(update={"feedback": True}),
            base.with_selection(x).model_copy(update={"feedback": True}),
        ]
        for g in variants:
            assert g != base
            for measured_only in (True, False):
                assert adjustment_sets(g, x, y, measured_only=measured_only) == adjustment_sets(
                    base, x, y, measured_only=measured_only
                )
                assert minimal_adjustment_sets(
                    g, x, y, measured_only=measured_only
                ) == minimal_adjustment_sets(base, x, y, measured_only=measured_only)
                assert canonical_adjustment_set(
                    g, x, y, measured_only=measured_only
                ) == canonical_adjustment_set(base, x, y, measured_only=measured_only)
            assert requires_unmeasured(g, x, y) == requires_unmeasured(base, x, y)
            assert roles(g, x, y) == roles(base, x, y)
            assert assign_roles(g, x, y).roles == assign_roles(base, x, y).roles


# -- ordering and determinism ------------------------------------------------------------


def test_sets_are_sorted_by_size_then_lexicographically() -> None:
    out = adjustment_sets(M_BIAS, "X", "Y", measured_only=False)
    assert isinstance(out, tuple)
    keys = [(len(s), tuple(sorted(s))) for s in out]
    assert keys == sorted(keys)
    assert keys[:3] == [(0, ()), (1, ("U1",)), (1, ("U2",))]
    # edge order in the source text does not change the answer
    shuffled = CausalGraph.from_edges(
        "X -> Y, U2 -> Y, U2 -> M, U1 -> M, U1 -> X", unmeasured=["U2", "U1"]
    )
    assert shuffled == M_BIAS
    assert adjustment_sets(shuffled, "X", "Y", measured_only=False) == out


def test_two_confounders_and_an_upstream_cause() -> None:
    g = CausalGraph.from_edges("A -> X, A -> Y, B -> X, B -> Y, W -> A, X -> Y")
    assert adjustment_sets(g, "X", "Y") == sets("A B", "A B W")
    assert minimal_adjustment_sets(g, "X", "Y") == sets("A B")
    assert canonical_adjustment_set(g, "X", "Y") == F({"A", "B", "W"})
    r = roles(g, "X", "Y")
    assert r["A"] == r["B"] == r["W"] == "confounder"


# -- RoleAssignment spec ---------------------------------------------------------------


def test_role_assignment_roundtrips_and_hashes_graph() -> None:
    ra = assign_roles(M_BIAS, "X", "Y")
    assert isinstance(ra, RoleAssignment)
    assert ra.graph_hash == M_BIAS.content_hash()
    assert ra.treatment == "X" and ra.outcome == "Y"
    assert ra.roles == roles(M_BIAS, "X", "Y")
    assert ra.with_role("collider") == ("M",)
    assert ra.with_role("instrument") == ("U1",)
    assert ra.with_role("mediator") == ()
    assert RoleAssignment.from_json(ra.to_json()) == ra
    assert assign_roles(M_BIAS, "X", "Y").content_hash() == ra.content_hash()
    assert assign_roles(M_BIAS.with_unmeasured("M"), "X", "Y").graph_hash != ra.graph_hash


# -- brute-force cross-checks on random DAGs ---------------------------------------------


def _random_dag(
    rng: np.random.Generator, n: int, p: float, p_bi: float
) -> tuple[CausalGraph, str, str]:
    """A random semi-Markovian DAG plus a measured treatment–outcome pair."""
    names = [f"N{i}" for i in range(n)]
    order = rng.permutation(n)
    edges: list[str] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = names[order[i]], names[order[j]]
            if rng.random() < p:
                edges.append(f"{a} -> {b}")
            elif rng.random() < p_bi:
                edges.append(f"{a} <-> {b}")
    x, y = (str(v) for v in rng.choice(names, size=2, replace=False))
    unmeasured = [nm for nm in names if nm not in (x, y) and rng.random() < 0.25]
    return CausalGraph.from_edges(", ".join(edges), nodes=names, unmeasured=unmeasured), x, y


def _brute_force(g: CausalGraph, x: str, y: str, *, measured_only: bool) -> set[frozenset[str]]:
    pool = (g.measured if measured_only else set(g.nodes)) - g.descendants(x, include_self=True)
    pool -= {y}
    ordered = sorted(pool)
    return {
        F(c)
        for k in range(len(ordered) + 1)
        for c in combinations(ordered, k)
        if backdoor_admissible(g, x, y, c)
    }


@pytest.mark.parametrize("seed", range(40))
def test_enumeration_and_existence_agree_with_brute_force(seed: int) -> None:
    rng = np.random.default_rng(seed)
    g, x, y = _random_dag(rng, n=int(rng.integers(3, 8)), p=0.35, p_bi=0.15)
    for measured_only in (True, False):
        expected = _brute_force(g, x, y, measured_only=measured_only)
        got = adjustment_sets(g, x, y, measured_only=measured_only)
        assert isinstance(got, tuple)
        assert set(got) == expected
        assert list(got) == sorted(got, key=lambda s: (len(s), tuple(sorted(s))))
        assert admissible_set_exists(g, x, y, measured_only=measured_only) == bool(expected)
        mins = minimal_adjustment_sets(g, x, y, measured_only=measured_only)
        assert isinstance(mins, tuple)
        assert set(mins) == {s for s in expected if not any(t < s for t in expected)}
        canonical = canonical_adjustment_set(g, x, y, measured_only=measured_only)
        if expected:
            assert canonical in expected
            assert all(m <= canonical for m in mins)
            # minimality is decided inside the canonical set, whatever the pool size
            assert (
                minimal_adjustment_sets(
                    g, x, y, measured_only=measured_only, max_candidates=len(canonical)
                )
                == mins
            )
        else:
            assert canonical is None
        for s in got:
            assert not (s & g.descendants(x)) and x not in s and y not in s
    assert requires_unmeasured(g, x, y) == (
        admissible_set_exists(g, x, y, measured_only=False)
        and not admissible_set_exists(g, x, y, measured_only=True)
    )


@pytest.mark.parametrize("seed", range(60))
def test_role_invariants_on_random_dags(seed: int) -> None:
    rng = np.random.default_rng(seed)
    g, x, y = _random_dag(rng, n=int(rng.integers(3, 9)), p=0.35, p_bi=0.2)
    g_under = g.remove_edges_out_of(x)
    anc_x, anc_y_under = g.ancestors(x), g_under.ancestors(y)
    r = roles(g, x, y)
    assert set(r) == set(g.nodes)
    assert r[x] == "treatment" and r[y] == "outcome"
    assert set(r.values()) <= set(ROLE_PRECEDENCE)
    # instruments are exactly frontdoor's unconditional instruments
    assert instrument_nodes(g, x, y) == frontdoor.instruments(g, x, y, measured_only=False)
    for n, role in r.items():
        if n in (x, y):
            continue
        # confounder iff common cause (mediators cannot be ancestors of x)
        assert (role == "confounder") == (n in anc_x and n in anc_y_under)
        if role == "mediator":
            assert n in g.descendants(x) and n in g.ancestors(y)
        if role == "instrument":
            assert n not in g.descendants(x)
            assert not g.d_separated(n, x) and g_under.d_separated(n, y)
        if role == "collider":
            # never a non-collider on a back-door path ...
            assert n not in anc_x and n not in anc_y_under
            # ... and conditioning on it alone d-connects x and y through it; the walk
            # x -> n <- b needs the edge out of x kept when n is a child of x
            g_walk = CausalGraph(
                nodes=g.nodes,
                edges=[*g_under.edges, *([(x, n)] if x in g.parents(n) else [])],
                bidirected=g.bidirected,
            )
            assert not g_walk.d_separated(x, y, {n})
            # two distinct parents (latent ones included) on opposite sides
            parents = set(g.parents(n)) | {f"<{s}>" for s in g.siblings(n)}
            assert len(parents) >= 2
        if role == "proxy":
            assert g.is_measured(n) and n not in g.descendants(x)
        if role == "descendant_of_outcome":
            assert n in g.descendants(y)
