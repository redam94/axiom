from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from axiom.core import Unsupported, is_failure
from axiom.identify.frontdoor import (
    FrontDoorRoute,
    InstrumentRoute,
    conditional_instruments,
    frontdoor_admissible,
    frontdoor_sets,
    instrument_admissible,
    instruments,
)
from axiom.identify.graph import CausalGraph, GraphError

# -- front-door ----------------------------------------------------------------------------


def test_classic_frontdoor() -> None:
    g = CausalGraph.from_edges("X -> M, M -> Y, X <-> Y")
    assert frontdoor_admissible(g, "X", "Y", ["M"])
    assert frontdoor_admissible(g, "X", "Y", "M")
    assert frontdoor_sets(g, "X", "Y") == (frozenset({"M"}),)


def test_mediator_with_backdoor_from_treatment_fails() -> None:
    g = CausalGraph.from_edges("X -> M, M -> Y, U -> X, U -> M, X <-> Y", unmeasured=["U"])
    assert not frontdoor_admissible(g, "X", "Y", ["M"])
    assert frontdoor_sets(g, "X", "Y") == ()
    # The latent is shown as a bidirected edge: same verdict.
    g2 = CausalGraph.from_edges("X -> M, M -> Y, X <-> M, X <-> Y")
    assert not frontdoor_admissible(g2, "X", "Y", ["M"])


def test_mediator_with_backdoor_to_outcome_fails() -> None:
    g = CausalGraph.from_edges("X -> M, M -> Y, M <-> Y, X <-> Y")
    assert not frontdoor_admissible(g, "X", "Y", ["M"])
    assert frontdoor_sets(g, "X", "Y") == ()


def test_parallel_mediators_only_admissible_as_a_pair() -> None:
    g = CausalGraph.from_edges("X -> M1, M1 -> Y, X -> M2, M2 -> Y, X <-> Y")
    assert not frontdoor_admissible(g, "X", "Y", ["M1"])
    assert not frontdoor_admissible(g, "X", "Y", ["M2"])
    assert frontdoor_admissible(g, "X", "Y", ["M1", "M2"])
    assert frontdoor_sets(g, "X", "Y") == (frozenset({"M1", "M2"}),)


def test_direct_edge_cannot_be_intercepted() -> None:
    g = CausalGraph.from_edges("X -> M, M -> Y, X -> Y, X <-> Y")
    assert not frontdoor_admissible(g, "X", "Y", ["M"])
    assert frontdoor_sets(g, "X", "Y") == ()


def test_off_path_node_can_complete_a_mediator_set() -> None:
    # W blocks M's back-door path to Y and is itself unconfounded with X.
    g = CausalGraph.from_edges("X -> M, M -> Y, W -> M, W -> Y, X <-> Y")
    assert not frontdoor_admissible(g, "X", "Y", ["M"])
    assert frontdoor_admissible(g, "X", "Y", ["M", "W"])
    assert frontdoor_sets(g, "X", "Y") == (frozenset({"M", "W"}),)


def test_chain_of_mediators_sorted_by_size_then_name() -> None:
    g = CausalGraph.from_edges("X -> A, A -> B, B -> Y, X <-> Y")
    sets = frontdoor_sets(g, "X", "Y")
    assert sets == (frozenset({"A"}), frozenset({"B"}), frozenset({"A", "B"}))


def test_null_effect_has_no_frontdoor_route() -> None:
    g = CausalGraph.from_edges("X <-> Y", nodes=["W"])
    assert not frontdoor_admissible(g, "X", "Y", ["W"])
    assert not frontdoor_admissible(g, "X", "Y", [])
    assert frontdoor_sets(g, "X", "Y") == ()


def test_frontdoor_measured_only_filter() -> None:
    g = CausalGraph.from_edges("X -> M, M -> Y, X <-> Y", unmeasured=["M"])
    assert frontdoor_admissible(g, "X", "Y", ["M"])  # purely graphical
    assert frontdoor_sets(g, "X", "Y") == ()
    assert frontdoor_sets(g, "X", "Y", measured_only=False) == (frozenset({"M"}),)


def test_frontdoor_overflow_is_unsupported() -> None:
    extra = ", ".join(f"X -> N{i}" for i in range(13))
    g = CausalGraph.from_edges(f"X -> M, M -> Y, X <-> Y, {extra}")
    res = frontdoor_sets(g, "X", "Y")
    assert isinstance(res, Unsupported) and is_failure(res)
    assert "max_candidates" in res.reason
    ok = frontdoor_sets(g, "X", "Y", max_candidates=14)
    assert isinstance(ok, tuple) and frozenset({"M"}) in ok


def test_frontdoor_argument_errors() -> None:
    g = CausalGraph.from_edges("X -> M, M -> Y")
    with pytest.raises(GraphError):
        frontdoor_admissible(g, "X", "Y", ["Q"])
    with pytest.raises(GraphError):
        frontdoor_admissible(g, "X", "X", ["M"])
    with pytest.raises(GraphError):
        frontdoor_admissible(g, "X", "Y", ["X"])
    with pytest.raises(GraphError):
        frontdoor_sets(g, "X", "Q")


def test_mediator_that_is_an_ancestor_of_treatment_is_rejected() -> None:
    # A sits upstream of X; it intercepts nothing and has an open back-door path from X.
    g = CausalGraph.from_edges("A -> X, X -> M, M -> Y, X <-> Y")
    assert not frontdoor_admissible(g, "X", "Y", ["A"])
    assert not frontdoor_admissible(g, "X", "Y", ["A", "M"])
    assert frontdoor_sets(g, "X", "Y") == (frozenset({"M"}),)


def test_mediator_downstream_of_outcome_is_rejected() -> None:
    # D is a child of Y: it intercepts nothing, and conditioning on it opens X <-> Y -> D.
    g = CausalGraph.from_edges("X -> M, M -> Y, Y -> D, X <-> Y")
    assert not frontdoor_admissible(g, "X", "Y", ["D"])
    assert not frontdoor_admissible(g, "X", "Y", ["D", "M"])
    assert frontdoor_sets(g, "X", "Y") == (frozenset({"M"}),)


def test_null_effect_short_circuits_before_the_candidate_bound() -> None:
    extra = ", ".join(f"N{i} -> Y" for i in range(13))
    g = CausalGraph.from_edges(f"X <-> Y, {extra}")
    assert "Y" not in g.descendants("X")
    assert frontdoor_sets(g, "X", "Y") == ()
    assert frontdoor_sets(g, "X", "Y", max_candidates=0) == ()


def _random_graph(
    rng: np.random.Generator, n: int, *, p_dir: float = 0.4, p_bi: float = 0.15
) -> CausalGraph:
    """A random semi-Markovian DAG whose causal order is a random permutation of the names."""
    names = [f"V{i}" for i in range(n)]
    perm = rng.permutation(n)
    directed = [
        (names[perm[i]], names[perm[j]])
        for i in range(n)
        for j in range(i + 1, n)
        if rng.random() < p_dir
    ]
    bidirected = [
        (names[i], names[j]) for i in range(n) for j in range(i + 1, n) if rng.random() < p_bi
    ]
    return CausalGraph(nodes=names, edges=directed, bidirected=bidirected)


def _random_pair(rng: np.random.Generator, g: CausalGraph) -> tuple[str, str]:
    x, y = rng.choice(list(g.nodes), size=2, replace=False)
    return str(x), str(y)


def _set_formulation(g: CausalGraph, x: str, y: str, m: frozenset[str]) -> bool:
    """(i), (ii) as in the module; (iii) as do-calculus rule 2: (M ⊥ y | x) in G_underline_M."""
    if y not in g.descendants(x):
        return False
    g_m = g
    for node in m:
        g_m = g_m.remove_edges_out_of(node)
    if y in g_m.descendants(x):
        return False
    if not g.remove_edges_out_of(x).d_separated(x, m):
        return False
    return g_m.d_separated(m, y, {x})


def test_per_mediator_condition_matches_set_formulation() -> None:
    rng = np.random.default_rng(20260821)
    checked = 0
    for _ in range(250):
        g = _random_graph(rng, 6)
        x, y = _random_pair(rng, g)
        if y not in g.descendants(x):
            continue
        others = sorted(set(g.nodes) - {x, y})
        for size in range(1, len(others) + 1):
            for combo in combinations(others, size):
                m = frozenset(combo)
                assert frontdoor_admissible(g, x, y, m) == _set_formulation(g, x, y, m), (
                    g.to_text(),
                    x,
                    y,
                    sorted(m),
                )
                checked += 1
    assert checked > 500


# -- linear-Gaussian oracle ----------------------------------------------------------------


def _linear_sem(
    g: CausalGraph, rng: np.random.Generator
) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """A random linear SEM on ``g``: ``(index, Sigma, (I - B)^-1)`` over nodes plus latents.

    Every bidirected edge becomes an explicit exogenous latent parent of both
    endpoints, so ``Sigma = (I - B)^-1 D (I - B)^-T`` with ``D`` diagonal, and
    the total effect of ``x`` on ``y`` is ``(I - B)^-1[y, x]``.
    """
    latents = [f"U[{a},{b}]" for a, b in g.bidirected]
    names = [*g.nodes, *latents]
    idx = {n: i for i, n in enumerate(names)}
    k = len(names)
    b = np.zeros((k, k))

    def coef() -> float:
        return float(rng.uniform(0.5, 1.5) * rng.choice([-1.0, 1.0]))

    for a, c in g.edges:
        b[idx[c], idx[a]] = coef()
    for (a, c), u in zip(g.bidirected, latents, strict=True):
        b[idx[a], idx[u]] = coef()
        b[idx[c], idx[u]] = coef()
    inv = np.linalg.inv(np.eye(k) - b)
    sigma = inv @ np.diag(rng.uniform(0.5, 1.5, size=k)) @ inv.T
    return idx, sigma, inv


def _partial_cov(sigma: np.ndarray, a: int, b: int, w: list[int]) -> float:
    if not w:
        return float(sigma[a, b])
    return float(sigma[a, b] - sigma[a, w] @ np.linalg.solve(sigma[np.ix_(w, w)], sigma[b, w]))


def _frontdoor_functional(
    sigma: np.ndarray, idx: dict[str, int], x: str, y: str, m: list[str]
) -> float:
    """Linear front-door: sum over m of (slope of m on x) * (coef of m in y ~ M + x)."""
    xi, yi = idx[x], idx[y]
    mi = [idx[n] for n in m]
    regressors = [*mi, xi]
    coef = np.linalg.solve(sigma[np.ix_(regressors, regressors)], sigma[regressors, yi])
    gamma = coef[: len(mi)]
    slope_m_on_x = sigma[mi, xi] / sigma[xi, xi]
    return float(gamma @ slope_m_on_x)


def test_oracle_every_admissible_route_recovers_the_total_effect() -> None:
    rng = np.random.default_rng(20260822)
    fd_routes = iv_routes = 0
    for _ in range(300):
        n = int(rng.integers(4, 8))
        g = _random_graph(
            rng, n, p_dir=float(rng.uniform(0.25, 0.5)), p_bi=float(rng.uniform(0.1, 0.3))
        )
        x, y = _random_pair(rng, g)
        idx, sigma, inv = _linear_sem(g, rng)
        tau = float(inv[idx[y], idx[x]])
        sets = frontdoor_sets(g, x, y, max_candidates=20)
        assert isinstance(sets, tuple)
        for m in sets:
            est = _frontdoor_functional(sigma, idx, x, y, sorted(m))
            assert est == pytest.approx(tau, abs=1e-6), (g.to_text(), x, y, sorted(m), est, tau)
            fd_routes += 1
        res = conditional_instruments(g, x, y, max_candidates=20, max_conditioning=3)
        assert isinstance(res, tuple)
        for z, w in res:
            wi = [idx[node] for node in sorted(w)]
            den = _partial_cov(sigma, idx[z], idx[x], wi)
            assert abs(den) > 1e-9, ("relevance claimed but cov(z, x | W) = 0", g.to_text(), z, w)
            est = _partial_cov(sigma, idx[z], idx[y], wi) / den
            assert est == pytest.approx(tau, abs=1e-6), (g.to_text(), x, y, z, sorted(w), est, tau)
            iv_routes += 1
    assert fd_routes > 20, fd_routes
    assert iv_routes > 100, iv_routes


# -- instruments ---------------------------------------------------------------------------


def test_classic_instrument() -> None:
    g = CausalGraph.from_edges("Z -> X, X -> Y, X <-> Y")
    assert instrument_admissible(g, "X", "Y", "Z")
    assert instruments(g, "X", "Y") == ("Z",)
    assert conditional_instruments(g, "X", "Y") == (("Z", frozenset()),)


def test_instrument_linked_to_treatment_by_a_latent_is_valid() -> None:
    # Z <-> X is relevance through a common cause; Z still has no open path to Y once
    # the edges out of X are cut, so it identifies the total effect.
    g = CausalGraph.from_edges("Z <-> X, X -> Y, X <-> Y")
    assert instrument_admissible(g, "X", "Y", "Z")
    assert instruments(g, "X", "Y") == ("Z",)


def test_conditional_instrument_via_an_opened_collider() -> None:
    # Z is marginally independent of X; conditioning on the collider C makes it relevant
    # (Z -> C <- A -> X) without opening any path to Y in G_underline_X.
    g = CausalGraph.from_edges("Z -> C, A -> C, A -> X, X -> Y, X <-> Y")
    assert not instrument_admissible(g, "X", "Y", "Z")
    assert instrument_admissible(g, "X", "Y", "Z", ["C"])
    # A and its proxy C are unconditional instruments; Z only becomes one given C.
    assert instruments(g, "X", "Y") == ("A", "C")
    res = conditional_instruments(g, "X", "Y")
    assert isinstance(res, tuple)
    assert ("Z", frozenset({"C"})) in res
    assert ("Z", frozenset()) not in res


def test_instrument_admissible_when_outcome_is_not_a_descendant_of_treatment() -> None:
    # A structurally null effect: Z is relevant and excluded, so the IV estimand is 0 / b.
    g = CausalGraph.from_edges("Z -> X, X <-> Y")
    assert "Y" not in g.descendants("X")
    assert instrument_admissible(g, "X", "Y", "Z")
    assert instruments(g, "X", "Y") == ("Z",)
    assert conditional_instruments(g, "X", "Y") == (("Z", frozenset()),)


def test_conditioning_candidates_exclude_descendants_of_outcome() -> None:
    g = CausalGraph.from_edges("Z -> X, X -> Y, Y -> D, X <-> Y, A -> Y")
    assert not instrument_admissible(g, "X", "Y", "Z", ["D"])
    res = conditional_instruments(g, "X", "Y")
    assert res == (("Z", frozenset()), ("Z", frozenset({"A"})))


def test_exclusion_violation_is_not_an_instrument() -> None:
    g = CausalGraph.from_edges("Z -> X, Z -> Y, X -> Y, X <-> Y")
    assert not instrument_admissible(g, "X", "Y", "Z")
    assert instruments(g, "X", "Y") == ()


def test_confounded_instrument_is_not_an_instrument() -> None:
    g = CausalGraph.from_edges("Z -> X, X -> Y, Z <-> Y")
    assert not instrument_admissible(g, "X", "Y", "Z")
    assert instruments(g, "X", "Y") == ()


def test_irrelevant_candidate_is_not_an_instrument() -> None:
    g = CausalGraph.from_edges("X -> Y, X <-> Y", nodes=["Z"])
    assert not instrument_admissible(g, "X", "Y", "Z")


def test_instrument_through_a_mediator_is_for_the_total_effect() -> None:
    g = CausalGraph.from_edges("Z -> X, X -> M, M -> Y, X <-> Y")
    assert instrument_admissible(g, "X", "Y", "Z")
    assert instruments(g, "X", "Y") == ("Z",)


def test_descendant_of_treatment_is_never_an_instrument() -> None:
    # Without condition (c), Z would pass: it is d-connected to X and isolated in G_underline_X.
    g = CausalGraph.from_edges("X -> Z, X -> Y, X <-> Y")
    assert not instrument_admissible(g, "X", "Y", "Z")
    assert instruments(g, "X", "Y") == ()


def test_conditioning_on_a_descendant_of_treatment_is_rejected() -> None:
    g = CausalGraph.from_edges("Z -> X, X -> W, X -> Y, X <-> Y")
    assert instrument_admissible(g, "X", "Y", "Z")
    assert not instrument_admissible(g, "X", "Y", "Z", ["W"])
    assert conditional_instruments(g, "X", "Y") == (("Z", frozenset()),)


def test_conditional_instrument() -> None:
    g = CausalGraph.from_edges("W -> Z, W -> Y, Z -> X, X -> Y, X <-> Y")
    assert not instrument_admissible(g, "X", "Y", "Z")
    assert instrument_admissible(g, "X", "Y", "Z", ["W"])
    assert instrument_admissible(g, "X", "Y", "Z", "W")
    assert instruments(g, "X", "Y") == ()
    assert conditional_instruments(g, "X", "Y") == (("Z", frozenset({"W"})),)


def test_conditional_instruments_include_supersets_up_to_bound() -> None:
    g = CausalGraph.from_edges("W -> Z, W -> Y, Z -> X, X -> Y, X <-> Y, A -> Y, B -> Y")
    res = conditional_instruments(g, "X", "Y")
    assert isinstance(res, tuple)
    assert res == (
        ("Z", frozenset({"W"})),
        ("Z", frozenset({"A", "W"})),
        ("Z", frozenset({"B", "W"})),
    )
    res1 = conditional_instruments(g, "X", "Y", max_conditioning=1)
    assert res1 == (("Z", frozenset({"W"})),)


def test_instruments_are_sorted() -> None:
    g = CausalGraph.from_edges("B -> X, A -> X, X -> Y, X <-> Y")
    assert instruments(g, "X", "Y") == ("A", "B")


def test_instrument_measured_only_filter() -> None:
    g = CausalGraph.from_edges("Z -> X, X -> Y, X <-> Y", unmeasured=["Z"])
    assert instrument_admissible(g, "X", "Y", "Z")
    assert instruments(g, "X", "Y") == ()
    assert instruments(g, "X", "Y", measured_only=False) == ("Z",)
    assert conditional_instruments(g, "X", "Y") == ()
    assert conditional_instruments(g, "X", "Y", measured_only=False) == (("Z", frozenset()),)


def test_conditional_instruments_overflow_is_unsupported() -> None:
    extra = ", ".join(f"N{i} -> Y" for i in range(13))
    g = CausalGraph.from_edges(f"Z -> X, X -> Y, X <-> Y, {extra}")
    res = conditional_instruments(g, "X", "Y")
    assert isinstance(res, Unsupported) and is_failure(res)
    ok = conditional_instruments(g, "X", "Y", max_candidates=14, max_conditioning=1)
    assert isinstance(ok, tuple) and ok[0] == ("Z", frozenset())


def test_instrument_argument_errors() -> None:
    g = CausalGraph.from_edges("Z -> X, X -> Y, W -> Y")
    with pytest.raises(GraphError):
        instrument_admissible(g, "X", "Y", "Q")
    with pytest.raises(GraphError):
        instrument_admissible(g, "X", "Y", "X")
    with pytest.raises(GraphError):
        instrument_admissible(g, "X", "Y", "Z", ["Y"])
    with pytest.raises(GraphError):
        instrument_admissible(g, "X", "Y", "Z", ["Z"])
    with pytest.raises(GraphError):
        instruments(g, "X", "X")


# -- route records -------------------------------------------------------------------------


def test_frontdoor_route_record() -> None:
    r = FrontDoorRoute(mediators=("M2", "M1", "M1"), treatment="X", outcome="Y")
    assert r.mediators == ("M1", "M2")
    assert FrontDoorRoute.from_json(r.to_json()) == r
    with pytest.raises(ValueError):
        FrontDoorRoute(mediators=(), treatment="X", outcome="Y")
    with pytest.raises(ValueError):
        FrontDoorRoute(mediators=("X",), treatment="X", outcome="Y")
    with pytest.raises(ValueError):
        FrontDoorRoute(mediators=("M",), treatment="X", outcome="X")
    with pytest.raises(ValueError, match="mediators"):
        FrontDoorRoute(mediators=None, treatment="X", outcome="Y")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mediators"):
        FrontDoorRoute(mediators=3, treatment="X", outcome="Y")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mediators"):
        FrontDoorRoute(mediators=(1, 2), treatment="X", outcome="Y")  # type: ignore[arg-type]
    assert FrontDoorRoute(mediators="M", treatment="X", outcome="Y").mediators == ("M",)


def test_instrument_route_record() -> None:
    r = InstrumentRoute(instrument="Z", treatment="X", outcome="Y")
    assert r.conditioning == ()
    c = InstrumentRoute(instrument="Z", conditioning=["W2", "W1"], treatment="X", outcome="Y")
    assert c.conditioning == ("W1", "W2")
    assert InstrumentRoute.from_json(c.to_json()) == c
    assert r.content_hash() != c.content_hash()
    with pytest.raises(ValueError):
        InstrumentRoute(instrument="X", treatment="X", outcome="Y")
    with pytest.raises(ValueError):
        InstrumentRoute(instrument="Z", conditioning=("Z",), treatment="X", outcome="Y")
    with pytest.raises(ValueError):
        InstrumentRoute(instrument="Z", treatment="X", outcome="X")
    with pytest.raises(ValueError, match="conditioning"):
        InstrumentRoute(
            instrument="Z", conditioning=None, treatment="X", outcome="Y"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="conditioning"):
        InstrumentRoute(
            instrument="Z", conditioning=1.5, treatment="X", outcome="Y"  # type: ignore[arg-type]
        )
    one = InstrumentRoute(instrument="Z", conditioning="W", treatment="X", outcome="Y")
    assert one.conditioning == ("W",)
