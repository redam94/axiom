"""surface.optimize / surface.frontier: allocation on a toy two-treatment Hill surface.

The toy satisfies ``SupportsForward`` with an expression tree built from
``HillKernel`` nodes (one forward): ``y = a + Σ_i beta_i · hill(x_i / k_i; s_i)``.
Known ``theta`` lets every test compare against brute force or closed form.
The real ``Surface`` (``SurfaceSpec``) is exercised for what the toy cannot
show: carryover refusal and ``steady_state()``, per-unit intercepts through
``context``, and column validation against what the tree reads.
"""

from __future__ import annotations

import importlib
import itertools
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pytest

from axiom.core import (
    Add,
    D,
    Data,
    DesignMatrix,
    Model,
    Mul,
    Outcome,
    Param,
    Posterior,
    Treatment,
    Unsupported,
    Unverified,
    value,
)
from axiom.surface.carryover import GeometricCarryover
from axiom.surface.design import Bounds
from axiom.surface.frontier import Frontier, frontier
from axiom.surface.kernels import HillKernel
from axiom.surface.model import Surface, SurfaceSpec
from axiom.surface.optimize import Allocation, allocate

NAMES = ("x1", "x2")


class HillToy:
    """``y = a + Σ_i beta_i · hill_i(x_i)`` (+ optional ``gamma · hill_1 · hill_2``)."""

    def __init__(self, treatments: tuple[str, ...], *, interaction: bool = False) -> None:
        kernel = HillKernel()
        self.treatments = treatments
        terms: list[Model] = [Param(name="a", dimension=D.outcome)]
        sats = []
        for t in treatments:
            dose = Data(name=t, dimension=D.currency)
            terms.append(kernel.response(dose, t))
            sats.append(kernel.saturation(dose, t))
        if interaction:
            terms.append(Mul(factors=(Param(name="gamma", dimension=D.outcome), sats[0], sats[1])))
        self._expr: Model = Add(terms=tuple(terms))

    @property
    def expr(self) -> Model:
        return self._expr

    def forward(
        self, dose: Mapping[str, npt.ArrayLike], theta: Mapping[str, npt.ArrayLike]
    ) -> npt.NDArray[np.float64]:
        return value(self._expr, data=dose, params=theta)

    def linearize(
        self, dose: Mapping[str, npt.ArrayLike], theta_at: Mapping[str, npt.ArrayLike]
    ) -> DesignMatrix:  # pragma: no cover - not exercised by the allocator
        n = np.asarray(next(iter(dose.values()))).size
        return DesignMatrix(X=np.ones((n, 1)), columns=("a",), offset=np.zeros(n), at={})


TOY = HillToy(NAMES)
THETA = {
    "a": 1.0,
    "k_x1": 2.0,
    "s_x1": 1.0,
    "beta_x1": 10.0,
    "k_x2": 4.0,
    "s_x2": 1.0,
    "beta_x2": 8.0,
}
BOUNDS = Bounds(treatments=NAMES, low=(0.0, 0.0), high=(10.0, 10.0))


def _closed_form(x: npt.ArrayLike, theta: Mapping[str, float]) -> npt.NDArray[np.float64]:
    xv = np.asarray(x, dtype=float)
    out = np.full(xv.shape[:-1], theta["a"])
    for j, t in enumerate(NAMES):
        u = (xv[..., j] / theta[f"k_{t}"]) ** theta[f"s_{t}"]
        out = out + theta[f"beta_{t}"] * u / (1.0 + u)
    return np.asarray(out, dtype=np.float64)


def _grid_best(
    theta: Mapping[str, float], budget: float, bounds: Bounds, n: int = 801
) -> tuple[npt.NDArray[np.float64], float, float]:
    axes = [np.linspace(lo, hi, n) for lo, hi in zip(bounds.low, bounds.high, strict=True)]
    pts = np.asarray(list(itertools.product(*axes)))
    pts = pts[pts.sum(axis=1) <= budget + 1e-12]
    vals = _closed_form(pts, theta)
    i = int(np.argmax(vals))
    step = max(float(a[1] - a[0]) for a in axes)
    return pts[i], float(vals[i]), step


def test_toy_forward_matches_closed_form() -> None:
    x = np.array([[0.0, 0.0], [1.0, 3.0], [5.0, 5.0]])
    dose = {"x1": x[:, 0], "x2": x[:, 1]}
    np.testing.assert_allclose(TOY.forward(dose, THETA), _closed_form(x, THETA), rtol=1e-12)


def test_matches_brute_force_grid_and_budget_binds() -> None:
    budget = 6.0
    alloc = allocate(TOY, THETA, budget=budget, bounds=BOUNDS)
    assert isinstance(alloc, Allocation)
    grid_x, grid_v, step = _grid_best(THETA, budget, BOUNDS)
    # the grid is a subset of the feasible set: the solver can only do at least as well
    assert alloc.expected_outcome >= grid_v - 1e-9
    np.testing.assert_allclose(alloc.as_array(), grid_x, atol=step)
    assert alloc.expected_outcome == pytest.approx(
        float(_closed_form(alloc.as_array(), THETA)), rel=1e-12
    )
    # monotone surface: the whole budget is spent
    assert abs(alloc.slack) < 1e-6
    assert alloc.status == "converged" and alloc.method == "slsqp"
    assert alloc.detail["n_draws"] == 1.0


def test_kkt_equal_marginal_value_at_interior_optimum() -> None:
    budget = 6.0
    alloc = allocate(TOY, THETA, budget=budget, bounds=BOUNDS, tol=1e-14)
    assert isinstance(alloc, Allocation)
    x = alloc.as_array()
    # d/dx [beta u/(1+u)] with s = 1 is beta k / (k + x)^2; equal across treatments at the optimum
    m1 = THETA["beta_x1"] * THETA["k_x1"] / (THETA["k_x1"] + x[0]) ** 2
    m2 = THETA["beta_x2"] * THETA["k_x2"] / (THETA["k_x2"] + x[1]) ** 2
    assert m1 == pytest.approx(m2, rel=1e-5)


def test_bounds_are_respected() -> None:
    tight = Bounds(treatments=NAMES, low=(0.5, 0.0), high=(1.5, 10.0))
    alloc = allocate(TOY, THETA, budget=6.0, bounds=tight)
    assert isinstance(alloc, Allocation)
    x = alloc.as_array()
    assert np.all(x >= np.asarray(tight.low) - 1e-12) and np.all(
        x <= np.asarray(tight.high) + 1e-12
    )
    # x1 is more valuable at the margin than x2 at these doses, so its upper bound is hit
    assert x[0] == pytest.approx(1.5, abs=1e-7)
    assert x[1] == pytest.approx(4.5, abs=1e-6)


def test_symmetric_treatments_split_evenly() -> None:
    theta = dict(THETA, k_x2=2.0, beta_x2=10.0)
    alloc = allocate(TOY, theta, budget=5.0, bounds=BOUNDS)
    assert isinstance(alloc, Allocation)
    assert alloc.doses["x1"] == pytest.approx(alloc.doses["x2"], abs=1e-6)
    assert alloc.total_dose == pytest.approx(5.0, abs=1e-8)


def test_non_convergence_surfaces_as_unsupported() -> None:
    start = {"x1": 0.1, "x2": 5.9}
    result = allocate(TOY, THETA, budget=6.0, bounds=BOUNDS, maxiter=1, start=start)
    assert isinstance(result, Unsupported)
    assert "did not converge" in result.reason and "SLSQP" in result.reason
    assert "Iteration limit" in result.reason
    assert result.detail["method"] == "slsqp" and result.detail["maxiter"] == "1"


def test_infeasible_budget_is_unsupported() -> None:
    result = allocate(
        TOY, THETA, budget=1.0, bounds=Bounds(treatments=NAMES, low=(1.0, 1.0), high=(5.0, 5.0))
    )
    assert isinstance(result, Unsupported)
    assert "lower bounds" in result.reason


def test_typed_failure_posterior_propagates() -> None:
    bad = Unverified(reason="fit did not converge")
    assert allocate(TOY, bad, budget=6.0, bounds=BOUNDS) is bad
    assert frontier(TOY, bad, [1.0, 2.0], BOUNDS) is bad


def test_warm_start_and_multistart_agree() -> None:
    base = allocate(TOY, THETA, budget=6.0, bounds=BOUNDS)
    assert isinstance(base, Allocation)
    warm = allocate(TOY, THETA, budget=6.0, bounds=BOUNDS, start={"x1": 5.0, "x2": 1.0})
    multi = allocate(TOY, THETA, budget=6.0, bounds=BOUNDS, n_starts=4, seed=3)
    assert isinstance(warm, Allocation) and isinstance(multi, Allocation)
    # SLSQP stops on the objective (ftol), so doses agree to ~sqrt(ftol / curvature)
    np.testing.assert_allclose(warm.as_array(), base.as_array(), atol=1e-5)
    np.testing.assert_allclose(multi.as_array(), base.as_array(), atol=1e-5)
    # 3 random starts on top of the fixed set (2 corners + centre, the pair split being the
    # centre at k = 2) and the 3 promoted pre-search points
    assert multi.detail["n_random_starts"] == 3.0 and multi.detail["n_starts"] == 9.0
    assert multi.detail["n_presearch"] == 8.0 and multi.detail["n_promoted"] == 3.0
    assert multi.seed == 3
    # a warm start from a larger budget is shrunk to feasibility, not rejected
    over = allocate(TOY, THETA, budget=6.0, bounds=BOUNDS, start={"x1": 9.0, "x2": 9.0})
    assert isinstance(over, Allocation)
    np.testing.assert_allclose(over.as_array(), base.as_array(), atol=1e-5)


def test_zero_valued_start_point_is_scaled_sensibly() -> None:
    # f(start) == 0 must not collapse the objective scale (it is taken from a probe of the box)
    zero = allocate(
        TOY, dict(THETA, a=0.0), budget=6.0, bounds=BOUNDS, start={"x1": 0.0, "x2": 0.0}
    )
    base = allocate(TOY, THETA, budget=6.0, bounds=BOUNDS)
    assert isinstance(zero, Allocation) and isinstance(base, Allocation)
    np.testing.assert_allclose(zero.as_array(), base.as_array(), atol=1e-5)


def test_multistart_escapes_local_optimum_on_sigmoidal_surface() -> None:
    # s = 3: each response is sigmoidal, and a small budget is best spent all on one treatment
    theta = dict(THETA, s_x1=3.0, s_x2=3.0, k_x1=2.0, k_x2=2.0, beta_x1=10.0, beta_x2=10.0)
    budget = 3.0
    grid_x, grid_v, step = _grid_best(theta, budget, BOUNDS, n=601)
    assert min(grid_x) < step  # the brute-force optimum is a corner, not the even split
    multi = allocate(TOY, theta, budget=budget, bounds=BOUNDS, n_starts=8, seed=0)
    assert isinstance(multi, Allocation)
    assert multi.expected_outcome >= grid_v - 1e-9
    assert min(multi.as_array()) < step


def _posterior(n: int, seed: int, *, sd: float = 2.0) -> Posterior:
    rng = np.random.default_rng(seed)
    draws = {k: np.full((2, n // 2), v) for k, v in THETA.items()}
    draws["beta_x1"] = np.asarray(THETA["beta_x1"] + sd * rng.standard_normal((2, n // 2)))
    return Posterior(draws, provenance={"seed": seed})


def test_posterior_mean_objective_equals_plug_in_mean_when_linear_in_the_uncertain_parameter() -> (
    None
):
    post = _posterior(200, 1)
    alloc = allocate(TOY, post, budget=6.0, bounds=BOUNDS)
    assert isinstance(alloc, Allocation)
    assert alloc.detail["n_draws"] == 200.0
    beta_bar = float(post.flat("beta_x1").mean())
    plug = allocate(TOY, dict(THETA, beta_x1=beta_bar), budget=6.0, bounds=BOUNDS)
    assert isinstance(plug, Allocation)
    np.testing.assert_allclose(alloc.as_array(), plug.as_array(), atol=1e-6)
    assert alloc.expected_outcome == pytest.approx(plug.expected_outcome, rel=1e-10)


def test_quantile_objective_is_risk_averse_on_the_uncertain_treatment() -> None:
    post = _posterior(400, 2, sd=3.0)
    mean = allocate(TOY, post, budget=6.0, bounds=BOUNDS, objective="mean")
    low = allocate(TOY, post, budget=6.0, bounds=BOUNDS, objective="quantile", q=0.1)
    assert isinstance(mean, Allocation) and isinstance(low, Allocation)
    assert low.objective == "quantile" and low.q == 0.1
    assert low.expected_outcome < mean.expected_outcome
    # the 10% quantile of beta_x1 is below its mean, so the pessimistic plan shifts dose to x2
    assert low.doses["x1"] < mean.doses["x1"] - 0.05
    assert abs(low.slack) < 1e-6


def test_draw_subsampling_is_seeded_and_recorded() -> None:
    post = _posterior(300, 5)
    a = allocate(TOY, post, budget=6.0, bounds=BOUNDS, n_draws=50, seed=11)
    b = allocate(TOY, post, budget=6.0, bounds=BOUNDS, n_draws=50, seed=11)
    c = allocate(TOY, post, budget=6.0, bounds=BOUNDS, n_draws=50, seed=12)
    assert isinstance(a, Allocation) and isinstance(b, Allocation) and isinstance(c, Allocation)
    assert a.detail["n_draws"] == 50.0 and a.seed == 11
    assert a.as_array().tolist() == b.as_array().tolist()
    assert a.expected_outcome != c.expected_outcome
    with pytest.raises(ValueError, match="n_draws"):
        allocate(TOY, post, budget=6.0, bounds=BOUNDS, n_draws=0)


def test_vector_parameter_draws_take_the_per_draw_path() -> None:
    post = _posterior(60, 7)
    draws = {k: post.draws(k) for k in post.names()}
    draws["unused_vec"] = np.zeros((2, 30, 3))  # a vector parameter forces one forward per draw
    with_vec = Posterior(draws)
    a = allocate(TOY, post, budget=6.0, bounds=BOUNDS)
    b = allocate(TOY, with_vec, budget=6.0, bounds=BOUNDS)
    assert isinstance(a, Allocation) and isinstance(b, Allocation)
    np.testing.assert_allclose(a.as_array(), b.as_array(), atol=1e-8)
    assert a.expected_outcome == pytest.approx(b.expected_outcome, rel=1e-12)


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="q must"):
        allocate(TOY, THETA, budget=6.0, bounds=BOUNDS, objective="quantile", q=1.0)
    # q is validated whatever the objective: a silently ignored bad q is a latent wrong number
    with pytest.raises(ValueError, match="q must"):
        allocate(TOY, THETA, budget=6.0, bounds=BOUNDS, objective="mean", q=0.0)
    with pytest.raises(ValueError, match="q must"):
        frontier(TOY, THETA, [1.0], BOUNDS, q=1.5)
    with pytest.raises(ValueError, match="n_starts"):
        allocate(TOY, THETA, budget=6.0, bounds=BOUNDS, n_starts=0)
    with pytest.raises(ValueError, match="start lacks"):
        allocate(TOY, THETA, budget=6.0, bounds=BOUNDS, start={"x1": 1.0})
    with pytest.raises(ValueError, match="unknown method"):
        allocate(TOY, THETA, budget=6.0, bounds=BOUNDS, method="simplex")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="budget must be finite"):
        allocate(TOY, THETA, budget=float("inf"), bounds=BOUNDS)


def test_allocation_spec_round_trip() -> None:
    alloc = allocate(TOY, THETA, budget=6.0, bounds=BOUNDS, seed=4)
    assert isinstance(alloc, Allocation)
    again = Allocation.from_json(alloc.to_json())
    assert again == alloc and again.content_hash() == alloc.content_hash()
    assert again.treatments == NAMES
    with pytest.raises(ValueError, match="finite"):
        Allocation(
            doses={"x1": float("nan")},
            expected_outcome=1.0,
            budget=1.0,
            objective="mean",
            method="slsqp",
        )


def test_default_start_set_escapes_the_even_split_trap() -> None:
    # symmetric sigmoidal responses (s = 3): at budgets 2 and 2.5 the even split is a stationary
    # point SLSQP accepts from a proportional-fill start, but all-in on one treatment is better
    # ({1, 1} -> 1 + 2 * 10 * (1/8)/(9/8) = 3.22 vs {0, 2} -> 1 + 10 * 1/2 = 6.0)
    theta = dict(THETA, s_x1=3.0, s_x2=3.0, k_x1=2.0, k_x2=2.0, beta_x1=10.0, beta_x2=10.0)
    for budget in (2.0, 2.5):
        alloc = allocate(TOY, theta, budget=budget, bounds=BOUNDS)
        assert isinstance(alloc, Allocation)
        grid_x, grid_v, step = _grid_best(theta, budget, BOUNDS, n=601)
        assert min(grid_x) < step  # the brute-force optimum is a corner
        assert alloc.expected_outcome >= grid_v - 1e-9
        assert min(alloc.as_array()) < step
        # 2 corners + centre (= the pair split at k = 2) + 3 promoted pre-search points
        assert alloc.detail["n_starts"] == 6.0 and alloc.detail["n_random_starts"] == 0.0
        assert alloc.detail["n_presearch"] == 8.0 and alloc.detail["n_promoted"] == 3.0
    even = allocate(TOY, theta, budget=2.0, bounds=BOUNDS)
    assert isinstance(even, Allocation)
    assert even.expected_outcome == pytest.approx(6.0, abs=1e-9)
    # the frontier reaches the brute-force optimum at every budget, including the trap budgets
    budgets = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
    fr = frontier(TOY, theta, budgets, BOUNDS)
    assert isinstance(fr, Frontier)
    for b, y in zip(fr.budgets, fr.outcomes, strict=True):
        _, grid_v, _ = _grid_best(theta, b, BOUNDS, n=401)
        assert y >= grid_v - 1e-9
    assert np.all(np.diff(np.asarray(fr.outcomes)) > 0)


def test_an_intercept_that_dwarfs_the_range_does_not_stall_the_solver() -> None:
    # alpha = 1e9 against a response range of ~10: the objective scale is the spread of the probe
    # values (never the level), so the solver moves off the start and reaches the same optimum
    base = allocate(TOY, THETA, budget=6.0, bounds=BOUNDS)
    huge = allocate(TOY, dict(THETA, a=1e9), budget=6.0, bounds=BOUNDS)
    assert isinstance(base, Allocation) and isinstance(huge, Allocation)
    assert huge.detail["n_iter"] > 1.0
    np.testing.assert_allclose(huge.as_array(), base.as_array(), atol=2e-3)
    assert huge.expected_outcome - 1e9 == pytest.approx(base.expected_outcome - 1.0, abs=1e-5)
    assert abs(huge.slack) < 1e-6


def test_bounds_and_surface_columns_are_checked_both_ways() -> None:
    # a treatment the surface does not read would be silently funded
    extra = Bounds(treatments=("x1", "x2", "x9"), low=(0.0,) * 3, high=(10.0,) * 3)
    with pytest.raises(ValueError, match=r"does not read.*\['x9'\]"):
        allocate(TOY, THETA, budget=6.0, bounds=extra)
    # a column the surface reads but Bounds and context do not supply
    partial = Bounds(treatments=("x1",), low=(0.0,), high=(10.0,))
    with pytest.raises(ValueError, match=r"\['x2'\].*context="):
        allocate(TOY, THETA, budget=6.0, bounds=partial)
    # context may not override a treatment, supply an unread column, or hold more than one value
    with pytest.raises(ValueError, match="treatments in Bounds"):
        allocate(TOY, THETA, budget=6.0, bounds=BOUNDS, context={"x1": 1.0})
    with pytest.raises(ValueError, match=r"does not read.*\['t'\]"):
        allocate(TOY, THETA, budget=6.0, bounds=BOUNDS, context={"t": 1.0})
    fixed = allocate(TOY, THETA, budget=6.0, bounds=partial, context={"x2": 2.0})
    assert isinstance(fixed, Allocation) and fixed.treatments == ("x1",)
    assert fixed.doses["x1"] == pytest.approx(6.0, abs=1e-6)
    assert fixed.expected_outcome == pytest.approx(float(_closed_form([6.0, 2.0], THETA)))


def test_point_theta_must_be_scalars() -> None:
    with pytest.raises(ValueError, match=r"theta\['beta_x1'\].*Posterior"):
        allocate(TOY, dict(THETA, beta_x1=np.array([10.0, 11.0])), budget=6.0, bounds=BOUNDS)
    with pytest.raises(ValueError, match=r"theta\['a'\]"):
        allocate(TOY, dict(THETA, a=np.ones((2, 1))), budget=6.0, bounds=BOUNDS)


# -- three treatments: the corners are not enough -----------------------------------------

NAMES3 = ("x1", "x2", "x3")
TOY3 = HillToy(NAMES3)
THETA3 = {
    "a": 1.0,
    **{f"{p}_{t}": v for t in NAMES3 for p, v in (("k", 2.0), ("s", 3.0), ("beta", 10.0))},
}
BOUNDS3 = Bounds(treatments=NAMES3, low=(0.0,) * 3, high=(10.0,) * 3)


def _closed_form_k(
    x: npt.ArrayLike, theta: Mapping[str, float], names: tuple[str, ...]
) -> npt.NDArray[np.float64]:
    xv = np.asarray(x, dtype=float)
    out = np.full(xv.shape[:-1], theta["a"])
    for j, t in enumerate(names):
        u = (xv[..., j] / theta[f"k_{t}"]) ** theta[f"s_{t}"]
        out = out + theta[f"beta_{t}"] * u / (1.0 + u)
    return np.asarray(out, dtype=np.float64)


def _simplex_grid_best(
    theta: Mapping[str, float], budget: float, bounds: Bounds, names: tuple[str, ...], n: int
) -> tuple[npt.NDArray[np.float64], float, float]:
    """Brute force over the budget simplex: every grid point of the box with Σ x <= budget."""
    axes = [np.linspace(lo, hi, n) for lo, hi in zip(bounds.low, bounds.high, strict=True)]
    pts = np.asarray(list(itertools.product(*axes)))
    pts = pts[pts.sum(axis=1) <= budget + 1e-12]
    vals = _closed_form_k(pts, theta, names)
    i = int(np.argmax(vals))
    step = max(float(a[1] - a[0]) for a in axes)
    return pts[i], float(vals[i]), step


def test_toy3_forward_matches_closed_form() -> None:
    x = np.array([[0.0, 2.0, 2.0], [4.0, 0.0, 0.0], [5 / 3, 5 / 3, 5 / 3]])
    dose = {t: x[:, j] for j, t in enumerate(NAMES3)}
    np.testing.assert_allclose(
        TOY3.forward(dose, THETA3), _closed_form_k(x, THETA3, NAMES3), rtol=1e-12
    )


@pytest.mark.parametrize(
    ("budget", "exact"),
    [
        # all-in on two of three at budget b: 1 + 2 * 10 * u / (1 + u), u = (b / 4)^3
        (4.0, 11.0),
        (5.0, 1.0 + 20.0 * 1.25**3 / (1.0 + 1.25**3)),
        (6.0, 1.0 + 20.0 * 1.5**3 / (1.0 + 1.5**3)),
    ],
)
def test_three_sigmoidal_treatments_match_brute_force_with_defaults(
    budget: float, exact: float
) -> None:
    # k = 3 with s = 3: each all-in corner is a genuine KKT point and so is the even split, yet
    # the optimum funds exactly two treatments equally — reachable from neither the corners nor
    # the centre; the two-way splits, the pre-search pool, and the perturbation check find it
    grid_x, grid_v, step = _simplex_grid_best(THETA3, budget, BOUNDS3, NAMES3, n=81)
    assert grid_v == pytest.approx(exact, rel=1e-12)
    assert sorted(grid_x)[0] < step < sorted(grid_x)[1]  # one treatment unfunded, two equal
    alloc = allocate(TOY3, THETA3, budget=budget, bounds=BOUNDS3)
    assert isinstance(alloc, Allocation)
    assert alloc.expected_outcome >= grid_v - 1e-9
    assert alloc.expected_outcome == pytest.approx(exact, rel=1e-9)
    np.testing.assert_allclose(np.sort(alloc.as_array()), np.sort(grid_x), atol=step)
    assert abs(alloc.slack) < 1e-6
    assert alloc.detail["n_presearch"] == 12.0  # max(8, 4k)
    assert alloc.detail["n_perturbations"] > 0.0
    # deterministic under the seed and stable across seeds: the optimum is not a lucky draw
    for seed in (0, 1, 2):
        again = allocate(TOY3, THETA3, budget=budget, bounds=BOUNDS3, seed=seed)
        assert isinstance(again, Allocation)
        assert again.expected_outcome == pytest.approx(exact, rel=1e-9)


def test_three_treatment_frontier_is_monotone_and_matches_brute_force() -> None:
    budgets = [2.0, 3.0, 4.0, 5.0, 6.0]
    fr = frontier(TOY3, THETA3, budgets, BOUNDS3)
    assert isinstance(fr, Frontier)
    for b, y in zip(fr.budgets, fr.outcomes, strict=True):
        _, grid_v, _ = _simplex_grid_best(THETA3, b, BOUNDS3, NAMES3, n=61)
        assert y >= grid_v - 1e-9
    assert np.all(np.diff(np.asarray(fr.outcomes)) > 0)


def test_frontier_rejects_a_drop_in_the_optimum(monkeypatch: pytest.MonkeyPatch) -> None:
    # the feasible set only grows with the budget, so an optimum below the previous budget's is
    # a missed basin: the builder refuses it, typed, naming both budgets (an allocator stub
    # returns the drop, since the real one no longer produces it on these surfaces)
    frontier_module = importlib.import_module("axiom.surface.frontier")  # the package re-exports
    # the function under the same name, so ``import ... as`` would bind the function, not the module

    fr = frontier(TOY, THETA, [2.0, 4.0], BOUNDS)
    assert isinstance(fr, Frontier)
    worse = fr.allocations[1].model_copy(update={"expected_outcome": fr.outcomes[0] - 1.0})
    canned = {2.0: fr.allocations[0], 4.0: worse}

    def stub(*args: object, budget: float, **kwargs: object) -> Allocation:
        return canned[budget]

    monkeypatch.setattr(frontier_module, "allocate", stub)
    result = frontier(TOY, THETA, [2.0, 4.0], BOUNDS)
    assert isinstance(result, Unsupported)
    assert result.reason.startswith("frontier failed at budget 4") and "below" in result.reason
    assert result.detail["previous_budget"] == "2"


def test_budget_equal_to_the_sum_of_lower_bounds_returns_the_lower_bounds() -> None:
    tight = Bounds(treatments=NAMES, low=(1.0, 1.0), high=(5.0, 5.0))
    for budget in (2.0, 2.0 + 1e-9):
        alloc = allocate(TOY, THETA, budget=budget, bounds=tight)
        assert isinstance(alloc, Allocation)
        assert alloc.as_array().tolist() == [1.0, 1.0]
        assert alloc.expected_outcome == pytest.approx(float(_closed_form([1.0, 1.0], THETA)))
        assert alloc.status == "converged" and alloc.budget == budget
        assert alloc.detail["budget_equals_sum_low"] == 1.0 and alloc.detail["n_draws"] == 1.0
    uneven = Bounds(treatments=NAMES, low=(0.5, 1.5), high=(5.0, 5.0))
    alloc = allocate(TOY, THETA, budget=2.0, bounds=uneven, method="cvxpy")
    assert isinstance(alloc, Allocation) and alloc.as_array().tolist() == [0.5, 1.5]
    assert alloc.method == "cvxpy"
    zero = allocate(TOY, THETA, budget=0.0, bounds=BOUNDS)
    assert isinstance(zero, Allocation) and zero.as_array().tolist() == [0.0, 0.0]
    assert zero.expected_outcome == pytest.approx(THETA["a"])
    # just above the tolerance the solver runs, and the frontier may start at the floor
    post = _posterior(40, 9)
    over = allocate(TOY, post, budget=2.0 + 1e-6, bounds=tight)
    assert isinstance(over, Allocation) and "budget_equals_sum_low" not in over.detail
    fr = frontier(TOY, THETA, [2.0, 3.0, 6.0], tight)
    assert isinstance(fr, Frontier) and fr.doses()[0].tolist() == [1.0, 1.0]
    assert np.all(np.diff(np.asarray(fr.outcomes)) > 0)
    # below the floor is still infeasible
    assert isinstance(allocate(TOY, THETA, budget=2.0 - 1e-6, bounds=tight), Unsupported)


# -- the real Surface -------------------------------------------------------------------

SURFACE_THETA = {
    "alpha": 1.0,
    "k_a": 2.0,
    "s_a": 1.0,
    "beta_a": 10.0,
    "k_b": 4.0,
    "s_b": 1.0,
    "beta_b": 8.0,
    "lam_a": 0.5,
    "sigma": 1.0,
}
SURFACE_BOUNDS = Bounds(treatments=("a", "b"), low=(0.0, 0.0), high=(10.0, 10.0))


def _spec(**overrides: object) -> SurfaceSpec:
    fields: dict[str, object] = {
        "name": "toy",
        "treatments": (
            Treatment(name="a", dimension=D.currency, unit="usd"),
            Treatment(name="b", dimension=D.currency, unit="usd"),
        ),
        "outcome": Outcome(name="y", dimension=D.outcome, unit="count"),
    }
    fields.update(overrides)
    return SurfaceSpec.model_validate(fields)


def test_real_surface_matches_the_toy() -> None:
    surface = Surface(_spec())
    alloc = allocate(surface, SURFACE_THETA, budget=6.0, bounds=SURFACE_BOUNDS)
    toy = allocate(TOY, THETA, budget=6.0, bounds=BOUNDS)
    assert isinstance(alloc, Allocation) and isinstance(toy, Allocation)
    assert alloc.treatments == ("a", "b")
    np.testing.assert_allclose(alloc.as_array(), toy.as_array(), atol=1e-5)
    assert alloc.expected_outcome == pytest.approx(toy.expected_outcome, rel=1e-10)
    rows = {"a": alloc.as_array()[:1], "b": alloc.as_array()[1:]}
    assert alloc.expected_outcome == pytest.approx(
        float(surface.forward(rows, SURFACE_THETA)[0]), rel=1e-12
    )


def test_carryover_surface_is_refused_and_steady_state_is_accepted() -> None:
    carried = Surface(_spec(carryover={"a": GeometricCarryover(max_lag=4)}))
    result = allocate(carried, SURFACE_THETA, budget=6.0, bounds=SURFACE_BOUNDS)
    assert isinstance(result, Unsupported)
    assert "carryover" in result.reason and "steady_state()" in result.reason
    assert "convolve" in result.detail["node"]
    fr = frontier(carried, SURFACE_THETA, [2.0, 6.0], SURFACE_BOUNDS)
    assert isinstance(fr, Unsupported) and "steady_state()" in fr.reason
    # at steady state the carryover is the identity (weights sum to one): same optimum as the
    # spec without carryover, and the carryover parameter is simply not read
    steady = carried.steady_state()
    alloc = allocate(steady, SURFACE_THETA, budget=6.0, bounds=SURFACE_BOUNDS)
    plain = allocate(Surface(_spec()), SURFACE_THETA, budget=6.0, bounds=SURFACE_BOUNDS)
    assert isinstance(alloc, Allocation) and isinstance(plain, Allocation)
    np.testing.assert_allclose(alloc.as_array(), plain.as_array(), atol=1e-6)
    assert alloc.expected_outcome == pytest.approx(plain.expected_outcome, rel=1e-10)
    fr = frontier(steady, SURFACE_THETA, [2.0, 6.0], SURFACE_BOUNDS)
    assert isinstance(fr, Frontier) and fr.outcomes[1] == pytest.approx(alloc.expected_outcome)


def test_per_unit_intercept_through_context() -> None:
    spec = _spec(intercept="per_unit", unit_labels=("u0", "u1"))
    surface = Surface(spec)
    theta = {**SURFACE_THETA, "alpha_unit": np.array([0.0, 5.0])}
    with pytest.raises(ValueError, match=r"\['unit'\].*context="):
        allocate(surface, theta, budget=6.0, bounds=SURFACE_BOUNDS)
    alloc = allocate(surface, theta, budget=6.0, bounds=SURFACE_BOUNDS, context={"unit": 1})
    assert isinstance(alloc, Allocation)
    x = alloc.as_array()
    direct = surface.forward({"a": x[:1], "b": x[1:], "unit": np.array([1])}, theta)
    assert alloc.expected_outcome == pytest.approx(float(direct[0]), rel=1e-12)
    # the intercept shifts the level only: the doses are the toy's
    toy = allocate(TOY, THETA, budget=6.0, bounds=BOUNDS)
    assert isinstance(toy, Allocation)
    np.testing.assert_allclose(x, toy.as_array(), atol=1e-5)
    assert alloc.expected_outcome == pytest.approx(toy.expected_outcome + 4.0, rel=1e-10)
    # the declared shape is enforced, and a scalar-only theta names the parameter
    with pytest.raises(ValueError, match=r"theta\['alpha_unit'\].*\(2,\)"):
        allocate(
            surface,
            {**theta, "alpha_unit": np.zeros(3)},
            budget=6.0,
            bounds=SURFACE_BOUNDS,
            context={"unit": 1},
        )
    with pytest.raises(ValueError, match=r"context\['unit'\]"):
        allocate(surface, theta, budget=6.0, bounds=SURFACE_BOUNDS, context={"unit": [0, 1]})
    # the steady-state surface keeps the per-unit intercept and its context
    carried = Surface(
        _spec(
            intercept="per_unit",
            unit_labels=("u0", "u1"),
            carryover={"b": GeometricCarryover(max_lag=3)},
        )
    )
    via_steady = allocate(
        carried.steady_state(), theta, budget=6.0, bounds=SURFACE_BOUNDS, context={"unit": 1}
    )
    assert isinstance(via_steady, Allocation)
    np.testing.assert_allclose(via_steady.as_array(), x, atol=1e-6)


def test_gather_index_context_must_be_an_integer_in_range() -> None:
    # ``unit`` is read through a Gather over alpha_unit (shape (2,)): a fractional, negative, or
    # out-of-range index is an error naming the column — never truncated, wrapped, or IndexError
    surface = Surface(_spec(intercept="per_unit", unit_labels=("u0", "u1")))
    theta = {**SURFACE_THETA, "alpha_unit": np.array([0.0, 5.0])}

    def run(unit: object) -> Allocation | object:
        return allocate(
            surface, theta, budget=6.0, bounds=SURFACE_BOUNDS, context={"unit": unit}, seed=0
        )

    for bad in (1.7, 0.5, float("nan"), True, "1"):
        with pytest.raises(ValueError, match=r"context\['unit'\].*integer"):
            run(bad)
    with pytest.raises(ValueError, match=r"context\['unit'\] = -1.*non-negative"):
        run(-1)
    with pytest.raises(ValueError, match=r"context\['unit'\] = 2 is out of range.*0\.\.1"):
        run(2)
    with pytest.raises(ValueError, match=r"context\['unit'\] = 7 is out of range"):
        run(np.int64(7))
    # integral floats and numpy integers are accepted and select the same unit
    one, one_f, one_np = run(1), run(1.0), run(np.int32(1))
    assert isinstance(one, Allocation) and isinstance(one_f, Allocation)
    assert isinstance(one_np, Allocation)
    assert one_f == one and one_np == one
    zero = run(0)
    assert isinstance(zero, Allocation)
    assert one.expected_outcome == pytest.approx(zero.expected_outcome + 5.0, rel=1e-10)
    # the frontier validates the same way
    with pytest.raises(ValueError, match=r"context\['unit'\]"):
        frontier(surface, theta, [2.0, 6.0], SURFACE_BOUNDS, context={"unit": 2})


# -- cvxpy -----------------------------------------------------------------------------

cvxpy = pytest.importorskip("cvxpy")


def test_cvxpy_matches_slsqp_on_concave_additive_surface() -> None:
    slsqp = allocate(TOY, THETA, budget=6.0, bounds=BOUNDS)
    convex = allocate(TOY, THETA, budget=6.0, bounds=BOUNDS, method="cvxpy", n_grid=401, seed=0)
    assert isinstance(slsqp, Allocation) and isinstance(convex, Allocation)
    assert convex.method == "cvxpy" and convex.detail["n_grid"] == 401.0
    # a piecewise-linear surrogate on a 401-point grid lands within one grid cell
    np.testing.assert_allclose(convex.as_array(), slsqp.as_array(), atol=0.05)
    assert convex.expected_outcome == pytest.approx(slsqp.expected_outcome, abs=1e-4)
    assert convex.expected_outcome <= slsqp.expected_outcome + 1e-9
    assert abs(convex.slack) < 1e-6
    # the reported value is forward() at the doses, not the surrogate's
    assert convex.expected_outcome == pytest.approx(
        float(_closed_form(convex.as_array(), THETA)), rel=1e-12
    )
    # the cvxpy answer is a warm start the exact solver polishes
    polished = allocate(TOY, THETA, budget=6.0, bounds=BOUNDS, start=convex.doses)
    assert isinstance(polished, Allocation)
    np.testing.assert_allclose(polished.as_array(), slsqp.as_array(), atol=1e-5)


def test_cvxpy_refuses_non_concave_and_non_additive_surfaces() -> None:
    sigmoid = dict(THETA, s_x1=3.0)
    result = allocate(TOY, sigmoid, budget=6.0, bounds=BOUNDS, method="cvxpy")
    assert isinstance(result, Unsupported) and "concave" in result.reason
    assert result.detail["treatment"] == "x1"
    coupled = HillToy(NAMES, interaction=True)
    result = allocate(coupled, dict(THETA, gamma=3.0), budget=6.0, bounds=BOUNDS, method="cvxpy")
    assert isinstance(result, Unsupported) and "separable" in result.reason


def test_cvxpy_tolerances_follow_the_response_range_not_its_level() -> None:
    # a sigmoid does not become concave because the intercept is large
    sigmoid = dict(THETA, s_x1=3.0, a=1e7)
    result = allocate(TOY, sigmoid, budget=6.0, bounds=BOUNDS, method="cvxpy", seed=0)
    assert isinstance(result, Unsupported) and "concave" in result.reason
    assert result.detail["treatment"] == "x1"
    # a concave surface with the same level is still solved
    still = allocate(TOY, dict(THETA, a=1e7), budget=6.0, bounds=BOUNDS, method="cvxpy", seed=0)
    base = allocate(TOY, THETA, budget=6.0, bounds=BOUNDS, method="cvxpy", seed=0)
    assert isinstance(still, Allocation) and isinstance(base, Allocation)
    np.testing.assert_allclose(still.as_array(), base.as_array(), atol=1e-6)
    # a level so large that rounding swamps the concavity tolerance is refused as unverifiable
    result = allocate(TOY, dict(THETA, a=1e12), budget=6.0, bounds=BOUNDS, method="cvxpy", seed=0)
    assert isinstance(result, Unsupported) and "cannot verify" in result.reason


# -- frontier ---------------------------------------------------------------------------


def test_frontier_is_monotone_with_diminishing_shadow_prices() -> None:
    budgets = [1.0, 2.0, 4.0, 8.0, 12.0]
    fr = frontier(TOY, THETA, budgets, BOUNDS)
    assert isinstance(fr, Frontier)
    assert fr.budgets == tuple(budgets) and fr.n == 5 and fr.treatments == NAMES
    outcomes = np.asarray(fr.outcomes)
    assert np.all(np.diff(outcomes) > 0)
    prices = fr.shadow_prices()
    assert prices.shape == (5,) and np.all(prices > 0) and np.all(np.diff(prices) < 0)
    for a in fr.allocations:
        assert abs(a.slack) < 1e-6
    # each point is the same optimum allocate finds cold
    cold = allocate(TOY, THETA, budget=8.0, bounds=BOUNDS)
    assert isinstance(cold, Allocation)
    np.testing.assert_allclose(fr.doses()[3], cold.as_array(), atol=1e-5)
    frame = fr.as_frame()
    assert list(frame.columns) == ["budget", "expected_outcome", "shadow_price", "x1", "x2"]
    assert len(frame) == 5
    # budgets are sorted, order of input does not matter
    rev = frontier(TOY, THETA, budgets[::-1], BOUNDS)
    assert isinstance(rev, Frontier) and rev.budgets == fr.budgets


def test_frontier_beyond_the_box_saturates_at_the_upper_bounds() -> None:
    # both budgets exceed Σ high = 20: the box, not the budget, binds
    fr = frontier(TOY, THETA, [20.0, 25.0], BOUNDS)
    assert isinstance(fr, Frontier)
    np.testing.assert_allclose(fr.doses(), [BOUNDS.high, BOUNDS.high], atol=1e-6)
    assert fr.outcomes[1] == pytest.approx(fr.outcomes[0], abs=1e-9)
    assert fr.shadow_prices()[1] == pytest.approx(0.0, abs=1e-9)
    assert fr.allocations[1].slack == pytest.approx(5.0, abs=1e-6)


def test_frontier_failure_names_the_budget() -> None:
    result = frontier(TOY, THETA, [2.0, 6.0], BOUNDS, maxiter=1)
    assert isinstance(result, Unsupported)
    assert result.reason.startswith("frontier failed at budget 2")
    assert result.detail["budget"] == "2"
    with pytest.raises(ValueError, match="distinct"):
        frontier(TOY, THETA, [2.0, 2.0], BOUNDS)
    with pytest.raises(ValueError, match="at least one"):
        frontier(TOY, THETA, [], BOUNDS)


def test_frontier_spec_round_trip_and_validation() -> None:
    fr = frontier(TOY, THETA, [2.0, 4.0], BOUNDS)
    assert isinstance(fr, Frontier)
    assert Frontier.from_json(fr.to_json()) == fr
    with pytest.raises(ValueError, match="increasing"):
        Frontier(budgets=(4.0, 2.0), outcomes=fr.outcomes[::-1], allocations=fr.allocations[::-1])
    with pytest.raises(ValueError, match="disagrees"):
        Frontier(budgets=fr.budgets, outcomes=(0.0, 0.0), allocations=fr.allocations)
    single = frontier(TOY, THETA, [3.0], BOUNDS)
    assert isinstance(single, Frontier) and np.isnan(single.shadow_prices()).all()
