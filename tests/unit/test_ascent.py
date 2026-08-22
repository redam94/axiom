"""surface.ascent: derivatives, steepest ascent, canonical analysis on quadratic and Hill toys."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping

import numpy as np
import numpy.typing as npt
import pytest

from axiom.core import (
    Add,
    Const,
    Data,
    DesignMatrix,
    Div,
    Model,
    Mul,
    Param,
    Pow,
    Unsupported,
    dimensionless,
    jax_available,
    value,
)
from axiom.surface.ascent import (
    AscentPath,
    StationaryPoint,
    canonical_analysis,
    gradient,
    hessian,
    steepest_ascent,
)
from axiom.surface.design import Bounds

NAMES = ("x1", "x2")


class QuadraticToy:
    """``y = c + b'x + x'Ax`` expressed in the second-order basis with expr nodes.

    Parameters are the basis coefficients: ``b_{i}`` = b_i, ``b_{i}_{i}`` = A_ii,
    ``b_{i}_{j}`` = 2 A_ij. Analytic gradient ``2Ax + b``, Hessian ``2A``.
    """

    def __init__(self, treatments: tuple[str, ...]) -> None:
        dl = dimensionless()
        basis: dict[str, Model] = {"b0": Const(value=1.0, dimension=dl)}
        for t in treatments:
            basis[f"b_{t}"] = Data(name=t, dimension=dl)
        for i, ti in enumerate(treatments):
            for tj in treatments[i:]:
                basis[f"b_{ti}_{tj}"] = Mul(
                    factors=(Data(name=ti, dimension=dl), Data(name=tj, dimension=dl))
                )
        self.treatments = treatments
        self.basis = basis
        self._expr: Model = Add(
            terms=tuple(Mul(factors=(Param(name=c, dimension=dl), e)) for c, e in basis.items())
        )

    @property
    def expr(self) -> Model:
        return self._expr

    def forward(
        self, dose: Mapping[str, npt.ArrayLike], theta: Mapping[str, npt.ArrayLike]
    ) -> npt.NDArray[np.float64]:
        return value(self._expr, data=dose, params=theta)

    def linearize(
        self, dose: Mapping[str, npt.ArrayLike], theta_at: Mapping[str, npt.ArrayLike]
    ) -> DesignMatrix:
        n = np.asarray(next(iter(dose.values()))).size
        cols = [np.broadcast_to(value(e, data=dose), (n,)) for e in self.basis.values()]
        return DesignMatrix(
            X=np.column_stack(cols), columns=tuple(self.basis), offset=np.zeros(n), at={}
        )

    def theta(self, c: float, b: npt.ArrayLike, A: npt.ArrayLike) -> dict[str, float]:
        bv, Am = np.asarray(b, dtype=float), np.asarray(A, dtype=float)
        out = {"b0": c}
        for i, ti in enumerate(self.treatments):
            out[f"b_{ti}"] = float(bv[i])
            for j, tj in enumerate(self.treatments[i:], start=i):
                out[f"b_{ti}_{tj}"] = float(Am[i, i] if i == j else 2.0 * Am[i, j])
        return out


class HillToy:
    """``y = b0 + b1 * (x/k)^s / (1 + (x/k)^s) + b2 * z``: a saturating dose plus a linear one.

    ``x`` enters through ``x/k`` only, so scaling ``x`` and ``k`` together by a
    factor ``c`` leaves every value unchanged: the gradient in ``x`` scales by
    ``1/c`` and the Hessian by ``1/c^2``, exactly.
    """

    def __init__(self) -> None:
        dl = dimensionless()
        x_over_k = Div(
            numerator=Data(name="x", dimension=dl), denominator=Param(name="k", dimension=dl)
        )
        s = Param(name="s", dimension=dl)
        hill = Div(
            numerator=Pow(base=x_over_k, exponent=s),
            denominator=Add(terms=(Const(value=1.0, dimension=dl), Pow(base=x_over_k, exponent=s))),
        )
        self._expr: Model = Add(
            terms=(
                Mul(factors=(Param(name="b0", dimension=dl), Const(value=1.0, dimension=dl))),
                Mul(factors=(Param(name="b1", dimension=dl), hill)),
                Mul(factors=(Param(name="b2", dimension=dl), Data(name="z", dimension=dl))),
            )
        )

    @property
    def expr(self) -> Model:
        return self._expr

    def forward(
        self, dose: Mapping[str, npt.ArrayLike], theta: Mapping[str, npt.ArrayLike]
    ) -> npt.NDArray[np.float64]:
        return value(self._expr, data=dose, params=theta)

    def linearize(
        self, dose: Mapping[str, npt.ArrayLike], theta_at: Mapping[str, npt.ArrayLike]
    ) -> DesignMatrix:
        raise NotImplementedError("not needed by the ascent tests")


def hill_closed_form(x: float, k: float, s: float, b1: float) -> tuple[float, float]:
    """Analytic ``d/dx`` and ``d2/dx2`` of ``b1 * u^s / (1 + u^s)``, ``u = x / k``."""
    u = x / k
    us = u**s
    g = b1 * s * us / (x * (1.0 + us) ** 2)
    h = b1 * s * us * ((s - 1.0) * (1.0 + us) - 2.0 * s * us) / (x**2 * (1.0 + us) ** 3)
    return g, h


TOY = QuadraticToy(NAMES)
HILL = HillToy()
HILL_THETA = {"b0": 1.0, "b1": 2.0, "b2": -0.1, "k": 2.0, "s": 0.8}
A_MAX = np.array([[-2.0, 0.5], [0.5, -1.0]])
B_MAX = np.array([3.0, -1.0])
C0 = 4.0
BIG_BOX = Bounds(treatments=NAMES, low=(-5.0, -5.0), high=(5.0, 5.0))


def _stationary(b: npt.ArrayLike, A: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(-0.5 * np.linalg.solve(np.asarray(A), np.asarray(b)), dtype=np.float64)


def grad(
    surface: object,
    theta: Mapping[str, float],
    x: Mapping[str, float],
    *,
    method: str = "auto",
    bounds: Bounds | None = None,
) -> dict[str, float]:
    """``gradient`` that must succeed: a typed failure is a test failure, not a number."""
    g = gradient(surface, theta, x, method=method, bounds=bounds)  # type: ignore[arg-type]
    assert not isinstance(g, Unsupported), g.reason
    return g


def hess(
    surface: object,
    theta: Mapping[str, float],
    x: Mapping[str, float],
    *,
    method: str = "auto",
    bounds: Bounds | None = None,
) -> npt.NDArray[np.float64]:
    """``hessian`` that must succeed."""
    H = hessian(surface, theta, x, method=method, bounds=bounds)  # type: ignore[arg-type]
    assert not isinstance(H, Unsupported), H.reason
    return H


@contextlib.contextmanager
def _x64_on() -> Iterator[None]:
    if not jax_available():
        pytest.skip("jax not installed")
    import jax

    was = bool(getattr(jax.config, "jax_enable_x64", False))
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", was)


@pytest.fixture(params=["finite", "jax"])
def method(request: pytest.FixtureRequest) -> Iterator[str]:
    m = str(request.param)
    if m == "jax":
        with _x64_on():
            yield m
    else:
        yield m


@pytest.fixture
def x64() -> Iterator[None]:
    with _x64_on():
        yield


# -- derivatives -------------------------------------------------------------------------


def test_gradient_and_hessian_match_analytic(method: str) -> None:
    theta = TOY.theta(C0, B_MAX, A_MAX)
    x = {"x1": 0.7, "x2": -1.3}
    xv = np.array([0.7, -1.3])
    g = grad(TOY, theta, x, method=method)
    assert tuple(g) == NAMES
    np.testing.assert_allclose(np.array(list(g.values())), 2 * A_MAX @ xv + B_MAX, atol=1e-7)
    H = hess(TOY, theta, x, method=method)
    np.testing.assert_allclose(H, 2 * A_MAX, atol=1e-6)
    # key order of x fixes the treatment order
    g_rev = grad(TOY, theta, {"x2": -1.3, "x1": 0.7}, method=method)
    assert tuple(g_rev) == ("x2", "x1") and g_rev["x1"] == pytest.approx(g["x1"])
    # bounds only change the step scale, not the answer
    gb = grad(TOY, theta, x, method=method, bounds=BIG_BOX)
    np.testing.assert_allclose(np.array(list(gb.values())), 2 * A_MAX @ xv + B_MAX, atol=1e-7)
    with pytest.raises(ValueError, match="exactly the treatments"):
        grad(TOY, theta, x, bounds=Bounds(treatments=("x1",), low=(0.0,), high=(1.0,)))


def test_auto_method_matches_finite_without_x64() -> None:
    if jax_available():
        import jax

        if bool(getattr(jax.config, "jax_enable_x64", False)):
            pytest.skip("x64 is on; auto resolves to jax")
    theta = TOY.theta(C0, B_MAX, A_MAX)
    x = {"x1": 0.2, "x2": 0.9}
    assert grad(TOY, theta, x) == grad(TOY, theta, x, method="finite")


def test_jax_method_without_x64_is_an_error() -> None:
    if not jax_available():
        pytest.skip("jax not installed")
    import jax

    was = bool(getattr(jax.config, "jax_enable_x64", False))
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(ValueError, match="jax_enable_x64"):
            grad(TOY, TOY.theta(C0, B_MAX, A_MAX), {"x1": 0.2, "x2": 0.9}, method="jax")
        with pytest.raises(ValueError, match="jax_enable_x64"):
            hess(TOY, TOY.theta(C0, B_MAX, A_MAX), {"x1": 0.2, "x2": 0.9}, method="jax")
    finally:
        jax.config.update("jax_enable_x64", was)


def test_finite_differences_track_jax_on_hill(x64: None) -> None:
    """Measure the finite-difference truncation error against exact (x64) jax derivatives."""
    worst_g, worst_h = 0.0, 0.0
    for x in (0.5, 2.0, 7.0):
        pt = {"x": x, "z": 1.0}
        gf = grad(HILL, HILL_THETA, pt, method="finite")
        gj = grad(HILL, HILL_THETA, pt, method="jax")
        g_true, h_true = hill_closed_form(x, 2.0, 0.8, 2.0)
        assert gj["x"] == pytest.approx(g_true, rel=1e-12)
        assert gj["z"] == pytest.approx(-0.1, rel=1e-12)
        Hf = hess(HILL, HILL_THETA, pt, method="finite")
        Hj = hess(HILL, HILL_THETA, pt, method="jax")
        assert Hj[0, 0] == pytest.approx(h_true, rel=1e-12)
        worst_g = max(worst_g, abs(gf["x"] - gj["x"]) / abs(gj["x"]))
        worst_h = max(worst_h, abs(Hf[0, 0] - Hj[0, 0]) / abs(Hj[0, 0]))
        assert gf["z"] == pytest.approx(-0.1, rel=1e-9)
        assert abs(Hf[0, 1]) < 1e-9  # mixed term of a separable surface: roundoff only
    print(
        f"\nfinite-difference vs jax on Hill: gradient {worst_g:.2e}, Hessian {worst_h:.2e} (rel)"
    )
    assert worst_g < 1e-8  # recorded: 7.7e-11 (relative step 1e-5, central)
    assert worst_h < 1e-5  # recorded: 5.5e-07 (relative step 1e-3, central)


@pytest.mark.parametrize("scale", [1e-3, 1e3])
def test_finite_differences_are_unit_invariant(scale: float) -> None:
    """Rescaling the dose unit (x and k together) rescales g by 1/c and H by 1/c^2.

    Exactly so whenever the step in ``x`` is set in ``x``'s own units: always
    with bounds, and without them while ``|x|`` is at least 1e-2 of the largest
    coordinate. At ``scale = 1e-3`` the dose ``2e-3`` sits below 1e-2 of ``z = 1``
    and borrows ``z``'s scale, so the two Hessians then agree only to truncation
    order (recorded: 8.5e-6 relative) — but each still matches the closed form
    within the method's documented accuracy.
    """
    x = 2.0
    base = {"x": x, "z": 1.0}
    scaled_theta = dict(HILL_THETA, k=HILL_THETA["k"] * scale)
    scaled = {"x": x * scale, "z": 1.0}
    own_units = x * scale >= 1e-2 * max(x * scale, 1.0)
    g = grad(HILL, HILL_THETA, base, method="finite")
    gs = grad(HILL, scaled_theta, scaled, method="finite")
    assert gs["x"] * scale == pytest.approx(g["x"], rel=1e-8)
    assert gs["z"] == pytest.approx(g["z"], rel=1e-8)
    H = hess(HILL, HILL_THETA, base, method="finite")
    Hs = hess(HILL, scaled_theta, scaled, method="finite")
    assert Hs[0, 0] * scale**2 == pytest.approx(H[0, 0], rel=1e-7 if own_units else 1e-4)
    # mixed term of a separable surface: round-off relative to the resolved curvature
    assert abs(H[0, 1]) <= 1e-6 * abs(H[0, 0])
    assert abs(Hs[0, 1]) <= 1e-6 * abs(Hs[0, 0])
    # both agree with the closed form at their own scale
    g_true, h_true = hill_closed_form(x, 2.0, 0.8, 2.0)
    assert g["x"] == pytest.approx(g_true, rel=1e-8)
    assert gs["x"] == pytest.approx(g_true / scale, rel=1e-8)
    assert H[0, 0] == pytest.approx(h_true, rel=1e-5)
    assert Hs[0, 0] == pytest.approx(h_true / scale**2, rel=1e-5)
    # with bounds the step is in each treatment's own units at every scale: exact
    box = Bounds(treatments=("x", "z"), low=(0.0, -1.0), high=(10.0, 3.0))
    box_s = Bounds(treatments=("x", "z"), low=(0.0, -1.0), high=(10.0 * scale, 3.0))
    gb = grad(HILL, HILL_THETA, base, method="finite", bounds=box)
    gbs = grad(HILL, scaled_theta, scaled, method="finite", bounds=box_s)
    assert gbs["x"] * scale == pytest.approx(gb["x"], rel=1e-8)
    Hb = hess(HILL, HILL_THETA, base, method="finite", bounds=box)
    Hbs = hess(HILL, scaled_theta, scaled, method="finite", bounds=box_s)
    assert Hbs[0, 0] * scale**2 == pytest.approx(Hb[0, 0], rel=1e-7)


def test_finite_differences_need_a_scale_at_the_origin() -> None:
    """All-zero x without bounds defines no dose scale: Unsupported, never a number."""
    theta = TOY.theta(C0, B_MAX, A_MAX)
    origin = {"x1": 0.0, "x2": 0.0}
    for r in (
        gradient(TOY, theta, origin, method="finite"),
        hessian(TOY, theta, origin, method="finite"),
        canonical_analysis(TOY, theta, origin, method="finite"),
    ):
        assert isinstance(r, Unsupported) and "dose scale" in r.reason and "bounds" in r.reason
    walk = steepest_ascent(TOY, theta, origin, step=0.1, n_steps=3, method="finite")
    assert isinstance(walk, Unsupported) and "dose scale" in walk.reason
    assert walk.detail["step"] == "0"
    # with bounds the half-width is the scale and the origin is an ordinary point
    g = grad(TOY, theta, origin, method="finite", bounds=BIG_BOX)
    np.testing.assert_allclose(np.array(list(g.values())), B_MAX, atol=1e-8)
    H = hess(TOY, theta, origin, method="finite", bounds=BIG_BOX)
    np.testing.assert_allclose(H, 2 * A_MAX, atol=1e-7)
    at0 = canonical_analysis(TOY, theta, origin, method="finite", bounds=BIG_BOX)
    assert isinstance(at0, StationaryPoint) and at0.kind == "maximum"


def test_zero_coordinate_without_bounds_borrows_the_scale_of_the_largest() -> None:
    """A zero coordinate used to get a 1e-8-relative step, so its second difference was
    round-off: hessian at (0, 1) came back [[0, 1.02], [1.02, -2]] for a true [[-4, 1], [1, -2]]
    and canonical_analysis silently reported a saddle. The step is now 1e-2 of max|x|."""
    theta = TOY.theta(C0, B_MAX, A_MAX)
    x = {"x1": 0.0, "x2": 1.0}
    true_H = 2 * A_MAX
    H = hess(TOY, theta, x, method="finite")
    scale = float(np.max(np.abs(true_H)))
    np.testing.assert_allclose(H, true_H, rtol=0.0, atol=1e-5 * scale)
    g = grad(TOY, theta, x, method="finite")
    true_g = 2 * A_MAX @ np.array([0.0, 1.0]) + B_MAX
    np.testing.assert_allclose(
        np.array(list(g.values())), true_g, rtol=0.0, atol=1e-7 * float(np.max(np.abs(true_g)))
    )
    r = canonical_analysis(TOY, theta, x, method="finite")
    assert isinstance(r, StationaryPoint) and r.kind == "maximum"
    xs = _stationary(B_MAX, A_MAX)
    np.testing.assert_allclose(
        np.array(r.point), xs, rtol=0.0, atol=1e-5 * float(np.max(np.abs(xs)))
    )
    np.testing.assert_allclose(
        np.array(r.eigenvalues), np.linalg.eigvalsh(true_H), rtol=0.0, atol=1e-5 * scale
    )
    # the same point with x1 and x2 swapped in the key order
    H_rev = hess(TOY, theta, {"x2": 1.0, "x1": 0.0}, method="finite")
    np.testing.assert_allclose(H_rev, true_H[::-1, ::-1], rtol=0.0, atol=1e-5 * scale)


def test_non_finite_derivatives_are_unsupported() -> None:
    """Hill with s < 1 just above dose 0 without bounds: the central stencil steps to x < 0 (NaN).

    At x = 1e-8 the coordinate borrows its scale from z (1e-2), so the gradient
    step 1e-7 and the Hessian step 1e-5 both cross zero, where x^0.8 is NaN.
    """
    near_zero = {"x": 1e-8, "z": 1.0}
    with np.errstate(invalid="ignore"):
        g = gradient(HILL, HILL_THETA, near_zero, method="finite")
        H = hessian(HILL, HILL_THETA, near_zero, method="finite")
        c = canonical_analysis(HILL, HILL_THETA, near_zero, method="finite")
    assert isinstance(g, Unsupported) and "gradient is not finite" in g.reason
    assert "bounds" in g.reason and "nan" in g.detail["gradient"]
    assert isinstance(H, Unsupported) and "hessian is not finite" in H.reason
    assert "nan" in H.detail["hessian"]
    assert isinstance(c, Unsupported) and "not finite" in c.reason


def _all_moves(
    surface: object, theta: Mapping[str, float], x: dict[str, float], **kw: object
) -> list[object]:
    """gradient, hessian, steepest_ascent, canonical_analysis at ``x`` with the same options."""
    return [
        gradient(surface, theta, x, **kw),  # type: ignore[arg-type]
        hessian(surface, theta, x, **kw),  # type: ignore[arg-type]
        steepest_ascent(surface, theta, x, step=0.5, n_steps=3, **kw),  # type: ignore[arg-type]
        canonical_analysis(surface, theta, x, **kw),  # type: ignore[arg-type]
    ]


def test_infinite_slope_at_a_boundary_is_unsupported_for_both_methods(x64: None) -> None:
    """Case A/B: Hill with s = 0.8 at dose 0. The true d/dx is +inf; jax's guarded ``Pow``
    reports 0 and a one-sided stencil inside bounds (0, 10) reported 9.4 — neither is a
    number to return. Every move refuses, with and without bounds, for both methods."""
    at_zero = {"x": 0.0, "z": 1.0}
    box = Bounds(treatments=("x", "z"), low=(0.0, 0.0), high=(10.0, 10.0))
    for method in ("jax", "finite"):
        for bounds in (None, box):
            with np.errstate(invalid="ignore"):
                results = _all_moves(HILL, HILL_THETA, at_zero, method=method, bounds=bounds)
            for r in results:
                assert isinstance(r, Unsupported), (method, bounds, r)
                assert "derivative not finite at the boundary for x" in r.reason
                assert "start strictly inside" in r.reason
                assert r.detail["coordinate"] == "x"
            walk = results[2]
            assert isinstance(walk, Unsupported) and walk.detail["step"] == "0"
    # the secants recorded grow by 10^(1 - s) = 1.58 per decade (the 1 + u^s denominator
    # adds 1e-3 at the largest step)
    g = gradient(HILL, HILL_THETA, at_zero, method="jax", bounds=box)
    assert isinstance(g, Unsupported)
    s = np.array([float(v) for v in g.detail["secants"].split(", ")])
    np.testing.assert_allclose(s[1:] / s[:-1], 10**0.2, rtol=5e-3)
    # a finite boundary slope passes the check: the quadratic at a corner and at a zero
    theta = TOY.theta(C0, B_MAX, A_MAX)
    corner = {"x1": 0.0, "x2": 1.0}
    for method in ("jax", "finite"):
        g_c = grad(TOY, theta, corner, method=method, bounds=BIG_BOX)
        np.testing.assert_allclose(
            np.array(list(g_c.values())), 2 * A_MAX @ np.array([0.0, 1.0]) + B_MAX, atol=1e-7
        )
        g_0 = grad(TOY, theta, corner, method=method)
        np.testing.assert_allclose(np.array(list(g_0.values())), list(g_c.values()), atol=1e-7)
    # on the upper bound the check looks inward (downwards): the linear z slope is finite
    top = Bounds(treatments=("x", "z"), low=(0.0, 0.0), high=(10.0, 1.0))
    g_top = grad(HILL, HILL_THETA, {"x": 2.0, "z": 1.0}, method="finite", bounds=top)
    assert g_top["z"] == pytest.approx(-0.1, rel=1e-7)  # one-sided stencil, h = 5e-6


def test_curvature_resolution_is_per_coordinate(x64: None) -> None:
    """Case 5: offset 1e8 on A = diag(-100, -0.01) at (0.7, -1.3). The x2 second difference
    is -0.015625 for a true -0.02 (its floor is 0.013); x1's -199.94 used to let the matrix
    through and canonical_analysis put the maximum at (0.01356, -63.85) for a true
    (0.015, -50). Now x2 alone is named; jax (exact) still resolves both."""
    x = {"x1": 0.7, "x2": -1.3}
    A = np.diag([-100.0, -0.01])
    theta = TOY.theta(1e8, B_MAX, A)
    H = hessian(TOY, theta, x, method="finite")
    assert isinstance(H, Unsupported) and "cannot resolve curvature in x2 " in H.reason
    assert H.detail["coordinates"] == "x2"
    c = canonical_analysis(TOY, theta, x, method="finite")
    assert isinstance(c, Unsupported) and "cannot resolve curvature in x2 " in c.reason
    # the gradient itself is resolved there: (-137, -0.974) against floors of order 3e-3
    g = grad(TOY, theta, x, method="finite")
    np.testing.assert_allclose(
        np.array(list(g.values())), 2 * A @ np.array([0.7, -1.3]) + B_MAX, rtol=1e-3
    )
    np.testing.assert_allclose(hess(TOY, theta, x, method="jax"), 2 * A, rtol=1e-9)
    exact = canonical_analysis(TOY, theta, x, method="jax")
    assert isinstance(exact, StationaryPoint) and exact.kind == "maximum"
    np.testing.assert_allclose(np.array(exact.point), [0.015, -50.0], rtol=1e-7)


def test_gradient_below_round_off_resolution_is_unsupported(x64: None) -> None:
    """Case C: offset 1e12 on A = diag(-2, -1) at (0.7, -1.3). forward() differs by ~1e-6
    across the stencil on a value whose ulp is 1e-4, so every first difference is exactly 0
    for a true (0.2, 1.6) and steepest_ascent used to stop "converged" at the start."""
    x = {"x1": 0.7, "x2": -1.3}
    A = np.diag([-2.0, -1.0])
    theta = TOY.theta(1e12, B_MAX, A)
    g = gradient(TOY, theta, x, method="finite")
    assert isinstance(g, Unsupported) and "cannot resolve the gradient in x1, x2 " in g.reason
    assert g.detail["coordinates"] == "x1, x2" and g.detail["gradient"] == "0, 0"
    walk = steepest_ascent(TOY, theta, x, step=0.1, n_steps=5, method="finite")
    assert isinstance(walk, Unsupported) and walk.detail["step"] == "0"
    assert "no gradient at step 0" in walk.reason and "cannot resolve the gradient" in walk.reason
    c = canonical_analysis(TOY, theta, x, method="finite")
    assert isinstance(c, Unsupported) and "cannot resolve the gradient" in c.reason
    gj = grad(TOY, theta, x, method="jax")
    np.testing.assert_allclose(np.array(list(gj.values())), [0.2, 1.6], rtol=1e-3)
    # per coordinate: at offset 1e9 x1 (-1.1, floor 0.03) is named and x2 (2.3, floor 0.017) not
    one = gradient(TOY, TOY.theta(1e9, B_MAX, A_MAX), x, method="finite")
    assert isinstance(one, Unsupported) and one.detail["coordinates"] == "x1"
    # a coordinate that is stationary beside a resolved one is negligible, not unresolved:
    # at (1, 1) the true gradient is (0, -2)
    theta_ok = TOY.theta(C0, B_MAX, A_MAX)
    g_ok = grad(TOY, theta_ok, {"x1": 1.0, "x2": 1.0}, method="finite")
    assert abs(g_ok["x1"]) <= 1e-8 and g_ok["x2"] == pytest.approx(-2.0, rel=1e-9)
    # at a stationary point nothing is resolved: finite differences cannot certify a zero
    # slope, so the gradient is Unsupported rather than the round-off it saw; jax is exact
    xs = dict(zip(NAMES, _stationary(B_MAX, A_MAX), strict=True))
    at_opt = gradient(TOY, theta_ok, xs, method="finite")
    assert isinstance(at_opt, Unsupported) and at_opt.detail["coordinates"] == "x1, x2"
    assert "may be stationary" in at_opt.reason
    exact = grad(TOY, theta_ok, xs, method="jax")
    assert np.allclose(list(exact.values()), 0.0, atol=1e-12)


def test_hessian_below_round_off_resolution_is_unsupported(x64: None) -> None:
    """A second difference the stencil cannot separate from round-off is not a number.

    A constant offset on a curvature-4 quadratic raises the round-off floor
    eps |f| / h_j^2 of each second difference without changing the Hessian. The
    step h_j = 1e-3 |x_j| is smaller in x1 (0.7) than in x2 (-1.3): at offset
    1.2e8 x1's entry (4) is below 100 x its floor (5.4) while x2's (2) is above
    its own (1.6), so x1 alone is named; by 1e9 both are, and the gradient is
    unresolved in x1 too. jax (exact) returns 2A throughout.
    """
    x = {"x1": 0.7, "x2": -1.3}
    one = hessian(TOY, TOY.theta(1.2e8, B_MAX, A_MAX), x, method="finite")
    assert isinstance(one, Unsupported) and "cannot resolve curvature in x1 " in one.reason
    assert one.detail["coordinates"] == "x1"
    theta = TOY.theta(1e9, B_MAX, A_MAX)
    H = hessian(TOY, theta, x, method="finite")
    assert isinstance(H, Unsupported) and "cannot resolve curvature" in H.reason
    assert H.detail["coordinates"] == "x1, x2"
    c = canonical_analysis(TOY, theta, x, method="finite")
    assert isinstance(c, Unsupported) and "cannot resolve the gradient in x1 " in c.reason
    np.testing.assert_allclose(hess(TOY, theta, x, method="jax"), 2 * A_MAX, rtol=1e-9)
    # a smaller offset is resolved, to within the documented noise relative to the largest entry
    ok = hess(TOY, TOY.theta(1e5, B_MAX, A_MAX), x, method="finite")
    np.testing.assert_allclose(
        ok, 2 * A_MAX, rtol=0.0, atol=1e-2 * float(np.max(np.abs(2 * A_MAX)))
    )
    # a surface with no curvature at all: every second difference is round-off, and with
    # nothing resolved there is nothing for it to be negligible beside
    flat = TOY.theta(C0, B_MAX, np.zeros((2, 2)))
    Hf = hessian(TOY, flat, x, method="finite")
    assert isinstance(Hf, Unsupported) and Hf.detail["coordinates"] == "x1, x2"
    np.testing.assert_allclose(hess(TOY, flat, x, method="jax"), np.zeros((2, 2)), atol=0.0)
    # a linear coordinate next to a curved one is negligible: its floor (5e-10) sits within
    # 1e-6 of the resolved curvature, even at x = 7 where that curvature is only 8.8e-3
    for xx in (0.5, 2.0, 7.0):
        Hz = hess(HILL, HILL_THETA, {"x": xx, "z": 1.0}, method="finite")
        assert Hz[1, 1] == pytest.approx(0.0, abs=1e-6 * abs(Hz[0, 0]))


def test_one_sided_stencils_at_the_boundary_stay_inside_and_stay_accurate(x64: None) -> None:
    theta = TOY.theta(C0, B_MAX, A_MAX)
    box = Bounds(treatments=NAMES, low=(0.0, 0.0), high=(1.0, 1.0))
    for corner in ({"x1": 0.0, "x2": 1.0}, {"x1": 1.0, "x2": 0.0}, {"x1": 0.0, "x2": 0.0}):
        xv = np.array([corner["x1"], corner["x2"]])
        g = grad(TOY, theta, corner, method="finite", bounds=box)
        np.testing.assert_allclose(np.array(list(g.values())), 2 * A_MAX @ xv + B_MAX, atol=1e-8)
        H = hess(TOY, theta, corner, method="finite", bounds=box)
        np.testing.assert_allclose(H, 2 * A_MAX, atol=1e-7)  # exact for a quadratic

    class Recording:
        """Wraps a surface and records every dose row forward() is asked for."""

        def __init__(self) -> None:
            self.rows: list[npt.NDArray[np.float64]] = []

        @property
        def expr(self) -> Model:
            return HILL.expr

        def forward(
            self, dose: Mapping[str, npt.ArrayLike], theta: Mapping[str, npt.ArrayLike]
        ) -> npt.NDArray[np.float64]:
            self.rows.append(np.column_stack([np.asarray(dose["x"]), np.asarray(dose["z"])]))
            return HILL.forward(dose, theta)

        def linearize(
            self, dose: Mapping[str, npt.ArrayLike], theta_at: Mapping[str, npt.ArrayLike]
        ) -> DesignMatrix:
            raise NotImplementedError

    rec = Recording()
    hb = Bounds(treatments=("x", "z"), low=(0.5, 0.0), high=(10.0, 2.0))
    at_corner = {"x": 0.5, "z": 2.0}  # lower bound in x, upper bound in z
    g0 = grad(rec, HILL_THETA, at_corner, method="finite", bounds=hb)
    H0 = hess(rec, HILL_THETA, at_corner, method="finite", bounds=hb)
    visited = np.vstack(rec.rows)  # includes the boundary check's secant points
    assert np.all(visited[:, 0] >= 0.5) and np.all(visited[:, 1] <= 2.0)
    assert np.all(np.isfinite(list(g0.values()))) and np.all(np.isfinite(H0))
    g_true, h_true = hill_closed_form(0.5, 2.0, 0.8, 2.0)
    assert g0["x"] == pytest.approx(g_true, rel=1e-6)
    assert g0["z"] == pytest.approx(-0.1, rel=1e-9)
    assert H0[0, 0] == pytest.approx(h_true, rel=1e-3)
    # away from the x^s singularity the one-sided stencil is second-order accurate
    inner = Bounds(treatments=("x", "z"), low=(3.0, 0.0), high=(10.0, 1.0))
    pt = {"x": 3.0, "z": 1.0}
    gj = grad(HILL, HILL_THETA, pt, method="jax")
    gf = grad(HILL, HILL_THETA, pt, method="finite", bounds=inner)
    assert gf["x"] == pytest.approx(gj["x"], rel=1e-7)
    Hj = hess(HILL, HILL_THETA, pt, method="jax")
    Hf = hess(HILL, HILL_THETA, pt, method="finite", bounds=inner)
    assert Hf[0, 0] == pytest.approx(Hj[0, 0], rel=1e-4)


# -- steepest ascent ---------------------------------------------------------------------


def test_steepest_ascent_default_is_a_straight_path(method: str) -> None:
    theta = TOY.theta(C0, B_MAX, A_MAX)
    start = {"x1": -3.0, "x2": 3.0}
    step = 0.02
    path = steepest_ascent(TOY, theta, start, step=step, n_steps=1000, method=method)
    assert isinstance(path, AscentPath)
    assert path.stop == "decrease"
    assert np.all(np.diff(path.values) > 0)
    P = np.array(path.points)
    d = np.diff(P, axis=0)
    g0 = np.array(list(grad(TOY, theta, start, method=method).values()))
    unit = g0 / np.linalg.norm(g0)
    np.testing.assert_allclose(d, np.broadcast_to(step * unit, d.shape), atol=1e-12)
    # the straight line stops where the value along it peaks, short of the 2-D optimum
    xs = _stationary(B_MAX, A_MAX)
    t = np.linspace(0.0, 10.0, 100_001)
    line = np.array([-3.0, 3.0])[None, :] + t[:, None] * unit[None, :]
    vals = TOY.forward({"x1": line[:, 0], "x2": line[:, 1]}, theta)
    t_peak = t[np.argmax(vals)]
    assert abs((path.n - 1) * step - t_peak) <= step
    assert np.linalg.norm(P[-1] - xs) > 0.5
    assert path.best() == dict(zip(NAMES, path.points[-1], strict=True))
    assert AscentPath.from_json(path.to_json()) == path


def test_steepest_ascent_recompute_reaches_concave_optimum(method: str) -> None:
    theta = TOY.theta(C0, B_MAX, A_MAX)
    xs = _stationary(B_MAX, A_MAX)
    step = 0.02
    path = steepest_ascent(
        TOY,
        theta,
        {"x1": -3.0, "x2": 3.0},
        step=step,
        n_steps=1000,
        method=method,
        recompute=True,
    )
    assert isinstance(path, AscentPath)
    assert path.stop == "decrease"
    assert np.all(np.diff(path.values) > 0)
    final = np.array(path.points[-1])
    assert np.linalg.norm(final - xs) < 2 * step
    straight = steepest_ascent(TOY, theta, {"x1": -3.0, "x2": 3.0}, step=step, n_steps=1000)
    assert isinstance(straight, AscentPath) and straight.values[-1] < path.values[-1]


def test_steepest_ascent_clips_to_bounds() -> None:
    theta = TOY.theta(C0, B_MAX, A_MAX)
    xs = _stationary(B_MAX, A_MAX)  # about (0.71, -0.14)
    bounds = Bounds(treatments=("x2", "x1"), low=(-5.0, -5.0), high=(5.0, 0.0))
    path = steepest_ascent(
        TOY,
        theta,
        {"x1": -3.0, "x2": 3.0},
        step=0.05,
        n_steps=500,
        bounds=bounds,
        recompute=True,
    )
    assert isinstance(path, AscentPath)
    arr = np.array(path.points)
    assert np.all(arr[:, 0] <= 0.0 + 1e-12) and np.all(arr[:, 1] >= -5.0)
    assert xs[0] > 0.0  # the unconstrained optimum is outside the box
    assert path.points[-1][0] == pytest.approx(0.0)
    assert path.stop in ("decrease", "boundary")
    with pytest.raises(ValueError):
        steepest_ascent(
            TOY,
            theta,
            {"x1": 0.0, "x2": 0.0},
            step=0.1,
            n_steps=3,
            bounds=Bounds(treatments=("x1",), low=(0.0,), high=(1.0,)),
        )
    with pytest.raises(ValueError):
        steepest_ascent(TOY, theta, {"x1": 0.0, "x2": 0.0}, step=0.0, n_steps=3)
    with pytest.raises(ValueError):
        steepest_ascent(TOY, theta, {"x1": 0.0, "x2": 0.0}, step=0.1, n_steps=0)


def test_steepest_ascent_converged_at_optimum(method: str) -> None:
    """jax certifies a zero gradient exactly; finite differences cannot (the round-off they
    see at a stationary point is what a real slope drowned by a large |f| looks like), so
    the finite walk refuses at step 0 instead of reporting "converged"."""
    theta = TOY.theta(C0, B_MAX, A_MAX)
    xs = _stationary(B_MAX, A_MAX)
    path = steepest_ascent(
        TOY, theta, dict(zip(NAMES, xs, strict=True)), step=0.1, n_steps=5, method=method
    )
    flat = TOY.theta(C0, np.zeros(2), np.zeros((2, 2)))
    still = steepest_ascent(TOY, flat, {"x1": 1.0, "x2": 1.0}, step=0.1, n_steps=5, method=method)
    if method == "jax":
        assert isinstance(path, AscentPath) and path.n == 1 and path.stop == "converged"
        assert isinstance(still, AscentPath) and still.stop == "converged" and still.n == 1
    else:
        for r in (path, still):
            assert isinstance(r, Unsupported) and "cannot resolve the gradient" in r.reason
            assert r.detail["step"] == "0" and r.detail["coordinates"] == "x1, x2"


def test_steepest_ascent_nan_gradient_is_unsupported_and_bounds_fix_it() -> None:
    """Hill with s < 1 just above dose 0: a central stencil steps to x < 0 where x^s is NaN.

    With bounds the stencil stays inside the box, but a start exactly on the
    zero-dose boundary has an infinite slope and is refused too (case B); a
    start strictly inside walks.
    """
    near_zero = {"x": 1e-8, "z": 1.0}
    with np.errstate(invalid="ignore"):
        r = steepest_ascent(HILL, HILL_THETA, near_zero, step=0.5, n_steps=5, method="finite")
    assert isinstance(r, Unsupported)
    assert "step 0" in r.reason and "gradient is not finite" in r.reason
    assert r.detail["step"] == "0" and "nan" in r.detail["gradient"]
    box = Bounds(treatments=("x", "z"), low=(0.0, 0.0), high=(10.0, 5.0))
    at_zero = steepest_ascent(
        HILL, HILL_THETA, {"x": 0.0, "z": 1.0}, step=0.5, n_steps=50, bounds=box, method="finite"
    )
    assert isinstance(at_zero, Unsupported) and "not finite at the boundary for x" in at_zero.reason
    start = {"x": 0.05, "z": 1.0}
    path = steepest_ascent(
        HILL, HILL_THETA, start, step=0.5, n_steps=50, bounds=box, method="finite"
    )
    assert isinstance(path, AscentPath)
    assert np.all(np.isfinite(path.values)) and np.all(np.diff(path.values) > 0)
    assert path.points[1][0] > 0.05  # moved up the dose axis
    arr = np.array(path.points)
    assert np.all(arr[:, 0] >= 0.0) and np.all(arr[:, 1] >= 0.0)
    assert path.stop in ("decrease", "boundary")
    with np.errstate(invalid="ignore"):
        again = steepest_ascent(
            HILL, HILL_THETA, near_zero, step=0.5, n_steps=5, method="finite", recompute=True
        )
    assert isinstance(again, Unsupported)


def test_steepest_ascent_jax_at_zero_dose_never_returns_a_non_finite_point(x64: None) -> None:
    """The true slope of x^0.8 at 0 is infinite. ``core.compile_jax`` guards ``Pow`` at a
    zero base (finite zero cotangent, see its ``Pow`` case), so jax alone would report a
    zero slope there and walk along z; the boundary check refuses instead."""
    box = Bounds(treatments=("x", "z"), low=(0.0, 0.0), high=(10.0, 5.0))
    r = steepest_ascent(
        HILL, HILL_THETA, {"x": 0.0, "z": 1.0}, step=0.5, n_steps=5, bounds=box, method="jax"
    )
    assert isinstance(r, Unsupported)
    assert "derivative not finite at the boundary for x" in r.reason
    assert r.detail["step"] == "0"


# -- canonical analysis ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("A", "b", "kind"),
    [
        (A_MAX, B_MAX, "maximum"),
        (np.array([[1.5, -0.2], [-0.2, 0.8]]), np.array([-1.0, 2.0]), "minimum"),
        (np.array([[1.0, 0.3], [0.3, -2.0]]), np.array([0.5, 0.5]), "saddle"),
    ],
)
def test_canonical_analysis_classifies_and_locates(
    method: str, A: npt.NDArray[np.float64], b: npt.NDArray[np.float64], kind: str
) -> None:
    theta = TOY.theta(C0, b, A)
    xs = _stationary(b, A)
    r = canonical_analysis(TOY, theta, {"x1": 1.0, "x2": -2.0}, method=method)
    assert isinstance(r, StationaryPoint)
    assert r.kind == kind
    np.testing.assert_allclose(np.array(r.point), xs, atol=1e-6)
    expected_value = float(TOY.forward({"x1": xs[[0]], "x2": xs[[1]]}, theta)[0])
    assert r.value == pytest.approx(expected_value, abs=1e-6)
    np.testing.assert_allclose(np.array(r.eigenvalues), np.linalg.eigvalsh(2 * A), atol=1e-6)
    V = np.array(r.eigenvectors).T
    np.testing.assert_allclose(V.T @ (2 * A) @ V, np.diag(r.eigenvalues), atol=1e-5)
    np.testing.assert_allclose(np.array(r.gradient), 2 * A @ np.array([1.0, -2.0]) + b, atol=1e-6)
    assert r.origin == (1.0, -2.0)
    assert StationaryPoint.from_json(r.to_json()) == r
    # from the origin, bounds supply the finite-difference scale
    at0 = canonical_analysis(TOY, theta, {"x1": 0.0, "x2": 0.0}, method=method, bounds=BIG_BOX)
    assert isinstance(at0, StationaryPoint) and at0.kind == kind
    np.testing.assert_allclose(np.array(at0.point), xs, atol=1e-6)


def test_canonical_analysis_singular_floor_is_method_aware() -> None:
    """At (0.3, 0.5) the gradient is (0.4, 0): x2's zero slope and zero curvature are
    negligible beside x1's resolved ones, so the finite Hessian is returned and found
    singular (at (0.5, 0.5) the gradient would be (0, 0) and unresolved first)."""
    theta = TOY.theta(C0, np.array([1.0, 0.0]), np.array([[-1.0, 0.0], [0.0, 0.0]]))
    r = canonical_analysis(TOY, theta, {"x1": 0.3, "x2": 0.5}, method="finite")
    assert isinstance(r, Unsupported) and "singular" in r.reason
    assert r.detail["method"] == "finite" and r.detail["floor"] == "1e-06"
    # A = [[-1, 1], [1, -1]] has an exact zero eigenvalue; finite-difference noise (~1e-9
    # relative) used to slip under an absolute 1e-12 guard and put the point at ~1e9
    rank1 = TOY.theta(C0, np.array([1.0, 0.0]), np.array([[-1.0, 1.0], [1.0, -1.0]]))
    f = canonical_analysis(TOY, rank1, {"x1": 1.0, "x2": 2.0}, method="finite")
    assert isinstance(f, Unsupported) and "singular" in f.reason
    # a ridge at 1e-7 relative is below the finite floor but far above the jax floor
    ridge = TOY.theta(C0, np.array([1.0, 0.0]), np.array([[-1.0, 0.0], [0.0, -1e-7]]))
    strict = canonical_analysis(TOY, ridge, {"x1": 0.3, "x2": 0.5}, tol=1e-9, method="finite")
    assert isinstance(strict, Unsupported) and "singular" in strict.reason
    with pytest.raises(ValueError):
        canonical_analysis(TOY, ridge, {"x1": 0.3, "x2": 0.5}, tol=-1.0)


def test_canonical_analysis_flat_direction_is_unsupported(x64: None) -> None:
    rank1 = TOY.theta(C0, np.array([1.0, 0.0]), np.array([[-1.0, 1.0], [1.0, -1.0]]))
    f = canonical_analysis(TOY, rank1, {"x1": 1.0, "x2": 2.0}, method="jax")
    assert isinstance(f, Unsupported) and "singular" in f.reason and f.detail["method"] == "jax"
    ridge = TOY.theta(C0, np.array([1.0, 0.0]), np.array([[-1.0, 0.0], [0.0, -1e-7]]))
    exact = canonical_analysis(TOY, ridge, {"x1": 0.5, "x2": 0.5}, tol=1e-9, method="jax")
    assert isinstance(exact, StationaryPoint) and exact.kind == "maximum"
    np.testing.assert_allclose(np.array(exact.point), [0.5, 0.0], atol=1e-9)
    flat = canonical_analysis(TOY, ridge, {"x1": 0.5, "x2": 0.5}, tol=1e-6, method="jax")
    assert isinstance(flat, Unsupported)
    assert "flat direction" in flat.reason and "ridge analysis" in flat.reason
    assert flat.detail["tol"] == "1e-06"
