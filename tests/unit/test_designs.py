"""surface.design: classical generators, criteria, Bayesian D-optimality, point exchange."""

from __future__ import annotations

import itertools
from collections.abc import Mapping

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
    value,
)
from axiom.surface.design import (
    Bounds,
    Design,
    a_criterion,
    bayesian_criterion,
    box_behnken,
    central_composite,
    d_criterion,
    defining_relation,
    e_criterion,
    equal_spacing,
    fractional_factorial,
    full_factorial,
    latin_hypercube,
    optimal_exchange,
)


class QuadraticToy:
    """``y = sum_c b_c * basis_c(x)`` over the full second-order basis, built from expr nodes."""

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


class HillToy:
    """``y = b0 + b1 * x^s / (k^s + x^s)``: linear in (b0, b1) at fixed (k, s)."""

    def __init__(self) -> None:
        dl = dimensionless()
        x_over_k = Div(
            numerator=Data(name="x", dimension=dl), denominator=Param(name="k", dimension=dl)
        )
        s = Param(name="s", dimension=dl)
        self.hill: Model = Div(
            numerator=Pow(base=x_over_k, exponent=s),
            denominator=Add(terms=(Const(value=1.0, dimension=dl), Pow(base=x_over_k, exponent=s))),
        )
        self._expr: Model = Add(
            terms=(
                Mul(factors=(Param(name="b0", dimension=dl), Const(value=1.0, dimension=dl))),
                Mul(factors=(Param(name="b1", dimension=dl), self.hill)),
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
        h = value(self.hill, data=dose, params=theta_at)
        return DesignMatrix(
            X=np.column_stack([np.ones_like(h), h]),
            columns=("b0", "b1"),
            offset=np.zeros_like(h),
            at={n: np.asarray(theta_at[n], dtype=float) for n in ("k", "s")},
        )


B2 = Bounds(treatments=("x1", "x2"), low=(0.0, 10.0), high=(4.0, 30.0))
B3 = Bounds(treatments=("x1", "x2", "x3"), low=(-1.0, 0.0, 0.0), high=(1.0, 2.0, 10.0))


def _coded_rows(bounds: Bounds, design: Design) -> set[tuple[float, ...]]:
    return {tuple(np.round(r, 9)) for r in bounds.to_coded(design.as_array())}


# -- specs -------------------------------------------------------------------------------


def test_bounds_validation() -> None:
    with pytest.raises(ValueError):
        Bounds(treatments=("a", "a"), low=(0.0, 0.0), high=(1.0, 1.0))
    with pytest.raises(ValueError):
        Bounds(treatments=("a",), low=(1.0,), high=(1.0,))
    with pytest.raises(ValueError):
        Bounds(treatments=("a", "b"), low=(0.0,), high=(1.0, 1.0))
    with pytest.raises(ValueError):
        Bounds(treatments=(), low=(), high=())
    assert B2.k == 2
    np.testing.assert_allclose(B2.from_coded([[-1.0, 1.0]]), [[0.0, 30.0]])
    np.testing.assert_allclose(B2.to_coded([[2.0, 20.0]]), [[0.0, 0.0]])
    np.testing.assert_allclose(B2.clip([[-3.0, 99.0]]), [[0.0, 30.0]])


def test_design_round_trip_and_accessors() -> None:
    d = central_composite(B2, center_points=2)
    back = Design.from_json(d.to_json())
    assert back == d and back.content_hash() == d.content_hash()
    assert d.n == 10 and d.k == 2
    assert d.as_array().shape == (10, 2)
    assert list(d.as_frame().columns) == ["x1", "x2"]
    doses = d.doses()
    np.testing.assert_array_equal(doses["x2"], d.as_array()[:, 1])
    with pytest.raises(ValueError):
        Design(treatments=("a", "b"), points=((0.0,),), kind="bad")
    with pytest.raises(ValueError):
        Design(treatments=("a",), points=(), kind="empty")


# -- classical generators ----------------------------------------------------------------


def test_central_composite_counts_and_alpha() -> None:
    d2 = central_composite(B2, center_points=4)
    assert d2.n == 4 + 4 + 4
    assert d2.detail["alpha"] == pytest.approx(4**0.25)
    assert d2.detail["n_cube"] == 4 and d2.detail["n_axial"] == 4 and d2.detail["n_center"] == 4
    coded = _coded_rows(B2, d2)
    assert {(-1.0, -1.0), (1.0, 1.0), (0.0, 0.0)} <= coded
    a = round(4**0.25, 9)
    assert {(a, 0.0), (-a, 0.0), (0.0, a), (0.0, -a)} <= coded
    d3 = central_composite(B3, center_points=6)
    assert d3.n == 8 + 6 + 6
    assert d3.detail["alpha"] == pytest.approx(8**0.25)
    face = central_composite(B2, alpha="face", center_points=1)
    assert face.detail["alpha"] == 1.0
    assert np.all(np.abs(B2.to_coded(face.as_array())) <= 1 + 1e-12)
    fixed = central_composite(B2, alpha=1.5, center_points=0)
    assert fixed.n == 8 and fixed.detail["alpha"] == 1.5
    ins = central_composite(B2, inscribed=True, center_points=0)
    assert np.all(np.abs(B2.to_coded(ins.as_array())) <= 1 + 1e-12)
    assert np.max(np.abs(B2.to_coded(ins.as_array()))) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        central_composite(B2, alpha=0.0)
    with pytest.raises(ValueError):
        central_composite(B2, center_points=-1)


def test_central_composite_fractional_cube() -> None:
    b5 = Bounds(treatments=tuple(f"t{i}" for i in range(5)), low=(0.0,) * 5, high=(1.0,) * 5)
    d = central_composite(b5, fraction=1, center_points=2)
    assert d.detail["n_cube"] == 16 and d.n == 16 + 10 + 2
    assert d.detail["resolution"] == 5
    assert d.detail["alpha"] == pytest.approx(16**0.25)
    b6 = Bounds(treatments=tuple(f"t{i}" for i in range(6)), low=(0.0,) * 6, high=(1.0,) * 6)
    d6 = central_composite(b6, fraction=2, center_points=0)
    assert d6.detail["n_cube"] == 16 and d6.detail["resolution"] == 4
    with pytest.raises(ValueError):
        central_composite(B2, fraction=1)


def test_box_behnken() -> None:
    d = box_behnken(B3, center_points=3)
    assert d.n == 12 + 3
    coded = B3.to_coded(d.as_array())
    edge = coded[:12]
    assert np.all(np.sum(np.abs(edge) > 1e-12, axis=1) == 2)
    assert np.all(np.isin(np.round(edge, 9), (-1.0, 0.0, 1.0)))
    assert len({tuple(r) for r in np.round(edge, 9)}) == 12
    np.testing.assert_allclose(coded[12:], 0.0, atol=1e-12)
    with pytest.raises(ValueError):
        box_behnken(B2)


def test_full_factorial_counts_and_levels() -> None:
    d = full_factorial(B2, (2, 3))
    assert d.n == 6 and d.kind == "full_factorial"
    arr = d.as_array()
    assert sorted(set(arr[:, 0])) == [0.0, 4.0]
    assert sorted(set(arr[:, 1])) == [10.0, 20.0, 30.0]
    assert len({tuple(r) for r in arr}) == 6
    d3 = full_factorial(B3, 3)
    assert d3.n == 27 and d3.detail["levels[x3]"] == 3
    with pytest.raises(ValueError):
        full_factorial(B2, (1, 2))
    with pytest.raises(ValueError):
        full_factorial(B2, (2, 2, 2))


def test_equal_spacing_is_a_regular_grid() -> None:
    d = equal_spacing(B2, 3)
    assert d.n == 9 and d.kind == "equal_spacing"
    assert sorted(set(d.as_array()[:, 0])) == [0.0, 2.0, 4.0]
    assert sorted(set(d.as_array()[:, 1])) == [10.0, 20.0, 30.0]


def test_fractional_factorial_parse_and_confounding() -> None:
    b4 = Bounds(treatments=("a", "b", "c", "d"), low=(0.0,) * 4, high=(1.0,) * 4)
    d = fractional_factorial(b4, ("D=ABC",))
    assert d.n == 8 and d.detail["resolution"] == 4 and d.detail["fraction"] == 1
    coded = b4.to_coded(d.as_array())
    np.testing.assert_allclose(coded[:, 3], coded[:, 0] * coded[:, 1] * coded[:, 2])
    assert len({tuple(r) for r in coded[:, :3]}) == 8
    assert defining_relation(("D=ABC",)) == ("ABCD",)
    neg = fractional_factorial(b4, ("D=-ABC",))
    neg_coded = b4.to_coded(neg.as_array())
    np.testing.assert_allclose(neg_coded[:, 3], -np.prod(neg_coded[:, :3], axis=1))
    b5 = Bounds(treatments=tuple("abcde"), low=(0.0,) * 5, high=(1.0,) * 5)
    d5 = fractional_factorial(b5, ("D=AB", "E=AC"))
    assert d5.n == 8 and d5.detail["resolution"] == 3
    assert defining_relation(("D=AB", "E=AC")) == ("ABD", "ACE", "BCDE")
    with pytest.raises(ValueError):
        fractional_factorial(b4, ("E=ABC",))
    with pytest.raises(ValueError):
        fractional_factorial(b4, ("D=ABC", "D=AB"))
    with pytest.raises(ValueError):
        fractional_factorial(b4, ("C=AB", "D=ABC"))
    with pytest.raises(ValueError):
        fractional_factorial(b4, ("DABC",))
    with pytest.raises(ValueError):
        fractional_factorial(b4, ())


def test_latin_hypercube_stratification_and_seed() -> None:
    n = 7
    d = latin_hypercube(B3, n, seed=3)
    assert d.n == n and d.detail["seed"] == 3
    u = (B3.to_coded(d.as_array()) + 1.0) / 2.0
    for j in range(3):
        assert sorted(np.floor(u[:, j] * n).astype(int)) == list(range(n))
    c = latin_hypercube(B3, n, seed=3, centered=True)
    uc = (B3.to_coded(c.as_array()) + 1.0) / 2.0
    mid = np.broadcast_to(((np.arange(n) + 0.5) / n)[:, None], (n, 3))
    np.testing.assert_allclose(np.sort(uc, axis=0), mid)
    assert latin_hypercube(B3, n, seed=3) == d
    assert latin_hypercube(B3, n, seed=4) != d
    assert d.detail["min_distance"] > 0
    single = latin_hypercube(B3, 1, seed=1)
    assert single.n == 1 and "min_distance" not in single.detail  # undefined, not 0.0
    with pytest.raises(ValueError):
        latin_hypercube(B3, 0, seed=1)


# -- criteria ----------------------------------------------------------------------------


def test_criteria_on_identity_and_singular_matrices() -> None:
    eye = np.eye(3)
    assert d_criterion(eye) == 0.0
    assert a_criterion(eye) == -3.0
    assert e_criterion(eye) == 1.0
    singular = np.column_stack([np.ones(4), np.ones(4)])
    assert d_criterion(singular) == -np.inf
    assert a_criterion(singular) == -np.inf
    assert e_criterion(singular) == 0.0
    dm = DesignMatrix(X=2.0 * eye, columns=("a", "b", "c"), offset=np.zeros(3), at={})
    assert d_criterion(dm) == pytest.approx(3 * np.log(4.0))
    assert e_criterion(dm) == 4.0
    assert d_criterion(np.array([[np.nan, 1.0]])) == -np.inf
    with pytest.raises(ValueError):
        d_criterion(np.ones(3))


def test_bayesian_criterion_one_draw_is_local() -> None:
    toy = HillToy()
    bounds = Bounds(treatments=("x",), low=(0.0,), high=(10.0,))
    design = equal_spacing(bounds, 6)
    th1 = {"k": 2.0, "s": 1.5}
    th2 = {"k": 6.0, "s": 0.8}
    local = d_criterion(toy.linearize(design.doses(), th1).X)
    assert bayesian_criterion(toy, design, [th1]) == pytest.approx(local)
    two = bayesian_criterion(toy, design, [th1, th2])
    other = d_criterion(toy.linearize(design.doses(), th2).X)
    assert two == pytest.approx(0.5 * (local + other))
    assert bayesian_criterion(toy, design, [th1], criterion="e") == pytest.approx(
        e_criterion(toy.linearize(design.doses(), th1).X)
    )
    with pytest.raises(ValueError):
        bayesian_criterion(toy, design, [])


# -- exchange ----------------------------------------------------------------------------


def test_d_optimal_exchange_beats_equal_spacing() -> None:
    toy = QuadraticToy(("x1", "x2"))
    candidates = full_factorial(B2, 5)
    naive = equal_spacing(B2, 4)
    theta = [{}]
    best = optimal_exchange(toy, candidates, naive.n, theta, seed=0)
    assert isinstance(best, Design)
    assert best.n == naive.n and best.kind == "d_optimal"
    naive_d = d_criterion(toy.linearize(naive.doses(), {}).X)
    best_d = d_criterion(toy.linearize(best.doses(), {}).X)
    margin = best_d - naive_d
    print(f"\nD-optimal exchange margin over equal spacing (log det): {margin:.4f}")
    assert margin > 0
    assert margin > 0.9  # recorded: 1.0203 (5x5 candidate grid, n=16, full quadratic, seed=0)
    assert best.detail["criterion"] == pytest.approx(best_d)
    assert best.detail["improvement"] >= 0
    cand_rows = {tuple(r) for r in candidates.points}
    assert all(p in cand_rows for p in best.points)
    assert optimal_exchange(toy, candidates, naive.n, theta, seed=0) == best


def test_exchange_bayesian_and_other_criteria() -> None:
    toy = HillToy()
    bounds = Bounds(treatments=("x",), low=(0.0,), high=(10.0,))
    candidates = equal_spacing(bounds, 11)
    draws = [{"k": 2.0, "s": 1.5}, {"k": 6.0, "s": 0.8}]
    d = optimal_exchange(toy, candidates, 6, draws, seed=1, n_restarts=2)
    assert isinstance(d, Design)
    assert d.detail["criterion"] == pytest.approx(bayesian_criterion(toy, d, draws))
    assert d.detail["n_draws"] == 2
    for crit in ("a", "e"):
        r = optimal_exchange(toy, candidates, 6, draws, criterion=crit, seed=1, n_restarts=1)
        assert isinstance(r, Design) and r.kind == f"{crit}_optimal"
        assert r.detail["criterion"] == pytest.approx(bayesian_criterion(toy, r, draws, crit))


def test_exchange_singular_candidates_is_unsupported() -> None:
    toy = QuadraticToy(("x1", "x2"))
    two_points = Design(treatments=("x1", "x2"), points=((0.0, 10.0), (4.0, 30.0)), kind="tiny")
    r = optimal_exchange(toy, two_points, 8, [{}], seed=0)
    assert isinstance(r, Unsupported)
    assert "no restart reached a nonsingular design" in r.reason
    assert r.detail["n_coefficients"] == "6" and r.detail["n_restarts"] == "3"
    for crit in ("a", "e"):
        assert isinstance(
            optimal_exchange(toy, two_points, 8, [{}], seed=0, criterion=crit), Unsupported
        )
    with pytest.raises(ValueError):
        optimal_exchange(toy, two_points, 0, [{}])
    with pytest.raises(ValueError):
        optimal_exchange(toy, two_points, 3, [])
    with pytest.raises(ValueError):
        optimal_exchange(toy, two_points, 3, [{}], criterion="z")  # type: ignore[arg-type]


def _collinear_candidates() -> Design:
    """Nine distinct points on the line ``x2 = 10 + 5 x1``: rank 3 for the 6-column quadratic."""
    x1 = np.linspace(0.0, 4.0, 9)
    return Design(
        treatments=("x1", "x2"),
        points=tuple((float(a), float(10.0 + 5.0 * a)) for a in x1),
        kind="line",
    )


def test_criteria_reject_rank_deficient_non_duplicate_rows() -> None:
    toy = QuadraticToy(("x1", "x2"))
    X = toy.linearize(_collinear_candidates().doses(), {}).X
    assert np.linalg.matrix_rank(X) == 3
    # slogdet of X'X is finite here (rounding noise), which is the bug being pinned down
    sign, logdet = np.linalg.slogdet(X.T @ X)
    assert np.isfinite(logdet)
    assert d_criterion(X) == -np.inf
    assert a_criterion(X) == -np.inf
    assert e_criterion(X) == 0.0
    # fewer rows than columns is singular by counting, whatever the numbers say
    X3 = toy.linearize(full_factorial(B2, 5).doses(), {}).X[:3]
    assert d_criterion(X3) == -np.inf and a_criterion(X3) == -np.inf and e_criterion(X3) == 0.0
    # the relative floor is shared: scaling X does not change the verdict
    assert d_criterion(1e3 * X) == -np.inf
    full = toy.linearize(full_factorial(B2, 3).doses(), {}).X
    assert np.isfinite(d_criterion(full))
    assert d_criterion(10.0 * full) == pytest.approx(d_criterion(full) + 6 * 2 * np.log(10.0))


def _scaled(bounds: Bounds, factor: float) -> Bounds:
    return Bounds(
        treatments=bounds.treatments,
        low=tuple(factor * v for v in bounds.low),
        high=tuple(factor * v for v in bounds.high),
    )


def _column_scale_shift(toy: QuadraticToy, bounds: Bounds, factor: float) -> float:
    """``2 sum log c_j``: how ``log det(X'X)`` moves when every dose is multiplied by ``factor``.

    Basis columns scale by ``factor^degree``, a constant diagonal ``S`` shared
    by every design on the bounds — so rankings are invariant to it.
    """
    X0 = toy.linearize(full_factorial(bounds, 3).doses(), {}).X
    X1 = toy.linearize(full_factorial(_scaled(bounds, factor), 3).doses(), {}).X
    c = np.max(np.abs(X1), axis=0) / np.max(np.abs(X0), axis=0)
    return float(2.0 * np.sum(np.log(c)))


def test_criteria_in_raw_dose_units_are_finite_and_accurate() -> None:
    """A 3^2 factorial on [0, 1000]^2 is well posed (lstsq recovers coefficients); cond ~2e6."""
    toy = QuadraticToy(("x1", "x2"))
    bounds = Bounds(treatments=("x1", "x2"), low=(0.0, 0.0), high=(1000.0, 1000.0))
    X = toy.linearize(full_factorial(bounds, 3).doses(), {}).X
    assert np.linalg.cond(X) > 1e6
    assert np.linalg.matrix_rank(X) == 6
    d, a, e = d_criterion(X), a_criterion(X), e_criterion(X)
    assert np.isfinite(d) and np.isfinite(a) and e > 0.0
    # independent references through the coded matrix: X = Xc S with S = diag(max-abs)
    S = np.max(np.abs(X), axis=0)
    Xc = X / S
    sign, logdet_c = np.linalg.slogdet(Xc.T @ Xc)
    assert sign > 0
    assert d == pytest.approx(logdet_c + 2.0 * np.sum(np.log(S)), rel=1e-12)
    pinv = np.linalg.pinv(Xc) / S[:, None]  # X^+ = S^-1 Xc^+
    assert a == pytest.approx(-np.sum(pinv * pinv), rel=1e-10)
    assert e == pytest.approx(1.0 / np.linalg.norm(pinv, 2) ** 2, rel=1e-10)
    # against unit bounds: same verdict, and the D shifts by the column-scale constant
    unit = toy.linearize(full_factorial(_scaled(bounds, 1e-3), 3).doses(), {}).X
    shift = _column_scale_shift(toy, _scaled(bounds, 1e-3), 1e3)
    assert d == pytest.approx(d_criterion(unit) + shift, rel=1e-12)
    # still singular in raw units when the geometry is singular
    line = Design(
        treatments=("x1", "x2"),
        points=tuple((float(a_), float(1000.0 * 0.25 * a_ / 4.0 + 2500.0)) for a_ in range(9)),
        kind="line",
    )
    Xl = toy.linearize(line.doses(), {}).X
    assert np.linalg.matrix_rank(Xl) == 3
    assert d_criterion(Xl) == -np.inf and a_criterion(Xl) == -np.inf and e_criterion(Xl) == 0.0
    assert d_criterion(X[:5]) == -np.inf  # n < p, whatever the units


def test_design_rankings_are_invariant_to_dose_units() -> None:
    toy = QuadraticToy(("x1", "x2"))
    for factor in (1e3, 1e-3):
        big = _scaled(B2, factor)
        shift = _column_scale_shift(toy, B2, factor)
        pairs = (
            (central_composite(B2, center_points=2), central_composite(big, center_points=2)),
            (equal_spacing(B2, 4), equal_spacing(big, 4)),
        )
        values = []
        for small_d, big_d in pairs:
            ds = d_criterion(toy.linearize(small_d.doses(), {}).X)
            db = d_criterion(toy.linearize(big_d.doses(), {}).X)
            assert np.isfinite(ds) and np.isfinite(db)
            assert db == pytest.approx(ds + shift, rel=1e-12)
            values.append((ds, db))
        (ds1, db1), (ds2, db2) = values
        assert (ds1 > ds2) == (db1 > db2)
        assert db1 - db2 == pytest.approx(ds1 - ds2, rel=1e-10)


def test_exchange_in_raw_dose_units_succeeds_and_matches_coded_units() -> None:
    toy = QuadraticToy(("x1", "x2"))
    unit = Bounds(treatments=("x1", "x2"), low=(0.0, 0.0), high=(1.0, 1.0))
    raw = _scaled(unit, 1e3)
    r_raw = optimal_exchange(toy, full_factorial(raw, 5), 9, [{}], seed=0)
    r_unit = optimal_exchange(toy, full_factorial(unit, 5), 9, [{}], seed=0)
    assert isinstance(r_raw, Design), r_raw
    assert isinstance(r_unit, Design), r_unit
    np.testing.assert_allclose(raw.to_coded(r_raw.as_array()), unit.to_coded(r_unit.as_array()))
    shift = _column_scale_shift(toy, unit, 1e3)
    assert r_raw.detail["criterion"] == pytest.approx(r_unit.detail["criterion"] + shift, rel=1e-12)
    assert r_raw.detail["improvement"] == pytest.approx(r_unit.detail["improvement"], rel=1e-9)
    # the scale-free distance from singularity is reported and is a property of the geometry
    assert r_raw.detail["condition"] == pytest.approx(r_unit.detail["condition"], rel=1e-9)
    assert 1.0 < r_raw.detail["condition"] < 1e3
    for crit in ("a", "e"):
        r = optimal_exchange(toy, full_factorial(raw, 5), 9, [{}], seed=0, criterion=crit)
        assert isinstance(r, Design), (crit, r)
    # the collinear line and n < p stay Unsupported in raw units too
    line = Design(
        treatments=("x1", "x2"),
        points=tuple((float(250.0 * i), float(1000.0 + 5.0 * 250.0 * i)) for i in range(9)),
        kind="line",
    )
    assert isinstance(optimal_exchange(toy, line, 8, [{}], seed=0), Unsupported)
    assert isinstance(optimal_exchange(toy, full_factorial(raw, 5), 3, [{}], seed=0), Unsupported)


def test_exchange_over_collinear_candidates_and_n_below_p_is_unsupported() -> None:
    toy = QuadraticToy(("x1", "x2"))
    for crit in ("d", "a", "e"):
        r = optimal_exchange(toy, _collinear_candidates(), 8, [{}], seed=0, criterion=crit)
        assert isinstance(r, Unsupported), crit
        assert "no restart reached a nonsingular design" in r.reason
    small = optimal_exchange(toy, full_factorial(B2, 5), 3, [{}], seed=0)
    assert isinstance(small, Unsupported)
    assert "n=3 runs cannot estimate 6 linear coefficients" in small.reason
    assert small.detail["n_coefficients"] == "6"
    exact = optimal_exchange(toy, full_factorial(B2, 5), 6, [{}], seed=0)
    assert isinstance(exact, Design) and np.isfinite(exact.detail["criterion"])


def test_exchange_escapes_a_singular_start_with_one_restart() -> None:
    """1-D quadratic (p=3), three candidates, n=4: over half the random starts are singular."""
    toy = QuadraticToy(("x",))
    bounds = Bounds(treatments=("x",), low=(0.0,), high=(2.0,))
    candidates = equal_spacing(bounds, 3)
    best_d = d_criterion(toy.linearize({"x": np.array([0.0, 1.0, 1.0, 2.0])}, {}).X)
    n_singular_starts = 0
    for seed in range(30):
        start = np.random.default_rng(seed).choice(3, size=4, replace=True)
        if len(set(start.tolist())) < 3:
            n_singular_starts += 1
        r = optimal_exchange(toy, candidates, 4, [{}], seed=seed, n_restarts=1)
        assert isinstance(r, Design), f"seed {seed} (start {start.tolist()}) -> {r}"
        assert r.detail["criterion"] == pytest.approx(best_d)
        if len(set(start.tolist())) < 3:
            assert "start_criterion" not in r.detail
    assert n_singular_starts >= 10  # the case is exercised, not dodged


def test_box_behnken_six_and_seven_are_the_published_block_designs() -> None:
    for k, n_runs, n_blocks in ((6, 48, 6), (7, 56, 7)):
        b = Bounds(treatments=tuple(f"t{i}" for i in range(k)), low=(0.0,) * k, high=(1.0,) * k)
        d = box_behnken(b, center_points=6)
        assert d.n == n_runs + 6
        assert d.detail["n_blocks"] == n_blocks and d.detail["block_size"] == 3
        coded = np.round(b.to_coded(d.as_array()), 9)
        runs = coded[:n_runs]
        assert np.all(np.sum(np.abs(runs) > 0, axis=1) == 3)
        assert np.all(np.isin(runs, (-1.0, 0.0, 1.0)))
        assert len({tuple(r) for r in runs}) == n_runs
        np.testing.assert_allclose(coded[n_runs:], 0.0)
        blocks = {tuple(np.flatnonzero(r)) for r in runs}
        assert len(blocks) == n_blocks
        per_factor = np.sum(np.abs(runs) > 0, axis=0)
        assert np.all(per_factor == 3 * 8)  # every treatment in exactly three blocks
        pair_counts = {
            pair: sum(1 for blk in blocks if set(pair) <= set(blk))
            for pair in itertools.combinations(range(k), 2)
        }
        if k == 7:
            assert set(pair_counts.values()) == {1}  # the Fano plane: a BIBD(7, 3, 1)
        else:
            twice = {p for p, c in pair_counts.items() if c == 2}
            assert twice == {(0, 3), (1, 4), (2, 5)} and set(pair_counts.values()) == {1, 2}
        # second-order model is estimable from the block runs alone
        toy = QuadraticToy(b.treatments)
        assert np.isfinite(d_criterion(toy.linearize(d.doses(), {}).X))
    b8 = Bounds(treatments=tuple(f"t{i}" for i in range(8)), low=(0.0,) * 8, high=(1.0,) * 8)
    with pytest.raises(ValueError, match="3 to 7 treatments"):
        box_behnken(b8)
    b5 = Bounds(treatments=tuple("abcde"), low=(0.0,) * 5, high=(1.0,) * 5)
    d5 = box_behnken(b5, center_points=0)
    assert d5.n == 40 and d5.detail["block_size"] == 2 and d5.detail["n_blocks"] == 10


def test_all_generators_share_bounds_treatments() -> None:
    for d in (
        central_composite(B3),
        box_behnken(B3),
        full_factorial(B3, 2),
        latin_hypercube(B3, 5, seed=0),
        equal_spacing(B3, 2),
    ):
        assert d.treatments == B3.treatments
        arr = d.as_array()
        assert arr.shape[1] == 3
    names = list(itertools.chain.from_iterable([B2.treatments, B3.treatments]))
    assert len(names) == 5
