"""``estimands.realize`` / ``evaluate``: recovery against ``sim`` truth, arithmetic against numpy,
typed degradation, identification, dimensions, and unit conversion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pytest
from scipy.integrate import quad

from axiom.core import (
    UNITS,
    Assumption,
    Blocked,
    Capability,
    D,
    DimensionError,
    Intervention,
    Outcome,
    Population,
    Posterior,
    PredictiveDraws,
    Spec,
    SupportsEstimands,
    TimeWindow,
    Treatment,
    UnitConversionError,
    Unsupported,
    Verdict,
    dimensionless,
    summarize,
)
from axiom.estimands import Estimand, Level, Quantity, QuantityKind
from axiom.estimands.evaluate import (
    QUADRATURE_NODES,
    EstimandResult,
    RealizedDraws,
    evaluate,
    realize,
)
from axiom.sim import DosePlan, arms_world, surface_world
from axiom.sim.surface_world import SurfaceWorld
from axiom.surface import FitResult, fit
from axiom.surface.carryover import GeometricCarryover

Array = npt.NDArray[np.float64]
Basis = Literal["per_period", "cumulative"]
LevelUnit = Literal["individual", "cluster", "aggregate"]

# -- worlds --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def arms() -> SurfaceWorld:
    return arms_world(
        n_units=30,
        treatments=("dose",),
        doses=DosePlan(scale=50.0, spread=0.6),
        truth={"beta_dose": 4.0, "alpha": 2.0},
        noise_sd=0.2,
        seed=1,
    )


@pytest.fixture(scope="module")
def arms_fit(arms: SurfaceWorld) -> FitResult:
    res = fit(arms.spec, arms.panel, backend="laplace", draws=150, chains=2, seed=2)
    assert isinstance(res.posterior, Posterior) and res.converged
    return res


@pytest.fixture(scope="module")
def panel() -> SurfaceWorld:
    return surface_world(
        n_units=4,
        n_periods=20,
        treatments=("a",),
        carryover={"a": GeometricCarryover(max_lag=3)},
        doses=DosePlan(scale=50.0, zero_fraction=0.1),
        noise_sd=0.2,
        seed=2,
    )


@pytest.fixture(scope="module")
def panel_fit(panel: SurfaceWorld) -> FitResult:
    res = fit(panel.spec, panel.panel, backend="laplace", draws=100, chains=2, seed=3)
    assert isinstance(res.posterior, Posterior)
    return res


@pytest.fixture(scope="module")
def long_carry() -> SurfaceWorld:
    """A six-lag carryover: the window edges are where a marginal's horizon matters."""
    return surface_world(
        n_units=3,
        n_periods=20,
        treatments=("a",),
        carryover={"a": GeometricCarryover(max_lag=6)},
        doses=DosePlan(scale=50.0),
        noise_sd=0.2,
        seed=7,
    )


@pytest.fixture(scope="module")
def long_carry_fit(long_carry: SurfaceWorld) -> FitResult:
    res = fit(long_carry.spec, long_carry.panel, backend="laplace", draws=40, chains=1, seed=7)
    assert isinstance(res.posterior, Posterior)
    return res


def _dim(kind: QuantityKind, outcome: Any, dose: Any) -> Any:
    return {
        "contrast": outcome,
        "marginal": outcome / dose,
        "ratio": outcome / dose,
        "elasticity": dimensionless(),
        "area": outcome * dose,
    }[kind]


def _estimand(
    treatment: Treatment,
    outcome: Outcome,
    n_periods: int,
    kind: QuantityKind = "contrast",
    **over: Any,
) -> Estimand:
    ref = None if kind in ("marginal", "elasticity") else Intervention(doses={treatment.name: 0.0})
    base: dict[str, Any] = dict(
        name=f"{kind}_probe",
        quantity=Quantity(kind=kind),
        treatment=treatment,
        intervention=Intervention(doses={treatment.name: 80.0}),
        reference=ref,
        outcome=outcome,
        population=Population(name="all"),
        window=TimeWindow(start=0, stop=n_periods),
        level=Level(unit="individual"),
        dimension=_dim(kind, outcome.dim, treatment.dim),
    )
    base.update(over)
    return Estimand(**base)


def _world_estimand(world: SurfaceWorld, kind: QuantityKind = "contrast", **over: Any) -> Estimand:
    return _estimand(world.spec.treatments[0], world.spec.outcome, world.n_periods, kind, **over)


def _covers(result: EstimandResult, truth: float) -> None:
    s = result.summary
    assert s.interval.contains(truth) or abs(s.mean - truth) <= 3.0 * s.sd + 0.02 * abs(
        truth
    ), f"truth {truth} vs {s.mean} ± {s.sd} {s.interval}"


def _agg(grid: Array, window: TimeWindow, level: LevelUnit) -> float:
    """The facet machinery in numpy: basis over the window's periods, then the level."""
    g = np.asarray(grid, dtype=float)[:, window.start : window.stop]
    per_unit = g.sum(axis=-1) if window.basis == "cumulative" else g.mean(axis=-1)
    return float(per_unit.mean() if level == "individual" else per_unit.sum())


# -- a hand-built producer with a closed form ------------------------------------------------


class _Linear:
    """``y[u, t] = alpha + beta · x[u, t]`` with (2, 5) draws of ``alpha`` and ``beta``.

    Satisfies ``SupportsEstimands`` so every aggregation rule can be checked
    against numpy in closed form. ``marginal_under`` is ``beta`` on the
    intervention's support and zero off it — the derivative of ``y`` under a
    common shift of the support's doses.
    """

    def __init__(
        self,
        x: Array,
        *,
        caps: frozenset[Capability] | None = None,
        report_doses: bool = True,
        labels: Sequence[str] | None = None,
        outcome_unit: str | None = "kg",
        dose_unit: str | None = "USD",
        marginal_unsupported: bool = False,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.alpha = 1.0 + rng.normal(0.0, 0.1, size=(2, 5))
        self.beta = 0.5 + rng.normal(0.0, 0.05, size=(2, 5))
        self.x = np.asarray(x, dtype=float)
        self.caps = (
            caps
            if caps is not None
            else frozenset(
                {
                    Capability.COUNTERFACTUAL,
                    Capability.MARGINAL,
                    Capability.TIME_WINDOW,
                    Capability.PER_UNIT,
                }
            )
        )
        self.report_doses = report_doses
        self.labels = tuple(labels or (f"u{i}" for i in range(self.x.shape[0])))
        self._outcome_unit = outcome_unit
        self._dose_unit = dose_unit
        self.marginal_unsupported = marginal_unsupported
        self.provenance: dict[str, Any] = {"kind": "linear double"}

    # SupportsPosterior
    def draws(self, name: str) -> Array:
        return {"alpha": self.alpha, "beta": self.beta}[name]

    def names(self) -> frozenset[str]:
        return frozenset({"alpha", "beta"})

    def coords(self) -> Mapping[str, Sequence[Any]]:
        return {}

    def n_draws(self) -> int:
        return 10

    # SupportsIntervention / SupportsEstimands
    @property
    def treatments(self) -> tuple[Treatment, ...]:
        return (Treatment(name="x", dimension=D.currency, unit=self._dose_unit),)

    @property
    def declared_estimands(self) -> tuple[str, ...]:
        return ()

    @property
    def outcome_unit(self) -> str | None:
        return self._outcome_unit

    def dose_unit(self, treatment: str) -> str | None:
        return self._dose_unit

    def capabilities(self) -> frozenset[Capability]:
        return self.caps

    def support(self, iv: Intervention) -> Array:
        out = np.zeros(self.x.shape)
        sl = slice(None) if iv.window is None else slice(iv.window.start, iv.window.stop)
        out[:, sl] = 1.0
        return out

    def realized(self, iv: Intervention) -> Array:
        x = self.x.copy()
        sl = slice(None) if iv.window is None else slice(iv.window.start, iv.window.stop)
        level = iv.doses["x"]
        if iv.mode == "set":
            x[:, sl] = level
        elif iv.mode == "scale":
            x[:, sl] *= level
        else:
            x[:, sl] += level
        return x

    def _pack(
        self, values: Array, iv: Intervention, window: TimeWindow | None, seed: int | None
    ) -> PredictiveDraws:
        sl = slice(None) if window is None else slice(window.start, window.stop)
        coords: dict[str, list[Any]] = {"unit": list(self.labels)}
        if self.report_doses:
            coords["x"] = [float(v) for v in self.realized(iv)[:, sl].reshape(-1)]
        return PredictiveDraws(
            values=values[..., sl], intervention=iv, window=window, coords=coords, seed=seed
        )

    def predict_under(
        self, iv: Intervention, window: TimeWindow | None = None, seed: int | None = None
    ) -> PredictiveDraws | Unsupported:
        x = self.realized(iv)
        values = self.alpha[:, :, None, None] + self.beta[:, :, None, None] * x[None, None]
        return self._pack(values, iv, window, seed)

    def marginal_under(
        self,
        iv: Intervention,
        treatment: str,
        window: TimeWindow | None = None,
        seed: int | None = None,
    ) -> PredictiveDraws | Unsupported:
        if self.marginal_unsupported:
            return Unsupported(reason="this double declines marginal_under", missing=("marginal",))
        values = self.beta[:, :, None, None] * self.support(iv)[None, None]
        return self._pack(np.asarray(values, dtype=np.float64), iv, window, seed)


_X = np.arange(18.0).reshape(3, 6) + 1.0  # 3 units × 6 periods, all doses positive
_T = Treatment(name="x", dimension=D.currency, unit="USD")
_Y = Outcome(name="y", dimension=D.outcome, unit="kg")


def _lin(kind: QuantityKind = "contrast", **over: Any) -> Estimand:
    treatment = over.pop("treatment", _T)
    outcome = over.pop("outcome", _Y)
    return _estimand(treatment, outcome, 6, kind, **over)


# -- recovery against sim truth ----------------------------------------------------------------


def test_contrast_recovers_truth_and_carries_provenance(
    arms: SurfaceWorld, arms_fit: FitResult
) -> None:
    e = _world_estimand(arms, "contrast")
    r = realize(e, arms_fit, definition="eti", mass=0.95, seed=0)
    assert isinstance(r, EstimandResult)
    truth = float(np.mean(arms.forward({"dose": 80.0}) - arms.forward({"dose": 0.0})))
    _covers(r, truth)
    assert r.summary.interval.definition == "eti" and r.summary.interval.mass == 0.95
    assert r.summary.n == r.n_draws == 300
    assert r.estimand_hash == e.content_hash() and r.estimand_name == e.name
    assert r.kind == "contrast" and r.dimension == D.outcome and r.unit is None
    assert r.status == "downgraded" and r.identification is None
    # the world's outcome has no unit: the equality of units is an assumption, on record
    assert [a.name for a in r.assumptions] == [
        "identification_not_checked",
        "units_assumed_equal",
    ]
    assert all(a.state == "unverified" for a in r.assumptions) and r.ledger == ()
    assert "outcome y" in r.assumptions[1].detail and "treatment" not in str(
        list(r.assumptions[1].detail)
    )
    assert r.producer_hash == arms_fit.provenance["model_hash"]
    assert r.detail["dose_iv"] == 80.0 and r.detail["dose_ref"] == 0.0
    assert r.detail["n_units"] == 30.0 and r.detail["n_periods"] == 1.0
    assert r.detail["basis"] == "cumulative" and r.detail["level"] == "individual"
    assert r.detail["intervention_mode"] == "set"
    assert Spec.from_json(r.to_json()) == r


def test_ratio_marginal_elasticity_area_recover_truth(
    arms: SurfaceWorld, arms_fit: FitResult
) -> None:
    y80 = float(np.mean(arms.forward({"dose": 80.0})))
    y0 = float(np.mean(arms.forward({"dose": 0.0})))
    m = arms.marginal("dose", dose={"dose": 80.0})
    assert not isinstance(m, Unsupported)
    m80 = float(np.mean(m))
    truths: dict[QuantityKind, float] = {
        "ratio": (y80 - y0) / 80.0,
        "marginal": m80,
        "elasticity": m80 * 80.0 / y80,
        "area": quad(lambda d: float(np.mean(arms.forward({"dose": d}))), 0.0, 80.0)[0],
    }
    for kind, truth in truths.items():
        r = realize(_world_estimand(arms, kind), arms_fit, mass=0.95)
        assert isinstance(r, EstimandResult), (kind, r)
        _covers(r, truth)
        assert r.dimension == _dim(kind, D.outcome, D.currency)
    area = realize(_world_estimand(arms, "area"), arms_fit)
    assert isinstance(area, EstimandResult)
    assert area.detail["n_nodes"] == float(QUADRATURE_NODES)
    # approx, not ==: the dose difference is a sum over quadrature nodes, so it
    # lands on 80.00000000000001 as readily as on 80.0 and which one depends on
    # the numpy build. Every sibling assertion on this key already uses approx.
    assert area.detail["dose_difference"] == pytest.approx(80.0)
    # the closed-form check on the kernel: marginal at the half-saturation dose is beta·s/(4k)
    r = realize(
        _world_estimand(arms, "marginal", intervention=Intervention(doses={"dose": 50.0})),
        arms_fit,
        mass=0.95,
    )
    assert isinstance(r, EstimandResult)
    _covers(r, 4.0 * 2.0 / (4.0 * 50.0))


@pytest.mark.parametrize("basis", ["cumulative", "per_period"])
@pytest.mark.parametrize("level", ["individual", "aggregate"])
def test_panel_world_recovery_under_every_basis_and_level(
    panel: SurfaceWorld, panel_fit: FitResult, basis: Basis, level: LevelUnit
) -> None:
    """ratio / elasticity / area / contrast / marginal against the world's truth aggregated the
    same way, on a carried panel, for both bases and both levels."""
    window = TimeWindow(start=0, stop=panel.n_periods, basis=basis)
    lvl = Level(unit=level)
    ones = np.ones(panel.mean.shape)
    y = {d: panel.forward({"a": d}) for d in (0.0, 80.0)}
    x = {d: np.full(panel.mean.shape, d) for d in (0.0, 80.0)}
    h = 1e-4
    d_agg_y = _agg(panel.forward({"a": 80.0 + h}), window, level) - _agg(y[80.0], window, level)
    truths: dict[QuantityKind, float] = {
        "contrast": _agg(y[80.0], window, level) - _agg(y[0.0], window, level),
        "ratio": (_agg(y[80.0], window, level) - _agg(y[0.0], window, level))
        / (_agg(x[80.0], window, level) - _agg(x[0.0], window, level)),
        "marginal": d_agg_y / (h * _agg(ones, window, level)),
        "elasticity": d_agg_y
        / (h * _agg(ones, window, level))
        * _agg(x[80.0], window, level)
        / _agg(y[80.0], window, level),
        "area": quad(lambda d: _agg(panel.forward({"a": d}), window, level), 0.0, 80.0)[0],
    }
    for kind, truth in truths.items():
        r = realize(_world_estimand(panel, kind, window=window, level=lvl), panel_fit, mass=0.95)
        assert isinstance(r, EstimandResult), (kind, r)
        _covers(r, truth)
        assert r.detail["basis"] == basis and r.detail["level"] == level
    # the basis-invariant functionals agree across bases to floating point, per draw
    invariant: tuple[QuantityKind, ...] = ("ratio", "marginal", "elasticity")
    for kind in invariant:
        a = realize(
            _world_estimand(panel, kind, window=window, level=lvl), panel_fit, keep_draws=True
        )
        other = window.model_copy(
            update={"basis": "per_period" if basis == "cumulative" else "cumulative"}
        )
        b = realize(
            _world_estimand(panel, kind, window=other, level=lvl), panel_fit, keep_draws=True
        )
        assert isinstance(a, RealizedDraws) and isinstance(b, RealizedDraws)
        assert np.allclose(a.draws, b.draws, rtol=1e-10), kind


def test_marginal_with_carryover_on_the_panel_world(
    panel: SurfaceWorld, panel_fit: FitResult
) -> None:
    e = _world_estimand(
        panel,
        "marginal",
        intervention=Intervention(doses={"a": 50.0}),
        level=Level(unit="aggregate"),
        window=TimeWindow(start=0, stop=20, basis="cumulative"),
    )
    r = realize(e, panel_fit, mass=0.95)
    assert isinstance(r, EstimandResult)
    # over the whole horizon the shift derivative sums to the same as the total marginal
    m = panel.marginal("a", dose={"a": 50.0}, horizon="total")
    assert not isinstance(m, Unsupported)
    _covers(r, float(np.mean(m)))
    h = 1e-4
    fd = float((panel.forward({"a": 50.0 + h}).sum() - panel.forward({"a": 50.0}).sum()) / (h * 80))
    _covers(r, fd)


def test_windowed_marginal_is_the_ratio_limit_under_carryover(
    long_carry: SurfaceWorld, long_carry_fit: FitResult
) -> None:
    """On a sub-window the marginal must be d agg(Y) / d agg(X) under a common shift — the
    limit of the ratio of two nearby set-interventions — not the total-horizon attribution."""
    eps = 1e-4
    for start, stop in ((0, 5), (15, 20)):
        levels: tuple[LevelUnit, ...] = ("individual", "aggregate")
        for level in levels:
            window = TimeWindow(start=start, stop=stop, basis="cumulative")
            lvl = Level(unit=level)
            marg = realize(
                _world_estimand(
                    long_carry,
                    "marginal",
                    intervention=Intervention(doses={"a": 50.0}),
                    window=window,
                    level=lvl,
                ),
                long_carry_fit,
                keep_draws=True,
            )
            ratio = realize(
                _world_estimand(
                    long_carry,
                    "ratio",
                    intervention=Intervention(doses={"a": 50.0 + eps}),
                    reference=Intervention(doses={"a": 50.0}),
                    window=window,
                    level=lvl,
                ),
                long_carry_fit,
                keep_draws=True,
            )
            assert isinstance(marg, RealizedDraws) and isinstance(ratio, RealizedDraws)
            assert np.allclose(marg.draws, ratio.draws, rtol=1e-3, atol=0.0), (start, level)
            # and the truth, aggregated the same way, is covered
            ones = np.ones(long_carry.mean.shape)
            truth = (
                _agg(long_carry.forward({"a": 50.0 + eps}), window, level)
                - _agg(long_carry.forward({"a": 50.0}), window, level)
            ) / (eps * _agg(ones, window, level))
            _covers(marg.result, truth)
    # the two windows see different carried doses, so the marginals differ
    early = realize(
        _world_estimand(
            long_carry,
            "marginal",
            intervention=Intervention(doses={"a": 50.0}),
            window=TimeWindow(start=0, stop=5),
            level=Level(unit="aggregate"),
        ),
        long_carry_fit,
    )
    late = realize(
        _world_estimand(
            long_carry,
            "marginal",
            intervention=Intervention(doses={"a": 50.0}),
            window=TimeWindow(start=15, stop=20),
            level=Level(unit="aggregate"),
        ),
        long_carry_fit,
    )
    assert isinstance(early, EstimandResult) and isinstance(late, EstimandResult)
    assert early.summary.mean != pytest.approx(late.summary.mean, rel=1e-3)


def test_hand_built_posterior_gives_identical_results(arms_fit: FitResult) -> None:
    post = arms_fit.posterior
    assert isinstance(post, Posterior)
    rebuilt = Posterior(
        {name: np.array(post.draws(name)) for name in post.names()},
        provenance={"method": "hand-built", "model_hash": "x" * 64},
    )
    twin = FitResult(arms_fit.surface, rebuilt, arms_fit.data, None, {"model_hash": "x" * 64})
    world = arms_world(
        n_units=30, treatments=("dose",), doses=DosePlan(scale=50.0, spread=0.6), seed=1
    )
    for kind in ("contrast", "ratio", "marginal", "elasticity", "area"):
        e = _world_estimand(world, kind)
        a = realize(e, arms_fit)
        b = realize(e, twin)
        assert isinstance(a, EstimandResult) and isinstance(b, EstimandResult)
        assert a.summary == b.summary and a.detail == b.detail
        assert b.producer_hash == "x" * 64 and a.producer_hash != b.producer_hash


# -- arithmetic against numpy on the linear double --------------------------------------------


def test_linear_double_satisfies_the_protocol() -> None:
    assert isinstance(_Linear(_X), SupportsEstimands)


def test_window_basis_and_level_arithmetic() -> None:
    p = _Linear(_X)
    beta = p.beta.reshape(-1)
    alpha = p.alpha.reshape(-1)
    dx = 80.0  # set 80 vs 0 in every cell
    cases: dict[tuple[Basis, LevelUnit], Array] = {
        ("per_period", "individual"): beta * dx,
        ("cumulative", "individual"): beta * dx * 6,
        ("per_period", "aggregate"): beta * dx * 3,
        ("cumulative", "aggregate"): beta * dx * 18,
        ("cumulative", "cluster"): beta * dx * 18,
    }
    for (basis, level), expect in cases.items():
        e = _lin(window=TimeWindow(start=0, stop=6, basis=basis), level=Level(unit=level))
        out = realize(e, p, keep_draws=True)
        assert isinstance(out, RealizedDraws)
        assert np.allclose(out.draws, expect), (basis, level)
        want = summarize(expect, definition="hdi", mass=0.9)
        assert out.result.summary.mean == pytest.approx(want.mean)
        assert out.result.summary.sd == pytest.approx(want.sd)
        assert out.result.summary.interval.lower == pytest.approx(want.interval.lower)
        assert out.result.summary.interval.definition == "hdi"
        assert out.result.unit == "kg"
        assert [a.name for a in out.result.assumptions] == ["identification_not_checked"]
    # a sub-window: [1, 4) has three periods
    e = _lin(window=TimeWindow(start=1, stop=4, basis="cumulative"), level=Level(unit="aggregate"))
    out = realize(e, p, keep_draws=True)
    assert isinstance(out, RealizedDraws) and np.allclose(out.draws, beta * dx * 9)
    # ratio divides by the dose aggregated the same way; marginal is d agg(Y)/d agg(X);
    # elasticity and area follow in closed form
    ratio = realize(_lin("ratio", level=Level(unit="aggregate")), p, keep_draws=True)
    assert isinstance(ratio, RealizedDraws) and np.allclose(ratio.draws, beta)
    assert ratio.result.detail["dose_difference"] == 3 * 6 * dx
    assert ratio.result.detail["basis"] == "cumulative"
    marg = realize(
        _lin(
            "marginal",
            window=TimeWindow(start=0, stop=6, basis="cumulative"),
            level=Level(unit="aggregate"),
        ),
        p,
        keep_draws=True,
    )
    assert isinstance(marg, RealizedDraws) and np.allclose(marg.draws, beta)
    el = realize(_lin("elasticity"), p, keep_draws=True)
    assert isinstance(el, RealizedDraws)
    assert np.allclose(el.draws, beta * 80.0 / (alpha + beta * 80.0))
    assert el.result.detail["dose_aggregated"] == 6 * 80.0  # cumulative over six periods
    area = realize(_lin("area"), p, keep_draws=True)
    assert isinstance(area, RealizedDraws)
    # ∫₀⁸⁰ agg(Y) dd with the cumulative basis summing six periods; exact for a polynomial
    assert np.allclose(area.draws, 6 * (alpha * 80.0 + beta * 80.0**2 / 2))
    assert area.result.dimension == D.outcome * D.currency
    assert area.result.unit == "kg·USD"


def test_ratio_marginal_elasticity_are_basis_invariant_contrast_area_scale() -> None:
    p = _Linear(_X)
    iv = Intervention(doses={"x": 1.5}, mode="scale")
    ref = Intervention(doses={"x": 1.0}, mode="scale")
    kinds: tuple[QuantityKind, ...] = ("ratio", "marginal", "elasticity", "contrast", "area")
    bases: tuple[Basis, ...] = ("cumulative", "per_period")
    for kind in kinds:
        out = {}
        for basis in bases:
            e = _lin(
                kind,
                intervention=iv,
                reference=None if kind in ("marginal", "elasticity") else ref,
                window=TimeWindow(start=0, stop=6, basis=basis),
            )
            r = realize(e, p, keep_draws=True)
            assert isinstance(r, RealizedDraws), (kind, r)
            out[basis] = r.draws
        if kind in ("ratio", "marginal", "elasticity"):
            assert np.allclose(out["cumulative"], out["per_period"]), kind
        else:
            assert np.allclose(out["cumulative"], 6 * out["per_period"]), kind


def test_scale_and_shift_modes_use_the_realized_doses() -> None:
    p = _Linear(_X)
    beta = p.beta.reshape(-1)
    iv = Intervention(doses={"x": 1.5}, mode="scale")
    ref = Intervention(doses={"x": 1.0}, mode="scale")
    e = _lin(
        "ratio",
        intervention=iv,
        reference=ref,
        window=TimeWindow(start=0, stop=6, basis="cumulative"),
        level=Level(unit="aggregate"),
    )
    out = realize(e, p, keep_draws=True)
    assert isinstance(out, RealizedDraws)
    assert out.result.detail["dose_difference"] == pytest.approx(0.5 * _X.sum())
    assert out.result.detail["intervention_mode"] == "scale"
    assert np.allclose(out.draws, beta)
    contrast = realize(_lin("contrast", intervention=iv, reference=ref), p, keep_draws=True)
    assert isinstance(contrast, RealizedDraws)
    assert np.allclose(contrast.draws, beta * 0.5 * _X.sum() / 3)  # cumulative, unit mean
    # a support restricts where the dose moves
    sup = TimeWindow(start=2, stop=4)
    shifted = realize(
        _lin(
            "contrast",
            intervention=Intervention(doses={"x": 10.0}, mode="shift", window=sup),
            reference=Intervention(doses={"x": 0.0}, mode="shift", window=sup),
            window=TimeWindow(start=0, stop=6, basis="cumulative"),
        ),
        p,
        keep_draws=True,
    )
    assert isinstance(shifted, RealizedDraws) and np.allclose(shifted.draws, beta * 10.0 * 2)
    # a producer that does not report realized doses cannot answer a scale-mode ratio
    silent = _Linear(_X, report_doses=False)
    out2 = realize(e, silent)
    assert isinstance(out2, Unsupported) and "does not report" in out2.reason
    assert isinstance(realize(_lin("ratio"), silent), EstimandResult)  # set mode reconstructs
    # area needs one straight path
    mixed = _lin("area", intervention=iv, reference=Intervention(doses={"x": 0.0}))
    out3 = realize(mixed, p)
    assert isinstance(out3, Unsupported) and "same mode" in out3.reason


def test_marginal_under_a_supported_shift_divides_by_the_reached_cells() -> None:
    """``marginal = agg(M) / agg(1_S)``: the derivative of the aggregate outcome with respect
    to the aggregate dose, so a support narrower than the window still gives ``beta``."""
    p = _Linear(_X)
    beta = p.beta.reshape(-1)
    sup = TimeWindow(start=2, stop=4)
    levels: tuple[LevelUnit, ...] = ("individual", "aggregate")
    bases: tuple[Basis, ...] = ("cumulative", "per_period")
    for level in levels:
        for basis in bases:
            e = _lin(
                "marginal",
                intervention=Intervention(doses={"x": 1.0}, mode="shift", window=sup),
                window=TimeWindow(start=0, stop=6, basis=basis),
                level=Level(unit=level),
            )
            out = realize(e, p, keep_draws=True)
            assert isinstance(out, RealizedDraws) and np.allclose(out.draws, beta), (level, basis)
    # the reporting window [1, 3) sees one of the two supported periods: still beta
    e = _lin(
        "marginal",
        intervention=Intervention(doses={"x": 1.0}, mode="shift", window=sup),
        window=TimeWindow(start=1, stop=3),
    )
    out = realize(e, p, keep_draws=True)
    assert isinstance(out, RealizedDraws) and np.allclose(out.draws, beta)
    # a support that never reaches the window has no dose change to differentiate against
    with pytest.raises(ValueError, match="does not reach the window"):
        realize(
            _lin(
                "marginal",
                intervention=Intervention(doses={"x": 1.0}, mode="shift", window=sup),
                window=TimeWindow(start=4, stop=6),
            ),
            p,
        )


def test_unit_weights_and_population_strata() -> None:
    p = _Linear(_X)
    beta = p.beta.reshape(-1)
    # doubling the observed dose: the per-unit (per-period mean) contrast is beta · mean_t x[u, t]
    iv = Intervention(doses={"x": 2.0}, mode="scale")
    ref = Intervention(doses={"x": 1.0}, mode="scale")
    per_unit = beta[:, None] * _X.mean(axis=1)[None, :]
    assert not np.allclose(per_unit[:, 0], per_unit[:, 1])  # the weights matter here
    per_period = TimeWindow(start=0, stop=6, basis="per_period")
    e = _lin(intervention=iv, reference=ref, window=per_period)
    uniform = realize(e, p, keep_draws=True)
    assert isinstance(uniform, RealizedDraws) and np.allclose(uniform.draws, per_unit.mean(axis=1))
    w = {"u0": 0.2, "u1": 0.3, "u2": 0.5}
    out = realize(e, p, unit_weights=w, keep_draws=True)
    assert isinstance(out, RealizedDraws)
    assert np.allclose(out.draws, per_unit @ np.array([0.2, 0.3, 0.5]))
    out = realize(e, p, unit_weights={"u0": 2.0, "u1": 3.0, "u2": 5.0}, keep_draws=True)
    assert isinstance(out, RealizedDraws) and np.allclose(out.draws, per_unit @ [0.2, 0.3, 0.5])
    with pytest.raises(ValueError, match="lack entries"):
        realize(e, p, unit_weights={"u0": 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        realize(e, p, unit_weights={"u0": -1.0, "u1": 1.0, "u2": 1.0})
    # weights belong to a weighted mean: the summing levels refuse them
    summing: tuple[LevelUnit, ...] = ("aggregate", "cluster")
    for level in summing:
        with pytest.raises(ValueError, match="individual.*only"):
            realize(e.model_copy(update={"level": Level(unit=level)}), p, unit_weights=w)
    # strata: the population's composition sets the weights, unit_strata maps units to strata
    pop = Population(name="target", strata={"soil": {"clay": 0.25, "loam": 0.75}})
    strata = {"u0": "clay", "u1": "loam", "u2": "loam"}
    es = _lin(intervention=iv, reference=ref, population=pop, window=per_period)
    out = realize(es, p, unit_strata=strata, keep_draws=True)
    assert isinstance(out, RealizedDraws)
    assert np.allclose(out.draws, per_unit @ np.array([0.25, 0.375, 0.375]))
    agg = realize(
        es.model_copy(update={"level": Level(unit="aggregate")}),
        p,
        unit_strata=strata,
        keep_draws=True,
    )
    assert isinstance(agg, RealizedDraws)
    assert np.allclose(agg.draws, per_unit @ (3 * np.array([0.25, 0.375, 0.375])))
    blocked = realize(es, p)
    assert isinstance(blocked, Blocked) and "unit_strata" in blocked.reason
    blocked = realize(es, p, unit_strata={"u0": "clay"})
    assert isinstance(blocked, Blocked) and "no stratum" in blocked.reason
    blocked = realize(es, p, unit_strata={"u0": "clay", "u1": "clay", "u2": "clay"})
    assert isinstance(blocked, Blocked) and "no unit" in blocked.reason
    blocked = realize(es, p, unit_strata={"u0": "clay", "u1": "loam", "u2": "sand"})
    assert isinstance(blocked, Blocked) and "does not declare" in blocked.reason
    with pytest.raises(ValueError, match="not both"):
        realize(es, p, unit_strata=strata, unit_weights=w)
    two = Population(name="t2", strata={"soil": {"clay": 1.0}, "slope": {"flat": 1.0}})
    out2 = realize(_lin(population=two), p, unit_strata=strata)
    assert isinstance(out2, Unsupported) and "more than one covariate" in out2.reason


def test_not_yet_realized_shapes_are_typed() -> None:
    p = _Linear(_X)
    out = realize(_lin(conditioning=("soil",)), p)
    assert isinstance(out, Unsupported) and "Phase 6" in out.reason
    out = realize(_lin(quantity=Quantity(kind="contrast", scale="log")), p)
    assert isinstance(out, Unsupported) and "log" in out.reason


# -- capabilities, windows ------------------------------------------------------------------


def test_missing_capabilities_are_unsupported_never_numbers() -> None:
    no_cf = _Linear(
        _X, caps=frozenset({Capability.MARGINAL, Capability.TIME_WINDOW, Capability.PER_UNIT})
    )
    for kind in ("contrast", "ratio", "area", "elasticity"):
        out = realize(_lin(kind), no_cf)
        assert isinstance(out, Unsupported) and out.missing == ("counterfactual",), kind
        assert "lacks" in out.reason and out.detail["estimand"] == f"{kind}_probe"
    assert isinstance(realize(_lin("marginal"), no_cf), EstimandResult)
    no_marg = _Linear(
        _X, caps=frozenset({Capability.COUNTERFACTUAL, Capability.TIME_WINDOW, Capability.PER_UNIT})
    )
    derivative_kinds: tuple[QuantityKind, ...] = ("marginal", "elasticity")
    for kind in derivative_kinds:
        out = realize(_lin(kind), no_marg)
        assert isinstance(out, Unsupported) and out.missing == ("marginal",)
    # a producer whose marginal_under declines is passed through, typed
    declines = _Linear(_X, marginal_unsupported=True)
    for kind in derivative_kinds:
        out = realize(_lin(kind), declines)
        assert isinstance(out, Unsupported) and "declines" in out.reason
    no_unit = _Linear(
        _X, caps=frozenset({Capability.COUNTERFACTUAL, Capability.MARGINAL, Capability.TIME_WINDOW})
    )
    out = realize(_lin(level=Level(unit="individual")), no_unit)
    assert isinstance(out, Unsupported) and out.missing == ("per_unit",)
    assert isinstance(realize(_lin(level=Level(unit="aggregate")), no_unit), EstimandResult)
    # several at once are all named, sorted
    out = realize(_lin("elasticity"), _Linear(_X, caps=frozenset()))
    assert isinstance(out, Unsupported)
    assert out.missing == ("counterfactual", "marginal", "per_unit")


def test_time_window_capability(panel_fit: FitResult) -> None:
    no_tw = _Linear(
        _X, caps=frozenset({Capability.COUNTERFACTUAL, Capability.MARGINAL, Capability.PER_UNIT})
    )
    whole = realize(_lin(window=TimeWindow(start=0, stop=6)), no_tw)
    assert isinstance(whole, EstimandResult)  # the whole horizon needs no windowing
    part = realize(_lin(window=TimeWindow(start=1, stop=4)), no_tw)
    assert isinstance(part, Unsupported) and part.missing == ("time_window",)
    with pytest.raises(ValueError, match="runs past"):
        realize(_lin(window=TimeWindow(start=0, stop=9)), no_tw)
    with pytest.raises(ValueError, match="returned 6 periods for window"):
        realize(_lin(window=TimeWindow(start=0, stop=9)), _Linear(_X))  # the double over-slices
    # on a fitted surface the producer selects the window; realize checks the length and sums
    e = _estimand(
        panel_fit.treatments[0],
        panel_fit.outcome,
        panel_fit.n_periods,
        "contrast",
        window=TimeWindow(start=3, stop=8, basis="cumulative"),
        level=Level(unit="aggregate"),
    )
    out = realize(e, panel_fit, keep_draws=True)
    assert isinstance(out, RealizedDraws)
    assert e.reference is not None
    iv = panel_fit.predict_under(e.intervention)
    ref = panel_fit.predict_under(e.reference)
    assert isinstance(iv, PredictiveDraws) and isinstance(ref, PredictiveDraws)
    expect = (iv.values - ref.values)[..., 3:8].sum(axis=(-1, -2)).reshape(-1)
    assert np.allclose(out.draws, expect)
    crippled = panel_fit.restrict([Capability.TIME_WINDOW])
    out2 = realize(e, crippled)
    assert isinstance(out2, Unsupported) and out2.missing == ("time_window",)


# -- identification ---------------------------------------------------------------------------


def test_identification_verdicts() -> None:
    p = _Linear(_X)
    e = _lin()
    ok = Verdict(status="identified", route="backdoor")
    r = realize(e, p, verdict=ok)
    assert isinstance(r, EstimandResult) and r.status == "identified"
    assert r.identification == ok and r.assumptions == () and r.ledger == ()
    down = Verdict(
        status="downgraded",
        assumptions=(Assumption(name="no_unmeasured", facet="population", statement="U absent"),),
    )
    blocked = realize(e, p, verdict=down)
    assert isinstance(blocked, Blocked) and "downgraded" in blocked.reason
    assert "assume_identified" in blocked.reason and blocked.detail["verdict"] == "downgraded"
    hard = Verdict(status="blocked", reason="no admissible set")
    blocked = realize(e, p, verdict=hard)
    assert isinstance(blocked, Blocked) and "no admissible set" in blocked.reason
    forced = realize(e, p, verdict=down, assume_identified=True)
    assert isinstance(forced, EstimandResult) and forced.status == "downgraded"
    assert [a.name for a in forced.assumptions] == ["no_unmeasured", "assumed_identified"]
    assert forced.assumptions[-1].state == "asserted"
    assert len(forced.ledger) == 1 and forced.ledger[0].kind == "assume_identified"
    assert forced.ledger[0].assumption == forced.assumptions[-1]
    assert forced.ledger[0].source == e.content_hash()
    assert forced.identification == down
    # no verdict + assertion: both the unchecked flag and the assertion are on record
    asserted = realize(e, p, assume_identified=True)
    assert isinstance(asserted, EstimandResult) and asserted.status == "downgraded"
    assert [a.name for a in asserted.assumptions] == [
        "identification_not_checked",
        "assumed_identified",
    ]
    assert asserted.ledger[0].detail["verdict"] == "none"
    # assume_identified on an identified verdict changes nothing
    assert realize(e, p, verdict=ok, assume_identified=True) == r


def test_unknown_units_are_an_assumption_that_downgrades() -> None:
    ok = Verdict(status="identified", route="backdoor")
    # the producer's outcome unit is unknown: equality is assumed, on record, and downgrades
    p = _Linear(_X, outcome_unit=None)
    r = realize(_lin(), p, verdict=ok)
    assert isinstance(r, EstimandResult) and r.status == "downgraded"
    assert [a.name for a in r.assumptions] == ["units_assumed_equal"]
    assert r.assumptions[0].state == "unverified" and r.assumptions[0].facet == "outcome"
    assert r.assumptions[0].detail == {"outcome y": "estimand kg, producer unknown"}
    assert r.ledger == () and r.identification == ok
    # the estimand's dose unit is unknown and the level is a dose: same, on the treatment facet
    no_dose_unit = _lin(treatment=Treatment(name="x", dimension=D.currency))
    r = realize(no_dose_unit, _Linear(_X), verdict=ok)
    assert isinstance(r, EstimandResult) and r.status == "downgraded"
    assert [a.name for a in r.assumptions] == ["units_assumed_equal"]
    assert r.assumptions[0].facet == "treatment"
    assert r.assumptions[0].detail == {"treatment x": "estimand unknown, producer USD"}
    # an elasticity under a dimensionless (scale) level crosses no unit boundary at all
    scale = _lin(
        "elasticity",
        treatment=Treatment(name="x", dimension=D.currency),
        intervention=Intervention(doses={"x": 2.0}, mode="scale"),
    )
    r = realize(scale, _Linear(_X, outcome_unit=None), verdict=ok)
    assert isinstance(r, EstimandResult) and r.status == "identified" and r.assumptions == ()
    # but a scale-mode ratio reports per dose unit, which must be known
    r = realize(
        _lin(
            "ratio",
            treatment=Treatment(name="x", dimension=D.currency),
            intervention=Intervention(doses={"x": 2.0}, mode="scale"),
            reference=Intervention(doses={"x": 1.0}, mode="scale"),
        ),
        _Linear(_X),
        verdict=ok,
    )
    assert isinstance(r, EstimandResult) and r.status == "downgraded"
    assert r.assumptions[0].name == "units_assumed_equal"


def test_result_cannot_claim_identified_without_a_verdict() -> None:
    p = _Linear(_X)
    r = realize(_lin(), p, verdict=Verdict(status="identified"))
    assert isinstance(r, EstimandResult)
    payload = r.model_dump()
    with pytest.raises(ValueError, match="needs an identification verdict"):
        EstimandResult.model_validate({**payload, "identification": None})
    with pytest.raises(ValueError, match="at least one assumption"):
        EstimandResult.model_validate({**payload, "status": "downgraded"})
    with pytest.raises(ValueError, match="unverified"):
        EstimandResult.model_validate(
            {
                **payload,
                "assumptions": [Assumption(name="x", facet="f", statement="s").model_dump()],
            }
        )


# -- dimensions and units -----------------------------------------------------------------------


def test_dimension_mismatch_raises(arms_fit: FitResult) -> None:
    p = _Linear(_X)
    t_time = Treatment(name="x", dimension=D.time, unit="day")
    with pytest.raises(DimensionError, match="declared in T but the producer fit it in"):
        realize(_estimand(t_time, _Y, 6), p)
    other = Outcome(name="y", dimension=D.currency)
    with pytest.raises(DimensionError, match="outcome 'y' is declared in"):
        realize(_estimand(arms_fit.treatments[0], other, 1), arms_fit)
    with pytest.raises(ValueError, match="does not have"):
        realize(_estimand(Treatment(name="zzz", dimension=D.currency), _Y, 6), p)


def _declare_currency_and_mass_units() -> None:
    UNITS.declare("USD", "currency")
    UNITS.declare("cent", "currency")
    UNITS.declare("kg", "outcome")
    UNITS.declare("g", "outcome")
    UNITS.register("USD", "cent", 100)
    UNITS.register("kg", "g", 1000)


def test_intervention_levels_are_converted_into_the_producer_unit() -> None:
    """An estimand in cents setting 8000 cents against a producer fit in USD is the same
    question as 80 USD: the level goes in converted, the draws come out converted."""
    _declare_currency_and_mass_units()
    p = _Linear(_X)  # USD, kg
    beta = p.beta.reshape(-1)
    alpha = p.alpha.reshape(-1)
    cent = Treatment(name="x", dimension=D.currency, unit="cent")
    gram = Outcome(name="y", dimension=D.outcome, unit="g")
    factor: dict[QuantityKind, float] = {
        "contrast": 1000.0,
        "marginal": 1000.0 / 100.0,
        "ratio": 1000.0 / 100.0,
        "elasticity": 1.0,
        "area": 1000.0 * 100.0,
    }
    lines = {"contrast": 2, "marginal": 3, "ratio": 3, "elasticity": 1, "area": 3}
    for kind in factor:
        usd = _lin(kind)  # set 80 USD (vs 0)
        cents = _estimand(
            cent, gram, 6, kind, intervention=Intervention(doses={"x": 8000.0})
        )  # set 8000 cents (vs 0)
        a = realize(usd, p, keep_draws=True)
        b = realize(cents, p, keep_draws=True)
        assert isinstance(a, RealizedDraws) and isinstance(b, RealizedDraws)
        assert a.result.ledger == ()
        assert np.allclose(b.draws, a.draws * factor[kind]), kind
        assert b.result.summary.mean == pytest.approx(a.result.summary.mean * factor[kind])
        # both sides' units are known: nothing is assumed about them
        assert [x.name for x in a.result.assumptions] == ["identification_not_checked"]
        assert [x.name for x in b.result.assumptions] == ["identification_not_checked"]
        kinds = [line.kind for line in b.result.ledger]
        assert kinds == ["unit_conversion"] * lines[kind], kind
        level_line = b.result.ledger[0]
        assert level_line.detail == {
            "from_unit": "cent",
            "to_unit": "USD",
            "factor": "1/100",
            "applies_to": "dose_levels",
        }
        assert "intervention dose levels" in level_line.statement
        if kind != "elasticity":
            assert b.result.ledger[1].detail == {
                "from_unit": "kg",
                "to_unit": "g",
                "factor": "1000",
            }
        if lines[kind] == 3:
            assert b.result.ledger[2].detail["from_unit"] == "USD"
            assert b.result.detail["dose_factor"] == 100.0
        # detail is in the estimand's unit
        assert b.result.detail["dose_iv"] == 8000.0 and b.result.detail["level_factor"] == 0.01
        assert (
            b.result.unit
            == {
                "contrast": "g",
                "marginal": "g/cent",
                "ratio": "g/cent",
                "elasticity": None,
                "area": "g·cent",
            }[kind]
        )
    # the closed forms, in the estimand's units: 8000 cents is 80 USD
    ratio = realize(_estimand(cent, gram, 6, "ratio"), p, keep_draws=True)
    assert isinstance(ratio, RealizedDraws)
    assert np.allclose(ratio.draws, beta * 1000.0 / 100.0)  # g per cent
    # 80 cents set vs 0, summed over six periods (cumulative), in the estimand's unit
    assert ratio.result.detail["dose_difference"] == pytest.approx(6 * 80.0)
    el = realize(
        _estimand(cent, gram, 6, "elasticity", intervention=Intervention(doses={"x": 8000.0})),
        p,
        keep_draws=True,
    )
    assert isinstance(el, RealizedDraws)
    assert np.allclose(el.draws, beta * 80.0 / (alpha + beta * 80.0))
    assert el.result.detail["dose_aggregated"] == pytest.approx(6 * 8000.0)
    # shift levels convert too; scale levels are dimensionless and do not
    shift = realize(
        _estimand(
            cent,
            gram,
            6,
            "contrast",
            intervention=Intervention(doses={"x": 1000.0}, mode="shift"),
            reference=Intervention(doses={"x": 0.0}, mode="shift"),
        ),
        p,
        keep_draws=True,
    )
    assert isinstance(shift, RealizedDraws)
    assert np.allclose(shift.draws, beta * 10.0 * 6 * 1000.0)  # +10 USD per cell, six periods, g
    scale = realize(
        _estimand(
            cent,
            gram,
            6,
            "ratio",
            intervention=Intervention(doses={"x": 1.5}, mode="scale"),
            reference=Intervention(doses={"x": 1.0}, mode="scale"),
            level=Level(unit="aggregate"),
        ),
        p,
        keep_draws=True,
    )
    assert isinstance(scale, RealizedDraws)
    assert [line.detail.get("applies_to") for line in scale.result.ledger] == [None, None]
    assert np.allclose(scale.draws, beta * 10.0)
    assert scale.result.detail["dose_difference"] == pytest.approx(0.5 * _X.sum() * 100.0)
    assert "level_factor" not in scale.result.detail
    # the area path is walked in the producer's unit: 16 nodes between 0 and 80 USD
    area = realize(
        _estimand(cent, gram, 6, "area", intervention=Intervention(doses={"x": 8000.0})),
        p,
        keep_draws=True,
    )
    assert isinstance(area, RealizedDraws)
    assert np.allclose(area.draws, 6 * (alpha * 80.0 + beta * 80.0**2 / 2) * 1000.0 * 100.0)
    assert area.result.detail["dose_difference"] == pytest.approx(8000.0)


def test_unit_conversion_on_a_fitted_surface_with_a_ledger_line() -> None:
    _declare_currency_and_mass_units()
    world = arms_world(
        n_units=12,
        treatments=("dose",),
        outcome=Outcome(name="y", dimension=D.outcome, unit="kg"),
        doses=DosePlan(scale=50.0),
        seed=4,
    )
    res = fit(world.spec, world.panel, backend="laplace", draws=40, chains=1, seed=4)
    assert isinstance(res.posterior, Posterior)
    usd, kg = world.spec.treatments[0], world.spec.outcome
    cent = usd.model_copy(update={"unit": "cent"})
    g = kg.model_copy(update={"unit": "g"})
    for kind in ("contrast", "marginal", "ratio", "elasticity", "area"):
        base = realize(_estimand(usd, kg, 1, kind), res, keep_draws=True)
        conv = realize(
            _estimand(cent, g, 1, kind, intervention=Intervention(doses={"dose": 8000.0})),
            res,
            keep_draws=True,
        )
        assert isinstance(base, RealizedDraws) and isinstance(conv, RealizedDraws)
        assert base.result.ledger == ()
        factor = {
            "contrast": 1000.0,
            "marginal": 1000.0 / 100.0,
            "ratio": 1000.0 / 100.0,
            "elasticity": 1.0,
            "area": 1000.0 * 100.0,
        }[kind]
        assert np.allclose(conv.draws, base.draws * factor), kind
        # the same level in a different unit is a different question
        other = realize(_estimand(cent, g, 1, kind), res, keep_draws=True)
        assert isinstance(other, RealizedDraws)
        assert not np.allclose(other.draws, base.draws * factor), kind
    # the same dimension in an unregistered unit is an error, never a silent number
    UNITS.declare("EUR", "currency")
    with pytest.raises(UnitConversionError, match="no conversion"):
        realize(_estimand(usd.model_copy(update={"unit": "EUR"}), kg, 1, "marginal"), res)
    with pytest.raises(UnitConversionError, match="not declared"):
        realize(_estimand(usd.model_copy(update={"unit": "zorkmid"}), kg, 1, "ratio"), res)
    # a set level in an undeclared unit cannot be applied to a USD producer, even for a contrast
    with pytest.raises(UnitConversionError, match="not declared"):
        realize(_estimand(usd.model_copy(update={"unit": "zorkmid"}), kg, 1, "contrast"), res)
    # a scale-mode contrast never touches the dose unit, so an undeclared one is not its problem
    out = realize(
        _estimand(
            usd.model_copy(update={"unit": "zorkmid"}),
            kg,
            1,
            "contrast",
            intervention=Intervention(doses={"dose": 2.0}, mode="scale"),
            reference=Intervention(doses={"dose": 1.0}, mode="scale"),
        ),
        res,
    )
    assert isinstance(out, EstimandResult)


# -- evaluate ----------------------------------------------------------------------------------


def test_evaluate_keys_by_name_and_passes_verdicts() -> None:
    p = _Linear(_X)
    a = _lin("contrast", name="lift")
    b = _lin("marginal", name="slope")
    c = _lin("ratio", name="per_unit_dose", conditioning=("soil",))
    out = evaluate([a, b, c], p, verdicts={"lift": Verdict(status="identified")}, mass=0.8)
    assert list(out) == ["lift", "slope", "per_unit_dose"]
    lift, slope, cond = out["lift"], out["slope"], out["per_unit_dose"]
    assert isinstance(lift, EstimandResult) and lift.status == "identified"
    assert isinstance(slope, EstimandResult) and slope.status == "downgraded"
    assert lift.summary.interval.mass == 0.8
    assert isinstance(cond, Unsupported)
    with pytest.raises(ValueError, match="distinct"):
        evaluate([a, a], p)
    assert evaluate([], p) == {}


def test_keep_draws_returns_the_draws_behind_the_summary(arms_fit: FitResult) -> None:
    world = arms_world(
        n_units=30, treatments=("dose",), doses=DosePlan(scale=50.0, spread=0.6), seed=1
    )
    out = realize(_world_estimand(world, "contrast"), arms_fit, keep_draws=True, definition="eti")
    assert isinstance(out, RealizedDraws)
    assert out.draws.shape == (300,) and out.result.n_draws == 300
    assert out.result.summary == summarize(out.draws, definition="eti", mass=0.9)
    plain = realize(_world_estimand(world, "contrast"), arms_fit, definition="eti")
    assert plain == out.result


# -- residual defects from verification --------------------------------------------------------


def test_scale_mode_elasticity_reports_the_aggregated_dose_in_the_estimands_unit() -> None:
    """A scale factor is dimensionless, so no level is converted going in and no draw coming
    out; the ``dose_aggregated`` figure in ``detail`` is still a dose and is reported in the
    estimand's unit (cents), not the producer's (USD)."""
    _declare_currency_and_mass_units()
    p = _Linear(_X)  # USD, kg
    cent = Treatment(name="x", dimension=D.currency, unit="cent")
    iv = Intervention(doses={"x": 1.5}, mode="scale")
    usd = realize(_lin("elasticity", intervention=iv), p, keep_draws=True)
    cents = realize(_lin("elasticity", treatment=cent, intervention=iv), p, keep_draws=True)
    assert isinstance(usd, RealizedDraws) and isinstance(cents, RealizedDraws)
    # the number itself is dimensionless and identical
    assert np.allclose(cents.draws, usd.draws)
    # cumulative over six periods, uniform mean over units, in the producer's unit...
    agg_usd = 1.5 * _X.sum(axis=1).mean()
    assert usd.result.detail["dose_aggregated"] == pytest.approx(agg_usd)
    # ...and in the estimand's for the cent estimand
    assert cents.result.detail["dose_aggregated"] == pytest.approx(agg_usd * 100.0)
    # no level went in and no draw came out converted: nothing to ledger, nothing assumed
    assert cents.result.ledger == () and "level_factor" not in cents.result.detail
    assert [a.name for a in cents.result.assumptions] == ["identification_not_checked"]


def test_other_treatments_levels_are_a_treatment_facet_assumption_named_once() -> None:
    """Levels set for treatments other than the estimand's are taken in the producer's units;
    the assumption sits on the treatment facet and names each treatment once, even when both
    arms set it."""
    p = _Linear(_X)
    e = _lin(
        "contrast",
        intervention=Intervention(doses={"x": 80.0, "z": 1.0, "w": 2.0}),
        reference=Intervention(doses={"x": 0.0, "z": 1.0, "w": 2.0}),
    )
    r = realize(e, p, verdict=Verdict(status="identified", route="backdoor"))
    assert isinstance(r, EstimandResult) and r.status == "downgraded"
    assert [a.name for a in r.assumptions] == ["units_assumed_equal"]
    a = r.assumptions[0]
    assert a.facet == "treatment"
    assert list(a.detail) == ["other treatments"]
    assert a.detail["other treatments"].startswith("the levels of 'w', 'z' are taken")
    assert "for 'x' only" in a.detail["other treatments"]
    assert a.statement.count("'w'") == 1 and a.statement.count("'z'") == 1
    assert "'w', 'z'" in a.statement
    # the estimand's own treatment unit is known on both sides: no entry for it
    assert not any(k.startswith("treatment ") for k in a.detail)
