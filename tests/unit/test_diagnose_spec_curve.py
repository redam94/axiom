"""Specification curve: axes, option application, the cartesian product, and the summary."""

from __future__ import annotations

import numpy as np
import pytest

from axiom.core import Interval, Unsupported
from axiom.diagnose.spec_curve import (
    FitSettings,
    SpecCurve,
    SpecCurveSummary,
    SpecificationAxis,
    SpecOption,
    SpecRow,
    apply_option,
    default_estimand,
    realized_point,
    specification_curve,
)
from axiom.sim import DosePlan, surface_world
from axiom.sim.surface_world import SurfaceWorld
from axiom.surface import FitResult, HillKernel, LogisticKernel, fit
from axiom.surface.carryover import GeometricCarryover


@pytest.fixture(scope="module")
def world() -> SurfaceWorld:
    return surface_world(
        n_units=4,
        n_periods=12,
        treatments=("a",),
        intercept="shared",
        doses=DosePlan(scale=50.0, zero_fraction=0.2),
        noise_sd=0.2,
        seed=11,
    )


@pytest.fixture(scope="module")
def res(world: SurfaceWorld) -> FitResult:
    return fit(world.spec, world.panel, backend="laplace", draws=40, chains=1, seed=1)


# -- axes and options ------------------------------------------------------------------------


def test_option_converts_spec_values_to_payloads_and_rejects_unknown_fields() -> None:
    opt = SpecOption(label="logistic", spec_update={"kernels": {"a": LogisticKernel()}})
    assert opt.spec_update["kernels"]["a"]["name"] == "logistic"
    assert opt.to_json()  # serializable: options are data, not callables
    with pytest.raises(ValueError, match="does not have"):
        SpecOption(label="x", spec_update={"no_such_field": 1})
    with pytest.raises(ValueError, match="may not change"):
        SpecOption(label="x", spec_update={"outcome": {"name": "z"}})
    with pytest.raises(ValueError, match="FitSettings"):
        SpecOption(label="x", fit_update={"n_draws": 3})


def test_axis_needs_distinct_labels() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        SpecificationAxis(name="k", options=(SpecOption(label="a"), SpecOption(label="a")))
    with pytest.raises(ValueError):
        SpecificationAxis(name="k", options=())


def test_apply_option_merges_kernels_and_replaces_scalars(world: SurfaceWorld) -> None:
    base = world.spec
    settings = FitSettings()
    opt = SpecOption(
        label="steady-logistic",
        spec_update={
            "kernels": {"a": LogisticKernel(reference_dose=50.0)},
            "intercept_scale": 3.0,
        },
        fit_update={"draws": 7, "backend": "laplace"},
    )
    spec, fs = apply_option(base, settings, opt)
    assert isinstance(spec.kernel_of("a"), LogisticKernel)
    assert spec.intercept_scale == 3.0
    assert spec.treatments == base.treatments and spec.outcome == base.outcome
    assert fs.draws == 7 and fs.backend == "laplace"
    # a merged mapping keeps other treatments' entries
    two = surface_world(n_units=2, n_periods=3, treatments=("a", "b"), seed=0).spec
    spec2, _ = apply_option(
        two,
        settings,
        SpecOption(label="x", spec_update={"carryover": {"a": GeometricCarryover(max_lag=2)}}),
    )
    assert spec2.carried == ("a",)
    assert isinstance(spec2.kernel_of("b"), HillKernel)


def test_apply_option_that_breaks_the_spec_raises_before_fitting(world: SurfaceWorld) -> None:
    opt = SpecOption(label="bad", spec_update={"intercept": "hierarchical", "unit_labels": ()})
    with pytest.raises(ValueError, match="unit_labels"):
        apply_option(world.spec, FitSettings(), opt)


# -- the default estimand and realization -----------------------------------------------------


def test_default_estimand_is_a_per_unit_cumulative_contrast(world: SurfaceWorld) -> None:
    e = default_estimand(world.spec, world.panel)
    assert e.quantity.kind == "contrast"
    assert e.treatment.name == "a"
    assert e.level.unit == "individual"
    assert e.window.start == 0 and e.window.stop == world.n_periods
    assert e.window.basis == "cumulative"
    doses = world.panel.array("a")
    assert e.intervention.doses["a"] == pytest.approx(float(doses[doses > 0].mean()))
    assert e.reference is not None and e.reference.doses["a"] == 0.0
    with pytest.raises(ValueError, match="positive"):
        default_estimand(world.spec, world.panel, dose=-1.0)
    with pytest.raises(KeyError):
        default_estimand(world.spec, world.panel, "zzz")


def test_realized_point_returns_draws_or_a_typed_failure(
    world: SurfaceWorld, res: FitResult
) -> None:
    e = default_estimand(world.spec, world.panel)
    got = realized_point(e, res, definition="eti", mass=0.9)
    assert not isinstance(got, Unsupported)
    assert got.result.summary.interval.definition == "eti"
    assert got.result.summary.interval.mass == 0.9
    assert got.draws.shape == (res.n_draws(),)
    # a fit that dropped the counterfactual capability cannot realize the contrast
    failed = realized_point(e, res.restrict(["counterfactual"]), definition="eti", mass=0.9)
    assert isinstance(failed, Unsupported)
    assert "counterfactual" in failed.reason
    # a fit whose backend is missing has no posterior
    missing = fit(world.spec, world.panel, backend="no-such-backend")
    assert isinstance(missing.posterior, Unsupported)
    failed2 = realized_point(e, missing, definition="eti", mass=0.9)
    assert isinstance(failed2, Unsupported) and "posterior" in failed2.missing


# -- the curve ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def curve(world: SurfaceWorld) -> SpecCurve:
    axes = (
        SpecificationAxis(
            name="kernel",
            options=(
                SpecOption(label="hill"),
                SpecOption(
                    label="logistic",
                    spec_update={"kernels": {"a": LogisticKernel(reference_dose=50.0)}},
                ),
            ),
        ),
        SpecificationAxis(
            name="intercept_scale",
            options=(
                SpecOption(label="tight", spec_update={"intercept_scale": 0.5}),
                SpecOption(label="wide", spec_update={"intercept_scale": 5.0}),
            ),
        ),
    )
    return specification_curve(world.spec, world.panel, axes, draws=40, chains=1, seed=3)


def test_curve_fits_every_combination_with_labels_and_intervals(
    world: SurfaceWorld, curve: SpecCurve
) -> None:
    assert curve.n_total == 4 and curve.n_dropped == 0 and len(curve.rows) == 4
    labels = [(r.labels["kernel"], r.labels["intercept_scale"]) for r in curve.rows]
    assert labels == [
        ("hill", "tight"),
        ("hill", "wide"),
        ("logistic", "tight"),
        ("logistic", "wide"),
    ]
    for row in curve.rows:
        assert row.failure is None
        assert isinstance(row.interval, Interval)
        assert row.interval.definition == "eti" and row.interval.mass == 0.9
        assert row.estimate is not None and row.interval.lower <= row.estimate <= row.interval.upper
        assert row.n_draws == 40
        assert row.converged
    # specs differ per row and the hill/tight row is not the base spec (scale changed)
    assert len({r.spec_hash for r in curve.rows}) == 4
    assert curve.base_spec_hash == world.spec.content_hash()
    assert curve.estimand_name == "contrast_a"
    # the real effect is positive and every specification agrees on the sign
    assert all(r.estimate is not None and r.estimate > 0 for r in curve.rows)


def test_curve_summary_and_roundtrip(curve: SpecCurve) -> None:
    s = curve.summary()
    assert isinstance(s, SpecCurveSummary)
    assert s.n == 4 and s.n_failed == 0 and s.n_converged == 4
    pts = np.asarray(curve.estimates)
    assert s.median == pytest.approx(float(np.median(pts)))
    assert s.iqr_lower is not None and s.iqr_upper is not None and s.iqr_lower <= s.iqr_upper
    assert s.minimum == pytest.approx(pts.min()) and s.maximum == pytest.approx(pts.max())
    assert s.share_excluding_zero == 1.0 and s.share_positive == 1.0
    back = SpecCurve.from_json(curve.to_json())
    assert back == curve and back.content_hash() == curve.content_hash()


def test_curve_caps_the_product_and_reports_the_drop(world: SurfaceWorld) -> None:
    axes = (
        SpecificationAxis(
            name="scale",
            options=tuple(
                SpecOption(label=f"s{i}", spec_update={"intercept_scale": float(i)})
                for i in (1, 2, 3)
            ),
        ),
    )
    c = specification_curve(world.spec, world.panel, axes, draws=20, max_specs=2, seed=0)
    assert c.n_total == 3 and c.n_dropped == 1 and len(c.rows) == 2
    assert [r.labels["scale"] for r in c.rows] == ["s1", "s2"]
    assert c.summary().n == 2


def test_curve_records_failed_rows_instead_of_dropping_them(world: SurfaceWorld) -> None:
    axes = (
        SpecificationAxis(
            name="backend",
            options=(
                SpecOption(label="laplace"),
                SpecOption(label="missing", fit_update={"backend": "no-such-backend"}),
            ),
        ),
    )
    c = specification_curve(world.spec, world.panel, axes, draws=20, seed=0)
    assert len(c.rows) == 2
    ok, bad = c.rows
    assert ok.failure is None and ok.estimate is not None
    assert bad.failure is not None and bad.estimate is None and bad.interval is None
    assert not bad.converged
    s = c.summary()
    assert s.n == 2 and s.n_failed == 1 and s.n_converged == 1


def test_curve_argument_validation(world: SurfaceWorld) -> None:
    axis = SpecificationAxis(name="k", options=(SpecOption(label="a"),))
    with pytest.raises(ValueError, match="at least one axis"):
        specification_curve(world.spec, world.panel, ())
    with pytest.raises(ValueError, match="distinct"):
        specification_curve(world.spec, world.panel, (axis, axis))
    with pytest.raises(ValueError, match="max_specs"):
        specification_curve(world.spec, world.panel, (axis,), max_specs=0)


def test_spec_row_is_either_a_point_or_a_failure() -> None:
    with pytest.raises(ValueError, match="either"):
        SpecRow(index=0, labels={}, spec_hash="h", spec_name="s")
    with pytest.raises(ValueError, match="either"):
        SpecRow(
            index=0,
            labels={},
            spec_hash="h",
            spec_name="s",
            estimate=1.0,
            interval=Interval(lower=0.0, upper=2.0, definition="eti", mass=0.9),
            failure="x",
        )
    row = SpecRow(
        index=0,
        labels={"k": "a"},
        spec_hash="h",
        spec_name="s",
        estimate=1.0,
        interval=Interval(lower=0.5, upper=2.0, definition="eti", mass=0.9),
    )
    assert row.excludes_zero is True
    assert (
        SpecRow(index=1, labels={}, spec_hash="h", spec_name="s", failure="f").excludes_zero is None
    )
