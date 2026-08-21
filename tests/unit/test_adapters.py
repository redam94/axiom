"""``axiom.adapters.marketing``: frames → ``Panel`` with the right roles and units, the MFF
long format, the surface preset, and the return-on-spend estimands on a fitted world."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom.adapters import (
    EXPOSURE,
    KPI,
    Channel,
    Geo,
    MarketingRoles,
    contribution,
    impressions,
    marginal_roas,
    marketing_spec,
    panel_from_marketing_frame,
    panel_from_mff,
    roas,
    roi,
    role_map,
    spend,
)
from axiom.core import D, Outcome, Posterior, Treatment, Unit, Unsupported, dimensionless
from axiom.estimands import EstimandResult
from axiom.sim import DosePlan, SurfaceWorld, surface_world
from axiom.surface import FitResult, GeometricCarryover, HillKernel, fit

# -- frames ------------------------------------------------------------------------------------


def _frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for geo in ("east", "north", "west"):
        for week in range(8):
            rows.append(
                {
                    "geo": geo,
                    "week": week,
                    "revenue": 100.0 + rng.normal(0.0, 1.0),
                    "tv": float(abs(rng.normal(50.0, 10.0))),
                    "search": float(abs(rng.normal(20.0, 5.0))),
                    "tv_impressions": float(rng.integers(100, 200)),
                    "holiday": float(week == 3),
                    "ignored": "text",
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def roles() -> MarketingRoles:
    return MarketingRoles(
        kpi="revenue",
        channels=("tv", "search"),
        geo="geo",
        date="week",
        impressions=("tv_impressions",),
        controls=("holiday",),
        kpi_dimension="currency",
    )


def test_aliases_are_the_general_entities() -> None:
    assert Channel is Treatment and Geo is Unit and KPI is Outcome
    d = spend("tv")
    assert d.dimension == D.currency and d.unit == "USD" and d.numeraire == "USD"
    assert impressions("tv_imp").dimension == EXPOSURE
    assert EXPOSURE == D.exposure_count


def test_roles_translate_to_a_role_map(roles: MarketingRoles) -> None:
    rm = role_map(roles)
    assert rm.unit == "geo" and rm.time == "week"
    assert rm.outcome == ("revenue", Outcome(name="revenue", dimension=D.currency, unit="USD"))
    assert rm.treatments["tv"] == Treatment(name="tv", dimension=D.currency, unit="USD")
    assert rm.covariates["tv_impressions"].dimension == EXPOSURE
    assert rm.covariates["holiday"].dimension == dimensionless()
    with pytest.raises(ValueError, match="more than one role"):
        MarketingRoles(kpi="tv", channels=("tv",))
    with pytest.raises(ValueError, match="at least one channel"):
        MarketingRoles(kpi="y", channels=())


def test_marketing_frame_becomes_a_panel_with_units(roles: MarketingRoles) -> None:
    panel = panel_from_marketing_frame(_frame(), roles)
    assert panel.units == ("east", "north", "west")
    assert panel.completeness().balanced and len(panel.periods) == 8
    assert panel.roles.unit_of("tv") == "USD" and panel.roles.unit_of("revenue") == "USD"
    assert panel.roles.dimension_of("tv_impressions") == EXPOSURE
    assert "ignored" not in panel.frame.columns
    count_roles = roles.model_copy(update={"kpi_dimension": "outcome", "kpi_unit": "orders"})
    p2 = panel_from_marketing_frame(_frame(), count_roles)
    assert p2.roles.outcome[1].dimension == D.outcome and p2.roles.unit_of("revenue") == "orders"
    with pytest.raises(ValueError, match="lacks columns"):
        panel_from_marketing_frame(_frame().drop(columns=["search"]), roles)


def test_mff_long_format_gives_the_same_panel(roles: MarketingRoles) -> None:
    wide = _frame().drop(columns=["ignored"])
    long = wide.melt(id_vars=["geo", "week"], var_name="variable", value_name="value")
    long = pd.concat(
        [long, pd.DataFrame({"geo": ["east"], "week": [0], "variable": ["extra"], "value": [1.0]})]
    )
    panel = panel_from_mff(long, roles)
    assert panel.content_hash() == panel_from_marketing_frame(wide, roles).content_hash()
    with pytest.raises(ValueError, match="duplicate"):
        panel_from_mff(pd.concat([long, long.iloc[:1]]), roles)
    with pytest.raises(ValueError, match="lacks column"):
        panel_from_mff(long.rename(columns={"value": "v"}), roles)
    with pytest.raises(ValueError, match="lacks columns"):
        panel_from_mff(long[long["variable"] != "tv"], roles)


def test_marketing_spec_is_a_surface_spec_through_the_builder(roles: MarketingRoles) -> None:
    panel = panel_from_marketing_frame(_frame(), roles)
    spec = marketing_spec(panel, seasonality=(4.0, 1), trend=True)
    assert spec.treatment_names == ("tv", "search")
    assert spec.treatment("tv") == panel.roles.treatments["tv"]
    assert spec.outcome == panel.roles.outcome[1]
    assert spec.intercept == "hierarchical" and spec.unit_labels == panel.units
    tv = panel.column("tv")
    assert spec.kernels["tv"] == HillKernel(reference_dose=float(tv[tv > 0].mean()))
    assert spec.carryover["tv"] == GeometricCarryover(max_lag=4)
    assert len(spec.nuisance.terms) == 2
    only = marketing_spec(panel, ["search"], carryover=None, intercept="shared", kernel="linear")
    assert only.treatment_names == ("search",) and only.carryover == {}
    assert only.kernel_of("search").name == "linear"
    with pytest.raises(ValueError, match="not treatment columns"):
        marketing_spec(panel, ["radio"])


# -- return on spend on a fitted world --------------------------------------------------------


@pytest.fixture(scope="module")
def revenue_world() -> SurfaceWorld:
    return surface_world(
        n_units=3,
        n_periods=10,
        treatments=("tv",),
        outcome=Outcome(name="revenue", dimension=D.currency, unit="USD"),
        doses=DosePlan(scale=50.0),
        intercept="shared",
        truth={"beta_tv": 2.0, "alpha": 1.0},
        noise_sd=0.1,
        seed=11,
    )


@pytest.fixture(scope="module")
def revenue_fit(revenue_world: SurfaceWorld) -> FitResult:
    res = fit(
        revenue_world.spec, revenue_world.panel, backend="laplace", draws=200, chains=2, seed=5
    )
    assert isinstance(res.posterior, Posterior)
    return res


def _truth(world: SurfaceWorld) -> tuple[float, float]:
    """True (contribution, ROAS) of ``tv`` over the whole panel, through the world's forward."""
    zero = dict(world.data)
    zero["tv"] = np.zeros_like(world.data["tv"])
    baseline = world.surface.forward(zero, world.theta)
    contrib = float(world.mean.sum() - baseline.sum())
    return contrib, contrib / float(world.data["tv"].sum())


def _true_marginal(world: SurfaceWorld, *, start: int, stop: int, h: float = 1e-3) -> float:
    """``d(Σ_window Y)/dδ / n_cells`` under a common shift of every dose, by central difference
    through the world's forward (no carryover, so the shift acts cell by cell)."""
    up, down = dict(world.data), dict(world.data)
    up["tv"] = world.data["tv"] + h
    down["tv"] = world.data["tv"] - h
    diff = world.surface.forward(up, world.theta) - world.surface.forward(down, world.theta)
    window = diff[:, start:stop]
    return float(window.sum() / (2.0 * h) / window.size)


def test_roas_roi_contribution_marginal_carry_units(
    revenue_world: SurfaceWorld, revenue_fit: FitResult
) -> None:
    true_contrib, true_roas = _truth(revenue_world)
    r = roas(revenue_fit, "tv", mass=0.9)
    assert isinstance(r, EstimandResult)
    assert r.kind == "ratio" and r.dimension == dimensionless()
    assert r.status == "downgraded" and "assumed_identified" in {a.name for a in r.assumptions}
    assert r.summary.interval.mass == 0.9 and r.summary.interval.definition == "hdi"
    assert abs(r.summary.mean - true_roas) <= 4.0 * r.summary.sd + 0.05 * abs(true_roas)

    c = contribution(revenue_fit, "tv", mass=0.9, definition="eti")
    assert isinstance(c, EstimandResult)
    assert c.kind == "contrast" and c.dimension == D.currency and c.unit == "USD"
    assert c.summary.interval.definition == "eti"
    assert abs(c.summary.mean - true_contrib) <= 4.0 * c.summary.sd + 0.05 * abs(true_contrib)

    i = roi(revenue_fit, "tv", mass=0.9, seed=0)
    r_same = roas(revenue_fit, "tv", mass=0.9, seed=0)
    assert isinstance(i, EstimandResult) and isinstance(r_same, EstimandResult)
    assert i.estimand_name == "roi_tv" and i.dimension == dimensionless()
    assert i.summary.mean == pytest.approx(r_same.summary.mean - 1.0)
    assert i.summary.interval.lower == pytest.approx(r_same.summary.interval.lower - 1.0)
    assert i.detail["derived_from"] == "roas_tv"

    m = marginal_roas(revenue_fit, "tv", window=(2, 10))
    assert isinstance(m, EstimandResult)
    assert m.kind == "marginal" and m.dimension == dimensionless()
    true_marginal = _true_marginal(revenue_world, start=2, stop=10)
    assert abs(m.summary.mean - true_marginal) <= 4.0 * m.summary.sd + 0.05 * abs(true_marginal)


def test_roas_refuses_a_count_kpi() -> None:
    world = surface_world(n_units=2, n_periods=4, treatments=("tv",), intercept="shared", seed=1)
    res = fit(world.spec, world.panel, backend="laplace", draws=20, chains=1, seed=2)
    out = roas(res, "tv")
    assert isinstance(out, Unsupported) and out.missing == ("currency_outcome",)
    assert isinstance(roi(res, "tv"), Unsupported)
    assert isinstance(marginal_roas(res, "tv"), Unsupported)
    c = contribution(res, "tv")
    assert isinstance(c, EstimandResult) and c.dimension == D.outcome
    with pytest.raises(KeyError):
        contribution(res, "radio")
