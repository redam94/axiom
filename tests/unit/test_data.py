from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from axiom.core import Covariate, D, Outcome, Treatment, dimensionless
from axiom.data import Panel, PanelError, RoleMap, fit_scaling


@pytest.fixture
def roles() -> RoleMap:
    return RoleMap(
        unit="plot",
        time="week",
        outcome=("y", Outcome(name="yield_total", dimension=D.outcome, unit="kg")),
        treatments={"fert": Treatment(name="fertilizer", dimension=D.currency, unit="USD")},
        covariates={"rain": Covariate(name="rain", dimension=dimensionless())},
    )


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "plot": np.repeat(["p1", "p2", "p3"], 8),
            "week": np.tile(range(8), 3),
            "y": rng.gamma(3, 2, 24),
            "fert": rng.uniform(0, 50, 24),
            "rain": rng.normal(0, 1, 24),
        }
    ).sample(frac=1, random_state=1)


def test_rolemap_validation(roles: RoleMap) -> None:
    assert roles.columns == ("plot", "week", "y", "fert", "rain")
    assert roles.kind_of("fert") == "treatment" and roles.dimension_of("fert") == D.currency
    assert roles.unit_of("y") == "kg" and roles.unit_of("plot") is None
    with pytest.raises(KeyError):
        roles.dimension_of("plot")
    with pytest.raises(ValueError, match="more than one role"):
        RoleMap(unit="a", time="a", outcome=("y", Outcome(name="y", dimension=D.outcome)))
    with pytest.raises(ValueError, match="entity names"):
        RoleMap(
            unit="u",
            time="t",
            outcome=("y", Outcome(name="same", dimension=D.outcome)),
            treatments={"x": Treatment(name="same", dimension=D.currency)},
        )


def test_panel_sorts_validates_and_reports(roles: RoleMap, frame: pd.DataFrame) -> None:
    p = Panel(frame, roles)
    assert p.units == ("p1", "p2", "p3") and p.periods == tuple(range(8))
    assert p.frame["week"].tolist()[:8] == list(range(8))
    c = p.completeness()
    assert c.balanced and c.missing_cells == 0 and c.n_rows == 24
    assert p.array("y").shape == (3, 8)
    assert p.select_units(["p2"]).units == ("p2",)
    with pytest.raises(PanelError, match="lacks columns"):
        Panel(frame.drop(columns=["rain"]), roles)
    with pytest.raises(PanelError, match="no role"):
        Panel(frame.assign(extra=1), roles)
    with pytest.raises(PanelError, match="numeric"):
        Panel(frame.assign(y="a"), roles)


def test_unbalanced_panel_is_reported_not_fixed(roles: RoleMap, frame: pd.DataFrame) -> None:
    p = Panel(frame.iloc[1:], roles)
    c = p.completeness()
    assert not c.balanced and c.missing_cells == 1 and sum(c.gaps.values()) == 1
    assert np.isnan(p.array("y")).sum() == 1
    with pytest.raises(PanelError, match="balanced panel is required"):
        p.require_balanced(context="a test")
    dup = Panel(pd.concat([frame, frame.iloc[:1]]), roles)
    assert dup.completeness().duplicate_rows == 1 and not dup.completeness().balanced


def test_content_hash_is_order_invariant(roles: RoleMap, frame: pd.DataFrame) -> None:
    a = Panel(frame, roles)
    b = Panel(frame.sample(frac=1, random_state=9), roles)
    assert a.content_hash() == b.content_hash()
    c = Panel(frame.assign(y=frame["y"] + 1), roles)
    assert a.content_hash() != c.content_hash()


def test_scaling_inverts_exactly(roles: RoleMap, frame: pd.DataFrame) -> None:
    p = Panel(frame, roles)
    sp = fit_scaling(p)
    assert sp.columns["y"].method == "max" and sp.columns["rain"].method == "standardize"
    z = sp.scale(p)
    assert z.frame["fert"].abs().max() == pytest.approx(1.0)
    assert z.frame["rain"].std(ddof=1) == pytest.approx(1.0)
    back = sp.unscale(z)
    np.testing.assert_allclose(back.frame["y"], p.frame["y"])
    np.testing.assert_allclose(sp.unscale_column("rain", z.frame["rain"]), p.frame["rain"])
    assert sp.content_hash() == sp.from_json(sp.to_json()).content_hash()


def test_scaling_refuses_degenerate_columns(roles: RoleMap, frame: pd.DataFrame) -> None:
    p = Panel(frame.assign(rain=1.0), roles)
    with pytest.raises(ValueError, match="constant"):
        fit_scaling(p)
    p0 = Panel(frame.assign(fert=0.0), roles)
    with pytest.raises(ValueError, match="all zeros"):
        fit_scaling(p0)
    assert fit_scaling(p0, treatments="none").columns["fert"].method == "none"
