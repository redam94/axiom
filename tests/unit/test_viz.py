"""``axiom.viz``: every figure is a plotly ``Figure`` with plotly installed and a typed
``Unsupported`` when the import is absent; no marketing vocabulary in the labels."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from axiom.core import AcceptanceRegion, Interval, Posterior, Unsupported, clopper_pearson
from axiom.meta import forest_data, funnel_data, random_effects
from axiom.sim import DosePlan, surface_world
from axiom.surface import FitResult, fit
from axiom.viz import (
    available,
    backtest_plot,
    coverage_plot,
    forest,
    funnel,
    response_curve,
    sbc_ranks,
    spec_curve_plot,
)

plotly = pytest.importorskip("plotly")
import plotly.graph_objects as go  # type: ignore[import-untyped]  # noqa: E402

_Y = np.array([0.42, 0.55, 0.31, 0.67, 0.48])
_SE = np.array([0.10, 0.15, 0.12, 0.20, 0.11])


@pytest.fixture(scope="module")
def fitted() -> FitResult:
    world = surface_world(
        n_units=2,
        n_periods=6,
        treatments=("a",),
        doses=DosePlan(scale=10.0),
        intercept="shared",
        seed=3,
    )
    res = fit(world.spec, world.panel, backend="laplace", draws=40, chains=1, seed=4)
    assert isinstance(res.posterior, Posterior)
    return res


def _sbc() -> SimpleNamespace:
    return SimpleNamespace(
        parameters=(
            SimpleNamespace(
                name="beta_a",
                histogram=(5, 6, 4, 5),
                bins=4,
                n=20,
                n_ranks=99,
                passed=True,
                ecdf_band=0.3,
            ),
            SimpleNamespace(
                name="k_a",
                histogram=(12, 3, 2, 3),
                bins=4,
                n=20,
                n_ranks=99,
                passed=False,
                ecdf_band=0.3,
            ),
        )
    )


def _coverage() -> SimpleNamespace:
    region: AcceptanceRegion = clopper_pearson(30, 0.9, 0.05)
    return SimpleNamespace(
        mass=0.9,
        parameters=(
            SimpleNamespace(name="beta_a", rate=0.9, region=region, passed=True),
            SimpleNamespace(name="k_a", rate=0.6, region=region, passed=False),
        ),
    )


def _curve() -> SimpleNamespace:
    def row(i: int, est: float | None, fail: str | None = None) -> SimpleNamespace:
        iv = (
            None
            if est is None
            else Interval(lower=est - 0.3, upper=est + 0.3, definition="hdi", mass=0.9)
        )
        return SimpleNamespace(
            estimate=est,
            interval=iv,
            labels={"kernel": f"k{i}", "carryover": "none"},
            converged=est is not None,
            failure=fail,
        )

    return SimpleNamespace(
        rows=(row(0, 1.2), row(1, 0.1), row(2, None, "backend declined"), row(3, 0.8)),
        mass=0.9,
        definition="hdi",
        estimand_name="contrast_a",
    )


def _backtest() -> SimpleNamespace:
    return SimpleNamespace(
        mass=0.9,
        scores=tuple(
            SimpleNamespace(
                step=h, mae=0.1 * h, rmse=0.12 * h, crps=0.05 * h, coverage=0.9 - 0.02 * h
            )
            for h in (1, 2, 3)
        ),
    )


def test_available_with_plotly_installed() -> None:
    assert available() is True


def test_every_figure_is_a_plotly_figure(fitted: FitResult) -> None:
    pooled = random_effects(_Y, _SE)
    figs = {
        "response_curve": response_curve(fitted, "a", n_grid=4, mass=0.8),
        "forest": forest(forest_data(_Y, _SE, pooled, labels=list("abcde"))),
        "funnel": funnel(funnel_data(_Y, _SE, pooled)),
        "sbc_ranks": sbc_ranks(_sbc()),
        "coverage_plot": coverage_plot(_coverage()),
        "spec_curve_plot": spec_curve_plot(_curve()),
        "backtest_plot": backtest_plot(_backtest()),
    }
    for name, fig in figs.items():
        assert isinstance(fig, go.Figure), (name, fig)
        labels = [fig.layout.title.text or ""]
        for axis in ("xaxis", "yaxis", "xaxis2", "yaxis2"):
            title = getattr(fig.layout, axis).title.text if axis in fig.layout else None
            labels.append(title or "")
        labels += [str(t.name) for t in fig.data]
        for banned in ("channel", "spend", "roas", "kpi", "geo"):
            assert banned not in " ".join(labels).lower(), (name, banned)


def test_response_curve_labels_and_shape(fitted: FitResult) -> None:
    fig = response_curve(fitted, "a", n_grid=5)
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.title.text == "a dose (USD)"
    assert fig.layout.yaxis.title.text == "expected y"
    band, mean = fig.data
    assert len(mean.x) == 5 and len(band.x) == 10
    assert mean.x[0] == 0.0 and mean.x[-1] == pytest.approx(float(np.max(fitted.data["a"])))
    assert all(
        lo <= m <= hi for lo, m, hi in zip(band.y[5:][::-1], mean.y, band.y[:5], strict=True)
    )
    with pytest.raises(ValueError, match="no treatment"):
        response_curve(fitted, "zzz")
    with pytest.raises(ValueError, match="n_grid"):
        response_curve(fitted, "a", n_grid=1)


def test_duck_typed_inputs_missing_attributes_are_typed_failures() -> None:
    out = coverage_plot(SimpleNamespace(mass=0.9, parameters=(SimpleNamespace(name="x"),)))
    assert isinstance(out, Unsupported) and "rate" in out.missing
    out2 = backtest_plot(SimpleNamespace(mass=0.9))
    assert isinstance(out2, Unsupported) and out2.missing == ("scores",)
    empty = spec_curve_plot(SimpleNamespace(rows=(), mass=0.9, definition="eti"))
    assert isinstance(empty, Unsupported)


def test_without_plotly_every_figure_is_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    monkeypatch.setitem(sys.modules, "plotly.subplots", None)
    assert available() is False
    calls = [
        lambda: response_curve(object(), "a"),
        lambda: forest(object()),
        lambda: funnel(object()),
        lambda: sbc_ranks(object()),
        lambda: coverage_plot(object()),
        lambda: spec_curve_plot(object()),
        lambda: backtest_plot(object()),
    ]
    for call in calls:
        out = call()
        assert isinstance(out, Unsupported)
        assert out.reason == "plotly not installed" and out.missing == ("viz",)
