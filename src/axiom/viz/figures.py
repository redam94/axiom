"""Plotly figures over axiom results, behind the ``[viz]`` extra.

plotly is imported lazily inside each figure function; ``axiom.viz``
imports without it and every figure returns ``Unsupported(reason="plotly
not installed", missing=("viz",))`` when it is absent. Figures are thin:
the numbers they draw come from the result objects, which carry their own
provenance (interval definition and mass, N, acceptance regions), and the
labels use the domain-general vocabulary (treatment, dose, outcome).

Each function documents the attributes it reads, so a result written by a
concurrent module only has to satisfy those:

* ``response_curve`` — a ``surface.FitResult`` (``predict_under``,
  ``surface.spec``, ``data``, ``n_periods``).
* ``forest`` — ``meta.ForestData`` (``rows[].label/estimate/interval/weight``,
  ``pooled_estimate``, ``pooled_interval``, ``prediction_interval``,
  ``mass``).
* ``funnel`` — ``meta.FunnelData`` (``y``, ``se``, ``pooled``,
  ``contours[].mass/se/lower/upper``).
* ``sbc_ranks`` — ``diagnose.SBCResult`` (``parameters[].name/histogram/
  bins/n/n_ranks/passed/ecdf_band``).
* ``coverage_plot`` — ``diagnose.CoverageResult``: ``parameters[]`` with
  ``name``, ``rate``, ``passed``, and a count-valued ``region`` with ``n``, ``lower`` /
  ``upper`` (``core.AcceptanceRegion``); ``mass`` on the result.
* ``spec_curve_plot`` — ``diagnose.SpecCurve`` (``rows[].estimate/interval/
  labels/converged/failure``, ``mass``, ``definition``, ``estimand_name``).
* ``backtest_plot`` — ``diagnose.Backtest``: ``scores[]`` with
  ``step``, ``mae``, ``rmse``, ``crps``, ``coverage``; ``mass``.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from axiom.core import Intervention, TimeWindow, Unsupported, eti, hdi
from axiom.core.intervals import IntervalDefinition

__all__ = [
    "available",
    "backtest_plot",
    "coverage_plot",
    "forest",
    "funnel",
    "response_curve",
    "sbc_ranks",
    "spec_curve_plot",
]

Array = npt.NDArray[np.float64]

_MISSING = Unsupported(
    reason="plotly not installed",
    missing=("viz",),
    detail={"install": "pip install 'axiom[viz]'"},
)


def _graph_objects() -> Any | Unsupported:
    """``plotly.graph_objects`` or the typed failure; never raises on a missing extra."""
    # plotly ships no type information; go through importlib so mypy sees ``Any``.
    try:
        go: Any = importlib.import_module("plotly.graph_objects")
    except ImportError:
        return _MISSING
    return go


def _subplots() -> Any | Unsupported:
    try:
        module: Any = importlib.import_module("plotly.subplots")
    except ImportError:
        return _MISSING
    return module.make_subplots


def available() -> bool:
    """True when plotly can be imported."""
    return not isinstance(_graph_objects(), Unsupported)


def _missing_attributes(obj: object, names: Sequence[str], what: str) -> Unsupported | None:
    absent = [n for n in names if not hasattr(obj, n)]
    if absent:
        return Unsupported(
            reason=f"{what} lacks attributes the figure reads: {absent}",
            missing=tuple(absent),
            detail={"type": type(obj).__name__},
        )
    return None


def _band(
    go: Any, x: Sequence[float], lower: Sequence[float], upper: Sequence[float], name: str
) -> Any:
    xs = list(x) + list(x)[::-1]
    ys = list(upper) + list(lower)[::-1]
    return go.Scatter(
        x=xs, y=ys, fill="toself", mode="lines", line={"width": 0}, opacity=0.25, name=name
    )


# -- surface ----------------------------------------------------------------------------------


def response_curve(
    result: Any,
    treatment: str,
    *,
    n_grid: int = 25,
    mass: float = 0.9,
    definition: IntervalDefinition = "eti",
    window: TimeWindow | None = None,
    seed: int | None = None,
) -> Any | Unsupported:
    """Expected outcome against the dose of ``treatment``, with a posterior band.

    Each grid dose is evaluated through ``result.predict_under`` (one
    forward per draw — the same ``forward`` the likelihood used) with the
    other treatments at their observed doses; the curve is the mean over
    units and the window's periods, and the band is the ``definition``
    interval at ``mass`` over draws. The grid runs from zero to the largest
    observed dose.
    """
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(result, ("predict_under", "surface", "data", "n_periods"), "result")
    if bad is not None:
        return bad
    spec = result.surface.spec
    if treatment not in spec.treatment_names:
        raise ValueError(f"no treatment {treatment!r}; have {list(spec.treatment_names)}")
    if n_grid < 2:
        raise ValueError("n_grid must be at least 2")
    observed = np.asarray(result.data[treatment], dtype=float)
    top = float(np.nanmax(observed)) if observed.size else 1.0
    grid = np.linspace(0.0, top if top > 0 else 1.0, n_grid)
    win = window if window is not None else TimeWindow(start=0, stop=int(result.n_periods))
    means: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for level in grid:
        draws = result.predict_under(
            Intervention(doses={treatment: float(level)}, mode="set"), window=win, seed=seed
        )
        if isinstance(draws, Unsupported):
            return draws
        values = np.asarray(draws.values, dtype=float)
        per_draw = values.reshape(values.shape[0] * values.shape[1], -1).mean(axis=1)
        iv = eti(per_draw, mass) if definition == "eti" else hdi(per_draw, mass)
        means.append(float(per_draw.mean()))
        lower.append(iv.lower)
        upper.append(iv.upper)
    entity = spec.treatment(treatment)
    dose_unit = f" ({entity.unit})" if entity.unit else ""
    out_unit = f" ({spec.outcome.unit})" if spec.outcome.unit else ""
    fig = go.Figure()
    fig.add_trace(_band(go, grid.tolist(), lower, upper, f"{definition} {mass:.0%}"))
    fig.add_trace(go.Scatter(x=grid.tolist(), y=means, mode="lines", name="posterior mean"))
    fig.update_layout(
        title=f"Response of {spec.outcome.name} to {treatment} dose",
        xaxis_title=f"{treatment} dose{dose_unit}",
        yaxis_title=f"expected {spec.outcome.name}{out_unit}",
    )
    return fig


# -- meta ------------------------------------------------------------------------------------


def forest(data: Any) -> Any | Unsupported:
    """A forest plot: one row per study with its interval, the pooled row, and the
    prediction interval when the data carries one."""
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(data, ("rows", "pooled_estimate", "mass"), "forest data")
    if bad is not None:
        return bad
    labels = [r.label for r in data.rows]
    est = [r.estimate for r in data.rows]
    lo = [r.estimate - r.interval.lower for r in data.rows]
    hi = [r.interval.upper - r.estimate for r in data.rows]
    sizes = [8.0 + 24.0 * (r.weight or 0.0) for r in data.rows]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=est,
            y=labels,
            mode="markers",
            marker={"size": sizes, "symbol": "square"},
            error_x={"type": "data", "symmetric": False, "array": hi, "arrayminus": lo},
            name="studies",
        )
    )
    pooled_iv = getattr(data, "pooled_interval", None)
    err = (
        {
            "type": "data",
            "symmetric": False,
            "array": [pooled_iv.upper - data.pooled_estimate],
            "arrayminus": [data.pooled_estimate - pooled_iv.lower],
        }
        if pooled_iv is not None
        else None
    )
    fig.add_trace(
        go.Scatter(
            x=[data.pooled_estimate],
            y=["pooled"],
            mode="markers",
            marker={"size": 14, "symbol": "diamond"},
            error_x=err,
            name="pooled",
        )
    )
    pred = getattr(data, "prediction_interval", None)
    if pred is not None:
        fig.add_trace(
            go.Scatter(
                x=[pred.lower, pred.upper],
                y=["prediction", "prediction"],
                mode="lines",
                line={"width": 4},
                name=f"prediction interval ({pred.mass:.0%})",
            )
        )
    fig.add_vline(x=0.0, line={"dash": "dot"})
    fig.update_layout(
        title=f"Forest plot ({data.mass:.0%} intervals)",
        xaxis_title="estimate",
        yaxis_title="study",
    )
    return fig


def funnel(data: Any) -> Any | Unsupported:
    """A funnel plot: study estimates against their standard errors (precision upward)
    with the pseudo-confidence contours around the pooled estimate."""
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(data, ("y", "se", "pooled", "contours"), "funnel data")
    if bad is not None:
        return bad
    fig = go.Figure()
    for c in data.contours:
        fig.add_trace(
            go.Scatter(
                x=list(c.lower) + list(c.upper)[::-1],
                y=list(c.se) + list(c.se)[::-1],
                mode="lines",
                line={"dash": "dash", "width": 1},
                name=f"{c.mass:.0%} contour",
            )
        )
    fig.add_trace(go.Scatter(x=list(data.y), y=list(data.se), mode="markers", name="studies"))
    fig.add_vline(x=data.pooled, line={"dash": "dot"})
    fig.update_layout(
        title="Funnel plot",
        xaxis_title="estimate",
        yaxis_title="standard error",
        yaxis={"autorange": "reversed"},
    )
    return fig


# -- diagnose ---------------------------------------------------------------------------------


def sbc_ranks(result: Any, *, columns: int = 3) -> Any | Unsupported:
    """Rank histograms per parameter from an SBC run, with the expected count per bin and
    the binomial band around it; failed parameters are titled as such."""
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(result, ("parameters",), "SBC result")
    if bad is not None:
        return bad
    make_subplots = _subplots()
    if isinstance(make_subplots, Unsupported):
        return make_subplots
    params = list(result.parameters)
    if not params:
        return Unsupported(reason="SBC result has no ranked parameters", missing=("parameters",))
    columns = max(1, min(columns, len(params)))
    rows = int(np.ceil(len(params) / columns))
    titles = [f"{p.name} ({'pass' if p.passed else 'FAIL'})" for p in params]
    fig = make_subplots(rows=rows, cols=columns, subplot_titles=titles)
    for i, p in enumerate(params):
        r, c = divmod(i, columns)
        expected = p.n / p.bins
        # binomial band around the expected count per bin (normal approximation)
        sd = float(np.sqrt(p.n * (1.0 / p.bins) * (1.0 - 1.0 / p.bins)))
        fig.add_trace(
            go.Bar(x=list(range(p.bins)), y=list(p.histogram), name=p.name, showlegend=False),
            row=r + 1,
            col=c + 1,
        )
        for level in (expected - 2.0 * sd, expected, expected + 2.0 * sd):
            fig.add_hline(y=level, line={"dash": "dot", "width": 1}, row=r + 1, col=c + 1)
    fig.update_layout(title="SBC rank histograms", xaxis_title="rank bin", yaxis_title="count")
    return fig


def coverage_plot(result: Any) -> Any | Unsupported:
    """Per-parameter coverage against the nominal mass with each parameter's exact
    acceptance region (``region.lower``/``region.upper``)."""
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(result, ("parameters", "mass"), "coverage result")
    if bad is not None:
        return bad
    params = list(result.parameters)
    for p in params:
        bad = _missing_attributes(p, ("name", "rate", "region", "passed"), "parameter coverage")
        if bad is not None:
            return bad
    names = [p.name for p in params]
    # ``rate`` is covered / n; the acceptance region is in counts, so divide by its n.
    cov = [float(p.rate) for p in params]
    lo = [float(p.rate) - float(p.region.lower) / float(p.region.n) for p in params]
    hi = [float(p.region.upper) / float(p.region.n) - float(p.rate) for p in params]
    colors = ["#2a9d8f" if p.passed else "#e76f51" for p in params]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=names,
            y=cov,
            mode="markers",
            marker={"size": 10, "color": colors},
            error_y={"type": "data", "symmetric": False, "array": hi, "arrayminus": lo},
            name="coverage (acceptance region)",
        )
    )
    fig.add_hline(y=float(result.mass), line={"dash": "dot"})
    fig.update_layout(
        title=f"Interval coverage at nominal {float(result.mass):.0%}",
        xaxis_title="parameter",
        yaxis_title="coverage",
        yaxis={"range": [0.0, 1.0]},
    )
    return fig


def spec_curve_plot(curve: Any) -> Any | Unsupported:
    """The specification curve: estimates sorted with their intervals, coloured by whether
    the interval excludes zero; failed specifications are omitted and counted in the title."""
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(curve, ("rows", "mass", "definition"), "specification curve")
    if bad is not None:
        return bad
    fitted = [r for r in curve.rows if r.estimate is not None and r.interval is not None]
    n_failed = len(curve.rows) - len(fitted)
    if not fitted:
        return Unsupported(reason="no specification produced an estimate", missing=("rows",))
    order = sorted(range(len(fitted)), key=lambda i: fitted[i].estimate)
    est = [fitted[i].estimate for i in order]
    lo = [fitted[i].estimate - fitted[i].interval.lower for i in order]
    hi = [fitted[i].interval.upper - fitted[i].estimate for i in order]
    text = [
        "<br>".join(f"{k}: {v}" for k, v in fitted[i].labels.items())
        + ("" if fitted[i].converged else "<br>(not converged)")
        for i in order
    ]
    colors = [
        "#2a9d8f" if (fitted[i].interval.lower > 0 or fitted[i].interval.upper < 0) else "#8d99ae"
        for i in order
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(order))),
            y=est,
            mode="markers",
            marker={"color": colors, "size": 8},
            error_y={"type": "data", "symmetric": False, "array": hi, "arrayminus": lo},
            text=text,
            hoverinfo="text+y",
            name=f"{curve.definition} {float(curve.mass):.0%}",
        )
    )
    fig.add_hline(y=0.0, line={"dash": "dot"})
    name = getattr(curve, "estimand_name", "estimand")
    fig.update_layout(
        title=f"Specification curve for {name} ({len(fitted)} fitted, {n_failed} failed)",
        xaxis_title="specification (sorted by estimate)",
        yaxis_title=name,
    )
    return fig


def backtest_plot(backtest: Any) -> Any | Unsupported:
    """Per-horizon scores of a rolling-origin backtest — MAE, RMSE, CRPS — and the interval
    coverage against the nominal mass."""
    go = _graph_objects()
    if isinstance(go, Unsupported):
        return go
    bad = _missing_attributes(backtest, ("scores", "mass"), "backtest")
    if bad is not None:
        return bad
    make_subplots = _subplots()
    if isinstance(make_subplots, Unsupported):
        return make_subplots
    rows = list(backtest.scores)
    for r in rows:
        bad = _missing_attributes(r, ("step", "mae", "rmse", "crps", "coverage"), "horizon score")
        if bad is not None:
            return bad
    h = [int(r.step) for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("error by horizon", "interval coverage"))
    for attr in ("mae", "rmse", "crps"):
        fig.add_trace(
            go.Scatter(
                x=h, y=[float(getattr(r, attr)) for r in rows], mode="lines+markers", name=attr
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(x=h, y=[float(r.coverage) for r in rows], mode="lines+markers", name="coverage"),
        row=1,
        col=2,
    )
    fig.add_hline(y=float(backtest.mass), line={"dash": "dot"}, row=1, col=2)
    fig.update_xaxes(title_text="horizon (periods)")
    fig.update_yaxes(title_text="outcome error", row=1, col=1)
    fig.update_yaxes(title_text="coverage", range=[0.0, 1.0], row=1, col=2)
    fig.update_layout(title="Rolling-origin backtest")
    return fig
