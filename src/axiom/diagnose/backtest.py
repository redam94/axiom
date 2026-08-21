"""Rolling-origin backtest and the freeze-and-replay predictor.

Extracted from the parent's ``validation/backtest.py`` and
``validation/frozen_predictor.py`` (ledger rows "diagnose/backtest"). The
one thing that must survive the port is the *graph-faithful saturation*
fix (parent issue #202): a forecast is the model's own ``forward`` over the
**full dose path** — training periods followed by the horizon — so that the
carryover state built up during training is carried into the forecast
periods exactly as the fitted likelihood saw it. The naive version evaluated
a linearized or horizon-only copy, which silently forecast the carried
treatments as if their dose history were zero: unsaturated and wrong.
``forecast`` therefore calls ``axiom.surface.forward.predict`` on
``prepare(spec, panel[:origin + horizon])`` and keeps the last ``horizon``
columns (rule 3: one forward).

``rolling_origin`` refits the model on ``panel[:origin]`` at every origin
(through ``axiom.surface.fit``), forecasts ``horizon`` periods ahead, and
scores each horizon step ``h = 1..H`` across origins and units:

* ``mae`` and ``rmse`` of the posterior-mean forecast against the observed
  outcome;
* ``crps`` from the predictive draws (outcome-scale, likelihood noise
  included), the empirical form ``E|X − y| − ½ E|X − X'|`` (Gneiting &
  Raftery 2007) computed per cell from the sorted draws and averaged;
* ``coverage`` of the ``mass`` predictive interval: the share of cells whose
  observed outcome falls inside it, with the exact binomial acceptance
  region ``clopper_pearson(n, mass, alpha)`` and ``passed``.

A refit that fails (typed failure from the backend) is recorded per origin
in ``failures`` and counted in ``n_failed_fits``; it is never dropped
silently, and a backtest with no successful origin is an ``Unsupported``.

``FrozenPredictor`` freezes a fit's spec, posterior, and nuisance
conventions so the same posterior can be replayed on a new panel through
``prepare`` + ``predict`` — the same forward again — with a content hash
over the spec and the draws for provenance. The new panel must carry the
outcome column (``prepare`` reads it for layout; fill it with zeros when
the outcome is unknown) and, when the spec indexes units, exactly the
spec's units.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, model_validator

from axiom.core import (
    AcceptanceRegion,
    Interval,
    Posterior,
    PredictiveDraws,
    Spec,
    Unsupported,
    clopper_pearson,
    eti,
    hdi,
)
from axiom.core.intervals import IntervalDefinition
from axiom.data import Panel
from axiom.surface import FitResult, Surface, SurfaceSpec, fit, prepare, resolve_conventions
from axiom.surface.forward import predict
from axiom.surface.model import Conventions

__all__ = [
    "Backtest",
    "FrozenPredictor",
    "HorizonScore",
    "OriginFailure",
    "OriginForecast",
    "crps",
    "forecast",
    "freeze",
    "rolling_origin",
    "training_panel",
]

log = logging.getLogger(__name__)

Array = npt.NDArray[np.float64]


# -- panels ---------------------------------------------------------------------------------


def training_panel(panel: Panel, n_periods: int) -> Panel:
    """The first ``n_periods`` periods of ``panel`` (every unit, roles unchanged)."""
    periods = list(panel.periods)
    if not 1 <= n_periods <= len(periods):
        raise ValueError(
            f"n_periods must be in [1, {len(periods)}] for this panel, got {n_periods}"
        )
    keep = set(periods[:n_periods])
    frame = panel.frame
    mask = frame[panel.roles.time].isin(keep)
    return Panel(frame.loc[mask], panel.roles)


# -- the forecast: one forward over the full dose path ---------------------------------------


def forecast(
    spec: SurfaceSpec,
    posterior: Posterior,
    panel: Panel,
    *,
    origin: int,
    horizon: int,
    conventions: Conventions | None = None,
    seed: int | None = None,
    noise: bool = False,
) -> PredictiveDraws:
    """Predictive draws for periods ``[origin, origin + horizon)`` with the carryover state
    carried from the periods before ``origin``.

    The surface is evaluated on ``prepare(spec, panel[:origin + horizon])``
    — the doses of the training periods *and* the horizon — and the last
    ``horizon`` columns are returned, ``values`` shaped ``(chain, draw,
    n_units, horizon)``. ``conventions`` lays the nuisance trend out on the
    fitted basis (``FitResult.provenance["nuisance_conventions"]``);
    ``noise=True`` adds likelihood noise per draw (the outcome-scale
    predictive). ``coords`` holds the unit labels, the period labels of the
    horizon, and per treatment the horizon doses flattened in C order.
    """
    n_total = len(panel.periods)
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if origin < 1:
        raise ValueError(f"origin must be at least 1 (one training period), got {origin}")
    if origin + horizon > n_total:
        raise ValueError(
            f"origin {origin} + horizon {horizon} runs past the panel's {n_total} periods"
        )
    path = training_panel(panel, origin + horizon)
    data = prepare(spec, path, conventions=conventions)
    draws = predict(Surface(spec), posterior, data, seed=seed, noise=noise)
    values = np.asarray(draws.values, dtype=np.float64)
    n_units = int(np.asarray(data[spec.treatment_names[0]]).shape[0])
    chains, per_chain = int(values.shape[0]), int(values.shape[1])
    grid = np.broadcast_to(values, (chains, per_chain, n_units, origin + horizon))
    labels = spec.unit_labels or path.units
    coords: dict[str, list[Any]] = {
        "unit": list(labels),
        "period": list(path.periods[origin:]),
    }
    for t in spec.treatment_names:
        dose_grid = np.asarray(data[t], dtype=np.float64)
        coords[t] = [float(x) for x in dose_grid[:, origin:].reshape(-1)]
    return PredictiveDraws(
        values=np.ascontiguousarray(grid[..., origin:]),
        intervention=draws.intervention,
        window=None,
        coords=coords,
        seed=seed,
    )


# -- scoring ----------------------------------------------------------------------------------


def crps(draws: npt.ArrayLike, observed: npt.ArrayLike) -> Array:
    """Empirical CRPS per cell: ``E|X − y| − ½ E|X − X'|`` from draws along the first axis.

    ``draws`` is ``(n_draws, *shape)``, ``observed`` broadcastable to
    ``shape``. The pairwise term uses the sorted-sample identity
    ``½ E|X − X'| = (1/n²) Σ_i (2i − n − 1) x_(i)``, which is ``O(n log n)``.
    """
    x = np.asarray(draws, dtype=np.float64)
    y = np.asarray(observed, dtype=np.float64)
    if x.ndim < 1 or x.shape[0] < 1:
        raise ValueError("crps needs at least one draw along the first axis")
    n = x.shape[0]
    term1 = np.mean(np.abs(x - y[None, ...]), axis=0)
    xs = np.sort(x, axis=0)
    weights = (2.0 * np.arange(1, n + 1) - n - 1.0) / (n * n)
    term2 = np.tensordot(weights, xs, axes=(0, 0))
    return np.asarray(term1 - term2, dtype=np.float64)


class HorizonScore(Spec):
    """Scores for one step ahead, pooled over origins and units (``n`` cells)."""

    step: int = Field(ge=1)
    n: int = Field(ge=1)
    mae: float
    rmse: float
    crps: float
    coverage: float = Field(ge=0.0, le=1.0)
    coverage_region: AcceptanceRegion
    passed: bool
    bias: float


class OriginForecast(Spec):
    """One origin's forecast: the posterior-mean path, its interval, and what was observed.

    Grids are ``(n_units, horizon)`` as nested lists; ``periods`` are the
    numeric period labels of the horizon.
    """

    origin: int = Field(ge=1)
    periods: tuple[float, ...]
    units: tuple[str, ...]
    observed: tuple[tuple[float, ...], ...]
    mean: tuple[tuple[float, ...], ...]
    lower: tuple[tuple[float, ...], ...]
    upper: tuple[tuple[float, ...], ...]
    converged: bool


class OriginFailure(Spec):
    """An origin whose refit produced no posterior, with the backend's reason."""

    origin: int = Field(ge=1)
    reason: str


class Backtest(Spec):
    """A rolling-origin backtest: per-horizon scores, per-origin forecasts, and the failures."""

    spec_hash: str
    spec_name: str
    panel_hash: str
    origins: tuple[int, ...]
    horizon: int = Field(ge=1)
    n_units: int = Field(ge=1)
    mass: float = Field(gt=0.0, lt=1.0)
    definition: IntervalDefinition
    alpha: float = Field(gt=0.0, lt=1.0)
    scores: tuple[HorizonScore, ...]
    forecasts: tuple[OriginForecast, ...]
    failures: tuple[OriginFailure, ...] = ()
    n_failed_fits: int = Field(ge=0)
    backend: str
    draws: int = Field(ge=1)
    seed: int | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Backtest:
        if len(self.scores) != self.horizon:
            raise ValueError(f"expected one score per horizon step ({self.horizon})")
        if self.n_failed_fits != len(self.failures):
            raise ValueError("n_failed_fits must equal the number of recorded failures")
        if len(self.forecasts) + self.n_failed_fits != len(self.origins):
            raise ValueError("every origin is either a forecast or a recorded failure")
        return self

    @property
    def passed(self) -> bool:
        """Interval coverage inside its acceptance region at every horizon step."""
        return all(s.passed for s in self.scores)


def _interval_bounds(
    flat: Array, definition: IntervalDefinition, mass: float
) -> tuple[Array, Array]:
    """Per-cell interval bounds from ``(n_draws, n_units, h)`` draws."""
    n_units, h = flat.shape[1], flat.shape[2]
    lower = np.empty((n_units, h))
    upper = np.empty((n_units, h))
    for i in range(n_units):
        for j in range(h):
            iv: Interval = (
                eti(flat[:, i, j], mass) if definition == "eti" else hdi(flat[:, i, j], mass)
            )
            lower[i, j], upper[i, j] = iv.lower, iv.upper
    return lower, upper


def rolling_origin(
    spec: SurfaceSpec,
    panel: Panel,
    *,
    origins: Sequence[int],
    horizon: int,
    backend: str = "laplace",
    draws: int = 200,
    tune: int = 200,
    chains: int = 1,
    seed: int | None = 0,
    mass: float = 0.9,
    definition: IntervalDefinition = "eti",
    alpha: float = 0.01,
) -> Backtest | Unsupported:
    """Refit at each origin, forecast ``horizon`` periods through the one forward, and score.

    ``origins`` are numbers of training periods (each in ``[1, n_periods −
    horizon]``, distinct). Each origin's refit is seeded ``seed + origin``.
    The coverage criterion at each step is the exact binomial region for
    ``n = n_origins_ok × n_units`` cells at rate ``mass`` and level
    ``alpha``.
    """
    n_total = len(panel.periods)
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if not origins:
        raise ValueError("rolling_origin needs at least one origin")
    if len(set(origins)) != len(origins):
        raise ValueError(f"origins must be distinct: {list(origins)}")
    bad = [o for o in origins if not 1 <= o <= n_total - horizon]
    if bad:
        raise ValueError(
            f"origins {bad} are outside [1, {n_total - horizon}] for a panel of {n_total} "
            f"periods and horizon {horizon}"
        )
    outcome_column = panel.roles.outcome[0]
    observed_all = np.asarray(panel.array(outcome_column), dtype=np.float64)
    n_units = observed_all.shape[0]
    labels = spec.unit_labels or panel.units
    if spec.unit_labels:
        # ``prepare`` lays rows out in spec order; the panel grid is in sorted-unit order.
        order = [list(panel.units).index(u) for u in spec.unit_labels]
        observed_all = observed_all[order]
    periods_numeric = _numeric(panel.periods)

    point_err: list[Array] = []  # (n_units, horizon) per origin
    crps_cells: list[Array] = []
    covered: list[npt.NDArray[np.bool_]] = []
    forecasts: list[OriginForecast] = []
    failures: list[OriginFailure] = []
    for origin in origins:
        fit_seed = None if seed is None else seed + int(origin)
        train = training_panel(panel, origin)
        result = fit(
            spec, train, backend=backend, draws=draws, tune=tune, chains=chains, seed=fit_seed
        )
        if not isinstance(result.posterior, Posterior):
            reason = f"{type(result.posterior).__name__}: {result.posterior.reason}"
            log.warning("rolling_origin: origin %d refit failed: %s", origin, reason)
            failures.append(OriginFailure(origin=origin, reason=reason))
            continue
        conventions = result.provenance.get("nuisance_conventions")
        mean_draws = forecast(
            spec,
            result.posterior,
            panel,
            origin=origin,
            horizon=horizon,
            conventions=conventions,
            seed=fit_seed,
            noise=False,
        )
        noisy_draws = forecast(
            spec,
            result.posterior,
            panel,
            origin=origin,
            horizon=horizon,
            conventions=conventions,
            seed=fit_seed,
            noise=True,
        )
        mean_flat = mean_draws.values.reshape(-1, n_units, horizon)
        noisy_flat = noisy_draws.values.reshape(-1, n_units, horizon)
        mean_path = mean_flat.mean(axis=0)
        observed = observed_all[:, origin : origin + horizon]
        lower, upper = _interval_bounds(noisy_flat, definition, mass)
        point_err.append(mean_path - observed)
        crps_cells.append(crps(noisy_flat, observed))
        covered.append((observed >= lower) & (observed <= upper))
        forecasts.append(
            OriginForecast(
                origin=origin,
                periods=tuple(float(p) for p in periods_numeric[origin : origin + horizon]),
                units=tuple(labels),
                observed=_rows(observed),
                mean=_rows(mean_path),
                lower=_rows(lower),
                upper=_rows(upper),
                converged=result.converged,
            )
        )
    if not forecasts:
        return Unsupported(
            reason=f"rolling_origin: every one of {len(origins)} refits failed",
            detail={"n_origins": str(len(origins)), "n_failed": str(len(failures))},
        )
    err = np.stack(point_err)  # (n_ok, n_units, horizon)
    crps_all = np.stack(crps_cells)
    cov_all = np.stack(covered)
    scores: list[HorizonScore] = []
    for h in range(horizon):
        e = err[:, :, h].ravel()
        c = cov_all[:, :, h].ravel()
        n = int(e.size)
        region = clopper_pearson(n, mass, alpha)
        k = int(c.sum())
        scores.append(
            HorizonScore(
                step=h + 1,
                n=n,
                mae=float(np.mean(np.abs(e))),
                rmse=float(np.sqrt(np.mean(e**2))),
                crps=float(np.mean(crps_all[:, :, h])),
                coverage=k / n,
                coverage_region=region,
                passed=region.accepts(k),
                bias=float(np.mean(e)),
            )
        )
    return Backtest(
        spec_hash=spec.content_hash(),
        spec_name=spec.name,
        panel_hash=panel.content_hash(),
        origins=tuple(int(o) for o in origins),
        horizon=horizon,
        n_units=n_units,
        mass=mass,
        definition=definition,
        alpha=alpha,
        scores=tuple(scores),
        forecasts=tuple(forecasts),
        failures=tuple(failures),
        n_failed_fits=len(failures),
        backend=backend,
        draws=draws,
        seed=seed,
    )


def _rows(grid: Array) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(x) for x in row) for row in np.asarray(grid, dtype=np.float64))


def _numeric(periods: Sequence[object]) -> Array:
    series = pd.Series(list(periods))
    if pd.api.types.is_datetime64_any_dtype(series):
        epoch = pd.Timestamp("1970-01-01", tz=series.dt.tz)
        return np.asarray(((series - epoch) / pd.Timedelta(days=1)).to_numpy(), dtype=np.float64)
    return np.asarray(series.to_numpy(dtype=float), dtype=np.float64)


# -- freeze and replay ------------------------------------------------------------------------


def _draws_hash(posterior: Posterior) -> str:
    h = hashlib.blake2b(digest_size=32)
    for name in sorted(posterior.names()):
        a = np.ascontiguousarray(posterior.draws(name), dtype=np.float64)
        h.update(name.encode("utf-8"))
        h.update(str(a.shape).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class FrozenPredictor:
    """A fit's spec, posterior, and nuisance conventions, frozen for replay on new panels.

    ``predict(panel)`` is ``prepare`` + ``surface.forward.predict`` — the
    same forward the fit used — and returns ``Unsupported`` (never a
    number) when the panel does not carry the spec's units. ``content_hash``
    covers the spec and every posterior draw, so two predictors that give
    different numbers have different hashes.
    """

    spec: SurfaceSpec
    posterior: Posterior
    conventions: dict[str, dict[str, float]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        payload = f"{self.spec.content_hash()}:{_draws_hash(self.posterior)}"
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=32).hexdigest()

    @property
    def surface(self) -> Surface:
        return Surface(self.spec)

    def predict(
        self, panel: Panel, *, seed: int | None = None, noise: bool = False
    ) -> PredictiveDraws | Unsupported:
        """Replay the frozen posterior on ``panel``: ``values`` is ``(chain, draw, n_units,
        n_periods)`` of the mean outcome (``noise=True``: the outcome-scale predictive).

        The panel must be balanced, carry the outcome column (read for
        layout only), and hold exactly the spec's units when the spec
        indexes them; a unit mismatch is ``Unsupported`` naming the missing
        and extra units. Other layout violations raise as ``prepare`` does.
        """
        if self.spec.unit_labels:
            have, want = set(panel.units), set(self.spec.unit_labels)
            if have != want:
                return Unsupported(
                    reason=(
                        f"panel units do not match the frozen spec's: missing "
                        f"{sorted(want - have)}, unexpected {sorted(have - want)}"
                    ),
                    missing=("units",),
                    detail={"spec": self.spec.name, "predictor": self.content_hash},
                )
        data = prepare(self.spec, panel, conventions=self.conventions or None)
        draws = predict(self.surface, self.posterior, data, seed=seed, noise=noise)
        n_units = int(np.asarray(data[self.spec.treatment_names[0]]).shape[0])
        n_periods = int(np.asarray(data[self.spec.treatment_names[0]]).shape[1])
        values = np.asarray(draws.values, dtype=np.float64)
        grid = np.broadcast_to(values, (values.shape[0], values.shape[1], n_units, n_periods))
        labels = self.spec.unit_labels or panel.units
        coords: dict[str, list[Any]] = {
            "unit": list(labels),
            "period": list(panel.periods),
            **{k: list(v) for k, v in draws.coords.items()},
        }
        return PredictiveDraws(
            values=np.ascontiguousarray(grid),
            intervention=draws.intervention,
            window=None,
            coords=coords,
            seed=seed,
        )


def freeze(result: FitResult, *, panel: Panel | None = None) -> FrozenPredictor | Unsupported:
    """Freeze a ``FitResult``'s spec, posterior, and conventions; ``Unsupported`` without a
    posterior.

    ``panel`` is only consulted when the fit did not record nuisance
    conventions (it always does); then they are resolved against it.
    """
    if not isinstance(result.posterior, Posterior):
        return Unsupported(
            reason=(
                f"cannot freeze a fit without a posterior: "
                f"{type(result.posterior).__name__}({result.posterior.reason!r})"
            ),
            missing=("posterior",),
            detail={"spec": result.surface.spec.name},
        )
    spec = result.surface.spec
    recorded: Mapping[str, Mapping[str, float]] | None = result.provenance.get(
        "nuisance_conventions"
    )
    if recorded is None and panel is not None:
        recorded = resolve_conventions(spec, panel)
    conventions = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in (recorded or {}).items()}
    provenance = {
        "spec_hash": spec.content_hash(),
        "model_hash": result.provenance.get("model_hash"),
        "panel_hash": result.provenance.get("panel_hash"),
        "backend": result.provenance.get("backend"),
        "seed": result.provenance.get("seed"),
        "n_draws": result.posterior.n_draws(),
    }
    return FrozenPredictor(
        spec=spec, posterior=result.posterior, conventions=conventions, provenance=provenance
    )
