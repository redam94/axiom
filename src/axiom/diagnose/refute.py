"""Fit-based refutation checks: placebo treatment, permutation, random subset, added noise.

Extracted from the parent's ``validation/validator.py`` (ledger row
"diagnose/refute"), keeping only the four checks that refit the model and
none of the report scaffolding. Each check perturbs the *data* the fit saw,
refits through ``axiom.surface.fit`` (one forward, rule 3), realizes the
same estimand through ``axiom.estimands.realize``, and compares the refuted
distribution with the original estimate under a rule the ``Refutation``
states in words:

* ``placebo_treatment`` — the treatment's dose *series* is reassigned across
  units (unit ``i`` receives unit ``π(i)``'s doses, the outcome stays). A
  real effect must vanish. Because an amplitude prior is typically
  one-signed (a Hill ``beta`` is positive), a contrast's posterior cannot
  straddle zero and "the placebo interval contains zero" is not a rule it
  could ever pass; the rule is instead that the placebo distribution does
  not reach the original: ``p_value`` is the share of the pooled placebo
  posterior draws at least as large in magnitude as the original estimate,
  and the check passes when ``p_value < alpha`` (``alpha = 1 − mass``).
  ``detail["placebo_ratio"]`` reports ``refuted_mean / original``.
* ``permutation`` — the treatment's doses are shuffled across every (unit,
  period) cell, ``n`` times; the posterior means of the refits form the
  null distribution of the estimand. ``p_value = (1 + #{|null| ≥
  |original|}) / (n + 1)`` (Phipson & Smyth 2010) and the rule is
  ``p_value < alpha``: the original estimate is not something the null
  produces.
* ``random_subset`` — the model is refitted on ``n`` random subsets holding
  ``fraction`` of the units; ``added_noise`` — ``n`` refits with Gaussian
  noise of sd ``sd_fraction × σ̂`` (the posterior mean of the likelihood
  scale, else the outcome's sd) added to the outcome. For both the refuted
  distribution is the refit point estimates; ``p_value`` is the two-sided
  empirical location of the original among them, ``2·min(F(orig), 1 −
  F(orig))``, and the rule is ``p_value ≥ alpha``: the original is not in
  the tails of what perturbed data give. With ``n`` refits the resolution
  of this p-value is ``1 / n``, which ``n`` states.

A refit whose backend returned a typed failure, or whose estimand could not
be realized, is counted in ``n_failed`` and logged; it is never dropped
silently, and a check with no successful refit is an ``Unsupported``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, model_validator

from axiom.core import Interval, NonEmptyStr, Posterior, Spec, Unsupported, eti, hdi
from axiom.core.intervals import IntervalDefinition
from axiom.data import Panel
from axiom.diagnose.spec_curve import default_estimand, realized_point
from axiom.estimands import Estimand, RealizedDraws
from axiom.surface import FitResult, SurfaceSpec, fit

__all__ = [
    "RefutationKind",
    "Refutation",
    "added_noise",
    "permutation",
    "placebo_treatment",
    "random_subset",
]

log = logging.getLogger(__name__)

Array = npt.NDArray[np.float64]
RefutationKind = Literal["placebo_treatment", "permutation", "random_subset", "added_noise"]


class Refutation(Spec):
    """The outcome of one refutation check, with the rule it passed or failed under.

    ``original`` / ``original_interval`` are the estimand on the real fit;
    ``refuted_*`` summarize the refuted distribution (placebo: pooled
    posterior draws; the others: refit point estimates). ``n`` is the size
    of that distribution; ``n_refits`` the refits attempted and ``n_failed``
    how many of them produced no estimate. ``p_value`` is defined per check
    in the module docstring. ``rule`` states, in words, what ``passed``
    means.
    """

    kind: RefutationKind
    estimand_name: str
    estimand_hash: str
    treatment: str
    original: float
    original_interval: Interval
    refuted_mean: float
    refuted_sd: float
    refuted_interval: Interval
    refuted_estimates: tuple[float, ...] = ()
    p_value: float = Field(ge=0.0, le=1.0)
    n: int = Field(ge=1)
    n_refits: int = Field(ge=1)
    n_failed: int = Field(ge=0)
    alpha: float = Field(gt=0.0, lt=1.0)
    rule: NonEmptyStr
    passed: bool
    seed: int | None = None
    detail: dict[str, float | str] = {}

    @model_validator(mode="after")
    def _consistent(self) -> Refutation:
        if self.n_failed > self.n_refits:
            raise ValueError("n_failed cannot exceed n_refits")
        if self.n_failed == self.n_refits:
            raise ValueError("a Refutation needs at least one successful refit")
        return self


# -- shared machinery ---------------------------------------------------------------------


def _interval(draws: Array, definition: IntervalDefinition, mass: float) -> Interval:
    return eti(draws, mass) if definition == "eti" else hdi(draws, mass)


def _require_posterior(result: FitResult) -> Posterior:
    if not isinstance(result.posterior, Posterior):
        raise ValueError(
            f"the fit has no posterior: {type(result.posterior).__name__}"
            f"({result.posterior.reason!r}); refutation needs a fitted model"
        )
    return result.posterior


def _treatment(result: FitResult, treatment: str | None) -> str:
    names = result.surface.spec.treatment_names
    if treatment is None:
        return names[0]
    if treatment not in names:
        raise ValueError(f"no treatment {treatment!r}; have {list(names)}")
    return treatment


def _dose_column(panel: Panel, treatment: str) -> str:
    """The panel column storing a treatment (by column name, else by entity name)."""
    roles = panel.roles
    if treatment in roles.treatments:
        return treatment
    for column, entity in roles.treatments.items():
        if entity.name == treatment:
            return column
    raise KeyError(f"panel has no column for treatment {treatment!r}")


def _with_grid(panel: Panel, column: str, grid: Array) -> Panel:
    """``panel`` with one measured column replaced by an ``(n_units, n_periods)`` grid."""
    frame = panel.frame
    roles = panel.roles
    units, periods = list(panel.units), list(panel.periods)
    lookup = pd.DataFrame(grid, index=units, columns=periods).stack()
    keys = list(zip(frame[roles.unit].astype(str), frame[roles.time], strict=True))
    frame[column] = np.asarray([lookup[k] for k in keys], dtype=np.float64)
    return Panel(frame, roles)


def _original(
    result: FitResult, estimand: Estimand, definition: IntervalDefinition, mass: float, seed: int
) -> RealizedDraws:
    point = realized_point(estimand, result, definition=definition, mass=mass, seed=seed)
    if isinstance(point, Unsupported):
        raise ValueError(f"the original estimand could not be realized: {point.reason}")
    return point


def _refit(
    spec: SurfaceSpec,
    panel: Panel,
    estimand: Estimand,
    *,
    backend: str,
    draws: int,
    chains: int,
    seed: int,
    definition: IntervalDefinition,
    mass: float,
) -> RealizedDraws | Unsupported:
    result = fit(spec, panel, backend=backend, draws=draws, chains=chains, seed=seed)
    return realized_point(estimand, result, definition=definition, mass=mass, seed=seed)


def _points_from_refits(
    what: str, outcomes: Sequence[RealizedDraws | Unsupported]
) -> tuple[Array, int] | Unsupported:
    """Posterior-mean point per successful refit and the count of failures."""
    points: list[float] = []
    n_failed = 0
    for i, out in enumerate(outcomes):
        if isinstance(out, Unsupported):
            n_failed += 1
            log.warning("%s: refit %d failed: %s", what, i, out.reason)
            continue
        points.append(out.result.summary.mean)
    if not points:
        return Unsupported(
            reason=f"{what}: every one of {len(outcomes)} refits failed",
            detail={"n_refits": str(len(outcomes)), "n_failed": str(n_failed)},
        )
    return np.asarray(points, dtype=np.float64), n_failed


def _location_p(points: Array, original: float) -> float:
    """Two-sided empirical location of ``original`` among ``points``: ``2·min(F, 1−F)``."""
    below = float(np.mean(points <= original))
    above = float(np.mean(points >= original))
    return float(min(1.0, 2.0 * min(below, above)))


# -- placebo -------------------------------------------------------------------------------


def placebo_treatment(
    result: FitResult,
    *,
    estimand: Estimand | None = None,
    treatment: str | None = None,
    seed: int = 0,
    n: int = 1,
    backend: str = "laplace",
    draws: int = 200,
    chains: int = 1,
    definition: IntervalDefinition = "eti",
    mass: float = 0.9,
    panel: Panel | None = None,
) -> Refutation | Unsupported:
    """Reassign the treatment's dose series across units, refit, and expect no effect.

    The panel needs at least two units (``ValueError`` otherwise). ``n``
    independent reassignments are refitted and their posterior draws pooled
    into the refuted distribution. The rule: fewer than ``alpha = 1 − mass``
    of the pooled placebo draws are at least as large in magnitude as the
    original estimate (the placebo does not reach the real effect).
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    _require_posterior(result)
    spec = result.surface.spec
    name = _treatment(result, treatment)
    panel = panel if panel is not None else _panel_of(result)
    target = estimand if estimand is not None else default_estimand(spec, panel, name)
    original = _original(result, target, definition, mass, seed)
    n_units = len(panel.units)
    if n_units < 2:
        raise ValueError("placebo_treatment needs at least two units to reassign doses across")
    rng = np.random.default_rng(seed)
    column = _dose_column(panel, name)
    grid = np.asarray(panel.array(column), dtype=np.float64)
    outcomes: list[RealizedDraws | Unsupported] = []
    for i in range(n):
        perm = rng.permutation(n_units)
        # Ensure the placebo is a genuine reassignment: no unit keeps its own series.
        while n_units > 1 and np.any(perm == np.arange(n_units)):
            perm = rng.permutation(n_units)
        placebo = _with_grid(panel, column, grid[perm])
        outcomes.append(
            _refit(
                spec,
                placebo,
                target,
                backend=backend,
                draws=draws,
                chains=chains,
                seed=seed + i,
                definition=definition,
                mass=mass,
            )
        )
    pooled: list[Array] = []
    n_failed = 0
    estimates: list[float] = []
    for i, out in enumerate(outcomes):
        if isinstance(out, Unsupported):
            n_failed += 1
            log.warning("placebo_treatment: refit %d failed: %s", i, out.reason)
            continue
        pooled.append(np.asarray(out.draws, dtype=np.float64).ravel())
        estimates.append(out.result.summary.mean)
    if not pooled:
        return Unsupported(
            reason=f"placebo_treatment: every one of {n} refits failed",
            detail={"n_refits": str(n), "n_failed": str(n_failed)},
        )
    pooled_draws = np.concatenate(pooled)
    interval = _interval(pooled_draws, definition, mass)
    orig = original.result.summary.mean
    alpha = 1.0 - mass
    p_value = float(np.mean(np.abs(pooled_draws) >= abs(orig)))
    passed = bool(p_value < alpha)
    ratio = float(pooled_draws.mean() / orig) if orig != 0.0 else float("nan")
    return Refutation(
        kind="placebo_treatment",
        estimand_name=target.name,
        estimand_hash=target.content_hash(),
        treatment=name,
        original=orig,
        original_interval=original.result.summary.interval,
        refuted_mean=float(pooled_draws.mean()),
        refuted_sd=float(pooled_draws.std(ddof=1)) if pooled_draws.size > 1 else 0.0,
        refuted_interval=interval,
        refuted_estimates=tuple(estimates),
        p_value=p_value,
        n=int(pooled_draws.size),
        n_refits=n,
        n_failed=n_failed,
        alpha=alpha,
        rule=(
            f"fewer than {alpha:.0%} of the N = {pooled_draws.size} pooled posterior draws of "
            f"the placebo estimand (treatment {name!r} reassigned across units, {n} refit(s)) "
            "are at least as large in magnitude as the original estimate"
        ),
        passed=passed,
        seed=seed,
        detail={
            "n_units": float(n_units),
            "placebo_ratio": ratio if np.isfinite(ratio) else "nan",
            "backend": backend,
            "draws": float(draws),
        },
    )


# -- permutation ---------------------------------------------------------------------------


def permutation(
    result: FitResult,
    n: int = 20,
    seed: int = 0,
    *,
    estimand: Estimand | None = None,
    treatment: str | None = None,
    alpha: float = 0.05,
    backend: str = "laplace",
    draws: int = 200,
    chains: int = 1,
    definition: IntervalDefinition = "eti",
    mass: float = 0.9,
    panel: Panel | None = None,
) -> Refutation | Unsupported:
    """Null distribution of the estimand from ``n`` refits with the doses shuffled across cells.

    The rule: ``p_value < alpha`` with ``p_value = (1 + #{|null| ≥
    |original|}) / (n + 1)``, where the count is over successful refits.
    The smallest attainable p-value is ``1 / (n + 1)``, so ``n`` must be at
    least ``1 / alpha − 1`` for the check to be able to pass
    (``ValueError`` otherwise).
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if 1.0 / (n + 1) >= alpha:
        raise ValueError(
            f"n={n} permutations cannot reach p < {alpha}: the smallest p-value is "
            f"1/(n+1) = {1.0 / (n + 1):.3g}; use n >= {int(np.ceil(1.0 / alpha)) }"
        )
    _require_posterior(result)
    spec = result.surface.spec
    name = _treatment(result, treatment)
    panel = panel if panel is not None else _panel_of(result)
    target = estimand if estimand is not None else default_estimand(spec, panel, name)
    original = _original(result, target, definition, mass, seed)
    rng = np.random.default_rng(seed)
    column = _dose_column(panel, name)
    grid = np.asarray(panel.array(column), dtype=np.float64)
    outcomes: list[RealizedDraws | Unsupported] = []
    for i in range(n):
        shuffled = rng.permutation(grid.ravel()).reshape(grid.shape)
        outcomes.append(
            _refit(
                spec,
                _with_grid(panel, column, shuffled),
                target,
                backend=backend,
                draws=draws,
                chains=chains,
                seed=seed + i,
                definition=definition,
                mass=mass,
            )
        )
    got = _points_from_refits("permutation", outcomes)
    if isinstance(got, Unsupported):
        return got
    null, n_failed = got
    orig = original.result.summary.mean
    extreme = int(np.sum(np.abs(null) >= abs(orig)))
    p_value = (1.0 + extreme) / (null.size + 1.0)
    return Refutation(
        kind="permutation",
        estimand_name=target.name,
        estimand_hash=target.content_hash(),
        treatment=name,
        original=orig,
        original_interval=original.result.summary.interval,
        refuted_mean=float(null.mean()),
        refuted_sd=float(null.std(ddof=1)) if null.size > 1 else 0.0,
        refuted_interval=_interval(null, definition, mass),
        refuted_estimates=tuple(float(x) for x in null),
        p_value=float(p_value),
        n=int(null.size),
        n_refits=n,
        n_failed=n_failed,
        alpha=alpha,
        rule=(
            f"permutation p-value (1 + #{{|null| >= |original|}}) / (N + 1) over N = "
            f"{null.size} refits with the doses of {name!r} shuffled across cells is "
            f"below {alpha:g}"
        ),
        passed=bool(p_value < alpha),
        seed=seed,
        detail={"n_extreme": float(extreme), "backend": backend, "draws": float(draws)},
    )


# -- random subset and added noise ---------------------------------------------------------


def _stability(
    kind: RefutationKind,
    result: FitResult,
    target: Estimand,
    name: str,
    outcomes: Sequence[RealizedDraws | Unsupported],
    original: RealizedDraws,
    *,
    n_refits: int,
    alpha: float,
    definition: IntervalDefinition,
    mass: float,
    seed: int,
    rule: str,
    detail: dict[str, float | str],
) -> Refutation | Unsupported:
    got = _points_from_refits(kind, outcomes)
    if isinstance(got, Unsupported):
        return got
    points, n_failed = got
    orig = original.result.summary.mean
    p_value = _location_p(points, orig)
    return Refutation(
        kind=kind,
        estimand_name=target.name,
        estimand_hash=target.content_hash(),
        treatment=name,
        original=orig,
        original_interval=original.result.summary.interval,
        refuted_mean=float(points.mean()),
        refuted_sd=float(points.std(ddof=1)) if points.size > 1 else 0.0,
        refuted_interval=_interval(points, definition, mass),
        refuted_estimates=tuple(float(x) for x in points),
        p_value=p_value,
        n=int(points.size),
        n_refits=n_refits,
        n_failed=n_failed,
        alpha=alpha,
        rule=rule,
        passed=bool(p_value >= alpha),
        seed=seed,
        detail=detail,
    )


def random_subset(
    result: FitResult,
    fraction: float = 0.7,
    n: int = 10,
    seed: int = 0,
    *,
    estimand: Estimand | None = None,
    treatment: str | None = None,
    alpha: float = 0.1,
    backend: str = "laplace",
    draws: int = 200,
    chains: int = 1,
    definition: IntervalDefinition = "eti",
    mass: float = 0.9,
    panel: Panel | None = None,
) -> Refutation | Unsupported:
    """Refit on ``n`` random subsets holding ``fraction`` of the units; expect stability.

    Each subset keeps ``max(2, round(fraction · n_units))`` units and the
    spec's ``unit_labels`` are restricted to them (intercepts that index
    units are re-declared for the subset). The rule: the original estimate
    lies within the central ``1 − alpha`` of the subset estimates
    (``p_value ≥ alpha``). ``n`` refits give a p-value resolution of ``1/n``.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    _require_posterior(result)
    spec = result.surface.spec
    name = _treatment(result, treatment)
    panel = panel if panel is not None else _panel_of(result)
    target = estimand if estimand is not None else default_estimand(spec, panel, name)
    original = _original(result, target, definition, mass, seed)
    units = list(panel.units)
    if len(units) < 3:
        raise ValueError("random_subset needs at least three units")
    keep = max(2, int(round(fraction * len(units))))
    if keep >= len(units):
        raise ValueError(f"fraction={fraction} keeps every one of the {len(units)} units; lower it")
    rng = np.random.default_rng(seed)
    outcomes: list[RealizedDraws | Unsupported] = []
    for i in range(n):
        chosen = sorted(str(u) for u in rng.choice(units, size=keep, replace=False))
        sub_panel = panel.select_units(chosen)
        sub_spec = (
            SurfaceSpec.model_validate({**spec.model_dump(), "unit_labels": tuple(chosen)})
            if spec.unit_labels
            else spec
        )
        outcomes.append(
            _refit(
                sub_spec,
                sub_panel,
                target,
                backend=backend,
                draws=draws,
                chains=chains,
                seed=seed + i,
                definition=definition,
                mass=mass,
            )
        )
    return _stability(
        "random_subset",
        result,
        target,
        name,
        outcomes,
        original,
        n_refits=n,
        alpha=alpha,
        definition=definition,
        mass=mass,
        seed=seed,
        rule=(
            f"the original estimate lies within the central {1 - alpha:.0%} of the estimates "
            f"from {n} refits on random subsets of {keep} of {len(units)} units "
            f"(two-sided empirical p >= {alpha:g}; resolution 1/{n})"
        ),
        detail={
            "fraction": fraction,
            "units_kept": float(keep),
            "n_units": float(len(units)),
            "backend": backend,
            "draws": float(draws),
        },
    )


def added_noise(
    result: FitResult,
    sd_fraction: float = 0.5,
    n: int = 10,
    seed: int = 0,
    *,
    estimand: Estimand | None = None,
    treatment: str | None = None,
    alpha: float = 0.1,
    backend: str = "laplace",
    draws: int = 200,
    chains: int = 1,
    definition: IntervalDefinition = "eti",
    mass: float = 0.9,
    panel: Panel | None = None,
) -> Refutation | Unsupported:
    """Refit ``n`` times with Gaussian noise of sd ``sd_fraction · σ̂`` added to the outcome.

    ``σ̂`` is the posterior mean of the likelihood scale when the spec has
    one, else the sd of the observed outcome; it is reported in ``detail``.
    The rule is the same as ``random_subset``'s: the original estimate lies
    within the central ``1 − alpha`` of the noisy-refit estimates.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if not (np.isfinite(sd_fraction) and sd_fraction > 0.0):
        raise ValueError(f"sd_fraction must be positive, got {sd_fraction}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    posterior = _require_posterior(result)
    spec = result.surface.spec
    name = _treatment(result, treatment)
    panel = panel if panel is not None else _panel_of(result)
    target = estimand if estimand is not None else default_estimand(spec, panel, name)
    original = _original(result, target, definition, mass, seed)
    outcome_column = panel.roles.outcome[0]
    observed = np.asarray(panel.array(outcome_column), dtype=np.float64)
    scale_name = spec.likelihood.scale
    if scale_name is not None and scale_name in posterior.names():
        sigma_hat = float(np.mean(posterior.flat(scale_name)))
        source = f"posterior mean of {scale_name!r}"
    else:
        sigma_hat = float(observed.std(ddof=1))
        source = "sd of the observed outcome"
    noise_sd = sd_fraction * sigma_hat
    rng = np.random.default_rng(seed)
    outcomes: list[RealizedDraws | Unsupported] = []
    for i in range(n):
        noisy = observed + rng.normal(0.0, noise_sd, size=observed.shape)
        outcomes.append(
            _refit(
                spec,
                _with_grid(panel, outcome_column, noisy),
                target,
                backend=backend,
                draws=draws,
                chains=chains,
                seed=seed + i,
                definition=definition,
                mass=mass,
            )
        )
    return _stability(
        "added_noise",
        result,
        target,
        name,
        outcomes,
        original,
        n_refits=n,
        alpha=alpha,
        definition=definition,
        mass=mass,
        seed=seed,
        rule=(
            f"the original estimate lies within the central {1 - alpha:.0%} of the estimates "
            f"from {n} refits with N(0, ({sd_fraction:g} x {sigma_hat:.4g})^2) noise added to "
            f"the outcome (two-sided empirical p >= {alpha:g}; resolution 1/{n})"
        ),
        detail={
            "sd_fraction": sd_fraction,
            "sigma_hat": sigma_hat,
            "sigma_source": source,
            "noise_sd": noise_sd,
            "backend": backend,
            "draws": float(draws),
        },
    )


# -- the fitted panel ------------------------------------------------------------------------


def _panel_of(result: FitResult) -> Panel:
    """The panel the fit saw, rebuilt from the fitted data dict (when none is passed in).

    ``FitResult`` keeps the prepared arrays, not the ``Panel``; the checks
    need a ``Panel`` to perturb and refit, so it is reassembled from the
    dose grids, the outcome grid, the unit labels, and the numeric periods
    with the spec's entities as roles. Nuisance basis columns are not
    panel columns (``prepare`` derives them) and the conventions are
    re-resolved on the rebuilt panel — identical for a full-horizon panel.
    """
    from axiom.sim import panel_from_arrays

    spec = result.surface.spec
    data = result.data
    labels = list(result.unit_labels)
    periods_numeric = result.periods
    order = np.argsort(np.asarray(labels, dtype=str), kind="stable")
    sorted_labels = [labels[i] for i in order]
    periods: list[int] | None
    if all(float(p).is_integer() for p in periods_numeric):
        periods = [int(p) for p in periods_numeric]
    else:
        periods = None
    doses = {t.name: np.asarray(data[t.name], dtype=np.float64)[order] for t in spec.treatments}
    outcome = np.asarray(data[spec.outcome.name], dtype=np.float64)[order]
    return panel_from_arrays(
        doses=doses,
        outcome=outcome,
        treatments=spec.treatments,
        outcome_entity=spec.outcome,
        units=sorted_labels,
        periods=periods,
        unit_column=spec.unit_column,
        time_column=spec.time_column,
    )
