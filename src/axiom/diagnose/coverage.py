"""Fixed-truth recovery coverage: does a nominal ``mass`` interval cover the truth ``mass`` of
the time?

Over ``n`` replications of a world with known parameters, fit the model,
take each parameter's ``mass`` interval, and count how often it contains
the true value. If the estimator is calibrated the count is
``Binomial(n, mass)``, so the criterion is the exact two-sided
Clopper–Pearson acceptance region ``core.clopper_pearson(n, mass, alpha)``
— stated with its ``n`` and ``alpha`` on every result, never a bare
fraction (rule 4). This is the roadmap's Phase 8 exit criterion 2: nominal
90 % intervals cover 88–92 % on ``sim`` worlds, and a mis-specified world
fails the same check.

``coverage`` replicates over *worlds* — the factory is called with a
fresh seed each time, so the doses, the non-structural truth, and the
noise all vary — and fits each world's ``spec`` on its ``panel``. A
mis-specified check is a factory that returns a ``WorldView`` whose
``spec`` is *not* the one that generated the panel (``misspecify``):
truth generated with carryover and fitted without, say. The truth the
intervals are scored against is always the generating truth.

``estimand_coverage`` scores realized estimands instead of parameters:
each is realized on the fit and on a **two-draw point posterior at the
truth** (``truth_producer``) through the same ``estimands.realize`` path,
so the true estimand is computed by the code under test and nothing else.

Ported by specification from the parent's ``diagnostics/coverage.py``
(ledger row ``diagnose/coverage.py``, PORT); the SBC half of that module
lives in ``diagnose.sbc``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from axiom.core import (
    AcceptanceRegion,
    Blocked,
    Posterior,
    Spec,
    Unsupported,
    clopper_pearson,
    interval,
)
from axiom.core.intervals import IntervalDefinition
from axiom.data import Panel
from axiom.estimands import Estimand, EstimandResult, realize
from axiom.surface import FitResult, Surface, SurfaceSpec, fit, prepare

__all__ = [
    "CoverageResult",
    "EstimandCoverage",
    "EstimandCoverageResult",
    "Fitter",
    "ParameterCoverage",
    "SupportsWorld",
    "WorldFactory",
    "WorldView",
    "coverage",
    "estimand_coverage",
    "misspecify",
    "truth_producer",
]

Array = npt.NDArray[np.float64]


@runtime_checkable
class SupportsWorld(Protocol):
    """What a replication needs: the spec to fit, the panel to fit it on, the truth to score
    against. ``axiom.sim.SurfaceWorld`` satisfies it; so does ``WorldView``."""

    @property
    def spec(self) -> SurfaceSpec: ...

    @property
    def panel(self) -> Panel: ...

    @property
    def theta(self) -> Mapping[str, Array]: ...


@dataclass(frozen=True)
class WorldView:
    """A (spec, panel, theta) triple — a world seen through a possibly different spec."""

    spec: SurfaceSpec
    panel: Panel
    theta: dict[str, Array]


def misspecify(world: SupportsWorld, spec: SurfaceSpec) -> WorldView:
    """``world``'s panel and truth, to be fitted with ``spec`` instead of the generating spec.

    The truth keeps every generating parameter; only those ``spec`` also
    declares are scored. ``ValueError`` when ``spec`` shares no parameter
    name with the generating model — then there is nothing to score.
    """
    fitted = {p.name for p in Surface(spec).model.free}
    if not fitted & set(world.theta):
        raise ValueError(
            "the mis-specified spec shares no parameter name with the generating truth; "
            "nothing could be scored"
        )
    return WorldView(spec=spec, panel=world.panel, theta=dict(world.theta))


WorldFactory = Callable[[int], SupportsWorld]
"""``seed -> world``: one independent replication per seed."""
Fitter = Callable[[SupportsWorld, int], FitResult]
"""``(world, seed) -> FitResult``: how each replication is fitted."""


def _default_fitter(backend: str, draws: int) -> Fitter:
    def run(world: SupportsWorld, seed: int) -> FitResult:
        return fit(world.spec, world.panel, backend=backend, draws=draws, chains=1, seed=seed)

    return run


# -- results ----------------------------------------------------------------------------------


class ParameterCoverage(Spec):
    """Coverage of one (scalar slice of a) parameter over ``n`` fitted replications.

    ``covered`` of ``n`` intervals contained the truth; ``region`` is the
    exact binomial acceptance region at the nominal ``mass`` and the stated
    ``alpha``; ``passed`` is ``region.accepts(covered)``.
    """

    name: str
    n: int = Field(ge=1)
    covered: int = Field(ge=0)
    mass: float = Field(gt=0, lt=1)
    definition: IntervalDefinition
    region: AcceptanceRegion
    passed: bool

    @property
    def rate(self) -> float:
        return self.covered / self.n

    @model_validator(mode="after")
    def _consistent(self) -> ParameterCoverage:
        if self.covered > self.n:
            raise ValueError(f"{self.name}: covered {self.covered} exceeds n {self.n}")
        if self.region.n != self.n or self.region.p != self.mass:
            raise ValueError(f"{self.name}: the acceptance region must be for (n, mass)")
        if self.passed != self.region.accepts(self.covered):
            raise ValueError(f"{self.name}: passed must equal region.accepts(covered)")
        return self


class CoverageResult(Spec):
    """Per-parameter coverage plus the bookkeeping: ``n_fitted + n_failed_fits == n``.

    ``nominal_region`` is the acceptance region for the requested ``n``;
    each parameter carries the region for the ``n_fitted`` it was scored on.

    ``passed`` is true when every scored parameter's count fell inside its
    acceptance region; ``failed_parameters`` names the rest. A fit the
    backend declined is counted, with its reason, and does not contribute
    an interval.
    """

    n: int = Field(ge=1)
    mass: float = Field(gt=0, lt=1)
    definition: IntervalDefinition
    alpha: float = Field(gt=0, lt=1)
    n_fitted: int = Field(ge=0)
    n_failed_fits: int = Field(ge=0)
    failure_reasons: tuple[str, ...] = ()
    nominal_region: AcceptanceRegion
    parameters: tuple[ParameterCoverage, ...]
    failed_parameters: tuple[str, ...]
    passed: bool
    provenance: dict[str, Any] = {}

    @model_validator(mode="after")
    def _consistent(self) -> CoverageResult:
        if self.n_fitted + self.n_failed_fits != self.n:
            raise ValueError("n_fitted + n_failed_fits must equal n")
        if len(self.failure_reasons) != self.n_failed_fits:
            raise ValueError("one failure reason per failed fit")
        expected = tuple(p.name for p in self.parameters if not p.passed)
        if self.failed_parameters != expected:
            raise ValueError("failed_parameters must list exactly the parameters that failed")
        if self.passed != (not expected and self.n_fitted > 0):
            raise ValueError("passed must mean every parameter passed on at least one fit")
        return self


class EstimandCoverage(Spec):
    """Coverage of one estimand: ``covered`` of ``n`` realized intervals held the true value
    (realized at the truth through the same path). ``true_values`` is kept so a plot can show
    the spread of truths across replications."""

    name: str
    estimand_hash: str
    n: int = Field(ge=1)
    covered: int = Field(ge=0)
    mass: float = Field(gt=0, lt=1)
    definition: IntervalDefinition
    region: AcceptanceRegion
    passed: bool
    true_values: tuple[float, ...] = ()

    @property
    def rate(self) -> float:
        return self.covered / self.n

    @model_validator(mode="after")
    def _consistent(self) -> EstimandCoverage:
        if self.covered > self.n:
            raise ValueError(f"{self.name}: covered {self.covered} exceeds n {self.n}")
        if self.region.n != self.n or self.region.p != self.mass:
            raise ValueError(f"{self.name}: the acceptance region must be for (n, mass)")
        if self.passed != self.region.accepts(self.covered):
            raise ValueError(f"{self.name}: passed must equal region.accepts(covered)")
        if self.true_values and len(self.true_values) != self.n:
            raise ValueError(f"{self.name}: one true value per scored replication")
        return self


class EstimandCoverageResult(Spec):
    """Per-estimand coverage; ``n_fitted + n_failed_fits == n``. A replication on which an
    estimand could not be realized (``Unsupported`` / ``Blocked``) counts as a failed fit for
    that estimand and is named in ``failure_reasons``."""

    n: int = Field(ge=1)
    mass: float = Field(gt=0, lt=1)
    definition: IntervalDefinition
    alpha: float = Field(gt=0, lt=1)
    n_fitted: int = Field(ge=0)
    n_failed_fits: int = Field(ge=0)
    failure_reasons: tuple[str, ...] = ()
    nominal_region: AcceptanceRegion
    estimands: tuple[EstimandCoverage, ...]
    failed_estimands: tuple[str, ...]
    passed: bool
    provenance: dict[str, Any] = {}

    @model_validator(mode="after")
    def _consistent(self) -> EstimandCoverageResult:
        if self.n_fitted + self.n_failed_fits != self.n:
            raise ValueError("n_fitted + n_failed_fits must equal n")
        expected = tuple(e.name for e in self.estimands if not e.passed)
        if self.failed_estimands != expected:
            raise ValueError("failed_estimands must list exactly the estimands that failed")
        if self.passed != (not expected and self.n_fitted > 0):
            raise ValueError("passed must mean every estimand passed on at least one fit")
        return self


# -- parameter coverage -----------------------------------------------------------------------


def _slices(name: str, shape: tuple[int, ...]) -> tuple[str, ...]:
    """Labels of the scalar slices in C order: ``name`` for a scalar, ``name[i,j]`` per element."""
    if not shape:
        return (name,)
    return tuple(f"{name}[{','.join(str(i) for i in idx)}]" for idx in np.ndindex(*shape))


def _failure(result: FitResult, i: int) -> str | None:
    if isinstance(result.posterior, Posterior):
        return None
    return f"replication {i}: {type(result.posterior).__name__}: {result.posterior.reason}"


def coverage(
    world_factory: WorldFactory,
    *,
    n: int,
    mass: float = 0.9,
    definition: IntervalDefinition = "eti",
    parameters: Sequence[str] | None = None,
    fit: Fitter | None = None,
    backend: str = "laplace",
    draws: int = 200,
    seed: int = 0,
    alpha: float = 0.01,
) -> CoverageResult:
    """Fixed-truth coverage of the ``mass`` ``definition`` interval over ``n`` replications.

    Replication ``i`` is ``world_factory(seed + i)`` fitted by ``fit``
    (default: ``surface.fit`` with ``backend`` and ``draws``, one chain,
    seeded ``seed + i``). ``parameters`` restricts scoring (default: every
    free parameter of the fitted model that the truth names); a vector
    parameter is scored per element. ``ValueError`` for a named parameter
    the fitted model does not have, or one the truth does not provide.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if definition == "wald":
        raise ValueError("coverage scores posterior intervals; definition must be eti or hdi")
    region = clopper_pearson(n, mass, alpha)
    fitter = fit if fit is not None else _default_fitter(backend, draws)
    counts: dict[str, int] = {}
    reasons: list[str] = []
    names: tuple[str, ...] | None = None
    n_fitted = 0
    for i in range(n):
        world = world_factory(seed + i)
        result = fitter(world, seed + i)
        why = _failure(result, i)
        if why is not None:
            reasons.append(why)
            continue
        posterior = result.posterior
        assert isinstance(posterior, Posterior)
        if names is None:
            free = tuple(p.name for p in result.surface.model.free)
            if parameters is None:
                names = tuple(p for p in free if p in world.theta)
            else:
                missing = [p for p in parameters if p not in free]
                if missing:
                    raise ValueError(f"parameters {missing} are not free in the fitted model")
                absent = [p for p in parameters if p not in world.theta]
                if absent:
                    raise ValueError(f"the world's truth does not provide {absent}")
                names = tuple(parameters)
            if not names:
                raise ValueError("no parameter of the fitted model is named by the truth")
        n_fitted += 1
        for name in names:
            flat = np.asarray(posterior.flat(name), dtype=np.float64)
            columns = flat.reshape(flat.shape[0], -1)
            truths = np.asarray(world.theta[name], dtype=np.float64).reshape(-1)
            if truths.size != columns.shape[1]:
                raise ValueError(
                    f"truth for {name!r} has {truths.size} elements; the posterior has "
                    f"{columns.shape[1]}"
                )
            for j, label in enumerate(_slices(name, flat.shape[1:])):
                iv = interval(columns[:, j], definition=definition, mass=mass)
                counts[label] = counts.get(label, 0) + int(iv.contains(float(truths[j])))
    region_fitted = clopper_pearson(n_fitted, mass, alpha) if n_fitted else region
    scored = tuple(
        ParameterCoverage(
            name=label,
            n=n_fitted,
            covered=k,
            mass=mass,
            definition=definition,
            region=region_fitted,
            passed=region_fitted.accepts(k),
        )
        for label, k in counts.items()
    )
    failed = tuple(p.name for p in scored if not p.passed)
    return CoverageResult(
        n=n,
        mass=mass,
        definition=definition,
        alpha=alpha,
        n_fitted=n_fitted,
        n_failed_fits=len(reasons),
        failure_reasons=tuple(reasons),
        nominal_region=region,
        parameters=scored,
        failed_parameters=failed,
        passed=bool(not failed and n_fitted > 0),
        provenance={"seed": seed, "backend": backend, "draws": draws},
    )


# -- estimand coverage ------------------------------------------------------------------------


def truth_producer(world: SupportsWorld) -> FitResult:
    """A ``FitResult`` whose posterior is a two-draw point mass at the world's truth, so an
    estimand realized on it is the *true* estimand computed through ``estimands.realize``.
    The generating spec is used, not a mis-specified view's."""
    surface = Surface(world.spec)
    model = surface.model
    draws = {
        p.name: np.broadcast_to(
            np.asarray(world.theta[p.name], dtype=np.float64), (1, 2, *p.shape)
        ).copy()
        for p in model.free
        if p.name in world.theta
    }
    missing = [p.name for p in model.free if p.name not in world.theta]
    if missing:
        raise ValueError(f"the world's truth does not provide {missing}")
    posterior = Posterior(
        draws,
        provenance={"method": "truth", "model_hash": model.content_hash(), "converged": True},
    )
    data = prepare(world.spec, world.panel)
    return FitResult(
        surface,
        posterior,
        data,
        None,
        {
            "spec_hash": world.spec.content_hash(),
            "model_hash": model.content_hash(),
            "panel_hash": world.panel.content_hash(),
            "unit_labels": list(world.spec.unit_labels or world.panel.units),
            "backend": "truth",
        },
    )


def _realized(
    estimand: Estimand, producer: FitResult, definition: IntervalDefinition, mass: float
) -> EstimandResult | Unsupported | Blocked:
    out = realize(estimand, producer, assume_identified=True, definition=definition, mass=mass)
    if isinstance(out, EstimandResult | Unsupported | Blocked):
        return out
    raise TypeError("realize returned draws although keep_draws was not requested")


def estimand_coverage(
    world_factory: WorldFactory,
    *,
    estimands: Sequence[Estimand],
    n: int,
    mass: float = 0.9,
    definition: IntervalDefinition = "eti",
    fit: Fitter | None = None,
    backend: str = "laplace",
    draws: int = 200,
    seed: int = 0,
    alpha: float = 0.01,
    truth_world: Callable[[SupportsWorld], SupportsWorld] | None = None,
) -> EstimandCoverageResult:
    """Coverage of realized estimands over ``n`` replications.

    Each estimand is realized on the fit (``assume_identified=True``: this
    is a recovery check, not an identification claim) and on
    ``truth_producer(truth_world(world))`` — ``truth_world`` maps the
    fitted world back to its generating world when the factory returns a
    mis-specified view (default: identity). The interval is scored against
    the truth's point value.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    if definition == "wald":
        raise ValueError("coverage scores posterior intervals; definition must be eti or hdi")
    names = [e.name for e in estimands]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"estimand names must be non-empty and distinct: {names}")
    region = clopper_pearson(n, mass, alpha)
    fitter = fit if fit is not None else _default_fitter(backend, draws)
    to_truth = truth_world if truth_world is not None else (lambda w: w)
    counts: dict[str, int] = {e.name: 0 for e in estimands}
    scored_n: dict[str, int] = {e.name: 0 for e in estimands}
    truths: dict[str, list[float]] = {e.name: [] for e in estimands}
    reasons: list[str] = []
    n_fitted = 0
    for i in range(n):
        world = world_factory(seed + i)
        result = fitter(world, seed + i)
        why = _failure(result, i)
        if why is not None:
            reasons.append(why)
            continue
        n_fitted += 1
        truth = truth_producer(to_truth(world))
        for e in estimands:
            got = _realized(e, result, definition, mass)
            ref = _realized(e, truth, definition, mass)
            if isinstance(got, Unsupported | Blocked):
                reasons.append(f"replication {i}, estimand {e.name!r}: {got.reason}")
                continue
            if isinstance(ref, Unsupported | Blocked):
                reasons.append(f"replication {i}, estimand {e.name!r} (truth): {ref.reason}")
                continue
            scored_n[e.name] += 1
            truths[e.name].append(ref.summary.mean)
            counts[e.name] += int(got.summary.interval.contains(ref.summary.mean))
    scored = tuple(
        EstimandCoverage(
            name=e.name,
            estimand_hash=e.content_hash(),
            n=scored_n[e.name],
            covered=counts[e.name],
            mass=mass,
            definition=definition,
            region=clopper_pearson(scored_n[e.name], mass, alpha),
            passed=clopper_pearson(scored_n[e.name], mass, alpha).accepts(counts[e.name]),
            true_values=tuple(truths[e.name]),
        )
        for e in estimands
        if scored_n[e.name] > 0
    )
    failed = tuple(e.name for e in scored if not e.passed)
    n_failed = n - n_fitted
    return EstimandCoverageResult(
        n=n,
        mass=mass,
        definition=definition,
        alpha=alpha,
        n_fitted=n_fitted,
        n_failed_fits=n_failed,
        failure_reasons=tuple(reasons),
        nominal_region=region,
        estimands=scored,
        failed_estimands=failed,
        passed=bool(not failed and scored and n_fitted > 0),
        provenance={"seed": seed, "backend": backend, "draws": draws},
    )
