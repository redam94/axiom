"""Response-surface worlds with known ground truth: a panel world and its ARMS special case.

``surface_world`` declares a ``SurfaceSpec`` (treatments, kernels,
carryover, intercepts, nuisance), fixes the true parameters, draws doses,
and generates the outcome by evaluating **the same expression tree the
estimator fits**: ``surface.model.build(spec)`` gives the ``ModelSpec``,
``surface.model.prepare`` lays the panel out, and ``axiom.core.value(
model.mean, data, theta)`` gives the noise-free outcome. There is no second
implementation of the Hill curve or the carryover in the simulator
(rule 3): a recovery test that passes here says the estimator inverts the
generator, and nothing else. The truth helpers ``marginal`` delegate to
``axiom.surface.forward.marginal`` / ``marginal_total`` for the same reason.

``arms_world`` is the single-period case — one row per unit, a shared
intercept, no carryover — that an experiment with dose arms produces.

**Truth.** Every parameter of the built model gets a true value, resolved
in prior-hierarchy order (parents before children):

* an entry in ``truth`` wins;
* the likelihood scale is ``noise_sd`` when given, else ``truth``'s entry,
  else its prior centre — it is never drawn, in either mode (pass it to
  vary it); ``noise_sd`` and a ``truth`` entry that disagree are an error;
* a ``fixed`` prior is its value;
* in ``truth_mode="centre"`` (the default) the *structural* parameters —
  those the kernels and carryovers introduce — sit at their prior centre
  (the mean; the median ``exp(mu)`` for a lognormal, so ``k`` sits at the
  kernel's ``reference_dose``), and everything else (intercepts, hierarchy
  hyperparameters, nuisance and interaction coefficients) is drawn from its
  prior with the world's generator;
* in ``truth_mode="prior"`` every free parameter is drawn from its prior —
  the simulation-based-calibration setting.

Each parameter draws from its own child generator (``Generator.spawn``), so
the two modes give the *same* non-structural values for the same seed and
differ only in the structural names — a prior-mode world is the centre-mode
world with its kernels and carryovers perturbed.

Deterministic by ``seed``: the same arguments give the same panel and the
same truth. The world keeps the prepared data dictionary so its truth
helpers (``forward``, ``total_response``, ``marginal``) evaluate the surface
at the truth over the panel's doses or over any substituted dose grid.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from axiom.core import (
    D,
    Likelihood,
    ModelSpec,
    Outcome,
    Param,
    Prior,
    Treatment,
    Unsupported,
    dimension_of,
    value,
)
from axiom.data import Panel
from axiom.sim.panel import DosePlan, draw_doses_rng, panel_from_arrays, unit_labels
from axiom.surface.carryover import (
    DelayedCarryover,
    GeometricCarryover,
    NoCarryover,
    WeibullCarryover,
)
from axiom.surface.forward import marginal as surface_marginal
from axiom.surface.forward import marginal_total as surface_marginal_total
from axiom.surface.kernels import (
    ExponentialKernel,
    HillKernel,
    LinearKernel,
    LogisticKernel,
    PowerKernel,
)
from axiom.surface.model import InterceptKind, Surface, SurfaceSpec, build, prepare
from axiom.surface.nuisance import NuisanceSet

__all__ = [
    "Horizon",
    "SurfaceWorld",
    "TruthMode",
    "arms_world",
    "surface_world",
    "true_parameters",
]

Array = npt.NDArray[np.float64]
TruthMode = Literal["centre", "prior"]
"""``centre``: structural parameters at their prior centre; ``prior``: everything drawn."""
Horizon = Literal["period", "total"]
"""``period``: same-period marginal (lag-zero weight); ``total``: summed over all lags."""

Kernel = HillKernel | LogisticKernel | ExponentialKernel | PowerKernel | LinearKernel
Carryover = GeometricCarryover | DelayedCarryover | WeibullCarryover | NoCarryover

_MAX_NAMED_CELLS = 20
"""How many offending ``(unit, period)`` cells an error or failure names before eliding."""


# -- truth -------------------------------------------------------------------------------


def _hyper(prior: Prior, key: str, resolved: Mapping[str, Array]) -> Array:
    v = prior.hyper[key]
    return np.asarray(resolved[v] if isinstance(v, str) else v, dtype=np.float64)


def _centre(prior: Prior, resolved: Mapping[str, Array]) -> Array:
    """The prior's centre: its mean, except the median for a lognormal (the reference dose)."""

    def h(k: str) -> Array:
        return _hyper(prior, k, resolved)

    match prior.family:
        case "normal":
            return h("mu")
        case "halfnormal":
            return np.asarray(h("sigma") * math.sqrt(2.0 / math.pi))
        case "lognormal":
            return np.asarray(np.exp(h("mu")))
        case "gamma":
            return np.asarray(h("alpha") / h("beta"))
        case "beta":
            return np.asarray(h("alpha") / (h("alpha") + h("beta")))
        case "uniform":
            return np.asarray(0.5 * (h("low") + h("high")))
        case "fixed":
            return h("value")
    raise ValueError(f"unknown prior family {prior.family!r}")  # pragma: no cover


def _draw(
    prior: Prior, resolved: Mapping[str, Array], shape: tuple[int, ...], rng: np.random.Generator
) -> Array:
    """One draw from the prior, with hyperparameters resolved against earlier truths."""

    def h(k: str) -> Array:
        return _hyper(prior, k, resolved)

    match prior.family:
        case "normal":
            out = rng.normal(h("mu"), h("sigma"), size=shape)
        case "halfnormal":
            out = np.abs(rng.normal(0.0, h("sigma"), size=shape))
        case "lognormal":
            out = rng.lognormal(h("mu"), h("sigma"), size=shape)
        case "gamma":
            out = rng.gamma(h("alpha"), 1.0 / h("beta"), size=shape)
        case "beta":
            out = rng.beta(h("alpha"), h("beta"), size=shape)
        case "uniform":
            out = rng.uniform(h("low"), h("high"), size=shape)
        case "fixed":
            out = np.broadcast_to(h("value"), shape)
    return np.asarray(out, dtype=np.float64)


def _resolution_order(model: ModelSpec) -> tuple[Param, ...]:
    """Parameters with every hyper-reference resolved before them (``ModelSpec`` forbids cycles)."""
    pending = list(model.parameters)
    done: set[str] = set()
    ordered: list[Param] = []
    while pending:
        ready = [p for p in pending if p.prior is None or all(q in done for q in p.prior.parents)]
        if not ready:  # pragma: no cover - ModelSpec validation rejects cycles
            raise ValueError("prior hierarchy could not be ordered")
        for p in ready:
            ordered.append(p)
            done.add(p.name)
            pending.remove(p)
    return tuple(ordered)


def _check_names(what: str, names: Sequence[str] | Mapping[str, Any], model: ModelSpec) -> None:
    declared = {p.name for p in model.parameters}
    unknown = sorted(set(names) - declared)
    if unknown:
        raise KeyError(
            f"{what} names parameters the model does not declare: {unknown}; "
            f"declared are {sorted(declared)}"
        )


def _as_declared(p: Param, given: npt.ArrayLike) -> Array:
    """``given`` broadcast to the parameter's declared shape, naming the parameter on failure."""
    a = np.asarray(given, dtype=np.float64)
    try:
        return np.broadcast_to(a, p.shape).copy()
    except ValueError as exc:
        raise ValueError(
            f"truth for {p.name!r} has shape {a.shape}; the parameter is declared with shape "
            f"{p.shape} and the value must broadcast to it"
        ) from exc


def true_parameters(
    model: ModelSpec,
    *,
    structural: Sequence[str] = (),
    truth: Mapping[str, npt.ArrayLike] | None = None,
    mode: TruthMode = "centre",
    seed: int | None = None,
) -> dict[str, Array]:
    """A full true-parameter dictionary for ``model`` (see the module docstring for the rules).

    ``structural`` names the parameters held at their prior centre in
    ``mode="centre"``; ``truth`` overrides any parameter. A vector parameter
    is broadcast to its declared shape. Unknown names in ``truth`` or
    ``structural`` raise ``KeyError``; a value that does not broadcast to
    the declared shape raises ``ValueError`` naming the parameter.
    """
    return _true_parameters(model, structural, truth or {}, mode, np.random.default_rng(seed))


def _true_parameters(
    model: ModelSpec,
    structural: Sequence[str],
    truth: Mapping[str, npt.ArrayLike],
    mode: TruthMode,
    rng: np.random.Generator,
    *,
    pinned: Sequence[str] = (),
) -> dict[str, Array]:
    """``pinned`` names sit at their prior centre in *either* mode (the likelihood scale)."""
    _check_names("truth", truth, model)
    _check_names("structural", structural, model)
    _check_names("pinned", pinned, model)
    held = set(structural)
    always = set(pinned)
    out: dict[str, Array] = {}
    for p in _resolution_order(model):
        prior = p.prior
        assert prior is not None  # ModelSpec guarantees a prior on every parameter
        # One child generator per parameter, spawned whether or not it is used, so a
        # parameter's draw does not depend on which other parameters were drawn.
        (child,) = rng.spawn(1)
        if p.name in truth:
            v = _as_declared(p, truth[p.name])
        elif prior.family == "fixed" or p.name in always or (mode == "centre" and p.name in held):
            v = np.broadcast_to(_centre(prior, out), p.shape).copy()
        else:
            v = _draw(prior, out, p.shape, child)
        out[p.name] = v if p.shape else np.asarray(float(v), dtype=np.float64)
    return out


def _structural_names(spec: SurfaceSpec) -> tuple[str, ...]:
    """Every kernel and carryover parameter of the spec — the ones recovery tests check."""
    names: list[str] = []
    for t in spec.treatments:
        kernel = spec.kernel_of(t.name)
        names.extend(
            p.name for p in kernel.parameters(t.name, dimension_of(t), spec.outcome_dimension)
        )
        names.extend(p.name for p in spec.carryover_of(t.name).parameters(t.name))
    return tuple(names)


def _truth_hash(truth: Mapping[str, npt.ArrayLike]) -> str:
    """blake2b-256 of the canonical JSON of ``truth`` (values as nested float lists)."""
    payload = {k: np.asarray(v, dtype=np.float64).tolist() for k, v in truth.items()}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(text.encode("utf-8"), digest_size=32).hexdigest()


def _name_cells(
    mask: npt.NDArray[np.bool_], labels: Sequence[str], periods: Sequence[object]
) -> str:
    """``(unit, period)`` of the first few true cells of an ``(n_units, n_periods)`` mask."""
    units, times = np.nonzero(mask)
    cells = [(str(labels[int(u)]), periods[int(t)]) for u, t in zip(units, times, strict=True)]
    shown = ", ".join(f"({u!r}, {t!r})" for u, t in cells[:_MAX_NAMED_CELLS])
    more = len(cells) - _MAX_NAMED_CELLS
    return f"{len(cells)} cell(s) (unit, period): {shown}" + (
        f", … and {more} more" if more > 0 else ""
    )


# -- the world ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceWorld:
    """A ``SurfaceSpec``, its true parameters, and a balanced panel generated from them.

    ``model`` is ``build(spec)``; ``data`` is ``prepare(spec, panel)`` — the
    dictionary the model's mean reads — and ``mean`` is the noise-free
    outcome on the panel's ``(n_units, n_periods)`` grid, so
    ``panel.array(outcome) - mean`` is exactly the noise that was added.
    ``theta`` holds every parameter of the model, vector intercepts included.
    ``noise_sd`` is the true likelihood scale, ``None`` for a likelihood
    without one (poisson).
    """

    spec: SurfaceSpec
    model: ModelSpec
    theta: dict[str, Array]
    panel: Panel
    data: dict[str, Array]
    mean: Array
    seed: int | None
    noise_sd: float | None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def surface(self) -> Surface:
        """The ``SupportsForward`` over this world's spec."""
        return Surface(self.spec)

    @property
    def treatments(self) -> tuple[str, ...]:
        return self.spec.treatment_names

    @property
    def n_units(self) -> int:
        return int(self.mean.shape[0])

    @property
    def n_periods(self) -> int:
        return int(self.mean.shape[1])

    @property
    def structural(self) -> tuple[str, ...]:
        """Names of the kernel and carryover parameters — the ones recovery tests check."""
        return _structural_names(self.spec)

    @property
    def noise(self) -> Array:
        """The noise that was added: ``outcome − mean`` on the panel grid."""
        return np.asarray(self.panel.array(self.spec.outcome.name) - self.mean)

    def _grid(self, name: str, values: npt.ArrayLike) -> Array:
        """``values`` as an ``(n_units, n_periods)`` dose grid: a scalar, the full grid, or
        for a one-period world a per-unit vector. Nothing else is accepted — a 1-D array is
        ambiguous between units and periods when both axes could take it."""
        a = np.asarray(values, dtype=np.float64)
        shape = self.mean.shape
        if a.ndim == 0:
            out = np.full(shape, float(a))
        elif a.ndim == 1 and self.n_periods == 1 and a.shape[0] == self.n_units:
            out = a.reshape(shape).copy()
        elif a.shape == shape:
            out = a.copy()
        else:
            accepted = "a scalar or the full grid"
            if self.n_periods == 1:
                accepted += f" or a per-unit vector of length {self.n_units}"
            raise ValueError(
                f"doses for {name!r} have shape {a.shape}; need (n_units, n_periods) = {shape}: "
                f"{accepted}"
            )
        if not np.all(np.isfinite(out)):
            raise ValueError(f"doses for {name!r} must be finite")
        return out

    def _inputs(self, dose: Mapping[str, npt.ArrayLike] | None) -> dict[str, Array]:
        """The panel's data dictionary with the given treatment doses substituted."""
        out = dict(self.data)
        for name, values in (dose or {}).items():
            if name not in self.treatments:
                raise KeyError(f"{name!r} is not a treatment of this world; have {self.treatments}")
            out[name] = self._grid(name, values)
        return out

    def forward(self, dose: Mapping[str, npt.ArrayLike] | None = None) -> Array:
        """The mean outcome at the truth: on the panel's doses, or with ``dose`` substituted.

        ``dose`` maps treatment name to an ``(n_units, n_periods)`` array, a
        scalar, or — for a one-period world — a per-unit vector; unnamed
        treatments keep the panel's doses. Evaluates the model's mean
        through ``axiom.core.value``.
        """
        return np.asarray(
            value(self.model.mean, data=self._inputs(dose), params=self.theta), dtype=np.float64
        )

    def total_response(self, dose: Mapping[str, npt.ArrayLike] | None = None) -> Array:
        """The treatments' contribution: ``forward(dose) − forward(all doses zero)``.

        What remains when intercepts and nuisance terms are removed — the
        quantity a dose-response estimand is about. Per cell of the panel
        grid; sum over the last axis for a per-unit cumulative total.
        """
        zero = {name: np.zeros(self.mean.shape) for name in self.treatments}
        return np.asarray(self.forward(dose) - self.forward(zero), dtype=np.float64)

    def marginal(
        self,
        treatment: str,
        dose: Mapping[str, npt.ArrayLike] | None = None,
        *,
        horizon: Horizon = "period",
    ) -> Array | Unsupported:
        """``d outcome / d dose`` of ``treatment`` at the truth, per cell of the panel grid.

        Delegates to ``axiom.surface.forward``: ``horizon="period"`` is
        ``marginal`` — the effect of one more unit of dose in period ``t``
        on the period-``t`` outcome, ``w_0 · ∂mu_t/∂c_t``; ``horizon="total"``
        is ``marginal_total`` — its effect summed over the lags that fall
        inside the horizon, ``Σ_l w_l · ∂mu_{t+l}/∂c_{t+l}``. Both include
        the interaction cross terms. Without carryover the two coincide.

        A cell whose derivative is not finite comes back as ``Unsupported``
        naming the cells rather than as ``inf``: a ``PowerKernel`` with
        shape ``s < 1`` — which its default ``beta(2, 2)`` prior always
        gives, in ``"centre"`` mode as well as ``"prior"`` mode — has an
        infinite slope at zero carried dose.
        """
        if treatment not in self.treatments:
            raise KeyError(
                f"{treatment!r} is not a treatment of this world; have {self.treatments}"
            )
        fn = surface_marginal if horizon == "period" else surface_marginal_total
        # An infinite slope is reported as Unsupported below, not as a numpy warning here.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            raw = fn(self.surface, self.theta, self._inputs(dose), treatment)
        out = np.asarray(np.broadcast_to(raw, self.mean.shape), dtype=np.float64)
        bad = ~np.isfinite(out)
        if np.any(bad):
            named = _name_cells(bad, self.spec.unit_labels, list(range(self.n_periods)))
            return Unsupported(
                reason=f"marginal of {treatment!r} is not finite at {named}; a kernel with "
                "shape s < 1 (PowerKernel's default prior) has infinite slope at zero dose",
                detail={
                    "treatment": treatment,
                    "horizon": horizon,
                    "n_cells": str(int(np.count_nonzero(bad))),
                },
            )
        return out


# -- construction -------------------------------------------------------------------------


def _treatments(treatments: Sequence[Treatment | str]) -> tuple[Treatment, ...]:
    out = tuple(
        t if isinstance(t, Treatment) else Treatment(name=t, dimension=D.currency, unit="USD")
        for t in treatments
    )
    if not out:
        raise ValueError("a surface world needs at least one treatment")
    names = [t.name for t in out]
    if len(set(names)) != len(names):
        raise ValueError(f"treatment names must be distinct: {names}")
    return out


def _per_treatment[T](
    what: str, given: Mapping[str, T] | T | None, names: Sequence[str]
) -> dict[str, T]:
    """One value per treatment from a mapping (partial allowed) or a single value for all."""
    if given is None:
        return {}
    if isinstance(given, Mapping):
        unknown = sorted(set(given) - set(names))
        if unknown:
            raise KeyError(f"{what} given for unknown treatments {unknown}; have {list(names)}")
        return dict(given)
    return {n: given for n in names}


def _default_kernels(
    kernels: Mapping[str, Kernel], plans: Mapping[str, DosePlan], names: Sequence[str]
) -> dict[str, Kernel]:
    """A Hill kernel at the dose plan's scale for every treatment without a kernel."""
    out: dict[str, Kernel] = dict(kernels)
    for n in names:
        if n not in out:
            out[n] = HillKernel(reference_dose=plans[n].scale)
    return out


def _check_noise_sd(noise_sd: float | None) -> None:
    if noise_sd is not None and not (math.isfinite(noise_sd) and noise_sd > 0):
        raise ValueError(f"noise_sd must be positive and finite, got {noise_sd}")


def surface_world(
    *,
    n_units: int,
    n_periods: int,
    treatments: Sequence[Treatment | str] = ("a", "b"),
    kernels: Mapping[str, Kernel] | Kernel | None = None,
    carryover: Mapping[str, Carryover] | Carryover | None = None,
    doses: Mapping[str, DosePlan] | DosePlan | None = None,
    nuisance: NuisanceSet | None = None,
    intercept: InterceptKind = "hierarchical",
    interactions: Sequence[tuple[str, str]] = (),
    outcome: Outcome | None = None,
    likelihood: Likelihood | None = None,
    truth: Mapping[str, npt.ArrayLike] | None = None,
    truth_mode: TruthMode = "centre",
    noise_sd: float | None = None,
    intercept_scale: float = 1.0,
    seed: int | None = None,
    name: str = "surface_world",
) -> SurfaceWorld:
    """A units × periods panel world with saturation, carryover, intercepts, and nuisance.

    ``treatments`` are ``Treatment`` entities or names (a name gets the
    currency dimension in USD). ``kernels`` / ``carryover`` / ``doses`` are
    per-treatment mappings (partial is fine) or one value for every
    treatment; a treatment without an entry gets ``HillKernel(
    reference_dose=plan.scale)``, ``NoCarryover()``, and ``DosePlan()``.
    ``intercept`` is ``"hierarchical"`` by default (unit intercepts around a
    common mean, hyperparameters on the ``intercept_scale``); the world
    supplies the unit labels. ``likelihood`` defaults to the spec's (normal
    with scale ``"sigma"``); ``truth`` overrides any parameter of the built
    model by name; ``noise_sd`` is the true likelihood scale (``None``: the
    ``truth`` entry, else the prior centre; it must be ``None`` for a
    poisson likelihood and may not disagree with ``truth``).

    A lognormal or poisson likelihood needs a positive mean in every cell;
    a world whose truth gives a non-positive mean is a ``ValueError``
    naming the cells, not a panel with impossible data.
    """
    if n_units < 1 or n_periods < 1:
        raise ValueError(f"need n_units >= 1 and n_periods >= 1, got {n_units}, {n_periods}")
    _check_noise_sd(noise_sd)
    ts = _treatments(treatments)
    names = [t.name for t in ts]
    given_plans = _per_treatment("dose plans", doses, names)
    plans = {n: given_plans.get(n, DosePlan()) for n in names}
    spec = SurfaceSpec(
        name=name,
        treatments=ts,
        outcome=outcome if outcome is not None else Outcome(name="y", dimension=D.outcome),
        kernels=_default_kernels(_per_treatment("kernels", kernels, names), plans, names),
        carryover=_per_treatment("carryover", carryover, names),
        nuisance=nuisance if nuisance is not None else NuisanceSet(terms=()),
        intercept=intercept,
        interactions=tuple(interactions),
        unit_labels=unit_labels(n_units),
        intercept_scale=intercept_scale,
        likelihood=(
            likelihood if likelihood is not None else Likelihood(family="normal", scale="sigma")
        ),
    )
    return _generate(
        spec,
        plans=plans,
        n_periods=n_periods,
        truth=truth,
        truth_mode=truth_mode,
        noise_sd=noise_sd,
        seed=seed,
    )


def arms_world(
    *,
    n_units: int,
    treatments: Sequence[Treatment | str] = ("dose",),
    kernels: Mapping[str, Kernel] | Kernel | None = None,
    doses: Mapping[str, DosePlan] | DosePlan | Mapping[str, npt.ArrayLike] | None = None,
    outcome: Outcome | None = None,
    truth: Mapping[str, npt.ArrayLike] | None = None,
    truth_mode: TruthMode = "centre",
    noise_sd: float | None = None,
    intercept_scale: float = 1.0,
    seed: int | None = None,
    name: str = "arms_world",
) -> SurfaceWorld:
    """The single-period world: one row per unit, a shared intercept, no carryover.

    This is what a dose-finding experiment produces — ``n_units`` arms or
    replicates, each at one dose per treatment. ``doses`` may be dose plans
    (as in ``surface_world``) or a mapping from treatment name to an
    explicit length-``n_units`` dose vector (a design's columns), in which
    case the doses are not random and a default kernel's ``reference_dose``
    is their mean. The panel has one period, ``t = 0``. ``noise_sd`` is
    resolved as in ``surface_world``.
    """
    if n_units < 1:
        raise ValueError(f"need n_units >= 1, got {n_units}")
    _check_noise_sd(noise_sd)
    ts = _treatments(treatments)
    names = [t.name for t in ts]
    fixed: dict[str, Array] = {}
    if (
        isinstance(doses, Mapping)
        and doses
        and not all(isinstance(v, DosePlan) for v in doses.values())
    ):
        if any(isinstance(v, DosePlan) for v in doses.values()):
            raise TypeError("doses must be all DosePlans or all dose arrays, not a mixture")
        unknown = sorted(set(doses) - set(names))
        if unknown:
            raise KeyError(f"doses given for unknown treatments {unknown}; have {names}")
        for n in names:
            if n not in doses:
                raise KeyError(f"no doses given for treatment {n!r}")
            a = np.asarray(doses[n], dtype=np.float64).reshape(-1)
            if a.shape != (n_units,):
                raise ValueError(
                    f"doses for {n!r} must have length n_units={n_units}, got {a.shape[0]}"
                )
            if not np.all(np.isfinite(a)) or np.any(a < 0):
                raise ValueError(f"doses for {n!r} must be finite and non-negative")
            fixed[n] = a.reshape(n_units, 1)
        plans = {n: DosePlan(scale=max(float(np.mean(fixed[n])), 1e-12)) for n in names}
    else:
        given_plans: dict[str, DosePlan] = _per_treatment(
            "dose plans",
            doses,  # type: ignore[arg-type]
            names,
        )
        plans = {n: given_plans.get(n, DosePlan()) for n in names}
    spec = SurfaceSpec(
        name=name,
        treatments=ts,
        outcome=outcome if outcome is not None else Outcome(name="y", dimension=D.outcome),
        kernels=_default_kernels(_per_treatment("kernels", kernels, names), plans, names),
        carryover={n: NoCarryover() for n in names},
        intercept="shared",
        unit_labels=unit_labels(n_units),
        intercept_scale=intercept_scale,
    )
    return _generate(
        spec,
        plans=plans,
        n_periods=1,
        truth=truth,
        truth_mode=truth_mode,
        noise_sd=noise_sd,
        seed=seed,
        fixed_doses=fixed,
    )


def _resolve_scale(
    spec: SurfaceSpec, truth: Mapping[str, npt.ArrayLike], noise_sd: float | None
) -> tuple[dict[str, npt.ArrayLike], tuple[str, ...]]:
    """``truth`` with the likelihood scale settled, and the names held at their prior centre.

    ``noise_sd`` wins when given and must agree with any ``truth`` entry;
    otherwise the ``truth`` entry is used; otherwise the scale is held at
    its prior centre (never drawn). A likelihood without a scale (poisson)
    rejects ``noise_sd``.
    """
    overrides: dict[str, npt.ArrayLike] = dict(truth)
    scale = spec.likelihood.scale
    if scale is None:
        if noise_sd is not None:
            raise ValueError(
                f"{spec.likelihood.family} likelihood has no scale parameter; "
                f"noise_sd must be None, got {noise_sd}"
            )
        return overrides, ()
    if noise_sd is not None:
        if scale in truth:
            declared = float(np.asarray(truth[scale], dtype=np.float64).reshape(-1)[0])
            if not math.isclose(declared, noise_sd, rel_tol=0.0, abs_tol=0.0):
                raise ValueError(
                    f"noise_sd={noise_sd} disagrees with truth[{scale!r}]={declared}; "
                    "give the likelihood scale once"
                )
        overrides[scale] = noise_sd
        return overrides, ()
    if scale in truth:
        return overrides, ()
    return overrides, (scale,)


def _generate(
    spec: SurfaceSpec,
    *,
    plans: Mapping[str, DosePlan],
    n_periods: int,
    truth: Mapping[str, npt.ArrayLike] | None,
    truth_mode: TruthMode,
    noise_sd: float | None,
    seed: int | None,
    fixed_doses: Mapping[str, Array] | None = None,
) -> SurfaceWorld:
    """Draw doses, fix the truth, evaluate the one forward, add noise, return the world."""
    rng = np.random.default_rng(seed)
    model = build(spec)
    labels = spec.unit_labels
    n_units = len(labels)
    dose_arrays: dict[str, Array] = {}
    for t in spec.treatments:
        if fixed_doses and t.name in fixed_doses:
            dose_arrays[t.name] = np.asarray(fixed_doses[t.name], dtype=np.float64)
        else:
            dose_arrays[t.name] = draw_doses_rng(plans[t.name], n_units, n_periods, rng)
    given_truth: dict[str, npt.ArrayLike] = dict(truth or {})
    overrides, held_scale = _resolve_scale(spec, given_truth, noise_sd)
    structural = _structural_names(spec)
    theta = _true_parameters(model, structural, overrides, truth_mode, rng, pinned=held_scale)
    scale = spec.likelihood.scale
    true_noise_sd = float(theta[scale]) if scale is not None else None

    def assemble(outcome: Array) -> Panel:
        return panel_from_arrays(
            doses=dose_arrays,
            outcome=outcome,
            treatments=spec.treatments,
            outcome_entity=spec.outcome,
            units=labels,
            unit_column=spec.unit_column,
            time_column=spec.time_column,
        )

    # A provisional panel (outcome zero) lets ``prepare`` lay out the data dictionary —
    # unit index, time, nuisance basis columns — that the outcome is then generated from.
    data = prepare(spec, assemble(np.zeros((n_units, n_periods))))
    mean = np.asarray(value(model.mean, data=data, params=theta), dtype=np.float64)
    mean = np.broadcast_to(mean, (n_units, n_periods)).copy()
    panel = assemble(mean + _noise(spec, mean, theta, rng))
    provenance: dict[str, Any] = {
        "seed": seed,
        "spec_hash": spec.content_hash(),
        "model_hash": model.content_hash(),
        "panel_hash": panel.content_hash(),
        "truth_mode": truth_mode,
        "truth": {"names": sorted(given_truth), "hash": _truth_hash(given_truth)},
        "structural": list(structural),
        "noise_sd": true_noise_sd,
        "n_units": n_units,
        "n_periods": n_periods,
    }
    return SurfaceWorld(
        spec=spec,
        model=model,
        theta=theta,
        panel=panel,
        data=prepare(spec, panel),
        mean=mean,
        seed=seed,
        noise_sd=true_noise_sd,
        provenance=provenance,
    )


def _require_positive_mean(spec: SurfaceSpec, mean: Array) -> None:
    """A lognormal or poisson outcome is impossible where the mean is not positive."""
    bad = ~(mean > 0)
    if np.any(bad):
        named = _name_cells(bad, spec.unit_labels, list(range(mean.shape[-1])))
        raise ValueError(
            f"{spec.likelihood.family} likelihood needs a positive mean in every cell; "
            f"the truth gives a non-positive mean at {named}. Raise the intercept or the "
            "amplitudes through `truth`."
        )


def _noise(
    spec: SurfaceSpec, mean: Array, theta: Mapping[str, Array], rng: np.random.Generator
) -> Array:
    """Likelihood noise on the outcome scale for the spec's likelihood family."""
    lik = spec.likelihood
    scale = float(theta[lik.scale]) if lik.scale else 0.0
    match lik.family:
        case "normal":
            return np.asarray(rng.normal(0.0, scale, size=mean.shape), dtype=np.float64)
        case "student_t":
            df = float(lik.df or 0.0)
            return np.asarray(scale * rng.standard_t(df, size=mean.shape), dtype=np.float64)
        case "lognormal":
            _require_positive_mean(spec, mean)
            return np.asarray(
                mean * np.expm1(rng.normal(0.0, scale, size=mean.shape)), dtype=np.float64
            )
        case "poisson":
            _require_positive_mean(spec, mean)
            return np.asarray(rng.poisson(mean) - mean, dtype=np.float64)
    raise ValueError(f"unknown likelihood family {lik.family!r}")  # pragma: no cover
