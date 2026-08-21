"""The declarative Bayesian response surface: ``SurfaceSpec`` → ``ModelSpec`` → ``FitResult``.

``SurfaceSpec`` names the treatments and the outcome and, per treatment, a
saturation kernel and a carryover; plus an intercept structure, pairwise
interactions, nuisance terms, and a likelihood. ``build`` turns it into the
one ``ModelSpec`` whose mean the likelihood, the simulator, the design math,
and the optimizer all evaluate (rule 3). ``Surface`` composes the two and
satisfies ``SupportsForward``; ``prepare`` lays a ``Panel`` out the way the
tree reads it; ``fit`` runs a backend and returns a ``FitResult``.

This is the observational **panel** model (review D2): units × time with
carryover and saturation per treatment, shared / per-unit / hierarchical
intercepts, and a nuisance baseline. The designed-experiment (arms) surface
is its special case — one period, no carryover, 1-D dose arrays.

Layout contract (units × time)
------------------------------
``Convolve`` acts along the **last** axis, so a dose series is laid out with
time last and units leading: every measured column is an ``(n_units,
n_periods)`` array sorted by (unit, time), and the mean is evaluated on that
grid — carryover convolves within a unit and never across units. ``Gather``
on a ``(n_units,)`` intercept vector needs an index of shape ``(n_units,
1)`` so the gathered intercept broadcasts down the time axis; ``prepare``
returns exactly that under ``unit_column``. For a 1-D dose grid (one entry
per run) a 1-D integer unit index of the same length is the right shape.

A spec that declares any carryover therefore reads **every** dose array as
``(n_units, n_periods)``: ``Surface.forward`` and ``Surface.linearize``
raise ``ValueError`` for a dose array with fewer than two axes rather than
convolve a grid of independent rows as if they were consecutive periods of
one unit. Rows that *are* independent — design points, candidate
allocations — go through ``Surface.steady_state()``, the same spec with
``NoCarryover`` for every treatment, which is exact at a steady state
because the weights sum to one (``f(Σ_l w_l x) = f(x)``). Carryover also
needs equispaced periods; ``prepare`` refuses a panel whose periods are not.

Units are part of the contract: ``prepare`` requires the panel's treatment
and outcome entities to carry the same unit string as the spec's, not just
the same dimension (``USD`` and cents share a dimension and differ by a
factor of one hundred).

Parameter roles
---------------
Every parameter the surface declares has one of three roles, which is what
``linearize`` keys on: **linear** — the mean is linear in it (intercepts,
amplitudes ``beta``, interaction ``gamma``, nuisance coefficients);
**nonlinear** — it sits inside a transform (scales ``k``, shapes ``s``,
carryover parameters); **auxiliary** — it is not in the mean at all
(hierarchical hyperparameters, the likelihood scale).

Nuisance conventions
--------------------
A ``LinearTrend`` with ``origin`` / ``scale`` left ``None`` resolves them
against the frame it sees. So that a forecast panel is laid out on the
*same* basis as the fitted one, ``prepare`` resolves the conventions once
(``resolve_conventions``), ``fit`` records them under
``FitResult.provenance["nuisance_conventions"]``, and ``prepare(...,
conventions=...)`` reuses them for a prediction panel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, model_validator

from axiom.core import (
    Add,
    Data,
    DesignMatrix,
    Dimension,
    DimensionError,
    Expr,
    Gather,
    Likelihood,
    ModelSpec,
    Mul,
    NonEmptyStr,
    Outcome,
    Param,
    Posterior,
    Prior,
    Spec,
    Treatment,
    Unsupported,
    Unverified,
    dimension_of,
    dimensionless,
    value,
)
from axiom.data import Panel, PanelError, RoleMap
from axiom.infer import Backend, ConvergenceReport, diagnose, get_backend
from axiom.surface.carryover import AnyCarryover, CarryoverKernel, NoCarryover
from axiom.surface.kernels import AnyKernel, HillKernel, ResponseKernel
from axiom.surface.linearize import design_matrix
from axiom.surface.nuisance import LinearTrend, NuisanceSet

__all__ = [
    "INTERCEPT",
    "INTERCEPT_MEAN",
    "INTERCEPT_SD",
    "UNIT_INTERCEPT",
    "Conventions",
    "DataDict",
    "FitResult",
    "InterceptKind",
    "ParameterRole",
    "Surface",
    "SurfaceSpec",
    "build",
    "fit",
    "interaction_name",
    "parameter_roles",
    "prepare",
    "resolve_conventions",
]

Array = npt.NDArray[np.float64]
DataDict = dict[str, npt.NDArray[Any]]
"""What ``prepare`` returns: float arrays plus the integer unit index."""
Conventions = Mapping[str, Mapping[str, float]]
"""Nuisance conventions: ``{trend column: {"origin": ..., "scale": ...}}`` in time units."""
ParameterRole = Literal["linear", "nonlinear", "auxiliary"]
InterceptKind = Literal["none", "shared", "per_unit", "hierarchical"]

INTERCEPT = "alpha"
"""Name of the shared intercept parameter."""
UNIT_INTERCEPT = "alpha_unit"
"""Name of the per-unit intercept vector (shape ``(n_units,)``)."""
INTERCEPT_MEAN = "alpha_mean"
"""Hierarchical intercept: population mean."""
INTERCEPT_SD = "alpha_sd"
"""Hierarchical intercept: between-unit standard deviation."""
STEADY_STATE = "steady_state"
"""Suffix on a steady-state surface's spec name and the key in its provenance."""


def interaction_name(first: str, second: str) -> str:
    """The parameter name of the pairwise interaction ``gamma`` between two treatments."""
    return f"gamma_{first}_{second}"


# -- spec --------------------------------------------------------------------------------


class SurfaceSpec(Spec):
    """A response surface over named treatments: what is fit, declared once.

    * ``treatments`` — each carries the dose dimension and unit; the dose
      column the model reads is ``treatment.name``.
    * ``kernels`` / ``carryover`` — per treatment name; a treatment without
      an entry gets ``HillKernel()`` (set ``reference_dose`` to the data's
      scale) and ``NoCarryover()``.
    * ``intercept`` — ``"none"``, ``"shared"`` (``alpha``), ``"per_unit"``
      (``alpha_unit[i] ~ normal(0, intercept_scale)``), or
      ``"hierarchical"`` (``alpha_unit[i] ~ normal(alpha_mean, alpha_sd)``).
      The last two need ``unit_labels``; a unit's index is its position.
    * ``interactions`` — pairs of treatment names; each adds
      ``gamma · f_i · f_j`` with the dimensionless saturations and
      ``gamma ~ normal(0, interaction_scale)`` in the outcome dimension.
    * ``nuisance`` — baseline terms; their basis columns come from
      ``prepare`` and their coefficients are linear parameters.
    * ``likelihood`` — default normal with scale ``"sigma"`` and a
      ``halfnormal(noise_scale)`` prior.
    * ``time_column`` / ``unit_column`` — the names of the time and
      unit-index columns in the prepared data dict.

    Construction builds the model once, so a spec that exists is one that
    dimension-checks and whose parameter names are distinct.
    """

    name: NonEmptyStr
    treatments: tuple[Treatment, ...] = Field(min_length=1)
    outcome: Outcome
    kernels: dict[str, AnyKernel] = {}
    carryover: dict[str, AnyCarryover] = {}
    nuisance: NuisanceSet = NuisanceSet(terms=())
    intercept: InterceptKind = "shared"
    interactions: tuple[tuple[str, str], ...] = ()
    likelihood: Likelihood = Likelihood(family="normal", scale="sigma")
    time_column: NonEmptyStr = "t"
    unit_column: NonEmptyStr = "unit"
    unit_labels: tuple[str, ...] = ()
    intercept_scale: float = Field(default=1.0, gt=0)
    interaction_scale: float = Field(default=1.0, gt=0)
    noise_scale: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def _consistent(self) -> SurfaceSpec:
        names = self.treatment_names
        if len(set(names)) != len(names):
            raise ValueError(f"treatment names must be distinct: {names}")
        for field, keys in (("kernels", self.kernels), ("carryover", self.carryover)):
            unknown = sorted(set(keys) - set(names))
            if unknown:
                raise ValueError(f"{field} names treatments that do not exist: {unknown}")
        seen: set[frozenset[str]] = set()
        for pair in self.interactions:
            first, second = pair
            if first == second or first not in names or second not in names:
                raise ValueError(
                    f"interaction {pair} must name two distinct treatments among {names}"
                )
            key = frozenset(pair)
            if key in seen:
                raise ValueError(f"interaction {pair} is declared twice")
            seen.add(key)
        if self.intercept in ("per_unit", "hierarchical") and not self.unit_labels:
            raise ValueError(f"intercept={self.intercept!r} needs unit_labels")
        if len(set(self.unit_labels)) != len(self.unit_labels):
            raise ValueError("unit_labels must be distinct")
        columns = list(self.data_columns)
        dupes = sorted({c for c in columns if columns.count(c) > 1})
        if dupes:
            raise ValueError(f"data column names collide: {dupes}")
        build(self)  # dimension check and parameter-name closure, at construction
        return self

    # -- accessors ------------------------------------------------------------------

    @property
    def treatment_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.treatments)

    def treatment(self, name: str) -> Treatment:
        for t in self.treatments:
            if t.name == name:
                return t
        raise KeyError(f"no treatment {name!r}; have {list(self.treatment_names)}")

    def kernel_of(self, name: str) -> ResponseKernel:
        """The saturation kernel for a treatment (``HillKernel()`` when none is declared)."""
        self.treatment(name)
        kernel = self.kernels.get(name)
        return kernel if kernel is not None else HillKernel()

    def carryover_of(self, name: str) -> CarryoverKernel:
        """The carryover for a treatment (``NoCarryover()`` when none is declared)."""
        self.treatment(name)
        carryover = self.carryover.get(name)
        return carryover if carryover is not None else NoCarryover()

    @property
    def carried(self) -> tuple[str, ...]:
        """Treatments that declare a carryover other than ``NoCarryover``, in spec order."""
        return tuple(
            t.name
            for t in self.treatments
            if not isinstance(self.carryover_of(t.name), NoCarryover)
        )

    @property
    def outcome_dimension(self) -> Dimension:
        return dimension_of(self.outcome)

    def dose_dimension(self, name: str) -> Dimension:
        return dimension_of(self.treatment(name))

    @property
    def n_units(self) -> int:
        """Number of unit labels (``0`` when the intercept does not index units)."""
        return len(self.unit_labels)

    @property
    def data_columns(self) -> tuple[str, ...]:
        """Every column of the prepared data dict, in ``prepare``'s order."""
        return (
            self.unit_column,
            self.time_column,
            self.outcome.name,
            *self.treatment_names,
            *self.nuisance.column_names(),
        )


# -- build -------------------------------------------------------------------------------


def _normal(scale: float) -> Prior:
    return Prior(family="normal", hyper={"mu": 0.0, "sigma": float(scale)})


def _halfnormal(scale: float) -> Prior:
    return Prior(family="halfnormal", hyper={"sigma": float(scale)})


def _kernel_role(kernel: ResponseKernel, param: Param, treatment: str) -> ParameterRole:
    stem = param.name.removesuffix(f"_{treatment}")
    if stem not in kernel.roles:  # pragma: no cover - kernel contract
        raise KeyError(f"kernel {kernel.name!r} declares {param.name!r} with no role")
    return "linear" if kernel.roles[stem] == "amplitude" else "nonlinear"


def _intercept(spec: SurfaceSpec) -> tuple[list[Expr], list[tuple[Param, ParameterRole]]]:
    dim = spec.outcome_dimension
    if spec.intercept == "none":
        return [], []
    if spec.intercept == "shared":
        alpha = Param(name=INTERCEPT, dimension=dim, prior=_normal(spec.intercept_scale))
        return [alpha], [(alpha, "linear")]
    index = Data(name=spec.unit_column, dimension=dimensionless())
    declared: list[tuple[Param, ParameterRole]] = []
    if spec.intercept == "hierarchical":
        mean = Param(name=INTERCEPT_MEAN, dimension=dim, prior=_normal(spec.intercept_scale))
        sd = Param(name=INTERCEPT_SD, dimension=dim, prior=_halfnormal(spec.intercept_scale))
        prior = Prior(family="normal", hyper={"mu": INTERCEPT_MEAN, "sigma": INTERCEPT_SD})
        declared += [(mean, "auxiliary"), (sd, "auxiliary")]
    else:
        prior = _normal(spec.intercept_scale)
    units = Param(name=UNIT_INTERCEPT, dimension=dim, prior=prior, shape=(spec.n_units,))
    declared.append((units, "linear"))
    return [Gather(source=units, index=index)], declared


def _carried(spec: SurfaceSpec) -> dict[str, Expr]:
    """The carryover-transformed dose expression per treatment (the kernel's input)."""
    out: dict[str, Expr] = {}
    for t in spec.treatments:
        dose = Data(name=t.name, dimension=dimension_of(t))
        out[t.name] = spec.carryover_of(t.name).apply(dose, t.name)
    return out


def _assemble(spec: SurfaceSpec) -> tuple[ModelSpec, dict[str, ParameterRole]]:
    out_dim = spec.outcome_dimension
    terms, declared = _intercept(spec)
    carried = _carried(spec)
    for t in spec.treatments:
        kernel, carryover = spec.kernel_of(t.name), spec.carryover_of(t.name)
        terms.append(kernel.response(carried[t.name], t.name, out_dim))
        for p in kernel.parameters(t.name, dimension_of(t), out_dim):
            declared.append((p, _kernel_role(kernel, p, t.name)))
        declared.extend((p, "nonlinear") for p in carryover.parameters(t.name))
    for first, second in spec.interactions:
        gamma = Param(
            name=interaction_name(first, second),
            dimension=out_dim,
            prior=_normal(spec.interaction_scale),
        )
        terms.append(
            Mul(
                factors=(
                    gamma,
                    spec.kernel_of(first).saturation(carried[first], first),
                    spec.kernel_of(second).saturation(carried[second], second),
                )
            )
        )
        declared.append((gamma, "linear"))
    if spec.nuisance.terms:
        terms.append(spec.nuisance.expr(outcome_dimension=out_dim))
        declared.extend((p, "linear") for p in spec.nuisance.parameters(outcome_dimension=out_dim))
    if spec.likelihood.scale:
        family = spec.likelihood.family
        scale_dim = out_dim if family in ("normal", "student_t") else dimensionless()
        sigma = Param(
            name=spec.likelihood.scale, dimension=scale_dim, prior=_halfnormal(spec.noise_scale)
        )
        declared.append((sigma, "auxiliary"))
    mean: Expr = terms[0] if len(terms) == 1 else Add(terms=tuple(terms))
    model = ModelSpec(
        name=spec.name,
        mean=mean,
        outcome=Data(name=spec.outcome.name, dimension=out_dim),
        likelihood=spec.likelihood,
        parameters=tuple(p for p, _ in declared),
    )
    return model, {p.name: role for p, role in declared}


def build(spec: SurfaceSpec) -> ModelSpec:
    """The ``ModelSpec`` a backend fits: ``mean = intercept + Σ_i beta_i f_i(carry_i(dose_i))
    + Σ gamma_ij f_i f_j + nuisance``, every parameter with its prior, dimension-checked."""
    return _assemble(spec)[0]


def parameter_roles(spec: SurfaceSpec) -> dict[str, ParameterRole]:
    """``{parameter name: "linear" | "nonlinear" | "auxiliary"}`` in declaration order."""
    return _assemble(spec)[1]


# -- surface -----------------------------------------------------------------------------


def _require_panel_layout(spec: SurfaceSpec, dose: Mapping[str, npt.ArrayLike], what: str) -> None:
    """Refuse a dose array with fewer than two axes for a treatment that declares carryover.

    ``Convolve`` runs along the last axis, so a 1-D grid of independent rows
    would be treated as one unit's consecutive periods. A treatment absent
    from ``dose`` is left to the interpreter, which raises ``KeyError``
    naming it.
    """
    for name in spec.carried:
        if name not in dose:
            continue
        nd = int(np.ndim(dose[name]))
        if nd < 2:
            raise ValueError(
                f"{what}: treatment {name!r} declares carryover "
                f"{spec.carryover_of(name).name!r}, so its dose array must be laid out "
                f"(n_units, n_periods) with time last; got ndim={nd}. Independent rows — a "
                f"design grid, candidate allocations — need Surface.steady_state()."
            )


class Surface:
    """A ``SurfaceSpec`` with its built model: the ``SupportsForward`` the upper layers use.

    ``forward(dose, theta)`` is ``axiom.core.value`` over ``expr``; ``dose``
    maps every ``Data`` column the mean reads — treatment doses, the unit
    index for per-unit intercepts, nuisance basis columns — to arrays laid
    out as the module docstring describes. ``linearize(dose, theta_at)``
    returns the design matrix in the linear parameters (``surface.linear``)
    with the nonlinear ones held at ``theta_at``.

    When the spec declares carryover for any treatment, both methods
    require that treatment's dose array to have at least two axes
    (``(n_units, n_periods)``, time last) and raise ``ValueError`` naming
    the treatment otherwise; ``steady_state()`` is the surface to hand to
    callers that evaluate independent rows.
    """

    def __init__(self, spec: SurfaceSpec, *, provenance: Mapping[str, Any] | None = None) -> None:
        self._spec = spec
        self._model, self._roles = _assemble(spec)
        self._provenance: dict[str, Any] = (
            dict(provenance) if provenance is not None else {STEADY_STATE: False}
        )

    @property
    def spec(self) -> SurfaceSpec:
        return self._spec

    @property
    def model(self) -> ModelSpec:
        return self._model

    @property
    def expr(self) -> Expr:
        return self._model.mean

    @property
    def provenance(self) -> dict[str, Any]:
        """``{"steady_state": bool}`` plus, for a steady-state surface, the spec it came from."""
        return dict(self._provenance)

    @property
    def roles(self) -> dict[str, ParameterRole]:
        return dict(self._roles)

    @property
    def linear(self) -> tuple[str, ...]:
        """Parameters the mean is linear in — the columns of ``linearize``."""
        return tuple(n for n, r in self._roles.items() if r == "linear")

    @property
    def nonlinear(self) -> tuple[str, ...]:
        """Parameters inside a transform — the point ``linearize`` is taken at."""
        return tuple(n for n, r in self._roles.items() if r == "nonlinear")

    @property
    def auxiliary(self) -> tuple[str, ...]:
        """Parameters outside the mean: hyperparameters and the likelihood scale."""
        return tuple(n for n, r in self._roles.items() if r == "auxiliary")

    def steady_state(self) -> Surface:
        """The same spec with ``NoCarryover`` for every treatment.

        Exact at a steady state: the carryover weights sum to one, so a
        dose held constant over the lags gives ``Σ_l w_l · x = x`` and the
        kernel sees the dose itself. The carryover parameters (``lam_*``,
        ``theta_*``, ``kappa_*``) leave the model; everything else — kernels,
        intercepts, interactions, nuisance, likelihood — is unchanged. The
        spec name gains the suffix ``":steady_state"`` and ``provenance``
        records ``{"steady_state": True, "source_spec": name, "source_spec_hash":
        hash}``. Rows of a dose grid are then independent, so a 1-D grid is
        accepted.
        """
        spec = self._spec
        steady = spec.model_copy(
            update={
                "name": f"{spec.name}:{STEADY_STATE}",
                "carryover": {t: NoCarryover() for t in spec.treatment_names},
            }
        )
        return Surface(
            steady,
            provenance={
                STEADY_STATE: True,
                "source_spec": spec.name,
                "source_spec_hash": spec.content_hash(),
                "dropped_parameters": tuple(
                    p.name for t in spec.treatment_names for p in spec.carryover_of(t).parameters(t)
                ),
            },
        )

    def forward(
        self, dose: Mapping[str, npt.ArrayLike], theta: Mapping[str, npt.ArrayLike]
    ) -> Array:
        _require_panel_layout(self._spec, dose, "forward")
        return value(self._model.mean, data=dose, params=theta)

    def linearize(
        self, dose: Mapping[str, npt.ArrayLike], theta_at: Mapping[str, npt.ArrayLike]
    ) -> DesignMatrix:
        _require_panel_layout(self._spec, dose, "linearize")
        return design_matrix(self._model, self.linear, dose, theta_at)

    def __repr__(self) -> str:
        return (
            f"Surface({self._spec.name!r}, treatments={list(self._spec.treatment_names)}, "
            f"intercept={self._spec.intercept!r}, linear={len(self.linear)}, "
            f"nonlinear={len(self.nonlinear)})"
        )


# -- prepare -----------------------------------------------------------------------------


def _treatment_column(roles: RoleMap, name: str) -> str:
    """The panel column holding a treatment: the column of that name, else the entity's."""
    if name in roles.treatments:
        return name
    for column, entity in roles.treatments.items():
        if entity.name == name:
            return column
    raise KeyError(
        f"panel has no treatment column or entity named {name!r}; "
        f"treatment columns are {list(roles.treatments)}"
    )


def _numeric_periods(periods: Sequence[object]) -> Array:
    """Period labels as float64: numeric as-is, datetimes as days since 1970-01-01."""
    series = pd.Series(list(periods))
    if pd.api.types.is_datetime64_any_dtype(series):
        epoch = pd.Timestamp("1970-01-01", tz=series.dt.tz)
        days = (series - epoch) / pd.Timedelta(days=1)
        return np.asarray(days.to_numpy(), dtype=np.float64)
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError(f"time column must be numeric or datetime, got {series.dtype}")
    return np.asarray(series.to_numpy(), dtype=np.float64)


def _require_equispaced(spec: SurfaceSpec, periods: Sequence[object]) -> None:
    """Carryover counts lags in periods, so the periods must be equally spaced."""
    t = _numeric_periods(periods)
    steps = np.diff(t)
    if steps.size == 0:
        return
    if not np.allclose(steps, steps[0], rtol=1e-9, atol=1e-12):
        distinct = np.unique(np.round(steps, 9))
        raise PanelError(
            f"surface {spec.name!r} declares carryover for {list(spec.carried)}, which counts "
            f"lags in periods and needs equispaced periods; the period steps take "
            f"{distinct.size} distinct values {distinct[:6].tolist()}"
            f"{'...' if distinct.size > 6 else ''}"
        )


def _require_same_unit(what: str, panel_unit: str | None, spec_unit: str | None) -> None:
    if panel_unit != spec_unit:
        raise DimensionError(
            f"{what}: panel declares unit {panel_unit!r}, spec declares {spec_unit!r}; "
            "the unit string is part of the contract, not only the dimension"
        )


def _grid(
    frame: pd.DataFrame,
    roles: RoleMap,
    labels: Sequence[str],
    periods: Sequence[object],
    column: str,
) -> Array:
    wide = frame.pivot(index=roles.unit, columns=roles.time, values=column)
    wide = wide.reindex(index=list(labels), columns=list(periods))
    out = np.asarray(wide.to_numpy(dtype=float), dtype=np.float64)
    if not np.all(np.isfinite(out)):
        raise ValueError(
            f"column {column!r} has missing or non-finite values; the surface does not impute"
        )
    return out


def _term_prefix(i: int) -> str:
    """``NuisanceSet``'s documented per-term prefix: term ``i`` gets ``n{i}_``."""
    return f"n{i}_"


def resolve_conventions(
    spec: SurfaceSpec, panel: Panel, conventions: Conventions | None = None
) -> dict[str, dict[str, float]]:
    """The trend conventions ``prepare`` lays the panel out with, one entry per trend column.

    For every ``LinearTrend`` in ``spec.nuisance``: the entry in
    ``conventions`` when one is given (it must hold ``origin`` and
    ``scale``, must agree with any value the term fixes explicitly, and
    ``scale`` must be positive), else the term's own resolution against
    the panel's periods (explicit fields, otherwise the first observed time
    and the window length). ``conventions`` naming a column that is not a
    trend column of this spec is a ``ValueError``. The result is what
    ``fit`` records under ``provenance["nuisance_conventions"]`` and what
    ``prepare(..., conventions=...)`` accepts for a forecast panel.
    """
    times = _numeric_periods(panel.periods)
    given = {column: dict(values) for column, values in (conventions or {}).items()}
    known = spec.nuisance.column_names()
    out: dict[str, dict[str, float]] = {}
    for i, term in enumerate(spec.nuisance.terms):
        if not isinstance(term, LinearTrend):
            continue
        (column,) = term.column_names(_term_prefix(i))
        if column not in known:  # pragma: no cover - NuisanceSet prefix contract
            raise RuntimeError(f"trend column {column!r} is not among {known}")
        if column in given:
            supplied = given.pop(column)
            missing = sorted({"origin", "scale"} - set(supplied))
            if missing:
                raise ValueError(f"conventions for trend column {column!r} lack {missing}")
            origin, scale = float(supplied["origin"]), float(supplied["scale"])
            if not (np.isfinite(origin) and np.isfinite(scale)) or scale <= 0.0:
                raise ValueError(
                    f"conventions for trend column {column!r} must be finite with scale > 0; "
                    f"got origin={origin}, scale={scale}"
                )
            for field, declared, supplied_value in (
                ("origin", term.origin, origin),
                ("scale", term.scale, scale),
            ):
                if declared is not None and declared != supplied_value:
                    raise ValueError(
                        f"trend column {column!r}: the spec fixes {field}={declared} but the "
                        f"conventions say {supplied_value}"
                    )
        else:
            origin, scale = term.resolve(times)
        out[column] = {"origin": float(origin), "scale": float(scale)}
    if given:
        raise ValueError(
            f"conventions name columns that are not trend columns of surface {spec.name!r}: "
            f"{sorted(given)}; trend columns are {sorted(out)}"
        )
    return out


def _resolved_nuisance(
    spec: SurfaceSpec, resolved: Mapping[str, Mapping[str, float]]
) -> NuisanceSet:
    """``spec.nuisance`` with every trend's origin / scale fixed to the resolved conventions."""
    terms: list[Any] = []
    for i, term in enumerate(spec.nuisance.terms):
        if isinstance(term, LinearTrend):
            (column,) = term.column_names(_term_prefix(i))
            values = resolved[column]
            term = term.model_copy(update={"origin": values["origin"], "scale": values["scale"]})
        terms.append(term)
    return spec.nuisance.model_copy(update={"terms": tuple(terms)})


def prepare(spec: SurfaceSpec, panel: Panel, *, conventions: Conventions | None = None) -> DataDict:
    """The data dict the built model reads, laid out ``(n_units, n_periods)``.

    Keys: every treatment name (dose), ``spec.outcome.name``,
    ``spec.unit_column`` (zero-based unit index, int, shape ``(n_units,
    1)``), ``spec.time_column`` (numeric time), and each nuisance basis
    column from ``spec.nuisance.augment``. Rows follow ``spec.unit_labels``
    when given (the panel must hold exactly those units), else the panel's
    sorted units; columns follow the panel's sorted periods.

    Refused: an unbalanced panel (``PanelError``); nulls in a column the
    model reads (``ValueError``); a treatment or outcome whose dimension
    *or unit string* differs from the spec's (``DimensionError`` naming
    both); periods that are not equispaced when any treatment declares
    carryover (``PanelError``).

    ``conventions`` fixes the trend origin / scale (``{trend column:
    {"origin", "scale"}}``, as recorded in
    ``FitResult.provenance["nuisance_conventions"]``) so a forecast panel
    is laid out on the fitted basis; without it each trend resolves against
    this panel (``resolve_conventions``).
    """
    panel.require_balanced(context=f"surface {spec.name!r}")
    roles = panel.roles
    have = panel.units
    labels: tuple[str, ...] = spec.unit_labels or have
    if set(labels) != set(have):
        missing = sorted(set(labels) - set(have))
        extra = sorted(set(have) - set(labels))
        raise ValueError(
            f"panel units do not match spec.unit_labels: missing {missing}, unexpected {extra}"
        )
    outcome_column, outcome_entity = roles.outcome
    if spec.outcome.name not in (outcome_column, outcome_entity.name):
        raise ValueError(
            f"spec outcome {spec.outcome.name!r} is neither the panel's outcome column "
            f"{outcome_column!r} nor its entity {outcome_entity.name!r}"
        )
    if dimension_of(outcome_entity) != spec.outcome_dimension:
        raise DimensionError(
            f"outcome {spec.outcome.name!r}: panel declares {dimension_of(outcome_entity)}, "
            f"spec declares {spec.outcome_dimension}"
        )
    _require_same_unit(f"outcome {spec.outcome.name!r}", outcome_entity.unit, spec.outcome.unit)
    periods = panel.periods
    if spec.carried:
        _require_equispaced(spec, periods)
    resolved = resolve_conventions(spec, panel, conventions)
    frame = _resolved_nuisance(spec, resolved).augment(panel.frame, roles.time)
    out: DataDict = {
        spec.unit_column: np.arange(len(labels), dtype=np.int64)[:, None],
        spec.time_column: np.tile(_numeric_periods(periods), (len(labels), 1)),
        spec.outcome.name: _grid(frame, roles, labels, periods, outcome_column),
    }
    for t in spec.treatments:
        column = _treatment_column(roles, t.name)
        if roles.dimension_of(column) != dimension_of(t):
            raise DimensionError(
                f"treatment {t.name!r}: panel column {column!r} carries "
                f"{roles.dimension_of(column)}, spec declares {dimension_of(t)}"
            )
        _require_same_unit(
            f"treatment {t.name!r} (panel column {column!r})", roles.unit_of(column), t.unit
        )
        out[t.name] = _grid(frame, roles, labels, periods, column)
    for column in spec.nuisance.column_names():
        out[column] = _grid(frame, roles, labels, periods, column)
    return out


# -- fit ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class FitResult:
    """What ``fit`` returns: the surface, its posterior (or a typed failure), and the record.

    ``posterior`` is ``Unsupported`` when the backend is unavailable and
    ``Unverified`` when the backend declined to certify its draws; it is
    never a posterior that was not earned. ``report`` holds MCMC diagnostics
    for sampled posteriors and is ``None`` for Laplace draws (independent by
    construction) and for failures. ``data`` is what ``prepare`` produced.
    ``provenance["nuisance_conventions"]`` is what ``prepare`` resolved the
    trend terms with; pass it back as ``prepare(..., conventions=...)`` to
    lay out a prediction panel on the same basis.
    """

    surface: Surface
    posterior: Posterior | Unverified | Unsupported
    data: Mapping[str, npt.NDArray[Any]]
    report: ConvergenceReport | None
    provenance: dict[str, Any]

    @property
    def converged(self) -> bool:
        """MCMC: the report's verdict. Laplace: the mode converged and the Hessian was PD."""
        if not isinstance(self.posterior, Posterior):
            return False
        if self.report is not None:
            return self.report.converged
        prov = self.posterior.provenance
        return bool(prov.get("converged", False)) and bool(prov.get("hessian_pd", True))


def fit(
    spec: SurfaceSpec,
    panel: Panel,
    *,
    backend: Backend | str = "laplace",
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    seed: int | None = None,
) -> FitResult:
    """Prepare the panel, sample the built model through ``backend``, and record everything.

    A backend named by string is resolved with ``axiom.infer.get_backend``;
    a missing extra comes back as ``posterior=Unsupported``, and a backend
    that declines to certify its draws as ``posterior=Unverified`` with
    ``report=None``. Backend errors are not caught. ``provenance`` carries
    the spec, model, and panel hashes, the backend name, the sizes, the
    seed, and the nuisance conventions the panel was laid out with.
    """
    surface = Surface(spec)
    conventions = resolve_conventions(spec, panel)
    data = prepare(spec, panel, conventions=conventions)
    completeness = panel.completeness()
    provenance: dict[str, Any] = {
        "spec_hash": spec.content_hash(),
        "model_hash": surface.model.content_hash(),
        "panel_hash": panel.content_hash(),
        "n_units": completeness.n_units,
        "n_periods": completeness.n_periods,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "seed": seed,
        "nuisance_conventions": conventions,
    }
    if isinstance(backend, str):
        resolved = get_backend(backend)
        if isinstance(resolved, Unsupported):
            provenance["backend"] = backend
            return FitResult(surface, resolved, data, None, provenance)
        backend = resolved
    provenance["backend"] = backend.name
    posterior: Posterior | Unverified | Unsupported = backend.sample(
        surface.model, data, draws=draws, tune=tune, chains=chains, seed=seed
    )
    report: ConvergenceReport | None = None
    if isinstance(posterior, Posterior) and posterior.provenance.get("method") != "laplace":
        report = diagnose(posterior)
    return FitResult(surface, posterior, data, report, provenance)
