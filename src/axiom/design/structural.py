"""Fisher information of a response surface's parameters at a design, through ``forward()``.

The parent's ``planning/identification.py`` byte-mirrored the model's forward
op in numpy and drifted from it. Here nothing is re-implemented: the
Jacobian of the mean with respect to the **linear** parameters is exactly
the design matrix ``surface.linearize`` returns (rule 3 — it is built by
evaluating the tree with unit vectors), and the columns for the
**nonlinear** parameters are derivatives of ``surface.forward`` itself —
``jax.jacfwd`` over ``core.compile_jax(surface.expr)`` when jax is present
with 64-bit mode on, central finite differences of ``forward()`` otherwise.
A Gaussian likelihood with scale ``noise_sd`` then gives

    FI = J' J / noise_sd²

and any exponential family gives the same thing with one diagonal weight
per row, ``FI = J' W J``, ``w_i = 1 / (phi V(mu_i))`` — a ``Weighting``
(see ``design.weighting``, which derives the variance functions). The
Gaussian case is ``W = I / noise_sd²``; nothing downstream of the matrix
knows or needs to know which family produced it.

and the Laplace approximation to the expected posterior covariance under
independent Gaussian priors is ``(diag(1 / prior_sd²) + FI)⁻¹``. The
design that identifies a target parameter best is the one that minimizes
that covariance's diagonal entry for it; ``design_to_identify`` searches a
candidate set for it by point exchange, exploiting the additivity of the
information over independent rows.

Scale conventions
-----------------
* Finite-difference steps are relative to each parameter's own magnitude,
  ``h = 1e-5 · max(|θ_j|, scale_j)``; a parameter sitting exactly at zero
  with no ``parameter_scales`` entry has no scale, which is ``Unsupported``,
  not a guess. A whole column whose finite differences are below ``100``
  times their round-off floor ``eps · max|f| / h`` is reported as exactly
  zero — the mean does not move with that parameter at this design (a
  carryover decay under a constant schedule) — and named in
  ``detail["round_off_columns"]``.
* Singularity is decided on the correlation-form matrix ``D⁻¹ M D⁻¹``
  (``D = sqrt(diag M)``), so the decision is independent of the units the
  parameters are measured in: an eigenvalue below ``1e-12`` of the
  largest, or a zero diagonal entry, is singular. ``det`` and
  ``min_eigenvalue`` are reported for the matrix in the caller's units.

The numbers depend on ``theta`` (review B11: information on a nonlinear
surface is local); average over prior draws by calling these functions per
draw.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from axiom.core import (
    DesignMatrix,
    NonEmptyStr,
    Spec,
    SupportsForward,
    Unsupported,
    compile_jax,
    jax_available,
    require_jax,
)
from axiom.design.weighting import Weighting
from axiom.surface import Design

__all__ = [
    "DerivativeMethod",
    "FisherInformation",
    "IdentifiabilityRidge",
    "IdentifyingDesign",
    "design_to_identify",
    "expected_posterior_sd",
    "fisher_information",
    "identifiability_ridge",
    "ridge_of",
]

Array = npt.NDArray[np.float64]
Theta = Mapping[str, npt.ArrayLike]
DerivativeMethod = Literal["auto", "jax", "finite"]
Source = Literal["jax", "finite"]

_REL_STEP = 1e-5  # relative central-difference step, near eps ** (1/3)
_RESOLUTION = 100.0  # a finite difference must exceed this multiple of its round-off floor
_SINGULAR_REL = 1e-12  # eigenvalue floor, relative to the largest, on the correlation form
_EPS = float(np.finfo(np.float64).eps)


# -- specs -------------------------------------------------------------------------------


class FisherInformation(Spec):
    """``J' W J / noise_sd²`` for the named parameters at one design and one ``theta``.

    ``parameters`` are the scalar coordinates: a bare name for a scalar, and
    ``name[i]`` per element of a vector parameter, in ``linearize`` column
    order followed by the nonlinear parameters. ``det`` and
    ``min_eigenvalue`` are for ``matrix`` in the caller's units; ``singular``
    is decided on the correlation form (module docstring) and is what the
    inverse-based quantities key on. ``n_observations`` is the number of
    rows the design contributed.
    """

    parameters: tuple[NonEmptyStr, ...] = Field(min_length=1)
    matrix: tuple[tuple[float, ...], ...]
    noise_sd: float = Field(gt=0)
    weighting: Weighting = Weighting()
    n_observations: int = Field(ge=1)
    det: float
    min_eigenvalue: float
    singular: bool
    method: Source
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _square(self) -> FisherInformation:
        p = len(self.parameters)
        if len(self.matrix) != p or any(len(row) != p for row in self.matrix):
            raise ValueError(f"matrix must be {p}x{p} to match {p} parameters")
        for row in self.matrix:
            if not all(math.isfinite(v) for v in row):
                raise ValueError("matrix entries must be finite")
        return self

    @property
    def p(self) -> int:
        return len(self.parameters)

    def as_array(self) -> Array:
        return np.asarray(self.matrix, dtype=np.float64).reshape(self.p, self.p)

    def index(self, name: str) -> int:
        """Position of a parameter coordinate; ``KeyError`` naming it when absent."""
        try:
            return self.parameters.index(name)
        except ValueError:
            raise KeyError(
                f"no parameter {name!r} in the information matrix; have {list(self.parameters)}"
            ) from None

    def covariance(self) -> Array | Unsupported:
        """``FI⁻¹`` — the flat-prior Laplace covariance; ``Unsupported`` when singular."""
        return _inverse(self.as_array(), self.parameters, "Fisher information")

    def __add__(self, other: FisherInformation) -> FisherInformation:
        """Information is additive over independent observations at the same ``theta``."""
        if self.parameters != other.parameters:
            raise ValueError("can only add information matrices over the same parameters")
        if self.noise_sd != other.noise_sd:
            raise ValueError("can only add information matrices with the same noise_sd")
        if self.weighting != other.weighting:
            raise ValueError(
                "can only add information matrices with the same weighting; "
                f"{self.weighting.label!r} and {other.weighting.label!r} are different "
                "likelihoods and their rows are not in the same units of information"
            )
        return _fisher_from_matrix(
            self.as_array() + other.as_array(),
            self.parameters,
            self.noise_sd,
            self.n_observations + other.n_observations,
            self.method if self.method == other.method else "finite",
            {**other.detail, **self.detail},
            self.weighting,
        )


class IdentifiabilityRidge(Spec):
    """The flattest direction of the information matrix and the couplings it implies.

    ``direction`` is the unit eigenvector of the smallest eigenvalue of the
    correlation-form matrix, keyed by parameter, signed so its largest
    component is positive — the combination of parameters the design
    moves least (the ``beta``/``k`` equifinality ridge of a saturating
    kernel). ``correlations`` are the flat-prior posterior correlations
    ``cov_ij / sqrt(cov_ii cov_jj)`` with ``cov = FI⁻¹``, one per requested
    pair in the order of ``pairs``; ``condition_number`` is that of the
    correlation form (``inf`` when singular).
    """

    parameters: tuple[NonEmptyStr, ...] = Field(min_length=1)
    direction: dict[str, float]
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    pairs: tuple[tuple[NonEmptyStr, NonEmptyStr], ...] = ()
    correlations: tuple[float, ...] = ()
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _aligned(self) -> IdentifiabilityRidge:
        if set(self.direction) != set(self.parameters):
            raise ValueError("direction must have one entry per parameter")
        if len(self.correlations) != len(self.pairs):
            raise ValueError("one correlation per pair")
        return self

    @property
    def ridge_parameters(self) -> tuple[str, ...]:
        """Parameters carrying at least ``1/sqrt(2p)`` of the direction's weight, by weight."""
        floor = 1.0 / math.sqrt(2.0 * len(self.parameters))
        heavy = [(abs(v), k) for k, v in self.direction.items() if abs(v) >= floor]
        return tuple(k for _, k in sorted(heavy, reverse=True))


class IdentifyingDesign(Spec):
    """The ``n`` candidate rows that minimize a target parameter's expected posterior sd.

    ``design`` holds the chosen rows (replicates allowed); ``indices`` are
    their positions in the candidate design; ``expected_sd`` is the
    Laplace posterior sd of ``target`` under the prior sds given (flat for
    a parameter without one) and ``expected_sds`` the same for every
    parameter. ``prior_sds`` records what was assumed.
    """

    target: NonEmptyStr
    design: Design
    indices: tuple[int, ...] = Field(min_length=1)
    expected_sd: float = Field(gt=0)
    expected_sds: dict[str, float]
    prior_sds: dict[str, float] = {}
    noise_sd: float = Field(gt=0)
    weighting: Weighting = Weighting()
    seed: int | None = None
    n_restarts: int = Field(ge=1)
    passes: int = Field(ge=0)
    detail: dict[str, str] = {}


# -- linear algebra helpers --------------------------------------------------------------


def _correlation_form(M: Array) -> tuple[Array, Array] | None:
    """``(D⁻¹ M D⁻¹, D)`` with ``D = sqrt(diag M)``; ``None`` when a diagonal entry is not > 0."""
    d = np.sqrt(np.diag(M))
    if not bool(np.all(np.isfinite(d))) or bool(np.any(d <= 0.0)):
        return None
    return M / np.outer(d, d), d


def _is_singular(M: Array) -> bool:
    form = _correlation_form(M)
    if form is None:
        return True
    C, _ = form
    w = np.linalg.eigvalsh(0.5 * (C + C.T))
    if not bool(np.all(np.isfinite(w))):
        return True
    return bool(w[0] <= _SINGULAR_REL * max(float(w[-1]), 0.0))


def _inverse(M: Array, names: Sequence[str], what: str) -> Array | Unsupported:
    """Inverse through the correlation form; ``Unsupported`` naming the flat coordinates."""
    form = _correlation_form(M)
    if form is None:
        zero = [n for n, v in zip(names, np.diag(M), strict=True) if not v > 0.0]
        return Unsupported(
            reason=(
                f"{what} carries no information on {zero}: the diagonal entry is zero, so the "
                "matrix cannot be inverted; change the design or give a prior sd"
            ),
            detail={"zero_diagonal": ", ".join(zero)},
        )
    C, d = form
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    if not bool(np.all(np.isfinite(w))) or bool(w[0] <= _SINGULAR_REL * max(float(w[-1]), 0.0)):
        direction = V[:, 0]
        floor = 1.0 / math.sqrt(2.0 * len(names))
        heavy = [names[i] for i in np.argsort(-np.abs(direction)) if abs(direction[i]) >= floor]
        rel = float(w[0]) / max(float(w[-1]), _EPS)
        flat = ", ".join(f"{n}={v:.3g}" for n, v in zip(names, direction, strict=True))
        return Unsupported(
            reason=(
                f"{what} is singular (smallest relative eigenvalue {rel:.3g} of the correlation "
                f"form); the flat direction loads on {heavy} — the design does not separate "
                "these parameters"
            ),
            detail={"flat_direction": flat, "min_relative_eigenvalue": f"{rel:.6g}"},
        )
    Cinv = (V / w) @ V.T
    out: Array = Cinv / np.outer(d, d)
    return out


def _fisher_from_matrix(
    M: Array,
    names: tuple[str, ...],
    noise_sd: float,
    n_obs: int,
    method: Source,
    detail: Mapping[str, str],
    weighting: Weighting | None = None,
) -> FisherInformation:
    M = 0.5 * (M + M.T)
    w = np.linalg.eigvalsh(M)
    sign, logdet = np.linalg.slogdet(M)
    det = float(sign * np.exp(logdet)) if sign != 0 else 0.0
    return FisherInformation(
        parameters=names,
        matrix=tuple(tuple(float(v) for v in row) for row in M),
        noise_sd=noise_sd,
        weighting=weighting if weighting is not None else Weighting(),
        n_observations=n_obs,
        det=det,
        min_eigenvalue=float(w[0]),
        singular=_is_singular(M),
        method=method,
        detail=dict(detail),
    )


# -- the Jacobian ------------------------------------------------------------------------


def _x64() -> bool:
    import jax

    return bool(getattr(jax.config, "jax_enable_x64", False))


def _use_jax(method: DerivativeMethod) -> bool:
    if method == "finite":
        return False
    if method == "jax":
        missing = require_jax()
        if missing is not None:
            raise ValueError(missing.reason)
        if not _x64():
            raise ValueError(
                "method='jax' needs jax_enable_x64=True: float32 derivatives are too coarse for "
                "an information matrix; enable x64 or use method='finite'"
            )
        return True
    if method != "auto":
        raise ValueError(f"method must be 'auto', 'jax', or 'finite'; got {method!r}")
    return jax_available() and _x64()


def _coordinates(name: str, shape: tuple[int, ...]) -> list[tuple[str, tuple[int, ...]]]:
    if not shape:
        return [(name, ())]
    return [
        (f"{name}[{','.join(str(i) for i in idx)}]", tuple(int(i) for i in idx))
        for idx in np.ndindex(*shape)
    ]


def _forward_flat(surface: SupportsForward, dose: Theta, theta: Theta) -> Array:
    return np.asarray(surface.forward(dose, theta), dtype=np.float64).reshape(-1)


def _finite_columns(
    surface: SupportsForward,
    dose: Theta,
    theta: dict[str, Array],
    nonlinear: Sequence[str],
    scales: Mapping[str, float],
    n_rows: int,
) -> tuple[Array, tuple[str, ...], list[str]] | Unsupported:
    """Central differences of ``forward`` in each nonlinear coordinate; round-off columns zeroed."""
    names: list[str] = []
    cols: list[Array] = []
    round_off: list[str] = []
    for name in nonlinear:
        base = theta[name]
        for coord, idx in _coordinates(name, base.shape):
            x0 = float(base[idx]) if idx else float(base.reshape(-1)[0])
            scale = max(abs(x0), float(scales.get(coord, scales.get(name, 0.0))))
            if not scale > 0.0:
                return Unsupported(
                    reason=(
                        f"parameter {coord!r} is exactly zero and has no entry in "
                        "parameter_scales, so no finite-difference step is relative to a "
                        "scale; pass parameter_scales={...} or use method='jax'"
                    ),
                    detail={"parameter": coord},
                )
            h = _REL_STEP * scale
            plus, minus = base.copy(), base.copy()
            if idx:
                plus[idx] = x0 + h
                minus[idx] = x0 - h
            else:
                plus = np.asarray(x0 + h, dtype=np.float64).reshape(base.shape)
                minus = np.asarray(x0 - h, dtype=np.float64).reshape(base.shape)
            f_plus = _forward_flat(surface, dose, {**theta, name: plus})
            f_minus = _forward_flat(surface, dose, {**theta, name: minus})
            if f_plus.size != n_rows or f_minus.size != n_rows:
                raise ValueError(
                    f"forward() returned {f_plus.size} values when perturbing {coord!r}; "
                    f"the design has {n_rows} rows"
                )
            col = (f_plus - f_minus) / (2.0 * h)
            if not bool(np.all(np.isfinite(col))):
                return Unsupported(
                    reason=f"forward() is not finite when {coord!r} is perturbed by ±{h:.3g}",
                    detail={"parameter": coord, "step": f"{h:.6g}"},
                )
            floor = _EPS * float(np.max(np.abs(np.concatenate([f_plus, f_minus])))) / h
            if float(np.max(np.abs(col))) < _RESOLUTION * floor:
                col = np.zeros_like(col)
                round_off.append(coord)
            names.append(coord)
            cols.append(col)
    J = np.stack(cols, axis=1) if cols else np.zeros((n_rows, 0), dtype=np.float64)
    return J, tuple(names), round_off


def _jax_columns(
    surface: SupportsForward,
    dose: Theta,
    theta: dict[str, Array],
    nonlinear: Sequence[str],
    n_rows: int,
) -> tuple[Array, tuple[str, ...]] | Unsupported:
    import jax
    import jax.numpy as jnp

    f = compile_jax(surface.expr)
    data = {k: np.asarray(v) for k, v in dose.items()}
    params: dict[str, Any] = {k: jnp.asarray(v) for k, v in theta.items()}
    names: list[str] = []
    cols: list[Array] = []
    for name in nonlinear:
        shape = theta[name].shape

        def g(v: Any, _name: str = name, _shape: tuple[int, ...] = shape) -> Any:
            return jnp.reshape(f(data, {**params, _name: jnp.reshape(v, _shape)}), (-1,))

        J = np.asarray(jax.jacfwd(g)(jnp.asarray(theta[name].reshape(-1))), dtype=np.float64)
        if J.shape[0] != n_rows:
            raise ValueError(
                f"the jax forward returned {J.shape[0]} values for {name!r}; the design has "
                f"{n_rows} rows"
            )
        if not bool(np.all(np.isfinite(J))):
            return Unsupported(
                reason=f"the jax derivative of forward() in {name!r} is not finite at theta",
                detail={"parameter": name},
            )
        for j, (coord, _) in enumerate(_coordinates(name, shape)):
            names.append(coord)
            cols.append(J[:, j])
    Jn = np.stack(cols, axis=1) if cols else np.zeros((n_rows, 0), dtype=np.float64)
    return Jn, tuple(names)


def _jacobian(
    surface: SupportsForward,
    dose: Theta,
    theta: Theta,
    method: DerivativeMethod,
    parameter_scales: Mapping[str, float] | None,
) -> tuple[Array, tuple[str, ...], Source, dict[str, str]] | Unsupported:
    dm: DesignMatrix = surface.linearize(dose, theta)
    n_rows = int(dm.offset.size)
    full = {k: np.asarray(v, dtype=np.float64) for k, v in theta.items()}
    nonlinear = tuple(dm.at)
    missing = [n for n in nonlinear if n not in full]
    if missing:
        raise KeyError(f"theta lacks nonlinear parameters {missing}")
    use_jax = _use_jax(method)
    detail: dict[str, str] = {}
    if use_jax:
        got = _jax_columns(surface, dose, full, nonlinear, n_rows)
        if isinstance(got, Unsupported):
            return got
        J_nl, names_nl = got
        source: Source = "jax"
    else:
        fd = _finite_columns(surface, dose, full, nonlinear, parameter_scales or {}, n_rows)
        if isinstance(fd, Unsupported):
            return fd
        J_nl, names_nl, round_off = fd
        if round_off:
            detail["round_off_columns"] = ", ".join(round_off)
        source = "finite"
    J = np.concatenate([dm.X, J_nl], axis=1)
    names = tuple(dm.columns) + names_nl
    if len(set(names)) != len(names):
        raise ValueError(f"parameter coordinates are not distinct: {names}")
    return J, names, source, detail


# -- public ------------------------------------------------------------------------------


def fisher_information(
    surface: SupportsForward,
    design_doses: Theta,
    theta: Theta,
    noise_sd: float = 1.0,
    *,
    weighting: Weighting | None = None,
    parameters: Sequence[str] | None = None,
    method: DerivativeMethod = "auto",
    parameter_scales: Mapping[str, float] | None = None,
) -> FisherInformation | Unsupported:
    """``J' W J / noise_sd²`` at ``design_doses`` and ``theta`` for the mean's parameters.

    ``design_doses`` is the data mapping ``forward`` reads (every column the
    mean needs — treatment doses, the unit index, nuisance columns) laid out
    as the surface requires; ``theta`` holds every parameter of the mean.
    Linear columns come from ``surface.linearize`` exactly; nonlinear ones
    from ``jax.jacfwd`` over ``compile_jax(surface.expr)`` (``method="jax"``,
    or ``"auto"`` with jax installed and x64 on) or central finite
    differences of ``forward`` (``"finite"``, the ``"auto"`` fallback).
    ``parameters`` restricts and orders the coordinates reported (a name
    not in the mean is a ``ValueError``); ``parameter_scales`` gives a
    finite-difference scale to a parameter sitting at zero.

    ``weighting`` gives the likelihood a non-Gaussian variance function:
    ``W = diag(w_i)`` with ``w_i = 1 / (phi V(mu_i))`` evaluated at this
    design's own mean, so a Poisson or binomial design is weighted where it
    sits rather than at some reference point. The default ``W = I`` with
    ``noise_sd`` is the homoscedastic Gaussian case and is unchanged. The
    two are alternatives, not layers: passing a ``weighting`` *and* a
    ``noise_sd`` other than 1 would apply a dispersion twice, and raises.

    Returns ``Unsupported`` when a derivative cannot be formed honestly: a
    zero parameter with no scale, or a non-finite forward under perturbation.
    """
    if not noise_sd > 0.0 or not math.isfinite(noise_sd):
        raise ValueError(f"noise_sd must be positive and finite, got {noise_sd}")
    if weighting is not None and noise_sd != 1.0:
        raise ValueError(
            f"a {weighting.family} weighting already carries its dispersion; pass either "
            f"weighting= or noise_sd=, not both (got noise_sd={noise_sd})"
        )
    got = _jacobian(surface, design_doses, theta, method, parameter_scales)
    if isinstance(got, Unsupported):
        return got
    J, names, source, detail = got
    if parameters is not None:
        wanted = tuple(parameters)
        unknown = [p for p in wanted if p not in names]
        if unknown:
            raise ValueError(f"parameters {unknown} are not coordinates of the mean; have {names}")
        if len(set(wanted)) != len(wanted):
            raise ValueError("parameters must be distinct")
        pos = [names.index(p) for p in wanted]
        J = J[:, pos]
        names = wanted
    if not names:
        raise ValueError("the mean has no parameters to inform")
    if weighting is None or weighting.is_unit:
        M = (J.T @ J) / (noise_sd**2)
    else:
        # The weights are a function of the mean at *this* design, so they are
        # computed here rather than handed in. J is the unweighted Jacobian of
        # the mean; the weighting never touches the derivative.
        try:
            w = np.broadcast_to(
                np.asarray(
                    weighting.at(_forward_flat(surface, design_doses, theta), design_doses),
                    dtype=np.float64,
                ).ravel(),
                (J.shape[0],),
            )
        except ValueError as exc:
            return Unsupported(
                reason=f"the {weighting.family} weighting is undefined at this design: {exc}",
                detail={"weighting": weighting.label},
            )
        M = J.T @ (w[:, None] * J)
        detail = {**detail, "weighting": weighting.label}
    return _fisher_from_matrix(M, names, noise_sd, int(J.shape[0]), source, detail, weighting)


def _prior_precision(
    prior_sds: Mapping[str, float] | None, names: Sequence[str]
) -> tuple[Array, dict[str, float]]:
    """``diag(1 / sd²)``; a coordinate ``name[i]`` falls back to ``name``; missing is flat."""
    given = dict(prior_sds or {})
    prec = np.zeros(len(names), dtype=np.float64)
    used: dict[str, float] = {}
    for j, coord in enumerate(names):
        base = coord.partition("[")[0]
        sd = given.get(coord, given.get(base))
        if sd is None:
            continue
        if not (math.isfinite(sd) and sd > 0.0):
            raise ValueError(f"prior sd for {coord!r} must be positive and finite, got {sd}")
        prec[j] = 1.0 / sd**2
        used[coord] = float(sd)
    unknown = sorted(
        k for k in given if k not in names and k not in {c.partition("[")[0] for c in names}
    )
    if unknown:
        raise ValueError(f"prior_sds name parameters not in the information matrix: {unknown}")
    return prec, used


def expected_posterior_sd(
    prior_sds: Mapping[str, float] | None, fisher: FisherInformation
) -> dict[str, float] | Unsupported:
    """Laplace posterior sds ``sqrt(diag((diag(1/sd²) + FI)⁻¹))`` per parameter coordinate.

    A parameter without a prior sd is treated as flat (zero precision); a
    coordinate ``name[i]`` reads ``prior_sds[name]`` when no entry names it
    directly. ``Unsupported`` when the posterior precision is singular —
    some direction has neither prior nor design information.
    """
    prec, _ = _prior_precision(prior_sds, fisher.parameters)
    M = fisher.as_array() + np.diag(prec)
    cov = _inverse(M, fisher.parameters, "posterior precision (prior + Fisher information)")
    if isinstance(cov, Unsupported):
        return cov
    var = np.diag(cov)
    if bool(np.any(var <= 0.0)):
        return Unsupported(
            reason="the inverted posterior precision has a non-positive variance; the matrix is "
            "numerically indefinite at this design",
            detail={"variances": ", ".join(f"{v:.3g}" for v in var)},
        )
    return {n: float(math.sqrt(v)) for n, v in zip(fisher.parameters, var, strict=True)}


def identifiability_ridge(
    surface: SupportsForward,
    design_doses: Theta,
    theta: Theta,
    noise_sd: float = 1.0,
    *,
    weighting: Weighting | None = None,
    pairs: Sequence[tuple[str, str]] = (),
    parameters: Sequence[str] | None = None,
    method: DerivativeMethod = "auto",
    parameter_scales: Mapping[str, float] | None = None,
) -> IdentifiabilityRidge | Unsupported:
    """The flattest direction of the information matrix at this design, and pairwise couplings.

    The direction is the smallest-eigenvalue eigenvector of the correlation
    form of the Fisher matrix (scale-free: the units of ``beta`` and ``k``
    do not decide it). For each pair the flat-prior posterior correlation
    is reported; that needs the matrix to be nonsingular, so a singular
    design with ``pairs`` requested is ``Unsupported`` (the pair is then
    perfectly confounded, which the direction already says).
    """
    fi = fisher_information(
        surface,
        design_doses,
        theta,
        noise_sd,
        weighting=weighting,
        parameters=parameters,
        method=method,
        parameter_scales=parameter_scales,
    )
    if isinstance(fi, Unsupported):
        return fi
    return ridge_of(fi, pairs=pairs)


def ridge_of(
    fisher: FisherInformation, *, pairs: Sequence[tuple[str, str]] = ()
) -> IdentifiabilityRidge | Unsupported:
    """``identifiability_ridge`` from an information matrix already computed."""
    names = fisher.parameters
    M = fisher.as_array()
    form = _correlation_form(M)
    if form is None:
        zero = [n for n, v in zip(names, np.diag(M), strict=True) if not v > 0.0]
        direction = {n: (1.0 if n == zero[0] else 0.0) for n in names}
        w_min, w_max = 0.0, float(np.max(np.diag(M)))
        C_cond = math.inf
    else:
        C, _ = form
        w, V = np.linalg.eigh(0.5 * (C + C.T))
        v = V[:, 0]
        v = v * (1.0 if v[int(np.argmax(np.abs(v)))] >= 0.0 else -1.0)
        direction = {n: float(x) for n, x in zip(names, v, strict=True)}
        w_min, w_max = float(max(w[0], 0.0)), float(w[-1])
        C_cond = math.inf if w_min <= 0.0 else w_max / w_min
    correlations: list[float] = []
    if pairs:
        for first, second in pairs:
            fisher.index(first)
            fisher.index(second)
        cov = fisher.covariance()
        if isinstance(cov, Unsupported):
            return Unsupported(
                reason=(
                    "pairwise correlations need a nonsingular information matrix; " + cov.reason
                ),
                detail=cov.detail,
            )
        for first, second in pairs:
            i, j = fisher.index(first), fisher.index(second)
            correlations.append(float(cov[i, j] / math.sqrt(cov[i, i] * cov[j, j])))
    return IdentifiabilityRidge(
        parameters=names,
        direction=direction,
        min_eigenvalue=w_min,
        max_eigenvalue=w_max,
        condition_number=C_cond,
        pairs=tuple((a, b) for a, b in pairs),
        correlations=tuple(correlations),
        detail={"noise_sd": f"{fisher.noise_sd:.6g}", "n_observations": str(fisher.n_observations)},
    )


def _target_sd(M: Array, t: int) -> float | None:
    """``sqrt([M⁻¹]_tt)`` or ``None`` when ``M`` is singular (correlation-form decision)."""
    form = _correlation_form(M)
    if form is None:
        return None
    C, d = form
    C = 0.5 * (C + C.T)
    try:
        w = np.linalg.eigvalsh(C)
    except np.linalg.LinAlgError:
        return None
    if not bool(np.all(np.isfinite(w))) or bool(w[0] <= _SINGULAR_REL * max(float(w[-1]), 0.0)):
        return None
    e = np.zeros(C.shape[0])
    e[t] = 1.0
    var = float(e @ np.linalg.solve(C, e)) / float(d[t] ** 2)
    return math.sqrt(var) if var > 0.0 else None


def design_to_identify(
    surface: SupportsForward,
    candidates: Design,
    theta: Theta,
    noise_sd: float,
    target: str,
    n: int,
    *,
    weighting: Weighting | None = None,
    prior_sds: Mapping[str, float] | None = None,
    seed: int | None = None,
    n_restarts: int = 3,
    max_passes: int = 20,
    method: DerivativeMethod = "auto",
    parameter_scales: Mapping[str, float] | None = None,
) -> IdentifyingDesign | Unsupported:
    """The ``n`` candidate rows (replicates allowed) minimizing ``target``'s expected posterior sd.

    Each candidate row is an independent observation, so its information
    matrix is computed once (through ``fisher_information`` on that row)
    and a design's information is the sum over its rows plus the prior
    precision. Point exchange from ``n_restarts`` seeded random starts:
    every slot is tried against every candidate until a pass improves
    nothing or ``max_passes`` is reached. A surface with carryover must be
    handed in as ``steady_state()`` (its ``forward`` refuses 1-D rows).

    ``Unsupported`` when no design of ``n`` rows makes the posterior
    precision nonsingular (too few rows for the flat-prior parameters), or
    when a row's derivatives cannot be formed.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if n_restarts < 1 or max_passes < 1:
        raise ValueError("n_restarts and max_passes must be positive")
    rows = candidates.as_array()
    per_row: list[Array] = []
    names: tuple[str, ...] | None = None
    source: Source = "finite"
    round_off: set[str] = set()
    for i in range(candidates.n):
        dose = {
            t: np.asarray([rows[i, j]], dtype=np.float64)
            for j, t in enumerate(candidates.treatments)
        }
        fi = fisher_information(
            surface,
            dose,
            theta,
            noise_sd,
            weighting=weighting,
            method=method,
            parameter_scales=parameter_scales,
        )
        if isinstance(fi, Unsupported):
            where = dict(zip(candidates.treatments, rows[i], strict=True))
            return Unsupported(
                reason=f"candidate row {i} ({where}): " + fi.reason,
                detail={**fi.detail, "candidate": str(i)},
            )
        if names is None:
            names = fi.parameters
            source = fi.method
        elif fi.parameters != names:
            raise ValueError(
                f"candidate row {i} reports parameters {fi.parameters}, row 0 reported {names}"
            )
        if "round_off_columns" in fi.detail:
            round_off.update(fi.detail["round_off_columns"].split(", "))
        per_row.append(fi.as_array())
    assert names is not None
    if target not in names:
        raise ValueError(
            f"target {target!r} is not a parameter coordinate of the mean; have {names}"
        )
    t = names.index(target)
    prec, used = _prior_precision(prior_sds, names)
    P0 = np.diag(prec)
    stack = np.stack(per_row)  # (n_candidates, p, p)

    def sd_of(counts: Array) -> float | None:
        M = P0 + np.tensordot(counts, stack, axes=(0, 0))
        return _target_sd(M, t)

    rng = np.random.default_rng(seed)
    best_counts: Array | None = None
    best_sd = math.inf
    best_passes = 0
    for _ in range(n_restarts):
        pick = rng.integers(0, candidates.n, size=n)
        counts = np.bincount(pick, minlength=candidates.n).astype(np.float64)
        current = sd_of(counts)
        passes = 0
        improved = True
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for slot in np.flatnonzero(counts > 0):
                for cand in range(candidates.n):
                    if cand == slot:
                        continue
                    trial = counts.copy()
                    trial[slot] -= 1.0
                    trial[cand] += 1.0
                    sd = sd_of(trial)
                    if sd is None:
                        continue
                    if current is None or sd < current * (1.0 - 1e-12):
                        counts, current = trial, sd
                        improved = True
                        break
                if improved:
                    break
        if current is not None and current < best_sd:
            best_counts, best_sd, best_passes = counts, current, passes
    if best_counts is None:
        return Unsupported(
            reason=(
                f"no design of {n} rows from the {candidates.n} candidates makes the posterior "
                f"precision for {list(names)} nonsingular; add rows, widen the candidate set, or "
                "give prior sds for the parameters the rows cannot separate"
            ),
            detail={"target": target, "n": str(n), "parameters": ", ".join(names)},
        )
    M = P0 + np.tensordot(best_counts, stack, axes=(0, 0))
    cov = _inverse(M, names, "posterior precision")
    if isinstance(cov, Unsupported):  # pragma: no cover - sd_of already certified M
        return cov
    sds = {nm: float(math.sqrt(v)) for nm, v in zip(names, np.diag(cov), strict=True)}
    indices = tuple(int(i) for i in np.repeat(np.arange(candidates.n), best_counts.astype(int)))
    design = Design(
        treatments=candidates.treatments,
        points=tuple(tuple(float(v) for v in rows[i]) for i in indices),
        kind=f"identify:{target}",
        detail={"expected_sd": best_sd, "n": float(n)},
    )
    detail: dict[str, str] = {"method": source}
    if round_off:
        detail["round_off_columns"] = ", ".join(sorted(round_off))
    return IdentifyingDesign(
        target=target,
        design=design,
        indices=indices,
        expected_sd=best_sd,
        expected_sds=sds,
        prior_sds=used,
        noise_sd=noise_sd,
        weighting=weighting if weighting is not None else Weighting(),
        seed=seed,
        n_restarts=n_restarts,
        passes=best_passes,
        detail=detail,
    )
