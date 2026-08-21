"""Sampler-free Laplace approximation: mode + inverse Hessian, in unconstrained space.

The parent's bug, ported as a specification rather than as code: its Laplace
path took the covariance from BFGS's ``hess_inv`` approximation, which is
not curvature, and produced ``inf`` draws that surfaced downstream as
effects of exactly ``0.0`` and NaN intervals. Here

* the mode is found with ``scipy.optimize.minimize(method="trust-ncg")``
  using the real gradient and Hessian, then **polished with Newton steps on
  the real Hessian** until the Newton decrement ``g·H⁻¹·g`` is below
  ``1e-8``. The decrement is the squared Newton step measured in the
  Hessian metric — posterior standard deviations — so the test is invariant
  to the units of the parameters (review: scipy's absolute ``gtol`` stopped
  far from the mode on non-unit scales). ``trust-exact`` and ``BFGS`` are
  fallbacks for the *mode only*; an optimizer that reports zero iterations
  or lands on an indefinite Hessian (a saddle) is restarted from a seeded
  jitter before the next one is tried;
* ``converged`` means: the polish reached the decrement tolerance **and**
  the Hessian there is positive semi-definite. A saddle is never a mode;
* the covariance is the inverse of the **real Hessian of the negative log
  density at the mode**, recomputed there whatever optimizer found it;
* everything happens in the **unconstrained** parametrization (review B13):
  ``z = unconstrain(theta)``, Hessian in ``z``, draws ``z ~ N(mode, H^-1)``
  mapped back through ``axiom.core.constrain``;
* positive-definiteness is checked with one eigendecomposition, **relative
  to the spectrum**: "numerically PD" means the smallest eigenvalue exceeds
  ``1e-8`` times the largest absolute eigenvalue — a condition number past
  ``1e8`` is a flat direction in disguise. No absolute floor anywhere: a
  Hessian of ``5e-10`` on a parameter of scale ``1e5`` is perfectly PD;
* ``laplace`` returns ``Unverified`` — not a Posterior — when the mode
  search did not converge or the Hessian is not numerically PD. Passing
  ``allow_unverified=True`` returns the Gaussian anyway: eigenvalues below
  the relative floor are clipped **to** that floor (the jitter is relative
  too), ``provenance["hessian_pd"] is False``, and a ``WARNING`` is logged;
* ``nonfinite_draw_frac`` is computed on the *constrained* draws and
  reported on every result. Draws are never filtered.

Derivatives come from jax (``jax.grad`` / ``jax.hessian`` over
``axiom.core.compile_log_density``) when jax is importable, otherwise from
central finite differences on ``axiom.core.log_density``. A finite
difference needs a step, and a step needs a scale; the scale is **taken
from the model, per coordinate**, not assumed to be one: each free
parameter's prior spread in unconstrained coordinates (a normal's
``sigma``; a lognormal's ``sigma``; ``1/sqrt(alpha)`` for a gamma; one for
the log / logit scales of halfnormal, beta and uniform; a ``sigma`` that
names a parent parameter uses the parent's prior's typical size). The step
is ``eps**(1/3) · max(scale_j, 1e-3 · |z_j|)`` for the gradient and
``eps**(1/4) · max(scale_j, 1e-3 · |z_j|)`` for the Hessian, with ``eps``
the float64 machine epsilon. An absolute floor of one (the previous rule)
returned round-off noise as the curvature of a parameter with prior sd
``1e8`` — ``-1.9e-6`` for a true ``5e-10`` — which the polish then read as
a saddle and, after restarts, as a verified Gaussian with the wrong sd.

Finite differences can also fail silently, so at every candidate mode the
diagonal second differences are **checked before they are believed**: each
must exceed the round-off floor ``eps · |f|`` by a factor of 100, and must
agree to 10 % with the second difference at twice the step (a truncation
check). A coordinate that fails either is *unresolved*; ``find_mode``,
``hessian_at`` and ``laplace`` then return ``Unverified`` naming it (the
remedy is to install jax or rescale the parameter) unless
``allow_unverified=True``, in which case the result carries
``hessian_pd=False`` and ``converged=False`` because neither could be
checked.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy import optimize as _opt

from axiom.core.expr import Param, Prior, data_names
from axiom.core.interpret.jax import compile_log_density, jax_available
from axiom.core.model import ModelSpec, constrain, free_parameters, log_density, unconstrain
from axiom.core.posterior import Posterior
from axiom.core.result import Unverified
from axiom.infer.backend import PointEstimate

__all__ = [
    "Derivatives",
    "LaplaceBackend",
    "constrain_draws",
    "find_mode",
    "flat_layout",
    "hessian_at",
    "laplace",
]

log = logging.getLogger(__name__)

Array = npt.NDArray[np.float64]
Derivatives = Literal["auto", "jax", "finite_difference"]
"""Where gradients and Hessians come from. ``auto`` prefers jax when importable."""

_EPS = float(np.finfo(np.float64).eps)
_GRAD_STEP = _EPS ** (1.0 / 3.0)
_HESS_STEP = _EPS ** (1.0 / 4.0)
_RELATIVE_STEP = 1e-3
"""Fraction of ``|z_j|`` that takes over from the prior scale far from the origin."""
_RESOLUTION_FACTOR = 100.0
"""A diagonal second difference must exceed ``eps · |f|`` by this factor to be believed."""
_TRUNCATION_TOL = 0.1
"""Relative disagreement between the step-``h`` and step-``2h`` curvature that is tolerated."""
_PD_RELATIVE_TOL = 1e-8
"""Smallest eigenvalue over largest |eigenvalue| above which a Hessian is numerically PD."""
_NEWTON_SUBSPACE_TOL = 1e-12
"""Relative curvature floor for the Newton step and decrement (Levenberg-style)."""
_NEWTON_TOL = 1e-8
"""Newton decrement ``g·H⁻¹·g`` (squared step in posterior-sd units) that counts as converged."""
_MAX_NEWTON = 50
_MAX_RESTARTS = 2
_RESTART_SEED = 20240817
_OPTIMIZERS: tuple[str, ...] = ("trust-ncg", "trust-exact", "BFGS")
_SEED_MAX = 2**32 - 1
_UNRESOLVED_REMEDY = "install jax or rescale"


# -- layout: dict of unconstrained arrays <-> one flat vector ----------------------------


def flat_layout(model: ModelSpec) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """``(name, shape)`` of every free parameter, in the order they occupy the flat vector."""
    return tuple((p.name, p.shape) for p in free_parameters(model))


def _size(shape: tuple[int, ...]) -> int:
    out = 1
    for n in shape:
        out *= n
    return out


def _flatten(layout: tuple[tuple[str, tuple[int, ...]], ...], z: Mapping[str, Any]) -> Array:
    parts = [np.asarray(z[name], dtype=float).reshape(_size(shape)) for name, shape in layout]
    return np.concatenate(parts) if parts else np.zeros(0)


def _unflatten(layout: tuple[tuple[str, tuple[int, ...]], ...], vec: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    i = 0
    for name, shape in layout:
        k = _size(shape)
        out[name] = vec[i : i + k].reshape(shape) if shape else vec[i]
        i += k
    return out


def _offsets(layout: tuple[tuple[str, tuple[int, ...]], ...]) -> list[int]:
    out: list[int] = []
    i = 0
    for _, shape in layout:
        out.append(i)
        i += _size(shape)
    return out


def _flat_names(layout: tuple[tuple[str, tuple[int, ...]], ...]) -> tuple[str, ...]:
    """One label per flat coordinate: ``name`` for a scalar, ``name[i]`` (C order) for a vector."""
    out: list[str] = []
    for name, shape in layout:
        if shape:
            out.extend(f"{name}[{i}]" for i in range(_size(shape)))
        else:
            out.append(name)
    return tuple(out)


def _model_data(model: ModelSpec, data: Mapping[str, npt.ArrayLike]) -> dict[str, Array]:
    """Only the columns the model reads, as float arrays (integer index columns survive)."""
    names = set(data_names(model.mean)) | {model.outcome.name}
    for c in model.constraints:
        names |= set(data_names(c.expr))
    missing = sorted(n for n in names if n not in data)
    if missing:
        raise KeyError(f"model {model.name!r} needs data columns {missing} that were not supplied")
    out: dict[str, Array] = {}
    for n in names:
        a = np.asarray(data[n])
        out[n] = a if np.issubdtype(a.dtype, np.integer) else np.asarray(a, dtype=float)
    return out


def _fixed_values(model: ModelSpec) -> dict[str, float]:
    return {
        p.name: float(p.prior.hyper["value"])
        for p in model.parameters
        if p.prior is not None and p.prior.family == "fixed"
    }


# -- finite-difference scales, from the priors -------------------------------------------


def _numeric(v: float | str) -> float | None:
    return None if isinstance(v, str) else float(v)


def _positive_or_one(v: float | None) -> float:
    return v if v is not None and np.isfinite(v) and v > 0.0 else 1.0


def _typical_magnitude(prior: Prior) -> float:
    """Typical size of a parameter under ``prior`` in *constrained* coordinates; one if unknown.

    Used when a ``sigma`` hyperparameter names a parent parameter: the
    child's spread is then whatever the parent typically is.
    """
    h = prior.hyper
    match prior.family:
        case "fixed":
            v = _numeric(h["value"])
            return _positive_or_one(abs(v) if v is not None else None)
        case "normal" | "halfnormal":
            return _positive_or_one(_numeric(h["sigma"]))
        case "lognormal":
            mu = _numeric(h["mu"])
            return _positive_or_one(float(np.exp(mu)) if mu is not None else None)
        case "gamma":
            alpha, beta = _numeric(h["alpha"]), _numeric(h["beta"])
            return _positive_or_one(
                alpha / beta if alpha is not None and beta is not None else None
            )
        case _:
            return 1.0


def _z_spread(model: ModelSpec, prior: Prior) -> float:
    """Prior spread of one parameter in its unconstrained coordinate.

    ``normal``: ``sigma``; ``lognormal``: ``sigma`` (the sd of the log);
    ``gamma``: ``1/sqrt(alpha)`` (the sd of the log, to first order);
    ``halfnormal``, ``beta``, ``uniform``: one, the natural unit of the log /
    logit scale. A ``sigma`` that names a parent parameter contributes the
    parent's typical magnitude.
    """
    h = prior.hyper
    match prior.family:
        case "normal" | "lognormal":
            sigma = h["sigma"]
            if isinstance(sigma, str):
                parent = model.parameter(sigma).prior
                return _typical_magnitude(parent) if parent is not None else 1.0
            return _positive_or_one(float(sigma))
        case "gamma":
            alpha = _numeric(h["alpha"])
            return _positive_or_one(1.0 / np.sqrt(alpha) if alpha is not None else None)
        case _:
            return 1.0


def _scales(model: ModelSpec, layout: tuple[tuple[str, tuple[int, ...]], ...]) -> Array:
    """Per-coordinate finite-difference scale for the flat unconstrained vector."""
    parts: list[Array] = []
    for name, shape in layout:
        prior = model.parameter(name).prior
        s = _z_spread(model, prior) if prior is not None else 1.0
        parts.append(np.full(_size(shape), s))
    return np.concatenate(parts) if parts else np.zeros(0)


def _steps(x: Array, scales: Array, step: float) -> Array:
    return step * np.maximum(scales, _RELATIVE_STEP * np.abs(x))


# -- derivatives ---------------------------------------------------------------------


@dataclass(frozen=True)
class _Objective:
    """Negative log density over the flat unconstrained vector, with its derivatives.

    ``unresolved(x)`` names the flat coordinates whose diagonal curvature at
    ``x`` the derivative source cannot stand behind; ``None`` when the
    source is exact (jax).
    """

    f: Callable[[Array], float]
    grad: Callable[[Array], Array]
    hess: Callable[[Array], Array]
    source: str
    unresolved: Callable[[Array], tuple[int, ...]] | None = None


def _resolve_derivatives(derivatives: Derivatives) -> Literal["jax", "finite_difference"]:
    if derivatives == "auto":
        return "jax" if jax_available() else "finite_difference"
    if derivatives == "jax" and not jax_available():
        raise ImportError("derivatives='jax' requested but jax is not installed")
    return derivatives


def _numpy_objective(
    model: ModelSpec,
    data: Mapping[str, Array],
    layout: tuple[tuple[str, tuple[int, ...]], ...],
) -> _Objective:
    scales = _scales(model, layout)

    def f(vec: Array) -> float:
        return -log_density(model, data, _unflatten(layout, np.asarray(vec, dtype=float)))

    def grad(vec: Array) -> Array:
        x = np.asarray(vec, dtype=float)
        g = np.zeros_like(x)
        steps = _steps(x, scales, _GRAD_STEP)
        for i in range(x.size):
            e = np.zeros_like(x)
            e[i] = steps[i]
            g[i] = (f(x + e) - f(x - e)) / (2.0 * steps[i])
        return g

    def hess(vec: Array) -> Array:
        x = np.asarray(vec, dtype=float)
        k = x.size
        H = np.zeros((k, k))
        steps = _steps(x, scales, _HESS_STEP)
        f0 = f(x)
        for i in range(k):
            ei = np.zeros(k)
            ei[i] = steps[i]
            H[i, i] = (f(x + ei) - 2.0 * f0 + f(x - ei)) / steps[i] ** 2
            for j in range(i + 1, k):
                ej = np.zeros(k)
                ej[j] = steps[j]
                H[i, j] = H[j, i] = (
                    f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)
                ) / (4.0 * steps[i] * steps[j])
        return H

    def unresolved(vec: Array) -> tuple[int, ...]:
        """Flat coordinates whose diagonal second difference is round-off or truncation."""
        x = np.asarray(vec, dtype=float)
        steps = _steps(x, scales, _HESS_STEP)
        f0 = f(x)
        bad: list[int] = []
        for i in range(x.size):
            e = np.zeros_like(x)
            e[i] = steps[i]
            fp, fm, fp2, fm2 = f(x + e), f(x - e), f(x + 2.0 * e), f(x - 2.0 * e)
            if not np.all(np.isfinite([f0, fp, fm, fp2, fm2])):
                bad.append(i)
                continue
            d1 = fp - 2.0 * f0 + fm
            # the subtraction loses eps times the largest magnitude it touches
            floor = _EPS * max(abs(f0), abs(fp), abs(fm))
            if abs(d1) < _RESOLUTION_FACTOR * floor:
                bad.append(i)
                continue
            h1 = d1 / steps[i] ** 2
            h2 = (fp2 - 2.0 * f0 + fm2) / (2.0 * steps[i]) ** 2
            if abs(h1 - h2) > _TRUNCATION_TOL * max(abs(h1), abs(h2)):
                bad.append(i)
        return tuple(bad)

    return _Objective(f=f, grad=grad, hess=hess, source="finite_difference", unresolved=unresolved)


def _enable_x64(jax_module: Any) -> None:
    """Turn on float64 before anything is compiled; jax's config API is untyped."""
    jax_module.config.update("jax_enable_x64", True)


def _jax_objective(
    model: ModelSpec,
    data: Mapping[str, Array],
    layout: tuple[tuple[str, tuple[int, ...]], ...],
) -> _Objective:
    import jax
    import jax.numpy as jnp

    _enable_x64(jax)
    ld = compile_log_density(model)
    jdata = {k: jnp.asarray(v) for k, v in data.items()}

    def nld(vec: Any) -> Any:
        return -ld(jdata, _unflatten(layout, vec))

    f_j = jax.jit(nld)
    g_j = jax.jit(jax.grad(nld))
    h_j = jax.jit(jax.hessian(nld))

    def f(vec: Array) -> float:
        return float(f_j(jnp.asarray(vec, dtype=jnp.float64)))

    def grad(vec: Array) -> Array:
        return np.asarray(g_j(jnp.asarray(vec, dtype=jnp.float64)), dtype=float)

    def hess(vec: Array) -> Array:
        return np.asarray(h_j(jnp.asarray(vec, dtype=jnp.float64)), dtype=float)

    return _Objective(f=f, grad=grad, hess=hess, source="jax")


def _objective(
    model: ModelSpec,
    data: Mapping[str, npt.ArrayLike],
    *,
    derivatives: Derivatives,
) -> tuple[_Objective, tuple[tuple[str, tuple[int, ...]], ...]]:
    layout = flat_layout(model)
    if not layout:
        raise ValueError(f"model {model.name!r} has no free parameters")
    cols = _model_data(model, data)
    source = _resolve_derivatives(derivatives)
    obj = (
        _jax_objective(model, cols, layout)
        if source == "jax"
        else _numpy_objective(model, cols, layout)
    )
    return obj, layout


def _unresolved_names(
    obj: _Objective, layout: tuple[tuple[str, tuple[int, ...]], ...], x: Array
) -> tuple[str, ...]:
    if obj.unresolved is None:
        return ()
    names = _flat_names(layout)
    return tuple(names[i] for i in obj.unresolved(x))


def _unresolved_reason(what: str, names: tuple[str, ...]) -> str:
    return (
        f"{what}: finite-difference curvature unresolved for {', '.join(names)}; "
        f"{_UNRESOLVED_REMEDY}"
    )


def hessian_at(
    model: ModelSpec,
    data: Mapping[str, npt.ArrayLike],
    z: Mapping[str, npt.ArrayLike],
    *,
    derivatives: Derivatives = "auto",
) -> Array | Unverified:
    """Hessian of the negative log density at unconstrained ``z``, in ``flat_layout`` order.

    With finite differences the diagonal is checked against its round-off
    floor and a step-doubling truncation estimate; a coordinate that fails
    makes the result ``Unverified`` naming it, never a noisy matrix.
    """
    obj, layout = _objective(model, data, derivatives=derivatives)
    x = _flatten(layout, z)
    bad = _unresolved_names(obj, layout, x)
    if bad:
        return Unverified(
            reason=_unresolved_reason(f"hessian_at on {model.name!r}", bad),
            detail={"unresolved": ", ".join(bad), "derivatives": obj.source},
        )
    return obj.hess(x)


# -- curvature ------------------------------------------------------------------------


@dataclass(frozen=True)
class _Curvature:
    """Symmetrized Hessian with its spectrum and the relative PD / PSD verdicts."""

    H: Array
    eigvals: Array
    eigvecs: Array
    scale: float  # largest |eigenvalue|; 0.0 for an identically zero Hessian

    @property
    def min_eigenvalue(self) -> float:
        return float(self.eigvals[0])

    @property
    def finite(self) -> bool:
        return bool(np.all(np.isfinite(self.eigvals)))

    @property
    def psd(self) -> bool:
        """Round-off-negative is still PSD; a genuinely negative direction is not."""
        return (
            self.finite
            and self.scale > 0.0
            and self.min_eigenvalue >= -_PD_RELATIVE_TOL * self.scale
        )

    @property
    def pd(self) -> bool:
        return (
            self.finite and self.scale > 0.0 and self.min_eigenvalue > _PD_RELATIVE_TOL * self.scale
        )


def _curvature(H: Array) -> _Curvature:
    S = 0.5 * (H + H.T)
    if not np.all(np.isfinite(S)):
        nan = np.full(S.shape[0], np.nan)
        return _Curvature(H=S, eigvals=nan, eigvecs=np.eye(S.shape[0]), scale=float("nan"))
    lam, V = np.linalg.eigh(S)
    return _Curvature(H=S, eigvals=lam, eigvecs=V, scale=float(np.max(np.abs(lam))))


# -- mode ----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Mode:
    z: Array
    objective: float
    method: str
    optimizer_success: bool
    converged: bool
    n_iter: int
    n_newton: int
    n_restarts: int
    curvature: _Curvature
    newton_decrement: float  # nan when undefined (indefinite or zero Hessian)
    unresolved: tuple[str, ...] = ()
    """Coordinates whose finite-difference curvature failed its resolution check."""

    @property
    def hessian_pd(self) -> bool:
        """PD *and* resolved: an unresolved curvature cannot be called anything."""
        return self.curvature.pd and not self.unresolved


def _initial_point(
    model: ModelSpec,
    layout: tuple[tuple[str, tuple[int, ...]], ...],
    init: Mapping[str, npt.ArrayLike] | None,
) -> Array:
    if init is None:
        return np.zeros(sum(_size(s) for _, s in layout))
    # ``init`` is in constrained coordinates and must name every free parameter;
    # ``unconstrain`` also needs the fixed ones, because an interval bound may be
    # a fixed parameter.
    missing = [name for name, _ in layout if name not in init]
    if missing:
        raise KeyError(f"init is missing free parameters {missing}")
    return _flatten(layout, unconstrain(model, {**_fixed_values(model), **init}))


_scipy_minimize: Callable[..., Any] = _opt.minimize


def _minimize(obj: _Objective, x0: Array, method: str) -> Any:
    kwargs: dict[str, Any] = {"jac": obj.grad}
    if method in ("trust-ncg", "trust-exact"):
        kwargs["hess"] = obj.hess
    return _scipy_minimize(obj.f, x0, method=method, **kwargs)


@dataclass(frozen=True)
class _Polished:
    z: Array
    objective: float
    curvature: _Curvature
    newton_decrement: float
    n_newton: int


def _newton_polish(obj: _Objective, x0: Array) -> _Polished:
    """Newton steps on the real Hessian until the decrement is below ``_NEWTON_TOL``.

    Eigenvalues below ``_NEWTON_SUBSPACE_TOL`` of the largest are floored
    there, for the step and for the decrement alike: a flat direction with a
    zero gradient costs nothing, a flat direction with a gradient (an
    unbounded density) keeps the decrement large. Stops without converging
    at an indefinite Hessian (a saddle: Newton would climb) or when
    backtracking cannot decrease ``f``.
    """
    x = np.array(x0, dtype=float)
    fx = obj.f(x)
    n_newton = 0
    for it in range(_MAX_NEWTON + 1):
        g = obj.grad(x)
        curv = _curvature(obj.hess(x))
        if not (np.all(np.isfinite(g)) and curv.psd):
            return _Polished(x, fx, curv, float("nan"), n_newton)
        # Levenberg-style floor: directions flatter than the subspace tolerance are
        # stepped against the floor, so a non-zero gradient along a flat or unbounded
        # direction (a funnel) keeps the decrement large instead of vanishing from it.
        gv = curv.eigvecs.T @ g
        ratio = gv / np.maximum(curv.eigvals, _NEWTON_SUBSPACE_TOL * curv.scale)
        decrement = float(gv @ ratio)
        if decrement < _NEWTON_TOL or it == _MAX_NEWTON:
            return _Polished(x, fx, curv, decrement, n_newton)
        delta = curv.eigvecs @ ratio
        t = 1.0
        while t >= 2.0**-30:
            xn = x - t * delta
            fn = obj.f(xn)
            if np.isfinite(fn) and fn <= fx - 1e-4 * t * decrement:
                break
            t *= 0.5
        else:
            return _Polished(x, fx, curv, decrement, n_newton)
        x, fx = xn, fn
        n_newton += 1
    raise AssertionError("unreachable")  # pragma: no cover


def _jittered_start(obj: _Objective, x0: Array, rng: np.random.Generator) -> Array | None:
    """``x0`` plus unit normal noise in unconstrained coordinates, with a finite objective."""
    for _ in range(5):
        x = x0 + rng.standard_normal(x0.size)
        if np.isfinite(obj.f(x)):
            return x
    return None


def _find_mode(
    obj: _Objective,
    layout: tuple[tuple[str, tuple[int, ...]], ...],
    x0: Array,
    *,
    rng: np.random.Generator,
) -> _Mode | Unverified:
    f0 = obj.f(x0)
    if not np.isfinite(f0):
        raise ValueError(
            f"log density is not finite at the initial point ({-f0}); pass a finite init"
        )
    best: _Mode | None = None
    for method in _OPTIMIZERS:
        for attempt in range(_MAX_RESTARTS + 1):
            start: Array | None = x0 if attempt == 0 else _jittered_start(obj, x0, rng)
            if start is None:
                break
            try:
                res = _minimize(obj, start, method)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                # scipy's trust-region solvers raise on a non-finite step rather than
                # returning failure; that is the same event as a non-finite result
                log.warning(
                    "laplace: optimizer %s left the finite region (%s); restarting", method, exc
                )
                continue
            x = np.asarray(res.x, dtype=float)
            nit = int(getattr(res, "nit", 0))
            if not (np.isfinite(float(res.fun)) and np.all(np.isfinite(x))):
                log.warning("laplace: optimizer %s left the finite region; restarting", method)
                continue
            pol = _newton_polish(obj, x)
            unresolved = _unresolved_names(obj, layout, pol.z)
            decrement_ok = np.isfinite(pol.newton_decrement) and pol.newton_decrement < _NEWTON_TOL
            mode = _Mode(
                z=pol.z,
                objective=pol.objective,
                method=method,
                optimizer_success=bool(res.success),
                converged=bool(pol.curvature.psd and decrement_ok and not unresolved),
                n_iter=nit + pol.n_newton,
                n_newton=pol.n_newton,
                n_restarts=attempt,
                curvature=pol.curvature,
                newton_decrement=pol.newton_decrement if not unresolved else float("nan"),
                unresolved=unresolved,
            )
            if mode.converged:
                return mode
            if unresolved:
                # the verdict at this point is noise, and so would be the verdict after a
                # restart or under another optimizer: the step, not the start, is the problem
                log.warning(
                    "laplace: optimizer %s stopped at a point where the finite-difference "
                    "curvature is unresolved for %s; %s",
                    method,
                    ", ".join(unresolved),
                    _UNRESOLVED_REMEDY,
                )
                return mode
            if best is None or mode.objective < best.objective:
                best = mode
            why = (
                "indefinite Hessian (saddle)"
                if not pol.curvature.psd
                else f"Newton decrement {pol.newton_decrement:.3g} >= {_NEWTON_TOL:g}"
            )
            if nit == 0 or not pol.curvature.psd:
                log.warning(
                    "laplace: optimizer %s stopped after %d iterations at %s; "
                    "restarting from a seeded jitter",
                    method,
                    nit,
                    why,
                )
                continue
            log.warning(
                "laplace: optimizer %s did not converge (%s; %s); trying the next one",
                method,
                why,
                getattr(res, "message", ""),
            )
            break
    if best is None:
        return Unverified(
            reason="no optimizer produced a finite mode; the log density may be unbounded",
            detail={"optimizers": ", ".join(_OPTIMIZERS)},
        )
    return best


def _none_if_nan(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None


def _point_estimate(model: ModelSpec, mode: _Mode, layout: Any) -> PointEstimate:
    theta, _ = constrain(model, _unflatten(layout, mode.z))
    flat: dict[str, float | tuple[float, ...]] = {}
    for name, shape in layout:
        v = np.asarray(theta[name], dtype=float)
        flat[name] = tuple(float(x) for x in v.ravel()) if shape else float(v)
    resolved = not mode.unresolved
    return PointEstimate(
        theta=flat,
        log_density=-mode.objective,
        converged=mode.converged,
        method=mode.method,
        n_iter=mode.n_iter,
        hessian_pd=mode.hessian_pd,
        # an unresolved curvature has no eigenvalue and no decrement worth reporting
        min_eigenvalue=_none_if_nan(mode.curvature.min_eigenvalue) if resolved else None,
        newton_decrement=_none_if_nan(mode.newton_decrement) if resolved else None,
    )


def _mode_estimate(
    model: ModelSpec,
    data: Mapping[str, npt.ArrayLike],
    *,
    init: Mapping[str, npt.ArrayLike] | None,
    derivatives: Derivatives,
    seed: int | None,
    allow_unverified: bool = False,
) -> PointEstimate | Unverified:
    obj, layout = _objective(model, data, derivatives=derivatives)
    rng = np.random.default_rng(_RESTART_SEED if seed is None else seed)
    mode = _find_mode(obj, layout, _initial_point(model, layout, init), rng=rng)
    if isinstance(mode, Unverified):
        return mode
    if mode.unresolved and not allow_unverified:
        return _unverified_mode(
            mode, _unresolved_reason(f"find_mode on {model.name!r}", mode.unresolved), obj.source
        )
    return _point_estimate(model, mode, layout)


def find_mode(
    model: ModelSpec,
    data: Mapping[str, npt.ArrayLike],
    *,
    init: Mapping[str, npt.ArrayLike] | None = None,
    derivatives: Derivatives = "auto",
    seed: int | None = None,
    allow_unverified: bool = False,
) -> PointEstimate | Unverified:
    """Posterior mode in unconstrained space, reported in constrained coordinates.

    ``init`` is an optional starting point in *constrained* coordinates for
    every free parameter; the default starts at ``z = 0`` (which is ``1`` for
    positive parameters and ``0.5`` for unit-interval ones). ``seed`` only
    feeds the jittered restarts taken after a saddle or a zero-iteration
    stop; the search is deterministic for a given seed (and for ``None``).

    ``converged`` is true only when the Newton decrement at the returned
    point is below ``1e-8`` **and** the Hessian there is positive
    semi-definite; ``hessian_pd``, ``min_eigenvalue`` and ``newton_decrement``
    say why when it is not.

    Returns ``Unverified`` when no optimizer finds a finite point at all, or
    when the finite-difference curvature at the candidate mode fails its
    resolution check (round-off or truncation) — the reason names the
    coordinates and the remedy (install jax, or rescale). With
    ``allow_unverified=True`` the latter case is returned as a
    ``PointEstimate`` with ``converged=False``, ``hessian_pd=False`` and no
    eigenvalue or decrement, since neither could be checked.
    """
    return _mode_estimate(
        model,
        data,
        init=init,
        derivatives=derivatives,
        seed=seed,
        allow_unverified=allow_unverified,
    )


# -- draws ---------------------------------------------------------------------------


def constrain_draws(
    model: ModelSpec, z: Mapping[str, npt.ArrayLike]
) -> dict[str, npt.NDArray[np.float64]]:
    """Apply ``axiom.core.constrain`` to every draw: ``z[name]`` is ``(n, *shape)``.

    Returns only the free parameters, each ``(n, *shape)``. One call to
    ``constrain`` per draw keeps parameter-valued interval bounds correct.
    """
    names = [p.name for p in free_parameters(model)]
    arrays = {k: np.asarray(v, dtype=float) for k, v in z.items()}
    n = next(iter(arrays.values())).shape[0]
    out = {name: np.empty_like(arrays[name]) for name in names}
    for i in range(n):
        theta, _ = constrain(model, {k: v[i] for k, v in arrays.items()})
        for name in names:
            out[name][i] = theta[name]
    return out


def _coords(params: tuple[Param, ...]) -> dict[str, list[int]]:
    return {f"{p.name}_dim{i}": list(range(n)) for p in params for i, n in enumerate(p.shape)}


def _draw_seed() -> int:
    """A fresh seed when the caller passed ``None``, so provenance can still record one."""
    return int(np.random.default_rng().integers(0, _SEED_MAX))


def _unverified_mode(mode: _Mode, reason: str, derivatives: str) -> Unverified:
    resolved = not mode.unresolved
    return Unverified(
        reason=reason,
        detail={
            "min_eigenvalue": repr(mode.curvature.min_eigenvalue) if resolved else "unresolved",
            "max_abs_eigenvalue": repr(mode.curvature.scale) if resolved else "unresolved",
            "converged": str(mode.converged),
            "optimizer": mode.method,
            "optimizer_success": str(mode.optimizer_success),
            "hessian_pd": str(mode.hessian_pd),
            "newton_decrement": repr(mode.newton_decrement) if resolved else "unresolved",
            "n_iter": str(mode.n_iter),
            "log_density_at_mode": repr(-mode.objective),
            "derivatives": derivatives,
            "unresolved": ", ".join(mode.unresolved),
        },
    )


def laplace(
    model: ModelSpec,
    data: Mapping[str, npt.ArrayLike],
    *,
    draws: int,
    seed: int | None,
    init: Mapping[str, npt.ArrayLike] | None = None,
    chains: int = 1,
    derivatives: Derivatives = "auto",
    allow_unverified: bool = False,
) -> Posterior | Unverified:
    """Gaussian approximation at the mode, drawn and mapped back to constrained space.

    Returns ``chains × draws`` independent draws shaped ``(chains, draws,
    *shape)`` per free parameter — or ``Unverified`` when the mode search
    did not converge (no finite mode, a saddle, or a Newton decrement above
    ``1e-8``), the Hessian at the mode is not numerically positive definite
    (smallest eigenvalue below ``1e-8`` of the largest), or the
    finite-difference curvature there failed its resolution check (the
    reason names the coordinates; install jax or rescale). ``Unverified.detail``
    carries ``min_eigenvalue``, ``converged``, ``optimizer``, ``unresolved``
    and the rest of the verdict. ``allow_unverified=True`` returns the
    Gaussian anyway with eigenvalues clipped to the relative floor,
    ``hessian_pd`` and/or ``converged`` false in provenance, and a
    ``WARNING`` logged. A Hessian with no positive curvature at all cannot
    be clipped to anything and is ``Unverified`` regardless.

    Provenance always carries ``hessian_pd``, ``min_eigenvalue``,
    ``newton_decrement``, ``nonfinite_draw_frac``, ``optimizer``, ``seed``
    (drawn and recorded when ``None`` was passed), the derivative source and
    ``unresolved_curvature`` (empty when every coordinate resolved). Fixed
    parameters are reported under ``provenance["fixed"]`` rather than as
    constant draws.
    """
    if draws < 1 or chains < 1:
        raise ValueError("draws and chains must be positive")
    used_seed = _draw_seed() if seed is None else int(seed)
    rng = np.random.default_rng(used_seed)
    obj, layout = _objective(model, data, derivatives=derivatives)
    mode = _find_mode(obj, layout, _initial_point(model, layout, init), rng=rng)
    if isinstance(mode, Unverified):
        return mode
    curv = mode.curvature  # the real Hessian at the polished mode, whatever found it
    if not curv.finite or curv.scale == 0.0:
        why = "not finite" if not curv.finite else "identically zero"
        return _unverified_mode(
            mode,
            f"Hessian at the mode of {model.name!r} is {why}; there is no curvature to "
            "build a Gaussian from, and nothing to scale a jitter by",
            obj.source,
        )
    hessian_pd = mode.hessian_pd
    floor = _PD_RELATIVE_TOL * curv.scale
    lam = np.maximum(curv.eigvals, floor)
    jitter = 0.0 if curv.pd else float(floor - curv.min_eigenvalue)
    problems: list[str] = []
    if mode.unresolved:
        problems.append(
            f"finite-difference curvature unresolved for {', '.join(mode.unresolved)}; "
            f"{_UNRESOLVED_REMEDY}"
        )
    elif not mode.converged:
        problems.append(
            "mode search did not converge ("
            + (
                "indefinite Hessian"
                if not curv.psd
                else f"Newton decrement {mode.newton_decrement:.3g}"
            )
            + f"; optimizer {mode.method}, {mode.n_iter} iterations)"
        )
    if not hessian_pd and not mode.unresolved:
        problems.append(
            f"Hessian at the mode is not numerically positive definite (min eigenvalue "
            f"{curv.min_eigenvalue:.3g} against a spectrum of {curv.scale:.3g}; the posterior "
            "is flat or ill-posed in at least one direction)"
        )
    if problems:
        reason = f"laplace on {model.name!r}: " + "; ".join(problems)
        if not allow_unverified:
            return _unverified_mode(mode, reason, obj.source)
        log.warning(
            "%s. allow_unverified=True: eigenvalues clipped to %.3g (relative jitter %.3g); "
            "these draws inherit the problem.",
            reason,
            floor,
            jitter,
        )
    # covariance = H^-1 = V diag(1/lambda) V^T; sample z = mode + V diag(1/sqrt(lambda)) eps
    root_cov = curv.eigvecs / np.sqrt(lam)
    n = chains * draws
    eps = rng.standard_normal((n, mode.z.size))
    z_flat = mode.z + eps @ root_cov.T
    z_draws = {
        name: z_flat[:, i : i + _size(shape)].reshape((n, *shape))
        for (name, shape), i in zip(layout, _offsets(layout), strict=True)
    }
    theta = constrain_draws(model, z_draws)
    bad = np.zeros(n, dtype=bool)
    for v in theta.values():
        bad |= ~np.all(np.isfinite(v.reshape(n, -1)), axis=1)
    nonfinite_frac = float(bad.mean())
    if nonfinite_frac > 0:
        log.warning(
            "laplace: %.1f%% of constrained draws are non-finite; they are kept, not filtered",
            100 * nonfinite_frac,
        )
    free = free_parameters(model)
    resolved = not mode.unresolved
    return Posterior(
        {name: v.reshape((chains, draws, *v.shape[1:])) for name, v in theta.items()},
        coords=_coords(free),
        provenance={
            "method": "laplace",
            "backend": "laplace",
            "optimizer": mode.method,
            "optimizer_success": mode.optimizer_success,
            "derivatives": obj.source,
            "seed": used_seed,
            "draws": draws,
            "tune": 0,
            "chains": chains,
            "converged": mode.converged,
            "verified": mode.converged and hessian_pd,
            "n_iter": mode.n_iter,
            "n_newton": mode.n_newton,
            "n_restarts": mode.n_restarts,
            "newton_decrement": _none_if_nan(mode.newton_decrement) if resolved else None,
            "log_density_at_mode": -mode.objective,
            "hessian_pd": hessian_pd,
            "min_eigenvalue": curv.min_eigenvalue if resolved else None,
            "max_abs_eigenvalue": curv.scale if resolved else None,
            "unresolved_curvature": list(mode.unresolved),
            "jitter": jitter,
            "nonfinite_draw_frac": nonfinite_frac,
            "model_hash": model.content_hash(),
            "model_name": model.name,
            "fixed": _fixed_values(model),
        },
    )


# -- backend -------------------------------------------------------------------------


class LaplaceBackend:
    """The always-available backend: no sampler, a Gaussian at the mode.

    ``sample`` returns Laplace draws — provenance says ``"laplace draws, not
    MCMC"`` — so code written against ``Backend`` runs without the extra.
    Like ``laplace``, it returns ``Unverified`` rather than a Gaussian it
    cannot stand behind.
    """

    name: str = "laplace"

    def sample(
        self,
        model: ModelSpec,
        data: Mapping[str, npt.ArrayLike],
        *,
        draws: int,
        tune: int,
        chains: int,
        seed: int | None,
    ) -> Posterior | Unverified:
        post = laplace(model, data, draws=draws, seed=seed, chains=chains)
        if isinstance(post, Unverified):
            return post
        return post.with_provenance(
            tune=tune, note="laplace draws, not MCMC; tune is recorded but unused"
        )

    def optimize(
        self, model: ModelSpec, data: Mapping[str, npt.ArrayLike], *, seed: int | None
    ) -> PointEstimate:
        """``find_mode``; ``Backend.optimize`` has no ``Unverified`` return, so it is raised."""
        est = find_mode(model, data, seed=seed)
        if isinstance(est, Unverified):
            raise ValueError(est.reason)
        return est

    def laplace(
        self,
        model: ModelSpec,
        data: Mapping[str, npt.ArrayLike],
        *,
        draws: int,
        seed: int | None,
        allow_unverified: bool = False,
    ) -> Posterior | Unverified:
        return laplace(model, data, draws=draws, seed=seed, allow_unverified=allow_unverified)
