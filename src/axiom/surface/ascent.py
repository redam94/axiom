"""Classical response-surface moves on a fitted surface: steepest ascent and canonical analysis.

Both need derivatives of ``forward()`` with respect to the doses. With jax
installed *and* 64-bit mode enabled they come from ``core.compile_jax`` over
the surface's expression tree — the same tree the likelihood evaluates;
otherwise from finite differences of ``forward()`` itself. Float32 jax
derivatives are too coarse to locate a stationary point, so ``method="jax"``
without x64 is an error rather than a silent downgrade. Either way there is
one forward (rule 3): nothing here knows what the surface is.

Finite-difference steps are relative to a per-treatment dose scale, never to
an absolute number: ``h_j = rel * scale_j`` with ``scale_j`` the half-width
of ``bounds`` when given, else ``max(|x_j|, 1e-2 * max|x|)``. Rescaling a
dose unit therefore rescales the derivatives exactly whenever each
coordinate's step is set in its own units — always with ``bounds``, and
without them while ``|x_j|`` is at least ``1e-2`` of the largest
coordinate. At a boundary of ``bounds`` the stencil is one-sided
(second-order accurate), so the surface is never evaluated outside the
box. Without ``bounds`` a zero or much smaller coordinate borrows its scale
from the largest one (mixing dose units, which only moves the truncation
error), and an all-zero point has no scale at all: that is ``Unsupported``,
not a number. Give ``bounds`` to differentiate at the origin.

A derivative the method cannot resolve is ``Unsupported``, never a number.
That covers any non-finite entry, and — for both methods — a coordinate that
sits exactly at ``0`` or exactly on a bound whose one-sided secant slope
keeps growing as the step shrinks (a kernel shape ``s < 1`` at a zero dose:
the true derivative is infinite there, where jax's guarded ``Pow`` reports
``0`` and a one-sided stencil reports some large finite number). Finite
differences are additionally checked per coordinate against their own
round-off floor: ``eps * max|f| / h_j`` for a first difference and
``eps * max|f| / h_j^2`` for a second one. An entry is *resolved* when it
exceeds ``100`` times its floor, and *negligible* when its floor is within
the method's noise (``1e-6`` relative) of the largest resolved entry, both
taken in coded units (per dose scale, so the verdict does not depend on the
dose units) — a linear coordinate beside a curved one. Anything else is round-off, not
slope or curvature (a too-small step, a too-large ``|f|``, or a surface the
stencil cannot see vary), and is ``Unsupported`` naming the coordinate. A
finite-difference gradient at a stationary point is therefore also
``Unsupported``: the stencil cannot tell a zero slope from one below its
floor, and the one case where that matters (``|f|`` so large that a real
slope vanishes in round-off) is indistinguishable from it.

The singular and flat thresholds in ``canonical_analysis`` are method-aware:
finite-difference Hessians carry noise of order ``1e-6`` relative to the
largest eigenvalue, jax Hessians ``1e-12``. Canonical analysis is a
second-order expansion at a point plus an eigendecomposition (review C6); a
flat direction is reported as ``Unsupported`` because ridge analysis and
lack-of-fit are 1.1.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import model_validator

from axiom.core import (
    NonEmptyStr,
    Spec,
    SupportsForward,
    Unsupported,
    compile_jax,
    jax_available,
    require_jax,
)
from axiom.surface.design import Bounds

__all__ = [
    "AscentPath",
    "Method",
    "StationaryPoint",
    "canonical_analysis",
    "gradient",
    "hessian",
    "steepest_ascent",
]

Array = npt.NDArray[np.float64]
Theta = Mapping[str, npt.ArrayLike]
Method = Literal["auto", "jax", "finite"]
Kind = Literal["maximum", "minimum", "saddle"]
Stop = Literal["decrease", "converged", "boundary", "n_steps"]
Box = tuple[Array, Array]

_REL_GRAD = 1e-5  # relative first-difference step (near eps ** (1/3))
_REL_HESS = 1e-3  # relative second-difference step
_ZERO_SCALE = 1e-2  # floor for a zero coordinate, relative to max|x|, when no bounds are given
_RESOLUTION = 100.0  # a finite difference must exceed this multiple of its round-off floor
_FLOOR_FINITE = (
    1e-6  # relative eigenvalue floor below which a finite-difference Hessian is singular
)
_FLOOR_JAX = 1e-12  # the same for jax (x64) derivatives
_EPS = float(np.finfo(np.float64).eps)
# Boundary check: one-sided secant slopes a decade apart, relative to the dose scale.
_BOUNDARY_STEPS = (1e-4, 1e-5, 1e-6)
_BOUNDARY_GROWTH = 1.01  # a secant must grow by this factor per decade to count as exploding
_BOUNDARY_NOISE = 1e4  # and the smallest-step secant must exceed this multiple of its round-off

# Stencils in units of ``h``: offsets and weights for the first (``_D1``) and second
# (``_D2``) derivative, keyed by side: 0 central, +1 forward (at a lower bound),
# -1 backward (at an upper bound). All are second-order accurate.
_D1: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {
    0: ((-1.0, 1.0), (-0.5, 0.5)),
    1: ((0.0, 1.0, 2.0), (-1.5, 2.0, -0.5)),
    -1: ((0.0, -1.0, -2.0), (1.5, -2.0, 0.5)),
}
_D2: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {
    0: ((-1.0, 0.0, 1.0), (1.0, -2.0, 1.0)),
    1: ((0.0, 1.0, 2.0, 3.0), (2.0, -5.0, 4.0, -1.0)),
    -1: ((0.0, -1.0, -2.0, -3.0), (2.0, -5.0, 4.0, -1.0)),
}


class AscentPath(Spec):
    """The visited points of a steepest-ascent walk and the surface value at each."""

    treatments: tuple[NonEmptyStr, ...]
    points: tuple[tuple[float, ...], ...]
    values: tuple[float, ...]
    stop: Stop

    @model_validator(mode="after")
    def _consistent(self) -> AscentPath:
        if not self.points or len(self.points) != len(self.values):
            raise ValueError("points and values must be non-empty and equally long")
        if any(len(p) != len(self.treatments) for p in self.points):
            raise ValueError("every point needs one coordinate per treatment")
        return self

    @property
    def n(self) -> int:
        return len(self.points)

    def best(self) -> dict[str, float]:
        return dict(zip(self.treatments, self.points[-1], strict=True))


class StationaryPoint(Spec):
    """Canonical analysis at ``origin``: the stationary point of the local quadratic and its shape.

    ``eigenvectors[i]`` pairs with ``eigenvalues[i]`` (ascending); ``value`` is
    ``forward()`` at ``point``, not the quadratic's prediction.
    """

    treatments: tuple[NonEmptyStr, ...]
    origin: tuple[float, ...]
    point: tuple[float, ...]
    value: float
    kind: Kind
    gradient: tuple[float, ...]
    eigenvalues: tuple[float, ...]
    eigenvectors: tuple[tuple[float, ...], ...]

    @model_validator(mode="after")
    def _consistent(self) -> StationaryPoint:
        k = len(self.treatments)
        sizes = (len(self.origin), len(self.point), len(self.gradient), len(self.eigenvalues))
        if k == 0 or any(s != k for s in sizes):
            raise ValueError("origin, point, gradient, eigenvalues need one entry per treatment")
        if len(self.eigenvectors) != k or any(len(v) != k for v in self.eigenvectors):
            raise ValueError("eigenvectors must be a k x k matrix")
        return self


# -- derivatives ---------------------------------------------------------------------


def _fmt(values: Array) -> str:
    return ", ".join(f"{v:.6g}" for v in values)


def _forward_rows(
    surface: SupportsForward, theta: Theta, names: tuple[str, ...], pts: Array
) -> Array:
    """``forward`` on ``pts`` (m, k) -> (m,); one value per dose row."""
    dose = {name: np.ascontiguousarray(pts[:, j], dtype=np.float64) for j, name in enumerate(names)}
    out = np.asarray(surface.forward(dose, theta), dtype=np.float64).reshape(-1)
    if out.size != pts.shape[0]:
        raise ValueError(f"surface returned {out.size} values for {pts.shape[0]} dose rows")
    return out


def _x64() -> bool:
    import jax

    return bool(getattr(jax.config, "jax_enable_x64", False))


def _use_jax(method: Method) -> bool:
    """Resolve ``method``: jax only with 64-bit mode on; ``"jax"`` without it is an error."""
    if method == "finite":
        return False
    if method == "jax":
        missing = require_jax()
        if missing is not None:
            raise ValueError(missing.reason)
        if not _x64():
            raise ValueError(
                "method='jax' needs jax_enable_x64=True: float32 derivatives are too coarse "
                "to locate a stationary point; enable x64 or use method='finite'"
            )
        return True
    if method != "auto":
        raise ValueError(f"method must be 'auto', 'jax', or 'finite'; got {method!r}")
    return jax_available() and _x64()


def _jax_scalar(
    surface: SupportsForward, theta: Theta, names: tuple[str, ...]
) -> Callable[[object], object]:
    import jax.numpy as jnp

    f = compile_jax(surface.expr)
    params = {k: jnp.asarray(v) for k, v in theta.items()}

    def scalar(xv: object) -> object:
        dose = {name: xv[j][None] for j, name in enumerate(names)}  # type: ignore[index]
        return jnp.sum(f(dose, params))

    return scalar


def _vec(x: Mapping[str, float]) -> tuple[tuple[str, ...], Array]:
    names = tuple(x)
    if not names:
        raise ValueError("x must name at least one treatment")
    return names, np.asarray([float(x[n]) for n in names], dtype=np.float64)


def _box(names: tuple[str, ...], bounds: Bounds | None) -> Box | None:
    """``(low, high)`` in the order of ``names``; ``bounds`` must name exactly those treatments."""
    if bounds is None:
        return None
    if set(bounds.treatments) != set(names):
        raise ValueError("bounds must name exactly the treatments in x")
    pos = [bounds.treatments.index(n) for n in names]
    lo = np.asarray(bounds.low, dtype=np.float64)[pos]
    hi = np.asarray(bounds.high, dtype=np.float64)[pos]
    return lo, hi


def _scale(x: Array, box: Box | None) -> Array | Unsupported:
    """Per-treatment dose scale the finite-difference and boundary-check steps are relative to.

    The half-width of ``bounds`` when given (``Bounds.half_width`` in the order
    of ``x``); otherwise ``max(|x_j|, 1e-2 * max|x|)``. An all-zero ``x``
    without bounds defines no scale, so no step can be relative to one (D4).
    """
    if box is not None:
        lo, hi = box
        return (hi - lo) / 2.0
    big = float(np.max(np.abs(x)))
    if big == 0.0:
        return Unsupported(
            reason=(
                "no dose scale at x: x is all zeros and no bounds were given, so neither the "
                "finite-difference step nor the boundary check has anything to be relative "
                "to; give bounds"
            ),
            detail={"point": _fmt(x)},
        )
    return np.maximum(np.abs(x), _ZERO_SCALE * big)


def _at_boundary(x: Array, box: Box | None) -> list[tuple[int, float]]:
    """``(coordinate, inward direction)`` for each coordinate exactly at 0 or exactly on a bound."""
    out: list[tuple[int, float]] = []
    for j in range(x.size):
        if box is not None and x[j] == box[1][j]:
            out.append((j, -1.0))
        elif box is not None and x[j] == box[0][j]:
            out.append((j, 1.0))
        elif x[j] == 0.0:
            out.append((j, 1.0))
    return out


def _boundary(
    surface: SupportsForward, theta: Theta, names: tuple[str, ...], x: Array, box: Box | None
) -> Unsupported | None:
    """``Unsupported`` when a boundary coordinate's one-sided slope is not finite.

    For every coordinate exactly at ``0`` or exactly on a bound the one-sided
    secant ``(f(x + d t scale_j) - f(x)) / (t scale_j)`` is taken at
    ``t = 1e-4, 1e-5, 1e-6`` towards the inside. A finite derivative settles
    (the secants change by ``O(t)``); a power singularity ``x^s``, ``s < 1``,
    grows by the constant factor ``10^(1 - s)`` per decade. The slope is
    "exploding" when it grows by at least ``_BOUNDARY_GROWTH`` over each
    decade *and* geometrically (the second decade's growth is at least the
    square root of the first's — curvature-driven growth decays tenfold per
    decade and fails this), with the smallest-step secant above
    ``_BOUNDARY_NOISE`` times its round-off. Non-finite or vanishing secants
    are left to the differentiation itself to report.
    """
    at = _at_boundary(x, box)
    if not at:
        return None
    scale = _scale(x, box)
    if isinstance(scale, Unsupported):
        return scale
    steps = np.asarray(_BOUNDARY_STEPS)
    rows = [x.copy()]
    for j, d in at:
        for t in steps:
            p = x.copy()
            p[j] += d * t * scale[j]
            rows.append(p)
    f = _forward_rows(surface, theta, names, np.vstack(rows))
    f0 = float(f[0])
    for m, (j, d) in enumerate(at):
        vals = f[1 + 3 * m : 4 + 3 * m]
        if not (math.isfinite(f0) and bool(np.all(np.isfinite(vals)))):
            continue
        deltas = steps * scale[j]
        secants = np.abs(vals - f0) / deltas
        noise = _BOUNDARY_NOISE * _EPS * max(abs(f0), float(np.max(np.abs(vals)))) / deltas[-1]
        if secants[-1] <= noise or secants[0] == 0.0 or secants[1] == 0.0:
            continue
        r1, r2 = float(secants[1] / secants[0]), float(secants[2] / secants[1])
        if r1 >= _BOUNDARY_GROWTH and r2 >= _BOUNDARY_GROWTH and r2 * r2 >= r1:
            side = "lower" if d > 0 else "upper"
            return Unsupported(
                reason=(
                    f"derivative not finite at the boundary for {names[j]}: at ({_fmt(x)}) the "
                    f"one-sided slope of forward() from the {side} side keeps growing as the "
                    f"step shrinks ({_fmt(secants)} over steps {_fmt(deltas)}); start strictly "
                    "inside"
                ),
                detail={
                    "point": _fmt(x),
                    "coordinate": names[j],
                    "secants": _fmt(secants),
                    "steps": _fmt(deltas),
                },
            )
    return None


def _unresolved(names: tuple[str, ...], values: Array, floors: Array, units: Array) -> list[str]:
    """Coordinates whose finite difference is neither resolved nor negligible.

    Resolved: ``|value_j| >= _RESOLUTION * floor_j``. Negligible: ``floor_j``
    is within ``_FLOOR_FINITE`` (the method's documented noise) of the largest
    resolved entry, so whatever the entry is, it sits inside that noise. The
    entries of a gradient or Hessian carry different dose units, so that
    comparison is made in coded units: ``units_j`` is ``scale_j`` for a first
    difference and ``scale_j^2`` for a second, which makes the verdict
    independent of the unit each dose is measured in.
    """
    resolved = np.abs(values) >= _RESOLUTION * floors
    coded = np.abs(values) * units
    ref = float(np.max(coded[resolved])) if bool(np.any(resolved)) else 0.0
    return [
        names[j]
        for j in range(len(names))
        if not resolved[j] and not floors[j] * units[j] <= _FLOOR_FINITE * ref
    ]


def _not_finite(what: str, values: Array, x: Array, use_jax: bool, box: Box | None) -> Unsupported:
    """``Unsupported`` for a derivative with a non-finite entry, with a method-aware hint."""
    if use_jax:
        hint = "the surface's exact derivative is not finite here"
    elif box is None:
        hint = (
            "the central stencil may have stepped outside the surface's domain; give bounds "
            "to keep it inside the box"
        )
    else:
        hint = (
            "with bounds the stencil stays inside the box, so the surface itself has no "
            "finite derivative here"
        )
    return Unsupported(
        reason=f"{what} is not finite at ({_fmt(x)}): {hint}",
        detail={"point": _fmt(x), what: _fmt(values.ravel())},
    )


def _sides(x: Array, h: Array, box: Box | None) -> list[int]:
    """Per coordinate: 0 central, +1 forward at a lower bound, -1 backward at an upper bound."""
    if box is None:
        return [0] * x.size
    lo, hi = box
    out = []
    for j in range(x.size):
        if x[j] - h[j] < lo[j]:
            out.append(1)
        elif x[j] + h[j] > hi[j]:
            out.append(-1)
        else:
            out.append(0)
    return out


def _grad_fd(
    surface: SupportsForward, theta: Theta, names: tuple[str, ...], x: Array, box: Box | None
) -> Array | Unsupported:
    k = x.size
    scale = _scale(x, box)
    if isinstance(scale, Unsupported):
        return scale
    h = _REL_GRAD * scale
    sides = _sides(x, h, box)
    rows: list[Array] = []
    plan: list[tuple[list[int], Array]] = []
    for j in range(k):
        offsets, weights = _D1[sides[j]]
        idx = []
        for o in offsets:
            p = x.copy()
            p[j] += o * h[j]
            rows.append(p)
            idx.append(len(rows) - 1)
        plan.append((idx, np.asarray(weights) / h[j]))
    f = _forward_rows(surface, theta, names, np.vstack(rows))
    g = np.asarray([float(f[idx] @ w) for idx, w in plan], dtype=np.float64)
    if not bool(np.all(np.isfinite(g))):
        return _not_finite("gradient", g, x, False, box)
    # Round-off floor of each first difference: eps max|f| / h_j over its stencil.
    floors = np.asarray(
        [_EPS * float(np.max(np.abs(f[idx]))) / h[j] for j, (idx, _) in enumerate(plan)]
    )
    unresolved = _unresolved(names, g, floors, scale)
    if unresolved:
        return Unsupported(
            reason=(
                f"finite differences cannot resolve the gradient in {', '.join(unresolved)} at "
                f"({_fmt(x)}): forward() changes by less than {_RESOLUTION:g} x its round-off "
                "eps|f| over the step h_j, so the first difference there is round-off, not "
                "slope (the point may be stationary, or |f| too large for the step); widen "
                "the dose scale with bounds or use method='jax'"
            ),
            detail={
                "point": _fmt(x),
                "coordinates": ", ".join(unresolved),
                "floors": _fmt(floors),
                "steps": _fmt(h),
                "gradient": _fmt(g),
            },
        )
    return g


def _hess_fd(
    surface: SupportsForward, theta: Theta, names: tuple[str, ...], x: Array, box: Box | None
) -> Array | Unsupported:
    """Second differences at ``x``, or ``Unsupported`` when round-off is all the stencil sees.

    The round-off floor of the second difference in coordinate ``j`` is
    ``eps * max|f| / h_j^2`` over its stencil. Each diagonal entry is checked
    against its own floor (``_unresolved``): one that is neither resolved nor
    negligible beside the resolved ones is round-off, not curvature, and the
    matrix is ``Unsupported`` naming the coordinates rather than a number. The
    cross terms' floors ``eps |f| / (h_i h_j)`` are bounded by the larger of
    the two diagonal floors, so the diagonal check covers them.
    """
    k = x.size
    scale = _scale(x, box)
    if isinstance(scale, Unsupported):
        return scale
    h = _REL_HESS * scale
    sides = _sides(x, h, box)
    rows: list[Array] = []

    def add(shift: dict[int, float]) -> int:
        p = x.copy()
        for j, o in shift.items():
            p[j] += o * h[j]
        rows.append(p)
        return len(rows) - 1

    diag: list[tuple[list[int], Array]] = []
    for j in range(k):
        offsets, weights = _D2[sides[j]]
        diag.append(([add({j: o}) for o in offsets], np.asarray(weights) / h[j] ** 2))
    cross: list[tuple[int, int, list[int], Array]] = []
    for i in range(k):
        for j in range(i + 1, k):
            oi, wi = _D1[sides[i]]
            oj, wj = _D1[sides[j]]
            idx = [add({i: a, j: b}) for a in oi for b in oj]
            w = np.outer(np.asarray(wi) / h[i], np.asarray(wj) / h[j]).ravel()
            cross.append((i, j, idx, w))
    f = _forward_rows(surface, theta, names, np.vstack(rows))
    H = np.zeros((k, k))
    for j, (idx, w) in enumerate(diag):
        H[j, j] = f[idx] @ w
    for i, j, idx, w in cross:
        H[i, j] = H[j, i] = f[idx] @ w
    if not bool(np.all(np.isfinite(H))):
        return _not_finite("hessian", H, x, False, box)
    floors = np.asarray(
        [_EPS * float(np.max(np.abs(f[idx]))) / h[j] ** 2 for j, (idx, _) in enumerate(diag)]
    )
    unresolved = _unresolved(names, np.diag(H).copy(), floors, scale**2)
    if unresolved:
        return Unsupported(
            reason=(
                f"finite differences cannot resolve curvature in {', '.join(unresolved)} at "
                f"({_fmt(x)}): the second difference there is within a factor {_RESOLUTION:g} "
                "of its round-off floor eps|f|/h^2 and the floor is not negligible beside the "
                "resolved entries; widen the dose scale with bounds or use method='jax'"
            ),
            detail={
                "point": _fmt(x),
                "coordinates": ", ".join(unresolved),
                "floors": _fmt(floors),
                "steps": _fmt(h),
                "diagonal": _fmt(np.diag(H)),
            },
        )
    return H


def _grad(
    surface: SupportsForward,
    theta: Theta,
    names: tuple[str, ...],
    x: Array,
    use_jax: bool,
    box: Box | None,
) -> Array | Unsupported:
    edge = _boundary(surface, theta, names, x, box)
    if edge is not None:
        return edge
    if not use_jax:
        return _grad_fd(surface, theta, names, x, box)
    import jax

    g = np.asarray(jax.grad(_jax_scalar(surface, theta, names))(x), dtype=np.float64)
    if not bool(np.all(np.isfinite(g))):
        return _not_finite("gradient", g, x, True, box)
    return g


def _hess(
    surface: SupportsForward,
    theta: Theta,
    names: tuple[str, ...],
    x: Array,
    use_jax: bool,
    box: Box | None,
) -> Array | Unsupported:
    edge = _boundary(surface, theta, names, x, box)
    if edge is not None:
        return edge
    if not use_jax:
        return _hess_fd(surface, theta, names, x, box)
    import jax

    H = np.asarray(jax.hessian(_jax_scalar(surface, theta, names))(x), dtype=np.float64)
    if not bool(np.all(np.isfinite(H))):
        return _not_finite("hessian", H, x, True, box)
    return H


def gradient(
    surface: SupportsForward,
    theta: Theta,
    x: Mapping[str, float],
    *,
    method: Method = "auto",
    bounds: Bounds | None = None,
) -> dict[str, float] | Unsupported:
    """``d forward / d dose`` at ``x``, keyed like ``x``; the key order fixes treatment order.

    ``bounds`` sets the finite-difference scale (half-width per treatment) and
    makes the stencil one-sided at the box boundary; without it the step is
    relative to ``max(|x_j|, 1e-2 max|x|)``. Returns ``Unsupported`` when an
    entry is not finite, when a coordinate exactly at ``0`` or on a bound has
    an exploding one-sided slope (either method), when ``x`` is all zeros
    without ``bounds`` (no scale), or when a finite-difference entry is
    round-off rather than slope (below ``100`` times its floor
    ``eps |f| / h_j`` and not negligible beside the resolved entries — which
    includes every finite-difference gradient at a stationary point).
    """
    names, xv = _vec(x)
    g = _grad(surface, theta, names, xv, _use_jax(method), _box(names, bounds))
    if isinstance(g, Unsupported):
        return g
    return {n: float(v) for n, v in zip(names, g, strict=True)}


def hessian(
    surface: SupportsForward,
    theta: Theta,
    x: Mapping[str, float],
    *,
    method: Method = "auto",
    bounds: Bounds | None = None,
) -> Array | Unsupported:
    """Second derivatives of ``forward`` at ``x``; rows and columns follow ``x``'s key order.

    ``bounds`` plays the same role as in ``gradient``. Returns ``Unsupported``
    when an entry is not finite, when a boundary coordinate's one-sided slope
    explodes, when ``x`` is all zeros without ``bounds``, or when a
    finite-difference diagonal entry is round-off rather than curvature
    (below ``100`` times its own floor ``eps |f| / h_j^2`` and that floor not
    within ``1e-6`` of the largest resolved diagonal entry).
    """
    names, xv = _vec(x)
    return _hess(surface, theta, names, xv, _use_jax(method), _box(names, bounds))


# -- moves -------------------------------------------------------------------------------


def steepest_ascent(
    surface: SupportsForward,
    theta: Theta,
    start: Mapping[str, float],
    *,
    step: float,
    n_steps: int,
    bounds: Bounds | None = None,
    method: Method = "auto",
    recompute: bool = False,
) -> AscentPath | Unsupported:
    """Walk ``step`` dose units per move up the gradient until the value stops rising.

    The default is the Myers-Montgomery path of steepest ascent: the
    first-order gradient at ``start`` fixes one direction and the walk follows
    that straight line until a move lowers the value (``"decrease"``), clipping
    to ``bounds`` pins the point (``"boundary"``), or ``n_steps`` moves are
    taken. ``recompute=True`` re-evaluates the gradient at every point instead
    — fixed-step gradient ascent, which bends towards a nearby optimum but
    needs the surface's derivatives at every visited point. ``"converged"``
    means the gradient's predicted first-order gain over one step is below
    the floating-point resolution of the current value.

    Returns ``Unsupported`` when ``forward()`` at ``start`` is not finite or
    when ``gradient`` would (a non-finite entry, an exploding slope at a
    boundary coordinate, an all-zero point without ``bounds``, or a
    finite-difference gradient below its round-off resolution — a start that
    is already stationary, or a ``|f|`` too large for the step), naming the
    step. With ``bounds`` the finite-difference stencil is one-sided at the
    boundary, so the surface is never evaluated outside the box.
    """
    if not step > 0:
        raise ValueError("step must be positive")
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    names, x = _vec(start)
    box = _box(names, bounds)
    use_jax = _use_jax(method)
    if box is not None:
        x = np.clip(x, *box)
    v = float(_forward_rows(surface, theta, names, x[None, :])[0])
    if not math.isfinite(v):
        return Unsupported(
            reason="forward() is not finite at start",
            detail={"point": _fmt(x), "value": repr(v)},
        )
    points, values = [x], [v]
    stop: Stop = "n_steps"
    direction: Array | None = None
    for i in range(n_steps):
        if direction is None or recompute:
            g = _grad(surface, theta, names, x, use_jax, box)
            if isinstance(g, Unsupported):
                return Unsupported(
                    reason=f"no gradient at step {i}: {g.reason}",
                    detail={**g.detail, "step": str(i)},
                    missing=g.missing,
                )
            norm = float(np.linalg.norm(g))
            if not step * norm > _EPS * abs(v):
                stop = "converged"
                break
            direction = g / norm
        x_next = x + step * direction
        if box is not None:
            x_next = np.clip(x_next, *box)
        if np.allclose(x_next, x, rtol=0.0, atol=1e-12 * step):
            stop = "boundary"
            break
        v_next = float(_forward_rows(surface, theta, names, x_next[None, :])[0])
        if not math.isfinite(v_next):
            return Unsupported(
                reason=f"forward() is not finite at step {i + 1} (point {_fmt(x_next)})",
                detail={"step": str(i + 1), "point": _fmt(x_next), "value": repr(v_next)},
            )
        if v_next - v <= 1e-10 * max(abs(v), abs(v_next), 1e-300):
            # no increase beyond round-off: the walk has stopped rising (method-independent)
            stop = "decrease"
            break
        x, v = x_next, v_next
        points.append(x)
        values.append(v)
    return AscentPath(
        treatments=names,
        points=tuple(tuple(float(c) for c in p) for p in points),
        values=tuple(values),
        stop=stop,
    )


def canonical_analysis(
    surface: SupportsForward,
    theta: Theta,
    x0: Mapping[str, float],
    *,
    method: Method = "auto",
    tol: float = 1e-8,
    bounds: Bounds | None = None,
) -> StationaryPoint | Unsupported:
    """Second-order expansion at ``x0``: stationary point ``x0 - H^-1 g`` and its nature.

    ``kind`` follows the eigenvalue signs of ``H``: all negative is a maximum,
    all positive a minimum, mixed a saddle. ``H`` is numerically singular when
    its smallest ``|eigenvalue|`` is below a method-aware floor times the
    largest (``1e-6`` for finite differences, ``1e-12`` for jax); that and an
    eigenvalue within ``tol`` (relative) of zero — a flat direction, whose
    ridge analysis is 1.1 — both return ``Unsupported`` rather than a
    stationary point that does not exist or sits arbitrarily far away.
    ``bounds`` sets the finite-difference scale and keeps the stencil inside
    the box; the stationary point itself may lie outside it. Whatever makes
    ``gradient`` or ``hessian`` ``Unsupported`` (a non-finite entry, an
    exploding slope at a boundary coordinate, no dose scale at the origin, a
    slope or curvature below finite-difference resolution) is returned as is.
    """
    if not tol >= 0:
        raise ValueError("tol must be non-negative")
    names, x = _vec(x0)
    box = _box(names, bounds)
    use_jax = _use_jax(method)
    g = _grad(surface, theta, names, x, use_jax, box)
    if isinstance(g, Unsupported):
        return g
    H = _hess(surface, theta, names, x, use_jax, box)
    if isinstance(H, Unsupported):
        return H
    H = 0.5 * (H + H.T)
    eigenvalues, vectors = np.linalg.eigh(H)
    scale = float(np.max(np.abs(eigenvalues)))
    smallest = float(np.min(np.abs(eigenvalues)))
    floor = _FLOOR_JAX if use_jax else _FLOOR_FINITE
    detail = {
        "eigenvalues": _fmt(eigenvalues),
        "method": "jax" if use_jax else "finite",
        "floor": repr(floor),
    }
    if scale == 0.0 or smallest <= floor * scale:
        return Unsupported(
            reason=(
                "Hessian is singular at x0 (smallest |eigenvalue| within the "
                f"{detail['method']}-difference floor {floor:g} of the largest): the local "
                "quadratic has no unique stationary point"
            ),
            detail=detail,
        )
    if smallest <= tol * scale:
        return Unsupported(
            reason=(
                f"the local quadratic has a flat direction (|eigenvalue| <= {tol:g} x largest); "
                "ridge analysis is not implemented (1.1)"
            ),
            detail={**detail, "tol": repr(tol)},
        )
    xs = x - np.linalg.solve(H, g)
    kind: Kind
    if bool(np.all(eigenvalues < 0)):
        kind = "maximum"
    elif bool(np.all(eigenvalues > 0)):
        kind = "minimum"
    else:
        kind = "saddle"
    value = float(_forward_rows(surface, theta, names, xs[None, :])[0])
    if not math.isfinite(value):
        return Unsupported(
            reason="forward() is not finite at the stationary point",
            detail={**detail, "point": _fmt(xs), "kind": kind},
        )
    return StationaryPoint(
        treatments=names,
        origin=tuple(float(c) for c in x),
        point=tuple(float(c) for c in xs),
        value=value,
        kind=kind,
        gradient=tuple(float(c) for c in g),
        eigenvalues=tuple(float(c) for c in eigenvalues),
        eigenvectors=tuple(tuple(float(c) for c in vectors[:, i]) for i in range(len(names))),
    )
