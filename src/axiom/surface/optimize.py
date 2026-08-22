"""Budget allocation on a fitted response surface: maximize the posterior-expected outcome.

``allocate`` chooses one dose per treatment to maximize ``E[forward(dose)]``
over posterior draws — or a posterior *quantile* of it, for a risk-averse
plan — subject to a total budget ``Σ dose ≤ budget`` and a box ``Bounds``.
The surface is anything satisfying ``SupportsForward``; the allocator only
ever calls ``forward`` on a batch of allocation rows (one row = one candidate
allocation, ``dose`` a mapping of equal-length 1-D arrays), so there is one
``forward()`` (rule 3) and nothing here knows what the surface is.

Rows are independent candidates, not a time series. A surface with
carryover (a ``Convolve`` node in ``surface.expr``) would read the row axis
as time and convolve one candidate into the next, so ``allocate`` refuses it
with ``Unsupported`` and asks for ``surface.steady_state()`` — the same
surface with the carryover removed, exact at steady state because the
weights sum to one. Non-treatment columns the surface reads (a unit index
for per-unit intercepts, nuisance basis columns) are supplied through
``context`` as single values broadcast to every row.

Two solvers:

* ``"slsqp"`` (default): ``scipy.optimize.minimize(method="SLSQP")`` in coded
  ``[-1, 1]`` units with an analytic budget Jacobian and a batched central-
  difference objective gradient (``2k + 1`` rows per evaluation, one
  ``forward`` call). SLSQP is local, so it is wrapped in a cheap global
  search. The *fixed* start set holds the ``k`` all-in corners (everything
  the budget allows on one treatment, the rest at ``low``), the two-way
  splits (the budget shared equally by a pair, for ``k <= 8``), the centre
  (every treatment filled the same fraction of the way from ``low`` to
  ``high``), and ``start`` when given. A *pre-search* pool of
  ``max(8, 4k)`` seeded random feasible points (half uniform in the box and
  shrunk to the budget, half sparse Dirichlet splits of it) is evaluated in
  one ``forward`` call and its best three join the start set, as do
  ``n_starts - 1`` further seeded random starts; every start is solved and
  the best converged solve wins. A Hill with ``s > 1`` is sigmoidal: "all-in
  on a subset of the treatments" is typically the global optimum while the
  even split — and, for ``k >= 3``, each corner — is a genuine KKT point
  that traps a single solve. After the best solve a *perturbation check*
  evaluates random feasible perturbations of the solution at three scales
  of the movable budget; if any beats it by more than the objective
  tolerance the solver restarts from there (at most three times), and a
  restart that cannot reach the point that beat it is a typed failure, not
  a number. The objective is centred at the best probe value and divided by
  the *spread* of the probe values, never their level, so an intercept that
  dwarfs the response range cannot stall the line search; the difference
  step grows with ``level / spread`` to keep the gradient's rounding error
  down. A solve that stops within one iteration below the best probe value
  is a stall, not a solution, and is rejected. When the budget equals the
  sum of the lower bounds (to tolerance) the feasible set is the single
  point ``low`` and it is returned without a solve.
* ``"cvxpy"`` (optional, ``axiom[convex]``): a global solution of a
  piecewise-linear concave approximation. It first *verifies* that the
  surface is additively separable across treatments and concave along each
  one on its bounds — tolerances relative to the response *range* over the
  evaluated rows — and returns ``Unsupported`` when either fails or cannot
  be verified at that tolerance: the convex path never fits a concave hull
  to a sigmoid.

A result that did not converge is never returned as numbers (parent issue
#290): a non-converged solver, an infeasible answer, a non-finite
objective, or a "converged" solve below a known feasible point comes back
as ``Unsupported`` carrying the solver's own message.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator
from scipy import optimize as _opt

from axiom.core import (
    Blocked,
    Convolve,
    Failure,
    Gather,
    NonEmptyStr,
    Opaque,
    Param,
    Spec,
    SupportsForward,
    SupportsPosterior,
    Unsupported,
    Unverified,
    data_names,
    walk,
)
from axiom.surface.design import Bounds

__all__ = [
    "Allocation",
    "AllocationMethod",
    "Objective",
    "allocate",
]

Array = npt.NDArray[np.float64]
Theta = Mapping[str, npt.ArrayLike]
Context = Mapping[str, npt.ArrayLike]
"""Non-treatment data columns the surface reads, one value each, broadcast to every row."""
Objective = Literal["mean", "quantile"]
"""What is maximized over posterior draws: the mean outcome, or its ``q``-quantile."""
AllocationMethod = Literal["slsqp", "cvxpy"]
"""The solver. ``cvxpy`` needs the ``convex`` extra and an additive concave surface."""

# scipy's stubs type the SLSQP constraint dict as a TypedDict; alias to keep mypy strict quiet.
_minimize: Callable[..., Any] = _opt.minimize

_EPS = float(np.finfo(np.float64).eps)
_GRAD_STEP = _EPS ** (1.0 / 3.0)
_MAX_GRAD_STEP = 1e-2
_FEASIBILITY_TOL = 1e-8
_STALL_TOL = 1e-9
"""A solve below the best probe value by more than this × scale is a stall, not an optimum."""
_PRESEARCH_MIN = 8
"""The pre-search pool holds ``max(_PRESEARCH_MIN, 4k)`` random feasible points."""
_N_PROMOTED = 3
"""How many of the best pre-search points are solved from."""
_MAX_PAIR_K = 8
"""Two-way splits are enumerated up to this many treatments (28 pairs)."""
_PERTURB_SCALES = (0.05, 0.2, 0.5)
"""Perturbation radii as fractions of the movable budget."""
_MAX_RESTARTS = 3
"""Restarts from a perturbation that beats the best solve before giving up, typed."""


# -- specs -----------------------------------------------------------------------------


class Allocation(Spec):
    """One allocation: a dose per treatment and the objective it achieves.

    ``expected_outcome`` is the maximized objective (posterior mean, or the
    ``q``-quantile) evaluated at ``doses`` by ``forward`` itself — never a
    surrogate's value. ``status`` is always ``"converged"``: a solver that did
    not converge returns ``Unsupported`` instead of an ``Allocation``.
    ``detail`` carries the solver's numbers (iterations, evaluations, starts,
    draws, grid size, surrogate value).
    """

    doses: dict[NonEmptyStr, float]
    expected_outcome: float
    budget: float
    objective: Objective
    q: float = Field(default=0.5, gt=0.0, lt=1.0)
    method: NonEmptyStr
    status: Literal["converged"] = "converged"
    seed: int | None = None
    detail: dict[str, float] = {}

    @model_validator(mode="after")
    def _consistent(self) -> Allocation:
        if not self.doses:
            raise ValueError("an Allocation needs at least one treatment")
        for name, d in self.doses.items():
            if not math.isfinite(d):
                raise ValueError(f"dose for {name!r} is not finite")
        if not math.isfinite(self.expected_outcome):
            raise ValueError("expected_outcome must be finite")
        if not math.isfinite(self.budget):
            raise ValueError("budget must be finite")
        return self

    @property
    def treatments(self) -> tuple[str, ...]:
        return tuple(self.doses)

    @property
    def total_dose(self) -> float:
        return float(sum(self.doses.values()))

    @property
    def slack(self) -> float:
        """Unspent budget; ``0`` (to solver tolerance) when the constraint binds."""
        return self.budget - self.total_dose

    def as_array(self) -> Array:
        return np.asarray(list(self.doses.values()), dtype=np.float64)


# -- what the surface reads ------------------------------------------------------------


def carryover_failure(surface: SupportsForward) -> Unsupported | None:
    """``Unsupported`` when ``surface.expr`` convolves along the row axis; ``None`` otherwise.

    Allocation rows are independent candidates; a ``Convolve`` would read
    them as a time series. The fix is ``surface.steady_state()``.
    """
    for path, node in walk(surface.expr):
        if isinstance(node, Convolve):
            return Unsupported(
                reason="surface has carryover (a Convolve node at "
                f"{path!r}); allocation rows are independent candidates, not a time "
                "series — pass surface.steady_state()",
                detail={"node": path},
            )
    return None


def _index_columns(surface: SupportsForward) -> dict[str, int | None]:
    """The integer index columns ``surface.expr`` reads through ``Gather`` nodes.

    Maps each index column to the length of the gathered axis when the
    source is a ``Param`` with a declared shape (the tightest bound over
    every ``Gather`` reading that column), else ``None``.
    """
    out: dict[str, int | None] = {}
    for _, node in walk(surface.expr):
        if not isinstance(node, Gather):
            continue
        name = node.index.name
        n = node.source.shape[-1] if isinstance(node.source, Param) and node.source.shape else None
        prev = out.get(name)
        out[name] = n if prev is None else (prev if n is None else min(prev, n))
    return out


def _index_value(name: str, raw: npt.ArrayLike, length: int | None) -> Array:
    """One zero-based integer for a ``Gather`` index column; anything else names the column."""
    a = np.asarray(raw)
    if a.size != 1:
        raise ValueError(
            f"context[{name!r}] has shape {a.shape}; a context column is one value "
            "broadcast to every allocation row"
        )
    v = a.reshape(())
    if np.issubdtype(a.dtype, np.integer):
        i = int(v)
    elif np.issubdtype(a.dtype, np.floating) and math.isfinite(float(v)) and float(v).is_integer():
        i = int(float(v))
    else:
        raise ValueError(
            f"context[{name!r}] = {v.tolist()!r} indexes a Gather and must be a zero-based integer"
        )
    if i < 0:
        raise ValueError(f"context[{name!r}] = {i} indexes a Gather and must be non-negative")
    if length is not None and i >= length:
        raise ValueError(
            f"context[{name!r}] = {i} is out of range for a Gather over {length} entries "
            f"(valid: 0..{length - 1})"
        )
    return np.asarray(i, dtype=np.int64)


def _context_columns(
    surface: SupportsForward, bounds: Bounds, context: Context | None
) -> dict[str, Array]:
    """Validate ``bounds`` and ``context`` against the columns ``surface.expr`` reads.

    Every treatment in ``bounds`` must be a column the surface reads; every
    other column the surface reads must come from ``context``; every
    ``context`` entry must be a single value. A column the surface reads as
    a ``Gather`` index must be a zero-based integer inside the gathered
    axis (``1.7``, ``-1``, and ``n`` are errors naming the column, never a
    truncated, wrapped, or out-of-range index). A tree holding an
    ``Opaque`` node cannot enumerate its columns, so only the ``context``
    shapes are checked then.
    """
    index_cols = _index_columns(surface)
    ctx: dict[str, Array] = {}
    for name, raw in (context or {}).items():
        if name in index_cols:
            ctx[name] = _index_value(name, raw, index_cols[name])
            continue
        a = np.asarray(raw, dtype=np.float64)
        if a.size != 1:
            raise ValueError(
                f"context[{name!r}] has shape {a.shape}; a context column is one value "
                "broadcast to every allocation row"
            )
        ctx[name] = a.reshape(())
    clash = sorted(set(ctx) & set(bounds.treatments))
    if clash:
        raise ValueError(f"context supplies columns that are treatments in Bounds: {clash}")
    if any(isinstance(node, Opaque) for _, node in walk(surface.expr)):
        return ctx
    read = data_names(surface.expr)
    unknown = [t for t in bounds.treatments if t not in read]
    if unknown:
        raise ValueError(
            f"Bounds names treatments the surface does not read: {unknown}; "
            f"the surface reads {list(read)}"
        )
    unread = sorted(set(ctx) - set(read))
    if unread:
        raise ValueError(
            f"context supplies columns the surface does not read: {unread}; "
            f"the surface reads {list(read)}"
        )
    missing = [c for c in read if c not in bounds.treatments and c not in ctx]
    if missing:
        raise ValueError(
            f"the surface reads columns {missing} that are neither treatments in Bounds "
            "nor supplied through context=; pass one value per column in context="
        )
    return ctx


def _declared_shapes(surface: SupportsForward) -> dict[str, tuple[int, ...]]:
    return {n.name: n.shape for _, n in walk(surface.expr) if isinstance(n, Param)}


# -- evaluation over draws --------------------------------------------------------------


@dataclass(frozen=True)
class _Evaluator:
    """``rows(X)``: ``forward`` on ``(m, k)`` allocation rows for every draw -> ``(n_draws, m)``.

    Scalar parameters are stacked ``(n_draws, 1)`` so one ``forward`` call
    broadcasts against the ``(m,)`` dose columns; vector parameters fall back
    to one call per draw. ``context`` columns (0-d) ride along unchanged.
    """

    surface: SupportsForward
    names: tuple[str, ...]
    context: Mapping[str, Array]
    broadcast: Mapping[str, Array] | None
    per_draw: tuple[Mapping[str, Array], ...]
    n_draws: int

    def rows(self, X: Array) -> Array:
        m = X.shape[0]
        dose: dict[str, Array] = {
            name: np.ascontiguousarray(X[:, j], dtype=np.float64)
            for j, name in enumerate(self.names)
        }
        dose.update(self.context)
        if self.broadcast is not None:
            return self._shape(self.surface.forward(dose, self.broadcast), m)
        stacked = [self._shape(self.surface.forward(dose, th), m) for th in self.per_draw]
        return np.concatenate(stacked, axis=0)

    @staticmethod
    def _shape(out: npt.ArrayLike, m: int) -> Array:
        a = np.asarray(out, dtype=np.float64)
        if a.ndim == 0 or a.size % m != 0:
            raise ValueError(f"surface returned shape {a.shape} for {m} allocation rows")
        return a.reshape(-1, m) if a.ndim != 1 else a.reshape(1, m)


def _point_theta(surface: SupportsForward, theta: Theta) -> dict[str, Array]:
    """A point plan: one scalar per parameter, or exactly the shape the surface declares."""
    declared = _declared_shapes(surface)
    out: dict[str, Array] = {}
    for name, raw in theta.items():
        a = np.asarray(raw, dtype=np.float64)
        want = declared.get(name, ())
        if a.shape != want:
            if want:
                raise ValueError(
                    f"theta[{name!r}] has shape {a.shape}; the surface declares shape {want}"
                )
            raise ValueError(
                f"theta[{name!r}] has shape {a.shape}; a point theta takes one scalar per "
                "parameter — pass a Posterior to allocate over draws"
            )
        out[name] = a
    return out


def _evaluator(
    surface: SupportsForward,
    posterior_or_theta: SupportsPosterior | Theta,
    names: tuple[str, ...],
    context: Mapping[str, Array],
    n_draws: int | None,
    rng: np.random.Generator,
) -> _Evaluator:
    if not isinstance(posterior_or_theta, SupportsPosterior):
        theta = _point_theta(surface, posterior_or_theta)
        return _Evaluator(surface, names, context, theta, (), 1)
    post = posterior_or_theta
    total = post.n_draws()
    if n_draws is not None:
        if n_draws < 1:
            raise ValueError("n_draws must be positive")
        if n_draws < total:
            idx = np.sort(rng.choice(total, size=n_draws, replace=False))
        else:
            idx = np.arange(total)
    else:
        idx = np.arange(total)
    flat: dict[str, Array] = {}
    for name in sorted(post.names()):
        a = np.asarray(post.draws(name), dtype=np.float64)
        flat[name] = a.reshape(total, *a.shape[2:])[idx]
    n = int(idx.size)
    if all(v.ndim == 1 for v in flat.values()):
        stacked = {k: v[:, None] for k, v in flat.items()}
        return _Evaluator(surface, names, context, stacked, (), n)
    per_draw = tuple({k: v[d] for k, v in flat.items()} for d in range(n))
    return _Evaluator(surface, names, context, None, per_draw, n)


def _aggregator(objective: Objective, q: float) -> Callable[[Array], Array]:
    if objective == "mean":
        return lambda v: np.asarray(np.mean(v, axis=0), dtype=np.float64)
    if objective == "quantile":
        return lambda v: np.asarray(np.quantile(v, q, axis=0), dtype=np.float64)
    raise ValueError(f"unknown objective {objective!r}; use 'mean' or 'quantile'")


# -- feasibility and starts -------------------------------------------------------------


def _box(bounds: Bounds) -> tuple[Array, Array]:
    return np.asarray(bounds.low, dtype=np.float64), np.asarray(bounds.high, dtype=np.float64)


def _centre(lo: Array, hi: Array, budget: float) -> Array:
    """Fill every treatment the same fraction of the way from ``low`` to ``high``."""
    room = float(np.sum(hi - lo))
    alpha = min(1.0, (budget - float(np.sum(lo))) / room) if room > 0 else 0.0
    out: Array = lo + max(alpha, 0.0) * (hi - lo)
    return out


def _corners(lo: Array, hi: Array, budget: float) -> list[Array]:
    """All-in starts: everything the budget allows on one treatment, the rest at ``low``."""
    spare = max(budget - float(np.sum(lo)), 0.0)
    out = []
    for i in range(lo.size):
        x = lo.copy()
        x[i] = min(hi[i], lo[i] + spare)
        out.append(x)
    return out


def _pairs(lo: Array, hi: Array, budget: float) -> list[Array]:
    """Two-way splits: the spare budget shared equally by a pair, the rest at ``low``.

    When one of the pair is capped by ``high`` the remainder goes to the
    other, up to its own cap.
    """
    spare = max(budget - float(np.sum(lo)), 0.0)
    out = []
    for i in range(lo.size):
        for j in range(i + 1, lo.size):
            x = lo.copy()
            give_i = min(hi[i] - lo[i], spare / 2.0)
            give_j = min(hi[j] - lo[j], spare / 2.0)
            give_i = min(hi[i] - lo[i], give_i + (spare / 2.0 - give_j))
            give_j = min(hi[j] - lo[j], give_j + (spare / 2.0 - give_i))
            x[i], x[j] = lo[i] + give_i, lo[j] + give_j
            out.append(x)
    return out


def _shrink(x: Array, lo: Array, hi: Array, budget: float) -> Array:
    """Clip to the box and, if over budget, shrink toward ``low`` until it is spent exactly."""
    x = np.asarray(np.clip(x, lo, hi), dtype=np.float64)
    spent, floor = float(np.sum(x)), float(np.sum(lo))
    if spent > budget and spent > floor:
        x = lo + (x - lo) * (budget - floor) / (spent - floor)
    return np.asarray(x, dtype=np.float64)


def _random_start(lo: Array, hi: Array, budget: float, rng: np.random.Generator) -> Array:
    return _shrink(lo + rng.uniform(size=lo.size) * (hi - lo), lo, hi, budget)


def _sparse_start(lo: Array, hi: Array, budget: float, rng: np.random.Generator) -> Array:
    """A Dirichlet(1/2) split of the spare budget: favours a few treatments funded, the rest low."""
    spare = max(budget - float(np.sum(lo)), 0.0)
    w = rng.dirichlet(np.full(lo.size, 0.5))
    return np.asarray(np.minimum(hi, lo + w * spare), dtype=np.float64)


def _presearch_pool(lo: Array, hi: Array, budget: float, rng: np.random.Generator) -> list[Array]:
    """``max(8, 4k)`` seeded random feasible points: uniform-in-the-box and sparse, alternating."""
    n = max(_PRESEARCH_MIN, 4 * int(lo.size))
    return [
        _random_start(lo, hi, budget, rng) if i % 2 == 0 else _sparse_start(lo, hi, budget, rng)
        for i in range(n)
    ]


def _perturbations(
    x: Array, lo: Array, hi: Array, budget: float, rng: np.random.Generator
) -> list[Array]:
    """Random feasible perturbations of ``x`` at each ``_PERTURB_SCALES`` × the movable budget."""
    span = min(max(budget - float(np.sum(lo)), 0.0), float(np.sum(hi - lo)))
    if span <= 0.0:
        return []
    k = int(lo.size)
    n = max(_PRESEARCH_MIN, 4 * k)
    out = []
    for s in _PERTURB_SCALES:
        for _ in range(n):
            step = s * span * rng.standard_normal(k) / math.sqrt(k)
            out.append(_shrink(x + step, lo, hi, budget))
    return out


def _resolve_start(
    start: Mapping[str, float], bounds: Bounds, lo: Array, hi: Array, budget: float
) -> Array:
    missing = [t for t in bounds.treatments if t not in start]
    if missing:
        raise ValueError(f"start lacks doses for {missing}")
    x = np.asarray([float(start[t]) for t in bounds.treatments], dtype=np.float64)
    return _shrink(x, lo, hi, budget)  # a warm start from a larger budget shrinks toward low


def _distinct(points: list[Array]) -> list[Array]:
    out: list[Array] = []
    for p in points:
        if not any(np.allclose(p, q, rtol=0.0, atol=1e-12) for q in out):
            out.append(p)
    return out


# -- SLSQP -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Solve:
    x: Array
    value: float
    converged: bool
    message: str
    status: int
    n_iter: int
    n_eval: int


@dataclass(frozen=True)
class _Probe:
    """The objective over the feasible probe points (``low``, the corners, the starts).

    ``scale`` is the spread of the probe values, floored by the rounding
    noise of their level; ``best`` is the largest probe value — the bar any
    solve must reach; ``step`` is the central-difference step as a fraction
    of the half-width, grown with ``level / scale`` so the rounding error of
    the difference quotient stays small when an intercept dwarfs the range.
    """

    scale: float
    best: float
    step: float


def _objective_scale(f_probe: Array) -> float:
    """An O(1) divisor for the objective: its spread over the probe points, never its level."""
    finite = f_probe[np.isfinite(f_probe)]
    if finite.size == 0:
        return 1.0
    spread = float(np.max(finite) - np.min(finite))
    return max(spread, 1e3 * _EPS * float(np.max(np.abs(finite))), 1e-12)


def _probe(
    ev: _Evaluator, agg: Callable[[Array], Array], points: list[Array]
) -> tuple[_Probe | None, Array]:
    """Evaluate every probe point in one ``forward`` call; ``None`` when no value is finite."""
    f = agg(ev.rows(np.vstack(points)))
    finite = f[np.isfinite(f)]
    if finite.size == 0:
        return None, f
    scale = _objective_scale(f)
    level = float(np.max(np.abs(finite)))
    step = min(_MAX_GRAD_STEP, max(_GRAD_STEP, (_EPS * max(level / scale, 1.0)) ** (1.0 / 3.0)))
    return _Probe(scale=scale, best=float(np.max(finite)), step=step), f


def _best_finite(f: Array) -> int | None:
    """Index of the largest finite entry, or ``None``."""
    masked = np.where(np.isfinite(f), f, -np.inf)
    j = int(np.argmax(masked))
    return j if math.isfinite(float(masked[j])) else None


def _slsqp(
    ev: _Evaluator,
    agg: Callable[[Array], Array],
    lo: Array,
    hi: Array,
    budget: float,
    x0: Array,
    probe: _Probe,
    *,
    maxiter: int,
    tol: float,
) -> _Solve:
    k = lo.size
    center, hw = (lo + hi) / 2.0, (hi - lo) / 2.0
    bscale = max(1.0, abs(budget))
    f_start = float(agg(ev.rows(x0[None, :]))[0])
    if not math.isfinite(f_start):
        return _Solve(x0, f_start, False, "objective is not finite at the start point", -1, 0, 1)
    scale, ref = probe.scale, probe.best
    counter = {"n": 0}

    def fun(c: Array) -> tuple[float, Array]:
        x = center + hw * c
        rows = np.repeat(x[None, :], 2 * k + 1, axis=0)
        steps = np.empty(k)
        for i in range(k):
            h = probe.step * hw[i]
            up, down = min(x[i] + h, hi[i]), max(x[i] - h, lo[i])
            rows[1 + 2 * i, i], rows[2 + 2 * i, i] = up, down
            steps[i] = up - down
        f = agg(ev.rows(rows))
        counter["n"] += 1
        g_x = (f[1::2] - f[2::2]) / steps
        return -(float(f[0]) - ref) / scale, -(g_x * hw) / scale

    constraint = {
        "type": "ineq",
        "fun": lambda c: (budget - float(np.sum(center + hw * c))) / bscale,
        "jac": lambda c: -hw / bscale,
    }
    c0 = np.clip((x0 - center) / np.where(hw > 0, hw, 1.0), -1.0, 1.0)
    res = _minimize(
        fun,
        c0,
        jac=True,
        method="SLSQP",
        bounds=[(-1.0, 1.0)] * k,
        constraints=[constraint],
        options={"maxiter": maxiter, "ftol": tol},
    )
    x = np.asarray(np.clip(center + hw * np.asarray(res.x, dtype=np.float64), lo, hi))
    value = float(agg(ev.rows(x[None, :]))[0])
    converged = bool(res.success)
    message = str(res.message)
    n_iter = int(res.nit)
    if converged and float(np.sum(x)) - budget > _FEASIBILITY_TOL * bscale:
        converged = False
        message = f"solution exceeds the budget by {float(np.sum(x)) - budget:.3g}"
    if converged and not math.isfinite(value):
        converged = False
        message = "objective is not finite at the solution"
    if converged and n_iter <= 1 and value < ref - _STALL_TOL * scale:
        converged = False
        message = (
            f"stalled at the start point after {n_iter} iteration(s) with objective "
            f"{value:.6g}, below the best probe point {ref:.6g}"
        )
    return _Solve(
        x=x,
        value=value,
        converged=converged,
        message=message,
        status=int(res.status),
        n_iter=n_iter,
        n_eval=counter["n"],
    )


def _allocate_slsqp(
    ev: _Evaluator,
    agg: Callable[[Array], Array],
    bounds: Bounds,
    budget: float,
    *,
    start: Mapping[str, float] | None,
    n_starts: int,
    maxiter: int,
    tol: float,
    rng: np.random.Generator,
) -> tuple[_Solve, dict[str, float]] | Unsupported:
    lo, hi = _box(bounds)
    k = int(lo.size)
    # fixed starts: the warm start, the all-in corners, the two-way splits, the centre
    fixed: list[Array] = []
    if start is not None:
        fixed.append(_resolve_start(start, bounds, lo, hi, budget))
    fixed += _corners(lo, hi, budget)
    if k <= _MAX_PAIR_K:
        fixed += _pairs(lo, hi, budget)
    fixed.append(_centre(lo, hi, budget))
    fixed = _distinct(fixed)
    # the pre-search pool is only evaluated; its best few are promoted to starts
    pool = _presearch_pool(lo, hi, budget, rng)
    n_random = n_starts - 1
    requested = [_random_start(lo, hi, budget, rng) for _ in range(n_random)]
    probe, f_probe = _probe(ev, agg, [lo, *fixed, *pool, *requested])
    if probe is None:
        return Unsupported(
            reason="objective is not finite at any feasible probe point "
            "(the lower bounds, the corners, the splits, the centre, the random pool)",
            detail={"method": "slsqp", "n_probe": str(1 + len(fixed) + len(pool) + n_random)},
        )
    f_pool = f_probe[1 + len(fixed) : 1 + len(fixed) + len(pool)]
    order = np.argsort(np.where(np.isfinite(f_pool), f_pool, -np.inf))[::-1]
    promoted = [pool[int(j)] for j in order[:_N_PROMOTED] if math.isfinite(float(f_pool[j]))]
    starts = _distinct([*fixed, *promoted, *requested])
    solves = [_slsqp(ev, agg, lo, hi, budget, x0, probe, maxiter=maxiter, tol=tol) for x0 in starts]
    converged = [s for s in solves if s.converged]
    n_failed = len(solves) - len(converged)
    common = {
        "method": "slsqp",
        "n_starts": str(len(solves)),
        "n_random_starts": str(n_random),
        "n_presearch": str(len(pool)),
        "maxiter": str(maxiter),
    }
    if not converged:
        worst = solves[0]
        return Unsupported(
            reason=f"allocator did not converge (SLSQP): {worst.message}",
            detail={**common, "scipy_status": str(worst.status), "n_iter": str(worst.n_iter)},
        )
    best = max(converged, key=lambda s: s.value)
    if best.value < probe.best - _STALL_TOL * probe.scale:
        failed = [s.message for s in solves if not s.converged]
        return Unsupported(
            reason=f"allocator did not converge (SLSQP): the best converged solve "
            f"({best.value:.6g}) is below a feasible probe point ({probe.best:.6g}); "
            f"the solve from that point failed: {failed[0] if failed else 'unknown'}",
            detail={**common, "best_solve": f"{best.value:.6g}", "probe": f"{probe.best:.6g}"},
        )
    # perturbation check: a feasible point near the solution that beats it means a missed basin
    n_perturb, n_restarts = 0, 0
    for _round in range(_MAX_RESTARTS + 1):
        points = _perturbations(best.x, lo, hi, budget, rng)
        if not points:
            break
        f_pert = agg(ev.rows(np.vstack(points)))
        n_perturb += len(points)
        j = _best_finite(f_pert)
        if j is None or float(f_pert[j]) <= best.value + _STALL_TOL * probe.scale:
            break
        beat = float(f_pert[j])
        if _round == _MAX_RESTARTS:
            return Unsupported(
                reason=f"allocator did not converge (SLSQP): after {n_restarts} restart(s) a "
                f"feasible perturbation ({beat:.6g}) still beats the best converged solve "
                f"({best.value:.6g}); raise n_starts or maxiter",
                detail={**common, "best_solve": f"{best.value:.6g}", "perturbation": f"{beat:.6g}"},
            )
        n_restarts += 1
        again = _slsqp(
            ev,
            agg,
            lo,
            hi,
            budget,
            points[j],
            _Probe(scale=probe.scale, best=max(probe.best, beat), step=probe.step),
            maxiter=maxiter,
            tol=tol,
        )
        solves.append(again)
        if not again.converged or again.value < beat - _STALL_TOL * probe.scale:
            return Unsupported(
                reason=f"allocator did not converge (SLSQP): a feasible perturbation "
                f"({beat:.6g}) beats the best converged solve ({best.value:.6g}) and the solve "
                f"from it failed: {again.message}",
                detail={**common, "best_solve": f"{best.value:.6g}", "perturbation": f"{beat:.6g}"},
            )
        best = again
    detail = {
        "n_iter": float(best.n_iter),
        "n_eval": float(best.n_eval),
        "n_starts": float(len(solves)),
        "n_random_starts": float(n_random),
        "n_presearch": float(len(pool)),
        "n_promoted": float(len(promoted)),
        "n_failed_starts": float(n_failed),
        "n_perturbations": float(n_perturb),
        "n_restarts": float(n_restarts),
        "scipy_status": float(best.status),
    }
    return best, detail


# -- cvxpy: piecewise-linear concave surrogate ------------------------------------------


def _allocate_cvxpy(
    ev: _Evaluator,
    agg: Callable[[Array], Array],
    bounds: Bounds,
    budget: float,
    *,
    n_grid: int,
    rng: np.random.Generator,
) -> tuple[_Solve, dict[str, float]] | Unsupported:
    try:
        import cvxpy as _cvxpy
    except ImportError:
        return Unsupported(
            reason="method='cvxpy' needs cvxpy; install axiom[convex] or use method='slsqp'",
            missing=("cvxpy",),
        )
    cp: Any = _cvxpy  # cvxpy ships no type information
    if n_grid < 3:
        raise ValueError("n_grid must be at least 3")
    lo, hi = _box(bounds)
    k = lo.size
    grids = [np.linspace(lo[i], hi[i], n_grid) for i in range(k)]
    n_check = 8
    checks = np.asarray([lo + rng.uniform(size=k) * (hi - lo) for _ in range(n_check)])
    # rows: baseline, k * n_grid one-at-a-time, then each check point and its k projections
    rows = [lo[None, :]]
    for i in range(k):
        block = np.repeat(lo[None, :], n_grid, axis=0)
        block[:, i] = grids[i]
        rows.append(block)
    for c in checks:
        rows.append(c[None, :])
        proj = np.repeat(lo[None, :], k, axis=0)
        proj[np.arange(k), np.arange(k)] = c
        rows.append(proj)
    f = agg(ev.rows(np.concatenate(rows, axis=0)))
    if not np.all(np.isfinite(f)):
        return Unsupported(reason="objective is not finite on the bounds; cannot build a surrogate")
    base = float(f[0])
    # tolerances are relative to the response RANGE: relative to the level, an intercept
    # large enough would let a sigmoid pass as concave
    spread = max(float(np.max(f) - np.min(f)), 1e-12)
    level = float(np.max(np.abs(f)))
    additive_tol, concave_tol = 1e-6 * spread, 1e-8 * spread
    rounding = 16.0 * _EPS * level  # noise in a second difference of values at this level
    if concave_tol < rounding:
        return Unsupported(
            reason="method='cvxpy' cannot verify concavity: the response level "
            f"(~{level:.3g}) dwarfs its range over the bounds ({spread:.3g}) and rounding "
            "exceeds the tolerance — use method='slsqp'",
            detail={"level": f"{level:.3g}", "range": f"{spread:.3g}"},
        )
    responses = [f[1 + i * n_grid : 1 + (i + 1) * n_grid] - base for i in range(k)]
    pos = 1 + k * n_grid
    for _ in range(n_check):
        whole, parts = float(f[pos]), f[pos + 1 : pos + 1 + k]
        additive = base + float(np.sum(parts - base))
        if abs(whole - additive) > additive_tol:
            return Unsupported(
                reason="method='cvxpy' needs an additively separable surface "
                "(forward(dose) = Σ_i f_i(dose_i)); this one is not — use method='slsqp'",
                detail={"max_deviation": f"{abs(whole - additive):.3g}"},
            )
        pos += 1 + k
    for i, r in enumerate(responses):
        second = np.diff(r, n=2)
        if np.any(second > concave_tol):
            return Unsupported(
                reason=f"method='cvxpy' needs a concave response; {bounds.treatments[i]!r} "
                "is not concave on its bounds — use method='slsqp'",
                detail={"treatment": bounds.treatments[i]},
            )
    x = cp.Variable(k)
    t = cp.Variable(k)
    constraints: list[Any] = [cp.sum(x) <= budget, x >= lo, x <= hi]
    for i in range(k):
        g, r = grids[i], responses[i]
        slopes = np.diff(r) / np.diff(g)
        for j in range(n_grid - 1):
            constraints.append(t[i] <= r[j] + slopes[j] * (x[i] - g[j]))
    problem = cp.Problem(cp.Maximize(cp.sum(t)), constraints)
    problem.solve()
    status = str(problem.status)
    if status != "optimal" or x.value is None:
        return Unsupported(
            reason=f"allocator did not converge (cvxpy): status {status!r}",
            detail={"method": "cvxpy", "cvxpy_status": status},
        )
    xs = np.asarray(np.clip(np.asarray(x.value, dtype=np.float64).reshape(k), lo, hi))
    bscale = max(1.0, abs(budget))
    if float(np.sum(xs)) - budget > _FEASIBILITY_TOL * bscale:
        return Unsupported(
            reason=f"cvxpy solution exceeds the budget by {float(np.sum(xs)) - budget:.3g}",
            detail={"method": "cvxpy", "cvxpy_status": status},
        )
    value = float(agg(ev.rows(xs[None, :]))[0])
    if not math.isfinite(value):
        return Unsupported(reason="objective is not finite at the cvxpy solution")
    surrogate = base + float(problem.value)
    solve = _Solve(xs, value, True, status, 0, 0, 1)
    detail = {
        "n_grid": float(n_grid),
        "surrogate_value": surrogate,
        "surrogate_gap": value - surrogate,
        "solver_iters": float(getattr(problem.solver_stats, "num_iters", 0) or 0),
    }
    return solve, detail


# -- public -----------------------------------------------------------------------------


def allocate(
    surface: SupportsForward,
    posterior_or_theta: SupportsPosterior | Theta | Failure,
    *,
    budget: float,
    bounds: Bounds,
    objective: Objective = "mean",
    q: float = 0.5,
    seed: int | None = None,
    method: AllocationMethod = "slsqp",
    start: Mapping[str, float] | None = None,
    n_starts: int = 1,
    n_draws: int | None = None,
    maxiter: int = 200,
    tol: float = 1e-12,
    n_grid: int = 101,
    context: Context | None = None,
) -> Allocation | Failure:
    """Maximize the posterior-expected outcome subject to ``Σ dose ≤ budget`` and ``bounds``.

    ``posterior_or_theta`` is either a ``SupportsPosterior`` (the objective
    averages — or takes the ``q``-quantile of — ``forward`` over its draws,
    ``n_draws`` of them subsampled with ``seed`` when given) or a single
    parameter mapping (a point plan: one scalar per parameter, or the shape
    the surface declares). ``bounds`` names the treatments — every one a
    column the surface reads; every other column the surface reads (a unit
    index, nuisance basis columns) comes from ``context`` as one value
    broadcast to every candidate row; a column read as a ``Gather`` index
    must be a zero-based integer inside the gathered axis. ``start`` adds a
    warm start to the fixed start set (the all-in corners, the two-way
    splits, the centre, the best of a seeded random pre-search pool);
    ``n_starts - 1`` extra seeded random feasible starts guard further
    against local optima on non-concave surfaces, and a perturbation check
    on the solution restarts from any feasible point that beats it.
    ``maxiter`` / ``tol`` go to SLSQP; ``n_grid`` sizes the cvxpy surrogate.
    A budget equal to the sum of the lower bounds (to tolerance) leaves one
    feasible point, ``low``, which is returned without a solve
    (``detail["budget_equals_sum_low"] == 1``).

    Returns ``Unsupported`` — never numbers — when the surface has carryover
    (use ``surface.steady_state()``), the budget is below the sum of the
    lower bounds, the solver does not converge, the answer is infeasible or
    below a feasible probe or perturbation point, or the objective is not
    finite. A typed failure passed in place of the posterior (an
    ``Unverified`` fit) is returned unchanged. Mismatched columns, shapes,
    and settings raise ``ValueError`` naming the offender.
    """
    if isinstance(posterior_or_theta, Unsupported | Blocked | Unverified):
        return posterior_or_theta
    if not 0.0 < q < 1.0:
        raise ValueError("q must lie strictly inside (0, 1)")
    if not math.isfinite(budget):
        raise ValueError("budget must be finite")
    if n_starts < 1:
        raise ValueError("n_starts must be positive")
    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    agg = _aggregator(objective, q)
    if method not in ("slsqp", "cvxpy"):
        raise ValueError(f"unknown method {method!r}; use 'slsqp' or 'cvxpy'")
    carried = carryover_failure(surface)
    if carried is not None:
        return carried
    ctx = _context_columns(surface, bounds, context)
    rng = np.random.default_rng(seed)
    lo, hi = _box(bounds)
    floor = float(np.sum(lo))
    if floor > budget + _FEASIBILITY_TOL * max(1.0, abs(budget)):
        return Unsupported(
            reason=f"budget {budget:.6g} is below the sum of lower bounds {floor:.6g}; "
            "no feasible allocation exists",
            detail={"budget": f"{budget:.6g}", "sum_low": f"{floor:.6g}"},
        )
    ev = _evaluator(surface, posterior_or_theta, bounds.treatments, ctx, n_draws, rng)
    if budget - floor <= _FEASIBILITY_TOL * max(1.0, abs(budget)):
        # the feasible set is the single point ``low``: nothing to solve
        at_low = float(agg(ev.rows(lo[None, :]))[0])
        if not math.isfinite(at_low):
            return Unsupported(
                reason=f"budget {budget:.6g} equals the sum of lower bounds {floor:.6g}, so the "
                "only feasible allocation is the lower bounds, where the objective is not finite",
                detail={"budget": f"{budget:.6g}", "sum_low": f"{floor:.6g}"},
            )
        return Allocation(
            doses={t: float(v) for t, v in zip(bounds.treatments, lo, strict=True)},
            expected_outcome=at_low,
            budget=float(budget),
            objective=objective,
            q=q,
            method=method,
            seed=seed,
            detail={
                "budget_equals_sum_low": 1.0,
                "n_iter": 0.0,
                "n_eval": 1.0,
                "n_starts": 0.0,
                "n_draws": float(ev.n_draws),
            },
        )
    if method == "slsqp":
        solved = _allocate_slsqp(
            ev,
            agg,
            bounds,
            budget,
            start=start,
            n_starts=n_starts,
            maxiter=maxiter,
            tol=tol,
            rng=rng,
        )
    else:
        solved = _allocate_cvxpy(ev, agg, bounds, budget, n_grid=n_grid, rng=rng)
    if isinstance(solved, Unsupported):
        return solved
    best, detail = solved
    detail["n_draws"] = float(ev.n_draws)
    return Allocation(
        doses={t: float(v) for t, v in zip(bounds.treatments, best.x, strict=True)},
        expected_outcome=best.value,
        budget=float(budget),
        objective=objective,
        q=q,
        method=method,
        seed=seed,
        detail=detail,
    )
