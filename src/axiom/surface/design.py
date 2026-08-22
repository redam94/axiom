"""Classical and optimal experimental designs over ``k`` continuous treatments.

``Bounds`` names the treatments and their dose ranges; a ``Design`` is a list
of dose points plus the recipe that produced it. The classical generators
(central composite, Box-Behnken, full and fractional factorial, Latin
hypercube, equal spacing) are deterministic given their seed. Optimal designs
come from Fedorov point exchange over a candidate ``Design`` under a D/A/E
criterion computed on the design matrix ``surface.linearize`` returns.

D-optimality on a nonlinear surface is local (review B11): the information
matrix depends on the nonlinear parameters the linearization is taken at.
``bayesian_criterion`` therefore averages the criterion over prior draws; a
single draw is the local special case. Criteria never raise on a singular
information matrix — they return ``-inf`` / ``-inf`` / ``0`` so an exchange
loop can step over rank-deficient candidates, and ``optimal_exchange``
returns ``Unsupported`` when no restart reaches a nonsingular design.

Singularity is decided on the singular values of the column-equilibrated
matrix ``X / scale`` (each column divided by its max-abs entry), never on
``X'X`` and never in raw dose units: ``X'X`` counts as singular when ``n < p``
or when the smallest singular value of the scaled matrix is below
``max(n, p) * eps`` of its largest — ``numpy.linalg.matrix_rank``'s floor.
Raw-unit conditioning (a dose range of ``1000`` puts ``1``, ``1e3`` and
``1e6`` in adjacent columns of a quadratic basis) is a property of the
units, not of the design, and must not fail a well-posed design. The
criteria themselves are still the values for ``X'X`` in the caller's units —
``log det(X'X)``, ``-trace((X'X)^-1)``, ``lambda_min(X'X)`` — computed from
the scaled decomposition so they stay accurate at raw-unit condition
numbers where ``slogdet``/``eigvalsh`` of ``X'X`` would not be.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import model_validator

from axiom.core import DesignMatrix, NonEmptyStr, Spec, SupportsForward, Unsupported

__all__ = [
    "Bounds",
    "Criterion",
    "Design",
    "a_criterion",
    "bayesian_criterion",
    "box_behnken",
    "central_composite",
    "d_criterion",
    "defining_relation",
    "e_criterion",
    "equal_spacing",
    "fractional_factorial",
    "full_factorial",
    "latin_hypercube",
    "optimal_exchange",
]

Array = npt.NDArray[np.float64]
Criterion = Literal["d", "a", "e"]
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_EPS = float(np.finfo(np.float64).eps)


# -- specs -----------------------------------------------------------------------------


class Bounds(Spec):
    """The experimental region: a box ``[low_i, high_i]`` per treatment, in dose units.

    Coded units put the box at ``[-1, 1]^k``; ``from_coded`` / ``to_coded``
    map between the two.
    """

    treatments: tuple[NonEmptyStr, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]

    @model_validator(mode="after")
    def _consistent(self) -> Bounds:
        k = len(self.treatments)
        if k == 0:
            raise ValueError("Bounds needs at least one treatment")
        if len(set(self.treatments)) != k:
            raise ValueError("treatment names must be unique")
        if len(self.low) != k or len(self.high) != k:
            raise ValueError(f"low and high must each have {k} entries, one per treatment")
        for name, lo, hi in zip(self.treatments, self.low, self.high, strict=True):
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError(f"{name}: bounds must be finite")
            if not lo < hi:
                raise ValueError(f"{name}: low {lo} must be strictly below high {hi}")
        return self

    @property
    def k(self) -> int:
        return len(self.treatments)

    def center(self) -> Array:
        lo, hi = np.asarray(self.low, dtype=np.float64), np.asarray(self.high, dtype=np.float64)
        return (lo + hi) / 2.0

    def half_width(self) -> Array:
        lo, hi = np.asarray(self.low, dtype=np.float64), np.asarray(self.high, dtype=np.float64)
        return (hi - lo) / 2.0

    def from_coded(self, coded: npt.ArrayLike) -> Array:
        """Map coded ``[-1, 1]`` coordinates (last axis = treatments) to dose units."""
        c = np.asarray(coded, dtype=np.float64)
        out: Array = self.center() + self.half_width() * c
        return out

    def to_coded(self, doses: npt.ArrayLike) -> Array:
        x = np.asarray(doses, dtype=np.float64)
        out: Array = (x - self.center()) / self.half_width()
        return out

    def clip(self, doses: npt.ArrayLike) -> Array:
        x = np.asarray(doses, dtype=float)
        return np.asarray(np.clip(x, np.asarray(self.low), np.asarray(self.high)), dtype=np.float64)


class Design(Spec):
    """A set of dose points (``points[i][j]`` is treatment ``j`` at run ``i``) and its recipe.

    ``kind`` names the generator; ``detail`` carries its numbers (alpha,
    resolution, criterion value, seed). Replicated runs are allowed.
    """

    treatments: tuple[NonEmptyStr, ...]
    points: tuple[tuple[float, ...], ...]
    kind: NonEmptyStr
    detail: dict[str, float] = {}

    @model_validator(mode="after")
    def _consistent(self) -> Design:
        k = len(self.treatments)
        if k == 0:
            raise ValueError("Design needs at least one treatment")
        if len(set(self.treatments)) != k:
            raise ValueError("treatment names must be unique")
        if not self.points:
            raise ValueError("Design needs at least one point")
        for i, p in enumerate(self.points):
            if len(p) != k:
                raise ValueError(f"point {i} has {len(p)} coordinates; expected {k}")
            if not all(math.isfinite(v) for v in p):
                raise ValueError(f"point {i} has a non-finite coordinate")
        return self

    @property
    def n(self) -> int:
        return len(self.points)

    @property
    def k(self) -> int:
        return len(self.treatments)

    def as_array(self) -> Array:
        return np.asarray(self.points, dtype=np.float64).reshape(self.n, self.k)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.as_array(), columns=list(self.treatments))

    def doses(self) -> dict[str, Array]:
        """The ``dose`` mapping ``SupportsForward.forward`` / ``linearize`` take."""
        arr = self.as_array()
        return {t: arr[:, j].copy() for j, t in enumerate(self.treatments)}


def _design(bounds: Bounds, coded: Array, kind: str, detail: Mapping[str, float]) -> Design:
    doses = bounds.from_coded(coded)
    return Design(
        treatments=bounds.treatments,
        points=tuple(tuple(float(v) for v in row) for row in doses),
        kind=kind,
        detail={k: float(v) for k, v in detail.items()},
    )


# -- two-level machinery ---------------------------------------------------------------


def _parse_generator(text: str, k: int) -> tuple[int, int, frozenset[int]]:
    """``"D=ABC"`` or ``"D=-ABC"`` -> (target index, sign, base indices)."""
    lhs, sep, rhs = text.replace(" ", "").upper().partition("=")
    if not sep or len(lhs) != 1 or not rhs:
        raise ValueError(f"generator {text!r} must look like 'D=ABC'")
    sign = 1
    if rhs[0] in "+-":
        sign = -1 if rhs[0] == "-" else 1
        rhs = rhs[1:]
    letters = [lhs, *rhs]
    if any(c not in _LETTERS or _LETTERS.index(c) >= k for c in letters):
        raise ValueError(f"generator {text!r} names a factor outside A..{_LETTERS[k - 1]}")
    if len(set(rhs)) != len(rhs) or lhs in rhs:
        raise ValueError(f"generator {text!r} repeats a factor")
    return _LETTERS.index(lhs), sign, frozenset(_LETTERS.index(c) for c in rhs)


def _words(generators: Sequence[str], k: int) -> tuple[frozenset[int], ...]:
    """All words of the defining relation: products of non-empty subsets of the generators."""
    parsed = [_parse_generator(g, k) for g in generators]
    targets = [t for t, _, _ in parsed]
    if len(set(targets)) != len(targets):
        raise ValueError("two generators define the same factor")
    for t, _, base in parsed:
        if base & set(targets):
            raise ValueError(f"generator for {_LETTERS[t]} uses a generated factor on its right")
    gen_words = [base | {t} for t, _, base in parsed]
    words: list[frozenset[int]] = []
    for r in range(1, len(gen_words) + 1):
        for combo in itertools.combinations(gen_words, r):
            w = frozenset[int]()
            for gw in combo:
                w = w ^ gw
            if not w:
                raise ValueError("generators are inconsistent: a product collapses to I")
            words.append(w)
    return tuple(words)


def defining_relation(generators: Sequence[str]) -> tuple[str, ...]:
    """The words of ``I = ...`` implied by two-level generators, e.g. ``("ABCD",)`` for ``D=ABC``.

    The resolution of the fraction is the length of the shortest word.
    """
    if not generators:
        return ()
    k = 1 + max(
        _LETTERS.index(c) for g in generators for c in g.replace(" ", "").upper() if c in _LETTERS
    )
    words = _words(generators, k)
    names = {"".join(_LETTERS[i] for i in sorted(w)) for w in words}
    return tuple(sorted(names, key=lambda w: (len(w), w)))


def _two_level(k: int, generators: Sequence[str]) -> tuple[Array, int | None]:
    """Coded runs of a 2^(k-p) design (p = len(generators)) and its resolution."""
    if k > len(_LETTERS):
        raise ValueError("two-level designs are lettered A..Z; at most 26 treatments")
    parsed = [_parse_generator(g, k) for g in generators]
    resolution = min(len(w) for w in _words(generators, k)) if generators else None
    targets = {t for t, _, _ in parsed}
    base = [i for i in range(k) if i not in targets]
    runs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(base))), dtype=np.float64)
    runs = runs.reshape(-1, len(base))[:, ::-1]  # first factor varies fastest
    coded = np.zeros((runs.shape[0], k))
    for j, i in enumerate(base):
        coded[:, i] = runs[:, j]
    for t, sign, rhs in parsed:
        coded[:, t] = sign * np.prod(coded[:, sorted(rhs)], axis=1)
    return coded, resolution


def _auto_generators(k: int, p: int) -> tuple[str, ...]:
    """Choose ``p`` generators for a 2^(k-p) fraction maximizing resolution (then word length)."""
    if p < 1 or k - p < 2:
        raise ValueError(f"fraction {p} leaves {k - p} base factors; need at least two")
    base, targets = _LETTERS[: k - p], _LETTERS[k - p : k]
    subsets = ["".join(c) for r in range(len(base), 1, -1) for c in itertools.combinations(base, r)]
    if len(subsets) < p:
        raise ValueError(f"cannot build {p} distinct generators from {len(base)} base factors")

    def score(gens: tuple[str, ...]) -> tuple[int, int]:
        words = _words(gens, k)
        return min(len(w) for w in words), sum(len(w) for w in words)

    if math.comb(len(subsets), p) <= 20_000:
        best = max(
            (
                tuple(f"{t}={s}" for t, s in zip(targets, combo, strict=True))
                for combo in itertools.combinations(subsets, p)
            ),
            key=score,
        )
        return best
    chosen: list[str] = []
    for t in targets:  # greedy fallback for large fractions
        chosen.append(max((f"{t}={s}" for s in subsets), key=lambda g: score((*chosen, g))))
    return tuple(chosen)


# -- classical generators ----------------------------------------------------------------


def central_composite(
    bounds: Bounds,
    *,
    alpha: float | Literal["rotatable", "face"] = "rotatable",
    center_points: int = 4,
    fraction: int | None = None,
    inscribed: bool = False,
) -> Design:
    """Cube (2^k, or 2^(k-p) with ``fraction=p``), 2k axial points at ``±alpha``, centre runs.

    Coded ``[-1, 1]`` is the box; a rotatable ``alpha = n_cube ** 1/4`` puts
    the axial points outside it (circumscribed). ``inscribed=True`` rescales
    so the axial points land on the box instead.
    """
    k = bounds.k
    if center_points < 0:
        raise ValueError("center_points must be non-negative")
    gens = _auto_generators(k, fraction) if fraction is not None else ()
    cube, resolution = _two_level(k, gens)
    n_cube = cube.shape[0]
    if isinstance(alpha, str):
        if alpha not in ("rotatable", "face"):
            raise ValueError("alpha must be a positive number, 'rotatable', or 'face'")
        a = n_cube**0.25 if alpha == "rotatable" else 1.0
    else:
        a = float(alpha)
        if not a > 0:
            raise ValueError("alpha must be positive")
    axial = np.zeros((2 * k, k))
    for i in range(k):
        axial[2 * i, i], axial[2 * i + 1, i] = -a, a
    coded = np.vstack([cube, axial, np.zeros((center_points, k))])
    if inscribed:
        coded = coded / max(a, 1.0)
    detail: dict[str, float] = {
        "alpha": a,
        "n_cube": n_cube,
        "n_axial": 2 * k,
        "n_center": center_points,
        "fraction": fraction or 0,
    }
    if resolution is not None:
        detail["resolution"] = resolution
    return _design(bounds, coded, "central_composite", detail)


_BB_TRIPLES: dict[int, tuple[tuple[int, int, int], ...]] = {
    # Box & Behnken (1960), Table: blocks of three treatments run as a 2^3 factorial.
    # k=6: the cyclic development of {1, 2, 4} mod 6 (a partially balanced block design;
    # pairs (1,4), (2,5), (3,6) occur twice, every other pair once): 6 x 8 = 48 runs.
    6: ((0, 1, 3), (1, 2, 4), (2, 3, 5), (0, 3, 4), (1, 4, 5), (0, 2, 5)),
    # k=7: the Fano plane, {1, 2, 4} mod 7 (the unique BIBD(7, 3, 1)): 7 x 8 = 56 runs.
    7: ((0, 1, 3), (1, 2, 4), (2, 3, 5), (3, 4, 6), (0, 4, 5), (1, 5, 6), (0, 2, 6)),
}


def box_behnken(bounds: Bounds, *, center_points: int = 3) -> Design:
    """The published Box-Behnken design for ``k`` in 3..7: blocked two-level factorials.

    ``k <= 5`` pairs every two treatments in a ``2^2`` factorial with the rest at
    ``0`` (edge midpoints of the cube: 12, 24, 40 runs). ``k = 6`` and ``7`` use
    Box and Behnken's three-treatment blocks run as ``2^3`` factorials (48 and
    56 runs). Larger ``k`` raises: the published tables for 9-16 treatments
    are not transcribed, and an all-pairs construction under this name would
    not be the Box-Behnken design.
    """
    k = bounds.k
    if k < 3:
        raise ValueError("a Box-Behnken design needs at least three treatments")
    if k > 7:
        raise ValueError(
            f"Box-Behnken designs are implemented for 3 to 7 treatments (got {k}); "
            "use central_composite or optimal_exchange for more"
        )
    if center_points < 0:
        raise ValueError("center_points must be non-negative")
    blocks: tuple[tuple[int, ...], ...]
    if k in _BB_TRIPLES:
        blocks = _BB_TRIPLES[k]
    else:
        blocks = tuple(itertools.combinations(range(k), 2))
    rows = []
    for block in blocks:
        for signs in itertools.product((-1.0, 1.0), repeat=len(block)):
            row = np.zeros(k)
            row[list(block)] = signs
            rows.append(row)
    coded = np.vstack([np.asarray(rows), np.zeros((center_points, k))])
    detail = {
        "n_blocks": len(blocks),
        "block_size": len(blocks[0]),
        "n_block_runs": len(rows),
        "n_center": center_points,
    }
    return _design(bounds, coded, "box_behnken", detail)


def _grid(bounds: Bounds, levels: tuple[int, ...], kind: str) -> Design:
    if any(n < 2 for n in levels):
        raise ValueError("every treatment needs at least two levels")
    axes = [np.linspace(-1.0, 1.0, n) for n in levels]
    coded = np.asarray(list(itertools.product(*axes)), dtype=np.float64)
    detail = {f"levels[{t}]": n for t, n in zip(bounds.treatments, levels, strict=True)}
    return _design(bounds, coded, kind, detail)


def full_factorial(bounds: Bounds, levels: int | tuple[int, ...]) -> Design:
    """Every combination of equally spaced levels (``levels`` per treatment, or one count)."""
    lv = (levels,) * bounds.k if isinstance(levels, int) else tuple(levels)
    if len(lv) != bounds.k:
        raise ValueError(f"levels must have {bounds.k} entries, one per treatment")
    return _grid(bounds, lv, "full_factorial")


def equal_spacing(bounds: Bounds, n_per_treatment: int) -> Design:
    """The naive comparator: a regular grid with ``n_per_treatment`` levels on every axis."""
    return _grid(bounds, (n_per_treatment,) * bounds.k, "equal_spacing")


def fractional_factorial(bounds: Bounds, generators: tuple[str, ...]) -> Design:
    """A two-level 2^(k-p) fraction from generators like ``("D=ABC",)``; A is the first treatment.

    ``detail["resolution"]`` is the shortest word of the defining relation;
    ``defining_relation`` lists the words (the confounding pattern).
    """
    if not generators:
        raise ValueError("fractional_factorial needs at least one generator; use full_factorial")
    coded, resolution = _two_level(bounds.k, generators)
    detail = {
        "resolution": float(resolution or 0),
        "fraction": len(generators),
        "n_runs": coded.shape[0],
    }
    return _design(bounds, coded, "fractional_factorial", detail)


def _min_distance(u: Array) -> float:
    if u.shape[0] < 2:
        return math.inf
    d = np.linalg.norm(u[:, None, :] - u[None, :, :], axis=-1)
    return float(d[np.triu_indices(u.shape[0], k=1)].min())


def latin_hypercube(
    bounds: Bounds,
    n: int,
    *,
    seed: int | None = None,
    centered: bool = False,
    n_restarts: int = 10,
) -> Design:
    """``n`` points with one per stratum per treatment; maximin over ``n_restarts`` draws."""
    if n < 1:
        raise ValueError("n must be positive")
    if n_restarts < 1:
        raise ValueError("n_restarts must be positive")
    rng = np.random.default_rng(seed)
    k = bounds.k
    best: Array | None = None
    best_score = -math.inf
    for _ in range(n_restarts):
        u = np.empty((n, k))
        for j in range(k):
            offset = 0.5 if centered else rng.uniform(size=n)
            u[:, j] = (rng.permutation(n) + offset) / n
        score = _min_distance(u)
        if best is None or score > best_score:
            best, best_score = u, score
    assert best is not None
    detail: dict[str, float] = {"n_restarts": n_restarts}
    if math.isfinite(best_score):  # undefined for a single point: no key rather than 0.0
        detail["min_distance"] = best_score
    if seed is not None:
        detail["seed"] = seed
    return _design(bounds, 2.0 * best - 1.0, "latin_hypercube", detail)


# -- criteria ----------------------------------------------------------------------------

# (column scale, singular values of X / scale descending, right singular vectors as rows,
#  n, p) — the decomposition every criterion is computed from
_Spectrum = tuple[Array, Array, Array, int, int]


def _spectrum(X: DesignMatrix | npt.ArrayLike) -> _Spectrum | None:
    """SVD of the column-equilibrated design matrix ``X / scale``.

    ``scale`` is each column's max-abs entry (``1`` for an all-zero column, which
    stays zero and is caught as rank deficiency). Equilibrating first makes
    the rank decision a property of the design's geometry rather than of the
    dose units. ``None`` when ``X`` is empty, non-finite, or the
    decomposition fails.
    """
    A = np.asarray(X.X if isinstance(X, DesignMatrix) else X, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"design matrix must be 2-D; got shape {A.shape}")
    if A.size == 0 or not np.all(np.isfinite(A)):
        return None
    n, p = A.shape
    scale = np.max(np.abs(A), axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    try:
        _, s, vt = np.linalg.svd(A / scale, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(s)):
        return None
    return scale, np.asarray(s, dtype=np.float64), np.asarray(vt, dtype=np.float64), n, p


def _rank(spec: _Spectrum | None) -> int:
    """Rank of the scaled matrix under ``numpy.linalg.matrix_rank``'s ``max(n, p) eps`` floor."""
    if spec is None:
        return 0
    _, s, _, n, p = spec
    if s.size == 0 or not s[0] > 0:
        return 0
    return int(np.count_nonzero(s > max(n, p) * _EPS * s[0]))


def _condition(spec: _Spectrum | None) -> float:
    """Condition number of the scaled matrix; ``inf`` below full rank."""
    if spec is None or _rank(spec) < spec[4]:
        return math.inf
    return float(spec[1][0] / spec[1][-1])


def _value(criterion: str, spec: _Spectrum | None) -> float:
    """The criterion for ``X'X`` in the caller's units; ``-inf`` / ``-inf`` / ``0`` below full rank.

    With ``X = Xs S`` (``Xs`` scaled, ``S = diag(scale)``) and ``Xs = U diag(s) V'``:
    ``log det(X'X) = 2 sum log s + 2 sum log scale``, and ``X^+ = S^-1 V diag(1/s) U'``
    so ``trace((X'X)^-1) = ||X^+||_F^2 = ||W||_F^2`` and
    ``lambda_min(X'X) = 1 / ||X^+||_2^2 = 1 / ||W||_2^2`` with ``W = S^-1 V diag(1/s)``
    (``U`` has orthonormal columns, so it drops out of both norms). Every
    quantity is read off the well-conditioned scaled factorization; the
    smallest singular value of ``X`` is the largest of ``W``, which an SVD
    resolves to relative accuracy.
    """
    if spec is None or _rank(spec) < spec[4]:
        return 0.0 if criterion == "e" else -math.inf
    scale, s, vt, _, _ = spec
    if criterion == "d":
        return float(2.0 * (np.sum(np.log(s)) + np.sum(np.log(scale))))
    W = (vt.T / scale[:, None]) / s[None, :]
    if criterion == "a":
        return float(-np.sum(W * W))
    return float(1.0 / np.linalg.norm(W, 2) ** 2)


def d_criterion(X: DesignMatrix | npt.ArrayLike) -> float:
    """``log det(X'X)``; ``-inf`` when the information matrix is singular.

    Singular means ``n < p`` or a smallest singular value of the
    column-equilibrated ``X`` below ``max(n, p) * eps`` of its largest — the
    same scale-free floor ``a_criterion`` and ``e_criterion`` apply. The
    value is for ``X`` in the caller's units: scaling a column by ``c``
    adds ``2 log c``.
    """
    return _value("d", _spectrum(X))


def a_criterion(X: DesignMatrix | npt.ArrayLike) -> float:
    """``-trace((X'X)^-1)`` (larger is better); ``-inf`` when singular."""
    return _value("a", _spectrum(X))


def e_criterion(X: DesignMatrix | npt.ArrayLike) -> float:
    """Smallest eigenvalue of ``X'X``; ``0`` when singular (a zero eigenvalue, not a guess)."""
    return _value("e", _spectrum(X))


def _singular_score(criterion: str, v: float) -> bool:
    """Whether a criterion value marks a singular information matrix."""
    return v <= 0.0 if criterion == "e" else not math.isfinite(v)


def _better(v: float, ref: float) -> bool:
    """``v`` improves on ``ref`` beyond rounding; any finite value beats ``-inf``."""
    if not math.isfinite(ref):
        return math.isfinite(v) and v > ref
    return v > ref + 1e-12 * abs(ref)


Score = tuple[float, float]  # (mean rank, mean criterion) over draws


def _better_score(a: Score, b: Score) -> bool:
    """Rank first, so a singular start can climb out one exchange at a time; then the criterion."""
    if a[0] > b[0] + 1e-9:
        return True
    if a[0] < b[0] - 1e-9:
        return False
    return _better(a[1], b[1])


_CRITERIA: dict[str, Callable[[DesignMatrix | npt.ArrayLike], float]] = {
    "d": d_criterion,
    "a": a_criterion,
    "e": e_criterion,
}


def _criterion_fn(criterion: str) -> Callable[[DesignMatrix | npt.ArrayLike], float]:
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {sorted(_CRITERIA)}; got {criterion!r}")
    return _CRITERIA[criterion]


def bayesian_criterion(
    surface: SupportsForward,
    design: Design,
    theta_draws: Sequence[Mapping[str, npt.ArrayLike]],
    criterion: Criterion = "d",
) -> float:
    """Mean over ``theta_draws`` of the criterion on ``surface.linearize(design, theta).X``.

    One draw is the local criterion at that point (review B11).
    """
    if not theta_draws:
        raise ValueError("theta_draws must contain at least one draw")
    fn = _criterion_fn(criterion)
    doses = design.doses()
    return float(np.mean([fn(surface.linearize(doses, theta)) for theta in theta_draws]))


# -- point exchange ----------------------------------------------------------------------


def optimal_exchange(
    surface: SupportsForward,
    candidates: Design,
    n: int,
    theta_draws: Sequence[Mapping[str, npt.ArrayLike]],
    *,
    criterion: Criterion = "d",
    seed: int | None = None,
    max_passes: int = 20,
    n_restarts: int = 3,
) -> Design | Unsupported:
    """Fedorov point exchange: the best ``n`` candidate rows (replicates allowed) by the criterion.

    The candidate set is linearized once per draw, so the surface must be
    row-wise (run ``i`` of ``X`` depends only on dose row ``i``). Each restart
    draws a random start, then sweeps every design row against every
    candidate until a pass yields no improvement or ``max_passes`` is hit. A
    singular start (criterion ``-inf``, or ``0`` for ``"e"``) is not frozen:
    any exchange that makes the design nonsingular is accepted. Returns
    ``Unsupported`` when ``n`` is below the number of linear coefficients or
    when no restart reaches a nonsingular design.

    ``detail["condition"]`` is the condition number of the column-equilibrated
    design matrix at the returned design (the largest over draws): the
    scale-free distance from singularity the rank decision was made on,
    independent of the dose units.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if max_passes < 1 or n_restarts < 1:
        raise ValueError("max_passes and n_restarts must be positive")
    if not theta_draws:
        raise ValueError("theta_draws must contain at least one draw")
    _criterion_fn(criterion)
    doses = candidates.doses()
    mats = [np.asarray(surface.linearize(doses, th).X, dtype=np.float64) for th in theta_draws]
    for M in mats:
        if M.ndim != 2 or M.shape[0] != candidates.n or M.shape[1] != mats[0].shape[1]:
            raise ValueError(
                f"linearize must return one row per candidate ({candidates.n}); got {M.shape}"
            )
    n_cand, p = candidates.n, int(mats[0].shape[1])
    base_detail = {
        "n": str(n),
        "n_candidates": str(n_cand),
        "n_coefficients": str(p),
        "criterion": criterion,
    }
    if n < p:
        return Unsupported(
            reason=f"n={n} runs cannot estimate {p} linear coefficients; need n >= {p}",
            detail=base_detail,
        )

    def spectra(idx: npt.NDArray[np.int64]) -> list[_Spectrum | None]:
        return [_spectrum(M[idx]) for M in mats]

    def score(idx: npt.NDArray[np.int64]) -> Score:
        specs = spectra(idx)
        rank = float(np.mean([_rank(e) for e in specs]))
        return rank, float(np.mean([_value(criterion, e) for e in specs]))

    rng = np.random.default_rng(seed)
    best_idx: npt.NDArray[np.int64] | None = None
    best_score: Score = (0.0, -math.inf)
    best_start: Score = (0.0, -math.inf)
    best_passes = 0
    for _ in range(n_restarts):
        idx = np.asarray(rng.choice(n_cand, size=n, replace=n > n_cand), dtype=np.int64)
        current = start = score(idx)
        passes = 0
        while passes < max_passes:
            passes += 1
            improved = False
            for i in range(n):
                best_c, best_v = int(idx[i]), current
                for c in range(n_cand):
                    if c == idx[i]:
                        continue
                    trial = idx.copy()
                    trial[i] = c
                    v = score(trial)
                    if _better_score(v, best_v):
                        best_c, best_v = c, v
                if best_c != idx[i]:
                    idx[i], current, improved = best_c, best_v, True
            if not improved:
                break
        if best_idx is None or _better_score(current, best_score):
            best_idx, best_score, best_start, best_passes = idx, current, start, passes
    if best_idx is None or _singular_score(criterion, best_score[1]):
        return Unsupported(
            reason=(
                f"no restart reached a nonsingular design: {criterion}-criterion over "
                f"{n_restarts} restart(s) of n={n} from {n_cand} candidates for {p} coefficients"
            ),
            detail={**base_detail, "n_restarts": str(n_restarts)},
        )
    arr = candidates.as_array()[np.sort(best_idx)]
    order = np.lexsort(arr.T[::-1])
    detail: dict[str, float] = {
        "criterion": best_score[1],
        "condition": max(_condition(e) for e in spectra(best_idx)),
        "n_passes": best_passes,
        "n_restarts": n_restarts,
        "n_candidates": n_cand,
        "n_draws": len(theta_draws),
    }
    if not _singular_score(criterion, best_start[1]):
        detail["start_criterion"] = best_start[1]
        detail["improvement"] = best_score[1] - best_start[1]
    if seed is not None:
        detail["seed"] = seed
    return Design(
        treatments=candidates.treatments,
        points=tuple(tuple(float(v) for v in row) for row in arr[order]),
        kind=f"{criterion}_optimal",
        detail=detail,
    )
