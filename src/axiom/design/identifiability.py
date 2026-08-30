"""What a design can estimate, and what it can only estimate *in combination*.

``design.structural`` answers "how precisely will this design pin each
parameter down", which presumes each parameter is pinned down at all. For a
nonlinear model that presumption is often false, and false in a specific,
diagnosable way: some *combination* of the parameters is estimable and the
parameters separately are not. A saturating response measured only at low
dose is the canonical case — ``alpha x / (k + x)`` is ``(alpha / k) x`` there,
and no amount of data at low dose separates ``alpha`` from ``k``, because
doubling both changes nothing at all.

That last sentence is the whole method. If scaling several parameters
together leaves every prediction unchanged, the likelihood is flat along
that direction and the data cannot move it. In log-parameter coordinates
such a scaling is a *constant* direction, which makes the search for it
linear algebra:

* build the sensitivity matrix ``S_ij = d f_i / d log theta_j`` at a
  parameter point, over every observation the design would take;
* its null space is the set of scalings that change no prediction — the
  symmetries;
* its row space is what the design *can* see;
* an integer vector ``w`` in either space names a monomial
  ``prod_j theta_j ** w_j``: ``(1, -1)`` is ``alpha / k``, ``(1, 1)`` is
  ``alpha * k``. Those are the combinations to report and to put priors on.

Three things that get called "not identified"
--------------------------------------------
They want different remedies, and the report keeps them apart.

* **An accident of this parameter value.** Pass several points (prior draws)
  as ``at=`` and a direction flat at only some of them drops out of
  ``persistent_deficiency``.
* **A property of this design.** Flat at every parameter value, but a
  different dose, a later period or a second measured quantity would break
  it. This is what ``prescribe_measurements`` looks for, and the remedy is
  to measure elsewhere.
* **A symmetry of the model.** ``a * b * x`` exposes only ``a * b``, at every
  parameter value and under every design. The remedy is to reparameterize —
  report ``a * b`` and stop pretending ``a`` was ever a quantity.

The analysis can separate the first from the other two on its own; separating
the second from the third takes candidate designs, which is why
``prescribe_measurements`` exists and why ``persistent_deficiency`` is
named for what it measures rather than for what one would like it to mean.

Then, because a rank statement is not yet a decision:

* ``simulated_identifiability`` runs the experiment you are proposing —
  simulates the outcome at a known truth, refits, and profiles the
  likelihood of each target — so "not identified" becomes an interval that
  is infinite on one side rather than an eigenvalue near zero. This is the
  practical identifiability of Raue et al. (2009), and it catches the case
  the rank test misses: a parameter that is identified in principle and
  unbounded in practice at this noise level and this sample size.
* ``prescribe_measurements`` asks the constructive question. Given
  candidates — another dose, a later period, an intermediate quantity you
  could measure — which smallest set restores full rank, and which symmetry
  does each one break?

Method notes
------------
Derivatives are central differences in log-parameter space, ``theta_j
exp(±h)``, which is exactly the log derivative rather than a scaled linear
one. A column whose differences sit below ``100`` times their round-off
floor is reported as exactly zero and named in ``zero_columns``, the same
convention ``design.structural`` uses: a mean that does not move with a
parameter at this design is a finding, not a small number. Log coordinates
need positive parameters; ``scaling="absolute"`` drops that requirement and
gives linear combinations instead of monomials.

Naming a combination is a *search over small integer vectors*, scored the way
a statistician would score it rather than geometrically: a symmetry by how
little the predictions move along it, an estimable combination by how
precisely the data pins it down, both relative to the stiffest direction of
the design so neither depends on the parameters' units. The search can fail
to name a direction that is genuinely there; when it does, the numerical
``null_basis`` is still reported and ``search_limits_hit`` says so.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator
from scipy import optimize
from scipy import stats as _st

from axiom.core import (
    Expr,
    ModelSpec,
    NonEmptyStr,
    Spec,
    Unsupported,
    constrain,
    free_parameters,
    log_likelihood,
    unconstrain,
    value,
)
from axiom.design.weighting import Weighting

__all__ = [
    "Combination",
    "EstimabilityReport",
    "Observation",
    "Prescription",
    "ProfileReport",
    "Scaling",
    "SensitivityMatrix",
    "estimable_combinations",
    "observation_from_model",
    "prescribe_measurements",
    "profile_combination",
    "profile_likelihood",
    "sensitivity_matrix",
    "simulated_identifiability",
]

Array = npt.NDArray[np.float64]
Theta = Mapping[str, npt.ArrayLike]
Scaling = Literal["log", "absolute"]

_REL_STEP = 1e-4
"""Relative step for the central difference. Larger than ``design.structural``'s
``1e-5`` because the perturbation is multiplicative in log space, where the
truncation error is what dominates."""
_RESOLUTION = 100.0
_EPS = float(np.finfo(np.float64).eps)
_RANK_TOL = 1e-8
"""Singular values below this multiple of the largest count as zero."""
_SINGULAR_FLOOR = 1e-12
"""Machine-level floor. The *practical* rank tolerance decides what counts as flat;
this decides what counts as numerically absent, which is a different question — a
combination pointing along a direction with no information at all is not estimable
at any tolerance, while one pointing slightly along a nearly-flat direction merely
has an amplified standard error, and saying so is the whole point."""
_CHI2_95 = float(_st.chi2.ppf(0.95, df=1)) / 2.0
"""The profile-likelihood drop for a 95% interval on one parameter: 1.92."""


# -- what is observed ------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    """One measurable quantity, the data it is measured at, and its noise.

    Not a ``Spec``: it holds the design's columns, which are arrays. An
    observation is whatever a single expression evaluates to — a whole
    panel's worth of rows for a fitted model's mean, one row for a single
    extra dose, or a quantity you do not currently measure but could.
    """

    label: str
    expr: Expr
    data: Mapping[str, Any] = field(default_factory=dict)
    noise_sd: float = 1.0
    weighting: Weighting | None = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("an observation needs a label")
        if not self.noise_sd > 0:
            raise ValueError(f"observation {self.label!r}: noise_sd must be positive")
        if self.weighting is not None and self.noise_sd != 1.0:
            raise ValueError(
                f"observation {self.label!r}: a {self.weighting.family} weighting already "
                "carries its dispersion; set either weighting= or noise_sd=, not both"
            )

    def evaluate(self, theta: Theta) -> Array:
        return np.atleast_1d(
            np.asarray(value(self.expr, data=self.data, params=theta), float)
        ).ravel()

    def root_weights(self, theta: Theta) -> Array:
        """``sqrt(w)`` per row, so that ``S = sqrt(W) J`` and ``S'S = J' W J``.

        Applied to the finished difference quotient and never inside it: the
        weights are a function of the mean, so folding them in before
        differencing would add a spurious ``mu · d sqrt(w) / d theta`` term.
        With a constant ``noise_sd`` that distinction does not arise, which is
        why the Gaussian-only code could scale wherever it liked.
        """
        if self.weighting is None:
            return np.asarray(1.0 / self.noise_sd, dtype=np.float64).reshape(1)
        w = np.asarray(self.weighting.at(self.evaluate(theta), self.data), dtype=np.float64)
        return np.sqrt(np.atleast_1d(w).ravel())


def observation_from_model(
    model: ModelSpec,
    data: Mapping[str, Any],
    *,
    noise_sd: float | None = None,
    theta: Theta | None = None,
    label: str = "",
    weighted: bool = False,
) -> Observation:
    """The observation a fitted ``ModelSpec`` represents: its mean over the panel's rows.

    ``noise_sd`` defaults to the likelihood's scale parameter at ``theta``
    when the model names one and it is available, and to ``1.0`` otherwise —
    which only rescales the information, so the rank statements are
    unaffected and the precision ones are in units of that scale.

    ``weighted=True`` instead takes the model's likelihood at its word and
    weights each row by ``1 / (phi V(mu))`` (see ``design.weighting``). For a
    normal likelihood the two agree; for any other family the unweighted form
    is measuring a Gaussian that is not there, so the rank statements survive
    but the precision ones do not.
    """
    if weighted:
        if model.likelihood.family != "normal" and noise_sd is not None:
            raise ValueError(
                f"a weighted {model.likelihood.family} observation carries its own "
                "dispersion; do not also pass noise_sd"
            )
        return Observation(
            label=label or model.name or "mean",
            expr=model.mean,
            data=dict(data),
            weighting=Weighting.from_model(model, theta),
        )
    scale = noise_sd
    if scale is None and model.likelihood.scale and theta is not None:
        found = theta.get(model.likelihood.scale)
        if found is not None:
            scale = float(np.mean(np.asarray(found, dtype=float)))
    return Observation(
        label=label or model.name or "mean",
        expr=model.mean,
        data=dict(data),
        noise_sd=float(scale) if scale else 1.0,
    )


# -- sensitivity -----------------------------------------------------------------------


@dataclass(frozen=True)
class SensitivityMatrix:
    """``d f / d log theta`` (or ``d f / d theta``) stacked over observations.

    Not a ``Spec`` — it is ``rows x parameters`` of floats and specs hold no
    large arrays. What survives into a spec is the *analysis*:
    ``EstimabilityReport``.
    """

    matrix: Array
    parameters: tuple[str, ...]
    observations: tuple[str, ...]
    scaling: Scaling
    theta: dict[str, float]
    zero_columns: tuple[str, ...] = ()

    @property
    def rows(self) -> int:
        return int(self.matrix.shape[0])

    def information(self) -> Array:
        """``S' S`` — the Fisher information in these coordinates (noise already folded in)."""
        return self.matrix.T @ self.matrix


def _scalar_coordinates(
    theta: Theta, names: Sequence[str]
) -> tuple[list[str], list[tuple[str, int]]]:
    labels: list[str] = []
    index: list[tuple[str, int]] = []
    for name in names:
        flat = np.atleast_1d(np.asarray(theta[name], dtype=float)).ravel()
        for i in range(flat.size):
            labels.append(name if flat.size == 1 else f"{name}[{i}]")
            index.append((name, i))
    return labels, index


def _perturbed(
    theta: Theta, name: str, position: int, delta: float, *, scaling: Scaling
) -> dict[str, Any]:
    out = {k: np.array(np.asarray(v, dtype=float)) for k, v in theta.items()}
    flat = np.atleast_1d(out[name]).ravel().copy()
    if scaling == "log":
        flat[position] = flat[position] * math.exp(delta)
    else:
        flat[position] = flat[position] + delta
    out[name] = flat.reshape(np.asarray(theta[name], dtype=float).shape)
    return out


def sensitivity_matrix(
    observations: Sequence[Observation],
    theta: Theta,
    *,
    parameters: Sequence[str] | None = None,
    scaling: Scaling = "log",
    step: float = _REL_STEP,
    parameter_scales: Mapping[str, float] | None = None,
) -> SensitivityMatrix | Unsupported:
    """Central differences of every observation with respect to every parameter.

    ``scaling="log"`` differentiates with respect to ``log theta`` by
    perturbing ``theta_j exp(±step)``; every parameter must then be strictly
    positive, and one that is not comes back ``Unsupported`` naming it, since
    "the ratio is identified" is a statement about positive quantities.
    ``scaling="absolute"`` uses an additive step of ``step * max(|theta_j|,
    scale_j)`` and gives linear combinations instead of monomials.
    """
    if not observations:
        raise ValueError("no observations were given")
    names = list(parameters) if parameters is not None else list(theta)
    missing = [n for n in names if n not in theta]
    if missing:
        raise KeyError(f"theta does not give a value for {missing}")
    labels, index = _scalar_coordinates(theta, names)
    if not labels:
        raise ValueError("no parameter coordinates to differentiate")
    scales = dict(parameter_scales or {})

    if scaling == "log":
        bad = sorted(
            {name for name in names if np.any(np.asarray(theta[name], dtype=float) <= 0.0)}
        )
        if bad:
            return Unsupported(
                reason=(
                    f"log-scaled sensitivity needs positive parameters; {bad} are not positive "
                    "at this point. Use scaling='absolute' for linear combinations instead"
                ),
                missing=tuple(bad),
                detail={"scaling": "log"},
            )
    else:
        undecided = sorted(
            {
                name
                for name in names
                if np.any(np.asarray(theta[name], dtype=float) == 0.0) and name not in scales
            }
        )
        if undecided:
            return Unsupported(
                reason=(
                    f"{undecided} sit at zero and have no entry in parameter_scales, so there "
                    "is no scale to take a step relative to"
                ),
                missing=tuple(undecided),
            )

    # Unweighted throughout: J is the derivative of the observation itself, and the
    # rows are scaled by sqrt(w) once the differences are formed (Observation.
    # root_weights). The round-off test below is invariant to that scaling anyway --
    # both the difference and the floor carry the same factor -- so the zero-column
    # verdict is identical to the one the noise_sd-scaled code reached.
    base = np.concatenate([o.evaluate(theta) for o in observations])
    if not np.all(np.isfinite(base)):
        return Unsupported(
            reason="the observations are not finite at this parameter point",
            detail={
                "theta": ", ".join(
                    f"{k}={float(np.mean(np.asarray(v, float))):.6g}" for k, v in theta.items()
                )
            },
        )
    columns: list[Array] = []
    zero_columns: list[str] = []
    for label, (name, position) in zip(labels, index, strict=True):
        flat = np.atleast_1d(np.asarray(theta[name], dtype=float)).ravel()
        if scaling == "log":
            delta = step
            denominator = 2.0 * step
        else:
            delta = step * max(abs(float(flat[position])), scales.get(name, 0.0), 0.0)
            if delta == 0.0:  # pragma: no cover - guarded above
                delta = step
            denominator = 2.0 * delta
        up = np.concatenate(
            [
                o.evaluate(_perturbed(theta, name, position, delta, scaling=scaling))
                for o in observations
            ]
        )
        down = np.concatenate(
            [
                o.evaluate(_perturbed(theta, name, position, -delta, scaling=scaling))
                for o in observations
            ]
        )
        difference = up - down
        floor = _EPS * float(np.max(np.abs(base)) if base.size else 1.0)
        if float(np.max(np.abs(difference))) <= _RESOLUTION * floor:
            columns.append(np.zeros_like(base))
            zero_columns.append(label)
            continue
        columns.append(difference / denominator)
    matrix = np.column_stack(columns) if columns else np.zeros((base.size, 0))
    try:
        root_w = np.concatenate(
            [np.broadcast_to(o.root_weights(theta), o.evaluate(theta).shape) for o in observations]
        )
    except ValueError as exc:
        return Unsupported(
            reason=f"an observation's weighting is undefined at this parameter point: {exc}",
            detail={
                "weighting": ", ".join(
                    o.weighting.label for o in observations if o.weighting is not None
                )
            },
        )
    matrix = root_w[:, None] * matrix
    if not np.all(np.isfinite(matrix)):
        return Unsupported(
            reason="the sensitivity is not finite; the model is not differentiable at this point",
            detail={"parameters": ", ".join(labels)},
        )
    return SensitivityMatrix(
        matrix=matrix,
        parameters=tuple(labels),
        observations=tuple(o.label for o in observations),
        scaling=scaling,
        theta={
            label: float(np.atleast_1d(np.asarray(theta[name], float)).ravel()[position])
            for label, (name, position) in zip(labels, index, strict=True)
        },
        zero_columns=tuple(zero_columns),
    )


# -- combinations ----------------------------------------------------------------------


class Combination(Spec):
    """A monomial (or linear combination) of parameters, and how flat it is.

    ``exponents`` maps a parameter coordinate to an integer power under
    ``scaling="log"`` — ``{"alpha": 1, "k": -1}`` is ``alpha / k`` — and to a
    coefficient under ``scaling="absolute"``.

    ``score`` is what the search measured, and its meaning follows ``kind``:

    * a **symmetry**'s score is its *flatness*, ``|S w| / (sigma_max |w|)``
      — how much the predictions move along this direction relative to the
      direction they move most. Zero is an exact symmetry; the search keeps
      anything at or below the tolerance.
    * an **estimable** combination's score is its *noise amplification*,
      ``sigma_max sqrt(w' (S'S)^-1 w) / |w|`` — the standard error of this
      combination in units of the best-determined direction's. One is the
      best any combination can do; the search keeps anything at or below
      ``1 / tolerance``.

    Both are scale-free, so neither depends on the units the parameters
    are measured in.
    """

    exponents: dict[str, int]
    kind: Literal["symmetry", "estimable"]
    scaling: Scaling = "log"
    score: float = Field(ge=0.0)
    label: str = ""

    @model_validator(mode="after")
    def _nonzero(self) -> Combination:
        if not self.exponents or all(v == 0 for v in self.exponents.values()):
            raise ValueError("a combination needs at least one non-zero exponent")
        return self

    @property
    def support(self) -> tuple[str, ...]:
        return tuple(sorted(k for k, v in self.exponents.items() if v != 0))

    def describe(self) -> str:
        """The statement the combination makes, in the direction its ``kind`` means.

        An estimable combination is a quantity: ``alpha / k``. A symmetry is
        not a quantity but a *motion* — the thing you can do to the
        parameters without changing any prediction — so it renders as one:
        ``alpha -> alpha*c, k -> k*c``.
        """
        if self.kind == "estimable":
            return self.render()
        if self.scaling == "absolute":
            moves = [
                f"{n} -> {n} {'+' if v > 0 else '-'} {abs(v) if abs(v) != 1 else ''}c".replace(
                    "  ", " "
                )
                for n, v in sorted(self.exponents.items())
                if v
            ]
            return ", ".join(moves)
        moves = []
        for name, power in sorted(self.exponents.items()):
            if not power:
                continue
            factor = "c" if abs(power) == 1 else f"c^{abs(power)}"
            moves.append(
                f"{name} -> {name}*{factor}" if power > 0 else f"{name} -> {name}/{factor}"
            )
        return ", ".join(moves)

    def render(self) -> str:
        """``alpha / k``, ``alpha^2 * beta``, or ``alpha - k`` for absolute scaling."""
        items = sorted((k, v) for k, v in self.exponents.items() if v != 0)
        if self.scaling == "absolute":
            parts = []
            for i, (name, coefficient) in enumerate(items):
                sign = "-" if coefficient < 0 else ("+" if i else "")
                magnitude = abs(coefficient)
                term = name if magnitude == 1 else f"{magnitude}*{name}"
                parts.append(f"{sign} {term}" if i or sign else term)
            return " ".join(parts).strip()
        top = [(n, v) for n, v in items if v > 0]
        bottom = [(n, -v) for n, v in items if v < 0]

        def piece(pairs: list[tuple[str, int]]) -> str:
            return " * ".join(n if p == 1 else f"{n}^{p}" for n, p in pairs)

        if not bottom:
            return piece(top)
        if not top:
            return f"1 / {piece(bottom)}"
        return f"{piece(top)} / {piece(bottom)}"


class EstimabilityReport(Spec):
    """What this design can and cannot separate, at one parameter point and across several.

    ``rank`` and ``deficiency`` are at the reference point. ``symmetries``
    are directions the design does not move — reparameterize along them or
    measure something that breaks them; ``estimable`` are the combinations
    it does move, and are what a report should quote when the parameters
    themselves are not identified. ``persistent_deficiency`` counts the
    directions that stayed flat at *every* parameter point tried, which rules
    out the flatness being an accident of the values you happened to pick.
    It does **not** rule out its being a property of the design: a saturating
    response measured only in its linear regime is flat in ``alpha * k`` at
    every ``(alpha, k)`` you try, and the cure there is a different dose, not
    a different parameterization. Only a direction that survives varying the
    *design* as well — which is what ``prescribe_measurements`` tries to do —
    is a symmetry of the model itself. With ``n_points == 1`` the field
    necessarily equals ``deficiency`` and says nothing at all.

    ``singular_values`` is the spectrum, in units of the sensitivity's own
    scaling, so a reader can see whether the rank decision was clear-cut or
    a judgement call at the tolerance.
    """

    parameters: tuple[NonEmptyStr, ...] = Field(min_length=1)
    observations: tuple[str, ...] = ()
    scaling: Scaling = "log"
    n_points: int = Field(default=1, ge=1)
    rank: int = Field(ge=0)
    deficiency: int = Field(ge=0)
    persistent_deficiency: int = Field(ge=0)
    singular_values: tuple[float, ...] = ()
    condition_number: float = Field(ge=1.0)
    symmetries: tuple[Combination, ...] = ()
    estimable: tuple[Combination, ...] = ()
    null_basis: tuple[tuple[float, ...], ...] = ()
    zero_columns: tuple[str, ...] = ()
    tolerance: float = Field(gt=0)
    search_limits_hit: tuple[str, ...] = ()
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _consistent(self) -> EstimabilityReport:
        if self.rank + self.deficiency != len(self.parameters):
            raise ValueError("rank + deficiency must be the number of parameter coordinates")
        if self.persistent_deficiency > self.deficiency:
            raise ValueError("a structural null direction is null at the reference point too")
        for row in self.null_basis:
            if len(row) != len(self.parameters):
                raise ValueError("each null direction has one entry per parameter")
        return self

    @property
    def identified(self) -> bool:
        """True when the design moves every parameter direction — full rank."""
        return self.deficiency == 0

    def summary(self) -> str:
        if self.identified:
            return f"all {len(self.parameters)} parameters are locally identified"
        flat = "; ".join(c.describe() for c in self.symmetries) or "see null_basis"
        estimable = ", ".join(c.render() for c in self.estimable) or "none nameable"
        return (
            f"{self.deficiency} of {len(self.parameters)} directions are flat "
            f"[{flat}]; estimable: {estimable}"
        )


def _normalized_integer(vector: Sequence[int]) -> tuple[int, ...] | None:
    values = [int(v) for v in vector]
    if all(v == 0 for v in values):
        return None
    divisor = 0
    for v in values:
        divisor = math.gcd(divisor, abs(v))
    values = [v // divisor for v in values]
    first = next(v for v in values if v != 0)
    if first < 0:
        values = [-v for v in values]
    return tuple(values)


def _integer_candidates(
    dimension: int, max_support: int, max_coefficient: int, limit: int
) -> tuple[list[tuple[int, ...]], bool]:
    """Small integer vectors, by increasing support then increasing coefficient size."""
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[int, ...]] = []
    truncated = False
    coefficients = [c for c in range(-max_coefficient, max_coefficient + 1) if c != 0]
    for support in range(1, max_support + 1):
        for positions in combinations(range(dimension), support):
            for values in product(coefficients, repeat=support):
                vector = [0] * dimension
                for position, v in zip(positions, values, strict=True):
                    vector[position] = v
                normalized = _normalized_integer(vector)
                if normalized is None or normalized in seen:
                    continue
                seen.add(normalized)
                out.append(normalized)
                if len(out) >= limit:
                    return out, True
    return out, truncated


def _independent(vectors: Sequence[tuple[int, ...]], wanted: int) -> list[tuple[int, ...]]:
    """Greedily keep vectors that add a dimension, in the order given."""
    kept: list[tuple[int, ...]] = []
    basis = np.zeros((0, len(vectors[0]))) if vectors else np.zeros((0, 0))
    for vector in vectors:
        if len(kept) >= wanted:
            break
        trial = np.vstack([basis, np.asarray(vector, dtype=float)])
        if np.linalg.matrix_rank(trial, tol=1e-10) > basis.shape[0]:
            basis = trial
            kept.append(vector)
    return kept


def _search(
    singular: Array,
    right: Array,
    labels: Sequence[str],
    *,
    kind: Literal["symmetry", "estimable"],
    scaling: Scaling,
    wanted: int,
    tolerance: float,
    max_support: int,
    max_coefficient: int,
    limit: int,
) -> tuple[tuple[Combination, ...], bool]:
    """Small integer directions that are flat, or that are well determined.

    ``singular`` are the singular values of the sensitivity matrix and
    ``right`` its right singular vectors as *rows* (numpy's ``vt``), so a
    direction's coordinates in that basis are ``right @ w``.

    A symmetry is scored by how little the predictions move along it; an
    estimable combination by how precisely the data pins it down. Both
    scores are relative to the stiffest direction, so both are free of the
    parameters' units, and both use the same tolerance from opposite ends.
    """
    if wanted <= 0 or right.size == 0:
        return (), False
    largest = float(singular[0]) if singular.size else 0.0
    if largest <= 0:
        return (), False
    # There are min(rows, parameters) singular values but always `parameters`
    # right singular vectors; the missing ones are exact zeros — directions the
    # design has no row to see at all.
    spectrum = np.zeros(right.shape[0], dtype=np.float64)
    spectrum[: singular.size] = singular
    singular = spectrum
    keep = singular > _SINGULAR_FLOOR * largest
    candidates, truncated = _integer_candidates(len(labels), max_support, max_coefficient, limit)
    ceiling = tolerance if kind == "symmetry" else 1.0 / tolerance
    scored: list[tuple[int, float, tuple[int, ...]]] = []
    for vector in candidates:
        w = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(w))
        coordinates = right @ w
        if kind == "symmetry":
            score = float(np.linalg.norm(singular * coordinates)) / (largest * norm)
        else:
            dropped = coordinates[~keep] if keep.size else coordinates
            if dropped.size and float(np.max(np.abs(dropped))) > 1e-9 * norm:
                continue  # a component the design cannot see at all
            usable = coordinates[keep]
            variance = float(np.sum((usable / singular[keep]) ** 2))
            score = largest * math.sqrt(variance) / norm
        if score <= ceiling:
            cost = sum(1 for v in vector if v != 0) * 100 + int(np.sum(np.abs(w)))
            scored.append((cost, score, vector))
    scored.sort()
    chosen = _independent([v for _, _, v in scored], wanted)
    scores = {v: s for _, s, v in scored}
    out = []
    for vector in chosen:
        exponents = {label: int(v) for label, v in zip(labels, vector, strict=True) if v}
        combination = Combination(
            exponents=exponents,
            kind=kind,
            scaling=scaling,
            score=scores[vector],
        )
        out.append(combination.model_copy(update={"label": combination.describe()}))
    return tuple(out), truncated


def estimable_combinations(
    observations: Sequence[Observation],
    theta: Theta,
    *,
    at: Sequence[Theta] = (),
    parameters: Sequence[str] | None = None,
    scaling: Scaling = "log",
    tolerance: float = _RANK_TOL,
    direction_tolerance: float | None = None,
    max_support: int = 3,
    max_coefficient: int = 3,
    max_candidates: int = 200_000,
    step: float = _REL_STEP,
    parameter_scales: Mapping[str, float] | None = None,
) -> EstimabilityReport | Unsupported:
    """Rank, flat directions, and the named combinations this design can estimate.

    ``theta`` is the reference point. ``at`` adds further points — prior
    draws are the natural choice — and a direction counts as *structural*
    only if it is flat at every one of them; that is the difference between
    "this design cannot separate them" and "nothing can".

    ``tolerance`` decides two things at once and is the dial worth
    understanding. A singular value below ``tolerance`` times the largest
    counts as zero, so ``1e-8`` asks a *structural* question ("is this
    direction flat to machine precision?") and ``1e-2`` asks a *practical*
    one ("is it a hundred times flatter than the stiffest direction, so
    that no realistic sample separates it?"). The same number sets the bar
    for naming combinations, unless ``direction_tolerance`` overrides it: a
    symmetry must be flatter than it, and an estimable combination's
    standard error must be within ``1 / tolerance`` of the best any
    combination achieves.

    The search for named combinations covers integer vectors with at most
    ``max_support`` non-zero entries and coefficients up to
    ``max_coefficient``. It is a search, not a proof: when it finds nothing
    the numerical ``null_basis`` is still reported, and hitting the
    ``max_candidates`` ceiling is recorded in ``search_limits_hit`` rather
    than passed off as an empty result.

    Pass ``parameters=`` to restrict the analysis to the mean's parameters:
    a likelihood scale that the mean does not depend on shows up as an
    exactly flat direction, which is true but not what the question was.
    """
    reference = sensitivity_matrix(
        observations,
        theta,
        parameters=parameters,
        scaling=scaling,
        step=step,
        parameter_scales=parameter_scales,
    )
    if isinstance(reference, Unsupported):
        return reference
    labels = reference.parameters
    stacked = [reference.matrix]
    for point in at:
        other = sensitivity_matrix(
            observations,
            point,
            parameters=parameters,
            scaling=scaling,
            step=step,
            parameter_scales=parameter_scales,
        )
        if isinstance(other, Unsupported):
            return Unsupported(
                reason=f"an additional parameter point could not be differentiated: {other.reason}",
                detail=other.detail,
                missing=other.missing,
            )
        if other.parameters != labels:
            raise ValueError("every parameter point must give the same parameter coordinates")
        norm = float(np.linalg.norm(other.matrix))
        stacked.append(other.matrix / norm if norm > 0 else other.matrix)
    scale = float(np.linalg.norm(reference.matrix))
    stacked[0] = reference.matrix / scale if scale > 0 else reference.matrix

    singular: Array = (
        np.asarray(np.linalg.svd(reference.matrix, compute_uv=False), dtype=np.float64)
        if reference.matrix.size
        else np.zeros(0, dtype=np.float64)
    )
    largest = float(singular[0]) if singular.size else 0.0
    rank = int(np.sum(singular > tolerance * largest)) if largest > 0 else 0
    _, _, vt = np.linalg.svd(reference.matrix, full_matrices=True)
    null = vt[rank:].T if vt.size else np.zeros((len(labels), 0))
    row_space = vt[:rank].T if vt.size else np.zeros((len(labels), 0))

    together = np.vstack(stacked)
    joint_singular = np.linalg.svd(together, compute_uv=False) if together.size else np.zeros(0)
    joint_largest = float(joint_singular[0]) if joint_singular.size else 0.0
    joint_rank = int(np.sum(joint_singular > tolerance * joint_largest)) if joint_largest > 0 else 0
    structural = len(labels) - joint_rank

    condition = (
        float(largest / singular[rank - 1]) if rank > 0 and singular[rank - 1] > 0 else math.inf
    )
    limits: list[str] = []
    direction = direction_tolerance if direction_tolerance is not None else max(tolerance, 1e-6)
    symmetries, truncated_null = _search(
        singular,
        vt,
        labels,
        kind="symmetry",
        scaling=scaling,
        wanted=len(labels) - rank,
        tolerance=direction,
        max_support=max_support,
        max_coefficient=max_coefficient,
        limit=max_candidates,
    )
    estimable, truncated_row = _search(
        singular,
        vt,
        labels,
        kind="estimable",
        scaling=scaling,
        wanted=rank,
        tolerance=direction,
        max_support=max_support,
        max_coefficient=max_coefficient,
        limit=max_candidates,
    )
    if truncated_null or truncated_row:
        limits.append(f"integer search stopped at {max_candidates} candidates")
    if len(symmetries) < len(labels) - rank:
        limits.append(
            f"named {len(symmetries)} of {len(labels) - rank} flat directions; the rest are "
            "not small-integer monomials and are in null_basis"
        )
    return EstimabilityReport(
        parameters=labels,
        observations=reference.observations,
        scaling=scaling,
        n_points=1 + len(at),
        rank=rank,
        deficiency=len(labels) - rank,
        persistent_deficiency=min(structural, len(labels) - rank),
        singular_values=tuple(float(s) for s in singular),
        condition_number=max(condition, 1.0),
        symmetries=symmetries,
        estimable=estimable,
        null_basis=tuple(tuple(float(v) for v in null[:, j]) for j in range(null.shape[1])),
        zero_columns=reference.zero_columns,
        tolerance=float(tolerance),
        search_limits_hit=tuple(limits),
        detail={
            "row_space_dimension": str(int(row_space.shape[1])),
            "rows": str(reference.rows),
        },
    )


# -- practical identifiability from a simulated experiment ------------------------------


class ProfileReport(Spec):
    """Profile-likelihood intervals for named targets, from one simulated experiment.

    ``targets`` are parameter coordinates or rendered combinations;
    ``interval`` is the profile-likelihood interval at ``level`` with
    ``-inf`` / ``inf`` where the profile never rises by the threshold inside
    the search range — that is the signature of a practically
    non-identifiable target, and it is reported as an infinity rather than
    as the edge of the grid. ``recovered`` is the maximum-likelihood value.

    ``flat`` lists the targets whose interval is unbounded on at least one
    side: the list a design review should read first.
    """

    targets: tuple[NonEmptyStr, ...] = Field(min_length=1)
    truth: dict[str, float]
    recovered: dict[str, float]
    interval: dict[str, tuple[float, float]]
    level: float = Field(gt=0, lt=1)
    threshold: float = Field(gt=0)
    flat: tuple[str, ...] = ()
    n_observations: int = Field(ge=1)
    seed: int | None = None
    converged: bool = True
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _aligned(self) -> ProfileReport:
        if set(self.interval) != set(self.targets) or set(self.recovered) != set(self.targets):
            raise ValueError("one recovered value and one interval per target")
        return self

    @property
    def identified(self) -> bool:
        return not self.flat


def _free_names(model: ModelSpec) -> tuple[str, ...]:
    return tuple(p.name for p in free_parameters(model))


def _fit(
    model: ModelSpec,
    data: Mapping[str, Any],
    start: Mapping[str, Any],
    *,
    fixed: Mapping[str, float] | None = None,
) -> tuple[dict[str, Any], float, bool]:
    """Maximum likelihood in unconstrained coordinates, optionally holding some parameters.

    ``fixed`` names parameters to hold at a *constrained* value — that is
    what a profile does. They are transformed once and then simply left out
    of the optimizer's vector, so the likelihood being maximized is the same
    function either way.
    """
    held = {k: float(v) for k, v in (fixed or {}).items()}
    free = free_parameters(model)
    shapes = {p.name: p.shape for p in free}
    names = [p.name for p in free if p.name not in held]
    filled: dict[str, Any] = {}
    for p in free:
        if p.name in held:
            filled[p.name] = (
                np.full(shapes[p.name], held[p.name]) if shapes[p.name] else held[p.name]
            )
        else:
            filled[p.name] = np.asarray(start[p.name], dtype=float)
    unconstrained = unconstrain(model, filled)
    if not names:
        theta, _ = constrain(model, unconstrained)
        return theta, -float(log_likelihood(model, data, theta)), True
    sizes = {n: max(int(np.prod(shapes[n])), 1) for n in names}
    z0 = np.concatenate([np.atleast_1d(unconstrained[n]).ravel() for n in names])

    def unpack(z: Array) -> dict[str, Any]:
        out: dict[str, Any] = {n: unconstrained[n] for n in held}
        position = 0
        for name in names:
            chunk = z[position : position + sizes[name]]
            position += sizes[name]
            out[name] = chunk.reshape(shapes[name]) if shapes[name] else chunk[0]
        return out

    def objective(z: Array) -> float:
        theta, _ = constrain(model, unpack(z))
        total = log_likelihood(model, data, theta)
        return -float(total) if math.isfinite(total) else 1e12

    result = optimize.minimize(objective, z0, method="L-BFGS-B", options={"maxiter": 2000})
    theta, _ = constrain(model, unpack(np.asarray(result.x, dtype=float)))
    return theta, float(result.fun), bool(result.success)


def profile_likelihood(
    model: ModelSpec,
    data: Mapping[str, Any],
    start: Mapping[str, Any],
    target: str,
    *,
    grid: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """``(values, drops)``: the profile of one parameter, as the drop in log-likelihood.

    Each grid point holds ``target`` fixed and maximizes the likelihood over
    every other free parameter, warm-started from the neighbouring point so
    the profile is a curve rather than a scatter of local optima. ``drops``
    is ``l_max - l(value)``: zero at the optimum, rising away from it. A
    profile that stays flat is a parameter the data does not determine.
    """
    if target not in _free_names(model):
        raise KeyError(f"{target!r} is not a free parameter of {model.name!r}")
    values = [float(v) for v in grid]
    if not values:
        raise ValueError("the profile grid is empty")
    best_theta, best, _ = _fit(model, data, start)
    order = sorted(range(len(values)), key=lambda i: values[i])
    optimum = float(np.mean(np.asarray(best_theta[target], dtype=float)))
    centre = min(range(len(order)), key=lambda pos: abs(values[order[pos]] - optimum))
    drops = [0.0] * len(values)
    for span in (range(centre, len(order)), range(centre - 1, -1, -1)):
        warm: Mapping[str, Any] = best_theta
        for position in span:
            i = order[position]
            fitted, here, _ = _fit(model, data, warm, fixed={target: values[i]})
            drops[i] = float(here - best)
            warm = fitted
    return tuple(values), tuple(drops)


def _monomial(theta: Mapping[str, Any], exponents: Mapping[str, int]) -> float:
    """``prod_j theta_j ** w_j`` at scalar coordinates."""
    out = 1.0
    for name, power in exponents.items():
        value_j = float(np.mean(np.asarray(theta[name], dtype=float)))
        if value_j <= 0.0 and power != 0:
            raise ValueError(
                f"a monomial of the parameters needs them positive; {name} is {value_j:g}"
            )
        out *= value_j**power
    return out


def _fit_holding_combination(
    model: ModelSpec,
    data: Mapping[str, Any],
    start: Mapping[str, Any],
    combination: Combination,
    target: float,
) -> tuple[dict[str, Any], float, bool]:
    """Maximum likelihood with one monomial of the parameters held at ``target``.

    The constraint is imposed by elimination, not by a penalty: the
    parameter with the largest exponent is solved for from the others, so
    every point the optimizer visits satisfies the constraint exactly.
    """
    exponents = {k: v for k, v in combination.exponents.items() if v}
    pivot = max(exponents, key=lambda name: abs(exponents[name]))
    power = exponents[pivot]
    free = free_parameters(model)
    shapes = {p.name: p.shape for p in free}
    vectors = sorted(n for n in exponents if shapes[n])
    if vectors:
        raise ValueError(f"profiling a combination needs scalar parameters; {vectors} are vectors")
    names = [p.name for p in free if p.name != pivot]
    if pivot not in shapes:
        raise KeyError(f"{pivot!r} is not a free parameter of {model.name!r}")
    filled = {p.name: np.asarray(start[p.name], dtype=float) for p in free}
    unconstrained = unconstrain(model, filled)
    sizes = {n: max(int(np.prod(shapes[n])), 1) for n in names}
    z0 = np.concatenate([np.atleast_1d(unconstrained[n]).ravel() for n in names])

    def assemble(z: Array) -> dict[str, Any] | None:
        pieces: dict[str, Any] = {pivot: unconstrained[pivot]}
        position = 0
        for name in names:
            chunk = z[position : position + sizes[name]]
            position += sizes[name]
            pieces[name] = chunk.reshape(shapes[name]) if shapes[name] else chunk[0]
        theta, _ = constrain(model, pieces)
        rest = 1.0
        for name, w in exponents.items():
            if name == pivot:
                continue
            here = float(np.mean(np.asarray(theta[name], dtype=float)))
            if not math.isfinite(here) or here <= 0.0:
                return None  # the optimizer wandered where the monomial is undefined
            rest *= here**w
        if not math.isfinite(rest) or rest <= 0 or target <= 0:
            return None
        solved = (target / rest) ** (1.0 / power)
        if not math.isfinite(solved) or solved <= 0:
            return None
        theta[pivot] = np.asarray(solved, dtype=float)
        return theta

    def objective(z: Array) -> float:
        theta = assemble(z)
        if theta is None:
            return 1e12
        total = log_likelihood(model, data, theta)
        return -float(total) if math.isfinite(total) else 1e12

    result = optimize.minimize(objective, z0, method="L-BFGS-B", options={"maxiter": 2000})
    theta = assemble(np.asarray(result.x, dtype=float))
    if theta is None:  # pragma: no cover - the optimizer stayed in the feasible region
        raise RuntimeError(f"the constraint {combination.render()} = {target} could not be met")
    return theta, float(result.fun), bool(result.success)


def profile_combination(
    model: ModelSpec,
    data: Mapping[str, Any],
    start: Mapping[str, Any],
    combination: Combination,
    *,
    grid: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The profile of a *combination* of parameters — ``alpha / k`` rather than ``alpha``.

    This is what turns "``alpha`` is unbounded" into a usable statement.
    A design in a saturating response's linear regime leaves ``alpha`` and
    ``k`` each unbounded and ``alpha / k`` tightly determined; profiling the
    ratio is how you show that, and the ratio is what the report should
    quote.
    """
    if combination.scaling != "log":
        raise ValueError("only a log-scaled combination is a monomial that can be held fixed")
    values = [float(v) for v in grid]
    if not values:
        raise ValueError("the profile grid is empty")
    best_theta, best, _ = _fit(model, data, start)
    optimum = _monomial(best_theta, combination.exponents)
    order = sorted(range(len(values)), key=lambda i: values[i])
    centre = min(range(len(order)), key=lambda pos: abs(values[order[pos]] - optimum))
    drops = [0.0] * len(values)
    for span in (range(centre, len(order)), range(centre - 1, -1, -1)):
        warm: Mapping[str, Any] = best_theta
        for position in span:
            i = order[position]
            fitted, here, _ = _fit_holding_combination(model, data, warm, combination, values[i])
            drops[i] = float(here - best)
            warm = fitted
    return tuple(values), tuple(drops)


def simulated_identifiability(
    model: ModelSpec,
    data: Mapping[str, Any],
    truth: Mapping[str, Any],
    *,
    targets: Sequence[str] | None = None,
    combinations: Sequence[Combination] = (),
    seed: int | None = None,
    level: float = 0.95,
    range_factor: float = 20.0,
    n_grid: int = 21,
    noise_sd: float | None = None,
) -> ProfileReport | Unsupported:
    """Run the proposed experiment on simulated data and profile each target.

    The outcome is simulated at ``truth`` under the model's own likelihood,
    the model is refit, and each target's profile likelihood is walked
    outward by up to ``range_factor`` in each direction. A target whose
    profile never rises by the ``level`` threshold on a side has an interval
    that is unbounded on that side: identified in principle, not in practice,
    at this design and this noise.

    ``normal``, ``poisson``, ``gamma`` and ``lognormal`` outcomes are drawn
    from their own likelihood. ``binomial`` is ``Unsupported``: its outcome is
    a count against a trials column rather than a draw around the mean, so
    simulating one here would be inventing a column the caller did not give.
    ``student_t`` is likewise refused rather than approximated by a normal.
    """
    family = model.likelihood.family
    if family not in ("normal", "poisson", "gamma", "lognormal"):
        return Unsupported(
            reason=(
                f"simulating a {family} outcome is not implemented; "
                "supply simulated data and call profile_likelihood directly"
            ),
            detail={"family": family},
        )
    scale = noise_sd
    if scale is None and model.likelihood.scale:
        found = truth.get(model.likelihood.scale)
        scale = float(np.mean(np.asarray(found, dtype=float))) if found is not None else None
    if family != "poisson" and (scale is None or not scale > 0):
        return Unsupported(
            reason=(
                "simulation needs a positive noise scale: give noise_sd=, or a value for the "
                f"likelihood's scale parameter {model.likelihood.scale!r} in truth"
            ),
            missing=(model.likelihood.scale or "noise_sd",),
        )
    mean = np.atleast_1d(np.asarray(value(model.mean, data=data, params=truth), dtype=float))
    rng = np.random.default_rng(seed)
    simulated = dict(data)
    positive = ("poisson", "gamma", "lognormal")
    if family in positive and np.any(mean <= 0.0):
        return Unsupported(
            reason=(
                f"a {family} mean must be positive to simulate from; it is not at this "
                "design and truth"
            ),
            detail={"family": family},
        )
    match family:
        case "normal":
            assert scale is not None
            draw = mean + rng.normal(0.0, scale, size=mean.shape)
        case "poisson":
            draw = rng.poisson(mean).astype(float)
        case "lognormal":
            assert scale is not None
            draw = rng.lognormal(np.log(mean), scale, size=mean.shape)
        case _:  # gamma; ``scale`` is the coefficient of variation
            assert scale is not None
            shape = 1.0 / scale**2
            draw = rng.gamma(shape, mean / shape, size=mean.shape)
    simulated[model.outcome.name] = draw

    names = list(targets) if targets is not None else list(_free_names(model))
    unknown = [n for n in names if n not in _free_names(model)]
    if unknown:
        raise KeyError(f"{unknown} are not free parameters of {model.name!r}")
    labelled = [(c.render(), c) for c in combinations]
    clash = sorted({label for label, _ in labelled} & set(names))
    if clash:
        raise ValueError(f"{clash} name both a parameter and a combination")

    fitted, _, converged = _fit(model, simulated, truth)
    threshold = float(_st.chi2.ppf(level, df=1)) / 2.0
    recovered: dict[str, float] = {}
    interval: dict[str, tuple[float, float]] = {}
    flat: list[str] = []
    for name in names:
        centre = float(np.mean(np.asarray(fitted[name], dtype=float)))
        recovered[name] = centre
        if centre > 0:
            low = np.geomspace(centre / range_factor, centre, n_grid // 2 + 1)[:-1]
            high = np.geomspace(centre, centre * range_factor, n_grid // 2 + 1)[1:]
        else:
            span = max(abs(centre), 1.0) * range_factor
            low = np.linspace(centre - span, centre, n_grid // 2 + 1)[:-1]
            high = np.linspace(centre, centre + span, n_grid // 2 + 1)[1:]
        values, drops = profile_likelihood(
            model, simulated, fitted, name, grid=[*low, centre, *high]
        )
        lower, upper = _crossings(values, drops, threshold)
        interval[name] = (lower, upper)
        if not math.isfinite(lower) or not math.isfinite(upper):
            flat.append(name)
    truth_values = {n: float(np.mean(np.asarray(truth[n], dtype=float))) for n in names}
    for label, combination in labelled:
        centre = _monomial(fitted, combination.exponents)
        recovered[label] = centre
        truth_values[label] = _monomial(truth, combination.exponents)
        grid = [
            *np.geomspace(centre / range_factor, centre, n_grid // 2 + 1)[:-1],
            centre,
            *np.geomspace(centre, centre * range_factor, n_grid // 2 + 1)[1:],
        ]
        values, drops = profile_combination(model, simulated, fitted, combination, grid=grid)
        lower, upper = _crossings(values, drops, threshold)
        interval[label] = (lower, upper)
        if not math.isfinite(lower) or not math.isfinite(upper):
            flat.append(label)
    return ProfileReport(
        targets=(*names, *(label for label, _ in labelled)),
        truth=truth_values,
        recovered=recovered,
        interval=interval,
        level=level,
        threshold=threshold,
        flat=tuple(flat),
        n_observations=int(mean.size),
        seed=seed,
        converged=converged,
        detail={
            # A poisson outcome has no scale parameter: its dispersion is its mean.
            "noise_sd": f"{scale:.6g}" if scale is not None else "n/a (poisson)",
            "family": family,
            "range_factor": f"{range_factor:g}",
        },
    )


def _crossings(
    values: Sequence[float], drops: Sequence[float], threshold: float
) -> tuple[float, float]:
    """Where the profile first rises past ``threshold`` on each side of its minimum."""
    array = np.asarray(drops, dtype=float)
    grid = np.asarray(values, dtype=float)
    centre = int(np.argmin(array))
    lower = -math.inf
    for i in range(centre, 0, -1):
        if array[i - 1] >= threshold:
            lower = float(_interpolate(grid[i - 1], array[i - 1], grid[i], array[i], threshold))
            break
    upper = math.inf
    for i in range(centre, len(array) - 1):
        if array[i + 1] >= threshold:
            upper = float(_interpolate(grid[i], array[i], grid[i + 1], array[i + 1], threshold))
            break
    return lower, upper


def _interpolate(x0: float, y0: float, x1: float, y1: float, target: float) -> float:
    if y1 == y0:
        return x1
    return x0 + (target - y0) * (x1 - x0) / (y1 - y0)


# -- what to measure next ---------------------------------------------------------------


class Prescription(Spec):
    """The smallest set of extra measurements that restores rank, and what each one buys.

    ``added`` are the chosen candidates in the order they were chosen;
    ``rank_before`` and ``rank_after`` bracket what they achieved;
    ``broken`` names, per addition, the symmetries it removed. When no
    combination of the candidates reaches full rank, ``complete`` is false
    and ``still_flat`` says what is left — which is the useful answer, since
    it names what would have to be measured instead.
    """

    added: tuple[str, ...] = ()
    rank_before: int = Field(ge=0)
    rank_after: int = Field(ge=0)
    n_parameters: int = Field(ge=1)
    broken: tuple[tuple[str, ...], ...] = ()
    still_flat: tuple[str, ...] = ()
    considered: tuple[str, ...] = ()
    complete: bool = False
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _aligned(self) -> Prescription:
        if len(self.broken) != len(self.added):
            raise ValueError("one 'broken' entry per addition")
        if self.rank_after < self.rank_before:
            raise ValueError("adding an observation cannot lower the rank")
        if self.complete and self.rank_after != self.n_parameters:
            raise ValueError("a complete prescription reaches full rank")
        return self


def prescribe_measurements(
    observations: Sequence[Observation],
    candidates: Sequence[Observation],
    theta: Theta,
    *,
    parameters: Sequence[str] | None = None,
    scaling: Scaling = "log",
    tolerance: float = _RANK_TOL,
    max_additions: int = 4,
    step: float = _REL_STEP,
    parameter_scales: Mapping[str, float] | None = None,
) -> Prescription | Unsupported:
    """Greedily choose candidates until the design has full rank, or say what is left.

    At each round every remaining candidate is added on its own and the one
    that raises the rank most — breaking ties by the largest gain in the
    smallest singular value — is kept. Greedy is not optimal, and the
    ``considered`` list is reported so a reader can see what was on the
    table.
    """
    base = estimable_combinations(
        observations,
        theta,
        parameters=parameters,
        scaling=scaling,
        tolerance=tolerance,
        step=step,
        parameter_scales=parameter_scales,
    )
    if isinstance(base, Unsupported):
        return base
    n_parameters = len(base.parameters)
    current = list(observations)
    remaining = list(candidates)
    added: list[str] = []
    broken: list[tuple[str, ...]] = []
    rank = base.rank
    flat = {c.render() for c in base.symmetries}
    while rank < n_parameters and remaining and len(added) < max_additions:
        best_position = -1
        # Only a strict rank increase counts: this function promises the smallest set
        # that *restores rank*, not the set that tightens the estimates most.
        best_score = (rank, math.inf)
        best_report: EstimabilityReport | None = None
        for position, candidate in enumerate(remaining):
            trial = estimable_combinations(
                [*current, candidate],
                theta,
                parameters=parameters,
                scaling=scaling,
                tolerance=tolerance,
                step=step,
                parameter_scales=parameter_scales,
            )
            if isinstance(trial, Unsupported):
                continue
            # Rank first, then the smallest singular value it leaves: two candidates
            # that both restore the rank are separated by how much room they leave.
            smallest = trial.singular_values[trial.rank - 1] if trial.rank else 0.0
            score = (trial.rank, smallest)
            if score > best_score:
                best_position, best_score, best_report = position, score, trial
        if best_report is None:
            break
        chosen = remaining.pop(best_position)
        current.append(chosen)
        added.append(chosen.label)
        now_flat = {c.render() for c in best_report.symmetries}
        broken.append(tuple(sorted(flat - now_flat)))
        flat = now_flat
        rank = best_report.rank
    return Prescription(
        added=tuple(added),
        rank_before=base.rank,
        rank_after=rank,
        n_parameters=n_parameters,
        broken=tuple(broken),
        still_flat=tuple(sorted(flat)),
        considered=tuple(c.label for c in candidates),
        complete=rank == n_parameters,
        detail={"max_additions": str(max_additions), "scaling": scaling},
    )
