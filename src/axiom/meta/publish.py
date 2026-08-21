"""Differentially private release of a cell's clipped mean, gated by ``privacy``.

New in axiom (roadmap M7.6; no parent module). A release:

1. refuses the cell unless ``check_cell`` licenses it (k-anonymity, dominance);
2. charges ``epsilon`` against the ``EpsilonLedger`` (``Blocked`` if that would
   overspend; the ledger is returned unchanged in that case, i.e. not at all);
3. clips every value to ``[lo, hi]`` and takes the mean — a statistic whose
   ``L1`` sensitivity to one record is ``(hi − lo) / n``;
4. adds noise by mechanism:

   * **Laplace**: scale ``b = sensitivity / epsilon`` (Dwork et al. 2006), pure
     ε-DP;
   * **Gaussian**: the *analytic* Gaussian mechanism (Balle & Wang 2018) — the
     smallest ``σ`` with ``Φ(Δ/(2σ) − εσ/Δ) − e^ε · Φ(−Δ/(2σ) − εσ/Δ) ≤ δ``,
     which is (ε, δ)-DP for every ε > 0 and is tighter than the classical
     ``σ = Δ · sqrt(2 ln(1.25/δ)) / ε`` (only valid for ε < 1).

The seed, clip bounds, sensitivity, noise scale and mechanism are all recorded
on the ``Release``; the noise is drawn from ``numpy.random.default_rng(seed)``
so a release is reproducible. ``orthogonal_split`` divides an ε equally among
parts (sequential composition: the parts sum to ε). ``carry_forward`` re-uses a
previous release for free when the cell's contributor set has changed by less
than a Jaccard-distance threshold, and otherwise requires a new (charged)
release.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import model_validator
from scipy.optimize import brentq
from scipy.special import ndtr

from axiom.core import Blocked, Interval, NonEmptyStr, Spec
from axiom.meta.privacy import Cell, EpsilonLedger, Mechanism, PrivacyPolicy, charge, check_cell

__all__ = [
    "EpsilonSplit",
    "Release",
    "carry_forward",
    "gaussian_sigma",
    "gaussian_sigma_classical",
    "jaccard_distance",
    "laplace_scale",
    "orthogonal_split",
    "release",
]


# -- noise calibration --------------------------------------------------------------------


def laplace_scale(sensitivity: float, epsilon: float) -> float:
    """Laplace mechanism scale ``b = sensitivity / epsilon``."""
    if not (math.isfinite(sensitivity) and sensitivity >= 0):
        raise ValueError("sensitivity must be finite and non-negative")
    if not (math.isfinite(epsilon) and epsilon > 0):
        raise ValueError("epsilon must be finite and positive")
    return sensitivity / epsilon


def gaussian_sigma_classical(sensitivity: float, epsilon: float, delta: float) -> float:
    """Classical ``σ = Δ · sqrt(2 ln(1.25/δ)) / ε`` (Dwork & Roth 2014, Thm A.1; needs ε < 1)."""
    if not 0.0 < epsilon < 1.0:
        raise ValueError("the classical Gaussian calibration is only valid for 0 < epsilon < 1")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def _analytic_delta(sigma: float, sensitivity: float, epsilon: float) -> float:
    a = sensitivity / (2.0 * sigma)
    b = epsilon * sigma / sensitivity
    return float(ndtr(a - b) - math.exp(epsilon) * ndtr(-a - b))


def gaussian_sigma(
    sensitivity: float, epsilon: float, delta: float, *, tol: float = 1e-12
) -> float:
    """Analytic Gaussian mechanism (Balle & Wang 2018): the smallest ``σ`` with

    ``Φ(Δ/(2σ) − εσ/Δ) − e^ε · Φ(−Δ/(2σ) − εσ/Δ) ≤ δ``,

    found by bracketing root finding to relative tolerance ``tol``. Returns
    ``0`` when the sensitivity is zero.
    """
    if not (math.isfinite(sensitivity) and sensitivity >= 0):
        raise ValueError("sensitivity must be finite and non-negative")
    if not (math.isfinite(epsilon) and epsilon > 0):
        raise ValueError("epsilon must be finite and positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if sensitivity == 0.0:
        return 0.0
    lo, hi = 1e-12 * sensitivity, sensitivity / epsilon
    # delta(sigma) decreases in sigma; widen the upper bracket until it is below delta.
    while _analytic_delta(hi, sensitivity, epsilon) > delta:
        hi *= 2.0
        if hi > 1e12 * sensitivity:
            raise ValueError("failed to bracket the analytic Gaussian sigma")
    root = brentq(
        lambda s: _analytic_delta(s, sensitivity, epsilon) - delta, lo, hi, rtol=tol, xtol=1e-300
    )
    return float(root)


# -- release --------------------------------------------------------------------------------


class Release(Spec):
    """A published clipped mean with every input that made it reproducible and auditable.

    ``value`` is the noised statistic; ``interval`` is ``value ± q_mass`` where
    ``q_mass`` is the ``mass`` quantile of ``|noise|`` (labelled ``wald``:
    point ± critical value · scale). ``contributors`` is the sorted contributor
    set, kept for churn accounting by ``carry_forward``; it is part of the
    audit record, not of the published number.
    """

    release_id: NonEmptyStr
    cell_name: str = ""
    statistic: Literal["clipped_mean"] = "clipped_mean"
    value: float
    interval: Interval
    mass: float
    n: int
    clip_lower: float
    clip_upper: float
    n_clipped: int
    sensitivity: float
    epsilon: float
    delta: float
    mechanism: Mechanism
    noise_scale: float
    seed: int
    contributors: tuple[str, ...]
    carried_from: str = ""
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _consistent(self) -> Release:
        if self.clip_lower >= self.clip_upper:
            raise ValueError("clip_lower must be below clip_upper")
        if self.n < 1 or not 0 <= self.n_clipped <= self.n:
            raise ValueError("n must be positive and n_clipped within [0, n]")
        if not (math.isfinite(self.epsilon) and self.epsilon > 0):
            raise ValueError("epsilon must be finite and positive")
        if not 0.0 < self.mass < 1.0:
            raise ValueError("mass must be in (0, 1)")
        if self.mechanism == "laplace" and self.delta != 0.0:
            raise ValueError("the Laplace mechanism is pure epsilon-DP; delta must be 0")
        if self.mechanism == "gaussian" and not 0.0 < self.delta < 1.0:
            raise ValueError("the Gaussian mechanism needs delta in (0, 1)")
        return self


def _noise_quantile(mechanism: Mechanism, scale: float, mass: float) -> float:
    if scale == 0.0:
        return 0.0
    if mechanism == "laplace":
        return -scale * math.log(1.0 - mass)
    from scipy.special import ndtri

    return scale * float(ndtri((1.0 + mass) / 2.0))


def release(
    cell: Cell,
    policy: PrivacyPolicy,
    ledger: EpsilonLedger,
    *,
    release_id: str,
    epsilon: float,
    clip: tuple[float, float],
    seed: int,
    mechanism: Mechanism = "laplace",
    delta: float = 1e-6,
    mass: float = 0.95,
) -> tuple[Release, EpsilonLedger] | Blocked:
    """Publish the cell's clipped mean under ``policy`` and ``ledger``, or refuse.

    Returns ``Blocked`` (and charges nothing) when the cell fails the policy
    or the ledger cannot cover ``epsilon``.
    """
    lo, hi = float(clip[0]), float(clip[1])
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError(f"clip must be finite with lower < upper, got {clip}")
    if not 0.0 < mass < 1.0:
        raise ValueError("mass must be in (0, 1)")
    verdict = check_cell(cell, policy)
    if not verdict.licensed:
        return Blocked(
            reason=f"cell refused by privacy policy: {verdict.reason}",
            detail={"rule": verdict.route, "release_id": release_id},
        )
    used_delta = delta if mechanism == "gaussian" else 0.0
    charged = charge(
        ledger, release_id=release_id, epsilon=epsilon, mechanism=mechanism, delta=used_delta
    )
    if isinstance(charged, Blocked):
        return charged
    values = np.asarray(cell.values, dtype=np.float64)
    clipped = np.clip(values, lo, hi)
    n = int(values.size)
    n_clipped = int(np.count_nonzero(clipped != values))
    sensitivity = (hi - lo) / n
    rng = np.random.default_rng(seed)
    if mechanism == "laplace":
        scale = laplace_scale(sensitivity, epsilon)
        noise = float(rng.laplace(0.0, scale)) if scale > 0 else 0.0
    elif mechanism == "gaussian":
        scale = gaussian_sigma(sensitivity, epsilon, delta)
        noise = float(rng.normal(0.0, scale)) if scale > 0 else 0.0
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    value = float(clipped.mean()) + noise
    q = _noise_quantile(mechanism, scale, mass)
    rel = Release(
        release_id=release_id,
        cell_name=cell.name,
        value=value,
        interval=Interval(lower=value - q, upper=value + q, definition="wald", mass=mass),
        mass=mass,
        n=n,
        clip_lower=lo,
        clip_upper=hi,
        n_clipped=n_clipped,
        sensitivity=sensitivity,
        epsilon=epsilon,
        delta=used_delta,
        mechanism=mechanism,
        noise_scale=scale,
        seed=seed,
        contributors=tuple(sorted(cell.contributors)),
        detail={
            "sensitivity": "(clip_upper - clip_lower) / n",
            "noise_scale": (
                "sensitivity / epsilon"
                if mechanism == "laplace"
                else "analytic Gaussian sigma (Balle & Wang 2018)"
            ),
            "composition": "sequential; charged against the ledger",
        },
    )
    return rel, charged


# -- composition ------------------------------------------------------------------------------


class EpsilonSplit(Spec):
    """An ε divided equally among ``parts`` releases under sequential composition."""

    epsilon: float
    parts: int
    epsilons: tuple[float, ...]
    composition: str = "sequential: the parts sum to epsilon"

    @model_validator(mode="after")
    def _sums(self) -> EpsilonSplit:
        if self.parts < 1 or len(self.epsilons) != self.parts:
            raise ValueError("parts must be positive and match len(epsilons)")
        if not math.isclose(math.fsum(self.epsilons), self.epsilon, rel_tol=1e-12):
            raise ValueError("the parts must sum to epsilon")
        return self


def orthogonal_split(epsilon: float, parts: int) -> EpsilonSplit:
    """Equal split ``epsilon / parts`` per release; sequential composition sums to ``epsilon``."""
    if not (math.isfinite(epsilon) and epsilon > 0):
        raise ValueError("epsilon must be finite and positive")
    if parts < 1:
        raise ValueError("parts must be at least 1")
    each = epsilon / parts
    eps = [each] * parts
    # Absorb floating-point residue into the last part so the sum is exact.
    eps[-1] = epsilon - math.fsum(eps[:-1])
    return EpsilonSplit(epsilon=epsilon, parts=parts, epsilons=tuple(eps))


# -- carry-forward ----------------------------------------------------------------------------


def jaccard_distance(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    """``1 − |a ∩ b| / |a ∪ b|``; zero for two empty sets."""
    union = set(a) | set(b)
    if not union:
        return 0.0
    return 1.0 - len(set(a) & set(b)) / len(union)


def carry_forward(
    previous: Release,
    cell: Cell,
    policy: PrivacyPolicy,
    *,
    churn_threshold: float,
) -> Release | Blocked:
    """Re-use ``previous`` at no ε cost when contributor churn is below ``churn_threshold``.

    Churn is the Jaccard distance between the previous and current
    contributor sets. The current cell must still clear the policy; above the
    threshold the result is ``Blocked`` naming the churn, and a fresh
    ``release`` (charged) is required.
    """
    if not 0.0 <= churn_threshold <= 1.0:
        raise ValueError("churn_threshold must be in [0, 1]")
    verdict = check_cell(cell, policy)
    if not verdict.licensed:
        return Blocked(
            reason=f"cell refused by privacy policy: {verdict.reason}",
            detail={"rule": verdict.route},
        )
    churn = jaccard_distance(frozenset(previous.contributors), cell.contributors)
    if churn > churn_threshold:
        return Blocked(
            reason=(
                f"contributor churn {churn:.3f} exceeds threshold {churn_threshold}; "
                "a new release is required"
            ),
            detail={"churn": repr(churn), "threshold": repr(churn_threshold)},
        )
    return previous.model_copy(
        update={
            "carried_from": previous.release_id,
            "detail": {**previous.detail, "carry_forward_churn": repr(churn)},
        }
    )
