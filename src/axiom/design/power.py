"""Closed-form power, minimum detectable effect, and sample size under a normal approximation.

Two settings share one piece of arithmetic. A *difference in means* between a
treated and a control arm with ``n`` units in total, a common outcome standard
deviation ``sd`` and a treated share ``allocation`` has

    se = sd / sqrt(n · allocation · (1 − allocation)).

A *regression coefficient* has whatever design standard error the design gives
it; the ``coefficient_*`` functions take that ``se`` directly and, for sample
size, assume it shrinks as ``n^(-1/2)`` from a reference ``n``.

Given ``se``, the power to detect an effect ``delta`` at level ``alpha`` is
exact under the normal model:

    two-sided:  Φ(|δ|/se − z_{1−α/2}) + Φ(−|δ|/se − z_{1−α/2})
    one-sided:  Φ(|δ|/se − z_{1−α})

``mde`` and ``sample_size`` are exact inversions of that formula, not the
textbook ``(z_{1−α/2} + z_{power}) · se`` approximation, so that
``power(mde(...)) == power`` holds to solver tolerance. The textbook value is
the upper end of the root bracket and differs only in the far tail. Every
result is a ``Spec`` that carries ``alpha``, ``power`` and ``two_sided``: the
numbers never travel without the convention that produced them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import model_validator
from scipy import optimize, stats

from axiom.core import Spec, Unsupported

__all__ = [
    "MDE",
    "PowerCurve",
    "PowerResult",
    "SampleSize",
    "coefficient_mde",
    "coefficient_power",
    "coefficient_sample_size",
    "difference_se",
    "mde",
    "power",
    "power_curve",
    "power_from_se",
    "proportion_difference_se",
    "proportion_mde",
    "proportion_power",
    "proportion_sample_size",
    "sample_size",
]

Array = npt.NDArray[np.float64]
PowerDesign = Literal["difference_in_means", "difference_in_proportions", "coefficient"]
_N_MAX = 10**8


# -- specs -----------------------------------------------------------------------------


def _check_alpha_power(alpha: float, target_power: float | None) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if target_power is not None and not (alpha < target_power < 1.0):
        raise ValueError(f"power must be in (alpha, 1), got {target_power} with alpha={alpha}")


def _check_allocation(allocation: float) -> None:
    if not 0.0 < allocation < 1.0:
        raise ValueError(f"allocation (treated share) must be in (0, 1), got {allocation}")


def _check_sd(sd: float) -> None:
    if not (math.isfinite(sd) and sd > 0):
        raise ValueError(f"sd must be finite and positive, got {sd}")


class PowerResult(Spec):
    """Power to detect ``effect`` at ``alpha`` with standard error ``se``.

    ``n`` and ``allocation`` are recorded when the standard error came from a
    difference-in-means design and are ``None`` for a supplied coefficient SE.
    """

    design: PowerDesign
    power: float
    effect: float
    se: float
    alpha: float
    two_sided: bool
    n: int | None = None
    sd: float | None = None
    allocation: float | None = None

    @model_validator(mode="after")
    def _valid(self) -> PowerResult:
        _check_alpha_power(self.alpha, None)
        if not 0.0 <= self.power <= 1.0:
            raise ValueError(f"power must be in [0, 1], got {self.power}")
        if self.se < 0:
            raise ValueError("se must be non-negative")
        return self


class MDE(Spec):
    """The smallest effect detected with probability ``power`` at level ``alpha``."""

    design: PowerDesign
    effect: float
    se: float
    alpha: float
    power: float
    two_sided: bool
    n: int | None = None
    sd: float | None = None
    allocation: float | None = None

    @model_validator(mode="after")
    def _valid(self) -> MDE:
        _check_alpha_power(self.alpha, self.power)
        if self.effect < 0 or self.se < 0:
            raise ValueError("effect and se must be non-negative")
        return self


class SampleSize(Spec):
    """The smallest ``n`` whose achieved power reaches the target for ``effect``.

    ``n_treated`` / ``n_control`` are the integer arms at ``allocation`` and
    ``power`` is the power achieved with those integer arms (never below the
    target). For the coefficient design ``n`` is total observations and the
    arms are ``None``.
    """

    design: PowerDesign
    n: int
    effect: float
    se: float
    alpha: float
    power: float
    target_power: float
    two_sided: bool
    sd: float | None = None
    allocation: float | None = None
    n_treated: int | None = None
    n_control: int | None = None

    @model_validator(mode="after")
    def _valid(self) -> SampleSize:
        _check_alpha_power(self.alpha, self.target_power)
        if self.n < 1:
            raise ValueError("n must be positive")
        if self.power + 1e-12 < self.target_power:
            raise ValueError(f"achieved power {self.power} is below the target {self.target_power}")
        return self


class PowerCurve(Spec):
    """Power as a function of effect size at a fixed design (``n``, ``sd``, ``allocation``)."""

    design: PowerDesign
    effects: tuple[float, ...]
    powers: tuple[float, ...]
    se: float
    alpha: float
    two_sided: bool
    n: int | None = None
    sd: float | None = None
    allocation: float | None = None

    @model_validator(mode="after")
    def _valid(self) -> PowerCurve:
        if len(self.effects) != len(self.powers) or not self.effects:
            raise ValueError("effects and powers must be non-empty and equal in length")
        _check_alpha_power(self.alpha, None)
        return self

    def power_at(self, effect: float) -> float:
        """Linear interpolation of the curve at ``effect`` (clamped to the grid)."""
        return float(np.interp(effect, self.effects, self.powers))


# -- arithmetic ------------------------------------------------------------------------


def difference_se(n: int, sd: float, allocation: float = 0.5) -> float:
    """``sd / sqrt(n · p · (1 − p))`` — the SE of a two-arm difference in means.

    One standard deviation, shared by both arms. If the outcome is binary that
    assumption is false — a proportion's variance is a function of its own mean,
    so the arms differ unless the effect is zero — and no choice of ``sd``
    repairs it. Use :func:`proportion_difference_se` instead.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    _check_sd(sd)
    _check_allocation(allocation)
    return sd / math.sqrt(n * allocation * (1.0 - allocation))


def proportion_difference_se(
    p_control: float, p_treated: float, n: int, allocation: float = 0.5
) -> float:
    """``sqrt(p_t(1-p_t)/n_t + p_c(1-p_c)/n_c)`` — the SE of a difference in proportions.

    Not reachable through :func:`difference_se`. That one is
    ``sd / sqrt(n · a · (1 − a))``: one outcome standard deviation shared by
    both arms, which is what a homoscedastic Gaussian gives. A binary
    outcome's variance is a *function of its own mean*, so the two arms have
    different variances unless ``p_control == p_treated`` — and if they were
    equal there would be no effect to power for. Passing a single ``sd`` here
    is therefore wrong precisely in proportion to the effect being looked
    for, which is the wrong direction for the error to run.

    Feed the result to :func:`power_from_se`, or to the ``coefficient_*``
    functions, like any other design standard error.
    """
    for name, p in (("p_control", p_control), ("p_treated", p_treated)):
        if not 0.0 <= p <= 1.0 or not math.isfinite(p):
            raise ValueError(f"{name} must be a probability in [0, 1], got {p}")
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    _check_allocation(allocation)
    n_t, n_c = _arms(n, allocation)
    var = p_treated * (1.0 - p_treated) / n_t + p_control * (1.0 - p_control) / n_c
    if var <= 0.0:
        raise ValueError(
            "both proportions are at 0 or 1, so the difference has no sampling variance "
            "and no power calculation is meaningful"
        )
    return math.sqrt(var)


def _proportion_se(p_control: float, p_treated: float, n_t: int, n_c: int) -> float:
    return math.sqrt(p_treated * (1.0 - p_treated) / n_t + p_control * (1.0 - p_control) / n_c)


def proportion_power(
    p_control: float,
    p_treated: float,
    n: int,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
    allocation: float = 0.5,
) -> PowerResult:
    """Power to detect ``p_treated - p_control`` with ``n`` units split at ``allocation``."""
    se = proportion_difference_se(p_control, p_treated, n, allocation)
    return PowerResult(
        design="difference_in_proportions",
        power=_power_value(p_treated - p_control, se, alpha, two_sided),
        effect=p_treated - p_control,
        se=se,
        alpha=alpha,
        two_sided=two_sided,
        n=n,
        allocation=allocation,
    )


def proportion_mde(
    p_control: float,
    n: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
    allocation: float = 0.5,
    direction: Literal["increase", "decrease"] = "increase",
) -> MDE | Unsupported:
    """Smallest detectable change in a proportion from ``p_control``, at ``n`` units.

    Not :func:`mde` with a binomial ``sd``, and the difference is not
    cosmetic. :func:`mde` inverts a *fixed* standard error, which is correct
    when the outcome's variance does not depend on its mean. A proportion's
    does, so the treated arm's variance moves as the effect grows and the
    standard error has to be solved for jointly with it. Freezing it at the
    null is optimistic wherever ``p_control < 0.5`` — at ``p_control = 0.1``
    and ``n = 400`` it reports a detectable lift of 8.4 points that actually
    carries 68% power, not the 80% asked for. This function root-finds on the
    real thing.

    ``direction`` picks the sign: a rise and a fall of the same size are not
    equally detectable, because they land on different variances.
    ``Unsupported`` when no effect within ``[0, 1]`` reaches ``power``.
    """
    if not 0.0 <= p_control <= 1.0 or not math.isfinite(p_control):
        raise ValueError(f"p_control must be a probability in [0, 1], got {p_control}")
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    _check_allocation(allocation)
    _check_alpha_power(alpha, power)
    sign = 1.0 if direction == "increase" else -1.0
    n_t, n_c = _arms(n, allocation)
    reach = (1.0 - p_control) if direction == "increase" else p_control

    def achieved(delta: float) -> float:
        p_t = min(max(p_control + sign * delta, 0.0), 1.0)
        se = _proportion_se(p_control, p_t, n_t, n_c)
        return _power_value(delta, se, alpha, two_sided)

    if reach <= 0.0 or achieved(reach) < power:
        return Unsupported(
            reason=(
                f"no {direction} in a proportion from {p_control:g} reaches power {power:g} "
                f"at n = {n}: even the largest possible effect ({reach:g}) achieves "
                f"{achieved(reach) if reach > 0 else 0.0:.4g}"
            ),
            detail={"p_control": f"{p_control:g}", "n": str(n), "direction": direction},
        )
    root = optimize.brentq(
        lambda d: achieved(d) - power, 1e-12, reach, xtol=1e-14, rtol=1e-12, maxiter=200
    )
    effect = float(root)
    p_t = min(max(p_control + sign * effect, 0.0), 1.0)
    return MDE(
        design="difference_in_proportions",
        effect=effect,
        se=_proportion_se(p_control, p_t, n_t, n_c),
        alpha=alpha,
        power=power,
        two_sided=two_sided,
        n=n,
        allocation=allocation,
    )


def proportion_sample_size(
    p_control: float,
    p_treated: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
    allocation: float = 0.5,
) -> SampleSize | Unsupported:
    """Smallest total ``n`` (integer arms) reaching ``power`` for a known pair of proportions.

    Both proportions are given, so unlike :func:`proportion_mde` the variance
    is fixed and the standard error falls exactly as ``n**-0.5``; the
    continuous solution bounds the integer search, as in :func:`sample_size`.
    """
    for name, p in (("p_control", p_control), ("p_treated", p_treated)):
        if not 0.0 <= p <= 1.0 or not math.isfinite(p):
            raise ValueError(f"{name} must be a probability in [0, 1], got {p}")
    _check_allocation(allocation)
    _check_alpha_power(alpha, power)
    effect = p_treated - p_control
    if effect == 0.0:
        return Unsupported(
            reason="sample size for a zero difference in proportions is unbounded",
            detail={"p_control": f"{p_control:g}", "p_treated": f"{p_treated:g}"},
        )
    var_unit = p_treated * (1.0 - p_treated) / allocation + p_control * (1.0 - p_control) / (
        1.0 - allocation
    )
    if var_unit <= 0.0:
        return Unsupported(
            reason=(
                "both proportions are at 0 or 1, so the difference has no sampling variance "
                "and no sample size is meaningful"
            ),
            detail={"p_control": f"{p_control:g}", "p_treated": f"{p_treated:g}"},
        )
    z_a = float(stats.norm.ppf(1.0 - (alpha / 2.0 if two_sided else alpha)))
    z_b = float(stats.norm.ppf(power))
    n_continuous = var_unit * ((z_a + z_b) / effect) ** 2
    if n_continuous > _N_MAX:
        return Unsupported(
            reason=f"sample size exceeds {_N_MAX}: the difference in proportions is negligible",
            detail={"n_continuous": f"{n_continuous:.4g}"},
        )
    lo = 2
    hi = max(int(math.ceil(n_continuous)) + 2, lo + 1)
    while lo <= _N_MAX:
        for n in range(lo, hi + 1):
            n_t, n_c = _arms(n, allocation)
            se = _proportion_se(p_control, p_treated, n_t, n_c)
            achieved = _power_value(effect, se, alpha, two_sided)
            if achieved >= power:
                return SampleSize(
                    design="difference_in_proportions",
                    n=n,
                    effect=effect,
                    se=se,
                    alpha=alpha,
                    power=achieved,
                    target_power=power,
                    two_sided=two_sided,
                    allocation=allocation,
                    n_treated=n_t,
                    n_control=n_c,
                )
        lo, hi = hi + 1, min(hi * 2, _N_MAX)
    return Unsupported(  # pragma: no cover - guarded by the n_continuous bound above
        reason=f"sample size exceeds {_N_MAX}",
        detail={"n_continuous": f"{n_continuous:.4g}"},
    )


def _arms(n: int, allocation: float) -> tuple[int, int]:
    n_t = int(round(n * allocation))
    n_t = min(max(n_t, 1), n - 1)
    return n_t, n - n_t


def _integer_arm_se(n: int, sd: float, allocation: float) -> float:
    n_t, n_c = _arms(n, allocation)
    return sd * math.sqrt(1.0 / n_t + 1.0 / n_c)


def _power_value(effect: float, se: float, alpha: float, two_sided: bool) -> float:
    if se == 0.0:
        return 1.0 if effect != 0.0 else float(alpha)
    ratio = abs(effect) / se
    if two_sided:
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        return float(stats.norm.cdf(ratio - z) + stats.norm.cdf(-ratio - z))
    z = float(stats.norm.ppf(1.0 - alpha))
    return float(stats.norm.cdf(ratio - z))


def _power_array(effects: Array, se: float, alpha: float, two_sided: bool) -> Array:
    ratio = np.abs(effects) / se
    if two_sided:
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        return np.asarray(stats.norm.cdf(ratio - z) + stats.norm.cdf(-ratio - z), dtype=np.float64)
    z = float(stats.norm.ppf(1.0 - alpha))
    return np.asarray(stats.norm.cdf(ratio - z), dtype=np.float64)


def _mde_value(se: float, alpha: float, target_power: float, two_sided: bool) -> float:
    """Exact inversion of ``_power_value`` in ``effect`` for a fixed ``se``."""
    if se == 0.0:
        return 0.0
    z_a = float(stats.norm.ppf(1.0 - (alpha / 2.0 if two_sided else alpha)))
    z_b = float(stats.norm.ppf(target_power))
    textbook = (z_a + z_b) * se
    if not two_sided:
        return textbook
    # The other tail adds a positive term, so the exact root lies in [0, textbook].
    root = optimize.brentq(
        lambda d: _power_value(d, se, alpha, True) - target_power,
        0.0,
        textbook * (1.0 + 1e-9),
        xtol=1e-14 * max(se, 1.0),
        rtol=1e-12,
        maxiter=200,
    )
    return float(root)


# -- difference in means ---------------------------------------------------------------


def power_from_se(
    effect: float, se: float, *, alpha: float = 0.05, two_sided: bool = True
) -> PowerResult:
    """Power for an effect whose estimator has standard error ``se``."""
    _check_alpha_power(alpha, None)
    if not (math.isfinite(se) and se >= 0):
        raise ValueError(f"se must be finite and non-negative, got {se}")
    return PowerResult(
        design="coefficient",
        power=_power_value(effect, se, alpha, two_sided),
        effect=effect,
        se=se,
        alpha=alpha,
        two_sided=two_sided,
    )


def power(
    n: int,
    effect: float,
    sd: float,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
    allocation: float = 0.5,
) -> PowerResult:
    """Power of a two-arm difference in means with ``n`` units in total."""
    se = difference_se(n, sd, allocation)
    _check_alpha_power(alpha, None)
    return PowerResult(
        design="difference_in_means",
        power=_power_value(effect, se, alpha, two_sided),
        effect=effect,
        se=se,
        alpha=alpha,
        two_sided=two_sided,
        n=n,
        sd=sd,
        allocation=allocation,
    )


def mde(
    n: int,
    sd: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
    allocation: float = 0.5,
) -> MDE:
    """Minimum detectable effect of a two-arm difference in means with ``n`` units in total.

    Inverts a *fixed* standard error, which is what a constant ``sd`` means.
    For a binary outcome the standard error moves with the effect and this is
    optimistic below ``p = 0.5``; use :func:`proportion_mde`.
    """
    se = difference_se(n, sd, allocation)
    _check_alpha_power(alpha, power)
    return MDE(
        design="difference_in_means",
        effect=_mde_value(se, alpha, power, two_sided),
        se=se,
        alpha=alpha,
        power=power,
        two_sided=two_sided,
        n=n,
        sd=sd,
        allocation=allocation,
    )


def sample_size(
    effect: float,
    sd: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
    allocation: float = 0.5,
) -> SampleSize | Unsupported:
    """Smallest total ``n`` (integer arms at ``allocation``) reaching ``power`` for ``effect``.

    For a binary outcome use :func:`proportion_sample_size`, which takes the two
    proportions rather than one pooled ``sd``.

    Searches ``n`` upward from 2 in vectorized blocks; the continuous textbook
    ``n`` bounds the first block. Returns ``Unsupported`` when no ``n`` below
    ``10**8`` suffices (an effect that is zero or negligible against ``sd``).
    """
    _check_sd(sd)
    _check_allocation(allocation)
    _check_alpha_power(alpha, power)
    if not math.isfinite(effect) or effect == 0.0:
        return Unsupported(
            reason="sample size for a zero effect is unbounded",
            detail={"effect": repr(effect)},
        )
    z_a = float(stats.norm.ppf(1.0 - (alpha / 2.0 if two_sided else alpha)))
    z_b = float(stats.norm.ppf(power))
    n_continuous = (sd * (z_a + z_b) / effect) ** 2 / (allocation * (1.0 - allocation))
    if n_continuous > _N_MAX:
        return Unsupported(
            reason=f"sample size exceeds {_N_MAX}: effect is negligible against sd",
            detail={"n_continuous": f"{n_continuous:.4g}"},
        )
    lo = 2
    hi = max(int(math.ceil(n_continuous)) + 2, lo + 1)
    while lo <= _N_MAX:
        ns = np.arange(lo, hi + 1)
        n_t = np.clip(np.rint(ns * allocation), 1, ns - 1)
        n_c = ns - n_t
        ses = sd * np.sqrt(1.0 / n_t + 1.0 / n_c)
        achieved = np.asarray(
            [_power_value(effect, float(s), alpha, two_sided) for s in ses], dtype=np.float64
        )
        hits = np.flatnonzero(achieved >= power)
        if hits.size:
            n = int(ns[hits[0]])
            nt, nc = _arms(n, allocation)
            return SampleSize(
                design="difference_in_means",
                n=n,
                effect=effect,
                se=_integer_arm_se(n, sd, allocation),
                alpha=alpha,
                power=float(achieved[hits[0]]),
                target_power=power,
                two_sided=two_sided,
                sd=sd,
                allocation=allocation,
                n_treated=nt,
                n_control=nc,
            )
        lo, hi = hi + 1, min(2 * hi, _N_MAX)
    return Unsupported(reason=f"no sample size below {_N_MAX} reaches the target power")


def power_curve(
    n: int,
    sd: float,
    effects: Sequence[float],
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
    allocation: float = 0.5,
) -> PowerCurve:
    """Power over a grid of effect sizes for a fixed two-arm design."""
    se = difference_se(n, sd, allocation)
    _check_alpha_power(alpha, None)
    grid = np.asarray(list(effects), dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("effects must be a non-empty 1-D sequence")
    powers = _power_array(grid, se, alpha, two_sided)
    return PowerCurve(
        design="difference_in_means",
        effects=tuple(float(e) for e in grid),
        powers=tuple(float(p) for p in powers),
        se=se,
        alpha=alpha,
        two_sided=two_sided,
        n=n,
        sd=sd,
        allocation=allocation,
    )


# -- regression coefficient with a design SE -------------------------------------------


def coefficient_power(
    effect: float, se: float, *, alpha: float = 0.05, two_sided: bool = True
) -> PowerResult:
    """Power to detect a coefficient of size ``effect`` whose design SE is ``se``."""
    return power_from_se(effect, se, alpha=alpha, two_sided=two_sided)


def coefficient_mde(
    se: float, *, alpha: float = 0.05, power: float = 0.8, two_sided: bool = True
) -> MDE:
    """Minimum detectable coefficient given its design standard error."""
    _check_alpha_power(alpha, power)
    if not (math.isfinite(se) and se >= 0):
        raise ValueError(f"se must be finite and non-negative, got {se}")
    return MDE(
        design="coefficient",
        effect=_mde_value(se, alpha, power, two_sided),
        se=se,
        alpha=alpha,
        power=power,
        two_sided=two_sided,
    )


def coefficient_sample_size(
    effect: float,
    se: float,
    n: int,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> SampleSize | Unsupported:
    """Observations needed for a coefficient whose SE is ``se`` at ``n`` observations.

    Assumes ``se(m) = se · sqrt(n / m)``: the design is replicated, not
    re-optimized. Returns ``Unsupported`` for a zero effect or an ``n`` above
    ``10**8``.
    """
    _check_alpha_power(alpha, power)
    if n < 1:
        raise ValueError("n must be positive")
    if not (math.isfinite(se) and se > 0):
        raise ValueError(f"se must be finite and positive, got {se}")
    if not math.isfinite(effect) or effect == 0.0:
        return Unsupported(reason="sample size for a zero effect is unbounded")
    target_se = abs(effect) / (_mde_value(1.0, alpha, power, two_sided))
    m_continuous = n * (se / target_se) ** 2
    if m_continuous > _N_MAX:
        return Unsupported(
            reason=f"sample size exceeds {_N_MAX}: effect is negligible against se",
            detail={"n_continuous": f"{m_continuous:.4g}"},
        )
    m = max(1, int(math.ceil(m_continuous - 1e-9)))
    se_m = se * math.sqrt(n / m)
    achieved = _power_value(effect, se_m, alpha, two_sided)
    while achieved + 1e-12 < power:
        m += 1
        se_m = se * math.sqrt(n / m)
        achieved = _power_value(effect, se_m, alpha, two_sided)
    return SampleSize(
        design="coefficient",
        n=m,
        effect=effect,
        se=se_m,
        alpha=alpha,
        power=achieved,
        target_power=power,
        two_sided=two_sided,
    )
