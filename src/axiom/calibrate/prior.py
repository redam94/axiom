"""The prior route: randomized evidence becomes an explicit amplitude prior on the surface.

Ported from the parent's ``calibration/experiment.py`` (``design_factor``,
``mean_sd_to_gamma``, ``derive_channel_prior``; ledger row
``calibrate/{evidence,prior}.py``, PORT), with the vocabulary generalized
and the result made a typed, hashed ``SurfaceSpec`` rather than a mutated
model. Two stages (D6.2):

1. **Combine.** Measurements of the *same* target estimand (equal content
   hashes) are pooled by inverse variance
   (``evidence.combine_inverse_variance``). A measurement of a *different*
   estimand is refused with a typed ``Unsupported`` unless the caller
   supplies a multiplicative ``corrections[source]`` factor — the resolved
   output of ``calibrate.transfer`` — which maps its estimate and standard
   error onto the target and writes a ledger line.
2. **Map to the amplitude.** The surface's amplitude ``beta`` is not the
   measured quantity; the realized estimand is ``contribution = g(beta, k, s,
   doses, ...)``, and at fixed shape and scale it is proportional to
   ``beta``. The **design factor** is that proportionality measured on the
   posterior: ``design_factor = mean(contribution_draws / beta_draws)``, the
   *mean of per-draw ratios* (this is the form that reproduces the parent's
   golden value ``999.5551091541466`` at ``1e-12``; ratio of means, median
   of ratios, and regression through the origin do not). The target's
   ``(mean, se)`` divided by the factor is the implied ``(mean, sd)`` of
   ``beta``, which is moment-matched to a positive-support family:
   ``lognormal`` (``sigma² = log(1 + (sd/mean)²)``, ``mu = log(mean) −
   sigma²/2``) or ``gamma`` (``shape = (mean/sd)²``, ``rate = mean/sd²``).

The result is a new ``SurfaceSpec`` whose kernel for the treatment carries
``amplitude_prior``; ``surface.build`` uses it in place of the default
``halfnormal(amplitude_scale)``. Only the amplitude's prior changes — the
curve's shape and scale priors are untouched, which is the defining property
of the prior route (the likelihood route, ``calibrate.likelihood``, moves
the shape too). Every step writes a ``LedgerLine`` whose ``detail`` carries
``"counterfactual"`` (the number without the step) and ``"value"`` (with it).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from axiom.core import Assumption, LedgerLine, Prior, Spec, Unsupported
from axiom.estimands import Estimand
from axiom.surface import SurfaceSpec
from axiom.surface.kernels import AMPLITUDE_PRIOR_FAMILIES

from .evidence import Measurement, combine_inverse_variance

__all__ = [
    "CalibratedSpec",
    "PriorFamily",
    "amplitude_prior",
    "combine_measurements",
    "derive_prior",
    "design_factor",
    "lognormal_from_moments",
    "mean_sd_to_gamma",
]

PriorFamily = Literal["lognormal", "gamma"]
"""Moment-matched families for the amplitude prior (both have support on R+)."""


# -- closed forms ----------------------------------------------------------------------


def design_factor(beta_samples: npt.ArrayLike, contribution_samples: npt.ArrayLike) -> float:
    """``mean(contribution / beta)`` over paired posterior draws.

    ``beta_samples`` are draws of the amplitude parameter; ``contribution_samples``
    are draws of the realized target quantity (the estimand evaluated through
    ``surface.forward`` at the experiment's doses) under the *same* draws, so
    each ratio is the quantity per unit of amplitude at one posterior point.
    The mean of per-draw ratios is the parent's formula and the one that
    reproduces the golden fixture at ``1e-12``; ratio of means
    (``998.857``), median of ratios (``998.474``) and regression through the
    origin (``998.067``) were tried and rejected. Raises ``ValueError`` on
    mismatched or empty draws, non-finite values, or any ``beta == 0``.
    """
    b = np.asarray(beta_samples, dtype=float).ravel()
    c = np.asarray(contribution_samples, dtype=float).ravel()
    if b.size == 0:
        raise ValueError("design_factor needs at least one draw")
    if b.shape != c.shape:
        raise ValueError(f"beta and contribution draws differ in length: {b.size} vs {c.size}")
    if not (np.all(np.isfinite(b)) and np.all(np.isfinite(c))):
        raise ValueError("draws must be finite")
    if np.any(b == 0.0):
        raise ValueError("beta draws must be non-zero to form the ratio contribution / beta")
    return float(np.mean(c / b))


def mean_sd_to_gamma(mean: float, sd: float) -> tuple[float, float]:
    """``(shape, rate)`` of the gamma with the given mean and standard deviation.

    ``shape = (mean / sd)²``, ``rate = mean / sd²`` (``axiom.core.Prior`` uses
    the *rate* convention for ``gamma``'s ``beta``).
    """
    if not (math.isfinite(mean) and mean > 0):
        raise ValueError(f"mean must be finite and positive, got {mean}")
    if not (math.isfinite(sd) and sd > 0):
        raise ValueError(f"sd must be finite and positive, got {sd}")
    ratio = mean / sd
    return ratio * ratio, mean / (sd * sd)


def lognormal_from_moments(mean: float, sd: float) -> tuple[float, float]:
    """``(mu, sigma)`` of the lognormal with the given mean and standard deviation.

    ``sigma = sqrt(log(1 + (sd / mean)²))`` (the parent's
    ``lognormal_sigma_from_moments``) and ``mu = log(mean) − sigma² / 2`` so
    that ``E[X] = exp(mu + sigma²/2) = mean`` exactly.
    """
    if not (math.isfinite(mean) and mean > 0):
        raise ValueError(f"mean must be finite and positive, got {mean}")
    if not (math.isfinite(sd) and sd > 0):
        raise ValueError(f"sd must be finite and positive, got {sd}")
    cv = sd / mean
    sigma = math.sqrt(math.log1p(cv * cv))
    return math.log(mean) - 0.5 * sigma * sigma, sigma


def amplitude_prior(mean: float, sd: float, family: PriorFamily = "lognormal") -> Prior:
    """The moment-matched ``Prior`` for a positive amplitude with the given mean and sd."""
    if family == "lognormal":
        mu, sigma = lognormal_from_moments(mean, sd)
        return Prior(family="lognormal", hyper={"mu": mu, "sigma": sigma})
    if family == "gamma":
        shape, rate = mean_sd_to_gamma(mean, sd)
        return Prior(family="gamma", hyper={"alpha": shape, "beta": rate})
    raise ValueError(
        f"unknown prior family {family!r}; choose from lognormal, gamma "
        f"(positive support: {sorted(AMPLITUDE_PRIOR_FAMILIES)})"
    )


# -- results ---------------------------------------------------------------------------


class CalibratedSpec(Spec):
    """The prior route's output: a new surface spec and how it was reached.

    ``spec`` is the calibrated ``SurfaceSpec`` (the input with one kernel's
    ``amplitude_prior`` set); ``prior`` is that prior. ``target`` is the
    content hash of the estimand the evidence was pooled on; ``sources`` the
    measurement sources in order; ``plan_statuses`` maps each source to
    ``"identified"`` (same estimand) or ``"corrected"`` (mapped through a
    supplied correction factor). The numbers along the way are kept so a
    reader can recompute every step: pooled ``target_mean``/``target_se``,
    the ``design_factor``, and the implied ``amplitude_mean``/``amplitude_sd``.
    """

    spec: SurfaceSpec
    treatment: str
    parameter: str
    prior: Prior
    family: PriorFamily
    target: str
    sources: tuple[str, ...]
    plan_statuses: dict[str, str]
    target_mean: float
    target_se: float
    design_factor: float
    amplitude_mean: float
    amplitude_sd: float
    ledger_lines: tuple[LedgerLine, ...]


# -- ledger helpers --------------------------------------------------------------------


def _assumption(name: str, facet: str, statement: str, challenged_by: str) -> Assumption:
    return Assumption(name=name, facet=facet, statement=statement, challenged_by=challenged_by)


def _line(
    kind: str,
    statement: str,
    assumption: Assumption,
    counterfactual: float,
    value: float,
    *,
    source: str = "",
    target: str = "",
    **extra: str,
) -> LedgerLine:
    detail = {"counterfactual": repr(float(counterfactual)), "value": repr(float(value)), **extra}
    return LedgerLine(
        kind=kind,
        statement=statement,
        assumption=assumption,
        detail=detail,
        source=source,
        target=target,
    )


# -- stage 1: combine ------------------------------------------------------------------


def combine_measurements(
    measurements: Sequence[Measurement],
    *,
    target: Estimand | None = None,
    corrections: Mapping[str, float] | None = None,
) -> tuple[float, float, tuple[LedgerLine, ...], dict[str, str]] | Unsupported:
    """Pool measurements on one target estimand: ``(mean, se, ledger_lines, plan_statuses)``.

    ``target`` defaults to the first measurement's estimand. A measurement
    whose estimand hash differs from the target's must have a multiplicative
    ``corrections[source]`` factor (the resolved product of a
    ``calibrate.transfer`` plan); its estimate and standard error are both
    scaled by it and a ``transfer:correction`` line records the
    uncorrected estimate as the counterfactual. Without a factor the
    function returns ``Unsupported`` naming the sources it cannot read.
    The pooled line's counterfactual is the single most precise measurement.
    """
    if not measurements:
        return Unsupported(reason="no measurements to combine", missing=("measurements",))
    corrections = dict(corrections or {})
    goal = target if target is not None else measurements[0].estimand
    goal_hash = goal.content_hash()
    lines: list[LedgerLine] = []
    statuses: dict[str, str] = {}
    estimates: list[float] = []
    ses: list[float] = []
    unreadable: list[str] = []
    seen: set[str] = set()
    for m in measurements:
        if m.source in seen:
            raise ValueError(f"duplicate measurement source {m.source!r}")
        seen.add(m.source)
        if m.target == goal_hash:
            statuses[m.source] = "identified"
            estimates.append(m.estimate)
            ses.append(m.se)
            continue
        factor = corrections.get(m.source)
        if factor is None:
            unreadable.append(m.source)
            continue
        if not (math.isfinite(factor) and factor > 0):
            raise ValueError(f"correction for {m.source!r} must be finite and positive")
        statuses[m.source] = "corrected"
        estimates.append(m.estimate * factor)
        ses.append(m.se * factor)
        lines.append(
            _line(
                "transfer:correction",
                f"measurement {m.source!r} of estimand {m.estimand.name!r} read as "
                f"{goal.name!r} through a multiplicative factor {factor!r}",
                _assumption(
                    "correction_factor_valid",
                    "transfer",
                    "the supplied factor maps the measured estimand onto the target estimand "
                    "(resolved by calibrate.transfer or asserted by the analyst)",
                    "a surface-based resolution of the same plan giving a different factor",
                ),
                m.estimate,
                m.estimate * factor,
                source=m.target,
                target=goal_hash,
                measurement=m.source,
                factor=repr(float(factor)),
                facets=",".join(m.estimand.differing_facets(goal)),
            )
        )
    if unreadable:
        return Unsupported(
            reason=(
                "measurements estimate a different estimand than the target and carry no "
                f"correction factor: {unreadable}; resolve their transfer plan first"
            ),
            missing=tuple(f"correction:{s}" for s in unreadable),
            detail={"target": goal_hash, "target_name": goal.name},
        )
    mean, se = combine_inverse_variance(estimates, ses)
    best = int(np.argmin(ses))
    lines.append(
        _line(
            "evidence:combine",
            f"{len(estimates)} measurement(s) of {goal.name!r} pooled by inverse variance",
            _assumption(
                "independent_unbiased_studies",
                "quantity",
                "each measurement is an independent, unbiased estimate of the target estimand "
                "with a correctly reported standard error (fixed-effect pooling)",
                "heterogeneity beyond the reported standard errors (a Q statistic, or a "
                "random-effects pooling that widens the interval)",
            ),
            estimates[best],
            mean,
            target=goal_hash,
            se=repr(float(se)),
            counterfactual_source=measurements[best].source,
            counterfactual_se=repr(float(ses[best])),
            sources=",".join(m.source for m in measurements),
        )
    )
    return mean, se, tuple(lines), statuses


# -- stage 2: derive -------------------------------------------------------------------


def _amplitude_name(spec: SurfaceSpec, treatment: str) -> str:
    """The single amplitude the prior route rewrites, or a refusal naming why there isn't one."""
    kernel = spec.kernel_of(treatment)
    stems = [stem for stem, role in kernel.roles.items() if role == "amplitude"]
    if len(stems) == 1:
        return f"{stems[0]}_{treatment}"
    if not stems:  # pragma: no cover - every shipped kernel declares an amplitude
        raise ValueError(f"kernel {kernel.name!r} declares no amplitude parameter")
    raise ValueError(
        f"kernel {kernel.name!r} declares {len(stems)} amplitudes ({', '.join(stems)}); the "
        "prior route rewrites the one amplitude a randomized contrast identifies, and a basis "
        "family spreads the response over several coefficients with no such single target. "
        "Use the likelihood route (calibrate.attach) for a basis family: it constrains a "
        "function of the whole coefficient vector instead."
    )


def derive_prior(
    measurements: Sequence[Measurement],
    spec: SurfaceSpec,
    treatment: str,
    *,
    beta_draws: npt.ArrayLike,
    contribution_draws: npt.ArrayLike,
    family: PriorFamily = "lognormal",
    target: Estimand | None = None,
    corrections: Mapping[str, float] | None = None,
) -> CalibratedSpec | Unsupported:
    """Turn measurements of a target estimand into an amplitude prior on ``spec`` (D6.2).

    ``beta_draws`` are posterior (or prior-predictive) draws of the
    amplitude parameter of ``treatment``'s kernel; ``contribution_draws`` are
    the target estimand realized under the same draws, in the unit of the
    measurements' ``estimate``. Their per-draw ratio is the design factor
    that converts the pooled ``(mean, se)`` of the target into the implied
    ``(mean, sd)`` of the amplitude, which is moment-matched to ``family``.
    The returned ``CalibratedSpec.spec`` is ``spec`` with that kernel's
    ``amplitude_prior`` set and nothing else changed; the ledger carries a
    combine line, a design-factor line and a moment-match line (plus one
    correction line per corrected measurement). Returns ``Unsupported``
    when the measurements cannot be pooled; raises ``ValueError`` on
    malformed inputs or a non-positive implied amplitude mean (a
    positive-support family cannot represent it).
    """
    spec.treatment(treatment)
    combined = combine_measurements(measurements, target=target, corrections=corrections)
    if isinstance(combined, Unsupported):
        return combined
    target_mean, target_se, lines, statuses = combined
    goal = target if target is not None else measurements[0].estimand
    goal_hash = goal.content_hash()

    factor = design_factor(beta_draws, contribution_draws)
    if factor <= 0:
        raise ValueError(
            f"design factor {factor!r} is not positive: the target estimand decreases in the "
            "amplitude, so no positive-support amplitude prior represents the evidence"
        )
    amp_mean = target_mean / factor
    amp_sd = target_se / factor
    if amp_mean <= 0:
        raise ValueError(
            f"implied amplitude mean {amp_mean!r} is not positive (pooled target mean "
            f"{target_mean!r}); a {family} amplitude prior cannot represent this evidence"
        )
    parameter = _amplitude_name(spec, treatment)
    n_draws = int(np.asarray(beta_draws, dtype=float).size)
    lines = (
        *lines,
        _line(
            "prior:design_factor",
            f"pooled {goal.name!r} mapped to amplitude {parameter!r} through "
            f"design factor {factor!r} = mean(contribution / amplitude) over {n_draws} draws",
            _assumption(
                "amplitude_proportionality",
                "intervention",
                "at fixed shape and scale the target estimand is proportional to the "
                "amplitude, so the posterior mean ratio converts the evidence to the "
                "amplitude scale; the shape and scale posteriors are left as they are",
                "a strong posterior correlation between the amplitude and the shape or scale, "
                "or a design factor that varies across draws by more than the evidence's "
                "relative standard error",
            ),
            target_mean,
            amp_mean,
            target=goal_hash,
            design_factor=repr(float(factor)),
            n_draws=str(n_draws),
            se=repr(float(amp_sd)),
        ),
    )

    kernel = spec.kernel_of(treatment)
    default_param = next(
        p
        for p in kernel.parameters(
            treatment, spec.dose_dimension(treatment), spec.outcome_dimension
        )
        if p.name == parameter
    )
    before = default_param.prior
    assert before is not None  # kernel amplitudes always carry a prior
    prior = amplitude_prior(amp_mean, amp_sd, family)
    lines = (
        *lines,
        _line(
            "prior:moment_match",
            f"amplitude {parameter!r} prior set to {family}(mean {amp_mean!r}, sd {amp_sd!r}), "
            f"replacing {before.family}({before.hyper})",
            _assumption(
                "prior_family_adequate",
                "quantity",
                f"a {family} with the matched first two moments represents the evidence on "
                "the amplitude; the default prior is replaced, not multiplied in",
                "evidence whose interval is markedly asymmetric on the amplitude scale, or a "
                "default prior that carried independent information",
            ),
            _prior_sd(before),
            amp_sd,
            target=goal_hash,
            family=family,
            hyper=repr({k: float(v) for k, v in prior.hyper.items() if not isinstance(v, str)}),
            replaced=f"{before.family}({before.hyper})",
            mean=repr(float(amp_mean)),
        ),
    )

    if not isinstance(kernel, Spec):  # pragma: no cover - every shipped kernel is a Spec
        raise TypeError(f"kernel {kernel.name!r} is not a Spec and cannot be copied")
    new_kernel = kernel.model_copy(update={"amplitude_prior": prior})
    new_spec = SurfaceSpec.model_validate(
        {
            **spec.model_dump(),
            "kernels": {
                **{k: v.model_dump() for k, v in spec.kernels.items()},
                treatment: new_kernel.model_dump(),
            },
        }
    )
    return CalibratedSpec(
        spec=new_spec,
        treatment=treatment,
        parameter=parameter,
        prior=prior,
        family=family,
        target=goal_hash,
        sources=tuple(m.source for m in measurements),
        plan_statuses=statuses,
        target_mean=target_mean,
        target_se=target_se,
        design_factor=factor,
        amplitude_mean=amp_mean,
        amplitude_sd=amp_sd,
        ledger_lines=lines,
    )


def _prior_sd(prior: Prior) -> float:
    """The standard deviation of a numeric positive-support prior (the counterfactual width)."""
    h = {k: float(v) for k, v in prior.hyper.items() if not isinstance(v, str)}
    match prior.family:
        case "halfnormal":
            return h["sigma"] * math.sqrt(1.0 - 2.0 / math.pi)
        case "lognormal":
            s2 = h["sigma"] ** 2
            return math.sqrt((math.exp(s2) - 1.0) * math.exp(2.0 * h["mu"] + s2))
        case "gamma":
            return math.sqrt(h["alpha"]) / h["beta"]
    return math.nan  # pragma: no cover - amplitude priors are restricted to the three above
