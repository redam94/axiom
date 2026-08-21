"""Model-anchored effect size: is the experiment powered to detect what the model already believes?

Ported from the parent's ``planning/design_anchor.py`` by specification.
An experiment is usually powered at a minimum detectable effect (MDE) chosen
by convention. If a fitted model already puts most of its posterior mass
above that MDE, the experiment is powered to confirm a belief, not to test
it — and its MDE should be anchored to the posterior instead.

``anchor_effect`` takes any ``SupportsPosterior`` and the parameter name of
the estimand (a vector parameter needs ``index``), and reports, in
``AnchoredEffect``:

* the posterior mean, sd, and a ``core.Interval`` of the parameter;
* ``probability_exceeds_mde`` — the posterior probability the true effect
  exceeds the MDE, computed from the draws — and its Gaussian
  approximation ``1 − Φ((mde − mean) / sd)`` for reference;
* ``anchored_effect`` — the effect the model believes is exceeded with
  probability ``credence``: the ``1 − credence`` posterior quantile. Powering
  at this value detects what the model *doubts*, which is what a test is for;
* ``already_believed`` — ``probability_exceeds_mde ≥ credence``: the flag.

Missing parameter, non-finite draws, or a vector parameter without an
``index`` return ``Unsupported``.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from pydantic import field_validator
from scipy.special import ndtr

from axiom.core import (
    Interval,
    NonEmptyStr,
    Spec,
    SupportsPosterior,
    Unsupported,
    interval,
)
from axiom.core.intervals import IntervalDefinition

__all__ = ["AnchoredEffect", "anchor_draws", "anchor_effect"]

Array = npt.NDArray[np.float64]


def _probability(name: str, v: float) -> float:
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {v}")
    return v


class AnchoredEffect(Spec):
    """A posterior's view of an MDE.

    ``probability_exceeds_mde`` is the fraction of posterior draws above
    ``mde``; ``probability_exceeds_mde_gaussian`` the same under
    ``N(posterior_mean, posterior_sd²)``. ``anchored_effect`` is the
    ``1 − credence`` posterior quantile. ``already_believed`` is
    ``probability_exceeds_mde >= credence``.
    """

    parameter: NonEmptyStr
    posterior_mean: float
    posterior_sd: float
    posterior_interval: Interval
    n_draws: int
    mde: float
    probability_exceeds_mde: float
    probability_exceeds_mde_gaussian: float
    credence: float
    anchored_effect: float
    already_believed: bool

    @field_validator("probability_exceeds_mde", "probability_exceeds_mde_gaussian")
    @classmethod
    def _probabilities(cls, v: float) -> float:
        return _probability("probability", v)

    @field_validator("credence")
    @classmethod
    def _credence_open(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"credence must be in (0, 1), got {v}")
        return v

    @property
    def mde_ratio(self) -> float:
        """``mde / posterior_mean``: how the convention compares with the belief."""
        if self.posterior_mean == 0.0:
            return math.inf
        return self.mde / self.posterior_mean


def anchor_draws(
    draws: npt.ArrayLike,
    parameter: str,
    mde: float,
    *,
    credence: float = 0.9,
    definition: IntervalDefinition = "hdi",
    mass: float = 0.9,
) -> AnchoredEffect | Unsupported:
    """``anchor_effect`` on a flat array of draws of one scalar parameter."""
    if not (math.isfinite(mde) and mde > 0):
        raise ValueError(f"mde must be finite and positive, got {mde}")
    if not 0.0 < credence < 1.0:
        raise ValueError(f"credence must be in (0, 1), got {credence}")
    x = np.asarray(draws, dtype=np.float64).ravel()
    if x.size < 2:
        return Unsupported(
            reason=f"need at least two draws of {parameter!r} to anchor an effect, got {x.size}"
        )
    if not np.all(np.isfinite(x)):
        bad = int(np.count_nonzero(~np.isfinite(x)))
        return Unsupported(
            reason=f"{bad} of {x.size} draws of {parameter!r} are not finite",
            detail={"parameter": parameter, "non_finite": str(bad)},
        )
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    p_emp = float(np.mean(x > mde))
    p_gauss = 1.0 - float(ndtr((mde - mean) / sd)) if sd > 0 else float(mean > mde)
    anchored = float(np.quantile(x, 1.0 - credence))
    return AnchoredEffect(
        parameter=parameter,
        posterior_mean=mean,
        posterior_sd=sd,
        posterior_interval=interval(x, definition=definition, mass=mass),
        n_draws=int(x.size),
        mde=float(mde),
        probability_exceeds_mde=p_emp,
        probability_exceeds_mde_gaussian=p_gauss,
        credence=float(credence),
        anchored_effect=anchored,
        already_believed=p_emp >= credence,
    )


def anchor_effect(
    posterior: SupportsPosterior,
    parameter: str,
    mde: float,
    *,
    index: int | None = None,
    credence: float = 0.9,
    definition: IntervalDefinition = "hdi",
    mass: float = 0.9,
) -> AnchoredEffect | Unsupported:
    """Anchor an MDE against the posterior of ``parameter``.

    ``index`` selects one element of a vector parameter (in its flattened
    trailing shape); a vector parameter without ``index`` is ``Unsupported``.
    ``credence`` is the posterior probability at which a belief counts as
    "already believed" and the quantile level of ``anchored_effect``.
    """
    if parameter not in posterior.names():
        return Unsupported(
            reason=f"posterior has no parameter {parameter!r}",
            missing=(parameter,),
            detail={"available": ", ".join(sorted(posterior.names()))},
        )
    a = np.asarray(posterior.draws(parameter), dtype=np.float64)
    if a.ndim < 2:
        return Unsupported(
            reason=f"draws of {parameter!r} must be (chain, draw, *shape); got shape {a.shape}"
        )
    flat = a.reshape(a.shape[0] * a.shape[1], -1)
    if flat.shape[1] == 1 and index is None:
        x = flat[:, 0]
    elif index is None:
        return Unsupported(
            reason=(
                f"{parameter!r} has trailing shape {a.shape[2:]}; pass index to select one element"
            ),
            detail={"parameter": parameter, "shape": str(a.shape[2:])},
        )
    elif not 0 <= index < flat.shape[1]:
        return Unsupported(
            reason=f"index {index} out of range for {parameter!r} with {flat.shape[1]} elements",
            detail={"parameter": parameter, "size": str(flat.shape[1])},
        )
    else:
        x = flat[:, index]
    name = parameter if index is None else f"{parameter}[{index}]"
    return anchor_draws(x, name, mde, credence=credence, definition=definition, mass=mass)
