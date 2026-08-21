"""Prior predictive of the response surface: what the priors imply before any fit.

Ported from the parent's ``diagnostics/saturation.py``. Draws parameters
from the surface's priors (``axiom.diagnose.sbc.draw_prior``), pushes each
draw through ``Surface.forward`` at the panel's observed doses — the one
``forward()`` — and summarizes the implied response and each treatment's
implied contribution (the response with that treatment's dose zeroed,
subtracted from the response at the observed doses, through the same
forward). Two implausibility rules, both stated on the result:

* **sign** — a draw's mean contribution of a treatment has the wrong sign
  for ``expected_sign`` (``"positive"`` by default);
* **magnitude** — a draw's largest absolute contribution of a treatment
  exceeds ``magnitude_factor`` times the observed outcome's standard
  deviation.

The share of prior mass that trips either rule is reported per treatment;
``passed`` is whether the share of draws tripping any rule is at most
``tolerance``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator

from axiom.core import Interval, ModelSpec, Spec, Summary, Unsupported, eti, summarize
from axiom.data import Panel
from axiom.surface import Surface, SurfaceSpec, prepare

__all__ = ["ExpectedSign", "PriorPredictive", "prior_predictive"]

Array = npt.NDArray[np.float64]
ExpectedSign = Literal["positive", "negative", "any"]


class PriorPredictive(Spec):
    """Prior-implied responses at the observed doses and the share that is implausible.

    ``response`` summarizes the draw-level mean response over all cells,
    ``response_range`` is the equal-tailed interval of every cell over
    every draw; ``contribution[t]`` summarizes the draw-level mean
    contribution of treatment ``t``. ``share_wrong_sign[t]`` and
    ``share_implausible_magnitude[t]`` are fractions of the ``n`` draws;
    ``share_flagged`` is the fraction tripping either rule for any
    treatment.
    """

    spec_hash: str
    panel_hash: str
    n: int = Field(ge=2)
    seed: int | None
    n_units: int = Field(ge=1)
    n_periods: int = Field(ge=1)
    treatments: tuple[str, ...] = Field(min_length=1)
    outcome_sd: float = Field(ge=0)
    expected_sign: ExpectedSign
    magnitude_factor: float = Field(gt=0)
    tolerance: float = Field(ge=0, le=1)
    mass: float = Field(gt=0, lt=1)
    response: Summary
    response_range: Interval
    contribution: dict[str, Summary]
    share_wrong_sign: dict[str, float]
    share_implausible_magnitude: dict[str, float]
    share_flagged: float = Field(ge=0, le=1)
    passed: bool

    @model_validator(mode="after")
    def _consistent(self) -> PriorPredictive:
        keys = set(self.treatments)
        for name, d in (
            ("contribution", self.contribution),
            ("share_wrong_sign", self.share_wrong_sign),
            ("share_implausible_magnitude", self.share_implausible_magnitude),
        ):
            if set(d) != keys:
                raise ValueError(f"{name} must be keyed by every treatment")
        if self.passed != (self.share_flagged <= self.tolerance):
            raise ValueError("passed must be share_flagged <= tolerance")
        return self


def _prior_draws(model: ModelSpec, rng: np.random.Generator, n: int) -> Mapping[str, Array]:
    # Lazy for the same reason as ``learning``: keep this module importable alone.
    from axiom.diagnose.sbc import draw_prior

    return draw_prior(model, rng, n=n)


def prior_predictive(
    spec: SurfaceSpec,
    panel: Panel,
    *,
    n: int = 200,
    seed: int | None = 0,
    expected_sign: ExpectedSign = "positive",
    magnitude_factor: float = 10.0,
    tolerance: float = 0.2,
    mass: float = 0.9,
) -> PriorPredictive | Unsupported:
    """Push ``n`` prior draws through ``Surface.forward`` at the panel's doses and summarize.

    Every number goes through the spec's built model; a draw whose forward
    is not finite is a typed ``Unsupported`` (naming how many), never
    dropped.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if magnitude_factor <= 0:
        raise ValueError(f"magnitude_factor must be positive, got {magnitude_factor}")
    if not 0 <= tolerance <= 1:
        raise ValueError(f"tolerance must lie in [0, 1], got {tolerance}")
    surface = Surface(spec)
    data = prepare(spec, panel)
    observed = np.asarray(data[spec.outcome.name], dtype=float)
    outcome_sd = float(observed.std(ddof=1)) if observed.size > 1 else 0.0
    draws = _prior_draws(surface.model, np.random.default_rng(seed), n)
    names = spec.treatment_names
    zeroed = {t: {**data, t: np.zeros_like(np.asarray(data[t], dtype=float))} for t in names}

    responses: list[Array] = []
    contributions: dict[str, list[Array]] = {t: [] for t in names}
    bad = 0
    for i in range(n):
        theta = {k: np.asarray(v, dtype=float)[i] for k, v in draws.items()}
        y = np.asarray(surface.forward(data, theta), dtype=float)
        if not np.all(np.isfinite(y)):
            bad += 1
            continue
        responses.append(y)
        for t in names:
            contributions[t].append(y - np.asarray(surface.forward(zeroed[t], theta), dtype=float))
    if bad:
        return Unsupported(
            reason=f"{bad} of {n} prior draws gave a non-finite response",
            detail={"non_finite": str(bad), "n": str(n)},
        )
    resp = np.stack(responses)  # (n, n_units, n_periods)
    flagged = np.zeros(n, dtype=bool)
    contribution: dict[str, Summary] = {}
    wrong_sign: dict[str, float] = {}
    too_big: dict[str, float] = {}
    for t in names:
        c = np.stack(contributions[t])
        mean_c = c.reshape(n, -1).mean(axis=1)
        max_abs = np.abs(c).reshape(n, -1).max(axis=1)
        if expected_sign == "positive":
            sign_bad = mean_c < 0.0
        elif expected_sign == "negative":
            sign_bad = mean_c > 0.0
        else:
            sign_bad = np.zeros(n, dtype=bool)
        size_bad = max_abs > magnitude_factor * outcome_sd if outcome_sd > 0 else max_abs > 0.0
        flagged |= sign_bad | size_bad
        contribution[t] = summarize(mean_c, definition="eti", mass=mass)
        wrong_sign[t] = float(sign_bad.mean())
        too_big[t] = float(size_bad.mean())
    share = float(flagged.mean())
    return PriorPredictive(
        spec_hash=spec.content_hash(),
        panel_hash=panel.content_hash(),
        n=int(n),
        seed=seed,
        n_units=int(resp.shape[1]),
        n_periods=int(resp.shape[2]),
        treatments=names,
        outcome_sd=outcome_sd,
        expected_sign=expected_sign,
        magnitude_factor=float(magnitude_factor),
        tolerance=float(tolerance),
        mass=float(mass),
        response=summarize(resp.reshape(n, -1).mean(axis=1), definition="eti", mass=mass),
        response_range=eti(resp.reshape(-1), mass),
        contribution=contribution,
        share_wrong_sign=wrong_sign,
        share_implausible_magnitude=too_big,
        share_flagged=share,
        passed=share <= tolerance,
    )
