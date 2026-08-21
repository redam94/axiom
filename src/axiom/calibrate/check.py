"""Did the calibration take? Posterior-versus-measurement agreement (D6.6).

Ported from the parent's ``validation/calibration.py``: after a fit — by
either route, or with no calibration at all — realize the measurement's
estimand on the fitted producer (``estimands.realize``, the same arithmetic
every estimand goes through) and compare the measurement to the posterior.

The comparison is the two-sample normal approximation. With the
measurement ``m ± se`` and the posterior mean ``μ`` and standard deviation
``τ`` of the realized estimand,

    z = (m − μ) / sqrt(se² + τ²),        p = 2 · (1 − Φ(|z|)),

treating both as independent normals. ``inside`` says whether the
measurement's point lies in the posterior interval (the caller's
``definition`` and ``mass``, a ``core.Interval``); it is a different
question from ``z`` — a wide posterior contains a point that a tight
measurement still sits far from in ``z`` — so both are reported. The
verdict thresholds are explicit arguments: ``|z| < tension_at`` (default
1) is ``agrees``, ``|z| < disagrees_at`` (default 2) is ``tension``, and
at or beyond it ``disagrees``. Those are the parent's defaults and are
conventions, not tests; ``z`` and ``p`` are there so a reader can apply
their own.

A producer that cannot realize the estimand returns ``realize``'s typed
``Unsupported`` or ``Blocked`` unchanged. Identification is assumed (the
measurement is being *compared* to the posterior, not transported), which
``realize`` records as a downgraded result under an asserted assumption;
that result travels in ``Agreement.realized``.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from pydantic import Field, model_validator
from scipy.special import ndtr

from axiom.calibrate.evidence import Measurement
from axiom.core import Blocked, Interval, Spec, SupportsEstimands, Unsupported
from axiom.core.intervals import IntervalDefinition
from axiom.estimands import EstimandResult, RealizedDraws, realize

__all__ = ["Agreement", "AgreementVerdict", "agreement"]

AgreementVerdict = Literal["agrees", "tension", "disagrees"]


class Agreement(Spec):
    """How a measurement sits against the posterior of its own estimand.

    ``z`` and ``p`` are the two-sample normal comparison of the module
    docstring; ``inside`` whether ``estimate`` lies in ``interval`` (the
    posterior's, with its definition and mass); ``verdict`` the banded
    reading at the recorded ``tension_at`` / ``disagrees_at`` thresholds.
    ``realized`` is the full ``EstimandResult`` the comparison was made
    against, with its status, assumptions, and ledger.
    """

    estimand_name: str
    estimand_hash: str
    measurement_hash: str
    source: str
    estimate: float
    se: float = Field(gt=0)
    posterior_mean: float
    posterior_sd: float = Field(ge=0)
    interval: Interval
    z: float
    p: float = Field(ge=0, le=1)
    inside: bool
    verdict: AgreementVerdict
    tension_at: float = Field(gt=0)
    disagrees_at: float = Field(gt=0)
    n_draws: int = Field(gt=0)
    realized: EstimandResult

    @model_validator(mode="after")
    def _consistent(self) -> Agreement:
        if self.tension_at >= self.disagrees_at:
            raise ValueError(
                f"tension_at ({self.tension_at}) must be below disagrees_at ({self.disagrees_at})"
            )
        if not math.isfinite(self.z):
            raise ValueError("z must be finite")
        return self


def _verdict(z: float, tension_at: float, disagrees_at: float) -> AgreementVerdict:
    a = abs(z)
    if a < tension_at:
        return "agrees"
    if a < disagrees_at:
        return "tension"
    return "disagrees"


def agreement(
    producer: SupportsEstimands,
    measurement: Measurement,
    *,
    definition: IntervalDefinition = "hdi",
    mass: float = 0.9,
    seed: int | None = None,
    tension_at: float = 1.0,
    disagrees_at: float = 2.0,
) -> Agreement | Unsupported | Blocked:
    """Realize ``measurement.estimand`` on ``producer`` and compare (module docstring).

    ``definition`` / ``mass`` define the posterior interval ``inside`` is
    judged against; ``seed`` is passed to the producer. ``tension_at`` and
    ``disagrees_at`` are the ``|z|`` bands of the verdict and must satisfy
    ``0 < tension_at < disagrees_at`` (``ValueError``). A producer that
    cannot realize the estimand returns the typed failure ``realize``
    produced.
    """
    if not (0.0 < tension_at < disagrees_at):
        raise ValueError(f"need 0 < tension_at < disagrees_at; got {tension_at} and {disagrees_at}")
    out = realize(
        measurement.estimand,
        producer,
        assume_identified=True,
        definition=definition,
        mass=mass,
        seed=seed,
        keep_draws=True,
    )
    if isinstance(out, Unsupported | Blocked):
        return out
    assert isinstance(out, RealizedDraws)  # keep_draws=True
    result = out.result
    draws = np.asarray(out.draws, dtype=np.float64).reshape(-1)
    mu = float(draws.mean())
    tau = float(draws.std(ddof=1)) if draws.size > 1 else 0.0
    se = float(measurement.se)
    z = (float(measurement.estimate) - mu) / math.sqrt(se**2 + tau**2)
    p = float(2.0 * (1.0 - ndtr(abs(z))))
    interval = result.summary.interval
    return Agreement(
        estimand_name=measurement.estimand.name,
        estimand_hash=measurement.estimand.content_hash(),
        measurement_hash=measurement.content_hash(),
        source=measurement.source,
        estimate=float(measurement.estimate),
        se=se,
        posterior_mean=mu,
        posterior_sd=tau,
        interval=interval,
        z=z,
        p=min(1.0, max(0.0, p)),
        inside=bool(interval.lower <= measurement.estimate <= interval.upper),
        verdict=_verdict(z, tension_at, disagrees_at),
        tension_at=tension_at,
        disagrees_at=disagrees_at,
        n_draws=int(draws.size),
        realized=result,
    )
