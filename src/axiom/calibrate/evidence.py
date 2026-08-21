"""``Measurement``: one piece of randomized (or otherwise external) evidence, typed.

Ported from the parent's ``calibration/experiment.py`` (ledger row
``calibrate/{evidence,prior}.py``, PORT): the record an experiment hands to
the calibration routes, and the inverse-variance pooling of several records
on the same quantity.

A ``Measurement`` is a point estimate with a standard error *and* the
``Estimand`` it estimates. The estimand carries every scope facet
(intervention doses, population, window, level, ...) and the unit of the
estimate, so the prior route and the likelihood route can decide — by
content hash and by ``Estimand.transfer_to`` — whether two measurements are
the same quantity, whether a measurement is the quantity the surface
realizes, and what assumption bridges any gap. Nothing here converts units
or corrects scope; that is ``calibrate.transfer``, and every correction it
applies appends a ``LedgerLine``.

``combine_inverse_variance`` is the fixed-effect pooling of independent
estimates of one quantity: weights ``w_i = 1 / se_i²``, pooled mean
``Σ w_i t_i / Σ w_i``, pooled standard error ``sqrt(1 / Σ w_i)``. The golden
fixture records the pair as ``[mean, se]``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from axiom.core import (
    Assumption,
    Interval,
    NonEmptyStr,
    Spec,
    wald,
)
from axiom.core.intervals import IntervalDefinition
from axiom.estimands import Estimand

__all__ = [
    "Measurement",
    "combine_inverse_variance",
]


class Measurement(Spec):
    """One external estimate of an estimand, with its uncertainty and provenance (D6.1).

    ``estimate`` is in the unit the estimand's entities declare (the
    outcome's unit for a contrast, outcome per dose for a ratio or marginal,
    ...). ``se`` is its standard error on the same scale. ``definition`` and
    ``mass`` say how the reporting study summarized its uncertainty:
    ``"wald"`` for a frequentist ``estimate ± z·se``; ``"eti"``/``"hdi"``
    when the study reported a posterior whose summary was reduced to a
    (mean, sd) pair. ``method`` names the design (a ``design.methods`` name
    such as ``"difference_in_differences"``, or free text); ``n_units`` and
    ``n_periods`` are the experiment's size; ``design_factor`` is an
    optional pre-computed ratio between the measured quantity and the
    surface amplitude (when absent the prior route computes it from draws);
    ``source`` is the study identifier or content hash the evidence came
    from; ``assumptions`` are the design's own licensing assumptions (method
    assumptions from the registry, or ones the analyst asserts).
    """

    estimand: Estimand
    estimate: float
    se: float = Field(gt=0)
    definition: Literal["wald", "eti", "hdi"] = "wald"
    mass: float = Field(default=0.95, gt=0, lt=1)
    method: str = ""
    n_units: int | None = Field(default=None, gt=0)
    n_periods: int | None = Field(default=None, gt=0)
    design_factor: float | None = Field(default=None, gt=0)
    source: NonEmptyStr
    assumptions: tuple[Assumption, ...] = ()

    @model_validator(mode="after")
    def _finite(self) -> Measurement:
        if not math.isfinite(self.estimate):
            raise ValueError(f"estimate must be finite, got {self.estimate}")
        if not math.isfinite(self.se):
            raise ValueError(f"se must be finite, got {self.se}")
        if self.design_factor is not None and not math.isfinite(self.design_factor):
            raise ValueError(f"design_factor must be finite, got {self.design_factor}")
        return self

    @property
    def interval(self) -> Interval:
        """``estimate ± z_{(1+mass)/2} · se`` labelled with the study's own definition.

        The record holds a (mean, se) summary, so the interval is the normal
        approximation for every definition; the label is kept so the ledger
        shows how the study reported it rather than relabelling it Wald.
        """
        w = wald(self.estimate, self.se, self.mass)
        definition: IntervalDefinition = self.definition
        return Interval(lower=w.lower, upper=w.upper, definition=definition, mass=self.mass)

    @property
    def precision(self) -> float:
        """``1 / se²`` — the inverse-variance weight this measurement carries when pooled."""
        return 1.0 / (self.se * self.se)

    @property
    def target(self) -> str:
        """The content hash of the estimand this measurement estimates."""
        return self.estimand.content_hash()


def combine_inverse_variance(targets: Sequence[float], ses: Sequence[float]) -> tuple[float, float]:
    """Fixed-effect pooling of independent estimates of one quantity: ``(mean, se)``.

    ``w_i = 1 / se_i²``; ``mean = Σ w_i t_i / Σ w_i``; ``se = sqrt(1 / Σ w_i)``.
    One estimate pools to itself. Raises ``ValueError`` on empty or
    mismatched inputs or a non-positive / non-finite ``se``.
    """
    t = np.asarray(targets, dtype=float).ravel()
    s = np.asarray(ses, dtype=float).ravel()
    if t.size == 0:
        raise ValueError("combine_inverse_variance needs at least one estimate")
    if t.shape != s.shape:
        raise ValueError(f"targets and ses differ in length: {t.size} vs {s.size}")
    if not np.all(np.isfinite(t)):
        raise ValueError("targets must be finite")
    if not (np.all(np.isfinite(s)) and np.all(s > 0)):
        raise ValueError("standard errors must be finite and positive")
    w = 1.0 / (s * s)
    total = float(np.sum(w))
    return float(np.sum(w * t) / total), float(math.sqrt(1.0 / total))
