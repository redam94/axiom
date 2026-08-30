"""How a likelihood family weights each row of the information matrix.

The design math asks one question of a likelihood: how much does an
observation at this mean tell me about the parameters? For any exponential
family the answer is a single diagonal weight. With ``mu_i`` the mean, the
score is

    dl/dtheta = sum_i (y_i - mu_i) / (phi V(mu_i)) . dmu_i/dtheta

and since ``Var[y_i] = phi V(mu_i)``,

    FI = sum_i (dmu_i/dtheta)(dmu_i/dtheta)' / (phi V(mu_i)) = J' W J

with ``w_i = 1 / (phi V(mu_i))``. Two things about that are worth saying
out loud, because they are what make this cheap here:

* **The link function never appears.** axiom differentiates the *mean* with
  respect to theta, not a linear predictor with respect to a coefficient
  vector, so any link is already inside ``J`` — which ``surface.linearize``
  and ``jax.jacfwd`` compute exactly (rule 3). None of the usual GLM
  apparatus — inverse links, deviance, IRLS — is needed to get the design
  math right.
* **The Gaussian case is not special.** ``FI = J'J / noise_sd**2`` is
  ``V = 1, phi = noise_sd**2``. Everything that already reads an
  information matrix — D-optimality, the ridge, expected posterior sd,
  point exchange — reads ``J' W J`` without knowing which family produced
  it.

The weights depend on ``mu``, hence on ``theta``: a design chosen this way
is *locally* optimal. That is not new. ``structural``'s module docstring
already says the information on a nonlinear surface is local (review B11)
and prescribes averaging over prior draws, because a Hill kernel makes
``J`` depend on ``theta`` whatever the likelihood is. Non-Gaussian
families widen the dependence; they do not introduce it.

``Weighting`` holds no arrays, so it travels with a result as provenance.
Its variance functions are :func:`axiom.core.variance_weight` — the same
call ``core.information_weight`` makes for a fitted model, so there is one
place where a family's variance is written down.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from axiom.core import LikelihoodFamily, ModelSpec, NonEmptyStr, Spec, variance_weight

__all__ = ["Weighting"]

Array = npt.NDArray[np.float64]


class Weighting(Spec):
    """The likelihood's contribution to ``FI = J' W J``, without any data in it.

    ``scale`` is the family's dispersion parameter *value* — the standard
    deviation for ``normal`` and ``student_t``, the log-scale sd for
    ``lognormal``, the coefficient of variation for ``gamma`` — and is
    unused by ``poisson`` and ``binomial``. ``trials`` names the data
    column holding the binomial denominators; without it a binomial row is
    a single Bernoulli trial.

    This is a spec and not a plain weight vector on purpose: the weights
    are a function of the mean, so they differ at every candidate design a
    search visits. What is constant, and what a result should carry, is the
    *rule* for computing them.
    """

    family: LikelihoodFamily = "normal"
    scale: float | None = None
    df: float | None = None
    trials: NonEmptyStr | None = None

    @classmethod
    def from_model(
        cls, model: ModelSpec, theta: Mapping[str, npt.ArrayLike] | None = None
    ) -> Weighting:
        """The weighting a fitted model implies, reading its scale out of ``theta``.

        ``ValueError`` when the family needs a scale and ``theta`` does not
        carry one — a design weighted by a guessed dispersion is a wrong
        number with no warning attached, which rule 4 does not allow.
        """
        lik = model.likelihood
        scale: float | None = None
        if lik.scale:
            if theta is None or lik.scale not in theta:
                raise ValueError(
                    f"a {lik.family} weighting needs the scale parameter {lik.scale!r}; "
                    "pass theta containing it"
                )
            scale = float(np.mean(np.asarray(theta[lik.scale], dtype=float)))
        elif lik.scale_expr is not None:
            raise ValueError(
                "a scale_expr varies by row, so it is not a Weighting; build the weights "
                "directly with core.information_weight and pass them as an array"
            )
        return cls(family=lik.family, scale=scale, df=lik.df, trials=lik.trials)

    @property
    def label(self) -> str:
        """A short description of the rule, for provenance and the ``__add__`` guard."""
        bits: list[str] = [self.family]
        if self.scale is not None:
            bits.append(f"scale={self.scale:.6g}")
        if self.df is not None:
            bits.append(f"df={self.df:.6g}")
        if self.trials:
            bits.append(f"trials={self.trials}")
        return ", ".join(bits)

    @property
    def is_unit(self) -> bool:
        """True when ``W`` is the identity: a normal likelihood of scale 1."""
        return self.family == "normal" and (self.scale is None or self.scale == 1.0)

    def at(self, mu: npt.ArrayLike, data: Mapping[str, Any] | None = None) -> Array:
        """``w`` at a mean, broadcastable against it. ``data`` supplies binomial trials."""
        trials = None
        if self.family == "binomial" and self.trials:
            if data is None or self.trials not in data:
                raise ValueError(
                    f"a binomial weighting needs the trials column {self.trials!r} in the "
                    "design's data"
                )
            trials = np.asarray(data[self.trials], dtype=float)
        return variance_weight(self.family, mu, scale=self.scale, df=self.df, trials=trials)
