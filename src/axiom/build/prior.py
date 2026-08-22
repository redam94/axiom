"""``PriorBuilder``: a family and its hyperparameters, or moments, → ``core.Prior``.

Ported from the parent's ``builders/prior.py`` (ledger: PORT) with the
marketing-specific presets dropped. The moment-matching route goes through
``calibrate.lognormal_from_moments`` / ``calibrate.mean_sd_to_gamma`` so a
prior built from ``(mean, sd)`` is the same prior ``calibrate.derive_prior``
would produce from the same moments — one definition of "lognormal with
this mean and sd" in the codebase.

A hyperparameter may be the *name* of another parameter (a hierarchy), as
``core.Prior`` allows; ``PriorBuilder().normal(mu="alpha_mean",
sigma="alpha_sd")`` is the per-unit intercept's prior.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from axiom.build.base import BuildError, Fields
from axiom.calibrate import lognormal_from_moments, mean_sd_to_gamma
from axiom.core import PRIOR_HYPER, Prior, PriorFamily

__all__ = ["MomentFamily", "PriorBuilder"]

MomentFamily = Literal["lognormal", "gamma", "normal"]
Hyper = float | str


@dataclass(frozen=True)
class PriorBuilder:
    """Collect a family and hyperparameters; ``build()`` is a ``core.Prior``.

    Family shortcuts (``normal``, ``halfnormal``, ``lognormal``, ``gamma``,
    ``beta``, ``uniform``, ``fixed``) set the family and every
    hyperparameter at once; ``family`` + ``hyper`` set them separately;
    ``from_moments`` picks the family's hyperparameters to match a mean and
    standard deviation.
    """

    fields: Fields = Fields()

    # -- generic ------------------------------------------------------------------------

    def with_(self, **updates: Any) -> PriorBuilder:
        return replace(self, fields=self.fields.with_(**updates))

    def family(self, family: PriorFamily) -> PriorBuilder:
        if family not in PRIOR_HYPER:
            raise BuildError(f"unknown prior family {family!r}; known: {sorted(PRIOR_HYPER)}")
        return self.with_(family=family)

    def hyper(self, **hyper: Hyper) -> PriorBuilder:
        """Add hyperparameters (merged with any already set)."""
        current = dict(self.fields.get("hyper", {}))
        current.update({k: _hyper(k, v) for k, v in hyper.items()})
        return self.with_(hyper=current)

    # -- family shortcuts ---------------------------------------------------------------

    def normal(self, mu: Hyper = 0.0, sigma: Hyper = 1.0) -> PriorBuilder:
        return self.family("normal").with_(hyper={}).hyper(mu=mu, sigma=sigma)

    def halfnormal(self, sigma: Hyper = 1.0) -> PriorBuilder:
        return self.family("halfnormal").with_(hyper={}).hyper(sigma=sigma)

    def lognormal(self, mu: Hyper = 0.0, sigma: Hyper = 1.0) -> PriorBuilder:
        return self.family("lognormal").with_(hyper={}).hyper(mu=mu, sigma=sigma)

    def gamma(self, alpha: Hyper, beta: Hyper) -> PriorBuilder:
        """Shape ``alpha`` and *rate* ``beta`` (``core.Prior``'s convention)."""
        return self.family("gamma").with_(hyper={}).hyper(alpha=alpha, beta=beta)

    def beta(self, alpha: Hyper, beta: Hyper) -> PriorBuilder:
        return self.family("beta").with_(hyper={}).hyper(alpha=alpha, beta=beta)

    def uniform(self, low: Hyper, high: Hyper) -> PriorBuilder:
        return self.family("uniform").with_(hyper={}).hyper(low=low, high=high)

    def fixed(self, value: Hyper) -> PriorBuilder:
        return self.family("fixed").with_(hyper={}).hyper(value=value)

    # -- moments ------------------------------------------------------------------------

    @staticmethod
    def from_moments(mean: float, sd: float, family: MomentFamily = "lognormal") -> PriorBuilder:
        """The ``family`` prior with the given mean and standard deviation.

        ``lognormal`` and ``gamma`` need a positive mean (positive support);
        ``normal`` is ``N(mean, sd)``. Unknown families are a ``BuildError``.
        """
        if family == "lognormal":
            mu, sigma = lognormal_from_moments(mean, sd)
            return PriorBuilder().lognormal(mu=mu, sigma=sigma)
        if family == "gamma":
            shape, rate = mean_sd_to_gamma(mean, sd)
            return PriorBuilder().gamma(alpha=shape, beta=rate)
        if family == "normal":
            if not sd > 0:
                raise BuildError(f"sd must be positive, got {sd}")
            return PriorBuilder().normal(mu=mean, sigma=sd)
        raise BuildError(f"from_moments supports lognormal, gamma, normal; got {family!r}")

    # -- build --------------------------------------------------------------------------

    def build(self) -> Prior:
        self.fields.require("family", builder="PriorBuilder")
        family: PriorFamily = self.fields.get("family")
        hyper = dict(self.fields.get("hyper", {}))
        need = PRIOR_HYPER[family]
        missing = [h for h in need if h not in hyper]
        if missing:
            raise BuildError(f"PriorBuilder: {family} prior needs {list(need)}; missing {missing}")
        return Prior(family=family, hyper=hyper)


def _hyper(name: str, value: Hyper) -> Hyper:
    if isinstance(value, bool):
        raise BuildError(f"hyperparameter {name!r} cannot be bool")
    if isinstance(value, str):
        if not value.strip():
            raise BuildError(f"hyperparameter {name!r} names an empty parameter")
        return value
    return float(value)
