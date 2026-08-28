"""Two variance components without a sampler: the three-level REML.

``meta.pool`` with ``effect_key="nested"`` separates the between-study spread
inside a party from the spread between parties, and needs a sampler to do it
with a free ``tau_party`` — the party level is non-centered, so ``laplace``
lands on a multiplicative ridge and overstates it threefold
(``docs/notes/0029-the-party-a-study-came-from.md`` §D29.3).

That is the right refusal and it leaves a real gap. A house that wants the two
variance components and a Wald interval on the family mean, with nothing
installed beyond the core four dependencies, had nowhere to go —
``meta.classical`` stopped at the two-level DerSimonian-Laird / Paule-Mandel /
REML family.

This module is the sampler-free route. Under the same normal hierarchy the
marginal covariance of the records is

    Sigma = diag(se²) + tau_study²·(same study) + tau_party²·(same party)

which is block-diagonal by party, small, and perfectly ordinary to invert. The
restricted likelihood is profiled over the two variance components by direct
numerical optimization; the family mean and its standard error fall out of
generalized least squares at the optimum, and the fit is exact rather than
approximate.

**What it is not.** It gives point estimates of the variance components and no
uncertainty about them, which is exactly the classical/Bayesian trade the rest
of ``meta`` already makes: ``meta.classical.random_effects`` reports a ``tau2``
and no interval on it either. When the number of parties is small — and it
usually is — that omission matters more than it does at the study level, and
``meta.pool`` with a sampler is the answer that quantifies it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from pydantic import Field, model_validator
from scipy import optimize, stats

from axiom.core import Interval, LedgerLine, Spec

__all__ = [
    "ThreeLevel",
    "three_level",
    "variance_shares",
]

Array = npt.NDArray[np.float64]
Index = npt.NDArray[np.int64]


class ThreeLevel(Spec):
    """A family mean whose uncertainty reflects the number of parties, not of studies.

    ``tau_study`` is the between-study spread *within* a party and
    ``tau_party`` the spread between parties. ``se`` is the standard error of
    ``estimate`` under the fitted covariance — the number the two-level pool
    understates when the studies came from a handful of parties.
    """

    estimate: float
    se: float = Field(gt=0)
    interval: Interval
    tau_study: float = Field(ge=0)
    tau_party: float = Field(ge=0)
    k: int = Field(ge=2)
    n_parties: int = Field(ge=2)
    mass: float = Field(gt=0, lt=1)
    log_likelihood: float
    converged: bool = True
    detail: dict[str, str] = {}

    @model_validator(mode="after")
    def _consistent(self) -> ThreeLevel:
        if self.interval.mass != self.mass:
            raise ValueError("interval mass must equal the estimate's mass")
        if self.interval.definition != "wald":
            raise ValueError(f"a three-level interval is wald, not {self.interval.definition!r}")
        if self.n_parties > self.k:
            raise ValueError("there cannot be more parties than records")
        return self

    @property
    def tau2_study(self) -> float:
        return self.tau_study**2

    @property
    def tau2_party(self) -> float:
        return self.tau_party**2

    def shares(self, sampling_variance: float) -> tuple[float, float, float]:
        """How the total variance splits between sampling, study and party.

        The three-level analogue of ``I²``: pass the typical within-study
        sampling variance (``variance_shares`` computes it from the standard
        errors) and read off what share of the spread lives at each level.
        """
        return variance_shares(sampling_variance, self.tau2_study, self.tau2_party)

    def ledger_line(self) -> LedgerLine:
        return LedgerLine(
            kind="three_level_pool",
            statement=(
                f"{self.k} record(s) over {self.n_parties} parties: mean "
                f"{self.estimate:.4g} {self.interval.text()}, tau within a party "
                f"{self.tau_study:.4g}, tau between parties {self.tau_party:.4g}"
            ),
            detail={
                "k": str(self.k),
                "n_parties": str(self.n_parties),
                "tau_study": f"{self.tau_study:.6g}",
                "tau_party": f"{self.tau_party:.6g}",
                "se": f"{self.se:.6g}",
                "log_likelihood": f"{self.log_likelihood:.6g}",
                "converged": str(self.converged),
                **self.detail,
            },
        )

    def summary(self) -> str:
        return "\n".join(
            [
                f"{self.k} records over {self.n_parties} parties",
                f"  mean          {self.estimate:+.4g} {self.interval.text()}",
                f"  se            {self.se:.4g}",
                f"  tau within    {self.tau_study:.4g}",
                f"  tau between   {self.tau_party:.4g}",
            ]
        )


def variance_shares(
    sampling: float, tau2_study: float, tau2_party: float
) -> tuple[float, float, float]:
    """``(sampling, study, party)`` shares of the total variance, summing to one.

    Raises when the total is not positive: with no variance anywhere there is
    nothing to apportion, and returning thirds would be an invention.
    """
    for name, value in (
        ("sampling", sampling),
        ("tau2_study", tau2_study),
        ("tau2_party", tau2_party),
    ):
        if not (math.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name} must be finite and non-negative, got {value}")
    total = sampling + tau2_study + tau2_party
    if total <= 0.0:
        raise ValueError("the total variance is zero; there are no shares to report")
    return (sampling / total, tau2_study / total, tau2_party / total)


def _blocks(party: Index) -> list[Index]:
    return [np.flatnonzero(party == p).astype(np.int64) for p in sorted(set(party.tolist()))]


def _restricted_log_likelihood(
    y: Array, v: Array, study: Index, party: Index, tau2_study: float, tau2_party: float
) -> tuple[float, float, float]:
    """``(reml log-likelihood, mu, se)`` at the given variance components."""
    log_det = 0.0
    xtx = 0.0  # 1' Sigma^-1 1
    xty = 0.0  # 1' Sigma^-1 y
    yty = 0.0  # y' Sigma^-1 y
    for members in _blocks(party):
        block = np.diag(v[members])
        same_study = study[members][:, None] == study[members][None, :]
        block = block + tau2_study * same_study + tau2_party
        sign, value = np.linalg.slogdet(block)
        if sign <= 0:
            return -math.inf, math.nan, math.nan
        log_det += float(value)
        solved = np.linalg.solve(block, np.column_stack([np.ones(members.size), y[members]]))
        xtx += float(np.ones(members.size) @ solved[:, 0])
        xty += float(np.ones(members.size) @ solved[:, 1])
        yty += float(y[members] @ solved[:, 1])
    if xtx <= 0.0:
        return -math.inf, math.nan, math.nan
    mu = xty / xtx
    quadratic = yty - xtx * mu * mu
    reml = -0.5 * (log_det + math.log(xtx) + quadratic)
    return reml, mu, math.sqrt(1.0 / xtx)


def three_level(
    y: npt.ArrayLike,
    se: npt.ArrayLike,
    study: Sequence[int] | npt.ArrayLike,
    party: Sequence[int] | npt.ArrayLike,
    *,
    mass: float = 0.95,
    max_tau: float | None = None,
) -> ThreeLevel:
    """REML over ``(tau_study², tau_party²)``, with the mean by GLS at the optimum.

    ``study`` and ``party`` are integer group labels per record — several
    records may share a study (two reads of one effect), and several studies a
    party. Needs at least two parties and at least one party with two studies,
    or the two variance components are the same quantity and the two-level
    ``meta.classical.random_effects`` is the honest fit.
    """
    ya = np.asarray(y, dtype=np.float64).ravel()
    sa = np.asarray(se, dtype=np.float64).ravel()
    st = np.asarray(study, dtype=np.int64).ravel()
    pa = np.asarray(party, dtype=np.int64).ravel()
    if not (ya.size == sa.size == st.size == pa.size):
        raise ValueError(
            f"y, se, study and party must be the same length, got "
            f"{ya.size}, {sa.size}, {st.size}, {pa.size}"
        )
    if ya.size < 2:
        raise ValueError("a three-level fit needs at least two records")
    if not np.all(np.isfinite(ya)) or not np.all(np.isfinite(sa)):
        raise ValueError("y and se must be finite")
    if np.any(sa <= 0):
        raise ValueError("standard errors must be positive")
    parties = sorted(set(pa.tolist()))
    if len(parties) < 2:
        raise ValueError(
            "a three-level fit needs at least two parties; with one, the two levels are one "
            "and meta.classical.random_effects is the honest fit"
        )
    studies_per_party = [len({int(s) for s in st[pa == p]}) for p in parties]
    if max(studies_per_party) < 2:
        raise ValueError(
            "a three-level fit needs a party with at least two studies; with one each, the "
            "between-study and between-party spreads are the same quantity"
        )
    if not 0.0 < mass < 1.0:
        raise ValueError(f"mass must be in (0, 1), got {mass}")

    spread = float(np.var(ya, ddof=1)) if ya.size > 1 else 1.0
    ceiling = max_tau if max_tau is not None else max(10.0 * math.sqrt(max(spread, 1e-12)), 1.0)
    v = sa**2

    def negative(log_pair: npt.NDArray[np.float64]) -> float:
        tau2s, tau2p = np.exp(log_pair)
        reml, _, _ = _restricted_log_likelihood(ya, v, st, pa, float(tau2s), float(tau2p))
        return -reml if math.isfinite(reml) else 1e12

    floor = math.log(1e-12)
    roof = math.log(ceiling**2)
    best = optimize.minimize(
        negative,
        x0=np.asarray([math.log(max(spread, 1e-8) / 2.0), math.log(max(spread, 1e-8) / 2.0)]),
        method="Nelder-Mead",
        bounds=[(floor, roof), (floor, roof)],
        options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000},
    )
    tau2_study, tau2_party = (float(x) for x in np.exp(best.x))
    tau2_study = 0.0 if tau2_study <= 1e-11 else tau2_study
    tau2_party = 0.0 if tau2_party <= 1e-11 else tau2_party
    reml, mu, s = _restricted_log_likelihood(ya, v, st, pa, tau2_study, tau2_party)
    z = float(stats.norm.isf((1.0 - mass) / 2.0))
    return ThreeLevel(
        estimate=mu,
        se=s,
        interval=Interval(lower=mu - z * s, upper=mu + z * s, definition="wald", mass=mass),
        tau_study=math.sqrt(tau2_study),
        tau_party=math.sqrt(tau2_party),
        k=int(ya.size),
        n_parties=len(parties),
        mass=mass,
        log_likelihood=reml,
        converged=bool(best.success),
        detail={
            "method": "reml, profiled over both variance components by Nelder-Mead",
            "studies_per_party": ", ".join(str(c) for c in studies_per_party),
        },
    )
