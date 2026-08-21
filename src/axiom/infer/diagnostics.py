"""MCMC convergence diagnostics in numpy: split-R̂, bulk/tail ESS, MCSE, and a report.

Implements the rank-normalized diagnostics of Vehtari, Gelman, Simpson,
Carpenter & Bürkner (2021), *Rank-normalization, folding, and localization:
an improved R̂ for assessing convergence of MCMC*, following the reference
implementation in arviz-stats step for step (Geyer's initial positive and
monotone sequences, the ``1 / log10(N)`` floor on the autocorrelation time).
Every function takes draws shaped ``(chain, draw)``; ``diagnose`` walks a
``Posterior`` and returns a ``ConvergenceReport`` — a ``Spec``, so it
serializes with the thresholds it was judged against.

Undefined values (a variable with zero variance, a mean over non-finite
draws) are ``nan`` from the functions and ``None`` in the report; a ``None``
never counts as converged, and neither does a single non-finite draw.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import field_validator
from scipy import stats as _st

from axiom.core.protocols import SupportsPosterior
from axiom.core.spec import Spec

__all__ = [
    "ConvergenceReport",
    "ConvergenceThresholds",
    "ParameterDiagnostics",
    "diagnose",
    "ess_bulk",
    "ess_tail",
    "mcse_mean",
    "split_rhat",
]

Array = npt.NDArray[np.float64]


# -- primitives ----------------------------------------------------------------------


def _as_chains(draws: npt.ArrayLike) -> Array:
    x = np.asarray(draws, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"draws must be (chain, draw) or (draw,); got shape {x.shape}")
    return x


def _split(x: Array) -> Array:
    """Split every chain in half: ``(c, n)`` -> ``(2c, n // 2)``; an odd last draw is dropped."""
    c, n = x.shape
    half = n // 2
    if half < 1:
        return x
    return np.concatenate([x[:, :half], x[:, half : 2 * half]], axis=0)


def _rank_normalize(x: Array) -> Array:
    """Ranks over all chains jointly, mapped through the normal quantile (Blom offsets)."""
    n = x.size
    ranks = _st.rankdata(x.ravel(), method="average").reshape(x.shape)
    return np.asarray(_st.norm.ppf((ranks - 0.375) / (n + 0.25)), dtype=float)


def _fold(x: Array) -> Array:
    return np.abs(x - np.median(x))


def _rhat_basic(x: Array) -> float:
    """Split-R̂ of Gelman & Rubin with the (n-1)/n pooled-variance correction."""
    c, n = x.shape
    if n < 2 or c < 2 or not np.all(np.isfinite(x)):
        return float("nan")
    within = float(np.mean(np.var(x, axis=1, ddof=1)))
    between = float(n * np.var(np.mean(x, axis=1), ddof=1))
    if within == 0.0:
        return float("nan")
    var_hat = (n - 1) / n * within + between / n
    return float(np.sqrt(var_hat / within))


def _autocovariance(x: Array) -> Array:
    """Per-chain biased autocovariance via FFT: ``(c, n)`` -> ``(c, n)``."""
    c, n = x.shape
    xc = x - x.mean(axis=1, keepdims=True)
    m = 1
    while m < 2 * n:
        m *= 2
    f = np.fft.rfft(xc, m, axis=1)
    acov = np.fft.irfft(f * np.conjugate(f), m, axis=1)[:, :n] / n
    return np.asarray(acov, dtype=float)


def _ess_basic(x: Array) -> float:
    """ESS of ``(c, n)`` chains: Vehtari et al. (2021) §3.2, as arviz-stats computes it.

    The autocorrelation time ``tau`` is Geyer's initial positive sequence
    made monotone, floored at ``1 / log10(c·n)`` (so ``ess <= N log10 N`` for
    antithetic chains). ``nan`` for constant draws.
    """
    c, n = x.shape
    if n < 4 or not np.all(np.isfinite(x)):
        return float("nan")
    if np.all(x == x.flat[0]):
        return float("nan")
    acov = _autocovariance(x)
    mean_var = float(np.mean(acov[:, 0])) * n / (n - 1.0)
    var_plus = mean_var * (n - 1.0) / n
    if c > 1:
        var_plus += float(np.var(x.mean(axis=1), ddof=1))
    if var_plus <= 0.0 or not np.isfinite(var_plus):
        return float("nan")
    rho = 1.0 - (mean_var - acov.mean(axis=0)) / var_plus  # rho[0] == 1 by construction
    rho_hat = np.zeros(n)
    rho_hat[0] = 1.0
    rho_hat[1] = rho[1]
    even, odd = 1.0, float(rho[1])
    # Geyer's initial positive sequence: keep pairs while their sum is positive
    t = 1
    while t < n - 3 and even + odd > 0.0:
        even, odd = float(rho[t + 1]), float(rho[t + 2])
        if even + odd >= 0.0:
            rho_hat[t + 1] = even
            rho_hat[t + 2] = odd
        t += 2
    max_t = t - 2
    if even > 0.0:
        rho_hat[max_t + 1] = even
    # Geyer's initial monotone sequence: no pair sum may exceed the one before it
    t = 1
    while t <= max_t - 2:
        if rho_hat[t + 1] + rho_hat[t + 2] > rho_hat[t - 1] + rho_hat[t]:
            rho_hat[t + 1] = (rho_hat[t - 1] + rho_hat[t]) / 2.0
            rho_hat[t + 2] = rho_hat[t + 1]
        t += 2
    total = c * n
    tau = -1.0 + 2.0 * float(np.sum(rho_hat[: max_t + 1])) + float(rho_hat[max_t + 1])
    tau = max(tau, 1.0 / np.log10(total))
    if not np.all(np.isfinite(rho_hat)):
        return float("nan")
    return float(total / tau)


# -- public functions -----------------------------------------------------------------


def split_rhat(draws: npt.ArrayLike) -> float:
    """Rank-normalized split-R̂: the max of the bulk and the folded (tail) versions.

    Converges to 1 for mixed chains; > 1.01 is the usual alarm. ``nan`` when
    the draws have no variance.
    """
    x = _split(_as_chains(draws))
    bulk = _rhat_basic(_rank_normalize(x))
    tail = _rhat_basic(_rank_normalize(_fold(x)))
    if np.isnan(bulk) or np.isnan(tail):
        return float("nan")
    return max(bulk, tail)


def ess_bulk(draws: npt.ArrayLike) -> float:
    """Bulk ESS: ESS of the rank-normalized split chains."""
    x = _split(_as_chains(draws))
    return _ess_basic(_rank_normalize(x))


def ess_tail(draws: npt.ArrayLike) -> float:
    """Tail ESS: the smaller of the ESS of the 5% and 95% quantile indicators."""
    x = _as_chains(draws)
    q05, q95 = np.quantile(x, [0.05, 0.95])
    lo = _ess_basic(_split((x <= q05).astype(float)))
    hi = _ess_basic(_split((x <= q95).astype(float)))
    return float(min(lo, hi))


def mcse_mean(draws: npt.ArrayLike) -> float:
    """Monte-Carlo standard error of the posterior mean, ``sd / sqrt(ess)`` on the raw draws."""
    x = _as_chains(draws)
    ess = _ess_basic(_split(x))
    if np.isnan(ess):
        return float("nan")
    return float(np.std(x, ddof=1) / np.sqrt(ess))


# -- report --------------------------------------------------------------------------


class ConvergenceThresholds(Spec):
    """What ``converged`` means. Defaults follow Vehtari et al. (2021)."""

    rhat_max: float = 1.01
    ess_min: float = 400.0
    divergences_max: int = 0

    @field_validator("rhat_max")
    @classmethod
    def _rhat(cls, v: float) -> float:
        if v < 1.0:
            raise ValueError("rhat_max must be at least 1")
        return v

    @field_validator("ess_min")
    @classmethod
    def _ess(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("ess_min must be positive")
        return v

    @field_validator("divergences_max")
    @classmethod
    def _div(cls, v: int) -> int:
        if v < 0:
            raise ValueError("divergences_max must be non-negative")
        return v


class ParameterDiagnostics(Spec):
    """One row of the report: a scalar variable or one element of a vector one.

    ``n_nonfinite`` counts ``nan`` / ``inf`` draws; a row with any fails
    whatever its R̂ says. ``mean`` and ``sd`` are ``None`` when undefined
    (non-finite draws, or a single draw for ``sd``) — never ``0.0``.
    """

    name: str
    rhat: float | None
    ess_bulk: float | None
    ess_tail: float | None
    mcse_mean: float | None
    mean: float | None
    sd: float | None
    n_nonfinite: int = 0

    @field_validator("n_nonfinite")
    @classmethod
    def _non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("n_nonfinite must be non-negative")
        return v

    def passes(self, thresholds: ConvergenceThresholds) -> bool:
        return (
            self.n_nonfinite == 0
            and self.rhat is not None
            and self.ess_bulk is not None
            and self.ess_tail is not None
            and self.mcse_mean is not None
            and self.rhat < thresholds.rhat_max
            and self.ess_bulk > thresholds.ess_min
            and self.ess_tail > thresholds.ess_min
        )


class ConvergenceReport(Spec):
    """Per-parameter diagnostics, the divergence count, and a verdict with its thresholds.

    ``nonfinite_draw_frac`` is the larger of the fraction of joint draws with
    any non-finite component seen in the posterior and the producer's own
    ``provenance["nonfinite_draw_frac"]``; anything above zero fails.
    """

    rows: tuple[ParameterDiagnostics, ...]
    divergences: int
    thresholds: ConvergenceThresholds
    converged: bool
    n_chains: int
    n_draws: int
    nonfinite_draw_frac: float = 0.0

    @property
    def failing(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.rows if not r.passes(self.thresholds))

    def row(self, name: str) -> ParameterDiagnostics:
        for r in self.rows:
            if r.name == name:
                return r
        raise KeyError(f"no row {name!r}; have {[r.name for r in self.rows]}")

    def to_frame(self) -> pd.DataFrame:
        """The table as a ``DataFrame`` indexed by name (``None`` becomes ``NaN``)."""
        return pd.DataFrame(
            [r.model_dump(exclude={"name"}) for r in self.rows],
            index=pd.Index([r.name for r in self.rows], name="parameter"),
        )


def _finite_or_none(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None


def _rows_for(name: str, a: Array) -> list[ParameterDiagnostics]:
    """``a`` is ``(chain, draw, *shape)``; one row per trailing index."""
    c, n = a.shape[:2]
    flat = a.reshape(c, n, -1)
    labels = (
        [name]
        if a.ndim == 2
        else [f"{name}[{','.join(str(i) for i in idx)}]" for idx in np.ndindex(*a.shape[2:])]
    )
    rows = []
    for k, label in enumerate(labels):
        x = flat[:, :, k]
        n_nonfinite = int(np.sum(~np.isfinite(x)))
        finite = n_nonfinite == 0
        rows.append(
            ParameterDiagnostics(
                name=label,
                rhat=_finite_or_none(split_rhat(x)),
                ess_bulk=_finite_or_none(ess_bulk(x)),
                ess_tail=_finite_or_none(ess_tail(x)),
                mcse_mean=_finite_or_none(mcse_mean(x)),
                mean=float(np.mean(x)) if finite else None,
                sd=float(np.std(x, ddof=1)) if finite and x.size > 1 else None,
                n_nonfinite=n_nonfinite,
            )
        )
    return rows


def diagnose(
    posterior: SupportsPosterior,
    *,
    divergences: int | None = None,
    rhat_max: float = 1.01,
    ess_min: float = 400.0,
    divergences_max: int = 0,
) -> ConvergenceReport:
    """Diagnose every variable of ``posterior`` and judge it against the thresholds.

    ``divergences`` defaults to ``posterior.provenance["divergences"]`` when
    the posterior carries one, else ``0``. The verdict also fails on any
    non-finite draw — seen in the draws or declared by the producer's
    ``provenance["nonfinite_draw_frac"]``.
    """
    thresholds = ConvergenceThresholds(
        rhat_max=rhat_max, ess_min=ess_min, divergences_max=divergences_max
    )
    prov: Any = getattr(posterior, "provenance", {})
    if not isinstance(prov, dict):
        prov = {}
    if divergences is None:
        divergences = int(prov.get("divergences", 0))
    declared_frac = float(prov.get("nonfinite_draw_frac", 0.0) or 0.0)
    rows: list[ParameterDiagnostics] = []
    n_chains = n_draws = 0
    bad: Array | None = None
    for name in sorted(posterior.names()):
        a = np.asarray(posterior.draws(name), dtype=float)
        n_chains, n_draws = int(a.shape[0]), int(a.shape[1])
        joint_bad = ~np.all(np.isfinite(a.reshape(n_chains * n_draws, -1)), axis=1)
        bad = joint_bad if bad is None else (bad | joint_bad)
        rows.extend(_rows_for(name, a))
    seen_frac = float(np.mean(bad)) if bad is not None and bad.size else 0.0
    nonfinite_frac = max(seen_frac, declared_frac)
    converged = (
        all(r.passes(thresholds) for r in rows)
        and divergences <= thresholds.divergences_max
        and nonfinite_frac == 0.0
    )
    return ConvergenceReport(
        rows=tuple(rows),
        divergences=divergences,
        thresholds=thresholds,
        converged=converged,
        n_chains=n_chains,
        n_draws=n_draws,
        nonfinite_draw_frac=nonfinite_frac,
    )
