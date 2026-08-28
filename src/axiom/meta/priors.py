"""The handoff: a pooled posterior becomes a ``core.Prior`` for ``calibrate`` or ``surface``.

This closes the loop the parent's ``benchmarks/priors.py`` closed (ledger
row ``meta/priors.py``, PORT): evidence pooled across studies re-enters the
next study as its prior. Two targets:

* ``target="mu"`` — the prior on the *family mean*: ``N(E[mu], sd[mu])``.
  Right when the new study asks "what is the typical effect".
* ``target="predictive"`` — the prior on a *new study's effect*, which
  includes the between-study spread: ``N(E[mu], sqrt(E[tau²] + sd[mu]²))``
  with ``E[tau²] = E[tau]² + sd[tau]²`` (exactly ``tau²`` when ``tau`` was
  fixed). Right when the new study is another draw from the family — the
  usual case, and the one that is wider.

A pool with a party level (``effect_key="nested"``, note 0029) can tell two
more questions apart, and they have different answers:

* ``target="new_party"`` — the next study is for a party this corpus has never
  seen: ``N(E[mu], sqrt(E[tau_party²] + E[tau²] + sd[mu]²))``. It must cross
  both levels, so it is the widest of the four.
* ``target="same_party"`` — the next study is another study for a party already
  in the corpus, named by ``party``: ``N(E[alpha_p], sqrt(E[tau²] +
  sd[alpha_p]²))``. It crosses one level, and it is much tighter, which is the
  whole reason the party level was worth fitting.

Asking for either against a two-level pool raises rather than quietly returning
the two-level answer: a prior that says "for this party" when nothing in the
pool knew about parties is the failure the targets exist to prevent.

``family="normal"`` returns the prior on the pooled scale. ``family=
"lognormal"`` returns a prior on the *positive* quantity: when the pool was
on the log scale (``PoolResult.scale == "log"``) the normal on the log
scale *is* the lognormal's parametrization; when the pool was on the
natural scale the lognormal is moment-matched (``sigma² = log(1 + s²/m²)``,
``mu = log m − sigma²/2``) and requires a positive mean.

``meta`` returns core objects only (``Prior``, ``LedgerLine``); it never
imports ``calibrate``. The ledger line records the pool's model hash, the
target, the family, and the numbers, so a prior in a later spec can be
traced to the corpus that produced it.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from axiom.core import LedgerLine, Prior, format_measured
from axiom.meta.pool import PoolResult

__all__ = ["PriorTarget", "prior_from_pool"]

PriorTarget = Literal["mu", "predictive", "new_party", "same_party"]
PriorFamilyOut = Literal["normal", "lognormal"]


def _expected_square(mean: float, sd: float) -> float:
    return mean**2 + sd**2


def _party_summary(result: PoolResult, party: str):  # type: ignore[no-untyped-def]
    for summary in result.alphas:
        if summary.name.endswith(f"[{party}]"):
            return summary
    known = [s.name.split("[")[1][:-1] for s in result.alphas if "[" in s.name]
    raise KeyError(f"no party {party!r} in this pool; it has {known}")


def _location_and_spread(
    result: PoolResult, target: PriorTarget, party: str = ""
) -> tuple[float, float]:
    m = result.mu.mean
    if target == "mu":
        return m, result.mu.sd
    tau2 = _expected_square(result.tau.mean, result.tau.sd)
    if target == "predictive":
        return m, float(np.sqrt(tau2 + result.mu.sd**2))
    if result.tau_party is None:
        raise ValueError(
            f"target={target!r} needs a pool with a party level "
            "(PoolSpec(effect_key='nested')); this pool has one variance component, so it "
            "cannot tell a new party from another study of an old one"
        )
    if target == "new_party":
        tau_party2 = _expected_square(result.tau_party.mean, result.tau_party.sd)
        return m, float(np.sqrt(tau_party2 + tau2 + result.mu.sd**2))
    if not party:
        raise ValueError("target='same_party' needs the party it is for")
    alpha = _party_summary(result, party)
    return alpha.mean, float(np.sqrt(tau2 + alpha.sd**2))


def prior_from_pool(
    result: PoolResult,
    *,
    target: PriorTarget = "predictive",
    family: PriorFamilyOut = "normal",
    party: str = "",
) -> tuple[Prior, LedgerLine]:
    """Turn a ``PoolResult`` into a ``core.Prior`` plus the ledger line that records it.

    ``party`` names the party ``target="same_party"`` is for and is ignored
    otherwise. ``ValueError`` when the spread is not positive (a degenerate
    pool), a natural-scale pool with a non-positive mean is asked for a
    lognormal, or a party-level target is asked of a two-level pool.
    """
    if target not in ("mu", "predictive", "new_party", "same_party"):
        raise ValueError(
            "target must be 'mu', 'predictive', 'new_party' or 'same_party', " f"got {target!r}"
        )
    if family not in ("normal", "lognormal"):
        raise ValueError(f"family must be 'normal' or 'lognormal', got {family!r}")
    m, s = _location_and_spread(result, target, party)
    if not (np.isfinite(m) and np.isfinite(s) and s > 0.0):
        raise ValueError(f"pool gives a degenerate prior: mean {m}, sd {s}")
    detail = {
        "pool_model_hash": result.model_hash,
        "family_pooled": result.family,
        "quantity": result.quantity,
        "pool_scale": result.scale,
        "target": target,
        "prior_family": family,
        "k": str(result.k),
        "n_effects": str(result.n_effects),
        "mu_mean": repr(result.mu.mean),
        "mu_sd": repr(result.mu.sd),
        "tau_mean": repr(result.tau.mean),
        "tau_sd": repr(result.tau.sd),
        "backend": result.backend,
    }
    if party:
        detail["party"] = party
    if result.tau_party is not None:
        detail["tau_party_mean"] = repr(result.tau_party.mean)
        detail["n_parties"] = str(result.n_parties)
    if family == "normal":
        prior = Prior(family="normal", hyper={"mu": float(m), "sigma": float(s)})
        statement = (
            f"prior on {target} of {result.quantity!r} in family {result.family!r}: "
            f"N({format_measured(m, s)}, {format_measured(s, s)}) "
            f"from a pool of {result.k} records"
        )
    elif result.scale == "log":
        prior = Prior(family="lognormal", hyper={"mu": float(m), "sigma": float(s)})
        statement = (
            f"lognormal prior on {target} of {result.quantity!r} in family {result.family!r}: "
            f"LogNormal({format_measured(m, s)}, {format_measured(s, s)}) "
            f"— the pool was on the log scale"
        )
        detail["lognormal"] = "log-scale pool: normal on the log scale is the lognormal"
    else:
        if m <= 0.0:
            raise ValueError(
                f"a lognormal prior needs a positive pooled mean; got {m} on the natural scale"
            )
        sigma2 = float(np.log1p((s / m) ** 2))
        log_mu = float(np.log(m) - 0.5 * sigma2)
        prior = Prior(family="lognormal", hyper={"mu": log_mu, "sigma": float(np.sqrt(sigma2))})
        statement = (
            f"lognormal prior on {target} of {result.quantity!r} in family {result.family!r}: "
            f"moment-matched to mean {format_measured(m, s)}, sd {format_measured(s, s)}"
        )
        detail["lognormal"] = "natural-scale pool: moment-matched (mean and sd preserved)"
    line = LedgerLine(
        kind="prior_from_pool",
        statement=statement,
        detail=detail,
        source=result.model_hash,
    )
    return prior, line
