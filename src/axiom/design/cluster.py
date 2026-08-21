"""Cluster-randomized designs: the design effect, matched-pair assignment, and holdout trade-off.

The parent called these geographic designs. Nothing here is geographic: a
cluster is any group of individuals randomized together — a region, a
clinic, a classroom — and is represented by a ``core.Unit`` of
``kind="cluster"``.

Randomizing ``k`` clusters of ``m`` individuals each is worth fewer than
``k · m`` independent observations when outcomes within a cluster are
correlated. With intra-cluster correlation ``icc`` the inflation is the
*design effect*

    DE = 1 + (m − 1) · icc,

so a cluster design has the power of an individual design with ``n = k · m``
and ``sd · sqrt(DE)`` — equivalently, the cluster means have variance
``sd² · DE / m``. Every function here reduces to ``design.power`` through that
substitution; there is no second power formula.

Matched pairs: clusters are paired by pre-period similarity and one member of
each pair is randomized to treatment. Pairing removes between-pair variance
from the comparison; the balance numbers on the ``Assignment`` say how much
was removed and are not a proof of exchangeability.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import model_validator

from axiom.core import D, NonEmptyStr, Spec, Unit, Unsupported
from axiom.design.power import MDE, PowerResult, SampleSize, mde, power, sample_size

__all__ = [
    "Assignment",
    "ClusterDesign",
    "HoldoutTradeoff",
    "MatchMetric",
    "cluster_mde",
    "cluster_power",
    "clusters_needed",
    "design_effect",
    "effective_sample_size",
    "holdout_tradeoff",
    "match_clusters",
]

Array = npt.NDArray[np.float64]
MatchMetric = Literal["level", "trajectory"]


def _check_icc(icc: float) -> None:
    if not 0.0 <= icc < 1.0:
        raise ValueError(f"icc must be in [0, 1), got {icc}")


def _check_cluster_size(cluster_size: int) -> None:
    if cluster_size < 1:
        raise ValueError(f"cluster_size must be at least 1, got {cluster_size}")


# -- design effect ---------------------------------------------------------------------


def design_effect(cluster_size: int, icc: float) -> float:
    """``1 + (m − 1) · icc`` — variance inflation of a cluster-randomized mean."""
    _check_cluster_size(cluster_size)
    _check_icc(icc)
    return 1.0 + (cluster_size - 1) * icc


def effective_sample_size(n_clusters: int, cluster_size: int, icc: float) -> float:
    """``k · m / DE`` — the number of independent observations the design is worth."""
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive")
    return n_clusters * cluster_size / design_effect(cluster_size, icc)


class ClusterDesign(Spec):
    """A cluster-randomized design and its variance inflation.

    ``unit`` is the cluster entity (``kind="cluster"``); ``allocation`` is the
    treated share of clusters.
    """

    unit: Unit
    n_clusters: int
    cluster_size: int
    icc: float
    allocation: float = 0.5

    @model_validator(mode="after")
    def _valid(self) -> ClusterDesign:
        if self.unit.kind != "cluster":
            raise ValueError(f"unit {self.unit.name!r} must have kind='cluster'")
        if self.n_clusters < 2:
            raise ValueError("n_clusters must be at least 2")
        _check_cluster_size(self.cluster_size)
        _check_icc(self.icc)
        if not 0.0 < self.allocation < 1.0:
            raise ValueError("allocation must be in (0, 1)")
        return self

    @property
    def design_effect(self) -> float:
        return design_effect(self.cluster_size, self.icc)

    @property
    def n_individuals(self) -> int:
        return self.n_clusters * self.cluster_size


def _default_unit() -> Unit:
    return Unit(name="cluster", dimension=D.entity, kind="cluster")


def cluster_power(
    design: ClusterDesign,
    effect: float,
    sd: float,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> PowerResult:
    """Power to detect ``effect`` on the individual-level scale under ``design``.

    The result's ``n`` is the number of individuals and ``sd`` the inflated
    ``sd · sqrt(DE)`` that was used; the design's own numbers are on ``design``.
    """
    return power(
        design.n_individuals,
        effect,
        sd * math.sqrt(design.design_effect),
        alpha=alpha,
        two_sided=two_sided,
        allocation=design.allocation,
    )


def cluster_mde(
    design: ClusterDesign,
    sd: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> MDE:
    """Minimum detectable effect under ``design``; see ``cluster_power`` for the scale."""
    return mde(
        design.n_individuals,
        sd * math.sqrt(design.design_effect),
        alpha=alpha,
        power=power,
        two_sided=two_sided,
        allocation=design.allocation,
    )


def clusters_needed(
    effect: float,
    sd: float,
    cluster_size: int,
    icc: float,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
    allocation: float = 0.5,
) -> SampleSize | Unsupported:
    """Smallest number of clusters of size ``cluster_size`` reaching ``power`` for ``effect``.

    ``n`` on the result is the number of *clusters*; ``n_treated`` /
    ``n_control`` are clusters too. The search is over whole clusters, so the
    achieved power is computed with integer cluster arms.
    """
    _check_cluster_size(cluster_size)
    _check_icc(icc)
    de = design_effect(cluster_size, icc)
    # Cluster means have sd · sqrt(DE / m); the cluster-level design is a difference in
    # means over clusters with that sd.
    sd_cluster = sd * math.sqrt(de / cluster_size)
    out = sample_size(
        effect, sd_cluster, alpha=alpha, power=power, two_sided=two_sided, allocation=allocation
    )
    if isinstance(out, Unsupported):
        return out
    return out.model_copy(update={"sd": sd_cluster})


# -- matched-pair assignment -----------------------------------------------------------


class Assignment(Spec):
    """A matched-pair treatment assignment of clusters.

    ``pairs`` are index pairs ``(i, j)`` matched on pre-period similarity;
    within each pair the treated member was chosen at random with ``seed``.
    ``unpaired`` holds the odd cluster out (assigned to control). The balance
    numbers compare treated and control pre-period means: ``pre_mean_treated``,
    ``pre_mean_control`` and their standardized difference ``pre_smd``
    (difference over the pooled pre-period sd across clusters).
    """

    unit: Unit
    labels: tuple[NonEmptyStr, ...]
    treated: tuple[int, ...]
    control: tuple[int, ...]
    pairs: tuple[tuple[int, int], ...]
    unpaired: tuple[int, ...]
    metric: MatchMetric
    seed: int | None
    pre_mean_treated: float
    pre_mean_control: float
    pre_smd: float
    mean_pair_distance: float

    @model_validator(mode="after")
    def _valid(self) -> Assignment:
        n = len(self.labels)
        if len(set(self.labels)) != n:
            raise ValueError("labels must be distinct")
        everything = sorted(self.treated + self.control)
        if everything != list(range(n)):
            raise ValueError("treated and control must partition the clusters")
        for i, j in self.pairs:
            if not (i in self.treated) ^ (i in self.control) or i == j:
                raise ValueError(f"pair {(i, j)} is malformed")
            if (i in self.treated) == (j in self.treated):
                raise ValueError(f"pair {(i, j)} has both members on the same arm")
        return self

    @property
    def n_treated(self) -> int:
        return len(self.treated)

    @property
    def n_control(self) -> int:
        return len(self.control)


def _features(pre: Array, metric: MatchMetric) -> Array:
    if metric == "level":
        return np.asarray(pre.mean(axis=1, keepdims=True), dtype=np.float64)
    # Trajectory: the whole pre-period series, standardized per period so no single
    # high-variance period dominates the distance.
    scale = pre.std(axis=0, ddof=0)
    scale = np.where(scale > 0, scale, 1.0)
    return np.asarray((pre - pre.mean(axis=0)) / scale, dtype=np.float64)


def match_clusters(
    pre_outcomes: npt.ArrayLike,
    *,
    labels: Sequence[str] | None = None,
    unit: Unit | None = None,
    metric: MatchMetric = "level",
    seed: int | None = None,
) -> Assignment:
    """Pair clusters by pre-period similarity and randomize one member of each pair.

    ``pre_outcomes`` is ``(n_clusters, n_pre)`` (a 1-D vector is one pre-period
    summary per cluster). Pairing is greedy and deterministic: the unpaired
    cluster with the largest pre-period level is matched to its nearest
    unpaired neighbour under ``metric`` (``"level"``: distance between
    pre-period means; ``"trajectory"``: Euclidean distance between
    per-period-standardized series). Only the coin flip inside each pair uses
    ``seed``.
    """
    pre = np.asarray(pre_outcomes, dtype=np.float64)
    if pre.ndim == 1:
        pre = pre[:, None]
    if pre.ndim != 2 or pre.shape[0] < 2 or pre.shape[1] < 1:
        raise ValueError(
            f"pre_outcomes must be (n_clusters >= 2, n_pre >= 1), got shape {pre.shape}"
        )
    if not np.all(np.isfinite(pre)):
        raise ValueError("pre_outcomes must be finite")
    n = pre.shape[0]
    names = tuple(labels) if labels is not None else tuple(f"c{i}" for i in range(n))
    if len(names) != n:
        raise ValueError(f"{len(names)} labels for {n} clusters")
    entity = unit if unit is not None else _default_unit()
    if entity.kind != "cluster":
        raise ValueError(f"unit {entity.name!r} must have kind='cluster'")

    feats = _features(pre, metric)
    level = pre.mean(axis=1)
    dist = np.sqrt(((feats[:, None, :] - feats[None, :, :]) ** 2).sum(axis=-1))
    rng = np.random.default_rng(seed)
    remaining = set(range(n))
    pairs: list[tuple[int, int]] = []
    treated: list[int] = []
    control: list[int] = []
    distances: list[float] = []
    while len(remaining) >= 2:
        i = max(remaining, key=lambda k: (level[k], -k))
        remaining.discard(i)
        j = min(remaining, key=lambda k: (dist[i, k], k))
        remaining.discard(j)
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        treated.append(a)
        control.append(b)
        pairs.append((a, b))
        distances.append(float(dist[i, j]))
    unpaired = tuple(sorted(remaining))
    control.extend(unpaired)

    t_idx = np.asarray(sorted(treated), dtype=int)
    c_idx = np.asarray(sorted(control), dtype=int)
    pooled_sd = float(level.std(ddof=1)) if n > 1 else 0.0
    diff = float(level[t_idx].mean() - level[c_idx].mean())
    smd = diff / pooled_sd if pooled_sd > 0 else 0.0
    return Assignment(
        unit=entity,
        labels=names,
        treated=tuple(int(i) for i in t_idx),
        control=tuple(int(i) for i in c_idx),
        pairs=tuple(pairs),
        unpaired=unpaired,
        metric=metric,
        seed=seed,
        pre_mean_treated=float(level[t_idx].mean()),
        pre_mean_control=float(level[c_idx].mean()),
        pre_smd=smd,
        mean_pair_distance=float(np.mean(distances)) if distances else 0.0,
    )


# -- holdout trade-off -----------------------------------------------------------------


class HoldoutTradeoff(Spec):
    """MDE as a function of the share of clusters held out of treatment.

    ``fractions`` are holdout (control) shares; ``mdes`` the minimum detectable
    effect at each; ``relative_mde`` is ``mde / min(mde)`` so the price of a
    small holdout is read directly; ``n_holdout`` the integer number of
    clusters held out. ``best_fraction`` minimizes the MDE (always the most
    balanced split on the grid) — the decision trades that against the
    treatment effect forgone on the held-out share, which is
    ``design.economics``' business.
    """

    design: ClusterDesign
    fractions: tuple[float, ...]
    mdes: tuple[float, ...]
    relative_mde: tuple[float, ...]
    n_holdout: tuple[int, ...]
    best_fraction: float
    alpha: float
    power: float
    two_sided: bool

    @model_validator(mode="after")
    def _valid(self) -> HoldoutTradeoff:
        k = len(self.fractions)
        if k == 0 or len(self.mdes) != k or len(self.relative_mde) != k or len(self.n_holdout) != k:
            raise ValueError("fractions, mdes, relative_mde, n_holdout must be equal in length")
        return self


def holdout_tradeoff(
    design: ClusterDesign,
    sd: float,
    fractions: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5),
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> HoldoutTradeoff:
    """MDE at each holdout share for ``design`` (its own ``allocation`` is ignored).

    A holdout share ``f`` holds ``round(f · k)`` clusters out of treatment; a
    share that rounds to fewer than one cluster on either arm is rejected.
    """
    fr = tuple(float(f) for f in fractions)
    if not fr:
        raise ValueError("fractions must be non-empty")
    mdes: list[float] = []
    holdouts: list[int] = []
    for f in fr:
        if not 0.0 < f < 1.0:
            raise ValueError(f"holdout fraction must be in (0, 1), got {f}")
        n_h = int(round(f * design.n_clusters))
        if n_h < 1 or n_h > design.n_clusters - 1:
            raise ValueError(
                f"holdout fraction {f} leaves {n_h} of {design.n_clusters} clusters on one arm"
            )
        holdouts.append(n_h)
        treated_share = 1.0 - n_h / design.n_clusters
        mdes.append(
            cluster_mde(
                design.model_copy(update={"allocation": treated_share}),
                sd,
                alpha=alpha,
                power=power,
                two_sided=two_sided,
            ).effect
        )
    best = min(mdes)
    best_i = int(np.argmin(np.asarray(mdes)))
    return HoldoutTradeoff(
        design=design,
        fractions=fr,
        mdes=tuple(mdes),
        relative_mde=tuple(m / best for m in mdes),
        n_holdout=tuple(holdouts),
        best_fraction=fr[best_i],
        alpha=alpha,
        power=power,
        two_sided=two_sided,
    )
