"""Project-side record emission: turn a realized estimand (or a hand-entered summary)
into a ``StudyRecord``.

Ported from the parent's ``benchmarks/contribute.py`` (ledger row
``meta/contribute.py``, PORT). The parent emitted contribution records from
a fitted marketing model; here the input is an ``estimands.EstimandResult``,
which already carries the dimension, the unit, the estimand hash, and a
posterior ``Summary``.

**Standard error from a summary.** A record needs a standard error, and a
``Summary`` carries an interval plus a posterior sd. When the interval is
``wald`` (``estimate ± z·se``) the se is exactly ``width / (2 z)``. When it
is an ``eti`` of a normal posterior the same formula is exact; for any other
posterior it is an approximation. For an ``hdi`` the width relation holds
only under symmetry, so the posterior sd is used instead. ``detail["se_from"]``
records which rule produced the number.
"""

from __future__ import annotations

from collections.abc import Mapping

from scipy.special import ndtri

from axiom.core import Dimension, Summary, dimensionless
from axiom.estimands import EstimandResult
from axiom.meta.schema import ReadKind, StudyRecord, poolable_quantity

__all__ = ["record_from_result", "record_from_summary", "se_from_summary"]


def se_from_summary(summary: Summary) -> tuple[float, str]:
    """``(se, rule)``: interval half-width over ``z`` for ``wald``/``eti``, else the sd."""
    iv = summary.interval
    if iv.definition in ("wald", "eti"):
        z = float(ndtri((1.0 + iv.mass) / 2.0))
        se = iv.width / (2.0 * z)
        if se > 0:
            return float(se), f"{iv.definition}_width/(2z) at mass {iv.mass}"
    return float(summary.sd), "posterior_sd"


def record_from_result(
    result: EstimandResult,
    *,
    study: str,
    contributor: str,
    read: ReadKind,
    family: str,
    quantity: str | None = None,
    moderators: Mapping[str, float] | None = None,
    period: str = "",
    source: str = "",
) -> StudyRecord:
    """A ``StudyRecord`` from a realized estimand.

    ``quantity`` defaults to the result's ``kind`` when that is a poolable
    name (``elasticity``); contrasts, marginals and ratios carry units and
    must be named explicitly by the caller who knows how they were
    standardized. The dimension is the result's; when it is not dimensionless
    the result's unit becomes ``unit_scale`` so the record is constructible
    and ``normalize`` can demand a licensed transfer. ``detail`` records the
    se rule, the result's status, and its producer hash.
    """
    q = quantity if quantity is not None else result.kind
    poolable_quantity(q)
    se, rule = se_from_summary(result.summary)
    dim: Dimension = result.dimension
    unit_scale = "" if dim.is_dimensionless else (result.unit or str(dim))
    return StudyRecord(
        study=study,
        contributor=contributor,
        quantity=q,
        estimand_hash=result.estimand_hash,
        estimate=float(result.summary.mean),
        se=se,
        n=None,  # draws are not units; the study size is not known to a result
        read=read,
        family=family,
        moderators=dict(moderators or {}),
        dimension=dim,
        unit_scale=unit_scale,
        period=period,
        source=source or result.estimand_name,
        detail={
            "se_from": rule,
            "result_status": result.status,
            "producer_hash": result.producer_hash,
            "n_draws": str(result.n_draws),
        },
    )


def record_from_summary(
    *,
    study: str,
    contributor: str,
    quantity: str,
    estimate: float,
    se: float,
    read: ReadKind,
    family: str,
    n: int | None = None,
    moderators: Mapping[str, float] | None = None,
    dimension: Dimension | None = None,
    unit_scale: str = "",
    estimand_hash: str = "",
    period: str = "",
    source: str = "",
    detail: Mapping[str, str] | None = None,
) -> StudyRecord:
    """A ``StudyRecord`` from hand-entered numbers (a published table, a report)."""
    return StudyRecord(
        study=study,
        contributor=contributor,
        quantity=quantity,
        estimand_hash=estimand_hash,
        estimate=float(estimate),
        se=float(se),
        n=n,
        read=read,
        family=family,
        moderators=dict(moderators or {}),
        dimension=dimension if dimension is not None else dimensionless(),
        unit_scale=unit_scale,
        period=period,
        source=source,
        detail={"se_from": "reported", **(detail or {})},
    )
