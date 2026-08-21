"""Study normalization: records (or a frame) in, a ``Corpus`` or a typed refusal out.

Ported from the parent's ``benchmarks/ingest.py`` (ledger row ``meta/ingest.py``,
PORT). The parent's ingest coerced marketing study forms into one table; here
``normalize`` is the single gate every record passes before it can be pooled,
and it refuses rather than coerces:

* a record whose dimension is not dimensionless, or whose ``unit_scale`` is
  declared, is refused with a reason naming the study — unless
  ``plans[study]`` is a licensed ``TransferPlan``, in which case the record
  is admitted and a ``LedgerLine`` for the transfer is recorded in its
  ``detail`` (roadmap criterion 5);
* a log-scale quantity (``POOLABLE_QUANTITIES[q].scale == "log"``) with a
  non-positive estimate is refused — there is no log of it;
* duplicate study ids are refused.

``from_frame`` builds records from a pandas frame under an explicit column
mapping; nothing is inferred from column names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from axiom.core import LedgerLine, Unsupported
from axiom.estimands import TransferPlan
from axiom.meta.schema import Corpus, ReadKind, StudyRecord, poolable_quantity

__all__ = ["DEFAULT_COLUMNS", "from_frame", "normalize"]


DEFAULT_COLUMNS: dict[str, str] = {
    "study": "study",
    "contributor": "contributor",
    "quantity": "quantity",
    "estimate": "estimate",
    "se": "se",
    "read": "read",
    "family": "family",
}
"""Record field → frame column. Required fields; optional ones (``n``, ``period``,
``source``, ``estimand_hash``, ``unit_scale``) are taken only when mapped."""

_OPTIONAL = ("n", "period", "source", "estimand_hash", "unit_scale")


def from_frame(
    df: pd.DataFrame,
    *,
    columns: Mapping[str, str] | None = None,
    moderators: Sequence[str] = (),
) -> tuple[StudyRecord, ...]:
    """Build records from a frame under an explicit ``field → column`` mapping.

    ``columns`` overrides ``DEFAULT_COLUMNS`` entry by entry and may add the
    optional fields. ``moderators`` names frame columns copied into each
    record's ``moderators`` under their column name. Every record is
    dimensionless (a frame cannot carry a ``Dimension``); records on a unit
    scale declare it through a mapped ``unit_scale`` column and are then
    gated by ``normalize``.
    """
    mapping = {**DEFAULT_COLUMNS, **(columns or {})}
    missing = [c for c in mapping.values() if c not in df.columns]
    if missing:
        raise ValueError(f"frame lacks mapped columns {missing}; have {list(df.columns)}")
    absent = [m for m in moderators if m not in df.columns]
    if absent:
        raise ValueError(f"frame lacks moderator columns {absent}")
    out: list[StudyRecord] = []
    for _, row in df.iterrows():
        fields: dict[str, object] = {
            "study": str(row[mapping["study"]]),
            "contributor": str(row[mapping["contributor"]]),
            "quantity": str(row[mapping["quantity"]]),
            "estimate": float(row[mapping["estimate"]]),
            "se": float(row[mapping["se"]]),
            "read": _read(row[mapping["read"]]),
            "family": str(row[mapping["family"]]),
            "moderators": {m: float(row[m]) for m in moderators},
        }
        for opt in _OPTIONAL:
            if opt in mapping:
                v = row[mapping[opt]]
                if opt == "n":
                    fields[opt] = None if pd.isna(v) else int(v)
                else:
                    fields[opt] = "" if pd.isna(v) else str(v)
        out.append(StudyRecord(**fields))  # type: ignore[arg-type]
    return tuple(out)


def _read(v: object) -> ReadKind:
    s = str(v)
    if s == "model":
        return "model"
    if s == "experiment":
        return "experiment"
    raise ValueError(f"read must be 'model' or 'experiment', got {v!r}")


def normalize(
    records_or_frame: Sequence[StudyRecord] | Corpus | pd.DataFrame,
    *,
    plans: Mapping[str, TransferPlan] | None = None,
    name: str = "corpus",
    columns: Mapping[str, str] | None = None,
    moderators: Sequence[str] = (),
) -> Corpus | Unsupported:
    """Admit records into a ``Corpus`` or refuse with a reason naming the study.

    Refusals (all ``Unsupported``, with ``missing`` naming what would admit
    the record): a non-dimensionless or unit-scaled record without a
    licensed ``plans[study]``; a log-scale quantity with a non-positive
    estimate; a duplicate study id. A licensed plan admits the record and
    writes the transfer's ledger line into ``detail["transfer"]`` together
    with the plan's content hash, so the pool's provenance shows which
    numbers crossed a scale boundary and under which assumptions.
    """
    if isinstance(records_or_frame, pd.DataFrame):
        records: Sequence[StudyRecord] = from_frame(
            records_or_frame, columns=columns, moderators=moderators
        )
    elif isinstance(records_or_frame, Corpus):
        records = records_or_frame.records
        name = records_or_frame.name
    else:
        records = tuple(records_or_frame)
    plans = plans or {}

    seen: set[str] = set()
    admitted: list[StudyRecord] = []
    for r in records:
        if r.study in seen:
            return Unsupported(
                reason=f"duplicate study id {r.study!r}; each record needs a unique study id",
                missing=("unique_study_id",),
                detail={"study": r.study},
            )
        seen.add(r.study)

        q = poolable_quantity(r.quantity)
        if q.scale == "log" and r.estimate <= 0:
            return Unsupported(
                reason=(
                    f"study {r.study!r}: quantity {r.quantity!r} is pooled on the log scale "
                    f"and its estimate {r.estimate} is not positive"
                ),
                missing=("positive_estimate",),
                detail={"study": r.study, "quantity": r.quantity, "estimate": repr(r.estimate)},
            )

        needs_plan = not r.is_dimensionless or not q.dimensionless
        if needs_plan:
            plan = plans.get(r.study)
            if plan is None or not plan.licensed:
                why = (
                    "no TransferPlan supplied"
                    if plan is None
                    else f"its TransferPlan is {plan.status!r}: {plan.reason or 'not licensed'}"
                )
                return Unsupported(
                    reason=(
                        f"study {r.study!r} reports {r.quantity!r} on a scale "
                        f"(dimension {r.dimension}, unit_scale {r.unit_scale!r}); pooling is "
                        f"over dimensionless quantities and {why}"
                    ),
                    missing=("licensed_transfer_plan",),
                    detail={
                        "study": r.study,
                        "dimension": str(r.dimension),
                        "unit_scale": r.unit_scale,
                    },
                )
            line = LedgerLine(
                kind="meta_transfer",
                statement=(
                    f"study {r.study!r} admitted on scale {r.unit_scale or str(r.dimension)!r} "
                    f"under a {plan.status!r} transfer plan"
                ),
                detail={"assumptions": ",".join(a.name for a in plan.assumptions)},
                source=plan.source,
                target=plan.target,
            )
            r = r.model_copy(
                update={
                    "detail": {
                        **r.detail,
                        "transfer": line.statement,
                        "transfer_plan_hash": plan.content_hash(),
                        "transfer_status": plan.status,
                        "transfer_assumptions": line.detail["assumptions"],
                    }
                }
            )
        admitted.append(r)

    if not admitted:
        return Unsupported(reason="no records to normalize", missing=("records",))
    return Corpus(records=tuple(admitted), name=name)
