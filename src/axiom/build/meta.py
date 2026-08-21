"""``MetaBuilder``: study records → ``meta.Corpus``, and the pooling choices → ``meta.PoolSpec``.

Rewritten from the parent's ``builders/model.py`` (ledger: REWRITE). Records
are added one at a time (``study``), wholesale (``records``), or from a
frame through ``meta.from_frame`` (``from_frame``); the pooling side names
the family, moderators, bias term, priors, and interval convention.
``build()`` is the ``Corpus``; ``build_pool_spec()`` is the ``PoolSpec`` —
both ``Spec`` and both round-trip. The corpus validates unique study ids
itself; the builder refuses a duplicate at ``study`` time so the message
names the offending call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import pandas as pd

from axiom.build.base import BuildError, Fields
from axiom.meta import Corpus, PoolPriors, PoolSpec, ReadKind, StudyRecord, from_frame

__all__ = ["MetaBuilder"]


@dataclass(frozen=True)
class MetaBuilder:
    """Collect study records and pooling choices."""

    fields: Fields = Fields()

    def with_(self, **updates: Any) -> MetaBuilder:
        return replace(self, fields=self.fields.with_(**updates))

    # -- records ------------------------------------------------------------------------

    def name(self, name: str) -> MetaBuilder:
        return self.with_(name=name)

    def study(
        self,
        study: str,
        *,
        estimate: float,
        se: float,
        read: ReadKind,
        family: str | None = None,
        contributor: str | None = None,
        quantity: str = "elasticity",
        n: int | None = None,
        moderators: Mapping[str, float] | None = None,
        estimand_hash: str = "",
        source: str = "",
        period: str = "",
    ) -> MetaBuilder:
        """Add one record. ``family`` and ``contributor`` default to the builder's
        ``family`` and to the study id respectively."""
        fam = family if family is not None else self.fields.get("family")
        if fam is None:
            raise BuildError(f"study {study!r}: set family (on the record or via .family())")
        record = StudyRecord(
            study=study,
            contributor=contributor if contributor is not None else study,
            quantity=quantity,
            estimate=float(estimate),
            se=float(se),
            read=read,
            family=fam,
            estimand_hash=estimand_hash,
            n=n,
            moderators=dict(moderators or {}),
            source=source,
            period=period,
        )
        return self.records([record])

    def records(self, records: Sequence[StudyRecord]) -> MetaBuilder:
        current: tuple[StudyRecord, ...] = self.fields.get("records", ())
        seen = {r.study for r in current}
        for r in records:
            if r.study in seen:
                raise BuildError(f"study id {r.study!r} was already added")
            seen.add(r.study)
        return self.with_(records=(*current, *records))

    def from_frame(
        self,
        df: pd.DataFrame,
        *,
        columns: Mapping[str, str] | None = None,
        moderators: Sequence[str] = (),
    ) -> MetaBuilder:
        """Append ``meta.from_frame(df, ...)``'s records."""
        return self.records(from_frame(df, columns=columns, moderators=moderators))

    # -- pooling ------------------------------------------------------------------------

    def family(self, family: str) -> MetaBuilder:
        return self.with_(family=family)

    def moderators(self, *names: str) -> MetaBuilder:
        return self.with_(moderators=tuple(names))

    def bias_term(self, on: bool = True) -> MetaBuilder:
        return self.with_(bias_term=on)

    def effect_key(self, key: Literal["study", "contributor"]) -> MetaBuilder:
        return self.with_(effect_key=key)

    def priors(
        self,
        *,
        mu_scale: float = 1.0,
        tau_scale: float = 1.0,
        gamma_scale: float = 1.0,
        delta_scale: float = 1.0,
        tau_fixed: float | None = None,
    ) -> MetaBuilder:
        return self.with_(
            priors=PoolPriors(
                mu_scale=mu_scale,
                tau_scale=tau_scale,
                gamma_scale=gamma_scale,
                delta_scale=delta_scale,
                tau_fixed=tau_fixed,
            )
        )

    def parametrization(self, kind: Literal["auto", "centered", "marginal"]) -> MetaBuilder:
        return self.with_(parametrization=kind)

    def interval(
        self, *, mass: float = 0.95, definition: Literal["eti", "hdi"] = "eti"
    ) -> MetaBuilder:
        return self.with_(mass=float(mass), definition=definition)

    # -- builds -------------------------------------------------------------------------

    def build(self) -> Corpus:
        return self.build_corpus()

    def build_corpus(self) -> Corpus:
        self.fields.require("records", builder="MetaBuilder.build_corpus")
        return Corpus(records=self.fields.get("records"), name=self.fields.get("name", "corpus"))

    def build_pool_spec(self) -> PoolSpec:
        self.fields.require("family", builder="MetaBuilder.build_pool_spec")
        keys = set(PoolSpec.model_fields)
        payload = {k: v for k, v in self.fields.to_dict().items() if k in keys}
        return PoolSpec(**payload)
