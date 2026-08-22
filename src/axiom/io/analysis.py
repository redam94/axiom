"""``Analysis``: the frozen container a notebook holds and ``io`` writes.

Review D1 resolved D7 as "yes". It holds state and delegates; it contains no
math. It lives in ``io`` rather than ``core`` because it holds a ``Panel``
(note 0002.2).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from axiom.core.posterior import Posterior
from axiom.core.spec import Spec
from axiom.core.verdict import LedgerLine
from axiom.data.frame import Panel
from axiom.io.provenance import Provenance

__all__ = ["Analysis"]


@dataclass(frozen=True)
class Analysis:
    """Everything an analysis produced, keyed and hashed.

    ``specs`` maps a role name (``"graph"``, ``"surface"``, ``"estimand:roi"``)
    to a spec. ``evidence`` is the sequence of calibration records;
    ``ledger`` the assumption ledger. Both are append-only through
    ``with_evidence`` / ``with_ledger_line``.
    """

    specs: Mapping[str, Spec] = field(default_factory=dict)
    panel: Panel | None = None
    posterior: Posterior | None = None
    evidence: tuple[Spec, ...] = ()
    ledger: tuple[LedgerLine, ...] = ()
    provenance: Provenance | None = None

    def spec(self, role: str) -> Spec:
        try:
            return self.specs[role]
        except KeyError:
            raise KeyError(f"no spec with role {role!r}; have {sorted(self.specs)}") from None

    def with_spec(self, role: str, spec: Spec) -> Analysis:
        return replace(self, specs={**self.specs, role: spec})

    def with_panel(self, panel: Panel) -> Analysis:
        return replace(self, panel=panel)

    def with_posterior(self, posterior: Posterior) -> Analysis:
        return replace(self, posterior=posterior)

    def with_evidence(self, *records: Spec) -> Analysis:
        return replace(self, evidence=(*self.evidence, *records))

    def with_ledger_line(self, *lines: LedgerLine) -> Analysis:
        return replace(self, ledger=(*self.ledger, *lines))

    def hashes(self) -> dict[str, str]:
        """Content hashes of every hashable part, for the provenance record."""
        out = {f"spec:{k}": v.content_hash() for k, v in sorted(self.specs.items())}
        if self.panel is not None:
            out["panel"] = self.panel.content_hash()
        for i, e in enumerate(self.evidence):
            out[f"evidence:{i}"] = e.content_hash()
        for i, line in enumerate(self.ledger):
            out[f"ledger:{i}"] = line.content_hash()
        return out

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Analysis):
            return NotImplemented
        panels_equal = (self.panel is None) == (other.panel is None) and (
            self.panel is None
            or other.panel is None
            or self.panel.content_hash() == other.panel.content_hash()
        )
        return (
            dict(self.specs) == dict(other.specs)
            and panels_equal
            and self.posterior == other.posterior
            and self.evidence == other.evidence
            and self.ledger == other.ledger
            and self.provenance == other.provenance
        )

    def summary(self) -> dict[str, Any]:
        return {
            "specs": sorted(self.specs),
            "panel": None if self.panel is None else repr(self.panel),
            "posterior": None if self.posterior is None else repr(self.posterior),
            "evidence": len(self.evidence),
            "ledger": len(self.ledger),
        }
