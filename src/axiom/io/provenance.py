"""Provenance: the set of content hashes an analysis touched, plus where it ran."""

from __future__ import annotations

import datetime as _dt
import platform
import sys
from collections.abc import Mapping

import numpy
import pandas
import pydantic
import scipy

from axiom.core.spec import Spec

__all__ = ["Provenance", "environment_fingerprint"]


def environment_fingerprint() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": sys.platform,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "pydantic": pydantic.__version__,
    }


class Provenance(Spec):
    """Who made this, from what, with which seed.

    ``hashes`` maps a role (``"graph"``, ``"surface"``, ``"panel"``,
    ``"posterior"`` ...) to a content hash. ``created`` is an ISO-8601 UTC
    timestamp string, set explicitly so the record is reproducible in tests.
    """

    axiom_version: str
    created: str
    hashes: dict[str, str] = {}
    seed: int | None = None
    environment: dict[str, str] = {}
    notes: tuple[str, ...] = ()

    @classmethod
    def now(
        cls,
        axiom_version: str,
        *,
        hashes: Mapping[str, str] | None = None,
        seed: int | None = None,
        notes: tuple[str, ...] = (),
    ) -> Provenance:
        return cls(
            axiom_version=axiom_version,
            created=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            hashes=dict(hashes or {}),
            seed=seed,
            environment=environment_fingerprint(),
            notes=notes,
        )
