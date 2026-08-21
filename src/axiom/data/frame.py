"""``Panel``: a role-tagged units × time table. Validates and reports; never imputes."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt
import pandas as pd

from axiom.core.spec import Spec
from axiom.data.roles import RoleMap

__all__ = ["Completeness", "Panel", "PanelError"]


class PanelError(ValueError):
    """The frame does not satisfy the role map."""


class Completeness(Spec):
    """What the panel covers, so an estimator can decide whether it accepts it (review D5).

    ``balanced`` is true iff every unit has every period exactly once.
    ``missing_cells`` counts (unit, period) pairs absent from the frame;
    ``null_cells`` counts nulls in measured columns; ``gaps`` maps a unit to
    the number of periods it is missing.
    """

    n_units: int
    n_periods: int
    n_rows: int
    balanced: bool
    missing_cells: int
    null_cells: int
    gaps: dict[str, int] = {}
    duplicate_rows: int = 0


class Panel:
    """A ``DataFrame`` plus a ``RoleMap``. Sorted by (unit, time) on construction.

    Not a ``Spec``: it holds data. Its identity for provenance is
    ``content_hash()`` — blake2b over the role map's hash and the frame's
    row hashes — and ``io`` stores frame and roles side by side.
    """

    def __init__(self, frame: pd.DataFrame, roles: RoleMap) -> None:
        missing = [c for c in roles.columns if c not in frame.columns]
        if missing:
            raise PanelError(f"frame lacks columns required by the role map: {missing}")
        extra = sorted(set(frame.columns) - set(roles.columns))
        if extra:
            raise PanelError(f"frame has columns with no role: {extra}; drop them or assign a role")
        df = frame.loc[:, list(roles.columns)].copy()
        for col in roles.measured:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise PanelError(f"measured column {col!r} must be numeric, got {df[col].dtype}")
            df[col] = df[col].astype(float)
        if df[roles.unit].isna().any() or df[roles.time].isna().any():
            raise PanelError("unit and time columns may not contain nulls")
        df[roles.unit] = df[roles.unit].astype(str)
        df = df.sort_values([roles.unit, roles.time], kind="stable").reset_index(drop=True)
        self._frame = df
        self._roles = roles

    # -- access ---------------------------------------------------------------

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy()

    @property
    def roles(self) -> RoleMap:
        return self._roles

    @property
    def units(self) -> tuple[str, ...]:
        return tuple(str(u) for u in self._frame[self._roles.unit].unique())

    @property
    def periods(self) -> tuple[object, ...]:
        return tuple(sorted(self._frame[self._roles.time].unique()))

    def column(self, name: str) -> npt.NDArray[np.float64]:
        if name not in self._roles.measured:
            raise KeyError(f"{name!r} is not a measured column")
        return np.asarray(self._frame[name].to_numpy(dtype=float), dtype=np.float64)

    def wide(self, column: str) -> pd.DataFrame:
        """``units × periods`` table of one measured column; NaN where absent."""
        return self._frame.pivot(index=self._roles.unit, columns=self._roles.time, values=column)

    def array(self, column: str) -> npt.NDArray[np.float64]:
        """``(n_units, n_periods)`` array of one measured column; NaN where absent."""
        wide = self.wide(column).reindex(columns=list(self.periods))
        return np.asarray(wide.to_numpy(dtype=float), dtype=np.float64)

    def select_units(self, units: Iterable[str]) -> Panel:
        keep = set(units)
        mask = self._frame[self._roles.unit].astype(str).isin(keep)
        return Panel(self._frame.loc[mask], self._roles)

    def __len__(self) -> int:
        return len(self._frame)

    def __repr__(self) -> str:
        c = self.completeness()
        return (
            f"Panel(units={c.n_units}, periods={c.n_periods}, rows={c.n_rows}, "
            f"balanced={c.balanced}, treatments={list(self._roles.treatments)}, "
            f"outcome={self._roles.outcome[0]!r})"
        )

    # -- reporting ------------------------------------------------------------

    def completeness(self) -> Completeness:
        u, t = self._roles.unit, self._roles.time
        n_units = int(self._frame[u].nunique())
        n_periods = int(self._frame[t].nunique())
        dup = int(self._frame.duplicated([u, t]).sum())
        counts = self._frame.drop_duplicates([u, t]).groupby(u, sort=True)[t].nunique()
        gaps = {str(k): int(n_periods - v) for k, v in counts.items() if v < n_periods}
        missing = int(sum(gaps.values()))
        nulls = int(self._frame[list(self._roles.measured)].isna().sum().sum())
        return Completeness(
            n_units=n_units,
            n_periods=n_periods,
            n_rows=int(len(self._frame)),
            balanced=(missing == 0 and dup == 0),
            missing_cells=missing,
            null_cells=nulls,
            gaps=gaps,
            duplicate_rows=dup,
        )

    def require_balanced(self, *, context: str = "") -> None:
        c = self.completeness()
        if not c.balanced:
            where = f" for {context}" if context else ""
            raise PanelError(
                f"a balanced panel is required{where}: {c.missing_cells} missing cells, "
                f"{c.duplicate_rows} duplicate rows"
            )

    # -- identity -------------------------------------------------------------

    def content_hash(self) -> str:
        """blake2b over the role map's hash and the frame's canonical CSV text.

        The text form (17 significant digits) is what ``io`` stores, so a
        stored panel hashes identically after reload regardless of dtype
        details such as ``object`` vs ``string``.
        """
        h = hashlib.blake2b(digest_size=32)
        h.update(self._roles.content_hash().encode())
        h.update(self.to_csv().encode("utf-8"))
        return h.hexdigest()

    def to_csv(self) -> str:
        """Canonical CSV text: role-ordered columns, sorted rows, ``%.17g`` floats."""
        return str(self._frame.to_csv(index=False, float_format="%.17g", lineterminator="\n"))
