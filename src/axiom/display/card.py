"""What a rendered result looks like, before anything decides how to draw it.

Every renderer in this subpackage returns a ``Card``: a title, a status, some
labelled rows, and a note. Nothing else. The point of the intermediate form is
that layout is decided once and the two back-ends — plain text and rich — only
choose a typeface for it.

That also makes the layer testable. A test asserts on the ``Card``, not on
escape codes, so a change of colour is not a broken test and a change of
*content* is.

Status is the one piece of semantics the card carries, because it is the one
thing a reader takes from a result at a glance: did this hold, is it assumed,
did it fail. Rendering maps it to a colour and a mark; nothing else in the
package is allowed to invent a third meaning for green.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = ["Card", "Row", "Status", "STATUS_MARK", "status_from"]

Status = Literal["good", "assumed", "bad", "neutral"]
"""``good`` it holds · ``assumed`` it is taken on trust · ``bad`` it failed."""

#: The mark each status prints in plain text, where colour is not available.
#: Chosen from ASCII on purpose: a terminal that cannot do colour often cannot
#: do glyphs either, and a report that renders as boxes is worse than one that
#: renders as letters.
STATUS_MARK: dict[str, str] = {
    "good": "[ok]",
    "assumed": "[assumed]",
    "bad": "[!]",
    "neutral": "",
}

#: What each status becomes in a terminal that has colour.
STATUS_STYLE: dict[str, str] = {
    "good": "green",
    "assumed": "yellow",
    "bad": "red",
    "neutral": "dim",
}


def status_from(word: str, mapping: dict[str, Status], default: Status = "neutral") -> Status:
    """A ``Status`` from a lookup, keeping the literal type through the dict.

    ``mapping.get(word, default)`` widens to ``str``, which is correct for the
    type checker and useless here: every caller wants the literal back.
    """
    return mapping.get(word, default)


@dataclass(frozen=True)
class Row:
    """One labelled line. ``emphasis`` lifts the number a reader came for."""

    label: str
    value: str
    emphasis: bool = False


@dataclass
class Card:
    """One result, ready to be drawn either way."""

    title: str
    status: Status = "neutral"
    rows: list[Row] = field(default_factory=list)
    note: str = ""
    subtitle: str = ""

    def add(self, label: str, value: object, *, emphasis: bool = False) -> Card:
        """Add a row, skipping anything empty — a blank line says nothing."""
        text = "" if value is None else str(value)
        if text.strip():
            self.rows.append(Row(label, text, emphasis))
        return self

    def width(self) -> int:
        return max((len(r.label) for r in self.rows), default=0)
