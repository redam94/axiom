"""Showing a result, in a terminal or a notebook, without it becoming a repr.

Sixty-one public types in axiom report an outcome, and until this subpackage
existed every one of them printed as a pydantic repr: three hundred characters
of field names on one line, with the number a reader wanted somewhere in the
middle. A result that is hard to read is a result that gets skimmed, and a
skimmed interval is a point estimate.

    from axiom.display import show
    show(identify(graph, "dose", "pressure"))

    Identification — dose → pressure                              [ok]
      status                    identified
      route                     backdoor
      adjustment set            age
      age blocks the only back-door path

**rich is optional and lazily imported.** With it, a card is a panel with a
colour for its status; without it, the same card is aligned plain text with an
ASCII mark. The content is identical — that is what makes it testable — and
``axiom`` still imports with four dependencies, which is the constraint the
dependency budget exists to protect.

    pip install "axiom[display]"

**In a notebook, call ``enable()`` once.** It registers a formatter with
IPython, so every axiom result renders as a card from then on with no ``show``
in front of it. Registering a formatter rather than patching ``_repr_html_``
onto the types keeps the arrow pointing down: a ``Verdict`` in ``core`` never
learns that a display layer exists.
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

from axiom.display.card import STATUS_MARK, STATUS_STYLE, Card, Row, Status, status_from
from axiom.display.renderers import REGISTRY, card_for, generic_card, register_all, renders

__all__ = [
    "STATUS_MARK",
    "STATUS_STYLE",
    "Card",
    "REGISTRY",
    "Row",
    "Status",
    "available",
    "card_for",
    "enable",
    "generic_card",
    "render",
    "renders",
    "status_from",
    "show",
]

register_all()

#: How far the label column runs before the value starts. Wide enough for the
#: longest label the renderers use, narrow enough to leave room for a number.
_LABEL_WIDTH = 26


def available() -> bool:
    """Whether ``rich`` can be imported, and colour is therefore possible."""
    try:
        import rich  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def render(obj: object, *, width: int = _LABEL_WIDTH) -> str:
    """The result as plain text. Always available, always the same string.

    This is the form the tests assert on: no colour, no terminal detection, no
    dependence on whether an optional package happens to be installed.
    """
    card = obj if isinstance(obj, Card) else card_for(obj)
    mark = STATUS_MARK.get(card.status, "")
    head = card.title if not mark else f"{card.title}  {mark}"
    lines = [head]
    if card.subtitle:
        lines.append(f"  {card.subtitle}")
    for row in card.rows:
        label = row.label[: width - 2].ljust(width - 2)
        lines.append(f"  {label}{row.value}")
    if card.note:
        lines.append(f"  {card.note}")
    return "\n".join(lines)


def _rich_renderable(card: Card) -> Any:
    """The card as a rich panel. Only called when rich is importable."""
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="left", style="dim", no_wrap=True)
    grid.add_column(justify="left", overflow="fold")
    for row in card.rows:
        value = Text(row.value, style="bold" if row.emphasis else "")
        grid.add_row(row.label, value)

    style = STATUS_STYLE.get(card.status, "dim")
    title = Text(card.title, style=style if card.status != "neutral" else "")
    body: Any = grid
    if card.note:
        note = Table.grid()
        note.add_column(overflow="fold")
        note.add_row(grid)
        note.add_row(Text(card.note, style="italic dim"))
        body = note
    return Panel(body, title=title, title_align="left", border_style=style, expand=False)


def show(obj: object, *, file: TextIO | None = None, plain: bool = False) -> None:
    """Print ``obj`` as a card — coloured if rich is installed, plain if not.

    ``plain`` forces the text form, which is what a script writing to a file
    wants and what a doctest needs.
    """
    card = obj if isinstance(obj, Card) else card_for(obj)
    if plain or not available():
        print(render(card), file=file or sys.stdout)
        return
    from rich.console import Console

    Console(file=file).print(_rich_renderable(card))


def enable(*, plain: bool = False) -> bool:
    """Make every axiom result render as a card in this notebook.

    Registers a formatter with IPython for ``Spec`` and for the handful of
    result types that are not Specs. Returns False outside IPython, where there
    is nothing to register with and ``show`` is the way in.

    A formatter rather than a ``_repr_html_`` on the types themselves: those
    live in layers below this one and must not learn that a display layer
    exists.
    """
    try:
        from IPython.core.getipython import get_ipython
    except ModuleNotFoundError:
        return False
    shell = get_ipython()
    if shell is None or shell.display_formatter is None:
        return False

    from axiom.core import Spec

    formatters = shell.display_formatter.formatters["text/plain"]

    def format_card(obj: object, printer: Any, cycle: bool) -> None:
        if cycle:  # pragma: no cover - a self-referential result is not a thing
            printer.text("...")
            return
        printer.text(render(obj))

    formatters.for_type(Spec, format_card)
    for kind in REGISTRY:
        formatters.for_type(kind, format_card)
    return True
