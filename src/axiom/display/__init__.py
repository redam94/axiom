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

import io
import sys
from collections.abc import Iterable, Sequence
from typing import Any, TextIO, get_args

from axiom.core.expr import Equation, Expr, Model, ODESystem, System
from axiom.core.interpret.latex import latex_or_unsupported
from axiom.core.model import ModelSpec
from axiom.core.result import Unsupported
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
    "show_math",
    "table",
    "render_table",
]

register_all()

#: How far the label column runs before the value starts. Wide enough for the
#: longest label the renderers use, narrow enough to leave room for a number.
_LABEL_WIDTH = 26

#: How wide a card is drawn when it is exported to HTML for a notebook. A
#: notebook has no terminal to measure, and rich would otherwise guess 80.
_HTML_WIDTH = 110

#: The types whose best rendering is typeset mathematics rather than a card.
#: The nodes are read off the ``Expr`` union so this cannot drift from it.
_MATH_TYPES: tuple[type, ...] = (
    Equation,
    System,
    ODESystem,
    ModelSpec,
    *get_args(get_args(Expr)[0]),
)


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


def _html_card(card: Card) -> str:
    """The card as self-contained HTML. Only called when rich is importable."""
    from rich.console import Console

    # force_jupyter=False or rich publishes the panel to the notebook itself and
    # the recorded buffer comes back empty — a second, unstyled copy of the card.
    console = Console(
        record=True,
        file=io.StringIO(),
        width=_HTML_WIDTH,
        force_terminal=False,
        force_jupyter=False,
    )
    console.print(_rich_renderable(card))
    return console.export_html(
        inline_styles=True,
        code_format=(
            '<pre style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
            'line-height:1.3;white-space:pre;overflow-x:auto;margin:0">{code}</pre>'
        ),
    )


def _math_of(obj: object) -> str | Unsupported:
    """The LaTeX for a tree, or for the mean of a model spec."""
    return latex_or_unsupported(obj.mean if isinstance(obj, ModelSpec) else obj)  # type: ignore[arg-type]


def show_math(obj: Model | ModelSpec, *, file: TextIO | None = None, plain: bool = False) -> None:
    """Show an expression as typeset mathematics in a notebook, as TeX anywhere else.

    ``latex`` returns a string, which a notebook prints as source. This puts the
    string where the front-end will set it. A tree with an ``Opaque`` node cannot
    be written down faithfully, so it shows the typed refusal as a card instead —
    a rendering that quietly dropped the part it could not read would be worse
    than no rendering.
    """
    tex = _math_of(obj)
    if isinstance(tex, Unsupported):
        show(tex, file=file, plain=plain)
        return
    if plain or file is not None or not _in_notebook():
        print(f"$${tex}$$", file=file or sys.stdout)
        return
    from IPython.display import Math
    from IPython.display import display as ipython_display

    ipython_display(Math(tex))


def _in_notebook() -> bool:
    """Whether there is an IPython front-end to hand a rich representation to."""
    try:
        from IPython.core.getipython import get_ipython
    except ModuleNotFoundError:
        return False
    return get_ipython() is not None


def _numeric(text: str) -> bool:
    try:
        float(text.replace(",", "").replace("%", "").strip())
    except ValueError:
        return False
    return True


def _cells(rows: Iterable[Sequence[object]]) -> list[list[str]]:
    return [[("" if c is None else str(c)) for c in row] for row in rows]


def render_table(
    rows: Iterable[Sequence[object]], *, headers: Sequence[str] = (), title: str = ""
) -> str:
    """A table as plain text: columns padded to their widest cell, numbers right.

    The same relationship to ``table`` that ``render`` has to ``show`` — this is
    the string the tests assert on, and it does not depend on whether rich
    happens to be installed.
    """
    body = _cells(rows)
    width = max((len(r) for r in body), default=0)
    width = max(width, len(headers))
    grid = [[*r, *[""] * (width - len(r))] for r in body]
    widths = [
        max(
            len(headers[c]) if c < len(headers) else 0,
            max((len(r[c]) for r in grid), default=0),
        )
        for c in range(width)
    ]
    right = [bool(grid) and all(_numeric(r[c]) for r in grid if r[c].strip()) for c in range(width)]

    def line(cells: Sequence[str], pad: Sequence[bool]) -> str:
        parts = [
            cells[c].rjust(widths[c]) if pad[c] else cells[c].ljust(widths[c]) for c in range(width)
        ]
        return "  ".join(parts).rstrip()

    out = [title] if title else []
    if headers:
        out.append(line([*headers, *[""] * (width - len(headers))], right))
        out.append("  ".join("-" * w for w in widths).rstrip())
    out += [line(r, right) for r in grid]
    return "\n".join(out)


def table(
    rows: Iterable[Sequence[object]],
    *,
    headers: Sequence[str] = (),
    title: str = "",
    file: TextIO | None = None,
    plain: bool = False,
) -> None:
    """Print rows as a table — a rich table if rich is installed, aligned text if not.

    The loop that prints a family and its value on every iteration is the shape
    this replaces: a reader comparing rows wants them in columns, and a notebook
    that prints twelve ragged lines has hidden the comparison it was making.
    """
    body = _cells(rows)
    if plain or not available():
        print(render_table(body, headers=headers, title=title), file=file or sys.stdout)
        return
    from rich.console import Console
    from rich.table import Table

    width = max((len(r) for r in body), default=len(headers))
    width = max(width, len(headers))
    grid = Table(title=title or None, header_style="bold", show_header=bool(headers), expand=False)
    for c in range(width):
        cells = [r[c] for r in body if c < len(r) and r[c].strip()]
        numeric = bool(cells) and all(_numeric(v) for v in cells)
        grid.add_column(
            headers[c] if c < len(headers) else "", justify="right" if numeric else "left"
        )
    for row in body:
        grid.add_row(*row, *[""] * (width - len(row)))
    Console(file=file).print(grid)


def enable(*, plain: bool = False) -> bool:
    """Make every axiom result render itself in this notebook. Call it once.

    Registers three formatters with IPython, in the order the front-end prefers
    them:

    * ``text/latex`` for the model tree — an expression, an equation, a system,
      a ``ModelSpec`` — so a model is set as mathematics rather than printed as
      TeX source;
    * ``text/html`` for every other result, drawn by rich as the same card
      ``show`` prints, so a notebook is not a paler medium than a terminal;
    * ``text/plain`` under both, which is what a diff, a log, and a front-end
      without either will read.

    ``plain=True`` registers only the last, which is what a notebook executed
    into a text-only artifact wants. Without rich installed the HTML formatter
    is skipped and the same thing happens.

    Returns False outside IPython, where there is nothing to register with and
    ``show`` is the way in.

    Formatters rather than ``_repr_html_`` on the types themselves: those live
    in layers below this one and must not learn that a display layer exists.
    """
    try:
        from IPython.core.getipython import get_ipython
    except ModuleNotFoundError:
        return False
    shell = get_ipython()
    if shell is None or shell.display_formatter is None:
        return False

    from axiom.core import Spec

    formatters = shell.display_formatter.formatters
    text = formatters["text/plain"]

    def format_card(obj: object, printer: Any, cycle: bool) -> None:
        if cycle:  # pragma: no cover - a self-referential result is not a thing
            printer.text("...")
            return
        printer.text(render(obj))

    text.for_type(Spec, format_card)
    for kind in REGISTRY:
        text.for_type(kind, format_card)
    if plain:
        return True

    def as_math(obj: object) -> str | None:
        tex = _math_of(obj)
        return None if isinstance(tex, Unsupported) else f"$\\displaystyle {tex}$"

    latex_formatter = formatters["text/latex"]
    for kind in _MATH_TYPES:
        latex_formatter.for_type(kind, as_math)

    if not available():
        return True

    def as_html(obj: object) -> str | None:
        # A tree is mathematics; returning nothing here lets text/latex have it,
        # because a front-end offered both will always take the HTML.
        if isinstance(obj, _MATH_TYPES):
            return None
        return _html_card(card_for(obj))

    html = formatters["text/html"]
    html.for_type(Spec, as_html)
    for kind in REGISTRY:
        html.for_type(kind, as_html)
    return True
