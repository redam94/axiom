"""The narration layer the examples are written against.

An example is not a script that prints numbers. It is an argument: here is the
question, here is the step, here is why that step is the necessary one, here is
what we chose not to do and why, and here is what came out. This module is what
lets each example say all of that in one place, once, so the terminal reader and
the site reader get the same walkthrough rather than one of them getting a
transcript.

Two audiences, one source:

*   Run ``python examples/01_agriculture_response_surface.py`` and you get the
    prose, the decisions and the numbers, wrapped for a terminal.
*   Set ``AXIOM_WALKTHROUGH_JSON=<path>`` and the same run also writes a
    structured record — steps, narrative, readouts, tables and chart payloads —
    which ``site/_gen/generate.py`` turns into the walkthrough pages. The site
    therefore cannot drift from the code: a figure on the page exists because a
    run produced its data.

This is the one module the examples import from each other's directory. It is
deliberately stdlib-only and imports nothing from ``axiom``, so it can never be
the reason an example works or fails.

Writing a step::

    w = Walkthrough(
        field="Agronomy",
        title="Where the yield actually peaks",
        question="Where is the optimum, and how should a rationed budget be split?",
    )

    w.step(
        "Spend the plots where the curvature is",
        why="A saturating response is a shape, and a shape is estimated from "
            "runs placed where it bends.",
        instead="A 5x5 grid measures the same shape with twice the plots and "
                "still puts nothing at the centre.",
    )
    w.out(f"design: {design.kind}, {design.n} plots")
    w.figure("design", kind="scatter", data={...}, opt={"key": "@"}, title="...")

``opt`` values that begin with ``@`` are paths into this figure's own ``data``
— ``"@"`` is the payload itself, ``"@rows"`` is ``data["rows"]``. The site
rewrites them into absolute paths at build time, which is why an example never
has to know where its data will end up living.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import sys
import textwrap
from typing import Any

# Width of the terminal rendering. Narrow enough to read, wide enough for a table.
WIDTH = max(72, min(96, shutil.get_terminal_size((88, 24)).columns))

_ACTIVE: list[Walkthrough] = []


def _wrap(text: str, indent: str, width: int = WIDTH) -> str:
    """Re-flow a docstring-style paragraph block for the terminal."""
    out = []
    for para in text.strip().split("\n\n"):
        cleaned = " ".join(line.strip() for line in para.strip().splitlines())
        out.append(
            textwrap.fill(cleaned, width=width, initial_indent=indent, subsequent_indent=indent)
        )
    return "\n\n".join(out)


class Walkthrough:
    """One example, as a sequence of narrated steps.

    Every method both prints and records. Nothing is printed that is not
    captured, and nothing is captured that a reader at a terminal does not see —
    which is the property that keeps the generated page honest.
    """

    def __init__(self, *, field: str, title: str, question: str) -> None:
        self.field = field
        self.title = title
        self.question = question
        self.steps: list[dict[str, Any]] = []
        self.figures: dict[str, Any] = {}
        self.findings: list[str] = []
        _ACTIVE.append(self)

        print("=" * WIDTH)
        print(f"{field.upper()}  ---  {title}")
        print("=" * WIDTH)
        print(_wrap(question, "  "))

    # -- structure ----------------------------------------------------------

    def step(self, title: str, *, why: str, instead: str | None = None) -> None:
        """Open a numbered step.

        ``why`` is the argument for the step existing at all; ``instead`` is the
        alternative that a reasonable person would have reached for first, and
        the reason this example does not. A step with no honest ``instead`` is
        allowed — a step with a dishonest one is worse than none.
        """
        n = len(self.steps) + 1
        self.steps.append(
            {
                "n": n,
                "title": title,
                "why": " ".join(why.split()),
                "instead": " ".join(instead.split()) if instead else None,
                "blocks": [],
            }
        )
        print()
        print("-" * WIDTH)
        print(f"STEP {n}  {title}")
        print("-" * WIDTH)
        print(_wrap(why, "  "))
        if instead:
            print()
            print(_wrap("INSTEAD OF: " + " ".join(instead.split()), "  "))
        print()

    def say(self, text: str) -> None:
        """A paragraph of narrative inside the current step."""
        self._blocks().append({"type": "say", "text": " ".join(text.split())})
        print(_wrap(text, "  "))
        print()

    # -- numbers ------------------------------------------------------------

    def out(self, line: str = "") -> None:
        """One line of readout: printed indented, recorded verbatim.

        Consecutive calls collect into a single block, so the page renders them
        as one terminal panel rather than a stack of one-line boxes.
        """
        blocks = self._blocks()
        if not blocks or blocks[-1]["type"] != "out":
            blocks.append({"type": "out", "lines": []})
        blocks[-1]["lines"].append(line)
        print(("  " + line).rstrip())

    def table(self, columns: list[str], rows: list[list[Any]], *, caption: str = "") -> None:
        """A small table, printed aligned and recorded as real rows.

        The site renders it as a table rather than as preformatted text, which
        is the difference between a number a reader can scan and a number they
        have to count columns to find.
        """
        body = [[_cell(c) for c in row] for row in rows]
        widths = [
            max(len(str(columns[i])), *(len(r[i]) for r in body)) if body else len(columns[i])
            for i in range(len(columns))
        ]
        self._blocks().append(
            {"type": "table", "columns": list(columns), "rows": body, "caption": caption}
        )
        if caption:
            print(_wrap(caption, "  "))
        head = "  " + "  ".join(str(c).ljust(widths[i]) for i, c in enumerate(columns))
        print(head.rstrip())
        print("  " + "  ".join("-" * w for w in widths))
        for row in body:
            print(("  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(row))).rstrip())
        print()

    # -- pictures -----------------------------------------------------------

    def figure(
        self,
        name: str,
        *,
        kind: str,
        data: Any,
        opt: dict[str, Any],
        title: str,
        note: str = "",
        legend: tuple[tuple[str, str], ...] = (),
        height: int | None = None,
    ) -> None:
        """Record a chart: its payload, and how the site should draw it.

        ``kind`` names one of the site's chart types (``band``, ``lines``,
        ``intervals``, ``bars``, ``heatmap``, ``scatter``, ``hist``,
        ``dumbbell``, ``funnel``). ``legend`` is a sequence of
        ``(swatch, label)`` pairs where the swatch is a CSS custom-property
        name such as ``accent`` or ``ink-2``, optionally suffixed ``:dash``.

        At a terminal a figure prints as a one-line marker: the numbers behind
        it are already in the readout above, so repeating them as ASCII art
        would be noise, and pretending the terminal has the chart would be a
        lie.
        """
        if name in self.figures:
            raise ValueError(f"figure {name!r} already recorded in this walkthrough")
        self.figures[name] = data
        self._blocks().append(
            {
                "type": "figure",
                "name": name,
                "kind": kind,
                "opt": opt,
                "title": " ".join(title.split()),
                "note": " ".join(note.split()),
                "legend": [list(pair) for pair in legend],
                "height": height,
            }
        )
        print(_wrap(f"[figure: {title}]", "  "))
        print()

    # -- the close ----------------------------------------------------------

    def finding(self, text: str) -> None:
        """A closing paragraph: what the run actually showed.

        Separate from the steps because it is the only part written after
        looking at the output, and it is the part that most often disagrees
        with the setup.
        """
        if not self.findings:
            print()
            print("=" * WIDTH)
            print("WHAT IT ACTUALLY SHOWED")
            print("=" * WIDTH)
        self.findings.append(" ".join(text.split()))
        print(_wrap(text, "  "))
        print()

    # -- internals ----------------------------------------------------------

    def _blocks(self) -> list[dict[str, Any]]:
        if not self.steps:
            raise RuntimeError("call step() before recording anything into it")
        blocks: list[dict[str, Any]] = self.steps[-1]["blocks"]
        return blocks

    def record(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "title": self.title,
            "question": " ".join(self.question.split()),
            "steps": self.steps,
            "figures": self.figures,
            "findings": self.findings,
        }


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


@atexit.register
def _dump() -> None:
    """Write the structured record, if this run was asked for one.

    Guarded by an environment variable rather than a flag so that the examples
    stay ordinary scripts: nothing in the twelve of them knows the site exists.
    """
    path = os.environ.get("AXIOM_WALKTHROUGH_JSON")
    if not path or not _ACTIVE:
        return
    try:
        with open(path, "w") as fh:
            json.dump(_ACTIVE[-1].record(), fh)
    except OSError as exc:  # pragma: no cover - a broken capture must be loud
        print(f"walkthrough capture failed: {exc}", file=sys.stderr)
        raise
