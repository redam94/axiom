"""Running Python the model wrote, against the analysis that is already loaded.

This is the part of the design that needs stating plainly rather than burying.
A gap in a report is usually a missing *picture* — a surface the notebooks fit
and never plotted against the doses that matter, a boundary nobody drew. Closing
that gap means writing code, and the code has to run against the objects the
notebooks built, because rebuilding them from a description is exactly how a
report ends up illustrating something the analysis did not produce.

So the executor runs in the **live namespace**, and three things constrain it:

* **It must produce a figure, bound to a name it is told to use.** A cell that
  runs and binds nothing is a failure, not a success with no output.
* **It is given the names that exist**, so it composes what is there instead of
  inventing a data frame.
* **A traceback comes back as text**, not as an exception, so the caller can
  hand it to the model and ask for a correction — which is what a second attempt
  is worth doing with.

What it is not is a sandbox. It runs in this process with this process's
permissions, which is appropriate for a developer tool pointed at a repository
you already trust and inappropriate for anything else. ``allow_execution`` has
to be passed explicitly for that reason.
"""

from __future__ import annotations

import io
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Any

__all__ = ["ExecutionRefused", "Executed", "describe_namespace", "run_python"]

#: Names a generated cell may not rebind: doing so would rewrite the analysis
#: the report is about rather than draw a picture of it.
PROTECTED = frozenset({"__builtins__", "__name__"})

FIGURE_TYPES = ("Figure", "FigureWidget")


class ExecutionRefused(RuntimeError):
    """Raised when code execution was not enabled by the caller."""


@dataclass(frozen=True)
class Executed:
    """The outcome of one generated cell."""

    code: str
    ok: bool
    bound: str = ""
    kind: str = ""
    stdout: str = ""
    error: str = ""

    def feedback(self) -> str:
        """What to show a model so its second attempt is better than its first."""
        if self.ok:
            return f"bound {self.bound} ({self.kind})"
        return self.error or "the cell ran and bound nothing"


def describe_namespace(namespace: dict[str, Any], *, limit: int = 60) -> str:
    """The names available to generated code, with their types.

    Shown to the model so that it composes the analysis that exists. Values are
    never shown — a model that is given numbers will put them in prose, and the
    prose gates would then be checking the model against itself.
    """
    rows: list[str] = []
    for key, value in sorted(namespace.items()):
        if key.startswith("_") or key in PROTECTED:
            continue
        kind = type(value).__name__
        if kind in ("module", "function", "type", "builtin_function_or_method"):
            continue
        rows.append(f"{key}: {kind}")
        if len(rows) >= limit:
            rows.append(f"... and more ({len(namespace)} names in total)")
            break
    return "\n".join(rows)


def run_python(
    code: str,
    namespace: dict[str, Any],
    *,
    expect: str = "figure",
    allow_execution: bool = False,
) -> Executed:
    """Execute ``code`` in ``namespace`` and report what it bound to ``expect``.

    Never raises on the code's own failure: a traceback is the return value,
    because the caller's next move is to show it to whatever wrote the code.
    """
    if not allow_execution:
        raise ExecutionRefused(
            "generated code was not run: pass allow_execution=True to enable it. "
            "It executes in this process with this process's permissions."
        )
    before = dict(namespace)
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            exec(compile(code, "<generated>", "exec"), namespace)
    except Exception:  # noqa: BLE001 - returned as text for a repair attempt
        for key in set(namespace) - set(before):
            del namespace[key]
        return Executed(
            code=code,
            ok=False,
            stdout=buffer.getvalue(),
            error=traceback.format_exc(limit=3),
        )

    produced = namespace.get(expect)
    if produced is None:
        return Executed(
            code=code,
            ok=False,
            stdout=buffer.getvalue(),
            error=f"the cell ran but bound nothing to `{expect}`",
        )
    kind = type(produced).__name__
    if kind not in FIGURE_TYPES:
        return Executed(
            code=code,
            ok=False,
            stdout=buffer.getvalue(),
            error=f"`{expect}` is a {kind}; a figure was wanted",
        )
    return Executed(code=code, ok=True, bound=expect, kind=kind, stdout=buffer.getvalue())
