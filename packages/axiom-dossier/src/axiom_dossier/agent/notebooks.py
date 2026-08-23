"""Reading a notebook series and running it, which is where the facts come from.

The HYPER-3 case study is six notebooks, seventy-two prose cells and ninety-two
code cells, and **its outputs are not stored** — the repository strips them, so
every number in it exists only while the code is running. A reader who wants the
estimate has to execute the notebook. So does anything that wants to report on
it.

That is the whole reason this module exists and the reason the pipeline around
it is agentic rather than a parser. There is no file to scrape. The facts are
obtained by running the analysis, in order, in one namespace, and then looking
at what is left behind: ``IdentificationVerdict``s, ``LinearEstimate``s,
``Interval``s and the plotly figures the notebooks drew.

Two properties this keeps:

* **Prose and object stay attached.** A notebook's markdown says what the next
  cell is doing; ``Passage`` records that pairing, so a variable called ``itt``
  can be labelled from the sentence above it instead of from its own name.
* **A failure is reported, never skipped.** If a cell raises, execution of that
  notebook stops and the error is recorded with the cell that raised it. A
  report built from a half-run notebook would be a report about an analysis that
  did not finish.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "describe_figure",
    "Execution",
    "Notebook",
    "Passage",
    "execute",
    "harvest",
    "read_notebook",
    "read_series",
]

#: Types worth pulling out of a namespace as reportable quantities. Matched by
#: name rather than by import so this module stays independent of which axiom
#: subpackages happen to be installed.
REPORTABLE = (
    "LinearEstimate",
    "Interval",
    "EstimandResult",
    "Summary",
    "Pooled",
)

#: What a plotly figure calls itself. Harvested separately: a figure is an
#: exhibit, not a claim.
FIGURE_TYPES = ("Figure", "FigureWidget")


@dataclass(frozen=True)
class Passage:
    """One markdown cell, with the code that follows it.

    The pairing is the point. ``code`` is what the prose was introducing, and
    ``defines`` are the names that code bound — which is how a label for a bare
    variable can be recovered from the sentence that announced it.
    """

    notebook: str
    index: int
    text: str
    code: str = ""
    defines: tuple[str, ...] = ()

    def heading(self) -> str:
        """The passage's first markdown heading, if it opens with one."""
        for line in self.text.splitlines():
            if line.startswith("#"):
                return line.lstrip("# ").strip()
        return ""


@dataclass(frozen=True)
class Notebook:
    """A notebook as source: its prose, its code, and the order of both."""

    path: Path
    passages: tuple[Passage, ...]
    code_cells: tuple[str, ...]

    @property
    def name(self) -> str:
        return self.path.name

    def prose(self) -> str:
        return "\n\n".join(p.text for p in self.passages if p.text.strip())


@dataclass
class Execution:
    """What running one notebook produced, and what it failed to."""

    notebook: str
    cells_run: int
    cells_total: int
    error: str = ""
    produced: dict[str, str] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return not self.error and self.cells_run == self.cells_total

    def summary(self) -> str:
        state = "ran" if self.complete else f"stopped at cell {self.cells_run}"
        detail = f" — {self.error}" if self.error else ""
        return f"{self.notebook}: {state} ({self.cells_run}/{self.cells_total}){detail}"


def _source_of(cell: dict[str, Any]) -> str:
    """A cell's source with the shell and magic lines dropped."""
    text = "".join(cell["source"])
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(("%", "!")))


def _names_bound(code: str) -> tuple[str, ...]:
    """Top-level names a cell assigns. Best effort: a cell that will not parse
    binds nothing as far as this is concerned."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return ()
    out: list[str] = []
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                out.append(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                out.extend(e.id for e in target.elts if isinstance(e, ast.Name))
    return tuple(dict.fromkeys(out))


def read_notebook(path: Path) -> Notebook:
    """Parse one ``.ipynb`` into prose paired with the code it introduces."""
    cells = json.loads(Path(path).read_text())["cells"]
    passages: list[Passage] = []
    code_cells: list[str] = []
    pending: str | None = None
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "markdown":
            if pending is not None:
                passages.append(Passage(path.name, index, pending))
            pending = "".join(cell["source"]).strip()
            continue
        if cell["cell_type"] != "code":
            continue
        code = _source_of(cell)
        code_cells.append(code)
        if pending is not None:
            passages.append(
                Passage(path.name, index, pending, code=code, defines=_names_bound(code))
            )
            pending = None
    if pending is not None:
        passages.append(Passage(path.name, len(cells), pending))
    return Notebook(path=Path(path), passages=tuple(passages), code_cells=tuple(code_cells))


def read_series(directory: Path | str, pattern: str = "*.ipynb") -> tuple[Notebook, ...]:
    """Every notebook in a directory, in filename order — which is reading order."""
    root = Path(directory)
    found = sorted(p for p in root.glob(pattern) if not p.name.startswith("."))
    if not found:
        raise FileNotFoundError(f"no notebooks matching {pattern!r} under {root}")
    return tuple(read_notebook(p) for p in found)


@contextmanager
def _as_if_jupyter(directory: Path) -> Iterator[None]:
    """Run with the notebook's directory current and importable.

    Jupyter puts the notebook's own directory on ``sys.path`` and makes it the
    working directory. A notebook that does ``import hyper3`` relies on both, and
    without them every cell in the series fails on the first import.
    """
    was_cwd = Path.cwd()
    added = str(directory)
    sys.path.insert(0, added)
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(was_cwd)
        if added in sys.path:
            sys.path.remove(added)


def execute(notebook: Notebook, namespace: dict[str, Any]) -> Execution:
    """Run a notebook's code cells in ``namespace``, stopping at the first failure.

    The namespace is shared across a series on purpose: notebook 6 reads what
    notebook 1 built, exactly as a person reading them in order would.
    """
    before = set(namespace)
    namespace.setdefault("__name__", "__main__")
    with _as_if_jupyter(notebook.path.parent), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, code in enumerate(notebook.code_cells):
            try:
                exec(compile(code, f"{notebook.name}#{i}", "exec"), namespace)
            except Exception as exc:  # noqa: BLE001 - reported, never swallowed
                return Execution(
                    notebook=notebook.name,
                    cells_run=i,
                    cells_total=len(notebook.code_cells),
                    error=f"cell {i}: {type(exc).__name__}: {exc}",
                    produced={k: type(namespace[k]).__name__ for k in set(namespace) - before},
                )
    return Execution(
        notebook=notebook.name,
        cells_run=len(notebook.code_cells),
        cells_total=len(notebook.code_cells),
        produced={k: type(namespace[k]).__name__ for k in set(namespace) - before},
    )


def harvest(namespace: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a namespace into reportable quantities and drawable figures.

    Returns ``(quantities, figures)``, both keyed by the variable name the
    notebook used. The names are poor labels — ``itt``, ``band``, ``fig`` — and
    labelling them is a job for something that has read the prose, which is why
    this returns the raw pairing rather than pretending to name them.
    """
    quantities: dict[str, Any] = {}
    figures: dict[str, Any] = {}
    for key, value in namespace.items():
        if key.startswith("_"):
            continue
        kind = type(value).__name__
        if kind in FIGURE_TYPES:
            figures[key] = value
        elif kind in REPORTABLE:
            quantities[key] = value
    return quantities, figures


def describe_figure(figure: Any) -> str:
    """What a drawn figure contains: its traces and its axis titles.

    A gap stage shown only the variable name ``01_fig`` cannot tell whether the
    dose-response curve was drawn or not, so it either invents work or declares
    everything covered. Describing the figure by its contents is what lets it
    answer the question it was asked.
    """
    layout = getattr(figure, "layout", None)
    data = getattr(figure, "data", ()) or ()
    kinds: dict[str, int] = {}
    names: list[str] = []
    for trace in data:
        kind = getattr(trace, "type", "trace")
        kinds[kind] = kinds.get(kind, 0) + 1
        name = getattr(trace, "name", None)
        if name and name not in names:
            names.append(str(name))

    def axis_title(axis: str) -> str:
        holder = getattr(layout, axis, None)
        title = getattr(holder, "title", None)
        return str(getattr(title, "text", "") or "")

    parts = [", ".join(f"{n}x {k}" for k, n in sorted(kinds.items())) or "no traces"]
    x, y = axis_title("xaxis"), axis_title("yaxis")
    if x or y:
        parts.append(f"axes: x={x or '?'}, y={y or '?'}")
    if names:
        parts.append("series: " + ", ".join(names[:6]))
    title = getattr(getattr(layout, "title", None), "text", "") or ""
    if title:
        parts.append(f"titled: {title}")
    return "; ".join(parts)
