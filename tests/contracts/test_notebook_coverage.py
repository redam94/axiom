"""Gate 12: every public symbol of every subpackage is used in that subpackage's notebooks.

A symbol counts as covered when it appears as a ``Name`` or ``Attribute`` in
a code cell under ``nbs/<subpackage>/``. Markdown mentions do not count.
A subpackage with public symbols and no notebook directory fails outright.
"""

from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path

import pytest
from _walk import SRC

NBS = SRC.parent.parent / "nbs"


def _public(subpackage: str) -> list[str]:
    mod = importlib.import_module(f"axiom.{subpackage}")
    names = getattr(mod, "__all__", None)
    if names is None:
        names = [n for n in vars(mod) if not n.startswith("_")]
    return sorted(names)


def _referenced(nb_dir: Path) -> set[str]:
    seen: set[str] = set()
    for nb in sorted(nb_dir.glob("*.ipynb")):
        cells = json.loads(nb.read_text())["cells"]
        for cell in cells:
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell["source"])
            src = "\n".join(
                line for line in src.splitlines() if not line.lstrip().startswith(("%", "!"))
            )
            try:
                tree = ast.parse(src)
            except SyntaxError as e:
                raise AssertionError(f"{nb}: code cell does not parse: {e}") from e
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    seen.add(node.id)
                elif isinstance(node, ast.Attribute):
                    seen.add(node.attr)
                elif isinstance(node, ast.alias):
                    seen.add(node.name.split(".")[-1])
    return seen


SUBPACKAGES = sorted(p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").exists())


@pytest.mark.parametrize("subpackage", SUBPACKAGES)
def test_public_api_is_demonstrated(subpackage: str) -> None:
    public = _public(subpackage)
    if not public:
        pytest.skip(f"axiom.{subpackage} has no public API yet")
    nb_dir = NBS / subpackage
    assert (
        nb_dir.is_dir()
    ), f"axiom.{subpackage} exports {len(public)} symbols but nbs/{subpackage}/ does not exist"
    assert list(nb_dir.glob("*.ipynb")), f"nbs/{subpackage}/ has no notebooks"
    uncovered = sorted(set(public) - _referenced(nb_dir))
    assert (
        not uncovered
    ), f"axiom.{subpackage}: public symbols not used in any notebook: {uncovered}"
