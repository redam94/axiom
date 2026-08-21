"""Gate 2: imports point down only.

A module may import from any layer strictly below it. Same-layer imports are
forbidden (review A4). ``core`` imports nothing from axiom.
"""

from __future__ import annotations

import ast
from pathlib import Path

from _walk import SRC, module_name, source_files

LAYER: dict[str, int] = {
    "core": 0,
    "data": 1,
    "io": 2,
    "infer": 3,
    "surface": 4,
    "estimands": 4,
    "identify": 4,
    "sim": 5,
    "meta": 5,
    "calibrate": 5,
    "design": 5,
    "build": 6,
    "diagnose": 6,
    "viz": 7,
    "adapters": 7,
}

# Same-layer exceptions that the architecture names explicitly. None yet.
ALLOWED_SAME_LAYER: set[tuple[str, str]] = set()


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append(node.module)
    return [m for m in out if m == "axiom" or m.startswith("axiom.")]


def _subpackage(mod: str) -> str | None:
    parts = mod.split(".")
    return parts[1] if len(parts) > 1 else None


def test_imports_point_down() -> None:
    violations = []
    for path in source_files():
        me = module_name(path)
        mine = _subpackage(me)
        if mine is None:
            continue
        for imp in _imports(path):
            target = _subpackage(imp)
            if target is None or target == mine:
                continue
            if target not in LAYER or mine not in LAYER:
                violations.append(f"{me}: imports {imp} (unknown layer)")
                continue
            if LAYER[target] > LAYER[mine]:
                violations.append(
                    f"{me} (layer {LAYER[mine]}) imports {imp} (layer {LAYER[target]})"
                )
            elif LAYER[target] == LAYER[mine] and (mine, target) not in ALLOWED_SAME_LAYER:
                violations.append(f"{me} imports peer subpackage {imp}")
    assert not violations, "layering violations:\n  " + "\n  ".join(violations)


def test_core_imports_nothing_from_axiom() -> None:
    bad = []
    for path in (SRC / "core").rglob("*.py"):
        for imp in _imports(path):
            if not imp.startswith("axiom.core"):
                bad.append(f"{module_name(path)} imports {imp}")
    assert not bad, bad


def test_every_subpackage_has_a_layer() -> None:
    pkgs = sorted(p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").exists())
    missing = [p for p in pkgs if p not in LAYER]
    assert not missing, f"subpackages without a layer assignment: {missing}"
