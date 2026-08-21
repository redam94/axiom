"""Shared helpers: import every module under ``axiom`` and list source files."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import axiom

SRC = Path(axiom.__file__).parent


def source_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py"))


def import_all() -> list[str]:
    names = [axiom.__name__]
    for m in pkgutil.walk_packages(axiom.__path__, prefix="axiom."):
        importlib.import_module(m.name)
        names.append(m.name)
    return names


def module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)
