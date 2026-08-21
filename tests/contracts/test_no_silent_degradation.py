"""Gate 8: no broad ``except`` swallows an error.

Ruff's BLE001 catches blind ``except Exception`` at lint time. This gate
catches (a) any ``# noqa`` that disables it, and (b) bare ``except:`` or
``except Exception`` handlers that neither re-raise, nor log at warning or
above, nor return (a typed failure) — the three permitted forms.
"""

from __future__ import annotations

import ast
import re

from _walk import module_name, source_files

_NOQA = re.compile(r"#\s*noqa[:\s].*\bBLE001\b")
_LOG = {"warning", "error", "critical", "exception"}


def _is_broad(handler: ast.ExceptHandler) -> bool:
    t = handler.type
    if t is None:
        return True
    if isinstance(t, ast.Name) and t.id in ("Exception", "BaseException"):
        return True
    return isinstance(t, ast.Tuple) and any(
        isinstance(e, ast.Name) and e.id in ("Exception", "BaseException") for e in t.elts
    )


def _handles_honestly(handler: ast.ExceptHandler) -> bool:
    for node in ast.walk(handler):
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Return):
            return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _LOG
        ):
            return True
    return False


def test_no_noqa_ble001() -> None:
    hits = [
        f"{module_name(p)}:{i + 1}"
        for p in source_files()
        for i, line in enumerate(p.read_text().splitlines())
        if _NOQA.search(line)
    ]
    assert not hits, f"BLE001 suppressed at: {hits}"


def test_broad_handlers_reraise_log_or_return() -> None:
    bad = []
    for path in source_files():
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.ExceptHandler)
                and _is_broad(node)
                and not _handles_honestly(node)
            ):
                bad.append(f"{module_name(path)}:{node.lineno}")
    assert not bad, f"broad except handlers that swallow: {bad}"
