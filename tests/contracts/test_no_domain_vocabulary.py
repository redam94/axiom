"""Gate 3: no marketing vocabulary in identifiers outside ``adapters/``.

Matches identifier *tokens* (split on ``_`` and camelCase), not substrings
(review A5), so ``geometric`` and ``median`` are fine and ``geo`` is not.
Comments and docstrings are not checked.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from _walk import SRC, source_files

BANNED = {
    "channel",
    "channels",
    "spend",
    "spends",
    "roas",
    "roi",
    "kpi",
    "kpis",
    "geo",
    "geos",
    "dma",
    "dmas",
    "impression",
    "impressions",
    "creative",
    "creatives",
    "media",
    "campaign",
    "campaigns",
    "brand",
    "brands",
}
_CAMEL = re.compile(r"[A-Z]?[a-z0-9]+|[A-Z]+(?![a-z])")


def tokens(identifier: str) -> list[str]:
    out: list[str] = []
    for part in identifier.split("_"):
        out += [t.lower() for t in _CAMEL.findall(part)]
    return out


def _identifiers(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text())
    names: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute):
            names.append((node.lineno, node.attr))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.append((node.lineno, node.name))
        elif isinstance(node, ast.arg):
            names.append((node.lineno, node.arg))
        elif isinstance(node, ast.alias):
            names.append((getattr(node, "lineno", 0), node.asname or node.name))
        elif isinstance(node, ast.keyword) and node.arg:
            names.append((node.value.lineno, node.arg))
    return names


def test_no_domain_vocabulary_outside_adapters() -> None:
    hits = []
    for path in source_files():
        if "adapters" in path.relative_to(SRC).parts:
            continue
        for lineno, ident in _identifiers(path):
            bad = BANNED & set(tokens(ident))
            if bad:
                hits.append(f"{path.relative_to(SRC)}:{lineno}: {ident} ({sorted(bad)})")
    assert not hits, "marketing vocabulary in identifiers:\n  " + "\n  ".join(hits)


def test_tokenizer_splits_like_the_gate_says() -> None:
    assert tokens("geometric_adstock") == ["geometric", "adstock"]
    assert tokens("GeoLift") == ["geo", "lift"]
    assert tokens("median") == ["median"]
    assert tokens("channelSpend") == ["channel", "spend"]
