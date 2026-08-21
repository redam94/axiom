"""Gate 5: no pickle in ``axiom.io``, no ``__reduce__`` anywhere, no ``allow_pickle=True``."""

from __future__ import annotations

import ast

from _walk import SRC, module_name, source_files

FORBIDDEN_MODULES = {"pickle", "cloudpickle", "dill", "shelve", "marshal"}


def test_io_imports_no_pickle() -> None:
    bad = []
    for path in (SRC / "io").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                bad += [
                    f"{module_name(path)}: import {a.name}"
                    for a in node.names
                    if a.name.split(".")[0] in FORBIDDEN_MODULES
                ]
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0] in FORBIDDEN_MODULES
            ):
                bad.append(f"{module_name(path)}: from {node.module} import ...")
    assert not bad, bad


def test_no_reduce_and_no_allow_pickle_anywhere() -> None:
    bad = []
    for path in source_files():
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef) and node.name in ("__reduce__", "__reduce_ex__"):
                bad.append(f"{module_name(path)}:{node.lineno}: defines {node.name}")
            if isinstance(node, ast.keyword) and node.arg == "allow_pickle":
                if not (isinstance(node.value, ast.Constant) and node.value.value is False):
                    bad.append(f"{module_name(path)}:{node.value.lineno}: allow_pickle not False")
    assert not bad, bad
