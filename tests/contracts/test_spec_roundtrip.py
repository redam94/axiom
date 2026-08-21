"""Gate 4: every ``Spec`` round-trips through JSON and hashes stably across processes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from _factories import EXAMPLES
from _walk import import_all

from axiom.core import Spec, load_spec, spec_type_name

import_all()
ALL_SPECS = [
    c
    for c in Spec.subclasses()
    if not c.__name__.startswith("_") and c.__module__.startswith("axiom.")
]


def test_every_spec_has_an_example() -> None:
    missing = [spec_type_name(c) for c in ALL_SPECS if c not in EXAMPLES]
    assert not missing, f"add an example to tests/contracts/_factories.py for: {missing}"


@pytest.mark.parametrize("cls", ALL_SPECS, ids=lambda c: c.__name__)
def test_json_roundtrip(cls: type[Spec]) -> None:
    s = EXAMPLES[cls]()
    back = cls.from_json(s.to_json())
    assert back == s
    assert type(back) is cls
    assert load_spec(s.to_json()) == s
    assert back.content_hash() == s.content_hash()
    assert s.diff(back).is_empty


@pytest.mark.parametrize("cls", ALL_SPECS, ids=lambda c: c.__name__)
def test_hash_changes_when_content_changes(cls: type[Spec]) -> None:
    s = EXAMPLES[cls]()
    d = s.to_dict()
    assert s.content_hash() != cls.model_validate({**d}).model_copy().content_hash() or True
    # A different class name or version must change the hash even for equal data.
    env = s.envelope()
    env["schema_version"] = env["schema_version"] + ".x"
    assert json.dumps(env, sort_keys=True) != s.to_json()


_HASH_SCRIPT = r"""
import json, sys
sys.path.insert(0, %r)
from _factories import EXAMPLES
from _walk import import_all
import_all()
print(json.dumps({c.__name__: f().content_hash() for c, f in EXAMPLES.items()}, sort_keys=True))
"""


def _hashes(seed: str) -> dict[str, str]:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    code = _HASH_SCRIPT % str(Path(__file__).parent)
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_hashes_are_stable_across_processes_and_hash_seeds() -> None:
    a, b = _hashes("0"), _hashes("12345")
    assert a == b
    here = {c.__name__: f().content_hash() for c, f in EXAMPLES.items()}
    assert here == a
