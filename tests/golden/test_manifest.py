"""Phase 0 gate: the golden fixture exists, parses, and names its source commit.

Every other golden test reads this file. If it rots, the whole port loses its
reference. See docs/plan/04-contracts-and-testing.md.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "parent_values.json"


def load() -> dict:
    return json.loads(FIXTURE.read_text())


def test_fixture_parses_and_records_provenance() -> None:
    data = load()
    meta = data["_meta"]
    assert meta["source_repo"] == "mmm-framework"
    assert meta["source_commit"], "the fixture must name the parent commit it came from"
    for key in ("captured", "python", "numpy", "scipy", "pandas"):
        assert meta[key], f"missing provenance field: {key}"


def test_fixture_is_populated() -> None:
    data = load()
    cases = {k: v for k, v in data.items() if not k.startswith("_")}
    assert len(cases) >= 30, "golden capture looks truncated"
    for key, case in cases.items():
        assert "value" in case, f"{key} has no recorded value"
        assert "rtol" in case, f"{key} has no tolerance"
        assert "::" in key, f"{key} is not module::function::case_id"


def test_pending_list_is_tracked() -> None:
    """The second capture pass is enumerated, not forgotten."""
    data = load()
    assert data["_pending"], "the pending-capture list should not be empty before Phase 5"
