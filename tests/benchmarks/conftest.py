"""Put ``benchmarks/`` on the path so the registry is importable from the tests."""

import sys
from pathlib import Path

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"

if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))
