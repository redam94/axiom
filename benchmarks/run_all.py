"""Run every benchmark case study, skipping the ones whose data is not present.

    python benchmarks/run_all.py            # all of them
    python benchmarks/run_all.py bcg        # just one

Vendored datasets always run. Fetched ones report how to get them and are counted
as skipped rather than failed, so this is safe to run on a fresh clone.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from registry import DATASETS

HERE = Path(__file__).resolve().parent

# case file -> the datasets it needs
CASES = {
    "case_bcg.py": ["bcg"],
    "case_misra1a.py": ["misra1a"],
    "case_darfur.py": ["darfur"],
    "case_lalonde.py": ["lalonde_nsw", "psid_controls"],
}


def main(argv: list[str]) -> int:
    wanted = [a for a in argv if not a.startswith("-")]
    cases = {
        name: needs for name, needs in CASES.items() if not wanted or any(w in name for w in wanted)
    }
    if not cases:
        print(f"nothing matched {wanted}; have {sorted(CASES)}")
        return 2

    width = max(len(n) for n in cases)
    ran, skipped, failed = [], [], []

    for name, needs in cases.items():
        missing = [d for d in needs if not DATASETS[d].available]
        if missing:
            skipped.append(name)
            print(f"{name:<{width}}  skip  needs {', '.join(missing)}")
            continue
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(HERE / name)], capture_output=True, text=True, cwd=str(HERE)
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode == 0:
            ran.append(name)
            print(f"{name:<{width}}  ok    {elapsed:5.1f}s")
        else:
            failed.append(name)
            print(f"{name:<{width}}  FAIL  {elapsed:5.1f}s")
            for line in (proc.stderr or proc.stdout).strip().splitlines()[-12:]:
                print("      " + line)

    print(f"\n{len(ran)} ran, {len(skipped)} skipped, {len(failed)} failed")
    if skipped:
        print("To run the skipped ones (and accept their licences):")
        print("  python benchmarks/fetch.py --all")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
