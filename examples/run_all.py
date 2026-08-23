"""Run every example and report which ones work.

    python examples/run_all.py           # all of them
    python examples/run_all.py 02 07     # just those

Each numbered example is a standalone script — they import nothing from each other,
and the only shared module is ``_walkthrough.py``, the stdlib-only narration layer
they are all written against. This runner exists so the whole set can be checked in
one command, and so CI can fail if an example stops working.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def examples() -> list[Path]:
    """The numbered walkthroughs. Leading-underscore modules are support, not demos."""
    return sorted(p for p in HERE.glob("[0-9]*.py"))


def main() -> int:
    wanted = sys.argv[1:]
    scripts = [p for p in examples() if not wanted or any(w in p.name for w in wanted)]
    if not scripts:
        print(f"nothing matched {wanted}")
        return 2

    width = max(len(p.name) for p in scripts)
    failures = []
    total = 0.0

    for script in scripts:
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
        elapsed = time.perf_counter() - t0
        total += elapsed
        ok = proc.returncode == 0
        print(f"{script.name:<{width}}  {'ok  ' if ok else 'FAIL'}  {elapsed:5.1f}s")
        if not ok:
            failures.append(script.name)
            tail = (proc.stderr or proc.stdout).strip().splitlines()
            for line in tail[-12:]:
                print("      " + line)

    print(f"\n{len(scripts) - len(failures)}/{len(scripts)} passed in {total:.0f}s")
    if failures:
        print("failed: " + ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
