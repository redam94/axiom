"""Gate 1: ``import axiom`` pulls in no sampler and stays light.

Runs in a subprocess so it measures a cold import. The time bound is
relative to ``import pandas`` in the same interpreter (review C9); the
``sys.modules`` assertion is the real invariant.
"""

from __future__ import annotations

import json
import subprocess
import sys

FORBIDDEN = [
    "jax",
    "jaxlib",
    "numpyro",
    "pymc",
    "pytensor",
    "arviz",
    "plotly",
    "cvxpy",
    "statsmodels",
]

PROBE = r"""
import json, sys, time
t0 = time.perf_counter(); import pandas; t_pandas = time.perf_counter() - t0
t0 = time.perf_counter()
import axiom, axiom.core, axiom.data, axiom.io, axiom.infer
t_axiom = time.perf_counter() - t0
print(json.dumps({"t_pandas": t_pandas, "t_axiom": t_axiom, "modules": sorted(sys.modules)}))
"""


def _probe() -> dict:
    out = subprocess.run([sys.executable, "-c", PROBE], check=True, capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def _fastest_probe(n: int = 3) -> dict:
    """The least noise-contaminated of ``n`` probes, componentwise.

    A single sample is not a measurement here: CI runs this suite under
    ``pytest -n logical``, so every core is busy and one unlucky probe can be
    twice the honest cost. Scheduler noise is strictly additive, so the minimum
    of a few runs is the best estimate of each import's real cost -- the same
    reason ``timeit`` reports a minimum rather than a mean.
    """
    runs = [_probe() for _ in range(n)]
    return {
        "t_pandas": min(r["t_pandas"] for r in runs),
        "t_axiom": min(r["t_axiom"] for r in runs),
    }


def test_no_heavy_modules_imported() -> None:
    mods = set(_probe()["modules"])
    hits = sorted(m for m in mods if m.split(".")[0] in FORBIDDEN)
    assert not hits, f"import axiom pulled in heavy modules: {hits}"


def test_import_time_is_bounded() -> None:
    r = _fastest_probe()
    # axiom's own import (after pandas is already loaded) must cost less than
    # pandas did plus a fixed allowance; generous for slow CI runners.
    budget = r["t_pandas"] + 0.6
    assert r["t_axiom"] < budget, f"import axiom took {r['t_axiom']:.3f}s (budget {budget:.3f}s)"
