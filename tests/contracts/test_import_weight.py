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


# axiom's own import, measured against `import pandas` in the same interpreter.
# Neither a pure ratio nor a pure allowance describes it, because the cost has
# both a fixed part and a part that scales with the machine, and the three
# environments this runs in sit in different places:
#
#   developer laptop, dev venv    pandas 0.165s   axiom 0.428s   2.60x  +0.26s
#   GitHub runner,    dev venv    pandas 0.615s   axiom 1.384s   2.25x  +0.77s
#   GitHub runner,    core only   pandas 0.179s   axiom 0.546s   3.06x  +0.37s
#   GitHub runner,    core only   pandas 0.260s   axiom 0.833s   3.20x  +0.57s
#
# The last two are the same job on two different days: GitHub does not give you
# the same machine twice, so neither column is reproducible to better than about
# a third of itself. A constant allowance alone tightens on a slow runner (row
# two failed `t_pandas + 0.6`). A ratio alone tightens where pandas is cheap
# (rows three and four failed `3.0x`). The bound is the more generous of the
# two, and both terms have to be exceeded before it fails.
#
# The margins are wide on purpose. This is a coarse regression guard, not a
# benchmark, and the assertion that actually holds the dependency budget is
# `test_no_heavy_modules_imported` above -- that one is exact and names the
# offender. A subpackage acquiring a top-level `import plotly` or `import pymc`
# moves this one by whole multiples; nothing else should move it at all.
_IMPORT_RATIO_BUDGET = 4.0
_IMPORT_ALLOWANCE_S = 1.0


def test_import_time_is_bounded() -> None:
    r = _fastest_probe()
    budget = max(_IMPORT_RATIO_BUDGET * r["t_pandas"], r["t_pandas"] + _IMPORT_ALLOWANCE_S)
    assert r["t_axiom"] < budget, (
        f"import axiom took {r['t_axiom']:.3f}s, "
        f"{r['t_axiom'] / r['t_pandas']:.2f}x pandas' {r['t_pandas']:.3f}s "
        f"(budget {budget:.3f}s)"
    )
