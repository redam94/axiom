#!/usr/bin/env python3
"""Capture reference values from mmm-framework into tests/golden/parent_values.json.

Phase 0 deliverable. Run this from inside the *parent* repo's environment, while
it is pinned at the commit recorded in the manifest:

    cd ~/mmm-framework
    uv run python ~/Coding/axiom/scripts/capture_golden.py \
        --out ~/Coding/axiom/tests/golden/parent_values.json

Why it exists: the parent repo keeps moving. Once axiom starts porting, "does
this match the parent" stops being answerable by running the parent. Capture the
answers first. See docs/plan/04-contracts-and-testing.md.

Adding a case: append to CASES. A case is (key, callable, kwargs, rtol). The key
is ``module::function::case_id``. Values are JSON-normalized; numpy arrays become
nested lists. Nothing here imports axiom.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any, Callable, NamedTuple

import numpy as np


class Case(NamedTuple):
    key: str
    fn: Callable[..., Any]
    kwargs: dict[str, Any]
    rtol: float = 1e-12


def _jsonify(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, dict):
        return {k: _jsonify(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonify(x) for x in v]
    return v


def build_cases() -> list[Case]:
    from mmm_framework.calibration.experiment import (
        combine_inverse_variance,
        design_factor,
        mean_sd_to_gamma,
    )
    from mmm_framework.calibration.likelihood import lognormal_sigma_from_moments
    from mmm_framework.planning.cpa import cpa_interval, cpa_power, max_detectable_cpa
    from mmm_framework.planning.eig import (
        decayed_sigma,
        eig_gaussian,
        sigma_exp_for_design,
    )
    from mmm_framework.planning.evoi import preposterior_sd_ratio
    from mmm_framework.transforms.adstock import adstock_weights
    from mmm_framework.transforms.saturation import logistic_saturation, root_saturation

    rng = np.random.default_rng(20260820)
    beta = rng.lognormal(mean=0.0, sigma=0.3, size=2000)
    contrib = beta * 1000.0 * rng.lognormal(mean=0.0, sigma=0.05, size=2000)
    dose = np.linspace(0.0, 4.0, 21)

    cases: list[Case] = [
        # --- calibrate/ : the crown jewels of the prior route -----------------
        Case("calibration.experiment::design_factor::seeded_lognormal",
             design_factor, {"contribution_samples": contrib, "beta_samples": beta}),
        *[Case(f"calibration.experiment::mean_sd_to_gamma::m{m}_s{s}",
               mean_sd_to_gamma, {"mean": m, "sd": s})
          for m, s in [(2.5, 0.4), (1.0, 1.0), (0.3, 0.05), (10.0, 3.0)]],
        Case("calibration.experiment::combine_inverse_variance::three_studies",
             combine_inverse_variance,
             {"targets": [2.5, 3.1, 1.9], "ses": [0.4, 0.9, 0.3]}),
        *[Case(f"calibration.likelihood::lognormal_sigma_from_moments::v{v}_se{se}",
               lognormal_sigma_from_moments, {"value": v, "se": se})
          for v, se in [(2.5, 0.4), (1.0, 0.5), (0.8, 0.05)]],

        # --- design/ : information and precision ------------------------------
        *[Case(f"planning.eig::eig_gaussian::sk{sk}_se{se}",
               eig_gaussian, {"sigma_k": sk, "sigma_exp": se})
          for sk, se in [(1.0, 1.0), (2.0, 0.5), (0.5, 2.0), (1.5, 0.25)]],
        *[Case(f"planning.eig::sigma_exp_for_design::{d}_roi{r}",
               sigma_exp_for_design, {"design_type": d, "roi_median": r})
          for d, r in [("geo_holdout", 2.0), ("geo_holdout", 0.5), ("ghost_ads", 2.0)]],
        *[Case(f"planning.eig::decayed_sigma::sp{sp}_w{w}_hl{hl}",
               decayed_sigma, {"sigma_post": sp, "weeks_elapsed": w, "half_life_weeks": hl})
          for sp, w, hl in [(0.4, 26.0, 52.0), (0.4, 0.0, 52.0), (1.0, 104.0, 26.0)]],
        Case("planning.evoi::preposterior_sd_ratio::t1_se05",
             preposterior_sd_ratio, {"tau": 1.0, "sigma_exp": 0.5}),
        Case("planning.cpa::cpa_interval::base",
             cpa_interval, {"lift": 1200.0, "se_lift": 300.0, "cost": 50000.0}),
        Case("planning.cpa::cpa_power::base",
             cpa_power, {"cost": 50000.0, "se_lift": 300.0, "true_lift": 1200.0}),
        Case("planning.cpa::max_detectable_cpa::base",
             max_detectable_cpa, {"cost": 50000.0, "lift_mde": 800.0}),

        # --- surface/ : the transform chain -----------------------------------
        *[Case(f"transforms.adstock::adstock_weights::{k}_lmax{lm}",
               adstock_weights, {"kind": k, "l_max": lm})
          for k, lm in [("geometric", 8), ("delayed", 12), ("weibull", 12)]],
        Case("transforms.adstock::adstock_weights::geometric_a08_unnormalized",
             adstock_weights,
             {"kind": "geometric", "l_max": 10, "alpha": 0.8, "normalize": False}),
        *[Case(f"transforms.saturation::logistic_saturation::lam{lam}",
               logistic_saturation, {"x": dose, "lam": lam})
          for lam in [0.5, 1.0, 2.5]],
        *[Case(f"transforms.saturation::root_saturation::e{e}",
               root_saturation, {"x": dose, "exponent": e})
          for e in [0.3, 0.5, 0.7]],
    ]
    return cases


# TODO(Phase 0, second pass): these need fixture construction, not just kwargs.
# Each is listed in docs/plan/03-roadmap.md as required golden coverage.
PENDING = [
    "calibration.experiment::derive_channel_prior",
    "planning.eig::eig_monte_carlo",
    "planning.evoi::compute_evpi",
    "planning.budget::optimize_budget",
    "planning.methods::* (six estimators on a fixed panel)",
    "frequentist.design::build_design_matrix (assert the 1e-12 invariant)",
    "validation.confounding_sensitivity::* (Cinelli-Hazlett partial R^2)",
    "benchmarks.meta_model::fit_meta_model (seeded posterior summaries)",
    "dag_model_builder.identification::* (~25 DAG corpus)",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("refusing to capture: not in a git checkout of the parent repo", file=sys.stderr)
        return 2

    import pandas
    import pydantic
    import scipy

    out: dict[str, Any] = {
        "_meta": {
            "source_repo": "mmm-framework",
            "source_commit": sha,
            "captured": date.today().isoformat(),
            "python": ".".join(str(x) for x in sys.version_info[:3]),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pandas.__version__,
            "pydantic": pydantic.__version__,
        },
        "_pending": PENDING,
    }

    failures = 0
    for case in build_cases():
        try:
            value = case.fn(**case.kwargs)
        except Exception as exc:  # capture-time only: record and keep going
            print(f"FAIL {case.key}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1
            continue
        out[case.key] = {
            "kwargs": _jsonify(case.kwargs),
            "value": _jsonify(value),
            "rtol": case.rtol,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    captured = len(out) - 2
    print(f"captured {captured} values from {sha} -> {args.out}")
    if failures:
        print(f"{failures} case(s) failed to evaluate", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
