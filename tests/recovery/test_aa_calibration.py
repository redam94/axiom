"""Phase 5 gate: A/A calibration of every registered method, and A/B power against prediction.

Roadmap exit criteria 2 and 3. For each method in ``METHODS``, 500 A/A
panels at nominal 5 %: a method is calibrated when its false-positive count
lies in ``clopper_pearson(500, 0.05, 1e-3)`` *and* its rate in
``[0.03, 0.07]``. A calibrated method must be ``"stable"``; one that is not
must be ``"experimental"`` both in ``METHODS`` (shipped) and in
``calibrate_registry``'s output, so a miscalibrated estimator is never
shipped quietly. Difference-in-differences and cluster-based regression must
be calibrated. The table is printed.

The A/B check runs difference-in-differences at three effect sizes and
compares realized power with ``power.power_from_se`` at the exact DiD
standard error under the DGP (``simulate.difference_in_differences_se``;
derivation in the ``simulate`` module docstring): the unit intercepts cancel
in each unit's pre-to-post change, the common AR(1) shocks cancel in the
treated-minus-control contrast, and the remaining independent noise gives
``se = noise_sd · sqrt(1/n_pre + 1/n_post) · sqrt(1/n_treated + 1/n_control)``.
The A/B panel has 80 units: the prediction is normal-theory while the
method's interval uses the Student-t critical value at 78 degrees of freedom,
so the two agree to well under a point; 1000 panels keep the Monte-Carlo
noise near 1.5 points.
"""

from __future__ import annotations

import time

import pytest

from axiom.core import clopper_pearson
from axiom.design.methods.registry import METHODS
from axiom.design.power import power_from_se
from axiom.design.simulate import (
    SimulationSpec,
    calibrate_registry,
    difference_in_differences_se,
    simulated_power,
)

pytestmark = [pytest.mark.slow, pytest.mark.recovery]

N_AA = 500
N_AB = 1000
ALPHA = 0.05
RATE_BOUNDS = (0.03, 0.07)


def test_aa_false_positive_rate_every_method() -> None:
    """The roadmap rule, exactly: every method is calibrated-and-stable or flagged-and-experimental.

    A method is *inside* when it returned no ``Unsupported``, its count lies in
    the exact region, and its rate in ``RATE_BOUNDS``. Inside methods must be
    ``"stable"`` both in ``METHODS`` and in ``calibrate_registry``'s output;
    outside ones must be ``"experimental"`` in both (so the shipped registry
    already says so). Difference-in-differences and cluster-based regression
    must be inside.
    """
    spec = SimulationSpec(
        n_units=40, n_periods=24, n_pre=12, n_treated=20, n_simulations=N_AA, seed=2024
    )
    t0 = time.perf_counter()
    registry, results = calibrate_registry(spec, METHODS, alpha=ALPHA)
    elapsed = time.perf_counter() - t0
    region = clopper_pearson(N_AA, ALPHA, 1e-3)
    print(
        f"\nA/A calibration: {N_AA} panels, nominal {ALPHA:g}, exact region "
        f"[{region.lower}, {region.upper}], rate bounds {RATE_BOUNDS}"
    )
    violations: list[str] = []
    inside: dict[str, bool] = {}
    for r in results:
        ok = (
            r.n_unsupported == 0
            and region.accepts(r.false_positive_count)
            and RATE_BOUNDS[0] <= r.false_positive_rate <= RATE_BOUNDS[1]
        )
        inside[r.method] = ok
        shipped, calibrated = METHODS[r.method].status, registry[r.method].status
        print(
            f"  {r.method:28s} {r.false_positive_count:3d}/{r.n_evaluated} "
            f"rate {r.false_positive_rate:.3f} unsupported {r.n_unsupported} "
            f"{'inside ' if ok else 'OUTSIDE'} METHODS={shipped} calibrated={calibrated}"
        )
        expected = "stable" if ok else "experimental"
        if shipped != expected:
            violations.append(
                f"{r.method}: {'inside' if ok else 'outside'} the region but METHODS says "
                f"{shipped!r} (expected {expected!r}); {r.reason or 'no reason recorded'}"
            )
        if calibrated != expected:
            violations.append(
                f"{r.method}: {'inside' if ok else 'outside'} the region but calibrate_registry "
                f"returned {calibrated!r} (expected {expected!r}); {r.reason}"
            )
    print(f"  ({len(results)} methods x {N_AA} panels in {elapsed:.1f}s)")
    assert not violations, "status does not match calibration:\n" + "\n".join(violations)
    for must in ("difference_in_differences", "cluster_based_regression"):
        assert inside[must], f"{must} must be calibrated on the gate panel"
    assert METHODS["difference_in_differences"].status == "stable", "METHODS was mutated"


@pytest.mark.parametrize("effect", [0.18, 0.28, 0.40])
def test_ab_power_matches_prediction_for_did(effect: float) -> None:
    spec = SimulationSpec(
        n_units=80,
        n_periods=16,
        n_pre=8,
        n_treated=40,
        noise_sd=1.0,
        unit_sd=1.0,
        period_sd=0.5,
        rho=0.5,
        effect=effect,
        n_simulations=N_AB,
        seed=77,
        mass=1.0 - ALPHA,
    )
    predicted = power_from_se(effect, difference_in_differences_se(spec), alpha=ALPHA).power
    r = simulated_power("difference_in_differences", spec, predicted_power=predicted)
    print(
        f"DiD effect {effect}: realized power {r.power:.3f} vs predicted {predicted:.3f} "
        f"(bias {r.bias:+.4f}, coverage {r.coverage:.3f}, region {r.region})"
    )
    assert r.n_unsupported == 0
    assert (
        abs(r.power - predicted) <= 0.05
    ), f"realized power {r.power:.3f} is more than 5 points from predicted {predicted:.3f}"
    assert abs(r.bias) < 3.0 * difference_in_differences_se(spec) / N_AB**0.5
    assert 0.92 <= r.coverage <= 0.98
