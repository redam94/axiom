"""Phase 2 transport recovery.

The transported estimate hits the target truth; the raw one misses by the
predicted amount.
"""

from __future__ import annotations

import numpy as np
import pytest

from axiom.identify import (
    CausalGraph,
    directly_transportable,
    identify,
    ols,
    s_admissible,
    transport_verdict,
)
from axiom.sim import transport_pair

pytestmark = pytest.mark.recovery


def test_selection_diagram_verdicts() -> None:
    g = CausalGraph.from_edges("Z -> X, Z -> Y, X -> Y", selection=["Z"])
    assert not directly_transportable(g, "X", "Y")
    assert s_admissible(g, "X", "Y", ["Z"]) and not s_admissible(g, "X", "Y", [])
    unadjusted = transport_verdict(g, "X", "Y", given=())
    assert unadjusted.verdict.status == "blocked"
    adjusted = transport_verdict(g, "X", "Y")
    assert adjusted.verdict.status == "identified" and adjusted.s_admissible_set == ("Z",)
    v = identify(g, "X", "Y")
    assert v.transport is not None and v.transport.verdict.status == "identified"


def test_transported_estimate_recovers_target_truth() -> None:
    source, target = transport_pair()
    n = 40_000
    s = source.observed(source.simulate(n, seed=0))
    t = target.observed(target.simulate(n, seed=1))
    # E*[Y | do(x)] at x = 1 in the target population, exactly
    x0 = 1.0
    truth = target.interventional_mean("Y", intervene={"X": x0})
    # the transport formula in linear form: fit Y ~ X + Z in the source, standardize over Z*
    import numpy.linalg as la

    design = np.column_stack([np.ones(len(s)), s["X"], s["Z"]])
    a, b_x, b_z = la.lstsq(design, s["Y"].to_numpy(), rcond=None)[0]
    transported = a + b_x * x0 + b_z * t["Z"].mean()
    untransported = a + b_x * x0 + b_z * s["Z"].mean()
    predicted_bias = b_z * (t["Z"].mean() - s["Z"].mean())
    se = ols(s, "Y", "X", covariates=["Z"]).se * 3
    assert abs(transported - truth) < 0.05 + se
    assert abs((truth - untransported) - predicted_bias) < 0.05 + se
    assert abs(untransported - truth) > 1.0  # the shift in Z is large by construction
