"""Phase 8 exit criterion 3: a planted confounder is priced within 15 % of its true bias.

World (``LinearSCM``, every noise sd 1):

    X -> D: 0.5,  U -> D: 0.8,  D -> Y: 1.0,  U -> Y: 1.0,  X -> Y: 0.3

with ``U`` unmeasured. The naive regression ``Y ~ D + X`` is biased by
``b·a / (a² + σ_d²) = 0.8 / 1.64 ≈ 0.488``. The partial R² values of ``U``
are computed from the DGP, not from the sample:

    r2_dz_x  = a² / (a² + σ_d²)
    r2_yz_dx = b² v / (b² v + σ_y²),   v = Var(U | D, X) = σ_d² / (a² + σ_d²)

and ``bias_bounds`` at those values must reproduce the observed
``naive − true`` gap within 15 %. n = 2000, seed = 20260821.
"""

from __future__ import annotations

import numpy as np
import pytest

from axiom.diagnose.sensitivity import bias_bounds, robustness_value
from axiom.sim import LinearSCM

A, B, C, G, TAU = 0.8, 1.0, 0.5, 0.3, 1.0
N, SEED = 2000, 20260821


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = X.shape[0] - X.shape[1]
    cov = resid @ resid / df * np.linalg.inv(X.T @ X)
    return beta, np.sqrt(np.diag(cov)), df


@pytest.mark.recovery
def test_bias_bounds_prices_planted_confounder() -> None:
    scm = LinearSCM.from_text(
        f"X -> D: {C}, U -> D: {A}, D -> Y: {TAU}, U -> Y: {B}, X -> Y: {G}",
        unmeasured=("U",),
        noise_sd={"X": 1.0, "U": 1.0, "D": 1.0, "Y": 1.0},
    )
    assert scm.total_effect("D", "Y") == TAU
    frame = scm.simulate(N, seed=SEED)
    obs = scm.observed(frame)
    assert "U" not in obs.columns
    X = np.column_stack([np.ones(N), obs["D"], obs["X"]])
    beta, se, df = _ols(obs["Y"].to_numpy(), X)
    naive, se_d = float(beta[1]), float(se[1])
    gap = naive - TAU
    assert gap > 0.3  # the confounder bites

    # population partial R² of U from the DGP
    r2_dz_x = A**2 / (A**2 + 1.0)
    v = 1.0 / (A**2 + 1.0)
    r2_yz_dx = B**2 * v / (B**2 * v + 1.0)
    priced = bias_bounds(naive, se_d, df, r2_yz_dx=r2_yz_dx, r2_dz_x=r2_dz_x)
    assert priced.bias == pytest.approx(gap, rel=0.15)
    assert priced.bias == pytest.approx(A * B / (A**2 + 1.0), rel=0.15)
    assert priced.adjusted_estimate == pytest.approx(TAU, abs=0.15 * gap)

    # RV_q at q = gap / naive is the equal-strength confounder that removes exactly the
    # planted bias; the planted partials (0.390, 0.379) straddle it
    rv = robustness_value(naive, se_d, df, q=gap / naive)
    assert min(r2_dz_x, r2_yz_dx) - 0.02 < rv.rv < max(r2_dz_x, r2_yz_dx) + 0.02
    # and the full estimate is not explained away: both partials sit below RV_1
    assert max(r2_dz_x, r2_yz_dx) < robustness_value(naive, se_d, df).rv

    # in-sample identity: with U's sample partial R² the formula is exact
    Xf = np.column_stack([X, frame["U"]])
    bf, sef, dff = _ols(frame["Y"].to_numpy(), Xf)
    t_u = bf[3] / sef[3]
    r2_yu = t_u**2 / (t_u**2 + dff)
    bd, sed, dfd = _ols(obs["D"].to_numpy(), np.column_stack([np.ones(N), obs["X"], frame["U"]]))
    t_du = bd[2] / sed[2]
    r2_du = t_du**2 / (t_du**2 + dfd)
    exact = bias_bounds(naive, se_d, df, r2_yz_dx=r2_yu, r2_dz_x=r2_du)
    assert exact.adjusted_estimate == pytest.approx(float(bf[1]), rel=1e-6)
