"""Charter success criterion 8: a complete analysis round-trips through ``axiom.io``.

A graph, an estimand and its result, a surface fit, a design, a calibration
(measurement, calibrated spec, ledger) and a meta contribution (corpus, pooled
estimate) are saved with ``save_analysis`` and read back with
``load_analysis``. Every artifact's content hash is unchanged, every reported
number is bit-identical, and nothing on disk is a pickle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from axiom.calibrate import CalibratedSpec, Ledger, Measurement, derive_prior
from axiom.core import Intervention, Population, Posterior, TimeWindow
from axiom.data import Panel
from axiom.design import DesignCandidate, SimulationSpec
from axiom.estimands import Estimand, EstimandResult, Level, Quantity, RealizedDraws, realize
from axiom.identify import CausalGraph, identify
from axiom.io import Analysis, load_analysis, save_analysis
from axiom.meta import Corpus, PooledEstimate, StudyRecord, random_effects
from axiom.sim import surface_world
from axiom.surface import FitResult, HillKernel, SurfaceSpec, fit

PICKLE_SUFFIXES = {".pkl", ".pickle", ".dill", ".joblib"}


def _fit_world() -> tuple[Panel, SurfaceSpec, FitResult, Estimand, EstimandResult, Measurement]:
    world = surface_world(
        n_units=3,
        n_periods=12,
        treatments=("a",),
        kernels=HillKernel(reference_dose=50.0, amplitude_scale=10.0),
        intercept="shared",
        truth={"beta_a": 10.0, "k_a": 50.0, "s_a": 2.0, "alpha": 5.0},
        noise_sd=2.0,
        seed=0,
    )
    result = fit(world.spec, world.panel, backend="laplace", draws=200, seed=1)
    assert isinstance(result.posterior, Posterior)
    estimand = Estimand(
        name="lift_100_vs_0",
        quantity=Quantity(kind="contrast"),
        treatment=world.spec.treatment("a"),
        intervention=Intervention(doses={"a": 100.0}),
        reference=Intervention(doses={"a": 0.0}),
        outcome=world.spec.outcome,
        population=Population(name="panel_units"),
        window=TimeWindow(start=0, stop=world.n_periods, basis="cumulative"),
        level=Level(unit="individual"),
        dimension=world.spec.outcome_dimension,
    )
    verdict = identify(CausalGraph.from_edges("a -> y"), "a", "y").verdict
    realized = realize(estimand, result, verdict=verdict, keep_draws=True, seed=0)
    assert isinstance(realized, RealizedDraws)
    truth = float(np.mean(np.sum(world.forward({"a": 100.0}) - world.forward({"a": 0.0}), axis=1)))
    measurement = Measurement(
        estimand=estimand,
        estimate=truth,
        se=0.02 * truth,
        method="randomized_contrast",
        n_units=20,
        n_periods=12,
        source="rct-roundtrip",
    )
    return world.panel, world.spec, result, estimand, realized.result, measurement


@pytest.fixture(scope="module")
def analysis() -> tuple[Analysis, dict[str, float]]:
    panel, spec, result, estimand, estimand_result, measurement = _fit_world()
    assert isinstance(result.posterior, Posterior)

    graph = CausalGraph.from_edges("z -> a, z -> y, a -> y", name="roundtrip")

    calibrated = derive_prior(
        [measurement],
        spec,
        "a",
        beta_draws=result.posterior.flat("beta_a"),
        contribution_draws=np.full(result.posterior.n_draws(), measurement.estimate),
    )
    assert isinstance(calibrated, CalibratedSpec)
    ledger = Ledger(lines=calibrated.ledger_lines)

    design_sim = SimulationSpec(
        n_units=16, n_periods=12, n_pre=6, n_treated=8, noise_sd=1.0, n_simulations=20, seed=11
    )
    design_candidate = DesignCandidate(
        name="small_holdout",
        method="difference_in_differences",
        n_units=20,
        n_periods=6,
        holdout_fraction=0.2,
        experiment_se=0.35,
        cost=200.0,
        cooldown_periods=2,
    )

    corpus = Corpus(
        name="roundtrip",
        records=tuple(
            StudyRecord(
                study=f"s{i}",
                contributor=f"c{i}",
                quantity="elasticity",
                estimate=y,
                se=s,
                read="experiment" if i % 2 else "model",
                family="fertilizer",
            )
            for i, (y, s) in enumerate([(0.42, 0.10), (0.55, 0.15), (0.31, 0.12), (0.67, 0.20)])
        ),
    )
    pooled = random_effects(*corpus.arrays(), tau_method="reml")
    assert isinstance(pooled, PooledEstimate)

    built = (
        Analysis(
            specs={
                "graph": graph,
                "estimand:lift": estimand,
                "result:lift": estimand_result,
                "surface": spec,
                "surface:calibrated": calibrated.spec,
                "calibration": calibrated,
                "ledger": ledger,
                "design:simulation": design_sim,
                "design:candidate": design_candidate,
                "meta:corpus": corpus,
                "meta:pooled": pooled,
            }
        )
        .with_panel(panel)
        .with_posterior(result.posterior)
        .with_evidence(measurement)
        .with_ledger_line(*ledger.lines)
    )
    numbers = {
        "estimand_estimate": estimand_result.summary.mean,
        "estimand_lower": estimand_result.summary.interval.lower,
        "pooled_estimate": pooled.estimate,
        "pooled_se": pooled.se,
        "posterior_beta_mean": float(result.posterior.flat("beta_a").mean()),
        "posterior_beta_hdi_lower": result.posterior.summary("beta_a").interval.lower,
        "calibrated_amplitude_mean": calibrated.amplitude_mean,
    }
    return built, numbers


ARTIFACT_ROLES = (
    "graph",
    "estimand:lift",
    "result:lift",
    "surface",
    "surface:calibrated",
    "calibration",
    "ledger",
    "design:simulation",
    "design:candidate",
    "meta:corpus",
    "meta:pooled",
)


def test_every_artifact_kind_is_present(analysis: tuple[Analysis, dict[str, float]]) -> None:
    built, _ = analysis
    assert set(built.specs) == set(ARTIFACT_ROLES)
    assert built.panel is not None and built.posterior is not None
    assert len(built.evidence) == 1 and isinstance(built.evidence[0], Measurement)
    assert built.ledger


def test_roundtrip_preserves_hashes_and_numbers(
    analysis: tuple[Analysis, dict[str, float]], tmp_path: Path
) -> None:
    built, numbers = analysis
    root = tmp_path / "roundtrip.axiom"
    saved = save_analysis(built, root, seed=1)
    loaded = load_analysis(root)

    assert loaded == saved
    assert loaded.hashes() == saved.hashes() == built.hashes()
    for role in ARTIFACT_ROLES:
        assert loaded.spec(role).content_hash() == built.spec(role).content_hash(), role
        assert type(loaded.spec(role)) is type(built.spec(role)), role
    assert loaded.evidence[0].content_hash() == built.evidence[0].content_hash()
    assert loaded.panel is not None and built.panel is not None
    assert loaded.panel.content_hash() == built.panel.content_hash()

    assert isinstance(loaded.posterior, Posterior) and isinstance(built.posterior, Posterior)
    result = loaded.spec("result:lift")
    pooled = loaded.spec("meta:pooled")
    calibrated = loaded.spec("calibration")
    assert isinstance(result, EstimandResult)
    assert isinstance(pooled, PooledEstimate)
    assert isinstance(calibrated, CalibratedSpec)
    reloaded = {
        "estimand_estimate": result.summary.mean,
        "estimand_lower": result.summary.interval.lower,
        "pooled_estimate": pooled.estimate,
        "pooled_se": pooled.se,
        "posterior_beta_mean": float(loaded.posterior.flat("beta_a").mean()),
        "posterior_beta_hdi_lower": loaded.posterior.summary("beta_a").interval.lower,
        "calibrated_amplitude_mean": calibrated.amplitude_mean,
    }
    for key, value in numbers.items():
        assert reloaded[key] == value, (key, reloaded[key], value)  # bit-identical, no tolerance
    for name in loaded.posterior.names():
        assert np.array_equal(loaded.posterior.draws(name), built.posterior.draws(name))


def test_no_pickle_in_the_directory(
    analysis: tuple[Analysis, dict[str, float]], tmp_path: Path
) -> None:
    built, _ = analysis
    root = tmp_path / "nopickle.axiom"
    save_analysis(built, root)
    files = [p for p in root.rglob("*") if p.is_file()]
    assert files
    assert not [p for p in files if p.suffix in PICKLE_SUFFIXES]
    for p in files:
        head = p.read_bytes()[:2]
        assert head != b"\x80\x04" and head != b"\x80\x05", f"{p} starts like a pickle stream"
    # the npz holds only ndarray members, readable with allow_pickle=False
    with np.load(root / "posterior.npz", allow_pickle=False) as z:
        assert set(z.files) >= {"__meta__", "beta_a"}
