"""Generate every number and every chart series shown on the axiom site.

Nothing on the site is typed by hand: this script runs axiom, and writes what it
gets back to ``site/assets/data/*.json``. The pages read those files. If a claim
on a page has a number in it, the number came from here, which means it came from
a real run.

Usage::

    python site/_gen/generate.py            # everything
    python site/_gen/generate.py identify   # one section, while iterating
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site" / "assets" / "data"
SEED = 0

SECTIONS: dict[str, Any] = {}


def section(fn: Any) -> Any:
    SECTIONS[fn.__name__] = fn
    return fn


def jsonable(obj: Any) -> Any:
    """numpy and friends -> plain JSON."""
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else round(v, 6)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(x) for x in obj]
    return obj


def write(name: str, payload: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(jsonable(payload), indent=1) + "\n")
    print(f"  wrote {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.1f} kB)")


# ----------------------------------------------------------------------------------
# overview: what is actually in the package
# ----------------------------------------------------------------------------------


@section
def overview() -> Any:
    import importlib
    import subprocess

    order = (
        "core data io infer identify estimands surface sim design calibrate meta "
        "diagnose build adapters viz report"
    ).split()
    blurb = {
        "core": "Specs, dimensions, intervals, expressions — the vocabulary "
        "everything else is written in.",
        "data": "Panels and role maps: which column is the treatment, which is the outcome.",
        "io": "Save and load a whole analysis as JSON plus a posterior file. Never a pickle.",
        "infer": "The sampler seam. Laplace in core; NumPyro and PyMC behind extras.",
        "identify": "Causal graphs, back-door / front-door / IV routes, transport verdicts.",
        "estimands": "Declare the number you want as a nine-facet object, then realize it.",
        "surface": "Dose-response kernels, carryover, the one forward(), optimal allocation.",
        "sim": "Synthetic worlds with known truth, so every claim has a recovery test.",
        "design": "Power, MDE, value of information, method leaderboards, sequential boundaries.",
        "calibrate": "Fold a randomized result into an observational model, and log "
        "what that assumed.",
        "meta": "Pool a corpus of studies; heterogeneity, bias terms, privacy-budgeted release.",
        "diagnose": "SBC, coverage, posterior predictive checks, refutations, "
        "specification curves.",
        "build": "Fluent builders for graphs, priors, and corpora.",
        "adapters": "Domain vocabulary — marketing is one adapter, not the core.",
        "viz": "Plot helpers for the diagnostics that have a canonical picture.",
        "report": "Templated HTML / PPTX / PDF reports over the viz layer.",
    }
    layer = {
        "core": "foundation",
        "data": "foundation",
        "io": "foundation",
        "infer": "sampler seam",
        "identify": "domain",
        "estimands": "domain",
        "surface": "domain",
        "sim": "domain",
        "design": "pillar",
        "calibrate": "pillar",
        "meta": "pillar",
        "diagnose": "composition",
        "build": "composition",
        "adapters": "composition",
        "viz": "composition",
        "report": "composition",
    }

    packages = []
    total = 0
    for name in order:
        mod = importlib.import_module("axiom." + name)
        symbols = sorted(getattr(mod, "__all__", []))
        total += len(symbols)
        packages.append(
            {
                "name": name,
                "n": len(symbols),
                "symbols": symbols,
                "blurb": blurb[name],
                "layer": layer[name],
            }
        )

    # How heavy is the import, really? `import axiom` alone is a no-op (the top-level
    # __init__ is empty), so the honest measurement is the set of pillars you would
    # actually use for design, identification, surfaces, and meta-analysis. The claim
    # the site makes is that none of them drag a sampler in.
    workload = "import axiom.identify, axiom.design, axiom.surface, axiom.meta, axiom.estimands"
    code = (
        "import sys, json;"
        "before=set(sys.modules);"
        f"{workload};"
        "new={m.split('.')[0] for m in set(sys.modules)-before};"
        "heavy={'jax','jaxlib','numpyro','pymc','pytensor','arviz','torch','tensorflow',"
        "'matplotlib','plotly','scipy','numpy','pandas','pydantic'};"
        "print(json.dumps(sorted(new & heavy)))"
    )
    imported = json.loads(
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout
    )

    runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        subprocess.run([sys.executable, "-c", workload], check=True, capture_output=True)
        runs.append(time.perf_counter() - t0)
    import_seconds = min(runs)

    return {
        "packages": packages,
        "total_symbols": total,
        "n_packages": len(packages),
        "import_workload": workload,
        "import_pulls": imported,
        "samplers_pulled": [m for m in imported if m in {"jax", "numpyro", "pymc", "pytensor"}],
        "import_seconds": round(import_seconds, 2),
    }


# ----------------------------------------------------------------------------------
# identify: can the data answer the question at all?
# ----------------------------------------------------------------------------------


@section
def identify() -> Any:
    from axiom.diagnose import bias_bounds, robustness_value, tipping_point
    from axiom.identify import (
        CausalGraph,
        identify,
        minimal_adjustment_sets,
        ols,
        two_stage_least_squares,
        weak_instrument_check,
    )
    from axiom.sim import confounded_world, hidden_confounder_world, iv_world

    N = 5_000
    out: dict[str, Any] = {}

    # --- 1. the effect is identified, and adjustment recovers it -------------------
    world = confounded_world()
    truth = world.total_effect("X", "Y")
    v = identify(world.graph, "X", "Y")
    frame = world.observed(world.simulate(N, seed=SEED))
    naive = ols(frame, "Y", "X")
    adjusted = ols(frame, "Y", "X", covariates=v.adjustment_set)

    def est(label: str, e: Any) -> dict[str, Any]:
        ci = e.ci(0.95)
        return {
            "label": label,
            "estimate": e.estimate,
            "se": e.se,
            "lower": ci.lower,
            "upper": ci.upper,
            "error": e.estimate - truth,
            "method": e.method,
            "covariates": list(e.covariates),
        }

    out["backdoor"] = {
        "graph": world.graph.to_text(),
        "measured": sorted(world.graph.measured),
        "truth": truth,
        "status": v.status,
        "route": v.route,
        "adjustment_set": sorted(v.adjustment_set),
        "minimal_sets": [sorted(s) for s in minimal_adjustment_sets(world.graph, "X", "Y")],
        "assumptions": [{"name": a.name, "state": a.state} for a in v.verdict.assumptions],
        "estimates": [est("naive (no adjustment)", naive), est("back-door adjusted", adjusted)],
    }

    # --- 2. it is not identified, and axiom says so before you fit anything --------
    hidden = hidden_confounder_world()
    vh = identify(hidden.graph, "X", "Y")
    frame_h = hidden.observed(hidden.simulate(N, seed=SEED))
    naive_h = ols(frame_h, "Y", "X")
    blocked = CausalGraph.from_edges("X -> Y, X <-> Y")
    vb = identify(blocked, "X", "Y")
    out["not_identified"] = {
        "graph": hidden.graph.to_text(),
        "measured": sorted(hidden.graph.measured),
        "observed_columns": hidden.observed(hidden.simulate(3, seed=0)).columns.tolist(),
        "status": vh.status,
        "route": vh.route,
        "unmeasured_required": sorted(vh.unmeasured_required),
        "naive": est("naive (confounder unobserved)", naive_h),
        "truth": truth,
        "blocked": {"graph": blocked.to_text(), "status": vb.status, "reason": vb.verdict.reason},
    }

    # --- 3. an instrument rescues it ----------------------------------------------
    ivw = iv_world()
    vi = identify(ivw.graph, "X", "Y")
    frame_iv = ivw.observed(ivw.simulate(N, seed=SEED + 1))
    truth_iv = ivw.total_effect("X", "Y")
    naive_iv = ols(frame_iv, "Y", "X")
    iv = two_stage_least_squares(frame_iv, "Y", "X", instruments=[vi.instrument])
    relevance = weak_instrument_check(iv)

    def est_iv(label: str, e: Any) -> dict[str, Any]:
        ci = e.ci(0.95)
        return {
            "label": label,
            "estimate": e.estimate,
            "se": e.se,
            "lower": ci.lower,
            "upper": ci.upper,
            "error": e.estimate - truth_iv,
            "method": e.method,
        }

    out["instrument"] = {
        "graph": ivw.graph.to_text(),
        "status": vi.status,
        "route": vi.route,
        "instrument": vi.instrument,
        "truth": truth_iv,
        "estimates": [est_iv("naive OLS", naive_iv), est_iv("2SLS", iv)],
        "first_stage_f": iv.detail["first_stage_f"],
        "relevance": {"name": relevance.name, "state": relevance.state},
    }

    # --- 4. how strong would a hidden confounder have to be? ----------------------
    df = naive_h.n - 2
    rv = robustness_value(estimate=naive_h.estimate, se=naive_h.se, df=df, q=1.0, alpha=0.05)
    q_true = (naive_h.estimate - truth) / naive_h.estimate
    rv_q = robustness_value(estimate=naive_h.estimate, se=naive_h.se, df=df, q=q_true, alpha=0.05)

    # A contour of the adjusted estimate over confounder strength, for the heatmap.
    grid = np.linspace(0.0, 0.5, 21)
    contour = []
    for ry in grid:
        row = []
        for rd in grid:
            b = bias_bounds(
                estimate=naive_h.estimate,
                se=naive_h.se,
                df=df,
                r2_yz_dx=float(ry),
                r2_dz_x=float(rd),
                mass=0.95,
            )
            row.append(b.adjusted_estimate)
        contour.append(row)

    r2_dz_x = 0.64 / 1.64
    resid_z = 1.5**2 * (1.0 - r2_dz_x)
    r2_yz_dx = resid_z / (resid_z + 1.0)
    bounds_rows = []
    for label, rd, ry in (
        ("the confounder that is actually there", r2_dz_x, r2_yz_dx),
        ("half as strong", r2_dz_x / 2, r2_yz_dx / 2),
        ("a quarter as strong", r2_dz_x / 4, r2_yz_dx / 4),
    ):
        b = bias_bounds(
            estimate=naive_h.estimate, se=naive_h.se, df=df, r2_yz_dx=ry, r2_dz_x=rd, mass=0.95
        )
        bounds_rows.append(
            {
                "label": label,
                "r2_dz_x": rd,
                "r2_yz_dx": ry,
                "bias": b.bias,
                "adjusted": b.adjusted_estimate,
                "lower": b.adjusted_interval.lower,
                "upper": b.adjusted_interval.upper,
            }
        )

    rng = np.random.default_rng(SEED)
    draws = rng.normal(naive_h.estimate, naive_h.se, size=4000)
    tp = tipping_point(draws, 1.0, np.linspace(0.0, 2.5, 26), certainty=0.9)

    out["sensitivity"] = {
        "estimate": naive_h.estimate,
        "se": naive_h.se,
        "truth": truth,
        "actual_bias": naive_h.estimate - truth,
        "rv": rv.rv,
        "rv_alpha": rv.rv_alpha,
        "r2_yd_x": rv.r2_yd_x,
        "q_true": q_true,
        "rv_q": rv_q.rv,
        "grid": grid.tolist(),
        "contour": contour,
        "bounds": bounds_rows,
        "tipping": {
            "decision_at_zero": tp.decision_at_zero,
            "probability_at_zero": tp.probability_at_zero,
            "bias": tp.bias,
            "certainty": 0.9,
            "threshold": 1.0,
            "interval": [tp.interval.lower, tp.interval.upper],
        },
    }
    return out


# ----------------------------------------------------------------------------------
# design: what is worth measuring next, and how big does it have to be?
# ----------------------------------------------------------------------------------


@section
def design() -> Any:
    from axiom.core import Population, TimeWindow
    from axiom.design import (
        DecisionSpec,
        DesignCandidate,
        EconomicInputs,
        ValuePerOutcome,
        anchor_draws,
        difference_se,
        eig_gaussian,
        evaluate_candidate,
        evoi_gaussian,
        mde,
        pareto_front,
        perturb,
        power,
        power_curve,
        sample_size,
    )
    from axiom.estimands import Level, realize, standard_estimands
    from axiom.identify import CausalGraph, identify
    from axiom.sim import arms_world
    from axiom.surface import HillKernel, fit

    out: dict[str, Any] = {}

    # A pilot: 60 units across the dose range, with the outcome noise a real programme
    # would actually have. The quieter worlds used in the unit tests make the sample-size
    # arithmetic degenerate (n in the single digits), which is true but tells you nothing.
    prior_world = arms_world(
        n_units=60,
        treatments=("a",),
        kernels=HillKernel(reference_dose=50.0, amplitude_scale=10.0),
        doses={"a": np.linspace(0.0, 100.0, 60)},
        truth={"beta_a": 6.0, "k_a": 50.0, "s_a": 2.0, "alpha": 1.0},
        noise_sd=6.0,
        seed=SEED,
    )
    res = fit(prior_world.spec, prior_world.panel, backend="laplace", draws=2000, seed=SEED)
    sd = float(res.posterior.summary("sigma").mean)

    registry = standard_estimands(
        treatment=prior_world.spec.treatments[0],
        outcome=prior_world.spec.outcome,
        population=Population(name="all"),
        window=TimeWindow(start=0, stop=1),
        level=Level(unit="individual"),
        dose=60.0,
        reference_dose=0.0,
    )
    verdict = identify(CausalGraph.from_edges("a -> y"), "a", "y").verdict
    rd = realize(registry.get("contrast_at_dose"), res, verdict=verdict, keep_draws=True, mass=0.9)
    contrast_draws = np.asarray(rd.draws).ravel()
    prior_mean, prior_sd = float(contrast_draws.mean()), float(contrast_draws.std())
    truth_contrast = float(
        prior_world.forward({"a": 60.0}).mean() - prior_world.forward({"a": 0.0}).mean()
    )

    MDE_CONVENTIONAL = 2.0
    ae = anchor_draws(contrast_draws, "contrast_at_dose", mde=MDE_CONVENTIONAL, credence=0.9)
    effect = ae.anchored_effect if ae.already_believed else MDE_CONVENTIONAL

    out["prior"] = {
        "mean": prior_mean,
        "sd": prior_sd,
        "truth": truth_contrast,
        "interval": [rd.result.summary.interval.lower, rd.result.summary.interval.upper],
        "outcome_sd": sd,
        "mde_conventional": MDE_CONVENTIONAL,
        "probability_exceeds_mde": ae.probability_exceeds_mde,
        "already_believed": bool(ae.already_believed),
        "anchored_effect": ae.anchored_effect,
        "effect_powered_for": effect,
        # the histogram of what we already believe about the contrast
        "draws_hist": np.histogram(contrast_draws, bins=36)[0].tolist(),
        "draws_edges": np.histogram(contrast_draws, bins=36)[1].tolist(),
    }

    ss = sample_size(effect=effect, sd=sd, power=0.8, alpha=0.05)
    proposed_n = 80  # what the team proposed before anyone did the arithmetic
    m = mde(proposed_n, sd=sd, power=0.8)
    se_experiment = difference_se(ss.n, sd=sd, allocation=ss.n_treated / ss.n)

    effects = np.linspace(0.2, 8.0, 60)
    curves = []
    # well-separated sizes, so the curves are distinguishable rather than overlapping
    for n in (20, ss.n, proposed_n, 200):
        pc = power_curve(int(n), sd=sd, effects=effects.tolist())
        curves.append({"n": int(n), "effects": effects.tolist(), "power": list(pc.powers)})

    out["sizing"] = {
        "effect": effect,
        "sd": sd,
        "n": ss.n,
        "n_treated": ss.n_treated,
        "n_control": ss.n_control,
        "power": ss.power,
        "proposed_n": proposed_n,
        "mde_at_proposed": m.effect,
        "power_at_proposed": power(proposed_n, effect, sd).power,
        "se_experiment": se_experiment,
        "curves": curves,
    }

    # --- what is the experiment worth? -------------------------------------------
    # The decision is genuinely on a knife edge: the programme scales up only if the
    # contrast clears 6.0, and the prior sits just under that with real spread. This is
    # the only situation in which an experiment can be worth its cost — if the prior
    # already settled the decision, the value of information is near zero by construction.
    VALUE_PER_OUTCOME, PROGRAM_UNITS = 400.0, 500
    decision = DecisionSpec(
        name="scale_up",
        threshold=6.0,
        value_per_outcome_unit=VALUE_PER_OUTCOME * PROGRAM_UNITS,
        numeraire="USD",
    )
    ev = evoi_gaussian(decision, prior_mean, prior_sd, se_experiment)
    out["value"] = {
        "eig_nats": eig_gaussian(prior_sd, se_experiment),
        "evpi": ev.evpi,
        "evsi": ev.evsi,
        "preposterior_sd": ev.preposterior_sd,
        "numeraire": ev.numeraire,
        "threshold": decision.threshold,
        "value_per_outcome": VALUE_PER_OUTCOME,
        "program_units": PROGRAM_UNITS,
        # EVSI as the experiment gets more precise
        "se_grid": np.linspace(0.1, 2.5, 40).tolist(),
        "evsi_grid": [
            evoi_gaussian(decision, prior_mean, prior_sd, float(s)).evsi
            for s in np.linspace(0.1, 2.5, 40)
        ],
        "evpi_line": ev.evpi,
    }

    vpo = ValuePerOutcome(
        value=VALUE_PER_OUTCOME,
        outcome_unit="unit",
        numeraire="USD",
        source="stated by the decision owner, 2026",
    )
    ratio_draws = contrast_draws / 60.0
    economics = EconomicInputs(
        value_per_outcome=vpo,
        dose_per_period=60.0,
        discount_rate=0.01,
        dose_unit="USD",
        dose_cost_per_unit=1.0,
        marginal_value_ratio=float(ratio_draws.mean()),
    )
    candidates = [
        DesignCandidate(
            name="powered_holdout",
            method="difference_in_differences",
            n_units=ss.n,
            n_periods=12,
            holdout_fraction=ss.n_control / ss.n,
            experiment_se=se_experiment,
            cost=300.0,
            cooldown_periods=2,
        ),
        DesignCandidate(
            name="proposed_holdout",
            method="difference_in_differences",
            n_units=proposed_n,
            n_periods=12,
            holdout_fraction=0.5,
            experiment_se=difference_se(proposed_n, sd=sd),
            cost=550.0,
            cooldown_periods=2,
        ),
        DesignCandidate(
            name="switchback",
            method="switchback",
            n_units=ss.n,
            n_periods=16,
            holdout_fraction=0.5,
            experiment_se=0.8 * se_experiment,
            cost=420.0,
            cooldown_periods=1,
        ),
    ]
    scores = [
        evaluate_candidate(
            c, decision, prior_mean=prior_mean, prior_sd=prior_sd, economics=economics
        )
        for c in candidates
    ]
    front = {s.name for s in pareto_front(scores, objectives=("net_value", "-cost", "eig"))}
    out["leaderboard"] = {
        "rows": [
            {
                "name": s.name,
                "eig": s.eig,
                "evsi": s.evsi,
                "opportunity_cost": s.opportunity_cost,
                "cost": s.cost,
                "net_value": s.net_value,
                "power": s.power,
                "on_front": s.name in front,
            }
            for s in scores
        ],
        "winner": max(scores, key=lambda s: s.net_value).name,
        "ledger_line": vpo.ledger_line().statement,
    }

    sens = []
    for parameter, grid in (
        ("value_per_outcome", (0.25, 0.5, 1.0, 2.0, 4.0)),
        ("experiment_se", (0.5, 0.75, 1.0, 1.5, 2.0)),
    ):
        table = perturb(candidates, decision, prior_mean, prior_sd, economics, parameter, grid=grid)
        sens.append(
            {
                "parameter": parameter,
                "mode": table.mode,
                "grid": list(grid),
                "winners": list(table.winners),
                "base_winner": table.base_winner,
                "stable": bool(table.stable),
                "tipping_points": jsonable(table.tipping_points),
            }
        )
    out["sensitivity"] = sens
    return out


# ----------------------------------------------------------------------------------
# surface: what does the dose-response look like, and where is the optimum?
# ----------------------------------------------------------------------------------


@section
def surface() -> Any:
    from axiom.diagnose import posterior_predictive, weak_identification
    from axiom.sim import arms_world
    from axiom.surface import (
        Bounds,
        allocate,
        canonical_analysis,
        central_composite,
        fit,
        frontier,
        marginal_band,
        response_band,
        steepest_ascent,
    )

    out: dict[str, Any] = {}
    bounds = Bounds(treatments=("a", "b"), low=(0.0, 0.0), high=(100.0, 40.0))
    ccd = central_composite(bounds, alpha="rotatable", center_points=4, inscribed=True)
    truth = {
        "alpha": 2.0,
        "beta_a": 10.0,
        "k_a": 50.0,
        "s_a": 2.0,
        "beta_b": 6.0,
        "k_b": 12.0,
        "s_b": 1.5,
    }
    from axiom.surface import HillKernel

    arms = arms_world(
        n_units=ccd.n,
        treatments=("a", "b"),
        kernels={
            "a": HillKernel(reference_dose=50.0, amplitude_scale=10.0),
            "b": HillKernel(reference_dose=15.0, amplitude_scale=10.0),
        },
        doses=ccd.doses(),
        truth=truth,
        noise_sd=0.5,
        seed=SEED,
    )
    res = fit(arms.spec, arms.panel, backend="laplace", draws=2000, seed=SEED)
    post = res.posterior
    theta_hat = {name: float(post.summary(name).mean) for name in post.names()}
    surf = res.surface

    out["design"] = {
        "kind": ccd.kind,
        "n": ccd.n,
        "detail": jsonable(dict(ccd.detail)),
        "points": [list(map(float, p)) for p in ccd.points],
        "treatments": list(bounds.treatments),
        "low": list(bounds.low),
        "high": list(bounds.high),
    }
    recovery = []
    for n in ("alpha", "beta_a", "k_a", "s_a", "beta_b", "k_b", "s_b", "sigma"):
        s = post.summary(n, definition="hdi", mass=0.9)
        t = truth.get(n, float(arms.noise.std()))
        inside = bool(s.interval.lower <= t <= s.interval.upper)
        recovery.append(
            {
                "name": n,
                "truth": t,
                "mean": s.mean,
                "sd": post.summary(n).sd,
                "lower": s.interval.lower,
                "upper": s.interval.upper,
                # decided here rather than typed into the page, so the table cannot lie
                "contains": inside,
                "contains_label": "yes" if inside else "no",
                "cell_class": "num-good" if inside else "num-bad",
                "row_class": "" if inside else "bad",
            }
        )
    out["recovery"] = recovery
    out["recovery_hits"] = sum(r["contains"] for r in recovery)
    out["recovery_n"] = len(recovery)

    # --- the curve with its uncertainty band, against the truth -------------------
    # The truth line has to be built the way response_band builds its own grid:
    # the named treatment is set to each dose and every *other* treatment keeps its
    # observed doses. Holding the others at zero instead would put the two curves on
    # different counterfactuals and make the comparison meaningless.
    from axiom.core import Intervention
    from axiom.surface import counterfactual_doses

    def truth_at(name: str, dose: float) -> float:
        data = counterfactual_doses(arms.surface, arms.data, Intervention(doses={name: dose}))
        return float(arms.surface.forward(data, arms.theta).mean())

    refusals = []
    for name, hi in (("a", 100.0), ("b", 40.0)):
        doses = np.linspace(0.0, hi, 40)
        band = response_band(res, name, doses=doses.tolist(), mass=0.9, seed=SEED)
        true_curve = [truth_at(name, float(d)) for d in doses]
        base = truth_at(name, 0.0)
        entry = {
            "treatment": name,
            "doses": doses.tolist(),
            "mean": list(band.mean),
            "lower": list(band.lower),
            "upper": list(band.upper),
            "mass": 0.9,
            "truth": true_curve,
            "truth_base": base,
        }

        # The slope of a Hill curve with shape s < 1 is infinite at zero dose, so the
        # marginal is genuinely undefined there. axiom returns a typed failure saying so
        # rather than handing back an inf. Retry off zero, and keep the refusal —
        # it is worth showing.
        marg = marginal_band(res, name, doses=doses.tolist(), mass=0.9, seed=SEED)
        if hasattr(marg, "reason"):
            refusals.append(
                {
                    "call": f"marginal_band(result, {name!r}, doses=0..{hi:g})",
                    "type": type(marg).__name__,
                    "reason": marg.reason,
                }
            )
            m_doses = np.linspace(hi / 40.0, hi, 40)
            marg = marginal_band(res, name, doses=m_doses.tolist(), mass=0.9, seed=SEED)
        else:
            m_doses = doses
        if not hasattr(marg, "reason"):
            entry.update(
                {
                    "marginal_doses": m_doses.tolist(),
                    "marginal_mean": list(marg.mean),
                    "marginal_lower": list(marg.lower),
                    "marginal_upper": list(marg.upper),
                }
            )
        out[f"band_{name}"] = entry
    out["refusals"] = refusals

    # --- where is the optimum, and what does a budget buy ------------------------
    centre = {"a": 50.0, "b": 20.0}
    sp = canonical_analysis(surf, theta_hat, centre)
    path = steepest_ascent(
        surf, theta_hat, {"a": 10.0, "b": 2.0}, step=5.0, n_steps=30, bounds=bounds
    )
    out["ascent"] = {
        "stationary_kind": getattr(sp, "kind", None),
        "stationary_point": jsonable(getattr(sp, "point", None)),
        "n_steps": path.n,
        "stop": path.stop,
        "best": jsonable(path.best()),
        "points": (
            [[float(p["a"]), float(p["b"])] for p in path.points]
            if isinstance(path.points[0], dict)
            else jsonable(path.points)
        ),
        "values": list(path.values),
    }

    BUDGET = 60.0
    alloc = allocate(surf, post, budget=BUDGET, bounds=bounds, objective="mean", seed=SEED)
    oracle = allocate(
        arms.surface, arms.theta, budget=BUDGET, bounds=bounds, objective="mean", seed=SEED
    )
    true_at_fitted = float(arms.forward({k: float(v) for k, v in alloc.doses.items()}).mean())
    budgets = [10.0, 20.0, 40.0, 60.0, 90.0, 120.0, 140.0]
    fr = frontier(surf, theta_hat, budgets=budgets, bounds=bounds, seed=SEED)
    frame = fr.as_frame()
    out["allocation"] = {
        "budget": BUDGET,
        "fitted": {k: float(v) for k, v in alloc.doses.items()},
        "fitted_outcome": alloc.expected_outcome,
        "status": alloc.status,
        "oracle": {k: float(v) for k, v in oracle.doses.items()},
        "oracle_outcome": oracle.expected_outcome,
        "true_at_fitted": true_at_fitted,
        "regret": oracle.expected_outcome - true_at_fitted,
        "frontier": {
            "budgets": budgets,
            "columns": list(frame.columns),
            "rows": jsonable(frame.to_dict("records")),
            "shadow_prices": jsonable(fr.shadow_prices()),
        },
    }

    # --- a 2-D response surface for the heatmap ----------------------------------
    ga = np.linspace(0.0, 100.0, 36)
    gb = np.linspace(0.0, 40.0, 36)
    A, B = np.meshgrid(ga, gb, indexing="xy")
    z = [
        [float(arms.surface.forward({"a": float(a), "b": float(b)}, theta_hat).mean()) for a in ga]
        for b in gb
    ]
    out["heatmap"] = {"a": ga.tolist(), "b": gb.tolist(), "z": z}

    ppc = posterior_predictive(res, n_draws=100, seed=SEED)
    rep = weak_identification(res, rho_threshold=0.9)
    out["checks"] = {
        "ppc": [
            {
                "name": s.name,
                "observed": s.observed,
                "lower": s.interval.lower,
                "upper": s.interval.upper,
                "p": s.p_two_sided,
                "extreme": bool(s.extreme),
            }
            for s in ppc.statistics
        ],
        "extreme": list(ppc.extreme_statistics),
        "condition_number": rep.condition_number,
        "high_pairs": [[p, q, float(r)] for p, q, r in rep.high_pairs],
        "unlearned": list(rep.unlearned),
        "saturated": list(rep.saturated),
        "passed": bool(rep.passed),
    }
    return out


# ----------------------------------------------------------------------------------
# calibrate: fold an experiment into an observational model
# ----------------------------------------------------------------------------------


@section
def calibrate() -> Any:
    from axiom.calibrate import (
        aggregation_level,
        agreement,
        carryover_window_factor,
        derive_prior,
        fit_calibrated,
        resolve,
    )
    from axiom.core import Intervention, Population, TimeWindow
    from axiom.data import Panel
    from axiom.estimands import Estimand, Level, Quantity, realize
    from axiom.sim import DosePlan, surface_world
    from axiom.surface import GeometricCarryover, HillKernel, fit

    TRUTH = {"beta_a": 10.0, "alpha": 5.0, "k_a": 50.0, "s_a": 2.0, "lam_a": 0.5}
    HI, LO, MAX_LAG = 100.0, 0.0, 4
    out: dict[str, Any] = {}

    world = surface_world(
        n_units=4,
        n_periods=24,
        treatments=("a",),
        kernels=HillKernel(reference_dose=50.0, amplitude_scale=10.0),
        carryover={"a": GeometricCarryover(max_lag=MAX_LAG)},
        doses=DosePlan(scale=50.0, spread=0.8, zero_fraction=0.05),
        intercept="shared",
        truth=TRUTH,
        noise_sd=2.0,
        seed=0,
    )
    spec = world.spec
    rng = np.random.default_rng(123)
    frame = world.panel.frame
    dose = frame["a"].to_numpy(dtype=np.float64)
    u = (dose - dose.mean()) / dose.std() + 0.3 * rng.standard_normal(dose.size)
    observed = Panel(frame.assign(y=frame["y"] + 1.0 * u), world.panel.roles)

    def contrast(name: str, window: TimeWindow, level: Level) -> Estimand:
        return Estimand(
            name=name,
            quantity=Quantity(kind="contrast"),
            treatment=spec.treatment("a"),
            intervention=Intervention(doses={"a": HI}),
            reference=Intervention(doses={"a": LO}),
            outcome=spec.outcome,
            population=Population(name="panel_units"),
            window=window,
            level=level,
            dimension=spec.outcome_dimension,
        )

    experiment_estimand = contrast(
        "first_period_lift",
        TimeWindow(start=0, stop=1, basis="cumulative"),
        Level(unit="individual"),
    )
    decision_estimand = contrast(
        "steady_state_lift",
        TimeWindow(start=MAX_LAG, stop=world.n_periods, basis="per_period"),
        Level(unit="cluster"),
    )

    diff = world.forward({"a": HI}) - world.forward({"a": LO})
    truth_experiment = float(np.mean(diff[:, 0]))
    truth_decision = float(np.sum(np.mean(diff[:, MAX_LAG:], axis=1)))
    se = 0.01 * truth_experiment
    from axiom.calibrate import Measurement

    measurement = Measurement(
        estimand=experiment_estimand,
        estimate=truth_experiment + se * float(np.random.default_rng(7).standard_normal()),
        se=se,
        method="randomized_contrast",
        n_units=40,
        n_periods=1,
        source="rct-2026Q1",
    )

    out["setup"] = {
        "confounding_correlation": float(np.corrcoef(dose, u)[0, 1]),
        "truth": TRUTH,
        "truth_experiment": truth_experiment,
        "truth_decision": truth_decision,
        "measurement": {
            "estimate": measurement.estimate,
            "se": measurement.se,
            "lower": measurement.interval.lower,
            "upper": measurement.interval.upper,
            "method": measurement.method,
            "source": measurement.source,
            "target_hash": measurement.target[:12],
        },
        "n_units": world.n_units,
        "n_periods": world.n_periods,
        "max_lag": MAX_LAG,
    }

    def moments(result: Any) -> dict[str, list[float]]:
        return {
            n: [float(result.posterior.flat(n).mean()), float(result.posterior.flat(n).std(ddof=1))]
            for n in ("beta_a", "k_a", "s_a", "lam_a", "alpha")
        }

    biased = fit(spec, observed, backend="laplace", draws=1000, seed=1)
    realized = realize(experiment_estimand, biased, assume_identified=True, keep_draws=True)
    beta_draws = biased.posterior.flat("beta_a")
    contribution_draws = np.asarray(realized.draws, dtype=np.float64).reshape(-1)
    calibrated = derive_prior(
        [measurement], spec, "a", beta_draws=beta_draws, contribution_draws=contribution_draws
    )
    prior_route = fit(calibrated.spec, observed, backend="laplace", draws=1000, seed=1)
    likelihood_route = fit_calibrated(
        spec, observed, [measurement], backend="laplace", draws=1000, seed=1
    )

    routes = []
    for label, result in (
        ("uncalibrated", biased),
        ("prior route", prior_route),
        ("likelihood route", likelihood_route),
    ):
        ag = agreement(result, measurement, seed=0)
        m = moments(result)
        routes.append(
            {
                "label": label,
                "moments": m,
                "beta_a": m["beta_a"][0],
                "beta_a_sd": m["beta_a"][1],
                "sd_above_truth": (m["beta_a"][0] - TRUTH["beta_a"]) / m["beta_a"][1],
                "realized": ag.posterior_mean,
                "posterior_sd": ag.posterior_sd,
                "z": ag.z,
                "verdict": ag.verdict,
            }
        )
    out["routes"] = {
        "rows": routes,
        "truth": TRUTH,
        "design_factor": calibrated.design_factor,
        "amplitude_mean": calibrated.amplitude_mean,
        "amplitude_sd": calibrated.amplitude_sd,
        "parameter": calibrated.parameter,
        "prior": str(calibrated.prior),
        "prior_ledger": [
            {"kind": ln.kind, "assumption": ln.assumption.name, "statement": ln.statement}
            for ln in calibrated.ledger_lines
        ],
        "route_provenance": jsonable(likelihood_route.provenance["route"]),
        "constraint": jsonable(likelihood_route.provenance["constraints"][0]["name"]),
    }

    # --- the transfer, and its ledger --------------------------------------------
    plan = experiment_estimand.transfer_to(decision_estimand)
    lam_hat = float(likelihood_route.posterior.flat("lam_a").mean())
    window = carryover_window_factor(
        GeometricCarryover(max_lag=MAX_LAG), {"lam_a": lam_hat}, 1, treatment="a"
    )
    level = aggregation_level(
        experiment_estimand.level, decision_estimand.level, cluster_size=world.n_units, icc=0.05
    )
    resolved = resolve(plan, corrections=[window, level])
    per_member, per_member_se = resolved.apply(measurement.estimate, measurement.se)
    decision_read = realize(decision_estimand, likelihood_route, assume_identified=True, seed=0)
    ledger = resolved.ledger
    lframe = ledger.to_frame()

    out["transfer"] = {
        "plan_status": plan.status,
        "differing": list(plan.differing),
        "source": experiment_estimand.name,
        "target": decision_estimand.name,
        "window_factor": window.value,
        "window_counterfactual": window.counterfactual,
        "lam_hat": lam_hat,
        "level_se_factor": level.value,
        "level_point_factor": jsonable(level.detail["point_factor"]),
        "status": resolved.status,
        "licensed": bool(resolved.licensed),
        "factor": resolved.factor,
        "se_scale": resolved.se_scale,
        "per_member": per_member,
        "per_member_se": per_member_se,
        "per_cluster": per_member * world.n_units,
        "per_cluster_se": per_member_se * world.n_units,
        "truth_decision": truth_decision,
        "model_read": decision_read.summary.mean,
        "model_lower": decision_read.summary.interval.lower,
        "model_upper": decision_read.summary.interval.upper,
        "ledger": jsonable(
            lframe[
                ["facet", "assumption", "state", "counterfactual", "value", "correction"]
            ].to_dict("records")
        ),
        "ledger_summary": ledger.summary(),
        "facets_covered": jsonable(ledger.facets_covered()),
        "complete": jsonable(ledger.check_complete(plan).status),
    }
    return out


# ----------------------------------------------------------------------------------
# meta: what does the whole body of evidence say?
# ----------------------------------------------------------------------------------


@section
def meta() -> Any:
    from axiom.build import MetaBuilder
    from axiom.estimands import TransferPlan
    from axiom.meta import (
        Corpus,
        EpsilonLedger,
        PoolPriors,
        PoolSpec,
        PrivacyPolicy,
        baujat,
        cell_from_records,
        check_cell,
        delta_identification,
        egger,
        forest_data,
        funnel_data,
        heterogeneity,
        leave_one_out,
        normalize,
        pool,
        prediction_interval,
        prior_from_pool,
        random_effects,
        record_from_summary,
        release,
    )

    out: dict[str, Any] = {}
    rng = np.random.default_rng(5)
    rows = [
        ("s01", "lab_a", "experiment", 0.62, 0.08, 120),
        ("s02", "lab_a", "model", 0.81, 0.06, 400),
        ("s03", "lab_b", "experiment", 0.55, 0.12, 60),
        ("s04", "lab_b", "model", 0.74, 0.07, 300),
        ("s05", "lab_c", "experiment", 0.70, 0.10, 90),
        ("s06", "lab_d", "experiment", 0.48, 0.15, 40),
        ("s07", "lab_e", "model", 0.77, 0.05, 500),
        ("s08", "lab_f", "experiment", 0.66, 0.09, 100),
        ("s09", "lab_g", "model", 0.85, 0.08, 250),
        ("s10", "lab_h", "experiment", 0.58, 0.11, 70),
        ("s11", "lab_i", "experiment", 0.64, 0.07, 150),
    ]
    mb = MetaBuilder().name("saturation-contrast").family("fertilizer")
    for study, lab, read, est, se_, n in rows:
        mb = mb.study(
            study,
            estimate=est,
            se=se_,
            read=read,
            contributor=lab,
            quantity="standardized_contrast",
            n=n,
            moderators={"follow_up": float(rng.integers(4, 13))},
            source=f"report-{study}",
        )
    eleven = mb.build()

    per_acre = record_from_summary(
        study="s12",
        contributor="lab_j",
        quantity="standardized_contrast",
        estimate=1.32,
        se=0.20,
        read="experiment",
        family="fertilizer",
        n=80,
        unit_scale="per_acre",
        moderators={"follow_up": 6.0},
        source="report-s12",
    )
    refused = normalize([*eleven.records, per_acre], name="twelve")
    out["refusal"] = {"reason": refused.reason, "missing": jsonable(refused.missing)}

    plan = TransferPlan(
        status="identified",
        source="per_acre",
        target="dimensionless",
        differing=(),
        entries=(),
        assumptions=(),
        ledger_lines=(),
    )
    admitted = normalize([*eleven.records, per_acre], plans={"s12": plan}, name="twelve")
    s12 = admitted.records[-1]
    REFERENCE_AREA = 2.0
    converted = s12.model_copy(
        update={
            "estimate": s12.estimate / REFERENCE_AREA,
            "se": s12.se / REFERENCE_AREA,
            "unit_scale": "",
            "detail": {**s12.detail, "converted_by": f"/ {REFERENCE_AREA} acre"},
        }
    )
    corpus = Corpus(records=(*admitted.records[:-1], converted), name="twelve")
    y, se = corpus.arrays()

    methods = []
    for method in ("dl", "pm", "reml"):
        re = random_effects(y, se, tau_method=method)
        methods.append(
            {
                "method": method,
                "estimate": re.estimate,
                "se": re.se,
                "tau2": re.tau2,
                "lower": re.interval.lower,
                "upper": re.interval.upper,
            }
        )
    het = heterogeneity(y, se)
    kh = random_effects(y, se, tau_method="reml", knapp_hartung=True)
    pi = prediction_interval(kh)
    eg = egger(y, se)
    loo = leave_one_out(y, se, method="reml")
    bj = baujat(y, se)

    fd = forest_data(corpus, None, kh, mass=0.95)
    fu = funnel_data(y, se, kh)

    out["classical"] = {
        "k": len(corpus),
        "n_experiment": sum(r.read == "experiment" for r in corpus.records),
        "n_model": sum(r.read == "model" for r in corpus.records),
        "studies": [
            {
                "study": r.study,
                "contributor": r.contributor,
                "read": r.read,
                "estimate": r.estimate,
                "se": r.se,
                "n": r.n,
            }
            for r in corpus.records
        ],
        "methods": methods,
        "heterogeneity": {
            "q": het.q,
            "df": het.df,
            "p": het.p_value,
            "i2": het.i2,
            "tau2": getattr(het, "tau2", None),
        },
        "knapp_hartung": {
            "estimate": kh.estimate,
            "se": kh.se,
            "lower": kh.interval.lower,
            "upper": kh.interval.upper,
        },
        "prediction_interval": [pi.lower, pi.upper],
        "egger": {"intercept": eg.intercept, "se": eg.se, "t": eg.t, "df": eg.df, "p": eg.p},
        "leave_one_out": [
            {"study": r.study, "influence": float(v)}
            for r, v in zip(corpus.records, loo.influence, strict=True)
        ],
        "baujat": {
            "q_contribution": jsonable(bj.q_contribution),
            "influence": jsonable(bj.influence),
            "labels": [r.study for r in corpus.records],
        },
        "forest": {
            "rows": [
                {
                    "label": row.label,
                    "estimate": row.estimate,
                    "se": row.se,
                    "lower": row.interval.lower,
                    "upper": row.interval.upper,
                    "weight": getattr(row, "weight", None),
                    "read": corpus.records[i].read,
                    "contributor": corpus.records[i].contributor,
                }
                for i, row in enumerate(fd.rows)
            ],
            "mass": fd.mass,
            # nested too, in the shape the forest chart wants for its pooled diamond
            "pooled": {
                "estimate": fd.pooled_estimate,
                "lower": fd.pooled_interval.lower,
                "upper": fd.pooled_interval.upper,
            },
            "pooled_estimate": fd.pooled_estimate,
            "pooled_se": fd.pooled_se,
            "pooled_lower": fd.pooled_interval.lower,
            "pooled_upper": fd.pooled_interval.upper,
            "prediction_lower": fd.prediction_interval.lower,
            "prediction_upper": fd.prediction_interval.upper,
            "tau2": fd.tau2,
        },
        "funnel": {
            "y": jsonable(y),
            "se": jsonable(se),
            "labels": [r.study for r in corpus.records],
            "read": [r.read for r in corpus.records],
            "pooled": fu.pooled,
            "contours": [
                {
                    "mass": c.mass,
                    "se": jsonable(c.se),
                    "lower": jsonable(c.lower),
                    "upper": jsonable(c.upper),
                }
                for c in fu.contours
            ],
            "max_se": float(np.max(se)),
        },
    }

    verdict = delta_identification(corpus, "fertilizer")
    pool_spec = PoolSpec(
        family="fertilizer",
        bias_term=True,
        priors=PoolPriors(mu_scale=2.0, tau_scale=0.5, delta_scale=1.0),
        mass=0.9,
        definition="eti",
    )
    pooled = pool(pool_spec, corpus, backend="laplace", draws=2000, seed=0)
    res = pooled.result
    out["pooled"] = {
        "delta_identification": {"status": verdict.status, "route": verdict.route},
        "dual_read_contributors": jsonable(eleven.dual_read_contributors("fertilizer")),
        "mu": {
            "mean": res.mu.mean,
            "sd": res.mu.sd,
            "lower": res.mu.interval.lower,
            "upper": res.mu.interval.upper,
        },
        "tau": {"mean": res.tau.mean, "sd": res.tau.sd},
        "delta": {
            "mean": res.delta.mean,
            "sd": res.delta.sd,
            "identified": bool(res.delta.identified),
        },
        "k": res.k,
        "backend": res.backend,
        "reml_without_bias": random_effects(y, se, tau_method="reml").estimate,
        "by_read": {
            "experiment": jsonable([r.estimate for r in corpus.records if r.read == "experiment"]),
            "model": jsonable([r.estimate for r in corpus.records if r.read == "model"]),
        },
    }

    amplitude_prior, handoff = prior_from_pool(res, target="predictive", family="lognormal")
    out["handoff"] = {
        "prior": str(amplitude_prior),
        "hyper": jsonable(dict(amplitude_prior.hyper)),
        "kind": handoff.kind,
        "statement": handoff.statement,
        "prior_mean": float(
            np.exp(amplitude_prior.hyper["mu"] + amplitude_prior.hyper["sigma"] ** 2 / 2)
        ),
    }

    cell = cell_from_records(corpus.records, name="fertilizer/standardized_contrast")
    policy = PrivacyPolicy(
        k=3, dominance_p=0.5, dominance_top_n=2, dominance_top_p=0.8, epsilon_total=1.0
    )
    gate = check_cell(cell, policy)
    published, budget = release(
        cell,
        policy,
        EpsilonLedger(budget=policy.epsilon_total),
        release_id="fertilizer-2026Q3",
        epsilon=0.5,
        clip=(0.0, 1.5),
        seed=0,
        mechanism="laplace",
    )
    out["privacy"] = {
        "gate": {
            "status": gate.status,
            "route": gate.route,
            "contributors": len(cell.contributors),
        },
        "policy": {"k": policy.k, "epsilon_total": policy.epsilon_total},
        "released": published.value,
        "lower": published.interval.lower,
        "upper": published.interval.upper,
        "clipped_mean": float(np.mean(np.clip(cell.values, 0, 1.5))),
        "mechanism": published.mechanism,
        "noise_scale": published.noise_scale,
        "epsilon": published.epsilon,
        "used": jsonable(budget.used),
        "remaining": jsonable(budget.remaining),
    }
    return out


# ----------------------------------------------------------------------------------
# case study: HYPER-3, the trial that has to stop an arm
# ----------------------------------------------------------------------------------


@section
def casestudy() -> Any:
    sys.path.insert(0, str(ROOT / "nbs" / "case-studies" / "hypertension"))
    import hyper3 as h
    from scipy import stats as sps

    from axiom.design import (
        LookSchedule,
        StoppingRule,
        crossing_probabilities,
        difference_se,
        harm_boundary,
        information_fractions,
        monitor,
        operating_characteristics,
    )
    from axiom.identify import ols

    SD_WINDOW, HARM_MARGIN, HARM_PROBABILITY = 6.0, 2.0, 0.95
    trial = h.trial(seed=20260821)
    LOOK_WEEKS = list(range(12, 24, 2))
    INFORMATION = information_fractions([h.N_UNITS * (w - 6) / 16 for w in LOOK_WEEKS], h.N_UNITS)
    schedule = LookSchedule(labels=tuple(f"week_{w}" for w in LOOK_WEEKS), information=INFORMATION)

    N_ARM = {arm: int(h.N_UNITS * k / h.BLOCK) for arm, k in h.ALLOCATION.items()}
    N_CELL = {
        (s, a): int(h.N_UNITS * h.STRATUM_SHARE[s] * k / h.BLOCK)
        for s in h.STRATA
        for a, k in h.ALLOCATION.items()
    }
    CONTRASTS = [("all", arm) for arm in h.ARMS[1:]] + [
        (s, arm) for s in h.STRATA for arm in h.ARMS[1:]
    ]

    rules = {}
    for stratum, arm in CONTRASTS:
        if stratum == "all":
            n_dose, n_control = N_ARM[arm], N_ARM["standard_of_care"]
        else:
            n_dose, n_control = N_CELL[(stratum, arm)], N_CELL[(stratum, "standard_of_care")]
        se = difference_se(
            n_dose + n_control, sd=SD_WINDOW, allocation=n_dose / (n_dose + n_control)
        )
        rules[(stratum, arm)] = StoppingRule(
            name=f"harm|{stratum}|{arm}",
            looks=schedule,
            boundaries=(
                harm_boundary(
                    HARM_PROBABILITY, INFORMATION, margin=HARM_MARGIN, se_at_full_information=se
                ),
            ),
        )

    def look_statistics(week: int) -> dict[tuple[str, str], tuple[float, float, int]]:
        frame = h.look_frame(trial, week, endpoint="safety")
        res = {}
        for stratum, arm in CONTRASTS:
            where = None if stratum == "all" else stratum
            r = h.contrast_frame(frame, arm, stratum=where)
            e = ols(r, "change", "treated", h.ancova_covariates(where))
            res[(stratum, arm)] = (e.estimate, e.se, e.n)
        return res

    series = {w: look_statistics(w) for w in LOOK_WEEKS}
    z_paths = {k: [] for k in rules}
    effects = {k: [] for k in rules}
    ses = {k: [] for k in rules}
    for w in LOOK_WEEKS:
        for k in rules:
            e, s, _ = series[w][k]
            z_paths[k].append(-e / s)
            effects[k].append(e)
            ses[k].append(s)
    paths = {k: monitor(rules[k], z_paths[k], effects=effects[k], ses=ses[k]) for k in rules}

    monitoring = []
    for (stratum, arm), path in paths.items():
        stop = path.stop
        monitoring.append(
            {
                "stratum": stratum,
                "stratum_label": "all units" if stratum == "all" else h.STRATUM_LABEL[stratum],
                "arm": arm,
                "arm_label": h.ARM_LABEL[arm],
                "looks_taken": len(path.looks),
                "decision": path.decision,
                "stopped_week": None if stop is None else LOOK_WEEKS[stop.look],
                "z_at_stop": None if stop is None else stop.z,
                "lowest_z_seen": min(z_paths[(stratum, arm)][: len(path.looks)]),
                "z_if_continued": None if stop is None else min(z_paths[(stratum, arm)]),
            }
        )

    panels = []
    for stratum in ("all", "age_25_35", "age_36_50", "age_51_plus"):
        key = (stratum, "dose_40")
        boundary = rules[key].boundaries[0]
        taken = len(paths[key].looks)
        stop = paths[key].stop
        panels.append(
            {
                "stratum": stratum,
                "label": "all units" if stratum == "all" else h.STRATUM_LABEL[stratum],
                "information": list(INFORMATION),
                "boundary": list(boundary.z),
                "z": z_paths[key],
                "looks_taken": taken,
                "weeks": LOOK_WEEKS,
                "stopped": stop is not None,
                "stop_information": None if stop is None else stop.information,
                "stop_z": None if stop is None else stop.z,
                "stop_week": None if stop is None else LOOK_WEEKS[stop.look],
                "effects": effects[key],
                "ses": ses[key],
            }
        )

    stopped = paths[("age_51_plus", "dose_40")]
    line = stopped.ledger_line()
    posterior = float(sps.norm.sf((HARM_MARGIN - stopped.stop.effect) / stopped.stop.se))

    # operating characteristics of the rule under a few drifts
    oc_rows = []
    for drift in (0.0, -1.0, -2.0, -3.0, -4.0):
        rule = rules[("age_51_plus", "dose_40")]
        oc = operating_characteristics(rule, drift)
        cp = crossing_probabilities(rule, drift)
        per_look = list(cp.per_look["harm"])
        oc_rows.append(
            {
                "drift": drift,
                "per_look": per_look,
                "stop_probability": float(sum(per_look)),
                "continue_probability": cp.continue_probability,
                "expected_information": oc.expected_information,
                "expected_looks": oc.expected_looks,
            }
        )

    # The dose-response the trial is actually trying to recover. This must be the
    # intention-to-treat contrast, not benefit-minus-harm at the prescribed dose:
    # adherence is a consequence of assignment, and the pressor term is cubic in the
    # dose actually taken, so the per-protocol curve overstates harm by a lot and would
    # not agree with the monitoring statistics on the same page.
    # Sign convention matches the monitoring effects: negative is a fall in blood
    # pressure (good), positive is a rise (harm).
    doses = np.linspace(0.0, 40.0, 41)
    week = float(h.PRIMARY_WEEK)
    curves = []
    for s in h.STRATA:
        curves.append(
            {
                "stratum": s,
                "label": h.STRATUM_LABEL[s],
                "itt": [float(h.intent_to_treat_contrast(float(d), s, week)) for d in doses],
                "per_protocol": [float(h.per_protocol_contrast(float(d), s, week)) for d in doses],
            }
        )
    pooled_itt = [float(h.intent_to_treat_contrast(float(d), None, week)) for d in doses]
    pooled_pp = [float(h.per_protocol_contrast(float(d), None, week)) for d in doses]

    return {
        "protocol": {
            "n_units": h.N_UNITS,
            "arms": list(h.ARMS),
            "arm_labels": {a: h.ARM_LABEL[a] for a in h.ARMS},
            "dose": {a: h.DOSE[a] for a in h.ARMS},
            "dose_unit": h.DOSE_UNIT,
            "outcome_unit": h.OUTCOME_UNIT,
            "strata": list(h.STRATA),
            "stratum_labels": {s: h.STRATUM_LABEL[s] for s in h.STRATA},
            "stratum_share": {s: h.STRATUM_SHARE[s] for s in h.STRATA},
            "enrollment_weeks": h.ENROLLMENT_WEEKS,
            "follow_up_weeks": h.FOLLOW_UP_WEEKS,
            "primary_week": h.PRIMARY_WEEK,
            "n_arm": N_ARM,
        },
        "rule": {
            "look_weeks": LOOK_WEEKS,
            "information": list(INFORMATION),
            "n_looks": schedule.n_looks,
            "n_contrasts": len(rules),
            "margin": HARM_MARGIN,
            "probability": HARM_PROBABILITY,
            "sd_window": SD_WINDOW,
        },
        "monitoring": monitoring,
        "panels": panels,
        "stop": {
            "decision": stopped.decision,
            "look": stopped.stopped_at + 1,
            "n_looks": schedule.n_looks,
            "week": LOOK_WEEKS[stopped.stop.look],
            "information": stopped.information_used,
            "effect": stopped.stop.effect,
            "se": stopped.stop.se,
            "n": series[LOOK_WEEKS[stopped.stop.look]][("age_51_plus", "dose_40")][2],
            "z": stopped.stop.z,
            "threshold": stopped.threshold(),
            "posterior_probability": posterior,
            "ledger_statement": line.statement,
            "assumption_name": line.assumption.name,
            "assumption_statement": line.assumption.statement,
            "challenged_by": jsonable(line.assumption.challenged_by),
        },
        "operating_characteristics": oc_rows,
        "truth": {
            "doses": doses.tolist(),
            "curves": curves,
            "pooled_itt": pooled_itt,
            "pooled_pp": pooled_pp,
            "week": week,
            # the values at the top dose, which is what the argument turns on
            "at_40": {
                "pooled_itt": pooled_itt[-1],
                "by_stratum": {s: curves[i]["itt"][-1] for i, s in enumerate(h.STRATA)},
                "pooled_pp": pooled_pp[-1],
            },
            "at_20": {
                "pooled_itt": pooled_itt[20],
                "by_stratum": {s: curves[i]["itt"][20] for i, s in enumerate(h.STRATA)},
            },
        },
    }


# ----------------------------------------------------------------------------------
# tour: short snippets shown on the landing page, with their real captured output
# ----------------------------------------------------------------------------------


@section
def tour() -> Any:
    """Run each snippet and capture exactly what it prints.

    The landing page shows the code and the output side by side. Capturing the
    output here means the two cannot drift apart: if the library changes what it
    prints, rebuilding the site changes the page.
    """
    import contextlib
    import io as _io
    import textwrap

    snippets = {
        "identify": """
            from axiom.identify import identify, ols
            from axiom.sim import confounded_world

            world = confounded_world()      # Z confounds X and Y
            verdict = identify(world.graph, "X", "Y")
            print("status :", verdict.status)
            print("route  :", verdict.route)
            print("adjust :", sorted(verdict.adjustment_set))

            frame = world.observed(world.simulate(5_000, seed=0))
            naive = ols(frame, "Y", "X")
            fixed = ols(frame, "Y", "X", covariates=verdict.adjustment_set)
            print()
            print(f"truth            {world.total_effect('X', 'Y'):.3f}")
            print(f"naive            {naive.estimate:.3f}  {naive.ci(0.95)}")
            print(f"back-door        {fixed.estimate:.3f}  {fixed.ci(0.95)}")
        """,
        "refuse": """
            from axiom.identify import identify
            from axiom.sim import hidden_confounder_world

            world = hidden_confounder_world()   # the confounder is never measured
            verdict = identify(world.graph, "X", "Y")
            print("status :", verdict.status)
            print("route  :", verdict.route)
            print("needs  :", sorted(verdict.unmeasured_required), "which you do not have")
            print()
            for a in verdict.verdict.assumptions:
                print(f"assumption    {a.name}  [{a.state}]")
                print(f"  says        {a.statement}")
                print(f"  challenged  {a.challenged_by}")
        """,
        "estimand": """
            from axiom.core import Population, TimeWindow
            from axiom.estimands import Level, standard_estimands
            from axiom.sim import arms_world
            from axiom.surface import HillKernel

            world = arms_world(
                n_units=40, treatments=("a",),
                kernels=HillKernel(reference_dose=50.0, amplitude_scale=10.0),
                doses={"a": [0.0] * 20 + [60.0] * 20},
                truth={"beta_a": 6.0, "k_a": 50.0, "s_a": 2.0, "alpha": 1.0},
                noise_sd=1.5, seed=0,
            )
            registry = standard_estimands(
                treatment=world.spec.treatments[0], outcome=world.spec.outcome,
                population=Population(name="all"), window=TimeWindow(start=0, stop=1),
                level=Level(unit="individual"), dose=60.0, reference_dose=0.0,
            )
            estimand = registry.get("contrast_at_dose")
            print("name      :", estimand.name)
            print("quantity  :", estimand.quantity.kind)
            print("dose      :", estimand.intervention.doses, "vs", estimand.reference.doses)
            print("window    :", estimand.window.start, "->", estimand.window.stop)
            print("dimension :", estimand.dimension)
            print("hash      :", estimand.content_hash()[:16])
        """,
    }

    out: dict[str, Any] = {}
    for name, raw in snippets.items():
        code = textwrap.dedent(raw).strip("\n")
        buf = _io.StringIO()
        env: dict[str, Any] = {}
        with contextlib.redirect_stdout(buf):
            exec(compile(code, f"<tour:{name}>", "exec"), env)  # noqa: S102
        out[name] = {"code": code, "output": buf.getvalue().rstrip("\n")}
        print(f"    {name}: {len(out[name]['output'].splitlines())} lines of output")
    return out


# ----------------------------------------------------------------------------------
# examples: run every script in examples/ and capture what it really prints
# ----------------------------------------------------------------------------------


@section
def examples() -> Any:
    """Execute each example in a subprocess and keep its source and its output.

    Running them rather than transcribing them means the site cannot drift from the
    scripts, and a broken example shows up as a failed build instead of stale prose.
    """
    import subprocess

    # field / one-line question / pillars — the only editorial content here
    META = {
        "01_agriculture_response_surface": (
            "Agronomy",
            "Where does yield peak in nitrogen and irrigation, and how should a "
            "rationed budget be split?",
            ["surface"],
        ),
        "02_economics_instrument": (
            "Labour economics",
            "Does a training programme raise earnings, when enrolment is " "self-selected?",
            ["identify"],
        ),
        "03_education_selection_bias": (
            "Education",
            "How strong would an unmeasured confounder have to be to erase the " "tutoring effect?",
            ["identify", "diagnose"],
        ),
        "04_epidemiology_frontdoor": (
            "Epidemiology",
            "An exposure you cannot deconfound, recovered through a measured " "mediator.",
            ["identify"],
        ),
        "05_ecology_transport": (
            "Conservation ecology",
            "Does a result from one reserve apply to another, and what must you "
            "measure there first?",
            ["identify"],
        ),
        "06_public_health_cluster_trial": (
            "Public health",
            "Sizing a cluster-randomized trial, where the intra-cluster "
            "correlation eats the sample.",
            ["design"],
        ),
        "07_clinical_rwe_calibration": (
            "Clinical research",
            "Using a small clean trial to correct a large confounded registry.",
            ["calibrate"],
        ),
        "08_energy_panel_methods": (
            "Energy",
            "Which quasi-experimental method actually holds its size on this panel?",
            ["design"],
        ),
        "09_manufacturing_steepest_ascent": (
            "Process engineering",
            "Climbing to a yield optimum, Box-Wilson style.",
            ["surface"],
        ),
        "10_marketing_saturation": (
            "Marketing",
            "Contribution and marginal return, and proof the vocabulary is a thin " "adapter.",
            ["adapters", "surface"],
        ),
        "11_psychology_meta_analysis": (
            "Behavioural science",
            "Pooling a literature that disagrees with itself.",
            ["meta"],
        ),
        "12_software_experiment_value": (
            "Product analytics",
            "Is this A/B test worth running at all?",
            ["design"],
        ),
    }

    directory = ROOT / "examples"
    entries = []
    for path in sorted(directory.glob("*.py")):
        if path.name == "run_all.py":
            continue
        stem = path.stem
        if stem not in META:
            print(f"    WARNING: {stem} has no metadata; skipping")
            continue
        field, question, pillars = META[stem]
        source = path.read_text()
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(path)], capture_output=True, text=True, cwd=str(ROOT)
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            raise SystemExit(f"example {stem} failed:\n{proc.stderr[-2000:]}")
        # the module docstring is the framing; the code below it is the demo
        body = source.split('"""', 2)[2].lstrip("\n") if source.count('"""') >= 2 else source
        entries.append(
            {
                "stem": stem,
                "number": stem.split("_")[0],
                "field": field,
                "question": question,
                "pillars": pillars,
                "docstring": source.split('"""')[1].strip() if '"""' in source else "",
                "code": body.rstrip(),
                "output": proc.stdout.rstrip("\n"),
                "lines": len(source.splitlines()),
                "seconds": round(elapsed, 1),
            }
        )
        print(f"    {stem}: {elapsed:.1f}s, " f"{len(proc.stdout.splitlines())} lines of output")

    return {
        "entries": entries,
        "n": len(entries),
        "total_seconds": round(sum(e["seconds"] for e in entries), 1),
        "total_lines": sum(e["lines"] for e in entries),
    }


# ----------------------------------------------------------------------------------
# benchmarks: does axiom reproduce numbers other people published?
# ----------------------------------------------------------------------------------


@section
def benchmarks() -> Any:
    """Re-run the published-value comparisons and record every one.

    The comparisons are recomputed here rather than scraped from the case-study
    output, so the site's table is the same arithmetic the tests assert on.
    """
    import subprocess

    sys.path.insert(0, str(ROOT / "benchmarks"))
    import numpy as _np
    import pandas as _pd
    from registry import DATASETS, load, log_risk_ratios, model_based_i2

    from axiom.core import BASES, D, Outcome, Treatment
    from axiom.data import Panel, RoleMap
    from axiom.diagnose import robustness_value
    from axiom.identify import ols
    from axiom.meta import heterogeneity, random_effects
    from axiom.surface import ExponentialKernel, SurfaceSpec, fit

    def row(label: str, ours: float, theirs: float, fmt: str = ".4f") -> dict[str, Any]:
        delta = abs(float(ours) - float(theirs))
        scale = max(abs(float(theirs)), 1e-12)
        return {
            "label": label,
            "axiom": float(ours),
            "published": float(theirs),
            "delta": delta,
            "rel": delta / scale,
            "fmt": fmt,
        }

    entries: list[dict[str, Any]] = []

    # --- BCG, against R's metafor --------------------------------------------
    bcg_spec = DATASETS["bcg"]
    pub = bcg_spec.published
    frame = load("bcg")
    y, se = log_risk_ratios(frame)
    pooled = random_effects(y, se, tau_method="reml")
    het = heterogeneity(y, se)
    i2_model, h2_model = model_based_i2(pooled.tau2, se)
    kh = random_effects(y, se, tau_method="reml", knapp_hartung=True)
    from axiom.meta import prediction_interval

    pi = prediction_interval(kh)

    entries.append({
        "name": "bcg",
        "title": bcg_spec.title,
        "field": bcg_spec.field,
        "pillar": bcg_spec.pillar,
        "reference": "R metafor",
        "availability": bcg_spec.availability,
        "citation": bcg_spec.citation,
        "licence": bcg_spec.licence,
        "source": bcg_spec.source,
        "n_rows": len(frame),
        "scale": f"{int((frame.tpos + frame.tneg + frame.cpos + frame.cneg).sum()):,} people",
        "rows": [
            row("pooled log risk ratio", pooled.estimate, pub["estimate"]),
            row("standard error", pooled.se, pub["se"]),
            row("95% interval, lower", pooled.interval.lower, pub["ci_lower"]),
            row("95% interval, upper", pooled.interval.upper, pub["ci_upper"]),
            row("tau^2 (REML)", pooled.tau2, pub["tau2_reml"]),
            row("Cochran's Q", het.q, pub["q"], ".4f"),
            row("I^2 (model-based)", i2_model, pub["i2_model_based"]),
            row("H^2 (model-based)", h2_model, pub["h2_model_based"], ".3f"),
        ],
        # the forest of individual trials, for a chart
        "forest": [
            {
                "label": f"{frame.author[i][:22]} {int(frame.year[i])}",
                "estimate": float(y[i]),
                "lower": float(y[i] - 1.96 * se[i]),
                "upper": float(y[i] + 1.96 * se[i]),
                "latitude": int(frame.ablat[i]),
            }
            for i in range(len(frame))
        ],
        "pooled": {
            "estimate": pooled.estimate,
            "lower": pooled.interval.lower,
            "upper": pooled.interval.upper,
        },
        "i2_q_based": het.i2,
        "i2_model_based": i2_model,
        "prediction": [pi.lower, pi.upper],
    })

    # --- NIST Misra1a, against certified values ------------------------------
    nist = DATASETS["misra1a"]
    cert = nist.published
    BASES.declare("volume", symbol="V")
    BASES.declare("pressure", symbol="P")
    volume = Outcome(name="volume", dimension=D.volume, unit="cc")
    pressure = Treatment(name="pressure", dimension=D.pressure, unit="mmHg")
    mf = load("misra1a")
    tidy = _pd.DataFrame({
        "unit": ["specimen"] * len(mf),
        "t": _np.arange(len(mf)),
        "pressure": mf["pressure"].to_numpy(float),
        "volume": mf["volume"].to_numpy(float),
    })
    panel = Panel(
        tidy,
        RoleMap(unit="unit", time="t", outcome=("volume", volume),
                treatments={"pressure": pressure}),
    )
    nist_fit = fit(
        SurfaceSpec(
            name="misra1a",
            treatments=(pressure,),
            outcome=volume,
            kernels={"pressure": ExponentialKernel(reference_dose=1800.0, amplitude_scale=250.0)},
            intercept="none",
            unit_labels=("specimen",),
        ),
        panel, backend="laplace", draws=4000, seed=SEED,
    )
    post = nist_fit.posterior
    beta = post.summary("beta_pressure", definition="hdi", mass=0.95)
    kk = post.summary("k_pressure", definition="hdi", mass=0.95)
    x = tidy["pressure"].to_numpy(float)
    yy = tidy["volume"].to_numpy(float)
    rss = float(_np.sum((yy - beta.mean * (1.0 - _np.exp(-x / kk.mean))) ** 2))

    entries.append({
        "name": "misra1a",
        "title": nist.title,
        "field": nist.field,
        "pillar": nist.pillar,
        "reference": "NIST certified values",
        "availability": nist.availability,
        "citation": nist.citation,
        "licence": nist.licence,
        "source": nist.source,
        "n_rows": len(mf),
        "scale": "14 observations, 2 parameters",
        "rows": [
            row("beta  (NIST b1)", beta.mean, cert["axiom_beta"], ".4f"),
            row("k     (NIST 1/b2)", kk.mean, cert["axiom_k"], ".2f"),
            row("implied b2", 1.0 / kk.mean, cert["b2"], ".3e"),
            row("residual sum of squares", rss, cert["residual_sum_of_squares"], ".6f"),
        ],
        "hdi": {
            "beta": [beta.interval.lower, beta.interval.upper],
            "k": [kk.interval.lower, kk.interval.upper],
            "beta_certified": cert["axiom_beta"],
            "k_certified": cert["axiom_k"],
        },
        "rss_excess": rss - cert["residual_sum_of_squares"],
        "observed": {"pressure": x.tolist(), "volume": yy.tolist()},
    })

    # --- Darfur and LaLonde: only if fetched ---------------------------------
    covs = ["age", "farmer_dar", "herder_dar", "pastvoted", "hhsize_darfur", "female"]
    if DATASETS["darfur"].available:
        d = DATASETS["darfur"]
        pub = d.published
        df_frame = load("darfur")
        dummies = _pd.get_dummies(df_frame["village"], prefix="v", drop_first=True, dtype=float)
        design = _pd.concat([df_frame.drop(columns=["village"]), dummies], axis=1)
        est = ols(design, "peacefactor", "directlyharmed",
                  covariates=[*covs, *dummies.columns])
        dof = int(est.detail["df_resid"])
        rv = robustness_value(estimate=est.estimate, se=est.se, df=dof, q=1.0, alpha=0.05)
        naive = ols(df_frame, "peacefactor", "directlyharmed")
        no_village = ols(df_frame, "peacefactor", "directlyharmed", covariates=covs)
        entries.append({
            "name": "darfur",
            "title": d.title,
            "field": d.field,
            "pillar": d.pillar,
            "reference": "Cinelli & Hazlett (2020)",
            "availability": d.availability,
            "citation": d.citation,
            "licence": d.licence,
            "source": d.source,
            "n_rows": len(df_frame),
            "scale": f"{len(df_frame):,} respondents, {df_frame.village.nunique()} villages",
            "rows": [
                row("coefficient", est.estimate, pub["coefficient"]),
                row("standard error", est.se, pub["se"]),
                row("residual df", float(dof), float(pub["df"]), ".0f"),
                row("robustness value", rv.rv, pub["robustness_value"], ".3f"),
                row("RV at alpha = 0.05", rv.rv_alpha, pub["robustness_value_alpha"], ".3f"),
                row("partial R^2", rv.r2_yd_x, pub["partial_r2"], ".3f"),
            ],
            "specifications": [
                {"label": "no covariates", "estimate": naive.estimate, "se": naive.se},
                {"label": "covariates only", "estimate": no_village.estimate, "se": no_village.se},
                {"label": "published specification", "estimate": est.estimate, "se": est.se},
            ],
        })

    if DATASETS["lalonde_nsw"].available and DATASETS["psid_controls"].available:
        lal = DATASETS["lalonde_nsw"]
        pub = lal.published
        nsw = load("lalonde_nsw")
        psid = load("psid_controls")
        lcovs = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]
        exp = ols(nsw, "re78", "treat")
        obs = _pd.concat([nsw[nsw.treat == 1], psid], ignore_index=True)
        obs_naive = ols(obs, "re78", "treat")
        obs_adj = ols(obs, "re78", "treat", covariates=lcovs)
        entries.append({
            "name": "lalonde_nsw",
            "title": lal.title,
            "field": lal.field,
            "pillar": lal.pillar,
            "reference": "Dehejia & Wahba (1999)",
            "availability": lal.availability,
            "citation": lal.citation,
            "licence": lal.licence,
            "source": lal.source,
            "n_rows": len(nsw),
            "scale": f"{len(nsw)} randomized, {len(psid):,} PSID comparison",
            "rows": [
                row("experimental effect (USD)", exp.estimate, pub["experimental_ate"], ".0f"),
                row("n", float(len(nsw)), float(pub["n"]), ".0f"),
                row("n treated", float(int(nsw.treat.sum())), float(pub["n_treated"]), ".0f"),
            ],
            "specifications": [
                {"label": "experimental benchmark", "estimate": exp.estimate, "se": exp.se},
                {"label": "PSID controls, unadjusted", "estimate": obs_naive.estimate,
                 "se": obs_naive.se},
                {"label": "PSID controls, 8 covariates", "estimate": obs_adj.estimate,
                 "se": obs_adj.se},
            ],
            "truth": exp.estimate,
        })

    # capture the full printed output of each case study, as with the examples
    for entry in entries:
        script = ROOT / "benchmarks" / f"case_{entry['name']}.py"
        if entry["name"] == "lalonde_nsw":
            script = ROOT / "benchmarks" / "case_lalonde.py"
        proc = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True,
            cwd=str(ROOT / "benchmarks"),
        )
        if proc.returncode != 0:
            raise SystemExit(f"benchmark {entry['name']} failed:\n{proc.stderr[-1500:]}")
        entry["output"] = proc.stdout.rstrip("\n")
        entry["worst"] = max(r["delta"] for r in entry["rows"])
        entry["worst_rel"] = max(r["rel"] for r in entry["rows"])
        print(f"    {entry['name']}: {len(entry['rows'])} checks, "
              f"worst relative error {entry['worst_rel']:.1e}")

    return {
        "entries": entries,
        "n": len(entries),
        "n_checks": sum(len(e["rows"]) for e in entries),
        "worst_rel": max(e["worst_rel"] for e in entries),
    }


def main() -> int:
    wanted = sys.argv[1:] or list(SECTIONS)
    for name in wanted:
        if name not in SECTIONS:
            print(f"unknown section {name!r}; have {sorted(SECTIONS)}")
            return 2
        print(f"[{name}]")
        t0 = time.perf_counter()
        write(name, SECTIONS[name]())
        print(f"  {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
