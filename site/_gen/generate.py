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

import ast
import inspect
import json
import os
import re
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


# The package tables live here, at module level, because two sections read them:
# `overview` (the counts on the landing page) and `api` (the symbol map). One copy
# means the API page cannot list a subpackage the landing page has never heard of.
ORDER = (
    "core data io infer dynamics identify discover estimands surface sim design "
    "calibrate meta diagnose build adapters viz report"
).split()

BLURB = {
    "core": "Specs, dimensions, intervals, expressions — the vocabulary "
    "everything else is written in.",
    "data": "Panels and role maps: which column is the treatment, which is the outcome.",
    "io": "Save and load a whole analysis as JSON plus a posterior file. Never a pickle.",
    "infer": "The sampler seam. Laplace in core; NumPyro and PyMC behind extras.",
    "dynamics": "A language for systems that are simultaneous or have time structure, "
    "compiled into ordinary expression trees.",
    "identify": "Causal graphs, back-door / front-door / IV routes, transport verdicts.",
    "discover": "Learn the graph instead of assuming it: essential graphs, GES/GIES, "
    "bootstrap stability, and FCI when a common cause may be unmeasured.",
    "estimands": "Declare the number you want as a nine-facet object, then realize it.",
    "surface": "Dose-response kernels, carryover, the one forward(), optimal allocation.",
    "sim": "Synthetic worlds with known truth, so every claim has a recovery test.",
    "design": "Power, MDE, value of information, method leaderboards, sequential boundaries.",
    "calibrate": "Fold a randomized result into an observational model, and log "
    "what that assumed.",
    "meta": "Pool a corpus of studies; heterogeneity, bias terms, privacy-budgeted release.",
    "diagnose": "SBC, coverage, posterior predictive checks, refutations, " "specification curves.",
    "build": "Fluent builders for graphs, priors, and corpora.",
    "adapters": "Domain vocabulary — marketing is one adapter, not the core.",
    "viz": "Plot helpers for the diagnostics that have a canonical picture.",
    "report": "Templated HTML / PPTX / PDF reports over the viz layer.",
}

LAYER = {
    "core": "foundation",
    "data": "foundation",
    "io": "foundation",
    "infer": "sampler seam",
    "dynamics": "domain",
    "identify": "domain",
    "discover": "domain",
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


@section
def overview() -> Any:
    import importlib
    import subprocess

    order, blurb, layer = ORDER, BLURB, LAYER

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
        # Keyed by name as well as ordered, because the pages used to index this
        # list positionally -- and inserting a subpackage then silently relabelled
        # every count after it. Reference a package by its name.
        "by_name": {p["name"]: p for p in packages},
        "total_symbols": total,
        "n_packages": len(packages),
        "import_workload": workload,
        "import_pulls": imported,
        "samplers_pulled": [m for m in imported if m in {"jax", "numpyro", "pymc", "pytensor"}],
        "import_seconds": round(import_seconds, 2),
    }


# ----------------------------------------------------------------------------------
# api: every public symbol, with its signature and a line of real usage
# ----------------------------------------------------------------------------------
#
# The map used to be a list of names, which tells a reader that `frontier` exists
# and nothing about how to call it. What is missing is the shape of the call, and
# the honest source for that is not a hand-written example -- those rot -- but the
# notebook that already demonstrates the symbol. Gate 12 guarantees one exists.
#
# So for each symbol this section pulls three things out of the code itself: the
# signature, the first line of the docstring, and the shortest *executed* statement
# in that subpackage's notebooks that actually uses it. A symbol whose only
# appearance is inside an `import` line gets no snippet, and the page says so
# rather than inventing one -- gate 12 counts an import as coverage, and this is
# the page that shows where that is all the coverage there is.

ADDRESS = re.compile(r" at 0x[0-9a-f]+")
# Any dotted module path in front of a type name: axiom.core.spec.Spec -> Spec.
QUALIFIER = re.compile(r"\b(?:[a-z_]\w*\.)+(\w+)")
MAX_USAGE_LINES = 16
MAX_DEFAULT_CHARS = 40
MAX_FIELDS = 12


def _collapse_annotated(text: str) -> str:
    """``Annotated[T, <validators>]`` -> ``T``.

    Pydantic renders the whole validator chain into an annotation's repr, which is
    both unreadable and unstable across runs. Only the type is wanted.
    """
    while True:
        start = text.find("Annotated[")
        if start < 0:
            return text
        depth, close = 0, len(text) - 1
        for i in range(start + len("Annotated[") - 1, len(text)):
            if text[i] == "[":
                depth += 1
            elif text[i] == "]":
                depth -= 1
                if depth == 0:
                    close = i
                    break
        inner = text[start + len("Annotated[") : close]
        depth, cut = 0, len(inner)
        for i, ch in enumerate(inner):
            if ch in "[(":
                depth += 1
            elif ch in "])":
                depth -= 1
            elif ch == "," and depth == 0:
                cut = i
                break
        text = text[:start] + inner[:cut].strip() + text[close + 1 :]


def _readable_type(text: str) -> str:
    text = _collapse_annotated(text.replace("<class '", "").replace("'>", ""))
    text = ADDRESS.sub("", text)
    text = QUALIFIER.sub(r"\1", text)
    return text.replace("NoneType", "None")


def _short_default(value: Any) -> str:
    """A default long enough to bury the signature is shown as an ellipsis.

    ``design.simulated_power`` defaults its registry to the whole method table;
    printed in full it is six thousand characters of signature for one parameter.
    """
    shown = ADDRESS.sub("", repr(value))
    return shown if len(shown) <= MAX_DEFAULT_CHARS else "..."


def _model_fields(obj: Any) -> Any:
    """Pydantic fields, or None for anything that is not a model.

    The test is ``getattr_static`` rather than ``getattr`` on purpose: some classes
    here define a ``__getattr__`` that raises for an unknown name (``Dimensions``
    raises ``UndeclaredBaseError``), so merely asking whether the attribute exists
    would blow up. ``getattr_static`` reads the type without running any of that.
    Once it says yes, the ordinary lookup is safe and returns the real dict.
    """
    if not inspect.isclass(obj):
        return None
    if inspect.getattr_static(obj, "model_fields", None) is None:
        return None
    return getattr(obj, "model_fields", None)


def _signature(obj: Any) -> str:
    fields = _model_fields(obj)
    if fields is not None:
        parts = []
        for name, field in fields.items():
            if name.startswith("_"):
                continue
            shown = f"{name}: {_readable_type(str(field.annotation))}"
            if not field.is_required():
                shown += f" = {_short_default(field.default)}"
            parts.append(shown)
        if len(parts) > MAX_FIELDS:
            rest = len(parts) - MAX_FIELDS
            parts = parts[:MAX_FIELDS] + [f"... and {rest} more field" + ("s" if rest > 1 else "")]
        return "(" + ", ".join(parts) + ")"
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return ""
    parts = []
    for name, param in sig.parameters.items():
        if name.startswith("_"):
            continue  # a private dataclass field is not part of the public call
        shown = name
        if param.kind is param.VAR_POSITIONAL:
            shown = "*" + shown
        elif param.kind is param.VAR_KEYWORD:
            shown = "**" + shown
        if param.annotation is not param.empty:
            shown += f": {_readable_type(str(param.annotation))}"
        if param.default is not param.empty:
            shown += f" = {_short_default(param.default)}"
        parts.append(shown)
    if any(p.kind is p.KEYWORD_ONLY for p in sig.parameters.values()) and not any(
        p.kind is p.VAR_POSITIONAL for p in sig.parameters.values()
    ):
        cut = next(i for i, p in enumerate(sig.parameters.values()) if p.kind is p.KEYWORD_ONLY)
        parts.insert(cut, "*")
    rendered = "(" + ", ".join(parts) + ")"
    if sig.return_annotation is not sig.empty:
        rendered += f" -> {_readable_type(str(sig.return_annotation))}"
    return rendered


def _kind(obj: Any) -> str:
    if inspect.isclass(obj):
        if _model_fields(obj) is not None:
            return "spec"
        if issubclass(obj, BaseException):
            return "exception"
        return "class"
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "function"
    return "callable" if callable(obj) else "value"


def _summary(obj: Any) -> str:
    """The first paragraph of the docstring, joined into one line."""
    doc = inspect.getdoc(obj) or ""
    for para in doc.split("\n\n"):
        line = " ".join(x.strip() for x in para.splitlines() if x.strip())
        if line:
            return line
    return ""


def _code_cells(nb: Path) -> Any:
    for cell in json.loads(nb.read_text())["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell["source"])
        source = "\n".join(
            ln for ln in source.splitlines() if not ln.lstrip().startswith(("%", "!"))
        )
        try:
            yield source, ast.parse(source)
        except SyntaxError:
            continue


def _mentions(node: ast.AST, name: str) -> bool:
    return any(
        (isinstance(n, ast.Name) and n.id == name)
        or (isinstance(n, ast.Attribute) and n.attr == name)
        for n in ast.walk(node)
    )


def _invokes(node: ast.AST, name: str) -> bool:
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        if (isinstance(fn, ast.Name) and fn.id == name) or (
            isinstance(fn, ast.Attribute) and fn.attr == name
        ):
            return True
    return False


def _usage(name: str, notebooks: list[Path]) -> dict[str, str] | None:
    """The shortest executed statement that uses ``name``, preferring a call.

    Import statements are skipped on purpose: ``from axiom.surface import Hill``
    demonstrates nothing about how ``Hill`` is called.
    """
    best: tuple[tuple[int, int, int, int], dict[str, str]] | None = None
    for nb in notebooks:
        for source, tree in _code_cells(nb):
            # Every statement, not only the top-level ones: when the sole call sits
            # inside a helper the reader wants that line, not the whole function.
            for stmt in [n for n in ast.walk(tree) if isinstance(n, ast.stmt)]:
                if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                    continue
                if not _mentions(stmt, name):
                    continue
                segment = ast.get_source_segment(source, stmt)
                if not segment:
                    continue
                lines = segment.splitlines()
                if stmt.col_offset:
                    pad = " " * stmt.col_offset
                    lines = [lines[0]] + [
                        ln[len(pad) :] if ln.startswith(pad) else ln for ln in lines[1:]
                    ]
                    segment = "\n".join(lines)
                defines = isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                rank = (
                    0 if _invokes(stmt, name) else 1,
                    defines,
                    len(lines) > MAX_USAGE_LINES,
                    len(lines),
                )
                if best is None or rank < best[0]:
                    if len(lines) > MAX_USAGE_LINES:
                        segment = "\n".join(lines[:MAX_USAGE_LINES]) + "\n    # ..."
                    best = (rank, {"code": segment, "notebook": nb.relative_to(ROOT).as_posix()})
        if best is not None and best[0][0] == 0:
            break
    return best[1] if best else None


@section
def api() -> Any:
    import importlib

    nbs = ROOT / "nbs"
    packages = []
    total = with_usage = 0
    for name in ORDER:
        module = importlib.import_module("axiom." + name)
        own = sorted(nbs.glob(f"{name}/*.ipynb"))
        # A symbol demonstrated in another subpackage's series still counts; the
        # fallback is what makes "no worked usage anywhere" mean what it says.
        elsewhere = [p for p in sorted(nbs.rglob("*.ipynb")) if p not in own]
        symbols = []
        for symbol in sorted(getattr(module, "__all__", [])):
            obj = getattr(module, symbol, None)
            used = _usage(symbol, own) or _usage(symbol, elsewhere)
            total += 1
            with_usage += used is not None
            symbols.append(
                {
                    "name": symbol,
                    "kind": _kind(obj),
                    "signature": _signature(obj),
                    "summary": _summary(obj),
                    "usage": used["code"] if used else "",
                    "notebook": used["notebook"] if used else "",
                }
            )
        packages.append(
            {
                "name": name,
                "n": len(symbols),
                "layer": LAYER[name],
                "blurb": BLURB[name],
                "symbols": symbols,
                "n_notebooks": len(own),
            }
        )
        print(f"    axiom.{name}: {len(symbols)} symbols, {len(own)} notebooks")

    return {
        "packages": packages,
        "total_symbols": total,
        "n_packages": len(packages),
        "n_with_usage": with_usage,
        "n_without_usage": total - with_usage,
        "without_usage": [
            f"{p['name']}.{s['name']}" for p in packages for s in p["symbols"] if not s["usage"]
        ],
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
        # -- the same numbers, shaped for the charts that draw them ---------------------
        # The rule's power curve: how often it fires against how much harm is really
        # there. The table states five points of it; the chart is the shape between them,
        # which is what says whether the rule is trigger-happy or asleep.
        "oc_curve": {
            "drift": [row["drift"] for row in oc_rows],
            "stop_probability": [row["stop_probability"] for row in oc_rows],
            "expected_looks": [row["expected_looks"] for row in oc_rows],
        },
        # Every contrast the trial monitors, by the lowest Z it ever showed. One of the
        # twelve reaches its boundary; the pooled arms never come close, which is the
        # whole case study in one picture.
        "contrasts": [
            {
                "label": f"{row['arm_label']} · {row['stratum_label']}",
                "value": row["lowest_z_seen"],
                "colour": "boundary" if row["decision"].startswith("stop") else "ink-3",
                "display": f"{row['lowest_z_seen']:+.2f}",
                "note": (
                    f"stopped at week {row['stopped_week']}"
                    if row["decision"].startswith("stop")
                    else f"never fired in {row['looks_taken']} looks"
                ),
            }
            for row in sorted(monitoring, key=lambda r: r["lowest_z_seen"])
        ],
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
# rutherford: GEIGER-1911, the experiment designed before it is run
# ----------------------------------------------------------------------------------


@section
def rutherford() -> Any:
    """Run the GEIGER-1911 plan and emit what the page draws.

    Everything here comes out of ``nbs/case-studies/rutherford/scattering.py`` —
    the same module the five notebooks share — so the page and the notebooks
    cannot disagree about a number.

    Two encoding notes. The site's charts have a linear scale, and the rates
    here span fourteen orders of magnitude, so the series are emitted as
    ``log10`` and the axes say so. And ``jsonable`` rounds to six decimals,
    which would turn a 5.9e-9 steradian aperture into zero, so anything that
    small travels as a log or as a preformatted string.
    """
    import math

    sys.path.insert(0, str(ROOT / "nbs" / "case-studies" / "rutherford"))
    import scattering as s

    from axiom.design import (
        FisherInformation,
        LookSchedule,
        StoppingRule,
        expected_posterior_sd,
        fisher_information,
        information_fractions,
        obrien_fleming,
        operating_characteristics,
    )

    surface = s.surface()
    floor_m = s.D_CLOSEST / 2.0

    def log10(values: Any) -> Any:
        return [float(np.log10(max(float(v), 1e-30))) for v in np.atleast_1d(values)]

    # -- what the two atoms predict, per steradian per second --------------------------
    grid = np.geomspace(0.4, 175.0, 90)
    curve = {
        "angles": [float(a) for a in grid],
        "log10_hard": log10(s.rate_per_steradian(grid, s.HARD_CENTRE)),
        "log10_diffuse": log10(s.rate_per_steradian(grid, s.DIFFUSE)),
        "log10_background": float(np.log10(s.BACKGROUND_DENSITY)),
    }

    # -- an hour at each angle, believed and then argued with ---------------------------
    aperture = s.widest_aperture(grid, s.HARD_CENTRE)
    believed = s.weight_of_evidence(
        s.rate_per_steradian(grid, s.HARD_CENTRE) * aperture * 3600.0,
        s.rate_per_steradian(grid, s.DIFFUSE) * aperture * 3600.0,
    )
    attacked, _ = s.surviving_evidence(grid, aperture, 3600.0)
    peak = int(np.argmax(believed))
    evidence = {
        "angles": [float(a) for a in grid],
        "log10_believed": log10(believed),
        "log10_attacked": log10(attacked),
        "peak_deg": float(grid[peak]),
        "peak_nats": float(believed[peak]),
        "at_150_believed": float(np.interp(150.0, grid, believed)),
        "at_150_attacked": float(np.interp(150.0, grid, attacked)),
    }

    # -- station by station, and what each keeps ---------------------------------------
    stations = [st for st in s.PLAN if st.foil]
    angles = np.array([st.theta_deg for st in stations])
    omega = s.widest_aperture(angles, s.HARD_CENTRE)
    hours = np.array([st.hours for st in stations])
    per_hour_believed = s.weight_of_evidence(
        s.rate_per_steradian(angles, s.HARD_CENTRE) * omega * 3600.0,
        s.rate_per_steradian(angles, s.DIFFUSE) * omega * 3600.0,
    )
    per_hour_attacked, worst = s.surviving_evidence(angles, omega, 3600.0)
    total_attacked = float((per_hour_attacked * hours).sum())

    plan = []
    for st, ap, hb, ha, wf in zip(
        stations, omega, per_hour_believed, per_hour_attacked, worst, strict=True
    ):
        half, arc = s.slit_for(st.theta_deg, float(ap))
        plan.append(
            {
                "angle": st.theta_deg,
                "role": st.role,
                "hours": st.hours,
                "omega_text": f"{ap:.3g}",
                "log10_omega": float(np.log10(ap)),
                "half_width": round(half, 3),
                "arc": round(arc, 1),
                "bias_pct": 100.0 * s.aperture_bias(st.theta_deg, half, arc, s.HARD_CENTRE),
                "nats_per_hour_believed": float(hb),
                "nats_per_hour_attacked": float(ha),
                "kept": float(ha / hb),
                "worst_core": float(wf),
                "share_of_surviving": float(ha * st.hours / total_attacked),
            }
        )

    # -- how far back the witness has to sit, against a wider and wider core ------------
    scan = np.geomspace(2.0, 179.0, 240)
    scan_ap = s.widest_aperture(scan, s.HARD_CENTRE)
    scan_believed = s.weight_of_evidence(
        s.rate_per_steradian(scan, s.HARD_CENTRE) * scan_ap * 3600.0,
        s.rate_per_steradian(scan, s.DIFFUSE) * scan_ap * 3600.0,
    )
    immunity = []
    for cap in (3.0, 5.0, 10.0, 20.0, 30.0):
        keeps, _ = s.surviving_evidence(scan, scan_ap, 3600.0, factors=np.geomspace(0.3, cap, 48))
        breached = np.where(keeps / scan_believed <= 0.99)[0]
        beyond = (
            None
            if breached.size and breached[-1] + 1 >= scan.size
            else float(scan[breached[-1] + 1] if breached.size else scan[0])
        )
        immunity.append(
            {
                "allowance": cap,
                "rms_deg": math.degrees(s.CORE_WIDTH * cap),
                "beyond_deg": beyond,
            }
        )

    # -- the slit: the same solid angle bought two ways ---------------------------------
    from scipy.optimize import brentq

    slit = []
    for st, ap in zip(stations, omega, strict=True):
        if st.theta_deg < 5.0:
            continue  # a pinhole either way; the shape argument has nothing to say
        half, arc = s.slit_for(st.theta_deg, float(ap))
        hole = brentq(
            lambda d, a=st.theta_deg, o=float(ap): s.slit_solid_angle(a, d, 2 * d) - o,
            1e-7,
            min(st.theta_deg * 0.98, 60.0),
        )
        slit.append(
            {
                "angle": st.theta_deg,
                "slot_half": round(half, 3),
                "slot_arc": round(arc, 1),
                "slot_bias_pct": 100.0 * s.aperture_bias(st.theta_deg, half, arc, s.HARD_CENTRE),
                "hole_half": round(hole, 3),
                "hole_bias_pct": 100.0
                * s.aperture_bias(st.theta_deg, hole, 2 * hole, s.HARD_CENTRE),
            }
        )

    # -- what the plan expects to know, and the bound it reports ------------------------
    info = fisher_information(surface, s.plan_data(), s.truth(math.log(0.30)), 1.0, method="finite")
    assert isinstance(info, FisherInformation)
    posterior_sds = expected_posterior_sd(s.PRIOR_SDS, info)
    assert isinstance(posterior_sds, dict)

    seconds = np.array([st.seconds for st in stations])
    plan_data = s.station(angles, omega, seconds)
    mu_hard = surface.counts(plan_data, s.HARD_CENTRE)
    radii = np.geomspace(3.0e-14, 1.0e-12, 300)
    against = np.array(
        [
            float(
                s.weight_of_evidence(
                    mu_hard, surface.counts(plan_data, s.truth(s.lam_of(float(r))))
                ).sum()
            )
            for r in radii
        ]
    )
    crossed = np.where(against > 3.0)[0]
    bound_m = float(radii[crossed[0]])

    # -- the witness, watched --------------------------------------------------------
    look_hours = (5.0, 15.0, 30.0, 60.0, 110.0)
    looks = LookSchedule(
        labels=tuple(f"{h:g} h" for h in look_hours),
        information=information_fractions(look_hours),
    )
    boundary = obrien_fleming(0.05, looks, side="upper", kind="efficacy")
    rule = StoppingRule(name="witness_150", looks=looks, boundaries=(boundary,))
    signal = (
        float(s.rate_per_steradian([150.0], s.HARD_CENTRE)[0] - s.BACKGROUND_DENSITY) * s.OMEGA_MAX
    )
    background = s.BACKGROUND_DENSITY * s.OMEGA_MAX
    full = 110 * 3600.0
    sources = []
    for factor in (1.0, 1e-2, 1e-3, 3e-4, 1e-4):
        drift = signal * factor * full / math.sqrt((signal * factor + background) * full)
        oc = operating_characteristics(rule, drift=drift)
        sources.append(
            {
                "factor_text": f"{factor:g}",
                "log10_factor": float(np.log10(factor)),
                "drift": float(drift),
                "stop_probability": float(oc.crossings.stop_probability),
                "expected_hours": float(oc.expected_information * 110.0),
            }
        )

    # -- the witness, actually watched -------------------------------------------------
    # A source a thousand times weaker than the modelled one, so the crossing is a real
    # event partway through rather than a foregone conclusion at the first look. The
    # chart wants the same panel shape the HYPER-3 monitoring uses.
    weak = 1e-3
    elapsed = np.array(look_hours) * 3600.0
    drawn = np.random.default_rng(1911).poisson(
        np.diff(np.concatenate([[0.0], (signal * weak + background) * elapsed]))
    )
    cumulative = np.cumsum(drawn)
    z_path = (cumulative - background * elapsed) / np.sqrt(np.maximum(cumulative, 1))
    crossing = next(
        (i for i, (zz, th) in enumerate(zip(z_path, boundary.z, strict=True)) if zz >= th), None
    )
    taken = len(z_path) if crossing is None else crossing + 1
    witness_panel = {
        "stratum": "witness_150",
        "label": f"150° · a source {weak:g} times the modelled one",
        "information": list(looks.information),
        "boundary": [float(z) for z in boundary.z],
        "z": [float(v) for v in z_path],
        "looks_taken": taken,
        "weeks": [float(h) for h in look_hours],
        "stopped": crossing is not None,
        "stop_information": None if crossing is None else float(looks.information[crossing]),
        "stop_z": None if crossing is None else float(z_path[crossing]),
        "stop_week": None if crossing is None else float(look_hours[crossing]),
        "effects": [float(c - background * t) for c, t in zip(cumulative, elapsed, strict=True)],
        "ses": [float(np.sqrt(max(c, 1))) for c in cumulative],
    }

    # -- one run of it ----------------------------------------------------------------
    observed = np.random.default_rng(1911).poisson(surface.counts(s.plan_data(), s.HARD_CENTRE))
    if_diffuse = surface.counts(s.plan_data(), s.DIFFUSE)
    counts = [
        {
            "angle": st.theta_deg,
            "foil": bool(st.foil),
            "role": st.role,
            "hours": st.hours,
            "observed": int(n),
            "if_diffuse_text": f"{d:,.0f}",
        }
        for st, n, d in zip(s.PLAN, observed, if_diffuse, strict=True)
    ]

    return {
        "apparatus": {
            "energy_mev": s.E_ALPHA,
            "foil_um": s.FOIL_THICKNESS * 1e6,
            "beam_rate": s.BEAM_RATE,
            "atoms_crossed": s.n_encounters(),
            "max_count_rate": s.MAX_COUNT_RATE,
            "max_per_minute": s.MAX_COUNT_RATE * 60.0,
            "hours": sum(st.hours for st in s.PLAN),
            "d_closest_fm": s.D_CLOSEST * 1e15,
            "floor_fm": floor_m * 1e15,
            "slit_min_half": s.SLIT_MIN_HALF_WIDTH,
        },
        "atoms": {
            "diffuse": {
                "radius_fm": s.R_ATOM * 1e15,
                "radius_text": f"{s.R_ATOM:.3g}",
                "cutoff_deg": math.degrees(2 * math.asin(math.exp(s.LAM_DIFFUSE))),
            },
            "hard": {
                "radius_fm": s.R_NUCLEUS * 1e15,
                "radius_text": f"{s.R_NUCLEUS:.3g}",
                "decades_apart": (s.LAM_HARD - s.LAM_DIFFUSE) / math.log(10),
            },
        },
        "curve": curve,
        "evidence": evidence,
        "plan": plan,
        "immunity": immunity,
        "slit": slit,
        # -- the same numbers, shaped for the charts that draw them ---------------------
        # Evidence spans five orders of magnitude across the stations, so the paired
        # comparison is drawn in log10; the point of the picture is the *gap*, and a
        # linear axis would put every station but one on the baseline.
        "station_evidence": [
            {
                "label": f"{row['angle']:.1f}°  {row['role']}",
                "a": float(np.log10(max(row["nats_per_hour_believed"], 1e-3))),
                "b": float(np.log10(max(row["nats_per_hour_attacked"], 1e-3))),
            }
            for row in plan
        ],
        "slit_compare": [
            {"label": f"{row['angle']:.0f}°", "a": row["hole_bias_pct"], "b": row["slot_bias_pct"]}
            for row in slit
        ],
        "immunity_curve": {
            "allowance": [row["allowance"] for row in immunity],
            "beyond": [row["beyond_deg"] for row in immunity],
            "rms": [row["rms_deg"] for row in immunity],
        },
        "counts_compare": [
            {
                "label": f"{st.theta_deg:.1f}°",
                # The diffuse atom predicts exactly zero at several stations. Floored at a
                # tenth of a count so the axis stays finite; the caption says so.
                "a": float(np.log10(max(float(d), 0.1))),
                "b": float(np.log10(max(float(n), 0.1))),
            }
            for st, n, d in zip(s.PLAN, observed, if_diffuse, strict=True)
            if st.foil
        ],
        "panels": [witness_panel],
        "sequential": {
            "look_hours": list(look_hours),
            "information": list(looks.information),
            "thresholds": [float(z) for z in boundary.z],
            "sources": sources,
            "signal_per_hour": signal * 3600.0,
            "background_per_hour": background * 3600.0,
        },
        "answer": {
            "bound_fm": bound_m * 1e15,
            "bound_text": f"{bound_m:.3g}",
            "floor_fm": floor_m * 1e15,
            "rutherford_1911_fm": 34.0,
            "gap_pct": 100.0 * (bound_m - floor_m) / floor_m,
            "posterior_sds": {k: float(v) for k, v in posterior_sds.items()},
            "evidence_believed": float((per_hour_believed * hours).sum()),
            "evidence_attacked": total_attacked,
            "witness_share": float((per_hour_attacked[-2:] * hours[-2:]).sum() / total_attacked),
            "counts": counts,
        },
    }


# ----------------------------------------------------------------------------------
# tutorial: one problem carried through all eight phases, a step at a time
# ----------------------------------------------------------------------------------

#: Porting your own problem. The other two series both start from a world that
#: already exists — ``surface_world(...)`` in one and ``import scattering`` in the
#: other — which is convenient for telling a story and useless if what you have
#: is a dataframe. Nothing below imports ``axiom.sim``. Every object a user has
#: to author is authored on the page: the roles, the entities and their
#: dimensions, the spec, the estimand.
PORTING_STEPS: list[dict[str, Any]] = [
    {
        "slug": "porting-1-your-data",
        "title": "Start from the dataframe you already have",
        "asks": "What does axiom need to know about your table?",
        "lede": (
            "Long format: one row per unit per period, a column for the treatment "
            "and a column for the outcome. That is the whole data requirement, and "
            "it is deliberately the shape almost everyone already has."
        ),
        "beat": (
            "What axiom will not do is guess which column is which. A `RoleMap` "
            "names the unit, the time, the outcome and every treatment explicitly. "
            "Column-name conventions are how a spend column gets read as an outcome "
            "in somebody's third refactor, and the map is five lines against that."
        ),
        "code": """
            import numpy as np
            import pandas as pd

            # Whatever produced your table. This one is 24 stores over 40 weeks:
            # leaflet spend, and footfall that responds to it with diminishing
            # returns. Substitute your own read_csv here -- nothing below cares.
            rng = np.random.default_rng(7)
            rows = []
            for store in range(24):
                base = 40.0 + rng.normal(0, 6)          # stores differ
                for week in range(40):
                    spend = float(max(0.0, rng.normal(300, 160)))
                    rows.append({
                        "store": f"s{store:02d}",
                        "week": week,
                        "leaflets": spend,
                        "footfall": base + 9.0 * spend / (spend + 250.0)
                                    + rng.normal(0, 2),
                    })
            frame = pd.DataFrame(rows)
            print(frame.head(3).to_string(index=False))
            print(f"\\n{len(frame)} rows, {frame['store'].nunique()} stores, "
                  f"{frame['week'].nunique()} weeks")
        """,
    },
    {
        "slug": "porting-2-entities",
        "title": "Say what the columns mean",
        "asks": "Why does a column need a dimension and a unit?",
        "lede": (
            "A `Treatment` and an `Outcome` are not labels. They carry a dimension "
            "and a unit, and those travel with every number derived from them — "
            "into the estimand, into the transfer ledger, onto the report."
        ),
        "beat": (
            "This is what stops dollars being added to visits four steps later, in "
            "a function nobody is reading at the time. `D.currency` and `D.outcome` "
            "are different dimensions, and axiom refuses the arithmetic that would "
            "mix them rather than returning a number with no meaning."
        ),
        "code": """
            from axiom.core import D, Outcome, Treatment
            from axiom.data import Panel, RoleMap

            leaflets = Treatment(
                name="leaflets",
                dimension=D.currency,
                unit="USD/store-week",
                description="leaflet spend delivered to the store's catchment",
            )
            footfall = Outcome(
                name="footfall",
                dimension=D.outcome,
                unit="visits/week",
                description="counted entries through the door",
                aggregation="mean",
            )

            roles = RoleMap(
                unit="store",
                time="week",
                outcome=("footfall", footfall),
                treatments={"leaflets": leaflets},
            )
            panel = Panel(frame, roles)

            print(f"unit column      : {roles.unit}")
            print(f"time column      : {roles.time}")
            print(f"outcome column   : {roles.outcome[0]}  [{footfall.dimension}]")
            print(f"treatment columns: {list(roles.treatments)}  "
                  f"[{leaflets.dimension}]")
            print()
            print(f"the two dimensions are different objects: "
                  f"{D.currency != D.outcome}")
        """,
    },
    {
        "slug": "porting-3-the-spec",
        "title": "Write the spec",
        "asks": "What shape do you believe the response has, and how sure are you?",
        "lede": (
            "A `SurfaceSpec` is the model, and it is the object most worth "
            "understanding rather than copying. One kernel per treatment says what "
            "shape the dose-response is allowed to take; the intercept says what "
            "the units are allowed to differ by; the scales are your priors."
        ),
        "beat": (
            "None of these is a default you can skip past. `HillKernel` says "
            "saturating with a half-way point near the reference dose — say that "
            "and a linear response can no longer be fitted, which is the point of "
            "saying it. The spec hashes, so the model that produced a number is "
            "identifiable later by more than a filename."
        ),
        "code": """
            from axiom.surface import HillKernel, SurfaceSpec

            spec = SurfaceSpec(
                name="leaflets_footfall",
                treatments=(leaflets,),
                outcome=footfall,
                # saturating, with the half-way point expected near 250 USD and
                # a lift of order 10 visits at saturation
                kernels={"leaflets": HillKernel(
                    reference_dose=250.0, amplitude_scale=10.0,
                )},
                intercept="shared",
                unit_column="store",
                time_column="week",
                intercept_scale=8.0,
                noise_scale=3.0,
            )
            print(f"model      : {spec.name}")
            print(f"content hash: {spec.content_hash()[:16]}")
            print(f"treatments : {[t.name for t in spec.treatments]}")
            print(f"intercept  : {spec.intercept}")
            print()
            print("`shared` says the stores share one baseline. They do not -- the")
            print("next step shows what that costs, and it is visible in sigma.")
        """,
    },
    {
        "slug": "porting-4-fit-it",
        "title": "Fit it, and read what came back",
        "asks": "Did the model find the response, and what did the spec cost you?",
        "lede": (
            "`fit` takes the spec and the panel and nothing else. The posterior "
            "comes back with one named parameter per thing the spec declared, so "
            "there is no positional unpacking and no guessing which column of an "
            "array is the amplitude."
        ),
        "beat": (
            "Look at `sigma` before anything else. The stores really do differ by "
            "about 6 visits, and a shared intercept has nowhere to put that, so it "
            "lands in the residual. The fit is not wrong — it is answering the "
            "question the spec asked. Changing `intercept` is how you ask a "
            "different one."
        ),
        "code": """
            from axiom.surface import fit

            result = fit(spec, panel, backend="laplace", draws=800, seed=0)

            print(f"{'parameter':16s}{'posterior mean':>16}")
            for name in result.posterior.names():
                mean = float(result.posterior.flat(name).mean())
                print(f"{name:16s}{mean:>16.2f}")
            print()
            print("the world that produced the data used:")
            print("  baseline  ~40      amplitude 9.0     half-way 250")
            print("  within-store noise 2.0, between-store spread 6.0")
            print()
            print("beta and k came back close. sigma did not -- it is carrying the")
            print("between-store spread the shared intercept could not.")
        """,
    },
    {
        "slug": "porting-5-the-estimand",
        "title": "Declare what you want to know",
        "asks": "How do you get a decision's number out of a fitted surface?",
        "lede": (
            "A fitted surface is not an answer. The answer is a specific "
            "comparison, over a specific population, in a specific window, at a "
            "specific level — and axiom makes you write that down as an `Estimand` "
            "before it will produce a number for it."
        ),
        "beat": (
            "That looks like ceremony until the day the window in your head and the "
            "window in the fit disagree. `realize` compares them and carries a "
            "ledger of every step it took to cross between them; the ledger is the "
            "part you show someone who has to trust the number."
        ),
        "code": """
            from axiom.core import Intervention, Population, TimeWindow
            from axiom.estimands import Estimand, Level, Quantity, realize

            lift = Estimand(
                name="lift_at_400_per_week",
                quantity=Quantity(kind="contrast"),
                treatment=leaflets,
                intervention=Intervention(doses={"leaflets": 400.0}),
                reference=Intervention(doses={"leaflets": 0.0}),
                outcome=footfall,
                population=Population(name="the_24_stores"),
                window=TimeWindow(start=0, stop=1, basis="per_period"),
                level=Level(unit="individual"),
                dimension=D.outcome,
            )
            got = realize(lift, result, assume_identified=True, mass=0.9, seed=0)

            print(f"{lift.name}")
            print(f"  {got.summary.mean:+.2f} {footfall.unit}  {got.summary.interval}")
            print(f"  status : {got.status}")
            print()
            for line in got.ledger:
                print(f"  [{line.kind}] {line.statement}")
        """,
    },
    {
        "slug": "porting-6-what-is-missing",
        "title": "What you have not done yet",
        "asks": "The number came back downgraded. Why, and what fixes it?",
        "lede": (
            "`assume_identified=True` is the flag that got a number out, and the "
            "status says exactly what it cost: `downgraded`. Nothing above argued "
            "that leaflet spend is unconfounded with footfall. Stores that were "
            "already busy may well have been given more leaflets."
        ),
        "beat": (
            "This is the honest end of a porting exercise, not a failure of one. "
            "You now have a model of your own data that produces a decision's "
            "number with its provenance attached — and a precise statement of the "
            "one thing standing between it and a causal claim. That is where the "
            "first tutorial starts."
        ),
        "code": """
            from axiom.identify import CausalGraph, identify

            # Write down what you actually believe about how the spend was set.
            world = CausalGraph.from_edges(
                "busyness -> leaflets, busyness -> footfall, leaflets -> footfall",
                unmeasured=["busyness"],
                name="how_the_spend_was_really_set",
            )
            verdict = identify(world, "leaflets", "footfall")
            print(f"the graph says : {verdict.status}")
            print(f"it would need  : {sorted(verdict.unmeasured_required)}, "
                  f"which nobody recorded")
            print()
            print("so the `downgraded` in step 5 was not a formality. What you have")
            print("is a description of 24 stores, not the effect of leaflets on")
            print("footfall -- and the difference is the whole subject of axiom.")
            print()
            print("two honest ways forward, and no third:")
            print("  measure it   -- record what drove the spend, and adjust")
            print("  randomize it -- assign the leaflets, and the arrow disappears")
            print()
            print("both are the subject of the other two tutorials.")
        """,
    },
]


#: GEIGER-1911, a step to a page. The case study's own world module does the
#: physics (``nbs/case-studies/rutherford/scattering.py``); every step below is a
#: decision about the apparatus, taken against it.
SCATTERING_STEPS: list[dict[str, Any]] = [
    {
        "slug": "scattering-1-one-parameter",
        "title": "Two atoms, one number",
        "asks": "How do you get resolving power against a question nobody has measured?",
        "lede": (
            "In 1910 there were two pictures of the atom and no experiment between "
            "them. A hard positive centre and a charge smeared through the whole atom "
            "both fitted everything anyone had. You cannot design against a debate — "
            "so the first move is to find the number the two pictures disagree about."
        ),
        "beat": (
            "They are one model at two values of the radius R of the positive charge, "
            "six decades apart. An alpha stopped head-on gets no closer than D; one "
            "scattered through an angle gets no closer than a known multiple of it; so "
            "a charge of radius R kills the scattering past a cut-off angle. Turning "
            "the debate into a parameter is what makes everything after this possible."
            "\n\n"
            "`scattering.py` is the case study's own module and it holds the physics: "
            "the mean expression, the parameters and their priors, the detector "
            "geometry. It is a hand-built `ModelSpec` rather than anything axiom "
            "generates, which is what a problem with real physics in it looks like. "
            "The porting tutorial builds the easier kind, a `SurfaceSpec`, from "
            "scratch."
        ),
        "code": """
            import sys
            sys.path.insert(0, "nbs/case-studies/rutherford")
            import scattering as s

            print(f"closest approach D        : {s.D_CLOSEST * 1e15:.1f} fm")
            print(f"a hard centre would be    : {s.R_NUCLEUS * 1e15:.1f} fm")
            print(f"a diffuse atom would be   : {s.R_ATOM * 1e15:,.0f} fm")
            print()
            for name, radius in (("hard centre", s.R_NUCLEUS), ("diffuse atom", s.R_ATOM)):
                sine = s.cutoff_sine(radius)
                where = "no cut-off in range" if sine >= 1.0 else f"{sine:.2e}"
                print(f"{name:14s} sin(theta_cut/2) = {where}")
            print()
            print("the two hypotheses are two values of one parameter, lam:")
            print(f"  hard centre  lam = {s.LAM_HARD:+.2f}")
            print(f"  diffuse atom lam = {s.LAM_DIFFUSE:+.2f}")
        """,
    },
    {
        "slug": "scattering-2-which-angles",
        "title": "Where to point the counter",
        "asks": "Which angles carry the evidence — and which carry it under attack?",
        "lede": (
            "Evidence per hour is easy to compute: how many nats an hour at each angle "
            "buys, if you believe the model. Follow it and you put the whole budget "
            "where the counts are. That is the trap this step exists to name."
        ),
        "beat": (
            "Evidence-per-hour is computed *inside the model that is under attack*. An "
            "objector who says the multiple-scattering core is wider than you fitted "
            "takes most of it away. `surviving_evidence` asks what is left when they "
            "do — and the answer is a different set of angles, far out in the tail."
        ),
        "code": """
            import numpy as np

            angles = np.array([1.0, 2.5, 5.0, 10.0, 20.0, 45.0, 90.0, 150.0])
            omega = s.widest_aperture(angles, s.HARD_CENTRE)

            believed = s.weight_of_evidence(
                s.rate_per_steradian(angles, s.HARD_CENTRE) * omega * 3600.0,
                s.rate_per_steradian(angles, s.DIFFUSE) * omega * 3600.0,
            )
            attacked, _ = s.surviving_evidence(angles, omega, 3600.0)

            print(f"{'angle':>8} {'nats/h believed':>18} {'nats/h attacked':>18}")
            for a, b, k in zip(angles, believed, attacked):
                print(f"{a:7.1f}d {b:18,.0f} {k:18,.1f}")
            print()
            best_believed = angles[int(np.argmax(believed))]
            best_attacked = angles[int(np.argmax(attacked))]
            print(f"believe the model, and the best angle is {best_believed:.1f} degrees")
            print(f"let it be attacked, and it is       {best_attacked:.1f} degrees")
        """,
    },
    {
        "slug": "scattering-3-the-slit",
        "title": "What shape to cut the aperture",
        "asks": "Same solid angle, same counts — does the shape matter?",
        "lede": (
            "A counter needs a hole to look through, and a bigger hole means more "
            "counts. The obvious hole is round. The scattering does not depend on "
            "azimuth, though, which means the two directions of a hole are not "
            "equivalent at all."
        ),
        "beat": (
            "Only the radial half-width smears the angle; azimuth is free resolution. "
            "So hold the radial width at the workshop's floor and take every extra "
            "steradian out in arc. A round hole of the same area at 90 degrees would "
            "have to be many degrees wide and would report a rate belonging to no "
            "angle in particular — the same counts, at a fraction of the resolution."
        ),
        "code": """
            print(f"{'angle':>7} {'radial':>9} {'arc':>9} {'slot bias':>11} {'hole bias':>11}")
            for theta in (5.0, 45.0, 90.0, 150.0):
                omega_here = float(s.widest_aperture([theta], s.HARD_CENTRE)[0])
                half, arc = s.slit_for(theta, omega_here)
                slot = 100.0 * s.aperture_bias(theta, half, arc, s.HARD_CENTRE)
                # a round hole of the same solid angle: equal in both directions
                hole = np.degrees(np.sqrt(omega_here / np.pi))
                hole_bias = 100.0 * s.aperture_bias(theta, hole, 2 * hole, s.HARD_CENTRE)
                print(f"{theta:6.1f}d {half:8.2f}d {arc:8.1f}d "
                      f"{slot:10.3f}% {hole_bias:10.2f}%")
            print()
            print("same solid angle, same counts. the slot reports the angle it is at;")
            print("the hole reports an average over everything it can see.")
        """,
    },
    {
        "slug": "scattering-4-how-long",
        "title": "How long at each, and what the plan buys",
        "asks": "Two hundred hours. Where do they go, and what comes back?",
        "lede": (
            "The plan is nine stations in four roles: anchors that pin the "
            "multiple-scattering core, a bank where the information is, witnesses whose "
            "evidence survives the core being argued with, and a foil-out run for the "
            "background. The hours are the argument."
        ),
        "beat": (
            "A hundred and ten of the two hundred go to 150 degrees — the opposite of "
            "what evidence-per-hour says, and right for the reason step two gave. The "
            "bound that comes back is R < 33 fm, against a floor of 29.6 fm that no "
            "amount of counting can beat, because the beam energy alone sets it."
        ),
        "code": """
            print(f"{'angle':>8} {'role':>11} {'hours':>7}")
            for st in s.PLAN:
                where = f"{st.theta_deg:.1f}d" + ("" if st.foil else " out")
                print(f"{where:>8} {st.role:>11} {st.hours:7.0f}")
            print(f"{'':>8} {'total':>11} {sum(x.hours for x in s.PLAN):7.0f}")
            print()

            surface = s.surface()
            plan = s.plan_data()
            mu_hard = surface.counts(plan, s.HARD_CENTRE)
            radii = np.geomspace(3.0e-14, 1.0e-12, 300)
            against = np.array([
                float(s.weight_of_evidence(
                    mu_hard, surface.counts(plan, s.truth(s.lam_of(float(r))))
                ).sum())
                for r in radii
            ])
            bound = float(radii[np.where(against > 3.0)[0][0]])
            print(f"the plan can separate a charge larger than {bound * 1e15:.1f} fm")
            print(f"the beam's own floor is                    {s.D_CLOSEST / 2 * 1e15:.1f} fm")
        """,
    },
    {
        "slug": "scattering-5-when-to-stop",
        "title": "When to stop",
        "asks": "The witness runs for 110 hours. Do you have to wait?",
        "lede": (
            "A hundred and ten hours at one angle is a long time to learn nothing you "
            "could not have learned in thirty. The same sequential machinery that stops "
            "a clinical trial early applies here: look at the witness on a schedule, "
            "against a boundary that spends the error rate across the looks."
        ),
        "beat": (
            "The boundary is what makes looking legitimate. Peeking at a running count "
            "and stopping when it looks good is how you manufacture a discovery; an "
            "O'Brien-Fleming boundary is the price of being allowed to look at all."
        ),
        "code": """
            import math
            from axiom.design import (
                LookSchedule, StoppingRule, information_fractions,
                obrien_fleming, operating_characteristics,
            )

            look_hours = (5.0, 15.0, 30.0, 60.0, 110.0)
            looks = LookSchedule(
                labels=tuple(f"{h:g} h" for h in look_hours),
                information=information_fractions(look_hours),
            )
            boundary = obrien_fleming(0.05, looks, side="upper", kind="efficacy")
            rule = StoppingRule(name="witness_150", looks=looks, boundaries=(boundary,))

            print(f"{'look':>8} {'information':>12} {'stop above z':>13}")
            for label, frac, z in zip(looks.labels, looks.information, boundary.z):
                print(f"{label:>8} {frac:12.2f} {z:13.3f}")

            signal = float(
                s.rate_per_steradian([150.0], s.HARD_CENTRE)[0] - s.BACKGROUND_DENSITY
            ) * s.OMEGA_MAX
            background = s.BACKGROUND_DENSITY * s.OMEGA_MAX
            full = 110 * 3600.0
            print()
            for factor in (1.0, 1e-3, 1e-4):
                drift = signal * factor * full / math.sqrt((signal * factor + background) * full)
                oc = operating_characteristics(rule, drift=drift)
                print(f"a source {factor:7.0e} of the modelled one: "
                      f"P(stop) = {oc.crossings.stop_probability:.3f}")
        """,
    },
    {
        "slug": "scattering-6-running-it",
        "title": "Running it",
        "asks": "The counts come in. What does the experiment actually say?",
        "lede": (
            "Every station is a Poisson draw around what the apparatus would really "
            "have produced. Then the five parameters are profiled over the one the "
            "experiment is about, and the bound is read where the likelihood has "
            "climbed far enough above its minimum."
        ),
        "beat": (
            "It comes back one-sided, and that is the honest shape: there is no lower "
            "limit on the radius here. Everything below the bound fits the counts "
            "equally well, because the beam cannot resolve past D/2 whatever the "
            "counting time. Rutherford published 34 fm in 1911."
        ),
        "code": """
            import warnings
            from axiom.design import profile_likelihood

            counts = np.random.default_rng(1911).poisson(surface.counts(plan, s.HARD_CENTRE))
            measured = dict(plan)
            measured["root_count"] = 2.0 * np.sqrt(counts)

            inn = [bool(st.foil) for st in s.PLAN]
            if_diffuse = surface.counts(plan, s.DIFFUSE)
            print(f"at 150 degrees: {int(counts[inn][-1]):,} counts observed")
            print(f"  a diffuse atom predicts {float(if_diffuse[inn][-1]):.0f}")
            print(f"  that is a Poisson excess of "
                  f"{(counts[inn][-1] - if_diffuse[inn][-1]) / math.sqrt(counts[inn][-1]):,.0f} sd")

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                values, drops = profile_likelihood(
                    s.model(), measured, {**s.HARD_CENTRE, "lam": 0.0},
                    "lam", grid=np.linspace(-4, 4, 41),
                )
            drops = np.asarray(drops)
            lowest = int(np.argmin(drops))
            last = int(np.where(drops[:lowest] > 1.921)[0][-1])
            crossing = float(np.interp(
                1.921, [drops[last + 1], drops[last]], [values[last + 1], values[last]]
            ))
            print()
            print(f"R < {s.radius_of(crossing) * 1e15:.1f} fm   (95%, one-sided)")
            print(f"the beam could not have resolved below {s.D_CLOSEST / 2 * 1e15:.1f} fm")
            print("Rutherford published 34 fm in 1911.")
        """,
    },
]


#: The tutorial series. Each is one problem carried through every phase it has
#: to survive, a step to a page, and each step's snippet runs into the *same*
#: namespace as the ones before it in its own series. That sharing is the whole
#: point: a tutorial whose steps do not carry state is a tour with numbers in it.
#:
#: Two of them, because the two shapes of causal work read differently. The depot
#: problem is an *analysis* story — the data is already there and the question is
#: whether it can answer anything. GEIGER-1911 is a *design* story: there is no
#: data at all, and every decision is about what to go and measure.
TUTORIALS: list[dict[str, Any]] = [
    {
        "key": "porting",
        "title": "Bringing your own problem",
        "kind": "A porting story",
        "asks": "You have a dataframe. How do you get it into axiom at all?",
        "problem": (
            "Twenty-four stores, forty weeks, leaflet spend and footfall. How do you "
            "turn a table you already have into a model axiom can fit, an estimand it "
            "can read, and a number you could defend?"
        ),
        "lede": (
            "The other two series start from a world that already exists — a simulator "
            "in one, a physics module in the other — which is fine for telling a story "
            "and no help at all when what you have is a CSV. Nothing in this one "
            "imports axiom.sim. Every object you would have to write is written on the "
            "page: the roles, the entities and their dimensions, the spec, the estimand."
        ),
        "notebook": "nbs/surface/",
        "steps": PORTING_STEPS,
    },
    {
        "key": "depot",
        "title": "One question, carried all the way",
        "kind": "An analysis story",
        "asks": "The data is already here. Can it answer anything?",
        "problem": (
            "A logistics operator runs 40 distribution depots in one region and 500 "
            "nationally. Should the standing weekly maintenance schedule go from "
            "nothing to 60 hours per depot?"
        ),
        "lede": (
            "Identification refuses the panel on hand, so an experiment is designed, "
            "sized against the boundary the decision turns on, and run. Its answer is "
            "folded back into the model — and it reverses the decision the "
            "observational fit would have made."
        ),
        "notebook": "nbs/tutorial/01-the-whole-loop.ipynb",
        "steps": [
            {
                "slug": "tutorial-1-the-question",
                "title": "The question, written down",
                "asks": "What are we actually deciding, and what would 'yes' have to beat?",
                "lede": (
                    "A logistics operator runs 40 depots in one region and 500 nationally. "
                    "Should the standing weekly maintenance schedule go from nothing to 60 "
                    "hours per depot? Before any data is touched, the decision has to be "
                    "written down as a quantity — because the number that answers it is not "
                    "the number an experiment can most easily measure."
                ),
                "beat": (
                    "Two estimands, not one. The experiment can measure a first-week lift on "
                    "individual depots; the decision turns on a steady-state weekly lift "
                    "across the region. They are different windows and different levels, and "
                    "keeping them apart from the first line is what stops the report answering "
                    "the easy question and calling it the hard one.\n\n"
                    "`surface_world` is a simulator, and it hands back the `spec` for "
                    "free so this story can get moving. With your own data there is no "
                    "simulator and the spec is the first thing you write -- the porting "
                    "tutorial does exactly that, from a dataframe, with nothing hidden."
                ),
                "code": """
            from axiom.core import Intervention, Population, TimeWindow
            from axiom.estimands import Estimand, Level, Quantity
            from axiom.sim import DosePlan, surface_world
            from axiom.surface import GeometricCarryover, HillKernel

            DOSE, NO_DOSE = 60.0, 0.0        # maintenance hours per depot-week
            VALUE_PER_POINT, COST_PER_HOUR = 400.0, 45.0
            REGION_DEPOTS, MAX_LAG, SEED = 40, 4, 0

            world = surface_world(
                n_units=REGION_DEPOTS, n_periods=52, treatments=("a",),
                kernels=HillKernel(reference_dose=50.0, amplitude_scale=10.0),
                carryover={"a": GeometricCarryover(max_lag=MAX_LAG)},
                doses=DosePlan(scale=50.0, spread=0.8, zero_fraction=0.05),
                intercept="shared",
                truth={"beta_a": 10.0, "alpha": 5.0, "k_a": 50.0, "s_a": 2.0, "lam_a": 0.5},
                noise_sd=2.0, seed=SEED,
            )
            spec = world.spec

            def contrast(name, window, level):
                return Estimand(
                    name=name, quantity=Quantity(kind="contrast"),
                    treatment=spec.treatment("a"),
                    intervention=Intervention(doses={"a": DOSE}),
                    reference=Intervention(doses={"a": NO_DOSE}),
                    outcome=spec.outcome, population=Population(name="region_depots"),
                    window=window, level=level, dimension=spec.outcome_dimension,
                )

            experiment_estimand = contrast(
                "first_week_lift",
                TimeWindow(start=0, stop=1, basis="cumulative"),
                Level(unit="individual"),
            )
            decision_estimand = contrast(
                "steady_state_weekly_lift",
                TimeWindow(start=MAX_LAG, stop=52, basis="per_period"),
                Level(unit="cluster"),
            )

            BREAK_EVEN = DOSE * COST_PER_HOUR / VALUE_PER_POINT
            print(f"the experiment can measure : {experiment_estimand.name}")
            print(f"the decision turns on      : {decision_estimand.name}")
            print()
            print(f"{DOSE:.0f} hours at {COST_PER_HOUR:.0f} USD "
                  f"= {DOSE * COST_PER_HOUR:,.0f} USD per depot-week")
            print(f"break-even steady-state lift: {BREAK_EVEN:.2f} index points per depot-week")
        """,
            },
            {
                "slug": "tutorial-2-identification",
                "title": "Can the data we already have answer it?",
                "asks": "Is the effect identified from the panel already on hand?",
                "lede": (
                    "The operator has a panel: depots, weeks, maintenance hours, an outcome "
                    "index. The temptation is to fit it. The question axiom asks first is "
                    "whether that panel can produce the number the decision needs at all."
                ),
                "beat": (
                    "It cannot. Depots that were already doing well got more maintenance hours, "
                    "and the thing that drove both was never recorded. `identify` says so and "
                    "names what it would need — which is the answer that saves the money, "
                    "because the alternative is a fitted number nobody can defend."
                ),
                "code": """
            from axiom.identify import CausalGraph, identify

            observed = CausalGraph.from_edges(
                "capability -> hours, capability -> outcome, hours -> outcome",
                unmeasured=["capability"], name="the_panel_on_hand",
            )
            verdict = identify(observed, "hours", "outcome")
            print("status :", verdict.status)
            print("route  :", verdict.route or "none available")
            print("needs  :", sorted(verdict.unmeasured_required), "which is not recorded")
            print()
            for a in verdict.verdict.assumptions:
                print(f"  [{a.state}] {a.name}")
                print(f"     {a.statement}")
        """,
            },
            {
                "slug": "tutorial-3-the-prior",
                "title": "The belief we start from",
                "asks": "What does the observational fit say, and how much should we trust it?",
                "lede": (
                    "An unidentified number is still a belief, and pretending to have none is "
                    "not neutrality. The observational fit is run — and then inflated, because "
                    "a confounded estimate that reports its nominal precision is the most "
                    "dangerous object in the analysis."
                ),
                "beat": (
                    "This is the number the experiment will be measured against. Write it down "
                    "now, with its inflation stated, so that later there is something for the "
                    "measurement to disagree with."
                ),
                "code": """

            import numpy as np
            from axiom.data import Panel
            from axiom.estimands import realize
            from axiom.surface import fit

            # The panel the operator actually has: depots that were already in good
            # shape got more hours, and "good shape" was never written down.
            rng = np.random.default_rng(123)
            frame = world.panel.frame
            dose = frame["a"].to_numpy(dtype=np.float64)
            standardized = (dose - dose.mean()) / dose.std()
            condition = 0.6 * standardized + 0.8 * rng.standard_normal(dose.size)
            observed_panel = Panel(frame.assign(y=frame["y"] + 0.8 * condition), world.panel.roles)
            print(f"corr(hours, fleet condition) = "
                  f"{float(np.corrcoef(dose, condition)[0, 1]):.2f}  <- the confounding")

            biased = fit(spec, observed_panel, backend="laplace", draws=1000, seed=1)
            before = realize(decision_estimand, biased, assume_identified=True, mass=0.9, seed=SEED)

            # The estimand is region-level: it sums across the 40 depots. The
            # break-even is per depot-week, so the comparison has to be made on
            # one scale or the other -- and getting that wrong is how a decision
            # gets made against a number forty times too big.
            before_per_depot = before.summary.mean / REGION_DEPOTS
            print()
            print("the observational reading of the decision's estimand")
            print(f"  region : {before.summary.mean:+.1f} index points per week")
            print(f"  depot  : {before_per_depot:+.2f} index points per depot-week")
            print(f"  break-even is {BREAK_EVEN:.2f}")
            print(f"  -> on this reading, fund it.")
        """,
            },
            {
                "slug": "tutorial-4-design",
                "title": "Designing the experiment",
                "asks": "How big does the trial need to be, and is the answer worth its cost?",
                "lede": (
                    "Four questions in order: what effect to power for, how many depots that "
                    "takes, what the answer is worth, and which concrete design buys it "
                    "cheapest. Powering for the effect you hope for is how trials get "
                    "commissioned that cannot fail informatively."
                ),
                "beat": (
                    "The effect worth detecting is the break-even, not the point estimate — "
                    "the experiment has to distinguish 'worth funding' from 'not', and that "
                    "boundary is where the precision has to land."
                ),
                "code": """

            from axiom.design import DecisionSpec, ValuePerOutcome, evoi_gaussian, sample_size

            from axiom.calibrate import carryover_window_factor
            from axiom.surface import GeometricCarryover as GC

            # The experiment reads a first week; the decision lives in the steady
            # state. `carryover_window_factor` says how much of the effect has
            # landed by then, which is what converts the break-even onto the
            # scale the trial can actually measure.
            lam = float(biased.posterior.flat("lam_a").mean())
            share = carryover_window_factor(GC(max_lag=MAX_LAG), {"lam_a": lam}, 1, treatment="a")
            threshold_first_week = BREAK_EVEN * share.counterfactual
            print(f"{share.counterfactual:.3f} of the effect lands in the first week")
            print(f"break-even on the experiment's scale: {threshold_first_week:.3f}")
            print()

            decision = DecisionSpec(
                name="raise_the_schedule",
                threshold=threshold_first_week,
                value_per_outcome_unit=VALUE_PER_POINT * 500 * 52,
                numeraire="USD",
            )

            sd = float(observed_panel.frame["y"].std())
            ss = sample_size(effect=threshold_first_week, sd=sd, power=0.8, alpha=0.05)
            print(f"powering for the boundary, not the hope: {threshold_first_week:.3f}")
            print(f"  depots per arm : {ss.n}")

            first_week = realize(
                experiment_estimand, biased, assume_identified=True, mass=0.9, seed=SEED
            )
            ev = evoi_gaussian(
                decision,
                float(first_week.summary.mean),
                float(first_week.summary.sd),
                0.35,
            )
            print()
            print(f"perfect information would be worth : {ev.evpi:,.0f} USD")
            print(f"this experiment, at se=0.35        : {ev.evsi:,.0f} USD")
        """,
            },
            {
                "slug": "tutorial-5-measurement",
                "title": "The experiment lands",
                "asks": "What did it measure, and does the model we already had agree?",
                "lede": (
                    "The trial runs. Its result arrives as one typed `Measurement` — the "
                    "estimand it measured, the number, the interval, and the conditions it was "
                    "measured under — rather than as a slide with a percentage on it."
                ),
                "beat": (
                    "Before folding it in, ask whether the old model predicted it. `agreement` "
                    "compares what the observational fit expected against what the experiment "
                    "saw, and this is where the tutorial turns: they disagree."
                ),
                "code": """

            from axiom.calibrate import Measurement, agreement

            # What the world would really have shown. The analyst does not get to
            # see this; the experiment reads it through noise.
            diff = world.forward({"a": DOSE}) - world.forward({"a": NO_DOSE})
            truth_experiment = float(np.mean(diff[:, 0]))
            se_experiment = 0.35
            measurement = Measurement(
                estimand=experiment_estimand,
                estimate=truth_experiment,
                se=se_experiment,
                mass=0.9,
                source="depot-maintenance-rct-2026Q2",
            )
            print(f"the trial read {measurement.estimate:+.3f} "
                  f"{measurement.interval}")

            says = agreement(biased, measurement, seed=SEED)
            print()
            print(f"the model we already had expected {says.posterior_mean:+.3f}")
            print(f"the trial says                    {says.estimate:+.3f}")
            print(f"agreement : {says.verdict}  (z = {says.z:.1f})")
        """,
            },
            {
                "slug": "tutorial-6-calibration",
                "title": "Folding the experiment into the model",
                "asks": "What does the model say once it has to honour the measurement?",
                "lede": (
                    "Refit the surface under the constraint that it reproduce what the trial "
                    "saw. The result is a model that agrees with the experiment on the "
                    "experiment's own terms — and can then be asked the decision's question, "
                    "which the experiment never measured directly."
                ),
                "beat": (
                    "Every step of that transfer is written to a ledger: the window it crossed, "
                    "the level it aggregated to, the carryover it assumed. The decision's number "
                    "is reported with the ledger attached or it is not reported."
                ),
                "code": """

            from axiom.calibrate import fit_calibrated

            calibrated = fit_calibrated(
                spec, observed_panel, [measurement], backend="laplace", draws=1000, seed=1
            )
            after = realize(
                decision_estimand, calibrated, assume_identified=True, mass=0.9, seed=SEED
            )

            after_per_depot = after.summary.mean / REGION_DEPOTS
            print("the decision's estimand, per depot-week")
            print(f"  observational : {before_per_depot:+.2f}")
            print(f"  calibrated    : {after_per_depot:+.2f}")
            print(f"  break-even    : {BREAK_EVEN:+.2f}")
            print()
            said = "fund it" if before_per_depot > BREAK_EVEN else "do not fund it"
            says_now = "fund it" if after_per_depot > BREAK_EVEN else "do not fund it"
            print(f"the observational model said : {said}")
            print(f"the experiment says          : {says_now}")
        """,
            },
            {
                "slug": "tutorial-7-the-follow-up",
                "title": "Planning the follow-up",
                "asks": "Should we repeat this, when does it go stale, and what next?",
                "lede": (
                    "A decision made once is a decision that decays. The same design machinery "
                    "that sized the first trial says when this answer stops being usable and "
                    "what the next dose worth testing is."
                ),
                "beat": (
                    "The answer is not 'run it again annually because that is the budget "
                    "cycle'. It is a number: how long before the value of a fresh answer "
                    "exceeds what a fresh answer costs."
                ),
                "code": """

            from axiom.design import time_to_re_experiment

            timing = time_to_re_experiment(
                posterior_sd=float(after.summary.sd) / REGION_DEPOTS,
                half_life_periods=26.0,
                experiment_se=se_experiment,
                min_eig=0.5,
            )
            print(f"a repeat is worth running again after {timing.periods:.0f} weeks")
            print(f"  today it would gain {timing.eig_now:.3f} nats, "
                  f"against a bar of 0.5")
        """,
            },
            {
                "slug": "tutorial-8-the-report",
                "title": "The report",
                "asks": "How does the whole of this become a document someone can check?",
                "lede": (
                    "Every number above was produced by a typed result carrying its own "
                    "provenance. `axiom-dossier` turns that record into a document: methods "
                    "from the steps, results from the quantities, limitations from the "
                    "assumptions still standing."
                ),
                "beat": (
                    "The point is not that a report gets written. It is that the report "
                    "cannot describe an analysis nobody ran, cannot state a number the record "
                    "does not hold, and cannot quietly omit the assumption doing the most work."
                ),
                "code": """

            from axiom_dossier import EvidenceBuilder, build

            evidence = (
                EvidenceBuilder(
                    "Weekly maintenance schedule",
                    "Should the standing weekly schedule go from nothing to 60 hours a depot?",
                )
                .verdict(verdict, graph=observed)
                .step(
                    "design", "Design",
                    what=f"{ss.n} depots an arm, powered for the break-even.",
                    why="The boundary the decision turns on is where precision has to land.",
                    equations=("break_even = DOSE * COST_PER_HOUR / VALUE_PER_POINT",),
                )
                .finding(
                    "decision", after, label="Steady-state weekly lift, region",
                    unit="index points", precision=1,
                    threshold=BREAK_EVEN * REGION_DEPOTS, beneficial="higher",
                )
                .build()
            )
            built = build(evidence, style="journal", verbosity="standard")
            print(built.summary())
            print(f"missing context keys: {built.missing() or 'none'}")
            print()
            for q in evidence.findings:
                print(f"  {q.label}: {q.stated()}  [{q.against_threshold()}]")
        """,
            },
        ],
    },
    {
        "key": "scattering",
        "title": "Designing an experiment before there is any data",
        "kind": "A design story",
        "asks": "There is no data. What should we go and measure?",
        "problem": (
            "Is the positive charge of an atom concentrated in a small hard centre, or "
            "spread through the whole atom? Both models fit everything known in 1910. "
            "What experiment would tell them apart — and specifically, which scattering "
            "angles, what shape of aperture, and for how long?"
        ),
        "lede": (
            "Nothing here is an analysis. Every step decides what to go and measure, "
            "and the whole argument turns on noticing that the two rival atoms are one "
            "model at two values of one number. That is what gives the experiment a "
            "resolving power, a sensitivity curve, and a rule for when to stop."
        ),
        "notebook": "nbs/case-studies/rutherford/",
        "steps": SCATTERING_STEPS,
    },
]


@section
def tutorial() -> Any:
    """Run every series, a step at a time, and keep what each step printed.

    Each series gets its own ``env``, and every step inside it runs into that
    one: step 6 of the depot tutorial recalibrates the fit step 3 produced, and
    a reader following along in a REPL has exactly the state the page shows.

    A step that raises stops the build. That is deliberate and it is the whole
    value of generating these pages: a tutorial that has drifted from the library
    fails here rather than on somebody's first afternoon with it.
    """
    import contextlib
    import io as _io
    import textwrap

    series = []
    for spec in TUTORIALS:
        env: dict[str, Any] = {}
        steps = []
        print(f"    [{spec['key']}]")
        for i, step in enumerate(spec["steps"], start=1):
            code = textwrap.dedent(step["code"]).strip("\n")
            buf = _io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(code, f"<tutorial:{step['slug']}>", "exec"), env)  # noqa: S102
            except Exception as exc:
                raise SystemExit(
                    f"tutorial step {i} ({step['slug']}) failed: "
                    f"{type(exc).__name__}: {exc}\n"
                    f"  output before the failure:\n{buf.getvalue()}"
                ) from exc
            output = buf.getvalue().rstrip("\n")
            steps.append(
                {
                    "n": i,
                    "slug": step["slug"],
                    "title": step["title"],
                    "asks": step["asks"],
                    "lede": step["lede"],
                    "beat": step["beat"],
                    "code": code,
                    "output": output,
                }
            )
            print(f"      {i}. {step['title']}: {len(output.splitlines())} lines")
        series.append({k: v for k, v in spec.items() if k != "steps"} | {"steps": steps})
    return {"series": series, "n": len(series)}


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
    """Execute each example and keep its narration, its numbers and its figures.

    Running them rather than transcribing them means the site cannot drift from the
    scripts, and a broken example shows up as a failed build instead of stale prose.

    Each example narrates itself through ``examples/_walkthrough.py``. With
    ``AXIOM_WALKTHROUGH_JSON`` set, the same run that prints the walkthrough also
    writes it as structured steps — narrative, decisions, readouts, tables and
    chart payloads. The code shown against each step is sliced out of the source
    file by line number, so a step's code is literally the code that ran between
    that step's heading and the next one.
    """
    import subprocess
    import tempfile

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
            "Does a training programme raise earnings, when enrolment is self-selected?",
            ["identify"],
        ),
        "03_education_selection_bias": (
            "Education",
            "How strong would an unmeasured confounder have to be to erase the tutoring effect?",
            ["identify", "diagnose"],
        ),
        "04_epidemiology_frontdoor": (
            "Epidemiology",
            "An exposure you cannot deconfound, recovered through a measured mediator.",
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
            "Contribution and marginal return, and proof the vocabulary is a thin adapter.",
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
    figures: dict[str, Any] = {}
    tmp = Path(tempfile.mkdtemp(prefix="axiom-walkthrough-"))

    # Narrow the set while iterating on one example; the site build always takes
    # all of them, so a half-generated page can never be what ships.
    pattern = os.environ.get("AXIOM_EXAMPLES_GLOB", "[0-9]*.py")

    for path in sorted(directory.glob(pattern)):
        stem = path.stem
        if stem not in META:
            print(f"    WARNING: {stem} has no metadata; skipping")
            continue
        field, question, pillars = META[stem]
        source = path.read_text()

        capture = tmp / f"{stem}.json"
        env = {**os.environ, "AXIOM_WALKTHROUGH_JSON": str(capture)}
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            env=env,
        )
        elapsed = time.perf_counter() - t0
        if proc.returncode != 0:
            raise SystemExit(f"example {stem} failed:\n{proc.stderr[-2000:]}")
        if not capture.exists():
            raise SystemExit(
                f"example {stem} produced no walkthrough — it must narrate itself "
                "through examples/_walkthrough.py"
            )
        walk = json.loads(capture.read_text())

        setup_code, step_code = slice_steps(source)
        if len(step_code) != len(walk["steps"]):
            raise SystemExit(
                f"example {stem}: {len(step_code)} w.step(...) calls in the source but "
                f"{len(walk['steps'])} steps recorded — steps must be top-level statements"
            )
        for step, code in zip(walk["steps"], step_code, strict=True):
            step["code"] = code
            for block in step["blocks"]:
                if block["type"] == "figure":
                    # Which JSON file the browser fetches for this chart: one per
                    # example, so a walkthrough page downloads its own figures and
                    # not those of the other eleven.
                    block["file"] = f"figures-{stem}"
        figures[stem] = walk.pop("figures")

        # the module docstring is the framing; the code below it is the demo
        body = source.split('"""', 2)[2].lstrip("\n") if source.count('"""') >= 2 else source
        entries.append(
            {
                "stem": stem,
                "slug": "example-" + stem.replace("_", "-"),
                "number": stem.split("_")[0],
                "field": field,
                "question": question,
                "pillars": pillars,
                "docstring": source.split('"""')[1].strip() if '"""' in source else "",
                "code": body.rstrip(),
                "setup_code": setup_code,
                "output": proc.stdout.rstrip("\n"),
                "lines": len(source.splitlines()),
                "seconds": round(elapsed, 1),
                "n_steps": len(walk["steps"]),
                "n_figures": sum(
                    1 for s in walk["steps"] for b in s["blocks"] if b["type"] == "figure"
                ),
                "walkthrough": walk,
            }
        )
        print(
            f"    {stem}: {elapsed:.1f}s, {len(walk['steps'])} steps, "
            f"{entries[-1]['n_figures']} figures"
        )

    for stem, payload in figures.items():
        write(f"figures-{stem}", payload)

    return {
        "entries": entries,
        "n": len(entries),
        "total_seconds": round(sum(e["seconds"] for e in entries), 1),
        "total_lines": sum(e["lines"] for e in entries),
        "total_steps": sum(e["n_steps"] for e in entries),
        "total_figures": sum(e["n_figures"] for e in entries),
    }


def slice_steps(source: str) -> tuple[str, list[str]]:
    """Split an example's source at its ``w.step(...)`` calls.

    Returns the setup block — everything after the module docstring and before the
    first step — and then one code block per step, running from the end of that
    step's call to the start of the next one. Slicing by line number rather than by
    a marker comment means the page shows the code that actually ran, and cannot be
    fooled by a stale annotation.

    The one thing dropped is each figure's caption text — ``title``, ``note`` and
    ``legend`` — which the page renders with the figure itself a few lines below.
    Showing it twice buries the axiom calls the step is actually about. The
    elision is marked in the code, and the unedited file is on every page under
    "the whole file".
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    captions = caption_lines(tree)

    calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "step"
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "w"
    ]
    if not calls:
        return trim(lines, captions, 0), []

    doc_end = 0
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Constant)
    ):
        doc_end = tree.body[0].end_lineno or 0

    setup = trim(lines[doc_end : calls[0].lineno - 1], captions, doc_end)
    blocks = []
    for i, call in enumerate(calls):
        start = call.end_lineno or call.lineno
        stop = calls[i + 1].lineno - 1 if i + 1 < len(calls) else len(lines)
        blocks.append(trim(lines[start:stop], captions, start))
    return setup, blocks


CAPTION_KWARGS = ("title", "note", "legend")


def caption_lines(tree: ast.Module) -> dict[int, str | None]:
    """Line numbers of every figure caption argument, as ``{lineno: replacement}``.

    The first line of each elided run carries the marker comment; the rest map to
    ``None`` and are dropped.
    """
    out: dict[int, str | None] = {}
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "figure"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "w"
        ):
            continue
        first = True
        for kw in node.keywords:
            if kw.arg not in CAPTION_KWARGS:
                continue
            end = kw.value.end_lineno or kw.value.lineno
            for ln in range(kw.value.lineno, end + 1):
                if first:
                    out[ln] = "    # title, note and legend: the caption shown with the chart"
                    first = False
                else:
                    out[ln] = None
    return out


def trim(lines: list[str], captions: dict[int, str | None], offset: int) -> str:
    """Drop the rule comments between steps, the blanks around them, and captions.

    ``offset`` is the 0-based index of ``lines[0]`` in the source file, so the
    caption line numbers — which are 1-based and file-wide — line up.
    """
    kept = []
    for i, ln in enumerate(lines):
        if ln.startswith("# ---"):
            continue
        marker = captions.get(offset + i + 1, "keep")
        if marker == "keep":
            kept.append(ln)
        elif marker is not None:
            kept.append(marker)
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


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

    entries.append(
        {
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
        }
    )

    # --- NIST Misra1a, against certified values ------------------------------
    nist = DATASETS["misra1a"]
    cert = nist.published
    BASES.declare("volume", symbol="V")
    BASES.declare("pressure", symbol="P")
    volume = Outcome(name="volume", dimension=D.volume, unit="cc")
    pressure = Treatment(name="pressure", dimension=D.pressure, unit="mmHg")
    mf = load("misra1a")
    tidy = _pd.DataFrame(
        {
            "unit": ["specimen"] * len(mf),
            "t": _np.arange(len(mf)),
            "pressure": mf["pressure"].to_numpy(float),
            "volume": mf["volume"].to_numpy(float),
        }
    )
    panel = Panel(
        tidy,
        RoleMap(
            unit="unit", time="t", outcome=("volume", volume), treatments={"pressure": pressure}
        ),
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
        panel,
        backend="laplace",
        draws=4000,
        seed=SEED,
    )
    post = nist_fit.posterior
    beta = post.summary("beta_pressure", definition="hdi", mass=0.95)
    kk = post.summary("k_pressure", definition="hdi", mass=0.95)
    x = tidy["pressure"].to_numpy(float)
    yy = tidy["volume"].to_numpy(float)
    rss = float(_np.sum((yy - beta.mean * (1.0 - _np.exp(-x / kk.mean))) ** 2))

    entries.append(
        {
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
        }
    )

    # --- Darfur and LaLonde: only if fetched ---------------------------------
    covs = ["age", "farmer_dar", "herder_dar", "pastvoted", "hhsize_darfur", "female"]
    if DATASETS["darfur"].available:
        d = DATASETS["darfur"]
        pub = d.published
        df_frame = load("darfur")
        dummies = _pd.get_dummies(df_frame["village"], prefix="v", drop_first=True, dtype=float)
        design = _pd.concat([df_frame.drop(columns=["village"]), dummies], axis=1)
        est = ols(design, "peacefactor", "directlyharmed", covariates=[*covs, *dummies.columns])
        dof = int(est.detail["df_resid"])
        rv = robustness_value(estimate=est.estimate, se=est.se, df=dof, q=1.0, alpha=0.05)
        naive = ols(df_frame, "peacefactor", "directlyharmed")
        no_village = ols(df_frame, "peacefactor", "directlyharmed", covariates=covs)
        entries.append(
            {
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
                    {
                        "label": "covariates only",
                        "estimate": no_village.estimate,
                        "se": no_village.se,
                    },
                    {"label": "published specification", "estimate": est.estimate, "se": est.se},
                ],
            }
        )

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
        entries.append(
            {
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
                    {
                        "label": "PSID controls, unadjusted",
                        "estimate": obs_naive.estimate,
                        "se": obs_naive.se,
                    },
                    {
                        "label": "PSID controls, 8 covariates",
                        "estimate": obs_adj.estimate,
                        "se": obs_adj.se,
                    },
                ],
                "truth": exp.estimate,
            }
        )

    # capture the full printed output of each case study, as with the examples
    for entry in entries:
        script = ROOT / "benchmarks" / f"case_{entry['name']}.py"
        if entry["name"] == "lalonde_nsw":
            script = ROOT / "benchmarks" / "case_lalonde.py"
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=str(ROOT / "benchmarks"),
        )
        if proc.returncode != 0:
            raise SystemExit(f"benchmark {entry['name']} failed:\n{proc.stderr[-1500:]}")
        entry["output"] = proc.stdout.rstrip("\n")
        entry["worst"] = max(r["delta"] for r in entry["rows"])
        entry["worst_rel"] = max(r["rel"] for r in entry["rows"])
        print(
            f"    {entry['name']}: {len(entry['rows'])} checks, "
            f"worst relative error {entry['worst_rel']:.1e}"
        )

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
