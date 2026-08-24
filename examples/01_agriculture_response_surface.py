"""Agronomy — nitrogen and irrigation, and where the yield actually peaks.

The question
    A field trial varies nitrogen (kg/ha) and irrigation (mm) across plots. Yield
    responds to both, and saturates in both. Where is the optimum, and if water is
    rationed, how should the remaining budget be split?

Why a response surface and not a regression
    Fitting a straight line to a saturating response puts the predicted optimum at
    whichever corner of the design happens to be highest. The whole point of the
    surface layer is to keep the curvature, so the optimizer is asked about the
    shape the data actually support.

Pillars: surface (kernels, allocation, frontier)
"""

import numpy as np
from _walkthrough import Walkthrough

from axiom.core import Intervention
from axiom.sim import arms_world
from axiom.surface import (
    Bounds,
    HillKernel,
    allocate,
    central_composite,
    counterfactual_doses,
    fit,
    frontier,
    response_band,
)

SEED = 0

w = Walkthrough(
    field="Agronomy",
    title="Where the yield actually peaks",
    question="""A field trial varies nitrogen and irrigation across plots. Yield
        saturates in both. Where is the optimum, and when water is rationed, how
        should the remaining input budget be split?""",
)

# ----------------------------------------------------------------------------------

w.step(
    "State the box the doses can live in",
    why="""Bounds are not a plotting range. They are a claim about the world — no
        plot on this farm will receive 400 kg/ha of nitrogen, so no optimizer should
        ever propose it and no fitted curve should be trusted out there. Saying it
        once, in an object, means the design, the allocation and the frontier all
        obey the same box instead of each carrying its own copy of the farm's
        limits.""",
    instead="""Fitting on whatever doses happened to occur and reading the optimum
        off the curve. That silently extrapolates: a Hill curve is still increasing
        at every dose, so the unconstrained answer to 'how much nitrogen' is always
        'more'.""",
)
bounds = Bounds(treatments=("nitrogen", "irrigation"), low=(0.0, 0.0), high=(200.0, 120.0))
w.out(f"treatments : {bounds.treatments}")
w.out(f"nitrogen   : {bounds.low[0]:.0f} to {bounds.high[0]:.0f} kg/ha")
w.out(f"irrigation : {bounds.low[1]:.0f} to {bounds.high[1]:.0f} mm")

# ----------------------------------------------------------------------------------

w.step(
    "Spend the plots where the curvature is",
    why="""Curvature is what this whole analysis turns on, and curvature is estimated
        from runs placed to see it bending. A rotatable central composite design is
        a factorial cube, plus axial points pushed out along each treatment, plus
        replicated centre points. The axial points are what make the quadratic terms
        estimable; the replicated centre is what separates pure noise from lack of
        fit. 'Rotatable' means the prediction variance depends only on distance from
        the centre, so no direction in the dose box is privileged by accident.""",
    instead="""A 5x5 grid over the same box. It costs 25 plots instead of twelve,
        spends most of them re-measuring the flat regions, and still gives no
        replicated centre — so it cannot tell a bad model from a noisy one. One
        factor at a time is worse again: it cannot see an interaction at all.""",
)
design = central_composite(bounds, alpha="rotatable", center_points=4, inscribed=True)
w.out(f"design : {design.kind}, {design.n} plots")
w.out(f"detail : {dict(design.detail)}")

frame = design.as_frame().round(1)
w.table(
    list(frame.columns),
    frame.to_numpy().tolist(),
    caption="Every plot the trial will plant, in the units an agronomist works in.",
)

points = [[float(a), float(b)] for a, b in design.points]
counts: dict[tuple[float, float], int] = {}
for a, b in points:
    counts[(round(a, 3), round(b, 3))] = counts.get((round(a, 3), round(b, 3)), 0) + 1
w.figure(
    "design",
    kind="scatter",
    data={
        "x": [p[0] for p in counts],
        "y": [p[1] for p in counts],
        "labels": [f"x{n}" if n > 1 else "" for n in counts.values()],
    },
    opt={
        "key": "@",
        "x": "x",
        "y": "y",
        "labels": "labels",
        "xLabel": "nitrogen (kg/ha)",
        "yLabel": "irrigation (mm)",
        "height": 300,
    },
    title="The twelve plots, in dose space",
    note="""Four corners, four axial points pushed out along each treatment, and a
        centre planted four times over — that replication is the only thing in the
        design that can separate noise from a wrong model.""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Grow a field whose answer is already known",
    why="""Every number below is checkable only because the truth was written down
        first. A recovery test is the difference between a method that works and a
        method that has never been graded: simulate from a known set of parameters,
        fit, and ask whether the intervals contain what generated the data.""",
    instead="""A real dataset. It would be more convincing about agronomy and
        completely uninformative about whether the estimator is right — with real
        data there is nothing to compare the answer to.""",
)
truth = {
    "alpha": 3.0,  # baseline yield, t/ha
    "beta_nitrogen": 4.2,  # how much yield nitrogen can add at saturation
    "k_nitrogen": 90.0,  # the half-saturation dose
    "s_nitrogen": 2.0,  # how sharply it turns over
    "beta_irrigation": 2.6,
    "k_irrigation": 45.0,
    "s_irrigation": 1.6,
}
field = arms_world(
    n_units=design.n,
    treatments=("nitrogen", "irrigation"),
    kernels={
        "nitrogen": HillKernel(reference_dose=90.0, amplitude_scale=5.0),
        "irrigation": HillKernel(reference_dose=45.0, amplitude_scale=5.0),
    },
    doses=design.doses(),
    truth=truth,
    noise_sd=0.35,
    seed=SEED,
)
w.out(f"plots      : {field.n_units}")
w.out(
    f"true peak  : nitrogen k = {truth['k_nitrogen']:.0f}, "
    f"irrigation k = {truth['k_irrigation']:.0f}"
)
w.out("noise sd   : 0.35 t/ha, against a yield range of about 3 to 9")

# ----------------------------------------------------------------------------------

w.step(
    "Fit once, and check the truth is inside the interval",
    why="""The likelihood calls exactly the same forward() the allocation and the
        frontier will call later. That is a design rule rather than an efficiency
        note: the moment the optimizer gets its own copy of the transform chain, the
        two drift, and the drift shows up as a recommendation the model cannot
        actually justify.""",
    instead="""Reporting the point estimates and moving on. The comparison worth
        making is not 'is the mean close' but 'does the interval contain the truth' —
        an estimator can be nearly right and still badly overconfident, and only the
        second question catches that.""",
)
result = fit(field.spec, field.panel, backend="laplace", draws=2000, seed=SEED)
posterior = result.posterior
theta = {name: float(posterior.summary(name).mean) for name in posterior.names()}
w.out(f"converged : {result.converged}")
# sorted, because `names()` is a frozenset: joined unsorted it prints in a
# different order on every run, which makes this walkthrough -- and the page and
# the report generated from it -- differ from itself between builds.
w.out(f"parameters: {', '.join(sorted(posterior.names()))}")

shown = ("beta_nitrogen", "k_nitrogen", "beta_irrigation", "k_irrigation")
rows, chart_rows = [], []
for name in shown:
    s = posterior.summary(name, definition="hdi", mass=0.9)
    inside = s.interval.lower <= truth[name] <= s.interval.upper
    rows.append(
        [
            name,
            f"{truth[name]:.1f}",
            f"{s.mean:.1f}",
            f"[{s.interval.lower:.1f}, {s.interval.upper:.1f}]",
            "yes" if inside else "NO",
        ]
    )
    # Scaled by the truth so parameters of wildly different size share one axis —
    # a 90 kg/ha half-saturation dose and a 4.2 t/ha amplitude are not comparable
    # in their own units, and drawing them as if they were is how a forest plot lies.
    chart_rows.append(
        {
            "label": name,
            "estimate": (s.mean - truth[name]) / truth[name],
            "lower": (s.interval.lower - truth[name]) / truth[name],
            "upper": (s.interval.upper - truth[name]) / truth[name],
            "note": f"truth {truth[name]:.1f}, posterior mean {s.mean:.1f}",
            "bad": not inside,
        }
    )
w.table(["parameter", "truth", "posterior mean", "90% HDI", "contains truth"], rows)

w.figure(
    "recovery",
    kind="intervals",
    data={"rows": chart_rows},
    opt={
        "rows": "@rows",
        "truth": 0,
        "truthLabel": "the truth",
        "xLabel": "error, as a fraction of the true value",
        "mass": 90,
        "labelWidth": 150,
    },
    title="Parameter recovery, every parameter on one axis",
    note="""Each interval is scaled by its own true value, which is the only way to
        put a half-saturation dose in kg/ha and an amplitude in t/ha on the same
        picture. An interval that misses the dashed line is a recovery failure and
        would be drawn in red.""",
    legend=(("accent", "90% HDI, contains the truth"), ("ink-2:dash", "the truth")),
)

# ----------------------------------------------------------------------------------

w.step(
    "Ask where the response flattens out",
    why="""The band is not a fitted line with error bars bolted on. It is every
        posterior draw pushed through forward() and summarised at each dose, so the
        width at a given dose is what the model genuinely does not know there —
        widest where the design placed fewest plots, which is exactly where a
        recommendation should be least confident.""",
    instead="""Reading the optimum off the posterior mean curve alone. The mean
        crosses its own plateau somewhere; the band shows a whole region of doses
        the data cannot tell apart, and that region, not the point, is the honest
        answer to 'how much nitrogen'.""",
)


def truth_at(name: str, dose: float) -> float:
    """The true curve on the same counterfactual the band uses.

    ``response_band`` sets the named treatment to each dose and leaves every other
    treatment at its observed level. Holding the others at zero instead would put
    the two curves on different counterfactuals, and the overlay would be a
    comparison of two different questions.
    """
    data = counterfactual_doses(field.surface, field.data, Intervention(doses={name: dose}))
    return float(field.surface.forward(data, field.theta).mean())


doses = np.linspace(0.0, 200.0, 30)
band = response_band(result, "nitrogen", doses=doses.tolist(), mass=0.9, seed=SEED)
true_curve = [truth_at("nitrogen", float(d)) for d in doses]

w.table(
    ["nitrogen (kg/ha)", "posterior mean (t/ha)", "90% interval", "truth"],
    [
        [
            f"{doses[i]:.0f}",
            f"{band.mean[i]:.2f}",
            f"[{band.lower[i]:.2f}, {band.upper[i]:.2f}]",
            f"{true_curve[i]:.2f}",
        ]
        for i in range(0, len(doses), 4)
    ],
    caption="Every fourth grid point; the chart below has all thirty.",
)
w.figure(
    "band",
    kind="band",
    data={
        "treatment": "nitrogen",
        "doses": doses.tolist(),
        "mean": list(band.mean),
        "lower": list(band.lower),
        "upper": list(band.upper),
        "truth": true_curve,
        "mass": 0.9,
    },
    opt={"key": "@", "xLabel": "nitrogen (kg/ha)", "yLabel": "yield (t/ha)", "height": 320},
    title="Yield against nitrogen, with everything the model does not know",
    note="""The dashed line is the curve the simulation actually used. It stays inside
        the band across the whole dose range, which is what a passed recovery test
        looks like when you draw it instead of tabulating it.""",
    legend=(
        ("accent", "posterior mean"),
        ("accent-wash", "90% interval"),
        ("ink-2:dash", "the truth"),
    ),
)

# ----------------------------------------------------------------------------------

w.step(
    "Now ration the water and split the budget",
    why="""The decision is not 'where is the peak' — it is 'given a fixed total of
        input and a water cap, what mix'. allocate() searches the fitted surface
        under both constraints, using the posterior rather than a single fitted
        curve, so the recommendation reflects the uncertainty in the shape and not
        just its mean.""",
    instead="""Splitting the budget in proportion to each input's estimated
        amplitude. That is the right answer only for a linear response; with
        saturation it over-buys whichever input is already past its bend, which is
        precisely the mistake the surface layer exists to prevent.""",
)
water_limited = Bounds(treatments=("nitrogen", "irrigation"), low=(0.0, 0.0), high=(200.0, 60.0))
plan = allocate(
    result.surface, posterior, budget=180.0, bounds=water_limited, objective="mean", seed=SEED
)
unconstrained = allocate(
    result.surface, posterior, budget=180.0, bounds=bounds, objective="mean", seed=SEED
)
w.out("budget 180 units of input, water capped at 60 mm:")
w.out("  " + ", ".join(f"{k} {v:.1f}" for k, v in plan.doses.items()))
w.out(f"  expected yield {plan.expected_outcome:.3f} t/ha  ({plan.status})")
w.out("the same budget with no water cap:")
w.out("  " + ", ".join(f"{k} {v:.1f}" for k, v in unconstrained.doses.items()))
w.out(f"  expected yield {unconstrained.expected_outcome:.3f} t/ha")
w.out(f"  the cap costs {unconstrained.expected_outcome - plan.expected_outcome:.3f} t/ha")

grid_n = np.linspace(0.0, 200.0, 32)
grid_i = np.linspace(0.0, 120.0, 32)
w.figure(
    "surface",
    kind="heatmap",
    data={
        "nitrogen": grid_n.tolist(),
        "irrigation": grid_i.tolist(),
        "z": [
            [
                float(
                    result.surface.forward(
                        {"nitrogen": float(n), "irrigation": float(i)}, theta
                    ).mean()
                )
                for n in grid_n
            ]
            for i in grid_i
        ],
    },
    opt={
        "key": "@",
        "x": "nitrogen",
        "y": "irrigation",
        "z": "z",
        "xLabel": "nitrogen (kg/ha)",
        "yLabel": "irrigation (mm)",
        "zLabel": "yield (t/ha)",
        "height": 340,
        "marks": [
            {
                "x": float(plan.doses["nitrogen"]),
                "y": float(plan.doses["irrigation"]),
                "label": "under the cap",
            },
            {
                "x": float(unconstrained.doses["nitrogen"]),
                "y": float(unconstrained.doses["irrigation"]),
                "label": "uncapped",
            },
        ],
    },
    title="The whole fitted surface, and the two allocations",
    note="""Both marks sit on the same budget line — 180 units of input — and differ
        only in whether irrigation may exceed 60 mm. The surface is flat enough
        along its ridge that the cap costs surprisingly little.""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Price the next unit of input",
    why="""A planner cannot act on 'the optimum'. They can act on 'the next hundred
        units of input are worth this much yield, the hundred after that are worth
        this much less'. The frontier re-solves the allocation at each budget, and
        the shadow price is the slope between neighbouring solutions — the number
        that says when to stop buying.""",
    instead="""Quoting one average return per unit of input. On a saturating surface
        the average is a blend of a steep early region and a flat late one, so it
        overstates the value of the next unit whenever you are already past the
        bend — and understates it when you are below.""",
)
budgets = [60.0, 120.0, 180.0, 240.0, 300.0]
front = frontier(result.surface, theta, budgets=budgets, bounds=water_limited, seed=SEED)
front_frame = front.as_frame().round(3)
prices = [float(p) for p in front.shadow_prices()]
w.table(
    list(front_frame.columns),
    front_frame.to_numpy().tolist(),
    caption="The best achievable yield at each budget, and how it is spent.",
)
w.figure(
    "frontier",
    kind="lines",
    data={
        "budgets": budgets,
        "outcome": [float(v) for v in front_frame["expected_outcome"]],
    },
    opt={
        "series": [{"label": "best achievable yield", "x": "@budgets", "y": "@outcome"}],
        "xLabel": "input budget",
        "yLabel": "yield (t/ha)",
        "height": 260,
        "rightPad": 120,
    },
    title="What each budget buys",
    note="The curve bends: equal increments of budget stop buying equal increments of yield.",
)
w.figure(
    "shadow",
    kind="bars",
    data={
        "rows": [
            {
                "label": f"budget {b:.0f}",
                "value": price,
                "display": f"{price:+.4f}",
                "note": f"the next input unit at a budget of {b:.0f} returns " f"{price:.4f} t/ha",
            }
            for b, price in zip(budgets, prices, strict=True)
        ]
    },
    opt={"rows": "@rows", "xLabel": "t/ha per extra unit of input", "labelWidth": 120},
    title="The shadow price of the next unit, by budget",
    note="""This is the frontier's slope, and it is the number a budget conversation
        is actually about. It falls monotonically, and the budget worth funding is
        the largest one whose shadow price still beats the price of the input.""",
)

# ----------------------------------------------------------------------------------

w.finding("""The design was twelve plots and it recovered a two-treatment saturating
    surface well enough that the true curve sits inside the 90% band at every
    nitrogen dose. That is the case for spending plots on curvature rather than
    spreading them evenly: a 5x5 grid would have cost twice as much and learned the
    shape less well.""")
w.finding(f"""The water cap turned out to be cheap. Capping irrigation at 60 mm costs
    {unconstrained.expected_outcome - plan.expected_outcome:.3f} t/ha at a budget of
    180 — the surface has a broad ridge, so the constraint slides the answer along
    it rather than down it. Had the ridge run the other way the same cap would have
    been expensive, and nothing but the fitted shape tells you which case you are
    in.""")
w.finding(f"""The shadow price falls from {prices[0]:.4f} to {prices[-1]:.4f} t/ha per unit
    across the budgets considered — a factor of {prices[0] / prices[-1]:.1f}. A single
    average return per unit of input would have hidden that entirely, and it is the
    only number in this whole run that answers the question a planner actually
    asked: should we buy more?""")
