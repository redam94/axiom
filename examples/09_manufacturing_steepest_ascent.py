"""Process engineering — climbing to a yield optimum, the Box-Wilson way.

The question
    A reactor's yield depends on temperature and residence time. Current settings
    are somewhere on the slope, not at the peak. Where should the next batch of
    experimental runs be placed?

The classical answer, which axiom implements directly
    Response-surface methodology, as Box and Wilson set it out in 1951. Fit a cheap
    local design, follow the gradient uphill in steps until it stops paying, then
    run a second-order design at the new location and ask what kind of stationary
    point you have arrived at. A maximum is good news. A saddle or a rising ridge
    means the process has a direction along which you can keep improving for free,
    which is worth far more than the peak itself.

Pillars: surface (designs, steepest ascent, canonical analysis)
"""

import numpy as np
from _walkthrough import Walkthrough

from axiom.sim import arms_world
from axiom.surface import (
    Bounds,
    HillKernel,
    canonical_analysis,
    central_composite,
    fit,
    full_factorial,
    steepest_ascent,
)

SEED = 0
CURRENT = {"temperature": 140.0, "residence": 12.0}

w = Walkthrough(
    field="Process engineering",
    title="The optimum is outside the operating window",
    question="""A reactor is running somewhere on the slope. Where should the next
        batch of expensive experimental runs be placed, and once the climb stops,
        what kind of place have you arrived at?""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Write down the operating window before anything else",
    why="""120 to 200 degrees and 5 to 45 minutes are not modelling choices, they are
        what the plant can physically run. Encoding them as Bounds means the design,
        the ascent and the canonical analysis all obey the same constraint, and it
        means an optimizer's answer is always something the plant could actually
        do.""",
    instead="""Optimising unconstrained and checking feasibility at the end. On a
        saturating surface the unconstrained answer is frequently outside the window
        — as it turns out to be here — and discovering that after the fact loses the
        much more useful question of what the boundary is costing.""",
)
window = Bounds(treatments=("temperature", "residence"), low=(120.0, 5.0), high=(200.0, 45.0))
w.out(f"temperature : {window.low[0]:.0f} to {window.high[0]:.0f} C")
w.out(f"residence   : {window.low[1]:.0f} to {window.high[1]:.0f} min")
w.out(f"running now : {CURRENT['temperature']:.0f} C, {CURRENT['residence']:.0f} min")

# ----------------------------------------------------------------------------------

w.step(
    "Stage one: buy a direction, not a surface",
    why="""The first design only has to answer 'which way is uphill'. A 3x3 factorial
        does that for nine runs. Curvature is not needed yet and estimating it here
        would be paying for information the next decision does not use.""",
    instead="""Going straight to a central composite design at the current setpoint.
        It costs more runs and it characterises the neighbourhood of a point you are
        about to leave. Box and Wilson's whole insight is that the expensive design
        belongs at the end of the climb, not at the start.""",
)
screen = full_factorial(window, 3)
w.out(f"screening design : {screen.kind}, {screen.n} runs")
truth = {
    "alpha": 40.0,
    "beta_temperature": 22.0,
    "k_temperature": 165.0,
    "s_temperature": 3.0,
    "beta_residence": 14.0,
    "k_residence": 26.0,
    "s_residence": 2.4,
}
reactor = arms_world(
    n_units=screen.n,
    treatments=("temperature", "residence"),
    kernels={
        "temperature": HillKernel(reference_dose=165.0, amplitude_scale=20.0),
        "residence": HillKernel(reference_dose=26.0, amplitude_scale=20.0),
    },
    doses=screen.doses(),
    truth=truth,
    noise_sd=1.1,
    seed=SEED,
)
screen_fit = fit(reactor.spec, reactor.panel, backend="laplace", draws=1500, seed=SEED)
theta = {n: float(screen_fit.posterior.summary(n).mean) for n in screen_fit.posterior.names()}
w.out(f"converged        : {screen_fit.converged}")

grid_t = np.linspace(120.0, 200.0, 30)
grid_r = np.linspace(5.0, 45.0, 30)
surface_z = [
    [
        float(
            screen_fit.surface.forward(
                {"temperature": float(t), "residence": float(r)}, theta
            ).mean()
        )
        for t in grid_t
    ]
    for r in grid_r
]

# ----------------------------------------------------------------------------------

w.step(
    "Stage two: walk uphill in cheap steps",
    why="""Steepest ascent is a sequence of small local decisions rather than a
        global search. When a run costs a shift of plant time and the surface is
        smooth — which describes most physical processes — it remains the right first
        move, and it stops on its own when the gradient stops paying or the window
        gets in the way.""",
    instead="""Handing the fitted surface to a global optimizer. It would return the
        unconstrained peak, which is outside the window, in one call — and it would
        skip the information the path itself carries about how steep the climb was
        and where it flattened.""",
)
path = steepest_ascent(screen_fit.surface, theta, CURRENT, step=6.0, n_steps=40, bounds=window)
best = path.best()
w.out(f"steps taken : {path.n}")
w.out(f"stopped because : {path.stop}")
w.out("best point on the path: " + ", ".join(f"{k} {v:.1f}" for k, v in best.items()))
w.out(f"predicted yield there : {path.values[-1]:.3f}")

points = [[float(v) for v in p] for p in path.points]
w.table(
    ["step", "temperature", "residence", "predicted yield"],
    [
        [str(i), f"{points[i][0]:.1f}", f"{points[i][1]:.1f}", f"{path.values[i]:.3f}"]
        for i in sorted({0, 1, 2, path.n // 3, 2 * path.n // 3, path.n - 1})
        if 0 <= i < path.n
    ],
)
w.figure(
    "climb",
    kind="lines",
    data={"step": list(range(path.n)), "yield": [float(v) for v in path.values]},
    opt={
        "series": [{"label": "predicted yield", "x": "@step", "y": "@yield", "colour": "accent"}],
        "xLabel": "step along the ascent",
        "yLabel": "predicted yield",
        "height": 260,
        "rightPad": 110,
    },
    title="What each step of the climb bought",
    note="""The curve flattens long before the path stops. That flattening is the
        signal to stop spending runs on the direction and start asking what kind of
        place this is — which is what stage three does.""",
)
w.figure(
    "surface",
    kind="heatmap",
    data={"temperature": grid_t.tolist(), "residence": grid_r.tolist(), "z": surface_z},
    opt={
        "key": "@",
        "x": "temperature",
        "y": "residence",
        "z": "z",
        "xLabel": "temperature (C)",
        "yLabel": "residence time (min)",
        "zLabel": "predicted yield",
        "height": 340,
        "marks": [
            {
                "x": CURRENT["temperature"],
                "y": CURRENT["residence"],
                "label": "running today",
            },
            {
                "x": float(best["temperature"]),
                "y": float(best["residence"]),
                "label": "ascent stopped here",
            },
        ],
    },
    title="The fitted surface, and where the climb started and ended",
    note="""The ascent ran into the corner of the window. That is not the algorithm
        failing — it is the plant's limits being the binding constraint, which is a
        different and more actionable finding than 'the optimum is here'.""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Stage three: ask what kind of place you have arrived at",
    why="""A flat gradient is not one situation, it is three, and they imply
        completely different next moves. A maximum means confirm and stop. A rising
        ridge means keep walking along it, because there is free yield in that
        direction. A saddle means back off — you are between two better regions.
        The canonical analysis reads which one off the fitted quadratic's
        eigenvalues.""",
    instead="""Declaring victory at the end of the ascent. The path stopping tells
        you the gradient is small or the window is in the way; it says nothing about
        the shape of the neighbourhood, and the shape is what decides whether to
        spend the next batch of runs at all.""",
)
confirm = central_composite(window, alpha="rotatable", center_points=4, inscribed=True)
w.out(f"confirmation design : {confirm.kind}, {confirm.n} runs")
stationary = canonical_analysis(screen_fit.surface, theta, best)
if hasattr(stationary, "reason"):
    w.out(f"canonical analysis declined: {stationary.reason}")
    inside = False
else:
    w.out(f"stationary point is a {stationary.kind}")
    w.out(
        "at "
        + ", ".join(
            f"{name} {float(v):.1f}"
            for name, v in zip(window.treatments, stationary.point, strict=True)
        )
    )
    w.out(f"eigenvalues : {[round(float(e), 5) for e in stationary.eigenvalues]}")
    inside = all(
        lo <= float(v) <= hi
        for v, lo, hi in zip(stationary.point, window.low, window.high, strict=True)
    )
    w.out(f"inside the operating window? {inside}")
    w.say("""Both eigenvalues negative means a genuine maximum. Mixed signs would mean a
        saddle, and the positive direction would be yield available for free along a
        ridge — worth more than the peak itself, and invisible to anything that only
        reports the location of the stationary point.""")
    if not inside:
        w.say("""And it is outside the window. The unconstrained peak sits where the
            plant cannot run, which is why the ascent stopped on the boundary rather
            than at the stationary point. The operating recommendation is the corner
            of the window; the interesting question becomes what it would cost to
            widen the window, and that is now a costed engineering question rather
            than an open-ended one.""")

# ----------------------------------------------------------------------------------

w.step(
    "Finally, price the whole exercise against the old setpoint",
    why="""Everything above is on the fitted surface. This step goes back to the
        simulated reactor and evaluates the TRUE yield at the old setpoint and at the
        point the climb recommended, which is the only check that the recommendation
        was worth acting on rather than merely internally consistent.""",
    instead="""Reporting the predicted improvement. The model's prediction of its own
        recommendation is not evidence: it is the same surface evaluated twice, and
        it would look just as good if the fit were wrong.""",
)
true_current = float(reactor.forward(CURRENT).mean())
true_best = float(reactor.forward({k: float(v) for k, v in best.items()}).mean())
w.out(f"true yield at the old setpoint  : {true_current:.3f}")
w.out(f"true yield at the ascent's best : {true_best:.3f}")
w.out(f"improvement                     : {true_best - true_current:+.3f}")


def within_window(name: str, value: float) -> float:
    """Where a setting sits in its own operating range, as a fraction.

    Degrees and minutes do not belong on the same axis, and drawing them there
    would be the chart lying about a comparison. Position within the window is a
    quantity they genuinely share, and it is the one an operator cares about:
    how much room is left.
    """
    i = window.treatments.index(name)
    return (value - window.low[i]) / (window.high[i] - window.low[i])


w.figure(
    "gain",
    kind="dumbbell",
    data={
        "rows": [
            {
                "label": f"temperature ({CURRENT['temperature']:.0f} to "
                f"{float(best['temperature']):.0f} C)",
                "a": within_window("temperature", CURRENT["temperature"]),
                "b": within_window("temperature", float(best["temperature"])),
            },
            {
                "label": f"residence ({CURRENT['residence']:.0f} to "
                f"{float(best['residence']):.0f} min)",
                "a": within_window("residence", CURRENT["residence"]),
                "b": within_window("residence", float(best["residence"])),
            },
        ]
    },
    opt={
        "rows": "@rows",
        "xLabel": "position within the operating window (0 = lowest, 1 = the limit)",
        "aLabel": "running today",
        "bLabel": "recommended",
        "labelWidth": 190,
    },
    title="What the plant would actually change",
    note=f"""Both settings end at 1.0 — hard against the limit, with no room left in
        either direction. That is the same finding as the canonical analysis, seen
        from the operator's side, and it is worth {true_best - true_current:+.2f} in
        true yield ({true_current:.1f} to {true_best:.1f}), verified against the
        simulation rather than against the fit.""",
    legend=(("ink-2", "today"), ("accent", "recommended")),
)

# ----------------------------------------------------------------------------------

w.finding(f"""Nothing here needed a global optimizer or a large design. Nine screening runs
    bought a direction, {path.n} cheap steps followed it, and the true yield went
    from {true_current:.2f} to {true_best:.2f} — {true_best - true_current:+.2f}
    verified against the simulation's own truth rather than against the fit's opinion
    of itself.""")
w.finding("""The part worth keeping is stage three. Arriving somewhere the gradient is flat
    does not tell you what kind of place it is, and the three possibilities — confirm
    and stop, keep walking along a ridge, back off a saddle — imply completely
    different next batches of runs.""")
w.finding("""Here the stationary point turned out to lie outside what the plant can run.
    That reframes the whole exercise: the answer is a corner of the operating window,
    and the next question is what widening the window would cost. An unconstrained
    optimizer would have returned an infeasible setpoint and no question at all.""")
