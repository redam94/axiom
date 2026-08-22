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

# The operating window: temperature 120-200 C, residence time 5-45 minutes.
window = Bounds(treatments=("temperature", "residence"), low=(120.0, 5.0), high=(200.0, 45.0))

# Stage one: a cheap screening design to find the uphill direction.
screen = full_factorial(window, 3)
print(f"stage 1 - screening design: {screen.kind}, {screen.n} runs")

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
print(f"  converged: {screen_fit.converged}")

# Stage two: walk uphill from where the process is running today.
current = {"temperature": 140.0, "residence": 12.0}
print(
    f"\nstage 2 - steepest ascent from the current setpoint "
    f"({current['temperature']:.0f} C, {current['residence']:.0f} min)"
)

path = steepest_ascent(screen_fit.surface, theta, current, step=6.0, n_steps=40, bounds=window)
print(f"  took {path.n} steps, stopped because: {path.stop}")

print(f"\n  {'step':>5}{'temperature':>14}{'residence':>12}{'predicted yield':>18}")
show = [0, 1, 2, max(0, path.n // 3), max(0, 2 * path.n // 3), path.n - 1]
for i in sorted(set(s for s in show if 0 <= s < path.n)):
    # points come back in the surface's treatment order, as plain tuples
    temperature, residence = (float(v) for v in path.points[i])
    print(f"  {i:>5}{temperature:>14.1f}{residence:>12.1f}{path.values[i]:>18.3f}")

best = path.best()
print("\n  best point on the path: " + ", ".join(f"{k} {v:.1f}" for k, v in best.items()))
print(f"  predicted yield there: {path.values[-1]:.3f}")

# Stage three: a second-order design at the new location, then ask what kind of
# stationary point it is.
print("\nstage 3 - characterising the neighbourhood")
confirm = central_composite(window, alpha="rotatable", center_points=4, inscribed=True)
print(f"  confirmation design: {confirm.kind}, {confirm.n} runs")

stationary = canonical_analysis(screen_fit.surface, theta, best)
if hasattr(stationary, "reason"):
    print(f"  canonical analysis declined: {stationary.reason}")
else:
    print(f"  stationary point is a {stationary.kind}")
    at = ", ".join(
        f"{name} {float(v):.1f}"
        for name, v in zip(window.treatments, stationary.point, strict=True)
    )
    print(f"  at {at}")
    print(f"  eigenvalues: {[round(float(e), 5) for e in stationary.eigenvalues]}")
    print("    both negative -> a genuine maximum; mixed signs -> a saddle, and the")
    print("    positive direction is free yield you have not taken yet.")

    inside = all(
        lo <= float(v) <= hi
        for v, lo, hi in zip(stationary.point, window.low, window.high, strict=True)
    )
    print(f"  inside the operating window? {inside}")
    if not inside:
        print("    The unconstrained peak sits outside what the plant can physically run,")
        print("    which is why the ascent stopped on the boundary rather than at the")
        print("    stationary point. The operating recommendation is the corner of the")
        print("    window, and the interesting question becomes what it would cost to")
        print("    widen the window -- an engineering question, now correctly posed.")

# How much was left on the table by running at the old setpoint?
true_at_current = float(reactor.forward(current).mean())
true_at_best = float(reactor.forward({k: float(v) for k, v in best.items()}).mean())
print(f"\ntrue yield at the old setpoint : {true_at_current:.3f}")
print(f"true yield at the ascent's best: {true_at_best:.3f}")
print(f"improvement                    : {true_at_best - true_at_current:+.3f}")

print("""
Reading it
    Nothing here needed a global optimizer or a large design. Steepest ascent is a
    sequence of cheap local decisions, and it is still the right first move when
    runs are expensive and the surface is smooth -- which describes most physical
    processes.

    The part worth keeping is stage three. Arriving somewhere the gradient is flat
    does not tell you what kind of place it is, and the three possibilities imply
    completely different next moves: confirm and stop, or keep walking along a
    ridge, or back off because you are on a saddle between two better regions.""")
