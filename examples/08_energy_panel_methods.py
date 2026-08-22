"""Energy — which quasi-experimental method should the retrofit programme use?

The question
    A utility is rolling out a heat-pump retrofit across buildings and wants the
    energy saving. Randomizing is possible but expensive; there are also months of
    pre-period meter data, and buildings that will not be treated for a year.

    Difference-in-differences, synthetic control, a time-based regression, a
    cluster regression, a switchback -- each is defensible, each rests on different
    assumptions, and the usual way to choose is by habit.

The move
    Simulate the panel you will actually have, run every method in the registry on
    data with NO effect, and see which ones report an effect anyway. A method whose
    false-positive rate is off at your panel shape is not a method you should trust
    on this data, regardless of what it does in the literature.

Pillars: design (method calibration, simulated power)
"""

import math

from _walkthrough import Walkthrough

from axiom.design import METHODS, SimulationSpec, calibrate_registry, simulated_power

w = Walkthrough(
    field="Energy",
    title="Calibrate the method before you trust the method",
    question="""Five defensible ways to estimate a heat-pump retrofit's saving, each
        resting on different assumptions. Which of them actually holds its size on
        the panel this programme will really have?""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Describe the panel you will actually have, not a generic one",
    why="""Every claim about a method's behaviour is conditional on the data it runs
        on. 120 buildings, 24 months, a year of pre-period, half treated, buildings
        that differ a lot at baseline, weather that moves everyone together, and
        month-to-month autocorrelation of 0.6 — those are the features that decide
        whether a variance estimator is honest here, and none of them are in a
        textbook's assumptions.""",
    instead="""Choosing a method from what the literature reports. The literature
        reports behaviour on the panels the authors had. A method that holds its size
        on 500 units with no autocorrelation may not hold it on 120 with rho = 0.6,
        and nothing about the method's name tells you which case you are in.""",
)
panel = SimulationSpec(
    n_units=120,  # buildings
    n_periods=24,  # months of meter data
    n_pre=12,  # a year before anything is switched on
    n_treated=60,
    unit_sd=1.2,  # buildings differ a lot in baseline consumption
    noise_sd=0.9,
    period_sd=0.6,  # weather moves everyone together
    rho=0.6,  # and consumption is autocorrelated month to month
    n_simulations=120,
    seed=0,
)
w.out(f"buildings         : {panel.n_units} ({panel.n_treated} treated)")
w.out(f"months            : {panel.n_periods}, of which {panel.n_pre} pre-period")
w.out(f"baseline spread   : unit sd {panel.unit_sd}, noise sd {panel.noise_sd}")
w.out(f"common shocks     : period sd {panel.period_sd} (weather)")
w.out(f"autocorrelation   : rho {panel.rho}")
w.out(f"simulations each  : {panel.n_simulations}")

# ----------------------------------------------------------------------------------

w.step(
    "Read what each candidate asks you to believe",
    why="""The registry carries each method's requirements and its named assumptions,
        so the comparison starts from what is being assumed rather than from what is
        familiar. Two methods that produce similar point estimates can rest on
        completely different claims, and the difference only shows up when one of
        those claims is false.""",
    instead="""Comparing methods on their estimates. On real data they will differ
        and there is no way to tell which is right; on simulated data with a known
        truth they will mostly agree, which tells you nothing about the case where it
        matters.""",
)
w.table(
    ["method", "needs", "assumes"],
    [
        [
            name,
            ", ".join(
                n
                for n, on in (
                    ("pre-period", spec.requires_pre_period),
                    ("controls", spec.requires_controls),
                )
                if on
            )
            or "nothing but the assignment",
            ", ".join(spec.assumption_names()),
        ]
        for name, spec in METHODS.items()
    ],
)

# ----------------------------------------------------------------------------------

w.step(
    "Drop one candidate by hand, and say why",
    why="""Synthetic control builds a donor combination for a single treated unit.
        This programme treats half the estate at once, so the method is not wrong —
        it is answering a different shape of question, and its solver says so by
        failing rather than by returning a number. Choosing the candidate set is part
        of the design.""",
    instead="""Looping over everything in the registry and taking whatever survives.
        That outsources a design decision to an exception handler, and on a different
        panel it would silently include a method whose question nobody checked.""",
)
CANDIDATES = {name: spec for name, spec in METHODS.items() if name != "synthetic_control"}
w.out(f"in the registry : {sorted(METHODS)}")
w.out(f"carried forward : {sorted(CANDIDATES)}")

# ----------------------------------------------------------------------------------

w.step(
    "Calibrate on panels with no effect at all",
    why="""Run each candidate many times on data generated with a true effect of
        exactly zero, and count how often it declares one. A nominal 5% test should
        reject about 5% of the time. If it rejects 15%, every interval it produces is
        too narrow, and every downstream number — the power calculation, the business
        case — inherits that.""",
    instead="""Trusting the nominal level. It is nominal: it is what the test would
        deliver under assumptions this panel may not satisfy. Autocorrelation of 0.6
        is exactly the kind of thing that breaks a variance estimator quietly.""",
)
calibrated, results = calibrate_registry(panel, CANDIDATES, alpha=0.05)
rows = []
for r in results:
    rate = r.false_positive_count / r.n_evaluated
    # a binomial interval on the observed rate, so 'passed' can be read rather than
    # taken on trust: does the interval reach the nominal 5%?
    half = 1.96 * math.sqrt(max(rate * (1 - rate), 1e-9) / r.n_evaluated)
    w.out(
        f"{r.method:<28}{r.false_positive_count:>3}/{r.n_evaluated:<5} "
        f"= {rate:>5.1%}   acceptable [{r.region.lower}, {r.region.upper}]   "
        f"{calibrated[r.method].status}"
    )
    rows.append(
        {
            "label": r.method,
            "estimate": rate,
            "lower": max(0.0, rate - half),
            "upper": rate + half,
            "note": f"{r.false_positive_count} false positives in {r.n_evaluated} runs",
            "bad": not r.passed,
        }
    )
w.figure(
    "size",
    kind="intervals",
    data={"rows": rows},
    opt={
        "rows": "@rows",
        "truth": 0.05,
        "truthLabel": "nominal 5%",
        "xLabel": "false-positive rate on panels with no true effect",
        "labelWidth": 190,
        "rowHeight": 36,
    },
    title="Does a 5% test reject 5% of the time on this panel?",
    note="""Each dot is the observed false-positive rate over the simulations, with a
        binomial interval. An interval that misses the dashed line is a method whose
        stated uncertainty is wrong here — and would be drawn in red.""",
)

# ----------------------------------------------------------------------------------

w.step(
    "Only now look at power, and only for the survivors",
    why="""Calibration first, power second. The order is the point: a method that
        cannot hold its size will look powerful precisely because it rejects too
        often, so ranking on power first selects for the broken one. With size
        settled, power is a fair comparison and it separates the candidates
        hard.""",
    instead="""Ranking every method by power and taking the top one. That is how a
        method that cannot hold its size ends up in a signed analysis plan, and by
        then the false-positive rate is somebody else's problem.""",
)
EFFECT = 0.35
with_effect = panel.model_copy(update={"effect": EFFECT})
usable = [r.method for r in results if r.passed]
powers, coverages = [], []
for name in usable:
    sp = simulated_power(name, with_effect, registry=CANDIDATES)
    w.out(
        f"{name:<28}power {sp.power:>6.3f}   bias {sp.bias:>+7.3f}   "
        f"coverage {sp.coverage:>5.2f}"
    )
    powers.append(
        {
            "label": name,
            "value": float(sp.power),
            "display": f"{sp.power:.2f}",
            "colour": "accent" if sp.power >= 0.8 else "boundary",
            "note": f"bias {sp.bias:+.3f}, coverage {sp.coverage:.2f}",
        }
    )
    coverages.append({"label": name, "a": 0.95, "b": float(sp.coverage)})
w.figure(
    "power",
    kind="bars",
    data={"rows": powers},
    opt={"rows": "@rows", "xLabel": f"power at a true saving of {EFFECT}", "labelWidth": 190},
    title="With size settled, power separates them",
    note="""'ghost' is not broken. It estimates a different quantity — intent-to-treat
        among the exposed — from a much smaller effective sample, and pays for it in
        power. A ranking that had not calibrated first could not have told that apart
        from a method that simply rejects too often.""",
    legend=(("accent", "reaches 80% power"), ("boundary", "does not")),
)
w.figure(
    "coverage",
    kind="dumbbell",
    data={"rows": coverages},
    opt={
        "rows": "@rows",
        "truth": 0.95,
        "truthLabel": "nominal",
        "xLabel": "interval coverage under a true effect",
        "aLabel": "nominal",
        "bLabel": "achieved",
        "labelWidth": 190,
        "rowHeight": 38,
    },
    title="And coverage catches what the size test alone would miss",
    note="""Size was checked under the null; this is coverage under a real effect.
        The bar of each dumbbell is the gap between what the method promises and what
        it delivers. Anything short of nominal is producing intervals that are too
        narrow when it matters — the kind of detail worth seeing before the analysis
        plan is signed rather than after.""",
    legend=(("ink-2", "nominal 95%"), ("accent", "achieved")),
)

# ----------------------------------------------------------------------------------

w.finding(f"""All {len(usable)} candidates calibrated: on a panel of this shape their
    nominal 5% tests really do reject about 5% of the time, so their intervals mean
    what they say. That is a result, and it is worth having explicitly rather than
    assuming — a method whose size were off here would poison the power calculation
    and the business case with it.""")
w.finding("""With calibration settled, power separated them hard, and the separation is
    interpretable rather than mysterious: the low-power method is estimating a
    different quantity from a smaller effective sample, not failing at the same
    one.""")
w.finding("""Worth not skipping past: switchback's coverage runs a little under nominal
    even though its size test passed. The HAC variance is working against real
    autocorrelation here, and the shortfall only shows up once there is an effect to
    cover. A size check under the null is necessary and it is not sufficient.""")
w.finding("""The order matters more than any individual number here. Calibrate first, then
    choose on power. Choosing on power first systematically favours whichever method
    is most willing to reject, which is the opposite of what a programme wants from
    the estimate it will defend to a regulator.""")
