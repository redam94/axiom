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

from axiom.design import METHODS, SimulationSpec, calibrate_registry, simulated_power

# The panel the programme will actually have.
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

print("methods in the registry, and what each one asks you to believe:")
for name, spec in METHODS.items():
    needs = []
    if spec.requires_pre_period:
        needs.append("pre-period")
    if spec.requires_controls:
        needs.append("controls")
    print(f"\n  {name}")
    print(f"    needs      : {', '.join(needs) or 'nothing but the assignment'}")
    print(f"    assumes    : {', '.join(spec.assumption_names())}")

# Synthetic control is deliberately left out. It builds a donor combination for a
# single treated unit, and this programme treats half the estate at once -- the
# method is not wrong, it is answering a different shape of question, and its
# solver says so by failing rather than returning a number. Choosing the candidate
# set is part of the design, not something to leave to a loop over everything.
CANDIDATES = {name: spec for name, spec in METHODS.items() if name != "synthetic_control"}

# Calibrate: run every candidate on panels with no effect at all, and check the
# false-positive rate lands where a 5% test says it should.
print("\n\ncalibrating on panels with NO true effect " f"({panel.n_simulations} simulations each):")
calibrated, results = calibrate_registry(panel, CANDIDATES, alpha=0.05)
print(f"\n  {'method':<28}{'false positives':>17}{'acceptable region':>21}  status")
for r in results:
    fp = f"{r.false_positive_count}/{r.n_evaluated}"
    region = f"[{r.region.lower}, {r.region.upper}]"
    status = calibrated[r.method].status
    flag = "" if r.passed else "   <-- not usable here"
    print(f"  {r.method:<28}{fp:>17}{region:>21}  {status}{flag}")

# Now power, for the methods that survived, at an effect worth acting on.
EFFECT = 0.35
with_effect = panel.model_copy(update={"effect": EFFECT})
usable = [r.method for r in results if r.passed]

print(f"\n\npower at a true saving of {EFFECT}, for the methods that calibrated:")
print(f"  {'method':<28}{'power':>8}{'bias':>9}{'coverage':>10}")
for name in usable:
    sp = simulated_power(name, with_effect, registry=CANDIDATES)
    print(f"  {name:<28}{sp.power:>8.3f}{sp.bias:>+9.3f}{sp.coverage:>10.2f}")

print("""
Reading it
    All five candidates calibrated: on a panel of this shape their nominal 5% tests
    really do reject about 5% of the time, so their intervals mean what they say.
    That is the result, and it is worth having explicitly rather than assuming --
    a method whose false-positive rate is off here would poison every downstream
    number, the power calculation and the business case included.

    With calibration settled, power separates them, and it separates them hard.
    'ghost' is not broken; it estimates a different quantity from a much smaller
    effective sample -- intent-to-treat among the exposed -- and pays for it. Note
    also switchback's coverage running a little under nominal: the HAC variance is
    working against real autocorrelation here, and that is the kind of detail worth
    seeing before the analysis plan is signed rather than after.

    The order matters. Calibrate first, then choose on power. Choosing on power
    first is how a method that cannot hold its size ends up in a protocol.""")
