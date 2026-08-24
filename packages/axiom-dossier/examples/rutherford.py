"""GEIGER-1911 as a research report: an experiment designed, run, and read.

The case study under ``nbs/case-studies/rutherford/`` uses axiom to design an
experiment *before it is run* — where to point the counter, what shape to cut the
aperture, how long to leave it open, and when to stop — and then runs it. This
turns that into the document such an experiment produces: the design and its
reasoning up front, the bound after, and every number traced.

    python examples/rutherford.py                  # generated, no key needed
    python examples/rutherford.py --narrate        # prose through Gemini
    python examples/rutherford.py --apa            # an APA manuscript
    python examples/rutherford.py --apa --standalone   # HTML with plotly bundled

Writes ``out/geiger-full.{html,pdf,pptx}``, or ``out/geiger-apa.*`` with ``--apa``.

Nothing below is typed in. The station plan, the hours and the apertures come
from ``scattering.py``; the counts come from one Poisson draw at each station
through ``surface.counts``; and the bound comes from ``design.profile_likelihood``
against those counts. Change the seed and the bound changes with it.

**This is the report a design produces rather than a trial.** Two things follow.
The methods section is most of the document, because in a designed experiment the
argument *is* the design — an angle chosen for the wrong reason cannot be fixed by
the analysis. And the single finding is a one-sided bound, which is what an
experiment that excludes a hypothesis returns: there is no lower limit on the
radius here, and the report says so rather than printing a point estimate that
reads like one.
"""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "nbs" / "case-studies" / "rutherford"))

import scattering as S  # noqa: E402
from axiom.core import Assumption, Interval, LedgerLine, Unsupported, Verdict  # noqa: E402
from axiom.design import profile_likelihood  # noqa: E402

from axiom_dossier import EvidenceBuilder, Narrator, build  # noqa: E402

SEED = 1911
"""The seed notebook 5 runs the experiment at. Every count below is drawn with it."""

MASS = 0.95
"""One-sided, and the profile's 1.921 drop is exactly this. Printed where it is used."""

DROP = 1.921
"""The log-likelihood drop that bounds a one-parameter profile at 95 %, one-sided."""


# -- what the experiment rests on ------------------------------------------------------
#
# These are notebook 5's own four assumptions and its own verdict, transcribed
# rather than invented. Three were *measured* by a station the design bought for
# the purpose, which is the case study's point: the plan was built so that the
# things the bound leans on could be checked rather than assumed. The fourth was
# not, and that is why the verdict is `downgraded` rather than `identified`.

CORE_WIDTH = Assumption(
    name="core_width",
    facet="mechanism",
    statement="the multiple-scattering core is Gaussian with the fitted width",
    challenged_by="a wide-angle excess that a wider core would also explain",
    state="satisfied",
    detail={
        "measured_by": "the anchor stations at 1.0 and 2.5 degrees",
        "margin": "a 10% error would be caught with certainty",
    },
)
BACKGROUND = Assumption(
    name="background",
    facet="measurement",
    statement="counts with the foil out are the counts with it in, less scattering",
    challenged_by="a background that depends on the foil being there",
    state="satisfied",
    detail={"measured_by": "10 hours foil-out at the 90 degree aperture"},
)
APERTURE = Assumption(
    name="aperture",
    facet="measurement",
    statement="each slit reports the rate at its centre",
    challenged_by="a slit wide enough for the curvature of the law to matter",
    state="satisfied",
    detail={"largest_smearing": "0.05% at 150 degrees, a twentieth of the counting error"},
)
SINGLE_SCATTERING = Assumption(
    name="single_scattering",
    facet="mechanism",
    statement="a wide-angle count is one close encounter, not several",
    challenged_by="a thicker foil, where two encounters would compound",
    state="asserted",
    detail={"basis": "0.4 um of gold; the design never tested this and a thickness series would"},
)
VERDICT = Verdict(
    status="downgraded",
    reason=(
        "the single-scattering assumption is asserted from the foil thickness rather "
        "than measured; every other assumption the bound leans on was measured by a "
        "station the design bought for that purpose"
    ),
    assumptions=(CORE_WIDTH, BACKGROUND, APERTURE, SINGLE_SCATTERING),
    route="profile likelihood on the counts, through one forward()",
)

ONE_PARAMETER = Assumption(
    name="the_two_atoms_differ_only_in_the_radius_of_the_positive_charge",
    facet="method",
    statement=(
        "a hard centre and a charge spread through the atom are one model at two values "
        "of one number, the radius R of the positive charge"
    ),
    challenged_by="any difference between the two atoms that is not a matter of extent",
    state="asserted",
)
COUNTER_LINEAR = Assumption(
    name="the_counter_is_linear_below_its_rate_cap",
    facet="measurement",
    statement=(
        f"every aperture is opened only as wide as keeps the rate under "
        f"{S.MAX_COUNT_RATE:g} counts per second, where the counter does not miss particles"
    ),
    challenged_by=(
        "a station run wide enough to pile up, which would under-report the busiest angles"
    ),
    state="asserted",
)


@dataclass(frozen=True)
class Limit:
    """A one-sided bound: the limit itself, and the interval that states it.

    ``quantity_from`` reads a point and an interval off anything that carries
    them, which is the seam this uses. The point is the **bound**, not the
    midpoint of the interval and not where the profile bottoms: an experiment
    that excludes a hypothesis reports the edge it excluded past, and printing
    a point estimate here would imply a measurement nobody made.
    """

    value: float
    interval: Interval


def _run(world, seed: int = SEED):
    """One Poisson draw at every station: the experiment, as notebook 5 runs it."""
    surface = S.surface()
    plan = S.plan_data()
    counts = np.random.default_rng(seed).poisson(surface.counts(plan, world))
    measured = dict(plan)
    measured["root_count"] = 2.0 * np.sqrt(counts)
    return plan, counts, measured


def _bound(measured) -> tuple[float, float]:
    """The 95 % upper limit on R, and where the profile bottoms, both in metres.

    A larger ``lam`` is a smaller charge, so the edge that carries information is
    the last grid point above the drop *before* the minimum. Nothing past the
    minimum is read as a bound: the profile there is level to within a
    fluctuation, which is the whole reason this is reported one-sided.
    """
    with warnings.catch_warnings():
        # The optimizer walks through parameter values where exp() overflows on
        # its way to the optimum. Those points are scored and discarded.
        warnings.simplefilter("ignore", RuntimeWarning)
        values, drops = profile_likelihood(
            S.model(), measured, {**S.HARD_CENTRE, "lam": 0.0}, "lam", grid=np.linspace(-4, 4, 41)
        )
    drops = np.asarray(drops)
    lowest = int(np.argmin(drops))
    last = int(np.where(drops[:lowest] > DROP)[0][-1])
    crossing = float(
        np.interp(DROP, [drops[last + 1], drops[last]], [values[last + 1], values[last]])
    )
    return S.radius_of(crossing), S.radius_of(values[lowest])


def evidence():
    """Run the experiment and collect everything the report is entitled to say."""
    plan, counts, measured = _run(S.HARD_CENTRE)
    bound, profile_min = _bound(measured)
    floor = S.D_CLOSEST / 2.0

    inn = [bool(st.foil) for st in S.PLAN]
    if_diffuse = S.surface().counts(plan, S.DIFFUSE)
    witness = int(counts[inn][-1])
    witness_if_diffuse = float(if_diffuse[inn][-1])
    background = int(counts[[not f for f in inn]][0])
    hours = sum(st.hours for st in S.PLAN)

    by_role: dict[str, list[S.Station]] = {}
    for st in S.PLAN:
        by_role.setdefault(st.role, []).append(st)

    def angles(role: str) -> str:
        return ", ".join(f"{st.theta_deg:g}°" for st in by_role[role])

    def role_hours(role: str) -> str:
        return f"{sum(st.hours for st in by_role[role]):g}"

    deepest = f"{sum(st.hours for st in by_role['witness'] if st.theta_deg == 150.0):g}"

    # Written as the question the bound settles. The debate is "concentrated or
    # spread", but that is an either/or and a conclusions section cannot answer
    # it from an interval; "is it far smaller than the atom" is the same question
    # asked so that a bound answers it.
    builder = EvidenceBuilder(
        "GEIGER-1911",
        "Is the positive charge of a gold atom concentrated in a centre far smaller "
        "than the atom itself?",
    )
    builder.verdict(VERDICT)

    # -- the design, which in a planned experiment is the argument ---------------------
    builder.step(
        "parameter",
        "The two atoms as one parameter",
        what=(
            "A head-on alpha of this energy stops at "
            f"{S.D_CLOSEST * 1e15:.1f} fm, and a trajectory scattered through θ comes no "
            "closer than half that times (1 + 1/sin(θ/2)). A positive charge of radius R "
            "therefore kills the scattering past a cut-off angle, so the two atoms are "
            "two values of R — about "
            f"{S.R_NUCLEUS * 1e15:.1f} fm against {S.R_ATOM * 1e15:.0f} fm."
        ),
        why=(
            "A parameter is what an experiment can have resolving power against. Two "
            "rival stories with no number between them can be argued about but not "
            "designed for."
        ),
        instead=(
            "Treating the two atoms as separate models and asking which fits better. "
            "That returns a verdict on the data in hand and no sensitivity curve, so it "
            "cannot say which angles to buy or how long to buy them for."
        ),
        equations=(
            "D        = 2 z Z e^2 / (4 pi eps0 E)          closest approach, head-on",
            "r_min(t) = (D/2) (1 + 1/sin(t/2))             closest approach at angle t",
            "sin(t_cut/2) = 1 / (2R/D - 1)                 where a charge of radius R",
            "                                              kills the scattering",
            "lam      = log sin(t_cut/2)                   the fitted parameter",
        ),
        detail={
            "beam energy (MeV)": f"{S.E_ALPHA:g}",
            "closest approach D (fm)": f"{S.D_CLOSEST * 1e15:.2f}",
            "hard centre R (fm)": f"{S.R_NUCLEUS * 1e15:.1f}",
            "diffuse atom R (fm)": f"{S.R_ATOM * 1e15:.0f}",
            "foil areal density (atoms/m²)": f"{S.AREAL_DENSITY:.3g}",
        },
        assumptions=[ONE_PARAMETER],
    )
    builder.step(
        "stations",
        "Where to point the counter",
        what=(
            f"{len(S.PLAN)} stations in four roles: anchors at {angles('anchor')} to pin "
            f"the multiple-scattering core, a bank at {angles('bank')} where the "
            f"information is, witnesses at {angles('witness')} whose evidence survives "
            f"the core model being attacked, and a foil-out run at "
            f"{angles('background')} for the background."
        ),
        why=(
            "No single angle plays two of these roles. The bank carries the most "
            "information per hour, but only if the model of the core is believed; the "
            "witness carries less and does not depend on it."
        ),
        instead=(
            "Spending every hour where evidence-per-hour is highest. That number is "
            "computed inside the model that is under attack, so following it buys the "
            "most evidence of the kind an objector can decline to accept."
        ),
        detail={
            "stations": str(len(S.PLAN)),
            "anchor": f"{angles('anchor')} ({role_hours('anchor')} h)",
            "bank": f"{angles('bank')} ({role_hours('bank')} h)",
            "witness": f"{angles('witness')} ({role_hours('witness')} h)",
            "background": f"{angles('background')}, foil out ({role_hours('background')} h)",
        },
        assumptions=[SINGLE_SCATTERING, BACKGROUND, CORE_WIDTH],
    )
    builder.step(
        "aperture",
        "What shape to cut the aperture",
        what=(
            "An annular slot rather than a round hole. A slit covering azimuth φ at "
            "θ ± δ subtends φ·2·sin θ·sin δ, so the same counts can be bought in either "
            "direction — and only δ smears the angle. The radial half-width is held at "
            "the workshop floor and every extra steradian is taken out in azimuth."
        ),
        why=(
            "The scattering does not depend on azimuth, so azimuth is free resolution. "
            "Opening radially costs angular resolution and buys nothing the arc cannot "
            "buy first."
        ),
        instead=(
            "A round hole of the same area. At 90° it would have to be about ±13° wide "
            "and would report a rate five per cent high, belonging to no angle in "
            "particular. The slot reports it to a part in four thousand."
        ),
        equations=(
            "Omega(t, d, phi) = phi * 2 sin(t) sin(d)      an annular slot subtends",
            "  phi = azimuthal arc, d = radial half-width",
            "  only d smears the angle; phi is free resolution",
        ),
        detail={
            "widest aperture (sr)": f"{S.OMEGA_MAX:g}",
            "rate cap (counts/s)": f"{S.MAX_COUNT_RATE:g}",
            "beam rate (alphas/s)": f"{S.BEAM_RATE:.3g}",
        },
        assumptions=[APERTURE, COUNTER_LINEAR],
    )
    builder.step(
        "hours",
        "How long at each",
        what=(
            f"{hours:g} observing hours in all, of which {role_hours('witness')} go to "
            f"the witnesses and {deepest} to 150° alone."
        ),
        why=(
            "150° is the only station whose evidence cannot be talked out of: it is past "
            "what a core thirty times too wide could reach. Buying every constraint the "
            "argument will need as cheaply as it can be bought leaves the rest for there."
        ),
        instead=(
            "Allocating by evidence-per-hour, which puts the time in the bank. That is "
            "the opposite of this plan and it is wrong for the same reason the station "
            "choice is: the ranking is computed inside the model under attack."
        ),
        equations=(
            "N(t) ~ Poisson( Omega * T * [ f_tail(t) + f_core(t) + b ] )",
            "  f_tail(t) = exp(log_a) / sin^4(t/2)         single scattering",
            "  f_core(t) = exp(log_c) exp(-t^2 / 2 exp(log_w)^2)   multiple scattering",
            "  b         = exp(log_b)                      counter background",
            "  the tail is cut off above t_cut(lam)",
        ),
        detail={
            "total hours": f"{hours:g}",
            "hours at 150°": deepest,
            "seed": str(SEED),
        },
    )

    # -- what came back ----------------------------------------------------------------
    builder.finding(
        "radius",
        Limit(
            value=bound * 1e15,
            interval=Interval(lower=0.0, upper=bound * 1e15, definition="wald", mass=MASS),
        ),
        label="Upper limit on the radius of the positive charge of gold",
        unit="fm",
        precision=1,
        # The decision is between the two atoms, so the value it turns on is the
        # radius the diffuse atom would have. An interval wholly below it excludes
        # that atom; this one is below it by four decades.
        threshold=S.R_ATOM * 1e15,
        # Smaller is the answer "yes, concentrated": the bound lying below the
        # radius a diffuse atom would have is what settles the question.
        beneficial="lower",
        note=(
            "One-sided. The profile has no lower edge — everything below this bound is "
            "equally consistent with the counts — because the beam itself cannot resolve "
            f"past {floor * 1e15:.1f} fm, whatever the counting time."
        ),
        mass=MASS,
        source="design.profile_likelihood on lam",
    )

    builder.diagnostic(
        "profile_min",
        profile_min * 1e15,
        label="Radius at which the profile likelihood bottoms",
        unit="fm",
        precision=1,
    )
    builder.diagnostic(
        "floor",
        floor * 1e15,
        label="Radius the beam energy alone could never resolve past",
        unit="fm",
        precision=1,
    )
    builder.diagnostic(
        "witness_counts",
        float(witness),
        label="Counts at 150 degrees, foil in",
        precision=0,
    )
    builder.diagnostic(
        "witness_if_diffuse",
        witness_if_diffuse,
        label="Counts a diffuse atom predicts at 150 degrees",
        precision=1,
    )
    builder.diagnostic(
        "background_counts",
        float(background),
        label="Counts at 90 degrees with the foil out",
        precision=0,
    )
    builder.diagnostic(
        "total_counts",
        float(counts.sum()),
        label="Counts recorded across every station",
        precision=0,
    )

    builder.assume(ONE_PARAMETER, COUNTER_LINEAR)
    builder.ledger_lines(
        [
            LedgerLine(
                kind="assumption",
                statement=(
                    "The witness stations were bought so that the wide-angle evidence "
                    "does not rest on the model of the multiple-scattering core. They "
                    "are the reason the bound survives that model being wrong."
                ),
                assumption=SINGLE_SCATTERING,
            ),
            LedgerLine(
                kind="assumption",
                statement=(
                    "The bound is reported against the radius a diffuse atom would "
                    "have, not against zero: this experiment excludes an atom, it does "
                    "not measure a nucleus."
                ),
                assumption=ONE_PARAMETER,
            ),
        ]
    )
    builder.remark(
        f"At 150 degrees the counter recorded {witness:,} particles where a diffuse atom "
        f"predicts {witness_if_diffuse:.0f} — a Poisson excess of "
        f"{(witness - witness_if_diffuse) / math.sqrt(witness):,.0f} standard errors. "
        "There is no version of this that is a fluctuation.",
        "The hours went where the argument was weakest rather than where the information "
        "was densest, and that is what the bound is made of. Dropping the 150 degree "
        "witness and keeping every other station loses most of it.",
    )
    builder.step(
        "inference",
        "Reading the bound off the counts",
        what=(
            "The five parameters were profiled over the cut-off, and the 95 % one-sided "
            f"limit is where the log-likelihood has climbed {DROP} above its minimum on "
            "the side that carries information."
        ),
        why=(
            "A larger cut-off is a smaller charge, so only one side of the profile "
            "bounds anything. The other side is level to within a fluctuation, which is "
            "why the answer is a limit rather than a measurement."
        ),
        instead=(
            "Quoting the profile's minimum as the radius. Everything below the bound "
            "fits the counts equally well, so a point estimate there would be a number "
            "the experiment did not produce."
        ),
        equations=(
            f"R_95 : 2 [ l(lam_hat) - l(lam) ] = {DROP}     one-sided, one parameter",
            "  l(.)  = the Poisson log-likelihood over all stations",
            "  R     = (D/2) (1 + 1/exp(lam))               radius from the cut-off",
        ),
        detail={
            "profiled parameter": "lam",
            "nuisance parameters": "log_a, log_c, log_w, log_b",
            "grid": "41 points over lam in [-4, 4]",
            "drop at 95% one-sided": f"{DROP}",
        },
    )
    builder.provenance(
        seed=str(SEED),
        world="nbs/case-studies/rutherford/scattering.py",
        estimator="profile likelihood on lam, one-sided",
        interval_mass=f"{MASS:.0%}",
        observing_hours=f"{hours:g}",
    )
    return builder.build()


def main() -> int:
    narrate = "--narrate" in sys.argv
    apa = "--apa" in sys.argv
    stem = "geiger-apa" if apa else "geiger-full"
    ev = evidence()
    built = build(
        ev,
        narrator=Narrator() if narrate else None,
        style="apa" if apa else "journal",
        verbosity="full",
        authors=("Matthew Reda",) if apa else (),
        affiliation="axiom" if apa else "",
    )

    print(f"evidence hash : {ev.content_hash()}")
    print(f"document      : {built.summary()}")
    print(f"findings      : {len(ev.findings)}   diagnostics: {len(ev.diagnostics)}")
    print(f"unresolved    : {', '.join(ev.unresolved()) or 'none'}")
    print()
    for q in ev.findings:
        print(f"  {q.label:44s} {q.stated():34s} [{q.against_threshold()}]")
    print()
    for n in built.narrations:
        print(f"  {n.summary()}")
    if built.rejected():
        print("\n  ^ kept the generated draft for those; nothing wrong reached the page.")

    out = Path(__file__).parent / "out"
    print()
    for fmt in ("html", "pdf", "pptx"):
        result = built.write(str(out / f"{stem}.{fmt}"), inline_plotly="--standalone" in sys.argv)
        if isinstance(result, Unsupported):
            print(f"  {fmt:5s} skipped — {result.reason}")
        else:
            print(f"  {fmt:5s} {result} ({Path(result).stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
