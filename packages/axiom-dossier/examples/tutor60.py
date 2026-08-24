"""TUTOR-60 as a research report: one decision, from the budget to the memo.

The case study under ``nbs/case-studies/tutoring/`` runs the whole of axiom
against a single decision across five notebooks — what is being asked, how to
plan the trial, what it measured, and what to fund. This turns that into the
document the decision produces: the design and its reasoning, the estimated
response, and the recommendation with what would overturn it.

    python examples/tutor60.py                  # generated, no key needed
    python examples/tutor60.py --narrate        # prose through Gemini
    python examples/tutor60.py --apa            # an APA manuscript
    python examples/tutor60.py --apa --standalone   # HTML with plotly bundled

Writes ``out/tutor60-full.{html,pdf,pptx}``, or ``out/tutor60-apa.*`` with ``--apa``.

It fits the trial, which takes a minute or two: ``tutoring.run`` builds the
realized program and ``surface.fit`` samples the response through PyMC. Nothing
below is typed in — the allocation comes from ``surface.allocate`` against the
posterior, the advantage from a paired comparison inside every draw, and what
would overturn it from ``diagnose.tipping_point``.

**The finding is a comparison between two things you could buy**, not an effect
of a treatment. That is what a decision report has instead of a headline effect:
the number that matters is the difference between the recommendation and the
thing the room already believes, priced in the units the budget is written in.

The module is named ``tutor60`` rather than ``tutoring`` on purpose: the case
study's own world module is ``tutoring.py``, and this file has to be importable
from the same interpreter without shadowing it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "nbs" / "case-studies" / "tutoring"))

import tutoring as T  # noqa: E402
from axiom.core import Assumption, LedgerLine, Unsupported, summarize  # noqa: E402
from axiom.diagnose import tipping_point  # noqa: E402
from axiom.identify import CausalGraph, identify  # noqa: E402
from axiom.infer import PymcBackend  # noqa: E402
from axiom.surface import Allocation, Surface, allocate, fit, predict  # noqa: E402

from axiom_dossier import EvidenceBuilder, Narrator, build  # noqa: E402

SEED = 0
"""The seed notebook 5 fits at. Every posterior number below is drawn with it."""

MASS = 0.9
"""Every interval in this report is a 90 % one, and says so where it is printed."""

PILOT_TUTORING = 1800.0
"""USD/student-year — the pilot district's 90 weekly minutes, and the number to beat."""


# -- what the recommendation rests on --------------------------------------------------
#
# Transcribed from notebook 5's ledger rather than invented. Two were settled by
# the design, and two are the decision's own commitments: what a reading point is
# worth, and which response family was fitted.

RANDOMIZED = Assumption(
    name="schools_were_randomized_to_allocations",
    facet="design",
    statement="schools were randomized to allocations, so the adjustment set is empty",
    challenged_by="an imbalance across arms in a pre-assignment school characteristic",
    state="satisfied",
    detail={"route": "backdoor", "unit_of_assignment": "school"},
)
ASSIGNED_NOT_DELIVERED = Assumption(
    name="the_estimand_is_over_assigned_allocations",
    facet="estimand",
    statement=(
        "the recommendation is over assigned allocations; schools delivered 93% of "
        "assigned minutes and that shortfall is priced into the estimate"
    ),
    challenged_by="a delivery rate that differs across the allocations being compared",
    state="satisfied",
    detail={"version": "assigned"},
)
VALUE_PER_POINT = Assumption(
    # Names are rendered into prose with the underscores turned to spaces, so a
    # possessive in one arrives without its apostrophe: "the departments
    # valuation" is what "a_reading_point_is_worth_the_departments_valuation"
    # printed into the conclusions.
    name="a_reading_point_has_the_value_the_state_assigns_it",
    facet="decision",
    statement=(
        f"a reading point is valued at {T.VALUE_PER_POINT:,.0f} USD; the recommendation "
        "does not depend on this number, only on whether to fund at all"
    ),
    challenged_by="a valuation low enough to make the whole program not worth funding",
    state="asserted",
    detail={"source": "state cost-benefit memorandum 2026"},
)
RESPONSE_FAMILY = Assumption(
    name="the_response_family_does_not_decide_the_answer",
    facet="method",
    statement=(
        "a monotone response family fitted to the same data recommends the same "
        "allocation and a sixth more value; the families part company only at three "
        "times this budget"
    ),
    challenged_by="a budget large enough for the families to disagree about the optimum",
    state="satisfied",
    detail={"checked": "notebook 3 section 6"},
)
GRADE_BANDS = Assumption(
    name="the_cohort_average_answers_the_question",
    facet="population",
    statement=(
        "the recommendation is made on the cohort average; the grade-band breakdown the "
        "program office asked for is not estimable from a school-randomized trial"
    ),
    challenged_by="a response that differs across grade bands enough to change the allocation",
    state="unverified",
)
#: What the state's own records could have answered, and could not. ``resources``
#: is a school's unobserved capacity: it drives both how much tutoring the school
#: buys and how much its pupils gain, and nothing in the administrative data
#: measures it. The report carries this graph so a reader can see the arrow the
#: trial was bought to delete.
OBSERVATIONAL = CausalGraph.from_edges(
    "resources -> poverty_share, resources -> tutoring, poverty_share -> tutoring, "
    "resources -> gain, poverty_share -> gain, tutoring -> gain",
    unmeasured=["resources"],
    name="what_the_district_knows",
)

#: The same world with the assignment randomized: the two arrows into
#: ``tutoring`` are gone, and with them the need to adjust for anything.
RANDOMIZED_GRAPH = CausalGraph.from_edges(
    "resources -> poverty_share, resources -> gain, poverty_share -> gain, tutoring -> gain",
    unmeasured=["resources"],
    name="what_the_trial_knows",
)

OBSERVATIONAL_VERDICT = identify(OBSERVATIONAL, "tutoring", "gain")
TRIAL_VERDICT = identify(RANDOMIZED_GRAPH, "tutoring", "gain")


def _fitted():
    """Run the trial and fit the response surface — the expensive part."""
    program = T.run(T.chosen_design())
    spec = T.planning_spec()
    result = fit(
        spec,
        T.school_panel(program),
        backend=PymcBackend(nuts_sampler="nutpie"),
        draws=1000,
        tune=1000,
        chains=4,
        seed=SEED,
    )
    return program, spec, result


def _cohort_draws(result, spec, tut, msg):
    """Posterior draws of cohort reading points at every point of the allocation grid.

    The program's effect is the difference between spending this and spending
    nothing, taken **inside each draw** so the uncertainty that cancels does
    cancel. Multiplying by how many students the budget then reaches is what
    turns a per-student response into the thing the decision is about.
    """
    surface = Surface(spec)
    nothing = predict(
        surface,
        result.posterior,
        {"tutoring": np.zeros_like(tut), "messaging": np.zeros_like(msg)},
        seed=SEED,
    )
    something = predict(surface, result.posterior, {"tutoring": tut, "messaging": msg}, seed=SEED)
    effect = (np.asarray(something.values) - np.asarray(nothing.values)).reshape(-1, *tut.shape)
    return T.students_served(tut, msg)[None, :, :] * effect


def evidence():
    """Fit the trial and collect everything the report is entitled to say."""
    program, spec, result = _fitted()

    tutoring_grid = np.arange(0.0, T.TUTORING_MAX + 1.0, 25.0)
    messaging_grid = np.asarray(T.MESSAGING_LEVELS)
    tut, msg = np.meshgrid(tutoring_grid, messaging_grid, indexing="ij")

    cohort = _cohort_draws(result, spec, tut, msg)
    expected = cohort.mean(axis=0)
    best = np.unravel_index(int(np.argmax(expected)), expected.shape)
    chosen_tutoring = float(tutoring_grid[best[0]])
    chosen_messaging = float(messaging_grid[best[1]])
    served = float(T.students_served(tut, msg)[best])

    # The comparison the decision actually is: the recommendation against what
    # the room already believes, paired inside each draw so the uncertainty
    # common to both cancels rather than being added twice.
    at_cap = int(np.argmax(messaging_grid == T.MESSAGING_MAX))
    pilot_row = int(np.argmin(np.abs(tutoring_grid - PILOT_TUTORING)))
    difference = cohort[:, best[0], best[1]] - cohort[:, pilot_row, at_cap]
    advantage = summarize(difference, mass=MASS)
    value = summarize(difference * T.VALUE_PER_POINT, mass=MASS)
    probability = float((difference > 0).mean())
    pilot_served = float(T.students_served(PILOT_TUTORING, T.MESSAGING_MAX))

    tipping = tipping_point(
        difference,
        decision_threshold=0.0,
        bias_grid=np.linspace(0.0, 2.0, 101) * float(difference.mean()),
        name="recommendation minus pilot",
        certainty=0.5,
    )

    per_student = T.BUDGET / T.N_ELIGIBLE
    at_budget = allocate(
        Surface(spec), result.posterior, budget=per_student, bounds=T.bounds(), seed=SEED
    )
    flat = tutoring_grid[
        cohort[:, :, at_cap].mean(axis=0) >= 0.98 * cohort[:, :, at_cap].mean(axis=0).max()
    ]

    # The question is written as the one the lead finding settles. The decision
    # behind it is "how much should we buy", but a how-much question has no
    # threshold and therefore no answer a conclusions section can state; the
    # comparison against what the room already believes does.
    builder = EvidenceBuilder(
        "TUTOR-60",
        "Does buying fewer tutoring minutes for every eligible student beat buying the "
        "pilot's ninety weekly minutes for the fraction of them the budget reaches?",
    )
    builder.verdict(TRIAL_VERDICT, graph=RANDOMIZED_GRAPH)

    # -- the decision, the design, the fit, the recommendation -------------------------
    builder.step(
        "decision",
        "The decision as arithmetic",
        what=(
            f"{T.BUDGET:,.0f} USD for one school year and {T.N_ELIGIBLE:,} students "
            f"reading two or more grade levels behind — {per_student:,.0f} USD a student "
            "if every one of them is reached. Both treatments are dosed in dollars per "
            f"eligible student per year: {T.MINUTE_COST:,.0f} USD buys a weekly tutoring "
            f"minute and {T.MESSAGE_COST:,.0f} USD a weekly caregiver message, capped at "
            f"{T.MESSAGE_CAP_PER_WEEK}."
        ),
        why=(
            "Putting both treatments on one scale is what makes the budget an actual "
            "constraint rather than a footnote. The decision variable is the per-student "
            "allocation, so that is the unit the response is estimated in."
        ),
        instead=(
            "Dosing tutoring in minutes and messaging in messages, which is how each is "
            "sold. The two cannot then be traded against one another, and the constraint "
            "that makes this a decision disappears."
        ),
        detail={
            "budget (USD)": f"{T.BUDGET:,.0f}",
            "eligible students": f"{T.N_ELIGIBLE:,}",
            "per student if all are reached (USD)": f"{per_student:,.0f}",
            "tutoring dose range (USD)": f"0 to {T.TUTORING_MAX:,.0f}",
            "messaging dose range (USD)": f"0 to {T.MESSAGING_MAX:,.0f}",
        },
    )
    builder.step(
        "design",
        "Planning the trial",
        what=(
            f"{program.n_schools} schools were randomized to allocations drawn from a "
            f"lattice over the two doses, with {len(T.MESSAGING_LEVELS)} messaging levels "
            "and tutoring spread across the range the decision might land in."
        ),
        why=(
            "Randomizing schools rather than students is what the program can actually "
            "deliver, and it is what leaves the adjustment set empty. The allocations are "
            "spread rather than concentrated because the question is the *shape* of the "
            "response, not whether one dose beats zero."
        ),
        instead=(
            "The state's own records on 240 schools. In that graph a school's "
            "unmeasured capacity drives both how much tutoring it buys and how much "
            f"its pupils gain — {OBSERVATIONAL.to_text()} — so the effect is "
            f"{OBSERVATIONAL_VERDICT.status} and no adjustment available in the data "
            "recovers it. Randomizing the assignment deletes the two arrows into "
            "tutoring, which is the whole of what the trial buys. A two-arm trial at "
            "the pilot's 90 minutes would have answered 'does tutoring work', which "
            "the prior already settles, and nothing about how much to buy."
        ),
        equations=(
            "assignment: (tutoring_s, messaging_s) ~ Uniform(lattice)   per school s",
            "observational graph : " + OBSERVATIONAL.to_text(),
            "  status " + OBSERVATIONAL_VERDICT.status + " - requires the unmeasured node",
            "randomized graph    : " + RANDOMIZED_GRAPH.to_text(),
            "  status " + TRIAL_VERDICT.status + " via " + str(TRIAL_VERDICT.route),
        ),
        detail={
            "unit of assignment": "school",
            "schools": str(program.n_schools),
            "messaging levels": ", ".join(f"{v:,.0f}" for v in T.MESSAGING_LEVELS),
            "eligible per school": f"{T.ELIGIBLE_PER_SCHOOL[0]}–{T.ELIGIBLE_PER_SCHOOL[1]}",
        },
        assumptions=[RANDOMIZED, ASSIGNED_NOT_DELIVERED],
    )
    builder.step(
        "fit",
        "Estimating the response",
        what=(
            "A spline response in each dose was fitted to the school-level panel through "
            f"PyMC, {result.posterior.n_draws():,} posterior draws in all. Cohort reading "
            "points at any allocation are then the fitted per-student gain times the "
            "number of students that allocation's price reaches."
        ),
        why=(
            "The quantity the decision is about is not the per-student effect, it is the "
            "total the budget delivers. Every extra dollar a student is a student not "
            "served, and only the product of the two carries that trade."
        ),
        instead=(
            "Reporting the per-student response alone and letting the reader multiply. "
            "The multiplication has to happen inside each posterior draw; done afterwards "
            "on the means it drops the covariance and misstates the interval."
        ),
        equations=(
            "gain_s = f(tutoring_s) + g(messaging_s) + eps_s",
            "  f, g   = penalised splines, one per dose, on the school-level panel",
            "effect(a) = E[gain | dose = a] - E[gain | dose = 0]     within each draw",
            "served(a) = min(N_eligible, Budget / cost(a))",
            "cohort(a) = served(a) * effect(a)                       the objective",
            "cost(a)   = a_tutoring + a_messaging   (USD per eligible student-year)",
        ),
        detail={
            "response family": "spline in both doses",
            "sampler": "PyMC via nutpie",
            "posterior draws": f"{result.posterior.n_draws():,}",
            "fit hash": spec.content_hash()[:12],
            "interval mass": f"{MASS:.0%}",
        },
        assumptions=[RESPONSE_FAMILY, GRADE_BANDS],
    )
    builder.step(
        "recommendation",
        "Choosing the allocation",
        what=(
            f"The allocation with the highest expected cohort points is "
            f"{chosen_tutoring:,.0f} USD of tutoring "
            f"({chosen_tutoring / T.MINUTE_COST:.0f} weekly minutes) and "
            f"{chosen_messaging:,.0f} USD of messaging "
            f"({chosen_messaging / T.MESSAGE_COST:.0f} messages a week), which reaches "
            f"{served:,.0f} of the {T.N_ELIGIBLE:,} eligible students. It is compared with "
            f"the pilot's 90 weekly minutes, which the same budget stretches to only "
            f"{pilot_served:,.0f} students."
        ),
        why=(
            "The comparison is paired inside every posterior draw, so what is reported is "
            "the distribution of the *difference* rather than the difference of two "
            "distributions."
        ),
        instead=(
            "Funding the pilot's 90 minutes for as many students as it reaches. That is "
            "the alternative the room arrived with, which is why it is the one the report "
            "is measured against rather than a strawman of no program at all."
        ),
        equations=(
            "a*      = argmax_a  mean_draws cohort(a)",
            "Delta   = cohort(a*) - cohort(a_pilot)      paired inside each draw",
            "P       = Pr(Delta > 0)",
            "value   = Delta * VALUE_PER_POINT",
        ),
        detail={
            "recommended tutoring (USD)": f"{chosen_tutoring:,.0f}",
            "recommended messaging (USD)": f"{chosen_messaging:,.0f}",
            "weekly minutes": f"{chosen_tutoring / T.MINUTE_COST:.0f}",
            "messages a week": f"{chosen_messaging / T.MESSAGE_COST:.0f}",
            "students reached": f"{served:,.0f}",
            # Rounded: the optimiser returns 705.0000000000003, and a table of
            # protocol settings is not where a reader should meet the residue of
            # a line search.
            "allocate() at the budget": (
                ", ".join(f"{k} {v:,.0f}" for k, v in at_budget.doses.items())
                + f" ({at_budget.status})"
                if isinstance(at_budget, Allocation)
                else str(at_budget)
            ),
            "within 2% of the best (USD)": f"{flat.min():,.0f} to {flat.max():,.0f}",
        },
        assumptions=[VALUE_PER_POINT],
    )

    # -- what the trial found ----------------------------------------------------------
    builder.finding(
        "advantage",
        advantage,
        label="Cohort reading points over the pilot allocation",
        unit="points/year",
        precision=0,
        threshold=0.0,
        beneficial="higher",
        note=(
            "Paired inside each posterior draw. The comparison is against the pilot's 90 "
            "weekly minutes, not against no program: the question was never whether "
            "tutoring works."
        ),
        mass=MASS,
        source="paired posterior comparison over the allocation grid",
    )
    builder.finding(
        "advantage_value",
        value,
        label="Value of the difference at the department's own valuation",
        unit="USD",
        precision=0,
        threshold=0.0,
        beneficial="higher",
        note=(
            f"At {T.VALUE_PER_POINT:,.0f} USD a reading point. The recommendation does not "
            "turn on this figure — only the decision to fund anything at all does."
        ),
        mass=MASS,
        source="the same draws, priced",
    )

    builder.diagnostic(
        "probability_better",
        probability,
        label="Posterior probability the recommendation beats the pilot",
        precision=3,
    )
    builder.diagnostic(
        "students_reached", served, label="Students the recommendation reaches", precision=0
    )
    builder.diagnostic(
        "pilot_reached",
        pilot_served,
        label="Students the pilot allocation would reach",
        precision=0,
    )
    builder.diagnostic(
        "cost_per_student",
        float(T.cost_per_student(chosen_tutoring, chosen_messaging)),
        label="Cost of the recommended allocation per student",
        unit="USD",
        precision=0,
    )
    builder.diagnostic("schools", float(program.n_schools), label="Schools randomized", precision=0)
    builder.diagnostic(
        "draws", float(result.posterior.n_draws()), label="Posterior draws", precision=0
    )

    builder.assume(
        RANDOMIZED, ASSIGNED_NOT_DELIVERED, VALUE_PER_POINT, RESPONSE_FAMILY, GRADE_BANDS
    )
    builder.ledger_lines(
        [
            LedgerLine(
                kind="identification",
                statement=("Schools were randomized to allocations; the adjustment set is empty."),
                assumption=RANDOMIZED,
                detail={"route": "backdoor", "unit_of_assignment": "school"},
            ),
            LedgerLine(
                kind="estimand",
                statement=(
                    "The recommendation is over assigned allocations. Schools delivered "
                    "93% of assigned minutes, and that shortfall is inside the estimate "
                    "rather than corrected out of it."
                ),
                assumption=ASSIGNED_NOT_DELIVERED,
                detail={"version": "assigned"},
            ),
            LedgerLine(
                kind="interval",
                statement=(
                    f"Every figure and quantity at {MASS:.0%} over "
                    f"{result.posterior.n_draws():,} posterior draws."
                ),
            ),
            LedgerLine(
                kind="limitation",
                statement=(
                    "The grade-band breakdown the program office asked for is not "
                    "estimable from a school-randomized trial, and is not reported."
                ),
                assumption=GRADE_BANDS,
            ),
        ]
    )
    builder.remark(
        f"The recommendation is flat between {flat.min():,.0f} and {flat.max():,.0f} USD — "
        f"{flat.min() / T.MINUTE_COST:.0f} to {flat.max() / T.MINUTE_COST:.0f} weekly "
        "minutes are all within two per cent of the best. The decision that matters is not "
        "the exact figure, it is not funding 90 minutes.",
        (
            # Reported as a share, because the absolute figure alone invites the
            # reading that some specific number of points is at stake. What the
            # tipping point says is how much of the advantage would have to be
            # illusory, and at a half of the posterior that is all of it.
            f"For the pilot to be the better buy, the advantage would have to be "
            f"overstated by {tipping.bias / float(difference.mean()):.0%} of itself — "
            "that is, essentially the whole of it would have to be an artefact of the "
            "fit rather than a real difference between the two allocations."
            if tipping.flipped and tipping.bias is not None
            else "No overstatement anywhere on the grid tested makes the pilot the better buy."
        ),
        "The reason the recommendation is not 90 minutes is not that 90 minutes fails. It "
        "is that the response is steep at the bottom, the budget is fixed, and every extra "
        "dollar a student is a student not served.",
    )
    builder.provenance(
        seed=str(SEED),
        world="nbs/case-studies/tutoring/tutoring.py",
        estimator="spline response surface, PyMC via nutpie",
        fit_hash=spec.content_hash()[:12],
        interval_mass=f"{MASS:.0%}",
    )
    return builder.build()


def main() -> int:
    narrate = "--narrate" in sys.argv
    apa = "--apa" in sys.argv
    stem = "tutor60-apa" if apa else "tutor60-full"
    print("fitting the trial — this takes a minute\n")
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
        print(f"  {q.label:52s} {q.stated():40s} [{q.against_threshold()}]")
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
