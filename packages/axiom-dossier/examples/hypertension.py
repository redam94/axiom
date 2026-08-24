"""HYPER-3 as a full research report: the design, the analysis, and what stopped.

The case study under ``nbs/case-studies/hypertension/`` runs the whole of axiom
against one trial across six notebooks. This turns that into the document such a
trial actually has to produce — protocol and design up front, findings after,
and every number traced.

    python examples/hypertension.py                  # generated, no key needed
    python examples/hypertension.py --narrate        # prose through Gemini
    python examples/hypertension.py --apa            # an APA manuscript
    python examples/hypertension.py --apa --standalone   # HTML with plotly bundled

Writes ``out/hyper3-full.{html,pdf,pptx}``, or ``out/hyper3-apa.*`` with
``--apa``. The HTML carries the figures as live plotly charts; the PDF and PPTX
rasterise them.

What makes it worth having as an example rather than a fixture: nothing below is
typed in. The protocol constants come from ``hyper3.py``, the arm counts from the
realized randomization, the contrasts from ``identify.ols`` against the analysis
frames the trial actually produced, and the stopping event from walking the
realized path against the boundary. Change the seed and every number in the
report changes with it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "nbs" / "case-studies" / "hypertension"))

import hyper3 as h  # noqa: E402
from axiom.core import Assumption, LedgerLine, Unsupported  # noqa: E402
from axiom.identify import CausalGraph, identify, ols  # noqa: E402

from axiom_dossier import EvidenceBuilder, Narrator, build  # noqa: E402

MASS = 0.9
"""Every interval in this report is a 90 % one, and says so where it is printed."""

HARM_MARGIN = 2.0
"""mmHg. The protocol's clinical margin — the number the whole design is priced against."""


# -- the assumptions the trial rests on, stated once ------------------------------------

ITT_ONLY = Assumption(
    name="no_conditioning_on_post_randomization_variables",
    facet="method",
    statement=(
        "the analysis conditions on nothing measured after assignment; adherence is a "
        "mediator and dropout is a descendant of the outcome"
    ),
    challenged_by="any model that adjusts for adherence, or that analyses completers alone",
    state="satisfied",
)
RANDOMIZED = Assumption(
    name="assignment_is_random_within_stratum",
    facet="design",
    statement="arms were assigned by blocked randomization inside each age band",
    challenged_by="an imbalance in a pre-randomization covariate beyond chance",
    state="satisfied",
)
MISSINGNESS = Assumption(
    name="dropout_is_ignorable_given_baseline",
    facet="population",
    statement=(
        "units who left the study differ from those who stayed only through what was "
        "measured before randomization"
    ),
    challenged_by="a tipping-point analysis over plausible departures from ignorability",
    state="unverified",
)
POOLING = Assumption(
    name="the_pooled_contrast_answers_the_question",
    facet="population",
    statement=("a single number over all age bands describes the effect the decision is about"),
    challenged_by="heterogeneity across the pre-specified strata",
    state="violated",
)


def evidence():
    """Run the trial and collect everything the report is entitled to say."""
    trial = h.trial()
    counts = trial.arm_counts()
    retention = float(trial.completion().iloc[-1])

    # -- the analysis frame the primary endpoint is entitled to see ---------------------
    frame = h.look_frame(trial, h.TRIAL_WEEKS, endpoint="primary")

    def contrast(arm: str, stratum: str | None):
        rows = h.contrast_frame(frame, arm, stratum=stratum)
        return ols(rows, "change", "treated", h.ancova_covariates(stratum))

    pooled_40 = contrast("dose_40", None)
    oldest_40 = contrast("dose_40", "age_51_plus")
    youngest_40 = contrast("dose_40", "age_25_35")
    pooled_20 = contrast("dose_20", None)
    pooled_10 = contrast("dose_10", None)

    graph = CausalGraph.from_edges(
        "arm -> adherence, arm -> change, adherence -> change, "
        "baseline -> change, age -> change",
        name="HYPER-3",
    )
    verdict = identify(graph, "arm", "change")

    builder = EvidenceBuilder(
        "HYPER-3",
        "Which of three daily doses should go into phase III, and is any of them "
        "harming the people taking it?",
    )
    builder.verdict(verdict)

    # -- the experimental design, as method steps --------------------------------------
    allocation = ":".join(str(v) for v in h.ALLOCATION.values())
    builder.step(
        "protocol",
        "Protocol",
        what=(
            f"{trial.n_units} adults with elevated or stage-1 hypertension were "
            f"randomized {allocation} to standard of care or to one of three daily "
            "doses, stratified by age band."
        ),
        why=(
            "The question is which dose to take forward, so the design has to estimate "
            "three contrasts against a shared control rather than one against another."
        ),
        detail={
            "arms": ", ".join(h.ARMS),
            "allocation": allocation,
            "strata": ", ".join(h.STRATA),
            "units randomized": str(trial.n_units),
            "enrollment weeks": str(h.ENROLLMENT_WEEKS),
            "follow-up weeks": str(h.FOLLOW_UP_WEEKS),
            "primary endpoint week": str(h.PRIMARY_WEEK),
        },
        assumptions=[RANDOMIZED],
    )
    builder.step(
        "measurement",
        "Measurement",
        what=(
            "Seated systolic pressure was measured weekly. Both endpoints average the "
            f"readings inside a window — weeks {h.WINDOW['safety'][0]}–"
            f"{h.WINDOW['safety'][1]} for safety and {h.WINDOW['primary'][0]}–"
            f"{h.WINDOW['primary'][1]} for the primary — rather than reading a single "
            "visit."
        ),
        why=(
            "Averaging several correlated readings roughly halves the variance of a "
            "unit's endpoint, which is the difference between a safety boundary that "
            "can fire inside a stratum and one that cannot. It is what the weekly "
            "schedule buys."
        ),
        detail={
            "minimum visits in window": str(h.MIN_VISITS),
            "retention at final week": f"{retention:.1%}",
        },
    )
    builder.step(
        "analysis",
        "Analysis",
        what=(
            "Each dose was compared with control by ANCOVA on the change from "
            "baseline, adjusting for baseline pressure, and for age band when the "
            "contrast is pooled across bands."
        ),
        why=(
            "Randomization licenses the intent-to-treat contrast and nothing else. "
            "Adjusting for a pre-randomization covariate costs nothing and buys "
            "precision; adjusting for anything measured afterwards would forfeit the "
            "identification the randomization provided."
        ),
        detail={
            "estimator": "ANCOVA (OLS on change, baseline-adjusted)",
            "pooled covariates": ", ".join(h.ancova_covariates(None)),
            "within-stratum covariates": ", ".join(h.ancova_covariates("age_51_plus")),
            "interval mass": f"{MASS:.0%}",
        },
        assumptions=[ITT_ONLY, MISSINGNESS],
    )
    builder.step(
        "monitoring",
        "Safety monitoring",
        what=(
            "Twelve harm contrasts — each dose against control overall and inside each "
            "age band — were monitored at scheduled safety reviews against a "
            f"posterior-probability boundary with a {HARM_MARGIN:.0f} mmHg clinical "
            "margin."
        ),
        why=(
            "A dose that helps on average can still harm one band. Pre-specifying the "
            "bands as monitored contrasts is what makes that visible before the trial "
            "ends; no estimator applied at closeout would have found it earlier."
        ),
        detail={
            "monitored contrasts": "12",
            "clinical margin (mmHg)": f"{HARM_MARGIN:.0f}",
        },
    )

    # -- what the trial found ----------------------------------------------------------
    for key, est, label, note in (
        (
            "pooled_40",
            pooled_40,
            "40 mg vs control, pooled",
            "The number that should never be reported alone: it averages a band it "
            "helps with a band it harms.",
        ),
        ("pooled_20", pooled_20, "20 mg vs control, pooled", ""),
        ("pooled_10", pooled_10, "10 mg vs control, pooled", ""),
        (
            "oldest_40",
            oldest_40,
            "40 mg vs control, age 51+",
            "The contrast the pooled number hides, and the one the safety boundary "
            "was watching.",
        ),
        ("youngest_40", youngest_40, "40 mg vs control, age 25-35", ""),
    ):
        builder.finding(
            key,
            est,
            label=label,
            unit=h.OUTCOME_UNIT,
            precision=2,
            # a fall in pressure is the good direction, and zero is the value the
            # decision turns on for every one of these contrasts
            threshold=0.0,
            beneficial="lower",
            note=note,
            mass=MASS,
            source="identify.ols (ANCOVA)",
        )

    builder.diagnostic(
        "retention",
        retention,
        label="Units still on study at the final week",
        precision=3,
    )
    builder.diagnostic(
        "analysed",
        float(len(frame)),
        label="Units contributing to the primary endpoint",
        precision=0,
    )
    for stratum in h.STRATA:
        builder.diagnostic(
            f"n_{stratum}",
            float(int(counts.loc[stratum].sum())),
            label=f"Units randomized in band {stratum.replace('_', ' ')}",
            precision=0,
        )

    builder.assume(RANDOMIZED, ITT_ONLY, MISSINGNESS, POOLING)
    builder.ledger_lines(
        [
            LedgerLine(
                kind="assumption",
                statement=(
                    "Adherence was not adjusted for: it lies on the path from arm to "
                    "outcome, and conditioning on it would break the randomization."
                ),
                assumption=ITT_ONLY,
            ),
            LedgerLine(
                kind="assumption",
                statement=(
                    "The pooled 40 mg contrast is reported alongside the per-band "
                    "contrasts, never instead of them."
                ),
                assumption=POOLING,
            ),
        ]
    )
    builder.provenance(
        seed=str(20260821),
        world="nbs/case-studies/hypertension/hyper3.py",
        endpoint=f"mean change, weeks {h.WINDOW['primary'][0]}-{h.WINDOW['primary'][1]}",
        interval_mass=f"{MASS:.0%}",
    )
    return builder.build()


def main() -> int:
    narrate = "--narrate" in sys.argv
    apa = "--apa" in sys.argv
    stem = "hyper3-apa" if apa else "hyper3-full"
    ev = evidence()
    built = build(
        ev,
        narrator=Narrator() if narrate else None,
        style="apa" if apa else "journal",
        verbosity="full",
        # APA puts the author block under the title and gathers the exhibits
        # after the text; the journal style embeds them with the prose.
        authors=("Matthew Reda",) if apa else (),
        affiliation="axiom" if apa else "",
    )

    print(f"evidence hash : {ev.content_hash()}")
    print(f"document      : {built.summary()}")
    print(f"findings      : {len(ev.findings)}   diagnostics: {len(ev.diagnostics)}")
    print(f"unresolved    : {', '.join(ev.unresolved()) or 'none'}")
    print()
    for q in ev.findings:
        print(f"  {q.label:34s} {q.stated():38s} [{q.against_threshold()}]")
    print()
    for n in built.narrations:
        print(f"  {n.summary()}")
    if built.rejected():
        print("\n  ^ kept the generated draft for those; nothing wrong reached the page.")

    out = Path(__file__).parent / "out"
    print()
    for fmt in ("html", "pdf", "pptx"):
        # The committed samples link the plotting library rather than bundling
        # it: self-contained is the right default for a report you email, and
        # the wrong one for a file that lives in a repository forever at five
        # megabytes a copy. `--standalone` writes the bundled version.
        result = built.write(
            str(out / f"{stem}.{fmt}"),
            inline_plotly="--standalone" in sys.argv,
        )
        if isinstance(result, Unsupported):
            print(f"  {fmt:5s} skipped — {result.reason}")
        else:
            print(f"  {fmt:5s} {result} ({Path(result).stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
