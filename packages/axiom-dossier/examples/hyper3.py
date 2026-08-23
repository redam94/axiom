"""A full research report from one axiom analysis, in all three formats.

    python examples/hyper3.py                       # readout, no key needed
    python examples/hyper3.py --journal --full      # a paper, at length
    python examples/hyper3.py --journal --narrate   # and narrated by Gemini

Writes ``out/hyper3.{html,pdf,pptx}`` and prints what happened to each section,
including any narration rejected for containing a number the evidence does not
license or a claim it does not make.
"""

from __future__ import annotations

import sys
from pathlib import Path

from axiom.core import Assumption, Interval, LedgerLine, Unsupported
from axiom.identify import CausalGraph, identify

from axiom_dossier import EvidenceBuilder, Narrator, build

NO_CONFOUNDING = Assumption(
    name="no_unmeasured_confounding",
    facet="population",
    statement="age is the only common cause of dose and systolic pressure",
    challenged_by="a sensitivity analysis at plausible confounder strength",
    state="unverified",
)


def evidence():
    """Everything the report is allowed to say, collected from axiom's own results."""
    graph = CausalGraph.from_edges("age -> dose, age -> pressure, dose -> pressure", name="HYPER-3")
    return (
        EvidenceBuilder("HYPER-3", "Does the 40 mg arm lower systolic pressure?")
        # the identification verdict carries the route and the adjustment set,
        # so the methods section does not have to be told either
        .verdict(identify(graph, "dose", "pressure"))
        .step(
            "design",
            "Design",
            what="A two-arm parallel trial with 300 participants per arm.",
            why="Powered for a 5 mmHg difference at 80 % power.",
            detail={"arms": "2", "per arm": "300"},
        )
        .step(
            "estimation",
            "Estimation",
            what="Ordinary least squares with a robust variance estimator.",
            why="The response is linear in dose over the range studied.",
        )
        .finding(
            "contrast",
            Interval(lower=-16.7, upper=-8.1, definition="eti", mass=0.9),
            label="40 mg vs control",
            unit="mmHg",
            source="estimands.realize",
            precision=1,
            # the value the decision turns on: without it a conclusions section
            # can report the estimate but cannot say whether it settles anything
            threshold=-5.0,
            beneficial="lower",
        )
        .diagnostic("coverage", 0.94, label="Interval coverage in simulation")
        .assume(NO_CONFOUNDING)
        .ledger_lines(
            [
                LedgerLine(
                    kind="assumption",
                    statement="Age is the only measured confounder of dose and pressure.",
                    assumption=NO_CONFOUNDING,
                )
            ]
        )
        .provenance(seed="0", axiom_version="1.0.0")
        .build()
    )


def main() -> int:
    narrate = "--narrate" in sys.argv
    style = "journal" if "--journal" in sys.argv else "plain"
    verbosity = "full" if "--full" in sys.argv else "standard"
    ev = evidence()
    built = build(
        ev,
        narrator=Narrator() if narrate else None,
        style=style,
        verbosity=verbosity,
    )

    print(f"evidence hash : {ev.content_hash()}")
    print(f"document      : {built.summary()}")
    print(f"sections      : {', '.join(s.title for s in built.report.sections)}")
    for n in built.narrations:
        print(f"  {n.summary()}")
    if built.rejected():
        print("\n  ^ those sections kept their generated text; nothing wrong reached the page.")

    out = Path(__file__).parent / "out"
    print()
    for fmt in ("html", "pdf", "pptx"):
        result = built.write(str(out / f"hyper3.{fmt}"))
        if isinstance(result, Unsupported):
            print(f"  {fmt:5s} skipped — {result.reason}")
        else:
            print(f"  {fmt:5s} {result} ({Path(result).stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
