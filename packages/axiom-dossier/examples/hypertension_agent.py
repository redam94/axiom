"""The whole HYPER-3 case study, read and reported by the agent.

    python examples/hypertension_agent.py                 # deterministic stages only
    python examples/hypertension_agent.py --model         # + labelling and gap-finding
    python examples/hypertension_agent.py --model --execute   # + code that draws them

Six notebooks, ninety-two code cells, and no stored outputs — so the pipeline
executes the series in one namespace and reports on what it leaves behind, then
consolidates the planning and outcome notes beside it.

Writes ``out/hyper3-agent.{pdf,html}``.

What each flag buys, and what it costs:

* no flag — reads, runs, harvests and assembles. Every number is real; the
  labels are variable names, because naming needs something that read the prose.
* ``--model`` — labels each harvested object from the markdown that introduced
  it, and lists the figures the prose describes that the run never drew.
* ``--execute`` — lets the fill stage run the Python it wrote to draw them. This
  executes model-written code in this process, which is why it is a separate
  flag from ``--model``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from axiom.core import Unsupported

from axiom_dossier import Gemini, Narrator, build
from axiom_dossier.agent import run_pipeline

REPO = Path(__file__).resolve().parents[3]
SERIES = REPO / "nbs" / "case-studies" / "hypertension"
NOTES = (
    REPO / "docs" / "notes" / "0004-case-study-hypertension.md",
    REPO / "docs" / "notes" / "0005-basis-response-families.md",
    SERIES / "README.md",
)

QUESTION = (
    "Which of three daily doses should go into phase III, and is any of them "
    "harming the people taking it?"
)


def main() -> int:
    use_model = "--model" in sys.argv
    execute = "--execute" in sys.argv
    model = Gemini() if use_model else None

    notes = {p.name: p.read_text() for p in NOTES if p.exists()}
    print(f"series : {SERIES.relative_to(REPO)}")
    print(f"notes  : {', '.join(notes) or 'none found'}")
    print(f"model  : {model.name if model else 'none (deterministic stages only)'}")
    print(f"execute: {execute}\n")

    run = run_pipeline(
        SERIES,
        title="HYPER-3",
        question=QUESTION,
        notes=notes,
        model=model,
        allow_execution=execute,
    )

    print()
    for line in run.transcript:
        print("  " + line)
    print(f"\n{run.summary()}")

    if run.evidence is None:
        print("nothing was assembled")
        return 1

    print("\nfindings the series produced:")
    for q in run.evidence.findings:
        print(f"  {q.label[:52]:54s} {q.stated()}")

    if run.gaps:
        print("\ngaps the prose implied:")
        for gap in run.gaps:
            mark = "drawn" if gap.resolved else (gap.error[:40] or "left open")
            print(f"  [{mark}] {gap.what[:70]}")

    # every figure the notebooks drew, plus anything the fill stage added
    extra = {
        key: (figure, f"From {key.split('_')[0]}: {key.split('_', 1)[-1]}.")
        for key, figure in run.figures.items()
    }
    built = build(
        run.evidence,
        narrator=Narrator() if use_model else None,
        style="apa",
        verbosity="full",
        authors=("Matthew Reda",),
        affiliation="axiom",
        extra_figures=extra,
    )
    print(f"\ndocument: {built.summary()}")
    print(f"missing : {built.missing()}")

    out = Path(__file__).parent / "out"
    for fmt in ("pdf", "html"):
        result = built.write(str(out / f"hyper3-agent.{fmt}"), inline_plotly=False)
        if isinstance(result, Unsupported):
            print(f"  {fmt:4s} skipped — {result.reason}")
        else:
            print(f"  {fmt:4s} {result} ({Path(result).stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
