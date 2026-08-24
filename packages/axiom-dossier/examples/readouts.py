"""A full readout for every example in the axiom repository.

Runs each of the twelve scripts under ``examples/`` with its structured-record
capture switched on, turns each record into an ``Evidence``, and writes one
journal-shaped report per example.

    python examples/readouts.py                 # all twelve
    python examples/readouts.py 07 02           # just those
    python examples/readouts.py --narrate       # prose through Gemini
    python examples/readouts.py --standalone    # HTML with plotly bundled

Writes ``out/readouts/<stem>.{pdf,html}`` and a one-line summary per example.

Each example is run in a subprocess exactly as a reader would run it, so a
report can never describe a run that did not happen: if the script fails, the
report is not written and the failure is printed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from axiom.core import Unsupported

from axiom_dossier import JOURNAL_THEME, Narrator, build, evidence_from_record, exhibits_from_record

REPO = Path(__file__).resolve().parents[3]
EXAMPLES = REPO / "examples"
OUT = Path(__file__).parent / "out" / "readouts"


def scripts(selectors: list[str]) -> list[Path]:
    """The example scripts to run, filtered by any numeric prefixes given."""
    found = sorted(p for p in EXAMPLES.glob("*.py") if p.stem[:1].isdigit())
    if not selectors:
        return found
    wanted = tuple(s.lstrip("0") or "0" for s in selectors)
    return [p for p in found if p.stem.split("_")[0].lstrip("0") in wanted]


def record_for(script: Path) -> dict | None:
    """Run one example and return the record it wrote, or None if it failed."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "record.json"
        env = {**os.environ, "AXIOM_WALKTHROUGH_JSON": str(target)}
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(EXAMPLES),
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(f"  ! {script.stem} exited {proc.returncode}")
            print("   ", proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "")
            return None
        if not target.exists():
            print(f"  ! {script.stem} wrote no record")
            return None
        return json.loads(target.read_text())


def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    narrate = "--narrate" in sys.argv
    standalone = "--standalone" in sys.argv
    narrator = Narrator() if narrate else None

    chosen = scripts(argv)
    if not chosen:
        print(f"no examples matched {argv}")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{len(chosen)} example(s) -> {OUT}\n")

    failures = 0
    for script in chosen:
        record = record_for(script)
        if record is None:
            failures += 1
            continue
        evidence = evidence_from_record(record)
        # The example's own tables and charts are exhibits of the steps that
        # made them, so they go in before the sections are built rather than
        # into the context afterwards: a step names its figure, and `build`
        # keeps the naming only where there is data behind it.
        built = build(
            evidence,
            narrator=narrator,
            style="journal",
            verbosity="full",
            extra_exhibits=exhibits_from_record(record, theme=JOURNAL_THEME),
        )

        missing = built.missing()
        if missing:
            print(f"  ! {script.stem}: template wants {missing}")
            failures += 1
            continue

        written = []
        for fmt in ("pdf", "html"):
            # The committed HTML links plotly rather than bundling it: twelve
            # reports with the library inside are 58 MB of repository, and these
            # are read from a checkout. `--standalone` writes the emailable one.
            result = built.write(
                str(OUT / f"{script.stem}.{fmt}"),
                inline_plotly=standalone,
            )
            written.append(fmt if not isinstance(result, Unsupported) else f"{fmt}:skipped")
        rejected = len(built.rejected())
        note = f", {rejected} narration(s) rejected" if rejected else ""
        exhibits = sum(len(step.exhibits) for step in evidence.steps)
        print(
            f"  {script.stem:38s} {len(evidence.steps)} steps, "
            f"{exhibits} exhibits, {len(evidence.remarks)} remarks -> "
            f"{', '.join(written)}{note}"
        )

    print(f"\n{len(chosen) - failures}/{len(chosen)} written")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
