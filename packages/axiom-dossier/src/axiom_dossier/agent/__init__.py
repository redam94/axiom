"""A multi-stage agent that reads a notebook series and reports on it.

The problem this solves is specific. A case study like
``nbs/case-studies/hypertension`` is six notebooks, seventy-two prose cells and
ninety-two code cells, and **the repository stores none of its outputs**. Every
number in it exists only while the code is running. So a report on it cannot be
assembled by parsing anything: the analysis has to be executed, and then the
prose and the objects have to be put back together.

That is what the graph does — read, run, label, find the gaps, close them,
assemble, verify — and it is why the labelling and gap stages want a model. A
variable called ``itt`` is not a thing to put in a report; the sentence above it
in the notebook says what it is. A model reads that sentence.

What the model never does is compute. It names a variable, notices that a
picture the prose describes was never drawn, and writes the code that draws it.
The numbers come from executing the analysis, and everything that reaches prose
still passes the numeric and claim gates.

    from axiom_dossier.agent import run_pipeline
    from axiom_dossier import Gemini, build

    run = run_pipeline(
        "nbs/case-studies/hypertension",
        title="HYPER-3",
        question="Which dose goes into phase III?",
        model=Gemini(),
        allow_execution=True,
    )
    print(run.summary())
    built = build(run.evidence, style="apa")

``allow_execution`` is explicit because the fill stage runs model-written Python
in this process. That is right for a developer tool pointed at a repository you
already trust, and wrong for anything else.
"""

from __future__ import annotations

from axiom_dossier.agent.graph import (
    MAX_REPAIRS,
    build_graph,
    default_models,
    run_pipeline,
)
from axiom_dossier.agent.notebooks import (
    Execution,
    Notebook,
    Passage,
    execute,
    harvest,
    read_notebook,
    read_series,
)
from axiom_dossier.agent.state import DossierState, Gap, Harvested, Run
from axiom_dossier.agent.tools import (
    Executed,
    ExecutionRefused,
    describe_namespace,
    run_python,
)

__all__ = [
    "MAX_REPAIRS",
    "DossierState",
    "Executed",
    "Execution",
    "ExecutionRefused",
    "Gap",
    "Harvested",
    "Notebook",
    "Passage",
    "Run",
    "build_graph",
    "default_models",
    "describe_namespace",
    "execute",
    "harvest",
    "read_notebook",
    "read_series",
    "run_pipeline",
    "run_python",
]
