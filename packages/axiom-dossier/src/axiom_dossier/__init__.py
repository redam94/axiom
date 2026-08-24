"""Shareable research reports from axiom results.

axiom's charter says a report generator is someone else's job — it returns typed
results and stops. This is that someone else. It depends on axiom; axiom does
not know it exists.

What it adds over ``axiom.report``, which already renders HTML, PPTX and PDF
from a template, is the two things that template still needed a person for:

**The sections write themselves.** ``methods_section`` is generated from the
``Verdict`` and the ``Assumption`` s that licensed the analysis, so it cannot
describe a route nobody took; ``limitations_section`` is generated from the
assumptions still unresolved, so it cannot be shorter than the truth. None of
this needs a language model, a key, or a network.

**A model may improve the prose, but never the numbers.** ``Narrator`` rewrites
the generated draft through Gemini, and the result is then checked back against
the evidence: every numeral must trace to a recorded quantity. One that does not
gets the narration rejected and the correct draft kept. The distinction is
recorded in the document's own provenance section, so a reader who does not
trust a model can see exactly which paragraphs it touched.

    from axiom_dossier import EvidenceBuilder, Narrator, build

    evidence = (
        EvidenceBuilder("HYPER-3", "Does 40 mg lower systolic pressure?")
        .verdict(verdict)
        .finding("contrast", result, label="40 mg vs control", unit="mmHg")
        .build()
    )
    built = build(evidence, narrator=Narrator())   # or narrator=None, offline
    built.write("hyper3.pdf")
"""

from __future__ import annotations

from axiom_dossier.apa import (
    APA_SECTIONS,
    APA_THEME,
    apa_caption,
    exhibit_sections,
    running_head,
    title_block,
    title_page_section,
)
from axiom_dossier.charts import CHART_KINDS, chart_figure
from axiom_dossier.claims import CLAIM_WORDS, Claim, licensed_claims
from axiom_dossier.claims import unlicensed as unlicensed_claims
from axiom_dossier.dossier import (
    DEFAULT_SECTIONS,
    JOURNAL_SECTIONS,
    Dossier,
    Exhibits,
    Style,
    build,
    context_for,
)
from axiom_dossier.evidence import (
    Evidence,
    EvidenceBuilder,
    Exhibit,
    MethodStep,
    Quantity,
    quantity_from,
)
from axiom_dossier.figures import diagnostics_plot, figures_for, findings_plot
from axiom_dossier.interpret import (
    conclusions_section,
    contested,
    discussion_section,
    pivotal,
    reading_of,
)
from axiom_dossier.journal import (
    JOURNAL_THEME,
    abstract_section,
    introduction_section,
    numbered,
)
from axiom_dossier.language import (
    LIGHT_MODEL,
    PROSE_MODEL,
    Gemini,
    LanguageModel,
    Offline,
)
from axiom_dossier.narrate import (
    ABSTRACT_LICENCE,
    ESTABLISHED,
    FULL_LICENCE,
    LICENCE,
    SECTION_INSTRUCTION,
    Licence,
    Narration,
    Narrator,
    established_note,
    evidence_brief,
    exhibit_note,
    narrate_text,
)
from axiom_dossier.numbers import licensed_numbers, literals, unverified
from axiom_dossier.sections import (
    VERBOSITY,
    Verbosity,
    assumption_rows,
    diagnostics_section,
    limitations_section,
    literal,
    methods_section,
    plural,
    provenance_section,
    readout_text,
    remarks_section,
    results_section,
    sentence,
    standing_assumptions,
)
from axiom_dossier.tables import design_rows, diagnostics_rows, findings_rows, tables_for
from axiom_dossier.walkthrough import (
    evidence_from_record,
    exhibits_from_record,
    figures_from_record,
    tables_from_record,
)

#: The notebook-reading pipeline is a subpackage rather than a re-export:
#: it needs langgraph, and importing axiom_dossier must not.
#:
#:     from axiom_dossier.agent import run_pipeline
__version__ = "0.2.0"

__all__ = [
    "title_block",
    "title_page_section",
    "tables_for",
    "running_head",
    "findings_rows",
    "findings_plot",
    "figures_for",
    "exhibit_sections",
    "diagnostics_rows",
    "diagnostics_plot",
    "design_rows",
    "apa_caption",
    "Exhibits",
    "APA_THEME",
    "APA_SECTIONS",
    "CHART_KINDS",
    "CLAIM_WORDS",
    "Claim",
    "DEFAULT_SECTIONS",
    "Dossier",
    "Evidence",
    "EvidenceBuilder",
    "Exhibit",
    "Gemini",
    "JOURNAL_SECTIONS",
    "JOURNAL_THEME",
    "LIGHT_MODEL",
    "LanguageModel",
    "MethodStep",
    "Narration",
    "Narrator",
    "Offline",
    "PROSE_MODEL",
    "Quantity",
    "ABSTRACT_LICENCE",
    "ESTABLISHED",
    "FULL_LICENCE",
    "LICENCE",
    "Licence",
    "SECTION_INSTRUCTION",
    "Style",
    "VERBOSITY",
    "Verbosity",
    "abstract_section",
    "assumption_rows",
    "build",
    "chart_figure",
    "conclusions_section",
    "context_for",
    "contested",
    "diagnostics_section",
    "discussion_section",
    "established_note",
    "evidence_brief",
    "exhibit_note",
    "evidence_from_record",
    "exhibits_from_record",
    "figures_from_record",
    "introduction_section",
    "licensed_claims",
    "licensed_numbers",
    "limitations_section",
    "literal",
    "plural",
    "readout_text",
    "sentence",
    "literals",
    "methods_section",
    "narrate_text",
    "pivotal",
    "numbered",
    "provenance_section",
    "quantity_from",
    "reading_of",
    "remarks_section",
    "results_section",
    "standing_assumptions",
    "tables_from_record",
    "unlicensed_claims",
    "unverified",
]
