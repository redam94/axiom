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

from axiom_dossier.claims import CLAIM_WORDS, Claim, licensed_claims
from axiom_dossier.claims import unlicensed as unlicensed_claims
from axiom_dossier.dossier import (
    DEFAULT_SECTIONS,
    JOURNAL_SECTIONS,
    Dossier,
    Style,
    build,
    context_for,
)
from axiom_dossier.evidence import (
    Evidence,
    EvidenceBuilder,
    MethodStep,
    Quantity,
    quantity_from,
)
from axiom_dossier.interpret import conclusions_section, discussion_section, reading_of
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
    SECTION_INSTRUCTION,
    Narration,
    Narrator,
    evidence_brief,
    narrate_text,
)
from axiom_dossier.numbers import licensed_numbers, literals, unverified
from axiom_dossier.sections import (
    VERBOSITY,
    Verbosity,
    assumption_rows,
    diagnostics_section,
    limitations_section,
    methods_section,
    provenance_section,
    results_section,
    standing_assumptions,
)

__version__ = "0.1.0"

__all__ = [
    "CLAIM_WORDS",
    "Claim",
    "DEFAULT_SECTIONS",
    "Dossier",
    "Evidence",
    "EvidenceBuilder",
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
    "SECTION_INSTRUCTION",
    "Style",
    "VERBOSITY",
    "Verbosity",
    "abstract_section",
    "assumption_rows",
    "build",
    "conclusions_section",
    "context_for",
    "diagnostics_section",
    "discussion_section",
    "evidence_brief",
    "introduction_section",
    "licensed_claims",
    "licensed_numbers",
    "limitations_section",
    "literals",
    "methods_section",
    "narrate_text",
    "numbered",
    "provenance_section",
    "quantity_from",
    "reading_of",
    "results_section",
    "standing_assumptions",
    "unlicensed_claims",
    "unverified",
]
