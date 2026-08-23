"""Discussion and Conclusions — the sections where a report is most likely to lie.

Every other generated section restates something. These two *interpret*, which
is a different and more dangerous job: the reader wants to be told what the
result means, and "what it means" is exactly the sentence that gets written past
the evidence.

So interpretation here is mechanical. What a finding means is derived from three
things the record already holds — where its interval sits relative to the value
the decision turns on, whether the effect was identified at all, and which
assumptions are still standing — and the sentences are assembled from those.
There is no step at which a judgement is invented.

Three rules the generated text keeps:

- **An unsettled interval is reported as unsettled**, never as a null result. An
  interval spanning the threshold means the data does not say which side the
  effect is on; that is not the same as saying there is no effect, and the
  difference is the most commonly destroyed distinction in applied work.
- **A conclusion inherits the identification verdict.** If the effect is not
  identified, the conclusion says what the number is *associationally* and
  refuses the causal reading, rather than hedging with an adverb.
- **Nothing is called significant, robust or proven.** Those words are checked
  by ``claims.unlicensed`` for narration; the generated text does not use them
  in the first place.
"""

from __future__ import annotations

from axiom.report import Heading, Paragraph, Section

from axiom_dossier.evidence import Evidence, Quantity
from axiom_dossier.sections import VERBOSITY, Verbosity, literal

__all__ = ["conclusions_section", "contested", "discussion_section", "reading_of"]


def _direction(q: Quantity) -> str:
    return "an increase" if q.value > 0 else "a decrease" if q.value < 0 else "no change"


def reading_of(q: Quantity, *, causal: bool) -> str:
    """One finding, read against its threshold. The whole of the interpretation.

    ``causal`` comes from the identification verdict, and changes the verb: an
    identified effect *is* a change produced by the treatment, an unidentified
    one is a difference observed between groups.
    """
    # Every branch opens with this, so the causal framing travels with the
    # sentence rather than living only in the section's preamble. A reader who
    # quotes one line out of a discussion should not lose the qualifier.
    subject = "the effect" if causal else "the observed difference"
    unit = f" {q.unit}" if q.unit else ""
    side = q.against_threshold()

    if side in ("no threshold", "no interval"):
        return (
            f"{q.label}: {subject} is {q.stated()}. No decision threshold was recorded "
            "for this quantity, so the report states the estimate and stops short of "
            "saying whether it is large enough to act on."
        )

    threshold = f"{q.threshold:.{q.precision}f}{unit}"
    if side == "spans":
        return (
            f"{q.label}: the interval {q.stated()} for {subject} contains the threshold "
            f"of {threshold}, so these data do not settle which side of it "
            f"{subject} falls. That is an unsettled question rather than a finding of "
            "no effect: an interval this wide is consistent with a difference worth "
            "acting on and with one that is not."
        )

    good = q.is_beneficial()
    where = "entirely below" if side == "below" else "entirely above"
    sentence = (
        f"{q.label}: the interval {q.stated()} for {subject} lies {where} the threshold "
        f"of {threshold}, so on these data the question is settled in that direction."
    )
    if good is True:
        return sentence + " That is the favourable side of the threshold."
    if good is False:
        return sentence + " That is the unfavourable side of the threshold."
    return sentence + f" The estimate is {_direction(q)} relative to no effect."


def _causal(evidence: Evidence) -> bool:
    return evidence.verdict is not None and evidence.verdict.status == "identified"


def discussion_section(
    evidence: Evidence,
    *,
    title: str = "Discussion",
    verbosity: Verbosity = "standard",
) -> Section:
    """What each finding means, what the identification licenses, what is still open."""
    causal = _causal(evidence)
    detail = VERBOSITY[verbosity]
    blocks: list[object] = []

    if not evidence.findings:
        return Section(
            title=title,
            blocks=(Paragraph(text="There are no findings to interpret."),),
        )

    if evidence.verdict is None:
        blocks.append(
            Paragraph(
                text=(
                    "No identification verdict was recorded, so the quantities below are "
                    "read as descriptions of the data rather than as effects of the "
                    "treatment."
                )
            )
        )
    elif causal:
        blocks.append(
            Paragraph(
                text=(
                    f"The effect is identified via the {evidence.verdict.route or 'recorded'} "
                    "route, so the quantities below can be read as effects of the treatment "
                    "rather than as associations — under the assumptions listed in the "
                    "methods, and no further."
                )
            )
        )
    else:
        blocks.append(
            Paragraph(
                text=(
                    f"**The effect is {evidence.verdict.status}, not identified.** "
                    f"{evidence.verdict.reason} The quantities below therefore describe "
                    "a difference between groups. Reading them as the effect of the "
                    "treatment requires an assumption this analysis does not supply."
                )
            )
        )

    if detail["per_finding_heading"]:
        for q in evidence.findings:
            blocks.append(Heading(text=q.label, level=3))
            blocks.append(Paragraph(text=reading_of(q, causal=causal)))
            if q.note and detail["include_notes"]:
                blocks.append(Paragraph(text=q.note, emphasis=True))
    else:
        for q in evidence.findings:
            blocks.append(Paragraph(text=reading_of(q, causal=causal)))

    unsettled = [q for q in evidence.findings if q.against_threshold() == "spans"]
    if unsettled and detail["include_next_steps"]:
        names = ", ".join(q.label for q in unsettled)
        blocks.append(
            Paragraph(
                text=(
                    f"{names}: the interval is too wide to settle the question. What "
                    "narrows it is more data or a design that estimates the quantity more "
                    "precisely — not a different summary of the data already collected."
                )
            )
        )

    standing = evidence.unresolved()
    if standing and detail["include_assumption_recap"]:
        blocks.append(
            Paragraph(
                text=(
                    f"Every reading above is conditional on {len(standing)} unresolved "
                    f"assumption(s): {', '.join(n.replace('_', ' ') for n in standing)}. "
                    "The limitations section states what each one claims and what would "
                    "challenge it."
                )
            )
        )
    return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]


def contested(evidence: Evidence) -> tuple[tuple[Quantity, ...], tuple[Quantity, ...]]:
    """Findings that settle favourably, and findings that settle unfavourably.

    Both non-empty means the record contains a disagreement, and a conclusion
    drawn from whichever finding happens to be first would be misleading. This
    is not a hypothetical: a dose can lower pressure on average and raise it in
    one age band, and the average is the number that gets quoted.
    """
    good = tuple(q for q in evidence.findings if q.is_beneficial() is True)
    bad = tuple(q for q in evidence.findings if q.is_beneficial() is False)
    return good, bad


def conclusions_section(
    evidence: Evidence,
    *,
    title: str = "Conclusions",
    verbosity: Verbosity = "standard",
) -> Section:
    """The short answer to the question that was asked, and its scope.

    Deliberately terse at every verbosity: a conclusions section that runs long
    is one that has started arguing. It answers the question, names what the
    answer is conditional on, and stops.

    The one thing it will not do is answer from a single finding while another
    recorded finding points the other way. When the record disagrees with
    itself, saying so *is* the conclusion.
    """
    causal = _causal(evidence)
    blocks: list[object] = []

    if evidence.question:
        blocks.append(Paragraph(text=f"**{literal(evidence.question)}**"))

    if not evidence.findings:
        blocks.append(Paragraph(text="No finding was recorded, so no conclusion follows."))
        return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]

    favourable, unfavourable = contested(evidence)
    if favourable and unfavourable:
        harmed = "; ".join(f"{q.label} at {q.stated()}" for q in unfavourable)
        helped = (
            "; ".join(f"{q.label} at {q.stated()}" for q in favourable)
            if len(favourable) <= 2
            else (
                f"{len(favourable)} other findings "
                f"({', '.join(q.label for q in favourable)})"
            )
        )
        blocks.append(
            Paragraph(
                text=(
                    f"**Not with one number.** {harmed} sits on the unfavourable side "
                    f"of the threshold. {helped} sit on the favourable side. Any single "
                    "summary averages a harm with a benefit, and the average is the "
                    "number that gets quoted — so the disaggregated findings are the "
                    "result, and the pooled figure is not a substitute for them."
                )
            )
        )
        scope: list[str] = []
        if causal:
            scope.append(
                "Each is a causal reading, licensed by the "
                f"{evidence.verdict.route or 'recorded'} route."  # type: ignore[union-attr]
            )
        standing = evidence.unresolved()
        if standing:
            scope.append(
                f"All of them hold under {len(standing)} unresolved assumption(s) "
                f"({', '.join(n.replace('_', ' ') for n in standing)}), and not otherwise."
            )
        if scope:
            blocks.append(Paragraph(text=" ".join(scope)))
        return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]

    lead = evidence.findings[0]
    side = lead.against_threshold()
    if side == "spans":
        answer = (
            f"Not settled by these data. {lead.label} is {lead.stated()}, an interval that "
            f"contains the threshold of {lead.threshold:.{lead.precision}f}"
            f"{' ' + lead.unit if lead.unit else ''}."
        )
    elif side in ("below", "above"):
        good = lead.is_beneficial()
        verdict_word = {True: "Yes", False: "No", None: "Settled in one direction"}[good]
        answer = (
            f"{verdict_word}. {lead.label} is {lead.stated()}, which lies entirely "
            f"{'below' if side == 'below' else 'above'} the threshold of "
            f"{lead.threshold:.{lead.precision}f}{' ' + lead.unit if lead.unit else ''}."
        )
    else:
        answer = f"{lead.label} is {lead.stated()}. No threshold was recorded to judge it against."
    blocks.append(Paragraph(text=answer))

    scope = []
    if causal:
        scope.append(
            f"This is a causal reading, licensed by the "
            f"{evidence.verdict.route or 'recorded'} route."  # type: ignore[union-attr]
        )
    elif evidence.verdict is not None:
        scope.append(f"This is not a causal reading: the effect is {evidence.verdict.status}.")
    standing = evidence.unresolved()
    if standing:
        scope.append(
            f"It holds under {len(standing)} unresolved assumption(s) "
            f"({', '.join(n.replace('_', ' ') for n in standing)}), and not otherwise."
        )
    if scope:
        blocks.append(Paragraph(text=" ".join(scope)))
    return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]
