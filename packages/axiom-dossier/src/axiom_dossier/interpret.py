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
from axiom_dossier.sections import VERBOSITY, Verbosity, literal, plural, sentence

__all__ = [
    "conclusions_section",
    "contested",
    "discussion_section",
    "pivotal",
    "reading_of",
]


def _direction(q: Quantity) -> str:
    return "an increase" if q.value > 0 else "a decrease" if q.value < 0 else "no change"


def reading_of(q: Quantity, *, causal: bool, stats: bool = True) -> str:
    """One finding, read against its threshold. The whole of the interpretation.

    ``causal`` comes from the identification verdict, and changes the verb: an
    identified effect *is* a change produced by the treatment, an unidentified
    one is a difference observed between groups.

    ``stats`` decides whether the reading carries the numbers. It does where the
    numbers are the point, and it does not in the discussion, which the UCSD
    guide asks to "repeat the results section in simpler terms and without
    referring to stats". That is not a stylistic preference: a discussion that
    reprints every interval is the results table a second time, and the second
    copy is where a reader stops reading.
    """
    # Every branch opens with this, so the causal framing travels with the
    # sentence rather than living only in the section's preamble. A reader who
    # quotes one line out of a discussion should not lose the qualifier.
    # "the effect" against "the estimate": the distinction that has to survive is
    # causal against not-causal, and it is carried by the noun. It was carried by
    # "the observed difference", which says the same thing and additionally says
    # the quantity is a contrast between groups -- true of a trial arm and false
    # of a physical constant, which is how a bound on a nuclear radius came to be
    # reported as an observed difference.
    subject = "the effect" if causal else "the estimate"
    unit = f" {q.unit}" if q.unit else ""
    side = q.against_threshold()

    if side in ("no threshold", "no interval"):
        estimate = f" is {q.stated()}" if stats else " is reported"
        return (
            f"{q.label}: {subject}{estimate}. No decision threshold was recorded "
            "for this quantity, so the report states the estimate and stops short of "
            "saying whether it is large enough to act on."
        )

    interval = f" {q.stated()}" if stats else ""
    threshold = f" of {q.threshold:.{q.precision}f}{unit}" if stats else ""
    if side == "spans":
        return (
            f"{q.label}: the interval{interval} for {subject} contains the "
            f"threshold{threshold}, so these data do not settle which side of it "
            f"{subject} falls. That is an unsettled question rather than a finding of "
            "no effect: an interval this wide is consistent with a difference worth "
            "acting on and with one that is not."
        )

    good = q.is_beneficial()
    where = "entirely below" if side == "below" else "entirely above"
    sentence = (
        f"{q.label}: the interval{interval} for {subject} lies {where} the "
        f"threshold{threshold}, so on these data the question is settled in that "
        "direction."
    )
    if good is True:
        return sentence + " That is the favourable side of the threshold."
    if good is False:
        return sentence + " That is the unfavourable side of the threshold."
    # "relative to no effect" is only true when the threshold *is* no effect. A
    # discrimination against a non-zero value -- the radius a diffuse atom would
    # have, a minimum worthwhile difference -- has already been read by the
    # sentence above, and adding this clause to it states a comparison the
    # analysis never made.
    if q.threshold == 0.0:
        return sentence + f" The estimate is {_direction(q)} relative to no effect."
    return sentence


def pivotal(evidence: Evidence) -> Quantity | None:
    """The finding an answer turns on: the unfavourable one, or else the first.

    A record whose findings disagree has one that matters more than the rest,
    and it is not the first in the list — it is the one on the wrong side of the
    threshold. Answering from ``findings[0]`` while a later finding points the
    other way is exactly what ``contested`` exists to catch, and this is the
    same rule applied to the single quantity a conclusion is allowed to state.
    """
    if not evidence.findings:
        return None
    for q in evidence.findings:
        if q.is_beneficial() is False:
            return q
    return evidence.findings[0]


def _agreeing(findings: tuple[Quantity, ...]) -> list[tuple[str, tuple[Quantity, ...]]]:
    """Findings bucketed by the reading they share, in first-appearance order.

    What makes the discussion repetitive is not that it reads every finding —
    it must — but that it reads four findings that agree with one another in
    four identically shaped sentences. Grouping them lets agreement be stated
    once and leaves the words for the finding that disagrees, which is the only
    one a reader has to think about.
    """
    buckets: dict[tuple[str, bool | None], list[Quantity]] = {}
    for q in findings:
        buckets.setdefault((q.against_threshold(), q.is_beneficial()), []).append(q)
    out: list[tuple[str, tuple[Quantity, ...]]] = []
    for (side, good), group in buckets.items():
        names = ", ".join(q.label for q in group)
        if side == "spans":
            text = (
                f"{names}: each interval contains the threshold, so these data do not "
                "settle which side of it the quantity falls on. That is an unsettled "
                "question rather than a finding of no effect."
            )
        elif side in ("below", "above"):
            where = "below" if side == "below" else "above"
            good_word: dict[bool | None, str] = {
                True: ", on the favourable side",
                False: ", on the unfavourable side",
                None: "",
            }
            text = (
                f"{names}: each interval lies entirely {where} the "
                f"threshold{good_word[good]}, so on these data every one of "
                "those questions is settled in that direction."
            )
        else:
            text = (
                f"{names}: no decision threshold was recorded, so the report states "
                "these estimates and stops short of judging them."
            )
        out.append((text, tuple(group)))
    return out


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
                    f"{sentence(evidence.verdict.reason)} The quantities below therefore "
                    "describe "
                    "what was measured. Reading them as the effect of the treatment "
                    "requires an assumption this analysis does not supply."
                )
            )
        )

    # Without the numbers: the results section carries them, and the guide asks
    # this section to repeat that one "in simpler terms and without referring to
    # stats". A finding that agrees with its neighbours is read alongside them
    # in one sentence; one that stands alone gets the full reading.
    for text, group in _agreeing(evidence.findings):
        if len(group) > 1:
            blocks.append(Paragraph(text=text))
            continue
        q = group[0]
        if detail["per_finding_heading"]:
            blocks.append(Heading(text=q.label, level=3))
        blocks.append(Paragraph(text=reading_of(q, causal=causal, stats=False)))
        if q.note and detail["include_notes"]:
            blocks.append(Paragraph(text=literal(q.note), emphasis=True))

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
                    "Every reading above is conditional on "
                    f"{plural(len(standing), 'unresolved assumption')}. The limitations "
                    "section states what each one claims "
                    "and what would challenge it."
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
        blocks.append(Paragraph(text=literal(evidence.question)))

    if not evidence.findings:
        blocks.append(Paragraph(text="No finding was recorded, so no conclusion follows."))
        return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]

    favourable, unfavourable = contested(evidence)
    if favourable and unfavourable:
        # One number in this section, and it is the one the answer turns on.
        # The rest are named, because a conclusions section that restates every
        # finding has become a second results section -- and the reader has
        # just read the first.
        lead = pivotal(evidence)
        stated = f", at {lead.stated()}" if lead is not None else ""
        harmed = ", ".join(q.label for q in unfavourable)
        helped = ", ".join(q.label for q in favourable)
        blocks.append(
            Paragraph(
                text=(
                    f"Not with one number. {harmed} sits on the unfavourable side "
                    f"of the threshold{stated}. {helped} sit on the favourable side. "
                    "Any single summary averages a harm with a benefit, and the "
                    "average is the number that gets quoted — so the disaggregated "
                    "findings are the result, and the pooled figure is not a "
                    "substitute for them."
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
                f"All of them hold under {plural(len(standing), 'unresolved assumption')} "
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
            f"It holds under {plural(len(standing), 'unresolved assumption')} "
            f"({', '.join(n.replace('_', ' ') for n in standing)}), and not otherwise."
        )
    if scope:
        blocks.append(Paragraph(text=" ".join(scope)))
    return Section(title=title, blocks=tuple(blocks))  # type: ignore[arg-type]
