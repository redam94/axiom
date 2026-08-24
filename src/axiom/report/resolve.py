"""Template + context → renderer-neutral content, once, for all three formats.

The renderers differ in how they draw a paragraph; they must not differ in what
a paragraph *says*, which number a metric shows, or whether a surface arrives
with its band. So resolution happens here, once, and produces plain values the
renderers only have to lay out.

The two rules that live in this module rather than in any renderer:

* **A metric shows its interval.** Anything that carries one — ``Interval``,
  ``Summary``, ``EstimandResult`` — is resolved to a value *and* an interval
  with its definition and mass. There is no branch that drops it.
* **A figure of a surface shows its band.** A ``ResponseBand`` source is
  rendered through ``viz.response_curve``, which cannot draw one without it.

A source that is present but of a type a block cannot use is a ``ValueError``
naming the block, the key and the type — not a silently blank slide.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from axiom.core import Interval, LedgerLine, Unsupported, format_interval, format_measured
from axiom.report.spec import (
    AnyBlock,
    Figure,
    Heading,
    LedgerBlock,
    Metric,
    PageBreak,
    Paragraph,
    Report,
    Table,
)
from axiom.report.spec import missing as _missing

#: Decimals for a metric that carries neither a stated precision nor an
#: interval. Two, as before: with nothing to round against there is nothing to
#: derive, and the old default is the least surprising answer.
_UNSTATED_PRECISION = 2

__all__ = [
    "ResolvedBlock",
    "ResolvedFigure",
    "ResolvedLedger",
    "ResolvedMetric",
    "ResolvedReport",
    "ResolvedSection",
    "ResolvedTable",
    "ResolvedText",
    "format_text",
    "resolve",
]


@dataclass(frozen=True)
class ResolvedText:
    """A heading or paragraph with its placeholders filled in."""

    text: str
    level: int = 0
    emphasis: bool = False

    @property
    def is_heading(self) -> bool:
        return self.level > 0


@dataclass(frozen=True)
class ResolvedFigure:
    """A plotly figure, its caption, and how tall it should be drawn."""

    figure: Any
    caption: str = ""
    height: float | None = None
    full_page: bool = False


@dataclass(frozen=True)
class ResolvedTable:
    """Header row, body rows already formatted to strings, and what was truncated."""

    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    caption: str = ""
    truncated: int = 0


@dataclass(frozen=True)
class ResolvedMetric:
    """A number, its label, and the interval it came with — if it had one."""

    label: str
    value: float
    text: str
    interval: Interval | None = None
    unit: str = ""

    @property
    def interval_text(self) -> str:
        if self.interval is None:
            return ""
        bounds = format_interval(self.interval.lower, self.interval.upper, group=True)
        return f"{bounds} ({self.interval.mass:.0%} {self.interval.definition.upper()})"


@dataclass(frozen=True)
class ResolvedLedger:
    """The assumption trail, as lines of kind / statement / assumption."""

    lines: tuple[LedgerLine, ...]
    caption: str = ""
    show_detail: bool = False


ResolvedBlock = ResolvedText | ResolvedFigure | ResolvedTable | ResolvedMetric | ResolvedLedger


@dataclass(frozen=True)
class ResolvedSection:
    """One section's title, optional summary, and its blocks in reading order.

    ``breaks`` holds the indices *before* which a new page or slide starts, so a
    renderer does not have to carry the ``PageBreak`` marker through its layout.
    """

    title: str
    summary: str
    blocks: tuple[ResolvedBlock, ...] = ()
    breaks: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ResolvedReport:
    """Everything the renderers need and nothing they do not."""

    name: str
    title: str
    subtitle: str
    footer: str
    theme: Any
    sections: tuple[ResolvedSection, ...]
    source_hash: str


class _SafeMap(dict[str, Any]):
    def __missing__(self, key: str) -> Any:  # pragma: no cover - guarded by `missing`
        raise KeyError(key)


def format_text(template: str, context: Mapping[str, object], *, where: str) -> str:
    """Fill ``{placeholders}`` from the context, naming the block when one is absent."""
    try:
        return template.format_map(_SafeMap(context))
    except KeyError as e:
        raise ValueError(f"{where}: no context value named {e.args[0]!r}") from e
    except (IndexError, ValueError) as e:
        raise ValueError(f"{where}: cannot format {template!r}: {e}") from e


def _figure_of(value: object, block: Figure, theme: Any) -> Any:
    """A plotly figure from whatever the source held, with a surface's band intact."""
    from axiom.surface import ResponseBand

    if isinstance(value, ResponseBand):
        from axiom.viz import response_curve

        drawn = response_curve(value)
        if isinstance(drawn, Unsupported):
            raise ValueError(
                f"figure block {block.source!r} holds a ResponseBand but it cannot be "
                f"drawn: {drawn}"
            )
        return drawn
    if hasattr(value, "to_dict") and hasattr(value, "data") and hasattr(value, "layout"):
        return value
    raise ValueError(
        f"figure block {block.source!r} needs a plotly Figure or a surface.ResponseBand, "
        f"got {type(value).__name__}"
    )


def _rows_of(
    value: object, block: Table
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...], int]:
    frame = value
    if hasattr(frame, "to_dict") and hasattr(frame, "columns"):  # a pandas DataFrame
        columns = [str(c) for c in frame.columns]
        records = [{str(k): v for k, v in row.items()} for row in frame.to_dict(orient="records")]
    elif isinstance(frame, Sequence) and all(isinstance(r, Mapping) for r in frame):
        records = [{str(k): v for k, v in dict(r).items()} for r in frame]
        columns = list(dict.fromkeys(k for r in records for k in r))
    else:
        raise ValueError(
            f"table block {block.source!r} needs a DataFrame or a sequence of mappings, "
            f"got {type(value).__name__}"
        )
    if block.columns:
        unknown = [c for c in block.columns if c not in columns]
        if unknown:
            raise ValueError(f"table block {block.source!r} has no column(s) {unknown}")
        columns = list(block.columns)
    kept, truncated = records[: block.max_rows], max(0, len(records) - block.max_rows)

    def cell(v: object) -> str:
        if isinstance(v, float):
            return f"{v:,.{block.precision}f}"
        return "" if v is None else str(v)

    rows = tuple(tuple(cell(record.get(c)) for c in columns) for record in kept)
    return tuple(columns), rows, truncated


def _metric_of(value: object, block: Metric) -> ResolvedMetric:
    """A value and, when the source carried one, its interval."""
    interval: Interval | None = None
    point: float
    if isinstance(value, Interval):
        interval = value
        point = 0.5 * (value.lower + value.upper)
    elif hasattr(value, "summary") and hasattr(value.summary, "interval"):
        summary = value.summary  # an EstimandResult
        interval = summary.interval
        point = float(summary.mean)
    elif hasattr(value, "interval") and hasattr(value, "mean"):  # a core.Summary
        interval = value.interval
        point = float(value.mean)
    elif isinstance(value, int | float):
        point = float(value)
    else:
        raise ValueError(
            f"metric block {block.source!r} needs a number, Interval, Summary or "
            f"EstimandResult, got {type(value).__name__}"
        )
    unit = f" {block.unit}" if block.unit else ""
    if block.precision is not None:
        text = f"{point:,.{block.precision}f}"
    elif interval is not None:
        # the interval is the resolution: a point printed finer than its own
        # half-width claims digits the source never had
        text = format_measured(point, interval.half_width, group=True)
    else:
        text = f"{point:,.{_UNSTATED_PRECISION}f}"
    return ResolvedMetric(
        label=block.label,
        value=point,
        text=f"{text}{unit}",
        interval=interval,
        unit=block.unit,
    )


def _ledger_of(value: object, block: LedgerBlock) -> tuple[LedgerLine, ...]:
    if isinstance(value, LedgerLine):
        return (value,)
    if isinstance(value, Sequence) and all(isinstance(line, LedgerLine) for line in value):
        return tuple(value)
    raise ValueError(
        f"ledger block {block.source!r} needs a LedgerLine or a sequence of them, "
        f"got {type(value).__name__}"
    )


def _resolve_block(
    block: AnyBlock, context: Mapping[str, object], theme: Any
) -> ResolvedBlock | None:
    where = f"{block.block} block"
    match block:
        case Heading():
            return ResolvedText(
                text=format_text(block.text, context, where=where), level=block.level
            )
        case Paragraph():
            return ResolvedText(
                text=format_text(block.text, context, where=where), emphasis=block.emphasis
            )
        case Figure():
            figure = _figure_of(context[block.source], block, theme)
            return ResolvedFigure(
                figure=figure,
                caption=format_text(block.caption, context, where=where),
                height=block.height,
                full_page=block.full_page,
            )
        case Table():
            columns, rows, truncated = _rows_of(context[block.source], block)
            return ResolvedTable(
                columns=columns,
                rows=rows,
                caption=format_text(block.caption, context, where=where),
                truncated=truncated,
            )
        case Metric():
            return _metric_of(context[block.source], block)
        case LedgerBlock():
            return ResolvedLedger(
                lines=_ledger_of(context[block.source], block),
                caption=format_text(block.caption, context, where=where),
                show_detail=block.show_detail,
            )
    return None  # Divider and PageBreak carry no content


def resolve(report: Report, context: Mapping[str, object]) -> ResolvedReport | Unsupported:
    """Fill the template from the context, or say exactly which keys are absent.

    ``Unsupported`` when a source is missing — a report that cannot be built is
    a typed failure naming every missing key at once, not an exception on the
    first one. A source that is *present* but unusable is a ``ValueError``,
    because that is a mistake in the template rather than a gap in the data.
    """
    absent = _missing(report, context)
    if absent:
        return Unsupported(
            reason=(
                f"report {report.name!r} needs context value(s) "
                f"{', '.join(repr(a) for a in absent)}"
            ),
            missing=absent,
        )
    sections: list[ResolvedSection] = []
    for section in report.sections:
        blocks: list[ResolvedBlock] = []
        breaks: set[int] = set()
        for block in section.blocks:
            if isinstance(block, PageBreak):
                breaks.add(len(blocks))
                continue
            resolved = _resolve_block(block, context, report.theme)
            if resolved is not None:
                blocks.append(resolved)
        sections.append(
            ResolvedSection(
                title=format_text(section.title, context, where=f"section {section.title!r}"),
                summary=format_text(section.summary, context, where=f"section {section.title!r}"),
                blocks=tuple(blocks),
                breaks=frozenset(breaks),
            )
        )
    return ResolvedReport(
        name=report.name,
        title=format_text(report.title, context, where="report title"),
        subtitle=format_text(report.subtitle, context, where="report subtitle"),
        footer=format_text(report.footer, context, where="report footer"),
        theme=report.theme,
        sections=tuple(sections),
        source_hash=report.content_hash(),
    )
