"""How a report looks, as a value.

A ``Theme`` is a ``Spec`` like everything else here, which is the point: the
styling of a report is part of its content hash, so "the same report, restyled"
and "a different report" are distinguishable, and a house style is a value you
pass rather than a file someone edits.

Colours are hex strings and lengths are in points, the one unit all three
renderers understand (HTML converts to ``pt``, python-pptx to EMU, reportlab
uses points natively). Nothing here is renderer-specific: a theme that renders
in HTML renders in PPTX and PDF, and the three try to look like each other.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, model_validator

from axiom.core import NonEmptyStr, Spec

__all__ = ["PageSize", "Theme"]

PageSize = Literal["letter", "a4", "widescreen", "standard"]
"""``letter``/``a4`` for paged output; ``widescreen`` (16:9) / ``standard`` (4:3) for slides."""

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")

#: Page geometry in points, by name. Slides use the PowerPoint conventions.
GEOMETRY: dict[str, tuple[float, float]] = {
    "letter": (612.0, 792.0),
    "a4": (595.0, 842.0),
    "widescreen": (960.0, 540.0),
    "standard": (720.0, 540.0),
}


class Theme(Spec):
    """Fonts, colours and geometry for a rendered report.

    ``palette`` is used in order for series that need distinct colours; the
    renderers do not invent colours outside it. ``band_opacity`` is the one
    number that is about statistics rather than taste — it is the fill of an
    uncertainty band, and setting it to zero is not allowed, because a report
    that draws a surface draws its band.
    """

    name: NonEmptyStr = "axiom"
    font: NonEmptyStr = "Helvetica"
    mono_font: NonEmptyStr = "Courier"
    base_size: float = Field(default=11.0, gt=0)
    title_size: float = Field(default=24.0, gt=0)
    heading_sizes: tuple[float, float, float] = (18.0, 14.0, 12.0)
    text_color: str = "#1c1c1c"
    muted_color: str = "#5b6472"
    accent_color: str = "#2f7fd1"
    background_color: str = "#ffffff"
    rule_color: str = "#d8dbe0"
    palette: tuple[str, ...] = ("#2f7fd1", "#b5453b", "#3aa17e", "#8a63c4", "#c9a227", "#5b6472")
    page: PageSize = "letter"
    slide: PageSize = "widescreen"
    margin: float = Field(default=54.0, ge=0)
    figure_height: float = Field(default=300.0, gt=0)
    band_opacity: float = Field(default=0.22, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _valid(self) -> Theme:
        for field in (
            "text_color",
            "muted_color",
            "accent_color",
            "background_color",
            "rule_color",
        ):
            value = getattr(self, field)
            if not _HEX.match(value):
                raise ValueError(f"{field} must be a #rrggbb hex colour, got {value!r}")
        for colour in self.palette:
            if not _HEX.match(colour):
                raise ValueError(f"palette colours must be #rrggbb, got {colour!r}")
        if not self.palette:
            raise ValueError("a theme needs at least one palette colour")
        if self.page not in ("letter", "a4"):
            raise ValueError(f"page must be a paper size, got {self.page!r}")
        if self.slide not in ("widescreen", "standard"):
            raise ValueError(f"slide must be a slide size, got {self.slide!r}")
        return self

    def page_geometry(self) -> tuple[float, float]:
        """``(width, height)`` in points for paged output."""
        return GEOMETRY[self.page]

    def slide_geometry(self) -> tuple[float, float]:
        """``(width, height)`` in points for slides."""
        return GEOMETRY[self.slide]

    def heading_size(self, level: int) -> float:
        """Point size for a heading level, clamped to the three the theme defines."""
        return self.heading_sizes[min(max(level, 1), len(self.heading_sizes)) - 1]

    def colour(self, index: int) -> str:
        """The ``index``-th series colour, cycling."""
        return self.palette[index % len(self.palette)]
