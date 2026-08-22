"""Figures as PNG bytes, for the two renderers that cannot hold a live one.

PPTX and PDF are static formats, so a plotly figure has to become an image, and
that needs ``kaleido``. It is behind the ``report`` extra rather than a core
dependency for the usual reason: it ships a browser engine, and a user who only
wants HTML should not pay for it.

The theme is applied here rather than in each renderer, so a figure looks the
same in a slide as it does in a PDF as it does in the browser.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from axiom.core import Unsupported
from axiom.report.theme import Theme

__all__ = ["MISSING_KALEIDO", "figure_png", "styled"]

MISSING_KALEIDO = Unsupported(
    reason=(
        "static image export needs kaleido; install the report extra "
        "(pip install 'axiom[report]') or render to HTML, which keeps figures live"
    ),
    missing=("report",),
)


def styled(figure: Any, theme: Theme, height: float) -> Any:
    """The theme applied to a figure, so all three formats agree on how it looks."""
    figure.update_layout(
        height=height,
        width=height * 16.0 / 9.0,
        margin={"l": 60, "r": 25, "t": 40 if figure.layout.title.text else 14, "b": 48},
        font={"family": theme.font, "size": theme.base_size},
        paper_bgcolor=theme.background_color,
        plot_bgcolor=theme.background_color,
        colorway=list(theme.palette),
    )
    return figure


def figure_png(
    figure: Any, theme: Theme, *, height: float, scale: float = 2.0
) -> bytes | Unsupported:
    """A themed figure as PNG bytes, or a typed refusal naming the missing extra."""
    if importlib.util.find_spec("kaleido") is None:
        return MISSING_KALEIDO
    # A kaleido that is installed and then fails is a real error, not a missing extra.
    # Rule 5: it propagates rather than being flattened into a typed refusal that would
    # send the caller off to install something they already have.
    return bytes(styled(figure, theme, height).to_image(format="png", scale=scale))
