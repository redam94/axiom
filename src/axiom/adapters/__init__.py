"""adapters: domain vocabularies over the general core.

The only package where domain identifiers (channel, spend, geo, KPI, ROAS;
plot, yield, nutrient) are permitted (gate 3). See ``nbs/adapters/``.

Each domain is a **submodule**, and that is the interface: ``adapters.marketing``
and ``adapters.agronomy``. A flat union of domain vocabularies collides by
construction — every adapter wants to call its own translator ``role_map`` and
its own preset ``spec`` — so a new adapter is added here as a name, not as a
spray of symbols. The marketing names are re-exported flat as well because
they were public before that was understood; nothing new should follow them.
"""

from __future__ import annotations

from axiom.adapters import agronomy, marketing
from axiom.adapters.marketing import (
    EXPOSURE,
    KPI,
    Channel,
    Geo,
    MarketingRoles,
    contribution,
    impressions,
    marginal_roas,
    marketing_spec,
    panel_from_marketing_frame,
    panel_from_mff,
    roas,
    roi,
    role_map,
    spend,
)

__all__ = [
    "EXPOSURE",
    "KPI",
    "Channel",
    "Geo",
    "MarketingRoles",
    "agronomy",
    "contribution",
    "impressions",
    "marginal_roas",
    "marketing",
    "marketing_spec",
    "panel_from_marketing_frame",
    "panel_from_mff",
    "roas",
    "roi",
    "role_map",
    "spend",
]
