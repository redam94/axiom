"""adapters: domain vocabularies over the general core.

The only package where marketing identifiers (channel, spend, geo, KPI,
ROAS) are permitted (gate 3). See ``nbs/adapters/``.
"""

from __future__ import annotations

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
    "contribution",
    "impressions",
    "marginal_roas",
    "marketing_spec",
    "panel_from_marketing_frame",
    "panel_from_mff",
    "roas",
    "roi",
    "role_map",
    "spend",
]
