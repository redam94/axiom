"""data: the role-tagged ``Panel`` and its scaling. Imports only ``core``."""

from axiom.data.frame import Completeness, Panel, PanelError
from axiom.data.roles import RoleKind, RoleMap
from axiom.data.scale import ColumnScaling, ScalingMethod, ScalingParameters, fit_scaling

__all__ = [
    "ColumnScaling",
    "Completeness",
    "Panel",
    "PanelError",
    "RoleKind",
    "RoleMap",
    "ScalingMethod",
    "ScalingParameters",
    "fit_scaling",
]
