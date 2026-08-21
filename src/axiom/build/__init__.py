"""build: fluent, immutable builders over every ``Spec``.

A builder is a frozen dataclass holding a ``Fields`` mapping; every fluent
method returns a new builder and ``build()`` returns a ``Spec`` that
round-trips through ``load_spec``. See ``nbs/build/``.
"""

from __future__ import annotations

from axiom.build.base import Builder, BuildError, Fields
from axiom.build.graph import GraphBuilder
from axiom.build.meta import MetaBuilder
from axiom.build.prior import MomentFamily, PriorBuilder
from axiom.build.study import SchedulePattern, StudyBuilder
from axiom.build.surface import SurfaceBuilder
from axiom.build.variable import EntitySpec, VariableBuilder, VariableKind

__all__ = [
    "BuildError",
    "Builder",
    "EntitySpec",
    "Fields",
    "GraphBuilder",
    "MetaBuilder",
    "MomentFamily",
    "PriorBuilder",
    "SchedulePattern",
    "StudyBuilder",
    "SurfaceBuilder",
    "VariableBuilder",
    "VariableKind",
]
