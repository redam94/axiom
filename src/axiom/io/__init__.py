"""io: the ``analysis.axiom`` format, provenance, a content-addressed registry, the
scopes artifacts belong to, and the life of one experiment.

No pickle, cloudpickle, or dill anywhere in this package; gate 5 asserts it.
"""

from axiom.io.analysis import Analysis
from axiom.io.catalog import (
    Catalog,
    CatalogEntry,
    CrossScopeError,
    Program,
    ProgramStore,
)
from axiom.io.experiment import (
    Deviation,
    ExperimentRun,
    LifecycleError,
    Stage,
    Transition,
)
from axiom.io.provenance import Provenance, environment_fingerprint
from axiom.io.registry import ArtifactRegistry
from axiom.io.serialize import FORMAT_VERSION, FormatError, load_analysis, save_analysis

__all__ = [
    "FORMAT_VERSION",
    "Analysis",
    "ArtifactRegistry",
    "Catalog",
    "CatalogEntry",
    "CrossScopeError",
    "Deviation",
    "ExperimentRun",
    "FormatError",
    "LifecycleError",
    "Program",
    "ProgramStore",
    "Provenance",
    "Stage",
    "Transition",
    "environment_fingerprint",
    "load_analysis",
    "save_analysis",
]
