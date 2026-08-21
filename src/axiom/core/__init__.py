"""core: vocabulary, dimensions, specs, protocols, verdicts, and intervals.

The foundation layer. Imports nothing else from axiom; depends only on
numpy, scipy, and pydantic. Everything here is a ``Spec`` or a protocol.
"""

from axiom.core.dimensions import (
    BASES,
    UNITS,
    BaseRegistry,
    D,
    Dimension,
    DimensionError,
    UndeclaredBaseError,
    UnitConversionError,
    UnitSystem,
    dimensionless,
)
from axiom.core.entities import (
    Covariate,
    Dose,
    Entity,
    EntityName,
    Intervention,
    Outcome,
    Population,
    TimeWindow,
    Treatment,
    UndimensionedWarning,
    Unit,
    dimension_of,
)
from axiom.core.intervals import Interval, Summary, eti, hdi, interval, summarize
from axiom.core.posterior import Posterior
from axiom.core.protocols import (
    Capability,
    PredictiveDraws,
    SupportsIntervention,
    SupportsPosterior,
    missing_capabilities,
)
from axiom.core.result import Blocked, Failure, NonEmptyStr, Unsupported, Unverified, is_failure
from axiom.core.spec import (
    SchemaVersionError,
    Spec,
    SpecDiff,
    SpecError,
    UnknownSpecError,
    load_spec,
    spec_type_name,
)
from axiom.core.stats import (
    AcceptanceRegion,
    clopper_pearson,
    effective_sample_size,
    mc_standard_error,
    z_score,
)
from axiom.core.verdict import Assumption, LedgerLine, Verdict

__all__ = [
    "BASES",
    "D",
    "UNITS",
    "AcceptanceRegion",
    "Assumption",
    "BaseRegistry",
    "Blocked",
    "Capability",
    "Covariate",
    "Dimension",
    "DimensionError",
    "Dose",
    "Entity",
    "EntityName",
    "Failure",
    "Intervention",
    "Interval",
    "LedgerLine",
    "NonEmptyStr",
    "Outcome",
    "Population",
    "Posterior",
    "PredictiveDraws",
    "SchemaVersionError",
    "Spec",
    "SpecDiff",
    "SpecError",
    "Summary",
    "SupportsIntervention",
    "SupportsPosterior",
    "TimeWindow",
    "Treatment",
    "UndeclaredBaseError",
    "UndimensionedWarning",
    "Unit",
    "UnitConversionError",
    "UnitSystem",
    "UnknownSpecError",
    "Unsupported",
    "Unverified",
    "Verdict",
    "clopper_pearson",
    "dimension_of",
    "dimensionless",
    "effective_sample_size",
    "eti",
    "hdi",
    "interval",
    "is_failure",
    "load_spec",
    "mc_standard_error",
    "missing_capabilities",
    "spec_type_name",
    "summarize",
    "z_score",
]
