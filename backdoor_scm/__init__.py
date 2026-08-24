"""Canonical modular random back-door SCM generator."""

from .active_v1 import active_v1_manifest
from .artifacts import ArtifactResolver, InMemoryArtifactProvider, ResolvedArtifact
from .capabilities import TruthCapability, TruthLevel
from .exceptions import OutOfSupportError
from .families import build_builtin_registry, make_continuous_smoke_manifest
from .families.covariates import (
    EmpiricalRowBootstrapCovariates,
    EmpiricalRowBootstrapFactory,
    GaussianCopulaCovariateFactory,
    GaussianCopulaCovariates,
    IndependentProductCovariateFactory,
    IndependentProductCovariates,
    LowRankGaussianCovariateFactory,
    LowRankGaussianCovariates,
    StudentTCopulaCovariateFactory,
    StudentTCopulaCovariates,
)
from .families.roots import (
    RootLaw,
    RootSpec,
    build_root,
    root_family_weights,
    sample_root_spec,
)
from .manifest import (
    ActivePriorManifest,
    DesignChoice,
    DesignLaw,
    FamilyChoice,
    FamilyDesign,
    PriorManifest,
)
from .registry import ComponentRole, FamilyRegistry
from .schema import (
    CovariateSchema,
    SchemaSamplingPolicy,
    TypedCovariateBatch,
    VariableSpec,
    VariableType,
    active_v1_schema_policy,
    sample_schema,
)
from .specs import (
    BackdoorTaskSpec,
    FamilyRef,
    ProvenanceRecord,
    SemanticObservedBatch,
    SemanticQueryBatch,
    TruthBatch,
)
from .task import BackdoorTask, load_task, sample_task

__all__ = [
    "BackdoorTask",
    "BackdoorTaskSpec",
    "ArtifactResolver",
    "ActivePriorManifest",
    "ComponentRole",
    "CovariateSchema",
    "DesignChoice",
    "DesignLaw",
    "FamilyChoice",
    "FamilyDesign",
    "FamilyRef",
    "FamilyRegistry",
    "EmpiricalRowBootstrapCovariates",
    "EmpiricalRowBootstrapFactory",
    "GaussianCopulaCovariateFactory",
    "GaussianCopulaCovariates",
    "InMemoryArtifactProvider",
    "IndependentProductCovariateFactory",
    "IndependentProductCovariates",
    "LowRankGaussianCovariateFactory",
    "LowRankGaussianCovariates",
    "OutOfSupportError",
    "PriorManifest",
    "ProvenanceRecord",
    "SemanticObservedBatch",
    "SemanticQueryBatch",
    "SchemaSamplingPolicy",
    "TruthBatch",
    "TruthCapability",
    "TruthLevel",
    "TypedCovariateBatch",
    "VariableSpec",
    "VariableType",
    "active_v1_manifest",
    "active_v1_schema_policy",
    "build_root",
    "build_builtin_registry",
    "make_continuous_smoke_manifest",
    "load_task",
    "root_family_weights",
    "RootLaw",
    "RootSpec",
    "ResolvedArtifact",
    "sample_root_spec",
    "sample_schema",
    "sample_task",
    "StudentTCopulaCovariateFactory",
    "StudentTCopulaCovariates",
]
