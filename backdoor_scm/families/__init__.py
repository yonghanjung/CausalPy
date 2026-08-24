"""Explicit built-in registration and a small P1-P3 smoke manifest."""

from __future__ import annotations

from typing import Any

from ..manifest import FamilyChoice, PriorManifest
from ..registry import FamilyRegistry
from .covariates import (
    EmpiricalRowBootstrapFactory,
    GaussianCopulaCovariateFactory,
    IndependentProductCovariateFactory,
    IndependentStandardNormalFactory,
    LowRankGaussianCovariateFactory,
    StudentTCopulaCovariateFactory,
)
from .outcomes import (
    CenteredLaplaceAffineOutcomeFactory,
    GaussianAffineOutcomeFactory,
)
from .propensity import SparseAffinePropensityFactory


def build_builtin_registry(artifact_resolver: Any | None = None) -> FamilyRegistry:
    registry = FamilyRegistry()
    registry.register(IndependentStandardNormalFactory())
    registry.register(IndependentProductCovariateFactory())
    registry.register(GaussianCopulaCovariateFactory())
    registry.register(LowRankGaussianCovariateFactory())
    registry.register(StudentTCopulaCovariateFactory())
    registry.register(EmpiricalRowBootstrapFactory(artifact_resolver))
    registry.register(SparseAffinePropensityFactory())
    registry.register(GaussianAffineOutcomeFactory())
    registry.register(CenteredLaplaceAffineOutcomeFactory())
    return registry


def make_continuous_smoke_manifest(dimension: int = 3) -> PriorManifest:
    if dimension <= 0:
        raise ValueError("Dimension must be positive.")
    return PriorManifest(
        manifest_version="p1-p3-continuous-smoke-v2",
        covariate_choices=(
            FamilyChoice.create(
                "bdpfn.covariate.independent_standard_normal",
                "1.0.0",
                1.0,
                {"dimension": dimension},
            ),
        ),
        propensity_choices=(
            FamilyChoice.create(
                "bdpfn.propensity.sparse_affine",
                "1.0.0",
                1.0,
                {
                    "epsilon": 0.05,
                    "coefficient_scale": 1.0,
                    "nonzero_probability": 0.5,
                },
            ),
        ),
        outcome_choices=(
            FamilyChoice.create(
                "bdpfn.outcome.gaussian_affine",
                "2.0.0",
                0.5,
                {
                    "coefficient_scale": 1.0,
                    "noise_sd_min": 0.2,
                    "noise_sd_max": 1.5,
                },
            ),
            FamilyChoice.create(
                "bdpfn.outcome.centered_laplace_affine",
                "2.0.0",
                0.5,
                {
                    "coefficient_scale": 1.0,
                    "noise_sd_min": 0.2,
                    "noise_sd_max": 1.5,
                },
            ),
        ),
        outcome_type="continuous",
        same_likelihood_probability=0.70,
        different_compatible_likelihood_probability=0.30,
    )
