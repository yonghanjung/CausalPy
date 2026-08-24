"""Compatible continuous outcome likelihoods with exact affine means."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..registry import ComponentRole
from ..specs import FamilyRef


@dataclass(frozen=True)
class _AffineOutcomeKernel:
    intercept: float
    weights: tuple[float, ...]
    noise_sd: float
    outcome_type = "continuous"

    def __post_init__(self) -> None:
        if not self.weights:
            raise ValueError("Outcome weights cannot be empty.")
        if not np.isfinite(self.noise_sd) or self.noise_sd <= 0.0:
            raise ValueError("Outcome noise SD must be positive and finite.")
        if not np.isfinite(self.intercept):
            raise ValueError("Outcome intercept must be finite.")
        if not np.isfinite(np.asarray(self.weights, dtype=float)).all():
            raise ValueError("Outcome weights must be finite.")

    @property
    def dimension(self) -> int:
        return len(self.weights)

    def mean(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError("Outcome input has the wrong shape.")
        return self.intercept + x @ np.asarray(self.weights, dtype=float)

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, Any]):
        return cls(
            intercept=float(parameters["intercept"]),
            weights=tuple(float(value) for value in parameters["weights"]),
            noise_sd=float(parameters["noise_sd"]),
        )


class GaussianAffineOutcomeKernel(_AffineOutcomeKernel):
    def sample(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self.mean(x) + self.noise_sd * rng.normal(size=x.shape[0])


class CenteredLaplaceAffineOutcomeKernel(_AffineOutcomeKernel):
    def sample(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self.mean(x) + rng.laplace(
            loc=0.0,
            scale=self.noise_sd / np.sqrt(2.0),
            size=x.shape[0],
        )


class _AffineOutcomeFactory:
    role = ComponentRole.OUTCOME
    version = "2.0.0"
    outcome_type = "continuous"
    kernel_class: type[_AffineOutcomeKernel]

    def sample_spec(
        self,
        rng: np.random.Generator,
        hyperparameters: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> FamilyRef:
        dimension = int(context["dimension"])
        coefficient_scale = float(hyperparameters.get("coefficient_scale", 1.0))
        noise_sd_min = float(hyperparameters.get("noise_sd_min", 0.2))
        noise_sd_max = float(hyperparameters.get("noise_sd_max", 1.5))
        if not 0.0 < noise_sd_min <= noise_sd_max:
            raise ValueError("Invalid outcome noise SD range.")
        log_noise_sd = rng.uniform(np.log(noise_sd_min), np.log(noise_sd_max))
        return FamilyRef.create(
            role=self.role.value,
            family_id=self.family_id,
            version=self.version,
            semantic_class_id=self.semantic_class_id,
            parameters={
                "intercept": float(rng.normal(scale=coefficient_scale)),
                "weights": tuple(
                    float(value)
                    for value in rng.normal(
                        scale=coefficient_scale,
                        size=dimension,
                    )
                ),
                "noise_sd": float(np.exp(log_noise_sd)),
            },
        )

    def build(self, spec: FamilyRef) -> _AffineOutcomeKernel:
        return self.kernel_class.from_parameters(spec.parameter_dict())


class GaussianAffineOutcomeFactory(_AffineOutcomeFactory):
    family_id = "bdpfn.outcome.gaussian_affine"
    semantic_class_id = "continuous.gaussian"
    kernel_class = GaussianAffineOutcomeKernel


class CenteredLaplaceAffineOutcomeFactory(_AffineOutcomeFactory):
    family_id = "bdpfn.outcome.centered_laplace_affine"
    semantic_class_id = "continuous.laplace"
    kernel_class = CenteredLaplaceAffineOutcomeKernel
