"""Built-in exact propensity functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..registry import ComponentRole
from ..specs import FamilyRef


@dataclass(frozen=True)
class SparseAffinePropensity:
    intercept: float
    weights: tuple[float, ...]
    epsilon: float

    def __post_init__(self) -> None:
        if not 0.0 < self.epsilon < 0.5:
            raise ValueError("Positivity epsilon must lie in (0, 0.5).")
        if not self.weights:
            raise ValueError("Propensity weights cannot be empty.")
        if not np.isfinite(self.intercept):
            raise ValueError("Propensity intercept must be finite.")
        if not np.isfinite(np.asarray(self.weights, dtype=float)).all():
            raise ValueError("Propensity weights must be finite.")

    @property
    def dimension(self) -> int:
        return len(self.weights)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError("Propensity input has the wrong shape.")
        weights = np.asarray(self.weights, dtype=float)
        score = np.clip(self.intercept + x @ weights, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-score))
        bounded = self.epsilon + (1.0 - 2.0 * self.epsilon) * probability
        return np.clip(bounded, self.epsilon, 1.0 - self.epsilon)


class SparseAffinePropensityFactory:
    role = ComponentRole.PROPENSITY
    family_id = "bdpfn.propensity.sparse_affine"
    version = "1.0.0"
    semantic_class_id = "propensity.sparse_affine"
    outcome_type = None

    def sample_spec(
        self,
        rng: np.random.Generator,
        hyperparameters: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> FamilyRef:
        dimension = int(context["dimension"])
        coefficient_scale = float(hyperparameters.get("coefficient_scale", 1.0))
        nonzero_probability = float(hyperparameters.get("nonzero_probability", 0.5))
        weights = rng.normal(scale=coefficient_scale, size=dimension)
        mask = rng.random(dimension) < nonzero_probability
        if not mask.any():
            mask[int(rng.integers(0, dimension))] = True
        weights = weights * mask
        return FamilyRef.create(
            role=self.role.value,
            family_id=self.family_id,
            version=self.version,
            semantic_class_id=self.semantic_class_id,
            parameters={
                "intercept": float(rng.normal(scale=coefficient_scale)),
                "weights": tuple(float(value) for value in weights),
                "epsilon": float(hyperparameters.get("epsilon", 0.05)),
            },
        )

    def build(self, spec: FamilyRef) -> SparseAffinePropensity:
        parameters = spec.parameter_dict()
        return SparseAffinePropensity(
            intercept=float(parameters["intercept"]),
            weights=tuple(float(value) for value in parameters["weights"]),
            epsilon=float(parameters["epsilon"]),
        )
