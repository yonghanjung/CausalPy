"""Behavioral contracts for modular back-door task components."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import numpy as np

from .specs import FamilyRef


class ArtifactResolver(Protocol):
    def resolve(self, artifact_id: str, expected_sha256: str) -> Any: ...


class CovariateLaw(Protocol):
    @property
    def dimension(self) -> int: ...

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
    ) -> Any: ...

    def contains(self, x: np.ndarray) -> bool: ...


class PropensityFunction(Protocol):
    @property
    def dimension(self) -> int: ...

    def evaluate(self, x: np.ndarray) -> np.ndarray: ...


class OutcomeKernel(Protocol):
    @property
    def dimension(self) -> int: ...

    @property
    def outcome_type(self) -> str: ...

    def sample(
        self,
        x: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray: ...

    def mean(self, x: np.ndarray) -> np.ndarray: ...


class FamilyFactory(Protocol):
    role: Any
    family_id: str
    version: str
    semantic_class_id: str
    outcome_type: str | None

    def sample_spec(
        self,
        rng: np.random.Generator,
        hyperparameters: Mapping[str, Any],
        context: Mapping[str, Any],
    ) -> FamilyRef: ...

    def build(self, spec: FamilyRef) -> object: ...
