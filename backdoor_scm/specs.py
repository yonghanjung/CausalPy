"""Immutable, finite, canonically serializable task specifications."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np


FrozenValue = None | bool | int | float | str | tuple["FrozenValue", ...]
_MAPPING_TAG = "__bdpfn_mapping_v1__"


def freeze_value(value: Any) -> FrozenValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Specification values must be finite.")
        if result == 0.0:
            return 0.0
        return result
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Mapping):
        return (
            _MAPPING_TAG,
            tuple(
                (str(key), freeze_value(item))
                for key, item in sorted(
                    value.items(),
                    key=lambda pair: str(pair[0]),
                )
            ),
        )
    if isinstance(value, (tuple, list)):
        return tuple(freeze_value(item) for item in value)
    raise TypeError(f"Unsupported specification value: {type(value)!r}")


def freeze_mapping(mapping: Mapping[str, Any]) -> tuple[tuple[str, FrozenValue], ...]:
    return tuple(
        (str(key), freeze_value(value))
        for key, value in sorted(mapping.items(), key=lambda pair: str(pair[0]))
    )


def thaw_value(value: FrozenValue) -> Any:
    if isinstance(value, tuple):
        if len(value) == 2 and value[0] == _MAPPING_TAG:
            entries = value[1]
            if not isinstance(entries, tuple):
                raise ValueError("Malformed frozen mapping payload.")
            return {
                str(key): thaw_value(item)
                for key, item in entries
            }
        return [thaw_value(item) for item in value]
    return value


def thaw_mapping(
    items: tuple[tuple[str, FrozenValue], ...],
) -> dict[str, Any]:
    return {key: thaw_value(value) for key, value in items}


def canonical_json(data: Mapping[str, Any]) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_hash(data: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FamilyRef:
    role: str
    family_id: str
    version: str
    semantic_class_id: str
    parameters: tuple[tuple[str, FrozenValue], ...]

    def __post_init__(self) -> None:
        if not self.role or not self.family_id or not self.version:
            raise ValueError("Family role, ID, and version must be non-empty.")
        if not self.semantic_class_id:
            raise ValueError("Semantic class ID must be non-empty.")

    @classmethod
    def create(
        cls,
        role: str,
        family_id: str,
        version: str,
        semantic_class_id: str,
        parameters: Mapping[str, Any],
    ) -> "FamilyRef":
        return cls(
            role=str(role),
            family_id=family_id,
            version=version,
            semantic_class_id=semantic_class_id,
            parameters=freeze_mapping(parameters),
        )

    def parameter_dict(self) -> dict[str, Any]:
        return thaw_mapping(self.parameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "family_id": self.family_id,
            "version": self.version,
            "semantic_class_id": self.semantic_class_id,
            "parameters": self.parameter_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FamilyRef":
        return cls.create(
            role=str(data["role"]),
            family_id=str(data["family_id"]),
            version=str(data["version"]),
            semantic_class_id=str(data["semantic_class_id"]),
            parameters=data["parameters"],
        )


@dataclass(frozen=True)
class BackdoorTaskSpec:
    task_id: int
    global_seed: int
    source_id: str
    rng_algorithm: str
    rng_version: str
    numpy_runtime_version: str
    manifest_hash: str
    covariate: FamilyRef
    propensity: FamilyRef
    outcome0: FamilyRef
    outcome1: FamilyRef
    outcome_type: str
    outcome_likelihood_mode: str

    def __post_init__(self) -> None:
        if self.task_id < 0 or self.global_seed < 0:
            raise ValueError("Task and global seed IDs must be non-negative.")
        if not self.source_id or not self.manifest_hash:
            raise ValueError("Source ID and manifest hash must be non-empty.")
        if (
            not self.rng_algorithm
            or not self.rng_version
            or not self.numpy_runtime_version
        ):
            raise ValueError("RNG and NumPy runtime versions must be non-empty.")
        if self.outcome_likelihood_mode not in {"same", "different"}:
            raise ValueError("Unknown outcome likelihood mode.")
        expected_roles = (
            (self.covariate, "covariate"),
            (self.propensity, "propensity"),
            (self.outcome0, "outcome"),
            (self.outcome1, "outcome"),
        )
        if any(component.role != role for component, role in expected_roles):
            raise ValueError("Task component has the wrong role.")
        outcome0_identity = self.outcome0.semantic_class_id
        outcome1_identity = self.outcome1.semantic_class_id
        if (
            self.outcome_likelihood_mode == "same"
            and outcome0_identity != outcome1_identity
        ):
            raise ValueError("Same-likelihood mode requires matching families.")
        if (
            self.outcome_likelihood_mode == "different"
            and outcome0_identity == outcome1_identity
        ):
            raise ValueError("Different-likelihood mode requires distinct families.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "global_seed": self.global_seed,
            "source_id": self.source_id,
            "rng_algorithm": self.rng_algorithm,
            "rng_version": self.rng_version,
            "numpy_runtime_version": self.numpy_runtime_version,
            "manifest_hash": self.manifest_hash,
            "covariate": self.covariate.to_dict(),
            "propensity": self.propensity.to_dict(),
            "outcome0": self.outcome0.to_dict(),
            "outcome1": self.outcome1.to_dict(),
            "outcome_type": self.outcome_type,
            "outcome_likelihood_mode": self.outcome_likelihood_mode,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def task_spec_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def sampling_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "global_seed": self.global_seed,
            "rng_algorithm": self.rng_algorithm,
            "rng_version": self.rng_version,
            "manifest_hash": self.manifest_hash,
            "covariate": self.covariate.to_dict(),
            "propensity": self.propensity.to_dict(),
            "outcome0": self.outcome0.to_dict(),
            "outcome1": self.outcome1.to_dict(),
            "outcome_type": self.outcome_type,
            "outcome_likelihood_mode": self.outcome_likelihood_mode,
        }

    @property
    def sampling_identity(self) -> str:
        return canonical_hash(self.sampling_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BackdoorTaskSpec":
        return cls(
            task_id=int(data["task_id"]),
            global_seed=int(data["global_seed"]),
            source_id=str(data["source_id"]),
            rng_algorithm=str(data["rng_algorithm"]),
            rng_version=str(data["rng_version"]),
            numpy_runtime_version=str(data["numpy_runtime_version"]),
            manifest_hash=str(data["manifest_hash"]),
            covariate=FamilyRef.from_dict(data["covariate"]),
            propensity=FamilyRef.from_dict(data["propensity"]),
            outcome0=FamilyRef.from_dict(data["outcome0"]),
            outcome1=FamilyRef.from_dict(data["outcome1"]),
            outcome_type=str(data["outcome_type"]),
            outcome_likelihood_mode=str(data["outcome_likelihood_mode"]),
        )

    @classmethod
    def from_json(cls, encoded: str) -> "BackdoorTaskSpec":
        return cls.from_dict(json.loads(encoded))


@dataclass(frozen=True)
class SemanticObservedBatch:
    x: np.ndarray
    a: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class SemanticQueryBatch:
    x: np.ndarray

    def __post_init__(self) -> None:
        owned = np.array(self.x, copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "x", owned)


@dataclass(frozen=True)
class TruthBatch:
    propensity: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    tau: np.ndarray


@dataclass(frozen=True)
class ProvenanceRecord:
    manifest_hash: str
    task_spec_hash: str
    sampling_identity: str
    seed_ids: tuple[tuple[str, int], ...]
    component_families: tuple[tuple[str, str, str], ...]
    rng_algorithm: str
    rng_version: str
    numpy_runtime_version: str
    source_id: str

    def __post_init__(self) -> None:
        if len(self.sampling_identity) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.sampling_identity
        ):
            raise ValueError("Sampling identity must be a lowercase SHA-256 hash.")
