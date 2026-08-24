"""Immutable prior manifests, separate from installed family registries."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping

from .specs import (
    FrozenValue,
    canonical_hash,
    canonical_json,
    freeze_mapping,
    thaw_mapping,
)


@dataclass(frozen=True)
class FamilyChoice:
    family_id: str
    version: str
    weight: float
    hyperparameters: tuple[tuple[str, FrozenValue], ...]

    def __post_init__(self) -> None:
        if not self.family_id or not self.version:
            raise ValueError("Family ID and version must be non-empty.")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("Family weight must be positive and finite.")

    @classmethod
    def create(
        cls,
        family_id: str,
        version: str,
        weight: float,
        hyperparameters: Mapping[str, Any],
    ) -> "FamilyChoice":
        return cls(
            family_id=family_id,
            version=version,
            weight=float(weight),
            hyperparameters=freeze_mapping(hyperparameters),
        )

    def hyperparameter_dict(self) -> dict[str, Any]:
        return thaw_mapping(self.hyperparameters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "version": self.version,
            "weight": self.weight,
            "hyperparameters": self.hyperparameter_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FamilyChoice":
        return cls.create(
            str(data["family_id"]),
            str(data["version"]),
            float(data["weight"]),
            data["hyperparameters"],
        )


def _validate_choices(name: str, choices: tuple[FamilyChoice, ...]) -> None:
    if not choices:
        raise ValueError(f"{name} choices cannot be empty.")
    total = sum(choice.weight for choice in choices)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} weights must sum to one, got {total!r}.")
    identities = [(choice.family_id, choice.version) for choice in choices]
    if len(set(identities)) != len(identities):
        raise ValueError(f"{name} choices contain duplicate families.")


@dataclass(frozen=True)
class PriorManifest:
    manifest_version: str
    covariate_choices: tuple[FamilyChoice, ...]
    propensity_choices: tuple[FamilyChoice, ...]
    outcome_choices: tuple[FamilyChoice, ...]
    outcome_type: str
    same_likelihood_probability: float
    different_compatible_likelihood_probability: float

    def __post_init__(self) -> None:
        if not self.manifest_version or not self.outcome_type:
            raise ValueError("Manifest version and outcome type must be non-empty.")
        _validate_choices("covariate", self.covariate_choices)
        _validate_choices("propensity", self.propensity_choices)
        _validate_choices("outcome", self.outcome_choices)
        probabilities = (
            self.same_likelihood_probability,
            self.different_compatible_likelihood_probability,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in probabilities):
            raise ValueError("Likelihood probabilities must be finite and non-negative.")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("Likelihood probabilities must sum to one.")
        if probabilities[1] > 0.0 and len(self.outcome_choices) < 2:
            raise ValueError("Different-likelihood tasks require at least two families.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "covariate_choices": [item.to_dict() for item in self.covariate_choices],
            "propensity_choices": [item.to_dict() for item in self.propensity_choices],
            "outcome_choices": [item.to_dict() for item in self.outcome_choices],
            "outcome_type": self.outcome_type,
            "same_likelihood_probability": self.same_likelihood_probability,
            "different_compatible_likelihood_probability": (
                self.different_compatible_likelihood_probability
            ),
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PriorManifest":
        return cls(
            manifest_version=str(data["manifest_version"]),
            covariate_choices=tuple(
                FamilyChoice.from_dict(item) for item in data["covariate_choices"]
            ),
            propensity_choices=tuple(
                FamilyChoice.from_dict(item) for item in data["propensity_choices"]
            ),
            outcome_choices=tuple(
                FamilyChoice.from_dict(item) for item in data["outcome_choices"]
            ),
            outcome_type=str(data["outcome_type"]),
            same_likelihood_probability=float(data["same_likelihood_probability"]),
            different_compatible_likelihood_probability=float(
                data["different_compatible_likelihood_probability"]
            ),
        )


@dataclass(frozen=True)
class DesignChoice:
    label: str
    weight: float
    metadata: tuple[tuple[str, FrozenValue], ...] = ()

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("Design choice label must be non-empty.")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("Design choice weight must be positive and finite.")

    @classmethod
    def create(
        cls,
        label: str,
        weight: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> "DesignChoice":
        return cls(
            label=str(label),
            weight=float(weight),
            metadata=freeze_mapping({} if metadata is None else metadata),
        )

    def metadata_dict(self) -> dict[str, Any]:
        return thaw_mapping(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "weight": self.weight,
            "metadata": self.metadata_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DesignChoice":
        return cls.create(
            label=str(data["label"]),
            weight=float(data["weight"]),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class DesignLaw:
    law_id: str
    choices: tuple[DesignChoice, ...]
    metadata: tuple[tuple[str, FrozenValue], ...] = ()

    def __post_init__(self) -> None:
        if not self.law_id or not self.choices:
            raise ValueError("Design law ID and choices must be non-empty.")
        labels = tuple(choice.label for choice in self.choices)
        if len(set(labels)) != len(labels):
            raise ValueError("Design law contains duplicate choice labels.")
        total = sum(choice.weight for choice in self.choices)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Design law {self.law_id!r} weights must sum to one, got {total!r}."
            )

    @classmethod
    def create(
        cls,
        law_id: str,
        choices: tuple[DesignChoice, ...],
        metadata: Mapping[str, Any] | None = None,
    ) -> "DesignLaw":
        return cls(
            law_id=str(law_id),
            choices=tuple(choices),
            metadata=freeze_mapping({} if metadata is None else metadata),
        )

    def metadata_dict(self) -> dict[str, Any]:
        return thaw_mapping(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "law_id": self.law_id,
            "choices": [choice.to_dict() for choice in self.choices],
            "metadata": self.metadata_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DesignLaw":
        return cls.create(
            law_id=str(data["law_id"]),
            choices=tuple(
                DesignChoice.from_dict(choice) for choice in data["choices"]
            ),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class FamilyDesign:
    role: str
    family_id: str
    version: str
    partition: str
    conditional_weight: float | None
    weight_scope: str | None
    metadata: tuple[tuple[str, FrozenValue], ...] = ()

    def __post_init__(self) -> None:
        if not self.role or not self.family_id or not self.version:
            raise ValueError("Family design identity must be non-empty.")
        if self.partition not in {"train", "optional", "held_out", "excluded"}:
            raise ValueError("Unknown family design partition.")
        if self.partition == "train":
            if (
                self.conditional_weight is None
                or not math.isfinite(self.conditional_weight)
                or self.conditional_weight <= 0.0
                or not self.weight_scope
            ):
                raise ValueError(
                    "Train families require positive conditional mass and a scope."
                )
        elif self.conditional_weight is not None:
            if (
                not math.isfinite(self.conditional_weight)
                or self.conditional_weight <= 0.0
                or not self.weight_scope
            ):
                raise ValueError("Partition family mass must be positive when present.")

    @classmethod
    def create(
        cls,
        *,
        role: str,
        family_id: str,
        version: str,
        partition: str,
        conditional_weight: float | None = None,
        weight_scope: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "FamilyDesign":
        return cls(
            role=str(role),
            family_id=str(family_id),
            version=str(version),
            partition=str(partition),
            conditional_weight=(
                None if conditional_weight is None else float(conditional_weight)
            ),
            weight_scope=None if weight_scope is None else str(weight_scope),
            metadata=freeze_mapping({} if metadata is None else metadata),
        )

    def metadata_dict(self) -> dict[str, Any]:
        return thaw_mapping(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "family_id": self.family_id,
            "version": self.version,
            "partition": self.partition,
            "conditional_weight": self.conditional_weight,
            "weight_scope": self.weight_scope,
            "metadata": self.metadata_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FamilyDesign":
        return cls.create(
            role=str(data["role"]),
            family_id=str(data["family_id"]),
            version=str(data["version"]),
            partition=str(data["partition"]),
            conditional_weight=(
                None
                if data.get("conditional_weight") is None
                else float(data["conditional_weight"])
            ),
            weight_scope=data.get("weight_scope"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class ActivePriorManifest:
    manifest_version: str
    registry_snapshot_digest: str
    families: tuple[FamilyDesign, ...]
    laws: tuple[DesignLaw, ...]
    caps: tuple[tuple[str, FrozenValue], ...]
    required_admissions: tuple[str, ...]
    admission_certificates: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.manifest_version:
            raise ValueError("Active manifest version must be non-empty.")
        if len(self.registry_snapshot_digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.registry_snapshot_digest
        ):
            raise ValueError("Registry snapshot digest must be a SHA-256 digest.")
        family_ids = tuple(
            (family.role, family.family_id, family.version)
            for family in self.families
        )
        if len(set(family_ids)) != len(family_ids):
            raise ValueError("Family partitions must be structurally disjoint.")
        law_ids = tuple(law.law_id for law in self.laws)
        if len(set(law_ids)) != len(law_ids):
            raise ValueError("Active manifest contains duplicate design laws.")
        if len(set(self.required_admissions)) != len(self.required_admissions):
            raise ValueError("Required admission IDs must be unique.")
        certificate_ids = tuple(item[0] for item in self.admission_certificates)
        if len(set(certificate_ids)) != len(certificate_ids):
            raise ValueError("Admission certificate IDs must be unique.")
        for component_id, certificate_hash in self.admission_certificates:
            if not component_id or len(certificate_hash) != 64 or any(
                character not in "0123456789abcdef"
                for character in certificate_hash
            ):
                raise ValueError("Admission certificate entry is malformed.")

    @classmethod
    def create(
        cls,
        *,
        manifest_version: str,
        registry_snapshot_digest: str,
        families: tuple[FamilyDesign, ...],
        laws: tuple[DesignLaw, ...],
        caps: Mapping[str, Any],
        required_admissions: tuple[str, ...],
        admission_certificates: tuple[tuple[str, str], ...] = (),
    ) -> "ActivePriorManifest":
        return cls(
            manifest_version=str(manifest_version),
            registry_snapshot_digest=str(registry_snapshot_digest),
            families=tuple(families),
            laws=tuple(laws),
            caps=freeze_mapping(caps),
            required_admissions=tuple(str(item) for item in required_admissions),
            admission_certificates=tuple(
                (str(component_id), str(certificate_hash))
                for component_id, certificate_hash in admission_certificates
            ),
        )

    @property
    def training_ready(self) -> bool:
        certified = {item[0] for item in self.admission_certificates}
        return bool(self.required_admissions) and certified == set(
            self.required_admissions
        )

    def require_training_ready(self) -> None:
        if not self.training_ready:
            raise RuntimeError(
                "Active-v1 sampling requires all required admission certificates."
            )

    def _legacy_sampler_choices(self) -> tuple[FamilyChoice, ...]:
        self.require_training_ready()
        raise RuntimeError(
            "Active-v1 design families are not bound to the legacy P1-P3 sampler."
        )

    @property
    def covariate_choices(self) -> tuple[FamilyChoice, ...]:
        return self._legacy_sampler_choices()

    @property
    def propensity_choices(self) -> tuple[FamilyChoice, ...]:
        return self._legacy_sampler_choices()

    @property
    def outcome_choices(self) -> tuple[FamilyChoice, ...]:
        return self._legacy_sampler_choices()

    def family_partition(self, partition: str) -> tuple[FamilyDesign, ...]:
        if partition not in {"train", "optional", "held_out", "excluded"}:
            raise ValueError("Unknown family partition.")
        return tuple(
            family for family in self.families if family.partition == partition
        )

    def law(self, law_id: str) -> DesignLaw:
        for law in self.laws:
            if law.law_id == law_id:
                return law
        raise KeyError(f"Unknown design law: {law_id!r}")

    def law_weights(self, law_id: str) -> dict[str, float]:
        return {
            choice.label: choice.weight
            for choice in self.law(law_id).choices
        }

    def caps_dict(self) -> dict[str, Any]:
        return thaw_mapping(self.caps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "registry_snapshot_digest": self.registry_snapshot_digest,
            "families": [family.to_dict() for family in self.families],
            "laws": [law.to_dict() for law in self.laws],
            "caps": self.caps_dict(),
            "required_admissions": list(self.required_admissions),
            "admission_certificates": [
                {
                    "component_id": component_id,
                    "certificate_hash": certificate_hash,
                }
                for component_id, certificate_hash in self.admission_certificates
            ],
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ActivePriorManifest":
        return cls.create(
            manifest_version=str(data["manifest_version"]),
            registry_snapshot_digest=str(data["registry_snapshot_digest"]),
            families=tuple(
                FamilyDesign.from_dict(family) for family in data["families"]
            ),
            laws=tuple(DesignLaw.from_dict(law) for law in data["laws"]),
            caps=data["caps"],
            required_admissions=tuple(
                str(item) for item in data["required_admissions"]
            ),
            admission_certificates=tuple(
                (
                    str(item["component_id"]),
                    str(item["certificate_hash"]),
                )
                for item in data.get("admission_certificates", ())
            ),
        )
