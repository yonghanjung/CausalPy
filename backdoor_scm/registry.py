"""Explicit, role-aware family registry."""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from .protocols import FamilyFactory
from .specs import canonical_hash


class ComponentRole(str, Enum):
    COVARIATE = "covariate"
    PROPENSITY = "propensity"
    OUTCOME = "outcome"


class FamilyRegistry:
    def __init__(self) -> None:
        self._factories: dict[tuple[ComponentRole, str, str], FamilyFactory] = {}

    def register(self, factory: FamilyFactory) -> None:
        try:
            role = ComponentRole(factory.role)
        except (AttributeError, ValueError) as error:
            raise ValueError("Factory has an invalid component role.") from error
        if not factory.family_id or not factory.version:
            raise ValueError("Factory ID and version must be non-empty.")
        if not factory.semantic_class_id:
            raise ValueError("Factory semantic class ID must be non-empty.")
        key = (role, factory.family_id, factory.version)
        if key in self._factories:
            raise ValueError(f"Family already registered: {key!r}")
        self._factories[key] = factory

    def resolve(
        self,
        role: ComponentRole | str,
        family_id: str,
        version: str,
    ) -> FamilyFactory:
        key = (ComponentRole(role), family_id, version)
        try:
            return self._factories[key]
        except KeyError as error:
            raise KeyError(f"Unknown family: {key!r}") from error

    def keys(self) -> tuple[tuple[ComponentRole, str, str], ...]:
        return tuple(sorted(self._factories, key=lambda item: tuple(map(str, item))))

    def snapshot_digest(self) -> str:
        entries = []
        for key in self.keys():
            factory = self._factories[key]
            capability = getattr(factory, "truth_capability", None)
            entries.append(
                {
                    "role": key[0].value,
                    "family_id": key[1],
                    "version": key[2],
                    "semantic_class_id": factory.semantic_class_id,
                    "outcome_type": factory.outcome_type,
                    "capability_hash": (
                        None
                        if capability is None
                        else capability.capability_hash
                    ),
                }
            )
        return canonical_hash({"registry_entries": entries})

    def __len__(self) -> int:
        return len(self._factories)
