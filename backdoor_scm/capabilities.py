"""Versioned truth capabilities used by component admission."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Mapping

from .specs import canonical_hash, canonical_json


class TruthLevel(str, Enum):
    ANALYTIC_EXACT = "analytic_exact"
    ENUMERATION_EXACT = "enumeration_exact"
    CERTIFIED_NUMERIC = "certified_numeric"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class TruthCapability:
    capability_version: str
    truth_level: TruthLevel
    moments: tuple[str, ...]
    arbitrary_query: bool
    exact_support: bool
    query_cost_bound: int
    atol: float | None = None
    rtol: float | None = None

    def __post_init__(self) -> None:
        if not self.capability_version:
            raise ValueError("Capability version must be non-empty.")
        try:
            level = TruthLevel(self.truth_level)
        except ValueError as error:
            raise ValueError("Unknown truth capability level.") from error
        object.__setattr__(self, "truth_level", level)

        moments = tuple(str(moment) for moment in self.moments)
        if any(not moment for moment in moments):
            raise ValueError("Moment names must be non-empty.")
        if len(set(moments)) != len(moments):
            raise ValueError("Moment names must be unique.")
        object.__setattr__(self, "moments", moments)
        if type(self.query_cost_bound) is not int or self.query_cost_bound <= 0:
            raise ValueError("Query cost bound must be a positive integer.")

        tolerances = (self.atol, self.rtol)
        if level is TruthLevel.CERTIFIED_NUMERIC:
            if any(
                value is None or not math.isfinite(value) or value < 0.0
                for value in tolerances
            ):
                raise ValueError(
                    "Certified numeric truth requires finite non-negative tolerances."
                )
            if self.atol > 1e-8 or self.rtol > 1e-6:
                raise ValueError(
                    "Certified numeric tolerances exceed the active-v1 contract."
                )
        elif any(value is not None for value in tolerances):
            raise ValueError("Only certified numeric truth declares tolerances.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_version": self.capability_version,
            "truth_level": self.truth_level.value,
            "moments": list(self.moments),
            "arbitrary_query": self.arbitrary_query,
            "exact_support": self.exact_support,
            "query_cost_bound": self.query_cost_bound,
            "atol": self.atol,
            "rtol": self.rtol,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def capability_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TruthCapability":
        return cls(
            capability_version=str(data["capability_version"]),
            truth_level=TruthLevel(str(data["truth_level"])),
            moments=tuple(str(moment) for moment in data["moments"]),
            arbitrary_query=bool(data["arbitrary_query"]),
            exact_support=bool(data["exact_support"]),
            query_cost_bound=int(data["query_cost_bound"]),
            atol=None if data.get("atol") is None else float(data["atol"]),
            rtol=None if data.get("rtol") is None else float(data["rtol"]),
        )

    @classmethod
    def from_json(cls, encoded: str) -> "TruthCapability":
        return cls.from_dict(json.loads(encoded))
