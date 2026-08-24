"""Frozen typed covariate schemas and active-v1 schema sampling."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Mapping

import numpy as np

from .specs import canonical_hash, canonical_json


class VariableType(str, Enum):
    CONTINUOUS = "continuous"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"


@dataclass(frozen=True)
class VariableSpec:
    name: str
    variable_type: VariableType
    cardinality: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Variable name must be non-empty.")
        try:
            variable_type = VariableType(self.variable_type)
        except ValueError as error:
            raise ValueError("Unknown variable type.") from error
        object.__setattr__(self, "variable_type", variable_type)
        if variable_type is VariableType.CONTINUOUS:
            if self.cardinality is not None:
                raise ValueError("Continuous variables do not have cardinality.")
        elif variable_type is VariableType.BINARY:
            if self.cardinality != 2:
                raise ValueError("Binary variables must have cardinality two.")
        elif (
            type(self.cardinality) is not int
            or self.cardinality < 2
            or self.cardinality > 256
        ):
            raise ValueError("Finite variable cardinality must lie in [2, 256].")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "variable_type": self.variable_type.value,
            "cardinality": self.cardinality,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VariableSpec":
        return cls(
            name=str(data["name"]),
            variable_type=VariableType(str(data["variable_type"])),
            cardinality=(
                None if data.get("cardinality") is None else int(data["cardinality"])
            ),
        )


_PROFILES = {
    "continuous_only",
    "binary_only",
    "categorical_or_ordinal_only",
    "mixed",
}


@dataclass(frozen=True)
class CovariateSchema:
    schema_version: str
    profile: str
    dimension_stratum: str
    sampling_attempt: int
    variables: tuple[VariableSpec, ...]

    def __post_init__(self) -> None:
        if not self.schema_version or self.profile not in _PROFILES:
            raise ValueError("Schema version and profile must be valid.")
        if not self.dimension_stratum:
            raise ValueError("Dimension stratum must be non-empty.")
        if type(self.sampling_attempt) is not int or self.sampling_attempt < 0:
            raise ValueError("Sampling attempt must be a non-negative integer.")
        variables = tuple(self.variables)
        object.__setattr__(self, "variables", variables)
        if not 1 <= len(variables) <= 99:
            raise ValueError("Covariate dimension must lie in [1, 99].")
        stratum_bounds = {
            "1": (1, 1),
            "2-5": (2, 5),
            "6-10": (6, 10),
            "11-20": (11, 20),
            "21-50": (21, 50),
            "51-99": (51, 99),
        }
        try:
            stratum_low, stratum_high = stratum_bounds[self.dimension_stratum]
        except KeyError as error:
            raise ValueError("Unknown covariate dimension stratum.") from error
        if not stratum_low <= len(variables) <= stratum_high:
            raise ValueError("Dimension stratum disagrees with the schema dimension.")
        names = tuple(variable.name for variable in variables)
        if len(set(names)) != len(names):
            raise ValueError("Variable names must be unique.")
        kinds = tuple(variable.variable_type for variable in variables)
        if self.profile == "continuous_only" and any(
            kind is not VariableType.CONTINUOUS for kind in kinds
        ):
            raise ValueError("Continuous-only schema contains a non-continuous variable.")
        if self.profile == "binary_only" and any(
            kind is not VariableType.BINARY for kind in kinds
        ):
            raise ValueError("Binary-only schema contains a non-binary variable.")
        if self.profile == "categorical_or_ordinal_only" and any(
            kind not in {VariableType.CATEGORICAL, VariableType.ORDINAL}
            for kind in kinds
        ):
            raise ValueError("Categorical/ordinal schema contains an incompatible type.")
        if self.profile == "mixed" and (len(kinds) < 2 or len(set(kinds)) < 2):
            raise ValueError("Mixed schema requires at least two semantic types.")

    @property
    def dimension(self) -> int:
        return len(self.variables)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "dimension_stratum": self.dimension_stratum,
            "sampling_attempt": self.sampling_attempt,
            "variables": [variable.to_dict() for variable in self.variables],
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def schema_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CovariateSchema":
        return cls(
            schema_version=str(data["schema_version"]),
            profile=str(data["profile"]),
            dimension_stratum=str(data["dimension_stratum"]),
            sampling_attempt=int(data["sampling_attempt"]),
            variables=tuple(
                VariableSpec.from_dict(item) for item in data["variables"]
            ),
        )

    @classmethod
    def from_json(cls, encoded: str) -> "CovariateSchema":
        return cls.from_dict(json.loads(encoded))


def _normalize_column(variable: VariableSpec, values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("Typed covariate columns must be one-dimensional.")
    if variable.variable_type is VariableType.CONTINUOUS:
        try:
            normalized = np.asarray(array, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("Continuous values must be numeric.") from error
        if not np.isfinite(normalized).all():
            raise ValueError("Continuous values must be finite.")
    else:
        try:
            numeric = np.asarray(array, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("Discrete values must be numeric integer codes.") from error
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError("Discrete values must be finite integer-valued codes.")
        normalized = numeric.astype(np.int64)
        if (normalized < 0).any() or (normalized >= variable.cardinality).any():
            raise ValueError("Discrete value lies outside its declared cardinality.")
    result = np.array(normalized, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class TypedCovariateBatch:
    schema: CovariateSchema
    columns: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if len(self.columns) != self.schema.dimension:
            raise ValueError("Typed batch column count disagrees with its schema.")
        normalized = tuple(
            _normalize_column(variable, column)
            for variable, column in zip(self.schema.variables, self.columns)
        )
        row_counts = {column.shape[0] for column in normalized}
        if len(row_counts) > 1:
            raise ValueError("Typed batch columns have inconsistent row counts.")
        object.__setattr__(self, "columns", normalized)

    @property
    def n_rows(self) -> int:
        return 0 if not self.columns else self.columns[0].shape[0]

    def to_matrix(self) -> np.ndarray:
        """Return the one numeric boundary representation used by task assembly."""

        result = np.column_stack(self.columns).astype(np.float64, copy=False)
        result = np.array(result, dtype=np.float64, copy=True, order="C")
        result.setflags(write=False)
        return result

    @classmethod
    def from_matrix(
        cls,
        schema: CovariateSchema,
        values: np.ndarray,
    ) -> "TypedCovariateBatch":
        matrix = np.asarray(values)
        if matrix.ndim != 2 or matrix.shape[1] != schema.dimension:
            raise ValueError("Outer covariate matrix has the wrong shape.")
        return cls(schema, tuple(matrix[:, index] for index in range(schema.dimension)))


@dataclass(frozen=True)
class SchemaSamplingPolicy:
    policy_version: str
    dimension_strata: tuple[tuple[str, int, int, float], ...]
    profile_weights: tuple[tuple[str, float], ...]
    mixed_type_weights: tuple[tuple[str, float], ...]
    categorical_ordinal_weights: tuple[tuple[str, float], ...]
    cardinality_bands: tuple[tuple[str, int, int, float], ...]

    def __post_init__(self) -> None:
        if not self.policy_version:
            raise ValueError("Schema sampling policy version must be non-empty.")
        for weights in (
            self.dimension_weights(),
            self.profile_weights_dict(),
            self.mixed_type_weights_dict(),
            dict(self.categorical_ordinal_weights),
            self.cardinality_band_weights(),
        ):
            if any(not math.isfinite(value) or value <= 0.0 for value in weights.values()):
                raise ValueError("Schema sampling weights must be positive and finite.")
            if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("Schema sampling weights must sum to one.")
        for _, low, high, _ in self.dimension_strata:
            if not 1 <= low <= high <= 99:
                raise ValueError("Dimension stratum is outside [1, 99].")
        for _, low, high, _ in self.cardinality_bands:
            if not 2 <= low <= high <= 30:
                raise ValueError("Training cardinality band is outside [2, 30].")

    def dimension_weights(self) -> dict[str, float]:
        return {label: weight for label, _, _, weight in self.dimension_strata}

    def profile_weights_dict(self) -> dict[str, float]:
        return dict(self.profile_weights)

    def mixed_type_weights_dict(self) -> dict[str, float]:
        return dict(self.mixed_type_weights)

    def cardinality_band_weights(self) -> dict[str, float]:
        return {label: weight for label, _, _, weight in self.cardinality_bands}


def active_v1_schema_policy() -> SchemaSamplingPolicy:
    return SchemaSamplingPolicy(
        policy_version="bdpfn-active-v1-schema-policy-v1",
        dimension_strata=(
            ("1", 1, 1, 1.0 / 6.0),
            ("2-5", 2, 5, 1.0 / 6.0),
            ("6-10", 6, 10, 1.0 / 6.0),
            ("11-20", 11, 20, 1.0 / 6.0),
            ("21-50", 21, 50, 1.0 / 6.0),
            ("51-99", 51, 99, 1.0 / 6.0),
        ),
        profile_weights=(
            ("continuous_only", 0.30),
            ("binary_only", 0.10),
            ("categorical_or_ordinal_only", 0.10),
            ("mixed", 0.50),
        ),
        mixed_type_weights=(
            ("continuous", 0.50),
            ("binary", 0.20),
            ("categorical", 0.15),
            ("ordinal_or_integer", 0.15),
        ),
        categorical_ordinal_weights=(
            ("categorical", 0.50),
            ("ordinal", 0.50),
        ),
        cardinality_bands=(
            ("2", 2, 2, 0.40),
            ("3-9", 3, 9, 0.40),
            ("10-30", 10, 30, 0.20),
        ),
    )


def _weighted_index(weights: tuple[float, ...], rng: np.random.Generator) -> int:
    probabilities = np.asarray(weights, dtype=float)
    probabilities = probabilities / probabilities.sum()
    return int(rng.choice(len(probabilities), p=probabilities))


def _sample_cardinality(
    rng: np.random.Generator,
    policy: SchemaSamplingPolicy,
) -> int:
    band_index = _weighted_index(
        tuple(item[3] for item in policy.cardinality_bands),
        rng,
    )
    _, low, high, _ = policy.cardinality_bands[band_index]
    values = np.arange(low, high + 1, dtype=int)
    probabilities = 1.0 / values.astype(float)
    probabilities = probabilities / probabilities.sum()
    return int(rng.choice(values, p=probabilities))


def sample_schema(
    rng: np.random.Generator,
    policy: SchemaSamplingPolicy | None = None,
    max_attempts: int = 10000,
) -> CovariateSchema:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("Schema sampling requires a local NumPy Generator.")
    if type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("Maximum schema attempts must be positive.")
    policy = active_v1_schema_policy() if policy is None else policy
    for attempt in range(max_attempts):
        stratum_index = _weighted_index(
            tuple(item[3] for item in policy.dimension_strata), rng
        )
        stratum_label, low, high, _ = policy.dimension_strata[stratum_index]
        dimension = int(rng.integers(low, high + 1))
        profile_index = _weighted_index(
            tuple(item[1] for item in policy.profile_weights), rng
        )
        profile = policy.profile_weights[profile_index][0]
        if profile == "mixed" and dimension < 2:
            continue

        if profile == "continuous_only":
            kinds = (VariableType.CONTINUOUS,) * dimension
        elif profile == "binary_only":
            kinds = (VariableType.BINARY,) * dimension
        elif profile == "categorical_or_ordinal_only":
            labels = tuple(item[0] for item in policy.categorical_ordinal_weights)
            weights = tuple(item[1] for item in policy.categorical_ordinal_weights)
            kinds = tuple(
                VariableType(labels[_weighted_index(weights, rng)])
                for _ in range(dimension)
            )
        else:
            labels = tuple(item[0] for item in policy.mixed_type_weights)
            weights = tuple(item[1] for item in policy.mixed_type_weights)
            kind_map = {
                "continuous": VariableType.CONTINUOUS,
                "binary": VariableType.BINARY,
                "categorical": VariableType.CATEGORICAL,
                "ordinal_or_integer": VariableType.ORDINAL,
            }
            kinds = tuple(
                kind_map[labels[_weighted_index(weights, rng)]]
                for _ in range(dimension)
            )
            if len(set(kinds)) < 2:
                continue

        variables = []
        for index, kind in enumerate(kinds):
            if kind is VariableType.CONTINUOUS:
                cardinality = None
            elif kind is VariableType.BINARY:
                cardinality = 2
            else:
                cardinality = _sample_cardinality(rng, policy)
            variables.append(VariableSpec(f"x{index}", kind, cardinality))
        return CovariateSchema(
            schema_version="bdpfn-covariate-schema-v1",
            profile=profile,
            dimension_stratum=stratum_label,
            sampling_attempt=attempt,
            variables=tuple(variables),
        )
    raise RuntimeError("Could not sample a compatible schema within the attempt cap.")
