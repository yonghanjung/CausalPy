"""Frozen scalar-function specifications over exact typed feature maps."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Mapping

import numpy as np

from ..features import FrozenFeatureMapSpec, TypedFeatureMap
from ..schema import TypedCovariateBatch, VariableType
from ..specs import canonical_hash, canonical_json


_VERSION = "1.0.0"
_FAMILIES = {
    "bdpfn.scalar.constant",
    "bdpfn.scalar.projection",
    "bdpfn.scalar.sparse_affine",
    "bdpfn.scalar.dense_affine",
    "bdpfn.scalar.categorical_lookup",
}
_AMPLITUDE_LAWS = {
    "function": ((0.5, 0.25), (1.0, 0.50), (2.0, 0.25)),
    "treatment_effect": ((0.25, 0.20), (0.5, 0.40), (1.0, 0.30), (2.0, 0.10)),
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SPEC_FIELDS = frozenset(
    {
        "version",
        "family_id",
        "schema_hash",
        "feature_map_hash",
        "amplitude_mode",
        "amplitude",
        "variable_indices",
        "feature_indices",
        "coefficients",
        "intercept",
        "lookup_values",
    }
)


def _indices(values: tuple[int, ...], label: str) -> tuple[int, ...]:
    result = tuple(values)
    if any(type(value) is not int or value < 0 for value in result):
        raise ValueError(f"Scalar {label} must be non-negative integers.")
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"Scalar {label} must be strictly increasing and unique.")
    return result


@dataclass(frozen=True)
class ScalarFunctionSpec:
    version: str
    family_id: str
    schema_hash: str
    feature_map_hash: str
    amplitude_mode: str
    amplitude: float
    variable_indices: tuple[int, ...]
    feature_indices: tuple[int, ...]
    coefficients: tuple[float, ...]
    intercept: float
    lookup_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.version != _VERSION or self.family_id not in _FAMILIES:
            raise ValueError("Unsupported scalar family identity or version.")
        if not _SHA256.fullmatch(self.schema_hash) or not _SHA256.fullmatch(
            self.feature_map_hash
        ):
            raise ValueError("Scalar schema and feature-map hashes must be SHA-256 IDs.")
        if self.amplitude_mode not in _AMPLITUDE_LAWS:
            raise ValueError("Unknown scalar amplitude mode.")
        amplitude = float(self.amplitude)
        intercept = float(self.intercept)
        coefficients = tuple(float(value) for value in self.coefficients)
        lookup_values = tuple(float(value) for value in self.lookup_values)
        if amplitude not in {value for value, _ in amplitude_law(self.amplitude_mode)}:
            raise ValueError("Scalar amplitude is outside its frozen mode domain.")
        if not math.isfinite(intercept) or any(
            not math.isfinite(value) for value in coefficients + lookup_values
        ):
            raise ValueError("Scalar parameters must be finite.")
        variables = _indices(tuple(self.variable_indices), "variable indices")
        features = _indices(tuple(self.feature_indices), "feature indices")
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "intercept", intercept)
        object.__setattr__(self, "variable_indices", variables)
        object.__setattr__(self, "feature_indices", features)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "lookup_values", lookup_values)
        if len(coefficients) != len(features):
            raise ValueError("Scalar coefficients must match its engineered features.")
        self._validate_family_fields()

    def _validate_family_fields(self) -> None:
        empty = not self.variable_indices and not self.feature_indices
        if self.family_id == "bdpfn.scalar.constant":
            if not empty or self.coefficients or self.lookup_values:
                raise ValueError("Constant scalar cannot contain indexed parameters.")
        elif self.family_id == "bdpfn.scalar.projection":
            if (
                len(self.variable_indices) != 1
                or len(self.feature_indices) != 1
                or len(self.coefficients) != 1
                or self.lookup_values
            ):
                raise ValueError("Projection scalar requires exactly one engineered feature.")
        elif self.family_id in {
            "bdpfn.scalar.sparse_affine",
            "bdpfn.scalar.dense_affine",
        }:
            if not self.variable_indices or not self.feature_indices or self.lookup_values:
                raise ValueError("Affine scalar requires indexed coefficients only.")
        elif (
            len(self.variable_indices) != 1
            or self.feature_indices
            or self.coefficients
            or not self.lookup_values
        ):
            raise ValueError("Categorical lookup requires one variable and a raw table.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "family_id": self.family_id,
            "schema_hash": self.schema_hash,
            "feature_map_hash": self.feature_map_hash,
            "amplitude_mode": self.amplitude_mode,
            "amplitude": self.amplitude,
            "variable_indices": list(self.variable_indices),
            "feature_indices": list(self.feature_indices),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "lookup_values": list(self.lookup_values),
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def spec_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScalarFunctionSpec":
        if set(data) != _SPEC_FIELDS:
            raise ValueError("Scalar specification fields must match the frozen schema.")
        return cls(
            version=str(data["version"]),
            family_id=str(data["family_id"]),
            schema_hash=str(data["schema_hash"]),
            feature_map_hash=str(data["feature_map_hash"]),
            amplitude_mode=str(data["amplitude_mode"]),
            amplitude=float(data["amplitude"]),
            variable_indices=tuple(int(value) for value in data["variable_indices"]),
            feature_indices=tuple(int(value) for value in data["feature_indices"]),
            coefficients=tuple(float(value) for value in data["coefficients"]),
            intercept=float(data["intercept"]),
            lookup_values=tuple(float(value) for value in data["lookup_values"]),
        )

    @classmethod
    def from_json(cls, encoded: str) -> "ScalarFunctionSpec":
        return cls.from_dict(json.loads(encoded))


def amplitude_law(mode: str) -> tuple[tuple[float, float], ...]:
    try:
        return _AMPLITUDE_LAWS[mode]
    except KeyError as error:
        raise ValueError("Unknown scalar amplitude mode.") from error


def _sample_amplitude(rng: np.random.Generator, mode: str) -> float:
    law = amplitude_law(mode)
    values = np.asarray([value for value, _ in law])
    probabilities = np.asarray([probability for _, probability in law])
    return float(rng.choice(values, p=probabilities))


def _base_fields(
    feature_spec: FrozenFeatureMapSpec,
    rng: np.random.Generator,
    amplitude_mode: str,
) -> dict[str, Any]:
    if not isinstance(feature_spec, FrozenFeatureMapSpec):
        raise TypeError("Scalar sampling requires a FrozenFeatureMapSpec.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("Scalar sampling requires a local NumPy Generator.")
    return {
        "version": _VERSION,
        "schema_hash": feature_spec.schema.schema_hash,
        "feature_map_hash": feature_spec.spec_hash,
        "amplitude_mode": amplitude_mode,
        "amplitude": _sample_amplitude(rng, amplitude_mode),
    }


def sample_constant_spec(
    feature_spec: FrozenFeatureMapSpec,
    rng: np.random.Generator,
    amplitude_mode: str = "function",
) -> ScalarFunctionSpec:
    return ScalarFunctionSpec(
        **_base_fields(feature_spec, rng, amplitude_mode),
        family_id="bdpfn.scalar.constant",
        variable_indices=(),
        feature_indices=(),
        coefficients=(),
        intercept=float(rng.normal()),
        lookup_values=(),
    )


def _variable_for_feature(feature_spec: FrozenFeatureMapSpec, feature_index: int) -> int:
    for block in feature_spec.blocks:
        if block.output_start <= feature_index < block.output_start + block.output_size:
            return block.variable_index
    raise ValueError("Engineered feature index lies outside the feature map.")


def sample_projection_spec(
    feature_spec: FrozenFeatureMapSpec,
    rng: np.random.Generator,
    amplitude_mode: str = "function",
) -> ScalarFunctionSpec:
    base = _base_fields(feature_spec, rng, amplitude_mode)
    feature_index = int(rng.integers(0, feature_spec.output_dimension))
    return ScalarFunctionSpec(
        **base,
        family_id="bdpfn.scalar.projection",
        variable_indices=(_variable_for_feature(feature_spec, feature_index),),
        feature_indices=(feature_index,),
        coefficients=(float(rng.normal()),),
        intercept=float(rng.normal()),
        lookup_values=(),
    )


def _expected_features(
    feature_spec: FrozenFeatureMapSpec,
    variable_indices: tuple[int, ...],
) -> tuple[int, ...]:
    return tuple(
        feature_index
        for variable_index in variable_indices
        for feature_index in range(
            feature_spec.blocks[variable_index].output_start,
            feature_spec.blocks[variable_index].output_start
            + feature_spec.blocks[variable_index].output_size,
        )
    )


def _sample_affine_spec(
    family_id: str,
    feature_spec: FrozenFeatureMapSpec,
    rng: np.random.Generator,
    amplitude_mode: str,
    variable_indices: tuple[int, ...],
) -> ScalarFunctionSpec:
    feature_indices = _expected_features(feature_spec, variable_indices)
    coefficient_count = len(feature_indices)
    coefficients = rng.normal(
        scale=1.0 / math.sqrt(coefficient_count),
        size=coefficient_count,
    )
    return ScalarFunctionSpec(
        **_base_fields(feature_spec, rng, amplitude_mode),
        family_id=family_id,
        variable_indices=variable_indices,
        feature_indices=feature_indices,
        coefficients=tuple(float(value) for value in coefficients),
        intercept=float(rng.normal()),
        lookup_values=(),
    )


def sample_sparse_affine_spec(
    feature_spec: FrozenFeatureMapSpec,
    rng: np.random.Generator,
    amplitude_mode: str = "function",
) -> ScalarFunctionSpec:
    if not isinstance(feature_spec, FrozenFeatureMapSpec):
        raise TypeError("Scalar sampling requires a FrozenFeatureMapSpec.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("Scalar sampling requires a local NumPy Generator.")
    active_count = int(rng.integers(1, min(10, feature_spec.schema.dimension) + 1))
    selected = rng.choice(
        feature_spec.schema.dimension,
        size=active_count,
        replace=False,
    )
    variable_indices = tuple(sorted(int(value) for value in selected))
    return _sample_affine_spec(
        "bdpfn.scalar.sparse_affine",
        feature_spec,
        rng,
        amplitude_mode,
        variable_indices,
    )


def sample_dense_affine_spec(
    feature_spec: FrozenFeatureMapSpec,
    rng: np.random.Generator,
    amplitude_mode: str = "function",
) -> ScalarFunctionSpec:
    if not isinstance(feature_spec, FrozenFeatureMapSpec):
        raise TypeError("Scalar sampling requires a FrozenFeatureMapSpec.")
    return _sample_affine_spec(
        "bdpfn.scalar.dense_affine",
        feature_spec,
        rng,
        amplitude_mode,
        tuple(range(feature_spec.schema.dimension)),
    )


def sample_categorical_lookup_spec(
    feature_spec: FrozenFeatureMapSpec,
    rng: np.random.Generator,
    amplitude_mode: str = "function",
) -> ScalarFunctionSpec:
    if not isinstance(feature_spec, FrozenFeatureMapSpec):
        raise TypeError("Scalar sampling requires a FrozenFeatureMapSpec.")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("Scalar sampling requires a local NumPy Generator.")
    candidates = tuple(
        index
        for index, variable in enumerate(feature_spec.schema.variables)
        if variable.variable_type is VariableType.CATEGORICAL
    )
    if not candidates:
        raise ValueError("Categorical lookup requires a nominal categorical variable.")
    variable_index = int(rng.choice(np.asarray(candidates, dtype=np.int64)))
    cardinality = feature_spec.schema.variables[variable_index].cardinality
    offsets = rng.normal(scale=1.0 / math.sqrt(cardinality), size=cardinality)
    return ScalarFunctionSpec(
        **_base_fields(feature_spec, rng, amplitude_mode),
        family_id="bdpfn.scalar.categorical_lookup",
        variable_indices=(variable_index,),
        feature_indices=(),
        coefficients=(),
        intercept=float(rng.normal()),
        lookup_values=tuple(float(value) for value in offsets),
    )


@dataclass(frozen=True)
class ScalarFunction:
    specification: ScalarFunctionSpec
    feature_spec: FrozenFeatureMapSpec

    def evaluate(self, batch: TypedCovariateBatch) -> np.ndarray:
        features = TypedFeatureMap(self.feature_spec).transform(batch)
        spec = self.specification
        if spec.family_id == "bdpfn.scalar.constant":
            result = np.full(batch.n_rows, spec.intercept, dtype=np.float64)
        elif spec.family_id == "bdpfn.scalar.categorical_lookup":
            variable_index = spec.variable_indices[0]
            result = spec.intercept + np.asarray(spec.lookup_values)[
                batch.columns[variable_index]
            ]
        else:
            result = spec.intercept + features[:, spec.feature_indices] @ np.asarray(
                spec.coefficients
            )
        result = np.asarray(spec.amplitude * result, dtype=np.float64)
        if not np.isfinite(result).all():
            raise FloatingPointError("Scalar evaluation produced nonfinite values.")
        result.setflags(write=False)
        return result


def build_scalar_function(
    specification: ScalarFunctionSpec,
    feature_spec: FrozenFeatureMapSpec,
) -> ScalarFunction:
    if specification.schema_hash != feature_spec.schema.schema_hash:
        raise ValueError("Scalar schema hash disagrees with the feature map.")
    if specification.feature_map_hash != feature_spec.spec_hash:
        raise ValueError("Scalar feature-map hash disagrees with the feature map.")
    if any(index >= feature_spec.schema.dimension for index in specification.variable_indices):
        raise ValueError("Scalar variable index lies outside the schema.")
    if any(index >= feature_spec.output_dimension for index in specification.feature_indices):
        raise ValueError("Scalar engineered feature index lies outside the feature map.")
    if specification.family_id == "bdpfn.scalar.projection":
        if _variable_for_feature(
            feature_spec, specification.feature_indices[0]
        ) != specification.variable_indices[0]:
            raise ValueError("Projection feature disagrees with its semantic variable.")
    elif specification.family_id in {
        "bdpfn.scalar.sparse_affine",
        "bdpfn.scalar.dense_affine",
    }:
        if specification.feature_indices != _expected_features(
            feature_spec, specification.variable_indices
        ):
            raise ValueError("Affine scalar must contain whole semantic feature blocks.")
        if (
            specification.family_id == "bdpfn.scalar.dense_affine"
            and specification.variable_indices
            != tuple(range(feature_spec.schema.dimension))
        ):
            raise ValueError("Dense affine scalar must use every semantic variable.")
        if (
            specification.family_id == "bdpfn.scalar.sparse_affine"
            and len(specification.variable_indices)
            > min(10, feature_spec.schema.dimension)
        ):
            raise ValueError("Sparse affine scalar exceeds its semantic-variable cap.")
    elif specification.family_id == "bdpfn.scalar.categorical_lookup":
        variable = feature_spec.schema.variables[specification.variable_indices[0]]
        if variable.variable_type is not VariableType.CATEGORICAL:
            raise ValueError("Categorical lookup requires a nominal categorical variable.")
        if len(specification.lookup_values) != variable.cardinality:
            raise ValueError("Categorical lookup table disagrees with its cardinality.")
    return ScalarFunction(specification, feature_spec)
