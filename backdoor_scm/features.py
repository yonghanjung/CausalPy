"""Exact task-frozen feature maps for typed covariates."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping

import numpy as np

from .exceptions import OutOfSupportError
from .families.roots import RootSpec, build_root
from .schema import CovariateSchema, TypedCovariateBatch, VariableSpec, VariableType
from .specs import canonical_hash, canonical_json


@dataclass(frozen=True)
class FeatureBlockSpec:
    variable_index: int
    variable_name: str
    variable_type: VariableType
    cardinality: int | None
    encoding: str
    output_start: int
    output_size: int
    centers: tuple[float, ...]
    scales: tuple[float, ...]

    def __post_init__(self) -> None:
        variable_type = VariableType(self.variable_type)
        object.__setattr__(self, "variable_type", variable_type)
        object.__setattr__(self, "centers", tuple(float(value) for value in self.centers))
        object.__setattr__(self, "scales", tuple(float(value) for value in self.scales))
        if type(self.variable_index) is not int or self.variable_index < 0:
            raise ValueError("Feature block variable index must be non-negative.")
        if not self.variable_name or not self.encoding:
            raise ValueError("Feature block identity must be non-empty.")
        if type(self.output_start) is not int or self.output_start < 0:
            raise ValueError("Feature block output start must be non-negative.")
        if type(self.output_size) is not int or self.output_size <= 0:
            raise ValueError("Feature block output size must be positive.")
        if len(self.centers) != self.output_size or len(self.scales) != self.output_size:
            raise ValueError("Feature block moments disagree with its output size.")
        if any(not math.isfinite(value) for value in self.centers):
            raise ValueError("Feature block centers must be finite.")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.scales):
            raise ValueError("Feature block scales must be finite and positive.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "variable_index": self.variable_index,
            "variable_name": self.variable_name,
            "variable_type": self.variable_type.value,
            "cardinality": self.cardinality,
            "encoding": self.encoding,
            "output_start": self.output_start,
            "output_size": self.output_size,
            "centers": list(self.centers),
            "scales": list(self.scales),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FeatureBlockSpec":
        return cls(
            variable_index=int(data["variable_index"]),
            variable_name=str(data["variable_name"]),
            variable_type=VariableType(str(data["variable_type"])),
            cardinality=None if data.get("cardinality") is None else int(data["cardinality"]),
            encoding=str(data["encoding"]),
            output_start=int(data["output_start"]),
            output_size=int(data["output_size"]),
            centers=tuple(float(value) for value in data["centers"]),
            scales=tuple(float(value) for value in data["scales"]),
        )


def _expected_block(
    variable_index: int,
    variable: VariableSpec,
    root_spec: RootSpec,
    output_start: int,
) -> FeatureBlockSpec:
    root = build_root(root_spec)
    if variable.variable_type is VariableType.CATEGORICAL:
        probabilities = root.pmf
        centers = tuple(float(value) for value in probabilities)
        scales = tuple(
            math.sqrt(float(value) * (1.0 - float(value)))
            for value in probabilities
        )
        encoding = "categorical_centered_scaled_one_hot"
        output_size = variable.cardinality
    else:
        centers = (root.mean,)
        scales = (math.sqrt(root.variance),)
        encoding = f"{variable.variable_type.value}_exact_standardized"
        output_size = 1
    return FeatureBlockSpec(
        variable_index=variable_index,
        variable_name=variable.name,
        variable_type=variable.variable_type,
        cardinality=variable.cardinality,
        encoding=encoding,
        output_start=output_start,
        output_size=output_size,
        centers=centers,
        scales=scales,
    )


@dataclass(frozen=True)
class FrozenFeatureMapSpec:
    version: str
    schema: CovariateSchema
    root_specs: tuple[RootSpec, ...]
    blocks: tuple[FeatureBlockSpec, ...]

    def __post_init__(self) -> None:
        if self.version != "1.0.0":
            raise ValueError("Unsupported feature-map version.")
        roots = tuple(self.root_specs)
        blocks = tuple(self.blocks)
        object.__setattr__(self, "root_specs", roots)
        object.__setattr__(self, "blocks", blocks)
        if len(roots) != self.schema.dimension or len(blocks) != self.schema.dimension:
            raise ValueError("Feature-map components disagree with the schema dimension.")
        expected = []
        output_start = 0
        for index, (variable, root_spec) in enumerate(zip(self.schema.variables, roots)):
            if (
                variable.variable_type is not root_spec.variable_type
                or variable.cardinality != root_spec.cardinality
            ):
                raise ValueError("Feature-map root disagrees with its schema variable.")
            block = _expected_block(index, variable, root_spec, output_start)
            expected.append(block)
            output_start += block.output_size
        if blocks != tuple(expected):
            raise ValueError("Feature-map block metadata is not exact for its root laws.")

    @classmethod
    def from_roots(
        cls,
        schema: CovariateSchema,
        root_specs: tuple[RootSpec, ...],
    ) -> "FrozenFeatureMapSpec":
        roots = tuple(root_specs)
        blocks = []
        output_start = 0
        for index, (variable, root_spec) in enumerate(zip(schema.variables, roots)):
            block = _expected_block(index, variable, root_spec, output_start)
            blocks.append(block)
            output_start += block.output_size
        return cls("1.0.0", schema, roots, tuple(blocks))

    @property
    def output_dimension(self) -> int:
        return sum(block.output_size for block in self.blocks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "schema": self.schema.to_dict(),
            "root_specs": [root.to_dict() for root in self.root_specs],
            "blocks": [block.to_dict() for block in self.blocks],
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def spec_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrozenFeatureMapSpec":
        return cls(
            version=str(data["version"]),
            schema=CovariateSchema.from_dict(data["schema"]),
            root_specs=tuple(RootSpec.from_dict(item) for item in data["root_specs"]),
            blocks=tuple(FeatureBlockSpec.from_dict(item) for item in data["blocks"]),
        )

    @classmethod
    def from_json(cls, encoded: str) -> "FrozenFeatureMapSpec":
        return cls.from_dict(json.loads(encoded))


@dataclass(frozen=True)
class TypedFeatureMap:
    specification: FrozenFeatureMapSpec

    def transform(self, batch: TypedCovariateBatch) -> np.ndarray:
        if not isinstance(batch, TypedCovariateBatch):
            raise TypeError("Typed feature transformation requires a TypedCovariateBatch.")
        if batch.schema != self.specification.schema:
            raise ValueError("Feature-map schema disagrees with the typed batch.")
        result = np.empty(
            (batch.n_rows, self.specification.output_dimension),
            dtype=np.float64,
        )
        for root_spec, block, column in zip(
            self.specification.root_specs,
            self.specification.blocks,
            batch.columns,
        ):
            if not build_root(root_spec).contains(column):
                raise OutOfSupportError("Covariate column lies outside its frozen root support.")
            output_slice = slice(block.output_start, block.output_start + block.output_size)
            if block.variable_type is VariableType.CATEGORICAL:
                indicators = np.equal(
                    np.asarray(column)[:, None],
                    np.arange(block.cardinality)[None, :],
                ).astype(np.float64)
                result[:, output_slice] = (
                    indicators - np.asarray(block.centers)
                ) / np.asarray(block.scales)
            else:
                result[:, output_slice] = (
                    np.asarray(column, dtype=np.float64)[:, None] - block.centers[0]
                ) / block.scales[0]
        if not np.isfinite(result).all():
            raise FloatingPointError("Typed feature transformation produced nonfinite values.")
        result.setflags(write=False)
        return result
