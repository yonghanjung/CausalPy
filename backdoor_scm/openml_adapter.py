"""Explicit, opt-in OpenML materialization for empirical covariate artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Mapping

import numpy as np

from .artifacts import InMemoryArtifactProvider, ResolvedArtifact
from .schema import CovariateSchema, TypedCovariateBatch


@dataclass(frozen=True)
class OpenMLColumnContract:
    schema: CovariateSchema
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    ignore_columns: tuple[str, ...]
    row_id_columns: tuple[str, ...]
    weight_column: str | None = None

    def __post_init__(self) -> None:
        groups = (
            tuple(self.feature_columns),
            tuple(self.target_columns),
            tuple(self.ignore_columns),
            tuple(self.row_id_columns),
        )
        for name, values in zip(
            ("feature", "target", "ignore", "row-ID"), groups
        ):
            if any(not value for value in values) or len(set(values)) != len(values):
                raise ValueError(f"OpenML {name} columns must be unique and non-empty.")
        object.__setattr__(self, "feature_columns", groups[0])
        object.__setattr__(self, "target_columns", groups[1])
        object.__setattr__(self, "ignore_columns", groups[2])
        object.__setattr__(self, "row_id_columns", groups[3])
        assigned = [value for group in groups for value in group]
        if self.weight_column is not None:
            if not self.weight_column:
                raise ValueError("OpenML weight column must be non-empty when present.")
            assigned.append(self.weight_column)
        if len(set(assigned)) != len(assigned):
            raise ValueError("OpenML column roles must be pairwise disjoint.")
        if len(self.feature_columns) != self.schema.dimension:
            raise ValueError("OpenML feature order disagrees with the typed schema.")

    def assigned_columns(self) -> tuple[str, ...]:
        result = (
            self.feature_columns
            + self.target_columns
            + self.ignore_columns
            + self.row_id_columns
        )
        return result if self.weight_column is None else result + (self.weight_column,)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema.to_dict(),
            "feature_columns": list(self.feature_columns),
            "target_columns": list(self.target_columns),
            "ignore_columns": list(self.ignore_columns),
            "row_id_columns": list(self.row_id_columns),
            "weight_column": self.weight_column,
        }


def materialize_openml_artifact(
    *,
    dataset_id: int,
    artifact_id: str,
    columns: OpenMLColumnContract,
    provider: InMemoryArtifactProvider,
    provenance: Mapping[str, Any],
    expected_version: int | str,
    expected_md5_checksum: str,
) -> ResolvedArtifact:
    """Fetch exactly once and register only explicitly assigned columns."""

    if type(dataset_id) is not int or dataset_id <= 0:
        raise ValueError("OpenML dataset ID must be a positive integer.")
    if not isinstance(columns, OpenMLColumnContract):
        raise TypeError("OpenML materialization requires a column contract.")
    if not isinstance(provider, InMemoryArtifactProvider):
        raise TypeError("OpenML materialization requires an explicit artifact provider.")
    openml = importlib.import_module("openml")
    dataset = openml.datasets.get_dataset(dataset_id)
    actual_version = getattr(dataset, "version", None)
    actual_checksum = getattr(dataset, "md5_checksum", None)
    if actual_version is None or str(actual_version) == "":
        raise ValueError("OpenML dataset version metadata is required.")
    if actual_checksum is None or str(actual_checksum) == "":
        raise ValueError("OpenML dataset md5 checksum metadata is required.")
    expected_checksum = str(expected_md5_checksum).lower()
    actual_checksum = str(actual_checksum).lower()
    if len(expected_checksum) != 32 or any(
        character not in "0123456789abcdef" for character in expected_checksum
    ):
        raise ValueError("Expected OpenML md5 checksum must be 32 hexadecimal digits.")
    if str(actual_version) != str(expected_version):
        raise ValueError("OpenML dataset version does not match the pinned version.")
    if actual_checksum != expected_checksum:
        raise ValueError("OpenML checksum does not match the pinned checksum.")
    frame, _, _, _ = dataset.get_data(dataset_format="dataframe")
    actual_columns = tuple(str(value) for value in frame.columns)
    declared_columns = columns.assigned_columns()
    if len(set(actual_columns)) != len(actual_columns):
        raise ValueError("OpenML materialization returned duplicate column names.")
    if set(actual_columns) != set(declared_columns):
        raise ValueError(
            "Every OpenML column must have one explicit feature/target/ignore/row-ID/weight role."
        )
    batch = TypedCovariateBatch(
        columns.schema,
        tuple(np.asarray(frame[name].to_numpy()) for name in columns.feature_columns),
    )
    weights = (
        None
        if columns.weight_column is None
        else np.asarray(frame[columns.weight_column].to_numpy(), dtype=np.float64)
    )
    resolved_provenance = {
        **dict(provenance),
        "adapter": "backdoor_scm.openml.explicit-columns-v1",
        "openml_dataset_id": dataset_id,
        "openml_dataset_version": str(actual_version),
        "openml_md5_checksum": actual_checksum,
        "column_contract": columns.to_dict(),
    }
    return provider.register(
        artifact_id,
        batch,
        weights=weights,
        provenance=resolved_provenance,
    )
