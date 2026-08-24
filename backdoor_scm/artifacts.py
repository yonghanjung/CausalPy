"""Explicit, content-addressed artifacts for empirical covariate laws."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping

import numpy as np

from .schema import CovariateSchema, TypedCovariateBatch
from .protocols import ArtifactResolver
from .specs import FrozenValue, canonical_hash, freeze_mapping, thaw_mapping


def _column_manifest(batch: TypedCovariateBatch) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, (variable, source) in enumerate(
        zip(batch.schema.variables, batch.columns)
    ):
        if np.issubdtype(source.dtype, np.floating):
            values = np.asarray(source, dtype="<f8").copy(order="C")
            values[values == 0.0] = 0.0
        else:
            values = np.asarray(source, dtype="<i8").copy(order="C")
        result.append(
            {
                "column_index": index,
                "variable": variable.to_dict(),
                "dtype": values.dtype.str,
                "shape": list(values.shape),
                "sha256": hashlib.sha256(values.tobytes(order="C")).hexdigest(),
            }
        )
    return result


def _normalize_weights(
    weights: np.ndarray | None,
    n_rows: int,
) -> tuple[float, ...] | None:
    if weights is None:
        return None
    values = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != n_rows:
        raise ValueError("Artifact weights must have one value per row.")
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("Artifact weights must be finite and strictly positive.")
    total = float(np.sum(values, dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("Artifact weight sum must be finite and positive.")
    normalized = [float(value / total) for value in values]
    normalized[-1] = 1.0 - sum(normalized[:-1])
    if any(weight <= 0.0 or not math.isfinite(weight) for weight in normalized):
        raise ValueError("Normalized artifact weights must remain strictly positive.")
    result = tuple(normalized)
    if sum(result) != 1.0:
        raise ValueError("Normalized artifact weights must sum exactly to one.")
    cumulative = np.cumsum(np.asarray(result, dtype=np.float64))
    cumulative[-1] = 1.0
    if (np.diff(np.concatenate(([0.0], cumulative))) <= 0.0).any():
        raise ValueError("Normalized artifact weight CDF must be strictly increasing.")
    return result


def _provider_owned_batch(batch: TypedCovariateBatch) -> TypedCovariateBatch:
    """Deep-copy columns onto immutable byte buffers owned by the provider."""

    validated = TypedCovariateBatch(batch.schema, batch.columns)
    immutable_columns = []
    for column in validated.columns:
        contiguous = np.ascontiguousarray(column)
        payload = contiguous.tobytes(order="C")
        immutable = np.frombuffer(payload, dtype=contiguous.dtype).reshape(
            contiguous.shape
        )
        immutable_columns.append(immutable)
    object.__setattr__(validated, "columns", tuple(immutable_columns))
    return validated


@dataclass(frozen=True)
class ResolvedArtifact:
    artifact_id: str
    schema: CovariateSchema
    batch: TypedCovariateBatch
    normalized_weights: tuple[float, ...] | None
    provenance: tuple[tuple[str, FrozenValue], ...]
    table_sha256: str
    weights_sha256: str
    provenance_sha256: str
    artifact_sha256: str

    def provenance_dict(self) -> dict[str, Any]:
        return thaw_mapping(self.provenance)


def _create_artifact(
    artifact_id: str,
    batch: TypedCovariateBatch,
    weights: np.ndarray | None,
    provenance: Mapping[str, Any],
) -> ResolvedArtifact:
    if not artifact_id:
        raise ValueError("Artifact ID must be non-empty.")
    owned_batch = _provider_owned_batch(batch)
    if owned_batch.n_rows <= 0:
        raise ValueError("Empirical artifacts must contain at least one row.")
    normalized_weights = _normalize_weights(weights, owned_batch.n_rows)
    frozen_provenance = freeze_mapping(provenance)
    table_sha256 = canonical_hash(
        {
            "schema": owned_batch.schema.to_dict(),
            "n_rows": owned_batch.n_rows,
            "columns": _column_manifest(owned_batch),
        }
    )
    weights_sha256 = canonical_hash(
        {
            "weights": None
            if normalized_weights is None
            else list(normalized_weights)
        }
    )
    provenance_sha256 = canonical_hash(thaw_mapping(frozen_provenance))
    artifact_sha256 = canonical_hash(
        {
            "schema": owned_batch.schema.to_dict(),
            "table_sha256": table_sha256,
            "weights_sha256": weights_sha256,
            "provenance_sha256": provenance_sha256,
        }
    )
    return ResolvedArtifact(
        artifact_id=artifact_id,
        schema=owned_batch.schema,
        batch=owned_batch,
        normalized_weights=normalized_weights,
        provenance=frozen_provenance,
        table_sha256=table_sha256,
        weights_sha256=weights_sha256,
        provenance_sha256=provenance_sha256,
        artifact_sha256=artifact_sha256,
    )


class InMemoryArtifactProvider:
    """Instance-local artifact store used by tests and explicit callers."""

    def __init__(self) -> None:
        self._artifacts: dict[str, ResolvedArtifact] = {}

    def register(
        self,
        artifact_id: str,
        batch: TypedCovariateBatch,
        *,
        weights: np.ndarray | None = None,
        provenance: Mapping[str, Any],
    ) -> ResolvedArtifact:
        candidate = _create_artifact(artifact_id, batch, weights, provenance)
        current = self._artifacts.get(artifact_id)
        if current is not None:
            if current.artifact_sha256 != candidate.artifact_sha256:
                raise ValueError("Artifact ID is already bound to different content.")
            return current
        self._artifacts[artifact_id] = candidate
        return candidate

    def resolve(self, artifact_id: str, expected_sha256: str) -> ResolvedArtifact:
        try:
            artifact = self._artifacts[artifact_id]
        except KeyError as error:
            raise KeyError(f"Unknown artifact ID: {artifact_id}") from error
        recomputed = _create_artifact(
            artifact.artifact_id,
            artifact.batch,
            None
            if artifact.normalized_weights is None
            else np.asarray(artifact.normalized_weights, dtype=np.float64),
            artifact.provenance_dict(),
        )
        digest_fields = (
            "table_sha256",
            "weights_sha256",
            "provenance_sha256",
            "artifact_sha256",
        )
        if any(
            getattr(artifact, field) != getattr(recomputed, field)
            for field in digest_fields
        ):
            raise ValueError("Stored artifact failed its integrity revalidation.")
        if artifact.artifact_sha256 != expected_sha256:
            raise ValueError("Artifact digest does not match the requested identity.")
        return artifact
