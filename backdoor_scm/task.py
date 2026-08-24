"""Frozen back-door tasks, local row sampling, and exact truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .exceptions import OutOfSupportError
from .manifest import FamilyChoice, PriorManifest
from .protocols import CovariateLaw, OutcomeKernel, PropensityFunction
from .registry import ComponentRole, FamilyRegistry
from .rng import NUMPY_RUNTIME_VERSION, RNG_ALGORITHM, RNG_VERSION, keyed_rng
from .schema import TypedCovariateBatch
from .specs import (
    BackdoorTaskSpec,
    ProvenanceRecord,
    SemanticObservedBatch,
    SemanticQueryBatch,
    TruthBatch,
)


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array)
    result.setflags(write=False)
    return result


def _numeric_covariate_batch(values: object) -> np.ndarray:
    if isinstance(values, TypedCovariateBatch):
        return values.to_matrix()
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError("Covariate laws must return a two-dimensional batch.")
    return result


def _weighted_choice(
    choices: Sequence[FamilyChoice],
    rng: np.random.Generator,
) -> FamilyChoice:
    probabilities = np.asarray([choice.weight for choice in choices], dtype=float)
    probabilities = probabilities / probabilities.sum()
    return choices[int(rng.choice(len(choices), p=probabilities))]


def _validate_manifest_registry(
    manifest: PriorManifest,
    registry: FamilyRegistry,
) -> None:
    for choice in manifest.covariate_choices:
        registry.resolve(ComponentRole.COVARIATE, choice.family_id, choice.version)
    for choice in manifest.propensity_choices:
        registry.resolve(ComponentRole.PROPENSITY, choice.family_id, choice.version)
    for choice in manifest.outcome_choices:
        factory = registry.resolve(
            ComponentRole.OUTCOME,
            choice.family_id,
            choice.version,
        )
        if factory.outcome_type != manifest.outcome_type:
            raise ValueError("Manifest combines incompatible outcome types.")
    if manifest.different_compatible_likelihood_probability > 0.0:
        likelihood_classes = {
            registry.resolve(
                ComponentRole.OUTCOME,
                choice.family_id,
                choice.version,
            ).semantic_class_id
            for choice in manifest.outcome_choices
        }
        if len(likelihood_classes) < 2:
            raise ValueError(
                "Different-likelihood mass requires two semantic classes."
            )


def sample_task(
    manifest: PriorManifest,
    global_seed: int,
    task_id: int,
    registry: FamilyRegistry,
    source_id: str,
) -> "BackdoorTask":
    if global_seed < 0 or task_id < 0:
        raise ValueError("Seed and task IDs must be non-negative.")
    if not source_id:
        raise ValueError("A caller-supplied source ID is required.")
    _validate_manifest_registry(manifest, registry)
    base_key = (global_seed, manifest.manifest_hash, task_id)

    covariate_choice = _weighted_choice(
        manifest.covariate_choices,
        keyed_rng(*base_key, "select", "covariate"),
    )
    covariate_factory = registry.resolve(
        ComponentRole.COVARIATE,
        covariate_choice.family_id,
        covariate_choice.version,
    )
    covariate_spec = covariate_factory.sample_spec(
        keyed_rng(*base_key, "parameters", "covariate"),
        covariate_choice.hyperparameter_dict(),
        {},
    )
    dimension = int(covariate_factory.build(covariate_spec).dimension)
    context = {"dimension": dimension, "outcome_type": manifest.outcome_type}

    propensity_choice = _weighted_choice(
        manifest.propensity_choices,
        keyed_rng(*base_key, "select", "propensity"),
    )
    propensity_factory = registry.resolve(
        ComponentRole.PROPENSITY,
        propensity_choice.family_id,
        propensity_choice.version,
    )
    propensity_spec = propensity_factory.sample_spec(
        keyed_rng(*base_key, "parameters", "propensity"),
        propensity_choice.hyperparameter_dict(),
        context,
    )

    mode_rng = keyed_rng(*base_key, "select", "outcome_likelihood_mode")
    same_likelihood = mode_rng.random() < manifest.same_likelihood_probability
    outcome0_choice = _weighted_choice(
        manifest.outcome_choices,
        keyed_rng(*base_key, "select", "outcome0"),
    )
    if same_likelihood:
        outcome1_choice = outcome0_choice
        likelihood_mode = "same"
    else:
        outcome0_factory = registry.resolve(
            ComponentRole.OUTCOME,
            outcome0_choice.family_id,
            outcome0_choice.version,
        )
        compatible = tuple(
            choice
            for choice in manifest.outcome_choices
            if registry.resolve(
                ComponentRole.OUTCOME,
                choice.family_id,
                choice.version,
            ).semantic_class_id
            != outcome0_factory.semantic_class_id
        )
        if not compatible:
            raise ValueError("No compatible different outcome likelihood is available.")
        outcome1_choice = _weighted_choice(
            compatible,
            keyed_rng(*base_key, "select", "outcome1"),
        )
        likelihood_mode = "different"

    outcome_specs = []
    for arm, choice in enumerate((outcome0_choice, outcome1_choice)):
        factory = registry.resolve(ComponentRole.OUTCOME, choice.family_id, choice.version)
        outcome_specs.append(
            factory.sample_spec(
                keyed_rng(*base_key, "parameters", f"outcome{arm}"),
                choice.hyperparameter_dict(),
                {**context, "arm": arm},
            )
        )

    spec = BackdoorTaskSpec(
        task_id=task_id,
        global_seed=global_seed,
        source_id=source_id,
        rng_algorithm=RNG_ALGORITHM,
        rng_version=RNG_VERSION,
        numpy_runtime_version=NUMPY_RUNTIME_VERSION,
        manifest_hash=manifest.manifest_hash,
        covariate=covariate_spec,
        propensity=propensity_spec,
        outcome0=outcome_specs[0],
        outcome1=outcome_specs[1],
        outcome_type=manifest.outcome_type,
        outcome_likelihood_mode=likelihood_mode,
    )
    return BackdoorTask.from_spec(spec, registry)


@dataclass(frozen=True)
class BackdoorTask:
    _spec: BackdoorTaskSpec
    _covariate: CovariateLaw
    _propensity: PropensityFunction
    _outcomes: tuple[OutcomeKernel, OutcomeKernel]

    def __post_init__(self) -> None:
        outcome0, outcome1 = self._outcomes
        dimensions = {
            self._covariate.dimension,
            self._propensity.dimension,
            outcome0.dimension,
            outcome1.dimension,
        }
        if len(dimensions) != 1:
            raise ValueError("Task components have incompatible dimensions.")
        if (
            outcome0.outcome_type != self._spec.outcome_type
            or outcome1.outcome_type != self._spec.outcome_type
        ):
            raise ValueError("Task components have incompatible outcome types.")

    @classmethod
    def from_spec(
        cls,
        spec: BackdoorTaskSpec,
        registry: FamilyRegistry,
    ) -> "BackdoorTask":
        covariate_factory = registry.resolve(
            ComponentRole.COVARIATE,
            spec.covariate.family_id,
            spec.covariate.version,
        )
        propensity_factory = registry.resolve(
            ComponentRole.PROPENSITY,
            spec.propensity.family_id,
            spec.propensity.version,
        )
        outcome0_factory = registry.resolve(
            ComponentRole.OUTCOME,
            spec.outcome0.family_id,
            spec.outcome0.version,
        )
        outcome1_factory = registry.resolve(
            ComponentRole.OUTCOME,
            spec.outcome1.family_id,
            spec.outcome1.version,
        )
        for family_ref, factory in (
            (spec.covariate, covariate_factory),
            (spec.propensity, propensity_factory),
            (spec.outcome0, outcome0_factory),
            (spec.outcome1, outcome1_factory),
        ):
            if family_ref.semantic_class_id != factory.semantic_class_id:
                raise ValueError("Spec semantic class does not match its factory.")
        return cls(
            _spec=spec,
            _covariate=covariate_factory.build(spec.covariate),
            _propensity=propensity_factory.build(spec.propensity),
            _outcomes=(
                outcome0_factory.build(spec.outcome0),
                outcome1_factory.build(spec.outcome1),
            ),
        )

    def to_spec(self) -> BackdoorTaskSpec:
        return self._spec

    def provenance(self) -> ProvenanceRecord:
        return ProvenanceRecord(
            manifest_hash=self._spec.manifest_hash,
            task_spec_hash=self._spec.task_spec_hash,
            sampling_identity=self._spec.sampling_identity,
            seed_ids=(
                ("global_seed", self._spec.global_seed),
                ("task_id", self._spec.task_id),
            ),
            component_families=(
                (
                    "covariate",
                    self._spec.covariate.family_id,
                    self._spec.covariate.version,
                ),
                (
                    "propensity",
                    self._spec.propensity.family_id,
                    self._spec.propensity.version,
                ),
                ("outcome0", self._spec.outcome0.family_id, self._spec.outcome0.version),
                ("outcome1", self._spec.outcome1.family_id, self._spec.outcome1.version),
            ),
            rng_algorithm=self._spec.rng_algorithm,
            rng_version=self._spec.rng_version,
            numpy_runtime_version=self._spec.numpy_runtime_version,
            source_id=self._spec.source_id,
        )

    def _require_sampling_runtime(self) -> None:
        if (
            self._spec.rng_algorithm != RNG_ALGORITHM
            or self._spec.rng_version != RNG_VERSION
            or self._spec.numpy_runtime_version != NUMPY_RUNTIME_VERSION
        ):
            raise ValueError(
                "Task sampling provenance is incompatible with the current runtime."
            )

    def sample_rows(
        self,
        n: int,
        row_seed: int,
        start_row: int = 0,
    ) -> SemanticObservedBatch:
        self._require_sampling_runtime()
        if n < 0 or row_seed < 0 or start_row < 0:
            raise ValueError("Row counts and seed IDs must be non-negative.")
        dimension = self._covariate.dimension
        x = np.empty((n, dimension), dtype=float)
        a = np.empty(n, dtype=np.int8)
        y = np.empty(n, dtype=float)
        task_key = self._spec.sampling_identity
        for offset in range(n):
            row_id = start_row + offset
            row_x = _numeric_covariate_batch(
                self._covariate.sample(
                    1,
                    keyed_rng(task_key, row_seed, row_id, "x"),
                )
            )
            probability = float(self._propensity.evaluate(row_x)[0])
            arm = int(
                keyed_rng(task_key, row_seed, row_id, "a").random()
                < probability
            )
            outcome = float(
                self._outcomes[arm].sample(
                    row_x,
                    keyed_rng(task_key, row_seed, row_id, "y"),
                )[0]
            )
            x[offset] = row_x[0]
            a[offset] = arm
            y[offset] = outcome
        return SemanticObservedBatch(
            x=_readonly(x),
            a=_readonly(a),
            y=_readonly(y),
        )

    def sample_query(
        self,
        n: int,
        query_seed: int,
        start_query: int = 0,
    ) -> SemanticQueryBatch:
        self._require_sampling_runtime()
        if n < 0 or query_seed < 0 or start_query < 0:
            raise ValueError("Query counts and seed IDs must be non-negative.")
        x = np.empty((n, self._covariate.dimension), dtype=float)
        for offset in range(n):
            query_id = start_query + offset
            query_x = _numeric_covariate_batch(
                self._covariate.sample(
                    1,
                    keyed_rng(
                        self._spec.sampling_identity,
                        query_seed,
                        query_id,
                        "query-x",
                    ),
                )
            )
            x[offset] = self._validated_query(query_x)[0]
        return SemanticQueryBatch(x=x)

    def _validated_query(
        self,
        x: np.ndarray | SemanticQueryBatch,
    ) -> np.ndarray:
        values = x.x if isinstance(x, SemanticQueryBatch) else x
        try:
            query = np.asarray(values, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("Query must be a numeric two-dimensional array.") from error
        if query.ndim != 2 or query.shape[1] != self._covariate.dimension:
            raise ValueError("Query has the wrong rank or covariate dimension.")
        if not self._covariate.contains(query):
            raise OutOfSupportError("Query is outside the declared covariate support.")
        return query

    def truth(self, x: np.ndarray | SemanticQueryBatch) -> TruthBatch:
        query = self._validated_query(x)
        propensity = self._propensity.evaluate(query)
        mu0 = self._outcomes[0].mean(query)
        mu1 = self._outcomes[1].mean(query)
        return TruthBatch(
            propensity=_readonly(propensity.copy()),
            mu0=_readonly(mu0.copy()),
            mu1=_readonly(mu1.copy()),
            tau=_readonly((mu1 - mu0).copy()),
        )

    def propensity(self, x: np.ndarray | SemanticQueryBatch) -> np.ndarray:
        return self.truth(x).propensity

    def mu(self, arm: int, x: np.ndarray | SemanticQueryBatch) -> np.ndarray:
        if isinstance(arm, bool) or not isinstance(arm, (int, np.integer)):
            raise ValueError("Outcome arm must be integer 0 or 1.")
        if int(arm) not in (0, 1):
            raise ValueError("Outcome arm must be integer 0 or 1.")
        truth = self.truth(x)
        return truth.mu0 if int(arm) == 0 else truth.mu1

    def tau(self, x: np.ndarray | SemanticQueryBatch) -> np.ndarray:
        return self.truth(x).tau


def load_task(
    spec: BackdoorTaskSpec | str,
    registry: FamilyRegistry,
) -> BackdoorTask:
    if isinstance(spec, str):
        spec = BackdoorTaskSpec.from_json(spec)
    if not isinstance(spec, BackdoorTaskSpec):
        raise TypeError("Task input must be a BackdoorTaskSpec or canonical JSON.")
    return BackdoorTask.from_spec(spec, registry)
