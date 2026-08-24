import dataclasses
import importlib
import sys
import unittest
from unittest import mock

import numpy as np

import backdoor_scm
from backdoor_scm.artifacts import InMemoryArtifactProvider
from backdoor_scm.families.covariates import EmpiricalRowBootstrapFactory
from backdoor_scm.manifest import FamilyChoice
from backdoor_scm.openml_adapter import OpenMLColumnContract, materialize_openml_artifact
from backdoor_scm.schema import (
    CovariateSchema,
    TypedCovariateBatch,
    VariableSpec,
    VariableType,
)
from backdoor_scm.specs import FamilyRef


def _mixed_schema():
    return CovariateSchema(
        "bdpfn-covariate-schema-v1",
        "mixed",
        "2-5",
        0,
        (
            VariableSpec("continuous", VariableType.CONTINUOUS),
            VariableSpec("binary", VariableType.BINARY, 2),
            VariableSpec("category", VariableType.CATEGORICAL, 3),
        ),
    )


def _mixed_batch(negative_zero=False):
    zero = -0.0 if negative_zero else 0.0
    return TypedCovariateBatch(
        _mixed_schema(),
        (
            np.asarray([zero, 1.5, -2.0, 3.0]),
            np.asarray([0, 1, 0, 1]),
            np.asarray([0, 1, 2, 1]),
        ),
    )


class ArtifactProviderTests(unittest.TestCase):
    def test_digest_is_complete_canonical_and_resolver_fails_closed(self) -> None:
        provider = InMemoryArtifactProvider()
        first = provider.register(
            "artifact:mixed:v1",
            _mixed_batch(negative_zero=True),
            weights=np.asarray([1.0, 2.0, 3.0, 4.0]),
            provenance={"source": "unit-test", "version": 1},
        )
        self.assertEqual(sum(first.normalized_weights), 1.0)
        self.assertTrue(all(weight > 0.0 for weight in first.normalized_weights))
        self.assertEqual(
            first.artifact_sha256,
            InMemoryArtifactProvider()
            .register(
                "artifact:mixed:v1",
                _mixed_batch(negative_zero=False),
                weights=np.asarray([1.0, 2.0, 3.0, 4.0]),
                provenance={"version": 1, "source": "unit-test"},
            )
            .artifact_sha256,
        )
        self.assertIs(
            provider.resolve("artifact:mixed:v1", first.artifact_sha256),
            first,
        )
        with self.assertRaises(ValueError):
            provider.resolve("artifact:mixed:v1", "0" * 64)
        with self.assertRaises(KeyError):
            provider.resolve("missing", first.artifact_sha256)
        with self.assertRaises(ValueError):
            provider.register(
                "artifact:mixed:v1",
                TypedCovariateBatch(
                    _mixed_schema(),
                    (
                        np.asarray([0.0, 9.0, -2.0, 3.0]),
                        np.asarray([0, 1, 0, 1]),
                        np.asarray([0, 1, 2, 1]),
                    ),
                ),
                weights=np.asarray([1.0, 2.0, 3.0, 4.0]),
                provenance={"source": "unit-test", "version": 1},
            )

    def test_weights_must_be_strictly_positive_and_finite(self) -> None:
        for weights in (
            [1.0, 0.0, 2.0, 3.0],
            [1.0, -1.0, 2.0, 3.0],
            [1.0, np.inf, 2.0, 3.0],
        ):
            with self.assertRaises(ValueError):
                InMemoryArtifactProvider().register(
                    "bad",
                    _mixed_batch(),
                    weights=np.asarray(weights),
                    provenance={"source": "unit-test"},
                )

    def test_table_schema_normalized_weights_and_provenance_change_identity(self) -> None:
        base = InMemoryArtifactProvider().register(
            "base",
            _mixed_batch(),
            weights=np.asarray([1.0, 2.0, 3.0, 4.0]),
            provenance={"source": "unit-test", "version": 1},
        )
        changed_table = _mixed_batch().to_matrix().copy()
        changed_table[0, 0] = 8.0
        table = InMemoryArtifactProvider().register(
            "table",
            TypedCovariateBatch.from_matrix(_mixed_schema(), changed_table),
            weights=np.asarray([1.0, 2.0, 3.0, 4.0]),
            provenance={"source": "unit-test", "version": 1},
        )
        renamed_schema = dataclasses.replace(
            _mixed_schema(),
            variables=(
                VariableSpec("renamed", VariableType.CONTINUOUS),
                *_mixed_schema().variables[1:],
            ),
        )
        schema = InMemoryArtifactProvider().register(
            "schema",
            TypedCovariateBatch(renamed_schema, _mixed_batch().columns),
            weights=np.asarray([1.0, 2.0, 3.0, 4.0]),
            provenance={"source": "unit-test", "version": 1},
        )
        weights = InMemoryArtifactProvider().register(
            "weights",
            _mixed_batch(),
            weights=np.asarray([4.0, 3.0, 2.0, 1.0]),
            provenance={"source": "unit-test", "version": 1},
        )
        provenance = InMemoryArtifactProvider().register(
            "provenance",
            _mixed_batch(),
            weights=np.asarray([1.0, 2.0, 3.0, 4.0]),
            provenance={"source": "unit-test", "version": 2},
        )
        self.assertEqual(
            len(
                {
                    item.artifact_sha256
                    for item in (base, table, schema, weights, provenance)
                }
            ),
            5,
        )

    def test_provider_owns_immutable_storage_and_revalidates_on_resolve(self) -> None:
        caller_batch = _mixed_batch()
        provider = InMemoryArtifactProvider()
        artifact = provider.register(
            "owned",
            caller_batch,
            weights=np.asarray([1.0, 2.0, 3.0, 4.0]),
            provenance={"source": "unit-test"},
        )
        expected_hash = artifact.artifact_sha256
        caller_batch.columns[0].setflags(write=True)
        caller_batch.columns[0][0] = 999.0
        self.assertEqual(artifact.batch.columns[0][0], 0.0)
        with self.assertRaises(ValueError):
            artifact.batch.columns[0].setflags(write=True)
        exposed_matrix = artifact.batch.to_matrix()
        exposed_matrix.setflags(write=True)
        exposed_matrix[0, 0] = 777.0
        self.assertEqual(artifact.batch.columns[0][0], 0.0)
        self.assertEqual(
            provider.resolve("owned", expected_hash).artifact_sha256,
            expected_hash,
        )

        object.__setattr__(artifact, "table_sha256", "0" * 64)
        with self.assertRaisesRegex(ValueError, "integrity"):
            provider.resolve("owned", expected_hash)

    def test_normalized_weights_must_have_strictly_increasing_cdf(self) -> None:
        with self.assertRaises(ValueError):
            InMemoryArtifactProvider().register(
                "underflow",
                _mixed_batch(),
                weights=np.asarray([np.nextafter(0.0, 1.0), 1.0, 1.0, 1.0]),
                provenance={"source": "unit-test"},
            )


class EmpiricalRowBootstrapTests(unittest.TestCase):
    def _artifact(self):
        provider = InMemoryArtifactProvider()
        artifact = provider.register(
            "artifact:mixed:v1",
            _mixed_batch(),
            weights=np.asarray([1.0, 2.0, 3.0, 4.0]),
            provenance={"source": "unit-test", "version": 1},
        )
        return provider, artifact

    def test_spec_round_trip_and_uniform_joint_row_partition_invariance(self) -> None:
        provider, artifact = self._artifact()
        factory = EmpiricalRowBootstrapFactory(provider)
        hyperparameters = {
            "artifact_id": artifact.artifact_id,
            "artifact_sha256": artifact.artifact_sha256,
            "sampling_mode": "uniform",
        }
        spec = factory.sample_spec(np.random.default_rng(11), hyperparameters, {})
        self.assertEqual(spec, FamilyRef.from_dict(spec.to_dict()))
        self.assertEqual(
            spec,
            factory.sample_spec(np.random.default_rng(999), hyperparameters, {}),
        )
        law = factory.build(spec)
        self.assertEqual(law.dimension, 3)
        whole = law.sample(100, np.random.default_rng(7))
        split_rng = np.random.default_rng(7)
        left = law.sample(40, split_rng)
        right = law.sample(60, split_rng)
        for whole_column, left_column, right_column in zip(
            whole.columns, left.columns, right.columns
        ):
            np.testing.assert_array_equal(
                whole_column, np.concatenate((left_column, right_column))
            )
        table_rows = set(map(tuple, _mixed_batch().to_matrix()))
        self.assertTrue(all(tuple(row) in table_rows for row in whole.to_matrix()))

    def test_weighted_sampling_validation_frequency_and_support(self) -> None:
        provider, artifact = self._artifact()
        factory = EmpiricalRowBootstrapFactory(provider)
        spec = factory.sample_spec(
            np.random.default_rng(3),
            {
                "artifact_id": artifact.artifact_id,
                "artifact_sha256": artifact.artifact_sha256,
                "sampling_mode": "weighted",
            },
            {},
        )
        law = factory.build(spec)
        sample = law.sample(50_000, np.random.default_rng(41)).to_matrix()
        observed = np.asarray(
            [np.mean(np.all(sample == row, axis=1)) for row in _mixed_batch().to_matrix()]
        )
        np.testing.assert_allclose(observed, [0.1, 0.2, 0.3, 0.4], atol=0.01)
        self.assertTrue(law.contains(_mixed_batch()))
        whole = law.sample(90, np.random.default_rng(99))
        split_rng = np.random.default_rng(99)
        split = (law.sample(30, split_rng), law.sample(60, split_rng))
        for whole_column, left_column, right_column in zip(
            whole.columns, split[0].columns, split[1].columns
        ):
            np.testing.assert_array_equal(
                whole_column, np.concatenate((left_column, right_column))
            )
        off_table = _mixed_batch().to_matrix().copy()
        off_table[0, 0] = 99.0
        self.assertFalse(law.contains(off_table))

        unweighted_provider = InMemoryArtifactProvider()
        unweighted = unweighted_provider.register(
            "unweighted", _mixed_batch(), provenance={"source": "unit-test"}
        )
        with self.assertRaises(ValueError):
            EmpiricalRowBootstrapFactory(unweighted_provider).sample_spec(
                np.random.default_rng(1),
                {
                    "artifact_id": unweighted.artifact_id,
                    "artifact_sha256": unweighted.artifact_sha256,
                    "sampling_mode": "weighted",
                },
                {},
            )

    def test_missing_resolver_and_forged_provenance_fail_closed(self) -> None:
        provider, artifact = self._artifact()
        hyperparameters = {
            "artifact_id": artifact.artifact_id,
            "artifact_sha256": artifact.artifact_sha256,
            "sampling_mode": "uniform",
        }
        with self.assertRaisesRegex(ValueError, "resolver"):
            EmpiricalRowBootstrapFactory().sample_spec(
                np.random.default_rng(1), hyperparameters, {}
            )
        factory = EmpiricalRowBootstrapFactory(provider)
        spec = factory.sample_spec(np.random.default_rng(1), hyperparameters, {})
        for field in ("table_sha256", "weights_sha256", "provenance_sha256"):
            parameters = spec.parameter_dict()
            parameters[field] = "0" * 64
            forged = FamilyRef.create(
                spec.role,
                spec.family_id,
                spec.version,
                spec.semantic_class_id,
                parameters,
            )
            with self.assertRaises(ValueError):
                factory.build(forged)
        parameters = spec.parameter_dict()
        parameters["schema"] = dataclasses.replace(
            _mixed_schema(),
            variables=(
                VariableSpec("renamed", VariableType.CONTINUOUS),
                *_mixed_schema().variables[1:],
            ),
        ).to_dict()
        forged_schema = FamilyRef.create(
            spec.role,
            spec.family_id,
            spec.version,
            spec.semantic_class_id,
            parameters,
        )
        with self.assertRaises(ValueError):
            factory.build(forged_schema)


class RegistryAndTaskIntegrationTests(unittest.TestCase):
    def test_builtin_registry_resolves_all_covariates_and_injects_resolver(self) -> None:
        expected = {
            "bdpfn.covariate.independent_product",
            "bdpfn.covariate.gaussian_copula",
            "bdpfn.covariate.low_rank_gaussian",
            "bdpfn.covariate.student_t_copula",
            "bdpfn.covariate.empirical_row_bootstrap",
        }
        registry = backdoor_scm.build_builtin_registry()
        for family_id in expected:
            registry.resolve("covariate", family_id, "1.0.0")
        empirical = registry.resolve(
            "covariate", "bdpfn.covariate.empirical_row_bootstrap", "1.0.0"
        )
        with self.assertRaisesRegex(ValueError, "resolver"):
            empirical.sample_spec(
                np.random.default_rng(1),
                {
                    "artifact_id": "missing",
                    "artifact_sha256": "0" * 64,
                    "sampling_mode": "uniform",
                },
                {},
            )

    def test_mixed_empirical_task_uses_built_dimension_and_one_numeric_boundary(self) -> None:
        provider = InMemoryArtifactProvider()
        artifact = provider.register(
            "artifact:mixed:v1",
            _mixed_batch(),
            provenance={"source": "unit-test", "version": 1},
        )
        base = backdoor_scm.make_continuous_smoke_manifest(dimension=99)
        manifest = dataclasses.replace(
            base,
            manifest_version="empirical-mixed-e2e-v1",
            covariate_choices=(
                FamilyChoice.create(
                    "bdpfn.covariate.empirical_row_bootstrap",
                    "1.0.0",
                    1.0,
                    {
                        "artifact_id": artifact.artifact_id,
                        "artifact_sha256": artifact.artifact_sha256,
                        "sampling_mode": "uniform",
                    },
                ),
            ),
        )
        task = backdoor_scm.sample_task(
            manifest,
            global_seed=19,
            task_id=2,
            registry=backdoor_scm.build_builtin_registry(provider),
            source_id="causalpy@test-commit",
        )
        rows = task.sample_rows(12, row_seed=5)
        query = task.sample_query(6, query_seed=9)
        truth = task.truth(query)
        self.assertEqual(rows.x.shape, (12, 3))
        self.assertEqual(query.x.shape, (6, 3))
        self.assertEqual(truth.tau.shape, (6,))
        self.assertTrue(np.isfinite(rows.x).all())


class OpenMLAdapterTests(unittest.TestCase):
    def test_core_does_not_import_openml_and_contract_is_explicit(self) -> None:
        sys.modules.pop("openml", None)
        importlib.reload(backdoor_scm)
        self.assertNotIn("openml", sys.modules)

        class _Column:
            def __init__(self, values):
                self._values = np.asarray(values)

            def to_numpy(self):
                return self._values

        class _Frame:
            columns = ("row_id", "x0", "x1", "x2", "target", "weight")

            def __init__(self):
                matrix = _mixed_batch().to_matrix()
                self._columns = {
                    "row_id": _Column(np.arange(4)),
                    "x0": _Column(matrix[:, 0]),
                    "x1": _Column(matrix[:, 1]),
                    "x2": _Column(matrix[:, 2]),
                    "target": _Column([0, 1, 0, 1]),
                    "weight": _Column([1.0, 2.0, 3.0, 4.0]),
                }

            def __getitem__(self, key):
                return self._columns[key]

        class _Dataset:
            version = 4
            md5_checksum = "0123456789abcdef0123456789abcdef"

            def get_data(self, *, dataset_format):
                self.dataset_format = dataset_format
                return _Frame(), None, None, list(_Frame.columns)

        class _Datasets:
            @staticmethod
            def get_dataset(dataset_id):
                if dataset_id != 61:
                    raise AssertionError("wrong dataset")
                return _Dataset()

        class _OpenML:
            datasets = _Datasets()

        contract = OpenMLColumnContract(
            schema=_mixed_schema(),
            feature_columns=("x0", "x1", "x2"),
            target_columns=("target",),
            ignore_columns=(),
            row_id_columns=("row_id",),
            weight_column="weight",
        )
        provider = InMemoryArtifactProvider()
        with mock.patch("importlib.import_module", return_value=_OpenML()) as loader:
            artifact = materialize_openml_artifact(
                dataset_id=61,
                artifact_id="openml:61:v1",
                columns=contract,
                provider=provider,
                provenance={"requested_by": "unit-test"},
                expected_version=4,
                expected_md5_checksum="0123456789abcdef0123456789abcdef",
            )
        loader.assert_called_once_with("openml")
        self.assertEqual(artifact.schema, _mixed_schema())
        self.assertEqual(artifact.normalized_weights, (0.1, 0.2, 0.3, 0.4))

        incomplete = dataclasses.replace(contract, ignore_columns=("unknown",))
        with mock.patch("importlib.import_module", return_value=_OpenML()):
            with self.assertRaises(ValueError):
                materialize_openml_artifact(
                    dataset_id=61,
                    artifact_id="bad",
                    columns=incomplete,
                    provider=InMemoryArtifactProvider(),
                    provenance={"requested_by": "unit-test"},
                    expected_version=4,
                    expected_md5_checksum="0123456789abcdef0123456789abcdef",
                )

    def test_openml_version_and_checksum_are_required_and_fail_closed(self) -> None:
        class _Dataset:
            version = 7
            md5_checksum = "fedcba9876543210fedcba9876543210"

        class _Datasets:
            dataset = _Dataset()

            @classmethod
            def get_dataset(cls, dataset_id):
                return cls.dataset

        class _OpenML:
            datasets = _Datasets()

        contract = OpenMLColumnContract(
            schema=_mixed_schema(),
            feature_columns=("x0", "x1", "x2"),
            target_columns=("target",),
            ignore_columns=(),
            row_id_columns=("row_id",),
            weight_column="weight",
        )
        common = {
            "dataset_id": 61,
            "artifact_id": "openml:61:v7",
            "columns": contract,
            "provider": InMemoryArtifactProvider(),
            "provenance": {"requested_by": "unit-test"},
        }
        with mock.patch("importlib.import_module", return_value=_OpenML()):
            with self.assertRaises(ValueError):
                materialize_openml_artifact(
                    **common,
                    expected_version=6,
                    expected_md5_checksum=_Dataset.md5_checksum,
                )
            with self.assertRaises(ValueError):
                materialize_openml_artifact(
                    **common,
                    expected_version=7,
                    expected_md5_checksum="0" * 32,
                )
            delattr(_Dataset, "md5_checksum")
            with self.assertRaises(ValueError):
                materialize_openml_artifact(
                    **common,
                    expected_version=7,
                    expected_md5_checksum="fedcba9876543210fedcba9876543210",
                )


if __name__ == "__main__":
    unittest.main()
