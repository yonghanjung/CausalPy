import dataclasses
import json
import unittest

import numpy as np

from backdoor_scm.exceptions import OutOfSupportError
from backdoor_scm.features import FrozenFeatureMapSpec, TypedFeatureMap
from backdoor_scm.families.roots import RootSpec
from backdoor_scm.families.scalars import (
    ScalarFunctionSpec,
    amplitude_law,
    build_scalar_function,
    sample_categorical_lookup_spec,
    sample_constant_spec,
    sample_dense_affine_spec,
    sample_projection_spec,
    sample_sparse_affine_spec,
)
from backdoor_scm.schema import (
    CovariateSchema,
    TypedCovariateBatch,
    VariableSpec,
    VariableType,
)


def _mixed_fixture():
    schema = CovariateSchema(
        "bdpfn-covariate-schema-v1",
        "mixed",
        "2-5",
        0,
        (
            VariableSpec("continuous", VariableType.CONTINUOUS),
            VariableSpec("binary", VariableType.BINARY, 2),
            VariableSpec("nominal", VariableType.CATEGORICAL, 3),
            VariableSpec("ordinal", VariableType.ORDINAL, 4),
        ),
    )
    base = np.asarray([0.2, 0.3, 0.5])
    nominal_pmf = 0.9 * base + 0.1 / 3.0
    ordinal_pmf = np.arange(1, 5, dtype=float) ** -1.0
    ordinal_pmf /= ordinal_pmf.sum()
    roots = (
        RootSpec.create(
            "bdpfn.root.uniform",
            "1.0.0",
            VariableType.CONTINUOUS,
            None,
            {"low": -np.sqrt(3.0), "high": np.sqrt(3.0)},
        ),
        RootSpec.create(
            "bdpfn.root.bernoulli",
            "1.0.0",
            VariableType.BINARY,
            2,
            {"p": 0.25, "probability_mode": "uniform_0.1_0.9"},
        ),
        RootSpec.create(
            "bdpfn.root.categorical_dirichlet1",
            "1.0.0",
            VariableType.CATEGORICAL,
            3,
            {"base_pmf": base.tolist(), "pmf": nominal_pmf.tolist()},
        ),
        RootSpec.create(
            "bdpfn.root.ordinal_truncated_zipf",
            "1.0.0",
            VariableType.ORDINAL,
            4,
            {"exponent": 1.0, "pmf": ordinal_pmf.tolist()},
        ),
    )
    batch = TypedCovariateBatch(
        schema,
        (
            np.asarray([-np.sqrt(3.0), 0.0, np.sqrt(3.0)]),
            np.asarray([0, 1, 0]),
            np.asarray([0, 1, 2]),
            np.asarray([0, 2, 3]),
        ),
    )
    return schema, roots, batch, nominal_pmf, ordinal_pmf


class TypedFeatureMapTests(unittest.TestCase):
    def test_exact_typed_map_is_frozen_canonical_and_batch_invariant(self) -> None:
        schema, roots, batch, nominal_pmf, ordinal_pmf = _mixed_fixture()
        specification = FrozenFeatureMapSpec.from_roots(schema, roots)
        rebuilt = FrozenFeatureMapSpec.from_json(specification.canonical_json())
        self.assertEqual(specification, rebuilt)
        self.assertEqual(specification.spec_hash, rebuilt.spec_hash)
        self.assertEqual(specification.output_dimension, 6)

        feature_map = TypedFeatureMap(specification)
        actual = feature_map.transform(batch)
        ordinal_support = np.arange(4, dtype=float)
        ordinal_mean = float(ordinal_support @ ordinal_pmf)
        ordinal_sd = np.sqrt(
            float((ordinal_support * ordinal_support) @ ordinal_pmf)
            - ordinal_mean**2
        )
        nominal = np.eye(3)[batch.columns[2]]
        nominal = (nominal - nominal_pmf) / np.sqrt(
            nominal_pmf * (1.0 - nominal_pmf)
        )
        expected = np.column_stack(
            (
                batch.columns[0],
                (batch.columns[1] - 0.25) / np.sqrt(0.25 * 0.75),
                nominal,
                (batch.columns[3] - ordinal_mean) / ordinal_sd,
            )
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.dtype("float64"))
        self.assertFalse(actual.flags.writeable)
        self.assertTrue(np.isfinite(actual).all())

        first = TypedCovariateBatch(schema, tuple(column[:1] for column in batch.columns))
        rest = TypedCovariateBatch(schema, tuple(column[1:] for column in batch.columns))
        np.testing.assert_array_equal(
            actual,
            np.vstack((feature_map.transform(first), feature_map.transform(rest))),
        )
        reversed_batch = TypedCovariateBatch(
            schema,
            tuple(column[::-1] for column in batch.columns),
        )
        np.testing.assert_array_equal(feature_map.transform(reversed_batch), actual[::-1])

    def test_feature_map_recomputes_metadata_and_fails_closed(self) -> None:
        schema, roots, batch, _, _ = _mixed_fixture()
        specification = FrozenFeatureMapSpec.from_roots(schema, roots)
        tampered = dataclasses.replace(
            specification.blocks[0],
            centers=(9.0,),
        )
        with self.assertRaises(ValueError):
            dataclasses.replace(
                specification,
                blocks=(tampered,) + specification.blocks[1:],
            )

        outside = TypedCovariateBatch(
            schema,
            (np.asarray([2.0]),) + tuple(column[:1] for column in batch.columns[1:]),
        )
        with self.assertRaises(OutOfSupportError):
            TypedFeatureMap(specification).transform(outside)

        encoded = json.loads(specification.canonical_json())
        encoded["root_specs"][0]["variable_type"] = "binary"
        with self.assertRaises(ValueError):
            FrozenFeatureMapSpec.from_dict(encoded)


class ScalarFunctionSpecTests(unittest.TestCase):
    def test_spec_is_frozen_canonical_hash_bound_and_strict(self) -> None:
        schema, roots, _, _, _ = _mixed_fixture()
        feature_spec = FrozenFeatureMapSpec.from_roots(schema, roots)
        specification = ScalarFunctionSpec(
            version="1.0.0",
            family_id="bdpfn.scalar.constant",
            schema_hash=schema.schema_hash,
            feature_map_hash=feature_spec.spec_hash,
            amplitude_mode="function",
            amplitude=1.0,
            variable_indices=(),
            feature_indices=(),
            coefficients=(),
            intercept=0.75,
            lookup_values=(),
        )
        rebuilt = ScalarFunctionSpec.from_json(specification.canonical_json())
        self.assertEqual(rebuilt, specification)
        self.assertEqual(rebuilt.spec_hash, specification.spec_hash)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            specification.intercept = 2.0
        with self.assertRaises(ValueError):
            dataclasses.replace(specification, feature_map_hash="not-a-sha")
        with self.assertRaises(ValueError):
            dataclasses.replace(specification, amplitude=0.25)
        with self.assertRaises(ValueError):
            dataclasses.replace(specification, coefficients=(np.inf,))

    def test_constant_and_projection_samplers_are_exact_and_deterministic(self) -> None:
        schema, roots, batch, _, _ = _mixed_fixture()
        feature_spec = FrozenFeatureMapSpec.from_roots(schema, roots)
        feature_values = TypedFeatureMap(feature_spec).transform(batch)
        constant = sample_constant_spec(feature_spec, np.random.default_rng(19))
        self.assertEqual(
            constant,
            sample_constant_spec(feature_spec, np.random.default_rng(19)),
        )
        constant_values = build_scalar_function(constant, feature_spec).evaluate(batch)
        np.testing.assert_array_equal(
            constant_values,
            np.full(batch.n_rows, constant.amplitude * constant.intercept),
        )

        projection = sample_projection_spec(feature_spec, np.random.default_rng(23))
        self.assertEqual(
            projection,
            sample_projection_spec(feature_spec, np.random.default_rng(23)),
        )
        self.assertEqual(len(projection.feature_indices), 1)
        expected = projection.amplitude * (
            projection.intercept
            + projection.coefficients[0]
            * feature_values[:, projection.feature_indices[0]]
        )
        actual = build_scalar_function(projection, feature_spec).evaluate(batch)
        np.testing.assert_array_equal(actual, expected)
        self.assertFalse(actual.flags.writeable)

    def test_sparse_and_dense_affine_use_whole_semantic_blocks(self) -> None:
        schema, roots, batch, _, _ = _mixed_fixture()
        feature_spec = FrozenFeatureMapSpec.from_roots(schema, roots)
        features = TypedFeatureMap(feature_spec).transform(batch)
        sparse = sample_sparse_affine_spec(feature_spec, np.random.default_rng(31))
        expected_sparse_indices = tuple(
            feature_index
            for variable_index in sparse.variable_indices
            for feature_index in range(
                feature_spec.blocks[variable_index].output_start,
                feature_spec.blocks[variable_index].output_start
                + feature_spec.blocks[variable_index].output_size,
            )
        )
        self.assertEqual(sparse.feature_indices, expected_sparse_indices)
        self.assertLessEqual(len(sparse.variable_indices), min(10, schema.dimension))
        sparse_expected = sparse.amplitude * (
            sparse.intercept
            + features[:, sparse.feature_indices] @ np.asarray(sparse.coefficients)
        )
        np.testing.assert_array_equal(
            build_scalar_function(sparse, feature_spec).evaluate(batch),
            sparse_expected,
        )

        dense = sample_dense_affine_spec(feature_spec, np.random.default_rng(37))
        self.assertEqual(dense.variable_indices, tuple(range(schema.dimension)))
        self.assertEqual(dense.feature_indices, tuple(range(feature_spec.output_dimension)))
        dense_expected = dense.amplitude * (
            dense.intercept
            + features @ np.asarray(dense.coefficients)
        )
        np.testing.assert_array_equal(
            build_scalar_function(dense, feature_spec).evaluate(batch),
            dense_expected,
        )

    def test_categorical_lookup_is_a_direct_nominal_table(self) -> None:
        schema, roots, batch, _, _ = _mixed_fixture()
        feature_spec = FrozenFeatureMapSpec.from_roots(schema, roots)
        lookup = sample_categorical_lookup_spec(
            feature_spec,
            np.random.default_rng(41),
        )
        variable_index = lookup.variable_indices[0]
        variable = schema.variables[variable_index]
        self.assertIs(variable.variable_type, VariableType.CATEGORICAL)
        self.assertEqual(len(lookup.lookup_values), variable.cardinality)
        expected = lookup.amplitude * (
            lookup.intercept
            + np.asarray(lookup.lookup_values)[batch.columns[variable_index]]
        )
        np.testing.assert_array_equal(
            build_scalar_function(lookup, feature_spec).evaluate(batch),
            expected,
        )

        no_nominal_schema = CovariateSchema(
            "bdpfn-covariate-schema-v1",
            "continuous_only",
            "1",
            0,
            (VariableSpec("x0", VariableType.CONTINUOUS),),
        )
        no_nominal_root = RootSpec.create(
            "bdpfn.root.normal",
            "1.0.0",
            VariableType.CONTINUOUS,
            None,
            {},
        )
        no_nominal_features = FrozenFeatureMapSpec.from_roots(
            no_nominal_schema,
            (no_nominal_root,),
        )
        with self.assertRaises(ValueError):
            sample_categorical_lookup_spec(
                no_nominal_features,
                np.random.default_rng(41),
            )

    def test_build_enforces_dense_completeness_and_sparse_variable_cap(self) -> None:
        schema, roots, _, _, _ = _mixed_fixture()
        feature_spec = FrozenFeatureMapSpec.from_roots(schema, roots)
        dense = sample_dense_affine_spec(feature_spec, np.random.default_rng(53))
        incomplete_dense = dataclasses.replace(
            dense,
            variable_indices=dense.variable_indices[:-1],
            feature_indices=dense.feature_indices[:-1],
            coefficients=dense.coefficients[:-1],
        )
        with self.assertRaises(ValueError):
            build_scalar_function(incomplete_dense, feature_spec)

        large_schema = CovariateSchema(
            "bdpfn-covariate-schema-v1",
            "continuous_only",
            "11-20",
            0,
            tuple(
                VariableSpec(f"x{index}", VariableType.CONTINUOUS)
                for index in range(11)
            ),
        )
        large_roots = tuple(
            RootSpec.create(
                "bdpfn.root.normal",
                "1.0.0",
                VariableType.CONTINUOUS,
                None,
                {},
            )
            for _ in range(11)
        )
        large_features = FrozenFeatureMapSpec.from_roots(large_schema, large_roots)
        too_wide_sparse = ScalarFunctionSpec(
            version="1.0.0",
            family_id="bdpfn.scalar.sparse_affine",
            schema_hash=large_schema.schema_hash,
            feature_map_hash=large_features.spec_hash,
            amplitude_mode="function",
            amplitude=1.0,
            variable_indices=tuple(range(11)),
            feature_indices=tuple(range(11)),
            coefficients=tuple(0.1 for _ in range(11)),
            intercept=0.0,
            lookup_values=(),
        )
        with self.assertRaises(ValueError):
            build_scalar_function(too_wide_sparse, large_features)

    def test_from_dict_rejects_unknown_fields(self) -> None:
        schema, roots, _, _, _ = _mixed_fixture()
        feature_spec = FrozenFeatureMapSpec.from_roots(schema, roots)
        specification = sample_constant_spec(feature_spec, np.random.default_rng(59))
        encoded = specification.to_dict()
        encoded["silent_typo"] = True
        with self.assertRaises(ValueError):
            ScalarFunctionSpec.from_dict(encoded)

    def test_amplitude_law_is_one_exact_ordered_public_authority(self) -> None:
        self.assertEqual(
            amplitude_law("function"),
            ((0.5, 0.25), (1.0, 0.50), (2.0, 0.25)),
        )
        self.assertEqual(
            amplitude_law("treatment_effect"),
            ((0.25, 0.20), (0.5, 0.40), (1.0, 0.30), (2.0, 0.10)),
        )
        with self.assertRaises(ValueError):
            amplitude_law("unknown")

    def test_local_generators_isolate_global_rng_and_build_rejects_hash_mismatch(self) -> None:
        schema, roots, _, _, _ = _mixed_fixture()
        feature_spec = FrozenFeatureMapSpec.from_roots(schema, roots)
        samplers = (
            sample_constant_spec,
            sample_projection_spec,
            sample_sparse_affine_spec,
            sample_dense_affine_spec,
            sample_categorical_lookup_spec,
        )
        np.random.seed(410)
        before = np.random.get_state()
        for seed, sampler in enumerate(samplers):
            sampled = sampler(
                feature_spec,
                np.random.default_rng(seed),
                amplitude_mode="treatment_effect",
            )
            self.assertIn(sampled.amplitude, {0.25, 0.5, 1.0, 2.0})
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

        other_schema = dataclasses.replace(schema, sampling_attempt=1)
        other_feature_spec = FrozenFeatureMapSpec.from_roots(other_schema, roots)
        specification = sample_projection_spec(feature_spec, np.random.default_rng(67))
        with self.assertRaises(ValueError):
            build_scalar_function(specification, other_feature_spec)

        changed_roots = (
            RootSpec.create(
                "bdpfn.root.normal",
                "1.0.0",
                VariableType.CONTINUOUS,
                None,
                {},
            ),
        ) + roots[1:]
        changed_feature_spec = FrozenFeatureMapSpec.from_roots(schema, changed_roots)
        with self.assertRaises(ValueError):
            build_scalar_function(specification, changed_feature_spec)

if __name__ == "__main__":
    unittest.main()
