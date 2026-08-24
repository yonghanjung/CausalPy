import json
import unittest
from dataclasses import FrozenInstanceError

import numpy as np
import backdoor_scm

from backdoor_scm.schema import (
    CovariateSchema,
    active_v1_schema_policy,
    sample_schema,
    TypedCovariateBatch,
    VariableSpec,
    VariableType,
)
from backdoor_scm.active_v1 import active_v1_manifest
from backdoor_scm.families.covariates import (
    IndependentProductCovariateFactory,
    IndependentProductCovariates,
)
from backdoor_scm.families.roots import (
    RootSpec,
    build_root,
    root_family_weights,
    sample_root_spec,
)


class TypedSchemaTests(unittest.TestCase):
    def test_schema_is_frozen_canonical_and_preserves_semantic_types(self) -> None:
        schema = CovariateSchema(
            schema_version="bdpfn-covariate-schema-v1",
            profile="mixed",
            dimension_stratum="2-5",
            sampling_attempt=3,
            variables=(
                VariableSpec("x0", VariableType.CONTINUOUS),
                VariableSpec("x1", VariableType.BINARY, cardinality=2),
                VariableSpec("x2", VariableType.CATEGORICAL, cardinality=7),
                VariableSpec("x3", VariableType.ORDINAL, cardinality=5),
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            schema.profile = "continuous_only"
        rebuilt = CovariateSchema.from_dict(json.loads(schema.canonical_json()))
        self.assertEqual(schema, rebuilt)
        self.assertEqual(schema.schema_hash, rebuilt.schema_hash)
        self.assertEqual(schema.dimension, 4)
        self.assertEqual(
            tuple(variable.variable_type for variable in schema.variables),
            (
                VariableType.CONTINUOUS,
                VariableType.BINARY,
                VariableType.CATEGORICAL,
                VariableType.ORDINAL,
            ),
        )

    def test_schema_rejects_invalid_dimension_profile_and_cardinality(self) -> None:
        with self.assertRaises(ValueError):
            CovariateSchema(
                "bdpfn-covariate-schema-v1",
                "mixed",
                "1",
                0,
                (VariableSpec("x0", VariableType.CONTINUOUS),),
            )
        with self.assertRaises(ValueError):
            VariableSpec("x", VariableType.BINARY, cardinality=3)
        with self.assertRaises(ValueError):
            VariableSpec("x", VariableType.CATEGORICAL, cardinality=1)

    def test_schema_rejects_dimension_stratum_mismatching_actual_dimension(self) -> None:
        with self.assertRaises(ValueError):
            CovariateSchema(
                "bdpfn-covariate-schema-v1",
                "continuous_only",
                "6-10",
                0,
                (
                    VariableSpec("x0", VariableType.CONTINUOUS),
                    VariableSpec("x1", VariableType.CONTINUOUS),
                ),
            )

    def test_typed_batch_normalizes_integer_valued_outer_float_columns(self) -> None:
        schema = CovariateSchema(
            "bdpfn-covariate-schema-v1",
            "mixed",
            "2-5",
            0,
            (
                VariableSpec("x0", VariableType.CONTINUOUS),
                VariableSpec("x1", VariableType.CATEGORICAL, cardinality=3),
            ),
        )
        batch = TypedCovariateBatch.from_matrix(
            schema,
            np.asarray([[0.25, 1.0], [-1.5, 2.0]]),
        )
        self.assertEqual(batch.columns[0].dtype, np.dtype("float64"))
        self.assertEqual(batch.columns[1].dtype, np.dtype("int64"))
        with self.assertRaises(ValueError):
            TypedCovariateBatch.from_matrix(schema, np.asarray([[0.0, 1.5]]))
        with self.assertRaises(ValueError):
            TypedCovariateBatch.from_matrix(schema, np.asarray([[0.0, 3.0]]))


class SchemaSamplingTests(unittest.TestCase):
    def test_policy_matches_frozen_active_v1_manifest(self) -> None:
        policy = active_v1_schema_policy()
        manifest = active_v1_manifest()
        self.assertEqual(
            policy.dimension_weights(),
            manifest.law_weights("dimension.stratum"),
        )
        self.assertEqual(
            policy.profile_weights_dict(),
            manifest.law_weights("schema.profile"),
        )
        self.assertEqual(
            policy.mixed_type_weights_dict(),
            manifest.law_weights("schema.mixed_coordinate_type"),
        )
        self.assertEqual(
            policy.cardinality_band_weights(),
            manifest.law_weights("covariate.categorical_cardinality_band"),
        )

    def test_schema_sampling_is_deterministic_and_rejects_homogeneous_mixed(self) -> None:
        first = sample_schema(np.random.default_rng(721))
        second = sample_schema(np.random.default_rng(721))
        self.assertEqual(first, second)

        seen_profiles = set()
        for seed in range(200):
            schema = sample_schema(np.random.default_rng(seed))
            seen_profiles.add(schema.profile)
            if schema.profile == "mixed":
                self.assertGreaterEqual(schema.dimension, 2)
                self.assertGreaterEqual(
                    len({variable.variable_type for variable in schema.variables}),
                    2,
                )
            for variable in schema.variables:
                if variable.variable_type in {
                    VariableType.CATEGORICAL,
                    VariableType.ORDINAL,
                }:
                    self.assertGreaterEqual(variable.cardinality, 2)
                    self.assertLessEqual(variable.cardinality, 30)
        self.assertEqual(
            seen_profiles,
            {
                "continuous_only",
                "binary_only",
                "categorical_or_ordinal_only",
                "mixed",
            },
        )


class ContinuousRootTests(unittest.TestCase):
    def test_family_specific_support_is_exact_for_bounded_and_unbounded_roots(self) -> None:
        uniform = build_root(
            RootSpec.create(
                "bdpfn.root.uniform",
                "1.0.0",
                VariableType.CONTINUOUS,
                None,
                {"low": -np.sqrt(3.0), "high": np.sqrt(3.0)},
            )
        )
        self.assertTrue(uniform.contains(np.asarray([-np.sqrt(3.0), 0.0, np.sqrt(3.0)])))
        self.assertFalse(uniform.contains(np.asarray([np.sqrt(3.0) + 1e-12])))

        alpha, beta = 2.0, 3.0
        raw_mean = alpha / (alpha + beta)
        raw_variance = alpha * beta / (
            (alpha + beta) ** 2 * (alpha + beta + 1.0)
        )
        raw_sd = np.sqrt(raw_variance)
        beta_root = build_root(
            RootSpec.create(
                "bdpfn.root.beta_standardized",
                "1.0.0",
                VariableType.CONTINUOUS,
                None,
                {
                    "alpha": alpha,
                    "beta": beta,
                    "raw_mean": raw_mean,
                    "raw_sd": raw_sd,
                },
            )
        )
        beta_low = (0.0 - raw_mean) / raw_sd
        beta_high = (1.0 - raw_mean) / raw_sd
        self.assertTrue(beta_root.contains(np.asarray([beta_low, 0.0, beta_high])))
        self.assertFalse(beta_root.contains(np.asarray([beta_low - 1e-12])))

        for family_id, parameters in (
            ("bdpfn.root.normal", {}),
            (
                "bdpfn.root.laplace_standardized",
                {"location": 0.0, "scale": 1.0 / np.sqrt(2.0)},
            ),
            (
                "bdpfn.root.gaussian_mixture2_standardized",
                {
                    "weight": 0.5,
                    "separation": 1.0,
                    "sd0": 0.5,
                    "sd1": 0.5,
                    "raw_mean": 0.0,
                    "raw_sd": np.sqrt(0.5),
                },
            ),
        ):
            root = build_root(
                RootSpec.create(
                    family_id,
                    "1.0.0",
                    VariableType.CONTINUOUS,
                    None,
                    parameters,
                )
            )
            self.assertTrue(root.contains(np.asarray([-1e12, 1e12])))
            self.assertFalse(root.contains(np.asarray([np.inf])))
    def test_root_specs_reject_family_type_and_parameter_mismatches(self) -> None:
        with self.assertRaises(ValueError):
            RootSpec.create(
                "bdpfn.root.uniform",
                "1.0.0",
                VariableType.CONTINUOUS,
                None,
                {"low": 0.0, "high": 1.0},
            )
        with self.assertRaises(ValueError):
            RootSpec.create(
                "bdpfn.root.normal",
                "1.0.0",
                VariableType.BINARY,
                2,
                {},
            )
        with self.assertRaises(ValueError):
            RootSpec.create(
                "bdpfn.root.laplace_standardized",
                "1.0.0",
                VariableType.CONTINUOUS,
                None,
                {"location": 0.0},
            )
        with self.assertRaises(ValueError):
            RootSpec.create(
                "bdpfn.root.unknown",
                "1.0.0",
                VariableType.CONTINUOUS,
                None,
                {},
            )

    def test_fixed_continuous_roots_have_exact_moments_and_rebuild(self) -> None:
        specifications = (
            RootSpec.create(
                "bdpfn.root.normal",
                "1.0.0",
                VariableType.CONTINUOUS,
                None,
                {},
            ),
            RootSpec.create(
                "bdpfn.root.uniform",
                "1.0.0",
                VariableType.CONTINUOUS,
                None,
                {"low": -np.sqrt(3.0), "high": np.sqrt(3.0)},
            ),
            RootSpec.create(
                "bdpfn.root.laplace_standardized",
                "1.0.0",
                VariableType.CONTINUOUS,
                None,
                {"location": 0.0, "scale": 1.0 / np.sqrt(2.0)},
            ),
        )
        for specification in specifications:
            root = build_root(specification)
            self.assertEqual(root.mean, 0.0)
            self.assertEqual(root.variance, 1.0)
            rebuilt = RootSpec.from_dict(json.loads(specification.canonical_json()))
            self.assertEqual(specification, rebuilt)
            np.testing.assert_array_equal(
                root.sample(12, np.random.default_rng(41)),
                build_root(rebuilt).sample(12, np.random.default_rng(41)),
            )

    def test_beta_and_gmm2_use_validated_exact_standardization(self) -> None:
        beta_variance = 2.0 * 3.0 / ((2.0 + 3.0) ** 2 * (2.0 + 3.0 + 1.0))
        beta = RootSpec.create(
            "bdpfn.root.beta_standardized",
            "1.0.0",
            VariableType.CONTINUOUS,
            None,
            {
                "alpha": 2.0,
                "beta": 3.0,
                "raw_mean": 2.0 / 5.0,
                "raw_sd": np.sqrt(beta_variance),
            },
        )
        weight, delta, sd0, sd1 = 0.4, 2.0, 0.5, 0.8
        mean0, mean1 = -delta / 2.0, delta / 2.0
        raw_mean = weight * mean0 + (1.0 - weight) * mean1
        raw_second = weight * (sd0**2 + mean0**2) + (1.0 - weight) * (
            sd1**2 + mean1**2
        )
        mixture = RootSpec.create(
            "bdpfn.root.gaussian_mixture2_standardized",
            "1.0.0",
            VariableType.CONTINUOUS,
            None,
            {
                "weight": weight,
                "separation": delta,
                "sd0": sd0,
                "sd1": sd1,
                "raw_mean": raw_mean,
                "raw_sd": np.sqrt(raw_second - raw_mean**2),
            },
        )
        for specification in (beta, mixture):
            root = build_root(specification)
            self.assertEqual(root.mean, 0.0)
            self.assertEqual(root.variance, 1.0)
            values = root.sample(20, np.random.default_rng(18))
            np.testing.assert_array_equal(
                values,
                build_root(RootSpec.from_json(specification.canonical_json())).sample(
                    20, np.random.default_rng(18)
                ),
            )

        with self.assertRaises(ValueError):
            RootSpec.create(
                "bdpfn.root.beta_standardized",
                "1.0.0",
                VariableType.CONTINUOUS,
                None,
                {"alpha": 0.2, "beta": 3.0, "raw_mean": 0.1, "raw_sd": 1.0},
            )
        bad_mixture = mixture.parameter_dict()
        bad_mixture["raw_mean"] += 0.1
        with self.assertRaises(ValueError):
            RootSpec.create(
                mixture.family_id,
                mixture.version,
                mixture.variable_type,
                mixture.cardinality,
                bad_mixture,
            )


class FiniteRootTests(unittest.TestCase):
    def test_binary_and_finite_roots_have_exact_pmf_moments_and_support(self) -> None:
        bernoulli = RootSpec.create(
            "bdpfn.root.bernoulli",
            "1.0.0",
            VariableType.BINARY,
            2,
            {"p": 0.3, "probability_mode": "uniform_0.1_0.9"},
        )
        base = np.arange(1, 4, dtype=float) ** (-1.2)
        base /= base.sum()
        floored = 0.9 * base + 0.1 / 3.0
        categorical = RootSpec.create(
            "bdpfn.root.categorical_truncated_zipf",
            "1.0.0",
            VariableType.CATEGORICAL,
            3,
            {
                "exponent": 1.2,
                "base_pmf": base.tolist(),
                "pmf": floored.tolist(),
            },
        )
        ordinal_pmf = np.arange(1, 4, dtype=float) ** (-1.2)
        ordinal_pmf /= ordinal_pmf.sum()
        ordinal = RootSpec.create(
            "bdpfn.root.ordinal_truncated_zipf",
            "1.0.0",
            VariableType.ORDINAL,
            3,
            {"exponent": 1.2, "pmf": ordinal_pmf.tolist()},
        )
        for specification in (bernoulli, categorical, ordinal):
            root = build_root(specification)
            self.assertAlmostEqual(float(root.pmf.sum()), 1.0)
            values = root.sample(30, np.random.default_rng(5))
            self.assertTrue(np.issubdtype(values.dtype, np.integer))
            self.assertTrue((values >= 0).all())
            self.assertTrue((values < specification.cardinality).all())
        self.assertAlmostEqual(bernoulli.parameter_dict()["p"], build_root(bernoulli).mean)
        np.testing.assert_allclose(build_root(categorical).pmf, floored)
        self.assertGreaterEqual(float(build_root(categorical).pmf.min()), 0.1 / 3.0)
        self.assertTrue(build_root(categorical).contains(np.asarray([0.0, 1, 2])))
        self.assertFalse(build_root(categorical).contains(np.asarray([1.5])))
        self.assertFalse(build_root(categorical).contains(np.asarray([3])))

    def test_finite_root_specs_reject_bad_probability_and_floor_contracts(self) -> None:
        with self.assertRaises(ValueError):
            RootSpec.create(
                "bdpfn.root.bernoulli",
                "1.0.0",
                VariableType.BINARY,
                2,
                {"p": 1.2, "probability_mode": "uniform_0.1_0.9"},
            )
        with self.assertRaises(ValueError):
            RootSpec.create(
                "bdpfn.root.categorical_uniform",
                "1.0.0",
                VariableType.CATEGORICAL,
                3,
                {"base_pmf": [1 / 3] * 3, "pmf": [0.8, 0.1, 0.1]},
            )


class RootSamplingTests(unittest.TestCase):
    def test_root_sampling_policy_matches_active_v1_and_is_local(self) -> None:
        manifest = active_v1_manifest()
        for variable_type, law_id in (
            (VariableType.CONTINUOUS, "root.continuous"),
            (VariableType.BINARY, "root.binary"),
            (VariableType.CATEGORICAL, "root.categorical_pmf"),
            (VariableType.ORDINAL, "root.ordinal"),
        ):
            self.assertEqual(
                root_family_weights(variable_type),
                manifest.law_weights(law_id),
            )

        np.random.seed(241)
        global_before = np.random.get_state()
        first = sample_root_spec(
            VariableSpec("x", VariableType.CONTINUOUS),
            np.random.default_rng(991),
        )
        second = sample_root_spec(
            VariableSpec("x", VariableType.CONTINUOUS),
            np.random.default_rng(991),
        )
        global_after = np.random.get_state()
        self.assertEqual(first, second)
        self.assertEqual(global_before[0], global_after[0])
        np.testing.assert_array_equal(global_before[1], global_after[1])
        self.assertEqual(global_before[2:], global_after[2:])

    def test_random_root_parameters_are_frozen_valid_and_cover_all_families(self) -> None:
        variables = (
            VariableSpec("xc", VariableType.CONTINUOUS),
            VariableSpec("xb", VariableType.BINARY, 2),
            VariableSpec("xk", VariableType.CATEGORICAL, 7),
            VariableSpec("xo", VariableType.ORDINAL, 5),
        )
        seen = {variable.variable_type: set() for variable in variables}
        for seed in range(300):
            for variable in variables:
                specification = sample_root_spec(
                    variable,
                    np.random.default_rng(seed),
                )
                seen[variable.variable_type].add(specification.family_id)
                self.assertEqual(
                    specification,
                    RootSpec.from_json(specification.canonical_json()),
                )
                build_root(specification)
        self.assertEqual(len(seen[VariableType.CONTINUOUS]), 6)
        self.assertEqual(len(seen[VariableType.BINARY]), 1)
        self.assertEqual(len(seen[VariableType.CATEGORICAL]), 3)
        self.assertEqual(len(seen[VariableType.ORDINAL]), 2)

        for seed in range(100):
            binary = sample_root_spec(
                variables[1],
                np.random.default_rng(seed),
            ).parameter_dict()
            self.assertIn(binary["probability_mode"], {
                "point_mass_0.5",
                "uniform_0.1_0.9",
            })
            self.assertGreaterEqual(binary["p"], 0.1)
            self.assertLessEqual(binary["p"], 0.9)


class IndependentProductCovariateTests(unittest.TestCase):
    def test_product_contains_enforces_each_root_support_after_schema_normalization(self) -> None:
        schema = CovariateSchema(
            "bdpfn-covariate-schema-v1",
            "mixed",
            "2-5",
            0,
            (
                VariableSpec("x0", VariableType.CONTINUOUS),
                VariableSpec("x1", VariableType.CATEGORICAL, 3),
            ),
        )
        uniform = RootSpec.create(
            "bdpfn.root.uniform",
            "1.0.0",
            VariableType.CONTINUOUS,
            None,
            {"low": -np.sqrt(3.0), "high": np.sqrt(3.0)},
        )
        finite = RootSpec.create(
            "bdpfn.root.categorical_uniform",
            "1.0.0",
            VariableType.CATEGORICAL,
            3,
            {"base_pmf": [1 / 3] * 3, "pmf": [1 / 3] * 3},
        )
        law = IndependentProductCovariates(schema, (uniform, finite))
        self.assertTrue(
            law.contains(np.asarray([[-np.sqrt(3.0), 0.0], [np.sqrt(3.0), 2.0]]))
        )
        self.assertFalse(law.contains(np.asarray([[np.sqrt(3.0) + 1e-12, 1.0]])))

    def test_factory_draws_frozen_schema_and_roots_then_rebuilds_rng_free(self) -> None:
        factory = IndependentProductCovariateFactory()
        first = factory.sample_spec(np.random.default_rng(42), {}, {})
        second = factory.sample_spec(np.random.default_rng(42), {}, {})
        self.assertEqual(first, second)
        self.assertEqual(first.family_id, "bdpfn.covariate.independent_product")

        law = factory.build(first)
        rebuilt = factory.build(type(first).from_dict(first.to_dict()))
        self.assertEqual(law.schema, rebuilt.schema)
        self.assertEqual(law.root_specs, rebuilt.root_specs)
        rows = law.sample(64, np.random.default_rng(112))
        rebuilt_rows = rebuilt.sample(64, np.random.default_rng(112))
        self.assertIsInstance(rows, TypedCovariateBatch)
        self.assertEqual(rows.schema, law.schema)
        for left, right in zip(rows.columns, rebuilt_rows.columns):
            np.testing.assert_array_equal(left, right)
        self.assertTrue(law.contains(rows))

    def test_independent_product_support_is_semantic_for_mixed_outer_matrix(self) -> None:
        schema = CovariateSchema(
            "bdpfn-covariate-schema-v1",
            "mixed",
            "2-5",
            0,
            (
                VariableSpec("x0", VariableType.CONTINUOUS),
                VariableSpec("x1", VariableType.CATEGORICAL, 3),
            ),
        )
        factory = IndependentProductCovariateFactory()
        specification = factory.sample_spec(
            np.random.default_rng(8),
            {"schema": schema.to_dict()},
            {},
        )
        law = factory.build(specification)
        self.assertEqual(law.schema, schema)
        self.assertTrue(law.contains(np.asarray([[0.2, 1.0], [0.0, 2.0]])))
        self.assertFalse(law.contains(np.asarray([[0.2, 1.5]])))
        self.assertFalse(law.contains(np.asarray([[0.2, 3.0]])))
        self.assertFalse(law.contains(np.asarray([[np.inf, 1.0]])))


class TypedCovariatePublicSurfaceTests(unittest.TestCase):
    def test_public_exports_and_builtin_registry_resolve_independent_product(self) -> None:
        self.assertIs(backdoor_scm.CovariateSchema, CovariateSchema)
        self.assertIs(backdoor_scm.RootSpec, RootSpec)
        self.assertIs(backdoor_scm.sample_schema, sample_schema)
        self.assertIs(backdoor_scm.sample_root_spec, sample_root_spec)
        factory = backdoor_scm.build_builtin_registry().resolve(
            "covariate",
            "bdpfn.covariate.independent_product",
            "1.0.0",
        )
        self.assertIsInstance(factory, IndependentProductCovariateFactory)


if __name__ == "__main__":
    unittest.main()
