import math
import unittest

import numpy as np

from backdoor_scm.families.covariates import (
    GaussianCopulaCovariateFactory,
    LowRankGaussianCovariateFactory,
    StudentTCopulaCovariateFactory,
    build_covariates,
    covariate_dependence_law,
    sample_covariate_spec,
)
from backdoor_scm.families.roots import (
    RootSpec,
    build_root,
    root_family_weights,
    sample_root_spec,
)
from backdoor_scm.specs import FamilyRef
from backdoor_scm.schema import CovariateSchema, VariableSpec, VariableType


def _continuous_roots():
    beta_mean = 2.0 / 5.0
    beta_sd = math.sqrt(2.0 * 3.0 / (5.0**2 * 6.0))
    weight, separation, sd0, sd1 = 0.4, 1.5, 0.7, 1.2
    mean0, mean1 = -separation / 2.0, separation / 2.0
    mixture_mean = weight * mean0 + (1.0 - weight) * mean1
    second = weight * (sd0**2 + mean0**2) + (1.0 - weight) * (
        sd1**2 + mean1**2
    )
    mixture_sd = math.sqrt(second - mixture_mean**2)
    return (
        RootSpec.create("bdpfn.root.normal", "1.0.0", VariableType.CONTINUOUS, None, {}),
        RootSpec.create(
            "bdpfn.root.uniform",
            "1.0.0",
            VariableType.CONTINUOUS,
            None,
            {"low": -math.sqrt(3.0), "high": math.sqrt(3.0)},
        ),
        RootSpec.create(
            "bdpfn.root.laplace_standardized",
            "1.0.0",
            VariableType.CONTINUOUS,
            None,
            {"location": 0.0, "scale": 1.0 / math.sqrt(2.0)},
        ),
        RootSpec.create(
            "bdpfn.root.beta_standardized",
            "1.0.0",
            VariableType.CONTINUOUS,
            None,
            {"alpha": 2.0, "beta": 3.0, "raw_mean": beta_mean, "raw_sd": beta_sd},
        ),
        RootSpec.create(
            "bdpfn.root.gaussian_mixture2_standardized",
            "1.0.0",
            VariableType.CONTINUOUS,
            None,
            {
                "weight": weight,
                "separation": separation,
                "sd0": sd0,
                "sd1": sd1,
                "raw_mean": mixture_mean,
                "raw_sd": mixture_sd,
            },
        ),
    )


def _continuous_schema(dimension):
    return CovariateSchema(
        "bdpfn-covariate-schema-v1",
        "continuous_only",
        "2-5",
        0,
        tuple(
            VariableSpec(f"x{index}", VariableType.CONTINUOUS)
            for index in range(dimension)
        ),
    )


class ContinuousRootTransformTests(unittest.TestCase):
    def test_all_continuous_cdf_ppf_round_trip_without_batch_statistics(self) -> None:
        probabilities = np.asarray([1e-6, 0.1, 0.5, 0.9, 1.0 - 1e-6])
        for specification in _continuous_roots():
            root = build_root(specification)
            quantiles = root.ppf(probabilities)
            np.testing.assert_allclose(
                root.cdf(quantiles),
                probabilities,
                rtol=0.0,
                atol=2e-12,
            )
            self.assertFalse(quantiles.flags.writeable)
            self.assertFalse(root.cdf(quantiles).flags.writeable)
        with self.assertRaises(ValueError):
            build_root(_continuous_roots()[0]).ppf(np.asarray([-0.1]))

    def test_student_t_marginal_and_finite_generalized_quantile_are_exact(self) -> None:
        self.assertEqual(
            root_family_weights(VariableType.CONTINUOUS),
            {
                "normal": 0.20,
                "uniform": 0.15,
                "beta": 0.10,
                "laplace": 0.15,
                "gaussian_mixture2": 0.15,
                "student_t": 0.25,
            },
        )
        reached = {
            sample_root_spec(
                VariableSpec("x", VariableType.CONTINUOUS),
                np.random.default_rng(seed),
            ).family_id
            for seed in range(100)
        }
        self.assertIn("bdpfn.root.student_t_standardized", reached)
        df = 3.5
        scale = math.sqrt((df - 2.0) / df)
        student = build_root(
            RootSpec.create(
                "bdpfn.root.student_t_standardized",
                "1.0.0",
                VariableType.CONTINUOUS,
                None,
                {"df": df, "scale": scale},
            )
        )
        probability = np.asarray([1e-5, 0.1, 0.5, 0.9, 1.0 - 1e-5])
        np.testing.assert_allclose(
            student.cdf(student.ppf(probability)), probability, rtol=0.0, atol=2e-12
        )
        self.assertEqual(student.mean, 0.0)
        self.assertEqual(student.variance, 1.0)

        bernoulli = build_root(
            RootSpec.create(
                "bdpfn.root.bernoulli",
                "1.0.0",
                VariableType.BINARY,
                2,
                {"p": 0.5, "probability_mode": "point_mass_0.5"},
            )
        )
        np.testing.assert_array_equal(
            bernoulli.ppf(np.asarray([0.0, 0.49, 0.5, 0.5001, 1.0])),
            np.asarray([0, 0, 0, 1, 1]),
        )

        base = np.asarray([0.2, 0.3, 0.5])
        categorical_pmf = 0.9 * base + 0.1 / 3.0
        categorical = build_root(
            RootSpec.create(
                "bdpfn.root.categorical_dirichlet1",
                "1.0.0",
                VariableType.CATEGORICAL,
                3,
                {"base_pmf": base.tolist(), "pmf": categorical_pmf.tolist()},
            )
        )
        first_boundary = categorical_pmf[0]
        second_boundary = float(categorical_pmf[:2].sum())
        np.testing.assert_array_equal(
            categorical.ppf(
                np.asarray(
                    [
                        0.0,
                        first_boundary,
                        np.nextafter(first_boundary, 1.0),
                        second_boundary,
                        np.nextafter(second_boundary, 1.0),
                        1.0,
                    ]
                )
            ),
            np.asarray([0, 0, 1, 1, 2, 2]),
        )

        ordinal_pmf = np.asarray([6.0 / 11.0, 3.0 / 11.0, 2.0 / 11.0])
        ordinal = build_root(
            RootSpec.create(
                "bdpfn.root.ordinal_truncated_zipf",
                "1.0.0",
                VariableType.ORDINAL,
                3,
                {"exponent": 1.0, "pmf": ordinal_pmf.tolist()},
            )
        )
        first_boundary = ordinal_pmf[0]
        second_boundary = float(ordinal_pmf[:2].sum())
        np.testing.assert_array_equal(
            ordinal.ppf(
                np.asarray(
                    [
                        0.0,
                        first_boundary,
                        np.nextafter(first_boundary, 1.0),
                        second_boundary,
                        np.nextafter(second_boundary, 1.0),
                        1.0,
                    ]
                )
            ),
            np.asarray([0, 0, 1, 1, 2, 2]),
        )


class GaussianCopulaTests(unittest.TestCase):
    def test_full_gram_copula_is_frozen_valid_and_partition_invariant(self) -> None:
        schema = _continuous_schema(5)
        roots = _continuous_roots()
        hyperparameters = {
            "schema": schema.to_dict(),
            "root_specs": [root.to_dict() for root in roots],
        }
        factory = GaussianCopulaCovariateFactory()
        specification = factory.sample_spec(
            np.random.default_rng(101), hyperparameters, {}
        )
        self.assertEqual(
            specification,
            factory.sample_spec(np.random.default_rng(101), hyperparameters, {}),
        )
        parameters = specification.parameter_dict()
        correlation = np.asarray(parameters["correlation"])
        np.testing.assert_allclose(correlation, correlation.T, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.diag(correlation), 1.0, rtol=0.0, atol=1e-12)
        np.linalg.cholesky(correlation)
        self.assertEqual(np.asarray(parameters["gram_factor"]).shape, (5, 5))

        law = factory.build(specification)
        complete = law.sample(17, np.random.default_rng(103))
        split_rng = np.random.default_rng(103)
        first = law.sample(6, split_rng)
        second = law.sample(11, split_rng)
        for whole, left, right in zip(complete.columns, first.columns, second.columns):
            np.testing.assert_array_equal(whole, np.concatenate((left, right)))
        self.assertTrue(law.contains(complete))
        for root, column in zip(roots, complete.columns):
            self.assertTrue(build_root(root).contains(column))

        malformed = dict(parameters)
        malformed["correlation"] = np.eye(5).tolist()
        with self.assertRaises(ValueError):
            factory.build(
                FamilyRef.create(
                    specification.role,
                    specification.family_id,
                    specification.version,
                    specification.semantic_class_id,
                    malformed,
                )
            )


class LowRankGaussianTests(unittest.TestCase):
    def test_low_rank_parameters_are_frozen_recomputed_and_local(self) -> None:
        schema = _continuous_schema(5)
        roots = _continuous_roots()
        hyperparameters = {
            "schema": schema.to_dict(),
            "root_specs": [root.to_dict() for root in roots],
        }
        factory = LowRankGaussianCovariateFactory()
        np.random.seed(707)
        global_before = np.random.get_state()
        specification = factory.sample_spec(
            np.random.default_rng(109), hyperparameters, {}
        )
        global_after = np.random.get_state()
        self.assertEqual(global_before[0], global_after[0])
        np.testing.assert_array_equal(global_before[1], global_after[1])
        self.assertEqual(global_before[2:], global_after[2:])
        self.assertEqual(
            specification,
            factory.sample_spec(np.random.default_rng(109), hyperparameters, {}),
        )
        parameters = specification.parameter_dict()
        rank = int(parameters["rank"])
        self.assertGreaterEqual(rank, 1)
        self.assertLessEqual(rank, min(10, schema.dimension - 1))
        loadings = np.asarray(parameters["loadings"])
        residual = np.asarray(parameters["residual"])
        correlation = np.asarray(parameters["correlation"])
        self.assertEqual(loadings.shape, (schema.dimension, rank))
        np.testing.assert_array_equal(residual, np.ones(schema.dimension))
        covariance = loadings @ loadings.T + np.diag(residual)
        scale = np.sqrt(np.diag(covariance))
        raw_correlation = covariance / scale[:, None] / scale[None, :]
        alpha = float(parameters["strength_alpha"])
        expected = (1.0 - alpha) * np.eye(schema.dimension) + alpha * raw_correlation
        np.testing.assert_allclose(correlation, expected, rtol=0.0, atol=1e-12)
        np.linalg.cholesky(correlation)
        law = factory.build(specification)
        sample = law.sample(2000, np.random.default_rng(113))
        self.assertTrue(law.contains(sample))
        for root, column in zip(roots, sample.columns):
            self.assertAlmostEqual(float(np.mean(column)), build_root(root).mean, delta=0.12)
            self.assertAlmostEqual(float(np.var(column)), build_root(root).variance, delta=0.18)

    def test_gaussian_dependent_modules_support_mixed_schema(self) -> None:
        schema = CovariateSchema(
            "bdpfn-covariate-schema-v1",
            "mixed",
            "2-5",
            0,
            (
                VariableSpec("x0", VariableType.CONTINUOUS),
                VariableSpec("x1", VariableType.BINARY, 2),
            ),
        )
        roots = (
            _continuous_roots()[0],
            RootSpec.create(
                "bdpfn.root.bernoulli",
                "1.0.0",
                VariableType.BINARY,
                2,
                {"p": 0.5, "probability_mode": "point_mass_0.5"},
            ),
        )
        hyperparameters = {
            "schema": schema.to_dict(),
            "root_specs": [root.to_dict() for root in roots],
        }
        for factory in (
            GaussianCopulaCovariateFactory(),
            LowRankGaussianCovariateFactory(),
        ):
            specification = factory.sample_spec(
                np.random.default_rng(127), hyperparameters, {}
            )
            parameters = specification.parameter_dict()
            self.assertIn(
                parameters["dependence_strength"], {"weak", "moderate", "strong"}
            )
            law = factory.build(specification)
            self.assertTrue(law.contains(law.sample(31, np.random.default_rng(128))))


class StudentTCopulaTests(unittest.TestCase):
    def test_student_t_copula_is_frozen_heavy_tailed_and_mixed_typed(self) -> None:
        schema = CovariateSchema(
            "bdpfn-covariate-schema-v1",
            "mixed",
            "2-5",
            0,
            (
                VariableSpec("continuous", VariableType.CONTINUOUS),
                VariableSpec("binary", VariableType.BINARY, 2),
                VariableSpec("ordinal", VariableType.ORDINAL, 3),
            ),
        )
        roots = (
            _continuous_roots()[0],
            RootSpec.create(
                "bdpfn.root.bernoulli",
                "1.0.0",
                VariableType.BINARY,
                2,
                {"p": 0.4, "probability_mode": "uniform_0.1_0.9"},
            ),
            RootSpec.create(
                "bdpfn.root.ordinal_randint",
                "1.0.0",
                VariableType.ORDINAL,
                3,
                {"pmf": [1.0 / 3.0] * 3},
            ),
        )
        hyperparameters = {
            "schema": schema.to_dict(),
            "root_specs": [root.to_dict() for root in roots],
        }
        factory = StudentTCopulaCovariateFactory()
        specification = factory.sample_spec(
            np.random.default_rng(149), hyperparameters, {}
        )
        parameters = specification.parameter_dict()
        self.assertGreaterEqual(parameters["df"], 3.0)
        self.assertLessEqual(parameters["df"], 32.0)
        self.assertIn(parameters["dependence_strength"], {"weak", "moderate", "strong"})
        self.assertEqual(
            specification,
            factory.sample_spec(np.random.default_rng(149), hyperparameters, {}),
        )
        law = factory.build(specification)
        self.assertGreater(law.upper_tail_dependence(0, 1), 0.0)
        self.assertEqual(
            law.upper_tail_dependence(0, 1),
            law.upper_tail_dependence(1, 0),
        )
        complete = law.sample(101, np.random.default_rng(151))
        split_rng = np.random.default_rng(151)
        parts = (law.sample(40, split_rng), law.sample(61, split_rng))
        for whole, left, right in zip(
            complete.columns, parts[0].columns, parts[1].columns
        ):
            np.testing.assert_array_equal(whole, np.concatenate((left, right)))
        self.assertTrue(law.contains(complete))


class CovariateChooserTests(unittest.TestCase):
    def test_dependence_law_is_exact_ordered_and_schema_conditional(self) -> None:
        self.assertEqual(
            covariate_dependence_law(_continuous_schema(5)),
            (
                ("independent_product", 0.10),
                ("gaussian_copula", 0.30),
                ("low_rank_gaussian", 0.20),
                ("student_t_copula", 0.40),
            ),
        )
        one_dimensional = CovariateSchema(
            "bdpfn-covariate-schema-v1",
            "continuous_only",
            "1",
            0,
            (VariableSpec("x0", VariableType.CONTINUOUS),),
        )
        self.assertEqual(
            covariate_dependence_law(one_dimensional),
            (("independent_product", 1.0),),
        )
        mixed = CovariateSchema(
            "bdpfn-covariate-schema-v1",
            "mixed",
            "2-5",
            0,
            (
                VariableSpec("x0", VariableType.CONTINUOUS),
                VariableSpec("x1", VariableType.BINARY, 2),
            ),
        )
        self.assertEqual(
            covariate_dependence_law(mixed),
            (
                ("independent_product", 0.10),
                ("gaussian_copula", 0.30),
                ("low_rank_gaussian", 0.20),
                ("student_t_copula", 0.40),
            ),
        )

    def test_chooser_is_local_reproducible_reachable_and_buildable(self) -> None:
        schema = _continuous_schema(5)
        np.random.seed(811)
        before = np.random.get_state()
        first = sample_covariate_spec(np.random.default_rng(131), schema)
        second = sample_covariate_spec(np.random.default_rng(131), schema)
        self.assertEqual(first, second)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        law = build_covariates(first)
        sample = law.sample(31, np.random.default_rng(137))
        self.assertTrue(law.contains(sample))

        reached = {
            sample_covariate_spec(np.random.default_rng(seed), schema).family_id
            for seed in range(100)
        }
        self.assertEqual(
            reached,
            {
                "bdpfn.covariate.independent_product",
                "bdpfn.covariate.gaussian_copula",
                "bdpfn.covariate.low_rank_gaussian",
                "bdpfn.covariate.student_t_copula",
            },
        )

    def test_mixed_schema_reaches_all_dependence_regimes(self) -> None:
        schema = CovariateSchema(
            "bdpfn-covariate-schema-v1",
            "mixed",
            "2-5",
            0,
            (
                VariableSpec("x0", VariableType.CONTINUOUS),
                VariableSpec("x1", VariableType.BINARY, 2),
            ),
        )
        family_ids = {
            sample_covariate_spec(np.random.default_rng(seed), schema).family_id
            for seed in range(100)
        }
        self.assertEqual(
            family_ids,
            {
                "bdpfn.covariate.independent_product",
                "bdpfn.covariate.gaussian_copula",
                "bdpfn.covariate.low_rank_gaussian",
                "bdpfn.covariate.student_t_copula",
            },
        )


if __name__ == "__main__":
    unittest.main()
