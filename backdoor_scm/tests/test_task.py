import dataclasses
import unittest

import numpy as np
import backdoor_scm

from backdoor_scm import (
    BackdoorTask,
    BackdoorTaskSpec,
    FamilyChoice,
    FamilyRef,
    build_builtin_registry,
    make_continuous_smoke_manifest,
    sample_task,
)
from backdoor_scm.families.outcomes import (
    CenteredLaplaceAffineOutcomeKernel,
    GaussianAffineOutcomeFactory,
    GaussianAffineOutcomeKernel,
)


class _GaussianAffineOutcomeFactoryV2(GaussianAffineOutcomeFactory):
    family_id = "test.outcome.gaussian_affine_v2"
    version = "3.0.0"


class _UnitNormalRng:
    def normal(self, size):
        return np.ones(size)


class _LaplaceScaleSpy:
    def __init__(self):
        self.scale = None

    def laplace(self, loc, scale, size):
        self.scale = scale
        return np.zeros(size)


class TaskTests(unittest.TestCase):
    def setUp(self):
        self.registry = build_builtin_registry()
        self.manifest = make_continuous_smoke_manifest(dimension=3)
        self.task = sample_task(
            self.manifest,
            global_seed=123,
            task_id=17,
            registry=self.registry,
            source_id="causalpy@test-commit",
        )

    def test_task_and_rows_are_reproducible_and_partition_invariant(self):
        again = sample_task(
            self.manifest,
            global_seed=123,
            task_id=17,
            registry=self.registry,
            source_id="causalpy@test-commit",
        )
        self.assertEqual(self.task.to_spec(), again.to_spec())

        whole = self.task.sample_rows(12, row_seed=501)
        prefix = self.task.sample_rows(5, row_seed=501)
        suffix = self.task.sample_rows(7, row_seed=501, start_row=5)
        np.testing.assert_array_equal(whole.x, np.vstack([prefix.x, suffix.x]))
        np.testing.assert_array_equal(whole.a, np.concatenate([prefix.a, suffix.a]))
        np.testing.assert_array_equal(whole.y, np.concatenate([prefix.y, suffix.y]))
        self.assertEqual(set(vars(whole)), {"x", "a", "y"})
        self.assertEqual(whole.x.shape, (12, 3))
        self.assertTrue(np.isin(whole.a, [0, 1]).all())

    def test_sampling_identity_excludes_source_but_preserves_full_integrity_hash(self):
        spec = self.task.to_spec()
        other_source_spec = dataclasses.replace(
            spec,
            source_id="causalpy@same-law-other-source",
        )
        self.assertNotEqual(spec.task_spec_hash, other_source_spec.task_spec_hash)
        self.assertEqual(spec.sampling_identity, other_source_spec.sampling_identity)

        other_source_task = BackdoorTask.from_spec(other_source_spec, self.registry)
        original_rows = self.task.sample_rows(16, row_seed=808)
        other_rows = other_source_task.sample_rows(16, row_seed=808)
        np.testing.assert_array_equal(original_rows.x, other_rows.x)
        np.testing.assert_array_equal(original_rows.a, other_rows.a)
        np.testing.assert_array_equal(original_rows.y, other_rows.y)

        parameters = spec.propensity.parameter_dict()
        positive_propensity = FamilyRef.create(
            role=spec.propensity.role,
            family_id=spec.propensity.family_id,
            version=spec.propensity.version,
            semantic_class_id=spec.propensity.semantic_class_id,
            parameters={**parameters, "intercept": 0.0},
        )
        negative_propensity = FamilyRef.create(
            role=spec.propensity.role,
            family_id=spec.propensity.family_id,
            version=spec.propensity.version,
            semantic_class_id=spec.propensity.semantic_class_id,
            parameters={**parameters, "intercept": np.float64(-0.0)},
        )
        self.assertEqual(
            dataclasses.replace(
                spec,
                propensity=positive_propensity,
            ).sampling_identity,
            dataclasses.replace(
                spec,
                propensity=negative_propensity,
            ).sampling_identity,
        )

    def test_query_sampling_is_partitioned_readonly_and_row_stream_independent(self):
        whole = self.task.sample_query(12, query_seed=707)
        prefix = self.task.sample_query(5, query_seed=707)
        suffix = self.task.sample_query(7, query_seed=707, start_query=5)

        self.assertIsInstance(whole, backdoor_scm.SemanticQueryBatch)
        self.assertEqual(set(vars(whole)), {"x"})
        self.assertEqual(whole.x.shape, (12, 3))
        np.testing.assert_array_equal(whole.x, np.vstack([prefix.x, suffix.x]))
        self.assertFalse(whole.x.flags.writeable)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            whole.x = np.zeros_like(whole.x)
        with self.assertRaises(ValueError):
            whole.x[0, 0] = 999.0

        source = np.array([[1.0, 2.0, 3.0]])
        owned = backdoor_scm.SemanticQueryBatch(source)
        source[0, 0] = 999.0
        self.assertEqual(owned.x[0, 0], 1.0)
        self.assertFalse(owned.x.flags.writeable)

        self.task.sample_rows(9, row_seed=707)
        repeated = self.task.sample_query(12, query_seed=707)
        np.testing.assert_array_equal(whole.x, repeated.x)

        from_batch = self.task.truth(whole)
        from_array = self.task.truth(whole.x)
        np.testing.assert_array_equal(from_batch.propensity, from_array.propensity)
        np.testing.assert_array_equal(from_batch.mu0, from_array.mu0)
        np.testing.assert_array_equal(from_batch.mu1, from_array.mu1)
        np.testing.assert_array_equal(from_batch.tau, from_array.tau)

        incompatible = backdoor_scm.load_task(
            dataclasses.replace(
                self.task.to_spec(),
                numpy_runtime_version="0.0.0-test-runtime",
            ),
            self.registry,
        )
        with self.assertRaises(ValueError):
            incompatible.sample_query(1, query_seed=0)

    def test_truth_is_exact_permutation_invariant_and_in_support(self):
        x = np.array([[0.0, 1.0, -1.0], [2.0, 0.5, 0.25]])
        truth = self.task.truth(x)
        reversed_truth = self.task.truth(x[::-1])

        np.testing.assert_allclose(truth.tau, truth.mu1 - truth.mu0)
        np.testing.assert_allclose(truth.propensity[::-1], reversed_truth.propensity)
        np.testing.assert_allclose(truth.mu0[::-1], reversed_truth.mu0)
        np.testing.assert_allclose(truth.mu1[::-1], reversed_truth.mu1)
        self.assertTrue((truth.propensity > 0.0).all())
        self.assertTrue((truth.propensity < 1.0).all())

        with self.assertRaises(ValueError):
            self.task.truth(np.array([[1.0, 2.0]]))
        with self.assertRaises(ValueError):
            self.task.truth(np.array([[1.0, np.nan, 2.0]]))

    def test_truth_distinguishes_shape_errors_from_out_of_support(self):
        support_error = backdoor_scm.OutOfSupportError

        with self.assertRaises(support_error):
            self.task.truth(np.array([[1.0, np.nan, 2.0]]))
        with self.assertRaises(support_error):
            self.task.truth(np.array([[1.0, np.inf, 2.0]]))

        for malformed in (
            np.array([1.0, 2.0, 3.0]),
            np.array([[1.0, 2.0]]),
        ):
            with self.assertRaises(ValueError) as raised:
                self.task.truth(malformed)
            self.assertNotIsInstance(raised.exception, support_error)

    def test_positivity_uses_spec_epsilon_on_extreme_inputs(self):
        spec = self.task.to_spec()
        epsilon = float(spec.propensity.parameter_dict()["epsilon"])
        dimension = int(spec.covariate.parameter_dict()["dimension"])
        extreme = np.asarray(
            [
                [-1.0e12] * dimension,
                [0.0] * dimension,
                [1.0e12] * dimension,
            ]
        )
        truth = self.task.truth(extreme)

        for values in (truth.propensity, truth.mu0, truth.mu1, truth.tau):
            self.assertTrue(np.isfinite(values).all())
        self.assertTrue((truth.propensity >= epsilon).all())
        self.assertTrue((truth.propensity <= 1.0 - epsilon).all())

    def test_public_truth_methods_match_truth_batch_and_share_validation(self):
        x = np.array(
            [
                [0.0, 1.0, -1.0],
                [2.0, 0.5, 0.25],
                [-3.0, 1.25, 0.75],
            ]
        )
        truth = self.task.truth(x)
        np.testing.assert_array_equal(self.task.propensity(x), truth.propensity)
        np.testing.assert_array_equal(self.task.mu(0, x), truth.mu0)
        np.testing.assert_array_equal(self.task.mu(1, x), truth.mu1)
        np.testing.assert_array_equal(self.task.tau(x), truth.tau)
        np.testing.assert_array_equal(
            self.task.tau(x[::-1]),
            truth.tau[::-1],
        )

        for values in (
            self.task.propensity(x),
            self.task.mu(0, x),
            self.task.mu(1, x),
            self.task.tau(x),
        ):
            self.assertFalse(values.flags.writeable)
        for invalid_arm in (-1, 2, 0.5):
            with self.assertRaises(ValueError):
                self.task.mu(invalid_arm, x)
        for method in (
            self.task.propensity,
            self.task.tau,
            lambda query: self.task.mu(0, query),
        ):
            with self.assertRaises(backdoor_scm.OutOfSupportError):
                method(np.array([[1.0, np.nan, 2.0]]))
            with self.assertRaises(ValueError) as raised:
                method(np.array([[1.0, 2.0]]))
            self.assertNotIsInstance(
                raised.exception,
                backdoor_scm.OutOfSupportError,
            )

    def test_spec_round_trip_rebuilds_rows_truth_and_provenance(self):
        spec = BackdoorTaskSpec.from_json(self.task.to_spec().canonical_json())
        rebuilt = BackdoorTask.from_spec(spec, self.registry)
        rows = self.task.sample_rows(8, row_seed=91)
        rebuilt_rows = rebuilt.sample_rows(8, row_seed=91)
        np.testing.assert_array_equal(rows.x, rebuilt_rows.x)
        np.testing.assert_array_equal(rows.a, rebuilt_rows.a)
        np.testing.assert_array_equal(rows.y, rebuilt_rows.y)

        provenance = rebuilt.provenance()
        self.assertEqual(provenance.manifest_hash, self.manifest.manifest_hash)
        self.assertEqual(provenance.task_spec_hash, spec.task_spec_hash)
        self.assertEqual(provenance.sampling_identity, spec.sampling_identity)
        self.assertEqual(provenance.source_id, "causalpy@test-commit")
        self.assertEqual(dict(provenance.seed_ids)["global_seed"], 123)
        self.assertEqual(dict(provenance.seed_ids)["task_id"], 17)
        self.assertEqual(provenance.rng_algorithm, "PCG64")
        self.assertTrue(provenance.rng_version)
        self.assertEqual(len(provenance.component_families), 4)

    def test_provenance_rejects_malformed_sampling_identity(self):
        provenance = self.task.provenance()
        self.assertEqual(len(provenance.sampling_identity), 64)
        self.assertTrue(
            all(character in "0123456789abcdef" for character in provenance.sampling_identity)
        )
        with self.assertRaises(ValueError):
            dataclasses.replace(
                provenance,
                sampling_identity="not-a-sha256",
            )

    def test_rng_contract_and_numpy_runtime_provenance_are_separate(self):
        spec = self.task.to_spec()
        provenance = self.task.provenance()

        self.assertEqual(spec.rng_version, "bdpfn-keyed-pcg64-v1")
        self.assertEqual(spec.numpy_runtime_version, np.__version__)
        self.assertEqual(provenance.rng_version, spec.rng_version)
        self.assertEqual(
            provenance.numpy_runtime_version,
            spec.numpy_runtime_version,
        )

        other_runtime = dataclasses.replace(
            spec,
            numpy_runtime_version="0.0.0-test-runtime",
        )
        self.assertNotEqual(spec.task_spec_hash, other_runtime.task_spec_hash)
        self.assertEqual(spec.sampling_identity, other_runtime.sampling_identity)

    def test_spec_rejects_wrong_roles_and_inconsistent_likelihood_modes(self):
        spec = self.task.to_spec()
        with self.assertRaises(ValueError):
            dataclasses.replace(
                spec,
                covariate=dataclasses.replace(spec.covariate, role="outcome"),
            )

        same_class_new_version = dataclasses.replace(
            spec.outcome0,
            family_id="test.same_class_new_version",
            version="99.0.0",
        )
        same_spec = dataclasses.replace(
            spec,
            outcome_likelihood_mode="same",
            outcome1=same_class_new_version,
        )
        self.assertEqual(
            same_spec.outcome0.semantic_class_id,
            same_spec.outcome1.semantic_class_id,
        )
        with self.assertRaises(ValueError):
            dataclasses.replace(
                spec,
                outcome_likelihood_mode="different",
                outcome1=same_class_new_version,
            )

        different_class_id = (
            "continuous.laplace"
            if spec.outcome0.semantic_class_id == "continuous.gaussian"
            else "continuous.gaussian"
        )
        different_class = dataclasses.replace(
            spec.outcome0,
            semantic_class_id=different_class_id,
        )
        with self.assertRaises(ValueError):
            dataclasses.replace(
                spec,
                outcome_likelihood_mode="same",
                outcome1=different_class,
            )
        different_spec = dataclasses.replace(
            spec,
            outcome_likelihood_mode="different",
            outcome1=different_class,
        )
        self.assertNotEqual(
            different_spec.outcome0.semantic_class_id,
            different_spec.outcome1.semantic_class_id,
        )

    def test_different_mass_requires_two_registered_likelihood_classes(self):
        registry = build_builtin_registry()
        registry.register(_GaussianAffineOutcomeFactoryV2())
        gaussian_choice = next(
            choice
            for choice in self.manifest.outcome_choices
            if choice.family_id == "bdpfn.outcome.gaussian_affine"
        )
        same_class_manifest = dataclasses.replace(
            self.manifest,
            outcome_choices=(
                dataclasses.replace(gaussian_choice, weight=0.5),
                FamilyChoice.create(
                    "test.outcome.gaussian_affine_v2",
                    "3.0.0",
                    0.5,
                    gaussian_choice.hyperparameter_dict(),
                ),
            ),
        )
        with self.assertRaises(ValueError):
            sample_task(
                same_class_manifest,
                global_seed=123,
                task_id=0,
                registry=registry,
                source_id="causalpy@test-commit",
            )

    def test_load_allows_truth_but_sampling_rejects_rng_runtime_mismatch(self):
        spec = self.task.to_spec()
        query = np.array([[0.25, -0.5, 1.0]])
        expected_truth = self.task.truth(query)

        for incompatible_spec in (
            dataclasses.replace(spec, rng_algorithm="MT19937"),
            dataclasses.replace(spec, rng_version="incompatible-contract"),
            dataclasses.replace(
                spec,
                numpy_runtime_version="0.0.0-test-runtime",
            ),
        ):
            loaded = backdoor_scm.load_task(
                incompatible_spec.canonical_json(),
                self.registry,
            )
            loaded_truth = loaded.truth(query)
            np.testing.assert_array_equal(
                expected_truth.propensity,
                loaded_truth.propensity,
            )
            np.testing.assert_array_equal(expected_truth.mu0, loaded_truth.mu0)
            np.testing.assert_array_equal(expected_truth.mu1, loaded_truth.mu1)
            np.testing.assert_array_equal(expected_truth.tau, loaded_truth.tau)
            with self.assertRaises(ValueError):
                loaded.sample_rows(1, row_seed=0)

        missing_family_spec = dataclasses.replace(
            spec,
            covariate=dataclasses.replace(spec.covariate, version="missing"),
        )
        with self.assertRaises(KeyError):
            backdoor_scm.load_task(missing_family_spec, self.registry)

    def test_treatment_label_swap_is_exact(self):
        spec = self.task.to_spec()
        propensity_parameters = spec.propensity.parameter_dict()
        swapped_propensity = FamilyRef.create(
            role=spec.propensity.role,
            family_id=spec.propensity.family_id,
            version=spec.propensity.version,
            semantic_class_id=spec.propensity.semantic_class_id,
            parameters={
                **propensity_parameters,
                "intercept": -propensity_parameters["intercept"],
                "weights": tuple(-v for v in propensity_parameters["weights"]),
            },
        )
        swapped_spec = dataclasses.replace(
            spec,
            task_id=18,
            propensity=swapped_propensity,
            outcome0=spec.outcome1,
            outcome1=spec.outcome0,
        )
        swapped = BackdoorTask.from_spec(swapped_spec, self.registry)
        x = np.array([[0.1, -0.2, 0.3], [1.0, 2.0, -1.0]])
        original_truth = self.task.truth(x)
        swapped_truth = swapped.truth(x)

        self.assertEqual(swapped.to_spec().outcome0, spec.outcome1)
        self.assertEqual(swapped.to_spec().outcome1, spec.outcome0)
        np.testing.assert_allclose(
            swapped_truth.propensity,
            1.0 - original_truth.propensity,
        )
        np.testing.assert_allclose(swapped_truth.mu0, original_truth.mu1)
        np.testing.assert_allclose(swapped_truth.mu1, original_truth.mu0)
        np.testing.assert_allclose(swapped_truth.tau, -original_truth.tau)

    def test_both_continuous_likelihoods_are_centered_at_exact_mean(self):
        parameters = {
            "intercept": 2.0,
            "weights": (1.5, -0.5, 0.25),
            "noise_sd": 0.7,
        }
        x = np.array([[1.0, 2.0, -1.0], [0.0, 0.0, 0.0]])
        expected = np.array([2.25, 2.0])
        gaussian = GaussianAffineOutcomeKernel.from_parameters(parameters)
        laplace = CenteredLaplaceAffineOutcomeKernel.from_parameters(parameters)

        np.testing.assert_allclose(gaussian.mean(x), expected)
        np.testing.assert_allclose(laplace.mean(x), expected)
        np.testing.assert_allclose(
            gaussian.sample(x, _UnitNormalRng()) - expected,
            np.full(x.shape[0], parameters["noise_sd"]),
        )
        laplace_rng = _LaplaceScaleSpy()
        laplace.sample(x, laplace_rng)
        self.assertAlmostEqual(
            laplace_rng.scale,
            parameters["noise_sd"] / np.sqrt(2.0),
        )

        manifest = make_continuous_smoke_manifest(dimension=3)
        self.assertEqual(manifest.manifest_version, "p1-p3-continuous-smoke-v2")
        self.assertEqual(
            {choice.version for choice in manifest.outcome_choices},
            {"2.0.0"},
        )
        for choice in manifest.outcome_choices:
            hyperparameters = choice.hyperparameter_dict()
            self.assertIn("noise_sd_min", hyperparameters)
            self.assertIn("noise_sd_max", hyperparameters)
            self.assertNotIn("scale_min", hyperparameters)
            self.assertNotIn("scale_max", hyperparameters)
        for outcome_spec in (
            self.task.to_spec().outcome0,
            self.task.to_spec().outcome1,
        ):
            self.assertEqual(outcome_spec.version, "2.0.0")
            self.assertIn("noise_sd", outcome_spec.parameter_dict())
            self.assertNotIn("scale", outcome_spec.parameter_dict())

    def test_runtime_laws_and_task_composition_are_immutable(self):
        x = np.array([[0.25, -0.5, 1.0]])
        hash_before = self.task.to_spec().task_spec_hash
        truth_before = self.task.truth(x)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.task._covariate.dimension = 99
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.task._propensity.intercept = 0.0
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.task._propensity.weights = (0.0, 0.0, 0.0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.task._outcomes[0].noise_sd = 9.0
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.task._outcomes = tuple(reversed(self.task._outcomes))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.task._spec = dataclasses.replace(self.task.to_spec(), task_id=999)

        truth_after = self.task.truth(x)
        self.assertEqual(hash_before, self.task.to_spec().task_spec_hash)
        np.testing.assert_array_equal(truth_before.propensity, truth_after.propensity)
        np.testing.assert_array_equal(truth_before.mu0, truth_after.mu0)
        np.testing.assert_array_equal(truth_before.mu1, truth_after.mu1)
        np.testing.assert_array_equal(truth_before.tau, truth_after.tau)

    def test_task_hyperprior_realizes_same_and_compatible_different_modes(self):
        seen_modes = set()
        seen_families = set()
        for task_id in range(200):
            task = sample_task(
                self.manifest,
                global_seed=901,
                task_id=task_id,
                registry=self.registry,
                source_id="causalpy@test-commit",
            )
            spec = task.to_spec()
            seen_modes.add(spec.outcome_likelihood_mode)
            seen_families.update(
                (spec.outcome0.family_id, spec.outcome1.family_id)
            )
            if spec.outcome_likelihood_mode == "same":
                self.assertEqual(
                    spec.outcome0.semantic_class_id,
                    spec.outcome1.semantic_class_id,
                )
            else:
                self.assertNotEqual(
                    spec.outcome0.semantic_class_id,
                    spec.outcome1.semantic_class_id,
                )
            self.assertEqual(spec.outcome_type, "continuous")

        self.assertEqual(seen_modes, {"same", "different"})
        self.assertEqual(
            seen_families,
            {
                "bdpfn.outcome.gaussian_affine",
                "bdpfn.outcome.centered_laplace_affine",
            },
        )


if __name__ == "__main__":
    unittest.main()
