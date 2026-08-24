import json
import math
import unittest
from dataclasses import FrozenInstanceError

import backdoor_scm
from backdoor_scm.active_v1 import active_v1_manifest
from backdoor_scm.capabilities import TruthCapability, TruthLevel
from backdoor_scm.manifest import ActivePriorManifest
from backdoor_scm.registry import ComponentRole, FamilyRegistry


class _RegistryFactory:
    outcome_type = None

    def __init__(self, family_id: str) -> None:
        self.role = ComponentRole.COVARIATE
        self.family_id = family_id
        self.version = "1.0.0"
        self.semantic_class_id = family_id.replace("test.", "semantic.")

    def sample_spec(self, rng, hyperparameters, context):  # pragma: no cover
        raise NotImplementedError

    def build(self, spec):  # pragma: no cover
        raise NotImplementedError


class TruthCapabilityTests(unittest.TestCase):
    def test_capability_is_frozen_versioned_and_canonical(self) -> None:
        capability = TruthCapability(
            capability_version="bdpfn-truth-capability-v1",
            truth_level=TruthLevel.ANALYTIC_EXACT,
            moments=("mean", "variance"),
            arbitrary_query=True,
            exact_support=True,
            query_cost_bound=4096,
        )

        with self.assertRaises(FrozenInstanceError):
            capability.arbitrary_query = False
        encoded = capability.canonical_json()
        self.assertEqual(encoded, capability.canonical_json())
        self.assertEqual(
            capability,
            TruthCapability.from_dict(json.loads(encoded)),
        )
        self.assertEqual(capability.capability_hash, capability.capability_hash)

    def test_certified_numeric_requires_explicit_contract_tolerances(self) -> None:
        with self.assertRaises(ValueError):
            TruthCapability(
                capability_version="bdpfn-truth-capability-v1",
                truth_level=TruthLevel.CERTIFIED_NUMERIC,
                moments=("mean",),
                arbitrary_query=True,
                exact_support=True,
                query_cost_bound=4096,
            )

        capability = TruthCapability(
            capability_version="bdpfn-truth-capability-v1",
            truth_level=TruthLevel.CERTIFIED_NUMERIC,
            moments=("mean",),
            arbitrary_query=True,
            exact_support=True,
            query_cost_bound=4096,
            atol=1e-8,
            rtol=1e-6,
        )
        self.assertEqual(capability.atol, 1e-8)
        self.assertEqual(capability.rtol, 1e-6)


class RegistrySnapshotTests(unittest.TestCase):
    def test_snapshot_digest_is_registration_order_invariant(self) -> None:
        first = FamilyRegistry()
        second = FamilyRegistry()
        for family_id in ("test.alpha", "test.beta"):
            first.register(_RegistryFactory(family_id))
        for family_id in ("test.beta", "test.alpha"):
            second.register(_RegistryFactory(family_id))

        self.assertEqual(first.snapshot_digest(), second.snapshot_digest())
        self.assertEqual(len(first.snapshot_digest()), 64)

        second.register(_RegistryFactory("test.gamma"))
        self.assertNotEqual(first.snapshot_digest(), second.snapshot_digest())


class ActiveV1ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = active_v1_manifest()

    def test_manifest_is_frozen_canonical_and_round_trips(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            self.manifest.manifest_version = "changed"
        encoded = self.manifest.canonical_json()
        rebuilt = ActivePriorManifest.from_dict(json.loads(encoded))
        self.assertEqual(self.manifest, rebuilt)
        self.assertEqual(self.manifest.manifest_hash, rebuilt.manifest_hash)
        self.assertEqual(len(self.manifest.registry_snapshot_digest), 64)

    def test_every_design_law_is_normalized_and_key_probabilities_are_frozen(self) -> None:
        for law in self.manifest.laws:
            self.assertTrue(
                math.isclose(
                    sum(choice.weight for choice in law.choices),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                law.law_id,
            )

        self.assertEqual(
            self.manifest.law_weights("schema.profile"),
            {
                "continuous_only": 0.30,
                "binary_only": 0.10,
                "categorical_or_ordinal_only": 0.10,
                "mixed": 0.50,
            },
        )
        self.assertEqual(
            self.manifest.law_weights("outcome.type"),
            {
                "continuous": 0.40,
                "bounded_continuous": 0.10,
                "binary": 0.20,
                "count": 0.10,
                "numeric_categorical_or_ordinal": 0.10,
                "nominal_categorical": 0.10,
            },
        )
        self.assertEqual(
            self.manifest.law_weights("arm.likelihood_coupling"),
            {"same": 0.70, "different_compatible": 0.30},
        )
        self.assertEqual(
            self.manifest.law_weights("root.continuous"),
            {
                "normal": 0.20,
                "uniform": 0.15,
                "beta": 0.10,
                "laplace": 0.15,
                "gaussian_mixture2": 0.15,
                "student_t": 0.25,
            },
        )
        self.assertEqual(
            self.manifest.law_weights("covariate.coordinate_dependence"),
            {
                "independent": 0.10,
                "gaussian_copula": 0.30,
                "low_rank_gaussian": 0.20,
                "student_t_copula": 0.40,
            },
        )
        self.assertEqual(
            self.manifest.law_weights("scalar.family"),
            {
                "constant": 0.05,
                "projection": 0.05,
                "sparse_affine": 0.15,
                "dense_affine": 0.10,
                "categorical_lookup": 0.08,
                "threshold_or_stump": 0.08,
                "shallow_tree_or_forest": 0.08,
                "polynomial": 0.10,
                "gam_hinge_or_spline": 0.10,
                "rbf": 0.08,
                "rff_or_fourier": 0.08,
                "small_mlp": 0.05,
            },
        )

    def test_main_family_mass_is_positive_and_partitions_are_disjoint(self) -> None:
        partitions = {
            name: self.manifest.family_partition(name)
            for name in ("train", "optional", "held_out", "excluded")
        }
        for family in partitions["train"]:
            self.assertIsNotNone(family.conditional_weight)
            self.assertGreater(family.conditional_weight, 0.0)
            self.assertTrue(family.weight_scope)

        identities = {
            name: {
                (family.role, family.family_id, family.version)
                for family in families
            }
            for name, families in partitions.items()
        }
        for left_index, left in enumerate(identities):
            for right in tuple(identities)[left_index + 1 :]:
                self.assertTrue(
                    identities[left].isdisjoint(identities[right]),
                    (left, right),
                )

        train_ids = {family.family_id for family in partitions["train"]}
        for required in (
            "bdpfn.root.normal",
            "bdpfn.root.gaussian_mixture2_standardized",
            "bdpfn.root.student_t_standardized",
            "bdpfn.covariate.student_t_copula",
            "bdpfn.propensity.constant_rct",
            "bdpfn.scalar.categorical_lookup",
            "bdpfn.scalar.small_mlp",
            "bdpfn.outcome.continuous_centered_gaussian",
            "bdpfn.outcome.bounded_beta",
            "bdpfn.outcome.binary_bernoulli",
            "bdpfn.outcome.count_poisson",
            "bdpfn.outcome.numeric_categorical",
            "bdpfn.outcome.nominal_categorical",
        ):
            self.assertIn(required, train_ids)

    def test_each_train_family_scope_is_an_explicit_normalized_nested_law(self) -> None:
        scoped_mass: dict[str, float] = {}
        for family in self.manifest.family_partition("train"):
            scoped_mass[family.weight_scope] = (
                scoped_mass.get(family.weight_scope, 0.0)
                + family.conditional_weight
            )
        for scope, total in scoped_mass.items():
            self.manifest.law(scope)
            self.assertTrue(
                math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12),
                (scope, total),
            )

    def test_singleton_likelihood_pools_are_explicitly_same_only(self) -> None:
        for outcome_type in (
            "binary",
            "numeric_categorical_or_ordinal",
            "nominal_categorical",
        ):
            law = self.manifest.law(f"outcome.likelihood.{outcome_type}")
            self.assertEqual(len(law.choices), 1)
            self.assertEqual(
                law.metadata_dict()["different_coupling_policy"],
                "same_only_if_singleton",
            )

    def test_manifest_is_not_training_ready_and_rejects_sampling_activation(self) -> None:
        self.assertFalse(self.manifest.training_ready)
        self.assertEqual(self.manifest.admission_certificates, ())
        with self.assertRaisesRegex(RuntimeError, "admission certificates"):
            self.manifest.require_training_ready()
        with self.assertRaisesRegex(RuntimeError, "admission certificates"):
            backdoor_scm.sample_task(
                self.manifest,
                global_seed=0,
                task_id=0,
                registry=FamilyRegistry(),
                source_id="causalpy@test-commit",
            )

    def test_available_plugin_registration_cannot_mutate_manifest(self) -> None:
        registry = FamilyRegistry()
        before_hash = self.manifest.manifest_hash
        before_digest = self.manifest.registry_snapshot_digest
        registry.register(_RegistryFactory("test.plugin"))
        self.assertEqual(self.manifest.manifest_hash, before_hash)
        self.assertEqual(self.manifest.registry_snapshot_digest, before_digest)

    def test_versioned_design_contract_is_public_without_enabling_sampling(self) -> None:
        self.assertIs(backdoor_scm.ActivePriorManifest, ActivePriorManifest)
        self.assertIs(backdoor_scm.TruthCapability, TruthCapability)
        self.assertIs(backdoor_scm.TruthLevel, TruthLevel)
        self.assertEqual(backdoor_scm.active_v1_manifest(), self.manifest)


if __name__ == "__main__":
    unittest.main()
