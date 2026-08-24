import dataclasses
import json
import math
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from backdoor_scm import (
    ComponentRole,
    FamilyChoice,
    FamilyRef,
    PriorManifest,
    build_builtin_registry,
    make_continuous_smoke_manifest,
    sample_task,
)
from backdoor_scm.specs import canonical_hash, canonical_json, freeze_value


class _DummyFactory:
    role = ComponentRole.COVARIATE
    family_id = "test.covariate.dummy"
    version = "1.0.0"
    semantic_class_id = "covariate.dummy"
    outcome_type = None

    def sample_spec(self, rng, hyperparameters, context):  # pragma: no cover
        raise NotImplementedError

    def build(self, spec):  # pragma: no cover
        raise NotImplementedError


class ContractTests(unittest.TestCase):
    def test_manifest_is_frozen_canonical_and_round_trips(self):
        manifest = make_continuous_smoke_manifest(dimension=3)

        with self.assertRaises(FrozenInstanceError):
            manifest.outcome_type = "binary"

        encoded = manifest.canonical_json()
        self.assertEqual(encoded, manifest.canonical_json())
        self.assertFalse("NaN" in encoded or "Infinity" in encoded)
        round_trip = PriorManifest.from_dict(json.loads(encoded))
        self.assertEqual(manifest, round_trip)
        self.assertEqual(manifest.manifest_hash, round_trip.manifest_hash)

    def test_likelihood_hyperprior_and_smoke_mass_are_explicit(self):
        manifest = make_continuous_smoke_manifest(dimension=2)
        weights = {
            choice.family_id: choice.weight
            for choice in manifest.outcome_choices
        }

        self.assertEqual(manifest.outcome_type, "continuous")
        self.assertAlmostEqual(manifest.same_likelihood_probability, 0.70)
        self.assertAlmostEqual(
            manifest.different_compatible_likelihood_probability,
            0.30,
        )
        self.assertGreater(weights["bdpfn.outcome.gaussian_affine"], 0.0)
        self.assertGreater(
            weights["bdpfn.outcome.centered_laplace_affine"],
            0.0,
        )

    def test_manifest_rejects_nonfinite_or_invalid_probabilities(self):
        with self.assertRaises(ValueError):
            FamilyChoice.create(
                "bad.family",
                "1.0.0",
                math.nan,
                {},
            )

        manifest = make_continuous_smoke_manifest(dimension=1)
        data = manifest.to_dict()
        data["same_likelihood_probability"] = 0.8
        with self.assertRaises(ValueError):
            PriorManifest.from_dict(data)

    def test_nested_mapping_has_tagged_canonical_round_trip(self):
        choice = FamilyChoice.create(
            "test.family",
            "1.0.0",
            1.0,
            {
                "outer": {
                    "beta": 2.0,
                    "alpha": {"lower": 0.1, "upper": 0.9},
                }
            },
        )
        encoded = json.loads(json.dumps(choice.to_dict(), allow_nan=False))
        round_trip = FamilyChoice.from_dict(encoded)

        self.assertEqual(choice, round_trip)
        self.assertEqual(
            round_trip.hyperparameter_dict()["outer"]["alpha"],
            {"lower": 0.1, "upper": 0.9},
        )

    def test_negative_zero_has_one_canonical_identity_everywhere(self):
        positive = {"top": 0.0, "nested": {"numpy": np.float64(0.0)}}
        negative = {"top": -0.0, "nested": {"numpy": np.float64(-0.0)}}
        frozen_positive = {"payload": freeze_value(positive)}
        frozen_negative = {"payload": freeze_value(negative)}
        self.assertEqual(
            canonical_json(frozen_positive),
            canonical_json(frozen_negative),
        )
        self.assertEqual(canonical_hash(frozen_positive), canonical_hash(frozen_negative))

        manifest = make_continuous_smoke_manifest(dimension=2)
        positive_choice = FamilyChoice.create(
            manifest.covariate_choices[0].family_id,
            manifest.covariate_choices[0].version,
            1.0,
            {"dimension": 2, "nested": positive},
        )
        negative_choice = FamilyChoice.create(
            manifest.covariate_choices[0].family_id,
            manifest.covariate_choices[0].version,
            1.0,
            {"dimension": 2, "nested": negative},
        )
        self.assertEqual(
            dataclasses.replace(manifest, covariate_choices=(positive_choice,)).manifest_hash,
            dataclasses.replace(manifest, covariate_choices=(negative_choice,)).manifest_hash,
        )

        task = sample_task(
            manifest,
            global_seed=12,
            task_id=4,
            registry=build_builtin_registry(),
            source_id="causalpy@test-commit",
        )
        spec = task.to_spec()
        propensity_parameters = spec.propensity.parameter_dict()
        positive_propensity = FamilyRef.create(
            role=spec.propensity.role,
            family_id=spec.propensity.family_id,
            version=spec.propensity.version,
            semantic_class_id=spec.propensity.semantic_class_id,
            parameters={**propensity_parameters, "intercept": 0.0},
        )
        negative_propensity = FamilyRef.create(
            role=spec.propensity.role,
            family_id=spec.propensity.family_id,
            version=spec.propensity.version,
            semantic_class_id=spec.propensity.semantic_class_id,
            parameters={**propensity_parameters, "intercept": np.float64(-0.0)},
        )
        self.assertEqual(
            dataclasses.replace(spec, propensity=positive_propensity).task_spec_hash,
            dataclasses.replace(spec, propensity=negative_propensity).task_spec_hash,
        )

    def test_registry_is_explicit_and_does_not_change_manifest(self):
        manifest = make_continuous_smoke_manifest(dimension=2)
        before = manifest.manifest_hash
        first = build_builtin_registry()
        second = build_builtin_registry()

        self.assertIsNot(first, second)
        first.register(_DummyFactory())
        self.assertEqual(before, manifest.manifest_hash)
        with self.assertRaises(ValueError):
            first.register(_DummyFactory())
        with self.assertRaises(KeyError):
            first.resolve(
                ComponentRole.COVARIATE,
                "missing.family",
                "1.0.0",
            )


if __name__ == "__main__":
    unittest.main()
