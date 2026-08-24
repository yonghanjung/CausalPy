import dataclasses
import unittest

import numpy as np

from backdoor_scm import (
    build_builtin_registry,
    make_continuous_smoke_manifest,
    sample_task,
)


class P1P3GoldenCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = make_continuous_smoke_manifest(dimension=3)
        self.task = sample_task(
            self.manifest,
            global_seed=123,
            task_id=17,
            registry=build_builtin_registry(),
            source_id="causalpy@test-commit",
        )

    def test_manifest_and_task_sampling_identity_are_frozen(self) -> None:
        self.assertEqual(
            self.manifest.manifest_hash,
            "3fac9afbae781f1a68ba2817f0ab472dd214fff3b8ed0b11d73214b245d2c762",
        )
        self.assertEqual(
            self.task.to_spec().sampling_identity,
            "92f8678de11bae44e275f4f42dc5ceddfb83dc4c527e3a3d6bb3ba3d211de627",
        )

        normalized = dataclasses.replace(
            self.task.to_spec(),
            numpy_runtime_version="2.4.6",
        )
        self.assertEqual(
            normalized.task_spec_hash,
            "4283abf3469dc349313242c0633d6bb4cfcc07dbcecc116e814da17a75469e31",
        )

    def test_rows_and_truth_are_byte_for_byte_compatible(self) -> None:
        rows = self.task.sample_rows(4, row_seed=501)
        np.testing.assert_array_equal(
            rows.x,
            np.asarray(
                [
                    [-0.8773411111323608, 0.7474938926685939, -0.4230968655440296],
                    [-0.8780626314414774, 0.8373570206133486, -0.3184553608338138],
                    [0.7863133074466474, 0.10078798488472895, -2.1584856474642615],
                    [-0.9822476782124661, -2.090431038000231, 2.015269761843746],
                ]
            ),
        )
        np.testing.assert_array_equal(rows.a, np.asarray([0, 1, 0, 0], dtype=np.int8))
        np.testing.assert_array_equal(
            rows.y,
            np.asarray(
                [
                    -2.0738128952444237,
                    0.8625117935213454,
                    -0.654686356834526,
                    1.2670185756086554,
                ]
            ),
        )

        truth = self.task.truth(
            np.asarray([[0.0, 1.0, -1.0], [2.0, 0.5, 0.25]])
        )
        np.testing.assert_array_equal(
            truth.propensity,
            np.asarray([0.48681150886477415, 0.34349677634521747]),
        )
        np.testing.assert_array_equal(
            truth.mu0,
            np.asarray([-1.500061504288074, -0.5151582538361595]),
        )
        np.testing.assert_array_equal(
            truth.mu1,
            np.asarray([0.9941841791446363, -2.660768890770677]),
        )
        np.testing.assert_array_equal(
            truth.tau,
            np.asarray([2.4942456834327102, -2.145610636934517]),
        )


if __name__ == "__main__":
    unittest.main()
