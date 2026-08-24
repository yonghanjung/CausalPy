import random
import unittest

import numpy as np

from backdoor_scm.rng import keyed_rng


class KeyedRngTests(unittest.TestCase):
    def test_same_key_reproduces_and_namespaces_separate(self):
        first = keyed_rng(123, "task", 7, "x").normal(size=16)
        second = keyed_rng(123, "task", 7, "x").normal(size=16)
        other = keyed_rng(123, "task", 7, "a").normal(size=16)

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))

    def test_keyed_rng_does_not_mutate_global_rng_states(self):
        np.random.seed(314)
        random.seed(271)
        numpy_before = np.random.get_state()
        python_before = random.getstate()

        keyed_rng("isolated", 1).normal(size=32)

        numpy_after = np.random.get_state()
        python_after = random.getstate()
        self.assertEqual(numpy_before[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
        self.assertEqual(numpy_before[2:], numpy_after[2:])
        self.assertEqual(python_before, python_after)


if __name__ == "__main__":
    unittest.main()
