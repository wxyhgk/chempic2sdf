"""Tests for the fixed P3-E13 cardinality-curve protocol."""

import unittest

from chemvector.training.atoms.train_g3_p3e13_cardinality_curve import (
    COARSE_SHA256,
    EXPECTED_EXPOSURE,
    EXPECTED_STEPS,
    MODEL_TYPE,
    TRAIN_IMAGES,
)


class G3P3E13CardinalityCurveTests(unittest.TestCase):
    def test_fixed_protocol(self) -> None:
        self.assertEqual(TRAIN_IMAGES, 3316)
        self.assertEqual(EXPECTED_EXPOSURE, 3316 * 30)
        self.assertEqual(EXPECTED_STEPS, 1560)
        self.assertIn("p3e13-cardinality-curve", MODEL_TYPE)
        self.assertEqual(len(COARSE_SHA256), 64)


if __name__ == "__main__":
    unittest.main()
