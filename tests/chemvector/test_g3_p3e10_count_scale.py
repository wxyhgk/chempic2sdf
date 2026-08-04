"""Tests for P3-E10 Count scale/capacity protocol constants."""

import unittest

from chemvector.training.atoms.train_g3_p3e10_count_scale_ab import (
    ARM_CONFIGS,
    EXPECTED_EXPOSURE,
    TRAIN_IMAGES,
)


class G3P3E10CountScaleTests(unittest.TestCase):
    def test_arms_change_only_hidden_capacity(self) -> None:
        self.assertEqual(TRAIN_IMAGES, 3316)
        self.assertEqual(EXPECTED_EXPOSURE, 3316 * 30)
        self.assertEqual(ARM_CONFIGS["expanded64"]["hidden_dim"], 64)
        self.assertEqual(ARM_CONFIGS["expanded32"]["hidden_dim"], 32)


if __name__ == "__main__":
    unittest.main()
