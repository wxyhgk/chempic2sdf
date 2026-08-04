"""Tests for matched-exposure P3-E14 Count scale training."""

import unittest

from chemvector.training.atoms.train_g3_p3e14_style_count_scale import (
    EXPECTED_EXPOSURE,
    EXPECTED_STEPS,
    REPEATED_EXPOSURE,
    TRAIN_IMAGES,
    matched_exposure_batches,
)


class G3P3E14StyleCountScaleTests(unittest.TestCase):
    def test_schedule_matches_p3e10_compute(self) -> None:
        batches, audit = matched_exposure_batches(TRAIN_IMAGES)
        self.assertEqual(len(batches), EXPECTED_STEPS)
        self.assertEqual(sum(map(len, batches)), EXPECTED_EXPOSURE)
        self.assertEqual(audit["unique_images_seen"], TRAIN_IMAGES)
        self.assertEqual(audit["repeated_exposure"], REPEATED_EXPOSURE)
        self.assertEqual(REPEATED_EXPOSURE, 8_548)


if __name__ == "__main__":
    unittest.main()
