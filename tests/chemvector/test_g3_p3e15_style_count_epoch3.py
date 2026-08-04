"""Tests for fixed P3-E15 full-epoch Count continuation."""

import unittest

from chemvector.training.atoms.train_g3_p3e15_style_count_epoch3 import (
    CONTINUATION_EXPOSURE,
    CONTINUATION_STEPS,
    FINAL_GLOBAL_STEP,
    FINAL_SAMPLES_SEEN,
    TRAIN_IMAGES,
    continuation_batches,
)


class G3P3E15StyleCountEpoch3Tests(unittest.TestCase):
    def test_two_complete_unique_epochs(self) -> None:
        batches, audit = continuation_batches(TRAIN_IMAGES)
        self.assertEqual(len(batches), CONTINUATION_STEPS)
        self.assertEqual(sum(map(len, batches)), CONTINUATION_EXPOSURE)
        self.assertEqual(audit["epochs"], 2)
        self.assertEqual(audit["residual_batch_size"], 52)
        self.assertEqual(FINAL_GLOBAL_STEP, 4_402)
        self.assertEqual(FINAL_SAMPLES_SEEN, 281_344)


if __name__ == "__main__":
    unittest.main()
