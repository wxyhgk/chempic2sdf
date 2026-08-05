"""Tests for P3-E16 Count continuation and bucket metrics."""

import unittest

from chemvector.training.atoms.train_g3_p3e16_style_count_epoch6 import (
    CONTINUATION_EXPOSURE,
    CONTINUATION_STEPS,
    FINAL_GLOBAL_STEP,
    FINAL_SAMPLES_SEEN,
    TRAIN_IMAGES,
    bucketed_count_metrics,
    continuation_batches,
)


class G3P3E16StyleCountEpoch6Tests(unittest.TestCase):
    def test_three_complete_unique_epochs(self) -> None:
        batches, audit = continuation_batches(TRAIN_IMAGES)
        self.assertEqual(len(batches), CONTINUATION_STEPS)
        self.assertEqual(sum(map(len, batches)), CONTINUATION_EXPOSURE)
        self.assertEqual(audit["epochs"], 3)
        self.assertEqual(FINAL_GLOBAL_STEP, 8_665)
        self.assertEqual(FINAL_SAMPLES_SEEN, 554_140)

    def test_bucket_metrics_use_true_count_and_variant(self) -> None:
        metrics = bucketed_count_metrics(
            [10, 21, 39, 61],
            [10, 20, 40, 60],
            [9, 20, 42, 60],
            ["coordgen", "original", "rdkit", "stress"],
        )
        self.assertEqual(metrics["count_exact"], 1)
        self.assertEqual(metrics["by_count"]["1-20"]["images"], 2)
        self.assertEqual(metrics["by_count"]["1-20"]["count_exact"], 1)
        self.assertEqual(metrics["by_count"]["21-40"]["images"], 1)
        self.assertEqual(metrics["by_variant"]["stress"]["count_mae"], 1.0)


if __name__ == "__main__":
    unittest.main()
