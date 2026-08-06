"""Protocol tests for P3-E19 bucket-balanced Count continuation."""

import unittest

import torch

from chemvector.training.atoms.train_g3_p3e19_bucket_balanced_count import (
    EXPECTED_BUCKET_COUNTS,
    EXPOSURE,
    STEPS,
    bucket_weights,
    capability_gate,
    epoch_batches,
    evaluate_bucketed,
    weighted_count_loss,
)


class G3P3E19BucketBalancedCountTests(unittest.TestCase):
    def test_inverse_frequency_equalizes_bucket_contribution(self) -> None:
        records = []
        representatives = {
            "1-20": 10,
            "21-40": 30,
            "41-60": 50,
            "61-80": 70,
            "81-100": 90,
        }
        for name, count in EXPECTED_BUCKET_COUNTS.items():
            records.extend({"label": representatives[name]} for _ in range(count))
        weights, audit = bucket_weights(records)
        contributions = [
            EXPECTED_BUCKET_COUNTS[name] * weights[name] for name in weights
        ]
        self.assertAlmostEqual(max(contributions), min(contributions))
        self.assertAlmostEqual(audit["mean_sample_weight"], 1.0)

    def test_weighted_loss_applies_per_target_bucket(self) -> None:
        logits = torch.zeros(2, 129)
        target = torch.tensor([10, 70])
        weights = {
            "1-20": 1.0,
            "21-40": 1.0,
            "41-60": 1.0,
            "61-80": 2.0,
            "81-100": 1.0,
        }
        plain = torch.nn.functional.cross_entropy(logits, target)
        self.assertAlmostEqual(
            float(weighted_count_loss(logits, target, weights)),
            float(plain * 1.5),
            places=5,
        )

    def test_schedule_is_one_complete_unique_epoch(self) -> None:
        batches, audit = epoch_batches(EXPOSURE)
        self.assertEqual(len(batches), STEPS)
        self.assertEqual(audit["unique_images"], EXPOSURE)
        self.assertEqual(sum(map(len, batches)), EXPOSURE)

    def test_gate_protects_low_count_and_requires_high_count_gain(self) -> None:
        fresh = {
            "count_accuracy": 0.42,
            "count_mae": 1.70,
            "by_count": {
                "1-20": {"count_accuracy": 0.51},
                "21-40": {"count_accuracy": 0.48},
                "61-80": {"count_accuracy": 0.20},
            },
        }
        decision, checks = capability_gate(
            fresh, {"count_accuracy": 0.46, "count_mae": 1.1}
        )
        self.assertEqual(decision, "bucket_balanced_count_go")
        self.assertTrue(all(checks.values()))
        fresh["by_count"]["61-80"]["count_accuracy"] = 0.19
        decision, checks = capability_gate(
            fresh, {"count_accuracy": 0.46, "count_mae": 1.1}
        )
        self.assertEqual(decision, "bucket_balanced_count_no_go")
        self.assertFalse(checks["fresh_61_80_gain10pp"])

    def test_evaluation_rejects_invalid_batch_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "batch size must be positive"):
            evaluate_bucketed(None, [], device=torch.device("cpu"), batch_size=0)


if __name__ == "__main__":
    unittest.main()
