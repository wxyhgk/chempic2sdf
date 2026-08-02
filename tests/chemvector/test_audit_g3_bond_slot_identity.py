import unittest

import torch

from chemvector.audit_g3_bond_slot_identity import (
    _aligned_endpoint_errors_pixels,
    candidate_identity_retention,
    endpoint_error_summary,
    summarize_routed_predictions,
    validate_p0d_checkpoint,
)
from chemvector.g3_structural_data import OVERFIT32_MANIFEST_SHA256
from chemvector.train_g3_hierarchical_graph import DEFAULT_VISION_CHECKPOINT_SHA256


class G3BondSlotIdentityAuditTests(unittest.TestCase):
    def test_unordered_router_error_uses_worst_aligned_endpoint(self) -> None:
        predicted = torch.tensor([[[0.80, 0.50], [0.19, 0.50]]])
        expected = torch.tensor([[[0.20, 0.50], [0.80, 0.50]]])

        errors = _aligned_endpoint_errors_pixels(
            predicted, expected, image_size=100.0
        )

        self.assertTrue(torch.allclose(errors, torch.tensor([1.0]), atol=1e-5))

    def test_candidate_retention_is_targetwise_and_reports_changed_images(self) -> None:
        previous = [[(0, 1, 2), (1, 3, 4)], [(2, 5, 6)]]
        current = [[(0, 1, 2), (1, 3, 5)], [(2, 5, 7)]]

        metrics = candidate_identity_retention(previous, current)

        self.assertEqual(metrics["comparable_targets"], 3)
        self.assertEqual(metrics["retained_targets"], 1)
        self.assertEqual(metrics["changed_targets"], 2)
        self.assertEqual(metrics["changed_images"], 2)
        self.assertAlmostEqual(metrics["retention"], 1 / 3)
        self.assertAlmostEqual(metrics["churn"], 2 / 3)

    def test_micro_metrics_separate_pointer_and_type(self) -> None:
        target_mask = torch.tensor([[True, True, False]])
        target_endpoints = torch.tensor(
            [[
                [[0.10, 0.10], [0.30, 0.10]],
                [[0.40, 0.20], [0.60, 0.20]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]]
        )
        pointer_logits = torch.full((1, 3, 2, 4), -10.0)
        pointer_logits[0, 0, 0, 1] = 10.0
        pointer_logits[0, 0, 1, 0] = 10.0
        pointer_logits[0, 1, 0, 1] = 10.0
        pointer_logits[0, 1, 1, 3] = 10.0
        type_logits = torch.zeros(1, 3, 4)
        type_logits[0, 0, 2] = 10.0
        type_logits[0, 1, 0] = 10.0

        metrics, identities, errors = summarize_routed_predictions(
            router_endpoints=target_endpoints.clone(),
            target_endpoints=target_endpoints,
            target_mask=target_mask,
            candidate_ids=torch.tensor([[[0, 1, 2], [1, 3, 4], [-1, -1, -1]]]),
            pointer_logits=pointer_logits,
            type_logits=type_logits,
            target_atom_refs=torch.tensor([[[0, 1], [2, 3], [0, 0]]]),
            target_types=torch.tensor([[2, 1, 0]]),
        )

        self.assertEqual(identities, [[(0, 1, 2), (1, 3, 4)]])
        self.assertEqual(errors, [0.0, 0.0])
        self.assertEqual(metrics["pointer_endpoint_correct"], 3)
        self.assertEqual(metrics["pointer_both_endpoint_correct"], 1)
        self.assertEqual(metrics["type_correct"], 1)
        self.assertEqual(metrics["type_given_pointer_correct"], 1)
        self.assertAlmostEqual(metrics["pointer_endpoint_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["pointer_both_endpoint_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["type_accuracy"], 0.5)
        self.assertAlmostEqual(metrics["type_given_pointer_accuracy"], 1.0)

    def test_endpoint_summary_reports_fixed_thresholds(self) -> None:
        summary = endpoint_error_summary([2.0, 6.0, 12.0, 20.0])

        self.assertEqual(summary["within_4px_objects"], 1)
        self.assertEqual(summary["within_8px_objects"], 2)
        self.assertEqual(summary["within_16px_objects"], 3)
        self.assertEqual(summary["within_16px_coverage"], 0.75)

    def test_checkpoint_scope_rejects_non_p0d_or_non_overfit32(self) -> None:
        payload = {
            "global_step": 400,
            "structural_scope": {
                "gt_atom_slot_oracle": True,
                "gt_bond_slot_identity_oracle": True,
            },
            "sample_manifest": {"sha256": OVERFIT32_MANIFEST_SHA256},
            "source_vision_checkpoint_sha256": DEFAULT_VISION_CHECKPOINT_SHA256,
        }
        validate_p0d_checkpoint(payload, step=400)

        invalid = {**payload, "sample_manifest": {"sha256": "0" * 64}}
        with self.assertRaisesRegex(ValueError, "overfit32"):
            validate_p0d_checkpoint(invalid, step=400)

        invalid = {**payload, "structural_scope": {}}
        with self.assertRaisesRegex(ValueError, "only P0-D"):
            validate_p0d_checkpoint(invalid, step=400)


if __name__ == "__main__":
    unittest.main()
