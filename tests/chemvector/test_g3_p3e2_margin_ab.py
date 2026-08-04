"""Tests for P3-E2 coverage-calibrated structured margins."""

import unittest

import torch

from chemvector.training.atoms.train_g3_p3e2_margin_ab import (
    ARM_CONFIGS,
    latent_structured_loss,
    structured_margin,
)


class G3P3E2MarginABTests(unittest.TestCase):
    def test_coverage_margin_scales_by_missing_gt(self) -> None:
        self.assertEqual(structured_margin("fixed_whole_image", 20, 1), 1.0)
        self.assertEqual(structured_margin("gt_coverage_deficiency", 20, 1), 0.05)
        self.assertEqual(structured_margin("gt_coverage_deficiency", 20, 4), 0.2)

    def test_coverage_loss_is_less_strict_for_one_missing_gt(self) -> None:
        output = {
            "existence_logits": torch.tensor([[5.0, 4.0, 3.0]], requires_grad=True),
            "pointer_logits": torch.tensor([[ [
                8.0, 7.0, 0.0,
            ], [
                8.0, 7.0, 0.0,
            ], [
                0.0, 0.0, 8.0,
            ] ]], requires_grad=True),
        }
        candidates = torch.tensor([[[0.1, 0.1], [0.102, 0.1], [0.2, 0.2]]])
        targets = torch.tensor([[[0.1, 0.1], [0.2, 0.2]]])
        mask = torch.tensor([[True, True]])
        fixed = latent_structured_loss(
            output, candidates, targets, mask, margin_mode="fixed_whole_image"
        )
        coverage = latent_structured_loss(
            output,
            candidates,
            targets,
            mask,
            margin_mode="gt_coverage_deficiency",
        )
        self.assertGreater(
            float(fixed["loss"].detach()), float(coverage["loss"].detach())
        )
        self.assertEqual(float(coverage["margin_mean"]), 0.5)

    def test_invalid_margin_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            structured_margin("gt_coverage_deficiency", 4, 0)
        with self.assertRaises(ValueError):
            structured_margin("unknown", 4, 1)

    def test_fullset_arm_preserves_exposure_with_fewer_updates(self) -> None:
        config = ARM_CONFIGS["capacity-coverage-fullset"]
        self.assertEqual(config["expected_exposure"], 64 * 30)
        self.assertEqual(config["expected_steps"], 30)
        self.assertEqual(config["update_mode"], "fullset_gradient_accumulation")
        mixed = ARM_CONFIGS["mixed-coverage-fullset"]
        self.assertEqual(mixed["expected_exposure"], 256 * 30)
        self.assertEqual(mixed["expected_steps"], 30)


if __name__ == "__main__":
    unittest.main()
