"""Tests for P3-E12 molecule-style Count consistency."""

import unittest

import torch

from chemvector.training.atoms.train_g3_p3e12_style_consistent_count import (
    style_consistent_count_loss,
)


class G3P3E12StyleConsistentCountTests(unittest.TestCase):
    def test_identical_variant_logits_have_zero_consistency(self) -> None:
        logits = torch.randn(1, 129).repeat(4, 1)
        losses = style_consistent_count_loss(logits, torch.tensor([3, 3, 3, 3]))
        self.assertAlmostEqual(float(losses["style_consistency"]), 0.0, places=7)

    def test_inconsistent_group_labels_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "inconsistent Count"):
            style_consistent_count_loss(
                torch.zeros(4, 129), torch.tensor([3, 3, 3, 4])
            )


if __name__ == "__main__":
    unittest.main()
