"""Tests for the frozen P3-E7 G3 Atom base/expert router."""

import unittest

import torch

from chemvector.components.atoms.objects.g3_base_expert_router import (
    G3_BASE_EXPERT_THRESHOLD,
    decode_g3_base_expert_joint,
    g3_should_route_expert,
)


class G3BaseExpertRouterTests(unittest.TestCase):
    def test_frozen_threshold_is_inclusive(self) -> None:
        self.assertTrue(g3_should_route_expert(G3_BASE_EXPERT_THRESHOLD))
        self.assertTrue(g3_should_route_expert(G3_BASE_EXPERT_THRESHOLD - 0.1))
        self.assertFalse(g3_should_route_expert(G3_BASE_EXPERT_THRESHOLD + 0.1))

    def test_explicit_zero_count_keeps_base(self) -> None:
        output = {
            "existence_logits": torch.zeros(1, 2),
            "pointer_logits": torch.zeros(1, 2, 2),
            "count_logits": torch.zeros(1, 3),
        }
        decoded = decode_g3_base_expert_joint(
            output,
            output,
            torch.zeros(1, 2, 2),
            counts=torch.tensor([0]),
        )
        self.assertEqual(decoded[0]["source"], "base")
        self.assertEqual(decoded[0]["count"], 0)
        self.assertEqual(tuple(decoded[0]["anchors"].shape), (0, 2))

    def test_missing_count_logits_are_rejected(self) -> None:
        output = {
            "existence_logits": torch.zeros(1, 2),
            "pointer_logits": torch.zeros(1, 2, 2),
        }
        with self.assertRaisesRegex(ValueError, "lacks valid Count"):
            decode_g3_base_expert_joint(output, output, torch.zeros(1, 2, 2))


if __name__ == "__main__":
    unittest.main()
