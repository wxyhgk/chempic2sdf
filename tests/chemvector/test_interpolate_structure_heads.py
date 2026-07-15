"""Tests for restricted structure-head checkpoint interpolation."""

from __future__ import annotations

import unittest

import torch

from chemvector.interpolate_structure_heads import interpolate_state_dicts


class InterpolateStructureHeadsTests(unittest.TestCase):
    def test_only_changed_structure_heads_are_interpolated(self) -> None:
        first = {
            "vision.weight": torch.tensor((1.0,)),
            "vertex_head.weight": torch.tensor((0.0, 2.0)),
            "relation_head.weight": torch.tensor((2.0, 4.0)),
            "stroke_head.weight": torch.tensor((4.0, 6.0)),
        }
        second = {
            "vision.weight": torch.tensor((1.0,)),
            "vertex_head.weight": torch.tensor((2.0, 4.0)),
            "relation_head.weight": torch.tensor((4.0, 6.0)),
            "stroke_head.weight": torch.tensor((6.0, 8.0)),
        }
        result, changed = interpolate_state_dicts(first, second, 0.25)
        self.assertTrue(torch.equal(result["vision.weight"], torch.tensor((1.0,))))
        self.assertTrue(
            torch.equal(result["vertex_head.weight"], torch.tensor((0.5, 2.5)))
        )
        self.assertEqual(
            changed,
            [
                "vertex_head.weight",
                "relation_head.weight",
                "stroke_head.weight",
            ],
        )

    def test_non_structure_head_change_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Non-structure-head tensor changed"):
            interpolate_state_dicts(
                {"vision.weight": torch.tensor((1.0,))},
                {"vision.weight": torch.tensor((2.0,))},
                0.5,
            )

    def test_out_of_range_alpha_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "alpha must be"):
            interpolate_state_dicts({}, {}, 1.1)


if __name__ == "__main__":
    unittest.main()
