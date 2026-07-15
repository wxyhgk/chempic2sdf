"""Tests for OCR-only checkpoint interpolation."""

from __future__ import annotations

import unittest

import torch

from chemvector.interpolate_ocr import interpolate_state_dicts


class InterpolateOCRTests(unittest.TestCase):
    def test_only_ocr_tensors_are_interpolated(self) -> None:
        first = {
            "vision.weight": torch.tensor((1.0,)),
            "ocr_head.weight": torch.tensor((0.0, 2.0)),
        }
        second = {
            "vision.weight": torch.tensor((1.0,)),
            "ocr_head.weight": torch.tensor((2.0, 4.0)),
        }
        result = interpolate_state_dicts(first, second, 0.25)
        self.assertTrue(torch.equal(result["vision.weight"], torch.tensor((1.0,))))
        self.assertTrue(
            torch.equal(result["ocr_head.weight"], torch.tensor((0.5, 2.5)))
        )

    def test_non_ocr_change_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Non-OCR tensor changed"):
            interpolate_state_dicts(
                {"vision.weight": torch.tensor((1.0,))},
                {"vision.weight": torch.tensor((2.0,))},
                0.5,
            )


if __name__ == "__main__":
    unittest.main()
