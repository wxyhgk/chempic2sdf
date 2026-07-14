"""Tests for the generated chemical-label OCR curriculum."""

from __future__ import annotations

import random
import unittest

import torch

from chemvector.ocr import OCRTextCodec
from chemvector.synthetic_ocr import (
    SyntheticOCRConfig,
    SyntheticOCRDataset,
    collate_synthetic_ocr,
    render_synthetic_ocr_sample,
    sample_label,
)


class SyntheticOCRTests(unittest.TestCase):
    def test_render_is_deterministic_for_seed(self) -> None:
        config = SyntheticOCRConfig()
        first = render_synthetic_ocr_sample(
            "R1",
            rng=random.Random(7),
            config=config,
        )
        second = render_synthetic_ocr_sample(
            "R1",
            rng=random.Random(7),
            config=config,
        )
        self.assertEqual(first.tobytes(), second.tobytes())
        self.assertLess(first.convert("L").getextrema()[0], 255)

    def test_all_label_categories_can_be_selected(self) -> None:
        categories = {
            sample_label(random.Random(seed), SyntheticOCRConfig())[1]
            for seed in range(200)
        }
        self.assertEqual(categories, {"blank", "standard", "alias", "random"})

    def test_collator_builds_one_center_anchor_per_crop(self) -> None:
        dataset = SyntheticOCRDataset(3, seed=11)
        samples = [dataset[index] for index in range(3)]
        batch = collate_synthetic_ocr(samples, OCRTextCodec())
        self.assertEqual(batch["pixel_values"].shape, (3, 3, 128, 128))
        self.assertTrue(
            torch.equal(batch["ocr_anchor_mask"], torch.ones(3, 1, dtype=torch.bool))
        )
        self.assertTrue(
            torch.equal(batch["ocr_anchor_points"], torch.full((3, 1, 2), 0.5))
        )


if __name__ == "__main__":
    unittest.main()
