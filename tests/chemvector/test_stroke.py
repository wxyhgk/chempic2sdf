"""Tests for learned visual stroke pair scoring."""

from __future__ import annotations

import unittest

from PIL import Image
import torch

from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.stroke import STROKE_CLASS_TO_ID, StrokePairCollator, all_pair_indices


class StrokePairDataTests(unittest.TestCase):
    def test_all_pairs_do_not_apply_a_distance_cutoff(self) -> None:
        self.assertEqual(
            all_pair_indices(4),
            [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
        )

    def test_collator_contains_positive_and_two_hop_negative_pairs(self) -> None:
        sample = {
            "key": "sample",
            "image": Image.new("RGB", (64, 64), "white"),
            "render": {
                "width": 64,
                "height": 64,
                "atoms": [
                    {"atom_id": 0, "image_x": 8, "image_y": 32},
                    {"atom_id": 1, "image_x": 32, "image_y": 32},
                    {"atom_id": 2, "image_x": 56, "image_y": 32},
                ],
                "bonds": [
                    {"bond_id": 0, "display_type": "SINGLE"},
                    {"bond_id": 1, "display_type": "DOUBLE"},
                ],
            },
            "chemmol": {
                "bonds": [
                    {"id": 0, "begin_atom": 0, "end_atom": 1},
                    {"id": 1, "begin_atom": 1, "end_atom": 2},
                ]
            },
        }
        batch = StrokePairCollator(
            image_size=64,
            negative_ratio=2,
            anchor_jitter_pixels=0,
        )([sample])
        pairs = batch["pair_indices"][0][batch["pair_mask"][0]].tolist()
        labels = batch["pair_labels"][0][batch["pair_mask"][0]].tolist()
        mapping = {tuple(pair): label for pair, label in zip(pairs, labels)}
        self.assertEqual(mapping[(0, 1)], STROKE_CLASS_TO_ID["SINGLE"])
        self.assertEqual(mapping[(1, 2)], STROKE_CLASS_TO_ID["DOUBLE"])
        self.assertEqual(mapping[(0, 2)], STROKE_CLASS_TO_ID["NONE"])


class StrokeScorerTests(unittest.TestCase):
    def test_forward_scores_only_unmasked_pairs(self) -> None:
        config = ChemVectorConfig(
            vocab_size=64,
            d_model=64,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=128,
            max_target_length=32,
            vision_dims=(16, 32, 48, 64),
            vision_depths=(1, 1, 1, 1),
            stroke_enabled=True,
            stroke_feature_dim=16,
            stroke_samples=12,
            stroke_band=3,
        )
        model = ChemVectorModel(config)
        pixels = torch.randn(2, 3, 128, 128)
        anchors = torch.tensor(
            [
                [[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]],
                [[0.2, 0.3], [0.8, 0.7], [0.0, 0.0]],
            ]
        )
        pairs = torch.tensor(
            [
                [[0, 1], [1, 2], [0, 2]],
                [[0, 1], [0, 0], [0, 0]],
            ]
        )
        mask = torch.tensor([[True, True, True], [True, False, False]])
        features = model.vision_encoder.forward_stroke_resolution(pixels)

        logits = model.stroke_head(features, anchors, pairs, mask, pixels.shape[-2:])

        self.assertEqual(logits.shape, (4, config.stroke_classes))
        logits.sum().backward()
        self.assertIsNotNone(model.stroke_head.line_encoder[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
