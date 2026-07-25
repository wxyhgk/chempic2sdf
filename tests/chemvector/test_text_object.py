"""Tests for independent text-object center detection."""

import unittest

import torch

from chemvector.text_object import (
    TextObjectBBoxHead,
    TextObjectCenterHead,
    decode_text_objects,
    text_object_loss,
)


class TextObjectTests(unittest.TestCase):
    def test_bbox_head_keeps_anchor_and_offsets_crop_center(self) -> None:
        head = TextObjectBBoxHead(4, hidden_channels=8)
        with torch.no_grad():
            head.heatmap.weight.zero_()
            head.heatmap.bias.fill_(10.0)
            head.offset.weight.zero_()
            head.offset.bias.zero_()
            head.extent.weight.zero_()
            head.extent.bias.zero_()
            head.bbox_offset.bias.copy_(torch.tensor((0.5, -0.5)))
        decoded = decode_text_objects(head(torch.zeros((1, 4, 1, 1))), threshold=0.1)
        item = decoded[0][0]
        self.assertAlmostEqual(item["x"], 0.5, places=5)
        self.assertGreater(item["bbox_x"], item["x"])
        self.assertLess(item["bbox_y"], item["y"])

    def test_bbox_offset_loss_uses_relative_extent(self) -> None:
        output = {
            "heatmap_logits": torch.zeros((1, 1, 2, 2)),
            "offset_logits": torch.zeros((1, 2, 2, 2)),
            "extent_logits": torch.zeros((1, 2, 2, 2)),
            "bbox_offset_logits": torch.zeros((1, 2, 2, 2)),
        }
        losses = text_object_loss(
            output,
            torch.tensor([[[0.25, 0.25]]]),
            torch.tensor([[True]]),
            torch.tensor([[[0.1, 0.1]]]),
            torch.tensor([[[0.30, 0.25]]]),
        )
        self.assertGreater(float(losses["bbox_offset_loss"]), 0.0)

    def test_loss_and_decode_are_finite(self) -> None:
        head = TextObjectCenterHead(8, hidden_channels=16)
        output = head(torch.randn(2, 8, 16, 16))
        centers = torch.tensor([[[0.25, 0.25]], [[0.75, 0.75]]])
        mask = torch.ones((2, 1), dtype=torch.bool)
        extents = torch.full((2, 1, 2), 0.02)

        losses = text_object_loss(output, centers, mask, extents)
        losses["loss"].backward()
        decoded = decode_text_objects(output, threshold=0.0, maximum_objects=2)

        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertEqual([len(items) for items in decoded], [2, 2])

    def test_empty_text_batch_has_finite_loss(self) -> None:
        head = TextObjectCenterHead(4, hidden_channels=8)
        output = head(torch.randn(1, 4, 8, 8))
        losses = text_object_loss(
            output,
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, dtype=torch.bool),
            torch.zeros(1, 1, 2),
        )
        self.assertTrue(torch.isfinite(losses["loss"]))


if __name__ == "__main__":
    unittest.main()
