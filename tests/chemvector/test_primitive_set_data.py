"""Tests for Visual SVG primitive-set target construction."""

from __future__ import annotations

import unittest

import torch
from PIL import Image

from chemvector.primitive_set_data import (
    NO_OBJECT_TYPE_ID,
    PRIMITIVE_SET_QUERY_COUNT,
    PRIMITIVE_TYPE_TO_ID,
    PrimitiveSetCollator,
    build_primitive_set_target,
)
from chemvector.primitive_set import LINE, primitive_set_loss


def primitive(
    kind: str,
    points: list[list[int]],
    *,
    stroke_width_q: int | None = None,
) -> dict:
    value = {"kind": kind, "quantized_points": points}
    if stroke_width_q is not None:
        value["stroke_width_q"] = stroke_width_q
    return value


class PrimitiveSetTargetTests(unittest.TestCase):
    def test_encodes_all_geometry_protocols_and_padding(self) -> None:
        payload = {
            "primitives": [
                primitive("line", [[0, 2047], [1024, 512]], stroke_width_q=21),
                primitive(
                    "triangle",
                    [[0, 0], [2047, 0], [1024, 2047]],
                    stroke_width_q=6,
                ),
                primitive(
                    "quad",
                    [[0, 0], [2047, 0], [2047, 2047], [0, 2047]],
                    stroke_width_q=4,
                ),
                primitive("text_box", [[100, 200], [300, 600]]),
            ]
        }

        target = build_primitive_set_target(payload)

        self.assertEqual(target["types"].shape, (PRIMITIVE_SET_QUERY_COUNT,))
        self.assertEqual(
            target["geometry"].shape,
            (PRIMITIVE_SET_QUERY_COUNT, 8),
        )
        self.assertEqual(
            target["types"][:5].tolist(),
            [
                PRIMITIVE_TYPE_TO_ID["line"],
                PRIMITIVE_TYPE_TO_ID["triangle"],
                PRIMITIVE_TYPE_TO_ID["quad"],
                PRIMITIVE_TYPE_TO_ID["text_box"],
                NO_OBJECT_TYPE_ID,
            ],
        )
        self.assertEqual(target["geometry_mask"].sum(dim=1)[:5].tolist(), [4, 6, 8, 4, 0])
        self.assertEqual(target["target_mask"].sum().item(), 4)
        self.assertEqual(
            target["has_stroke"][:5].tolist(),
            [True, True, True, False, False],
        )
        self.assertAlmostEqual(target["stroke"][0].item(), 21 / 63)
        torch.testing.assert_close(
            target["geometry"][3, :4],
            torch.tensor(
                [
                    200 / 2047,
                    400 / 2047,
                    200 / 2047,
                    400 / 2047,
                ]
            ),
        )
        self.assertTrue(
            torch.equal(
                target["geometry"][4:],
                torch.zeros_like(target["geometry"][4:]),
            )
        )

    def test_capacity_overflow_fails_instead_of_truncating(self) -> None:
        line = primitive("line", [[0, 0], [1, 1]], stroke_width_q=15)
        payload = {"primitives": [dict(line) for _ in range(PRIMITIVE_SET_QUERY_COUNT + 1)]}

        with self.assertRaisesRegex(ValueError, "exceeds frozen query capacity 288"):
            build_primitive_set_target(payload)

    def test_rejects_invalid_text_box_geometry(self) -> None:
        payload = {"primitives": [primitive("text_box", [[100, 100], [100, 200]])]}

        with self.assertRaisesRegex(ValueError, "positive box"):
            build_primitive_set_target(payload)


class PrimitiveSetCollatorTests(unittest.TestCase):
    def test_collates_targets_and_preserves_sample_sidecars(self) -> None:
        bond_layer = Image.new("L", (2, 2), 127)
        meta = {"split": "train"}
        sample = {
            "key": "primitive-sample",
            "image": Image.new("RGB", (2, 2), "black"),
            "bond_layer": bond_layer,
            "primitives": {
                "primitives": [
                    primitive("line", [[0, 0], [2047, 2047]], stroke_width_q=15)
                ]
            },
            "meta": meta,
        }

        batch = PrimitiveSetCollator(image_size=4)([sample])

        self.assertEqual(batch["pixel_values"].shape, (1, 3, 4, 4))
        torch.testing.assert_close(
            batch["pixel_values"],
            torch.full((1, 3, 4, 4), -1.0),
        )
        self.assertEqual(batch["types"].shape, (1, PRIMITIVE_SET_QUERY_COUNT))
        self.assertEqual(batch["geometry"].shape, (1, PRIMITIVE_SET_QUERY_COUNT, 8))
        self.assertEqual(batch["keys"], ["primitive-sample"])
        self.assertIs(batch["meta"][0], meta)
        self.assertIs(batch["bond_layers"][0], bond_layer)

    def test_collator_targets_feed_the_set_loss_without_key_translation(self) -> None:
        sample = {
            "key": "loss-sample",
            "image": Image.new("RGB", (4, 4), "white"),
            "bond_layer": Image.new("L", (4, 4), 0),
            "primitives": {
                "primitives": [
                    primitive("line", [[100, 200], [300, 400]], stroke_width_q=15)
                ]
            },
            "meta": {"split": "train"},
        }
        batch = PrimitiveSetCollator(image_size=4)([sample])
        type_logits = torch.full((1, PRIMITIVE_SET_QUERY_COUNT, 5), -4.0)
        type_logits[..., 0] = 4.0
        type_logits[0, 0, 0] = -4.0
        type_logits[0, 0, LINE] = 4.0
        output = {
            "type_logits": type_logits,
            "geometry": batch["geometry"].clone(),
            "text_geometry": batch["geometry"].clone(),
            "stroke": batch["stroke"].clone(),
        }

        losses = primitive_set_loss(output, batch)

        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertEqual(len(losses["assignments"][0][0]), 1)


if __name__ == "__main__":
    unittest.main()
