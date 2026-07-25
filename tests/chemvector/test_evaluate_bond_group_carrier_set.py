from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch

from Data.sdf.visual_svg_v0_2 import parse_visual_svg_v0_2
from chemvector.bond_group_carrier_geometry import encode_bond_group_carrier
from chemvector.bond_group_carrier_set import TEXT_BOX
from chemvector.evaluate_bond_group_carrier_set import (
    carrier_rows_to_bond_group_rows,
)
from chemvector.evaluate_bond_group_set import (
    evaluate_decoded_sample,
    summarize_prediction_rows,
)
from chemvector.evaluate_visual_svg import render_bond_mask
from Data.sdf.visual_svg_v0_2 import flatten_visual_svg_v0_2_to_v0_1


FIXTURE = (
    Path(__file__).parents[1]
    / "data_sdf/fixtures/visual_svg_v0_2/all_group_kinds.json"
)


def _perfect_carrier_rows(sequence: str) -> list[dict]:
    parsed = parse_visual_svg_v0_2(sequence)
    rows = []
    for query_index, group in enumerate(parsed.bond_groups):
        encoded = encode_bond_group_carrier(group)
        rows.append(
            {
                "query_index": query_index,
                "score": 1.0,
                "type": int(encoded.kind_id),
                "kind": group.kind,
                "member_count": int(encoded.member_count),
                "carrier_points": encoded.carrier_points,
                "positive_params": encoded.positive_params,
                "signed_params": encoded.signed_params,
                "group_stroke": encoded.group_stroke,
            }
        )
    for text in parsed.text_boxes:
        rows.append(
            {
                "query_index": len(rows),
                "score": 1.0,
                "type": TEXT_BOX,
                "kind": "text_box",
                "member_count": 0,
                "text_geometry": torch.tensor(
                    (
                        (text.x0 + text.x1) / (2.0 * 2047.0),
                        (text.y0 + text.y1) / (2.0 * 2047.0),
                        (text.x1 - text.x0) / 2047.0,
                        (text.y1 - text.y0) / 2047.0,
                    )
                ),
            }
        )
    return rows


class EvaluateBondGroupCarrierSetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sequence = FIXTURE.read_text(encoding="utf-8")

    def test_perfect_carriers_reuse_all_existing_metric_layers(self) -> None:
        expanded = carrier_rows_to_bond_group_rows(
            _perfect_carrier_rows(self.sequence),
            max_members=32,
        )
        expected = parse_visual_svg_v0_2(self.sequence)
        mask = render_bond_mask(flatten_visual_svg_v0_2_to_v0_1(expected))
        bond_layer = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")

        row = evaluate_decoded_sample(
            sample_key="carrier-perfect",
            decode_mode="count_head",
            threshold=None,
            decoded_rows=list(expanded.rows),
            expected_v0_2_sequence=self.sequence,
            bond_layer=bond_layer,
            line_tolerance=8.0,
            text_box_iou=0.5,
            max_members=32,
        )
        summary = summarize_prediction_rows([row])

        self.assertFalse(expanded.invalid_objects)
        self.assertTrue(row["group_set_exact"])
        self.assertTrue(row["flatten_exact"])
        self.assertEqual(summary["group_set_exact"], 1.0)
        self.assertEqual(summary["flattened_v0_1"]["classes"]["line"]["f1"], 1.0)
        self.assertAlmostEqual(summary["bond_iou"]["mean"], 1.0, places=7)

    def test_expansion_is_differentiable_before_protocol_quantization(self) -> None:
        row = _perfect_carrier_rows(self.sequence)[2]
        row["carrier_points"] = row["carrier_points"].clone().requires_grad_(True)
        row["positive_params"] = row["positive_params"].clone().requires_grad_(True)
        row["signed_params"] = row["signed_params"].clone().requires_grad_(True)
        expanded = carrier_rows_to_bond_group_rows([row], max_members=32)

        loss = expanded.rows[0]["member_geometry"].sum()
        loss.backward()

        self.assertIsNotNone(row["carrier_points"].grad)
        self.assertIsNotNone(row["positive_params"].grad)
        self.assertIsNone(row["signed_params"].grad)

    def test_incompatible_member_count_is_reported_not_erased(self) -> None:
        row = _perfect_carrier_rows(self.sequence)[0]
        row["member_count"] = 2

        expanded = carrier_rows_to_bond_group_rows([row], max_members=32)

        self.assertEqual(len(expanded.rows), 1)
        self.assertEqual(len(expanded.invalid_objects), 1)
        self.assertEqual(expanded.rows[0]["member_count"], 2)
        self.assertEqual(tuple(expanded.rows[0]["member_geometry"].shape), (0, 8))
        self.assertIn("incompatible", expanded.invalid_objects[0]["error"])


if __name__ == "__main__":
    unittest.main()
