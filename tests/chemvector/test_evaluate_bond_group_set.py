from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from Data.sdf.visual_svg_v0_2 import (
    LineMemberV02,
    QuadMemberV02,
    TriangleMemberV02,
    flatten_visual_svg_v0_2_to_v0_1,
    parse_visual_svg_v0_2,
)
from chemvector.bond_group_set import BondGroupSetConfig, BondGroupSetPredictor
from chemvector.evaluate_bond_group_set import (
    decoded_rows_to_visual_svg_v0_2,
    evaluate_decoded_sample,
    load_bond_group_set_model,
    summarize_prediction_rows,
)
from chemvector.evaluate_visual_svg import render_bond_mask
from chemvector.model import ChemVectorConfig
from chemvector.train_bond_group_set import (
    CHECKPOINT_SCHEMA_VERSION,
    SEQUENCE_SCHEMA_VERSION,
)


FIXTURE = (
    Path(__file__).parents[1]
    / "data_sdf/fixtures/visual_svg_v0_2/all_group_kinds.json"
)


def _normalized_member(member) -> tuple[list[float], float]:
    if isinstance(member, LineMemberV02):
        values = (member.x0, member.y0, member.x1, member.y1)
    elif isinstance(member, TriangleMemberV02):
        values = (
            member.x0,
            member.y0,
            member.x1,
            member.y1,
            member.x2,
            member.y2,
        )
    elif isinstance(member, QuadMemberV02):
        values = (
            member.x0,
            member.y0,
            member.x1,
            member.y1,
            member.x2,
            member.y2,
            member.x3,
            member.y3,
        )
    else:
        raise AssertionError(type(member))
    geometry = [value / 2047.0 for value in values]
    geometry.extend([0.0] * (8 - len(geometry)))
    return geometry, member.stroke_width / 63.0


def _perfect_rows(sequence: str) -> list[dict]:
    parsed = parse_visual_svg_v0_2(sequence)
    rows = []
    for query_index, group in enumerate(parsed.bond_groups):
        values = [_normalized_member(member) for member in group.members]
        rows.append(
            {
                "query_index": query_index,
                "score": 1.0,
                "kind": group.kind,
                "member_count": len(group.members),
                "member_geometry": torch.tensor([value[0] for value in values]),
                "member_stroke": torch.tensor([value[1] for value in values]),
            }
        )
    for box in parsed.text_boxes:
        rows.append(
            {
                "query_index": len(rows),
                "score": 1.0,
                "kind": "text_box",
                "member_count": 0,
                "text_geometry": torch.tensor(
                    (
                        (box.x0 + box.x1) / (2.0 * 2047.0),
                        (box.y0 + box.y1) / (2.0 * 2047.0),
                        (box.x1 - box.x0) / 2047.0,
                        (box.y1 - box.y0) / 2047.0,
                    )
                ),
            }
        )
    return rows


class BondGroupSetEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sequence = FIXTURE.read_text(encoding="utf-8")

    def test_perfect_rows_roundtrip_strict_v0_2(self) -> None:
        converted = decoded_rows_to_visual_svg_v0_2(
            _perfect_rows(self.sequence),
            max_members=4,
        )

        self.assertTrue(converted.valid)
        self.assertEqual(converted.sequence, self.sequence)
        self.assertEqual(converted.parsed, parse_visual_svg_v0_2(self.sequence))
        self.assertEqual(len(converted.invalid_objects), 0)

    def test_cardinality_and_overflow_are_invalid_without_count_erasure(self) -> None:
        rows = [
            {
                "query_index": 0,
                "score": 1.0,
                "kind": "parallel_2",
                "member_count": 1,
                "member_geometry": torch.zeros(1, 8),
                "member_stroke": torch.ones(1),
            },
            {
                "query_index": 1,
                "score": 1.0,
                "kind": "hashed_wedge",
                "member_count": 5,
                "member_geometry": torch.zeros(2, 8),
                "member_stroke": torch.ones(2),
            },
            {
                "query_index": 2,
                "score": 1.0,
                "kind": "text_box",
                "member_count": 1,
                "text_geometry": torch.tensor((0.5, 0.5, 0.1, 0.1)),
            },
        ]

        converted = decoded_rows_to_visual_svg_v0_2(rows, max_members=4)

        self.assertEqual(converted.decoded_counts["parallel_2"], 1)
        self.assertEqual(converted.decoded_counts["hashed_wedge"], 1)
        self.assertEqual(converted.decoded_counts["text_box"], 1)
        self.assertEqual(len(converted.invalid_objects), 3)
        self.assertEqual(len(converted.overflow_objects), 1)
        self.assertFalse(converted.valid)

    def test_perfect_sample_reports_all_three_metric_layers(self) -> None:
        expected = parse_visual_svg_v0_2(self.sequence)
        expected_flat = flatten_visual_svg_v0_2_to_v0_1(expected)
        mask = render_bond_mask(expected_flat)
        bond_layer = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")

        row = evaluate_decoded_sample(
            sample_key="sample",
            decode_mode="count_head",
            threshold=None,
            decoded_rows=_perfect_rows(self.sequence),
            expected_v0_2_sequence=self.sequence,
            bond_layer=bond_layer,
            line_tolerance=8.0,
            text_box_iou=0.5,
            max_members=4,
        )
        summary = summarize_prediction_rows([row])

        self.assertTrue(row["group_set_exact"])
        self.assertTrue(row["flatten_exact"])
        self.assertEqual(summary["group_set_exact"], 1.0)
        self.assertEqual(summary["object_count"]["exact"], 1.0)
        self.assertEqual(summary["type_count_set_exact"], 1.0)
        self.assertEqual(summary["member_count_set_exact"], 1.0)
        self.assertEqual(summary["types"]["hashed_wedge"]["member_count"]["recall"], 1.0)
        self.assertEqual(summary["member_line_geometry"]["recall"]["1q"], 1.0)
        self.assertEqual(
            summary["group_aligned_member_line_geometry"]["max_endpoint_error_q"]["p90"],
            0.0,
        )
        self.assertEqual(summary["flattened_v0_1"]["classes"]["line"]["f1"], 1.0)
        self.assertAlmostEqual(summary["bond_iou"]["mean"], 1.0, places=7)
        self.assertEqual(summary["invalid"]["objects"], 0)

    def test_checkpoint_loader_is_strict(self) -> None:
        vision = ChemVectorConfig(
            vocab_size=31,
            pad_token_id=0,
            d_model=64,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=128,
            dropout=0.0,
            max_target_length=16,
            vision_dims=(8, 16, 32, 64),
            vision_depths=(1, 1, 1, 1),
        )
        config = BondGroupSetConfig(
            queries=2,
            max_members=3,
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            memory_grid_size=4,
        )
        model = BondGroupSetPredictor(vision, config)
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
            "model_type": model.model_type,
            "model": model.state_dict(),
            "vision_config": asdict(vision),
            "bond_group_set_config": asdict(config),
            "uses_internal_test": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(payload, path)

            restored, _ = load_bond_group_set_model(path, torch.device("cpu"))

            self.assertEqual(restored.config, config)
            payload["uses_internal_test"] = True
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                load_bond_group_set_model(path, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
