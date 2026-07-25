from __future__ import annotations

import unittest

from PIL import Image
import torch

from chemvector.bond_group_proposal_membership_data import (
    BondGroupProposalMembershipCollator,
    build_primitive_group_targets,
)
from chemvector.bond_group_set import HASHED_WEDGE, PARALLEL_2, SOLID_TRIANGLE
from chemvector.primitive_set import LINE, TEXT_BOX, TRIANGLE


def primitive(
    primitive_id: int,
    kind: str,
    points: list[list[int]],
    *,
    owner_kind: str,
    owner_id: int,
    stroke_width_q: int = 20,
) -> dict:
    value = {
        "primitive_id": primitive_id,
        "canonical_index": primitive_id,
        "kind": kind,
        "owner_kind": owner_kind,
        "owner_id": owner_id,
        "quantized_points": points,
    }
    if kind != "text_box":
        value["stroke_width_q"] = stroke_width_q
    return value


def sidecar() -> dict:
    primitives = [
        primitive(0, "line", [[100, 100], [300, 100]], owner_kind="bond", owner_id=10),
        primitive(1, "line", [[100, 112], [300, 112]], owner_kind="bond", owner_id=10),
        primitive(2, "line", [[100, 200], [300, 200]], owner_kind="bond", owner_id=11),
        primitive(3, "line", [[100, 212], [300, 212]], owner_kind="bond", owner_id=11),
        primitive(4, "line", [[120, 300], [160, 300]], owner_kind="bond", owner_id=12),
        primitive(5, "line", [[110, 316], [170, 316]], owner_kind="bond", owner_id=12),
        primitive(6, "line", [[100, 332], [180, 332]], owner_kind="bond", owner_id=12),
        primitive(
            7,
            "text_box",
            [[400, 100], [460, 140]],
            owner_kind="atom",
            owner_id=20,
        ),
    ]
    return {
        "schema_version": "visual-primitives-0.1",
        "primitives": primitives,
        "bond_groups": [
            {"bond_id": 10, "direction": "NONE", "primitive_ids": [0, 1]},
            {"bond_id": 11, "direction": "NONE", "primitive_ids": [2, 3]},
            {
                "bond_id": 12,
                "direction": "BEGINDASH",
                "primitive_ids": [4, 5, 6],
            },
        ],
        "text_instances": [{"atom_id": 20, "primitive_ids": [7]}],
    }


def one_group_sidecar(kind: str) -> dict:
    if kind == "triangle":
        physical = primitive(
            0,
            "triangle",
            [[100, 500], [180, 520], [100, 540]],
            owner_kind="bond",
            owner_id=1,
            stroke_width_q=6,
        )
    else:
        physical = primitive(
            0,
            "line",
            [[100, 100], [300, 100]],
            owner_kind="bond",
            owner_id=1,
        )
    return {
        "schema_version": "visual-primitives-0.1",
        "primitives": [physical],
        "bond_groups": [
            {"bond_id": 1, "direction": "NONE", "primitive_ids": [0]}
        ],
        "text_instances": [],
    }


def flat_output(
    rows: list[tuple[int, list[float], float]],
    *,
    queries: int = 8,
    feature_width: int = 4,
    text_queries: tuple[int, ...] = (),
) -> dict[str, torch.Tensor]:
    type_logits = torch.full((1, queries, 5), -8.0)
    type_logits[..., 0] = 8.0
    geometry = torch.zeros((1, queries, 8))
    stroke = torch.full((1, queries), 20 / 63)
    features = torch.arange(
        queries * feature_width,
        dtype=torch.float32,
    ).reshape(1, queries, feature_width)
    for query_index, values, object_logit in rows:
        type_logits[0, query_index] = -8.0
        type_logits[0, query_index, LINE] = object_logit
        geometry[0, query_index, : len(values)] = torch.tensor(values)
    for query_index in text_queries:
        type_logits[0, query_index] = -8.0
        type_logits[0, query_index, TEXT_BOX] = 8.0
        geometry[0, query_index, :4] = torch.tensor(
            [430 / 2047, 120 / 2047, 60 / 2047, 40 / 2047]
        )
    count_logits = torch.full((1, queries + 1), -8.0)
    count_logits[0, len(rows) + len(text_queries)] = 8.0
    return {
        "type_logits": type_logits,
        "count_logits": count_logits,
        "geometry": geometry,
        "text_geometry": geometry.clone(),
        "stroke": stroke,
        "object_features": features,
    }


def normalized_line(x0: int, y0: int, x1: int, y1: int) -> list[float]:
    return [x0 / 2047, y0 / 2047, x1 / 2047, y1 / 2047]


def sample(payload: dict, *, split: str = "train") -> dict:
    return {
        "key": "membership-sample",
        "image": Image.new("RGB", (16, 16), "white"),
        "bond_layer": Image.new("L", (16, 16), 0),
        "primitives": payload,
        "meta": {"source_split": split},
    }


class BondGroupProposalMembershipDataTest(unittest.TestCase):
    def test_builds_primitive_to_group_protocol_from_primitive_ids(self) -> None:
        targets = build_primitive_group_targets(sidecar())

        self.assertEqual(set(targets), set(range(7)))
        self.assertEqual(targets[0].group_id, targets[1].group_id)
        self.assertEqual(targets[0].group_type, PARALLEL_2)
        self.assertEqual(targets[0].group_count, 2)
        self.assertEqual(targets[4].group_type, HASHED_WEDGE)
        self.assertEqual(targets[4].group_count, 3)
        self.assertNotIn(7, targets)

    def test_threshold_oracle_keep_weights_shell_and_ignores_partial_fixed(self) -> None:
        payload = sidecar()
        output = flat_output(
            [
                (0, normalized_line(300, 100, 100, 100), 8.0),
                (1, normalized_line(100, 122, 300, 122), 8.0),
                (2, normalized_line(100, 200, 300, 200), 8.0),
                (3, normalized_line(120, 300, 160, 300), 8.0),
                (4, normalized_line(110, 316, 170, 316), 8.0),
                (5, normalized_line(700, 700, 800, 700), 8.0),
                (7, normalized_line(100, 212, 300, 212), 0.0),
            ],
            text_queries=(6,),
        )
        output["type_logits"][0, 7] = torch.tensor(
            [0.0, 0.1, -0.1, -0.1, -0.1]
        )
        collator = BondGroupProposalMembershipCollator(
            image_size=16,
            max_proposals=8,
            shell_weight=0.25,
        )

        batch = collator([sample(payload)], flat_output=output)

        self.assertEqual(int(batch["proposal_count"][0]), 5)
        self.assertEqual(
            batch["proposal_query_index"][0, :5].tolist(),
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            batch["group_membership"][0, :5].tolist(),
            [0, 0, 1, 2, 2],
        )
        self.assertEqual(
            batch["group_type"][0, :5].tolist(),
            [PARALLEL_2, PARALLEL_2, PARALLEL_2, HASHED_WEDGE, HASHED_WEDGE],
        )
        self.assertEqual(batch["group_count"][0, :5].tolist(), [2, 2, 2, 3, 3])
        self.assertEqual(
            batch["group_mask"][0, :5].tolist(),
            [True, True, False, True, True],
        )
        torch.testing.assert_close(
            batch["match_weight"][0, :5],
            torch.tensor([1.0, 0.25, 0.0, 1.0, 1.0]),
        )
        self.assertTrue(bool(batch["proposal_mask"][0, :5].all()))
        self.assertFalse(bool(batch["proposal_mask"][0, 5:].any()))
        self.assertIs(batch["primitives"][0], payload)
        self.assertIn("types", batch)
        self.assertIn("geometry", batch)

        # The input geometry and feature come from query 0, not from GT target 0.
        torch.testing.assert_close(
            batch["proposal_geometry"][0, 0, :4],
            torch.tensor(normalized_line(300, 100, 100, 100)),
        )
        torch.testing.assert_close(
            batch["proposal_features"][0, 0],
            output["object_features"][0, 0],
        )

    def test_polygon_matching_is_invariant_to_cycle_and_direction(self) -> None:
        payload = one_group_sidecar("triangle")
        target = [[100, 500], [180, 520], [100, 540]]
        reordered = [target[1], target[0], target[2]]
        values = [coordinate / 2047 for point in reordered for coordinate in point]
        output = flat_output([], queries=2)
        output["type_logits"][0, 0] = -8.0
        output["type_logits"][0, 0, TRIANGLE] = 8.0
        output["geometry"][0, 0, :6] = torch.tensor(values)
        output["stroke"][0, 0] = 6 / 63

        batch = BondGroupProposalMembershipCollator(
            image_size=16,
            max_proposals=2,
        )([sample(payload)], flat_output=output)

        self.assertEqual(int(batch["proposal_count"][0]), 1)
        self.assertEqual(int(batch["group_type"][0, 0]), SOLID_TRIANGLE)
        self.assertAlmostEqual(float(batch["match_distance_q"][0, 0]), 0.0)

    def test_count_head_is_optional_while_default_threshold_is_point_five(self) -> None:
        payload = one_group_sidecar("line")
        output = flat_output([], queries=2)
        output["type_logits"][0, 0] = torch.tensor([0.0, 0.1, -0.1, -0.1, -0.1])
        output["geometry"][0, 0, :4] = torch.tensor(
            normalized_line(100, 100, 300, 100)
        )
        output["count_logits"] = torch.tensor([[-8.0, 8.0, -8.0]])

        threshold_batch = BondGroupProposalMembershipCollator(
            image_size=16,
            max_proposals=2,
        )([sample(payload)], flat_output=output)
        count_batch = BondGroupProposalMembershipCollator(
            image_size=16,
            max_proposals=2,
            decode_mode="count_head",
        )([sample(payload)], flat_output=output)

        self.assertEqual(int(threshold_batch["proposal_count"][0]), 0)
        self.assertEqual(int(count_batch["proposal_count"][0]), 1)

    def test_rejects_internal_test_before_flat_model_attachment(self) -> None:
        collator = BondGroupProposalMembershipCollator(image_size=16)

        with self.assertRaisesRegex(ValueError, "internal_test"):
            collator([sample(one_group_sidecar("line"), split="internal_test")])


if __name__ == "__main__":
    unittest.main()
