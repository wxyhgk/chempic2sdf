from __future__ import annotations

import math
import unittest

import torch

from Data.sdf.visual_svg_v0_2 import parse_visual_svg_v0_2
from chemvector.bond_group_proposal_membership import (
    HASHED_WEDGE,
    PARALLEL_2,
    SINGLE_LINE,
)
from chemvector.bond_group_proposal_membership_pipeline import (
    PROPOSAL_DESCRIPTOR_DIM,
    PROPOSAL_DESCRIPTOR_FIELDS,
    aggregate_group_slot_targets,
    bond_only_expected_v0_2_sequence,
    build_proposal_descriptors,
    gather_flat_type_logits,
    membership_rows_to_evaluator_rows,
    prepare_proposal_membership_pipeline,
    summarize_restricted_partitions,
)
from chemvector.primitive_set import LINE, NO_OBJECT


def primitive(
    primitive_id: int,
    points: list[list[int]],
    *,
    owner_kind: str,
    owner_id: int,
    kind: str = "line",
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
        value["stroke_width_q"] = 20
    return value


def sidecar() -> dict:
    return {
        "schema_version": "visual-primitives-0.1",
        "primitives": [
            primitive(0, [[100, 100], [300, 100]], owner_kind="bond", owner_id=1),
            primitive(1, [[100, 112], [300, 112]], owner_kind="bond", owner_id=1),
            primitive(2, [[120, 300], [160, 300]], owner_kind="bond", owner_id=2),
            primitive(3, [[110, 316], [170, 316]], owner_kind="bond", owner_id=2),
            primitive(4, [[100, 332], [180, 332]], owner_kind="bond", owner_id=2),
            primitive(5, [[400, 400], [500, 400]], owner_kind="bond", owner_id=3),
            primitive(
                6,
                [[600, 100], [650, 140]],
                owner_kind="atom",
                owner_id=9,
                kind="text_box",
            ),
        ],
        "bond_groups": [
            {"bond_id": 1, "direction": "NONE", "primitive_ids": [0, 1]},
            {
                "bond_id": 2,
                "direction": "BEGINDASH",
                "primitive_ids": [2, 3, 4],
            },
            {"bond_id": 3, "direction": "NONE", "primitive_ids": [5]},
        ],
        "text_instances": [{"atom_id": 9, "primitive_ids": [6]}],
    }


def proposal_batch() -> dict:
    geometry = torch.zeros((1, 7, 8), dtype=torch.float32)
    geometry[0, 0, :4] = torch.tensor([0.1, 0.2, 0.4, 0.6])
    geometry[0, 1, :4] = torch.tensor([0.1, 0.22, 0.4, 0.62])
    geometry[0, 2, :4] = torch.tensor([0.2, 0.5, 0.3, 0.5])
    geometry[0, 3, :4] = torch.tensor([0.18, 0.55, 0.32, 0.55])
    geometry[0, 4, :4] = torch.tensor([0.7, 0.7, 0.8, 0.7])
    geometry[0, 5, :4] = torch.tensor([0.4, 0.8, 0.6, 0.8])
    return {
        "proposal_features": torch.arange(1 * 7 * 4).reshape(1, 7, 4).float(),
        "proposal_geometry": geometry,
        "proposal_stroke": torch.tensor([[0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.0]]),
        "proposal_score": torch.tensor([[0.9, 0.8, 0.95, 0.85, 0.7, 0.6, 0.0]]),
        "proposal_type": torch.tensor([[LINE, LINE, LINE, LINE, LINE, LINE, NO_OBJECT]]),
        "proposal_query_index": torch.tensor([[3, 1, 5, 2, 4, 0, -1]]),
        "proposal_mask": torch.tensor([[True, True, True, True, True, True, False]]),
        # group 10 is complete p2; group 20 is a two-support hashed count 3;
        # group 30 is a usable singleton; group 40 is an ignored partial p2.
        "group_membership": torch.tensor([[10, 10, 20, 20, 30, 40, -1]]),
        "group_type": torch.tensor(
            [[PARALLEL_2, PARALLEL_2, HASHED_WEDGE, HASHED_WEDGE, SINGLE_LINE, PARALLEL_2, 0]]
        ),
        "group_count": torch.tensor([[2, 2, 3, 3, 1, 2, 0]]),
        "group_mask": torch.tensor([[True, True, True, True, True, False, False]]),
        "match_weight": torch.tensor([[1.0, 0.25, 1.0, 0.5, 1.0, 0.0, 0.0]]),
        "primitives": [sidecar()],
    }


class BondGroupProposalMembershipPipelineTest(unittest.TestCase):
    def test_descriptor_has_strict_order_and_padding_is_zero(self) -> None:
        batch = proposal_batch()
        references = torch.tensor(
            [[0.05, 0.1], [0.15, 0.2], [0.25, 0.3], [0.35, 0.4], [0.45, 0.5], [0.55, 0.6]]
        )

        descriptor = build_proposal_descriptors(batch, references)

        self.assertEqual(descriptor.shape, (1, 7, PROPOSAL_DESCRIPTOR_DIM))
        self.assertEqual(len(PROPOSAL_DESCRIPTOR_FIELDS), PROPOSAL_DESCRIPTOR_DIM)
        torch.testing.assert_close(descriptor[0, 0, :8], batch["proposal_geometry"][0, 0])
        self.assertEqual(descriptor[0, 0, 8:16].tolist(), [1, 1, 1, 1, 0, 0, 0, 0])
        self.assertAlmostEqual(float(descriptor[0, 0, 16]), 0.2)
        self.assertAlmostEqual(float(descriptor[0, 0, 17]), 0.9)
        torch.testing.assert_close(descriptor[0, 0, 18:20], references[3])
        torch.testing.assert_close(descriptor[0, 0, 20:22], torch.tensor([0.25, 0.4]))
        self.assertAlmostEqual(float(descriptor[0, 0, 22]), 0.5, places=6)
        self.assertTrue(torch.equal(descriptor[0, 6], torch.zeros(23)))

    def test_flat_type_logits_are_gathered_by_query_index(self) -> None:
        batch = proposal_batch()
        logits = torch.arange(1 * 6 * 5).reshape(1, 6, 5).float()

        gathered = gather_flat_type_logits(
            logits,
            batch["proposal_query_index"],
            batch["proposal_mask"],
        )

        torch.testing.assert_close(gathered[0, 0], logits[0, 3])
        torch.testing.assert_close(gathered[0, 5], logits[0, 0])
        self.assertTrue(torch.equal(gathered[0, 6], torch.zeros(5)))

    def test_slot_aggregation_keeps_hashed_partial_and_ignores_fixed_partial(self) -> None:
        batch = proposal_batch()

        targets = aggregate_group_slot_targets(batch, group_slots=5)

        self.assertEqual(int(targets["known_counts"][0]), 3)
        self.assertEqual(targets["source_group_ids"][0, :3].tolist(), [10, 20, 30])
        self.assertEqual(
            targets["group_types"][0, :3].tolist(),
            [PARALLEL_2, HASHED_WEDGE, SINGLE_LINE],
        )
        self.assertEqual(targets["member_counts"][0, :3].tolist(), [2, 3, 1])
        self.assertEqual(
            targets["group_membership"][0, :3].sum(dim=-1).tolist(),
            [2, 2, 1],
        )
        torch.testing.assert_close(
            targets["group_weights"][0, :3],
            torch.tensor([0.625, 0.75, 1.0]),
        )
        self.assertEqual(
            targets["proposal_keep"][0].tolist(),
            [True, True, True, True, True, False, False],
        )
        self.assertEqual(
            targets["proposal_types"][0].tolist(),
            [LINE, LINE, LINE, LINE, LINE, NO_OBJECT, NO_OBJECT],
        )
        self.assertFalse(bool(targets["group_mask"][0, 3:].any()))

    def test_prepare_helper_returns_model_inputs_targets_counts_and_coverage(self) -> None:
        batch = proposal_batch()
        flat_output = {"type_logits": torch.randn(1, 6, 5)}
        references = torch.rand(6, 2)

        prepared = prepare_proposal_membership_pipeline(
            batch,
            flat_output,
            references,
            group_slots=5,
        )

        self.assertEqual(
            set(prepared["model_inputs"]),
            {
                "proposal_features",
                "proposal_descriptors",
                "flat_type_logits",
                "proposal_scores",
                "proposal_mask",
            },
        )
        self.assertEqual(int(prepared["known_counts"][0]), 3)
        self.assertIs(prepared["known_counts"], prepared["targets"]["known_counts"])
        coverage = prepared["oracle_coverage"]
        self.assertEqual(int(coverage["expected_physical"][0]), 6)
        self.assertEqual(int(coverage["oracle_matched"][0]), 6)
        self.assertEqual(int(coverage["oracle_keep"][0]), 5)
        self.assertEqual(int(coverage["oracle_type_correct"][0]), 5)

    def test_frozen_geometry_fits_fixed_and_expands_two_support_hashed(self) -> None:
        batch = proposal_batch()
        decoded = [
            {
                "query_index": 0,
                "score": 0.9,
                "type": PARALLEL_2,
                "kind": "parallel_2",
                "member_count": 2,
                "proposal_indices": [0, 1],
                "valid_membership": True,
            },
            {
                "query_index": 1,
                "score": 0.8,
                "type": HASHED_WEDGE,
                "kind": "hashed_wedge",
                "member_count": 5,
                "proposal_indices": [2, 3],
                "valid_membership": True,
            },
        ]

        result = membership_rows_to_evaluator_rows(
            decoded,
            batch,
            batch_index=0,
            max_members=8,
        )

        self.assertEqual(result.fit_failures, ())
        self.assertEqual(len(result.rows), 2)
        by_kind = {row["kind"]: row for row in result.rows}
        self.assertEqual(tuple(by_kind["parallel_2"]["member_geometry"].shape), (2, 8))
        self.assertEqual(tuple(by_kind["hashed_wedge"]["member_geometry"].shape), (5, 8))
        self.assertEqual(len(by_kind["hashed_wedge"]["member_stroke"]), 5)

    def test_bond_only_expected_sequence_removes_text(self) -> None:
        sequence = bond_only_expected_v0_2_sequence(sidecar())
        parsed = parse_visual_svg_v0_2(sequence)

        self.assertEqual(len(parsed.bond_groups), 3)
        self.assertEqual(parsed.text_boxes, ())

    def test_restricted_partition_is_slot_permutation_invariant(self) -> None:
        targets = aggregate_group_slot_targets(proposal_batch(), group_slots=5)
        decoded = [
            [
                {
                    "type": HASHED_WEDGE,
                    "member_count": 3,
                    "proposal_indices": [2, 3],
                    "valid_membership": True,
                },
                {
                    "type": SINGLE_LINE,
                    "member_count": 1,
                    "proposal_indices": [4],
                    "valid_membership": True,
                },
                {
                    "type": PARALLEL_2,
                    "member_count": 2,
                    "proposal_indices": [1, 0],
                    "valid_membership": True,
                },
            ]
        ]

        exact = summarize_restricted_partitions(decoded, targets)
        self.assertEqual(exact["partition_exact"], 1.0)
        self.assertEqual(exact["typed_partition_exact"], 1.0)
        self.assertEqual(exact["groups"]["f1"], 1.0)
        self.assertEqual(exact["pairs"]["f1"], 1.0)

        decoded[0][0]["proposal_indices"] = [0, 3]
        wrong = summarize_restricted_partitions(decoded, targets)
        self.assertEqual(wrong["partition_exact"], 0.0)
        self.assertLess(wrong["groups"]["f1"], 1.0)
        self.assertLess(wrong["pairs"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
