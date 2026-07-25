from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

from chemvector.bond_group_set import (
    HASHED_WEDGE,
    NO_OBJECT,
    PARALLEL_2,
    SINGLE_LINE,
    TEXT_BOX,
    BondGroupSetConfig,
    BondGroupSetPredictor,
    bond_group_set_assignments,
    bond_group_set_loss,
    decode_bond_group_set,
)
from chemvector.bond_group_set_data import build_bond_group_set_target
from chemvector.model import ChemVectorConfig


FIXTURE = (
    Path(__file__).parents[1]
    / "data_sdf/fixtures/visual_svg_v0_2/all_group_kinds.json"
)


def _batch_target(query_count: int = 8, max_members: int = 4) -> dict[str, torch.Tensor]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    target = build_bond_group_set_target(
        payload,
        query_count=query_count,
        max_members=max_members,
    )
    return {name: value.unsqueeze(0) for name, value in target.items()}


def _perfect_output(target: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    batch, queries = target["types"].shape
    members = target["member_geometry"].size(2)
    order = torch.arange(queries - 1, -1, -1)
    logits = torch.full((batch, queries, 8), -12.0)
    member_count_logits = torch.full((batch, queries, members + 1), -12.0)
    member_geometry = torch.zeros((batch, queries, members, 8))
    member_stroke = torch.zeros((batch, queries, members))
    text_geometry = torch.zeros((batch, queries, 4))
    for query_index, target_index in enumerate(order.tolist()):
        object_type = int(target["types"][0, target_index])
        logits[0, query_index, object_type] = 12.0
        count = int(target["member_counts"][0, target_index])
        member_count_logits[0, query_index, count] = 12.0
        member_geometry[0, query_index] = target["member_geometry"][0, target_index]
        member_stroke[0, query_index] = target["member_stroke"][0, target_index]
        text_geometry[0, query_index] = target["text_geometry"][0, target_index]
    expected_count = int(target["target_mask"].sum())
    count_logits = torch.full((batch, queries + 1), -12.0)
    count_logits[:, expected_count] = 12.0
    return {
        "type_logits": logits,
        "count_logits": count_logits,
        "member_count_logits": member_count_logits,
        "member_geometry": member_geometry,
        "member_stroke": member_stroke,
        "text_geometry": text_geometry,
    }


class BondGroupSetModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vision_config = ChemVectorConfig(
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

    def test_forward_shapes_local_refinement_and_backward(self) -> None:
        config = BondGroupSetConfig(
            queries=6,
            max_members=4,
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            memory_grid_size=4,
            local_query_kernel=3,
            endpoint_refine_kernel=3,
        )
        model = BondGroupSetPredictor(self.vision_config, config)

        output = model(torch.randn(2, 3, 64, 64))
        loss = (
            output["type_logits"].square().mean()
            + output["member_geometry"].mean()
            + output["member_count_logits"].square().mean()
        )
        loss.backward()

        self.assertEqual(tuple(output["type_logits"].shape), (2, 6, 8))
        self.assertEqual(tuple(output["count_logits"].shape), (2, 7))
        self.assertEqual(tuple(output["member_count_logits"].shape), (2, 6, 5))
        self.assertEqual(tuple(output["member_geometry"].shape), (2, 6, 4, 8))
        self.assertEqual(tuple(output["member_stroke"].shape), (2, 6, 4))
        self.assertEqual(tuple(output["text_geometry"].shape), (2, 6, 4))
        self.assertIn("coarse_member_geometry", output)
        self.assertIsNotNone(model.member_geometry_head[-1].weight.grad)
        self.assertIsNotNone(model.endpoint_refine_mlp[-1].weight.grad)

    def test_model_shape_contract_can_be_instantiated_at_143_members(self) -> None:
        config = BondGroupSetConfig(
            queries=2,
            max_members=143,
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            memory_grid_size=4,
        )
        model = BondGroupSetPredictor(self.vision_config, config)

        output = model(torch.randn(1, 3, 32, 32))

        self.assertEqual(tuple(output["member_geometry"].shape), (1, 2, 143, 8))
        self.assertEqual(tuple(output["member_count_logits"].shape), (1, 2, 144))

    def test_hungarian_and_loss_support_143_members_without_truncation(self) -> None:
        members = 143
        target_geometry = torch.zeros((1, 2, members, 8))
        for index in range(members):
            y = (index + 1) / (members + 2)
            target_geometry[0, 0, index, :4] = torch.tensor(
                (0.2, y, 0.8, y)
            )
        member_mask = torch.zeros((1, 2, members), dtype=torch.bool)
        member_mask[0, 0] = True
        target = {
            "types": torch.tensor([[HASHED_WEDGE, NO_OBJECT]]),
            "member_counts": torch.tensor([[members, 0]]),
            "member_geometry": target_geometry,
            "member_geometry_mask": member_mask[..., None].expand(-1, -1, -1, 8),
            "member_stroke": torch.full((1, 2, members), 11.0 / 63.0),
            "member_mask": member_mask,
            "text_geometry": torch.zeros((1, 2, 4)),
            "target_mask": torch.tensor([[True, False]]),
            "object_count": torch.tensor([1]),
        }
        type_logits = torch.full((1, 2, 8), -12.0)
        type_logits[0, 0, HASHED_WEDGE] = 12.0
        type_logits[0, 1, NO_OBJECT] = 12.0
        member_count_logits = torch.full((1, 2, members + 1), -12.0)
        member_count_logits[0, 0, members] = 12.0
        member_count_logits[0, 1, 0] = 12.0
        count_logits = torch.full((1, 3), -12.0)
        count_logits[0, 1] = 12.0
        output = {
            "type_logits": type_logits,
            "count_logits": count_logits,
            "member_count_logits": member_count_logits,
            "member_geometry": target_geometry.clone(),
            "member_stroke": target["member_stroke"].clone(),
            "text_geometry": torch.zeros((1, 2, 4)),
        }

        losses = bond_group_set_loss(output, target)

        self.assertAlmostEqual(float(losses["geometry_loss"]), 0.0, places=7)
        self.assertLess(float(losses["member_count_loss"]), 1e-6)

    def test_hungarian_loss_is_invariant_to_outer_object_order(self) -> None:
        target = _batch_target()
        output = _perfect_output(target)

        losses = bond_group_set_loss(output, target)

        self.assertLess(float(losses["classification_loss"]), 1e-6)
        self.assertAlmostEqual(float(losses["geometry_loss"]), 0.0, places=7)
        self.assertAlmostEqual(float(losses["text_geometry_loss"]), 0.0, places=7)
        self.assertAlmostEqual(float(losses["stroke_loss"]), 0.0, places=7)
        self.assertLess(float(losses["member_count_loss"]), 1e-6)
        self.assertEqual(len(losses["assignments"][0][0]), 7)

    def test_line_endpoint_reversal_does_not_change_assignment_cost(self) -> None:
        target = _batch_target()
        output = _perfect_output(target)
        line_queries = torch.where(
            (output["type_logits"].argmax(dim=-1)[0] == SINGLE_LINE)
            | (output["type_logits"].argmax(dim=-1)[0] == PARALLEL_2)
        )[0]
        for query_index in line_queries.tolist():
            geometry = output["member_geometry"][0, query_index]
            geometry[:, :4] = geometry[:, :4].reshape(-1, 2, 2).flip(1).flatten(1)

        assignments = bond_group_set_assignments(output, target)
        losses = bond_group_set_loss(output, target)

        self.assertEqual(len(assignments[0][0]), 7)
        self.assertAlmostEqual(float(losses["geometry_loss"]), 0.0, places=7)

    def test_decode_uses_independent_total_and_member_count_heads(self) -> None:
        logits = torch.full((1, 3, 8), -8.0)
        logits[0, 0, SINGLE_LINE] = 8.0
        logits[0, 1, TEXT_BOX] = 7.0
        logits[0, 2, NO_OBJECT] = 8.0
        count_logits = torch.full((1, 4), -8.0)
        count_logits[0, 2] = 8.0
        member_count_logits = torch.full((1, 3, 5), -8.0)
        member_count_logits[0, 0, 1] = 8.0
        member_count_logits[0, 1, 4] = 8.0
        output = {
            "type_logits": logits,
            "count_logits": count_logits,
            "member_count_logits": member_count_logits,
            "member_geometry": torch.rand(1, 3, 4, 8),
            "member_stroke": torch.rand(1, 3, 4),
            "text_geometry": torch.rand(1, 3, 4),
        }

        decoded = decode_bond_group_set(output, mode="count_head")

        self.assertEqual(len(decoded[0]), 2)
        by_kind = {row["kind"]: row for row in decoded[0]}
        self.assertEqual(by_kind["single_line"]["member_count"], 1)
        self.assertEqual(by_kind["text_box"]["member_count"], 0)


if __name__ == "__main__":
    unittest.main()
