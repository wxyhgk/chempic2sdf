import unittest

import torch

from chemvector.bond_group_carrier_geometry import carrier_parameter_variants
from chemvector.bond_group_carrier_set import (
    HASHED_WEDGE,
    NO_OBJECT,
    PARALLEL_2,
    PARALLEL_3,
    SINGLE_LINE,
    SOLID_QUAD,
    SOLID_TRIANGLE,
    TEXT_BOX,
    BondGroupCarrierSetConfig,
    BondGroupCarrierSetPredictor,
    bond_group_carrier_set_assignments,
    bond_group_carrier_set_loss,
    carrier_head_warm_start_state,
    decode_bond_group_carrier_set,
)
from chemvector.bond_group_set import BondGroupSetPredictor
from chemvector.model import ChemVectorConfig


KINDS = (
    SINGLE_LINE,
    PARALLEL_2,
    PARALLEL_3,
    HASHED_WEDGE,
    SOLID_TRIANGLE,
    SOLID_QUAD,
)


def _carrier_rows() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points = torch.tensor(
        [
            [0.10, 0.20, 0.70, 0.25, 0.0, 0.0, 0.0, 0.0],
            [0.15, 0.35, 0.80, 0.40, 0.16, 0.37, 0.79, 0.42],
            [0.20, 0.55, 0.85, 0.50, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.70, 0.78, 0.68, 0.0, 0.0, 0.0, 0.0],
            [0.20, 0.15, 0.38, 0.42, 0.08, 0.45, 0.0, 0.0],
            [0.55, 0.12, 0.82, 0.18, 0.78, 0.44, 0.50, 0.38],
        ],
        dtype=torch.float32,
    )
    positive = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.030, 0.0],
            [0.010, 0.065],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    signed = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return points, positive, signed


def _target() -> dict[str, torch.Tensor]:
    points, positive, signed = _carrier_rows()
    queries = len(KINDS) + 1
    return {
        "types": torch.tensor([(*KINDS, NO_OBJECT)], dtype=torch.long),
        "member_counts": torch.tensor([[1, 2, 3, 6, 1, 1, 0]]),
        "carrier_points": torch.cat(
            (points, torch.zeros(1, 8)),
            dim=0,
        ).unsqueeze(0),
        "positive_params": torch.cat(
            (positive, torch.zeros(1, 2)),
            dim=0,
        ).unsqueeze(0),
        "signed_params": torch.cat(
            (signed, torch.zeros(1, 2)),
            dim=0,
        ).unsqueeze(0),
        "group_stroke": torch.tensor(
            [[0.20, 0.21, 0.22, 0.12, 0.08, 0.08, 0.0]]
        ),
        "text_geometry": torch.zeros(1, queries, 4),
        "target_mask": torch.tensor([[True] * len(KINDS) + [False]]),
        "object_count": torch.tensor([len(KINDS)]),
    }


def _perfect_output(
    target: dict[str, torch.Tensor],
    *,
    variant_index: int = 0,
    reverse_outer: bool = True,
) -> dict[str, torch.Tensor]:
    _, queries = target["types"].shape
    type_logits = torch.full((1, queries, 8), -16.0)
    member_count_logits = torch.full((1, queries, 33), -16.0)
    points = torch.zeros(1, queries, 8)
    positive = torch.zeros(1, queries, 2)
    signed = torch.zeros(1, queries, 2)
    stroke = torch.zeros(1, queries)
    text = torch.zeros(1, queries, 4)
    target_order = list(range(queries))
    if reverse_outer:
        target_order.reverse()
    for query_index, target_index in enumerate(target_order):
        kind_id = int(target["types"][0, target_index])
        type_logits[0, query_index, kind_id] = 16.0
        count = int(target["member_counts"][0, target_index])
        member_count_logits[0, query_index, count] = 16.0
        stroke[0, query_index] = target["group_stroke"][0, target_index]
        text[0, query_index] = target["text_geometry"][0, target_index]
        if kind_id == NO_OBJECT:
            continue
        variants = carrier_parameter_variants(
            target["carrier_points"][0, target_index].unsqueeze(0),
            target["positive_params"][0, target_index].unsqueeze(0),
            target["signed_params"][0, target_index].unsqueeze(0),
            torch.tensor([kind_id]),
        )
        active = torch.where(variants.variant_mask[0])[0]
        selected = active[min(variant_index, len(active) - 1)]
        points[0, query_index] = variants.carrier_points[0, selected]
        positive[0, query_index] = variants.positive_params[0, selected]
        signed[0, query_index] = variants.signed_params[0, selected]
    count_logits = torch.full((1, queries + 1), -16.0)
    count_logits[0, len(KINDS)] = 16.0
    return {
        "type_logits": type_logits,
        "count_logits": count_logits,
        "member_count_logits": member_count_logits,
        "carrier_points": points,
        "positive_params": positive,
        "signed_params": signed,
        "group_stroke": stroke,
        "text_geometry": text,
    }


class BondGroupCarrierSetTest(unittest.TestCase):
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

    def test_forward_shapes_backward_and_four_point_refiner(self) -> None:
        config = BondGroupCarrierSetConfig(
            queries=6,
            max_members=32,
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            memory_grid_size=4,
            local_query_kernel=3,
            endpoint_refine_kernel=3,
        )
        model = BondGroupCarrierSetPredictor(self.vision_config, config)
        output = model(torch.randn(2, 3, 64, 64))
        loss = (
            output["type_logits"].square().mean()
            + output["carrier_points"].mean()
            + output["positive_params"].mean()
            + output["signed_params"].square().mean()
            + output["group_stroke"].mean()
        )
        loss.backward()

        self.assertEqual(output["type_logits"].shape, (2, 6, 8))
        self.assertEqual(output["count_logits"].shape, (2, 7))
        self.assertEqual(output["member_count_logits"].shape, (2, 6, 33))
        self.assertEqual(output["carrier_points"].shape, (2, 6, 8))
        self.assertEqual(output["positive_params"].shape, (2, 6, 2))
        self.assertEqual(output["signed_params"].shape, (2, 6, 2))
        self.assertTrue(torch.equal(output["signed_params"], torch.zeros_like(output["signed_params"])))
        self.assertEqual(output["group_stroke"].shape, (2, 6))
        self.assertEqual(output["text_geometry"].shape, (2, 6, 4))
        self.assertEqual(output["coarse_carrier_points"].shape, (2, 6, 8))
        self.assertIsNotNone(model.carrier_geometry_head[-1].weight.grad)
        self.assertIsNotNone(model.endpoint_refine_mlp[-1].weight.grad)

    def test_model_has_no_max_member_geometry_or_stroke_heads(self) -> None:
        config = BondGroupCarrierSetConfig(
            queries=2,
            max_members=32,
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            memory_grid_size=4,
        )
        model = BondGroupCarrierSetPredictor(self.vision_config, config)
        state = model.state_dict()

        self.assertFalse(hasattr(model, "member_embedding"))
        self.assertFalse(hasattr(model, "member_geometry_head"))
        self.assertFalse(hasattr(model, "member_stroke_head"))
        self.assertFalse(hasattr(model, "signed_params_head"))
        self.assertEqual(model.carrier_embedding.weight.shape, (1, 32))
        self.assertFalse(any("member_geometry" in name for name in state))
        self.assertFalse(any("member_stroke" in name for name in state))

    def test_matched_carrier_loss_backpropagates_all_parameter_heads(self) -> None:
        config = BondGroupCarrierSetConfig(
            queries=len(KINDS) + 1,
            max_members=32,
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            memory_grid_size=4,
        )
        model = BondGroupCarrierSetPredictor(self.vision_config, config)
        output = model(torch.randn(1, 3, 48, 48))

        losses = bond_group_carrier_set_loss(output, _target())
        losses["loss"].backward()

        self.assertTrue(torch.isfinite(losses["loss"]))
        for head in (
            model.carrier_geometry_head,
            model.positive_params_head,
            model.group_stroke_head,
        ):
            self.assertIsNotNone(head[-1].weight.grad)
            self.assertGreater(float(head[-1].weight.grad.abs().sum()), 0.0)
        self.assertEqual(float(losses["signed_param_loss"].detach()), 0.0)

    def test_slot_zero_heads_have_exact_warm_start_shapes(self) -> None:
        config = BondGroupCarrierSetConfig(
            queries=2,
            max_members=32,
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            memory_grid_size=4,
        )
        old = BondGroupSetPredictor(self.vision_config, config)
        new = BondGroupCarrierSetPredictor(self.vision_config, config)
        mapped = carrier_head_warm_start_state(old.state_dict())
        state = new.state_dict()
        state.update(mapped)
        new.load_state_dict(state, strict=True)
        hidden = torch.randn(2, config.d_model)
        old_features = hidden + old.member_embedding.weight[0]
        new_features = hidden + new.carrier_embedding.weight[0]

        torch.testing.assert_close(
            old.member_geometry_head(old_features),
            new.carrier_geometry_head(new_features),
        )
        torch.testing.assert_close(
            old.member_stroke_head(old_features),
            new.group_stroke_head(new_features),
        )

    def test_hungarian_loss_is_outer_order_and_typed_variant_invariant(self) -> None:
        target = _target()
        output = _perfect_output(target, variant_index=7, reverse_outer=True)

        assignments = bond_group_carrier_set_assignments(output, target)
        losses = bond_group_carrier_set_loss(output, target)

        self.assertEqual(len(assignments[0][0]), len(KINDS))
        self.assertLess(float(losses["classification_loss"]), 1e-6)
        self.assertAlmostEqual(float(losses["geometry_loss"]), 0.0, places=7)
        self.assertAlmostEqual(float(losses["stroke_loss"]), 0.0, places=7)
        self.assertLess(float(losses["member_count_loss"]), 1e-6)

    def test_every_flip_and_polygon_cycle_has_zero_geometry_loss(self) -> None:
        target = _target()
        for variant_index in range(8):
            with self.subTest(variant_index=variant_index):
                output = _perfect_output(
                    target,
                    variant_index=variant_index,
                    reverse_outer=False,
                )
                losses = bond_group_carrier_set_loss(output, target)
                self.assertAlmostEqual(
                    float(losses["geometry_loss"]),
                    0.0,
                    places=7,
                )

    def test_decode_emits_carrier_rows_without_member_expansion(self) -> None:
        logits = torch.full((1, 3, 8), -8.0)
        logits[0, 0, PARALLEL_2] = 8.0
        logits[0, 1, TEXT_BOX] = 8.0
        logits[0, 2, NO_OBJECT] = 8.0
        count_logits = torch.full((1, 4), -8.0)
        count_logits[0, 2] = 8.0
        member_count_logits = torch.full((1, 3, 33), -8.0)
        member_count_logits[0, 0, 2] = 8.0
        member_count_logits[0, 1, 0] = 8.0
        output = {
            "type_logits": logits,
            "count_logits": count_logits,
            "member_count_logits": member_count_logits,
            "carrier_points": torch.rand(1, 3, 8),
            "positive_params": torch.rand(1, 3, 2),
            "signed_params": torch.rand(1, 3, 2).mul(2).sub(1),
            "group_stroke": torch.rand(1, 3),
            "text_geometry": torch.rand(1, 3, 4),
        }

        decoded = decode_bond_group_carrier_set(output, mode="count_head")

        self.assertEqual(len(decoded[0]), 2)
        by_kind = {row["kind"]: row for row in decoded[0]}
        carrier = by_kind["parallel_2"]
        self.assertIn("carrier_points", carrier)
        self.assertIn("positive_params", carrier)
        self.assertIn("signed_params", carrier)
        self.assertIn("group_stroke", carrier)
        self.assertNotIn("member_geometry", carrier)
        self.assertEqual(by_kind["text_box"]["member_count"], 0)

    def test_decode_constrains_member_counts_by_carrier_type(self) -> None:
        logits = torch.full((1, 6, 8), -8.0)
        kinds = (
            SINGLE_LINE,
            PARALLEL_2,
            PARALLEL_3,
            HASHED_WEDGE,
            SOLID_TRIANGLE,
            SOLID_QUAD,
        )
        for query_index, kind in enumerate(kinds):
            logits[0, query_index, kind] = 8.0
        count_logits = torch.full((1, 7), -8.0)
        count_logits[0, 6] = 8.0
        member_count_logits = torch.full((1, 6, 33), -8.0)
        member_count_logits[..., 0] = 8.0
        member_count_logits[0, 3, 7] = 12.0
        output = {
            "type_logits": logits,
            "count_logits": count_logits,
            "member_count_logits": member_count_logits,
            "carrier_points": torch.rand(1, 6, 8),
            "positive_params": torch.rand(1, 6, 2),
            "signed_params": torch.rand(1, 6, 2).mul(2).sub(1),
            "group_stroke": torch.rand(1, 6),
            "text_geometry": torch.rand(1, 6, 4),
        }

        decoded = decode_bond_group_carrier_set(output, mode="count_head")[0]

        by_kind = {row["type"]: row for row in decoded}
        self.assertEqual(
            [by_kind[kind]["member_count"] for kind in kinds],
            [1, 2, 3, 7, 1, 1],
        )
        self.assertEqual(
            [by_kind[kind]["raw_member_count"] for kind in kinds],
            [0, 0, 0, 7, 0, 0],
        )


if __name__ == "__main__":
    unittest.main()
