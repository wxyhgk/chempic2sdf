import unittest
from unittest.mock import patch

import torch

import chemvector.g3_hierarchical_graph as g3_module
from chemvector.bond_group_query import bond_group_query_assignments
from chemvector.g3_hierarchical_graph import (
    FixedLearnedBondSlotDecoder,
    G3AtomOnlyPredictor,
    G3AtomPairBondIdentityPredictor,
    G3FixedLearnedBondSlotPredictor,
    G3HierarchicalGraphConfig,
    G3HierarchicalGraphPredictor,
    G3SetLearnedBondSlotPredictor,
    build_canonical_atom_slot_targets,
    build_g3_bond_targets,
    build_g3_atom_pair_targets,
    build_gt_bond_slot_identity_oracle,
    extract_g3_dense_bond_queries,
    g3_fixed_bond_slot_loss,
    g3_atom_only_loss,
    g3_canonical_atom_only_loss,
    g3_atom_pair_bond_loss,
    g3_graph_loss,
    g3_set_bond_slot_assignments,
    g3_set_bond_slot_loss,
    select_count_conditioned_atom_slots,
    spatial_atom_slot_loss,
)


class G3HierarchicalGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.config = G3HierarchicalGraphConfig(
            input_channels=8,
            hidden_channels=16,
            d_model=16,
            atom_slots_per_cell=3,
            maximum_atoms=4,
            bond_slots_per_cell=2,
            maximum_bonds=3,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=32,
            memory_stride=1,
            incidence_hidden_dim=16,
            incidence_pair_hidden_dim=8,
            dropout=0.0,
        )

    @staticmethod
    def _targets() -> dict[str, torch.Tensor]:
        return {
            "atom_anchors": torch.tensor(
                [[[0.20, 0.40], [0.80, 0.40], [0.0, 0.0], [0.0, 0.0]]]
            ),
            "atom_mask": torch.tensor([[True, True, False, False]]),
            "bond_atom_refs": torch.tensor([[[0, 1], [0, 0], [0, 0]]]),
            "bond_types": torch.tensor([[0, 0, 0]]),
            "bond_mask": torch.tensor([[True, False, False]]),
        }

    def _forward(self) -> tuple[G3HierarchicalGraphPredictor, dict]:
        model = G3HierarchicalGraphPredictor(self.config)
        output = model(
            torch.randn(1, 8, 6, 8),
            atom_counts=torch.tensor([2]),
            bond_counts=torch.tensor([1]),
        )
        return model, output

    def test_forward_shapes_counts_and_masks_on_non_square_features(self) -> None:
        model = G3HierarchicalGraphPredictor(self.config)
        output = model(
            torch.randn(2, 8, 6, 8),
            atom_counts=torch.tensor([2, 3]),
            bond_counts=torch.tensor([1, 2]),
        )

        self.assertEqual(
            tuple(output["atom_dense_output"]["presence_logits"].shape),
            (2, 3, 6, 8),
        )
        self.assertEqual(tuple(output["atom_anchors"].shape), (2, 4, 2))
        self.assertEqual(tuple(output["atom_features"].shape), (2, 4, 16))
        self.assertEqual(tuple(output["atom_geometry"].shape), (2, 4, 8))
        self.assertEqual(tuple(output["atom_slot_indices"].shape), (2, 4, 3))
        self.assertEqual(tuple(output["atom_count_logits"].shape), (2, 5))
        self.assertEqual(output["atom_mask"].sum(dim=1).tolist(), [2, 3])
        self.assertEqual(tuple(output["bond_endpoints"].shape), (2, 3, 2, 2))
        self.assertEqual(tuple(output["bond_type_logits"].shape), (2, 3, 4))
        self.assertNotIn("bond_direction_logits", output)
        self.assertEqual(tuple(output["bond_count_logits"].shape), (2, 4))
        self.assertEqual(output["bond_mask"].sum(dim=1).tolist(), [1, 2])
        self.assertEqual(
            tuple(output["endpoint_pointer_logits"].shape), (2, 3, 2, 4)
        )
        self.assertTrue(torch.isfinite(output["atom_anchors"]).all())
        self.assertTrue(torch.isfinite(output["bond_endpoints"]).all())
        self.assertTrue(
            ((output["atom_anchors"] >= 0.0) & (output["atom_anchors"] <= 1.0)).all()
        )
        valid_bond_coordinates = (output["bond_endpoints"] >= 0.0) & (
            output["bond_endpoints"] <= 1.0
        )
        self.assertTrue(valid_bond_coordinates.all())
        self.assertTrue(
            (output["endpoint_pointer_logits"][0, 1:] == -1e4).all()
        )
        self.assertTrue(
            (output["endpoint_pointer_logits"][0, 0, :, 2:] == -1e4).all()
        )

    def test_gt_atom_slot_oracle_preserves_target_order_and_image_features(
        self,
    ) -> None:
        model = G3HierarchicalGraphPredictor(self.config)
        anchors = torch.tensor(
            [[[0.80, 0.40], [0.20, 0.40], [0.0, 0.0], [0.0, 0.0]]]
        )
        mask = torch.tensor([[True, True, False, False]])

        output = model(
            torch.randn(1, 8, 6, 8),
            atom_counts=torch.tensor([2]),
            bond_counts=torch.tensor([1]),
            oracle_atom_anchors=anchors,
            oracle_atom_mask=mask,
        )

        torch.testing.assert_close(output["atom_anchors"], anchors)
        self.assertEqual(output["atom_mask"].tolist(), mask.tolist())
        self.assertEqual(output["atom_target_indices"].tolist(), [[0, 1, -1, -1]])
        self.assertEqual(tuple(output["atom_features"].shape), (1, 4, 16))
        self.assertGreater(
            float(output["atom_features"][0, :2].detach().abs().sum()), 0.0
        )
        self.assertEqual(
            float(output["atom_features"][0, 2:].detach().abs().sum()), 0.0
        )

        with self.assertRaisesRegex(ValueError, "provided together"):
            model(
                torch.randn(1, 8, 6, 8),
                atom_counts=torch.tensor([2]),
                bond_counts=torch.tensor([1]),
                oracle_atom_anchors=anchors,
            )

    def test_count_conditioned_atom_selection_respects_ranking_and_mask(self) -> None:
        logits = torch.tensor([[[[0.0, 9.0, 1.0]], [[8.0, 7.0, 6.0]]]])
        anchors = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 1, 3)
        anchors = anchors / anchors.max()
        output = {
            "presence_logits": logits,
            "anchors": anchors,
            "anchor_offsets": torch.zeros_like(anchors),
            "slot_features": torch.arange(24, dtype=torch.float32).reshape(
                1, 2, 4, 1, 3
            ),
        }
        candidate_mask = torch.ones_like(logits, dtype=torch.bool)
        candidate_mask[0, 0, 0, 1] = False

        selected = select_count_conditioned_atom_slots(
            output,
            torch.tensor([2]),
            maximum_atoms=4,
            candidate_mask=candidate_mask,
        )

        self.assertEqual(selected["atom_mask"].sum().item(), 2)
        self.assertEqual(
            selected["atom_presence_logits"][:, :2].tolist(), [[8.0, 7.0]]
        )
        self.assertNotIn(
            [0, 0, 1], selected["atom_slot_indices"][:, :2].reshape(-1, 3).tolist()
        )
        with self.assertRaisesRegex(ValueError, "exceeds valid dense candidates"):
            select_count_conditioned_atom_slots(
                output,
                torch.tensor([6]),
                maximum_atoms=6,
                candidate_mask=candidate_mask,
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            select_count_conditioned_atom_slots(
                output,
                torch.tensor([2.5]),
                maximum_atoms=4,
                candidate_mask=candidate_mask,
            )

    def test_g3_bond_query_extraction_does_not_apply_local_max_suppression(
        self,
    ) -> None:
        dense = {
            "heatmap_logits": torch.tensor([[[[9.0, 8.0, 0.0]]]]),
            "object_features": torch.randn(1, 16, 1, 3),
            "endpoints": torch.zeros(1, 1, 2, 2, 1, 3),
            "display_logits": torch.zeros(1, 1, 4, 1, 3),
            "chemical_logits": torch.zeros(1, 1, 4, 1, 3),
            "direction_logits": torch.zeros(1, 1, 5, 1, 3),
        }

        proposals = extract_g3_dense_bond_queries(dense, maximum_queries=2)

        self.assertEqual(proposals["base_logits"].tolist(), [[9.0, 8.0]])
        self.assertEqual(
            proposals["indices"].reshape(-1, 3).tolist(), [[0, 0, 0], [0, 0, 1]]
        )

    def test_gt_bond_router_only_reorders_predicted_proposals(self) -> None:
        endpoints = torch.tensor(
            [[
                [[0.70, 0.10], [0.90, 0.10]],
                [[0.40, 0.80], [0.60, 0.80]],
                [[0.10, 0.20], [0.30, 0.20]],
            ]],
            requires_grad=True,
        )
        logits = torch.tensor([[10.0, 20.0, 30.0]], requires_grad=True)
        proposals = {
            "base_endpoints": endpoints,
            "base_logits": logits,
            "base_scores": logits.sigmoid(),
            "base_display_logits": torch.randn(1, 3, 4),
            "base_chemical_logits": torch.randn(1, 3, 4),
            "base_direction_logits": torch.randn(1, 3, 5),
            "base_features": torch.randn(1, 3, 16),
            "mask": torch.ones(1, 3, dtype=torch.bool),
            "indices": torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 0, 2]]]),
            "router_metadata": torch.tensor(17),
        }
        targets = torch.tensor(
            [[
                [[0.11, 0.20], [0.31, 0.20]],
                [[0.71, 0.10], [0.91, 0.10]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]]
        )

        routed = build_gt_bond_slot_identity_oracle(
            proposals, targets, torch.tensor([[True, True, False]])
        )

        self.assertEqual(routed["oracle_permutation"].tolist(), [[2, 0, 1]])
        self.assertEqual(routed["base_logits"].tolist(), [[30.0, 10.0, 20.0]])
        self.assertEqual(routed["bond_target_indices"].tolist(), [[0, 1, -1]])
        self.assertEqual(routed["oracle_positive_mask"].tolist(), [[True, True, False]])
        self.assertEqual(routed["oracle_full_coverage"].tolist(), [True])
        self.assertEqual(int(routed["router_metadata"]), 17)
        self.assertFalse(torch.equal(routed["base_endpoints"][:, :2], targets[:, :2]))
        self.assertTrue(routed["base_endpoints"].requires_grad)
        (routed["base_endpoints"].sum() + routed["base_logits"].sum()).backward()
        self.assertEqual(float(endpoints.grad.sum()), 12.0)
        self.assertEqual(float(logits.grad.sum()), 3.0)

    def test_atom_slot_loss_supports_same_cell_objects_and_rejects_overflow(
        self,
    ) -> None:
        model = G3HierarchicalGraphPredictor(self.config)
        dense = model.atom_head(torch.zeros(1, 8, 8, 8))
        targets = {
            "atom_anchors": torch.tensor(
                [[[0.251, 0.500], [0.252, 0.501], [0.253, 0.502], [0.0, 0.0]]]
            ),
            "atom_mask": torch.tensor([[True, True, True, False]]),
        }

        losses = spatial_atom_slot_loss(dense, targets)
        losses["loss"].backward()

        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertGreater(float(model.atom_head.presence.weight.grad.abs().sum()), 0.0)
        self.assertGreater(
            float(model.atom_head.anchor_offset.weight.grad.abs().sum()), 0.0
        )
        overflow = {
            "atom_anchors": torch.tensor(
                [[
                    [0.2510, 0.5000],
                    [0.2515, 0.5005],
                    [0.2520, 0.5010],
                    [0.2525, 0.5015],
                ]]
            ),
            "atom_mask": torch.ones(1, 4, dtype=torch.bool),
        }
        with self.assertRaisesRegex(ValueError, "slot capacity"):
            spatial_atom_slot_loss(dense, overflow)

    def test_full_loss_backpropagates_through_each_g3_stage(self) -> None:
        model, output = self._forward()

        losses = g3_graph_loss(output, self._targets())
        losses["loss"].backward()

        self.assertTrue(torch.isfinite(losses["loss"]))
        names = (
            "atom_head.presence.weight",
            "atom_head.anchor_offset.weight",
            "atom_head.feature_projection.weight",
            "atom_head.count.2.weight",
            "dense_bond.heatmap.weight",
            "dense_bond.chemical.weight",
            "bond_decoder.base.decoder.layers.0.multihead_attn.in_proj_weight",
            "bond_decoder.atom_conditioner.layers.0.multihead_attn.in_proj_weight",
            "bond_decoder.presence_residual.weight",
            "bond_decoder.endpoint_residual.weight",
            "bond_decoder.chemical_residual.weight",
            "bond_decoder.incidence.atom_projection.0.weight",
            "bond_decoder.incidence.endpoint_projection.0.weight",
            "bond_count.3.weight",
        )
        parameters = dict(model.named_parameters())
        for name in names:
            gradient = parameters[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)

    def test_p0c_computes_one_assignment_shared_by_bond_and_pointer(self) -> None:
        _, output = self._forward()

        with patch(
            "chemvector.g3_hierarchical_graph.bond_group_query_assignments",
            wraps=bond_group_query_assignments,
        ) as matcher:
            losses = g3_graph_loss(
                output,
                self._targets(),
                shared_confidence_aware_bond_assignment=True,
            )

        self.assertEqual(matcher.call_count, 1)
        self.assertEqual(matcher.call_args.kwargs["presence_cost_weight"], 2.0)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_p0d_uses_direct_prefix_identity_without_loss_hungarian(self) -> None:
        model = G3HierarchicalGraphPredictor(self.config)
        targets = self._targets()
        normalized = build_g3_bond_targets(targets)
        output = model(
            torch.randn(1, 8, 6, 8),
            atom_counts=torch.tensor([2]),
            bond_counts=torch.tensor([1]),
            oracle_atom_anchors=targets["atom_anchors"],
            oracle_atom_mask=targets["atom_mask"],
            oracle_bond_endpoints=normalized["endpoints"],
            oracle_bond_mask=normalized["mask"],
        )

        with patch(
            "chemvector.g3_hierarchical_graph.bond_group_query_assignments",
            wraps=bond_group_query_assignments,
        ) as matcher, patch(
            "chemvector.g3_hierarchical_graph.bond_group_query_loss",
            wraps=g3_module.bond_group_query_loss,
        ) as bond_loss, patch(
            "chemvector.g3_hierarchical_graph._incidence_targets",
            wraps=g3_module._incidence_targets,
        ) as incidence:
            losses = g3_graph_loss(
                output,
                targets,
                gt_bond_slot_identity_oracle=True,
                normalized_bond_targets=normalized,
            )

        self.assertEqual(matcher.call_count, 0)
        self.assertIs(
            bond_loss.call_args.kwargs["assignments"],
            incidence.call_args.kwargs["bond_assignments"],
        )
        self.assertEqual(output["bond_mask"].tolist(), [[True, False, False]])
        self.assertEqual(
            output["bond_query_output"]["bond_target_indices"].tolist(),
            [[0, -1, -1]],
        )
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_p0d_model_requires_paired_atom_and_bond_oracles(self) -> None:
        model = G3HierarchicalGraphPredictor(self.config)
        normalized = build_g3_bond_targets(self._targets())

        with self.assertRaisesRegex(ValueError, "require oracle Atom slots"):
            model(
                torch.randn(1, 8, 6, 8),
                atom_counts=torch.tensor([2]),
                bond_counts=torch.tensor([1]),
                oracle_bond_endpoints=normalized["endpoints"],
                oracle_bond_mask=normalized["mask"],
            )

    def test_p0d_bond_selection_is_count_prefix_not_presence_topk(self) -> None:
        model = G3HierarchicalGraphPredictor(self.config)
        targets = self._targets()
        normalized = build_g3_bond_targets(targets)

        with patch(
            "chemvector.g3_hierarchical_graph._count_conditioned_mask",
            side_effect=AssertionError("presence top-k must not run for P0-D"),
        ):
            output = model(
                torch.randn(1, 8, 6, 8),
                atom_counts=torch.tensor([2]),
                bond_counts=torch.tensor([1]),
                oracle_atom_anchors=targets["atom_anchors"],
                oracle_atom_mask=targets["atom_mask"],
                oracle_bond_endpoints=normalized["endpoints"],
                oracle_bond_mask=normalized["mask"],
            )

        self.assertEqual(output["bond_mask"].tolist(), [[True, False, False]])

    def test_p0e_uses_fixed_three_role_slots_and_requires_only_atom_oracle(
        self,
    ) -> None:
        model = G3FixedLearnedBondSlotPredictor(self.config)
        self.assertIsInstance(model.bond_decoder, FixedLearnedBondSlotDecoder)
        self.assertEqual(tuple(model.bond_decoder.slot_embedding.weight.shape), (3, 16))
        self.assertEqual(tuple(model.bond_decoder.role_embedding.weight.shape), (3, 16))
        self.assertFalse(hasattr(model, "dense_bond"))

        with self.assertRaisesRegex(ValueError, "requires GT Atom-slot oracle"):
            model(torch.randn(1, 8, 6, 8))
        with self.assertRaises(TypeError):
            model(
                torch.randn(1, 8, 6, 8),
                oracle_atom_anchors=self._targets()["atom_anchors"],
                oracle_atom_mask=self._targets()["atom_mask"],
                oracle_bond_endpoints=torch.zeros(1, 3, 2, 2),
            )

    def test_p0h_contains_only_atom_head_and_reuses_original_atom_loss(self) -> None:
        model = G3AtomOnlyPredictor(self.config)
        targets = self._targets()
        output = model(torch.randn(1, 8, 6, 8), atom_counts=torch.tensor([2]))

        self.assertEqual(set(dict(model.named_children())), {"atom_head"})
        self.assertNotIn("bond_mask", output)
        p0h = g3_atom_only_loss(output, targets)
        original = spatial_atom_slot_loss(output["atom_dense_output"], targets)
        self.assertEqual(set(p0h), set(original))
        for name in p0h:
            torch.testing.assert_close(p0h[name], original[name])
        p0h["loss"].backward()
        self.assertGreater(
            float(model.atom_head.presence.weight.grad.abs().sum()), 0.0
        )
        with self.assertRaises(TypeError):
            model(
                torch.randn(1, 8, 6, 8),
                bond_counts=torch.tensor([1]),
            )

    def test_p0i_canonical_targets_are_anchor_independent_and_reject_ties(self) -> None:
        model = G3AtomOnlyPredictor(self.config)
        dense = model.atom_head(torch.zeros(1, 8, 8, 8))
        targets = {
            "atom_anchors": torch.tensor(
                [[[0.251, 0.501], [0.252, 0.502], [0.0, 0.0], [0.0, 0.0]]]
            ),
            "atom_mask": torch.tensor([[True, True, False, False]]),
        }
        altered = dict(dense)
        altered["anchors"] = dense["anchors"].flip(1)
        with patch(
            "chemvector.g3_hierarchical_graph.linear_sum_assignment",
            side_effect=AssertionError("P0-I must not use predicted-anchor Hungarian"),
        ):
            first = build_canonical_atom_slot_targets(dense, targets)
            second = build_canonical_atom_slot_targets(altered, targets)
            output = model(torch.zeros(1, 8, 8, 8), atom_counts=torch.tensor([2]))
            losses = g3_canonical_atom_only_loss(output, targets)

        self.assertTrue(torch.equal(first["positive"], second["positive"]))
        self.assertTrue(torch.equal(first["anchors"], second["anchors"]))
        cell_y = int(float(targets["atom_anchors"][0, 0, 1]) * 8)
        cell_x = int(float(targets["atom_anchors"][0, 0, 0]) * 8)
        self.assertEqual(
            first["positive"][0, :, cell_y, cell_x].tolist(),
            [True, True, False],
        )
        torch.testing.assert_close(
            first["anchors"][0, 0, :, cell_y, cell_x],
            targets["atom_anchors"][0, 0],
        )
        torch.testing.assert_close(
            first["anchors"][0, 1, :, cell_y, cell_x],
            targets["atom_anchors"][0, 1],
        )
        self.assertTrue(torch.isfinite(losses["loss"]))

        tied = {name: value.clone() for name, value in targets.items()}
        tied["atom_anchors"][0, 1] = tied["atom_anchors"][0, 0]
        with self.assertRaisesRegex(ValueError, "identical GT coordinates"):
            build_canonical_atom_slot_targets(dense, tied)

    def test_p0e_forward_and_direct_loss_never_use_hungarian_or_topk(self) -> None:
        model = G3FixedLearnedBondSlotPredictor(self.config)
        targets = self._targets()
        normalized = build_g3_bond_targets(targets)
        with patch(
            "chemvector.g3_hierarchical_graph.linear_sum_assignment",
            side_effect=AssertionError("P0-E must not use Hungarian"),
        ), patch(
            "chemvector.g3_hierarchical_graph._count_conditioned_mask",
            side_effect=AssertionError("P0-E must not use presence top-k"),
        ):
            output = model(
                torch.randn(1, 8, 6, 8),
                atom_counts=torch.tensor([2]),
                bond_counts=torch.tensor([1]),
                oracle_atom_anchors=targets["atom_anchors"],
                oracle_atom_mask=targets["atom_mask"],
            )
            losses = g3_fixed_bond_slot_loss(
                output,
                targets,
                normalized_bond_targets=normalized,
            )

        self.assertEqual(output["bond_mask"].tolist(), [[True, False, False]])
        self.assertNotIn("bond_presence_logits", output)
        self.assertNotIn("bond_query_output", output)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        for name in (
            "bond_decoder.endpoint_head.weight",
            "bond_decoder.incidence.endpoint_projection.0.weight",
            "bond_decoder.incidence.atom_projection.0.weight",
            "atom_head.feature_projection.weight",
        ):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)

    def test_p0f_forward_uses_presence_topk_without_gt_bond_inputs(self) -> None:
        model = G3SetLearnedBondSlotPredictor(self.config)
        targets = self._targets()
        output = model(
            torch.randn(1, 8, 6, 8),
            atom_counts=torch.tensor([2]),
            bond_counts=torch.tensor([1]),
            oracle_atom_anchors=targets["atom_anchors"],
            oracle_atom_mask=targets["atom_mask"],
        )

        self.assertEqual(int(output["bond_mask"].sum()), 1)
        selected = int(output["bond_presence_logits"].argmax(dim=-1))
        self.assertTrue(bool(output["bond_mask"][0, selected]))
        self.assertIn("set_bond_slot_output", output)
        with self.assertRaises(TypeError):
            model(
                torch.randn(1, 8, 6, 8),
                oracle_atom_anchors=targets["atom_anchors"],
                oracle_atom_mask=targets["atom_mask"],
                oracle_bond_endpoints=torch.zeros(1, 3, 2, 2),
            )

    def test_p0f_uses_one_assignment_for_all_set_losses(self) -> None:
        model = G3SetLearnedBondSlotPredictor(self.config)
        targets = self._targets()
        normalized = build_g3_bond_targets(targets)
        output = model(
            torch.randn(1, 8, 6, 8),
            atom_counts=torch.tensor([2]),
            bond_counts=torch.tensor([1]),
            oracle_atom_anchors=targets["atom_anchors"],
            oracle_atom_mask=targets["atom_mask"],
        )

        with patch(
            "chemvector.g3_hierarchical_graph.g3_set_bond_slot_assignments",
            wraps=g3_set_bond_slot_assignments,
        ) as matcher:
            losses = g3_set_bond_slot_loss(
                output,
                targets,
                normalized_bond_targets=normalized,
            )

        self.assertEqual(matcher.call_count, 1)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        for name in (
            "presence_head.weight",
            "base.bond_decoder.endpoint_head.weight",
            "base.bond_decoder.incidence.endpoint_projection.0.weight",
        ):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)

    def test_p0g_enumerates_every_pair_and_accepts_no_gt_bond_input(self) -> None:
        model = G3AtomPairBondIdentityPredictor(self.config)
        targets = self._targets()
        targets["atom_anchors"][0, 2] = torch.tensor([0.98, 0.95])
        targets["atom_mask"][0, 2] = True
        output = model(
            torch.randn(1, 8, 6, 8),
            atom_counts=torch.tensor([3]),
            oracle_atom_anchors=targets["atom_anchors"],
            oracle_atom_mask=targets["atom_mask"],
        )

        self.assertEqual(
            output["flat_atom_pair_indices"].tolist(),
            [[0, 1], [0, 2], [1, 2]],
        )
        pair_targets = build_g3_atom_pair_targets(output, targets)
        self.assertEqual(pair_targets["target_bonds"], 1)
        self.assertEqual(pair_targets["covered_bonds"], 1)
        with self.assertRaises(TypeError):
            model(
                torch.randn(1, 8, 6, 8),
                atom_counts=torch.tensor([3]),
                bond_counts=torch.tensor([1]),
                oracle_atom_anchors=targets["atom_anchors"],
                oracle_atom_mask=targets["atom_mask"],
            )

    def test_p0g_loss_uses_no_hungarian_and_reaches_pair_reader(self) -> None:
        model = G3AtomPairBondIdentityPredictor(self.config)
        targets = self._targets()
        output = model(
            torch.randn(1, 8, 6, 8),
            atom_counts=torch.tensor([2]),
            oracle_atom_anchors=targets["atom_anchors"],
            oracle_atom_mask=targets["atom_mask"],
        )
        with patch(
            "chemvector.g3_hierarchical_graph.linear_sum_assignment",
            side_effect=AssertionError("P0-G must not use Hungarian"),
        ):
            losses = g3_atom_pair_bond_loss(output, targets)

        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        for name in (
            "pair_bond.ribbon_encoder.0.weight",
            "pair_bond.presence.weight",
            "pair_bond.chemical.weight",
        ):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(float(gradient.abs().sum()), 0.0, name)

    def test_bond_decoder_is_atom_permutation_equivariant_and_masks_padding(
        self,
    ) -> None:
        model = G3HierarchicalGraphPredictor(self.config).eval()
        features = torch.randn(1, 8, 6, 8)
        dense_atoms = model.atom_head(features)
        atoms = select_count_conditioned_atom_slots(
            dense_atoms, torch.tensor([3]), maximum_atoms=4
        )
        proposals = extract_g3_dense_bond_queries(
            model.dense_bond(features), maximum_queries=3
        )
        first = model.bond_decoder(
            features,
            proposals,
            atoms["atom_features"],
            atoms["atom_geometry"],
            atoms["atom_mask"],
        )
        permutation = torch.tensor([2, 3, 0, 1])
        permuted = model.bond_decoder(
            features,
            proposals,
            atoms["atom_features"][:, permutation],
            atoms["atom_geometry"][:, permutation],
            atoms["atom_mask"][:, permutation],
        )

        for name in (
            "presence_logits",
            "endpoints",
            "display_logits",
            "chemical_logits",
        ):
            torch.testing.assert_close(
                first[name], permuted[name], atol=1e-6, rtol=1e-6
            )
        torch.testing.assert_close(
            first["endpoint_pointer_logits"][..., permutation],
            permuted["endpoint_pointer_logits"],
            atol=1e-5,
            rtol=1e-5,
        )

        padded_features = atoms["atom_features"].clone()
        padded_geometry = atoms["atom_geometry"].clone()
        padded_features[:, 3] = 1000.0
        padded_geometry[:, 3] = -1000.0
        padded = model.bond_decoder(
            features,
            proposals,
            padded_features,
            padded_geometry,
            atoms["atom_mask"],
        )
        for name in (
            "presence_logits",
            "endpoints",
            "display_logits",
            "chemical_logits",
        ):
            torch.testing.assert_close(first[name], padded[name], atol=1e-6, rtol=1e-6)
        self.assertTrue((padded["endpoint_pointer_logits"][..., 3] == -1e4).all())

    def test_graph_loss_is_invariant_to_atom_order_and_unordered_bond_endpoints(
        self,
    ) -> None:
        _, output = self._forward()
        targets = self._targets()
        baseline = g3_graph_loss(output, targets)

        atom_permuted = {
            **targets,
            "atom_anchors": targets["atom_anchors"][:, [1, 0, 2, 3]],
            "atom_mask": targets["atom_mask"][:, [1, 0, 2, 3]],
            "bond_atom_refs": torch.tensor([[[1, 0], [0, 0], [0, 0]]]),
        }
        permuted = g3_graph_loss(output, atom_permuted)
        endpoint_swapped = {
            **targets,
            "bond_atom_refs": targets["bond_atom_refs"].flip(-1),
        }
        swapped = g3_graph_loss(output, endpoint_swapped)

        for name, value in baseline.items():
            torch.testing.assert_close(value, permuted[name], atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(value, swapped[name], atol=1e-6, rtol=1e-6)

        padded_targets = {
            **targets,
            "bond_atom_refs": targets["bond_atom_refs"].clone(),
        }
        padded_targets["bond_atom_refs"][:, 1:] = 999
        padded = g3_graph_loss(output, padded_targets)
        for name, value in baseline.items():
            torch.testing.assert_close(value, padded[name], atol=1e-6, rtol=1e-6)

        atom_tie_output = {
            **output,
            "atom_anchors": output["atom_anchors"].clone(),
            "bond_query_output": {
                **output["bond_query_output"],
                "endpoint_pointer_logits": torch.tensor(
                    [[[[12.0, -12.0, -1e4, -1e4], [-12.0, 12.0, -1e4, -1e4]]] * 3]
                ),
            },
        }
        atom_tie_output["atom_anchors"][:, :2] = 0.5
        atom_tied = g3_graph_loss(atom_tie_output, targets)
        atom_tied_permuted = g3_graph_loss(atom_tie_output, atom_permuted)
        for name, value in atom_tied.items():
            torch.testing.assert_close(
                value, atom_tied_permuted[name], atol=1e-6, rtol=1e-6
            )

        tied_output = {
            **output,
            "bond_query_output": {
                **output["bond_query_output"],
                "endpoints": torch.full_like(
                    output["bond_query_output"]["endpoints"], 0.5
                ),
            },
        }
        tied = g3_graph_loss(tied_output, targets)
        tied_swapped = g3_graph_loss(tied_output, endpoint_swapped)
        for name, value in tied.items():
            torch.testing.assert_close(
                value, tied_swapped[name], atol=1e-6, rtol=1e-6
            )

        asymmetric_tie_output = {
            **output,
            "bond_query_output": {
                **output["bond_query_output"],
                "endpoints": torch.tensor(
                    [[[[0.000, 0.5], [0.125, 0.5]]] * self.config.maximum_bonds]
                ),
            },
        }
        asymmetric_targets = {
            **targets,
            "atom_anchors": torch.tensor(
                [[[0.25, 0.5], [0.75, 0.5], [0.0, 0.0], [0.0, 0.0]]]
            ),
        }
        asymmetric_swapped_targets = {
            **asymmetric_targets,
            "bond_atom_refs": asymmetric_targets["bond_atom_refs"].flip(-1),
        }
        asymmetric = g3_graph_loss(asymmetric_tie_output, asymmetric_targets)
        asymmetric_swapped = g3_graph_loss(
            asymmetric_tie_output, asymmetric_swapped_targets
        )
        for name, value in asymmetric.items():
            torch.testing.assert_close(
                value, asymmetric_swapped[name], atol=1e-6, rtol=1e-6
            )

    def test_zero_bond_sample_has_finite_loss(self) -> None:
        _, output = self._forward()
        targets = self._targets()
        targets["bond_mask"] = torch.zeros_like(targets["bond_mask"])

        losses = g3_graph_loss(output, targets)

        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertEqual(float(losses["incidence_loss"].detach()), 0.0)

    def test_batched_target_gather_and_graph_loss(self) -> None:
        model = G3HierarchicalGraphPredictor(self.config)
        output = model(
            torch.randn(2, 8, 6, 8),
            atom_counts=torch.tensor([2, 3]),
            bond_counts=torch.tensor([1, 2]),
        )
        targets = {
            "atom_anchors": torch.tensor(
                [
                    [[0.20, 0.40], [0.80, 0.40], [0.0, 0.0], [0.0, 0.0]],
                    [[0.15, 0.60], [0.50, 0.30], [0.85, 0.60], [0.0, 0.0]],
                ]
            ),
            "atom_mask": torch.tensor(
                [[True, True, False, False], [True, True, True, False]]
            ),
            "bond_atom_refs": torch.tensor(
                [
                    [[0, 1], [0, 0], [0, 0]],
                    [[0, 1], [1, 2], [0, 0]],
                ]
            ),
            "bond_types": torch.tensor([[0, 0, 0], [0, 1, 0]]),
            "bond_mask": torch.tensor(
                [[True, False, False], [True, True, False]]
            ),
        }

        bond_targets = build_g3_bond_targets(targets)
        losses = g3_graph_loss(output, targets)

        self.assertEqual(tuple(bond_targets["endpoints"].shape), (2, 3, 2, 2))
        torch.testing.assert_close(
            bond_targets["endpoints"][1, 1],
            targets["atom_anchors"][1, [1, 2]],
        )
        self.assertTrue(torch.isfinite(losses["loss"]))

        bond_permutation = torch.tensor([1, 0, 2])
        permuted_targets = {
            **targets,
            "bond_atom_refs": targets["bond_atom_refs"][:, bond_permutation],
            "bond_types": targets["bond_types"][:, bond_permutation],
            "bond_mask": targets["bond_mask"][:, bond_permutation],
        }
        permuted_bond_targets = build_g3_bond_targets(permuted_targets)
        permuted_losses = g3_graph_loss(output, permuted_targets)
        for name in ("endpoints", "atom_refs", "display", "chemical", "mask"):
            torch.testing.assert_close(
                bond_targets[name], permuted_bond_targets[name]
            )
        for name, value in losses.items():
            torch.testing.assert_close(
                value, permuted_losses[name], atol=1e-6, rtol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
