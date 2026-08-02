from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn

from chemvector.g3_hierarchical_graph import (
    G3AtomOnlyPredictor,
    G3AtomPairBondIdentityPredictor,
    G3FixedLearnedBondSlotPredictor,
    G3HierarchicalGraphConfig,
    G3HierarchicalGraphPredictor,
    G3SetLearnedBondSlotPredictor,
)
from chemvector.model import ChemVectorConfig, ChemVisionEncoder
from chemvector.train_g3_hierarchical_graph import (
    ATOM_ONLY_CAPACITY_OUTPUT,
    CANONICAL_CELL_SLOT_OUTPUT,
    DEFAULT_OUTPUT,
    FIXED_LEARNED_BOND_SLOTS_OUTPUT,
    GT_ATOM_PAIR_BOND_OUTPUT,
    GT_BOND_SLOT_IDENTITY_ORACLE_OUTPUT,
    GT_ATOM_SLOT_ORACLE_OUTPUT,
    SHARED_BOND_ASSIGNMENT_OUTPUT,
    SET_LEARNED_BOND_SLOTS_OUTPUT,
    _checkpoint_payload,
    _validate_args,
    build_p0h_atom_only_model,
    build_parser,
    cache_frozen_vision_features,
    cyclic_batch_indices,
    evaluate_g3_atom_only_count_modes,
    evaluate_g3_count_modes,
    g3_atom_only_training_step,
    g3_training_step,
    load_frozen_vision_checkpoint,
    maximum_cardinality_radius_match_count,
    module_state_sha256,
    validate_atom_only_optimizer,
)


def graph_targets(batch: int = 1) -> dict[str, torch.Tensor]:
    atom_anchors = torch.tensor(
        [[[0.20, 0.50], [0.80, 0.50], [0.0, 0.0], [0.0, 0.0]]]
    ).expand(batch, -1, -1).clone()
    atom_mask = torch.tensor([[True, True, False, False]]).expand(batch, -1).clone()
    bond_refs = torch.tensor([[[0, 1], [0, 0], [0, 0]]]).expand(
        batch, -1, -1
    ).clone()
    bond_types = torch.tensor([[2, 0, 0]]).expand(batch, -1).clone()
    bond_mask = torch.tensor([[True, False, False]]).expand(batch, -1).clone()
    return {
        "atom_anchors": atom_anchors,
        "atom_mask": atom_mask,
        "bond_atom_refs": bond_refs,
        "bond_types": bond_types,
        "bond_display": bond_types.clone(),
        "bond_mask": bond_mask,
    }


class CountAwareStructuralModel(nn.Module):
    def __init__(self, *, deploy_wrong: bool) -> None:
        super().__init__()
        self.deploy_wrong = deploy_wrong

    def forward(
        self,
        features,
        *,
        atom_counts=None,
        bond_counts=None,
        oracle_atom_anchors=None,
        oracle_atom_mask=None,
    ):
        batch = len(features)
        forced = atom_counts is not None and bond_counts is not None
        atom_mask = torch.tensor(
            [[True, True, False, False]], device=features.device
        ).expand(batch, -1).clone()
        bond_mask = torch.tensor(
            [[True, False, False]], device=features.device
        ).expand(batch, -1).clone()
        if self.deploy_wrong and not forced:
            atom_mask[:, 1] = False
            bond_mask[:, 0] = False
        anchors = torch.tensor(
            [[[0.20, 0.50], [0.80, 0.50], [0.0, 0.0], [0.0, 0.0]]],
            device=features.device,
        ).expand(batch, -1, -1)
        target_indices = None
        if oracle_atom_anchors is not None:
            anchors = oracle_atom_anchors
            atom_mask = oracle_atom_mask.bool()
            target_indices = torch.arange(
                atom_mask.size(1), device=features.device
            )[None].expand(batch, -1)
            target_indices = torch.where(
                atom_mask, target_indices, torch.full_like(target_indices, -1)
            )
        pointer_logits = torch.full(
            (batch, 3, 2, 4), -1e4, device=features.device
        )
        pointer_logits[:, 0, 0, 0] = 12.0
        pointer_logits[:, 0, 1, 1] = 12.0
        bond_type_logits = torch.zeros(batch, 3, 4, device=features.device)
        bond_type_logits[:, 0, 2] = 12.0
        result = {
            "atom_mask": atom_mask,
            "atom_anchors": anchors,
            "bond_mask": bond_mask,
            "endpoint_pointer_logits": pointer_logits,
            "bond_type_logits": bond_type_logits,
        }
        if target_indices is not None:
            result["atom_target_indices"] = target_indices
        return result


class TinyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 8, kernel_size=4, stride=4)

    def forward_stroke_resolution(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.projection(pixels)


class G3TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.config = G3HierarchicalGraphConfig(
            input_channels=8,
            hidden_channels=16,
            d_model=16,
            atom_slots_per_cell=2,
            maximum_atoms=4,
            bond_slots_per_cell=2,
            maximum_bonds=3,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=32,
            memory_stride=1,
            incidence_hidden_dim=16,
            incidence_pair_hidden_dim=8,
        )

    def test_frozen_vision_checkpoint_load_and_feature_cache_on_cpu(self) -> None:
        vision_config = ChemVectorConfig(
            d_model=32,
            decoder_heads=4,
            vision_dims=(8, 16, 32, 64),
            vision_depths=(1, 1, 1, 1),
            dropout=0.0,
        )
        source = ChemVisionEncoder(vision_config)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "vision.pt"
            torch.save(
                {
                    "model_type": "primitive-set-v0.1",
                    "vision_config": asdict(vision_config),
                    "model": {
                        f"vision_encoder.{name}": value
                        for name, value in source.state_dict().items()
                    },
                },
                checkpoint,
            )

            loaded, config, _, checksum = load_frozen_vision_checkpoint(
                checkpoint, torch.device("cpu")
            )
            features = cache_frozen_vision_features(
                loaded,
                torch.zeros(2, 3, 32, 32),
                device=torch.device("cpu"),
                batch_size=1,
                precision="fp32",
            )

        self.assertEqual(config.vision_dims[0], 8)
        self.assertEqual(len(checksum), 64)
        self.assertFalse(
            any(parameter.requires_grad for parameter in loaded.parameters())
        )
        self.assertFalse(loaded.training)
        self.assertEqual(tuple(features.shape), (2, 8, 8, 8))

    def test_cpu_training_step_updates_g3_parameters(self) -> None:
        model = G3HierarchicalGraphPredictor(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.atom_head.presence.weight.detach().clone()

        losses = g3_training_step(
            model,
            optimizer,
            torch.randn(1, 8, 6, 8),
            graph_targets(),
            precision="fp32",
        )

        self.assertTrue(torch.isfinite(torch.tensor(losses["loss"])))
        self.assertGreater(losses["gradient_norm"], 0.0)
        self.assertFalse(torch.equal(before, model.atom_head.presence.weight))

    def test_cpu_training_step_supports_gt_atom_slot_oracle(self) -> None:
        model = G3HierarchicalGraphPredictor(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses = g3_training_step(
            model,
            optimizer,
            torch.randn(1, 8, 6, 8),
            graph_targets(),
            precision="fp32",
            gt_atom_slot_oracle=True,
        )

        self.assertTrue(torch.isfinite(torch.tensor(losses["loss"])))
        self.assertGreater(losses["gradient_norm"], 0.0)

    def test_cpu_training_step_supports_p0c_shared_assignment(self) -> None:
        model = G3HierarchicalGraphPredictor(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses = g3_training_step(
            model,
            optimizer,
            torch.randn(1, 8, 6, 8),
            graph_targets(),
            precision="fp32",
            gt_atom_slot_oracle=True,
            shared_confidence_aware_bond_assignment=True,
        )

        self.assertTrue(torch.isfinite(torch.tensor(losses["loss"])))
        self.assertGreater(losses["gradient_norm"], 0.0)

    def test_cpu_training_step_supports_p0d_prefix_identity(self) -> None:
        model = G3HierarchicalGraphPredictor(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        losses = g3_training_step(
            model,
            optimizer,
            torch.randn(1, 8, 6, 8),
            graph_targets(),
            precision="fp32",
            gt_atom_slot_oracle=True,
            gt_bond_slot_identity_oracle=True,
        )

        self.assertTrue(torch.isfinite(torch.tensor(losses["loss"])))
        self.assertGreater(losses["gradient_norm"], 0.0)

    def test_cpu_training_step_supports_p0e_without_hungarian(self) -> None:
        model = G3FixedLearnedBondSlotPredictor(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        with patch(
            "chemvector.g3_hierarchical_graph.linear_sum_assignment",
            side_effect=AssertionError("P0-E training must not use Hungarian"),
        ):
            losses = g3_training_step(
                model,
                optimizer,
                torch.randn(1, 8, 6, 8),
                graph_targets(),
                precision="fp32",
                gt_atom_slot_oracle=True,
                fixed_learned_bond_slots=True,
            )

        self.assertTrue(torch.isfinite(torch.tensor(losses["loss"])))
        self.assertGreater(losses["gradient_norm"], 0.0)

    def test_p0e_evaluator_runs_without_hungarian_and_records_slot_mode(self) -> None:
        model = G3FixedLearnedBondSlotPredictor(self.config)
        with patch(
            "chemvector.g3_hierarchical_graph.linear_sum_assignment",
            side_effect=AssertionError("P0-E evaluation must not use Hungarian"),
        ), patch(
            "chemvector.train_g3_hierarchical_graph.linear_sum_assignment",
            side_effect=AssertionError("P0-E evaluation must not align Atom slots"),
        ):
            metrics = evaluate_g3_count_modes(
                model,
                torch.randn(1, 8, 6, 8),
                graph_targets(),
                batch_size=1,
                precision="fp32",
                gt_atom_slot_oracle=True,
                fixed_learned_bond_slots=True,
            )

        self.assertIn("p0e", metrics["scope"])
        self.assertEqual(
            metrics["bond_slot_mode"], "fixed_learned_canonical_prefix"
        )

    def test_cpu_training_step_and_evaluator_support_p0f_set_prediction(self) -> None:
        model = G3SetLearnedBondSlotPredictor(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = g3_training_step(
            model,
            optimizer,
            torch.randn(1, 8, 6, 8),
            graph_targets(),
            precision="fp32",
            gt_atom_slot_oracle=True,
            set_learned_bond_slots=True,
        )
        metrics = evaluate_g3_count_modes(
            model,
            torch.randn(1, 8, 6, 8),
            graph_targets(),
            batch_size=1,
            precision="fp32",
            gt_atom_slot_oracle=True,
            set_learned_bond_slots=True,
        )

        self.assertTrue(torch.isfinite(torch.tensor(losses["loss"])))
        self.assertGreater(losses["gradient_norm"], 0.0)
        self.assertIn("p0f", metrics["scope"])
        self.assertEqual(
            metrics["bond_slot_mode"], "fixed_learned_set_prediction"
        )

    def test_cpu_training_step_and_evaluator_support_p0g_atom_pairs(self) -> None:
        model = G3AtomPairBondIdentityPredictor(self.config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = g3_training_step(
            model,
            optimizer,
            torch.randn(1, 8, 6, 8),
            graph_targets(),
            precision="fp32",
            gt_atom_slot_oracle=True,
            direct_gt_atom_pair_bonds=True,
        )
        metrics = evaluate_g3_count_modes(
            model,
            torch.randn(1, 8, 6, 8),
            graph_targets(),
            batch_size=1,
            precision="fp32",
            gt_atom_slot_oracle=True,
            direct_gt_atom_pair_bonds=True,
        )

        self.assertTrue(torch.isfinite(torch.tensor(losses["loss"])))
        self.assertGreater(losses["gradient_norm"], 0.0)
        self.assertIn("p0g", metrics["scope"])
        self.assertEqual(metrics["atom_slot_mode"], "gt_target_order_oracle")
        self.assertEqual(metrics["gt_count"]["atom_pair_target_coverage"], 1.0)
        self.assertEqual(metrics["bond_slot_mode"], "all_unordered_gt_atom_pairs")

    def test_p0h_initialization_matches_original_p0_atom_sha(self) -> None:
        torch.manual_seed(20260713)
        model, audit = build_p0h_atom_only_model(self.config)

        self.assertTrue(audit["equal"])
        self.assertEqual(
            audit["reference_atom_sha256"], audit["candidate_atom_sha256"]
        )
        self.assertEqual(
            audit["candidate_atom_sha256"], module_state_sha256(model.atom_head)
        )

    def test_p0h_step_eval_and_optimizer_are_bond_independent(self) -> None:
        model = G3AtomOnlyPredictor(self.config)
        optimizer = torch.optim.AdamW(model.atom_head.parameters(), lr=1e-3)
        targets = graph_targets()
        atom_targets = {
            "atom_anchors": targets["atom_anchors"],
            "atom_mask": targets["atom_mask"],
        }
        validate_atom_only_optimizer(model, optimizer)

        with patch(
            "chemvector.train_g3_hierarchical_graph.build_g3_bond_targets",
            side_effect=AssertionError("P0-H must not read Bond targets"),
        ):
            losses = g3_atom_only_training_step(
                model,
                optimizer,
                torch.randn(1, 8, 6, 8),
                atom_targets,
                precision="fp32",
            )
            metrics = evaluate_g3_atom_only_count_modes(
                model,
                torch.randn(1, 8, 6, 8),
                atom_targets,
                batch_size=1,
                precision="fp32",
            )

        self.assertTrue(torch.isfinite(torch.tensor(losses["loss"])))
        self.assertGreater(losses["gradient_norm"], 0.0)
        self.assertIn("p0h", metrics["scope"])
        self.assertFalse(metrics["training_count_teacher_forcing"])
        self.assertIn("p0h_gate", metrics)
        self.assertNotIn(
            "deploy_atom_set_exact_6px", metrics["p0h_gate"]["checks"]
        )

        extra = nn.Parameter(torch.zeros(1))
        invalid = torch.optim.AdamW(
            [*model.atom_head.parameters(), extra], lr=1e-3
        )
        with self.assertRaisesRegex(ValueError, "only Atom-head"):
            validate_atom_only_optimizer(model, invalid)

    def test_p0i_step_uses_canonical_targets_and_has_independent_scope(self) -> None:
        model = G3AtomOnlyPredictor(self.config)
        optimizer = torch.optim.AdamW(model.atom_head.parameters(), lr=1e-3)
        targets = graph_targets()
        atom_targets = {
            "atom_anchors": targets["atom_anchors"],
            "atom_mask": targets["atom_mask"],
        }
        with patch(
            "chemvector.g3_hierarchical_graph.linear_sum_assignment",
            side_effect=AssertionError("P0-I training must not use Hungarian"),
        ):
            losses = g3_atom_only_training_step(
                model,
                optimizer,
                torch.randn(1, 8, 6, 8),
                atom_targets,
                precision="fp32",
                canonical_cell_slot_targets=True,
            )
        metrics = evaluate_g3_atom_only_count_modes(
            model,
            torch.randn(1, 8, 6, 8),
            atom_targets,
            batch_size=1,
            precision="fp32",
            canonical_cell_slot_targets=True,
        )

        self.assertTrue(torch.isfinite(torch.tensor(losses["loss"])))
        self.assertGreater(losses["gradient_norm"], 0.0)
        self.assertIn("p0i", metrics["scope"])
        self.assertEqual(metrics["atom_target_mode"], "canonical_cell_slot")
        self.assertIn("p0i_gate", metrics)
        self.assertNotIn("p0h_gate", metrics)
        self.assertIn("P0-I", metrics["sdf_valid"]["reason"])

    def test_p0h_atom_matching_maximizes_radius_cardinality_first(self) -> None:
        predicted = torch.tensor([[0.10, 0.10], [0.11, 0.10]])
        expected = torch.tensor([[0.10, 0.10], [0.12, 0.10]])

        matched = maximum_cardinality_radius_match_count(
            predicted,
            expected,
            radius_pixels=1.1,
            image_size=100.0,
        )

        self.assertEqual(matched, 2)

    def test_train_only_evaluator_separates_gt_and_deploy_counts(self) -> None:
        metrics = evaluate_g3_count_modes(
            CountAwareStructuralModel(deploy_wrong=True),
            torch.zeros(2, 8, 4, 4),
            graph_targets(batch=2),
            batch_size=1,
            precision="fp32",
        )

        self.assertEqual(metrics["gt_count"]["count_pair_exact"], 1.0)
        self.assertEqual(metrics["gt_count"]["atom_set_exact_6px"], 1.0)
        self.assertEqual(
            metrics["gt_count"]["endpoint_pointer_topology_exact"], 1.0
        )
        self.assertEqual(metrics["gt_count"]["bond_type_exact"], 1.0)
        self.assertEqual(metrics["gt_count"]["structural_graph_exact"], 1.0)
        self.assertEqual(metrics["deploy_count"]["count_pair_exact"], 0.0)
        self.assertEqual(metrics["deploy_count"]["structural_graph_exact"], 0.0)
        self.assertEqual(metrics["sdf_valid"]["status"], "not_applicable")
        self.assertEqual(metrics["p0_gate"]["status"], "failed")
        self.assertNotIn("bond_assignment_mode", metrics)
        self.assertNotIn(
            "bond_assignment_query_overlap_micro", metrics["deploy_count"]
        )
        self.assertTrue(
            metrics["p0_gate"]["checks"]["gt_structural_graph_exact"] is False
            or metrics["p0_gate"]["checks"]["image_count"] is False
        )

    def test_train_only_evaluator_records_gt_atom_slot_oracle(self) -> None:
        metrics = evaluate_g3_count_modes(
            CountAwareStructuralModel(deploy_wrong=True),
            torch.zeros(2, 8, 4, 4),
            graph_targets(batch=2),
            batch_size=1,
            precision="fp32",
            gt_atom_slot_oracle=True,
        )

        self.assertEqual(metrics["atom_slot_mode"], "gt_target_order_oracle")
        self.assertIn("p0b", metrics["scope"])
        self.assertEqual(metrics["gt_count"]["atom_set_exact_6px"], 1.0)
        self.assertEqual(metrics["gt_count"]["structural_graph_exact"], 1.0)
        self.assertEqual(metrics["deploy_count"]["atom_count_exact"], 1.0)
        self.assertEqual(metrics["deploy_count"]["bond_count_exact"], 0.0)
        self.assertNotIn(
            "bond_assignment_query_overlap_micro", metrics["deploy_count"]
        )

    def test_p0c_evaluator_reports_assignment_identity_and_gate(self) -> None:
        metrics = evaluate_g3_count_modes(
            G3HierarchicalGraphPredictor(self.config),
            torch.randn(1, 8, 6, 8),
            graph_targets(),
            batch_size=1,
            precision="fp32",
            gt_atom_slot_oracle=True,
            shared_confidence_aware_bond_assignment=True,
        )

        self.assertIn("p0c", metrics["scope"])
        self.assertEqual(metrics["bond_assignment_mode"], "shared_confidence_aware")
        deploy = metrics["deploy_count"]
        self.assertEqual(deploy["bond_assignment_mode"], "shared_confidence_aware")
        self.assertEqual(deploy["bond_assignment_presence_cost_weight"], 2.0)
        self.assertEqual(deploy["bond_assignment_shared_matched_queries"], 1)
        self.assertGreaterEqual(deploy["bond_assignment_precision_micro"], 0.0)
        self.assertLessEqual(deploy["bond_assignment_precision_micro"], 1.0)
        self.assertIn(
            "deploy_shared_assignment_overlap_micro",
            metrics["p0_gate"]["checks"],
        )
        self.assertIn(
            "deploy_shared_assignment_query_set_exact",
            metrics["p0_gate"]["checks"],
        )

    def test_p0d_evaluator_reports_router_coverage_and_gate(self) -> None:
        metrics = evaluate_g3_count_modes(
            G3HierarchicalGraphPredictor(self.config),
            torch.randn(1, 8, 6, 8),
            graph_targets(),
            batch_size=1,
            precision="fp32",
            gt_atom_slot_oracle=True,
            gt_bond_slot_identity_oracle=True,
        )

        self.assertIn("p0d", metrics["scope"])
        self.assertEqual(
            metrics["bond_assignment_mode"],
            "gt_visual_endpoint_router_prefix_identity",
        )
        self.assertEqual(metrics["gt_count"]["bond_oracle_full_coverage_images"], 1)
        self.assertEqual(metrics["gt_count"]["bond_oracle_matched_targets"], 1)
        self.assertEqual(metrics["gt_count"]["bond_oracle_target_count"], 1)
        self.assertIn(
            "gt_bond_router_full_coverage", metrics["p0_gate"]["checks"]
        )
        self.assertIn(
            "gt_bond_router_target_coverage", metrics["p0_gate"]["checks"]
        )

    def test_gt_atom_slot_oracle_requires_pre_registered_output(self) -> None:
        parser = build_parser()
        invalid = parser.parse_args(["--gt-atom-slot-oracle"])
        self.assertEqual(invalid.output_dir, DEFAULT_OUTPUT)
        with self.assertRaisesRegex(ValueError, "pre-registered output"):
            _validate_args(invalid)

        valid = parser.parse_args(
            [
                "--gt-atom-slot-oracle",
                "--output-dir",
                GT_ATOM_SLOT_ORACLE_OUTPUT,
            ]
        )
        _validate_args(valid)

    def test_p0c_requires_both_flags_and_pre_registered_output(self) -> None:
        parser = build_parser()
        missing_oracle = parser.parse_args(
            [
                "--shared-confidence-aware-bond-assignment",
                "--output-dir",
                SHARED_BOND_ASSIGNMENT_OUTPUT,
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires --gt-atom-slot-oracle"):
            _validate_args(missing_oracle)

        wrong_output = parser.parse_args(
            [
                "--gt-atom-slot-oracle",
                "--shared-confidence-aware-bond-assignment",
            ]
        )
        with self.assertRaisesRegex(ValueError, "pre-registered"):
            _validate_args(wrong_output)

        valid = parser.parse_args(
            [
                "--gt-atom-slot-oracle",
                "--shared-confidence-aware-bond-assignment",
                "--output-dir",
                SHARED_BOND_ASSIGNMENT_OUTPUT,
            ]
        )
        _validate_args(valid)

    def test_p0d_requires_atom_oracle_is_mutually_exclusive_and_isolated(self) -> None:
        parser = build_parser()
        missing_atom = parser.parse_args(
            [
                "--gt-bond-slot-identity-oracle",
                "--output-dir",
                GT_BOND_SLOT_IDENTITY_ORACLE_OUTPUT,
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires --gt-atom-slot-oracle"):
            _validate_args(missing_atom)

        incompatible = parser.parse_args(
            [
                "--gt-atom-slot-oracle",
                "--shared-confidence-aware-bond-assignment",
                "--gt-bond-slot-identity-oracle",
                "--output-dir",
                GT_BOND_SLOT_IDENTITY_ORACLE_OUTPUT,
            ]
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _validate_args(incompatible)

        wrong_output = parser.parse_args(
            ["--gt-atom-slot-oracle", "--gt-bond-slot-identity-oracle"]
        )
        with self.assertRaisesRegex(ValueError, "pre-registered"):
            _validate_args(wrong_output)

        valid = parser.parse_args(
            [
                "--gt-atom-slot-oracle",
                "--gt-bond-slot-identity-oracle",
                "--output-dir",
                GT_BOND_SLOT_IDENTITY_ORACLE_OUTPUT,
            ]
        )
        _validate_args(valid)

    def test_p0e_requires_atom_oracle_is_mutually_exclusive_and_isolated(self) -> None:
        parser = build_parser()
        missing_atom = parser.parse_args(
            [
                "--fixed-learned-bond-slots",
                "--output-dir",
                FIXED_LEARNED_BOND_SLOTS_OUTPUT,
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires --gt-atom-slot-oracle"):
            _validate_args(missing_atom)

        incompatible = parser.parse_args(
            [
                "--gt-atom-slot-oracle",
                "--shared-confidence-aware-bond-assignment",
                "--fixed-learned-bond-slots",
                "--output-dir",
                FIXED_LEARNED_BOND_SLOTS_OUTPUT,
            ]
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _validate_args(incompatible)

        wrong_output = parser.parse_args(
            ["--gt-atom-slot-oracle", "--fixed-learned-bond-slots"]
        )
        with self.assertRaisesRegex(ValueError, "pre-registered"):
            _validate_args(wrong_output)

        valid = parser.parse_args(
            [
                "--gt-atom-slot-oracle",
                "--fixed-learned-bond-slots",
                "--output-dir",
                FIXED_LEARNED_BOND_SLOTS_OUTPUT,
            ]
        )
        _validate_args(valid)

    def test_p0e_default_combined_parameter_count_is_below_51m(self) -> None:
        config = G3HierarchicalGraphConfig(
            input_channels=96,
            hidden_channels=256,
            d_model=384,
            atom_slots_per_cell=3,
            maximum_atoms=128,
            bond_slots_per_cell=3,
            maximum_bonds=128,
            decoder_layers=6,
            decoder_heads=8,
            decoder_ffn_dim=1536,
            memory_stride=4,
        )
        graph_parameters = sum(
            parameter.numel()
            for parameter in G3FixedLearnedBondSlotPredictor(config).parameters()
        )
        combined_parameters = 13_658_080 + graph_parameters

        self.assertEqual(graph_parameters, 36_055_958)
        self.assertEqual(combined_parameters, 49_714_038)
        self.assertLess(combined_parameters, 51_000_000)

    def test_p0f_requires_atom_oracle_and_pre_registered_output(self) -> None:
        parser = build_parser()
        missing_atom = parser.parse_args(
            [
                "--set-learned-bond-slots",
                "--output-dir",
                SET_LEARNED_BOND_SLOTS_OUTPUT,
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires --gt-atom-slot-oracle"):
            _validate_args(missing_atom)

        wrong_output = parser.parse_args(
            ["--gt-atom-slot-oracle", "--set-learned-bond-slots"]
        )
        with self.assertRaisesRegex(ValueError, "pre-registered"):
            _validate_args(wrong_output)

        valid = parser.parse_args(
            [
                "--gt-atom-slot-oracle",
                "--set-learned-bond-slots",
                "--output-dir",
                SET_LEARNED_BOND_SLOTS_OUTPUT,
            ]
        )
        _validate_args(valid)

    def test_p0g_requires_atom_oracle_and_pre_registered_output(self) -> None:
        parser = build_parser()
        missing_atom = parser.parse_args(
            [
                "--direct-gt-atom-pair-bonds",
                "--output-dir",
                GT_ATOM_PAIR_BOND_OUTPUT,
            ]
        )
        with self.assertRaisesRegex(ValueError, "require --gt-atom-slot-oracle"):
            _validate_args(missing_atom)

        wrong_output = parser.parse_args(
            ["--gt-atom-slot-oracle", "--direct-gt-atom-pair-bonds"]
        )
        with self.assertRaisesRegex(ValueError, "pre-registered"):
            _validate_args(wrong_output)

        valid = parser.parse_args(
            [
                "--gt-atom-slot-oracle",
                "--direct-gt-atom-pair-bonds",
                "--output-dir",
                GT_ATOM_PAIR_BOND_OUTPUT,
            ]
        )
        _validate_args(valid)

    def test_p0h_cli_scope_and_checkpoint_are_isolated(self) -> None:
        parser = build_parser()
        wrong_output = parser.parse_args(["--atom-only-capacity"])
        with self.assertRaisesRegex(ValueError, "pre-registered"):
            _validate_args(wrong_output)

        incompatible = parser.parse_args(
            [
                "--atom-only-capacity",
                "--gt-atom-slot-oracle",
                "--output-dir",
                ATOM_ONLY_CAPACITY_OUTPUT,
            ]
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _validate_args(incompatible)

        args = parser.parse_args(
            [
                "--atom-only-capacity",
                "--output-dir",
                ATOM_ONLY_CAPACITY_OUTPUT,
            ]
        )
        _validate_args(args)
        model = G3AtomOnlyPredictor(self.config)
        optimizer = torch.optim.AdamW(model.atom_head.parameters(), lr=1e-3)
        initialization = {
            "reference_atom_sha256": "a" * 64,
            "candidate_atom_sha256": "a" * 64,
            "equal": True,
        }
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=self.config,
            vision_config=ChemVectorConfig(
                d_model=32,
                decoder_heads=4,
                vision_dims=(8, 16, 32, 64),
                vision_depths=(1, 1, 1, 1),
            ),
            args=args,
            audit={
                "vision": {"path": "vision.pt", "sha256": "b" * 64},
                "sample_manifest": {"sha256": "c" * 64},
                "atom_initialization": initialization,
            },
            step=1,
            last_losses={"loss": 1.0},
            train_only_metrics={"scope": "p0h"},
        )

        scope = payload["structural_scope"]
        self.assertTrue(scope["atom_only_capacity"])
        self.assertFalse(scope["bond_refs_and_types"])
        self.assertFalse(scope["count_teacher_forcing"])
        self.assertEqual(payload["atom_initialization"], initialization)

    def test_p0i_cli_requires_atom_only_and_isolated_output(self) -> None:
        parser = build_parser()
        missing_atom_only = parser.parse_args(
            [
                "--canonical-cell-slot-targets",
                "--output-dir",
                CANONICAL_CELL_SLOT_OUTPUT,
            ]
        )
        with self.assertRaisesRegex(ValueError, "require --atom-only-capacity"):
            _validate_args(missing_atom_only)

        wrong_output = parser.parse_args(
            ["--atom-only-capacity", "--canonical-cell-slot-targets"]
        )
        with self.assertRaisesRegex(ValueError, "pre-registered"):
            _validate_args(wrong_output)

        valid = parser.parse_args(
            [
                "--atom-only-capacity",
                "--canonical-cell-slot-targets",
                "--output-dir",
                CANONICAL_CELL_SLOT_OUTPUT,
            ]
        )
        _validate_args(valid)

        missing_mode = parser.parse_args(
            ["--atom-only-capacity", "--output-dir", CANONICAL_CELL_SLOT_OUTPUT]
        )
        with self.assertRaisesRegex(ValueError, "pre-registered"):
            _validate_args(missing_mode)

    def test_cyclic_batches_preserve_manifest_order_without_shuffle(self) -> None:
        first = cyclic_batch_indices(
            1, sample_count=5, batch_size=3, device=torch.device("cpu")
        )
        second = cyclic_batch_indices(
            2, sample_count=5, batch_size=3, device=torch.device("cpu")
        )

        self.assertEqual(first.tolist(), [0, 1, 2])
        self.assertEqual(second.tolist(), [3, 4, 0])


if __name__ == "__main__":
    unittest.main()
