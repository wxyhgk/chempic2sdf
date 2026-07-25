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
    flatten_visual_svg_v0_2_to_v0_1,
    parse_visual_svg_v0_2,
)
from chemvector.bond_group_proposal_membership import SINGLE_LINE
from chemvector.bond_group_proposal_membership_data import (
    BondGroupProposalMembershipCollator,
)
from chemvector.bond_group_proposal_membership_pipeline import (
    bond_only_expected_v0_2_sequence,
    prepare_proposal_membership_pipeline,
)
from chemvector.bond_group_typed_affinity import (
    PAIR_NONE,
    BondGroupTypedAffinityConfig,
    BondGroupTypedAffinityPredictor,
)
from chemvector.bond_group_typed_affinity_packing import (
    TypedAffinityPackingConfig,
)
from chemvector.evaluate_bond_group_typed_affinity import (
    DECODE_MODE,
    EVALUATION_SCHEMA_VERSION,
    FORMAL_LEARNED_AFFINITY_THRESHOLDS,
    build_decode_result,
    build_evaluation_collator,
    build_parser,
    evaluate_typed_affinity_output_batch,
    load_evaluation_models,
    summarize_candidate_counts,
    summarize_solver,
    write_evaluation_outputs,
)
from chemvector.evaluate_visual_svg import render_bond_mask
from chemvector.model import ChemVectorConfig
from chemvector.primitive_set import LINE, PrimitiveSetConfig, PrimitiveSetPredictor
from chemvector.train_bond_group_typed_affinity import (
    CHECKPOINT_SCHEMA_VERSION,
    SEQUENCE_SCHEMA_VERSION,
    conditional_scope_contract,
    file_sha256,
    restrict_to_oracle_usable_pool,
)


def sidecar() -> dict:
    return {
        "schema_version": "visual-primitives-0.1",
        "primitives": [
            {
                "primitive_id": 0,
                "canonical_index": 0,
                "kind": "line",
                "owner_kind": "bond",
                "owner_id": 7,
                "quantized_points": [[205, 409], [819, 409]],
                "stroke_width_q": 19,
            }
        ],
        "bond_groups": [
            {"bond_id": 7, "direction": "NONE", "primitive_ids": [0]}
        ],
        "text_instances": [],
    }


def proposal_batch() -> dict:
    return {
        "proposal_features": torch.randn(1, 3, 16),
        "proposal_geometry": torch.tensor(
            [[[0.1, 0.2, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0]] * 3]
        ),
        "proposal_stroke": torch.tensor([[19 / 63, 0.0, 0.0]]),
        "proposal_score": torch.tensor([[0.9, 0.0, 0.0]]),
        "proposal_type": torch.tensor([[LINE, 0, 0]]),
        "proposal_query_index": torch.tensor([[0, -1, -1]]),
        "proposal_mask": torch.tensor([[True, False, False]]),
        "proposal_count": torch.tensor([1]),
        "group_membership": torch.tensor([[7, -1, -1]]),
        "group_type": torch.tensor([[SINGLE_LINE, 0, 0]]),
        "group_count": torch.tensor([[1, 0, 0]]),
        "group_mask": torch.tensor([[True, False, False]]),
        "match_weight": torch.tensor([[1.0, 0.0, 0.0]]),
        "primitives": [sidecar()],
    }


class EvaluateBondGroupTypedAffinityTest(unittest.TestCase):
    @staticmethod
    def vision_config() -> ChemVectorConfig:
        return ChemVectorConfig(
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

    @staticmethod
    def primitive_config() -> PrimitiveSetConfig:
        return PrimitiveSetConfig(
            queries=3,
            d_model=16,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=32,
            memory_grid_size=4,
        )

    @staticmethod
    def affinity_config() -> BondGroupTypedAffinityConfig:
        return BondGroupTypedAffinityConfig(
            proposal_feature_dim=16,
            proposal_descriptor_dim=23,
            flat_type_classes=5,
            d_model=16,
            proposal_encoder_layers=1,
            heads=4,
            ffn_dim=32,
            pair_dim=8,
            pair_hidden_dim=16,
        )

    def write_checkpoints(self, directory: Path) -> tuple[Path, Path]:
        primitive = PrimitiveSetPredictor(self.vision_config(), self.primitive_config())
        primitive_path = directory / "primitive.pt"
        torch.save(
            {
                "checkpoint_schema_version": "primitive-set-checkpoint-0.1",
                "sequence_schema_version": "visual-svg-0.1",
                "model_type": primitive.model_type,
                "model": primitive.state_dict(),
                "vision_config": asdict(primitive.vision_config),
                "primitive_set_config": asdict(primitive.config),
                "global_step": 11,
                "samples_seen": 44,
                "training_args": {
                    "train_shards": ["train.tar"],
                    "validation_shards": ["validation.tar"],
                },
                "uses_internal_test": False,
            },
            primitive_path,
        )
        digest = file_sha256(primitive_path)
        affinity = BondGroupTypedAffinityPredictor(self.affinity_config())
        affinity_path = directory / "affinity.pt"
        provenance = {
            "checkpoint": str(primitive_path),
            "checkpoint_resolved": str(primitive_path.resolve()),
            "checkpoint_sha256": digest,
            "checkpoint_global_step": 11,
            "model_type": primitive.model_type,
        }
        collator = BondGroupProposalMembershipCollator(
            image_size=768,
            max_proposals=3,
        )
        torch.save(
            {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
                "model_type": affinity.model_type,
                "model": affinity.state_dict(),
                "typed_affinity_config": asdict(affinity.config),
                "global_step": 7,
                "samples_seen": 28,
                "training_args": {
                    "train_shards": ["train.tar"],
                    "validation_shards": ["validation.tar"],
                    "image_size": 768,
                    "max_group_count": 3,
                },
                "conditional_scope": conditional_scope_contract(collator),
                "source_primitive_checkpoint": str(primitive_path),
                "source_primitive_checkpoint_sha256": digest,
                "primitive_provenance": provenance,
                "frozen_primitive_state_saved": False,
                "candidate_generator_state_saved": False,
                "exact_cover_solver_state_saved": False,
                "formal_sdf_evaluation_in_scope": False,
                "uses_internal_test": False,
            },
            affinity_path,
        )
        return affinity_path, primitive_path

    def test_strict_loader_and_conditional_scope_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            affinity_path, primitive_path = self.write_checkpoints(Path(temporary))
            loaded = load_evaluation_models(affinity_path, torch.device("cpu"))
            self.assertEqual(loaded.affinity_checkpoint["global_step"], 7)
            self.assertEqual(loaded.primitive_checkpoint["global_step"], 11)
            self.assertFalse(
                any(parameter.requires_grad for parameter in loaded.primitive_model.parameters())
            )
            collator = build_evaluation_collator(
                loaded.affinity_checkpoint,
                loaded.primitive_model,
                image_size=768,
            )
            self.assertEqual(collator.max_proposals, 3)

            payload = torch.load(
                affinity_path,
                map_location="cpu",
                weights_only=False,
            )
            payload["conditional_scope"]["end_to_end_metric"] = True
            torch.save(payload, affinity_path)
            with self.assertRaisesRegex(ValueError, "conditional scope"):
                load_evaluation_models(affinity_path, torch.device("cpu"))

            payload["conditional_scope"]["end_to_end_metric"] = False
            payload["source_primitive_checkpoint_sha256"] = "0" * 64
            torch.save(payload, affinity_path)
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_evaluation_models(affinity_path, torch.device("cpu"))

            self.assertTrue(primitive_path.is_file())

    def test_exact_cover_batch_uses_known_count_and_frozen_geometry(self) -> None:
        proposals = proposal_batch()
        flat_output = {"type_logits": torch.zeros((1, 3, 5))}
        prepared = prepare_proposal_membership_pipeline(
            proposals,
            flat_output,
            torch.tensor([[0.25, 0.2], [0.5, 0.5], [0.75, 0.75]]),
            group_slots=3,
        )
        conditional_inputs, full_mask, usable_mask = restrict_to_oracle_usable_pool(
            prepared["model_inputs"],
            prepared["targets"],
        )
        prepared = {
            **prepared,
            "model_inputs": conditional_inputs,
            "full_proposal_mask": full_mask,
            "oracle_usable_mask": usable_mask,
        }

        node = torch.full((1, 3, 7), -20.0)
        node[..., 0] = 20.0
        node[0, 0] = -20.0
        node[0, 0, SINGLE_LINE] = 20.0
        pair = torch.full((1, 3, 3, 4), -20.0)
        pair[..., PAIR_NONE] = 20.0
        expected = parse_visual_svg_v0_2(
            bond_only_expected_v0_2_sequence(sidecar())
        )
        mask = render_bond_mask(flatten_visual_svg_v0_2_to_v0_1(expected))
        source_batch = {
            "keys": ["sample"],
            "primitives": [sidecar()],
            "bond_layers": [Image.fromarray((mask.astype(np.uint8) * 255), mode="L")],
        }

        rows = evaluate_typed_affinity_output_batch(
            {"node_group_type_logits": node, "pair_logits": pair},
            prepared,
            proposals,
            source_batch,
            packing_config=TypedAffinityPackingConfig(
                learned_affinity_thresholds=(0.9, 0.95),
            ),
            line_tolerance=8.0,
            text_box_iou=0.5,
            max_members=143,
            source_primitive_sha256="a" * 64,
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row["solver"]["feasible"])
        self.assertEqual(row["solver"]["status"], "optimal")
        self.assertTrue(row["group_set_exact"])
        self.assertTrue(row["restricted_partition"]["partition_exact"])
        self.assertEqual(row["carrier_fit_failures"], [])
        self.assertEqual(row["known_usable_group_count"], 1)
        self.assertEqual(row["oracle_usable_proposal_count"], 1)
        self.assertTrue(row["known_usable_group_count_given_to_decoder"])
        self.assertFalse(row["end_to_end_deployment_metric"])
        self.assertEqual(row["decode_mode"], DECODE_MODE)

        result = build_decode_result(rows)
        self.assertEqual(result["headline_metrics"]["line_recall_r8"], 1.0)
        self.assertEqual(result["headline_metrics"]["bond_iou_mean"], 1.0)
        self.assertEqual(
            result["headline_metrics"]["per_type_group_exact_recall"][
                "single_line"
            ],
            1.0,
        )
        self.assertEqual(result["solver"]["infeasible_samples"], 0)

    def test_candidate_and_solver_summaries_report_distributions(self) -> None:
        rows = [
            {
                "candidate_counts": {"geometry": 3, "union": 5},
                "matched_proposal_count": 9,
                "oracle_usable_proposal_count": 8,
                "ignored_matched_proposal_count": 1,
                "solver": {
                    "status": "optimal",
                    "feasible": True,
                    "timeout": False,
                    "infeasible": False,
                    "latency_ms": 2.0,
                },
            },
            {
                "candidate_counts": {"geometry": 7, "union": 11},
                "matched_proposal_count": 12,
                "oracle_usable_proposal_count": 10,
                "ignored_matched_proposal_count": 2,
                "solver": {
                    "status": "limit_reached",
                    "feasible": False,
                    "timeout": True,
                    "infeasible": True,
                    "latency_ms": 10.0,
                },
            },
        ]

        candidates = summarize_candidate_counts(rows)
        solver = summarize_solver(rows)
        self.assertEqual(candidates["per_field"]["union"]["p50"], 8.0)
        self.assertEqual(candidates["per_field"]["geometry"]["max"], 7.0)
        self.assertEqual(solver["latency_ms"]["p50"], 6.0)
        self.assertEqual(solver["latency_ms"]["p95"], 9.6)
        self.assertEqual(solver["timeout_samples"], 1)
        self.assertEqual(solver["infeasible_samples"], 1)

    def test_formal_threshold_defaults_and_output_files(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--checkpoint",
                "checkpoint.pt",
                "--shard",
                "validation.tar",
                "--output-dir",
                "output",
            ]
        )
        self.assertEqual(
            tuple(args.learned_affinity_thresholds),
            FORMAL_LEARNED_AFFINITY_THRESHOLDS,
        )

        with tempfile.TemporaryDirectory() as temporary:
            metrics = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "conditional_on_gt_associated_proposals": True,
                "end_to_end_deployment_metric": False,
                "packing_config": {"learned_affinity_thresholds": [0.9, 0.95]},
            }
            rows = [{"sample_key": "sample", "solver": {"status": "optimal"}}]
            write_evaluation_outputs(temporary, metrics, rows)
            loaded = json.loads(
                (Path(temporary) / "evaluation_metrics.json").read_text()
            )
            prediction = json.loads(
                (Path(temporary) / "predictions.jsonl").read_text()
            )
            self.assertFalse(loaded["end_to_end_deployment_metric"])
            self.assertEqual(prediction["sample_key"], "sample")


if __name__ == "__main__":
    unittest.main()
