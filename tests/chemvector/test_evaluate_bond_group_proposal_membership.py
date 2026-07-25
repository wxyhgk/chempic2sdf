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
from chemvector.bond_group_proposal_membership import (
    SINGLE_LINE,
    BondGroupProposalMembershipConfig,
    BondGroupProposalMembershipPredictor,
)
from chemvector.bond_group_proposal_membership_pipeline import (
    bond_only_expected_v0_2_sequence,
    prepare_proposal_membership_pipeline,
)
from chemvector.evaluate_bond_group_proposal_membership import (
    DECODE_MODES,
    EVALUATION_SCHEMA_VERSION,
    evaluate_membership_output_batch,
    load_evaluation_models,
    summarize_fit_failures,
    summarize_restricted_rows,
    write_evaluation_outputs,
)
from chemvector.evaluate_visual_svg import render_bond_mask
from chemvector.model import ChemVectorConfig
from chemvector.primitive_set import LINE, PrimitiveSetConfig, PrimitiveSetPredictor
from chemvector.train_bond_group_proposal_membership import (
    CHECKPOINT_SCHEMA_VERSION,
    SEQUENCE_SCHEMA_VERSION,
    file_sha256,
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


class EvaluateBondGroupProposalMembershipTest(unittest.TestCase):
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
    def membership_config() -> BondGroupProposalMembershipConfig:
        return BondGroupProposalMembershipConfig(
            proposal_feature_dim=16,
            proposal_descriptor_dim=23,
            flat_type_classes=5,
            d_model=16,
            proposal_encoder_layers=1,
            group_decoder_layers=1,
            heads=4,
            ffn_dim=32,
            group_queries=2,
            max_members=8,
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
        membership = BondGroupProposalMembershipPredictor(self.membership_config())
        membership_path = directory / "membership.pt"
        provenance = {
            "checkpoint": str(primitive_path),
            "checkpoint_resolved": str(primitive_path.resolve()),
            "checkpoint_sha256": digest,
            "checkpoint_global_step": 11,
            "model_type": primitive.model_type,
        }
        torch.save(
            {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
                "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
                "model_type": membership.model_type,
                "model": membership.state_dict(),
                "proposal_membership_config": asdict(membership.config),
                "global_step": 7,
                "samples_seen": 28,
                "training_args": {
                    "train_shards": ["train.tar"],
                    "validation_shards": ["validation.tar"],
                },
                "source_primitive_checkpoint": str(primitive_path),
                "source_primitive_checkpoint_sha256": digest,
                "primitive_provenance": provenance,
                "frozen_primitive_state_saved": False,
                "uses_internal_test": False,
            },
            membership_path,
        )
        return membership_path, primitive_path

    def test_strict_loader_checks_membership_and_source_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            membership_path, primitive_path = self.write_checkpoints(Path(temporary))
            loaded = load_evaluation_models(
                membership_path,
                torch.device("cpu"),
            )
            self.assertEqual(loaded.membership_checkpoint["global_step"], 7)
            self.assertEqual(loaded.primitive_checkpoint["global_step"], 11)
            self.assertFalse(
                any(p.requires_grad for p in loaded.primitive_model.parameters())
            )

            payload = torch.load(
                membership_path,
                map_location="cpu",
                weights_only=False,
            )
            payload["source_primitive_checkpoint_sha256"] = "0" * 64
            torch.save(payload, membership_path)
            with self.assertRaisesRegex(ValueError, "provenance"):
                load_evaluation_models(membership_path, torch.device("cpu"))

            payload["source_primitive_checkpoint_sha256"] = file_sha256(primitive_path)
            payload["primitive_provenance"]["checkpoint_sha256"] = payload[
                "source_primitive_checkpoint_sha256"
            ]
            payload["uses_internal_test"] = True
            torch.save(payload, membership_path)
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                load_evaluation_models(membership_path, torch.device("cpu"))

    def test_both_modes_fit_frozen_geometry_and_reuse_existing_evaluator(self) -> None:
        proposals = proposal_batch()
        flat_output = {"type_logits": torch.zeros((1, 3, 5))}
        prepared = prepare_proposal_membership_pipeline(
            proposals,
            flat_output,
            torch.tensor([[0.25, 0.2], [0.5, 0.5], [0.75, 0.75]]),
            group_slots=2,
        )
        output = {
            "proposal_keep_logits": torch.tensor([[20.0, -20.0, -20.0]]),
            "proposal_type_logits": torch.tensor(
                [[[ -20.0, 20.0, -20.0, -20.0], [20.0] * 4, [20.0] * 4]]
            ),
            "group_type_logits": torch.tensor(
                [[[-20.0, 20.0, -20.0, -20.0, -20.0, -20.0, -20.0],
                  [20.0, -20.0, -20.0, -20.0, -20.0, -20.0, -20.0]]]
            ),
            "member_count_logits": torch.full((1, 2, 9), -20.0),
            "group_count_logits": torch.tensor([[-20.0, 20.0, -20.0]]),
            "membership_logits": torch.tensor([[[20.0, -20.0, -20.0], [-20.0] * 3]]),
        }
        output["member_count_logits"][0, 0, 1] = 20.0
        expected = parse_visual_svg_v0_2(bond_only_expected_v0_2_sequence(sidecar()))
        mask = render_bond_mask(flatten_visual_svg_v0_2_to_v0_1(expected))
        source_batch = {
            "keys": ["sample"],
            "primitives": [sidecar()],
            "bond_layers": [Image.fromarray((mask.astype(np.uint8) * 255), mode="L")],
        }

        evaluated = evaluate_membership_output_batch(
            output,
            prepared,
            proposals,
            source_batch,
            line_tolerance=8.0,
            text_box_iou=0.5,
            max_members=8,
            source_primitive_sha256="a" * 64,
        )

        self.assertEqual(set(evaluated), set(DECODE_MODES))
        for mode in DECODE_MODES:
            row = evaluated[mode][0]
            self.assertTrue(row["group_set_exact"])
            self.assertEqual(row["carrier_fit_failures"], [])
            self.assertTrue(row["restricted_partition"]["partition_exact"])
            self.assertEqual(row["source_primitive_checkpoint_sha256"], "a" * 64)
            self.assertFalse(row["uses_internal_test"])

    def test_outputs_record_metrics_predictions_and_fit_failures(self) -> None:
        rows = {
            mode: [
                {
                    "sample_key": "sample",
                    "decode_mode": mode,
                    "carrier_fit_failures": [],
                    "uses_internal_test": False,
                }
            ]
            for mode in DECODE_MODES
        }
        metrics = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "source_primitive_checkpoint_sha256": "a" * 64,
            "uses_internal_test": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            write_evaluation_outputs(output, metrics, rows)
            stored = json.loads((output / "evaluation_metrics.json").read_text())
            lines = (output / "predictions.jsonl").read_text().splitlines()
            self.assertEqual(stored["schema_version"], EVALUATION_SCHEMA_VERSION)
            self.assertEqual(len(lines), 2)
            self.assertEqual(
                summarize_fit_failures(rows[DECODE_MODES[0]])["objects"],
                0,
            )
            with self.assertRaisesRegex(RuntimeError, "not empty"):
                write_evaluation_outputs(output, metrics, rows)

    def test_restricted_summary_preserves_per_sample_f1_counts(self) -> None:
        rows = [
            {
                "restricted_partition": {
                    "partition_exact": True,
                    "typed_partition_exact": False,
                    "invalid_rows": 1,
                    "groups": {
                        "true_positive": 2,
                        "predicted": 3,
                        "expected": 2,
                    },
                    "typed_groups": {
                        "true_positive": 1,
                        "predicted": 3,
                        "expected": 2,
                    },
                    "pairs": {
                        "true_positive": 4,
                        "predicted": 5,
                        "expected": 4,
                    },
                }
            },
            {
                "restricted_partition": {
                    "partition_exact": False,
                    "typed_partition_exact": True,
                    "invalid_rows": 0,
                    "groups": {
                        "true_positive": 1,
                        "predicted": 1,
                        "expected": 2,
                    },
                    "typed_groups": {
                        "true_positive": 1,
                        "predicted": 1,
                        "expected": 2,
                    },
                    "pairs": {
                        "true_positive": 2,
                        "predicted": 2,
                        "expected": 3,
                    },
                }
            },
        ]

        summary = summarize_restricted_rows(rows)

        self.assertEqual(summary["samples"], 2)
        self.assertEqual(summary["partition_exact"], 0.5)
        self.assertEqual(summary["typed_partition_exact"], 0.5)
        self.assertEqual(summary["invalid_rows"], 1)
        self.assertEqual(summary["groups"]["true_positive"], 3)
        self.assertEqual(summary["groups"]["predicted"], 4)
        self.assertEqual(summary["groups"]["expected"], 4)
        self.assertEqual(summary["groups"]["f1"], 0.75)
        self.assertEqual(summary["typed_groups"]["true_positive"], 2)
        self.assertEqual(summary["pairs"]["true_positive"], 6)
        self.assertEqual(summary["pairs"]["predicted"], 7)
        self.assertEqual(summary["pairs"]["expected"], 7)


if __name__ == "__main__":
    unittest.main()
