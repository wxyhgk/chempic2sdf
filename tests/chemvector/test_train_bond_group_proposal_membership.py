from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from chemvector.bond_group_proposal_membership import (
    SINGLE_LINE,
    BondGroupProposalMembershipConfig,
    BondGroupProposalMembershipPredictor,
)
from chemvector.model import ChemVectorConfig
from chemvector.primitive_set import (
    LINE,
    PrimitiveSetConfig,
    PrimitiveSetPredictor,
)
from chemvector.train_bond_group_proposal_membership import (
    SOURCE_PRIMITIVE_CHECKPOINT,
    SOURCE_PRIMITIVE_SHA256,
    _finalize_partition_metrics,
    _finalize_coverage,
    _merge_coverage,
    _partition_counts,
    build_parser,
    compute_membership_loss,
    file_sha256,
    freeze_primitive_model,
    load_frozen_primitive_checkpoint,
    membership_config,
    oracle_keep_type_output,
    prepare_pipeline_batch,
    reject_internal_test_metadata,
    reject_internal_test_shards,
    save_checkpoint,
)


class TrainBondGroupProposalMembershipTest(unittest.TestCase):
    @staticmethod
    def _vision_config() -> ChemVectorConfig:
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
    def _primitive_config() -> PrimitiveSetConfig:
        return PrimitiveSetConfig(
            queries=8,
            d_model=16,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=32,
            memory_grid_size=4,
            dropout=0.0,
        )

    @staticmethod
    def _membership_config() -> BondGroupProposalMembershipConfig:
        return BondGroupProposalMembershipConfig(
            proposal_feature_dim=16,
            proposal_descriptor_dim=23,
            flat_type_classes=5,
            d_model=16,
            proposal_encoder_layers=1,
            group_decoder_layers=1,
            heads=4,
            ffn_dim=32,
            group_queries=3,
            max_members=8,
        )

    def _write_primitive_checkpoint(self, path: Path) -> PrimitiveSetPredictor:
        model = PrimitiveSetPredictor(
            self._vision_config(),
            self._primitive_config(),
        )
        torch.save(
            {
                "checkpoint_schema_version": "primitive-set-checkpoint-0.1",
                "sequence_schema_version": "visual-svg-0.1",
                "model_type": model.model_type,
                "model": model.state_dict(),
                "vision_config": asdict(model.vision_config),
                "primitive_set_config": asdict(model.config),
                "global_step": 11,
                "samples_seen": 44,
                "training_args": {
                    "train_shards": ["train.tar"],
                    "validation_shards": ["validation.tar"],
                },
                "uses_internal_test": False,
            },
            path,
        )
        return model

    def test_capacity_defaults_match_the_500_step_gate(self) -> None:
        args = build_parser().parse_args([])
        config = membership_config(proposal_feature_dim=256)

        self.assertEqual(Path(args.primitive_checkpoint), SOURCE_PRIMITIVE_CHECKPOINT)
        self.assertEqual(args.primitive_checkpoint_sha256, SOURCE_PRIMITIVE_SHA256)
        self.assertEqual(args.max_steps, 500)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.learning_rate, 3e-4)
        self.assertEqual(args.validate_every, 100)
        self.assertEqual(config.group_queries, 192)
        self.assertEqual(config.max_members, 143)
        self.assertEqual(config.proposal_descriptor_dim, 23)

    def test_strict_source_load_checks_sha_and_freezes_every_parameter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "primitive.pt"
            self._write_primitive_checkpoint(path)
            digest = file_sha256(path)
            model, checkpoint, provenance = load_frozen_primitive_checkpoint(
                path,
                expected_sha256=digest,
                device=torch.device("cpu"),
            )

            self.assertFalse(model.training)
            self.assertFalse(any(p.requires_grad for p in model.parameters()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertEqual(checkpoint["global_step"], 11)
            self.assertEqual(provenance["checkpoint_sha256"], digest)
            self.assertFalse(provenance["requires_grad"])
            self.assertTrue(provenance["eval_mode"])
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                load_frozen_primitive_checkpoint(
                    path,
                    expected_sha256="0" * 64,
                    device=torch.device("cpu"),
                )

    def test_freeze_helper_clears_existing_gradients(self) -> None:
        model = PrimitiveSetPredictor(
            self._vision_config(),
            self._primitive_config(),
        )
        first = next(model.parameters())
        first.grad = torch.ones_like(first)
        count = freeze_primitive_model(model)

        self.assertGreater(count, 0)
        self.assertFalse(model.training)
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_internal_test_is_rejected_by_path_key_and_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "internal_test"):
            reject_internal_test_shards(["data/internal_test/shard.tar"])
        with self.assertRaisesRegex(ValueError, "internal_test"):
            reject_internal_test_metadata({}, "internal_test/sample")
        with self.assertRaisesRegex(ValueError, "internal_test"):
            reject_internal_test_metadata(
                {"source_split": "internal_test"},
                "safe",
            )
        reject_internal_test_metadata({"source_split": "validation"}, "safe")

    def test_group_weights_are_forwarded_and_never_silently_dropped(self) -> None:
        weights = torch.tensor([[1.0, 0.25]])
        captured = {}

        def weighted_loss(output, targets, proposal_mask, *, group_weights=None):
            captured["weights"] = group_weights
            return {"loss": proposal_mask.float().sum() * 0.0}

        with patch(
            "chemvector.train_bond_group_proposal_membership."
            "bond_group_proposal_membership_loss",
            weighted_loss,
        ):
            result = compute_membership_loss(
                {},
                {"group_weights": weights},
                torch.ones((1, 2), dtype=torch.bool),
            )
        self.assertIn("loss", result)
        self.assertIs(captured["weights"], weights)

        def unweighted_loss(output, targets, proposal_mask):
            return {"loss": proposal_mask.float().sum() * 0.0}

        with patch(
            "chemvector.train_bond_group_proposal_membership."
            "bond_group_proposal_membership_loss",
            unweighted_loss,
        ):
            with self.assertRaisesRegex(RuntimeError, "silently discard"):
                compute_membership_loss(
                    {},
                    {"group_weights": weights},
                    torch.ones((1, 2), dtype=torch.bool),
                )

    def test_oracle_mode_replaces_only_keep_and_proposal_type(self) -> None:
        output = {
            "proposal_keep_logits": torch.zeros((1, 3)),
            "proposal_type_logits": torch.zeros((1, 3, 4)),
            "group_type_logits": torch.randn(1, 2, 7),
            "membership_logits": torch.randn(1, 2, 3),
        }
        targets = {
            "proposal_keep": torch.tensor([[True, False, True]]),
            "proposal_types": torch.tensor([[LINE, 0, LINE]]),
        }
        mask = torch.tensor([[True, True, False]])
        oracle = oracle_keep_type_output(output, targets, mask)

        self.assertGreater(float(oracle["proposal_keep_logits"][0, 0]), 0)
        self.assertLess(float(oracle["proposal_keep_logits"][0, 1]), 0)
        self.assertLess(float(oracle["proposal_keep_logits"][0, 2]), 0)
        self.assertEqual(
            int(oracle["proposal_type_logits"][0, 0].argmax()),
            LINE,
        )
        self.assertEqual(
            int(oracle["proposal_type_logits"][0, 2].argmax()),
            0,
        )
        self.assertIs(oracle["group_type_logits"], output["group_type_logits"])
        self.assertIs(oracle["membership_logits"], output["membership_logits"])

    def test_partition_metrics_require_exact_type_and_membership(self) -> None:
        decoded = [
            [
                {
                    "type": SINGLE_LINE,
                    "proposal_indices": [0],
                    "valid_membership": True,
                },
                {
                    "type": SINGLE_LINE,
                    "proposal_indices": [1],
                    "valid_membership": True,
                },
            ]
        ]
        targets = {
            "group_mask": torch.tensor([[True, True]]),
            "group_types": torch.tensor([[SINGLE_LINE, SINGLE_LINE]]),
            "group_membership": torch.tensor(
                [[[True, False, False], [False, True, False]]]
            ),
        }
        counts = _partition_counts(
            decoded,
            targets,
            torch.tensor([[True, True, False]]),
        )
        metrics = _finalize_partition_metrics(counts)

        self.assertEqual(metrics["group_exact"], 2)
        self.assertEqual(metrics["membership_f1"], 1.0)
        self.assertEqual(metrics["sample_exact_rate"], 1.0)
        decoded[0][1]["proposal_indices"] = [0]
        broken = _finalize_partition_metrics(
            _partition_counts(
                decoded,
                targets,
                torch.tensor([[True, True, False]]),
            )
        )
        self.assertLess(broken["membership_f1"], 1.0)
        self.assertEqual(broken["sample_exact_rate"], 0.0)

    def test_checkpoint_contains_membership_state_and_source_sha_only(self) -> None:
        model = BondGroupProposalMembershipPredictor(self._membership_config())
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        args = Namespace(
            train_shards=["train.tar"],
            validation_shards=["validation.tar"],
        )
        provenance = {
            "checkpoint": "primitive.pt",
            "checkpoint_sha256": "a" * 64,
            "frozen_parameter_count": 123,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "membership.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                global_step=7,
                samples_seen=28,
                args=args,
                validation={"loss": 1.0},
                primitive_provenance=provenance,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(checkpoint["model_type"], model.model_type)
        self.assertEqual(checkpoint["source_primitive_checkpoint_sha256"], "a" * 64)
        self.assertFalse(checkpoint["frozen_primitive_state_saved"])
        self.assertNotIn("primitive_model", checkpoint)
        self.assertNotIn("primitive_state", checkpoint)
        self.assertEqual(set(checkpoint["model"]), set(model.state_dict()))

    def test_pipeline_batch_attaches_flat_output_before_descriptor_build(self) -> None:
        primitive_model = PrimitiveSetPredictor(
            self._vision_config(),
            self._primitive_config(),
        )
        proposal_batch = {
            "proposal_features": torch.randn(1, 3, 16),
            "proposal_geometry": torch.tensor(
                [[[0.1, 0.2, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0]] * 3]
            ),
            "proposal_stroke": torch.tensor([[0.3, 0.0, 0.0]]),
            "proposal_score": torch.tensor([[0.9, 0.0, 0.0]]),
            "proposal_type": torch.tensor([[LINE, 0, 0]]),
            "proposal_query_index": torch.tensor([[0, -1, -1]]),
            "proposal_mask": torch.tensor([[True, False, False]]),
            "group_membership": torch.tensor([[7, -1, -1]]),
            "group_type": torch.tensor([[SINGLE_LINE, 0, 0]]),
            "group_count": torch.tensor([[1, 0, 0]]),
            "group_mask": torch.tensor([[True, False, False]]),
            "match_weight": torch.tensor([[1.0, 0.0, 0.0]]),
            "primitives": [
                {
                    "schema_version": "visual-primitives-0.1",
                    "primitives": [
                        {
                            "primitive_id": 0,
                            "kind": "line",
                            "quantized_points": [[205, 409], [819, 409]],
                            "stroke_width_q": 19,
                            "owner_kind": "bond",
                            "owner_id": 7,
                        }
                    ],
                    "bond_groups": [
                        {
                            "bond_id": 7,
                            "direction": "NONE",
                            "primitive_ids": [0],
                        }
                    ],
                    "text_instances": [],
                }
            ],
        }

        class FakeCollator:
            def __init__(self):
                self.called = False

            def attach_flat_output(self, batch, flat_output):
                self.called = True
                return proposal_batch

        collator = FakeCollator()
        flat_output = {"type_logits": torch.randn(1, 8, 5)}
        prepared = prepare_pipeline_batch(
            {"primitives": proposal_batch["primitives"]},
            flat_output,
            primitive_model,
            self._membership_config(),
            collator,
        )

        self.assertTrue(collator.called)
        self.assertEqual(
            prepared["model_inputs"]["proposal_descriptors"].shape,
            (1, 3, 23),
        )
        self.assertEqual(int(prepared["known_counts"][0]), 1)
        self.assertEqual(float(prepared["targets"]["group_weights"][0, 0]), 1.0)

    def test_tensor_coverage_is_accumulated_with_sum_count_and_mean(self) -> None:
        totals = {}
        _merge_coverage(totals, {"keep_recall": torch.tensor([0.5, 1.0])})
        _merge_coverage(totals, {"keep_recall": torch.tensor([0.25])})
        result = _finalize_coverage(totals)

        self.assertEqual(result["keep_recall"]["count"], 3)
        self.assertAlmostEqual(result["keep_recall"]["sum"], 1.75)
        self.assertAlmostEqual(result["keep_recall"]["mean"], 1.75 / 3.0)


if __name__ == "__main__":
    unittest.main()
