from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from chemvector.bond_group_proposal_pointer import (
    BondGroupProposalPointerPredictor,
)
from chemvector.model import ChemVectorConfig
from chemvector.primitive_set import (
    PrimitiveSetConfig,
    PrimitiveSetPredictor,
)
from chemvector.train_bond_group_proposal_pointer import (
    SOURCE_PRIMITIVE_CHECKPOINT,
    SOURCE_PRIMITIVE_SHA256,
    _finalize_root_diagnostics,
    assert_frozen_flat_output,
    build_parser,
    compute_pointer_loss,
    file_sha256,
    freeze_primitive_model,
    load_frozen_primitive_checkpoint,
    oracle_pool_contract,
    pointer_config,
    reject_internal_test_metadata,
    reject_internal_test_shards,
    save_checkpoint,
)


class TrainBondGroupProposalPointerTest(unittest.TestCase):
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
    def _pointer_config():
        return pointer_config(
            proposal_feature_dim=16,
            tiny=True,
            max_members=8,
            max_group_count=4,
        )

    def _write_primitive_checkpoint(
        self,
        path: Path,
        *,
        train_shard: str = "train.tar",
    ) -> PrimitiveSetPredictor:
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
                    "train_shards": [train_shard],
                    "validation_shards": ["validation.tar"],
                },
                "uses_internal_test": False,
            },
            path,
        )
        return model

    def test_capacity_defaults_match_v096_500_step_gate(self) -> None:
        args = build_parser().parse_args([])
        config = pointer_config(proposal_feature_dim=256)

        self.assertEqual(Path(args.primitive_checkpoint), SOURCE_PRIMITIVE_CHECKPOINT)
        self.assertEqual(args.primitive_checkpoint_sha256, SOURCE_PRIMITIVE_SHA256)
        self.assertEqual(args.max_steps, 500)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.learning_rate, 3e-4)
        self.assertEqual(args.validate_every, 100)
        self.assertEqual(args.max_group_count, 192)
        self.assertEqual(config.max_group_count, 192)
        self.assertEqual(config.max_members, 143)
        self.assertEqual(config.proposal_descriptor_dim, 23)
        self.assertTrue(config.predict_group_count)

    def test_strict_source_load_checks_sha_eval_no_grad_and_lineage(self) -> None:
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
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                load_frozen_primitive_checkpoint(
                    path,
                    expected_sha256="0" * 64,
                    device=torch.device("cpu"),
                )

            unsafe = Path(temporary) / "unsafe.pt"
            self._write_primitive_checkpoint(
                unsafe,
                train_shard="data/internal-test/train.tar",
            )
            with self.assertRaisesRegex(ValueError, "internal_test"):
                load_frozen_primitive_checkpoint(
                    unsafe,
                    expected_sha256=file_sha256(unsafe),
                    device=torch.device("cpu"),
                )

    def test_frozen_output_guard_rejects_training_and_autograd_leaks(self) -> None:
        model = PrimitiveSetPredictor(
            self._vision_config(),
            self._primitive_config(),
        )
        freeze_primitive_model(model)
        assert_frozen_flat_output(model, {"features": torch.zeros(1)})

        model.train()
        with self.assertRaisesRegex(RuntimeError, "eval mode"):
            assert_frozen_flat_output(model, {"features": torch.zeros(1)})
        model.eval()
        with self.assertRaisesRegex(RuntimeError, "require gradients"):
            assert_frozen_flat_output(
                model,
                {"features": torch.zeros(1, requires_grad=True)},
            )

    def test_internal_test_is_rejected_in_path_key_and_metadata(self) -> None:
        for path in (
            "data/internal_test/shard.tar",
            "data/internal-test/shard.tar",
        ):
            with self.assertRaisesRegex(ValueError, "internal_test"):
                reject_internal_test_shards([path])
        with self.assertRaisesRegex(ValueError, "internal_test"):
            reject_internal_test_metadata({}, "internal-test/sample")
        with self.assertRaisesRegex(ValueError, "internal_test"):
            reject_internal_test_metadata(
                {"source_split": "internal_test"},
                "safe",
            )

    def test_loss_forwards_group_weights_and_latent_root_configuration(self) -> None:
        captured = {}
        weights = torch.tensor([[1.0, 0.25]])

        def fake_loss(
            output,
            targets,
            proposal_mask,
            *,
            group_weights,
            root_mode,
            softmin_temperature,
        ):
            captured.update(
                {
                    "weights": group_weights,
                    "root_mode": root_mode,
                    "temperature": softmin_temperature,
                }
            )
            return {"loss": proposal_mask.float().sum() * 0.0}

        with patch(
            "chemvector.train_bond_group_proposal_pointer."
            "bond_group_proposal_pointer_loss",
            fake_loss,
        ):
            result = compute_pointer_loss(
                {},
                {"group_weights": weights},
                torch.ones((1, 2), dtype=torch.bool),
                root_mode="softmin",
                softmin_temperature=0.2,
            )

        self.assertIn("loss", result)
        self.assertIs(captured["weights"], weights)
        self.assertEqual(captured["root_mode"], "softmin")
        self.assertEqual(captured["temperature"], 0.2)

    def test_checkpoint_saves_pointer_state_and_source_sha_only(self) -> None:
        model = BondGroupProposalPointerPredictor(self._pointer_config())
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
        pool = {"source": "v0.9.5_gt_associated_flat_proposals"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pointer.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                global_step=7,
                samples_seen=28,
                args=args,
                validation={"loss": 1.0},
                primitive_provenance=provenance,
                pool_contract=pool,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(checkpoint["model_type"], model.model_type)
        self.assertEqual(checkpoint["source_primitive_checkpoint_sha256"], "a" * 64)
        self.assertEqual(checkpoint["oracle_proposal_pool"], pool)
        self.assertFalse(checkpoint["frozen_primitive_state_saved"])
        self.assertNotIn("primitive_model", checkpoint)
        self.assertNotIn("primitive_state", checkpoint)
        self.assertEqual(set(checkpoint["model"]), set(model.state_dict()))

    def test_pool_contract_records_exact_v095_association_defaults(self) -> None:
        from chemvector.bond_group_proposal_membership_data import (
            BondGroupProposalMembershipCollator,
        )

        contract = oracle_pool_contract(BondGroupProposalMembershipCollator())
        self.assertEqual(contract["decode_mode"], "threshold")
        self.assertEqual(contract["proposal_threshold"], 0.5)
        self.assertEqual(contract["high_confidence_tolerance_q"], 8.0)
        self.assertEqual(contract["association_tolerance_q"], 32.0)
        self.assertEqual(contract["shell_weight"], 0.25)
        self.assertFalse(contract["unmatched_flat_false_positives_in_pool"])

    def test_root_diagnostic_rates_report_collisions_and_affinity_gap(self) -> None:
        metrics = _finalize_root_diagnostics(
            {
                "samples": 2,
                "expected_groups": 4,
                "selected_roots": 4,
                "distinct_target_groups": 3,
                "root_collisions": 1,
                "distinct_group_coverage_exact_samples": 1,
                "affinity_self_preferred_roots": 2,
                "affinity_majority_agrees_roots": 3,
                "assigned_members": 8,
                "assigned_members_same_as_root_group": 6,
            }
        )
        self.assertEqual(metrics["distinct_target_group_coverage"], 0.75)
        self.assertEqual(metrics["root_collision_rate"], 0.25)
        self.assertEqual(metrics["affinity_self_preference_rate"], 0.5)
        self.assertEqual(metrics["affinity_majority_agreement_rate"], 0.75)
        self.assertEqual(metrics["root_group_member_purity"], 0.75)


if __name__ == "__main__":
    unittest.main()
