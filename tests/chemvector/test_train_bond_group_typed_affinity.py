from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from chemvector.bond_group_proposal_membership import (
    HASHED_WEDGE,
    PARALLEL_2,
    PARALLEL_3,
    SINGLE_LINE,
)
from chemvector.bond_group_proposal_membership_data import (
    BondGroupProposalMembershipCollator,
)
from chemvector.bond_group_typed_affinity import (
    PAIR_HASHED,
    PAIR_NONE,
    BondGroupTypedAffinityPredictor,
    build_typed_affinity_targets,
)
from chemvector.model import ChemVectorConfig
from chemvector.primitive_set import PrimitiveSetConfig, PrimitiveSetPredictor
from chemvector.train_bond_group_typed_affinity import (
    CHECKPOINT_SCHEMA_VERSION,
    CONDITIONAL_SCOPE_NAME,
    SOURCE_PRIMITIVE_CHECKPOINT,
    SOURCE_PRIMITIVE_SHA256,
    _empty_node_totals,
    _empty_pair_totals,
    assert_frozen_flat_output,
    build_parser,
    compute_typed_affinity_loss,
    conditional_scope_contract,
    file_sha256,
    finalize_node_type_metrics,
    finalize_unordered_pair_metrics,
    load_frozen_primitive_checkpoint,
    reject_internal_test_metadata,
    reject_internal_test_shards,
    restrict_to_oracle_usable_pool,
    save_checkpoint,
    typed_affinity_config,
    update_conditional_metric_totals,
)


def _aggregate_targets(
    *,
    device: torch.device = torch.device("cpu"),
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    group_types = torch.tensor(
        [[PARALLEL_2, PARALLEL_3, HASHED_WEDGE, SINGLE_LINE, 0]],
        device=device,
    )
    membership = torch.zeros((1, 5, 8), device=device, dtype=torch.bool)
    membership[0, 0, 0:2] = True
    membership[0, 1, 2:5] = True
    membership[0, 2, 5:7] = True
    membership[0, 3, 7] = True
    proposal_mask = torch.ones((1, 8), device=device, dtype=torch.bool)
    return {
        "group_types": group_types,
        "group_mask": group_types != 0,
        "group_membership": membership,
        "proposal_keep": membership.any(dim=1),
        "group_weights": torch.tensor(
            [[1.0, 1.0, 0.8, 1.0, 0.0]],
            device=device,
        ),
    }, proposal_mask


def _perfect_output(
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    affinity_targets = build_typed_affinity_targets(targets, proposal_mask)
    batch, proposals = proposal_mask.shape
    node_logits = torch.full(
        (batch, proposals, 7),
        -8.0,
        device=proposal_mask.device,
    )
    node_logits.scatter_(
        -1,
        affinity_targets["node_group_type_labels"].unsqueeze(-1),
        8.0,
    )
    pair_logits = torch.full(
        (batch, proposals, proposals, 4),
        -8.0,
        device=proposal_mask.device,
    )
    safe_pair_labels = affinity_targets["pair_labels"].clamp_min(PAIR_NONE)
    pair_logits.scatter_(-1, safe_pair_labels.unsqueeze(-1), 8.0)
    return {
        "node_group_type_logits": node_logits,
        "pair_logits": pair_logits,
    }


class TrainBondGroupTypedAffinityTest(unittest.TestCase):
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

    def _write_primitive_checkpoint(self, path: Path, train_shard: str) -> None:
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

    def test_capacity_defaults_match_v097_contract(self) -> None:
        args = build_parser().parse_args([])
        config = typed_affinity_config(proposal_feature_dim=256)

        self.assertEqual(Path(args.primitive_checkpoint), SOURCE_PRIMITIVE_CHECKPOINT)
        self.assertEqual(args.primitive_checkpoint_sha256, SOURCE_PRIMITIVE_SHA256)
        self.assertEqual(args.image_size, 768)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(args.max_steps, 500)
        self.assertEqual(args.learning_rate, 3e-4)
        self.assertEqual(args.seed, 20260721)
        self.assertEqual(args.validate_every, 100)
        self.assertEqual(args.save_every, 100)
        self.assertEqual(args.negative_top_k, 64)
        self.assertEqual(config.proposal_descriptor_dim, 23)
        self.assertEqual(config.d_model, 192)
        self.assertEqual(config.pair_dim, 96)

    def test_oracle_pool_replaces_attention_mask_and_rejects_empty_samples(self) -> None:
        inputs = {
            "proposal_features": torch.randn(1, 3, 4),
            "proposal_mask": torch.tensor([[True, True, True]]),
        }
        targets = {"proposal_keep": torch.tensor([[True, False, True]])}
        conditional, full, usable = restrict_to_oracle_usable_pool(inputs, targets)

        self.assertTrue(torch.equal(full, torch.tensor([[True, True, True]])))
        self.assertTrue(torch.equal(usable, torch.tensor([[True, False, True]])))
        self.assertTrue(torch.equal(conditional["proposal_mask"], usable))
        self.assertTrue(torch.equal(inputs["proposal_mask"], full))

        with self.assertRaisesRegex(ValueError, "no usable proposals"):
            restrict_to_oracle_usable_pool(
                inputs,
                {"proposal_keep": torch.zeros((1, 3), dtype=torch.bool)},
            )

    def test_conditional_scope_excludes_generator_solver_and_formal_sdf(self) -> None:
        contract = conditional_scope_contract(
            BondGroupProposalMembershipCollator()
        )
        self.assertEqual(contract["name"], CONDITIONAL_SCOPE_NAME)
        self.assertTrue(contract["conditional"])
        self.assertFalse(contract["candidate_generator_in_scope"])
        self.assertFalse(contract["exact_cover_solver_in_scope"])
        self.assertFalse(contract["formal_sdf_in_scope"])
        self.assertFalse(contract["end_to_end_metric"])
        self.assertEqual(
            contract["proposal_mask"],
            "targets.proposal_keep & frozen_proposal_mask",
        )

    def test_perfect_metrics_use_unordered_pairs_and_report_all_classes(self) -> None:
        targets, proposal_mask = _aggregate_targets()
        output = _perfect_output(targets, proposal_mask)
        node_totals = _empty_node_totals()
        pair_totals = _empty_pair_totals()
        per_image = []

        update_conditional_metric_totals(
            node_totals,
            pair_totals,
            per_image,
            output,
            targets,
            proposal_mask,
            ["sample-1"],
            negative_top_k=7,
        )
        node = finalize_node_type_metrics(node_totals)
        pair = finalize_unordered_pair_metrics(pair_totals)

        self.assertEqual(node["overall_accuracy"], 1.0)
        self.assertEqual(node["macro_accuracy"], 1.0)
        self.assertEqual(pair["total"], 28)
        self.assertEqual(pair["target_positive_pairs"], 5)
        self.assertEqual(pair["target_none_pairs"], 23)
        self.assertEqual(pair["mined_hard_negative_pairs"], 7)
        self.assertEqual(pair["macro_f1"], 1.0)
        self.assertEqual(pair["by_type"]["parallel_2"]["target"], 1)
        self.assertEqual(pair["by_type"]["parallel_3"]["target"], 3)
        self.assertEqual(pair["by_type"]["hashed"]["target"], 1)
        self.assertEqual(pair["p2"]["f1"], 1.0)
        self.assertEqual(pair["p3"]["f1"], 1.0)
        self.assertEqual(pair["hashed"]["f1"], 1.0)
        self.assertEqual(per_image[0]["unordered_pairs"], 28)
        self.assertEqual(per_image[0]["sample_key"], "sample-1")

    def test_pair_metrics_report_false_positive_and_false_negative(self) -> None:
        targets, proposal_mask = _aggregate_targets()
        output = _perfect_output(targets, proposal_mask)
        output["pair_logits"][0, 0, 1] = -8.0
        output["pair_logits"][0, 0, 1, PAIR_NONE] = 8.0
        output["pair_logits"][0, 1, 0] = output["pair_logits"][0, 0, 1]
        output["pair_logits"][0, 0, 2] = -8.0
        output["pair_logits"][0, 0, 2, PAIR_HASHED] = 8.0
        output["pair_logits"][0, 2, 0] = output["pair_logits"][0, 0, 2]
        node_totals = _empty_node_totals()
        pair_totals = _empty_pair_totals()

        update_conditional_metric_totals(
            node_totals,
            pair_totals,
            [],
            output,
            targets,
            proposal_mask,
            ["sample-1"],
            negative_top_k=None,
        )
        pair = finalize_unordered_pair_metrics(pair_totals)

        self.assertEqual(pair["by_type"]["parallel_2"]["false_negative"], 1)
        self.assertEqual(pair["by_type"]["hashed"]["false_positive"], 1)
        self.assertLess(pair["macro_f1"], 1.0)

    def test_loss_wrapper_forwards_all_conditional_weights(self) -> None:
        captured = {}

        def fake_loss(output, targets, proposal_mask, **kwargs):
            captured.update(kwargs)
            return {"loss": proposal_mask.float().sum() * 0.0}

        with patch(
            "chemvector.train_bond_group_typed_affinity."
            "bond_group_typed_affinity_loss",
            fake_loss,
        ):
            compute_typed_affinity_loss(
                {},
                {},
                torch.ones((1, 2), dtype=torch.bool),
                node_type_weight=1.2,
                node_background_weight=0.0,
                pair_positive_weight=1.3,
                pair_negative_weight=0.4,
                negative_top_k=9,
            )

        self.assertEqual(captured["node_type_weight"], 1.2)
        self.assertEqual(captured["node_background_weight"], 0.0)
        self.assertEqual(captured["pair_positive_weight"], 1.3)
        self.assertEqual(captured["pair_negative_weight"], 0.4)
        self.assertEqual(captured["negative_top_k"], 9)

    def test_checkpoint_is_conditional_and_does_not_save_flat_weights(self) -> None:
        model = BondGroupTypedAffinityPredictor(
            typed_affinity_config(proposal_feature_dim=16, tiny=True)
        )
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
        scope = conditional_scope_contract(BondGroupProposalMembershipCollator())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "typed-affinity.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                global_step=7,
                samples_seen=28,
                args=args,
                validation={"loss": 1.0},
                primitive_provenance=provenance,
                scope_contract=scope,
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        self.assertEqual(
            checkpoint["checkpoint_schema_version"],
            CHECKPOINT_SCHEMA_VERSION,
        )
        self.assertEqual(checkpoint["conditional_scope"], scope)
        self.assertFalse(checkpoint["frozen_primitive_state_saved"])
        self.assertFalse(checkpoint["candidate_generator_state_saved"])
        self.assertFalse(checkpoint["exact_cover_solver_state_saved"])
        self.assertFalse(checkpoint["formal_sdf_evaluation_in_scope"])
        self.assertNotIn("primitive_model", checkpoint)
        self.assertNotIn("primitive_state", checkpoint)
        self.assertNotIn("flat_model", checkpoint)
        self.assertEqual(set(checkpoint["model"]), set(model.state_dict()))

    def test_strict_source_sha_freeze_and_internal_test_isolation_are_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "primitive.pt"
            self._write_primitive_checkpoint(path, "train.tar")
            digest = file_sha256(path)
            model, _, provenance = load_frozen_primitive_checkpoint(
                path,
                expected_sha256=digest,
                device=torch.device("cpu"),
            )
            self.assertFalse(model.training)
            self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
            self.assertEqual(provenance["checkpoint_sha256"], digest)
            assert_frozen_flat_output(model, {"features": torch.zeros(1)})
            with self.assertRaisesRegex(ValueError, "SHA256 differs"):
                load_frozen_primitive_checkpoint(
                    path,
                    expected_sha256="0" * 64,
                    device=torch.device("cpu"),
                )

            unsafe = Path(temporary) / "unsafe.pt"
            self._write_primitive_checkpoint(unsafe, "data/internal-test/train.tar")
            with self.assertRaisesRegex(ValueError, "internal_test"):
                load_frozen_primitive_checkpoint(
                    unsafe,
                    expected_sha256=file_sha256(unsafe),
                    device=torch.device("cpu"),
                )

        for value in ("internal_test", "internal-test"):
            with self.assertRaisesRegex(ValueError, "internal_test"):
                reject_internal_test_shards([f"data/{value}/shard.tar"])
        with self.assertRaisesRegex(ValueError, "internal_test"):
            reject_internal_test_metadata(
                {"source_split": "internal-test"},
                "safe",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_one_step_smoke(self) -> None:
        device = torch.device("cuda")
        config = typed_affinity_config(proposal_feature_dim=16, tiny=True)
        model = BondGroupTypedAffinityPredictor(config).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        targets, proposal_mask = _aggregate_targets(device=device)
        generator = torch.Generator(device=device).manual_seed(20260721)
        inputs = {
            "proposal_features": torch.randn(
                1, 8, 16, device=device, generator=generator
            ),
            "proposal_descriptors": torch.randn(
                1, 8, 23, device=device, generator=generator
            ),
            "flat_type_logits": torch.randn(
                1, 8, 5, device=device, generator=generator
            ),
            "proposal_scores": torch.rand(
                1, 8, device=device, generator=generator
            ),
            "proposal_mask": proposal_mask,
        }

        output = model(**inputs)
        losses = compute_typed_affinity_loss(
            output,
            targets,
            proposal_mask,
            node_type_weight=1.0,
            node_background_weight=0.0,
            pair_positive_weight=1.0,
            pair_negative_weight=0.25,
            negative_top_k=16,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        gradient_count = sum(
            parameter.grad is not None for parameter in model.parameters()
        )
        self.assertGreater(gradient_count, 0)
        self.assertTrue(torch.isfinite(losses["loss"]))
        optimizer.step()


if __name__ == "__main__":
    unittest.main()
