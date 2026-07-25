from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from chemvector.bond_group_set import (
    NO_OBJECT,
    SINGLE_LINE,
    BondGroupSetConfig,
    BondGroupSetPredictor,
)
from chemvector.model import ChemVectorConfig
from chemvector.primitive_set import PrimitiveSetConfig, PrimitiveSetPredictor
from chemvector.train_bond_group_set import (
    CAPACITY_VISION_SHA256,
    CHECKPOINT_SCHEMA_VERSION,
    ENDPOINT_REFINER_FREEZE_MODE,
    ENDPOINT_REFINER_PARAMETER_PREFIXES,
    FULL_MODEL_FREEZE_MODE,
    PRIMITIVE_SET_CHECKPOINT_SCHEMA_VERSION,
    PRIMITIVE_SET_SEQUENCE_SCHEMA_VERSION,
    SEQUENCE_SCHEMA_VERSION,
    atomic_torch_save,
    bond_group_config,
    build_parser,
    configure_trainable_parameters,
    extend_resume_lineage,
    is_primitive_set_warmstart_key,
    load_primitive_set_init_checkpoint,
    load_primitive_set_warmstart_modules,
    load_resume_checkpoint,
    primitive_set_warmstart_provenance,
    reject_internal_test_shards,
    save_checkpoint,
    validate,
    validate_initialization_cli_contract,
)


class StaticBondGroupModel(nn.Module):
    def __init__(self, real_output: dict, white_output: dict) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.real_output = real_output
        self.white_output = white_output

    def forward(self, pixels: torch.Tensor) -> dict[str, torch.Tensor]:
        source = self.white_output if bool((pixels == 1.0).all()) else self.real_output
        return {
            name: value.to(pixels.device) + self.anchor * 0.0
            for name, value in source.items()
        }


class TrainBondGroupSetTest(unittest.TestCase):
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
    def _primitive_config(queries: int = 7) -> PrimitiveSetConfig:
        return PrimitiveSetConfig(
            queries=queries,
            d_model=64,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=128,
            memory_grid_size=8,
            dropout=0.0,
            local_query_kernel=3,
            endpoint_refine_kernel=3,
        )

    @staticmethod
    def _group_config(queries: int = 5) -> BondGroupSetConfig:
        return BondGroupSetConfig(
            queries=queries,
            max_members=4,
            d_model=64,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=128,
            memory_grid_size=8,
            dropout=0.0,
            local_query_kernel=3,
            endpoint_refine_kernel=3,
        )

    def test_default_capacity_contract_is_frozen(self) -> None:
        args = build_parser().parse_args([])
        config = bond_group_config(
            queries=args.queries,
            max_members=args.max_members,
            local_query_kernel=args.local_query_kernel,
            endpoint_refine_kernel=args.endpoint_refine_kernel,
        )

        self.assertEqual(config.queries, 192)
        self.assertEqual(config.max_members, 32)
        self.assertEqual(config.local_query_kernel, 5)
        self.assertEqual(config.endpoint_refine_kernel, 25)
        self.assertEqual(config.d_model, 256)

    def test_tiny_mode_preserves_query_and_member_capacity(self) -> None:
        config = bond_group_config(
            tiny=True,
            queries=11,
            max_members=143,
            local_query_kernel=3,
            endpoint_refine_kernel=3,
        )

        self.assertEqual(config.queries, 11)
        self.assertEqual(config.max_members, 143)
        self.assertEqual(config.d_model, 64)

    def test_rejects_internal_test_shard_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "internal_test"):
            reject_internal_test_shards(
                ["Data/sdf/visual_svg_v0_1/pilot_10k/shards/internal_test-000000.tar"]
            )

    def test_primitive_warmstart_loads_only_semantic_shared_modules(self) -> None:
        torch.manual_seed(1)
        source = PrimitiveSetPredictor(
            self._vision_config(),
            self._primitive_config(queries=7),
        )
        torch.manual_seed(2)
        target = BondGroupSetPredictor(
            self._vision_config(),
            self._group_config(queries=5),
        )
        before_query = target.query_embedding.weight.detach().clone()
        before_references = target.reference_points.detach().clone()
        before_type = target.type_head.weight.detach().clone()
        before_text = target.text_geometry_head[0].weight.detach().clone()
        before_count = target.count_head[1].weight.detach().clone()

        checkpoint = {"model": source.state_dict()}
        loaded_keys = load_primitive_set_warmstart_modules(target, checkpoint)
        target_state = target.state_dict()
        source_state = source.state_dict()

        self.assertTrue(loaded_keys)
        self.assertTrue(all(is_primitive_set_warmstart_key(k) for k in loaded_keys))
        self.assertNotIn("query_embedding.weight", loaded_keys)
        self.assertNotIn("reference_points", loaded_keys)
        self.assertFalse(any(key.startswith("type_head.") for key in loaded_keys))
        self.assertFalse(any(key.startswith("count_head.") for key in loaded_keys))
        self.assertFalse(any(key.startswith("geometry_head.") for key in loaded_keys))
        self.assertFalse(any(key.startswith("text_geometry_head.") for key in loaded_keys))
        self.assertFalse(any(key.startswith("stroke_head.") for key in loaded_keys))
        for key in loaded_keys:
            self.assertTrue(torch.equal(target_state[key], source_state[key]), key)
        self.assertTrue(torch.equal(target.query_embedding.weight, before_query))
        self.assertTrue(torch.equal(target.reference_points, before_references))
        self.assertTrue(torch.equal(target.type_head.weight, before_type))
        self.assertTrue(torch.equal(target.text_geometry_head[0].weight, before_text))
        self.assertTrue(torch.equal(target.count_head[1].weight, before_count))

    def test_primitive_warmstart_rejects_shape_mismatch(self) -> None:
        source = PrimitiveSetPredictor(
            self._vision_config(),
            self._primitive_config(),
        )
        target = BondGroupSetPredictor(
            self._vision_config(),
            self._group_config(),
        )
        state = dict(source.state_dict())
        state["raw_ink_projection.weight"] = torch.zeros((1, 1, 1, 1))

        with self.assertRaisesRegex(ValueError, "tensor shapes differ"):
            load_primitive_set_warmstart_modules(target, {"model": state})

    def test_primitive_warmstart_freezes_source_sha_and_records_provenance(self) -> None:
        vision = self._vision_config()
        source_config = self._primitive_config(queries=7)
        target_config = self._group_config(queries=5)
        payload = {
            "checkpoint_schema_version": PRIMITIVE_SET_CHECKPOINT_SCHEMA_VERSION,
            "sequence_schema_version": PRIMITIVE_SET_SEQUENCE_SCHEMA_VERSION,
            "model_type": PrimitiveSetPredictor.model_type,
            "source_vision_checkpoint": "vision-source.pt",
            "source_vision_checkpoint_sha256": CAPACITY_VISION_SHA256,
            "model": {},
            "vision_config": asdict(vision),
            "primitive_set_config": asdict(source_config),
            "training_args": {"train_shards": [], "validation_shards": []},
            "global_step": 11000,
            "samples_seen": 44000,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "primitive.pt"
            torch.save(payload, path)

            checkpoint, loaded_vision, digest = load_primitive_set_init_checkpoint(
                path,
                target_config,
            )
            provenance = primitive_set_warmstart_provenance(
                path,
                digest,
                checkpoint,
                ["decoder.norm.weight"],
            )

            self.assertEqual(loaded_vision, vision)
            self.assertEqual(
                provenance["source_vision_checkpoint_sha256"],
                CAPACITY_VISION_SHA256,
            )
            self.assertEqual(provenance["checkpoint_sha256"], digest)
            self.assertEqual(provenance["loaded_keys"], ["decoder.norm.weight"])

            payload["source_vision_checkpoint_sha256"] = "wrong"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                load_primitive_set_init_checkpoint(path, target_config)

    def test_resume_and_primitive_initialization_are_mutually_exclusive(self) -> None:
        args = build_parser().parse_args(
            [
                "--resume-checkpoint",
                "resume.pt",
                "--init-primitive-set-checkpoint",
                "primitive.pt",
            ]
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_initialization_cli_contract(args)

    def test_endpoint_refiner_only_requires_resume(self) -> None:
        without_resume = build_parser().parse_args(["--endpoint-refiner-only"])
        primitive_init = build_parser().parse_args(
            [
                "--init-primitive-set-checkpoint",
                "primitive.pt",
                "--endpoint-refiner-only",
            ]
        )
        with_resume = build_parser().parse_args(
            ["--resume-checkpoint", "group.pt", "--endpoint-refiner-only"]
        )

        with self.assertRaisesRegex(ValueError, "requires --resume-checkpoint"):
            validate_initialization_cli_contract(without_resume)
        with self.assertRaisesRegex(ValueError, "requires --resume-checkpoint"):
            validate_initialization_cli_contract(primitive_init)
        validate_initialization_cli_contract(with_resume)

    def test_endpoint_refiner_only_freezes_every_other_parameter(self) -> None:
        model = BondGroupSetPredictor(
            self._vision_config(),
            self._group_config(),
        )

        names, count = configure_trainable_parameters(
            model,
            endpoint_refiner_only=True,
        )

        expected = [
            name
            for name, _ in model.named_parameters()
            if name.startswith(ENDPOINT_REFINER_PARAMETER_PREFIXES)
        ]
        self.assertEqual(names, expected)
        self.assertEqual(
            count,
            sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if name in expected
            ),
        )
        self.assertTrue(names)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name in expected, name)

        all_names, all_count = configure_trainable_parameters(
            model,
            endpoint_refiner_only=False,
        )
        self.assertEqual(all_names, [name for name, _ in model.named_parameters()])
        self.assertEqual(all_count, sum(p.numel() for p in model.parameters()))

    def test_resume_contract_is_strict_and_uses_fresh_lineage(self) -> None:
        args = build_parser().parse_args(
            [
                "--tiny",
                "--queries",
                "11",
                "--max-members",
                "143",
                "--local-query-kernel",
                "3",
                "--endpoint-refine-kernel",
                "3",
                "--image-size",
                "64",
                "--max-steps",
                "2000",
            ]
        )
        config = bond_group_config(
            tiny=True,
            queries=11,
            max_members=143,
            local_query_kernel=3,
            endpoint_refine_kernel=3,
        )
        payload = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
            "model_type": "bond-group-set-v0.1",
            "source_vision_checkpoint": "source.pt",
            "source_vision_checkpoint_sha256": CAPACITY_VISION_SHA256,
            "model": {},
            "vision_config": {},
            "bond_group_set_config": asdict(config),
            "training_args": {"image_size": 64},
            "global_step": 1000,
            "samples_seen": 2000,
            "uses_internal_test": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step-001000.pt"
            torch.save(payload, path)

            checkpoint, loaded_config, digest = load_resume_checkpoint(path, args)
            lineage = extend_resume_lineage(checkpoint, path, digest)

            self.assertEqual(loaded_config, config)
            self.assertEqual(lineage[-1]["global_step"], 1000)
            self.assertEqual(lineage[-1]["samples_seen"], 2000)
            mismatched = argparse.Namespace(**vars(args))
            mismatched.max_members = 32
            with self.assertRaisesRegex(ValueError, "architecture differs"):
                load_resume_checkpoint(path, mismatched)

    def test_atomic_checkpoint_has_no_visible_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"

            atomic_torch_save({"value": torch.tensor(7)}, path)

            self.assertTrue(path.is_file())
            self.assertFalse(path.with_name(path.name + ".tmp").exists())
            self.assertEqual(int(torch.load(path, weights_only=False)["value"]), 7)

    def test_checkpoint_preserves_primitive_initialization_provenance(self) -> None:
        model = BondGroupSetPredictor(
            self._vision_config(),
            self._group_config(),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        args = build_parser().parse_args(
            [
                "--tiny",
                "--queries",
                "5",
                "--max-members",
                "4",
                "--local-query-kernel",
                "3",
                "--endpoint-refine-kernel",
                "3",
            ]
        )
        provenance = {
            "mode": "primitive_set_warmstart",
            "checkpoint": "primitive.pt",
            "checkpoint_sha256": "abc123",
            "source_vision_checkpoint_sha256": CAPACITY_VISION_SHA256,
            "loaded_keys": ["decoder.norm.weight"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                global_step=1,
                samples_seen=2,
                args=args,
                validation=None,
                source_checkpoint=Path("vision-source.pt"),
                source_checkpoint_sha256=CAPACITY_VISION_SHA256,
                initialization_provenance=provenance,
                freeze_mode=ENDPOINT_REFINER_FREEZE_MODE,
                trainable_parameter_names=["endpoint_refine_mlp.2.bias"],
                trainable_parameter_count=2,
            )

            saved = torch.load(path, map_location="cpu", weights_only=False)

            self.assertEqual(saved["initialization_provenance"], provenance)
            self.assertEqual(saved["freeze_mode"], ENDPOINT_REFINER_FREEZE_MODE)
            self.assertEqual(
                saved["trainable_parameter_names"],
                ["endpoint_refine_mlp.2.bias"],
            )
            self.assertEqual(saved["trainable_parameter_count"], 2)
            self.assertNotEqual(saved["freeze_mode"], FULL_MODEL_FREEZE_MODE)
            self.assertEqual(
                saved["source_vision_checkpoint_sha256"],
                CAPACITY_VISION_SHA256,
            )

    def test_validation_reports_white_image_loss_ratio(self) -> None:
        queries = 2
        members = 1
        target_geometry = torch.zeros((1, queries, members, 8))
        target_geometry[0, 0, 0, :4] = torch.tensor((0.1, 0.2, 0.8, 0.7))
        member_mask = torch.tensor([[[True], [False]]])
        batch = {
            "pixel_values": torch.zeros((1, 3, 16, 16)),
            "types": torch.tensor([[SINGLE_LINE, NO_OBJECT]]),
            "member_counts": torch.tensor([[1, 0]]),
            "member_geometry": target_geometry,
            "member_geometry_mask": member_mask[..., None].expand(-1, -1, -1, 8),
            "member_stroke": torch.tensor([[[11.0 / 63.0], [0.0]]]),
            "member_mask": member_mask,
            "text_geometry": torch.zeros((1, queries, 4)),
            "target_mask": torch.tensor([[True, False]]),
            "object_count": torch.tensor([1]),
        }
        type_logits = torch.full((1, queries, 8), -8.0)
        type_logits[0, 0, SINGLE_LINE] = 8.0
        type_logits[0, 1, NO_OBJECT] = 8.0
        member_count_logits = torch.full((1, queries, members + 1), -8.0)
        member_count_logits[0, 0, 1] = 8.0
        member_count_logits[0, 1, 0] = 8.0
        count_logits = torch.full((1, queries + 1), -8.0)
        count_logits[0, 1] = 8.0
        real_output = {
            "type_logits": type_logits,
            "count_logits": count_logits,
            "member_count_logits": member_count_logits,
            "member_geometry": target_geometry.clone(),
            "member_stroke": batch["member_stroke"].clone(),
            "text_geometry": batch["text_geometry"].clone(),
        }
        white_output = {
            **real_output,
            "type_logits": torch.zeros_like(type_logits),
            "count_logits": torch.zeros_like(count_logits),
            "member_count_logits": torch.zeros_like(member_count_logits),
            "member_geometry": torch.zeros_like(target_geometry),
        }
        model = StaticBondGroupModel(real_output, white_output)

        metrics = validate(model, [batch], torch.device("cpu"))

        self.assertGreater(metrics["white_image_loss_ratio"], 1.0)
        self.assertGreater(metrics["white_image_loss"], metrics["loss"])


if __name__ == "__main__":
    unittest.main()
