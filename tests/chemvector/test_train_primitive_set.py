"""Focused tests for Primitive Set training and checkpoint contracts."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from chemvector.model import ChemVectorConfig
from chemvector.primitive_set import (
    LINE,
    PrimitiveSetConfig,
    PrimitiveSetPredictor,
    primitive_set_loss,
)
from chemvector.train_primitive_set import (
    CAPACITY_VISION_SHA256,
    CHECKPOINT_SCHEMA_VERSION,
    COARSE_GEOMETRY_FREEZE_MODE,
    COARSE_GEOMETRY_PARAMETER_PREFIXES,
    ENDPOINT_REFINER_FREEZE_MODE,
    ENDPOINT_REFINER_PARAMETER_PREFIXES,
    FRESH_OPTIMIZER_MODE,
    FULL_MODEL_FREEZE_MODE,
    OUTPUT_HEAD_PARAMETER_PREFIXES,
    OUTPUT_HEADS_FREEZE_MODE,
    build_parser,
    configure_trainable_parameters,
    extend_resume_lineage,
    load_capacity_vision_source,
    load_resume_checkpoint,
    loss_kwargs_from_args,
    prepare_output_directory,
    primitive_config,
    reject_internal_test_metadata,
    reject_internal_test_shards,
    reject_train_validation_overlap,
    save_checkpoint,
    validate,
    validate_capacity_checkpoint,
)


class StaticPrimitiveModel(nn.Module):
    model_type = "primitive-set-test"

    def __init__(
        self,
        output: dict[str, torch.Tensor],
        *,
        white_output: dict[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.output = output
        self.white_output = white_output or output

    def forward(self, pixels: torch.Tensor) -> dict[str, torch.Tensor]:
        source = self.white_output if bool((pixels == 1.0).all()) else self.output
        return {
            name: value.to(pixels.device) + self.anchor * 0.0
            for name, value in source.items()
        }


class PrimitiveCheckpointModel(nn.Module):
    model_type = "primitive-set-v0.1"

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.vision_config = ChemVectorConfig(
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            vision_dims=(8, 16, 32, 32),
            vision_depths=(1, 1, 1, 1),
        )
        self.config = PrimitiveSetConfig(
            queries=8,
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            memory_grid_size=4,
            local_query_kernel=5,
            endpoint_refine_kernel=25,
            line_center_vector=True,
        )


class TrainPrimitiveSetTests(unittest.TestCase):
    def test_default_primitive_configuration_is_frozen(self) -> None:
        config = primitive_config()

        self.assertEqual(config.queries, 288)
        self.assertEqual(config.d_model, 256)
        self.assertEqual(config.decoder_layers, 4)
        self.assertEqual(config.decoder_heads, 8)
        self.assertEqual(config.decoder_ffn_dim, 1024)
        self.assertEqual(config.memory_grid_size, 48)
        self.assertEqual(config.local_query_kernel, 0)
        self.assertEqual(config.endpoint_refine_kernel, 0)
        self.assertFalse(config.line_center_vector)
        defaults = build_parser().parse_args([])
        self.assertFalse(defaults.tiny)
        self.assertEqual(defaults.local_query_kernel, 0)
        self.assertEqual(defaults.endpoint_refine_kernel, 0)
        self.assertFalse(defaults.line_center_vector)
        self.assertFalse(defaults.line_geometry_objective)
        self.assertEqual(defaults.class_cost_weight, 2.0)
        self.assertIsNone(defaults.resume_checkpoint)
        self.assertFalse(defaults.endpoint_refiner_only)
        self.assertFalse(defaults.output_heads_only)
        self.assertFalse(defaults.coarse_geometry_only)

    def test_line_geometry_cli_builds_one_shared_loss_protocol(self) -> None:
        args = build_parser().parse_args(
            ["--line-geometry-objective", "--class-cost-weight", "0.5"]
        )

        self.assertEqual(
            loss_kwargs_from_args(args),
            {
                "line_geometry_objective": True,
                "class_cost_weight": 0.5,
            },
        )

    def test_local_query_kernel_is_reproducible_in_config_and_cli(self) -> None:
        args = build_parser().parse_args(["--local-query-kernel", "5"])
        config = primitive_config(local_query_kernel=args.local_query_kernel)

        self.assertEqual(config.local_query_kernel, 5)
        with self.assertRaisesRegex(ValueError, "zero or odd"):
            primitive_config(local_query_kernel=2)

    def test_endpoint_refine_kernel_is_reproducible_in_config_and_cli(self) -> None:
        args = build_parser().parse_args(["--endpoint-refine-kernel", "25"])
        config = primitive_config(
            endpoint_refine_kernel=args.endpoint_refine_kernel
        )

        self.assertEqual(config.endpoint_refine_kernel, 25)
        with self.assertRaisesRegex(ValueError, "zero or odd"):
            primitive_config(endpoint_refine_kernel=24)

    def test_line_center_vector_is_reproducible_in_config_and_cli(self) -> None:
        args = build_parser().parse_args(["--line-center-vector"])
        config = primitive_config(line_center_vector=args.line_center_vector)

        self.assertTrue(config.line_center_vector)

    def test_rejects_sealed_internal_test_shards(self) -> None:
        with self.assertRaisesRegex(ValueError, "internal_test"):
            reject_internal_test_shards(["internal_test/shards/part-000000.tar"])
        with self.assertRaisesRegex(ValueError, "internal_test"):
            reject_internal_test_shards(["internal-test/shards/part-000000.tar"])

    def test_rejects_internal_test_key_and_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "internal_test"):
            reject_internal_test_metadata({}, "internal-test-000001")
        with self.assertRaisesRegex(ValueError, "internal_test"):
            reject_internal_test_metadata({"split": "INTERNAL_TEST"}, "safe")

    def test_train_and_validation_shards_must_be_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "shared.tar"
            shard.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "overlap"):
                reject_train_validation_overlap([str(shard)], [str(shard)])

        defaults = build_parser().parse_args([])
        self.assertFalse(defaults.allow_train_validation_overlap)

    def test_output_directory_must_be_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output"
            prepare_output_directory(path)
            (path / "occupied").write_text("x", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "not empty"):
                prepare_output_directory(path)

    def test_capacity_checkpoint_metadata_is_strict(self) -> None:
        checkpoint = {
            "checkpoint_schema_version": "visual-svg-checkpoint-0.2",
            "sequence_schema_version": "visual-svg-0.1",
            "model_type": "visual-prefix-v0.1",
            "visual_grid_size": 12,
            "ink_subgrid": 8,
            "config": {},
            "model": {},
        }
        validate_capacity_checkpoint(checkpoint)
        checkpoint["model_type"] = "other"
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            validate_capacity_checkpoint(checkpoint)

    def test_capacity_checkpoint_sha_is_checked_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(b"not a checkpoint")

            with self.assertRaisesRegex(ValueError, "restricted"):
                load_capacity_vision_source(
                    path,
                    expected_sha256=CAPACITY_VISION_SHA256,
                )

    def test_validation_reports_matched_types_and_count_mae(self) -> None:
        logits = torch.full((1, 3, 5), -6.0)
        logits[0, 0, LINE] = 6.0
        logits[0, 1:, 0] = 6.0
        geometry = torch.zeros((1, 3, 8))
        geometry[0, 0, :4] = torch.tensor([0.1, 0.2, 0.8, 0.7])
        output = {
            "type_logits": logits,
            "geometry": geometry,
            "text_geometry": geometry.clone(),
            "stroke": torch.tensor([[15.0 / 63.0, 0.0, 0.0]]),
            "count_logits": torch.tensor([[-6.0, 6.0, -6.0, -6.0]]),
        }
        white_logits = torch.full((1, 3, 5), -6.0)
        white_logits[..., 0] = 6.0
        white_output = {
            **output,
            "type_logits": white_logits,
            "count_logits": torch.tensor([[6.0, -6.0, -6.0, -6.0]]),
        }
        model = StaticPrimitiveModel(output, white_output=white_output)
        batch = {
            "pixel_values": torch.zeros((1, 3, 8, 8)),
            "types": torch.tensor([[LINE, 0, 0]]),
            "geometry": geometry.clone(),
            "geometry_mask": torch.tensor(
                [[[True, True, True, True, False, False, False, False]] * 3]
            ),
            "stroke": torch.tensor([[15.0 / 63.0, 0.0, 0.0]]),
            "has_stroke": torch.tensor([[True, False, False]]),
            "target_mask": torch.tensor([[True, False, False]]),
        }

        metrics = validate(model, [batch], torch.device("cpu"), threshold=0.5)

        self.assertEqual(metrics["samples"], 1)
        self.assertEqual(metrics["matched_type_accuracy"], 1.0)
        self.assertEqual(metrics["matched_type_total"], 1)
        self.assertEqual(metrics["count_mae"], 0.0)
        self.assertEqual(metrics["threshold_count_exact_rate"], 1.0)
        self.assertEqual(metrics["threshold_count_mae"], 0.0)
        self.assertEqual(metrics["count_head_exact_rate"], 1.0)
        self.assertEqual(metrics["count_head_mae"], 0.0)
        self.assertEqual(metrics["count_head_topk_count_mean"], 1.0)
        self.assertGreater(metrics["white_image_loss_ratio"], 1.0)
        self.assertAlmostEqual(metrics["target_count_mean"], 1.0)
        for name in (
            "loss",
            "classification_loss",
            "coordinate_loss",
            "shape_loss",
            "line_endpoint_loss",
            "line_center_loss",
            "line_length_loss",
            "line_orientation_loss",
            "stroke_loss",
            "cardinality_loss",
            "count_classification_loss",
        ):
            self.assertTrue(torch.isfinite(torch.tensor(metrics[name])))

    def test_validation_uses_identical_line_loss_kwargs_for_real_and_white(self) -> None:
        logits = torch.tensor(
            [[[-6.0, 6.0, -6.0, -6.0, -6.0], [6.0, -6.0, -6.0, -6.0, -6.0]]]
        )
        geometry = torch.zeros((1, 2, 8))
        geometry[0, 0, :4] = torch.tensor([0.1, 0.2, 0.8, 0.7])
        output = {
            "type_logits": logits,
            "geometry": geometry,
            "text_geometry": geometry.clone(),
            "stroke": torch.zeros(1, 2),
            "count_logits": torch.tensor([[-6.0, 6.0, -6.0]]),
        }
        model = StaticPrimitiveModel(output)
        batch = {
            "pixel_values": torch.zeros((1, 3, 8, 8)),
            "types": torch.tensor([[LINE, 0]]),
            "geometry": geometry.clone(),
            "geometry_mask": torch.tensor(
                [[[True, True, True, True, False, False, False, False]] * 2]
            ),
            "stroke": torch.zeros(1, 2),
            "has_stroke": torch.tensor([[False, False]]),
            "target_mask": torch.tensor([[True, False]]),
        }
        kwargs = {
            "line_geometry_objective": True,
            "class_cost_weight": 0.5,
        }
        calls = []

        def capture(output, targets, **received):
            calls.append(received)
            return primitive_set_loss(output, targets, **received)

        with patch(
            "chemvector.train_primitive_set.primitive_set_loss",
            side_effect=capture,
        ):
            metrics = validate(
                model,
                [batch],
                torch.device("cpu"),
                threshold=0.5,
                loss_kwargs=kwargs,
            )

        self.assertEqual(calls, [kwargs, kwargs])
        for name in (
            "line_endpoint_loss",
            "line_center_loss",
            "line_length_loss",
            "line_orientation_loss",
        ):
            self.assertIn(name, metrics)

    def test_checkpoint_contains_required_contract_and_is_atomic(self) -> None:
        model = PrimitiveCheckpointModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        args = argparse.Namespace(max_steps=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            save_checkpoint(
                path,
                model,
                optimizer,
                global_step=1,
                samples_seen=2,
                args=args,
                validation={"loss": 1.0},
                source_checkpoint=Path("capacity.pt"),
                source_checkpoint_sha256="abc123",
            )

            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(
                payload["checkpoint_schema_version"], CHECKPOINT_SCHEMA_VERSION
            )
            self.assertEqual(payload["model_type"], model.model_type)
            self.assertIn("model", payload)
            self.assertIn("vision_config", payload)
            self.assertIn("primitive_set_config", payload)
            self.assertEqual(
                payload["primitive_set_config"]["local_query_kernel"],
                5,
            )
            self.assertEqual(
                payload["primitive_set_config"]["endpoint_refine_kernel"],
                25,
            )
            self.assertTrue(
                payload["primitive_set_config"]["line_center_vector"]
            )
            self.assertEqual(payload["source_vision_checkpoint_sha256"], "abc123")
            self.assertEqual(payload["resume_lineage"], [])
            self.assertEqual(payload["freeze_mode"], FULL_MODEL_FREEZE_MODE)
            self.assertEqual(payload["optimizer_mode"], FRESH_OPTIMIZER_MODE)
            self.assertEqual(payload["trainable_parameter_names"], ["weight"])
            self.assertEqual(payload["trainable_parameter_count"], 1)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def _resume_args(self, *extra: str) -> argparse.Namespace:
        return build_parser().parse_args(
            [
                "--tiny",
                "--local-query-kernel",
                "3",
                "--endpoint-refine-kernel",
                "3",
                "--line-geometry-objective",
                "--class-cost-weight",
                "0.5",
                *extra,
            ]
        )

    def _resume_model(self) -> PrimitiveSetPredictor:
        vision_config = ChemVectorConfig(
            d_model=32,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=64,
            vision_dims=(8, 16, 32, 32),
            vision_depths=(1, 1, 1, 1),
        )
        return PrimitiveSetPredictor(
            vision_config,
            primitive_config(
                tiny=True,
                local_query_kernel=3,
                endpoint_refine_kernel=3,
            ),
        )

    def _write_resume_checkpoint(
        self,
        path: Path,
        model: PrimitiveSetPredictor,
        args: argparse.Namespace,
        *,
        global_step: int = 5,
        samples_seen: int = 20,
    ) -> None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        save_checkpoint(
            path,
            model,
            optimizer,
            global_step=global_step,
            samples_seen=samples_seen,
            args=args,
            validation={"loss": 0.5},
            source_checkpoint=Path("capacity.pt"),
            source_checkpoint_sha256=CAPACITY_VISION_SHA256,
        )

    def test_resume_strictly_rejects_architecture_and_loss_mismatch(self) -> None:
        model = self._resume_model()
        args = self._resume_args()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            self._write_resume_checkpoint(path, model, args)

            loaded, _, config, _ = load_resume_checkpoint(path, args)
            self.assertEqual(config, model.config)
            self.assertEqual(loaded["global_step"], 5)

            architecture_mismatch = self._resume_args(
                "--local-query-kernel",
                "5",
            )
            with self.assertRaisesRegex(ValueError, "architecture differs"):
                load_resume_checkpoint(path, architecture_mismatch)

            loss_mismatch = build_parser().parse_args(
                [
                    "--tiny",
                    "--local-query-kernel",
                    "3",
                    "--endpoint-refine-kernel",
                    "3",
                ]
            )
            with self.assertRaisesRegex(ValueError, "loss protocol differs"):
                load_resume_checkpoint(path, loss_mismatch)

    def test_endpoint_refiner_only_selects_exact_parameter_set(self) -> None:
        model = self._resume_model()

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
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name in expected, name)

        no_refiner = PrimitiveSetPredictor(
            model.vision_config,
            primitive_config(tiny=True),
        )
        with self.assertRaisesRegex(ValueError, "requires a checkpoint"):
            configure_trainable_parameters(
                no_refiner,
                endpoint_refiner_only=True,
            )

    def test_output_heads_only_selects_exact_parameter_set(self) -> None:
        model = self._resume_model()

        names, count = configure_trainable_parameters(
            model,
            endpoint_refiner_only=False,
            output_heads_only=True,
        )

        expected = [
            name
            for name, _ in model.named_parameters()
            if name.startswith(OUTPUT_HEAD_PARAMETER_PREFIXES)
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
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name in expected, name)

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            configure_trainable_parameters(
                model,
                endpoint_refiner_only=True,
                output_heads_only=True,
            )
        self.assertEqual(OUTPUT_HEADS_FREEZE_MODE, "output_heads_only")

    def test_coarse_geometry_only_selects_exact_parameter_set(self) -> None:
        model = self._resume_model()

        names, count = configure_trainable_parameters(
            model,
            endpoint_refiner_only=False,
            coarse_geometry_only=True,
        )

        expected = [
            name
            for name, _ in model.named_parameters()
            if name.startswith(COARSE_GEOMETRY_PARAMETER_PREFIXES)
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
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name in expected, name)
        self.assertEqual(COARSE_GEOMETRY_FREEZE_MODE, "coarse_geometry_only")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for resume test")
    def test_cuda_resume_runs_one_fresh_optimizer_refiner_step(self) -> None:
        args = self._resume_args(
            "--endpoint-refiner-only",
            "--max-steps",
            "6",
        )
        source_model = self._resume_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            self._write_resume_checkpoint(path, source_model, args)
            checkpoint, vision_config, set_config, digest = load_resume_checkpoint(
                path,
                args,
            )
            model = PrimitiveSetPredictor(vision_config, set_config).cuda()
            model.load_state_dict(checkpoint["model"], strict=True)
            lineage = extend_resume_lineage(checkpoint, path, digest)
            names, _ = configure_trainable_parameters(
                model,
                endpoint_refiner_only=True,
            )
            trainable = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.0)
            self.assertFalse(optimizer.state)
            before = {
                name: parameter.detach().clone()
                for name, parameter in model.named_parameters()
            }

            output = model(torch.randn(1, 3, 64, 64, device="cuda"))
            targets = {
                "types": torch.zeros(1, 288, dtype=torch.long, device="cuda"),
                "geometry": torch.zeros(1, 288, 8, device="cuda"),
                "stroke": torch.zeros(1, 288, device="cuda"),
                "has_stroke": torch.zeros(
                    1, 288, dtype=torch.bool, device="cuda"
                ),
                "target_mask": torch.zeros(
                    1, 288, dtype=torch.bool, device="cuda"
                ),
            }
            targets["types"][0, 0] = LINE
            targets["geometry"][0, 0, :4] = torch.tensor(
                [0.1, 0.2, 0.8, 0.7],
                device="cuda",
            )
            targets["target_mask"][0, 0] = True
            losses = primitive_set_loss(
                output,
                targets,
                line_geometry_objective=True,
                class_cost_weight=0.5,
            )
            losses["loss"].backward()
            optimizer.step()

            changed = []
            for name, parameter in model.named_parameters():
                if not torch.equal(before[name], parameter.detach()):
                    changed.append(name)
            self.assertTrue(changed)
            self.assertTrue(all(name in names for name in changed))
            self.assertEqual(int(checkpoint["global_step"]) + 1, args.max_steps)

            resumed_path = Path(directory) / "resumed.pt"
            save_checkpoint(
                resumed_path,
                model,
                optimizer,
                global_step=args.max_steps,
                samples_seen=int(checkpoint["samples_seen"]) + 1,
                args=args,
                validation={"loss": float(losses["loss"].detach())},
                source_checkpoint=Path(checkpoint["source_vision_checkpoint"]),
                source_checkpoint_sha256=checkpoint[
                    "source_vision_checkpoint_sha256"
                ],
                resume_lineage=lineage,
                freeze_mode=ENDPOINT_REFINER_FREEZE_MODE,
                optimizer_mode=FRESH_OPTIMIZER_MODE,
            )
            resumed = torch.load(resumed_path, map_location="cpu", weights_only=False)
            self.assertEqual(resumed["global_step"], 6)
            self.assertEqual(resumed["resume_lineage"], lineage)
            self.assertEqual(resumed["freeze_mode"], ENDPOINT_REFINER_FREEZE_MODE)
            self.assertEqual(resumed["optimizer_mode"], FRESH_OPTIMIZER_MODE)
            self.assertEqual(resumed["trainable_parameter_names"], names)
            self.assertEqual(
                resumed["trainable_parameter_count"],
                sum(parameter.numel() for parameter in trainable),
            )


if __name__ == "__main__":
    unittest.main()
