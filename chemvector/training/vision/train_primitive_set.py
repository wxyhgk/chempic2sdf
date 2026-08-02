#!/usr/bin/env python3
"""Train the unordered Visual SVG primitive-set predictor."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import glob
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from chemvector.model import ChemVectorConfig
from chemvector.primitive_set import (
    NO_OBJECT,
    PRIMITIVE_TYPES,
    PrimitiveSetConfig,
    PrimitiveSetPredictor,
    load_visual_prefix_vision,
    primitive_set_loss,
)
from chemvector.primitive_set_data import PrimitiveSetCollator
from chemvector.visual_svg_data import VisualSVGWebDataset


CAPACITY_VISION_CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-v090-visual-prefix-capacity-final-clean-1000/"
    "best-teacher.pt"
)
CAPACITY_VISION_SHA256 = (
    "df9468b525dff285e2b70988332be354a541ec972476068c8bf6bf137e914ddf"
)
CHECKPOINT_SCHEMA_VERSION = "primitive-set-checkpoint-0.1"
VALIDATION_SCHEMA_VERSION = "primitive-set-validation-0.2"
SEQUENCE_SCHEMA_VERSION = "visual-svg-0.1"
SOURCE_MODEL_TYPE = "visual-prefix-v0.1"
DEFAULT_OUTPUT_DIR = "outputs/checkpoints/chemvector-v092-primitive-set-pilot10k-300"
TARGET_FIELDS = (
    "types",
    "geometry",
    "geometry_mask",
    "stroke",
    "has_stroke",
    "target_mask",
)
LOSS_FIELDS = (
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
)
ENDPOINT_REFINER_PARAMETER_PREFIXES = (
    "endpoint_refine_projection.",
    "endpoint_refine_depthwise.",
    "endpoint_refine_mlp.",
    "endpoint_point_embedding.",
)
OUTPUT_HEAD_PARAMETER_PREFIXES = (
    "type_head.",
    "text_geometry_head.",
    "stroke_head.",
    "count_head.",
)
COARSE_GEOMETRY_PARAMETER_PREFIXES = (
    "geometry_head.",
    "line_geometry_head.",
)
FULL_MODEL_FREEZE_MODE = "all_parameters_trainable"
ENDPOINT_REFINER_FREEZE_MODE = "endpoint_refiner_only"
OUTPUT_HEADS_FREEZE_MODE = "output_heads_only"
COARSE_GEOMETRY_FREEZE_MODE = "coarse_geometry_only"
FRESH_OPTIMIZER_MODE = "fresh"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand_shards(patterns: list[str]) -> list[str]:
    shards = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            shards.extend(matches)
        elif Path(pattern).is_file():
            shards.append(pattern)
    unique = list(dict.fromkeys(shards))
    if not unique:
        raise ValueError(f"No shards matched: {patterns}")
    reject_internal_test_shards(unique)
    return unique


def _normalized_security_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        default=str,
    ).lower().replace("-", "_")


def reject_internal_test_shards(shards: list[str]) -> None:
    forbidden = []
    for path in shards:
        values = (str(Path(path)), str(Path(path).expanduser().resolve()))
        if any("internal_test" in value.lower().replace("-", "_") for value in values):
            forbidden.append(path)
    if forbidden:
        raise ValueError(
            "Primitive Set training must not read the sealed internal_test split: "
            + ", ".join(forbidden)
        )


def reject_internal_test_metadata(meta: Any, sample_key: str = "") -> None:
    normalized_key = sample_key.lower().replace("-", "_")
    if "internal_test" in normalized_key or "internal_test" in _normalized_security_text(meta):
        raise ValueError(f"Refusing sealed internal_test sample: {sample_key}")


def validate_batch_security(batch: dict[str, Any]) -> None:
    keys = batch.get("keys")
    metadata = batch.get("meta")
    if not isinstance(keys, list) or not isinstance(metadata, list):
        raise ValueError("Primitive Set batch lacks security keys or metadata")
    if len(keys) != len(metadata):
        raise ValueError("Primitive Set batch keys and metadata differ")
    for key, meta in zip(keys, metadata):
        reject_internal_test_metadata(meta, str(key))


def reject_train_validation_overlap(
    train_shards: list[str],
    validation_shards: list[str],
) -> None:
    train = {Path(path).expanduser().resolve() for path in train_shards}
    validation = {Path(path).expanduser().resolve() for path in validation_shards}
    overlap = sorted(str(path) for path in train & validation)
    if overlap:
        raise ValueError(
            "Primitive Set train/validation shards overlap: " + ", ".join(overlap)
        )


def prepare_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def validate_capacity_checkpoint(checkpoint: dict) -> None:
    expected = {
        "checkpoint_schema_version": "visual-svg-checkpoint-0.2",
        "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
        "model_type": SOURCE_MODEL_TYPE,
        "visual_grid_size": 12,
        "ink_subgrid": 8,
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Capacity vision checkpoint metadata differs: {mismatches}")
    if not isinstance(checkpoint.get("config"), dict):
        raise ValueError("Capacity vision checkpoint has no model config")
    if not isinstance(checkpoint.get("model"), dict):
        raise ValueError("Capacity vision checkpoint has no model state")


def vision_config_from_checkpoint(checkpoint: dict) -> ChemVectorConfig:
    valid = {field.name for field in fields(ChemVectorConfig)}
    values = {
        name: value
        for name, value in checkpoint["config"].items()
        if name in valid
    }
    return ChemVectorConfig(**values)


def load_capacity_vision_source(
    path: Path,
    *,
    expected_sha256: str = CAPACITY_VISION_SHA256,
) -> tuple[dict, ChemVectorConfig, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Capacity vision checkpoint does not exist: {path}")
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise ValueError(
            "Primitive Set initialization is restricted to the frozen capacity-final "
            f"checkpoint: expected {expected_sha256}, received {digest}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_capacity_checkpoint(checkpoint)
    return checkpoint, vision_config_from_checkpoint(checkpoint), digest


def validate_resume_checkpoint(checkpoint: dict) -> None:
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
        "model_type": PrimitiveSetPredictor.model_type,
        "source_vision_checkpoint_sha256": CAPACITY_VISION_SHA256,
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Resume checkpoint metadata differs: {mismatches}")
    for name in ("model", "vision_config", "primitive_set_config", "training_args"):
        if not isinstance(checkpoint.get(name), dict):
            raise ValueError(f"Resume checkpoint has no {name}")
    if not isinstance(checkpoint.get("source_vision_checkpoint"), str):
        raise ValueError("Resume checkpoint has no source vision checkpoint")
    for name in ("global_step", "samples_seen"):
        value = checkpoint.get(name)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Resume checkpoint has invalid {name}: {value!r}")
    lineage = checkpoint.get("resume_lineage", [])
    if lineage is not None and not isinstance(lineage, list):
        raise ValueError("Resume checkpoint lineage must be a list")


def resume_checkpoint_configs(
    checkpoint: dict,
) -> tuple[ChemVectorConfig, PrimitiveSetConfig]:
    valid_vision = {field.name for field in fields(ChemVectorConfig)}
    vision_values = {
        name: value
        for name, value in checkpoint["vision_config"].items()
        if name in valid_vision
    }
    return (
        ChemVectorConfig(**vision_values),
        PrimitiveSetConfig(**checkpoint["primitive_set_config"]),
    )


def validate_resume_cli_contract(
    checkpoint: dict,
    args: argparse.Namespace,
    checkpoint_config: PrimitiveSetConfig,
) -> None:
    requested_config = primitive_config(
        tiny=args.tiny,
        local_query_kernel=args.local_query_kernel,
        endpoint_refine_kernel=args.endpoint_refine_kernel,
        line_center_vector=args.line_center_vector,
    )
    if checkpoint_config != requested_config:
        raise ValueError(
            "Resume checkpoint architecture differs from CLI: "
            f"checkpoint={asdict(checkpoint_config)} cli={asdict(requested_config)}"
        )
    previous_args = checkpoint["training_args"]
    previous_protocol = {
        "line_geometry_objective": bool(
            previous_args.get("line_geometry_objective", False)
        ),
        "class_cost_weight": float(previous_args.get("class_cost_weight", 2.0)),
    }
    requested_protocol = loss_kwargs_from_args(args)
    if previous_protocol != requested_protocol:
        raise ValueError(
            "Resume checkpoint loss protocol differs from CLI: "
            f"checkpoint={previous_protocol} cli={requested_protocol}"
        )


def load_resume_checkpoint(
    path: Path,
    args: argparse.Namespace,
) -> tuple[dict, ChemVectorConfig, PrimitiveSetConfig, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    digest = file_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_resume_checkpoint(checkpoint)
    vision_config, set_config = resume_checkpoint_configs(checkpoint)
    validate_resume_cli_contract(checkpoint, args, set_config)
    return checkpoint, vision_config, set_config, digest


def extend_resume_lineage(
    checkpoint: dict,
    path: Path,
    sha256: str,
) -> list[dict]:
    lineage = list(checkpoint.get("resume_lineage") or [])
    lineage.append(
        {
            "checkpoint": str(path),
            "sha256": sha256,
            "global_step": int(checkpoint["global_step"]),
            "samples_seen": int(checkpoint["samples_seen"]),
        }
    )
    return lineage


def configure_trainable_parameters(
    model: PrimitiveSetPredictor,
    *,
    endpoint_refiner_only: bool,
    output_heads_only: bool = False,
    coarse_geometry_only: bool = False,
) -> tuple[list[str], int]:
    selected_modes = sum(
        (endpoint_refiner_only, output_heads_only, coarse_geometry_only)
    )
    if selected_modes > 1:
        raise ValueError(
            "Frozen-subset training modes are mutually exclusive"
        )
    if endpoint_refiner_only and not model.config.endpoint_refine_kernel:
        raise ValueError(
            "--endpoint-refiner-only requires a checkpoint with endpoint refinement"
        )
    selected_prefixes = None
    if endpoint_refiner_only:
        selected_prefixes = ENDPOINT_REFINER_PARAMETER_PREFIXES
    elif output_heads_only:
        selected_prefixes = OUTPUT_HEAD_PARAMETER_PREFIXES
    elif coarse_geometry_only:
        selected_prefixes = COARSE_GEOMETRY_PARAMETER_PREFIXES
    trainable_names = []
    trainable_count = 0
    for name, parameter in model.named_parameters():
        trainable = selected_prefixes is None or name.startswith(selected_prefixes)
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
            trainable_count += parameter.numel()
    if not trainable_names:
        raise RuntimeError("Primitive Set training selected no trainable parameters")
    return trainable_names, trainable_count


def primitive_config(
    *,
    tiny: bool = False,
    local_query_kernel: int = 0,
    endpoint_refine_kernel: int = 0,
    line_center_vector: bool = False,
) -> PrimitiveSetConfig:
    if tiny:
        return PrimitiveSetConfig(
            queries=288,
            d_model=64,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=128,
            memory_grid_size=8,
            dropout=0.0,
            local_query_kernel=local_query_kernel,
            endpoint_refine_kernel=endpoint_refine_kernel,
            line_center_vector=line_center_vector,
        )
    return PrimitiveSetConfig(
        queries=288,
        d_model=256,
        decoder_layers=4,
        decoder_heads=8,
        decoder_ffn_dim=1024,
        memory_grid_size=48,
        dropout=0.0,
        local_query_kernel=local_query_kernel,
        endpoint_refine_kernel=endpoint_refine_kernel,
        line_center_vector=line_center_vector,
    )


def make_loader(
    shards: list[str],
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        VisualSVGWebDataset(
            shards,
            shardshuffle=shuffle,
            sample_shuffle_buffer=100 if shuffle else 0,
        ),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=PrimitiveSetCollator(),
    )


def targets_to_device(batch: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in TARGET_FIELDS
    }


def autocast_context(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


def loss_kwargs_from_args(args: argparse.Namespace) -> dict[str, bool | float]:
    """Build the single loss protocol shared by train and validation paths."""

    return {
        "line_geometry_objective": bool(args.line_geometry_objective),
        "class_cost_weight": float(args.class_cost_weight),
    }


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    threshold: float,
    loss_kwargs: dict[str, bool | float] | None = None,
    require_security_metadata: bool = False,
) -> dict:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Primitive Set validation threshold must be in [0, 1]")
    loss_kwargs = {} if loss_kwargs is None else dict(loss_kwargs)
    model.eval()
    loss_totals = {name: 0.0 for name in LOSS_FIELDS}
    matched_correct = 0
    matched_total = 0
    type_correct = {name: 0 for name in PRIMITIVE_TYPES[1:]}
    type_total = {name: 0 for name in PRIMITIVE_TYPES[1:]}
    threshold_count_error = 0.0
    threshold_count_exact = 0
    soft_count_error = 0.0
    threshold_count_total = 0.0
    count_head_error = 0.0
    count_head_exact = 0
    count_head_total = 0.0
    target_count_total = 0.0
    white_image_loss_total = 0.0
    samples = 0
    batches = 0

    for batch in loader:
        if require_security_metadata:
            validate_batch_security(batch)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = targets_to_device(batch, device)
        with autocast_context(device):
            output = model(pixels)
            losses = primitive_set_loss(output, targets, **loss_kwargs)
            white_output = model(torch.ones_like(pixels))
            white_losses = primitive_set_loss(
                white_output,
                targets,
                **loss_kwargs,
            )
        batch_size = int(pixels.size(0))
        for name in LOSS_FIELDS:
            loss_totals[name] += float(losses[name]) * batch_size
        white_image_loss_total += float(white_losses["loss"]) * batch_size

        predicted_types = output["type_logits"].argmax(dim=-1)
        for batch_index, (query_indices, target_indices) in enumerate(
            losses["assignments"]
        ):
            if not len(query_indices):
                continue
            expected = targets["types"][batch_index, target_indices].long()
            predicted = predicted_types[batch_index, query_indices]
            correct = predicted == expected
            matched_correct += int(correct.sum())
            matched_total += len(expected)
            for type_id, type_name in enumerate(PRIMITIVE_TYPES[1:], 1):
                mask = expected == type_id
                type_total[type_name] += int(mask.sum())
                type_correct[type_name] += int((correct & mask).sum())

        probabilities = output["type_logits"].float().softmax(dim=-1)
        object_scores = probabilities[..., 1:].amax(dim=-1)
        threshold_counts = (
            (object_scores >= threshold)
            & (object_scores > probabilities[..., NO_OBJECT])
        ).sum(dim=1)
        soft_counts = (1.0 - probabilities[..., NO_OBJECT]).sum(dim=1)
        target_counts = targets["target_mask"].sum(dim=1)
        count_logits = output.get("count_logits")
        if count_logits is None:
            raise ValueError("Primitive Set output is missing count_logits")
        if count_logits.ndim != 2 or count_logits.size(0) != batch_size:
            raise ValueError("Primitive Set count_logits have an incompatible shape")
        count_head_counts = count_logits.argmax(dim=-1).clamp_max(
            output["type_logits"].size(1)
        )
        threshold_count_error += float(
            (threshold_counts - target_counts).abs().sum()
        )
        threshold_count_exact += int((threshold_counts == target_counts).sum())
        soft_count_error += float((soft_counts - target_counts).abs().sum())
        threshold_count_total += float(threshold_counts.sum())
        count_head_error += float((count_head_counts - target_counts).abs().sum())
        count_head_exact += int((count_head_counts == target_counts).sum())
        count_head_total += float(count_head_counts.sum())
        target_count_total += float(target_counts.sum())
        samples += batch_size
        batches += 1

    if not samples:
        raise RuntimeError("Primitive Set validation loader produced no samples")
    matched_loss_total = loss_totals["loss"]
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "samples": samples,
        "batches": batches,
        "threshold": threshold,
        **{name: value / samples for name, value in loss_totals.items()},
        "matched_type_accuracy": matched_correct / max(matched_total, 1),
        "matched_type_correct": matched_correct,
        "matched_type_total": matched_total,
        "matched_type_accuracy_by_type": {
            name: type_correct[name] / max(type_total[name], 1)
            for name in PRIMITIVE_TYPES[1:]
        },
        "matched_type_total_by_type": type_total,
        "white_image_loss": white_image_loss_total / samples,
        "white_image_loss_ratio": white_image_loss_total
        / max(matched_loss_total, 1e-12),
        "threshold_count_exact_rate": threshold_count_exact / samples,
        "threshold_count_mae": threshold_count_error / samples,
        "threshold_predicted_count_mean": threshold_count_total / samples,
        "count_mae": threshold_count_error / samples,
        "soft_count_mae": soft_count_error / samples,
        "predicted_count_mean": threshold_count_total / samples,
        "count_head_exact_rate": count_head_exact / samples,
        "count_head_mae": count_head_error / samples,
        "count_head_topk_count_mean": count_head_total / samples,
        "target_count_mean": target_count_total / samples,
    }


def save_checkpoint(
    path: Path,
    model: PrimitiveSetPredictor,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    samples_seen: int,
    args: argparse.Namespace,
    validation: dict | None,
    source_checkpoint: Path,
    source_checkpoint_sha256: str,
    resume_lineage: list[dict] | None = None,
    freeze_mode: str = FULL_MODEL_FREEZE_MODE,
    optimizer_mode: str = FRESH_OPTIMIZER_MODE,
    trainable_parameter_names: list[str] | None = None,
    trainable_parameter_count: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if trainable_parameter_names is None:
        trainable_parameter_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
    if trainable_parameter_count is None:
        trainable_parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    torch.save(
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
            "model_type": model.model_type,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "vision_config": asdict(model.vision_config),
            "primitive_set_config": asdict(model.config),
            "global_step": global_step,
            "samples_seen": samples_seen,
            "training_args": vars(args),
            "validation": validation,
            "source_vision_checkpoint": str(source_checkpoint),
            "source_vision_checkpoint_sha256": source_checkpoint_sha256,
            "resume_lineage": list(resume_lineage or []),
            "freeze_mode": freeze_mode,
            "optimizer_mode": optimizer_mode,
            "trainable_parameter_names": list(trainable_parameter_names),
            "trainable_parameter_count": int(trainable_parameter_count),
        },
        temporary,
    )
    temporary.replace(path)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train fixed-query Visual SVG primitive set prediction."
    )
    parser.add_argument(
        "--train-shards",
        nargs="+",
        default=["Data/sdf/visual_svg_v0_1/pilot_10k/shards/train-*.tar"],
    )
    parser.add_argument(
        "--validation-shards",
        nargs="+",
        default=["Data/sdf/visual_svg_v0_1/pilot_10k/shards/validation-*.tar"],
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--init-vision-checkpoint",
        default=str(CAPACITY_VISION_CHECKPOINT),
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--local-query-kernel", type=int, default=0)
    parser.add_argument("--endpoint-refine-kernel", type=int, default=0)
    parser.add_argument("--line-center-vector", action="store_true")
    parser.add_argument("--line-geometry-objective", action="store_true")
    parser.add_argument("--class-cost-weight", type=float, default=2.0)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--allow-train-validation-overlap", action="store_true")
    parser.add_argument("--endpoint-refiner-only", action="store_true")
    parser.add_argument("--output-heads-only", action="store_true")
    parser.add_argument("--coarse-geometry-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--tiny", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Primitive Set training")
    if min(
        args.max_steps,
        args.batch_size,
        args.gradient_accumulation,
        args.validate_every,
        args.save_every,
        args.log_every,
    ) < 1:
        raise ValueError("Primitive Set training counts must be positive")
    if args.workers < 0 or args.learning_rate <= 0.0:
        raise ValueError("Primitive Set workers and learning rate are invalid")
    if args.class_cost_weight < 0.0:
        raise ValueError("--class-cost-weight must be nonnegative")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    subset_modes = sum(
        (
            args.endpoint_refiner_only,
            args.output_heads_only,
            args.coarse_geometry_only,
        )
    )
    if subset_modes > 1:
        raise ValueError(
            "Frozen-subset training modes are mutually exclusive"
        )
    if subset_modes and not args.resume_checkpoint:
        raise ValueError("Frozen-subset training requires --resume-checkpoint")
    set_config = primitive_config(
        tiny=args.tiny,
        local_query_kernel=args.local_query_kernel,
        endpoint_refine_kernel=args.endpoint_refine_kernel,
        line_center_vector=args.line_center_vector,
    )
    loss_kwargs = loss_kwargs_from_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    train_shards = expand_shards(args.train_shards)
    validation_shards = expand_shards(args.validation_shards)
    if not args.allow_train_validation_overlap:
        reject_train_validation_overlap(train_shards, validation_shards)
    output_dir = Path(args.output_dir)
    prepare_output_directory(output_dir)
    device = torch.device("cuda")
    resume_lineage = []
    if args.endpoint_refiner_only:
        freeze_mode = ENDPOINT_REFINER_FREEZE_MODE
    elif args.output_heads_only:
        freeze_mode = OUTPUT_HEADS_FREEZE_MODE
    elif args.coarse_geometry_only:
        freeze_mode = COARSE_GEOMETRY_FREEZE_MODE
    else:
        freeze_mode = FULL_MODEL_FREEZE_MODE
    optimizer_mode = FRESH_OPTIMIZER_MODE
    global_step = 0
    samples_seen = 0
    best_loss = float("inf")
    resume_path = Path(args.resume_checkpoint) if args.resume_checkpoint else None
    if resume_path is not None:
        resume_checkpoint, vision_config, set_config, resume_sha256 = (
            load_resume_checkpoint(resume_path, args)
        )
        model = PrimitiveSetPredictor(vision_config, set_config).to(device)
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        source_path = Path(resume_checkpoint["source_vision_checkpoint"])
        source_sha256 = resume_checkpoint["source_vision_checkpoint_sha256"]
        global_step = int(resume_checkpoint["global_step"])
        samples_seen = int(resume_checkpoint["samples_seen"])
        resume_lineage = extend_resume_lineage(
            resume_checkpoint,
            resume_path,
            resume_sha256,
        )
        previous_validation = resume_checkpoint.get("validation")
        previous_validation_shards = resume_checkpoint["training_args"].get(
            "validation_shards", []
        )
        same_validation_scope = False
        if isinstance(previous_validation_shards, list) and all(
            isinstance(path, str) for path in previous_validation_shards
        ):
            previous_matches = []
            for pattern in previous_validation_shards:
                matches = sorted(glob.glob(pattern))
                previous_matches.extend(matches or [pattern])
            same_validation_scope = {
                Path(path).expanduser().resolve() for path in previous_matches
            } == {
                Path(path).expanduser().resolve() for path in validation_shards
            }
        if same_validation_scope and isinstance(previous_validation, dict) and isinstance(
            previous_validation.get("loss"), (int, float)
        ):
            best_loss = float(previous_validation["loss"])
        del resume_checkpoint
    else:
        source_path = Path(args.init_vision_checkpoint)
        source_checkpoint, vision_config, source_sha256 = load_capacity_vision_source(
            source_path
        )
        model = PrimitiveSetPredictor(vision_config, set_config).to(device)
        load_visual_prefix_vision(model, source_checkpoint)
        del source_checkpoint
    if args.max_steps < global_step:
        raise ValueError(
            f"--max-steps {args.max_steps} precedes resume global step {global_step}"
        )
    trainable_names, trainable_count = configure_trainable_parameters(
        model,
        endpoint_refiner_only=args.endpoint_refiner_only,
        output_heads_only=args.output_heads_only,
        coarse_geometry_only=args.coarse_geometry_only,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    write_json(output_dir / "config.json", model.config_dict())
    write_json(output_dir / "training_args.json", vars(args))
    write_json(
        output_dir / "vision_source.json",
        {
            "checkpoint": str(source_path),
            "sha256": source_sha256,
            "required_sha256": CAPACITY_VISION_SHA256,
            "resume_checkpoint": str(resume_path) if resume_path else None,
            "resume_lineage": resume_lineage,
        },
    )
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"parameters={model.parameter_count():,}")
    print(f"trainable_parameters={trainable_count:,}")
    print(f"freeze_mode={freeze_mode} optimizer_mode={optimizer_mode}")
    if subset_modes:
        print("trainable_parameter_names=" + json.dumps(trainable_names))
    if resume_path is not None:
        print(f"resume_checkpoint={resume_path} resume_global_step={global_step}")
    print(f"train_shards={len(train_shards)} validation_shards={len(validation_shards)}")
    print(f"vision_source_sha256={source_sha256}")

    train_loader = make_loader(
        train_shards,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
    )
    validation_loader = make_loader(
        validation_shards,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
    )
    optimizer.zero_grad(set_to_none=True)
    accumulated = 0
    validation_history = []
    last_validation = None
    running = {name: 0.0 for name in LOSS_FIELDS}
    running_batches = 0
    started = time.perf_counter()

    while global_step < args.max_steps:
        batches_this_epoch = 0
        for batch in train_loader:
            batches_this_epoch += 1
            validate_batch_security(batch)
            model.train()
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            targets = targets_to_device(batch, device)
            with autocast_context(device):
                output = model(pixels)
                losses = primitive_set_loss(output, targets, **loss_kwargs)
                loss = losses["loss"] / args.gradient_accumulation
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(
                    f"Non-finite Primitive Set loss before optimizer step {global_step + 1}"
                )
            loss.backward()
            accumulated += 1
            samples_seen += int(pixels.size(0))
            running_batches += 1
            for name in LOSS_FIELDS:
                running[name] += float(losses[name].detach())
            if accumulated < args.gradient_accumulation:
                continue

            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            global_step += 1

            if global_step % args.log_every == 0:
                values = " ".join(
                    f"{name}={running[name] / max(running_batches, 1):.6f}"
                    for name in LOSS_FIELDS
                )
                print(
                    f"step={global_step} samples={samples_seen} {values} "
                    f"elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
                running = {name: 0.0 for name in LOSS_FIELDS}
                running_batches = 0
                started = time.perf_counter()

            if global_step % args.validate_every == 0:
                last_validation = validate(
                    model,
                    validation_loader,
                    device,
                    threshold=args.threshold,
                    loss_kwargs=loss_kwargs,
                    require_security_metadata=True,
                )
                validation_history.append(
                    {"global_step": global_step, "metrics": last_validation}
                )
                print(
                    "validation=" + json.dumps(last_validation, sort_keys=True),
                    flush=True,
                )
                if last_validation["loss"] < best_loss:
                    best_loss = last_validation["loss"]
                    save_checkpoint(
                        output_dir / "best.pt",
                        model,
                        optimizer,
                        global_step=global_step,
                        samples_seen=samples_seen,
                        args=args,
                        validation=last_validation,
                        source_checkpoint=source_path,
                        source_checkpoint_sha256=source_sha256,
                        resume_lineage=resume_lineage,
                        freeze_mode=freeze_mode,
                        optimizer_mode=optimizer_mode,
                        trainable_parameter_names=trainable_names,
                        trainable_parameter_count=trainable_count,
                    )

            if global_step % args.save_every == 0:
                save_checkpoint(
                    output_dir / f"step-{global_step:06d}.pt",
                    model,
                    optimizer,
                    global_step=global_step,
                    samples_seen=samples_seen,
                    args=args,
                    validation=last_validation,
                    source_checkpoint=source_path,
                    source_checkpoint_sha256=source_sha256,
                    resume_lineage=resume_lineage,
                    freeze_mode=freeze_mode,
                    optimizer_mode=optimizer_mode,
                    trainable_parameter_names=trainable_names,
                    trainable_parameter_count=trainable_count,
                )
            if global_step >= args.max_steps:
                break
        if batches_this_epoch == 0:
            raise RuntimeError("Primitive Set training loader produced no samples")

    if accumulated:
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("Primitive Set stopped with an incomplete accumulated step")
    if not validation_history or validation_history[-1]["global_step"] != global_step:
        last_validation = validate(
            model,
            validation_loader,
            device,
            threshold=args.threshold,
            loss_kwargs=loss_kwargs,
            require_security_metadata=True,
        )
        validation_history.append(
            {"global_step": global_step, "metrics": last_validation}
        )
    assert last_validation is not None
    if last_validation["loss"] < best_loss or not (output_dir / "best.pt").exists():
        best_loss = last_validation["loss"]
        save_checkpoint(
            output_dir / "best.pt",
            model,
            optimizer,
            global_step=global_step,
            samples_seen=samples_seen,
            args=args,
            validation=last_validation,
            source_checkpoint=source_path,
            source_checkpoint_sha256=source_sha256,
            resume_lineage=resume_lineage,
            freeze_mode=freeze_mode,
            optimizer_mode=optimizer_mode,
            trainable_parameter_names=trainable_names,
            trainable_parameter_count=trainable_count,
        )
    save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        global_step=global_step,
        samples_seen=samples_seen,
        args=args,
        validation=last_validation,
        source_checkpoint=source_path,
        source_checkpoint_sha256=source_sha256,
        resume_lineage=resume_lineage,
        freeze_mode=freeze_mode,
        optimizer_mode=optimizer_mode,
        trainable_parameter_names=trainable_names,
        trainable_parameter_count=trainable_count,
    )
    write_json(output_dir / "validation.json", last_validation)
    write_json(
        output_dir / "summary.json",
        {
            "schema_version": "primitive-set-training-summary-0.1",
            "model_type": model.model_type,
            "global_step": global_step,
            "samples_seen": samples_seen,
            "best_validation_loss": best_loss,
            "history": validation_history,
            "source_vision_checkpoint": str(source_path),
            "source_vision_checkpoint_sha256": source_sha256,
            "resume_lineage": resume_lineage,
            "freeze_mode": freeze_mode,
            "optimizer_mode": optimizer_mode,
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": trainable_count,
            "uses_internal_test": False,
        },
    )
    print("final_validation=" + json.dumps(last_validation, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
