#!/usr/bin/env python3
"""Train the unordered Visual SVG v0.2 bond-group set predictor."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import glob
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from chemvector.bond_group_set import (
    NO_OBJECT,
    BondGroupSetConfig,
    BondGroupSetPredictor,
    bond_group_set_loss,
)
from chemvector.bond_group_set_data import BondGroupSetCollator
from chemvector.model import ChemVectorConfig
from chemvector.primitive_set import (
    PrimitiveSetConfig,
    PrimitiveSetPredictor,
    load_visual_prefix_vision,
)
from chemvector.train_primitive_set import (
    CAPACITY_VISION_CHECKPOINT,
    CAPACITY_VISION_SHA256,
    CHECKPOINT_SCHEMA_VERSION as PRIMITIVE_SET_CHECKPOINT_SCHEMA_VERSION,
    ENDPOINT_REFINER_FREEZE_MODE,
    ENDPOINT_REFINER_PARAMETER_PREFIXES,
    FULL_MODEL_FREEZE_MODE,
    SEQUENCE_SCHEMA_VERSION as PRIMITIVE_SET_SEQUENCE_SCHEMA_VERSION,
    file_sha256,
    load_capacity_vision_source,
    prepare_output_directory,
)
from chemvector.visual_svg_data import VisualSVGWebDataset


CHECKPOINT_SCHEMA_VERSION = "bond-group-set-checkpoint-0.1"
SEQUENCE_SCHEMA_VERSION = "visual-svg-0.2"
DEFAULT_CAPACITY_SHARD = (
    "Data/sdf/visual_svg_v0_1/capacity_100/shards/capacity-000000.tar"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v093-bond-group-set-capacity100-p1"
)
TARGET_FIELDS = (
    "types",
    "member_counts",
    "member_geometry",
    "member_geometry_mask",
    "member_stroke",
    "member_mask",
    "text_geometry",
    "target_mask",
    "object_count",
)
LOSS_FIELDS = (
    "loss",
    "classification_loss",
    "geometry_loss",
    "text_geometry_loss",
    "stroke_loss",
    "member_count_loss",
    "cardinality_loss",
    "count_classification_loss",
    "line_endpoint_loss",
    "line_center_loss",
    "line_length_loss",
    "line_orientation_loss",
)
FRESH_OPTIMIZER_MODE = "fresh"
PRIMITIVE_SET_WARMSTART_PREFIXES = (
    "vision_encoder.",
    "memory_projection.",
    "raw_ink_projection.",
    "local_query_projection.",
    "query_position_projection.",
    "decoder.",
    "endpoint_refine_projection.",
    "endpoint_refine_depthwise.",
    "endpoint_refine_mlp.",
    "endpoint_point_embedding.",
)
PRIMITIVE_SET_SHARED_CONFIG_FIELDS = (
    "d_model",
    "decoder_layers",
    "decoder_heads",
    "decoder_ffn_dim",
    "memory_grid_size",
    "dropout",
    "local_query_kernel",
    "endpoint_refine_kernel",
    "line_center_vector",
)


def reject_internal_test_shards(shards: list[str]) -> None:
    forbidden = [path for path in shards if "internal_test" in path.lower()]
    if forbidden:
        raise ValueError(
            "Bond Group Set training must not read sealed internal_test shards: "
            + ", ".join(forbidden)
        )


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


def is_primitive_set_warmstart_key(name: str) -> bool:
    """Return whether one flat-model tensor is semantically reusable."""

    return any(name.startswith(prefix) for prefix in PRIMITIVE_SET_WARMSTART_PREFIXES)


def validate_primitive_set_init_checkpoint(checkpoint: dict) -> None:
    expected = {
        "checkpoint_schema_version": PRIMITIVE_SET_CHECKPOINT_SCHEMA_VERSION,
        "sequence_schema_version": PRIMITIVE_SET_SEQUENCE_SCHEMA_VERSION,
        "model_type": PrimitiveSetPredictor.model_type,
        "source_vision_checkpoint_sha256": CAPACITY_VISION_SHA256,
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise ValueError(
            "Primitive Set warm-start metadata differs: " + repr(mismatches)
        )
    if checkpoint.get("uses_internal_test") is True:
        raise ValueError("Primitive Set warm-start checkpoint used internal_test")
    for name in (
        "model",
        "vision_config",
        "primitive_set_config",
        "training_args",
    ):
        if not isinstance(checkpoint.get(name), dict):
            raise ValueError(f"Primitive Set warm-start checkpoint has no {name}")
    if not isinstance(checkpoint.get("source_vision_checkpoint"), str):
        raise ValueError(
            "Primitive Set warm-start checkpoint has no source vision checkpoint"
        )
    for shard_field in ("train_shards", "validation_shards"):
        shards = checkpoint["training_args"].get(shard_field, [])
        if not isinstance(shards, list) or not all(
            isinstance(path, str) for path in shards
        ):
            raise ValueError(
                f"Primitive Set warm-start has invalid {shard_field}: {shards!r}"
            )
        reject_internal_test_shards(shards)


def validate_primitive_set_shared_config(
    source: PrimitiveSetConfig,
    target: BondGroupSetConfig,
) -> None:
    mismatches = {
        name: (getattr(source, name), getattr(target, name))
        for name in PRIMITIVE_SET_SHARED_CONFIG_FIELDS
        if getattr(source, name) != getattr(target, name)
    }
    if mismatches:
        raise ValueError(
            "Primitive Set warm-start shared architecture differs: "
            + repr(mismatches)
        )


def load_primitive_set_init_checkpoint(
    path: Path,
    target_config: BondGroupSetConfig,
) -> tuple[dict, ChemVectorConfig, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Primitive Set warm-start checkpoint does not exist: {path}"
        )
    digest = file_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_primitive_set_init_checkpoint(checkpoint)
    source_config = PrimitiveSetConfig(**checkpoint["primitive_set_config"])
    validate_primitive_set_shared_config(source_config, target_config)
    return checkpoint, ChemVectorConfig(**checkpoint["vision_config"]), digest


def load_primitive_set_warmstart_modules(
    model: BondGroupSetPredictor,
    checkpoint: dict,
) -> list[str]:
    """Load the complete reusable module whitelist and no output/query state."""

    source_state = checkpoint["model"]
    target_state = model.state_dict()
    source_keys = {
        name for name in source_state if is_primitive_set_warmstart_key(name)
    }
    target_keys = {
        name for name in target_state if is_primitive_set_warmstart_key(name)
    }
    if source_keys != target_keys:
        raise ValueError(
            "Primitive Set warm-start reusable key set differs: "
            f"missing={sorted(target_keys - source_keys)} "
            f"unexpected={sorted(source_keys - target_keys)}"
        )
    shape_mismatches = {
        name: (tuple(source_state[name].shape), tuple(target_state[name].shape))
        for name in sorted(target_keys)
        if source_state[name].shape != target_state[name].shape
    }
    if shape_mismatches:
        raise ValueError(
            "Primitive Set warm-start tensor shapes differ: "
            + repr(shape_mismatches)
        )
    loaded_keys = sorted(target_keys)
    incompatible = model.load_state_dict(
        {name: source_state[name] for name in loaded_keys},
        strict=False,
    )
    expected_missing = sorted(set(target_state) - target_keys)
    if sorted(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(
            "Primitive Set warm-start produced an unexpected missing-key set"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Primitive Set warm-start produced unexpected keys: "
            + repr(incompatible.unexpected_keys)
        )
    return loaded_keys


def primitive_set_warmstart_provenance(
    path: Path,
    sha256: str,
    checkpoint: dict,
    loaded_keys: list[str],
) -> dict:
    return {
        "mode": "primitive_set_warmstart",
        "checkpoint": str(path),
        "checkpoint_resolved": str(path.resolve()),
        "checkpoint_sha256": sha256,
        "checkpoint_schema_version": checkpoint["checkpoint_schema_version"],
        "model_type": checkpoint["model_type"],
        "global_step": int(checkpoint.get("global_step", 0)),
        "samples_seen": int(checkpoint.get("samples_seen", 0)),
        "source_vision_checkpoint": checkpoint["source_vision_checkpoint"],
        "source_vision_checkpoint_sha256": checkpoint[
            "source_vision_checkpoint_sha256"
        ],
        "loaded_keys": list(loaded_keys),
        "loaded_key_count": len(loaded_keys),
    }


def configure_trainable_parameters(
    model: BondGroupSetPredictor,
    *,
    endpoint_refiner_only: bool,
) -> tuple[list[str], int]:
    names = []
    count = 0
    for name, parameter in model.named_parameters():
        selected = not endpoint_refiner_only or name.startswith(
            ENDPOINT_REFINER_PARAMETER_PREFIXES
        )
        parameter.requires_grad_(selected)
        if selected:
            names.append(name)
            count += parameter.numel()
    if not names:
        raise ValueError("Bond Group Set training selected no parameters")
    return names, count


def bond_group_config(
    *,
    tiny: bool = False,
    queries: int = 192,
    max_members: int = 32,
    local_query_kernel: int = 5,
    endpoint_refine_kernel: int = 25,
) -> BondGroupSetConfig:
    if tiny:
        return BondGroupSetConfig(
            queries=queries,
            max_members=max_members,
            d_model=64,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=128,
            memory_grid_size=8,
            dropout=0.0,
            local_query_kernel=local_query_kernel,
            endpoint_refine_kernel=endpoint_refine_kernel,
        )
    return BondGroupSetConfig(
        queries=queries,
        max_members=max_members,
        d_model=256,
        decoder_layers=4,
        decoder_heads=8,
        decoder_ffn_dim=1024,
        memory_grid_size=48,
        dropout=0.0,
        local_query_kernel=local_query_kernel,
        endpoint_refine_kernel=endpoint_refine_kernel,
    )


def make_loader(
    shards: list[str],
    *,
    config: BondGroupSetConfig,
    batch_size: int,
    workers: int,
    image_size: int,
    shuffle: bool,
    limit: int | None = None,
) -> DataLoader:
    reject_internal_test_shards(shards)
    return DataLoader(
        VisualSVGWebDataset(
            shards,
            shardshuffle=shuffle,
            sample_shuffle_buffer=100 if shuffle else 0,
            limit=limit,
        ),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=BondGroupSetCollator(
            image_size=image_size,
            query_count=config.queries,
            max_members=config.max_members,
        ),
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


@torch.inference_mode()
def validate(
    model: BondGroupSetPredictor,
    loader,
    device: torch.device,
) -> dict[str, float | int | str]:
    model.eval()
    totals = {name: 0.0 for name in LOSS_FIELDS}
    samples = 0
    matched_objects = 0
    matched_type_correct = 0
    matched_member_count_correct = 0
    count_absolute_error = 0.0
    count_exact = 0
    white_loss_total = 0.0
    started = time.time()
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = targets_to_device(batch, device)
        with autocast_context(device):
            output = model(pixels)
            losses = bond_group_set_loss(output, targets, image_size=pixels.size(-1))
            white_output = model(torch.ones_like(pixels))
            white_losses = bond_group_set_loss(
                white_output,
                targets,
                image_size=pixels.size(-1),
            )
        batch_size = pixels.size(0)
        samples += batch_size
        for name in LOSS_FIELDS:
            totals[name] += float(losses[name]) * batch_size
        white_loss_total += float(white_losses["loss"]) * batch_size
        predicted_total = output["count_logits"].float().argmax(dim=-1)
        expected_total = targets["target_mask"].sum(dim=-1)
        count_absolute_error += float((predicted_total - expected_total).abs().sum())
        count_exact += int((predicted_total == expected_total).sum())
        for batch_index, (query_indices, target_indices) in enumerate(
            losses["assignments"]
        ):
            if not len(query_indices):
                continue
            predicted_types = output["type_logits"][
                batch_index,
                query_indices,
            ].argmax(dim=-1)
            expected_types = targets["types"][batch_index, target_indices]
            predicted_members = output["member_count_logits"][
                batch_index,
                query_indices,
            ].argmax(dim=-1)
            expected_members = targets["member_counts"][batch_index, target_indices]
            matched_objects += len(query_indices)
            matched_type_correct += int((predicted_types == expected_types).sum())
            matched_member_count_correct += int(
                (predicted_members == expected_members).sum()
            )
    if not samples:
        raise RuntimeError("Bond Group Set validation loader produced no samples")
    mean_loss = totals["loss"] / samples
    mean_white_loss = white_loss_total / samples
    return {
        "schema_version": "bond-group-set-validation-0.1",
        "samples": samples,
        **{name: value / samples for name, value in totals.items()},
        "matched_objects": matched_objects,
        "matched_type_accuracy": matched_type_correct / max(matched_objects, 1),
        "matched_member_count_accuracy": matched_member_count_correct
        / max(matched_objects, 1),
        "count_head_mae": count_absolute_error / samples,
        "count_head_exact_rate": count_exact / samples,
        "white_image_loss": mean_white_loss,
        "white_image_loss_ratio": mean_white_loss / max(mean_loss, 1e-8),
        "elapsed_seconds": time.time() - started,
        "uses_internal_test": False,
    }


def save_checkpoint(
    path: Path,
    model: BondGroupSetPredictor,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    samples_seen: int,
    args: argparse.Namespace,
    validation: dict | None,
    source_checkpoint: Path,
    source_checkpoint_sha256: str,
    resume_lineage: list[dict] | None = None,
    initialization_provenance: dict | None = None,
    freeze_mode: str = FULL_MODEL_FREEZE_MODE,
    trainable_parameter_names: list[str] | None = None,
    trainable_parameter_count: int | None = None,
) -> None:
    if trainable_parameter_names is None:
        trainable_parameter_names = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
    if trainable_parameter_count is None:
        trainable_parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    atomic_torch_save(
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
            "model_type": model.model_type,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "vision_config": asdict(model.vision_config),
            "bond_group_set_config": asdict(model.config),
            "global_step": global_step,
            "samples_seen": samples_seen,
            "training_args": vars(args),
            "validation": validation,
            "source_vision_checkpoint": str(source_checkpoint),
            "source_vision_checkpoint_sha256": source_checkpoint_sha256,
            "resume_lineage": list(resume_lineage or []),
            "initialization_provenance": initialization_provenance,
            "freeze_mode": freeze_mode,
            "optimizer_mode": FRESH_OPTIMIZER_MODE,
            "trainable_parameter_names": list(trainable_parameter_names),
            "trainable_parameter_count": int(trainable_parameter_count),
            "uses_internal_test": False,
        },
        path,
    )


def atomic_torch_save(payload: dict, path: Path) -> None:
    """Write one checkpoint completely before replacing the visible path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RuntimeError(f"Atomic checkpoint temporary path already exists: {temporary}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_resume_checkpoint(checkpoint: dict) -> None:
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
        "model_type": BondGroupSetPredictor.model_type,
        "source_vision_checkpoint_sha256": CAPACITY_VISION_SHA256,
        "uses_internal_test": False,
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Bond Group Set resume metadata differs: {mismatches}")
    for name in (
        "model",
        "vision_config",
        "bond_group_set_config",
        "training_args",
    ):
        if not isinstance(checkpoint.get(name), dict):
            raise ValueError(f"Bond Group Set resume checkpoint has no {name}")
    for name in ("global_step", "samples_seen"):
        value = checkpoint.get(name)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Bond Group Set resume has invalid {name}: {value!r}")
    if not isinstance(checkpoint.get("source_vision_checkpoint"), str):
        raise ValueError("Bond Group Set resume has no source vision checkpoint")
    lineage = checkpoint.get("resume_lineage", [])
    if lineage is not None and not isinstance(lineage, list):
        raise ValueError("Bond Group Set resume lineage must be a list")
    provenance = checkpoint.get("initialization_provenance")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError(
            "Bond Group Set resume initialization provenance must be a dictionary"
        )


def load_resume_checkpoint(
    path: Path,
    args: argparse.Namespace,
) -> tuple[dict, BondGroupSetConfig, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Bond Group Set resume checkpoint does not exist: {path}")
    digest = file_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_resume_checkpoint(checkpoint)
    config = BondGroupSetConfig(**checkpoint["bond_group_set_config"])
    requested = bond_group_config(
        tiny=args.tiny,
        queries=args.queries,
        max_members=args.max_members,
        local_query_kernel=args.local_query_kernel,
        endpoint_refine_kernel=args.endpoint_refine_kernel,
    )
    if config != requested:
        raise ValueError(
            "Bond Group Set resume architecture differs from CLI: "
            f"checkpoint={asdict(config)} cli={asdict(requested)}"
        )
    previous_image_size = int(checkpoint["training_args"].get("image_size", 768))
    if previous_image_size != args.image_size:
        raise ValueError(
            "Bond Group Set resume image size differs from CLI: "
            f"checkpoint={previous_image_size} cli={args.image_size}"
        )
    return checkpoint, config, digest


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


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shards", nargs="+", default=[DEFAULT_CAPACITY_SHARD])
    parser.add_argument(
        "--validation-shards",
        nargs="+",
        default=[DEFAULT_CAPACITY_SHARD],
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--vision-checkpoint",
        default=str(CAPACITY_VISION_CHECKPOINT),
    )
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--init-primitive-set-checkpoint")
    parser.add_argument("--endpoint-refiner-only", action="store_true")
    parser.add_argument("--queries", type=int, default=192)
    parser.add_argument("--max-members", type=int, default=32)
    parser.add_argument("--local-query-kernel", type=int, default=5)
    parser.add_argument("--endpoint-refine-kernel", type=int, default=25)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--tiny", action="store_true")
    return parser


def validate_initialization_cli_contract(args: argparse.Namespace) -> None:
    if args.resume_checkpoint and args.init_primitive_set_checkpoint:
        raise ValueError(
            "--resume-checkpoint and --init-primitive-set-checkpoint are mutually exclusive"
        )
    if args.endpoint_refiner_only and not args.resume_checkpoint:
        raise ValueError("--endpoint-refiner-only requires --resume-checkpoint")


def main() -> None:
    args = build_parser().parse_args()
    validate_initialization_cli_contract(args)
    if min(
        args.batch_size,
        args.max_steps,
        args.validate_every,
        args.save_every,
        args.image_size,
    ) < 1:
        raise ValueError("Positive training dimensions are required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    train_shards = expand_shards(args.train_shards)
    validation_shards = expand_shards(args.validation_shards)
    output_dir = Path(args.output_dir)
    prepare_output_directory(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_lineage: list[dict] = []
    initialization_provenance = None
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        resume, config, resume_sha256 = load_resume_checkpoint(resume_path, args)
        vision_config = ChemVectorConfig(**resume["vision_config"])
        model = BondGroupSetPredictor(vision_config, config).to(device)
        model.load_state_dict(resume["model"], strict=True)
        source_path = Path(resume["source_vision_checkpoint"])
        source_sha256 = str(resume["source_vision_checkpoint_sha256"])
        global_step = int(resume["global_step"])
        samples_seen = int(resume["samples_seen"])
        resume_lineage = extend_resume_lineage(
            resume,
            resume_path,
            resume_sha256,
        )
        initialization_provenance = resume.get("initialization_provenance")
        if args.max_steps <= global_step:
            raise ValueError(
                "--max-steps is the final global step and must exceed the resume "
                f"step {global_step}"
            )
    elif args.init_primitive_set_checkpoint:
        config = bond_group_config(
            tiny=args.tiny,
            queries=args.queries,
            max_members=args.max_members,
            local_query_kernel=args.local_query_kernel,
            endpoint_refine_kernel=args.endpoint_refine_kernel,
        )
        primitive_path = Path(args.init_primitive_set_checkpoint)
        primitive, vision_config, primitive_sha256 = (
            load_primitive_set_init_checkpoint(primitive_path, config)
        )
        model = BondGroupSetPredictor(vision_config, config).to(device)
        loaded_keys = load_primitive_set_warmstart_modules(model, primitive)
        source_path = Path(primitive["source_vision_checkpoint"])
        source_sha256 = str(primitive["source_vision_checkpoint_sha256"])
        initialization_provenance = primitive_set_warmstart_provenance(
            primitive_path,
            primitive_sha256,
            primitive,
            loaded_keys,
        )
        global_step = 0
        samples_seen = 0
        del primitive
    else:
        source_path = Path(args.vision_checkpoint)
        source, vision_config, source_sha256 = load_capacity_vision_source(source_path)
        config = bond_group_config(
            tiny=args.tiny,
            queries=args.queries,
            max_members=args.max_members,
            local_query_kernel=args.local_query_kernel,
            endpoint_refine_kernel=args.endpoint_refine_kernel,
        )
        model = BondGroupSetPredictor(vision_config, config).to(device)
        load_visual_prefix_vision(model, source)
        global_step = 0
        samples_seen = 0
    freeze_mode = (
        ENDPOINT_REFINER_FREEZE_MODE
        if args.endpoint_refiner_only
        else FULL_MODEL_FREEZE_MODE
    )
    trainable_names, trainable_count = configure_trainable_parameters(
        model,
        endpoint_refiner_only=args.endpoint_refiner_only,
    )
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = make_loader(
        train_shards,
        config=config,
        batch_size=args.batch_size,
        workers=args.workers,
        image_size=args.image_size,
        shuffle=True,
    )
    validation_loader = make_loader(
        validation_shards,
        config=config,
        batch_size=args.batch_size,
        workers=args.workers,
        image_size=args.image_size,
        shuffle=False,
        limit=args.validation_limit,
    )

    best_loss = float("inf")
    last_validation = None
    history = []
    while global_step < args.max_steps:
        batches = 0
        for batch in train_loader:
            batches += 1
            model.train()
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            targets = targets_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                output = model(pixels)
                losses = bond_group_set_loss(
                    output,
                    targets,
                    image_size=args.image_size,
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            global_step += 1
            samples_seen += pixels.size(0)

            if global_step % args.validate_every == 0 or global_step >= args.max_steps:
                last_validation = validate(model, validation_loader, device)
                history.append({"global_step": global_step, "metrics": last_validation})
                print(
                    f"step={global_step} loss={last_validation['loss']:.6f} "
                    f"type={last_validation['matched_type_accuracy']:.4f} "
                    f"member_count={last_validation['matched_member_count_accuracy']:.4f}",
                    flush=True,
                )
                if float(last_validation["loss"]) < best_loss:
                    best_loss = float(last_validation["loss"])
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
                        initialization_provenance=initialization_provenance,
                        freeze_mode=freeze_mode,
                        trainable_parameter_names=trainable_names,
                        trainable_parameter_count=trainable_count,
                    )
            if global_step % args.save_every == 0 or global_step >= args.max_steps:
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
                    initialization_provenance=initialization_provenance,
                    freeze_mode=freeze_mode,
                    trainable_parameter_names=trainable_names,
                    trainable_parameter_count=trainable_count,
                )
            if global_step >= args.max_steps:
                break
        if not batches:
            raise RuntimeError("Bond Group Set training loader produced no samples")

    assert last_validation is not None
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
        initialization_provenance=initialization_provenance,
        freeze_mode=freeze_mode,
        trainable_parameter_names=trainable_names,
        trainable_parameter_count=trainable_count,
    )
    write_json(output_dir / "validation.json", last_validation)
    write_json(
        output_dir / "summary.json",
        {
            "schema_version": "bond-group-set-training-summary-0.1",
            "model_type": model.model_type,
            "global_step": global_step,
            "samples_seen": samples_seen,
            "best_validation_loss": best_loss,
            "history": history,
            "source_vision_checkpoint": str(source_path),
            "source_vision_checkpoint_sha256": source_sha256,
            "config": asdict(config),
            "resume_lineage": resume_lineage,
            "initialization_provenance": initialization_provenance,
            "freeze_mode": freeze_mode,
            "optimizer_mode": FRESH_OPTIMIZER_MODE,
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": trainable_count,
            "uses_internal_test": False,
        },
    )


if __name__ == "__main__":
    main()
