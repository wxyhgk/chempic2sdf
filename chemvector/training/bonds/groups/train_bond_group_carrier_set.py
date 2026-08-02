#!/usr/bin/env python3
"""Train the compact carrier representation for Visual SVG v0.2 groups."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from chemvector.bond_group_carrier_set import (
    BondGroupCarrierSetConfig,
    BondGroupCarrierSetPredictor,
    bond_group_carrier_set_loss,
    carrier_head_warm_start_state,
)
from chemvector.bond_group_carrier_set_data import BondGroupCarrierSetCollator
from chemvector.bond_group_set import BondGroupSetConfig, BondGroupSetPredictor
from chemvector.model import ChemVectorConfig
from chemvector.train_bond_group_set import (
    CHECKPOINT_SCHEMA_VERSION as BOND_GROUP_CHECKPOINT_SCHEMA_VERSION,
    SEQUENCE_SCHEMA_VERSION,
    atomic_torch_save,
    autocast_context,
    expand_shards,
)
from chemvector.train_primitive_set import (
    CAPACITY_VISION_SHA256,
    ENDPOINT_REFINER_PARAMETER_PREFIXES,
    FRESH_OPTIMIZER_MODE,
    file_sha256,
    prepare_output_directory,
)
from chemvector.visual_svg_data import VisualSVGWebDataset


CHECKPOINT_SCHEMA_VERSION = "bond-group-carrier-set-checkpoint-0.1"
VALIDATION_SCHEMA_VERSION = "bond-group-carrier-set-validation-0.1"
TRAINING_SUMMARY_SCHEMA_VERSION = "bond-group-carrier-set-training-summary-0.1"
DEFAULT_CAPACITY_SHARD = (
    "Data/sdf/visual_svg_v0_1/capacity_100/shards/capacity-000000.tar"
)
SOURCE_BOND_GROUP_CHECKPOINT = Path(
    "outputs/checkpoints/"
    "chemvector-v093-bond-group-set-capacity100-cold-full-step7000/"
    "step-007000.pt"
)
SOURCE_BOND_GROUP_SHA256 = (
    "661c84fbd5decece785da8ded3ed794bc59c6a9e6940d8a4d53d61787730fe39"
)
SOURCE_BOND_GROUP_MAX_MEMBERS = 32
DEFAULT_MAX_MEMBERS = 143
EXPANDED_MEMBER_COUNT_BIAS = -1.0e4
EXPANDED_MEMBER_COUNT_KEYS = (
    "member_count_head.2.weight",
    "member_count_head.2.bias",
)
DEFAULT_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v094-bond-group-carrier-set-capacity100-p1"
)
FULL_MODEL_FREEZE_MODE = "all_parameters_trainable"
CARRIER_GEOMETRY_FREEZE_MODE = "carrier_geometry_and_endpoint_refiner_only"
CARRIER_GEOMETRY_PARAMETER_PREFIXES = (
    "carrier_embedding.",
    "carrier_geometry_head.",
    "positive_params_head.",
    "group_stroke_head.",
    *ENDPOINT_REFINER_PARAMETER_PREFIXES,
)
MAPPED_CARRIER_PREFIXES = (
    "carrier_embedding.",
    "carrier_geometry_head.",
    "group_stroke_head.",
)
RANDOM_CARRIER_PREFIXES = (
    "positive_params_head.",
)
TARGET_FIELDS = (
    "types",
    "member_counts",
    "carrier_points",
    "point_mask",
    "positive_params",
    "positive_param_mask",
    "signed_params",
    "signed_param_mask",
    "group_stroke",
    "text_geometry",
    "target_mask",
    "object_count",
)
LOSS_FIELDS = (
    "loss",
    "classification_loss",
    "geometry_loss",
    "carrier_point_loss",
    "positive_param_loss",
    "signed_param_loss",
    "text_geometry_loss",
    "stroke_loss",
    "member_count_loss",
    "cardinality_loss",
    "count_classification_loss",
)


def carrier_set_config(
    *,
    tiny: bool = False,
    queries: int = 192,
    max_members: int = DEFAULT_MAX_MEMBERS,
    local_query_kernel: int = 5,
    endpoint_refine_kernel: int = 25,
) -> BondGroupCarrierSetConfig:
    common = {
        "queries": queries,
        "max_members": max_members,
        "local_query_kernel": local_query_kernel,
        "endpoint_refine_kernel": endpoint_refine_kernel,
        "dropout": 0.0,
    }
    if tiny:
        return BondGroupCarrierSetConfig(
            **common,
            d_model=64,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=128,
            memory_grid_size=8,
        )
    return BondGroupCarrierSetConfig(
        **common,
        d_model=256,
        decoder_layers=4,
        decoder_heads=8,
        decoder_ffn_dim=1024,
        memory_grid_size=48,
    )


def reject_internal_test_metadata(meta: Any, sample_key: str = "") -> None:
    """Reject sealed data by both sample key and arbitrary metadata content."""

    serialized = json.dumps(
        meta,
        ensure_ascii=True,
        sort_keys=True,
        default=str,
    ).lower()
    if "internal_test" in sample_key.lower() or "internal_test" in serialized:
        raise ValueError(f"Refusing sealed internal_test sample: {sample_key}")


def _reject_checkpoint_shards(checkpoint: dict[str, Any], label: str) -> None:
    training_args = checkpoint.get("training_args")
    if not isinstance(training_args, dict):
        raise ValueError(f"{label} checkpoint has no training_args")
    for field in ("train_shards", "validation_shards"):
        shards = training_args.get(field, [])
        if not isinstance(shards, list) or not all(
            isinstance(path, str) for path in shards
        ):
            raise ValueError(f"{label} checkpoint has invalid {field}: {shards!r}")
        forbidden = [path for path in shards if "internal_test" in path.lower()]
        if forbidden:
            raise ValueError(
                f"{label} checkpoint references sealed internal_test shards: "
                + ", ".join(forbidden)
            )


def validate_bond_group_source_checkpoint(checkpoint: dict[str, Any]) -> None:
    expected = {
        "checkpoint_schema_version": BOND_GROUP_CHECKPOINT_SCHEMA_VERSION,
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
        raise ValueError(f"Bond Group Set source metadata differs: {mismatches}")
    for field in ("model", "vision_config", "bond_group_set_config"):
        if not isinstance(checkpoint.get(field), dict):
            raise ValueError(f"Bond Group Set source checkpoint has no {field}")
    for field in ("global_step", "samples_seen"):
        value = checkpoint.get(field)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Bond Group Set source has invalid {field}: {value!r}")
    lineage = checkpoint.get("resume_lineage")
    if not isinstance(lineage, list):
        raise ValueError("Bond Group Set source resume_lineage must be a list")
    _reject_checkpoint_shards(checkpoint, "Bond Group Set source")


def _validate_source_architecture(
    source: BondGroupSetConfig,
    target: BondGroupCarrierSetConfig,
) -> None:
    source_payload = asdict(source)
    target_payload = asdict(target)
    mismatches = {
        name: (source_payload[name], target_payload[name])
        for name in source_payload
        if name != "max_members" and source_payload[name] != target_payload[name]
    }
    if mismatches:
        raise ValueError(
            "Bond Group Set source architecture differs from Carrier Set: "
            + repr(mismatches)
        )
    if source.max_members != SOURCE_BOND_GROUP_MAX_MEMBERS:
        raise ValueError(
            "Bond Group Set source max_members differs from the frozen source: "
            f"checkpoint={source.max_members} expected={SOURCE_BOND_GROUP_MAX_MEMBERS}"
        )
    if target.max_members < source.max_members:
        raise ValueError(
            "Carrier Set max_members cannot shrink the source count head: "
            f"source={source.max_members} target={target.max_members}"
        )


def load_bond_group_source_checkpoint(
    path: Path,
    target_config: BondGroupCarrierSetConfig,
    *,
    expected_sha256: str = SOURCE_BOND_GROUP_SHA256,
) -> tuple[dict[str, Any], ChemVectorConfig, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Bond Group Set source does not exist: {path}")
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise ValueError(
            "Bond Group Set source SHA256 differs: "
            f"checkpoint={digest} frozen={expected_sha256}"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_bond_group_source_checkpoint(checkpoint)
    source_config = BondGroupSetConfig(**checkpoint["bond_group_set_config"])
    _validate_source_architecture(source_config, target_config)
    return checkpoint, ChemVectorConfig(**checkpoint["vision_config"]), digest


def load_bond_group_warmstart_modules(
    model: BondGroupCarrierSetPredictor,
    checkpoint: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Copy every shared tensor and map the old physical member slot 0."""

    source_state = checkpoint["model"]
    target_state = model.state_dict()
    mapped_state = carrier_head_warm_start_state(source_state)
    shared_candidates = set(source_state) & set(target_state)
    source_only = set(source_state) - set(target_state)
    target_only = set(target_state) - set(source_state)
    expected_source_only = {
        "member_embedding.weight",
        "member_geometry_head.0.weight",
        "member_geometry_head.0.bias",
        "member_geometry_head.2.weight",
        "member_geometry_head.2.bias",
        "member_stroke_head.0.weight",
        "member_stroke_head.0.bias",
        "member_stroke_head.2.weight",
        "member_stroke_head.2.bias",
    }
    expected_target_only = set(mapped_state) | {
        name
        for name in target_state
        if name.startswith(RANDOM_CARRIER_PREFIXES)
    }
    if source_only != expected_source_only or target_only != expected_target_only:
        raise ValueError(
            "Bond Group Set to Carrier Set key contract differs: "
            f"source_only={sorted(source_only)} target_only={sorted(target_only)}"
        )
    shape_mismatches = {
        name: (tuple(source_state[name].shape), tuple(target_state[name].shape))
        for name in shared_candidates
        if source_state[name].shape != target_state[name].shape
        and name not in EXPANDED_MEMBER_COUNT_KEYS
    }
    shape_mismatches.update(
        {
            name: (tuple(value.shape), tuple(target_state[name].shape))
            for name, value in mapped_state.items()
            if value.shape != target_state[name].shape
        }
    )
    if shape_mismatches:
        raise ValueError(
            "Bond Group Set warm-start tensor shapes differ: "
            + repr(shape_mismatches)
        )
    expanded_keys = sorted(
        name
        for name in EXPANDED_MEMBER_COUNT_KEYS
        if source_state[name].shape != target_state[name].shape
    )
    if expanded_keys and set(expanded_keys) != set(EXPANDED_MEMBER_COUNT_KEYS):
        raise ValueError(
            "Member count output weight and bias must expand together: "
            + repr(expanded_keys)
        )
    expanded_state: dict[str, torch.Tensor] = {}
    if expanded_keys:
        source_weight = source_state["member_count_head.2.weight"]
        source_bias = source_state["member_count_head.2.bias"]
        target_weight = target_state["member_count_head.2.weight"]
        target_bias = target_state["member_count_head.2.bias"]
        if (
            source_weight.ndim != 2
            or target_weight.ndim != 2
            or source_bias.ndim != 1
            or target_bias.ndim != 1
            or source_weight.size(0) != source_bias.size(0)
            or target_weight.size(0) != target_bias.size(0)
            or source_weight.size(1) != target_weight.size(1)
            or target_weight.size(0) <= source_weight.size(0)
        ):
            raise ValueError(
                "Member count output head cannot be expanded safely: "
                f"source_weight={tuple(source_weight.shape)} "
                f"source_bias={tuple(source_bias.shape)} "
                f"target_weight={tuple(target_weight.shape)} "
                f"target_bias={tuple(target_bias.shape)}"
            )
        expanded_weight = torch.zeros_like(target_weight)
        expanded_bias = torch.full_like(target_bias, EXPANDED_MEMBER_COUNT_BIAS)
        source_rows = source_weight.size(0)
        expanded_weight[:source_rows] = source_weight.to(expanded_weight.device)
        expanded_bias[:source_rows] = source_bias.to(expanded_bias.device)
        expanded_state = {
            "member_count_head.2.weight": expanded_weight,
            "member_count_head.2.bias": expanded_bias,
        }
    shared_keys = sorted(shared_candidates - set(expanded_keys))
    random_keys = sorted(
        name for name in target_state if name.startswith(RANDOM_CARRIER_PREFIXES)
    )
    initial_random = {name: target_state[name].detach().clone() for name in random_keys}
    complete_state = {name: value for name, value in target_state.items()}
    complete_state.update({name: source_state[name] for name in shared_keys})
    complete_state.update(mapped_state)
    complete_state.update(expanded_state)
    model.load_state_dict(complete_state, strict=True)
    loaded_state = model.state_dict()
    for name in shared_keys:
        if not torch.equal(
            loaded_state[name].detach().cpu(),
            source_state[name].detach().cpu(),
        ):
            raise RuntimeError(f"Shared warm-start tensor was not copied exactly: {name}")
    for name, value in mapped_state.items():
        if not torch.equal(
            loaded_state[name].detach().cpu(),
            value.detach().cpu(),
        ):
            raise RuntimeError(f"Mapped carrier tensor was not copied exactly: {name}")
    for name, value in expanded_state.items():
        if not torch.equal(
            loaded_state[name].detach().cpu(),
            value.detach().cpu(),
        ):
            raise RuntimeError(f"Expanded count tensor was not loaded exactly: {name}")
    for name, value in initial_random.items():
        if not torch.equal(loaded_state[name], value):
            raise RuntimeError(f"New carrier tensor was unexpectedly overwritten: {name}")
    return shared_keys, sorted(mapped_state), expanded_keys, random_keys


def bond_group_warmstart_provenance(
    path: Path,
    sha256: str,
    checkpoint: dict[str, Any],
    shared_keys: list[str],
    mapped_keys: list[str],
    expanded_keys: list[str],
    random_keys: list[str],
    *,
    target_max_members: int,
) -> dict[str, Any]:
    source_lineage = list(checkpoint["resume_lineage"])
    source_lineage.append(
        {
            "checkpoint": str(path),
            "checkpoint_resolved": str(path.resolve()),
            "sha256": sha256,
            "global_step": int(checkpoint["global_step"]),
            "samples_seen": int(checkpoint["samples_seen"]),
        }
    )
    expansion = None
    if expanded_keys:
        expansion = {
            "source_max_members": int(
                checkpoint["bond_group_set_config"]["max_members"]
            ),
            "target_max_members": int(target_max_members),
            "copied_rows": [0, SOURCE_BOND_GROUP_MAX_MEMBERS],
            "new_rows": [
                SOURCE_BOND_GROUP_MAX_MEMBERS + 1,
                int(target_max_members),
            ],
            "new_weight_value": 0.0,
            "new_bias_value": EXPANDED_MEMBER_COUNT_BIAS,
            "activation_status": (
                "inactive_during_geometry_capacity_training; reset new rows and "
                "unfreeze member_count_head before count-capacity training"
            ),
        }
    return {
        "mode": "bond_group_set_slot0_warmstart",
        "checkpoint": str(path),
        "checkpoint_resolved": str(path.resolve()),
        "checkpoint_sha256": sha256,
        "checkpoint_schema_version": checkpoint["checkpoint_schema_version"],
        "model_type": checkpoint["model_type"],
        "global_step": int(checkpoint["global_step"]),
        "samples_seen": int(checkpoint["samples_seen"]),
        "source_vision_checkpoint": checkpoint["source_vision_checkpoint"],
        "source_vision_checkpoint_sha256": checkpoint[
            "source_vision_checkpoint_sha256"
        ],
        "source_lineage": source_lineage,
        "shared_keys": list(shared_keys),
        "mapped_carrier_keys": list(mapped_keys),
        "expanded_keys": list(expanded_keys),
        "expanded_member_count_head": expansion,
        "randomly_initialized_keys": list(random_keys),
    }


def configure_trainable_parameters(
    model: BondGroupCarrierSetPredictor,
    *,
    carrier_geometry_only: bool,
) -> tuple[list[str], int]:
    names: list[str] = []
    count = 0
    for name, parameter in model.named_parameters():
        selected = not carrier_geometry_only or name.startswith(
            CARRIER_GEOMETRY_PARAMETER_PREFIXES
        )
        parameter.requires_grad_(selected)
        if selected:
            names.append(name)
            count += parameter.numel()
    if not names:
        raise ValueError("Carrier Set training selected no parameters")
    return names, count


def make_loader(
    shards: list[str],
    *,
    config: BondGroupCarrierSetConfig,
    batch_size: int,
    workers: int,
    image_size: int,
    shuffle: bool,
    limit: int | None = None,
) -> DataLoader:
    return DataLoader(
        VisualSVGWebDataset(
            shards,
            shardshuffle=shuffle,
            sample_shuffle_buffer=100 if shuffle else 0,
            limit=limit,
        ),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=BondGroupCarrierSetCollator(
            image_size=image_size,
            query_count=config.queries,
            max_members=config.max_members,
        ),
    )


def targets_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in TARGET_FIELDS
    }


def _validate_batch_metadata(batch: dict[str, Any]) -> None:
    for key, meta in zip(batch["keys"], batch["meta"]):
        reject_internal_test_metadata(meta, key)


@torch.inference_mode()
def validate(
    model: BondGroupCarrierSetPredictor,
    loader,
    device: torch.device,
) -> dict[str, float | int | str | bool]:
    model.eval()
    totals = {name: 0.0 for name in LOSS_FIELDS}
    white_totals = {name: 0.0 for name in LOSS_FIELDS}
    samples = 0
    matched_objects = 0
    matched_type_correct = 0
    matched_member_count_correct = 0
    count_absolute_error = 0.0
    count_exact = 0
    started = time.time()
    for batch in loader:
        _validate_batch_metadata(batch)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        targets = targets_to_device(batch, device)
        with autocast_context(device):
            output = model(pixels)
            losses = bond_group_carrier_set_loss(
                output,
                targets,
                image_size=pixels.size(-1),
            )
            white_output = model(torch.ones_like(pixels))
            white_losses = bond_group_carrier_set_loss(
                white_output,
                targets,
                image_size=pixels.size(-1),
            )
        batch_size = pixels.size(0)
        samples += batch_size
        for name in LOSS_FIELDS:
            totals[name] += float(losses[name]) * batch_size
            white_totals[name] += float(white_losses[name]) * batch_size
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
                batch_index, query_indices
            ].argmax(dim=-1)
            expected_types = targets["types"][batch_index, target_indices]
            predicted_members = output["member_count_logits"][
                batch_index, query_indices
            ].argmax(dim=-1)
            expected_members = targets["member_counts"][batch_index, target_indices]
            matched_objects += len(query_indices)
            matched_type_correct += int((predicted_types == expected_types).sum())
            matched_member_count_correct += int(
                (predicted_members == expected_members).sum()
            )
    if not samples:
        raise RuntimeError("Carrier Set validation loader produced no samples")
    mean_loss = totals["loss"] / samples
    mean_white_loss = white_totals["loss"] / samples
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "samples": samples,
        **{name: value / samples for name, value in totals.items()},
        "white_image_losses": {
            name: value / samples for name, value in white_totals.items()
        },
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
    model: BondGroupCarrierSetPredictor,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    samples_seen: int,
    args: argparse.Namespace,
    validation: dict[str, Any] | None,
    initialization_provenance: dict[str, Any],
    resume_lineage: list[dict[str, Any]],
    trainable_parameter_names: list[str],
    trainable_parameter_count: int,
    freeze_mode: str,
) -> None:
    atomic_torch_save(
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
            "model_type": model.model_type,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "vision_config": asdict(model.vision_config),
            "bond_group_carrier_set_config": asdict(model.config),
            "global_step": int(global_step),
            "samples_seen": int(samples_seen),
            "training_args": vars(args),
            "validation": validation,
            "source_bond_group_checkpoint": initialization_provenance[
                "checkpoint"
            ],
            "source_bond_group_checkpoint_sha256": initialization_provenance[
                "checkpoint_sha256"
            ],
            "source_bond_group_lineage": initialization_provenance[
                "source_lineage"
            ],
            "source_vision_checkpoint": initialization_provenance[
                "source_vision_checkpoint"
            ],
            "source_vision_checkpoint_sha256": initialization_provenance[
                "source_vision_checkpoint_sha256"
            ],
            "initialization_provenance": initialization_provenance,
            "resume_lineage": list(resume_lineage),
            "freeze_mode": freeze_mode,
            "optimizer_mode": FRESH_OPTIMIZER_MODE,
            "trainable_parameter_names": list(trainable_parameter_names),
            "trainable_parameter_count": int(trainable_parameter_count),
            "uses_internal_test": False,
        },
        path,
    )


def validate_resume_checkpoint(checkpoint: dict[str, Any]) -> None:
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
        "model_type": BondGroupCarrierSetPredictor.model_type,
        "source_bond_group_checkpoint_sha256": SOURCE_BOND_GROUP_SHA256,
        "source_vision_checkpoint_sha256": CAPACITY_VISION_SHA256,
        "optimizer_mode": FRESH_OPTIMIZER_MODE,
        "uses_internal_test": False,
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Carrier Set resume metadata differs: {mismatches}")
    for field in (
        "model",
        "vision_config",
        "bond_group_carrier_set_config",
        "training_args",
        "initialization_provenance",
    ):
        if not isinstance(checkpoint.get(field), dict):
            raise ValueError(f"Carrier Set resume checkpoint has no {field}")
    for field in ("global_step", "samples_seen"):
        value = checkpoint.get(field)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Carrier Set resume has invalid {field}: {value!r}")
    if not isinstance(checkpoint.get("resume_lineage"), list):
        raise ValueError("Carrier Set resume_lineage must be a list")
    if not isinstance(checkpoint.get("source_bond_group_lineage"), list):
        raise ValueError("Carrier Set source_bond_group_lineage must be a list")
    provenance = checkpoint["initialization_provenance"]
    if provenance.get("checkpoint_sha256") != SOURCE_BOND_GROUP_SHA256:
        raise ValueError("Carrier Set initialization provenance SHA256 differs")
    if checkpoint["source_bond_group_lineage"] != provenance.get("source_lineage"):
        raise ValueError("Carrier Set frozen source lineage differs")
    config_payload = checkpoint["bond_group_carrier_set_config"]
    target_max_members = config_payload.get("max_members")
    if not isinstance(target_max_members, int) or target_max_members < 1:
        raise ValueError("Carrier Set resume has invalid max_members")
    expected_expanded_keys = (
        sorted(EXPANDED_MEMBER_COUNT_KEYS)
        if target_max_members > SOURCE_BOND_GROUP_MAX_MEMBERS
        else []
    )
    if provenance.get("expanded_keys") != expected_expanded_keys:
        raise ValueError("Carrier Set expanded count-head key provenance differs")
    expansion = provenance.get("expanded_member_count_head")
    if expected_expanded_keys:
        expected_expansion = {
            "source_max_members": SOURCE_BOND_GROUP_MAX_MEMBERS,
            "target_max_members": target_max_members,
            "copied_rows": [0, SOURCE_BOND_GROUP_MAX_MEMBERS],
            "new_rows": [SOURCE_BOND_GROUP_MAX_MEMBERS + 1, target_max_members],
            "new_weight_value": 0.0,
            "new_bias_value": EXPANDED_MEMBER_COUNT_BIAS,
            "activation_status": (
                "inactive_during_geometry_capacity_training; reset new rows and "
                "unfreeze member_count_head before count-capacity training"
            ),
        }
        if expansion != expected_expansion:
            raise ValueError("Carrier Set count-head expansion provenance differs")
    elif expansion is not None:
        raise ValueError("Carrier Set has unexpected count-head expansion provenance")
    _reject_checkpoint_shards(checkpoint, "Carrier Set resume")


def load_resume_checkpoint(
    path: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], BondGroupCarrierSetConfig, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Carrier Set resume checkpoint does not exist: {path}")
    digest = file_sha256(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_resume_checkpoint(checkpoint)
    config = BondGroupCarrierSetConfig(
        **checkpoint["bond_group_carrier_set_config"]
    )
    requested = carrier_set_config(
        tiny=args.tiny,
        queries=args.queries,
        max_members=args.max_members,
        local_query_kernel=args.local_query_kernel,
        endpoint_refine_kernel=args.endpoint_refine_kernel,
    )
    if config != requested:
        raise ValueError(
            "Carrier Set resume architecture differs from CLI: "
            f"checkpoint={asdict(config)} cli={asdict(requested)}"
        )
    previous_image_size = int(checkpoint["training_args"].get("image_size", 768))
    if previous_image_size != args.image_size:
        raise ValueError(
            "Carrier Set resume image size differs from CLI: "
            f"checkpoint={previous_image_size} cli={args.image_size}"
        )
    return checkpoint, config, digest


def extend_resume_lineage(
    checkpoint: dict[str, Any],
    path: Path,
    sha256: str,
) -> list[dict[str, Any]]:
    lineage = list(checkpoint["resume_lineage"])
    lineage.append(
        {
            "checkpoint": str(path),
            "checkpoint_resolved": str(path.resolve()),
            "sha256": sha256,
            "global_step": int(checkpoint["global_step"]),
            "samples_seen": int(checkpoint["samples_seen"]),
        }
    )
    return lineage


def write_json(path: Path, payload: dict[str, Any]) -> None:
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
        "--init-bond-group-checkpoint",
        default=str(SOURCE_BOND_GROUP_CHECKPOINT),
    )
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--carrier-geometry-only", action="store_true")
    parser.add_argument("--queries", type=int, default=192)
    parser.add_argument("--max-members", type=int, default=DEFAULT_MAX_MEMBERS)
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
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser


def _device_from_args(args: argparse.Namespace) -> torch.device:
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if args.device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = build_parser().parse_args()
    if min(
        args.batch_size,
        args.max_steps,
        args.validate_every,
        args.save_every,
        args.image_size,
        args.queries,
        args.max_members,
        args.local_query_kernel,
        args.endpoint_refine_kernel,
    ) < 1:
        raise ValueError("Positive training dimensions are required")
    if args.workers < 0 or args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Workers/learning rate/weight decay are invalid")
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
    device = _device_from_args(args)
    resume_lineage: list[dict[str, Any]] = []
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        resume, config, resume_sha256 = load_resume_checkpoint(resume_path, args)
        vision_config = ChemVectorConfig(**resume["vision_config"])
        model = BondGroupCarrierSetPredictor(vision_config, config).to(device)
        model.load_state_dict(resume["model"], strict=True)
        global_step = int(resume["global_step"])
        samples_seen = int(resume["samples_seen"])
        initialization_provenance = resume["initialization_provenance"]
        resume_lineage = extend_resume_lineage(
            resume,
            resume_path,
            resume_sha256,
        )
        if args.max_steps <= global_step:
            raise ValueError(
                "--max-steps is the final global step and must exceed the resume "
                f"step {global_step}"
            )
    else:
        config = carrier_set_config(
            tiny=args.tiny,
            queries=args.queries,
            max_members=args.max_members,
            local_query_kernel=args.local_query_kernel,
            endpoint_refine_kernel=args.endpoint_refine_kernel,
        )
        source_path = Path(args.init_bond_group_checkpoint)
        source, vision_config, source_sha256 = load_bond_group_source_checkpoint(
            source_path,
            config,
        )
        model = BondGroupCarrierSetPredictor(vision_config, config).to(device)
        shared_keys, mapped_keys, expanded_keys, random_keys = (
            load_bond_group_warmstart_modules(model, source)
        )
        initialization_provenance = bond_group_warmstart_provenance(
            source_path,
            source_sha256,
            source,
            shared_keys,
            mapped_keys,
            expanded_keys,
            random_keys,
            target_max_members=config.max_members,
        )
        global_step = 0
        samples_seen = 0
        del source

    freeze_mode = (
        CARRIER_GEOMETRY_FREEZE_MODE
        if args.carrier_geometry_only
        else FULL_MODEL_FREEZE_MODE
    )
    trainable_names, trainable_count = configure_trainable_parameters(
        model,
        carrier_geometry_only=args.carrier_geometry_only,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
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
    last_validation: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    while global_step < args.max_steps:
        batches = 0
        for batch in train_loader:
            batches += 1
            _validate_batch_metadata(batch)
            model.train()
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            targets = targets_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                output = model(pixels)
                losses = bond_group_carrier_set_loss(
                    output,
                    targets,
                    image_size=args.image_size,
                )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.gradient_clip,
            )
            optimizer.step()
            global_step += 1
            samples_seen += pixels.size(0)

            if global_step % args.validate_every == 0 or global_step >= args.max_steps:
                last_validation = validate(model, validation_loader, device)
                history.append({"global_step": global_step, "metrics": last_validation})
                print(
                    f"step={global_step} loss={last_validation['loss']:.6f} "
                    f"carrier={last_validation['geometry_loss']:.6f} "
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
                        initialization_provenance=initialization_provenance,
                        resume_lineage=resume_lineage,
                        trainable_parameter_names=trainable_names,
                        trainable_parameter_count=trainable_count,
                        freeze_mode=freeze_mode,
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
                    initialization_provenance=initialization_provenance,
                    resume_lineage=resume_lineage,
                    trainable_parameter_names=trainable_names,
                    trainable_parameter_count=trainable_count,
                    freeze_mode=freeze_mode,
                )
            if global_step >= args.max_steps:
                break
        if not batches:
            raise RuntimeError("Carrier Set training loader produced no samples")

    assert last_validation is not None
    save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        global_step=global_step,
        samples_seen=samples_seen,
        args=args,
        validation=last_validation,
        initialization_provenance=initialization_provenance,
        resume_lineage=resume_lineage,
        trainable_parameter_names=trainable_names,
        trainable_parameter_count=trainable_count,
        freeze_mode=freeze_mode,
    )
    write_json(output_dir / "validation.json", last_validation)
    write_json(
        output_dir / "summary.json",
        {
            "schema_version": TRAINING_SUMMARY_SCHEMA_VERSION,
            "model_type": model.model_type,
            "global_step": global_step,
            "samples_seen": samples_seen,
            "best_validation_loss": best_loss,
            "history": history,
            "config": asdict(config),
            "initialization_provenance": initialization_provenance,
            "resume_lineage": resume_lineage,
            "freeze_mode": freeze_mode,
            "optimizer_mode": FRESH_OPTIMIZER_MODE,
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": trainable_count,
            "uses_internal_test": False,
        },
    )


if __name__ == "__main__":
    main()
