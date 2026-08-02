#!/usr/bin/env python3
"""Train v0.9.5 group membership over frozen flat primitive proposals."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import glob
import inspect
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.utils.data import DataLoader

from chemvector.bond_group_proposal_membership import (
    BondGroupProposalMembershipConfig,
    BondGroupProposalMembershipPredictor,
    bond_group_proposal_membership_loss,
    decode_bond_group_proposal_membership,
)
from chemvector.bond_group_proposal_membership_data import (
    BondGroupProposalMembershipCollator,
)
from chemvector.evaluate_primitive_set import load_primitive_set_model
from chemvector.primitive_set import PrimitiveSetPredictor
from chemvector.train_bond_group_set import atomic_torch_save
from chemvector.train_primitive_set import file_sha256, prepare_output_directory
from chemvector.visual_svg_data import VisualSVGWebDataset

try:
    from chemvector.bond_group_proposal_membership_pipeline import (
        PROPOSAL_DESCRIPTOR_DIM,
        prepare_proposal_membership_pipeline,
    )
except ModuleNotFoundError:
    # The pipeline is developed independently.  Importing this module remains
    # useful for checkpoint and freeze-contract tests while that file lands.
    PROPOSAL_DESCRIPTOR_DIM = 23
    prepare_proposal_membership_pipeline = None


CHECKPOINT_SCHEMA_VERSION = "bond-group-proposal-membership-checkpoint-0.1"
VALIDATION_SCHEMA_VERSION = "bond-group-proposal-membership-validation-0.1"
TRAINING_SUMMARY_SCHEMA_VERSION = (
    "bond-group-proposal-membership-training-summary-0.1"
)
SEQUENCE_SCHEMA_VERSION = "visual-svg-0.2"
DEFAULT_CAPACITY_SHARD = (
    "Data/sdf/visual_svg_v0_1/capacity_100/shards/capacity-000000.tar"
)
SOURCE_PRIMITIVE_CHECKPOINT = Path(
    "outputs/checkpoints/"
    "chemvector-v092-primitive-set-capacity100-endpoint-only-lr1e5-step11000/"
    "step-011000.pt"
)
SOURCE_PRIMITIVE_SHA256 = (
    "0678a4bd4754ad94dfb084a7f447768bf4ee5ef2d31f0f79e0ab8884f1ef041a"
)
DEFAULT_OUTPUT_DIR = (
    "outputs/checkpoints/"
    "chemvector-v095-bond-group-proposal-membership-capacity100-step500"
)
LOSS_FIELDS = (
    "loss",
    "group_type_loss",
    "member_count_loss",
    "group_count_loss",
    "membership_positive_loss",
    "membership_negative_loss",
    "membership_dice_loss",
    "unmatched_membership_loss",
    "exclusivity_loss",
    "proposal_keep_loss",
    "proposal_type_loss",
)
WEIGHT_TARGET_FIELDS = ("group_weights",)


def membership_config(
    *,
    proposal_feature_dim: int,
    tiny: bool = False,
    group_queries: int = 192,
    max_members: int = 143,
) -> BondGroupProposalMembershipConfig:
    common = {
        "proposal_feature_dim": proposal_feature_dim,
        "proposal_descriptor_dim": PROPOSAL_DESCRIPTOR_DIM,
        "flat_type_classes": 5,
        "group_queries": group_queries,
        "max_members": max_members,
        "dropout": 0.0,
    }
    if tiny:
        return BondGroupProposalMembershipConfig(
            **common,
            d_model=32,
            proposal_encoder_layers=1,
            group_decoder_layers=1,
            heads=4,
            ffn_dim=64,
        )
    return BondGroupProposalMembershipConfig(
        **common,
        d_model=192,
        proposal_encoder_layers=2,
        group_decoder_layers=2,
        heads=6,
        ffn_dim=768,
    )


def reject_internal_test_metadata(meta: Any, sample_key: str = "") -> None:
    serialized = json.dumps(
        meta,
        ensure_ascii=True,
        sort_keys=True,
        default=str,
    ).lower()
    if "internal_test" in sample_key.lower() or "internal_test" in serialized:
        raise ValueError(f"Refusing sealed internal_test sample: {sample_key}")


def reject_internal_test_shards(shards: list[str]) -> None:
    forbidden = [path for path in shards if "internal_test" in path.lower()]
    if forbidden:
        raise ValueError(
            "Proposal Membership training must not read sealed internal_test shards: "
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


def _reject_checkpoint_shards(checkpoint: dict[str, Any]) -> None:
    training_args = checkpoint.get("training_args", {})
    if not isinstance(training_args, dict):
        raise ValueError("Frozen primitive checkpoint has invalid training_args")
    for field in ("train_shards", "validation_shards"):
        values = training_args.get(field, [])
        if values is None:
            continue
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise ValueError(f"Frozen primitive checkpoint has invalid {field}")
        reject_internal_test_shards(values)


def freeze_primitive_model(model: PrimitiveSetPredictor) -> int:
    """Put the source model in permanent inference-only mode."""

    model.eval()
    count = 0
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
        count += parameter.numel()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen primitive model still has trainable parameters")
    return count


def validate_frozen_primitive_checkpoint(checkpoint: dict[str, Any]) -> None:
    expected = {
        "checkpoint_schema_version": "primitive-set-checkpoint-0.1",
        "sequence_schema_version": "visual-svg-0.1",
        "model_type": PrimitiveSetPredictor.model_type,
    }
    mismatches = {
        field: (checkpoint.get(field), value)
        for field, value in expected.items()
        if checkpoint.get(field) != value
    }
    if mismatches:
        raise ValueError(f"Frozen primitive checkpoint metadata differs: {mismatches}")
    for field in ("model", "vision_config", "primitive_set_config"):
        if not isinstance(checkpoint.get(field), dict):
            raise ValueError(f"Frozen primitive checkpoint has no {field}")
    if checkpoint.get("uses_internal_test") not in (None, False):
        raise ValueError("Frozen primitive checkpoint used internal_test")
    _reject_checkpoint_shards(checkpoint)


def load_frozen_primitive_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    device: torch.device,
) -> tuple[PrimitiveSetPredictor, dict[str, Any], dict[str, Any]]:
    """Strictly load, verify and freeze one exact primitive checkpoint."""

    if not path.is_file():
        raise FileNotFoundError(f"Frozen primitive checkpoint does not exist: {path}")
    if len(expected_sha256) != 64:
        raise ValueError("Expected primitive checkpoint SHA256 must contain 64 hex chars")
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise ValueError(
            "Frozen primitive checkpoint SHA256 differs: "
            f"expected {expected_sha256}, received {digest}"
        )
    model, checkpoint = load_primitive_set_model(path, device)
    validate_frozen_primitive_checkpoint(checkpoint)
    frozen_parameter_count = freeze_primitive_model(model)
    provenance = {
        "checkpoint": str(path),
        "checkpoint_resolved": str(path.resolve()),
        "checkpoint_sha256": digest,
        "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
        "checkpoint_samples_seen": int(checkpoint.get("samples_seen", 0)),
        "model_type": model.model_type,
        "primitive_set_config": asdict(model.config),
        "vision_config": asdict(model.vision_config),
        "frozen_parameter_count": frozen_parameter_count,
        "requires_grad": False,
        "eval_mode": True,
    }
    return model, checkpoint, provenance


def make_loader(
    shards: list[str],
    *,
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
        collate_fn=BondGroupProposalMembershipCollator(image_size=image_size),
    )


def _validate_batch_metadata(batch: dict[str, Any]) -> None:
    keys = batch.get("keys")
    metadata = batch.get("meta")
    if not isinstance(keys, list) or not isinstance(metadata, list):
        raise ValueError("Proposal Membership batch lacks keys or metadata")
    if len(keys) != len(metadata):
        raise ValueError("Proposal Membership keys and metadata differ")
    for key, meta in zip(keys, metadata):
        reject_internal_test_metadata(meta, str(key))


def _device_tensor_dict(
    values: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        name: (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for name, value in values.items()
    }


def autocast_context(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


def prepare_pipeline_batch(
    batch: dict[str, Any],
    flat_output: dict[str, torch.Tensor],
    primitive_model: PrimitiveSetPredictor,
    config: BondGroupProposalMembershipConfig,
    collator: BondGroupProposalMembershipCollator,
) -> dict[str, Any]:
    if prepare_proposal_membership_pipeline is None:
        raise RuntimeError(
            "chemvector.bond_group_proposal_membership_pipeline is unavailable"
        )
    proposal_batch = collator.attach_flat_output(batch, flat_output)
    prepared = prepare_proposal_membership_pipeline(
        proposal_batch,
        flat_output,
        primitive_model.reference_points.detach(),
        group_slots=config.group_queries,
    )
    required = {"model_inputs", "targets", "known_counts", "oracle_coverage"}
    missing = sorted(required - prepared.keys())
    if missing:
        raise ValueError(f"Proposal pipeline result is missing fields: {missing}")
    return prepared


def compute_membership_loss(
    output: dict[str, torch.Tensor],
    targets: dict[str, Any],
    proposal_mask: torch.Tensor,
) -> dict[str, Any]:
    """Forward every provided weighting tensor or fail explicitly."""

    parameters = inspect.signature(
        bond_group_proposal_membership_loss
    ).parameters
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs = {}
    for field in WEIGHT_TARGET_FIELDS:
        if field not in targets:
            continue
        if field not in parameters and not accepts_kwargs:
            raise RuntimeError(
                f"Pipeline supplied {field}, but the core membership loss does "
                "not consume it; refusing to silently discard supervision weights"
            )
        kwargs[field] = targets[field]
    return bond_group_proposal_membership_loss(
        output,
        targets,
        proposal_mask,
        **kwargs,
    )


def oracle_keep_type_output(
    output: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Replace only proposal keep/type logits for the isolated validation mode."""

    result = dict(output)
    keep = torch.full_like(output["proposal_keep_logits"], -20.0)
    positive = targets["proposal_keep"].bool() & proposal_mask.bool()
    keep[positive] = 20.0
    corrected = torch.full_like(output["proposal_type_logits"], -20.0)
    type_targets = targets["proposal_types"].long().clamp(
        min=0,
        max=corrected.size(-1) - 1,
    )
    corrected.scatter_(-1, type_targets.unsqueeze(-1), 20.0)
    corrected = corrected.masked_fill(~proposal_mask.bool().unsqueeze(-1), -20.0)
    corrected[..., 0] = torch.where(
        proposal_mask.bool(),
        corrected[..., 0],
        corrected[..., 0].new_full((), 20.0),
    )
    result["proposal_keep_logits"] = keep
    result["proposal_type_logits"] = corrected
    return result


def _partition_counts(
    decoded: list[list[dict[str, Any]]],
    targets: dict[str, torch.Tensor],
    proposal_mask: torch.Tensor,
) -> dict[str, int]:
    totals = {
        "samples": 0,
        "count_exact": 0,
        "count_absolute_error": 0,
        "predicted_groups": 0,
        "expected_groups": 0,
        "matched_groups": 0,
        "type_correct": 0,
        "group_exact": 0,
        "sample_exact": 0,
        "membership_true_positive": 0,
        "membership_false_positive": 0,
        "membership_false_negative": 0,
        "valid_groups": 0,
    }
    for batch_index, rows in enumerate(decoded):
        valid_proposals = proposal_mask[batch_index].bool()
        target_indices = torch.where(targets["group_mask"][batch_index].bool())[0]
        target_types = targets["group_types"][batch_index, target_indices].long()
        target_membership = targets["group_membership"][
            batch_index,
            target_indices,
        ].bool() & valid_proposals.unsqueeze(0)
        predicted_membership = torch.zeros(
            (len(rows), proposal_mask.size(1)),
            dtype=torch.bool,
            device=proposal_mask.device,
        )
        predicted_types = []
        for row_index, row in enumerate(rows):
            predicted_types.append(int(row["type"]))
            indices = [int(value) for value in row.get("proposal_indices", [])]
            if any(index < 0 or index >= proposal_mask.size(1) for index in indices):
                raise ValueError("Decoded proposal membership index is out of range")
            if indices:
                predicted_membership[row_index, indices] = True
            totals["valid_groups"] += int(bool(row.get("valid_membership", False)))
        predicted_types_tensor = torch.tensor(
            predicted_types,
            device=proposal_mask.device,
            dtype=torch.long,
        )
        predicted_count = len(rows)
        expected_count = len(target_indices)
        totals["samples"] += 1
        totals["predicted_groups"] += predicted_count
        totals["expected_groups"] += expected_count
        totals["count_absolute_error"] += abs(predicted_count - expected_count)
        totals["count_exact"] += int(predicted_count == expected_count)
        sample_exact = predicted_count == expected_count

        matched_predicted: set[int] = set()
        matched_target: set[int] = set()
        exact_matches = 0
        if predicted_count and expected_count:
            intersections = (
                predicted_membership[:, None, :]
                & target_membership[None, :, :]
            ).sum(dim=-1).float()
            unions = (
                predicted_membership[:, None, :]
                | target_membership[None, :, :]
            ).sum(dim=-1).float().clamp_min(1.0)
            membership_cost = 1.0 - intersections / unions
            type_cost = (
                predicted_types_tensor[:, None] != target_types[None, :]
            ).float()
            rows_index, columns_index = linear_sum_assignment(
                (membership_cost + type_cost).cpu().numpy()
            )
            for predicted_index, target_index in zip(rows_index, columns_index):
                predicted_index = int(predicted_index)
                target_index = int(target_index)
                matched_predicted.add(predicted_index)
                matched_target.add(target_index)
                predicted_mask = predicted_membership[predicted_index]
                target_mask = target_membership[target_index]
                totals["membership_true_positive"] += int(
                    (predicted_mask & target_mask).sum()
                )
                totals["membership_false_positive"] += int(
                    (predicted_mask & ~target_mask).sum()
                )
                totals["membership_false_negative"] += int(
                    (~predicted_mask & target_mask).sum()
                )
                type_correct = bool(
                    predicted_types_tensor[predicted_index]
                    == target_types[target_index]
                )
                exact = type_correct and torch.equal(predicted_mask, target_mask)
                totals["matched_groups"] += 1
                totals["type_correct"] += int(type_correct)
                totals["group_exact"] += int(exact)
                exact_matches += int(exact)
        for predicted_index in set(range(predicted_count)) - matched_predicted:
            totals["membership_false_positive"] += int(
                predicted_membership[predicted_index].sum()
            )
        for target_index in set(range(expected_count)) - matched_target:
            totals["membership_false_negative"] += int(
                target_membership[target_index].sum()
            )
        sample_exact = sample_exact and exact_matches == expected_count
        totals["sample_exact"] += int(sample_exact)
    return totals


def _merge_integer_totals(target: dict[str, int], values: dict[str, int]) -> None:
    for name, value in values.items():
        target[name] = target.get(name, 0) + int(value)


def _finalize_partition_metrics(totals: dict[str, int]) -> dict[str, Any]:
    samples = max(totals.get("samples", 0), 1)
    predicted = totals.get("predicted_groups", 0)
    expected = totals.get("expected_groups", 0)
    matched = totals.get("matched_groups", 0)
    tp = totals.get("membership_true_positive", 0)
    fp = totals.get("membership_false_positive", 0)
    fn = totals.get("membership_false_negative", 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        **totals,
        "count_exact_rate": totals.get("count_exact", 0) / samples,
        "count_mae": totals.get("count_absolute_error", 0) / samples,
        "group_precision": totals.get("group_exact", 0) / max(predicted, 1),
        "group_recall": totals.get("group_exact", 0) / max(expected, 1),
        "matched_type_accuracy": totals.get("type_correct", 0) / max(matched, 1),
        "sample_exact_rate": totals.get("sample_exact", 0) / samples,
        "membership_precision": precision,
        "membership_recall": recall,
        "membership_f1": 2.0 * precision * recall / max(precision + recall, 1e-8),
        "valid_group_rate": totals.get("valid_groups", 0) / max(predicted, 1),
    }


def _merge_coverage(target: dict[str, Any], values: Any) -> None:
    if not isinstance(values, dict):
        return
    for name, value in values.items():
        if isinstance(value, dict):
            child = target.setdefault(name, {})
            if isinstance(child, dict):
                _merge_coverage(child, value)
        elif isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value.float()).all()):
                raise ValueError(f"Oracle coverage {name} contains non-finite values")
            entry = target.setdefault(name, {"sum": 0.0, "count": 0})
            if not isinstance(entry, dict):
                raise ValueError(f"Oracle coverage accumulator collision at {name}")
            entry["sum"] += float(value.float().sum())
            entry["count"] += int(value.numel())
        elif isinstance(value, bool):
            target[name] = bool(target.get(name, False) or value)
        elif isinstance(value, (int, float)):
            entry = target.setdefault(name, {"sum": 0.0, "count": 0})
            if not isinstance(entry, dict):
                raise ValueError(f"Oracle coverage accumulator collision at {name}")
            entry["sum"] += float(value)
            entry["count"] += 1


def _finalize_coverage(values: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name, value in values.items():
        if isinstance(value, dict) and set(value) == {"sum", "count"}:
            count = int(value["count"])
            result[name] = {
                "sum": float(value["sum"]),
                "count": count,
                "mean": float(value["sum"]) / max(count, 1),
            }
        elif isinstance(value, dict):
            result[name] = _finalize_coverage(value)
        else:
            result[name] = value
    return result


@torch.inference_mode()
def validate(
    model: BondGroupProposalMembershipPredictor,
    primitive_model: PrimitiveSetPredictor,
    loader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    primitive_model.eval()
    if any(parameter.requires_grad for parameter in primitive_model.parameters()):
        raise RuntimeError("Validation primitive model is not frozen")
    loss_totals = {name: 0.0 for name in LOSS_FIELDS}
    oracle_totals: dict[str, int] = {}
    learned_totals: dict[str, int] = {}
    coverage_totals: dict[str, Any] = {}
    proposal_keep_correct = 0
    proposal_keep_total = 0
    proposal_type_correct = 0
    proposal_type_total = 0
    samples = 0
    started = time.time()
    for batch in loader:
        _validate_batch_metadata(batch)
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        with autocast_context(device):
            flat_output = primitive_model(pixels)
        prepared = prepare_pipeline_batch(
            batch,
            flat_output,
            primitive_model,
            model.config,
            loader.collate_fn,
        )
        del flat_output
        model_inputs = _device_tensor_dict(prepared["model_inputs"], device)
        targets = _device_tensor_dict(prepared["targets"], device)
        known_counts = prepared["known_counts"].to(device, non_blocking=True)
        proposal_mask = model_inputs["proposal_mask"].bool()
        with autocast_context(device):
            output = model(**model_inputs)
            losses = compute_membership_loss(output, targets, proposal_mask)
        batch_size = pixels.size(0)
        samples += batch_size
        for name in LOSS_FIELDS:
            loss_totals[name] += float(losses[name]) * batch_size

        oracle_output = oracle_keep_type_output(output, targets, proposal_mask)
        oracle_decoded = decode_bond_group_proposal_membership(
            oracle_output,
            proposal_mask,
            mode="known_count",
            known_counts=known_counts,
        )
        learned_decoded = decode_bond_group_proposal_membership(
            output,
            proposal_mask,
            mode="learned_count",
        )
        _merge_integer_totals(
            oracle_totals,
            _partition_counts(oracle_decoded, targets, proposal_mask),
        )
        _merge_integer_totals(
            learned_totals,
            _partition_counts(learned_decoded, targets, proposal_mask),
        )
        _merge_coverage(coverage_totals, prepared["oracle_coverage"])

        keep_prediction = output["proposal_keep_logits"].sigmoid() >= 0.5
        keep_target = targets["proposal_keep"].bool()
        proposal_keep_correct += int(
            ((keep_prediction == keep_target) & proposal_mask).sum()
        )
        proposal_keep_total += int(proposal_mask.sum())
        type_scope = proposal_mask & keep_target
        type_prediction = output["proposal_type_logits"].argmax(dim=-1)
        proposal_type_correct += int(
            ((type_prediction == targets["proposal_types"]) & type_scope).sum()
        )
        proposal_type_total += int(type_scope.sum())
    if not samples:
        raise RuntimeError("Proposal Membership validation loader produced no samples")
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "samples": samples,
        **{name: value / samples for name, value in loss_totals.items()},
        "known_count_oracle_keep_type": _finalize_partition_metrics(oracle_totals),
        "learned_count": _finalize_partition_metrics(learned_totals),
        "proposal_keep_accuracy": proposal_keep_correct / max(proposal_keep_total, 1),
        "proposal_type_accuracy_on_kept": proposal_type_correct
        / max(proposal_type_total, 1),
        "oracle_coverage": _finalize_coverage(coverage_totals),
        "elapsed_seconds": time.time() - started,
        "uses_internal_test": False,
    }


def save_checkpoint(
    path: Path,
    model: BondGroupProposalMembershipPredictor,
    optimizer: torch.optim.Optimizer,
    *,
    global_step: int,
    samples_seen: int,
    args: argparse.Namespace,
    validation: dict[str, Any] | None,
    primitive_provenance: dict[str, Any],
) -> None:
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    atomic_torch_save(
        {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "sequence_schema_version": SEQUENCE_SCHEMA_VERSION,
            "model_type": model.model_type,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "proposal_membership_config": asdict(model.config),
            "global_step": int(global_step),
            "samples_seen": int(samples_seen),
            "training_args": vars(args),
            "validation": validation,
            "source_primitive_checkpoint": primitive_provenance["checkpoint"],
            "source_primitive_checkpoint_sha256": primitive_provenance[
                "checkpoint_sha256"
            ],
            "primitive_provenance": primitive_provenance,
            "frozen_primitive_state_saved": False,
            "optimizer_mode": "fresh",
            "trainable_parameter_names": trainable_names,
            "trainable_parameter_count": trainable_count,
            "uses_internal_test": False,
        },
        path,
    )


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
        "--primitive-checkpoint",
        default=str(SOURCE_PRIMITIVE_CHECKPOINT),
    )
    parser.add_argument(
        "--primitive-checkpoint-sha256",
        default=SOURCE_PRIMITIVE_SHA256,
    )
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--validation-limit", type=int, default=100)
    parser.add_argument("--group-queries", type=int, default=192)
    parser.add_argument("--max-members", type=int, default=143)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        args.image_size,
        args.batch_size,
        args.group_queries,
        args.max_members,
        args.max_steps,
        args.validate_every,
        args.save_every,
    )
    if min(positive) < 1:
        raise ValueError("Positive Proposal Membership dimensions are required")
    if args.workers < 0:
        raise ValueError("Workers must be nonnegative")
    if args.validation_limit is not None and args.validation_limit < 1:
        raise ValueError("Validation limit must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("Learning rate and weight decay are invalid")
    if args.gradient_clip <= 0.0:
        raise ValueError("Gradient clip must be positive")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    _seed_everything(args.seed)
    train_shards = expand_shards(args.train_shards)
    validation_shards = expand_shards(args.validation_shards)
    output_dir = Path(args.output_dir)
    prepare_output_directory(output_dir)
    device = _device_from_args(args)
    primitive_model, _, primitive_provenance = load_frozen_primitive_checkpoint(
        Path(args.primitive_checkpoint),
        expected_sha256=args.primitive_checkpoint_sha256,
        device=device,
    )
    config = membership_config(
        proposal_feature_dim=primitive_model.config.d_model,
        tiny=args.tiny,
        group_queries=args.group_queries,
        max_members=args.max_members,
    )
    model = BondGroupProposalMembershipPredictor(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_loader = make_loader(
        train_shards,
        batch_size=args.batch_size,
        workers=args.workers,
        image_size=args.image_size,
        shuffle=True,
    )
    validation_loader = make_loader(
        validation_shards,
        batch_size=args.batch_size,
        workers=args.workers,
        image_size=args.image_size,
        shuffle=False,
        limit=args.validation_limit,
    )
    write_json(output_dir / "config.json", model.config_dict())
    write_json(output_dir / "training_args.json", vars(args))
    write_json(output_dir / "primitive_source.json", primitive_provenance)

    global_step = 0
    samples_seen = 0
    best_loss = float("inf")
    last_validation: dict[str, Any] | None = None
    history = []
    while global_step < args.max_steps:
        batches = 0
        for batch in train_loader:
            batches += 1
            _validate_batch_metadata(batch)
            model.train()
            primitive_model.eval()
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            with torch.inference_mode(), autocast_context(device):
                flat_output = primitive_model(pixels)
            prepared = prepare_pipeline_batch(
                batch,
                flat_output,
                primitive_model,
                config,
                train_loader.collate_fn,
            )
            del flat_output
            model_inputs = _device_tensor_dict(prepared["model_inputs"], device)
            targets = _device_tensor_dict(prepared["targets"], device)
            proposal_mask = model_inputs["proposal_mask"].bool()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                output = model(**model_inputs)
                losses = compute_membership_loss(output, targets, proposal_mask)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            if any(parameter.grad is not None for parameter in primitive_model.parameters()):
                raise RuntimeError("Frozen primitive model received gradients")
            global_step += 1
            samples_seen += pixels.size(0)

            if global_step % args.validate_every == 0 or global_step >= args.max_steps:
                last_validation = validate(
                    model,
                    primitive_model,
                    validation_loader,
                    device,
                )
                history.append({"global_step": global_step, "metrics": last_validation})
                oracle = last_validation["known_count_oracle_keep_type"]
                learned = last_validation["learned_count"]
                print(
                    f"step={global_step} loss={last_validation['loss']:.6f} "
                    f"oracle_exact={oracle['sample_exact_rate']:.4f} "
                    f"learned_exact={learned['sample_exact_rate']:.4f} "
                    f"learned_f1={learned['membership_f1']:.4f}",
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
                        primitive_provenance=primitive_provenance,
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
                    primitive_provenance=primitive_provenance,
                )
            if global_step >= args.max_steps:
                break
        if not batches:
            raise RuntimeError("Proposal Membership training loader produced no samples")

    if last_validation is None:
        raise RuntimeError("Proposal Membership training completed without validation")
    save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        global_step=global_step,
        samples_seen=samples_seen,
        args=args,
        validation=last_validation,
        primitive_provenance=primitive_provenance,
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
            "primitive_provenance": primitive_provenance,
            "frozen_primitive_state_saved": False,
            "optimizer_mode": "fresh",
            "trainable_parameter_count": model.parameter_count(),
            "uses_internal_test": False,
        },
    )


if __name__ == "__main__":
    main()
