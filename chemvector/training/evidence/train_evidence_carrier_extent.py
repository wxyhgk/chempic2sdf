#!/usr/bin/env python3
"""Train and evaluate v0.9.9 P0 evidence-carrier extent prediction.

P0 is deliberately conditional.  Full-raster LSD carriers are associated with
ordinary target lines after proposal generation.  The model sees only the full
raster and each carrier; GT association supplies geometry supervision and the
``target_line_id`` equivalence relation used by validation.  Keep/drop is out of
scope: the keep head is frozen and its loss weight is exactly zero.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import glob
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from chemvector.audit_evidence_carrier_extent_oracle import (
    unordered_endpoint_error_q,
)
from chemvector.audit_evidence_carrier_proposals import TargetLine, extract_targets
from chemvector.evidence_carrier_extent import (
    EvidenceCarrierExtentConfig,
    EvidenceCarrierExtentPredictor,
    decode_evidence_carrier_extent,
    evidence_carrier_extent_loss,
)
from chemvector.evidence_carrier_extent_data import (
    IN_SCOPE_GROUP_KINDS,
    EvidenceCarrierExtentCollator,
    EvidenceCarrierExtentTargetConfig,
    EvidenceCarrierExtentWebDataset,
    reject_internal_test_sample,
)


CHECKPOINT_SCHEMA_VERSION = "evidence-carrier-extent-checkpoint-0.1"
VALIDATION_SCHEMA_VERSION = "evidence-carrier-extent-validation-0.1"
TRAINING_SUMMARY_SCHEMA_VERSION = "evidence-carrier-extent-training-summary-0.1"
DEFAULT_TRAIN_SHARDS = (
    "Data/sdf/visual_svg_v0_1/pilot_10k/shards/train-*.tar",
)
DEFAULT_VALIDATION_SHARDS = (
    "Data/sdf/visual_svg_v0_1/pilot_10k/shards/validation-*.tar",
)
DEFAULT_OUTPUT_DIR = (
    "outputs/checkpoints/"
    "chemvector-v099-evidence-carrier-extent-pilot9k-p0-step1000"
)
P0_EXTENT_BINS = 385
P0_KEEP_WEIGHT = 0.0
P0_NORMAL_OFFSET_WEIGHT = 0.25
P0_EXTENT_WEIGHT = 1.0
DEFAULT_TOLERANCES_Q = (8.0, 16.0, 32.0)
BEST_CHECKPOINT_METRIC_ORDER = (
    ("ordinary_q8_member_recall", "max"),
    ("structural_q8_sample_exact_rate", "max"),
    ("long_q8_group_coverage", "max"),
    ("loss", "min"),
)
LOSS_FIELDS = (
    "loss",
    "keep_loss",
    "normal_offset_loss",
    "extent_loss",
    "left_extent_loss",
    "right_extent_loss",
)
GROUP_ALIASES = {
    "single_line": "single",
    "parallel_2": "p2",
    "parallel_3": "p3",
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _security_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        default=str,
    ).lower().replace("-", "_")


def _contains_internal_test(value: Any) -> bool:
    return "internal_test" in str(value).lower().replace("-", "_")


def reject_internal_test_paths(paths: Iterable[str | Path]) -> None:
    forbidden = []
    for path in paths:
        candidate = Path(path).expanduser()
        values = (str(candidate), str(candidate.resolve(strict=False)))
        if any(_contains_internal_test(value) for value in values):
            forbidden.append(str(path))
    if forbidden:
        raise ValueError(
            "Evidence Carrier Extent must not read sealed internal_test shards: "
            + ", ".join(forbidden)
        )


def expand_shards(patterns: Iterable[str]) -> list[str]:
    shards: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            shards.extend(matches)
        elif Path(pattern).is_file():
            shards.append(str(Path(pattern)))
    unique = list(dict.fromkeys(shards))
    if not unique:
        raise ValueError(f"No Evidence Carrier Extent shards matched: {list(patterns)}")
    reject_internal_test_paths(unique)
    return unique


def reject_train_validation_overlap(
    train_shards: Iterable[str | Path],
    validation_shards: Iterable[str | Path],
) -> None:
    train = {Path(path).expanduser().resolve() for path in train_shards}
    validation = {
        Path(path).expanduser().resolve() for path in validation_shards
    }
    overlap = sorted(str(path) for path in train & validation)
    if overlap:
        raise ValueError(
            "Evidence Carrier Extent train/validation shards overlap: "
            + ", ".join(overlap)
        )


def validate_batch_security(batch: dict[str, Any]) -> None:
    keys = batch.get("keys")
    metadata = batch.get("meta")
    if not isinstance(keys, list) or not isinstance(metadata, list):
        raise ValueError("Evidence Carrier Extent batch lacks keys or metadata")
    if len(keys) != len(metadata):
        raise ValueError("Evidence Carrier Extent batch keys and metadata differ")
    for key, meta in zip(keys, metadata):
        reject_internal_test_sample({"key": str(key), "meta": meta})


def assert_unscaled_full_raster_batch(batch: dict[str, Any]) -> None:
    """Fail before model forward if the collator resized source geometry."""

    pixels = batch.get("pixel_values")
    layers = batch.get("layers")
    if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4:
        raise ValueError("Evidence Carrier Extent batch lacks raster pixels")
    if not isinstance(layers, list) or len(layers) != pixels.size(0):
        raise ValueError("Evidence Carrier Extent batch lacks layer dimensions")
    tensor_size = (int(pixels.size(-2)), int(pixels.size(-1)))
    for batch_index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError("Evidence Carrier Extent layer sidecar is invalid")
        source_size = (int(layer.get("height", 0)), int(layer.get("width", 0)))
        if source_size != tensor_size:
            raise ValueError(
                "P0 requires the original full raster without resize before model "
                f"forward: sample={batch_index} source={source_size} "
                f"tensor={tensor_size}"
            )


def shard_manifest(shards: Iterable[str | Path]) -> list[dict[str, Any]]:
    result = []
    for path_value in shards:
        path = Path(path_value).expanduser().resolve()
        reject_internal_test_paths((path,))
        if not path.is_file():
            raise FileNotFoundError(f"Evidence Carrier Extent shard is missing: {path}")
        result.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return result


def evidence_carrier_extent_config(*, tiny: bool = False) -> EvidenceCarrierExtentConfig:
    """Return the fixed P0 geometry protocol with an optional small encoder."""

    if tiny:
        return EvidenceCarrierExtentConfig(
            longitudinal_samples=33,
            band_samples=5,
            band_radius_pixels=8.0,
            normal_offset_radius_pixels=4.0,
            normal_offset_step_pixels=0.5,
            extent_bins=P0_EXTENT_BINS,
            hidden_dim=16,
            dropout=0.0,
        )
    return EvidenceCarrierExtentConfig()


def evidence_carrier_target_config() -> EvidenceCarrierExtentTargetConfig:
    return EvidenceCarrierExtentTargetConfig()


def assert_extent_config_contract(
    model_config: EvidenceCarrierExtentConfig,
    target_config: EvidenceCarrierExtentTargetConfig,
) -> None:
    """Reject any silent mismatch between model bins and target geometry."""

    mismatches: dict[str, tuple[Any, Any]] = {}
    if model_config.extent_bins != P0_EXTENT_BINS:
        mismatches["model.extent_bins"] = (
            model_config.extent_bins,
            P0_EXTENT_BINS,
        )
    if target_config.extent_bins != P0_EXTENT_BINS:
        mismatches["target.extent_bins"] = (
            target_config.extent_bins,
            P0_EXTENT_BINS,
        )
    if model_config.extent_bins != target_config.extent_bins:
        mismatches["extent_bins"] = (
            model_config.extent_bins,
            target_config.extent_bins,
        )
    if model_config.normal_offset_bins != target_config.normal_offset_bins:
        mismatches["normal_offset_bins"] = (
            model_config.normal_offset_bins,
            target_config.normal_offset_bins,
        )
    if not math.isclose(
        model_config.normal_offset_radius_pixels,
        target_config.normal_offset_limit_pixels,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        mismatches["normal_offset_range_pixels"] = (
            model_config.normal_offset_radius_pixels,
            target_config.normal_offset_limit_pixels,
        )
    target_step = (
        2.0
        * target_config.normal_offset_limit_pixels
        / float(target_config.normal_offset_bins - 1)
    )
    if not math.isclose(
        model_config.normal_offset_step_pixels,
        target_step,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        mismatches["normal_offset_step_pixels"] = (
            model_config.normal_offset_step_pixels,
            target_step,
        )
    if mismatches:
        raise ValueError(
            "Evidence Carrier Extent model/target configuration differs: "
            + repr(mismatches)
        )


def freeze_p0_keep_head(model: EvidenceCarrierExtentPredictor) -> None:
    for parameter in model.keep_head.parameters():
        parameter.requires_grad_(False)


def p0_trainable_parameter_names(
    model: EvidenceCarrierExtentPredictor,
) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def make_loader(
    shards: list[str],
    *,
    target_config: EvidenceCarrierExtentTargetConfig,
    batch_size: int,
    workers: int,
    shuffle: bool,
    image_size: int = 768,
    limit: int | None = None,
) -> DataLoader:
    reject_internal_test_paths(shards)
    return DataLoader(
        EvidenceCarrierExtentWebDataset(
            shards,
            shardshuffle=shuffle,
            sample_shuffle_buffer=256 if shuffle else 0,
            limit=limit,
        ),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=EvidenceCarrierExtentCollator(
            image_size=image_size,
            oracle_bond_layer=False,
            target_config=target_config,
        ),
    )


def batch_tensors_to_device(
    batch: dict[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    names = (
        "pixel_values",
        "carrier_segments",
        "carrier_mask",
        "keep_labels",
        "keep_loss_mask",
        "equivalence_weight",
        "normal_offset_bin",
        "extent_left_bin",
        "extent_right_bin",
        "geometry_loss_mask",
        "target_line_id",
        "group_id",
        "group_type",
    )
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in names
    }


def compute_p0_loss(
    output: dict[str, torch.Tensor],
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return evidence_carrier_extent_loss(
        output,
        tensors["keep_labels"],
        tensors["normal_offset_bin"],
        tensors["extent_left_bin"],
        tensors["extent_right_bin"],
        keep_loss_mask=tensors["keep_loss_mask"],
        geometry_loss_mask=tensors["geometry_loss_mask"],
        equivalence_weight=tensors["equivalence_weight"],
        keep_weight=P0_KEEP_WEIGHT,
        normal_offset_weight=P0_NORMAL_OFFSET_WEIGHT,
        extent_weight=P0_EXTENT_WEIGHT,
    )


def best_equivalent_line_predictions(
    predicted_segments_pixels: np.ndarray | torch.Tensor,
    target_line_id: np.ndarray | torch.Tensor,
    geometry_loss_mask: np.ndarray | torch.Tensor,
    target_lines: list[TargetLine],
    *,
    width: int,
    height: int,
) -> dict[int, dict[str, float | int]]:
    """Take the lowest endpoint error inside each ``target_line_id`` class.

    Predicted segments have already been reconstructed solely from model bins.
    GT is used here only for evaluation and for grouping equivalent positive
    carriers; no GT normal offset or extent enters a prediction.
    """

    if isinstance(predicted_segments_pixels, torch.Tensor):
        predicted = predicted_segments_pixels.detach().float().cpu().numpy()
    else:
        predicted = np.asarray(predicted_segments_pixels, dtype=np.float64)
    if isinstance(target_line_id, torch.Tensor):
        line_ids = target_line_id.detach().cpu().numpy()
    else:
        line_ids = np.asarray(target_line_id)
    if isinstance(geometry_loss_mask, torch.Tensor):
        mask = geometry_loss_mask.detach().cpu().numpy().astype(bool)
    else:
        mask = np.asarray(geometry_loss_mask, dtype=bool)
    if predicted.ndim != 3 or predicted.shape[1:] != (2, 2):
        raise ValueError("Predicted Evidence Carrier lines must have shape [N,2,2]")
    if line_ids.shape != (len(predicted),) or mask.shape != (len(predicted),):
        raise ValueError("Evidence Carrier equivalence arrays differ in shape")
    if width < 2 or height < 2:
        raise ValueError("Evidence Carrier evaluation canvas must be at least 2x2")

    ordinary = {
        line_id: line
        for line_id, line in enumerate(target_lines)
        if line.group_kind in IN_SCOPE_GROUP_KINDS
    }
    best: dict[int, dict[str, float | int]] = {
        line_id: {"error_q": math.inf, "carrier_index": -1}
        for line_id in ordinary
    }
    for carrier_index in np.flatnonzero(mask):
        line_id = int(line_ids[carrier_index])
        if line_id not in ordinary:
            raise ValueError(
                "Positive Evidence Carrier references an out-of-scope target line"
            )
        segment = predicted[carrier_index]
        error = (
            unordered_endpoint_error_q(
                segment,
                ordinary[line_id].points_q,
                width=width,
                height=height,
            )
            if np.isfinite(segment).all()
            else math.inf
        )
        if error < float(best[line_id]["error_q"]):
            best[line_id] = {
                "error_q": float(error),
                "carrier_index": int(carrier_index),
            }
    return best


def evaluate_sample_geometry(
    predicted_segments_pixels: np.ndarray | torch.Tensor,
    target_line_id: np.ndarray | torch.Tensor,
    geometry_loss_mask: np.ndarray | torch.Tensor,
    primitives: dict[str, Any],
    layers: dict[str, Any],
    *,
    target_config: EvidenceCarrierExtentTargetConfig,
    tolerances_q: tuple[float, ...] = DEFAULT_TOLERANCES_Q,
) -> dict[str, Any]:
    width = int(layers.get("width", 0))
    height = int(layers.get("height", 0))
    target_lines, target_groups, _ = extract_targets(
        primitives,
        layers,
        long_bond_pixels=target_config.long_bond_pixels,
    )
    best = best_equivalent_line_predictions(
        predicted_segments_pixels,
        target_line_id,
        geometry_loss_mask,
        target_lines,
        width=width,
        height=height,
    )
    ordinary_groups = [
        group
        for group in target_groups
        if group.kind in IN_SCOPE_GROUP_KINDS and group.line_indices
    ]
    result: dict[str, Any] = {
        "ordinary_target_lines": len(best),
        "equivalence_classes_with_predictions": sum(
            math.isfinite(float(value["error_q"])) for value in best.values()
        ),
        "tolerances": {},
    }
    for tolerance in tolerances_q:
        name = f"q{int(tolerance)}"
        matched = {
            line_id: float(values["error_q"]) <= tolerance
            for line_id, values in best.items()
        }
        groups = {}
        for kind, alias in GROUP_ALIASES.items():
            selected = [group for group in ordinary_groups if group.kind == kind]
            groups[alias] = {
                "matched": sum(
                    all(matched.get(line_id, False) for line_id in group.line_indices)
                    for group in selected
                ),
                "expected": len(selected),
            }
        long_groups = [group for group in ordinary_groups if group.long_group]
        member_matches = sum(matched.values())
        result["tolerances"][name] = {
            "ordinary_members": {
                "matched": member_matches,
                "expected": len(best),
            },
            "groups": groups,
            "long_groups": {
                "matched": sum(
                    all(matched.get(line_id, False) for line_id in group.line_indices)
                    for group in long_groups
                ),
                "expected": len(long_groups),
            },
            "structural_sample_eligible": bool(best),
            "structural_sample_exact": bool(best) and member_matches == len(best),
        }
    return result


def summarize_geometry_records(
    records: list[dict[str, Any]],
    tolerances_q: tuple[float, ...] = DEFAULT_TOLERANCES_Q,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for tolerance in tolerances_q:
        name = f"q{int(tolerance)}"
        member_matched = 0
        member_expected = 0
        structural_exact = 0
        structural_expected = 0
        structural_excluded_no_ordinary_target = 0
        group_totals = {
            alias: {"matched": 0, "expected": 0}
            for alias in GROUP_ALIASES.values()
        }
        long_matched = 0
        long_expected = 0
        for record in records:
            values = record["tolerances"][name]
            member_matched += values["ordinary_members"]["matched"]
            member_expected += values["ordinary_members"]["expected"]
            if values["structural_sample_eligible"]:
                structural_expected += 1
                structural_exact += int(values["structural_sample_exact"])
            else:
                structural_excluded_no_ordinary_target += 1
            for alias in group_totals:
                group_totals[alias]["matched"] += values["groups"][alias]["matched"]
                group_totals[alias]["expected"] += values["groups"][alias]["expected"]
            long_matched += values["long_groups"]["matched"]
            long_expected += values["long_groups"]["expected"]
        for values in group_totals.values():
            values["coverage"] = values["matched"] / max(values["expected"], 1)
        summary[name] = {
            "ordinary_members": {
                "matched": member_matched,
                "expected": member_expected,
                "recall": member_matched / max(member_expected, 1),
            },
            "groups": group_totals,
            "long_groups": {
                "matched": long_matched,
                "expected": long_expected,
                "coverage": long_matched / max(long_expected, 1),
            },
            "structural_samples": {
                "exact": structural_exact,
                "expected": structural_expected,
                "excluded_no_ordinary_target": (
                    structural_excluded_no_ordinary_target
                ),
                "total": len(records),
                "exact_rate": (
                    structural_exact / structural_expected
                    if structural_expected
                    else None
                ),
            },
        }
    return summary


def validation_checkpoint_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    """Rank P0 checkpoints by route metrics, with loss only as a tie-breaker."""

    values = []
    for name, direction in BEST_CHECKPOINT_METRIC_ORDER:
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(
                f"Evidence Carrier validation metric {name!r} is not finite"
            )
        numeric = float(value)
        values.append(numeric if direction == "max" else -numeric)
    return tuple(values)


def _weighted_accuracy(
    predicted: torch.Tensor,
    expected: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[float, float, int, int]:
    selected_weights = weights[mask].float()
    correct = predicted[mask] == expected[mask]
    return (
        float((correct.float() * selected_weights).sum()),
        float(selected_weights.sum()),
        int(correct.sum()),
        int(correct.numel()),
    )


@torch.inference_mode()
def validate(
    model: EvidenceCarrierExtentPredictor,
    loader: DataLoader,
    device: torch.device,
    *,
    target_config: EvidenceCarrierExtentTargetConfig,
    tolerances_q: tuple[float, ...] = DEFAULT_TOLERANCES_Q,
) -> dict[str, Any]:
    assert_extent_config_contract(model.config, target_config)
    model.eval()
    loss_numerators = {name: 0.0 for name in LOSS_FIELDS}
    loss_denominators = {name: 0.0 for name in LOSS_FIELDS}
    bin_totals = {
        name: {
            "weighted_correct": 0.0,
            "weight": 0.0,
            "correct": 0,
            "count": 0,
        }
        for name in ("normal", "left", "right")
    }
    records: list[dict[str, Any]] = []
    samples = 0
    batches = 0
    carrier_count = 0
    max_carriers_per_sample = 0
    positive_carrier_count = 0
    label_distribution = {"positive": 0, "negative": 0, "ignored": 0}

    for batch in loader:
        validate_batch_security(batch)
        assert_unscaled_full_raster_batch(batch)
        tensors = batch_tensors_to_device(batch, device)
        pixels = tensors["pixel_values"]
        with autocast_context(device):
            output = model(
                pixels,
                tensors["carrier_segments"],
                tensors["carrier_mask"],
                coordinate_mode="normalized",
            )
            losses = compute_p0_loss(output, tensors)
        decoded = decode_evidence_carrier_extent(output)
        geometry_weight = float(losses["geometry_supervision_weight"])
        keep_weight = float(losses["keep_supervision_weight"])
        for name in LOSS_FIELDS:
            denominator = keep_weight if name == "keep_loss" else geometry_weight
            loss_numerators[name] += float(losses[name]) * denominator
            loss_denominators[name] += denominator

        geometry_mask = tensors["geometry_loss_mask"] & tensors["carrier_mask"]
        predicted_bins = {
            "normal": decoded["normal_offset_bins"],
            "left": decoded["left_extent_bins"],
            "right": decoded["right_extent_bins"],
        }
        expected_bins = {
            "normal": tensors["normal_offset_bin"],
            "left": tensors["extent_left_bin"],
            "right": tensors["extent_right_bin"],
        }
        for name in predicted_bins:
            values = _weighted_accuracy(
                predicted_bins[name],
                expected_bins[name],
                geometry_mask,
                tensors["equivalence_weight"],
            )
            for field, value in zip(
                ("weighted_correct", "weight", "correct", "count"), values
            ):
                bin_totals[name][field] += value

        batch_size = int(pixels.size(0))
        if not (
            len(batch["primitives"]) == len(batch["layers"]) == batch_size
        ):
            raise ValueError("Evidence Carrier Extent sidecar batch sizes differ")
        predicted_lines = decoded["refined_segments_pixels"].detach().float().cpu()
        line_ids = tensors["target_line_id"].detach().cpu()
        geometry_cpu = geometry_mask.detach().cpu()
        for batch_index in range(batch_size):
            layers = batch["layers"][batch_index]
            width = int(layers.get("width", 0))
            height = int(layers.get("height", 0))
            if (height, width) != tuple(pixels.shape[-2:]):
                raise AssertionError("Full-raster size changed after the pre-forward check")
            records.append(
                evaluate_sample_geometry(
                    predicted_lines[batch_index],
                    line_ids[batch_index],
                    geometry_cpu[batch_index],
                    batch["primitives"][batch_index],
                    layers,
                    target_config=target_config,
                    tolerances_q=tolerances_q,
                )
            )

        samples += batch_size
        batches += 1
        carrier_count += int(tensors["carrier_mask"].sum())
        max_carriers_per_sample = max(
            max_carriers_per_sample,
            int(tensors["carrier_mask"].sum(dim=1).max()),
        )
        positive_carrier_count += int(geometry_mask.sum())
        valid_labels = tensors["keep_labels"][tensors["carrier_mask"]]
        label_distribution["positive"] += int((valid_labels == 1).sum())
        label_distribution["negative"] += int((valid_labels == 0).sum())
        label_distribution["ignored"] += int((valid_labels == -1).sum())

    if not samples:
        raise RuntimeError("Evidence Carrier Extent validation produced no samples")
    if loss_denominators["loss"] <= 0.0:
        raise RuntimeError("Evidence Carrier Extent validation has no positive carriers")
    geometry = summarize_geometry_records(records, tolerances_q)
    bin_accuracy = {}
    for name, values in bin_totals.items():
        bin_accuracy[name] = {
            **values,
            "accuracy": values["weighted_correct"] / max(values["weight"], 1e-12),
            "unweighted_accuracy": values["correct"] / max(values["count"], 1),
        }
    result = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "model_type": model.model_type,
        "samples": samples,
        "batches": batches,
        "carrier_count": carrier_count,
        "carrier_count_mean": carrier_count / samples,
        "max_carriers_per_sample": max_carriers_per_sample,
        "positive_carrier_count": positive_carrier_count,
        "label_distribution": label_distribution,
        "positive_equivalence_class_count": loss_denominators["loss"],
        **{
            name: loss_numerators[name] / max(loss_denominators[name], 1e-12)
            for name in LOSS_FIELDS
        },
        "bin_accuracy": bin_accuracy,
        "geometry": geometry,
        "evaluation_protocol": {
            "raster_source": "full_png",
            "oracle_bond_layer": False,
            "conditional_positive_carriers": True,
            "prediction_reconstruction": "predicted normal/left/right bins only",
            "equivalence_group": "target_line_id",
            "equivalence_selection": "minimum endpoint error within equivalent carriers",
            "gt_geometry_used_as_model_input": False,
            "keep_in_scope": False,
            "p1_keep_warning": (
                "P0 association supplies almost no ordinary negatives; do not reuse "
                "this keep BCE protocol for P1."
            ),
        },
        "uses_internal_test": False,
    }
    for name, values in geometry.items():
        result[f"ordinary_{name}_member_recall"] = values["ordinary_members"][
            "recall"
        ]
        result[f"structural_{name}_sample_exact_rate"] = values[
            "structural_samples"
        ]["exact_rate"]
        result[f"long_{name}_group_coverage"] = values["long_groups"]["coverage"]
    return result


def _plain_training_args(args: argparse.Namespace | dict[str, Any]) -> dict[str, Any]:
    values = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    return json.loads(json.dumps(values, default=str))


def validate_checkpoint_payload(checkpoint: dict[str, Any]) -> None:
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": EvidenceCarrierExtentPredictor.model_type,
        "uses_internal_test": False,
        "raster_source": "full_png",
        "oracle_bond_layer": False,
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Evidence Carrier Extent checkpoint differs: {mismatches}")
    for name in (
        "model",
        "evidence_carrier_extent_config",
        "target_config",
        "training_args",
        "loss_protocol",
    ):
        if not isinstance(checkpoint.get(name), dict):
            raise ValueError(f"Evidence Carrier Extent checkpoint has no {name}")
    for name in ("global_step", "samples_seen"):
        value = checkpoint.get(name)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"Evidence Carrier Extent checkpoint has invalid {name}")
    protocol = checkpoint["loss_protocol"]
    if protocol.get("keep_weight") != P0_KEEP_WEIGHT:
        raise ValueError("P0 Evidence Carrier checkpoint must have keep_weight=0")
    if protocol.get("normal_offset_weight") != P0_NORMAL_OFFSET_WEIGHT:
        raise ValueError(
            "P0 Evidence Carrier checkpoint has a different normal-offset weight"
        )
    if protocol.get("extent_weight") != P0_EXTENT_WEIGHT:
        raise ValueError(
            "P0 Evidence Carrier checkpoint has a different extent weight"
        )
    if protocol.get("equivalence_weight") != "target_line_id_balanced":
        raise ValueError("P0 Evidence Carrier checkpoint lacks equivalence balancing")
    if _contains_internal_test(_security_text(checkpoint["training_args"])):
        raise ValueError("Evidence Carrier Extent checkpoint references internal_test")
    for manifest_name in ("train_shards", "validation_shards"):
        manifest = checkpoint.get(manifest_name, [])
        if not isinstance(manifest, list):
            raise ValueError(f"Evidence Carrier Extent {manifest_name} is not a list")
        paths = []
        for record in manifest:
            if not isinstance(record, dict):
                raise ValueError(f"Evidence Carrier Extent {manifest_name} is invalid")
            path = record.get("path")
            digest = record.get("sha256")
            if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"Evidence Carrier Extent {manifest_name} record is invalid")
            paths.append(path)
        reject_internal_test_paths(paths)
    model_config = EvidenceCarrierExtentConfig(
        **checkpoint["evidence_carrier_extent_config"]
    )
    target_config = EvidenceCarrierExtentTargetConfig(**checkpoint["target_config"])
    assert_extent_config_contract(model_config, target_config)


def save_checkpoint(
    path: str | Path,
    model: EvidenceCarrierExtentPredictor,
    optimizer: torch.optim.Optimizer | None,
    *,
    target_config: EvidenceCarrierExtentTargetConfig,
    global_step: int,
    samples_seen: int,
    args: argparse.Namespace | dict[str, Any],
    validation: dict[str, Any] | None,
    train_shards: list[dict[str, Any]] | None = None,
    validation_shards: list[dict[str, Any]] | None = None,
) -> str:
    assert_extent_config_contract(model.config, target_config)
    payload = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": model.model_type,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "evidence_carrier_extent_config": asdict(model.config),
        "target_config": asdict(target_config),
        "global_step": int(global_step),
        "samples_seen": int(samples_seen),
        "training_args": _plain_training_args(args),
        "validation": validation,
        "train_shards": list(train_shards or []),
        "validation_shards": list(validation_shards or []),
        "loss_protocol": {
            "keep_weight": P0_KEEP_WEIGHT,
            "normal_offset_weight": P0_NORMAL_OFFSET_WEIGHT,
            "extent_weight": P0_EXTENT_WEIGHT,
            "equivalence_weight": "target_line_id_balanced",
        },
        "raster_source": "full_png",
        "oracle_bond_layer": False,
        "trainable_parameter_names": p0_trainable_parameter_names(model),
        "parameter_count": model.parameter_count(),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "trainer_sha256": file_sha256(__file__),
        "uses_internal_test": False,
    }
    validate_checkpoint_payload(payload)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return file_sha256(output)


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[
    EvidenceCarrierExtentPredictor,
    EvidenceCarrierExtentTargetConfig,
    dict[str, Any],
]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Evidence Carrier Extent checkpoint must be a dictionary")
    validate_checkpoint_payload(checkpoint)
    model_config = EvidenceCarrierExtentConfig(
        **checkpoint["evidence_carrier_extent_config"]
    )
    target_config = EvidenceCarrierExtentTargetConfig(**checkpoint["target_config"])
    model = EvidenceCarrierExtentPredictor(model_config)
    model.load_state_dict(checkpoint["model"], strict=True)
    freeze_p0_keep_head(model)
    model.to(torch.device(device))
    return model, target_config, checkpoint


def prepare_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _device_from_name(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _validate_cli_args(args: argparse.Namespace) -> None:
    positive = (
        "max_steps",
        "batch_size",
        "workers",
        "gradient_accumulation",
        "validate_every",
        "save_every",
        "log_every",
        "image_size",
    )
    for name in positive:
        value = getattr(args, name)
        minimum = 0 if name == "workers" else 1
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("Optimizer values are invalid")
    if args.max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be positive")
    if args.validation_limit is not None and args.validation_limit < 1:
        raise ValueError("validation_limit must be positive")
    if args.image_size != 768 and not args.tiny:
        raise ValueError("Full P0 training requires the native 768x768 raster")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shards", nargs="+", default=list(DEFAULT_TRAIN_SHARDS))
    parser.add_argument(
        "--validation-shards",
        nargs="+",
        default=list(DEFAULT_VALIDATION_SHARDS),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--evaluate-only",
        metavar="CHECKPOINT",
        help="Evaluate one P0 checkpoint without training.",
    )
    return parser


def _apply_smoke_settings(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.tiny = True
    args.max_steps = min(args.max_steps, 2)
    args.batch_size = 1
    args.workers = 0
    args.gradient_accumulation = 1
    args.validate_every = 1
    args.save_every = 1
    args.log_every = 1
    args.validation_limit = min(args.validation_limit or 2, 2)


def _evaluate_only(args: argparse.Namespace, device: torch.device) -> None:
    checkpoint_path = Path(args.evaluate_only).expanduser().resolve()
    reject_internal_test_paths((checkpoint_path,))
    model, target_config, checkpoint = load_checkpoint(
        checkpoint_path,
        device=device,
    )
    validation_shards = expand_shards(args.validation_shards)
    loader = make_loader(
        validation_shards,
        target_config=target_config,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        image_size=args.image_size,
        limit=args.validation_limit,
    )
    metrics = validate(model, loader, device, target_config=target_config)
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["checkpoint_sha256"] = file_sha256(checkpoint_path)
    metrics["checkpoint_global_step"] = checkpoint["global_step"]
    metrics["validation_shards"] = shard_manifest(validation_shards)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evaluation.json", metrics)
    print(json.dumps(metrics, sort_keys=True), flush=True)


def main() -> None:
    args = build_parser().parse_args()
    _apply_smoke_settings(args)
    _validate_cli_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = _device_from_name(args.device)

    if args.evaluate_only:
        _evaluate_only(args, device)
        return

    train_shards = expand_shards(args.train_shards)
    validation_shards = expand_shards(args.validation_shards)
    reject_train_validation_overlap(train_shards, validation_shards)
    output_dir = Path(args.output_dir)
    prepare_output_directory(output_dir)

    target_config = evidence_carrier_target_config()
    model_config = evidence_carrier_extent_config(tiny=args.tiny)
    assert_extent_config_contract(model_config, target_config)
    model = EvidenceCarrierExtentPredictor(model_config).to(device)
    freeze_p0_keep_head(model)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_manifest = shard_manifest(train_shards)
    validation_manifest = shard_manifest(validation_shards)
    train_loader = make_loader(
        train_shards,
        target_config=target_config,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=True,
        image_size=args.image_size,
    )
    validation_loader = make_loader(
        validation_shards,
        target_config=target_config,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        image_size=args.image_size,
        limit=args.validation_limit,
    )

    global_step = 0
    samples_seen = 0
    accumulated = 0
    best_checkpoint_key: tuple[float, ...] | None = None
    best_checkpoint_step: int | None = None
    best_validation: dict[str, Any] | None = None
    last_validation: dict[str, Any] | None = None
    validation_history: list[dict[str, Any]] = []
    checkpoint_sha256: dict[str, str] = {}
    observed_train_carriers = {
        "samples": 0,
        "carrier_count": 0,
        "max_carriers_per_sample": 0,
        "label_distribution": {"positive": 0, "negative": 0, "ignored": 0},
    }
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()

    while global_step < args.max_steps:
        batches_this_epoch = 0
        for batch in train_loader:
            validate_batch_security(batch)
            assert_unscaled_full_raster_batch(batch)
            tensors = batch_tensors_to_device(batch, device)
            valid_labels = tensors["keep_labels"][tensors["carrier_mask"]]
            observed_train_carriers["samples"] += int(
                tensors["pixel_values"].size(0)
            )
            observed_train_carriers["carrier_count"] += int(
                tensors["carrier_mask"].sum()
            )
            observed_train_carriers["max_carriers_per_sample"] = max(
                observed_train_carriers["max_carriers_per_sample"],
                int(tensors["carrier_mask"].sum(dim=1).max()),
            )
            observed_train_carriers["label_distribution"]["positive"] += int(
                (valid_labels == 1).sum()
            )
            observed_train_carriers["label_distribution"]["negative"] += int(
                (valid_labels == 0).sum()
            )
            observed_train_carriers["label_distribution"]["ignored"] += int(
                (valid_labels == -1).sum()
            )
            with autocast_context(device):
                output = model(
                    tensors["pixel_values"],
                    tensors["carrier_segments"],
                    tensors["carrier_mask"],
                    coordinate_mode="normalized",
                )
                losses = compute_p0_loss(output, tensors)
                scaled_loss = losses["loss"] / args.gradient_accumulation
            scaled_loss.backward()
            accumulated += 1
            samples_seen += int(tensors["pixel_values"].size(0))
            batches_this_epoch += 1
            if accumulated < args.gradient_accumulation:
                continue

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                args.max_grad_norm,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("Evidence Carrier Extent gradient norm is non-finite")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            global_step += 1

            if global_step == 1 or global_step % args.log_every == 0:
                print(
                    json.dumps(
                        {
                            "step": global_step,
                            "samples_seen": samples_seen,
                            "loss": float(losses["loss"].detach()),
                            "normal_offset_loss": float(
                                losses["normal_offset_loss"].detach()
                            ),
                            "extent_loss": float(losses["extent_loss"].detach()),
                            "positive_equivalence_classes": float(
                                losses["geometry_supervision_weight"].detach()
                            ),
                            "gradient_norm": float(gradient_norm.detach()),
                            "elapsed_seconds": time.monotonic() - started,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            if global_step % args.validate_every == 0:
                last_validation = validate(
                    model,
                    validation_loader,
                    device,
                    target_config=target_config,
                )
                validation_history.append(
                    {"global_step": global_step, "metrics": last_validation}
                )
                checkpoint_key = validation_checkpoint_key(last_validation)
                if (
                    best_checkpoint_key is None
                    or checkpoint_key > best_checkpoint_key
                ):
                    best_checkpoint_key = checkpoint_key
                    best_checkpoint_step = global_step
                    best_validation = last_validation
                    checkpoint_sha256["best.pt"] = save_checkpoint(
                        output_dir / "best.pt",
                        model,
                        optimizer,
                        target_config=target_config,
                        global_step=global_step,
                        samples_seen=samples_seen,
                        args=args,
                        validation=last_validation,
                        train_shards=train_manifest,
                        validation_shards=validation_manifest,
                    )
                print(
                    "validation=" + json.dumps(last_validation, sort_keys=True),
                    flush=True,
                )

            if global_step % args.save_every == 0:
                name = f"step-{global_step:06d}.pt"
                checkpoint_sha256[name] = save_checkpoint(
                    output_dir / name,
                    model,
                    optimizer,
                    target_config=target_config,
                    global_step=global_step,
                    samples_seen=samples_seen,
                    args=args,
                    validation=last_validation,
                    train_shards=train_manifest,
                    validation_shards=validation_manifest,
                )
            if global_step >= args.max_steps:
                break
        if batches_this_epoch == 0:
            raise RuntimeError("Evidence Carrier Extent training produced no samples")

    if accumulated:
        raise RuntimeError("P0 training stopped with an incomplete accumulated step")
    if not validation_history or validation_history[-1]["global_step"] != global_step:
        last_validation = validate(
            model,
            validation_loader,
            device,
            target_config=target_config,
        )
        validation_history.append(
            {"global_step": global_step, "metrics": last_validation}
        )
    assert last_validation is not None
    final_checkpoint_key = validation_checkpoint_key(last_validation)
    if (
        best_checkpoint_key is None
        or final_checkpoint_key > best_checkpoint_key
        or not (output_dir / "best.pt").is_file()
    ):
        best_checkpoint_key = final_checkpoint_key
        best_checkpoint_step = global_step
        best_validation = last_validation
        checkpoint_sha256["best.pt"] = save_checkpoint(
            output_dir / "best.pt",
            model,
            optimizer,
            target_config=target_config,
            global_step=global_step,
            samples_seen=samples_seen,
            args=args,
            validation=last_validation,
            train_shards=train_manifest,
            validation_shards=validation_manifest,
        )
    final_step_name = f"step-{global_step:06d}.pt"
    checkpoint_sha256[final_step_name] = save_checkpoint(
        output_dir / final_step_name,
        model,
        optimizer,
        target_config=target_config,
        global_step=global_step,
        samples_seen=samples_seen,
        args=args,
        validation=last_validation,
        train_shards=train_manifest,
        validation_shards=validation_manifest,
    )
    checkpoint_sha256["last.pt"] = save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        target_config=target_config,
        global_step=global_step,
        samples_seen=samples_seen,
        args=args,
        validation=last_validation,
        train_shards=train_manifest,
        validation_shards=validation_manifest,
    )
    assert best_checkpoint_key is not None
    assert best_checkpoint_step is not None
    assert best_validation is not None
    write_json(output_dir / "validation.json", last_validation)
    write_json(
        output_dir / "summary.json",
        {
            "schema_version": TRAINING_SUMMARY_SCHEMA_VERSION,
            "model_type": model.model_type,
            "global_step": global_step,
            "samples_seen": samples_seen,
            "best_validation_loss": float(best_validation["loss"]),
            "best_checkpoint_selection": {
                "global_step": best_checkpoint_step,
                "metric_order": [
                    {"metric": name, "direction": direction}
                    for name, direction in BEST_CHECKPOINT_METRIC_ORDER
                ],
                "key": list(best_checkpoint_key),
                "metrics": {
                    name: float(best_validation[name])
                    for name, _ in BEST_CHECKPOINT_METRIC_ORDER
                },
            },
            "history": validation_history,
            "train_shards": train_manifest,
            "validation_shards": validation_manifest,
            "checkpoint_sha256": checkpoint_sha256,
            "observed_train_carriers": {
                **observed_train_carriers,
                "carrier_count_mean": observed_train_carriers["carrier_count"]
                / max(observed_train_carriers["samples"], 1),
            },
            "final_validation_carriers": {
                "samples": last_validation["samples"],
                "carrier_count": last_validation["carrier_count"],
                "carrier_count_mean": last_validation["carrier_count_mean"],
                "max_carriers_per_sample": last_validation[
                    "max_carriers_per_sample"
                ],
                "label_distribution": last_validation["label_distribution"],
            },
            "training_args": _plain_training_args(args),
            "raster_source": "full_png",
            "oracle_bond_layer": False,
            "uses_internal_test": False,
        },
    )
    print("final_validation=" + json.dumps(last_validation, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
