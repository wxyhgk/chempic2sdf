#!/usr/bin/env python3
"""Train the v0.9.10 pair scorer conditioned on frozen P0 evidence.

The command joins the frozen geometry-only pair cache with its frozen-P0
augmentation sidecar.  Model weights and descriptor normalization use only the
``fit`` partition.  Checkpoint selection and the gain threshold use only the
train-derived ``calibration`` partition.  Validation remains a separate,
one-pass operation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from chemvector.evidence_carrier_conditional_pair_data import (
    PAIR_FEATURE_DIM,
    PAIR_FEATURE_NAMES,
    PAIR_NEGATIVE_LABEL,
    PAIR_POSITIVE_LABEL,
    PAIRING_PROFILE_TRAINING_WIDE,
    conditional_pair_pairing_config,
)
from chemvector.evidence_carrier_p0_pair_cache import (
    CACHE_SCHEMA_VERSION as P0_PAIR_CACHE_SCHEMA_VERSION,
    DEFAULT_OUTPUT as DEFAULT_P0_PAIR_AUGMENTATION_CACHE,
    joined_p0_pair_records,
    load_evidence_carrier_p0_pair_cache,
)
from chemvector.evidence_carrier_p0_pair_scorer import (
    P0_PAIR_CONSISTENCY_FEATURE_DIM,
    P0_PAIR_CONSISTENCY_FEATURE_NAMES,
    P0_PAIR_SCORER_PARAMETER_COUNT,
    EvidenceCarrierP0PairScorer,
    EvidenceCarrierP0PairScorerConfig,
)
from chemvector.evidence_carrier_pair_cache import (
    CACHE_SCHEMA_VERSION as BASE_PAIR_CACHE_SCHEMA_VERSION,
)
from chemvector.evidence_carrier_pair_evaluation import (
    calibrate_evidence_carrier_pair_threshold,
)
from chemvector.evidence_carrier_pair_scorer import (
    EvidenceCarrierPairFeatureTransformConfig,
    evidence_carrier_pair_scorer_loss,
)
from chemvector.train_evidence_carrier_extent import (
    CHECKPOINT_SCHEMA_VERSION as P0_EXTENT_CHECKPOINT_SCHEMA_VERSION,
    VALIDATION_SCHEMA_VERSION as P0_EXTENT_VALIDATION_SCHEMA_VERSION,
)
from chemvector.train_evidence_carrier_pair_scorer import (
    load_train_only_cache,
    pair_scorer_train_cal_gate,
)


CHECKPOINT_SCHEMA_VERSION = "evidence-carrier-p0-pair-scorer-checkpoint-0.2"
SUMMARY_SCHEMA_VERSION = "evidence-carrier-p0-pair-scorer-train-cal-summary-0.2"
DEFAULT_BASE_CACHE = (
    "outputs/cache/"
    "chemvector-v0910-evidence-carrier-pair-training-wide-all-variants.pt"
)
DEFAULT_AUGMENTATION_CACHE = DEFAULT_P0_PAIR_AUGMENTATION_CACHE
DEFAULT_OUTPUT_DIR = (
    "outputs/checkpoints/"
    "chemvector-v0910-evidence-carrier-p0-step1000-fp32-"
    "pair-scorer-pilot9k-train-cal"
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_internal_test(value: Any) -> bool:
    return "internal_test" in str(value).lower().replace("-", "_")


def _plain_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, argparse.Namespace):
        return {name: _plain_json(item) for name, item in vars(value).items()}
    if isinstance(value, Mapping):
        return {str(name): _plain_json(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    return value


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_plain_json(payload), ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def _record_tensor(
    record: Mapping[str, Any],
    name: str,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if name not in record:
        raise ValueError(f"Frozen-P0 pair record lacks {name}")
    value = record[name]
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.to(dtype=dtype)


def _validate_joined_record(
    record: Mapping[str, Any],
    *,
    anchor_feature_dim: int,
    check_values: bool,
) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("Frozen-P0 pair records must be mappings")
    anchors = _record_tensor(record, "anchor_features", dtype=torch.float32)
    features = _record_tensor(record, "pair_features", dtype=torch.float32)
    consistency = _record_tensor(
        record, "pair_consistency_features", dtype=torch.float32
    )
    indices = _record_tensor(record, "pair_indices", dtype=torch.long)
    labels = _record_tensor(record, "pair_labels", dtype=torch.long)
    if anchors.ndim != 2 or anchors.size(-1) != anchor_feature_dim:
        raise ValueError("Frozen-P0 anchor features have an invalid shape")
    if features.ndim != 2 or features.size(-1) != PAIR_FEATURE_DIM:
        raise ValueError("Frozen-P0 pair features must have shape [pairs,8]")
    pair_count = len(features)
    if consistency.shape != (pair_count, P0_PAIR_CONSISTENCY_FEATURE_DIM):
        raise ValueError("Frozen-P0 consistency features must have shape [pairs,12]")
    if indices.shape != (pair_count, 2):
        raise ValueError("Frozen-P0 pair indices must have shape [pairs,2]")
    if labels.shape != (pair_count,):
        raise ValueError("Frozen-P0 pair labels must have shape [pairs]")
    if pair_count and (
        bool((indices < 0).any())
        or bool((indices >= len(anchors)).any())
        or bool((indices[:, 0] == indices[:, 1]).any())
    ):
        raise ValueError("Frozen-P0 pair indices exceed this sample's anchors")
    valid_labels = (labels == PAIR_NEGATIVE_LABEL) | (labels == PAIR_POSITIVE_LABEL)
    if pair_count and not bool(valid_labels.all()):
        raise ValueError("Frozen-P0 pair labels must be binary")
    if check_values and (
        not bool(torch.isfinite(anchors).all())
        or not bool(torch.isfinite(features).all())
        or not bool(torch.isfinite(consistency).all())
    ):
        raise ValueError("Frozen-P0 pair records contain non-finite values")


def load_train_only_p0_pair_caches(
    base_cache_path: str | Path,
    augmentation_cache_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_paths = (str(base_cache_path), str(augmentation_cache_path))
    resolved_paths = tuple(
        Path(path).expanduser().resolve(strict=False) for path in raw_paths
    )
    if any(_contains_internal_test(path) for path in (*raw_paths, *resolved_paths)):
        raise ValueError("Frozen-P0 pair training cannot read internal_test caches")
    base_cache = load_train_only_cache(resolved_paths[0])
    augmentation_cache = load_evidence_carrier_p0_pair_cache(
        resolved_paths[1],
        base_cache=base_cache,
        require_complete=True,
    )
    if augmentation_cache["base_cache_sha256"] != base_cache["cache_sha256"]:
        raise ValueError("Frozen-P0 augmentation is not bound to the loaded base cache")
    return base_cache, augmentation_cache


def verify_frozen_p0_provenance(
    augmentation_cache: Mapping[str, Any],
) -> dict[str, Any]:
    raw_path = augmentation_cache.get("p0_checkpoint_path")
    expected_sha = augmentation_cache.get("p0_checkpoint_sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("Frozen-P0 augmentation lacks its P0 checkpoint path")
    resolved_path = Path(raw_path).expanduser().resolve(strict=False)
    if _contains_internal_test(raw_path) or _contains_internal_test(resolved_path):
        raise ValueError("Frozen P0 checkpoint cannot reference internal_test")
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Frozen P0 checkpoint is missing: {resolved_path}")
    actual_sha = file_sha256(resolved_path)
    if not _valid_sha256(expected_sha) or actual_sha != expected_sha:
        raise ValueError("Frozen P0 checkpoint SHA differs from the augmentation cache")
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Frozen P0 checkpoint must be a dictionary")
    if checkpoint.get("uses_internal_test") not in (None, False):
        raise ValueError("Frozen P0 checkpoint used internal_test")
    expected_selected = augmentation_cache.get(
        "p0_checkpoint_selected_by_validation"
    )
    expected_contains_metrics = augmentation_cache.get(
        "p0_checkpoint_contains_validation_metrics"
    )
    if expected_selected is not False:
        raise ValueError("Frozen P0 sidecar must declare validation-independent selection")
    if expected_contains_metrics is not True:
        raise ValueError("Frozen P0 sidecar must declare contained validation metrics")
    if augmentation_cache.get(
        "p0_checkpoint_schema_version"
    ) != P0_EXTENT_CHECKPOINT_SCHEMA_VERSION or augmentation_cache.get(
        "p0_validation_metrics_schema_version"
    ) != P0_EXTENT_VALIDATION_SCHEMA_VERSION:
        raise ValueError("Frozen P0 sidecar checkpoint/validation schema differs")
    global_step = augmentation_cache.get("p0_global_step")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 1
        or resolved_path.name != f"step-{global_step:06d}.pt"
    ):
        raise ValueError("Frozen P0 sidecar path/global_step provenance differs")
    validation_metrics = checkpoint.get("validation")
    contains_validation_metrics = isinstance(validation_metrics, dict) and bool(
        validation_metrics
    )
    if contains_validation_metrics is not expected_contains_metrics:
        raise ValueError("Frozen P0 validation-metrics provenance differs from sidecar")
    if validation_metrics.get("schema_version") != augmentation_cache.get(
        "p0_validation_metrics_schema_version"
    ):
        raise ValueError("Frozen P0 validation schema differs from sidecar")
    expected = {
        "checkpoint_schema_version": augmentation_cache.get(
            "p0_checkpoint_schema_version"
        ),
        "model_type": augmentation_cache.get("p0_model_type"),
        "global_step": augmentation_cache.get("p0_global_step"),
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Frozen P0 checkpoint metadata differs from sidecar: {mismatches}")
    return {
        "path": str(resolved_path),
        "sha256": actual_sha,
        "model_type": checkpoint.get("model_type"),
        "global_step": checkpoint.get("global_step"),
        "checkpoint_schema_version": checkpoint.get("checkpoint_schema_version"),
        "contains_validation_metrics": contains_validation_metrics,
        "selected_by_validation": expected_selected,
        "upstream_validation_read": contains_validation_metrics,
        "checkpoint_selected_by_validation": expected_selected,
        "uses_internal_test": False,
    }


def joined_training_records(
    base_cache: Mapping[str, Any],
    augmentation_cache: Mapping[str, Any],
    partition: str,
) -> list[dict[str, Any]]:
    records = joined_p0_pair_records(base_cache, augmentation_cache, partition)
    feature_dim = int(augmentation_cache["anchor_feature_dim"])
    for record in records:
        _validate_joined_record(
            record,
            anchor_feature_dim=feature_dim,
            check_values=True,
        )
    return records


def collate_p0_pair_training_records(
    records: Sequence[Mapping[str, Any]],
    *,
    anchor_feature_dim: int,
) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("Cannot collate an empty frozen-P0 pair batch")
    for record in records:
        _validate_joined_record(
            record,
            anchor_feature_dim=anchor_feature_dim,
            check_values=False,
        )
    anchor_counts = [len(record["anchor_features"]) for record in records]
    pair_counts = [len(record["pair_labels"]) for record in records]
    anchor_capacity = max(max(anchor_counts), 1)
    pair_capacity = max(max(pair_counts), 1)
    batch_size = len(records)
    result = {
        "anchor_features": torch.zeros(
            (batch_size, anchor_capacity, anchor_feature_dim), dtype=torch.float32
        ),
        "pair_indices": torch.full(
            (batch_size, pair_capacity, 2), -1, dtype=torch.long
        ),
        "pair_features": torch.zeros(
            (batch_size, pair_capacity, PAIR_FEATURE_DIM), dtype=torch.float32
        ),
        "pair_consistency_features": torch.zeros(
            (batch_size, pair_capacity, P0_PAIR_CONSISTENCY_FEATURE_DIM),
            dtype=torch.float32,
        ),
        "pair_labels": torch.full(
            (batch_size, pair_capacity), -1, dtype=torch.long
        ),
        "pair_mask": torch.zeros(
            (batch_size, pair_capacity), dtype=torch.bool
        ),
        "anchor_count": torch.tensor(anchor_counts, dtype=torch.long),
        "pair_count": torch.tensor(pair_counts, dtype=torch.long),
    }
    for batch_index, record in enumerate(records):
        anchor_count = anchor_counts[batch_index]
        pair_count = pair_counts[batch_index]
        if anchor_count:
            result["anchor_features"][batch_index, :anchor_count] = _record_tensor(
                record, "anchor_features", dtype=torch.float32
            )
        if pair_count:
            result["pair_indices"][batch_index, :pair_count] = _record_tensor(
                record, "pair_indices", dtype=torch.long
            )
            result["pair_features"][batch_index, :pair_count] = _record_tensor(
                record, "pair_features", dtype=torch.float32
            )
            result["pair_consistency_features"][batch_index, :pair_count] = (
                _record_tensor(
                    record, "pair_consistency_features", dtype=torch.float32
                )
            )
            result["pair_labels"][batch_index, :pair_count] = _record_tensor(
                record, "pair_labels", dtype=torch.long
            )
            result["pair_mask"][batch_index, :pair_count] = True
    return result


def fit_p0_pair_feature_normalization(
    model: EvidenceCarrierP0PairScorer,
    fit_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pair_rows = []
    consistency_rows = []
    for record in fit_records:
        pair_features = _record_tensor(record, "pair_features", dtype=torch.float32)
        if not len(pair_features):
            continue
        pair_rows.append(pair_features)
        consistency_rows.append(
            _record_tensor(
                record, "pair_consistency_features", dtype=torch.float32
            )
        )
    if not pair_rows:
        raise ValueError("Frozen-P0 pair fit partition has no candidate rows")
    features = torch.cat(pair_rows, dim=0).unsqueeze(0)
    consistency = torch.cat(consistency_rows, dim=0).unsqueeze(0)
    mask = torch.ones(features.shape[:2], dtype=torch.bool)
    return model.fit_feature_normalization(
        features,
        consistency,
        mask,
        source_partition="train-fit",
    )


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _to_device(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device)
        for name, value in batch.items()
        if name not in {"anchor_count", "pair_count"}
    }


def _hard_negative_loss(
    pair_gains: torch.Tensor,
    pair_labels: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, int]:
    zero = pair_gains.sum() * 0.0
    parts = []
    for batch_index in range(len(pair_gains)):
        negative = pair_mask[batch_index] & (
            pair_labels[batch_index] == PAIR_NEGATIVE_LABEL
        )
        if not bool(negative.any()):
            continue
        scores = pair_gains[batch_index, negative].float()
        hardest = scores.topk(min(top_k, len(scores))).values
        parts.append(F.softplus(hardest).mean())
    return (torch.stack(parts).mean() if parts else zero), len(parts)


def p0_pair_scorer_loss(
    pair_gains: torch.Tensor,
    pair_indices: torch.Tensor,
    pair_labels: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    hard_negative_top_k: int,
    hard_negative_weight: float,
) -> dict[str, Any]:
    base = evidence_carrier_pair_scorer_loss(
        pair_gains,
        pair_indices,
        pair_labels,
        pair_mask,
    )
    hard_negative, image_count = _hard_negative_loss(
        pair_gains,
        pair_labels,
        pair_mask,
        top_k=hard_negative_top_k,
    )
    return {
        **base,
        "base_loss": base["loss"],
        "hard_negative_loss": hard_negative,
        "images_with_hard_negatives": image_count,
        "loss": base["loss"] + float(hard_negative_weight) * hard_negative,
    }


def _loss_for_batch(
    model: EvidenceCarrierP0PairScorer,
    batch: Mapping[str, torch.Tensor],
    *,
    hard_negative_top_k: int,
    hard_negative_weight: float,
) -> dict[str, Any]:
    output = model(
        batch["anchor_features"],
        batch["pair_indices"],
        batch["pair_features"],
        batch["pair_consistency_features"],
        batch["pair_mask"],
    )
    return p0_pair_scorer_loss(
        output["pair_gains"],
        batch["pair_indices"],
        batch["pair_labels"],
        batch["pair_mask"],
        hard_negative_top_k=hard_negative_top_k,
        hard_negative_weight=hard_negative_weight,
    )


def _batches(
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    order: Sequence[int] | None = None,
) -> list[Sequence[Mapping[str, Any]]]:
    indices = list(range(len(records))) if order is None else list(order)
    return [
        [records[index] for index in indices[start : start + batch_size]]
        for start in range(0, len(indices), batch_size)
    ]


def train_p0_pair_scorer_epoch(
    model: EvidenceCarrierP0PairScorer,
    optimizer: torch.optim.Optimizer,
    fit_records: Sequence[Mapping[str, Any]],
    *,
    anchor_feature_dim: int,
    batch_size: int,
    device: torch.device,
    rng: random.Random,
    max_grad_norm: float,
    hard_negative_top_k: int,
    hard_negative_weight: float,
) -> dict[str, float]:
    model.train()
    order = list(range(len(fit_records)))
    rng.shuffle(order)
    totals = {
        "loss": 0.0,
        "base_loss": 0.0,
        "balanced_bce": 0.0,
        "ranking_loss": 0.0,
        "hard_negative_loss": 0.0,
    }
    batch_count = 0
    for records in _batches(fit_records, batch_size=batch_size, order=order):
        batch = _to_device(
            collate_p0_pair_training_records(
                records, anchor_feature_dim=anchor_feature_dim
            ),
            device,
        )
        losses = _loss_for_batch(
            model,
            batch,
            hard_negative_top_k=hard_negative_top_k,
            hard_negative_weight=hard_negative_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        for name in totals:
            totals[name] += float(losses[name].detach().cpu())
        batch_count += 1
    return {
        **{name: value / max(batch_count, 1) for name, value in totals.items()},
        "batches": float(batch_count),
    }


@torch.no_grad()
def evaluate_p0_pair_scorer_loss(
    model: EvidenceCarrierP0PairScorer,
    records: Sequence[Mapping[str, Any]],
    *,
    anchor_feature_dim: int,
    batch_size: int,
    device: torch.device,
    hard_negative_top_k: int,
    hard_negative_weight: float,
) -> dict[str, float]:
    model.eval()
    weighted_sums = {
        "balanced_bce": 0.0,
        "positive_bce": 0.0,
        "negative_bce": 0.0,
        "ranking_loss": 0.0,
        "hard_negative_loss": 0.0,
    }
    denominators = {name: 0 for name in weighted_sums}
    count_fields = {
        "balanced_bce": "images_with_pairs",
        "positive_bce": "images_with_positive_pairs",
        "negative_bce": "images_with_negative_pairs",
        "ranking_loss": "images_with_ranking_pairs",
        "hard_negative_loss": "images_with_hard_negatives",
    }
    batch_count = 0
    for current in _batches(records, batch_size=batch_size):
        batch = _to_device(
            collate_p0_pair_training_records(
                current, anchor_feature_dim=anchor_feature_dim
            ),
            device,
        )
        losses = _loss_for_batch(
            model,
            batch,
            hard_negative_top_k=hard_negative_top_k,
            hard_negative_weight=hard_negative_weight,
        )
        for name, count_name in count_fields.items():
            count = int(losses[count_name])
            weighted_sums[name] += float(losses[name].detach().cpu()) * count
            denominators[name] += count
        batch_count += 1
    metrics = {
        name: weighted_sums[name] / denominators[name]
        if denominators[name]
        else 0.0
        for name in weighted_sums
    }
    metrics["base_loss"] = metrics["balanced_bce"] + 0.5 * metrics["ranking_loss"]
    metrics["loss"] = (
        metrics["base_loss"]
        + float(hard_negative_weight) * metrics["hard_negative_loss"]
    )
    return {
        **metrics,
        "images_with_pairs": float(denominators["balanced_bce"]),
        "images_with_ranking_pairs": float(denominators["ranking_loss"]),
        "images_with_hard_negatives": float(denominators["hard_negative_loss"]),
        "batches": float(batch_count),
    }


@torch.no_grad()
def score_p0_pair_records(
    model: EvidenceCarrierP0PairScorer,
    records: Sequence[Mapping[str, Any]],
    *,
    anchor_feature_dim: int,
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    model.eval()
    scores: list[torch.Tensor] = []
    for current in _batches(records, batch_size=batch_size):
        collated = collate_p0_pair_training_records(
            current, anchor_feature_dim=anchor_feature_dim
        )
        counts = collated["pair_count"].tolist()
        batch = _to_device(collated, device)
        gains = model(
            batch["anchor_features"],
            batch["pair_indices"],
            batch["pair_features"],
            batch["pair_consistency_features"],
            batch["pair_mask"],
        )["pair_gains"]
        for row, count in zip(gains.detach().cpu(), counts, strict=True):
            scores.append(row[: int(count)].clone())
    if len(scores) != len(records):
        raise RuntimeError("Frozen-P0 pair scorer returned the wrong sample count")
    return scores


def _class_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    positive = negative = pairs = 0
    for record in records:
        labels = _record_tensor(record, "pair_labels", dtype=torch.long)
        positive += int((labels == PAIR_POSITIVE_LABEL).sum())
        negative += int((labels == PAIR_NEGATIVE_LABEL).sum())
        pairs += len(labels)
    return {
        "pairs": pairs,
        "same_target_positive": positive,
        "distinct_target_hard_negative": negative,
    }


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _checkpoint_payload(
    model: EvidenceCarrierP0PairScorer,
    *,
    args: argparse.Namespace,
    base_cache: Mapping[str, Any],
    augmentation_cache: Mapping[str, Any],
    p0_provenance: Mapping[str, Any],
    best_epoch: int,
    optimizer_steps: int,
    normalization: Mapping[str, Any],
    best_calibration_loss: Mapping[str, Any],
    threshold_calibration: Mapping[str, Any],
    gate: Mapping[str, Any],
    checkpoint_selection: Mapping[str, Any],
) -> dict[str, Any]:
    selected_calibration = {
        key: value
        for key, value in threshold_calibration.items()
        if key not in {"sample_evaluations", "threshold_trace"}
    }
    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": model.model_type,
        "model": model.state_dict(),
        "model_config": model.config_dict(),
        "parameter_count": model.parameter_count(),
        "pair_feature_names": list(PAIR_FEATURE_NAMES),
        "pair_consistency_feature_names": list(P0_PAIR_CONSISTENCY_FEATURE_NAMES),
        "normalization": dict(normalization),
        "loss_config": {
            "base_loss": "balanced-per-image-bce-plus-shared-anchor-ranking",
            "hard_negative_top_k": int(args.hard_negative_top_k),
            "hard_negative_weight": float(args.hard_negative_weight),
        },
        "gain_threshold": float(threshold_calibration["threshold"]),
        "threshold_calibration": selected_calibration,
        "train_cal_gate": dict(gate),
        "checkpoint_selection": _plain_json(checkpoint_selection),
        "best_epoch": int(best_epoch),
        "optimizer_steps": int(optimizer_steps),
        "best_calibration_loss": dict(best_calibration_loss),
        "training_args": _plain_json(args),
        "base_cache_path": base_cache["cache_path"],
        "base_cache_sha256": base_cache["cache_sha256"],
        "base_cache_schema_version": base_cache.get("cache_schema_version"),
        "augmentation_cache_path": augmentation_cache["cache_path"],
        "augmentation_cache_sha256": augmentation_cache["cache_sha256"],
        "augmentation_cache_schema_version": augmentation_cache.get(
            "cache_schema_version"
        ),
        "augmentation_cache_complete": augmentation_cache.get("complete"),
        "frozen_p0_checkpoint_path": p0_provenance["path"],
        "frozen_p0_checkpoint_sha256": p0_provenance["sha256"],
        "p0_frozen": augmentation_cache["p0_frozen"],
        "p0_model_type": augmentation_cache.get("p0_model_type"),
        "p0_global_step": augmentation_cache.get("p0_global_step"),
        "p0_checkpoint_schema_version": p0_provenance[
            "checkpoint_schema_version"
        ],
        "p0_checkpoint_contains_validation_metrics": p0_provenance[
            "contains_validation_metrics"
        ],
        "p0_checkpoint_selected_by_validation": p0_provenance[
            "selected_by_validation"
        ],
        "upstream_p0_validation_read": p0_provenance[
            "upstream_validation_read"
        ],
        "upstream_p0_checkpoint_selected_by_validation": p0_provenance[
            "checkpoint_selected_by_validation"
        ],
        "anchor_feature_dim": augmentation_cache["anchor_feature_dim"],
        "pairing_profile": base_cache.get("pairing_profile", "training_wide"),
        "pairing_config": _plain_json(
            base_cache.get("config", {}).get("pairing_config", {})
        ),
        "uses_internal_test": False,
        "validation_read": False,
    }


def save_p0_pair_scorer_checkpoint(
    path: str | Path,
    payload: Mapping[str, Any],
) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(output)
    return file_sha256(output)


def load_p0_pair_scorer_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    verify_source_hashes: bool = True,
) -> tuple[EvidenceCarrierP0PairScorer, dict[str, Any]]:
    raw_path = str(path)
    resolved_path = Path(path).expanduser().resolve(strict=False)
    if _contains_internal_test(raw_path) or _contains_internal_test(resolved_path):
        raise ValueError("Frozen-P0 pair checkpoint cannot reference internal_test")
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Frozen-P0 pair checkpoint must be a dictionary")
    if not isinstance(verify_source_hashes, bool):
        raise ValueError("verify_source_hashes must be boolean")
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": EvidenceCarrierP0PairScorer.model_type,
        "uses_internal_test": False,
        "validation_read": False,
        "pair_feature_names": list(PAIR_FEATURE_NAMES),
        "pair_consistency_feature_names": list(P0_PAIR_CONSISTENCY_FEATURE_NAMES),
        "base_cache_schema_version": BASE_PAIR_CACHE_SCHEMA_VERSION,
        "augmentation_cache_schema_version": P0_PAIR_CACHE_SCHEMA_VERSION,
        "augmentation_cache_complete": True,
        "p0_frozen": True,
        "p0_checkpoint_contains_validation_metrics": True,
        "p0_checkpoint_selected_by_validation": False,
        "upstream_p0_validation_read": True,
        "upstream_p0_checkpoint_selected_by_validation": False,
        "p0_checkpoint_schema_version": P0_EXTENT_CHECKPOINT_SCHEMA_VERSION,
        "p0_model_type": "evidence-carrier-extent-v0.1",
        "pairing_profile": PAIRING_PROFILE_TRAINING_WIDE,
        "pairing_config": asdict(
            conditional_pair_pairing_config(PAIRING_PROFILE_TRAINING_WIDE)
        ),
    }
    mismatches = {
        name: (checkpoint.get(name), expected_value)
        for name, expected_value in expected.items()
        if checkpoint.get(name) != expected_value
    }
    if mismatches:
        raise ValueError(f"Frozen-P0 pair checkpoint contract differs: {mismatches}")
    p0_global_step = checkpoint.get("p0_global_step")
    if (
        isinstance(p0_global_step, bool)
        or not isinstance(p0_global_step, int)
        or p0_global_step < 1
    ):
        raise ValueError("Frozen-P0 pair checkpoint p0_global_step is invalid")
    p0_path = Path(checkpoint.get("frozen_p0_checkpoint_path", ""))
    if p0_path.name != f"step-{p0_global_step:06d}.pt":
        raise ValueError("Frozen-P0 pair checkpoint P0 filename and global_step differ")
    threshold = checkpoint.get("gain_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("Frozen-P0 pair gain threshold must be finite")
    provenance = (
        ("base_cache_path", "base_cache_sha256"),
        ("augmentation_cache_path", "augmentation_cache_sha256"),
        ("frozen_p0_checkpoint_path", "frozen_p0_checkpoint_sha256"),
    )
    for path_name, sha_name in provenance:
        source_path = checkpoint.get(path_name)
        source_sha = checkpoint.get(sha_name)
        if (
            not isinstance(source_path, str)
            or not source_path
            or _contains_internal_test(source_path)
            or not _valid_sha256(source_sha)
        ):
            raise ValueError(
                f"Frozen-P0 pair checkpoint provenance {path_name} is invalid"
            )
        if verify_source_hashes:
            resolved_source = Path(source_path).expanduser().resolve(strict=False)
            if _contains_internal_test(resolved_source):
                raise ValueError(
                    f"Frozen-P0 pair checkpoint provenance {path_name} is invalid"
                )
            if not resolved_source.is_file():
                raise FileNotFoundError(
                    f"Frozen-P0 pair provenance file is missing: {resolved_source}"
                )
            if file_sha256(resolved_source) != source_sha:
                raise ValueError(
                    f"Frozen-P0 pair checkpoint {sha_name} no longer matches"
                )
    if not isinstance(checkpoint.get("train_cal_gate"), dict):
        raise ValueError("Frozen-P0 pair checkpoint lacks train-cal gate metadata")
    if not isinstance(checkpoint.get("checkpoint_selection"), dict):
        raise ValueError("Frozen-P0 pair checkpoint lacks selection metadata")
    if not isinstance(checkpoint.get("loss_config"), dict):
        raise ValueError("Frozen-P0 pair checkpoint lacks loss metadata")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Frozen-P0 pair checkpoint model config is invalid")
    try:
        config = EvidenceCarrierP0PairScorerConfig(**model_config["config"])
        pair_transform = EvidenceCarrierPairFeatureTransformConfig(
            **model_config["pair_transform"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Frozen-P0 pair checkpoint model config is malformed") from error
    model = EvidenceCarrierP0PairScorer(config, pair_transform)
    if checkpoint.get("parameter_count") != model.parameter_count():
        raise ValueError("Frozen-P0 pair checkpoint parameter count differs")
    if config == EvidenceCarrierP0PairScorerConfig() and (
        model.parameter_count() != P0_PAIR_SCORER_PARAMETER_COUNT
    ):
        raise ValueError("Frozen-P0 pair checkpoint default parameter contract differs")
    if checkpoint.get("anchor_feature_dim") != config.anchor_feature_dim:
        raise ValueError("Frozen-P0 pair checkpoint anchor feature width differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    if model_config != model.config_dict():
        raise ValueError("Frozen-P0 pair checkpoint model config differs from weights")
    if not bool(model.normalization_fitted) or int(model.normalization_sample_count) < 2:
        raise ValueError("Frozen-P0 pair checkpoint normalization is not fitted")
    if (
        not bool(torch.isfinite(model.descriptor_mean).all())
        or not bool(torch.isfinite(model.descriptor_scale).all())
        or bool((model.descriptor_scale <= 0.0).any())
    ):
        raise ValueError("Frozen-P0 pair normalization buffers are invalid")
    normalization = checkpoint.get("normalization")
    if (
        not isinstance(normalization, dict)
        or normalization.get("source_partition") != "train-fit"
        or normalization.get("sample_count") != int(model.normalization_sample_count)
    ):
        raise ValueError("Frozen-P0 pair normalization metadata is invalid")
    try:
        declared_mean = torch.tensor(
            normalization["descriptor_mean"], dtype=torch.float32
        )
        declared_scale = torch.tensor(
            normalization["descriptor_scale"], dtype=torch.float32
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Frozen-P0 pair normalization metadata is malformed") from error
    if not torch.equal(declared_mean, model.descriptor_mean.cpu()) or not torch.equal(
        declared_scale, model.descriptor_scale.cpu()
    ):
        raise ValueError("Frozen-P0 pair normalization metadata differs from buffers")
    model.to(torch.device(device))
    return model, checkpoint


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Frozen-P0 pair output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_cache, augmentation_cache = load_train_only_p0_pair_caches(
        args.base_cache,
        args.augmentation_cache,
    )
    p0_provenance = verify_frozen_p0_provenance(augmentation_cache)
    fit_records = joined_training_records(base_cache, augmentation_cache, "fit")
    calibration_records = joined_training_records(
        base_cache, augmentation_cache, "calibration"
    )
    anchor_feature_dim = int(augmentation_cache["anchor_feature_dim"])
    model = EvidenceCarrierP0PairScorer(
        EvidenceCarrierP0PairScorerConfig(
            **augmentation_cache["pair_consistency_config"]
        )
    )
    normalization = fit_p0_pair_feature_normalization(model, fit_records)
    device = _device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    rng = random.Random(args.seed)
    history = []
    checkpoint_candidates: list[dict[str, Any]] = []
    best_loss = math.inf
    best_loss_epoch = 0
    stale = 0
    optimizer_steps = 0
    for epoch in range(1, args.max_epochs + 1):
        train_metrics = train_p0_pair_scorer_epoch(
            model,
            optimizer,
            fit_records,
            anchor_feature_dim=anchor_feature_dim,
            batch_size=args.batch_size,
            device=device,
            rng=rng,
            max_grad_norm=args.max_grad_norm,
            hard_negative_top_k=args.hard_negative_top_k,
            hard_negative_weight=args.hard_negative_weight,
        )
        optimizer_steps += int(train_metrics["batches"])
        calibration_loss = evaluate_p0_pair_scorer_loss(
            model,
            calibration_records,
            anchor_feature_dim=anchor_feature_dim,
            batch_size=args.batch_size,
            device=device,
            hard_negative_top_k=args.hard_negative_top_k,
            hard_negative_weight=args.hard_negative_weight,
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "calibration_loss": calibration_loss,
        }
        history.append(row)
        print(json.dumps(_plain_json(row), ensure_ascii=True), flush=True)
        current = float(calibration_loss["loss"])
        checkpoint_candidates.append(
            {
                "epoch": epoch,
                "optimizer_steps": optimizer_steps,
                "calibration_loss": dict(calibration_loss),
                "state_dict": deepcopy(model.state_dict()),
            }
        )
        checkpoint_candidates.sort(
            key=lambda candidate: float(candidate["calibration_loss"]["loss"])
        )
        del checkpoint_candidates[args.checkpoint_candidates :]
        if current < best_loss - args.minimum_improvement:
            best_loss = current
            best_loss_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if not checkpoint_candidates:
        raise RuntimeError("Frozen-P0 pair training produced no checkpoint")
    calibration_component_count = base_cache.get("identity_partition", {}).get(
        "calibration_component_count"
    )
    structurally_scored = []
    for candidate_index, candidate in enumerate(checkpoint_candidates, start=1):
        print(
            json.dumps(
                {
                    "stage": "matching_calibration_start",
                    "candidate_index": candidate_index,
                    "candidate_count": len(checkpoint_candidates),
                    "epoch": int(candidate["epoch"]),
                    "calibration_loss": float(
                        candidate["calibration_loss"]["loss"]
                    ),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        model.load_state_dict(candidate["state_dict"], strict=True)
        calibration_scores = score_p0_pair_records(
            model,
            calibration_records,
            anchor_feature_dim=anchor_feature_dim,
            batch_size=args.batch_size,
            device=device,
        )
        calibration = calibrate_evidence_carrier_pair_threshold(
            calibration_records,
            calibration_scores,
            max_threshold_candidates=args.threshold_candidates,
        )
        gate = pair_scorer_train_cal_gate(
            calibration,
            calibration_records=calibration_records,
            calibration_component_count=calibration_component_count,
        )
        metrics = calibration["selected_metrics"]
        print(
            json.dumps(
                {
                    "stage": "matching_calibration_complete",
                    "candidate_index": candidate_index,
                    "candidate_count": len(checkpoint_candidates),
                    "epoch": int(candidate["epoch"]),
                    "threshold": float(calibration["threshold"]),
                    "gate": gate["status"],
                    "selected_cross_target_pairs": int(
                        metrics["selected_cross_target_pairs"]
                    ),
                    "multiplicity_2_recall": metrics["multiplicity_2_recall"],
                    "pairable_sample_exact": metrics["pairable_sample_exact"],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        structural_key = (
            int(metrics["selected_cross_target_pairs"] == 0),
            int(metrics["parallel_2_violated_group_count"] == 0),
            int(metrics["multiplicity_2_matched_count"]),
            int(metrics["pairable_sample_exact_count"]),
            -int(metrics["ordinary_support_lost_target_count"]),
            float(metrics["selected_pair_precision"] or 0.0),
            -float(candidate["calibration_loss"]["loss"]),
        )
        structurally_scored.append(
            {
                **candidate,
                "threshold_calibration": calibration,
                "gate": gate,
                "selection_key": structural_key,
            }
        )
    selected = max(
        structurally_scored,
        key=lambda candidate: candidate["selection_key"],
    )
    model.load_state_dict(selected["state_dict"], strict=True)
    best_epoch = int(selected["epoch"])
    selected_optimizer_steps = int(selected["optimizer_steps"])
    best_calibration_loss = selected["calibration_loss"]
    threshold_calibration = selected["threshold_calibration"]
    gate = selected["gate"]
    candidate_summary = [
        {
            "epoch": int(candidate["epoch"]),
            "optimizer_steps": int(candidate["optimizer_steps"]),
            "calibration_loss": candidate["calibration_loss"],
            "gain_threshold": candidate["threshold_calibration"]["threshold"],
            "selected_metrics": candidate["threshold_calibration"][
                "selected_metrics"
            ],
            "gate_status": candidate["gate"]["status"],
            "selection_key": list(candidate["selection_key"]),
            "selected": candidate is selected,
        }
        for candidate in structurally_scored
    ]
    checkpoint_selection = {
        "candidate_count": len(candidate_summary),
        "candidate_source": "lowest train-cal loss checkpoints",
        "metric_order": [
            "selected_cross_target_pairs == 0",
            "parallel_2_violated_group_count == 0",
            "maximize multiplicity_2_matched_count",
            "maximize pairable_sample_exact_count",
            "minimize ordinary_support_lost_target_count",
            "maximize selected_pair_precision",
            "minimize calibration loss",
        ],
        "candidates": candidate_summary,
    }
    payload = _checkpoint_payload(
        model,
        args=args,
        base_cache=base_cache,
        augmentation_cache=augmentation_cache,
        p0_provenance=p0_provenance,
        best_epoch=best_epoch,
        optimizer_steps=selected_optimizer_steps,
        normalization=normalization,
        best_calibration_loss=best_calibration_loss,
        threshold_calibration=threshold_calibration,
        gate=gate,
        checkpoint_selection=checkpoint_selection,
    )
    checkpoint_sha = save_p0_pair_scorer_checkpoint(
        output_dir / "best.pt", payload
    )
    write_json(output_dir / "calibration.json", threshold_calibration)
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": gate["status"],
        "model_type": model.model_type,
        "parameter_count": model.parameter_count(),
        "device": str(device),
        "best_epoch": best_epoch,
        "best_loss_epoch": best_loss_epoch,
        "epochs_completed": len(history),
        "optimizer_steps": selected_optimizer_steps,
        "total_optimizer_steps": optimizer_steps,
        "fit_samples": len(fit_records),
        "calibration_samples": len(calibration_records),
        "fit_class_counts": _class_counts(fit_records),
        "calibration_class_counts": _class_counts(calibration_records),
        "normalization": normalization,
        "loss_config": payload["loss_config"],
        "best_calibration_loss": best_calibration_loss,
        "gain_threshold": threshold_calibration["threshold"],
        "calibration_metrics": threshold_calibration["selected_metrics"],
        "train_cal_gate": gate,
        "checkpoint_selection": checkpoint_selection,
        "history": history,
        "checkpoint": "best.pt",
        "checkpoint_sha256": checkpoint_sha,
        "base_cache_path": base_cache["cache_path"],
        "base_cache_sha256": base_cache["cache_sha256"],
        "augmentation_cache_path": augmentation_cache["cache_path"],
        "augmentation_cache_sha256": augmentation_cache["cache_sha256"],
        "frozen_p0_checkpoint_path": p0_provenance["path"],
        "frozen_p0_checkpoint_sha256": p0_provenance["sha256"],
        "upstream_p0_validation_read": p0_provenance[
            "upstream_validation_read"
        ],
        "upstream_p0_checkpoint_selected_by_validation": p0_provenance[
            "checkpoint_selected_by_validation"
        ],
        "validation_read": False,
        "uses_internal_test": False,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", default=DEFAULT_BASE_CACHE)
    parser.add_argument("--augmentation-cache", default=DEFAULT_AUGMENTATION_CACHE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=9911)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--minimum-improvement", type=float, default=1e-5)
    parser.add_argument("--checkpoint-candidates", type=int, default=5)
    parser.add_argument("--threshold-candidates", type=int, default=32)
    parser.add_argument("--hard-negative-top-k", type=int, default=4)
    parser.add_argument("--hard-negative-weight", type=float, default=0.5)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    integer_positive = (
        "max_epochs",
        "batch_size",
        "patience",
        "checkpoint_candidates",
        "threshold_candidates",
        "hard_negative_top_k",
    )
    for name in integer_positive:
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("Optimizer values are invalid")
    if args.max_grad_norm <= 0.0 or args.minimum_improvement < 0.0:
        raise ValueError("Gradient and improvement values are invalid")
    if (
        not math.isfinite(float(args.hard_negative_weight))
        or args.hard_negative_weight < 0.0
    ):
        raise ValueError("hard_negative_weight must be finite and nonnegative")
    paths = (args.base_cache, args.augmentation_cache, args.output_dir)
    resolved = tuple(Path(path).expanduser().resolve(strict=False) for path in paths)
    if any(_contains_internal_test(path) for path in (*paths, *resolved)):
        raise ValueError("Frozen-P0 pair train-cal paths cannot reference internal_test")


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    summary = train(args)
    print(json.dumps(_plain_json(summary), ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "collate_p0_pair_training_records",
    "evaluate_p0_pair_scorer_loss",
    "fit_p0_pair_feature_normalization",
    "joined_training_records",
    "load_p0_pair_scorer_checkpoint",
    "load_train_only_p0_pair_caches",
    "p0_pair_scorer_loss",
    "save_p0_pair_scorer_checkpoint",
    "score_p0_pair_records",
    "train_p0_pair_scorer_epoch",
    "verify_frozen_p0_provenance",
]
