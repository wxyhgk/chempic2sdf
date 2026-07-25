#!/usr/bin/env python3
"""Train the v0.9.10 geometry-only Evidence Carrier pair scorer.

This command consumes a train-only feature cache.  It fits model weights on the
``fit`` partition, selects the checkpoint and relative-gain threshold on the
train-derived ``calibration`` partition, and has no validation-shard argument.
Public validation is intentionally a separate, explicit evaluation stage.
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

from chemvector.evidence_carrier_conditional_pair_data import (
    PAIR_FEATURE_DIM,
    PAIR_FEATURE_NAMES,
    PAIR_NEGATIVE_LABEL,
    PAIR_POSITIVE_LABEL,
    PAIRING_PROFILE_TRAINING_WIDE,
    conditional_pair_pairing_config,
)
from chemvector.evidence_carrier_pair_cache import (
    CACHE_SCHEMA_VERSION,
    load_evidence_carrier_pair_cache,
)
from chemvector.evidence_carrier_pair_evaluation import (
    calibrate_evidence_carrier_pair_threshold,
)
from chemvector.evidence_carrier_pair_scorer import (
    PAIR_SCORER_PARAMETER_COUNT,
    EvidenceCarrierPairFeatureTransformConfig,
    EvidenceCarrierPairScorer,
    evidence_carrier_pair_scorer_loss,
)


CHECKPOINT_SCHEMA_VERSION = "evidence-carrier-pair-scorer-checkpoint-0.1"
SUMMARY_SCHEMA_VERSION = "evidence-carrier-pair-scorer-train-cal-summary-0.1"
DEFAULT_OUTPUT_DIR = (
    "outputs/checkpoints/"
    "chemvector-v0910-evidence-carrier-pair-scorer-pilot9k-train-cal"
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
        raise ValueError(f"Pair cache record lacks {name}")
    value = record[name]
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.to(dtype=dtype)


def _validate_train_record(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise ValueError("Pair cache records must be mappings")
    features = _record_tensor(record, "pair_features", dtype=torch.float32)
    indices = _record_tensor(record, "pair_indices", dtype=torch.long)
    labels = _record_tensor(record, "pair_labels", dtype=torch.long)
    if features.ndim != 2 or features.size(-1) != PAIR_FEATURE_DIM:
        raise ValueError("Pair cache features must have shape [pairs,8]")
    if indices.shape != (len(features), 2):
        raise ValueError("Pair cache indices must have shape [pairs,2]")
    if labels.shape != (len(features),):
        raise ValueError("Pair cache labels must have shape [pairs]")
    if len(features) and not bool(torch.isfinite(features).all()):
        raise ValueError("Pair cache features contain non-finite values")
    if len(indices) and bool((indices < 0).any()):
        raise ValueError("Pair cache indices must be nonnegative")
    valid_labels = (labels == PAIR_NEGATIVE_LABEL) | (labels == PAIR_POSITIVE_LABEL)
    if len(labels) and not bool(valid_labels.all()):
        raise ValueError("Pair cache labels must be binary")


def pair_cache_records(payload: Mapping[str, Any], partition: str) -> list[dict[str, Any]]:
    if partition not in {"fit", "calibration"}:
        raise ValueError("Pair cache partition must be fit or calibration")
    records: Any = None
    nested = payload.get("records")
    if isinstance(nested, Mapping):
        records = nested.get(partition)
    if records is None:
        records = payload.get(f"{partition}_records")
    if not isinstance(records, (tuple, list)) or not records:
        raise ValueError(f"Pair cache has no nonempty {partition} records")
    result = [dict(record) for record in records]
    for record in result:
        _validate_train_record(record)
    return result


def load_train_only_cache(path: str | Path) -> dict[str, Any]:
    cache_path = Path(path).expanduser().resolve()
    if _contains_internal_test(cache_path):
        raise ValueError("Pair scorer training cannot read internal_test caches")
    payload = load_evidence_carrier_pair_cache(
        cache_path,
        validate_source_hashes=True,
    )
    pair_cache_records(payload, "fit")
    pair_cache_records(payload, "calibration")
    payload["cache_path"] = str(cache_path)
    payload["cache_sha256"] = file_sha256(cache_path)
    return payload


def collate_pair_training_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("Cannot collate an empty pair record batch")
    for record in records:
        _validate_train_record(record)
    counts = [len(_record_tensor(record, "pair_labels", dtype=torch.long)) for record in records]
    capacity = max(max(counts), 1)
    batch = len(records)
    result = {
        "pair_features": torch.zeros(
            (batch, capacity, PAIR_FEATURE_DIM), dtype=torch.float32
        ),
        "pair_indices": torch.full(
            (batch, capacity, 2), -1, dtype=torch.long
        ),
        "pair_labels": torch.full((batch, capacity), -1, dtype=torch.long),
        "pair_mask": torch.zeros((batch, capacity), dtype=torch.bool),
        "pair_count": torch.tensor(counts, dtype=torch.long),
    }
    for batch_index, (record, count) in enumerate(zip(records, counts, strict=True)):
        if not count:
            continue
        result["pair_features"][batch_index, :count] = _record_tensor(
            record, "pair_features", dtype=torch.float32
        )
        result["pair_indices"][batch_index, :count] = _record_tensor(
            record, "pair_indices", dtype=torch.long
        )
        result["pair_labels"][batch_index, :count] = _record_tensor(
            record, "pair_labels", dtype=torch.long
        )
        result["pair_mask"][batch_index, :count] = True
    return result


def fit_pair_feature_normalization(
    model: EvidenceCarrierPairScorer,
    fit_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = [
        _record_tensor(record, "pair_features", dtype=torch.float32)
        for record in fit_records
        if len(_record_tensor(record, "pair_features", dtype=torch.float32))
    ]
    if not rows:
        raise ValueError("Pair scorer fit partition has no candidate rows")
    features = torch.cat(rows, dim=0).unsqueeze(0)
    mask = torch.ones(features.shape[:2], dtype=torch.bool)
    return model.fit_feature_normalization(
        features,
        mask,
        source_partition="train-fit",
    )


def _device(name: str) -> torch.device:
    if name == "auto":
        # This model has only 833 parameters; CPU avoids per-image CUDA sync in
        # the structured ranking loss and is normally faster.
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device)
        for name, value in batch.items()
        if name != "pair_count"
    }


def _loss_for_batch(
    model: EvidenceCarrierPairScorer,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    output = model(batch["pair_features"], batch["pair_mask"])
    return evidence_carrier_pair_scorer_loss(
        output["pair_gains"],
        batch["pair_indices"],
        batch["pair_labels"],
        batch["pair_mask"],
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


def train_pair_scorer_epoch(
    model: EvidenceCarrierPairScorer,
    optimizer: torch.optim.Optimizer,
    fit_records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
    rng: random.Random,
    max_grad_norm: float,
) -> dict[str, float]:
    model.train()
    order = list(range(len(fit_records)))
    rng.shuffle(order)
    totals: dict[str, float] = {
        "loss": 0.0,
        "balanced_bce": 0.0,
        "ranking_loss": 0.0,
    }
    batch_count = 0
    for records in _batches(fit_records, batch_size=batch_size, order=order):
        batch = _to_device(collate_pair_training_records(records), device)
        losses = _loss_for_batch(model, batch)
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
def evaluate_pair_scorer_loss(
    model: EvidenceCarrierPairScorer,
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    weighted_sums: dict[str, float] = {
        "balanced_bce": 0.0,
        "positive_bce": 0.0,
        "negative_bce": 0.0,
        "ranking_loss": 0.0,
    }
    denominators: dict[str, int] = {
        "balanced_bce": 0,
        "positive_bce": 0,
        "negative_bce": 0,
        "ranking_loss": 0,
    }
    count_fields = {
        "balanced_bce": "images_with_pairs",
        "positive_bce": "images_with_positive_pairs",
        "negative_bce": "images_with_negative_pairs",
        "ranking_loss": "images_with_ranking_pairs",
    }
    batch_count = 0
    for current in _batches(records, batch_size=batch_size):
        batch = _to_device(collate_pair_training_records(current), device)
        losses = _loss_for_batch(model, batch)
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
    metrics["loss"] = metrics["balanced_bce"] + 0.5 * metrics["ranking_loss"]
    return {
        **metrics,
        "images_with_pairs": float(denominators["balanced_bce"]),
        "images_with_ranking_pairs": float(denominators["ranking_loss"]),
        "batches": float(batch_count),
    }


@torch.no_grad()
def score_pair_records(
    model: EvidenceCarrierPairScorer,
    records: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> list[torch.Tensor]:
    model.eval()
    scores: list[torch.Tensor] = []
    for current in _batches(records, batch_size=batch_size):
        collated = collate_pair_training_records(current)
        counts = collated["pair_count"].tolist()
        batch = _to_device(collated, device)
        gains = model(batch["pair_features"], batch["pair_mask"])["pair_gains"]
        for row, count in zip(gains.detach().cpu(), counts, strict=True):
            scores.append(row[: int(count)].clone())
    if len(scores) != len(records):
        raise RuntimeError("Pair scorer produced a different number of sample score rows")
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


def pair_scorer_train_cal_gate(
    calibration: Mapping[str, Any],
    *,
    calibration_records: Sequence[Mapping[str, Any]],
    calibration_component_count: int,
) -> dict[str, Any]:
    metrics = calibration["selected_metrics"]
    class_counts = _class_counts(calibration_records)
    identities = {
        str(record.get("identity", {}).get("molecule_id", record.get("molecule_id", "")))
        for record in calibration_records
    }
    identities.discard("")
    if (
        isinstance(calibration_component_count, bool)
        or not isinstance(calibration_component_count, int)
        or calibration_component_count < 1
    ):
        raise ValueError("calibration_component_count must be a positive integer")
    checks = {
        "selected_cross_target_equal_0": metrics["selected_cross_target_pairs"] == 0,
        "parallel_2_groups_present": metrics["parallel_2_group_count"] > 0,
        "parallel_2_preservation_equal_1": (
            metrics["parallel_2_distinct_preservation"] == 1.0
        ),
        "selected_precision_at_least_0_999": (
            (metrics["selected_pair_precision"] or 0.0) >= 0.999
        ),
        "multiplicity_2_recall_at_least_0_995": (
            (metrics["multiplicity_2_recall"] or 0.0) >= 0.995
        ),
        "pairable_sample_exact_at_least_0_95": (
            (metrics["pairable_sample_exact"] or 0.0) >= 0.95
        ),
        "pairable_samples_present": metrics["pairable_sample_count"] > 0,
        "support_loss_at_most_0_1pp": (
            (metrics["ordinary_support_loss_pp"] or 0.0) <= 0.1
        ),
        "all_matching_optimal": (
            metrics["matching_optimal_sample_count"] == metrics["sample_count"]
        ),
        "permutation_determinism_equal_1": (
            metrics["permutation_deterministic"] is True
        ),
        "matching_p95_below_20ms": (
            (metrics["matching_latency_ms"]["p95"] or math.inf) < 20.0
        ),
        "at_least_5000_cross_candidates": (
            class_counts["distinct_target_hard_negative"] >= 5000
        ),
        "at_least_100_identity_components": calibration_component_count >= 100,
    }
    return {
        "status": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "thresholds": {
            "selected_cross_target_pairs": 0,
            "parallel_2_distinct_preservation": 1.0,
            "selected_pair_precision": 0.999,
            "multiplicity_2_recall": 0.995,
            "pairable_sample_exact": 0.95,
            "maximum_support_loss_pp": 0.1,
            "permutation_determinism": 1.0,
            "maximum_matching_p95_ms": 20.0,
            "minimum_cross_candidates": 5000,
            "minimum_identity_components": 100,
        },
        "calibration_class_counts": class_counts,
        "calibration_molecule_count": len(identities),
        "calibration_component_count": calibration_component_count,
    }


def _checkpoint_payload(
    model: EvidenceCarrierPairScorer,
    *,
    args: argparse.Namespace,
    cache: Mapping[str, Any],
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
        "feature_transform_config": asdict(model.transform_config),
        "parameter_count": model.parameter_count(),
        "pair_feature_names": list(PAIR_FEATURE_NAMES),
        "normalization": dict(normalization),
        "gain_threshold": float(threshold_calibration["threshold"]),
        "threshold_calibration": selected_calibration,
        "train_cal_gate": dict(gate),
        "checkpoint_selection": _plain_json(checkpoint_selection),
        "best_epoch": int(best_epoch),
        "optimizer_steps": int(optimizer_steps),
        "best_calibration_loss": dict(best_calibration_loss),
        "training_args": _plain_json(args),
        "cache_path": cache["cache_path"],
        "cache_sha256": cache["cache_sha256"],
        "cache_schema_version": cache.get("cache_schema_version"),
        "pairing_profile": cache.get("pairing_profile", "training_wide"),
        "pairing_config": _plain_json(cache.get("config", {}).get("pairing_config", {})),
        "uses_internal_test": False,
        "validation_read": False,
    }


def save_pair_scorer_checkpoint(path: str | Path, payload: Mapping[str, Any]) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(output)
    return file_sha256(output)


def load_pair_scorer_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[EvidenceCarrierPairScorer, dict[str, Any]]:
    raw_path = str(path)
    resolved_path = Path(path).expanduser().resolve(strict=False)
    if _contains_internal_test(raw_path) or _contains_internal_test(resolved_path):
        raise ValueError("Pair scorer checkpoint path cannot reference internal_test")
    checkpoint = torch.load(resolved_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Pair scorer checkpoint must be a dictionary")
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": EvidenceCarrierPairScorer.model_type,
        "uses_internal_test": False,
        "validation_read": False,
        "pair_feature_names": list(PAIR_FEATURE_NAMES),
        "parameter_count": PAIR_SCORER_PARAMETER_COUNT,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "pairing_profile": PAIRING_PROFILE_TRAINING_WIDE,
        "pairing_config": asdict(
            conditional_pair_pairing_config(PAIRING_PROFILE_TRAINING_WIDE)
        ),
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Pair scorer checkpoint contract differs: {mismatches}")
    threshold = checkpoint.get("gain_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
    ):
        raise ValueError("Pair scorer checkpoint gain_threshold must be finite")
    cache_path = checkpoint.get("cache_path")
    cache_sha = checkpoint.get("cache_sha256")
    if (
        not isinstance(cache_path, str)
        or not cache_path
        or _contains_internal_test(cache_path)
        or not isinstance(cache_sha, str)
        or len(cache_sha) != 64
        or any(character not in "0123456789abcdef" for character in cache_sha.lower())
    ):
        raise ValueError("Pair scorer checkpoint cache provenance is invalid")
    if not isinstance(checkpoint.get("train_cal_gate"), dict):
        raise ValueError("Pair scorer checkpoint lacks the train-cal gate")
    if not isinstance(checkpoint.get("checkpoint_selection"), dict):
        raise ValueError("Pair scorer checkpoint lacks checkpoint-selection metadata")
    transform = EvidenceCarrierPairFeatureTransformConfig(
        **checkpoint["feature_transform_config"]
    )
    model = EvidenceCarrierPairScorer(transform)
    model.load_state_dict(checkpoint["model"], strict=True)
    if checkpoint.get("model_config") != model.config_dict():
        raise ValueError("Pair scorer checkpoint model config differs from its weights")
    if not bool(model.normalization_fitted):
        raise ValueError("Pair scorer checkpoint normalization is not fitted")
    if int(model.normalization_sample_count) < 2:
        raise ValueError("Pair scorer checkpoint normalization sample count is invalid")
    if not bool(torch.isfinite(model.feature_mean).all()) or not bool(
        torch.isfinite(model.feature_scale).all()
    ) or bool((model.feature_scale <= 0.0).any()):
        raise ValueError("Pair scorer checkpoint normalization buffers are invalid")
    normalization = checkpoint.get("normalization")
    if (
        not isinstance(normalization, dict)
        or normalization.get("source_partition") != "train-fit"
        or normalization.get("sample_count") != int(model.normalization_sample_count)
    ):
        raise ValueError("Pair scorer checkpoint normalization metadata is invalid")
    try:
        declared_mean = torch.tensor(normalization["feature_mean"], dtype=torch.float32)
        declared_scale = torch.tensor(normalization["feature_scale"], dtype=torch.float32)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Pair scorer normalization metadata is malformed") from error
    if not torch.equal(declared_mean, model.feature_mean.cpu()) or not torch.equal(
        declared_scale, model.feature_scale.cpu()
    ):
        raise ValueError("Pair scorer normalization metadata differs from buffers")
    model.to(torch.device(device))
    return model, checkpoint


def train(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Pair scorer output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_train_only_cache(args.cache)
    fit_records = pair_cache_records(cache, "fit")
    calibration_records = pair_cache_records(cache, "calibration")
    device = _device(args.device)
    model = EvidenceCarrierPairScorer().to(device)
    normalization = fit_pair_feature_normalization(model, fit_records)
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
        train_metrics = train_pair_scorer_epoch(
            model,
            optimizer,
            fit_records,
            batch_size=args.batch_size,
            device=device,
            rng=rng,
            max_grad_norm=args.max_grad_norm,
        )
        optimizer_steps += int(train_metrics["batches"])
        calibration_loss = evaluate_pair_scorer_loss(
            model,
            calibration_records,
            batch_size=args.batch_size,
            device=device,
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
        raise RuntimeError("Pair scorer training never produced a checkpoint")
    partition_metadata = cache.get("identity_partition", {})
    calibration_component_count = partition_metadata.get(
        "calibration_component_count"
    )
    structurally_scored = []
    for candidate in checkpoint_candidates:
        model.load_state_dict(candidate["state_dict"], strict=True)
        calibration_scores = score_pair_records(
            model,
            calibration_records,
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
    selected = max(structurally_scored, key=lambda candidate: candidate["selection_key"])
    model.load_state_dict(selected["state_dict"], strict=True)
    best_epoch = int(selected["epoch"])
    selected_optimizer_steps = int(selected["optimizer_steps"])
    best_calibration_loss = selected["calibration_loss"]
    threshold_calibration = selected["threshold_calibration"]
    gate = selected["gate"]
    checkpoint_candidate_summary = [
        {
            "epoch": int(candidate["epoch"]),
            "optimizer_steps": int(candidate["optimizer_steps"]),
            "calibration_loss": candidate["calibration_loss"],
            "gain_threshold": candidate["threshold_calibration"]["threshold"],
            "selected_metrics": candidate["threshold_calibration"]["selected_metrics"],
            "gate_status": candidate["gate"]["status"],
            "selection_key": list(candidate["selection_key"]),
            "selected": candidate is selected,
        }
        for candidate in structurally_scored
    ]
    checkpoint_selection = {
        "candidate_count": len(checkpoint_candidate_summary),
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
        "candidates": checkpoint_candidate_summary,
    }
    payload = _checkpoint_payload(
        model,
        args=args,
        cache=cache,
        best_epoch=best_epoch,
        optimizer_steps=selected_optimizer_steps,
        normalization=normalization,
        best_calibration_loss=best_calibration_loss,
        threshold_calibration=threshold_calibration,
        gate=gate,
        checkpoint_selection=checkpoint_selection,
    )
    checkpoint_sha = save_pair_scorer_checkpoint(output_dir / "best.pt", payload)
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
        "best_calibration_loss": best_calibration_loss,
        "gain_threshold": threshold_calibration["threshold"],
        "calibration_metrics": threshold_calibration["selected_metrics"],
        "train_cal_gate": gate,
        "checkpoint_selection": checkpoint_selection,
        "history": history,
        "checkpoint": "best.pt",
        "checkpoint_sha256": checkpoint_sha,
        "cache_path": cache["cache_path"],
        "cache_sha256": cache["cache_sha256"],
        "validation_read": False,
        "uses_internal_test": False,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=9910)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--minimum-improvement", type=float, default=1e-5)
    parser.add_argument("--checkpoint-candidates", type=int, default=3)
    parser.add_argument("--threshold-candidates", type=int, default=32)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    integer_positive = (
        "max_epochs",
        "batch_size",
        "patience",
        "checkpoint_candidates",
        "threshold_candidates",
    )
    for name in integer_positive:
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("Optimizer values are invalid")
    if args.max_grad_norm <= 0.0 or args.minimum_improvement < 0.0:
        raise ValueError("Gradient and improvement values are invalid")
    if _contains_internal_test(args.cache) or _contains_internal_test(args.output_dir):
        raise ValueError("Pair scorer train-cal paths cannot reference internal_test")


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
    "collate_pair_training_records",
    "evaluate_pair_scorer_loss",
    "fit_pair_feature_normalization",
    "load_pair_scorer_checkpoint",
    "load_train_only_cache",
    "pair_cache_records",
    "pair_scorer_train_cal_gate",
    "save_pair_scorer_checkpoint",
    "score_pair_records",
    "train_pair_scorer_epoch",
]
