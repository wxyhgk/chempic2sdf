#!/usr/bin/env python3
"""Train and one-shot evaluate Evidence Carrier v0.9.17."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

import chemvector.build_evidence_carrier_pair_residual_fit_dev as manifest_module
import chemvector.evidence_carrier_pair_conditioned_residual as residual_module
from chemvector.evidence_carrier_pair_conditioned_residual import (
    PAIR_RESIDUAL_HARD_NEGATIVE_TOP_K,
    PAIR_RESIDUAL_HARD_NEGATIVE_WEIGHT,
    PAIR_RESIDUAL_L2_WEIGHT,
    PAIR_RESIDUAL_PARAMETER_COUNT,
    EvidenceCarrierPairConditionedResidual,
    pair_conditioned_residual_loss,
)
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
    evaluate_evidence_carrier_pair_sample,
)
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,
    SAMPLE_THRESHOLD_PAIR_TOKEN_DIM,
)
CHECKPOINT_SCHEMA_VERSION = (
    "evidence-carrier-pair-conditioned-residual-checkpoint-0.1"
)
DEV_SCHEMA_VERSION = "evidence-carrier-pair-conditioned-residual-dev900-0.1"
SUMMARY_SCHEMA_VERSION = (
    "evidence-carrier-pair-conditioned-residual-train-summary-0.1"
)
DEFAULT_MANIFEST = manifest_module.DEFAULT_OUTPUT
DEFAULT_MANIFEST_AUDIT = manifest_module.DEFAULT_AUDIT_OUTPUT
DEFAULT_OUTPUT_DIR = (
    "outputs/checkpoints/chemvector-v0917-pair-conditioned-residual"
)
DEFAULT_DEV_OUTPUT = (
    "outputs/evaluations/chemvector-v0917-pair-conditioned-residual/dev900.json"
)
DEFAULT_DEV_CONSUMPTION_MARKER = (
    "outputs/evaluations/chemvector-v0917-pair-conditioned-residual/"
    "dev900-consumption.json"
)
TRAINING_STEPS = 2_000
TRAINING_BATCH_SIZE = 64
TRAINING_SEED = 20_260_722
TRAINING_LEARNING_RATE = 1e-3
TRAINING_WEIGHT_DECAY = 1e-4
TRAINING_MAX_GRAD_NORM = 1.0
EXPECTED_FIT_SAMPLE_COUNT = 7_200
EXPECTED_DEV_SAMPLE_COUNT = 900


class PairResidualTrainingError(RuntimeError):
    """Raised when a v0.9.17 training or evaluation contract is violated."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(name): _plain(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                _plain(payload),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                _plain(payload),
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def implementation_sha256() -> dict[str, str]:
    return {
        "trainer": file_sha256(Path(__file__).resolve()),
        "model": file_sha256(Path(residual_module.__file__).resolve()),
        "manifest_builder": file_sha256(Path(manifest_module.__file__).resolve()),
    }


def load_pair_residual_manifest(
    path: str | Path,
    *,
    audit_path: str | Path = DEFAULT_MANIFEST_AUDIT,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Load the index and fit sidecar without opening or hashing dev."""

    resolved = Path(path).expanduser().resolve(strict=True)
    audit_resolved = Path(audit_path).expanduser().resolve(strict=True)
    if manifest_module._contains_forbidden(resolved) or manifest_module._contains_forbidden(
        audit_resolved
    ):
        raise PairResidualTrainingError("manifest path references forbidden data")
    lock_path = resolved.with_suffix(resolved.suffix + ".lock")
    if not lock_path.is_file():
        raise PairResidualTrainingError("manifest completion lock is missing")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PairResidualTrainingError("manifest audit or lock is unreadable") from error
    manifest_sha = file_sha256(resolved)
    audit_sha = file_sha256(audit_resolved)
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PairResidualTrainingError("manifest completion lock lacks artifacts")
    if (
        lock.get("status") != "complete"
        or artifacts.get("index")
        != {"path": str(resolved), "sha256": manifest_sha}
        or artifacts.get("audit")
        != {"path": str(audit_resolved), "sha256": audit_sha}
        or audit.get("index_path") != str(resolved)
        or audit.get("index_sha256") != manifest_sha
    ):
        raise PairResidualTrainingError("manifest audit/lock SHA chain differs")
    index = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(index, Mapping):
        raise PairResidualTrainingError("pair residual manifest must be a mapping")
    index = dict(index)
    manifest_module.validate_index_payload(index)
    if index.get("implementation_sha256") != manifest_module._implementation_sha256():
        raise PairResidualTrainingError("manifest implementation SHA differs")
    source = index.get("source_sha256")
    expected_source = manifest_module._expected_source_sha256()
    if source != expected_source:
        raise PairResidualTrainingError("manifest frozen source SHA differs")
    if (
        audit.get("schema_version") != manifest_module.AUDIT_SCHEMA_VERSION
        or audit.get("status") != manifest_module.STATUS_COMPLETE
        or audit.get("formal_eligible") is not True
        or audit.get("source_sha256") != source
        or audit.get("implementation_sha256")
        != index["implementation_sha256"]
        or lock.get("source_sha256") != source
        or lock.get("implementation_sha256")
        != index["implementation_sha256"]
    ):
        raise PairResidualTrainingError("manifest audit/lock provenance differs")
    input_contract = index.get("input")
    if not isinstance(input_contract, Mapping):
        raise PairResidualTrainingError("manifest input contract is missing")
    for name in (
        "base_cache",
        "augmentation_cache",
        "frozen_pair_scorer_checkpoint",
        "frozen_p0_checkpoint",
    ):
        current = input_contract.get(name)
        if not isinstance(current, Mapping):
            raise PairResidualTrainingError(f"manifest input {name} is missing")
        source_path = Path(current["path"]).expanduser().resolve(strict=True)
        if (
            current.get("sha256") != expected_source[name]
            or file_sha256(source_path) != expected_source[name]
        ):
            raise PairResidualTrainingError(f"live {name} SHA differs")
    fit_metadata = index["partitions"]["fit"]
    dev_metadata = index["partitions"]["dev"]
    expected_fit_path = os.path.abspath(
        os.path.expanduser(manifest_module.DEFAULT_FIT_SIDECAR)
    )
    expected_dev_path = os.path.abspath(
        os.path.expanduser(manifest_module.DEFAULT_DEV_SIDECAR)
    )
    if (
        fit_metadata.get("records_path") != expected_fit_path
        or dev_metadata.get("records_path") != expected_dev_path
    ):
        raise PairResidualTrainingError("sidecar paths differ from frozen protocol")
    fit_sidecar_path = Path(fit_metadata["records_path"]).expanduser().resolve(
        strict=True
    )
    fit_sidecar_sha = file_sha256(fit_sidecar_path)
    if (
        fit_sidecar_sha != fit_metadata["records_sha256"]
        or artifacts.get("fit_sidecar")
        != {"path": str(fit_sidecar_path), "sha256": fit_sidecar_sha}
        or audit.get("fit_sidecar_path") != str(fit_sidecar_path)
        or audit.get("fit_sidecar_sha256") != fit_sidecar_sha
    ):
        raise PairResidualTrainingError("fit sidecar SHA chain differs")
    dev_sidecar_path = str(dev_metadata["records_path"])
    dev_sidecar_sha = str(dev_metadata["records_sha256"])
    if (
        artifacts.get("dev_sidecar")
        != {"path": dev_sidecar_path, "sha256": dev_sidecar_sha}
        or audit.get("dev_sidecar_path") != dev_sidecar_path
        or audit.get("dev_sidecar_sha256") != dev_sidecar_sha
    ):
        raise PairResidualTrainingError("declared dev sidecar SHA chain differs")
    fit_sidecar = torch.load(
        fit_sidecar_path, map_location="cpu", weights_only=False
    )
    if not isinstance(fit_sidecar, Mapping):
        raise PairResidualTrainingError("fit sidecar must be a mapping")
    manifest_module.validate_records_sidecar(
        fit_sidecar, expected_partition="fit"
    )
    manifest_module.validate_index_payload(index, fit_sidecar=fit_sidecar)
    if (
        fit_sidecar.get("implementation_sha256") != index["implementation_sha256"]
        or fit_sidecar.get("source_sha256") != index["source_sha256"]
        or fit_sidecar.get("split_contract") != index["split_contract"]
    ):
        raise PairResidualTrainingError("fit sidecar provenance differs from index")
    fit_records = fit_sidecar.get("records")
    if not isinstance(fit_records, list):
        raise PairResidualTrainingError("fit sidecar records are missing")
    return index, fit_records, {
        "path": str(resolved),
        "sha256": manifest_sha,
        "audit_path": str(audit_resolved),
        "audit_sha256": audit_sha,
        "lock_path": str(lock_path),
        "lock_sha256": file_sha256(lock_path),
        "fit_sidecar_path": str(fit_sidecar_path),
        "fit_sidecar_sha256": fit_sidecar_sha,
        "dev_sidecar_path": dev_sidecar_path,
        "dev_sidecar_sha256": dev_sidecar_sha,
    }


def load_dev_sidecar_after_consumption(
    index: Mapping[str, Any],
    manifest_metadata: Mapping[str, Any],
    marker_path: str | Path,
) -> list[dict[str, Any]]:
    """Verify and open dev only after the one-shot marker exists."""

    try:
        marker = Path(marker_path).expanduser().resolve(strict=True)
        state = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PairResidualTrainingError("dev consumption marker is unreadable") from error
    if (
        state.get("status") != "started"
        or state.get("dev_evaluation_count") != 1
        or state.get("manifest_sha256") != manifest_metadata["sha256"]
        or state.get("dev_sidecar_path") != manifest_metadata["dev_sidecar_path"]
        or state.get("dev_sidecar_sha256")
        != manifest_metadata["dev_sidecar_sha256"]
    ):
        raise PairResidualTrainingError("dev consumption marker contract differs")
    dev_path = Path(manifest_metadata["dev_sidecar_path"]).expanduser().resolve(
        strict=True
    )
    dev_sha = file_sha256(dev_path)
    if dev_sha != manifest_metadata["dev_sidecar_sha256"]:
        raise PairResidualTrainingError("dev sidecar SHA differs after consumption")
    sidecar = torch.load(dev_path, map_location="cpu", weights_only=False)
    if not isinstance(sidecar, Mapping):
        raise PairResidualTrainingError("dev sidecar must be a mapping")
    manifest_module.validate_records_sidecar(sidecar, expected_partition="dev")
    if (
        sidecar.get("implementation_sha256") != index["implementation_sha256"]
        or sidecar.get("source_sha256") != index["source_sha256"]
        or sidecar.get("split_contract") != index["split_contract"]
    ):
        raise PairResidualTrainingError("dev sidecar provenance differs from index")
    records = sidecar.get("records")
    if not isinstance(records, list):
        raise PairResidualTrainingError("dev sidecar records are missing")
    return records


def collate_pair_residual_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("cannot collate an empty pair residual batch")
    pair_counts = [len(record["target_free"]["pair_tokens"]) for record in records]
    capacity = max(max(pair_counts), 1)
    batch = len(records)
    result = {
        "pair_tokens": torch.zeros(
            (batch, capacity, SAMPLE_THRESHOLD_PAIR_TOKEN_DIM), dtype=torch.float32
        ),
        "pair_indices": torch.full(
            (batch, capacity, 2), -1, dtype=torch.long
        ),
        "pair_labels": torch.full((batch, capacity), -1, dtype=torch.long),
        "pair_mask": torch.zeros((batch, capacity), dtype=torch.bool),
        "global_count_features": torch.zeros(
            (batch, SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM), dtype=torch.float32
        ),
        "pair_count": torch.tensor(pair_counts, dtype=torch.long),
    }
    for index, record in enumerate(records):
        target_free = record.get("target_free")
        target_aware = record.get("target_aware")
        if not isinstance(target_free, Mapping) or not isinstance(target_aware, Mapping):
            raise ValueError("pair residual record layers are invalid")
        count = pair_counts[index]
        tokens = torch.as_tensor(target_free["pair_tokens"], dtype=torch.float32)
        pairs = torch.as_tensor(target_free["pair_indices"], dtype=torch.long)
        labels = torch.as_tensor(target_aware["pair_labels"], dtype=torch.long)
        global_features = torch.as_tensor(
            target_free["global_count_features"], dtype=torch.float32
        )
        if (
            tokens.shape != (count, SAMPLE_THRESHOLD_PAIR_TOKEN_DIM)
            or pairs.shape != (count, 2)
            or labels.shape != (count,)
            or global_features.shape
            != (SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,)
        ):
            raise ValueError("pair residual record tensor shapes differ")
        if count:
            result["pair_tokens"][index, :count] = tokens
            result["pair_indices"][index, :count] = pairs
            result["pair_labels"][index, :count] = labels
            result["pair_mask"][index, :count] = True
        result["global_count_features"][index] = global_features
    return result


def deterministic_fit_batch_indices(
    record_count: int,
    *,
    steps: int = TRAINING_STEPS,
    batch_size: int = TRAINING_BATCH_SIZE,
    seed: int = TRAINING_SEED,
) -> Iterator[tuple[int, ...]]:
    if record_count < 1 or steps < 1 or batch_size < 1:
        raise ValueError("record_count, steps, and batch_size must be positive")
    rng = random.Random(seed)
    order: list[int] = []
    cursor = 0
    for _ in range(steps):
        batch = []
        while len(batch) < batch_size:
            if cursor == len(order):
                order = list(range(record_count))
                rng.shuffle(order)
                cursor = 0
            take = min(batch_size - len(batch), len(order) - cursor)
            batch.extend(order[cursor : cursor + take])
            cursor += take
        yield tuple(batch)


def _to_device(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def train_pair_residual_steps(
    model: EvidenceCarrierPairConditionedResidual,
    optimizer: torch.optim.Optimizer,
    fit_records: Sequence[Mapping[str, Any]],
    *,
    steps: int = TRAINING_STEPS,
    batch_size: int = TRAINING_BATCH_SIZE,
    seed: int = TRAINING_SEED,
    max_grad_norm: float = TRAINING_MAX_GRAD_NORM,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    if not fit_records or any(record.get("partition") != "fit" for record in fit_records):
        raise ValueError("optimizer may consume only fit records")
    model.train()
    loss_values: list[float] = []
    classification_values: list[float] = []
    ranking_values: list[float] = []
    hard_values: list[float] = []
    l2_values: list[float] = []
    gradient_norms: list[float] = []
    final_residual_summary: dict[str, float] | None = None
    for indices in deterministic_fit_batch_indices(
        len(fit_records), steps=steps, batch_size=batch_size, seed=seed
    ):
        current = [fit_records[index] for index in indices]
        batch = _to_device(collate_pair_residual_records(current), device)
        output = model(
            batch["pair_tokens"],
            batch["pair_mask"],
            batch["global_count_features"],
        )
        losses = pair_conditioned_residual_loss(
            output["margin"],
            output["residual"],
            batch["pair_indices"],
            batch["pair_labels"],
            batch["pair_mask"],
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        loss_values.append(float(losses["loss"].detach().cpu()))
        classification_values.append(
            float(losses["balanced_bce"].detach().cpu())
        )
        ranking_values.append(float(losses["ranking_loss"].detach().cpu()))
        hard_values.append(float(losses["hard_negative_loss"].detach().cpu()))
        l2_values.append(float(losses["residual_l2_loss"].detach().cpu()))
        gradient_norms.append(float(gradient.detach().cpu()))
        active_residual = output["residual"][batch["pair_mask"]].detach().float()
        if len(active_residual):
            final_residual_summary = {
                "mean": float(active_residual.mean().cpu()),
                "rms": float(active_residual.square().mean().sqrt().cpu()),
                "maximum_absolute": float(active_residual.abs().max().cpu()),
            }
    return {
        "optimizer_steps": steps,
        "batch_size": batch_size,
        "mean_loss": float(np.mean(loss_values)),
        "final_loss": loss_values[-1],
        "mean_balanced_classification_loss": float(np.mean(classification_values)),
        "mean_ranking_loss": float(np.mean(ranking_values)),
        "mean_hard_negative_loss": float(np.mean(hard_values)),
        "mean_residual_l2_loss": float(np.mean(l2_values)),
        "maximum_preclip_gradient_norm": max(gradient_norms),
        "final_batch_residual": final_residual_summary,
    }


def _assert_finite_metrics(value: Any, *, path: str = "training_metrics") -> None:
    if isinstance(value, Mapping):
        for name, item in value.items():
            _assert_finite_metrics(item, path=f"{path}.{name}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_finite_metrics(item, path=f"{path}[{index}]")
        return
    if isinstance(value, (int, float, np.generic)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise PairResidualTrainingError(f"{path} is non-finite")
        return
    if value is None:
        raise PairResidualTrainingError(f"{path} is unexpectedly null")


@torch.inference_mode()
def audit_fit_model_finite(
    model: EvidenceCarrierPairConditionedResidual,
    fit_records: Sequence[Mapping[str, Any]],
    training_metrics: Mapping[str, Any],
    *,
    batch_size: int = TRAINING_BATCH_SIZE,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    """Reject non-finite training state before the one-shot dev marker exists."""

    if not fit_records or batch_size < 1:
        raise PairResidualTrainingError("fit finite audit requires records and batch size")
    if any(record.get("partition") != "fit" for record in fit_records):
        raise PairResidualTrainingError("fit finite audit received a non-fit record")
    _assert_finite_metrics(training_metrics)
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise PairResidualTrainingError(f"model parameter {name} is non-finite")

    was_training = model.training
    model.eval()
    sample_count = pair_count = 0
    residual_sum = residual_square_sum = 0.0
    residual_maximum_absolute = 0.0
    try:
        for start in range(0, len(fit_records), batch_size):
            current = fit_records[start : start + batch_size]
            batch = _to_device(collate_pair_residual_records(current), device)
            output = model(
                batch["pair_tokens"],
                batch["pair_mask"],
                batch["global_count_features"],
            )
            active = batch["pair_mask"]
            for name in ("residual", "corrected_gain", "margin"):
                if not bool(torch.isfinite(output[name][active]).all()):
                    raise PairResidualTrainingError(
                        f"fit forward {name} contains non-finite values"
                    )
            if not bool(torch.isfinite(output["image_context"]).all()):
                raise PairResidualTrainingError(
                    "fit forward image context contains non-finite values"
                )
            residual = output["residual"][active].double()
            sample_count += len(current)
            pair_count += int(active.sum())
            if len(residual):
                residual_sum += float(residual.sum())
                residual_square_sum += float(residual.square().sum())
                residual_maximum_absolute = max(
                    residual_maximum_absolute,
                    float(residual.abs().max()),
                )
    finally:
        model.train(was_training)
    if sample_count != len(fit_records) or pair_count < 1:
        raise PairResidualTrainingError("fit finite audit coverage differs")
    return {
        "status": "FIT-FINITE-AUDIT-PASSED",
        "sample_count": sample_count,
        "pair_count": pair_count,
        "parameter_count": model.parameter_count(),
        "residual_mean": residual_sum / pair_count,
        "residual_rms": math.sqrt(residual_square_sum / pair_count),
        "residual_maximum_absolute": residual_maximum_absolute,
        "dev_sidecar_loaded": False,
    }


def _latency_summary(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()) if len(array) else None,
        "p50": float(np.percentile(array, 50)) if len(array) else None,
        "p95": float(np.percentile(array, 95)) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
    }


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()) if len(array) else None,
        "mean": float(array.mean()) if len(array) else None,
        "p50": float(np.percentile(array, 50)) if len(array) else None,
        "p95": float(np.percentile(array, 95)) if len(array) else None,
        "maximum": float(array.max()) if len(array) else None,
    }


def _structural_key(metrics: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -int(metrics["selected_cross_target_pairs"]),
        -int(metrics["parallel_2_violated_group_count"]),
        int(metrics["multiplicity_2_matched_count"]),
        int(metrics["pairable_sample_exact_count"]),
        -int(metrics["ordinary_support_lost_target_count"]),
        float(metrics.get("selected_pair_precision") or 0.0),
    )


def pair_residual_capability_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    checkpoint_sha_unchanged: bool = True,
) -> dict[str, Any]:
    baseline_key = _structural_key(baseline)
    candidate_key = _structural_key(candidate)
    checks = {
        "dev_sample_count_equal_900": candidate["sample_count"] == EXPECTED_DEV_SAMPLE_COUNT,
        "selected_cross_target_equal_0": candidate["selected_cross_target_pairs"] == 0,
        "selected_precision_at_least_0_999": (
            (candidate["selected_pair_precision"] or 0.0) >= 0.999
        ),
        "multiplicity_2_recall_at_least_0_995": (
            (candidate["multiplicity_2_recall"] or 0.0) >= 0.995
        ),
        "pairable_sample_exact_at_least_0_95": (
            (candidate["pairable_sample_exact"] or 0.0) >= 0.95
        ),
        "parallel_2_groups_present": candidate["parallel_2_group_count"] > 0,
        "parallel_2_preservation_equal_1": (
            candidate["parallel_2_distinct_preservation"] == 1.0
        ),
        "support_loss_at_most_0_1pp": (
            candidate["ordinary_support_loss_pp"] is not None
            and candidate["ordinary_support_loss_pp"] <= 0.1
        ),
        "permutation_determinism_equal_1": (
            candidate["permutation_deterministic"] is True
        ),
        "all_matching_optimal": (
            candidate["matching_optimal_sample_count"] == EXPECTED_DEV_SAMPLE_COUNT
        ),
        "checkpoint_sha_unchanged": checkpoint_sha_unchanged is True,
        "strictly_better_than_frozen_baseline": candidate_key > baseline_key,
    }
    return {
        "status": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "baseline_structural_key": list(baseline_key),
        "candidate_structural_key": list(candidate_key),
        "thresholds": {
            "sample_count": EXPECTED_DEV_SAMPLE_COUNT,
            "selected_cross_target_pairs": 0,
            "selected_pair_precision": 0.999,
            "multiplicity_2_recall": 0.995,
            "pairable_sample_exact": 0.95,
            "parallel_2_distinct_preservation": 1.0,
            "maximum_support_loss_pp": 0.1,
            "permutation_determinism": 1.0,
            "matching_optimal_sample_count": EXPECTED_DEV_SAMPLE_COUNT,
            "checkpoint_sha_unchanged": True,
            "candidate_structural_key_strictly_greater": True,
        },
    }


@torch.inference_mode()
def evaluate_dev_once(
    model: EvidenceCarrierPairConditionedResidual,
    dev_records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    if len(dev_records) != EXPECTED_DEV_SAMPLE_COUNT:
        raise PairResidualTrainingError("dev manifest must contain exactly 900 records")
    model.eval()
    baseline_evaluations = []
    candidate_evaluations = []
    sample_evaluations = []
    forward_latencies = []
    residual_values: list[float] = []
    changed_sample_count = 0
    for record in dev_records:
        target_free = record.get("target_free")
        target_aware = record.get("target_aware")
        if not isinstance(target_free, Mapping) or not isinstance(
            target_aware, Mapping
        ):
            raise PairResidualTrainingError("dev record target layers are missing")
        sealed_target = target_aware.get("evaluation_target")
        if not isinstance(sealed_target, Mapping):
            raise PairResidualTrainingError("dev record lacks its sealed evaluation target")
        source = {
            **dict(sealed_target),
            "pair_indices": target_free["pair_indices"],
            "pair_labels": target_aware["pair_labels"],
        }
        batch = _to_device(collate_pair_residual_records([record]), device)
        started = perf_counter()
        output = model(
            batch["pair_tokens"],
            batch["pair_mask"],
            batch["global_count_features"],
        )
        forward_latencies.append((perf_counter() - started) * 1000.0)
        count = int(batch["pair_count"][0])
        corrected = output["corrected_gain"][0, :count].detach().cpu()
        residual = output["residual"][0, :count].detach().cpu()
        raw = batch["pair_tokens"][0, :count, -1].detach().cpu()
        residual_values.extend(map(float, residual.tolist()))
        baseline_result = evaluate_evidence_carrier_pair_sample(
            source,
            raw,
            threshold=SAMPLE_THRESHOLD_BASELINE_TAU,
            verify_permutation=True,
        )
        candidate_result = evaluate_evidence_carrier_pair_sample(
            source,
            corrected,
            threshold=SAMPLE_THRESHOLD_BASELINE_TAU,
            verify_permutation=True,
        )
        baseline_evaluations.append(baseline_result)
        candidate_evaluations.append(candidate_result)
        changed = (
            baseline_result["selected_anchor_pairs"]
            != candidate_result["selected_anchor_pairs"]
        )
        changed_sample_count += int(changed)
        sample_evaluations.append(
            {
                "identity": dict(record["identity"]),
                "changed_selection": changed,
                "residual": _distribution(list(map(float, residual.tolist()))),
                "baseline": baseline_result,
                "candidate": candidate_result,
            }
        )
    baseline = aggregate_evidence_carrier_pair_evaluations(baseline_evaluations)
    candidate = aggregate_evidence_carrier_pair_evaluations(candidate_evaluations)
    gate = pair_residual_capability_gate(baseline, candidate)
    status = (
        "DEV900-PAIR-RESIDUAL-GO"
        if gate["status"] == "GO"
        else "DEV900-PAIR-RESIDUAL-NO-GO"
    )
    return {
        "schema_version": DEV_SCHEMA_VERSION,
        "status": status,
        "tau0": SAMPLE_THRESHOLD_BASELINE_TAU,
        "baseline_selected_metrics": baseline,
        "candidate_selected_metrics": candidate,
        "gate": gate,
        "changed_selection_sample_count": changed_sample_count,
        "residual_distribution": _distribution(residual_values),
        "residual_forward_latency_ms_per_sample": _latency_summary(
            forward_latencies
        ),
        "diagnostic_checks": {
            "residual_forward_p95_below_1ms": (
                (_latency_summary(forward_latencies)["p95"] or math.inf) < 1.0
            ),
            "candidate_matching_p95_below_20ms": (
                (candidate["matching_latency_ms"]["p95"] or math.inf) < 20.0
            ),
        },
        "diagnostics_excluded_from_capability_gate": [
            "residual_forward_p95_below_1ms",
            "candidate_matching_p95_below_20ms",
        ],
        "sample_evaluations": sample_evaluations,
        "original_calibration_records_accessed": False,
        "validation_read": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
    }


def _begin_dev_consumption(
    marker_path: Path,
    *,
    manifest_metadata: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> None:
    _write_json_no_overwrite(
        marker_path,
        {
            "schema_version": "evidence-carrier-pair-residual-dev-consumption-0.1",
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "dev_evaluation_count": 1,
            "manifest_path": manifest_metadata["path"],
            "manifest_sha256": manifest_metadata["sha256"],
            "dev_sidecar_path": manifest_metadata["dev_sidecar_path"],
            "dev_sidecar_sha256": manifest_metadata["dev_sidecar_sha256"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256_before_dev": checkpoint_sha256,
            "original_calibration_records_accessed": False,
        },
    )


def _complete_dev_consumption(
    marker_path: Path,
    *,
    dev_sidecar_path: Path,
    dev_sidecar_sha256: str,
    report_path: Path,
    report_sha256: str,
    checkpoint_sha256_after: str,
    status: str,
) -> str:
    started = json.loads(marker_path.read_text(encoding="utf-8"))
    if started.get("status") != "started" or started.get("dev_evaluation_count") != 1:
        raise PairResidualTrainingError("dev consumption marker state differs")
    if started.get("checkpoint_sha256_before_dev") != checkpoint_sha256_after:
        raise PairResidualTrainingError("checkpoint changed during dev consumption")
    if (
        started.get("dev_sidecar_path") != str(dev_sidecar_path)
        or started.get("dev_sidecar_sha256") != dev_sidecar_sha256
    ):
        raise PairResidualTrainingError("verified dev sidecar differs from marker")
    complete = {
        **started,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256_after_dev": checkpoint_sha256_after,
        "checkpoint_sha256_unchanged": True,
        "dev_sidecar_path": str(dev_sidecar_path),
        "dev_sidecar_sha256_verified": dev_sidecar_sha256,
        "report_path": str(report_path),
        "report_sha256": report_sha256,
        "dev_status": status,
    }
    _replace_json(marker_path, complete)
    return file_sha256(marker_path)


def load_pair_residual_checkpoint(
    path: str | Path,
    *,
    device: torch.device = torch.device("cpu"),
    verify_provenance: bool = True,
) -> tuple[EvidenceCarrierPairConditionedResidual, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("pair residual checkpoint must be a mapping")
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": EvidenceCarrierPairConditionedResidual.model_type,
        "parameter_count": PAIR_RESIDUAL_PARAMETER_COUNT,
        "optimizer_steps": TRAINING_STEPS,
        "checkpoint_selection": "last_step_only",
        "dev_sidecar_loaded": False,
        "dev_used_for_training": False,
        "dev_used_for_checkpoint_selection": False,
        "dev_evaluated": False,
        "original_calibration_records_accessed": False,
        "validation_read": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise ValueError(f"pair residual checkpoint contract differs: {mismatches}")
    model = EvidenceCarrierPairConditionedResidual()
    if checkpoint.get("model_config") != model.config_dict():
        raise ValueError("pair residual checkpoint model config differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    if checkpoint.get("implementation_sha256") != implementation_sha256():
        raise ValueError("pair residual checkpoint implementation SHA differs")
    if checkpoint.get("source_sha256") != manifest_module._expected_source_sha256():
        raise ValueError("pair residual checkpoint source commitment differs")
    if verify_provenance:
        for path_name, sha_name in (
            ("manifest_path", "manifest_sha256"),
            ("manifest_audit_path", "manifest_audit_sha256"),
            ("manifest_lock_path", "manifest_lock_sha256"),
            ("fit_sidecar_path", "fit_sidecar_sha256"),
        ):
            source = Path(checkpoint[path_name]).expanduser().resolve(strict=True)
            if file_sha256(source) != checkpoint[sha_name]:
                raise ValueError(f"checkpoint provenance {sha_name} differs")
        dev_path = checkpoint.get("dev_sidecar_path")
        dev_sha = checkpoint.get("dev_sidecar_sha256")
        if (
            not isinstance(dev_path, str)
            or not dev_path
            or not isinstance(dev_sha, str)
            or len(dev_sha) != 64
            or any(character not in "0123456789abcdef" for character in dev_sha)
        ):
            raise ValueError("checkpoint declared dev sidecar metadata is invalid")
    model.to(device).eval()
    return model, dict(checkpoint)


def train(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    dev_path = Path(DEFAULT_DEV_OUTPUT).expanduser().resolve(strict=False)
    marker_path = Path(DEFAULT_DEV_CONSUMPTION_MARKER).expanduser().resolve(
        strict=False
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"training output directory is not empty: {output_dir}")
    for path in (dev_path, marker_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite formal artifact: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    run_lock = output_dir / "run.lock"
    _write_json_no_overwrite(
        run_lock,
        {
            "schema_version": "evidence-carrier-pair-residual-run-lock-0.1",
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "optimizer_steps": TRAINING_STEPS,
            "dev_evaluation_count": 1,
        },
    )
    manifest, fit_records, manifest_metadata = load_pair_residual_manifest(
        args.manifest
    )
    if len(fit_records) != EXPECTED_FIT_SAMPLE_COUNT:
        raise PairResidualTrainingError("formal fit record count differs")

    random.seed(TRAINING_SEED)
    np.random.seed(TRAINING_SEED)
    torch.manual_seed(TRAINING_SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cpu")
    model = EvidenceCarrierPairConditionedResidual().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAINING_LEARNING_RATE,
        weight_decay=TRAINING_WEIGHT_DECAY,
    )
    frozen_implementation = implementation_sha256()
    training_metrics = train_pair_residual_steps(
        model,
        optimizer,
        fit_records,
        device=device,
    )
    if implementation_sha256() != frozen_implementation:
        raise PairResidualTrainingError("implementation changed during training")
    fit_finite_audit = audit_fit_model_finite(
        model,
        fit_records,
        training_metrics,
        device=device,
    )
    checkpoint = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": model.model_type,
        "model": model.state_dict(),
        "model_config": model.config_dict(),
        "parameter_count": model.parameter_count(),
        "optimizer_steps": TRAINING_STEPS,
        "checkpoint_selection": "last_step_only",
        "training_protocol": {
            "optimizer": "AdamW",
            "learning_rate": TRAINING_LEARNING_RATE,
            "weight_decay": TRAINING_WEIGHT_DECAY,
            "batch_size": TRAINING_BATCH_SIZE,
            "max_grad_norm": TRAINING_MAX_GRAD_NORM,
            "seed": TRAINING_SEED,
            "learning_rate_schedule": "constant",
            "hard_negative_top_k": PAIR_RESIDUAL_HARD_NEGATIVE_TOP_K,
            "hard_negative_weight": PAIR_RESIDUAL_HARD_NEGATIVE_WEIGHT,
            "residual_l2_weight": PAIR_RESIDUAL_L2_WEIGHT,
            "loss": "balanced-cls-plus-shared-anchor-ranking-plus-hard-negative-plus-residual-l2",
            "numeric_threads": 1,
        },
        "training_metrics": training_metrics,
        "fit_finite_audit": fit_finite_audit,
        "implementation_sha256": frozen_implementation,
        "manifest_path": manifest_metadata["path"],
        "manifest_sha256": manifest_metadata["sha256"],
        "fit_sidecar_path": manifest_metadata["fit_sidecar_path"],
        "fit_sidecar_sha256": manifest_metadata["fit_sidecar_sha256"],
        "dev_sidecar_path": manifest_metadata["dev_sidecar_path"],
        "dev_sidecar_sha256": manifest_metadata["dev_sidecar_sha256"],
        "manifest_audit_path": manifest_metadata["audit_path"],
        "manifest_audit_sha256": manifest_metadata["audit_sha256"],
        "manifest_lock_path": manifest_metadata["lock_path"],
        "manifest_lock_sha256": manifest_metadata["lock_sha256"],
        "source_sha256": dict(manifest["source_sha256"]),
        "fit_sample_count": len(fit_records),
        "dev_sample_count": EXPECTED_DEV_SAMPLE_COUNT,
        "dev_sidecar_loaded": False,
        "dev_used_for_training": False,
        "dev_used_for_checkpoint_selection": False,
        "dev_evaluated": False,
        "original_calibration_records_accessed": False,
        "validation_read": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
    }
    checkpoint_path = output_dir / "final.pt"
    temporary = checkpoint_path.with_suffix(f".pt.tmp.{os.getpid()}")
    try:
        torch.save(checkpoint, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, checkpoint_path)
        _fsync_directory(checkpoint_path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    checkpoint_sha_before = file_sha256(checkpoint_path)
    sealed_model, sealed_checkpoint = load_pair_residual_checkpoint(
        checkpoint_path, device=device, verify_provenance=True
    )
    if sealed_checkpoint["implementation_sha256"] != frozen_implementation:
        raise PairResidualTrainingError("sealed implementation commitment differs")
    _begin_dev_consumption(
        marker_path,
        manifest_metadata=manifest_metadata,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha_before,
    )
    dev_records = load_dev_sidecar_after_consumption(
        manifest,
        manifest_metadata,
        marker_path,
    )
    dev = evaluate_dev_once(
        sealed_model,
        dev_records,
        device=device,
    )
    if implementation_sha256() != frozen_implementation:
        raise PairResidualTrainingError("implementation changed during dev evaluation")
    checkpoint_sha_after = file_sha256(checkpoint_path)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise PairResidualTrainingError("checkpoint changed during dev evaluation")
    dev_payload = {
        **dev,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256_before_dev": checkpoint_sha_before,
        "checkpoint_sha256_after_dev": checkpoint_sha_after,
        "checkpoint_sha256_unchanged": True,
        "manifest_path": manifest_metadata["path"],
        "manifest_sha256": manifest_metadata["sha256"],
        "manifest_audit_path": manifest_metadata["audit_path"],
        "manifest_audit_sha256": manifest_metadata["audit_sha256"],
        "manifest_lock_path": manifest_metadata["lock_path"],
        "manifest_lock_sha256": manifest_metadata["lock_sha256"],
        "fit_sidecar_path": manifest_metadata["fit_sidecar_path"],
        "fit_sidecar_sha256": manifest_metadata["fit_sidecar_sha256"],
        "dev_sidecar_path": manifest_metadata["dev_sidecar_path"],
        "dev_sidecar_sha256": manifest_metadata["dev_sidecar_sha256"],
        "source_sha256": dict(manifest["source_sha256"]),
        "implementation_sha256": frozen_implementation,
        "optimizer_steps": TRAINING_STEPS,
        "checkpoint_selection": "last_step_only_before_dev",
        "dev_used_for_optimizer": False,
        "dev_used_for_checkpoint_selection": False,
    }
    _write_json_no_overwrite(dev_path, dev_payload)
    dev_sha = file_sha256(dev_path)
    marker_sha = _complete_dev_consumption(
        marker_path,
        dev_sidecar_path=Path(manifest_metadata["dev_sidecar_path"]),
        dev_sidecar_sha256=manifest_metadata["dev_sidecar_sha256"],
        report_path=dev_path,
        report_sha256=dev_sha,
        checkpoint_sha256_after=checkpoint_sha_after,
        status=dev["status"],
    )
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": dev["status"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256_before_dev": checkpoint_sha_before,
        "checkpoint_sha256_after_dev": checkpoint_sha_after,
        "checkpoint_sha256_unchanged": True,
        "manifest_path": manifest_metadata["path"],
        "manifest_sha256": manifest_metadata["sha256"],
        "fit_sidecar_path": manifest_metadata["fit_sidecar_path"],
        "fit_sidecar_sha256": manifest_metadata["fit_sidecar_sha256"],
        "dev_sidecar_path": manifest_metadata["dev_sidecar_path"],
        "dev_sidecar_sha256": manifest_metadata["dev_sidecar_sha256"],
        "training_metrics": training_metrics,
        "fit_finite_audit": fit_finite_audit,
        "dev_path": str(dev_path),
        "dev_sha256": dev_sha,
        "dev_consumption_marker": str(marker_path),
        "dev_consumption_marker_sha256": marker_sha,
        "dev_evaluation_count": 1,
        "dev_gate": dev["gate"],
        "baseline_selected_metrics": dev["baseline_selected_metrics"],
        "candidate_selected_metrics": dev["candidate_selected_metrics"],
        "residual_distribution": dev["residual_distribution"],
        "residual_forward_latency_ms_per_sample": dev[
            "residual_forward_latency_ms_per_sample"
        ],
        "optimizer_steps": TRAINING_STEPS,
        "checkpoint_selection": "last_step_only",
        "dev_used_for_optimizer": False,
        "dev_used_for_checkpoint_selection": False,
        "original_calibration_records_accessed": False,
        "validation_read": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
    }
    summary_path = output_dir / "summary.json"
    _write_json_no_overwrite(summary_path, summary)
    summary_sha = file_sha256(summary_path)
    _replace_json(
        run_lock,
        {
            "schema_version": "evidence-carrier-pair-residual-run-lock-0.1",
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha_after,
            "fit_sidecar_path": manifest_metadata["fit_sidecar_path"],
            "fit_sidecar_sha256": manifest_metadata["fit_sidecar_sha256"],
            "dev_sidecar_path": manifest_metadata["dev_sidecar_path"],
            "dev_sidecar_sha256_verified": manifest_metadata[
                "dev_sidecar_sha256"
            ],
            "summary_path": str(summary_path),
            "summary_sha256": summary_sha,
            "dev_path": str(dev_path),
            "dev_sha256": dev_sha,
            "dev_consumption_marker": str(marker_path),
            "dev_consumption_marker_sha256": marker_sha,
        },
    )
    return {**summary, "summary_sha256": summary_sha}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(_plain(train(args)), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "DEFAULT_DEV_CONSUMPTION_MARKER",
    "DEFAULT_DEV_OUTPUT",
    "DEFAULT_MANIFEST",
    "DEFAULT_MANIFEST_AUDIT",
    "DEFAULT_OUTPUT_DIR",
    "DEV_SCHEMA_VERSION",
    "PairResidualTrainingError",
    "SUMMARY_SCHEMA_VERSION",
    "TRAINING_BATCH_SIZE",
    "TRAINING_LEARNING_RATE",
    "TRAINING_MAX_GRAD_NORM",
    "TRAINING_SEED",
    "TRAINING_STEPS",
    "TRAINING_WEIGHT_DECAY",
    "audit_fit_model_finite",
    "collate_pair_residual_records",
    "deterministic_fit_batch_indices",
    "evaluate_dev_once",
    "implementation_sha256",
    "load_dev_sidecar_after_consumption",
    "load_pair_residual_checkpoint",
    "load_pair_residual_manifest",
    "pair_residual_capability_gate",
    "train_pair_residual_steps",
]
