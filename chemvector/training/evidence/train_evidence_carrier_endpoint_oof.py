#!/usr/bin/env python3
"""Train and evaluate the three fit-only endpoint-incidence OOF models."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

import chemvector.build_evidence_carrier_endpoint_oof as data_module
import chemvector.evidence_carrier_endpoint_incidence_residual as model_module
from chemvector.build_evidence_carrier_endpoint_oof import (
    DEFAULT_AUDIT,
    DEFAULT_FOLD_PATHS,
    DEFAULT_OUTPUT_INDEX,
    EXPECTED_SAMPLES_PER_FOLD,
    validate_fold_sidecar,
    validate_index,
)
from chemvector.evidence_carrier_endpoint_incidence_residual import (
    ENDPOINT_RESIDUAL_PARAMETER_COUNT,
    EvidenceCarrierEndpointIncidenceResidual,
    endpoint_safe_residual_loss,
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


CHECKPOINT_SCHEMA_VERSION = "evidence-carrier-endpoint-oof-checkpoint-0.1"
REPORT_SCHEMA_VERSION = "evidence-carrier-endpoint-oof-report-0.1"
STATUS_GO = "FIT-OOF-ENDPOINT-SAFE-RESIDUAL-GO"
STATUS_NO_GO = "FIT-OOF-ENDPOINT-SAFE-RESIDUAL-NO-GO"
DEFAULT_OUTPUT_DIR = "outputs/checkpoints/chemvector-v0918-endpoint-oof"
DEFAULT_REPORT = "outputs/evaluations/chemvector-v0918-endpoint-oof/oof.json"
EXPECTED_INDEX_SHA256 = (
    "c413455ea949a855d972b68a7d550b1b97c756379ba4973a779408b8606763c4"
)
EXPECTED_AUDIT_SHA256 = (
    "2ea8e39c2747d224aba3544e76790c9913be00365b779c869985e30b9061e392"
)
TRAINING_STEPS = 2_000
TRAINING_BATCH_SIZE = 64
TRAINING_SEEDS = (20_260_722, 20_260_723, 20_260_724)
TRAINING_LEARNING_RATE = 1e-3
TRAINING_WEIGHT_DECAY = 1e-4
TRAINING_MAX_GRAD_NORM = 1.0
EXPECTED_TRAIN_SAMPLE_COUNT = 4_800
EXPECTED_OOF_SAMPLE_COUNT = 2_400


class EndpointOOFTrainingError(RuntimeError):
    """Raised when the frozen endpoint OOF training contract is violated."""


def file_sha256(path: str | Path) -> str:
    return data_module.file_sha256(path)


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
        "model": file_sha256(Path(model_module.__file__).resolve()),
        "data_builder": file_sha256(Path(data_module.__file__).resolve()),
    }


def load_oof_index(
    index_path: str | Path = DEFAULT_OUTPUT_INDEX,
    *,
    audit_path: str | Path = DEFAULT_AUDIT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    index_resolved = Path(index_path).expanduser().resolve(strict=True)
    audit_resolved = Path(audit_path).expanduser().resolve(strict=True)
    lock_path = index_resolved.with_suffix(index_resolved.suffix + ".lock")
    if not lock_path.is_file():
        raise EndpointOOFTrainingError("OOF data completion lock is missing")
    if file_sha256(index_resolved) != EXPECTED_INDEX_SHA256:
        raise EndpointOOFTrainingError("OOF index SHA differs")
    if file_sha256(audit_resolved) != EXPECTED_AUDIT_SHA256:
        raise EndpointOOFTrainingError("OOF audit SHA differs")
    try:
        audit = json.loads(audit_resolved.read_text(encoding="utf-8"))
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EndpointOOFTrainingError("OOF audit or lock is unreadable") from error
    index = torch.load(index_resolved, map_location="cpu", weights_only=False)
    if not isinstance(index, Mapping):
        raise EndpointOOFTrainingError("OOF index must be a mapping")
    index = dict(index)
    validate_index(index)
    expected_implementation = data_module.implementation_sha256()
    if index.get("implementation_sha256") != expected_implementation:
        raise EndpointOOFTrainingError("OOF data implementation SHA differs")
    if (
        audit.get("index_path") != str(index_resolved)
        or audit.get("index_sha256") != EXPECTED_INDEX_SHA256
        or audit.get("implementation_sha256") != expected_implementation
        or lock.get("status") != "complete"
        or lock.get("index_path") != str(index_resolved)
        or lock.get("index_sha256") != EXPECTED_INDEX_SHA256
    ):
        raise EndpointOOFTrainingError("OOF data provenance chain differs")
    return index, {
        "index_path": str(index_resolved),
        "index_sha256": EXPECTED_INDEX_SHA256,
        "audit_path": str(audit_resolved),
        "audit_sha256": EXPECTED_AUDIT_SHA256,
        "lock_path": str(lock_path),
        "lock_sha256": file_sha256(lock_path),
    }


def load_oof_fold(
    index: Mapping[str, Any], fold: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if fold not in (0, 1, 2):
        raise ValueError("fold must be 0, 1, or 2")
    reference = index["folds"][fold]
    expected_path = str(Path(DEFAULT_FOLD_PATHS[fold]).expanduser().resolve())
    if reference.get("sidecar_path") != expected_path:
        raise EndpointOOFTrainingError("fold sidecar path differs")
    path = Path(expected_path).resolve(strict=True)
    sha = file_sha256(path)
    if sha != reference.get("sidecar_sha256"):
        raise EndpointOOFTrainingError("fold sidecar SHA differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise EndpointOOFTrainingError("fold sidecar must be a mapping")
    validate_fold_sidecar(payload, expected_fold=fold)
    if (
        payload.get("implementation_sha256") != index["implementation_sha256"]
        or payload.get("source_sha256") != index["source_sha256"]
        or payload.get("fold_contract") != index["fold_contract"]
    ):
        raise EndpointOOFTrainingError("fold sidecar provenance differs")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != EXPECTED_SAMPLES_PER_FOLD:
        raise EndpointOOFTrainingError("fold sidecar records differ")
    return records, {"path": str(path), "sha256": sha}


def collate_endpoint_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("cannot collate an empty endpoint batch")
    pair_counts = [len(record["target_free"]["pair_tokens"]) for record in records]
    capacity = max(max(pair_counts), 1)
    batch = len(records)
    result = {
        "pair_tokens": torch.zeros(
            (batch, capacity, SAMPLE_THRESHOLD_PAIR_TOKEN_DIM), dtype=torch.float32
        ),
        "pair_indices": torch.full((batch, capacity, 2), -1, dtype=torch.long),
        "pair_labels": torch.full((batch, capacity), -1, dtype=torch.long),
        "pair_mask": torch.zeros((batch, capacity), dtype=torch.bool),
        "global_count_features": torch.zeros(
            (batch, SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM), dtype=torch.float32
        ),
        "pair_count": torch.tensor(pair_counts, dtype=torch.long),
    }
    for index, record in enumerate(records):
        target_free = record["target_free"]
        target_aware = record["target_aware"]
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
            or global_features.shape != (SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,)
        ):
            raise ValueError("endpoint record tensor shapes differ")
        if count:
            result["pair_tokens"][index, :count] = tokens
            result["pair_indices"][index, :count] = pairs
            result["pair_labels"][index, :count] = labels
            result["pair_mask"][index, :count] = True
        result["global_count_features"][index] = global_features
    return result


def deterministic_batch_indices(
    record_count: int,
    *,
    steps: int,
    batch_size: int,
    seed: int,
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


def train_endpoint_steps(
    model: EvidenceCarrierEndpointIncidenceResidual,
    optimizer: torch.optim.Optimizer,
    records: Sequence[Mapping[str, Any]],
    *,
    steps: int = TRAINING_STEPS,
    batch_size: int = TRAINING_BATCH_SIZE,
    seed: int,
    device: torch.device = torch.device("cpu"),
) -> dict[str, Any]:
    if len(records) != EXPECTED_TRAIN_SAMPLE_COUNT:
        raise ValueError("endpoint trainer requires exactly 4,800 records")
    model.train()
    names = (
        "loss",
        "positive_loss",
        "negative_loss",
        "worst_negative_loss",
        "negative_upward_loss",
        "ranking_loss",
        "residual_l2_loss",
    )
    values = {name: [] for name in names}
    gradient_norms = []
    for indices in deterministic_batch_indices(
        len(records), steps=steps, batch_size=batch_size, seed=seed
    ):
        current = [records[index] for index in indices]
        batch = _to_device(collate_endpoint_records(current), device)
        output = model(
            batch["pair_tokens"],
            batch["pair_indices"],
            batch["pair_mask"],
            batch["global_count_features"],
        )
        losses = endpoint_safe_residual_loss(
            output["margin"],
            output["residual"],
            batch["pair_indices"],
            batch["pair_labels"],
            batch["pair_mask"],
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        gradient = torch.nn.utils.clip_grad_norm_(
            model.parameters(), TRAINING_MAX_GRAD_NORM, error_if_nonfinite=True
        )
        optimizer.step()
        for name in names:
            values[name].append(float(losses[name].detach().cpu()))
        gradient_norms.append(float(gradient.detach().cpu()))
    return {
        "optimizer_steps": steps,
        "batch_size": batch_size,
        "seed": seed,
        **{f"mean_{name}": float(np.mean(items)) for name, items in values.items()},
        **{f"final_{name}": items[-1] for name, items in values.items()},
        "maximum_preclip_gradient_norm": max(gradient_norms),
    }


@torch.inference_mode()
def audit_train_finite(
    model: EvidenceCarrierEndpointIncidenceResidual,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    for name, parameter in model.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise EndpointOOFTrainingError(f"model parameter {name} is non-finite")
    residual_sum = residual_square_sum = 0.0
    maximum_absolute = 0.0
    pair_count = 0
    model.eval()
    for start in range(0, len(records), TRAINING_BATCH_SIZE):
        batch = _to_device(
            collate_endpoint_records(records[start : start + TRAINING_BATCH_SIZE]),
            device,
        )
        output = model(
            batch["pair_tokens"],
            batch["pair_indices"],
            batch["pair_mask"],
            batch["global_count_features"],
        )
        active = batch["pair_mask"]
        for name in ("residual", "corrected_gain", "margin", "endpoint_context"):
            tensor = output[name]
            checked = tensor[active] if tensor.ndim == 2 else tensor[active]
            if not bool(torch.isfinite(checked).all()):
                raise EndpointOOFTrainingError(f"train finite audit failed for {name}")
        residual = output["residual"][active].double()
        pair_count += len(residual)
        residual_sum += float(residual.sum())
        residual_square_sum += float(residual.square().sum())
        if len(residual):
            maximum_absolute = max(maximum_absolute, float(residual.abs().max()))
    return {
        "status": "TRAIN-FINITE-AUDIT-PASSED",
        "sample_count": len(records),
        "pair_count": pair_count,
        "residual_mean": residual_sum / pair_count,
        "residual_rms": math.sqrt(residual_square_sum / pair_count),
        "residual_maximum_absolute": maximum_absolute,
    }


def _evaluation_target(record: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = record["target_aware"]["evaluation_target"]
    return {
        **dict(evaluation),
        "pair_indices": record["target_free"]["pair_indices"],
        "pair_labels": record["target_aware"]["pair_labels"],
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


def oof_gate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    expected_sample_count: int,
) -> dict[str, Any]:
    checks = {
        "sample_count": candidate["sample_count"] == expected_sample_count,
        "cross_equal_0": candidate["selected_cross_target_pairs"] == 0,
        "precision_at_least_0_999": (
            (candidate["selected_pair_precision"] or 0.0) >= 0.999
        ),
        "m2_recall_at_least_0_995": (
            (candidate["multiplicity_2_recall"] or 0.0) >= 0.995
        ),
        "pairable_exact_at_least_0_95": (
            (candidate["pairable_sample_exact"] or 0.0) >= 0.95
        ),
        "parallel_2_preservation_equal_1": (
            candidate["parallel_2_distinct_preservation"] == 1.0
        ),
        "support_loss_at_most_0_1pp": (
            candidate["ordinary_support_loss_pp"] is not None
            and candidate["ordinary_support_loss_pp"] <= 0.1
        ),
        "matching_optimal": (
            candidate["matching_optimal_sample_count"] == expected_sample_count
        ),
        "permutation_deterministic": candidate["permutation_deterministic"] is True,
        "strictly_better_than_baseline": (
            _structural_key(candidate) > _structural_key(baseline)
        ),
    }
    return {
        "status": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "baseline_structural_key": list(_structural_key(baseline)),
        "candidate_structural_key": list(_structural_key(candidate)),
    }


@torch.inference_mode()
def evaluate_oof_records(
    model: EvidenceCarrierEndpointIncidenceResidual,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    baseline_rows = []
    candidate_rows = []
    changed_rows = []
    residual_values = []
    forward_latency = []
    exact_fixed = exact_regressed = introduced_cross = 0
    model.eval()
    for record in records:
        batch = _to_device(collate_endpoint_records([record]), device)
        started = perf_counter()
        output = model(
            batch["pair_tokens"],
            batch["pair_indices"],
            batch["pair_mask"],
            batch["global_count_features"],
        )
        forward_latency.append((perf_counter() - started) * 1000.0)
        count = int(batch["pair_count"][0])
        raw = batch["pair_tokens"][0, :count, -1].detach().cpu()
        corrected = output["corrected_gain"][0, :count].detach().cpu()
        residual = output["residual"][0, :count].detach().cpu()
        residual_values.extend(map(float, residual.tolist()))
        target = _evaluation_target(record)
        baseline = evaluate_evidence_carrier_pair_sample(
            target,
            raw,
            threshold=SAMPLE_THRESHOLD_BASELINE_TAU,
            verify_permutation=True,
        )
        candidate = evaluate_evidence_carrier_pair_sample(
            target,
            corrected,
            threshold=SAMPLE_THRESHOLD_BASELINE_TAU,
            verify_permutation=True,
        )
        baseline_rows.append(baseline)
        candidate_rows.append(candidate)
        before_exact = baseline["pairable_sample_exact"] is True
        after_exact = candidate["pairable_sample_exact"] is True
        exact_fixed += int(not before_exact and after_exact)
        exact_regressed += int(before_exact and not after_exact)
        introduced_cross += int(
            baseline["selected_cross_target_pairs"] == 0
            and candidate["selected_cross_target_pairs"] > 0
        )
        if baseline["selected_anchor_pairs"] != candidate["selected_anchor_pairs"]:
            changed_rows.append(
                {
                    "identity": dict(record["identity"]),
                    "baseline": baseline,
                    "candidate": candidate,
                    "residual_minimum": float(residual.min()) if len(residual) else None,
                    "residual_maximum": float(residual.max()) if len(residual) else None,
                }
            )
    baseline_metrics = aggregate_evidence_carrier_pair_evaluations(baseline_rows)
    candidate_metrics = aggregate_evidence_carrier_pair_evaluations(candidate_rows)
    return {
        "baseline_rows": baseline_rows,
        "candidate_rows": candidate_rows,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "changed_sample_count": len(changed_rows),
        "changed_samples": changed_rows,
        "pairable_exact_fixed_count": exact_fixed,
        "pairable_exact_regressed_count": exact_regressed,
        "introduced_cross_sample_count": introduced_cross,
        "residual_distribution": _distribution(residual_values),
        "forward_latency_ms": _distribution(forward_latency),
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


def _save_checkpoint(
    path: Path,
    payload: Mapping[str, Any],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(path)


def _load_checkpoint(
    path: Path,
    *,
    device: torch.device,
    expected_implementation: Mapping[str, str],
) -> EvidenceCarrierEndpointIncidenceResidual:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": EvidenceCarrierEndpointIncidenceResidual.model_type,
        "parameter_count": ENDPOINT_RESIDUAL_PARAMETER_COUNT,
        "optimizer_steps": TRAINING_STEPS,
        "checkpoint_selection": "last_step_only",
        "implementation_sha256": dict(expected_implementation),
        "v0917_dev_sidecar_read": False,
        "v0917_dev_report_read": False,
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise EndpointOOFTrainingError(f"checkpoint contract differs: {mismatches}")
    model = EvidenceCarrierEndpointIncidenceResidual()
    if checkpoint.get("model_config") != model.config_dict():
        raise EndpointOOFTrainingError("checkpoint model config differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model


def train(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    report_path = Path(args.report).expanduser().resolve(strict=False)
    run_lock = output_dir / "run.lock"
    summary_path = output_dir / "summary.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"OOF output directory is not empty: {output_dir}")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite OOF report: {report_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_no_overwrite(
        run_lock,
        {
            "schema_version": "evidence-carrier-endpoint-oof-run-lock-0.1",
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "steps_per_fold": TRAINING_STEPS,
        },
    )
    index, index_metadata = load_oof_index(args.index, audit_path=args.audit)
    frozen_implementation = implementation_sha256()
    random.seed(TRAINING_SEEDS[0])
    np.random.seed(TRAINING_SEEDS[0])
    torch.manual_seed(TRAINING_SEEDS[0])
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cpu")

    fold_reports = []
    all_baseline_rows = []
    all_candidate_rows = []
    for eval_fold in range(3):
        train_fold_ids = [fold for fold in range(3) if fold != eval_fold]
        train_records = []
        train_sidecars = []
        for fold in train_fold_ids:
            records, metadata = load_oof_fold(index, fold)
            train_records.extend(records)
            train_sidecars.append({"fold": fold, **metadata})
        train_records.sort(key=lambda record: record["identity"]["sample_key"])
        if len(train_records) != EXPECTED_TRAIN_SAMPLE_COUNT:
            raise EndpointOOFTrainingError("OOF train sample count differs")
        seed = TRAINING_SEEDS[eval_fold]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = EvidenceCarrierEndpointIncidenceResidual().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAINING_LEARNING_RATE,
            weight_decay=TRAINING_WEIGHT_DECAY,
        )
        metrics = train_endpoint_steps(
            model,
            optimizer,
            train_records,
            seed=seed,
            device=device,
        )
        if implementation_sha256() != frozen_implementation:
            raise EndpointOOFTrainingError("implementation changed during training")
        finite_audit = audit_train_finite(model, train_records, device=device)
        fold_dir = output_dir / f"fold{eval_fold}"
        checkpoint_path = fold_dir / "final.pt"
        checkpoint = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_type": model.model_type,
            "model": model.state_dict(),
            "model_config": model.config_dict(),
            "parameter_count": model.parameter_count(),
            "optimizer_steps": TRAINING_STEPS,
            "checkpoint_selection": "last_step_only",
            "eval_fold": eval_fold,
            "train_folds": train_fold_ids,
            "seed": seed,
            "training_metrics": metrics,
            "train_finite_audit": finite_audit,
            "train_sidecars": train_sidecars,
            "index_metadata": index_metadata,
            "source_sha256": dict(index["source_sha256"]),
            "implementation_sha256": frozen_implementation,
            "v0917_dev_sidecar_read": False,
            "v0917_dev_report_read": False,
            "original_calibration_records_accessed": False,
            "fresh_read": False,
            "sealed_read": False,
            "internal_test_read": False,
            "mr_tadf_read": False,
        }
        checkpoint_sha_before = _save_checkpoint(checkpoint_path, checkpoint)
        sealed_model = _load_checkpoint(
            checkpoint_path,
            device=device,
            expected_implementation=frozen_implementation,
        )

        eval_records, eval_metadata = load_oof_fold(index, eval_fold)
        if len(eval_records) != EXPECTED_OOF_SAMPLE_COUNT:
            raise EndpointOOFTrainingError("OOF eval sample count differs")
        evaluation = evaluate_oof_records(sealed_model, eval_records, device=device)
        checkpoint_sha_after = file_sha256(checkpoint_path)
        if checkpoint_sha_after != checkpoint_sha_before:
            raise EndpointOOFTrainingError("checkpoint changed during OOF evaluation")
        gate = oof_gate(
            evaluation["baseline_metrics"],
            evaluation["candidate_metrics"],
            expected_sample_count=EXPECTED_OOF_SAMPLE_COUNT,
        )
        all_baseline_rows.extend(evaluation.pop("baseline_rows"))
        all_candidate_rows.extend(evaluation.pop("candidate_rows"))
        fold_report = {
            "fold": eval_fold,
            "status": gate["status"],
            "train_folds": train_fold_ids,
            "train_sample_count": len(train_records),
            "eval_sample_count": len(eval_records),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256_before_eval": checkpoint_sha_before,
            "checkpoint_sha256_after_eval": checkpoint_sha_after,
            "checkpoint_sha256_unchanged": True,
            "eval_sidecar": {"fold": eval_fold, **eval_metadata},
            "training_metrics": metrics,
            "train_finite_audit": finite_audit,
            "gate": gate,
            **evaluation,
        }
        fold_report_path = fold_dir / "oof.json"
        _write_json_no_overwrite(fold_report_path, fold_report)
        fold_report["report_path"] = str(fold_report_path)
        fold_report["report_sha256"] = file_sha256(fold_report_path)
        fold_reports.append(fold_report)

    aggregate_baseline = aggregate_evidence_carrier_pair_evaluations(all_baseline_rows)
    aggregate_candidate = aggregate_evidence_carrier_pair_evaluations(all_candidate_rows)
    aggregate_gate = oof_gate(
        aggregate_baseline,
        aggregate_candidate,
        expected_sample_count=3 * EXPECTED_OOF_SAMPLE_COUNT,
    )
    status = (
        STATUS_GO
        if aggregate_gate["status"] == "GO"
        and all(report["gate"]["status"] == "GO" for report in fold_reports)
        else STATUS_NO_GO
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "fold_reports": fold_reports,
        "aggregate_baseline_metrics": aggregate_baseline,
        "aggregate_candidate_metrics": aggregate_candidate,
        "aggregate_gate": aggregate_gate,
        "all_fold_gates_go": all(
            report["gate"]["status"] == "GO" for report in fold_reports
        ),
        "index_metadata": index_metadata,
        "implementation_sha256": frozen_implementation,
        "source_sha256": dict(index["source_sha256"]),
        "v0917_dev_sidecar_read": False,
        "v0917_dev_report_read": False,
        "original_calibration_records_accessed": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
    }
    _write_json_no_overwrite(report_path, report)
    report_sha = file_sha256(report_path)
    summary = {
        "schema_version": "evidence-carrier-endpoint-oof-summary-0.1",
        "status": status,
        "report_path": str(report_path),
        "report_sha256": report_sha,
        "aggregate_baseline_metrics": aggregate_baseline,
        "aggregate_candidate_metrics": aggregate_candidate,
        "aggregate_gate": aggregate_gate,
        "folds": [
            {
                "fold": report["fold"],
                "status": report["status"],
                "checkpoint_path": report["checkpoint_path"],
                "checkpoint_sha256": report["checkpoint_sha256_after_eval"],
                "report_path": report["report_path"],
                "report_sha256": report["report_sha256"],
            }
            for report in fold_reports
        ],
        "implementation_sha256": frozen_implementation,
    }
    _write_json_no_overwrite(summary_path, summary)
    summary_sha = file_sha256(summary_path)
    _replace_json(
        run_lock,
        {
            "schema_version": "evidence-carrier-endpoint-oof-run-lock-0.1",
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "result_status": status,
            "report_path": str(report_path),
            "report_sha256": report_sha,
            "summary_path": str(summary_path),
            "summary_sha256": summary_sha,
        },
    )
    return {**summary, "summary_sha256": summary_sha}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=DEFAULT_OUTPUT_INDEX)
    parser.add_argument("--audit", default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(_plain(train(args)), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "EndpointOOFTrainingError",
    "collate_endpoint_records",
    "deterministic_batch_indices",
    "evaluate_oof_records",
    "load_oof_fold",
    "load_oof_index",
    "oof_gate",
    "train_endpoint_steps",
]
