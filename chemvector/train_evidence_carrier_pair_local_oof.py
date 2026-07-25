#!/usr/bin/env python3
"""Train and evaluate v0.9.19 pair-local raster residual with 3-fold OOF."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import chemvector.evidence_carrier_pair_local_residual as model_module
import chemvector.train_evidence_carrier_endpoint_oof as base
from chemvector.build_evidence_carrier_pair_local_oof import (
    DEFAULT_INDEX as DEFAULT_LOCAL_INDEX,
    INDEX_SCHEMA_VERSION as LOCAL_INDEX_SCHEMA_VERSION,
    STATUS_COMPLETE as LOCAL_STATUS_COMPLETE,
    file_sha256,
    validate_pair_local_sidecar,
)
from chemvector.evidence_carrier_endpoint_incidence_residual import (
    endpoint_safe_residual_loss,
)
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
    evaluate_evidence_carrier_pair_sample,
)
from chemvector.evidence_carrier_pair_local_residual import (
    PAIR_LOCAL_RESIDUAL_PARAMETER_COUNT,
    EvidenceCarrierPairLocalResidual,
)
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,
    SAMPLE_THRESHOLD_PAIR_TOKEN_DIM,
)


CHECKPOINT_SCHEMA_VERSION = "evidence-carrier-pair-local-oof-checkpoint-0.1"
REPORT_SCHEMA_VERSION = "evidence-carrier-pair-local-oof-report-0.1"
STATUS_GO = "FIT-OOF-PAIR-LOCAL-RASTER-GO"
STATUS_NO_GO = "FIT-OOF-PAIR-LOCAL-RASTER-NO-GO"
DEFAULT_OUTPUT_DIR = "outputs/checkpoints/chemvector-v0919-pair-local-oof"
DEFAULT_REPORT = "outputs/evaluations/chemvector-v0919-pair-local-oof/oof.json"
TRAINING_STEPS = 2_000
TRAINING_BATCH_SIZE = 32
TRAINING_SEEDS = (20_260_725, 20_260_726, 20_260_727)
TRAINING_LEARNING_RATE = 1e-3
TRAINING_WEIGHT_DECAY = 1e-4
TRAINING_MAX_GRAD_NORM = 1.0


class PairLocalOOFTrainingError(RuntimeError):
    """Raised when pair-local OOF provenance or training differs."""


def implementation_sha256() -> dict[str, str]:
    return {
        "trainer": file_sha256(Path(__file__).resolve()),
        "model": file_sha256(Path(model_module.__file__).resolve()),
        "base_oof_trainer": file_sha256(Path(base.__file__).resolve()),
    }


def _write_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return file_sha256(path)


def load_local_index(path: str | Path = DEFAULT_LOCAL_INDEX) -> tuple[dict, dict]:
    resolved = Path(path).expanduser().resolve(strict=True)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise PairLocalOOFTrainingError("pair-local index must be a mapping")
    payload = dict(payload)
    if (
        payload.get("schema_version") != LOCAL_INDEX_SCHEMA_VERSION
        or payload.get("status") != LOCAL_STATUS_COMPLETE
        or payload.get("counts") != {"sample_count": 7200, "pair_count": 368573}
    ):
        raise PairLocalOOFTrainingError("pair-local index contract differs")
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise PairLocalOOFTrainingError("pair-local index folds differ")
    return payload, {"path": str(resolved), "sha256": file_sha256(resolved)}


def load_local_fold(index: Mapping[str, Any], fold: int) -> tuple[dict[str, torch.Tensor], dict]:
    if fold not in (0, 1, 2):
        raise ValueError("fold must be 0, 1, or 2")
    reference = index["folds"][fold]
    path = Path(reference["path"]).resolve(strict=True)
    sha = file_sha256(path)
    if sha != reference["sha256"]:
        raise PairLocalOOFTrainingError("pair-local fold SHA differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_pair_local_sidecar(payload, expected_fold=fold)
    records = payload["records"]
    mapping = {str(record["sample_key"]): record["pair_local_rasters"] for record in records}
    if len(mapping) != len(records):
        raise PairLocalOOFTrainingError("pair-local fold sample keys overlap")
    return mapping, {"path": str(path), "sha256": sha}


def merge_pair_local_records(
    records: Sequence[Mapping[str, Any]], crops_by_key: Mapping[str, torch.Tensor]
) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for record in records:
        key = str(record["identity"]["sample_key"])
        crops = crops_by_key.get(key)
        if crops is None:
            raise PairLocalOOFTrainingError(f"sample {key} lacks pair-local crops")
        pair_count = len(record["target_free"]["pair_tokens"])
        if crops.dtype != torch.uint8 or crops.shape != (pair_count, 2, 24, 48):
            raise PairLocalOOFTrainingError(f"sample {key} pair-local crop shape differs")
        merged.append({**record, "pair_local_rasters": crops})
        seen.add(key)
    if seen != set(crops_by_key):
        raise PairLocalOOFTrainingError("pair-local fold contains unmatched samples")
    return merged


def collate_pair_local_records(records: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("cannot collate an empty pair-local batch")
    pair_counts = [len(record["target_free"]["pair_tokens"]) for record in records]
    capacity = max(max(pair_counts), 1)
    batch = len(records)
    result = {
        "pair_tokens": torch.zeros((batch, capacity, SAMPLE_THRESHOLD_PAIR_TOKEN_DIM)),
        "pair_local_rasters": torch.zeros(
            (batch, capacity, 2, 24, 48), dtype=torch.uint8
        ),
        "pair_indices": torch.full((batch, capacity, 2), -1, dtype=torch.long),
        "pair_labels": torch.full((batch, capacity), -1, dtype=torch.long),
        "pair_mask": torch.zeros((batch, capacity), dtype=torch.bool),
        "global_count_features": torch.zeros(
            (batch, SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM)
        ),
        "pair_count": torch.tensor(pair_counts, dtype=torch.long),
    }
    for index, record in enumerate(records):
        count = pair_counts[index]
        target_free = record["target_free"]
        tensors = {
            "pair_tokens": torch.as_tensor(target_free["pair_tokens"], dtype=torch.float32),
            "pair_local_rasters": torch.as_tensor(
                record["pair_local_rasters"], dtype=torch.uint8
            ),
            "pair_indices": torch.as_tensor(target_free["pair_indices"], dtype=torch.long),
            "pair_labels": torch.as_tensor(
                record["target_aware"]["pair_labels"], dtype=torch.long
            ),
        }
        expected = {
            "pair_tokens": (count, 21),
            "pair_local_rasters": (count, 2, 24, 48),
            "pair_indices": (count, 2),
            "pair_labels": (count,),
        }
        if any(tuple(tensors[name].shape) != expected[name] for name in expected):
            raise PairLocalOOFTrainingError("pair-local record tensor shapes differ")
        if count:
            for name, tensor in tensors.items():
                result[name][index, :count] = tensor
            result["pair_mask"][index, :count] = True
        global_features = torch.as_tensor(
            target_free["global_count_features"], dtype=torch.float32
        )
        if global_features.shape != (4,):
            raise PairLocalOOFTrainingError("global count feature shape differs")
        result["global_count_features"][index] = global_features
    return result


def _to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    result = {name: value.to(device, non_blocking=True) for name, value in batch.items()}
    result["pair_local_rasters"] = result["pair_local_rasters"].to(torch.float32).div_(255.0)
    return result


def _forward(model: EvidenceCarrierPairLocalResidual, batch: Mapping[str, torch.Tensor]):
    return model(
        batch["pair_tokens"],
        batch["pair_local_rasters"],
        batch["pair_indices"],
        batch["pair_mask"],
        batch["global_count_features"],
    )


def train_steps(
    model: EvidenceCarrierPairLocalResidual,
    optimizer: torch.optim.Optimizer,
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if len(records) != 4800:
        raise ValueError("pair-local trainer requires exactly 4,800 records")
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
    gradients = []
    for indices in base.deterministic_batch_indices(
        len(records),
        steps=TRAINING_STEPS,
        batch_size=TRAINING_BATCH_SIZE,
        seed=seed,
    ):
        batch = _to_device(
            collate_pair_local_records([records[index] for index in indices]), device
        )
        output = _forward(model, batch)
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
        gradients.append(float(gradient.detach().cpu()))
    return {
        "optimizer_steps": TRAINING_STEPS,
        "batch_size": TRAINING_BATCH_SIZE,
        "seed": seed,
        **{f"mean_{name}": float(np.mean(items)) for name, items in values.items()},
        **{f"final_{name}": items[-1] for name, items in values.items()},
        "maximum_preclip_gradient_norm": max(gradients),
    }


@torch.inference_mode()
def audit_train_finite(
    model: EvidenceCarrierPairLocalResidual,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    pair_count = 0
    residual_sum = residual_square_sum = maximum_absolute = 0.0
    unsafe_images = 0
    for start in range(0, len(records), TRAINING_BATCH_SIZE):
        batch = _to_device(
            collate_pair_local_records(records[start : start + TRAINING_BATCH_SIZE]),
            device,
        )
        output = _forward(model, batch)
        if not all(
            bool(torch.isfinite(output[name]).all())
            for name in ("residual", "corrected_gain", "margin", "local_features")
        ):
            raise PairLocalOOFTrainingError("train finite audit failed")
        active = batch["pair_mask"]
        residual = output["residual"][active].double()
        pair_count += len(residual)
        residual_sum += float(residual.sum())
        residual_square_sum += float(residual.square().sum())
        if len(residual):
            maximum_absolute = max(maximum_absolute, float(residual.abs().max()))
        negative = (batch["pair_labels"] == 0) & active
        unsafe_images += int(
            (((output["margin"] >= 0.0) & negative).any(dim=1)).sum().detach().cpu()
        )
    return {
        "status": "TRAIN-FINITE-AUDIT-PASSED",
        "sample_count": len(records),
        "pair_count": pair_count,
        "unsafe_image_count": unsafe_images,
        "residual_mean": residual_sum / pair_count,
        "residual_rms": math.sqrt(residual_square_sum / pair_count),
        "residual_maximum_absolute": maximum_absolute,
    }


@torch.inference_mode()
def evaluate_oof_records(
    model: EvidenceCarrierPairLocalResidual,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    baseline_rows = []
    candidate_rows = []
    changed_rows = []
    residual_values = []
    forward_latency = []
    fixed = regressed = introduced_cross = 0
    for start in range(0, len(records), TRAINING_BATCH_SIZE):
        current = records[start : start + TRAINING_BATCH_SIZE]
        batch = _to_device(collate_pair_local_records(current), device)
        started = perf_counter()
        output = _forward(model, batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (perf_counter() - started) * 1000.0
        forward_latency.extend([elapsed / len(current)] * len(current))
        for image_index, record in enumerate(current):
            count = int(batch["pair_count"][image_index])
            raw = batch["pair_tokens"][image_index, :count, -1].detach().cpu()
            corrected = output["corrected_gain"][image_index, :count].detach().cpu()
            residual = output["residual"][image_index, :count].detach().cpu()
            residual_values.extend(map(float, residual.tolist()))
            target = base._evaluation_target(record)
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
            before = baseline["pairable_sample_exact"] is True
            after = candidate["pairable_sample_exact"] is True
            fixed += int(not before and after)
            regressed += int(before and not after)
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
    return {
        "baseline_rows": baseline_rows,
        "candidate_rows": candidate_rows,
        "baseline_metrics": aggregate_evidence_carrier_pair_evaluations(baseline_rows),
        "candidate_metrics": aggregate_evidence_carrier_pair_evaluations(candidate_rows),
        "changed_sample_count": len(changed_rows),
        "changed_samples": changed_rows,
        "pairable_exact_fixed_count": fixed,
        "pairable_exact_regressed_count": regressed,
        "introduced_cross_sample_count": introduced_cross,
        "residual_distribution": base._distribution(residual_values),
        "forward_latency_ms": base._distribution(forward_latency),
    }


def _load_checkpoint(
    path: Path, *, device: torch.device, implementation: Mapping[str, str]
) -> EvidenceCarrierPairLocalResidual:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": EvidenceCarrierPairLocalResidual.model_type,
        "parameter_count": PAIR_LOCAL_RESIDUAL_PARAMETER_COUNT,
        "optimizer_steps": TRAINING_STEPS,
        "checkpoint_selection": "last_step_only",
        "implementation_sha256": dict(implementation),
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise PairLocalOOFTrainingError(f"checkpoint contract differs: {mismatches}")
    model = EvidenceCarrierPairLocalResidual()
    if checkpoint.get("model_config") != model.config_dict():
        raise PairLocalOOFTrainingError("checkpoint model config differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def train(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if report_path.exists():
        raise FileExistsError(report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_no_overwrite(
        output_dir / "run.lock",
        {
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "steps_per_fold": TRAINING_STEPS,
        },
    )
    base_index, base_metadata = base.load_oof_index(args.base_index, audit_path=args.base_audit)
    local_index, local_metadata = load_local_index(args.local_index)
    frozen_implementation = implementation_sha256()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PairLocalOOFTrainingError("CUDA was requested but is unavailable")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

    fold_reports = []
    all_baseline_rows = []
    all_candidate_rows = []
    for eval_fold in range(3):
        train_records = []
        train_sidecars = []
        for fold in range(3):
            if fold == eval_fold:
                continue
            records, base_fold_meta = base.load_oof_fold(base_index, fold)
            crops, local_fold_meta = load_local_fold(local_index, fold)
            train_records.extend(merge_pair_local_records(records, crops))
            train_sidecars.append(
                {"fold": fold, "base": base_fold_meta, "pair_local": local_fold_meta}
            )
        train_records.sort(key=lambda record: record["identity"]["sample_key"])
        seed = TRAINING_SEEDS[eval_fold]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = EvidenceCarrierPairLocalResidual().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=TRAINING_LEARNING_RATE, weight_decay=TRAINING_WEIGHT_DECAY
        )
        metrics = train_steps(model, optimizer, train_records, seed=seed, device=device)
        if implementation_sha256() != frozen_implementation:
            raise PairLocalOOFTrainingError("implementation changed during training")
        finite = audit_train_finite(model, train_records, device=device)
        checkpoint_path = output_dir / f"fold{eval_fold}" / "final.pt"
        checkpoint = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_type": model.model_type,
            "model": model.state_dict(),
            "model_config": model.config_dict(),
            "parameter_count": model.parameter_count(),
            "optimizer_steps": TRAINING_STEPS,
            "checkpoint_selection": "last_step_only",
            "eval_fold": eval_fold,
            "train_folds": [fold for fold in range(3) if fold != eval_fold],
            "seed": seed,
            "device": str(device),
            "training_metrics": metrics,
            "train_finite_audit": finite,
            "train_sidecars": train_sidecars,
            "base_index_metadata": base_metadata,
            "local_index_metadata": local_metadata,
            "implementation_sha256": frozen_implementation,
            "validation_read": False,
            "fresh_read": False,
            "sealed_read": False,
            "internal_test_read": False,
            "mr_tadf_read": False,
        }
        checkpoint_sha = _save_checkpoint(checkpoint_path, checkpoint)
        sealed = _load_checkpoint(
            checkpoint_path, device=device, implementation=frozen_implementation
        )
        eval_records, eval_base_meta = base.load_oof_fold(base_index, eval_fold)
        eval_crops, eval_local_meta = load_local_fold(local_index, eval_fold)
        eval_records = merge_pair_local_records(eval_records, eval_crops)
        evaluation = evaluate_oof_records(sealed, eval_records, device=device)
        if file_sha256(checkpoint_path) != checkpoint_sha:
            raise PairLocalOOFTrainingError("checkpoint changed during OOF evaluation")
        gate = base.oof_gate(
            evaluation["baseline_metrics"],
            evaluation["candidate_metrics"],
            expected_sample_count=2400,
        )
        all_baseline_rows.extend(evaluation.pop("baseline_rows"))
        all_candidate_rows.extend(evaluation.pop("candidate_rows"))
        fold_report = {
            "fold": eval_fold,
            "status": gate["status"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "training_metrics": metrics,
            "train_finite_audit": finite,
            "eval_base_sidecar": eval_base_meta,
            "eval_pair_local_sidecar": eval_local_meta,
            "gate": gate,
            **evaluation,
        }
        fold_report_path = checkpoint_path.parent / "oof.json"
        _write_json_no_overwrite(fold_report_path, fold_report)
        fold_report["report_path"] = str(fold_report_path)
        fold_report["report_sha256"] = file_sha256(fold_report_path)
        fold_reports.append(fold_report)
        del model, sealed, optimizer, train_records, eval_records
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate_baseline = aggregate_evidence_carrier_pair_evaluations(all_baseline_rows)
    aggregate_candidate = aggregate_evidence_carrier_pair_evaluations(all_candidate_rows)
    aggregate_gate = base.oof_gate(
        aggregate_baseline, aggregate_candidate, expected_sample_count=7200
    )
    all_folds_go = all(report["gate"]["status"] == "GO" for report in fold_reports)
    status = STATUS_GO if aggregate_gate["status"] == "GO" and all_folds_go else STATUS_NO_GO
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "fold_reports": fold_reports,
        "aggregate_baseline_metrics": aggregate_baseline,
        "aggregate_candidate_metrics": aggregate_candidate,
        "aggregate_gate": aggregate_gate,
        "all_fold_gates_go": all_folds_go,
        "base_index_metadata": base_metadata,
        "local_index_metadata": local_metadata,
        "implementation_sha256": frozen_implementation,
        "validation_read": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
    }
    _write_json_no_overwrite(report_path, report)
    summary = {
        "schema_version": "evidence-carrier-pair-local-oof-summary-0.1",
        "status": status,
        "report_path": str(report_path),
        "report_sha256": file_sha256(report_path),
        "parameter_count": PAIR_LOCAL_RESIDUAL_PARAMETER_COUNT,
        "device": str(device),
        "aggregate_baseline_metrics": aggregate_baseline,
        "aggregate_candidate_metrics": aggregate_candidate,
        "aggregate_gate": aggregate_gate,
        "folds": [
            {
                "fold": row["fold"],
                "status": row["status"],
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "report_path": row["report_path"],
                "report_sha256": row["report_sha256"],
            }
            for row in fold_reports
        ],
    }
    _write_json_no_overwrite(output_dir / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-index", default=base.DEFAULT_OUTPUT_INDEX)
    parser.add_argument("--base-audit", default=base.DEFAULT_AUDIT)
    parser.add_argument("--local-index", default=DEFAULT_LOCAL_INDEX)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    print(json.dumps(train(build_arg_parser().parse_args()), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "collate_pair_local_records",
    "load_local_fold",
    "load_local_index",
    "merge_pair_local_records",
]
