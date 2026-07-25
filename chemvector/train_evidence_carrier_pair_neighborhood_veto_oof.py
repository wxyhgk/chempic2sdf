#!/usr/bin/env python3
"""Train v0.9.24 ordered-neighborhood veto with consensus OOF evaluation."""

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

import chemvector.evidence_carrier_pair_neighborhood_veto as model_module
import chemvector.train_evidence_carrier_endpoint_oof as base
import chemvector.train_evidence_carrier_pair_junction_veto_oof as consensus_helper
import chemvector.train_evidence_carrier_pair_local_oof as helper
from chemvector.build_evidence_carrier_pair_local_oof import file_sha256
from chemvector.build_evidence_carrier_pair_neighborhood_oof import (
    DEFAULT_INDEX as DEFAULT_NEIGHBORHOOD_INDEX,
    INDEX_SCHEMA_VERSION as NEIGHBORHOOD_INDEX_SCHEMA_VERSION,
    STATUS_COMPLETE as NEIGHBORHOOD_STATUS_COMPLETE,
    validate_neighborhood_sidecar,
)
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
    evaluate_evidence_carrier_pair_sample,
)
from chemvector.evidence_carrier_pair_local_residual import (
    EvidenceCarrierPairLocalResidual,
)
from chemvector.evidence_carrier_pair_local_spatial_residual import (
    EvidenceCarrierPairLocalSpatialResidual,
)
from chemvector.evidence_carrier_pair_local_veto import pair_local_veto_loss
from chemvector.evidence_carrier_pair_neighborhood_veto import (
    PAIR_NEIGHBORHOOD_VETO_PARAMETER_COUNT,
    EvidenceCarrierPairNeighborhoodVeto,
)
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
)


CHECKPOINT_SCHEMA_VERSION = (
    "evidence-carrier-pair-neighborhood-veto-oof-checkpoint-0.1"
)
REPORT_SCHEMA_VERSION = "evidence-carrier-pair-neighborhood-veto-oof-report-0.1"
STATUS_GO = "FIT-OOF-PAIR-NEIGHBORHOOD-VETO-GO"
STATUS_NO_GO = "FIT-OOF-PAIR-NEIGHBORHOOD-VETO-NO-GO"
DEFAULT_OUTPUT_DIR = "outputs/checkpoints/chemvector-v0924-pair-neighborhood-veto-oof"
DEFAULT_REPORT = "outputs/evaluations/chemvector-v0924-pair-neighborhood-veto-oof/oof.json"
TRAINING_STEPS = 2_000
TRAINING_BATCH_SIZE = 32
TRAINING_SEEDS = (20_260_740, 20_260_741, 20_260_742)
TRAINING_LEARNING_RATE = 1e-3
TRAINING_WEIGHT_DECAY = 1e-4
TRAINING_MAX_GRAD_NORM = 1.0


class PairNeighborhoodVetoOOFTrainingError(RuntimeError):
    """Raised when the neighborhood-veto OOF protocol differs."""


def implementation_sha256() -> dict[str, str]:
    return {
        "trainer": file_sha256(Path(__file__).resolve()),
        "model": file_sha256(Path(model_module.__file__).resolve()),
        "pair_local_helper": file_sha256(Path(helper.__file__).resolve()),
        "consensus_helper": file_sha256(Path(consensus_helper.__file__).resolve()),
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


def load_neighborhood_index(
    path: str | Path = DEFAULT_NEIGHBORHOOD_INDEX,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise PairNeighborhoodVetoOOFTrainingError(
            "pair-neighborhood index must be a mapping"
        )
    payload = dict(payload)
    if (
        payload.get("schema_version") != NEIGHBORHOOD_INDEX_SCHEMA_VERSION
        or payload.get("status") != NEIGHBORHOOD_STATUS_COMPLETE
        or payload.get("counts") != {"sample_count": 7200, "pair_count": 368573}
    ):
        raise PairNeighborhoodVetoOOFTrainingError(
            "pair-neighborhood index contract differs"
        )
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise PairNeighborhoodVetoOOFTrainingError(
            "pair-neighborhood index folds differ"
        )
    return payload, {"path": str(resolved), "sha256": file_sha256(resolved)}


def load_neighborhood_fold(
    index: Mapping[str, Any], fold: int
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    if fold not in (0, 1, 2):
        raise ValueError("fold must be 0, 1, or 2")
    reference = index["folds"][fold]
    path = Path(reference["path"]).resolve(strict=True)
    sha = file_sha256(path)
    if sha != reference["sha256"]:
        raise PairNeighborhoodVetoOOFTrainingError(
            "pair-neighborhood fold SHA differs"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_neighborhood_sidecar(payload, expected_fold=fold)
    records = payload["records"]
    mapping = {
        str(record["sample_key"]): (
            record["neighbor_features"],
            record["neighbor_mask"],
        )
        for record in records
    }
    if len(mapping) != len(records):
        raise PairNeighborhoodVetoOOFTrainingError(
            "pair-neighborhood fold sample keys overlap"
        )
    return mapping, {"path": str(path), "sha256": sha}


def merge_neighborhood_records(
    records: Sequence[Mapping[str, Any]],
    neighborhoods_by_key: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for record in records:
        key = str(record["identity"]["sample_key"])
        neighborhood = neighborhoods_by_key.get(key)
        if neighborhood is None:
            raise PairNeighborhoodVetoOOFTrainingError(
                f"sample {key} lacks pair-neighborhood features"
            )
        features, mask = neighborhood
        pair_count = len(record["target_free"]["pair_tokens"])
        if features.dtype != torch.float16 or features.shape != (pair_count, 12, 8):
            raise PairNeighborhoodVetoOOFTrainingError(
                f"sample {key} pair-neighborhood feature shape differs"
            )
        if mask.dtype != torch.bool or mask.shape != (pair_count, 12):
            raise PairNeighborhoodVetoOOFTrainingError(
                f"sample {key} pair-neighborhood mask shape differs"
            )
        merged.append(
            {
                **record,
                "neighbor_features": features,
                "neighbor_mask": mask,
            }
        )
        seen.add(key)
    if seen != set(neighborhoods_by_key):
        raise PairNeighborhoodVetoOOFTrainingError(
            "pair-neighborhood fold contains unmatched samples"
        )
    return merged


def collate_neighborhood_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, torch.Tensor]:
    result = helper.collate_pair_local_records(records)
    batch, capacity = result["pair_mask"].shape
    result["neighbor_features"] = torch.zeros(
        (batch, capacity, 12, 8), dtype=torch.float16
    )
    result["neighbor_mask"] = torch.zeros(
        (batch, capacity, 12), dtype=torch.bool
    )
    for index, record in enumerate(records):
        count = int(result["pair_count"][index])
        features = torch.as_tensor(record["neighbor_features"], dtype=torch.float16)
        mask = torch.as_tensor(record["neighbor_mask"], dtype=torch.bool)
        if features.shape != (count, 12, 8) or mask.shape != (count, 12):
            raise PairNeighborhoodVetoOOFTrainingError(
                "pair-neighborhood record tensor shapes differ"
            )
        if count:
            result["neighbor_features"][index, :count] = features
            result["neighbor_mask"][index, :count] = mask
    return result


def _to_device(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    result = helper._to_device(batch, device)
    result["neighbor_features"] = result["neighbor_features"].to(torch.float32)
    return result


def _forward(
    model: EvidenceCarrierPairNeighborhoodVeto,
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return model(
        batch["pair_tokens"],
        batch["pair_local_rasters"],
        batch["pair_indices"],
        batch["pair_mask"],
        batch["global_count_features"],
        batch["neighbor_features"],
        batch["neighbor_mask"],
    )


def train_steps(
    model: EvidenceCarrierPairNeighborhoodVeto,
    optimizer: torch.optim.Optimizer,
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if len(records) != 4800:
        raise ValueError("neighborhood veto trainer requires exactly 4,800 records")
    names = (
        "loss",
        "positive_loss",
        "worst_positive_loss",
        "negative_loss",
        "hard_negative_loss",
        "logit_l2_loss",
    )
    values = {name: [] for name in names}
    gradients = []
    model.train()
    for indices in base.deterministic_batch_indices(
        len(records), steps=TRAINING_STEPS, batch_size=TRAINING_BATCH_SIZE, seed=seed
    ):
        batch = _to_device(
            collate_neighborhood_records([records[index] for index in indices]),
            device,
        )
        output = _forward(model, batch)
        losses = pair_local_veto_loss(
            output["veto_logits"],
            output["raw_gain"],
            batch["pair_labels"],
            batch["pair_mask"],
            tau0=float(model.tau0),
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
def audit_train_veto(
    model: EvidenceCarrierPairNeighborhoodVeto,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    logit_sum = logit_square_sum = 0.0
    pair_count = positive_veto = negative_veto = missed_active_negative = 0
    for start in range(0, len(records), TRAINING_BATCH_SIZE):
        batch = _to_device(
            collate_neighborhood_records(records[start : start + TRAINING_BATCH_SIZE]),
            device,
        )
        output = _forward(model, batch)
        active = batch["pair_mask"]
        logits = output["veto_logits"][active].double()
        if not bool(torch.isfinite(logits).all()):
            raise PairNeighborhoodVetoOOFTrainingError(
                "train neighborhood veto logits are non-finite"
            )
        pair_count += len(logits)
        logit_sum += float(logits.sum())
        logit_square_sum += float(logits.square().sum())
        positive = (batch["pair_labels"] == 1) & active
        negative = (batch["pair_labels"] == 0) & active
        raw_active_negative = negative & (
            output["raw_gain"] > model.tau0.to(output["raw_gain"].dtype)
        )
        positive_veto += int((output["veto_mask"] & positive).sum().cpu())
        negative_veto += int((output["veto_mask"] & negative).sum().cpu())
        missed_active_negative += int(
            (raw_active_negative & ~output["veto_mask"]).sum().cpu()
        )
    return {
        "status": "TRAIN-NEIGHBORHOOD-VETO-AUDIT-PASSED",
        "sample_count": len(records),
        "pair_count": pair_count,
        "positive_veto_count": positive_veto,
        "negative_veto_count": negative_veto,
        "missed_active_negative_count": missed_active_negative,
        "logit_mean": logit_sum / pair_count,
        "logit_rms": math.sqrt(logit_square_sum / pair_count),
    }


@torch.inference_mode()
def evaluate_oof_records(
    veto_model: EvidenceCarrierPairNeighborhoodVeto,
    avg_model: EvidenceCarrierPairLocalResidual,
    spatial_model: EvidenceCarrierPairLocalSpatialResidual,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    veto_model.eval()
    avg_model.eval()
    spatial_model.eval()
    raw_rows = []
    consensus_rows = []
    candidate_rows = []
    changed_rows = []
    vetoed_positive = vetoed_negative = vetoed_samples = 0
    latency = []
    tau = float(SAMPLE_THRESHOLD_BASELINE_TAU)
    for start in range(0, len(records), TRAINING_BATCH_SIZE):
        current = records[start : start + TRAINING_BATCH_SIZE]
        batch = _to_device(collate_neighborhood_records(current), device)
        started = perf_counter()
        avg = helper._forward(avg_model, batch)
        spatial = helper._forward(spatial_model, batch)
        veto = _forward(veto_model, batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (perf_counter() - started) * 1000.0
        latency.extend([elapsed / len(current)] * len(current))
        for image_index, record in enumerate(current):
            count = int(batch["pair_count"][image_index])
            raw = batch["pair_tokens"][image_index, :count, -1].detach().cpu()
            consensus = torch.minimum(
                avg["corrected_gain"][image_index, :count],
                spatial["corrected_gain"][image_index, :count],
            ).detach().cpu()
            logits = veto["veto_logits"][image_index, :count].detach().cpu()
            remove = (consensus > tau) & (logits > 0.0)
            candidate = torch.where(
                remove, torch.full_like(consensus, tau - 1.0), consensus
            )
            labels = torch.as_tensor(
                record["target_aware"]["pair_labels"], dtype=torch.long
            )
            vetoed_positive += int((remove & (labels == 1)).sum())
            vetoed_negative += int((remove & (labels == 0)).sum())
            vetoed_samples += int(bool(remove.any()))
            target = base._evaluation_target(record)
            raw_eval = evaluate_evidence_carrier_pair_sample(
                target, raw, threshold=tau, verify_permutation=True
            )
            consensus_eval = evaluate_evidence_carrier_pair_sample(
                target, consensus, threshold=tau, verify_permutation=True
            )
            candidate_eval = evaluate_evidence_carrier_pair_sample(
                target, candidate, threshold=tau, verify_permutation=True
            )
            raw_rows.append(raw_eval)
            consensus_rows.append(consensus_eval)
            candidate_rows.append(candidate_eval)
            if consensus_eval["selected_anchor_pairs"] != candidate_eval[
                "selected_anchor_pairs"
            ]:
                changed_rows.append(
                    {
                        "identity": dict(record["identity"]),
                        "consensus": consensus_eval,
                        "candidate": candidate_eval,
                        "vetoed_candidate_indices": torch.where(remove)[0].tolist(),
                    }
                )
    return {
        "raw_rows": raw_rows,
        "consensus_rows": consensus_rows,
        "candidate_rows": candidate_rows,
        "raw_metrics": aggregate_evidence_carrier_pair_evaluations(raw_rows),
        "consensus_metrics": aggregate_evidence_carrier_pair_evaluations(
            consensus_rows
        ),
        "candidate_metrics": aggregate_evidence_carrier_pair_evaluations(
            candidate_rows
        ),
        "changed_sample_count": len(changed_rows),
        "changed_samples": changed_rows,
        "vetoed_positive_pair_count": vetoed_positive,
        "vetoed_negative_pair_count": vetoed_negative,
        "vetoed_sample_count": vetoed_samples,
        "forward_latency_ms": base._distribution(latency),
    }


def _load_checkpoint(
    path: Path, *, device: torch.device, implementation: Mapping[str, str]
) -> EvidenceCarrierPairNeighborhoodVeto:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": EvidenceCarrierPairNeighborhoodVeto.model_type,
        "parameter_count": PAIR_NEIGHBORHOOD_VETO_PARAMETER_COUNT,
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
        raise PairNeighborhoodVetoOOFTrainingError(
            f"neighborhood checkpoint differs: {mismatches}"
        )
    model = EvidenceCarrierPairNeighborhoodVeto()
    if checkpoint.get("model_config") != model.config_dict():
        raise PairNeighborhoodVetoOOFTrainingError(
            "neighborhood checkpoint config differs"
        )
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
            "neighbor_order": "signed_normal_offset",
            "group_type_in_forward": False,
            "rescue": "min(v0.9.19,v0.9.20)",
        },
    )
    base_index, base_metadata = base.load_oof_index(
        args.base_index, audit_path=args.base_audit
    )
    local_index, local_metadata = helper.load_local_index(args.local_index)
    neighborhood_index, neighborhood_metadata = load_neighborhood_index(
        args.neighborhood_index
    )
    frozen_implementation = implementation_sha256()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PairNeighborhoodVetoOOFTrainingError(
            "CUDA was requested but is unavailable"
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high")

    fold_reports = []
    all_raw_rows = []
    all_consensus_rows = []
    all_candidate_rows = []
    for eval_fold in range(3):
        train_records = []
        train_sidecars = []
        for fold in range(3):
            if fold == eval_fold:
                continue
            records, base_meta = base.load_oof_fold(base_index, fold)
            crops, local_meta = helper.load_local_fold(local_index, fold)
            neighborhoods, neighborhood_meta = load_neighborhood_fold(
                neighborhood_index, fold
            )
            records = helper.merge_pair_local_records(records, crops)
            train_records.extend(merge_neighborhood_records(records, neighborhoods))
            train_sidecars.append(
                {
                    "fold": fold,
                    "base": base_meta,
                    "pair_local": local_meta,
                    "pair_neighborhood": neighborhood_meta,
                }
            )
        train_records.sort(key=lambda record: record["identity"]["sample_key"])
        seed = TRAINING_SEEDS[eval_fold]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = EvidenceCarrierPairNeighborhoodVeto().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAINING_LEARNING_RATE,
            weight_decay=TRAINING_WEIGHT_DECAY,
        )
        metrics = train_steps(model, optimizer, train_records, seed=seed, device=device)
        if implementation_sha256() != frozen_implementation:
            raise PairNeighborhoodVetoOOFTrainingError(
                "implementation changed during training"
            )
        train_audit = audit_train_veto(model, train_records, device=device)
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
            "train_veto_audit": train_audit,
            "train_sidecars": train_sidecars,
            "base_index_metadata": base_metadata,
            "local_index_metadata": local_metadata,
            "neighborhood_index_metadata": neighborhood_metadata,
            "implementation_sha256": frozen_implementation,
            "group_type_in_forward": False,
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
        avg_path = Path(args.avg_checkpoint_dir).resolve() / f"fold{eval_fold}" / "final.pt"
        spatial_path = (
            Path(args.spatial_checkpoint_dir).resolve()
            / f"fold{eval_fold}"
            / "final.pt"
        )
        avg_model, avg_sha = consensus_helper._load_teacher(
            EvidenceCarrierPairLocalResidual, avg_path, device
        )
        spatial_model, spatial_sha = consensus_helper._load_teacher(
            EvidenceCarrierPairLocalSpatialResidual, spatial_path, device
        )
        eval_records, eval_base_meta = base.load_oof_fold(base_index, eval_fold)
        eval_crops, eval_local_meta = helper.load_local_fold(local_index, eval_fold)
        eval_neighborhoods, eval_neighborhood_meta = load_neighborhood_fold(
            neighborhood_index, eval_fold
        )
        eval_records = helper.merge_pair_local_records(eval_records, eval_crops)
        eval_records = merge_neighborhood_records(eval_records, eval_neighborhoods)
        evaluation = evaluate_oof_records(
            sealed, avg_model, spatial_model, eval_records, device=device
        )
        if file_sha256(checkpoint_path) != checkpoint_sha:
            raise PairNeighborhoodVetoOOFTrainingError(
                "checkpoint changed during evaluation"
            )
        gate = base.oof_gate(
            evaluation["raw_metrics"],
            evaluation["candidate_metrics"],
            expected_sample_count=2400,
        )
        all_raw_rows.extend(evaluation.pop("raw_rows"))
        all_consensus_rows.extend(evaluation.pop("consensus_rows"))
        all_candidate_rows.extend(evaluation.pop("candidate_rows"))
        fold_report = {
            "fold": eval_fold,
            "status": gate["status"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "teacher_checkpoints": {
                "avg": {"path": str(avg_path), "sha256": avg_sha},
                "spatial": {"path": str(spatial_path), "sha256": spatial_sha},
            },
            "training_metrics": metrics,
            "train_veto_audit": train_audit,
            "eval_base_sidecar": eval_base_meta,
            "eval_pair_local_sidecar": eval_local_meta,
            "eval_pair_neighborhood_sidecar": eval_neighborhood_meta,
            "gate": gate,
            **evaluation,
        }
        fold_report_path = checkpoint_path.parent / "oof.json"
        _write_json_no_overwrite(fold_report_path, fold_report)
        fold_report["report_path"] = str(fold_report_path)
        fold_report["report_sha256"] = file_sha256(fold_report_path)
        fold_reports.append(fold_report)
        del model, sealed, avg_model, spatial_model, optimizer, train_records, eval_records
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw_metrics = aggregate_evidence_carrier_pair_evaluations(all_raw_rows)
    consensus_metrics = aggregate_evidence_carrier_pair_evaluations(
        all_consensus_rows
    )
    candidate_metrics = aggregate_evidence_carrier_pair_evaluations(
        all_candidate_rows
    )
    aggregate_gate = base.oof_gate(
        raw_metrics, candidate_metrics, expected_sample_count=7200
    )
    all_folds_go = all(row["gate"]["status"] == "GO" for row in fold_reports)
    status = (
        STATUS_GO
        if aggregate_gate["status"] == "GO" and all_folds_go
        else STATUS_NO_GO
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "fold_reports": fold_reports,
        "aggregate_raw_metrics": raw_metrics,
        "aggregate_consensus_metrics": consensus_metrics,
        "aggregate_candidate_metrics": candidate_metrics,
        "aggregate_gate": aggregate_gate,
        "all_fold_gates_go": all_folds_go,
        "base_index_metadata": base_metadata,
        "local_index_metadata": local_metadata,
        "neighborhood_index_metadata": neighborhood_metadata,
        "implementation_sha256": frozen_implementation,
        "validation_read": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
    }
    _write_json_no_overwrite(report_path, report)
    summary = {
        "schema_version": "evidence-carrier-pair-neighborhood-veto-oof-summary-0.1",
        "status": status,
        "report_path": str(report_path),
        "report_sha256": file_sha256(report_path),
        "parameter_count": PAIR_NEIGHBORHOOD_VETO_PARAMETER_COUNT,
        "device": str(device),
        "aggregate_raw_metrics": raw_metrics,
        "aggregate_consensus_metrics": consensus_metrics,
        "aggregate_candidate_metrics": candidate_metrics,
        "aggregate_gate": aggregate_gate,
        "folds": [
            {
                "fold": row["fold"],
                "status": row["status"],
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "report_path": row["report_path"],
                "report_sha256": row["report_sha256"],
                "vetoed_positive_pair_count": row[
                    "vetoed_positive_pair_count"
                ],
                "vetoed_negative_pair_count": row[
                    "vetoed_negative_pair_count"
                ],
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
    parser.add_argument("--local-index", default=helper.DEFAULT_LOCAL_INDEX)
    parser.add_argument("--neighborhood-index", default=DEFAULT_NEIGHBORHOOD_INDEX)
    parser.add_argument(
        "--avg-checkpoint-dir", default=consensus_helper.AVG_CHECKPOINT_DIR
    )
    parser.add_argument(
        "--spatial-checkpoint-dir", default=consensus_helper.SPATIAL_CHECKPOINT_DIR
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    print(json.dumps(train(build_arg_parser().parse_args()), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "collate_neighborhood_records",
    "load_neighborhood_fold",
    "load_neighborhood_index",
    "merge_neighborhood_records",
    "train_steps",
]
