#!/usr/bin/env python3
"""Train v0.9.26 shared-anchor structured component matching with 3-fold OOF."""

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
from torch.nn import functional as F

import chemvector.evidence_carrier_component_structured as model_module
import chemvector.train_evidence_carrier_endpoint_oof as base
import chemvector.train_evidence_carrier_pair_local_oof as helper
from chemvector.build_evidence_carrier_component_matching_oof import (
    DEFAULT_INDEX as DEFAULT_COMPONENT_INDEX,
    INDEX_SCHEMA_VERSION as COMPONENT_INDEX_SCHEMA_VERSION,
    STATUS_COMPLETE as COMPONENT_STATUS_COMPLETE,
    file_sha256,
    validate_component_matching_sidecar,
)
from chemvector.evidence_carrier_component_matching import (
    apply_component_matching_selection,
    component_structured_loss,
    optimal_component_matchings,
)
from chemvector.evidence_carrier_component_structured import (
    COMPONENT_STRUCTURED_PARAMETER_COUNT,
    EvidenceCarrierComponentStructured,
)
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
    evaluate_evidence_carrier_pair_sample,
)


CHECKPOINT_SCHEMA_VERSION = "evidence-carrier-component-structured-oof-checkpoint-0.1"
REPORT_SCHEMA_VERSION = "evidence-carrier-component-structured-oof-report-0.1"
STATUS_GO = "FIT-OOF-COMPONENT-STRUCTURED-GO"
STATUS_NO_GO = "FIT-OOF-COMPONENT-STRUCTURED-NO-GO"
DEFAULT_OUTPUT_DIR = "outputs/checkpoints/chemvector-v0926-component-structured-oof"
DEFAULT_REPORT = "outputs/evaluations/chemvector-v0926-component-structured-oof/oof.json"
TRAINING_STEPS = 2_000
TRAINING_BATCH_SIZE = 32
TRAINING_SEEDS = (20_260_743, 20_260_744, 20_260_745)
TRAINING_LEARNING_RATE = 1e-3
TRAINING_WEIGHT_DECAY = 1e-4
TRAINING_MAX_GRAD_NORM = 1.0
STRUCTURED_LOSS_WEIGHT = 2.0
CLASSIFICATION_LOSS_WEIGHT = 1.0
POSITIVE_SAFETY_WEIGHT = 2.0
WORST_POSITIVE_WEIGHT = 4.0
ACTIVE_NEGATIVE_WEIGHT = 4.0
RESIDUAL_L2_WEIGHT = 0.001
ENERGY_MARGIN = 0.25
HARD_RECORD_PERIOD = 4


class ComponentStructuredOOFTrainingError(RuntimeError):
    """Raised when the structured component OOF protocol differs."""


def implementation_sha256() -> dict[str, str]:
    return {
        "trainer": file_sha256(Path(__file__).resolve()),
        "model": file_sha256(Path(model_module.__file__).resolve()),
        "pair_local_helper": file_sha256(Path(helper.__file__).resolve()),
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


def load_component_index(
    path: str | Path = DEFAULT_COMPONENT_INDEX,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ComponentStructuredOOFTrainingError("component index must be a mapping")
    payload = dict(payload)
    if (
        payload.get("schema_version") != COMPONENT_INDEX_SCHEMA_VERSION
        or payload.get("status") != COMPONENT_STATUS_COMPLETE
        or payload.get("counts", {}).get("sample_count") != 7200
        or payload.get("counts", {}).get("pair_count") != 368573
    ):
        raise ComponentStructuredOOFTrainingError("component index contract differs")
    folds = payload.get("folds")
    if not isinstance(folds, list) or len(folds) != 3:
        raise ComponentStructuredOOFTrainingError("component index folds differ")
    return payload, {"path": str(resolved), "sha256": file_sha256(resolved)}


def load_component_fold(
    index: Mapping[str, Any], fold: int
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    if fold not in (0, 1, 2):
        raise ValueError("fold must be 0, 1, or 2")
    reference = index["folds"][fold]
    path = Path(reference["path"]).resolve(strict=True)
    sha = file_sha256(path)
    if sha != reference["sha256"]:
        raise ComponentStructuredOOFTrainingError("component fold SHA differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    validate_component_matching_sidecar(payload, expected_fold=fold)
    mapping = {str(record["sample_key"]): record for record in payload["records"]}
    if len(mapping) != len(payload["records"]):
        raise ComponentStructuredOOFTrainingError("component sample keys overlap")
    return mapping, {"path": str(path), "sha256": sha}


def merge_component_records(
    records: Sequence[Mapping[str, Any]],
    component_by_key: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for record in records:
        key = str(record["identity"]["sample_key"])
        sidecar = component_by_key.get(key)
        if sidecar is None:
            raise ComponentStructuredOOFTrainingError(
                f"sample {key} lacks component matching features"
            )
        count = len(record["target_free"]["pair_tokens"])
        scores = sidecar["baseline_scores"]
        anchor_counts = sidecar["component_anchor_counts"]
        edge_counts = sidecar["component_edge_counts"]
        if scores.dtype != torch.float32 or scores.shape != (count,):
            raise ComponentStructuredOOFTrainingError(
                f"sample {key} baseline score shape differs"
            )
        if anchor_counts.dtype != torch.uint8 or anchor_counts.shape != (count,):
            raise ComponentStructuredOOFTrainingError(
                f"sample {key} component anchor count shape differs"
            )
        if edge_counts.dtype != torch.uint8 or edge_counts.shape != (count,):
            raise ComponentStructuredOOFTrainingError(
                f"sample {key} component edge count shape differs"
            )
        merged.append(
            {
                **record,
                "baseline_scores": scores,
                "component_anchor_counts": anchor_counts,
                "component_edge_counts": edge_counts,
                "components": sidecar["components"],
            }
        )
        seen.add(key)
    if seen != set(component_by_key):
        raise ComponentStructuredOOFTrainingError(
            "component fold contains unmatched samples"
        )
    return merged


def collate_component_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = helper.collate_pair_local_records(records)
    batch, capacity = result["pair_mask"].shape
    result["baseline_scores"] = torch.zeros((batch, capacity), dtype=torch.float32)
    result["component_anchor_counts"] = torch.zeros(
        (batch, capacity), dtype=torch.float32
    )
    result["component_edge_counts"] = torch.zeros(
        (batch, capacity), dtype=torch.float32
    )
    result["components"] = []
    for index, record in enumerate(records):
        count = int(result["pair_count"][index])
        result["baseline_scores"][index, :count] = torch.as_tensor(
            record["baseline_scores"], dtype=torch.float32
        )
        result["component_anchor_counts"][index, :count] = torch.as_tensor(
            record["component_anchor_counts"], dtype=torch.float32
        )
        result["component_edge_counts"][index, :count] = torch.as_tensor(
            record["component_edge_counts"], dtype=torch.float32
        )
        result["components"].append(record["components"])
    return result


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    components = batch["components"]
    tensors = {name: value for name, value in batch.items() if name != "components"}
    result: dict[str, Any] = helper._to_device(tensors, device)
    result["components"] = components
    return result


def _forward(
    model: EvidenceCarrierComponentStructured, batch: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    return model(
        batch["pair_tokens"],
        batch["pair_local_rasters"],
        batch["pair_indices"],
        batch["pair_mask"],
        batch["global_count_features"],
        batch["baseline_scores"],
        batch["component_anchor_counts"],
        batch["component_edge_counts"],
    )


def component_training_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
) -> dict[str, Any]:
    residual = output["edge_residual"]
    edge_energy = output["edge_energy"]
    labels = batch["pair_labels"]
    pair_mask = batch["pair_mask"]
    multi = pair_mask & (batch["component_anchor_counts"] > 2.0)
    zero = residual.sum() * 0.0
    if bool(multi.any()):
        classification_loss = F.binary_cross_entropy_with_logits(
            residual[multi].float(), labels[multi].float()
        )
        residual_l2_loss = residual[multi].float().square().mean()
    else:
        classification_loss = zero
        residual_l2_loss = zero

    structured_parts = []
    positive_parts = []
    worst_positive_parts = []
    active_negative_parts = []
    active_positive_count = active_negative_count = component_count = 0
    for image_index, components in enumerate(batch["components"]):
        for component in components:
            indices = torch.as_tensor(
                component["active_pair_indices"],
                dtype=torch.long,
                device=residual.device,
            )
            masks = torch.as_tensor(
                component["matching_masks"],
                dtype=torch.bool,
                device=residual.device,
            )
            component_labels = labels[image_index, indices]
            optimal = optimal_component_matchings(
                masks.detach().cpu(), component_labels.detach().cpu()
            ).to(residual.device)
            energies = edge_energy[image_index, indices]
            structured_parts.append(component_structured_loss(energies, masks, optimal))
            positive = component_labels == 1
            negative = component_labels == 0
            if bool(positive.any()):
                positive_energy = energies[positive]
                positive_parts.append(
                    F.relu(ENERGY_MARGIN - positive_energy).square().mean()
                )
                worst_positive_parts.append(
                    F.relu(ENERGY_MARGIN - positive_energy.min()).square()
                )
                active_positive_count += int(positive.sum())
            if bool(negative.any()):
                negative_energy = energies[negative]
                active_negative_parts.append(
                    F.relu(negative_energy + ENERGY_MARGIN).square().mean()
                )
                active_negative_count += int(negative.sum())
            component_count += 1

    def mean_or_zero(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).mean() if values else zero

    structured_loss = mean_or_zero(structured_parts)
    positive_safety_loss = mean_or_zero(positive_parts)
    worst_positive_loss = mean_or_zero(worst_positive_parts)
    active_negative_loss = mean_or_zero(active_negative_parts)
    loss = (
        STRUCTURED_LOSS_WEIGHT * structured_loss
        + CLASSIFICATION_LOSS_WEIGHT * classification_loss
        + POSITIVE_SAFETY_WEIGHT * positive_safety_loss
        + WORST_POSITIVE_WEIGHT * worst_positive_loss
        + ACTIVE_NEGATIVE_WEIGHT * active_negative_loss
        + RESIDUAL_L2_WEIGHT * residual_l2_loss
    )
    return {
        "loss": loss,
        "structured_loss": structured_loss,
        "classification_loss": classification_loss,
        "positive_safety_loss": positive_safety_loss,
        "worst_positive_loss": worst_positive_loss,
        "active_negative_loss": active_negative_loss,
        "residual_l2_loss": residual_l2_loss,
        "component_count": component_count,
        "active_positive_count": active_positive_count,
        "active_negative_count": active_negative_count,
    }


def _hard_record_indices(records: Sequence[Mapping[str, Any]]) -> list[int]:
    result = []
    for record_index, record in enumerate(records):
        labels = torch.as_tensor(record["target_aware"]["pair_labels"], dtype=torch.long)
        if any(
            bool((labels[component["active_pair_indices"].long()] == 0).any())
            for component in record["components"]
        ):
            result.append(record_index)
    return result


def train_steps(
    model: EvidenceCarrierComponentStructured,
    optimizer: torch.optim.Optimizer,
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if len(records) != 4800:
        raise ValueError("component trainer requires exactly 4,800 records")
    hard_indices = _hard_record_indices(records)
    if not hard_indices:
        raise ComponentStructuredOOFTrainingError(
            "training folds contain no active negative component"
        )
    names = (
        "loss",
        "structured_loss",
        "classification_loss",
        "positive_safety_loss",
        "worst_positive_loss",
        "active_negative_loss",
        "residual_l2_loss",
    )
    values = {name: [] for name in names}
    gradients = []
    hard_injections = 0
    model.train()
    batches = base.deterministic_batch_indices(
        len(records), steps=TRAINING_STEPS, batch_size=TRAINING_BATCH_SIZE, seed=seed
    )
    for step, source_indices in enumerate(batches):
        indices = list(source_indices)
        if step % HARD_RECORD_PERIOD == 0:
            indices[0] = hard_indices[(step // HARD_RECORD_PERIOD) % len(hard_indices)]
            hard_injections += 1
        batch = _to_device(
            collate_component_records([records[index] for index in indices]), device
        )
        output = _forward(model, batch)
        losses = component_training_loss(output, batch)
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
        "hard_record_count": len(hard_indices),
        "hard_record_period": HARD_RECORD_PERIOD,
        "hard_record_injections": hard_injections,
        **{f"mean_{name}": float(np.mean(items)) for name, items in values.items()},
        **{f"final_{name}": items[-1] for name, items in values.items()},
        "maximum_preclip_gradient_norm": max(gradients),
    }


@torch.inference_mode()
def evaluate_oof_records(
    model: EvidenceCarrierComponentStructured,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    model.eval()
    baseline_rows = []
    candidate_rows = []
    changed_rows = []
    removed_positive = removed_negative = changed_samples = 0
    latency = []
    for start in range(0, len(records), TRAINING_BATCH_SIZE):
        current = records[start : start + TRAINING_BATCH_SIZE]
        batch = _to_device(collate_component_records(current), device)
        started = perf_counter()
        output = _forward(model, batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (perf_counter() - started) * 1000.0
        latency.extend([elapsed / len(current)] * len(current))
        for image_index, record in enumerate(current):
            count = int(batch["pair_count"][image_index])
            baseline_scores = batch["baseline_scores"][image_index, :count]
            residual = output["edge_residual"][image_index, :count]
            candidate_scores = apply_component_matching_selection(
                baseline_scores,
                residual,
                record["components"],
                threshold=threshold,
            )
            removed = (baseline_scores > threshold) & (
                candidate_scores <= threshold
            )
            labels = torch.as_tensor(
                record["target_aware"]["pair_labels"],
                dtype=torch.long,
                device=device,
            )
            removed_positive += int((removed & (labels == 1)).sum().cpu())
            removed_negative += int((removed & (labels == 0)).sum().cpu())
            changed_samples += int(bool(removed.any()))
            target = base._evaluation_target(record)
            baseline_eval = evaluate_evidence_carrier_pair_sample(
                target,
                baseline_scores.detach().cpu(),
                threshold=threshold,
                verify_permutation=True,
            )
            candidate_eval = evaluate_evidence_carrier_pair_sample(
                target,
                candidate_scores.detach().cpu(),
                threshold=threshold,
                verify_permutation=True,
            )
            baseline_rows.append(baseline_eval)
            candidate_rows.append(candidate_eval)
            if baseline_eval["selected_anchor_pairs"] != candidate_eval[
                "selected_anchor_pairs"
            ]:
                changed_rows.append(
                    {
                        "identity": dict(record["identity"]),
                        "baseline": baseline_eval,
                        "candidate": candidate_eval,
                        "removed_candidate_indices": torch.where(removed)[0]
                        .detach()
                        .cpu()
                        .tolist(),
                    }
                )
    return {
        "baseline_rows": baseline_rows,
        "candidate_rows": candidate_rows,
        "baseline_metrics": aggregate_evidence_carrier_pair_evaluations(
            baseline_rows
        ),
        "candidate_metrics": aggregate_evidence_carrier_pair_evaluations(
            candidate_rows
        ),
        "changed_sample_count": len(changed_rows),
        "changed_samples": changed_rows,
        "removed_positive_pair_count": removed_positive,
        "removed_negative_pair_count": removed_negative,
        "removed_sample_count": changed_samples,
        "forward_latency_ms": base._distribution(latency),
    }


def _load_checkpoint(
    path: Path, *, device: torch.device, implementation: Mapping[str, str]
) -> EvidenceCarrierComponentStructured:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": EvidenceCarrierComponentStructured.model_type,
        "parameter_count": COMPONENT_STRUCTURED_PARAMETER_COUNT,
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
        raise ComponentStructuredOOFTrainingError(
            f"component checkpoint differs: {mismatches}"
        )
    model = EvidenceCarrierComponentStructured()
    if checkpoint.get("model_config") != model.config_dict():
        raise ComponentStructuredOOFTrainingError("component checkpoint config differs")
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
            "decision_unit": "enumerated_multi_anchor_component_matching",
            "isolated_pair_policy": "v0.9.23_passthrough",
            "score_increase_possible": False,
        },
    )
    base_index, base_metadata = base.load_oof_index(
        args.base_index, audit_path=args.base_audit
    )
    local_index, local_metadata = helper.load_local_index(args.local_index)
    component_index, component_metadata = load_component_index(args.component_index)
    threshold = float(component_index["threshold"])
    frozen_implementation = implementation_sha256()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ComponentStructuredOOFTrainingError("CUDA was requested but is unavailable")
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
            records, base_meta = base.load_oof_fold(base_index, fold)
            crops, local_meta = helper.load_local_fold(local_index, fold)
            components, component_meta = load_component_fold(component_index, fold)
            records = helper.merge_pair_local_records(records, crops)
            train_records.extend(merge_component_records(records, components))
            train_sidecars.append(
                {
                    "fold": fold,
                    "base": base_meta,
                    "pair_local": local_meta,
                    "component": component_meta,
                }
            )
        train_records.sort(key=lambda record: record["identity"]["sample_key"])
        seed = TRAINING_SEEDS[eval_fold]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = EvidenceCarrierComponentStructured().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TRAINING_LEARNING_RATE,
            weight_decay=TRAINING_WEIGHT_DECAY,
        )
        metrics = train_steps(model, optimizer, train_records, seed=seed, device=device)
        if implementation_sha256() != frozen_implementation:
            raise ComponentStructuredOOFTrainingError(
                "implementation changed during training"
            )
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
            "threshold": threshold,
            "training_metrics": metrics,
            "train_sidecars": train_sidecars,
            "base_index_metadata": base_metadata,
            "local_index_metadata": local_metadata,
            "component_index_metadata": component_metadata,
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
        eval_crops, eval_local_meta = helper.load_local_fold(local_index, eval_fold)
        eval_components, eval_component_meta = load_component_fold(
            component_index, eval_fold
        )
        eval_records = helper.merge_pair_local_records(eval_records, eval_crops)
        eval_records = merge_component_records(eval_records, eval_components)
        evaluation = evaluate_oof_records(
            sealed, eval_records, device=device, threshold=threshold
        )
        if file_sha256(checkpoint_path) != checkpoint_sha:
            raise ComponentStructuredOOFTrainingError(
                "checkpoint changed during evaluation"
            )
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
            "eval_base_sidecar": eval_base_meta,
            "eval_pair_local_sidecar": eval_local_meta,
            "eval_component_sidecar": eval_component_meta,
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

    baseline_metrics = aggregate_evidence_carrier_pair_evaluations(all_baseline_rows)
    candidate_metrics = aggregate_evidence_carrier_pair_evaluations(all_candidate_rows)
    aggregate_gate = base.oof_gate(
        baseline_metrics, candidate_metrics, expected_sample_count=7200
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
        "aggregate_baseline_metrics": baseline_metrics,
        "aggregate_candidate_metrics": candidate_metrics,
        "aggregate_gate": aggregate_gate,
        "all_fold_gates_go": all_folds_go,
        "threshold": threshold,
        "base_index_metadata": base_metadata,
        "local_index_metadata": local_metadata,
        "component_index_metadata": component_metadata,
        "implementation_sha256": frozen_implementation,
        "validation_read": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
    }
    _write_json_no_overwrite(report_path, report)
    summary = {
        "schema_version": "evidence-carrier-component-structured-oof-summary-0.1",
        "status": status,
        "report_path": str(report_path),
        "report_sha256": file_sha256(report_path),
        "parameter_count": COMPONENT_STRUCTURED_PARAMETER_COUNT,
        "device": str(device),
        "aggregate_baseline_metrics": baseline_metrics,
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
                "removed_positive_pair_count": row[
                    "removed_positive_pair_count"
                ],
                "removed_negative_pair_count": row[
                    "removed_negative_pair_count"
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
    parser.add_argument("--component-index", default=DEFAULT_COMPONENT_INDEX)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    print(json.dumps(train(build_arg_parser().parse_args()), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "collate_component_records",
    "component_training_loss",
    "load_component_fold",
    "load_component_index",
    "merge_component_records",
    "train_steps",
]
