#!/usr/bin/env python3
"""Train v0.9.27 score-blind counterfactual component matching with OOF."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

import chemvector.evidence_carrier_component_score_blind as model_module
import chemvector.train_evidence_carrier_component_structured_oof as previous
import chemvector.train_evidence_carrier_endpoint_oof as base
import chemvector.train_evidence_carrier_pair_local_oof as helper
from chemvector.build_evidence_carrier_pair_local_oof import file_sha256
from chemvector.evidence_carrier_component_matching import (
    apply_component_matching_selection,
    component_structured_loss,
    enumerate_component_matchings,
    optimal_component_matchings,
)
from chemvector.evidence_carrier_component_score_blind import (
    COMPONENT_SCORE_BLIND_PARAMETER_COUNT,
    EvidenceCarrierComponentScoreBlind,
)
from chemvector.evidence_carrier_pair_components import pair_component_metadata
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
    evaluate_evidence_carrier_pair_sample,
)


CHECKPOINT_SCHEMA_VERSION = "evidence-carrier-component-score-blind-oof-checkpoint-0.1"
REPORT_SCHEMA_VERSION = "evidence-carrier-component-score-blind-oof-report-0.1"
STATUS_GO = "FIT-OOF-COMPONENT-SCORE-BLIND-GO"
STATUS_NO_GO = "FIT-OOF-COMPONENT-SCORE-BLIND-NO-GO"
DEFAULT_OUTPUT_DIR = "outputs/checkpoints/chemvector-v0927-component-score-blind-oof"
DEFAULT_REPORT = "outputs/evaluations/chemvector-v0927-component-score-blind-oof/oof.json"
TRAINING_STEPS = 2_000
TRAINING_BATCH_SIZE = 32
TRAINING_SEEDS = (20_260_746, 20_260_747, 20_260_748)
TRAINING_LEARNING_RATE = 1e-3
TRAINING_WEIGHT_DECAY = 1e-4
TRAINING_MAX_GRAD_NORM = 1.0
COUNTERFACTUAL_BASE_ENERGY = 0.5
ENERGY_MARGIN = 0.25
COUNTERFACTUAL_STRUCTURED_WEIGHT = 2.0
REAL_STRUCTURED_WEIGHT = 1.0
CLASSIFICATION_WEIGHT = 1.0
POSITIVE_SAFETY_WEIGHT = 4.0
WORST_POSITIVE_WEIGHT = 8.0
NEGATIVE_REJECTION_WEIGHT = 4.0
RESIDUAL_L2_WEIGHT = 0.001


class ComponentScoreBlindOOFTrainingError(RuntimeError):
    """Raised when the score-blind OOF protocol differs."""


def implementation_sha256() -> dict[str, str]:
    return {
        "trainer": file_sha256(Path(__file__).resolve()),
        "model": file_sha256(Path(model_module.__file__).resolve()),
        "previous_data_helper": file_sha256(Path(previous.__file__).resolve()),
    }


def prepare_counterfactual_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    prepared = []
    for record in records:
        pairs = torch.as_tensor(record["target_free"]["pair_indices"], dtype=torch.long)
        metadata = pair_component_metadata(pairs)
        components = []
        multi_ids = metadata["component_ids"][
            metadata["multi_anchor_component"]
        ].unique(sorted=True)
        for component_id in multi_ids.tolist():
            indices = torch.where(metadata["component_ids"] == int(component_id))[0]
            components.append(
                {
                    "component_id": int(component_id),
                    "pair_indices": indices.to(torch.int32).contiguous(),
                    "matching_masks": enumerate_component_matchings(pairs[indices]),
                }
            )
        prepared.append({**record, "counterfactual_components": components})
    return prepared


def collate_score_blind_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = previous.collate_component_records(records)
    result["counterfactual_components"] = [
        record["counterfactual_components"] for record in records
    ]
    return result


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    lists = {
        "components": batch["components"],
        "counterfactual_components": batch["counterfactual_components"],
    }
    tensors = {
        name: value
        for name, value in batch.items()
        if name not in {"components", "counterfactual_components"}
    }
    result: dict[str, Any] = helper._to_device(tensors, device)
    result.update(lists)
    return result


def _forward(
    model: EvidenceCarrierComponentScoreBlind, batch: Mapping[str, Any]
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


def score_blind_training_loss(
    output: Mapping[str, torch.Tensor], batch: Mapping[str, Any]
) -> dict[str, Any]:
    residual = output["edge_residual"]
    real_energy = output["edge_energy"]
    labels = batch["pair_labels"]
    multi = batch["pair_mask"] & (batch["component_anchor_counts"] > 2.0)
    zero = residual.sum() * 0.0
    classification_loss = (
        F.binary_cross_entropy_with_logits(residual[multi].float(), labels[multi].float())
        if bool(multi.any())
        else zero
    )
    residual_l2_loss = residual[multi].float().square().mean() if bool(multi.any()) else zero
    real_parts = []
    counterfactual_parts = []
    positive_parts = []
    worst_positive_parts = []
    negative_parts = []
    real_component_count = counterfactual_component_count = 0
    counterfactual_positive_count = counterfactual_negative_count = 0

    for image_index, components in enumerate(batch["components"]):
        for component in components:
            indices = torch.as_tensor(
                component["active_pair_indices"], dtype=torch.long, device=residual.device
            )
            masks = torch.as_tensor(
                component["matching_masks"], dtype=torch.bool, device=residual.device
            )
            component_labels = labels[image_index, indices]
            optimal = optimal_component_matchings(
                masks.detach().cpu(), component_labels.detach().cpu()
            ).to(residual.device)
            real_parts.append(
                component_structured_loss(real_energy[image_index, indices], masks, optimal)
            )
            real_component_count += 1

    for image_index, components in enumerate(batch["counterfactual_components"]):
        for component in components:
            indices = torch.as_tensor(
                component["pair_indices"], dtype=torch.long, device=residual.device
            )
            masks = torch.as_tensor(
                component["matching_masks"], dtype=torch.bool, device=residual.device
            )
            component_labels = labels[image_index, indices]
            optimal = optimal_component_matchings(
                masks.detach().cpu(), component_labels.detach().cpu()
            ).to(residual.device)
            energies = COUNTERFACTUAL_BASE_ENERGY + residual[image_index, indices]
            counterfactual_parts.append(
                component_structured_loss(energies, masks, optimal)
            )
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
                counterfactual_positive_count += int(positive.sum())
            if bool(negative.any()):
                negative_energy = energies[negative]
                negative_parts.append(
                    F.relu(negative_energy + ENERGY_MARGIN).square().mean()
                )
                counterfactual_negative_count += int(negative.sum())
            counterfactual_component_count += 1

    def mean_or_zero(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).mean() if values else zero

    real_structured_loss = mean_or_zero(real_parts)
    counterfactual_structured_loss = mean_or_zero(counterfactual_parts)
    positive_safety_loss = mean_or_zero(positive_parts)
    worst_positive_loss = mean_or_zero(worst_positive_parts)
    negative_rejection_loss = mean_or_zero(negative_parts)
    loss = (
        COUNTERFACTUAL_STRUCTURED_WEIGHT * counterfactual_structured_loss
        + REAL_STRUCTURED_WEIGHT * real_structured_loss
        + CLASSIFICATION_WEIGHT * classification_loss
        + POSITIVE_SAFETY_WEIGHT * positive_safety_loss
        + WORST_POSITIVE_WEIGHT * worst_positive_loss
        + NEGATIVE_REJECTION_WEIGHT * negative_rejection_loss
        + RESIDUAL_L2_WEIGHT * residual_l2_loss
    )
    return {
        "loss": loss,
        "counterfactual_structured_loss": counterfactual_structured_loss,
        "real_structured_loss": real_structured_loss,
        "classification_loss": classification_loss,
        "positive_safety_loss": positive_safety_loss,
        "worst_positive_loss": worst_positive_loss,
        "negative_rejection_loss": negative_rejection_loss,
        "residual_l2_loss": residual_l2_loss,
        "real_component_count": real_component_count,
        "counterfactual_component_count": counterfactual_component_count,
        "counterfactual_positive_count": counterfactual_positive_count,
        "counterfactual_negative_count": counterfactual_negative_count,
    }


def train_steps(
    model: EvidenceCarrierComponentScoreBlind,
    optimizer: torch.optim.Optimizer,
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if len(records) != 4800:
        raise ValueError("score-blind trainer requires exactly 4,800 records")
    names = (
        "loss",
        "counterfactual_structured_loss",
        "real_structured_loss",
        "classification_loss",
        "positive_safety_loss",
        "worst_positive_loss",
        "negative_rejection_loss",
        "residual_l2_loss",
    )
    values = {name: [] for name in names}
    gradients = []
    model.train()
    for indices in base.deterministic_batch_indices(
        len(records), steps=TRAINING_STEPS, batch_size=TRAINING_BATCH_SIZE, seed=seed
    ):
        batch = _to_device(
            collate_score_blind_records([records[index] for index in indices]), device
        )
        output = _forward(model, batch)
        losses = score_blind_training_loss(output, batch)
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
        "counterfactual_base_energy": COUNTERFACTUAL_BASE_ENERGY,
        **{f"mean_{name}": float(np.mean(items)) for name, items in values.items()},
        **{f"final_{name}": items[-1] for name, items in values.items()},
        "maximum_preclip_gradient_norm": max(gradients),
    }


@torch.inference_mode()
def evaluate_oof_records(
    model: EvidenceCarrierComponentScoreBlind,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    threshold: float,
) -> dict[str, Any]:
    model.eval()
    baseline_rows = []
    candidate_rows = []
    changed_rows = []
    removed_positive = removed_negative = removed_samples = 0
    latency = []
    for start in range(0, len(records), TRAINING_BATCH_SIZE):
        current = records[start : start + TRAINING_BATCH_SIZE]
        batch = _to_device(collate_score_blind_records(current), device)
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
            removed = (baseline_scores > threshold) & (candidate_scores <= threshold)
            labels = torch.as_tensor(
                record["target_aware"]["pair_labels"], dtype=torch.long, device=device
            )
            removed_positive += int((removed & (labels == 1)).sum().cpu())
            removed_negative += int((removed & (labels == 0)).sum().cpu())
            removed_samples += int(bool(removed.any()))
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
        "baseline_metrics": aggregate_evidence_carrier_pair_evaluations(baseline_rows),
        "candidate_metrics": aggregate_evidence_carrier_pair_evaluations(candidate_rows),
        "changed_sample_count": len(changed_rows),
        "changed_samples": changed_rows,
        "removed_positive_pair_count": removed_positive,
        "removed_negative_pair_count": removed_negative,
        "removed_sample_count": removed_samples,
        "forward_latency_ms": base._distribution(latency),
    }


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> str:
    return previous._save_checkpoint(path, payload)


def _load_checkpoint(
    path: Path, *, device: torch.device, implementation: Mapping[str, str]
) -> EvidenceCarrierComponentScoreBlind:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": EvidenceCarrierComponentScoreBlind.model_type,
        "parameter_count": COMPONENT_SCORE_BLIND_PARAMETER_COUNT,
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
        raise ComponentScoreBlindOOFTrainingError(
            f"score-blind checkpoint differs: {mismatches}"
        )
    model = EvidenceCarrierComponentScoreBlind()
    if checkpoint.get("model_config") != model.config_dict():
        raise ComponentScoreBlindOOFTrainingError("score-blind checkpoint config differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def branch_gate(
    baseline_metrics: Mapping[str, Any], candidate_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    checks = {
        "cross_at_most_8": candidate_metrics["selected_cross_target_pairs"] <= 8,
        "m2_recall_at_least_0_995": candidate_metrics["multiplicity_2_recall"] >= 0.995,
        "pairable_exact_not_below_baseline": candidate_metrics["pairable_sample_exact"]
        >= baseline_metrics["pairable_sample_exact"],
        "precision_at_least_0_999": candidate_metrics["selected_pair_precision"] >= 0.999,
        "parallel_2_not_worse": candidate_metrics["parallel_2_violated_group_count"]
        <= baseline_metrics["parallel_2_violated_group_count"],
    }
    return {"status": "GO" if all(checks.values()) else "NO-GO", "checks": checks}


def train(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if report_path.exists():
        raise FileExistsError(report_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    previous._write_json_no_overwrite(
        output_dir / "run.lock",
        {
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "steps_per_fold": TRAINING_STEPS,
            "score_blind_forward": True,
            "counterfactual_base_energy": COUNTERFACTUAL_BASE_ENERGY,
            "branch_gate": "cross<=8_and_no_exact_regression",
        },
    )
    base_index, base_metadata = base.load_oof_index(
        args.base_index, audit_path=args.base_audit
    )
    local_index, local_metadata = helper.load_local_index(args.local_index)
    component_index, component_metadata = previous.load_component_index(
        args.component_index
    )
    threshold = float(component_index["threshold"])
    frozen_implementation = implementation_sha256()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ComponentScoreBlindOOFTrainingError("CUDA was requested but is unavailable")
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
            components, component_meta = previous.load_component_fold(
                component_index, fold
            )
            records = helper.merge_pair_local_records(records, crops)
            records = previous.merge_component_records(records, components)
            train_records.extend(prepare_counterfactual_records(records))
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
        model = EvidenceCarrierComponentScoreBlind().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=TRAINING_LEARNING_RATE, weight_decay=TRAINING_WEIGHT_DECAY
        )
        metrics = train_steps(model, optimizer, train_records, seed=seed, device=device)
        if implementation_sha256() != frozen_implementation:
            raise ComponentScoreBlindOOFTrainingError(
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
        eval_components, eval_component_meta = previous.load_component_fold(
            component_index, eval_fold
        )
        eval_records = helper.merge_pair_local_records(eval_records, eval_crops)
        eval_records = previous.merge_component_records(eval_records, eval_components)
        eval_records = prepare_counterfactual_records(eval_records)
        evaluation = evaluate_oof_records(
            sealed, eval_records, device=device, threshold=threshold
        )
        standard_gate = base.oof_gate(
            evaluation["baseline_metrics"],
            evaluation["candidate_metrics"],
            expected_sample_count=2400,
        )
        decision_gate = branch_gate(
            evaluation["baseline_metrics"], evaluation["candidate_metrics"]
        )
        all_baseline_rows.extend(evaluation.pop("baseline_rows"))
        all_candidate_rows.extend(evaluation.pop("candidate_rows"))
        fold_report = {
            "fold": eval_fold,
            "status": decision_gate["status"],
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "training_metrics": metrics,
            "eval_base_sidecar": eval_base_meta,
            "eval_pair_local_sidecar": eval_local_meta,
            "eval_component_sidecar": eval_component_meta,
            "standard_gate": standard_gate,
            "branch_gate": decision_gate,
            **evaluation,
        }
        fold_report_path = checkpoint_path.parent / "oof.json"
        previous._write_json_no_overwrite(fold_report_path, fold_report)
        fold_report["report_path"] = str(fold_report_path)
        fold_report["report_sha256"] = file_sha256(fold_report_path)
        fold_reports.append(fold_report)
        del model, sealed, optimizer, train_records, eval_records
        if device.type == "cuda":
            torch.cuda.empty_cache()

    baseline_metrics = aggregate_evidence_carrier_pair_evaluations(all_baseline_rows)
    candidate_metrics = aggregate_evidence_carrier_pair_evaluations(all_candidate_rows)
    standard_gate = base.oof_gate(
        baseline_metrics, candidate_metrics, expected_sample_count=7200
    )
    decision_gate = branch_gate(baseline_metrics, candidate_metrics)
    status = STATUS_GO if decision_gate["status"] == "GO" else STATUS_NO_GO
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "fold_reports": fold_reports,
        "aggregate_baseline_metrics": baseline_metrics,
        "aggregate_candidate_metrics": candidate_metrics,
        "aggregate_standard_gate": standard_gate,
        "aggregate_branch_gate": decision_gate,
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
    previous._write_json_no_overwrite(report_path, report)
    summary = {
        "schema_version": "evidence-carrier-component-score-blind-oof-summary-0.1",
        "status": status,
        "report_path": str(report_path),
        "report_sha256": file_sha256(report_path),
        "parameter_count": COMPONENT_SCORE_BLIND_PARAMETER_COUNT,
        "device": str(device),
        "aggregate_baseline_metrics": baseline_metrics,
        "aggregate_candidate_metrics": candidate_metrics,
        "aggregate_standard_gate": standard_gate,
        "aggregate_branch_gate": decision_gate,
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
    previous._write_json_no_overwrite(output_dir / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-index", default=base.DEFAULT_OUTPUT_INDEX)
    parser.add_argument("--base-audit", default=base.DEFAULT_AUDIT)
    parser.add_argument("--local-index", default=helper.DEFAULT_LOCAL_INDEX)
    parser.add_argument("--component-index", default=previous.DEFAULT_COMPONENT_INDEX)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    print(json.dumps(train(build_arg_parser().parse_args()), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "branch_gate",
    "collate_score_blind_records",
    "prepare_counterfactual_records",
    "score_blind_training_loss",
    "train_steps",
]
