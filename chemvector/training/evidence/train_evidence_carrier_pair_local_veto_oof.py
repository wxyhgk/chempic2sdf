#!/usr/bin/env python3
"""Train and evaluate the v0.9.21 conservative pair veto with 3-fold OOF."""

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

import chemvector.evidence_carrier_pair_local_veto as model_module
import chemvector.train_evidence_carrier_endpoint_oof as base
import chemvector.train_evidence_carrier_pair_local_oof as helper
from chemvector.build_evidence_carrier_pair_local_oof import file_sha256
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
    evaluate_evidence_carrier_pair_sample,
)
from chemvector.evidence_carrier_pair_local_veto import (
    PAIR_LOCAL_VETO_PARAMETER_COUNT,
    EvidenceCarrierPairLocalVeto,
    pair_local_veto_loss,
)
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
)


CHECKPOINT_SCHEMA_VERSION = "evidence-carrier-pair-local-veto-oof-checkpoint-0.1"
REPORT_SCHEMA_VERSION = "evidence-carrier-pair-local-veto-oof-report-0.1"
STATUS_GO = "FIT-OOF-PAIR-LOCAL-VETO-GO"
STATUS_NO_GO = "FIT-OOF-PAIR-LOCAL-VETO-NO-GO"
DEFAULT_OUTPUT_DIR = "outputs/checkpoints/chemvector-v0921-pair-local-veto-oof"
DEFAULT_REPORT = "outputs/evaluations/chemvector-v0921-pair-local-veto-oof/oof.json"
TRAINING_STEPS = 2_000
TRAINING_BATCH_SIZE = 32
TRAINING_SEEDS = (20_260_731, 20_260_732, 20_260_733)
TRAINING_LEARNING_RATE = 1e-3
TRAINING_WEIGHT_DECAY = 1e-4
TRAINING_MAX_GRAD_NORM = 1.0


class PairLocalVetoOOFTrainingError(RuntimeError):
    """Raised when the frozen veto OOF protocol differs."""


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


def _forward(model: EvidenceCarrierPairLocalVeto, batch: Mapping[str, torch.Tensor]):
    return model(
        batch["pair_tokens"],
        batch["pair_local_rasters"],
        batch["pair_indices"],
        batch["pair_mask"],
        batch["global_count_features"],
    )


def train_steps(
    model: EvidenceCarrierPairLocalVeto,
    optimizer: torch.optim.Optimizer,
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if len(records) != 4800:
        raise ValueError("veto trainer requires exactly 4,800 records")
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
        batch = helper._to_device(
            helper.collate_pair_local_records([records[index] for index in indices]),
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
    model: EvidenceCarrierPairLocalVeto,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    logit_sum = logit_square_sum = 0.0
    pair_count = positive_veto = negative_veto = missed_active_negative = 0
    for start in range(0, len(records), TRAINING_BATCH_SIZE):
        batch = helper._to_device(
            helper.collate_pair_local_records(records[start : start + TRAINING_BATCH_SIZE]),
            device,
        )
        output = _forward(model, batch)
        active = batch["pair_mask"]
        logits = output["veto_logits"][active].double()
        if not bool(torch.isfinite(logits).all()):
            raise PairLocalVetoOOFTrainingError("train veto logits are non-finite")
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
        "status": "TRAIN-VETO-AUDIT-PASSED",
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
    model: EvidenceCarrierPairLocalVeto,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    baseline_rows = []
    candidate_rows = []
    changed_rows = []
    latency = []
    vetoed_positive = vetoed_negative = vetoed_samples = 0
    for start in range(0, len(records), TRAINING_BATCH_SIZE):
        current = records[start : start + TRAINING_BATCH_SIZE]
        batch = helper._to_device(helper.collate_pair_local_records(current), device)
        started = perf_counter()
        output = _forward(model, batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (perf_counter() - started) * 1000.0
        latency.extend([elapsed / len(current)] * len(current))
        for image_index, record in enumerate(current):
            count = int(batch["pair_count"][image_index])
            raw = output["raw_gain"][image_index, :count].detach().cpu()
            corrected = output["corrected_gain"][image_index, :count].detach().cpu()
            veto = output["veto_mask"][image_index, :count].detach().cpu()
            labels = torch.as_tensor(
                record["target_aware"]["pair_labels"], dtype=torch.long
            )
            vetoed_positive += int((veto & (labels == 1)).sum())
            vetoed_negative += int((veto & (labels == 0)).sum())
            vetoed_samples += int(bool(veto.any()))
            if bool((corrected > raw).any()):
                raise PairLocalVetoOOFTrainingError("veto increased a candidate score")
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
            if baseline["selected_anchor_pairs"] != candidate["selected_anchor_pairs"]:
                changed_rows.append(
                    {
                        "identity": dict(record["identity"]),
                        "baseline": baseline,
                        "candidate": candidate,
                        "vetoed_candidate_indices": torch.where(veto)[0].tolist(),
                    }
                )
    return {
        "baseline_rows": baseline_rows,
        "candidate_rows": candidate_rows,
        "baseline_metrics": aggregate_evidence_carrier_pair_evaluations(baseline_rows),
        "candidate_metrics": aggregate_evidence_carrier_pair_evaluations(candidate_rows),
        "changed_sample_count": len(changed_rows),
        "changed_samples": changed_rows,
        "vetoed_positive_pair_count": vetoed_positive,
        "vetoed_negative_pair_count": vetoed_negative,
        "vetoed_sample_count": vetoed_samples,
        "forward_latency_ms": base._distribution(latency),
    }


def _load_checkpoint(
    path: Path, *, device: torch.device, implementation: Mapping[str, str]
) -> EvidenceCarrierPairLocalVeto:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": EvidenceCarrierPairLocalVeto.model_type,
        "parameter_count": PAIR_LOCAL_VETO_PARAMETER_COUNT,
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
        raise PairLocalVetoOOFTrainingError(f"veto checkpoint differs: {mismatches}")
    model = EvidenceCarrierPairLocalVeto()
    if checkpoint.get("model_config") != model.config_dict():
        raise PairLocalVetoOOFTrainingError("veto checkpoint config differs")
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
            "score_increase_possible": False,
        },
    )
    base_index, base_metadata = base.load_oof_index(args.base_index, audit_path=args.base_audit)
    local_index, local_metadata = helper.load_local_index(args.local_index)
    frozen_implementation = implementation_sha256()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PairLocalVetoOOFTrainingError("CUDA was requested but is unavailable")
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
            train_records.extend(helper.merge_pair_local_records(records, crops))
            train_sidecars.append(
                {"fold": fold, "base": base_meta, "pair_local": local_meta}
            )
        train_records.sort(key=lambda record: record["identity"]["sample_key"])
        seed = TRAINING_SEEDS[eval_fold]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = EvidenceCarrierPairLocalVeto().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=TRAINING_LEARNING_RATE, weight_decay=TRAINING_WEIGHT_DECAY
        )
        metrics = train_steps(model, optimizer, train_records, seed=seed, device=device)
        if implementation_sha256() != frozen_implementation:
            raise PairLocalVetoOOFTrainingError("implementation changed during training")
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
        eval_records = helper.merge_pair_local_records(eval_records, eval_crops)
        evaluation = evaluate_oof_records(sealed, eval_records, device=device)
        if file_sha256(checkpoint_path) != checkpoint_sha:
            raise PairLocalVetoOOFTrainingError("checkpoint changed during evaluation")
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
            "train_veto_audit": train_audit,
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
    all_folds_go = all(row["gate"]["status"] == "GO" for row in fold_reports)
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
        "schema_version": "evidence-carrier-pair-local-veto-oof-summary-0.1",
        "status": status,
        "report_path": str(report_path),
        "report_sha256": file_sha256(report_path),
        "parameter_count": PAIR_LOCAL_VETO_PARAMETER_COUNT,
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
                "vetoed_positive_pair_count": row["vetoed_positive_pair_count"],
                "vetoed_negative_pair_count": row["vetoed_negative_pair_count"],
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
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    print(json.dumps(train(build_arg_parser().parse_args()), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["audit_train_veto", "evaluate_oof_records", "train_steps"]
