#!/usr/bin/env python3
"""Train v0.9.22 junction-negative veto and evaluate consensus rescue OOF."""

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

import chemvector.evidence_carrier_pair_junction_veto as model_module
import chemvector.train_evidence_carrier_endpoint_oof as base
import chemvector.train_evidence_carrier_pair_local_oof as helper
import chemvector.train_evidence_carrier_pair_local_veto_oof as veto_helper
from chemvector.build_evidence_carrier_pair_local_oof import file_sha256
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
    evaluate_evidence_carrier_pair_sample,
)
from chemvector.evidence_carrier_pair_junction_veto import (
    PAIR_JUNCTION_VETO_PARAMETER_COUNT,
    EvidenceCarrierPairJunctionVeto,
    pair_junction_veto_loss,
)
from chemvector.evidence_carrier_pair_local_residual import (
    EvidenceCarrierPairLocalResidual,
)
from chemvector.evidence_carrier_pair_local_spatial_residual import (
    EvidenceCarrierPairLocalSpatialResidual,
)
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
)


CHECKPOINT_SCHEMA_VERSION = "evidence-carrier-pair-junction-veto-oof-checkpoint-0.1"
REPORT_SCHEMA_VERSION = "evidence-carrier-pair-junction-veto-oof-report-0.1"
STATUS_GO = "FIT-OOF-PAIR-JUNCTION-VETO-GO"
STATUS_NO_GO = "FIT-OOF-PAIR-JUNCTION-VETO-NO-GO"
DEFAULT_OUTPUT_DIR = "outputs/checkpoints/chemvector-v0922-pair-junction-veto-oof"
DEFAULT_REPORT = "outputs/evaluations/chemvector-v0922-pair-junction-veto-oof/oof.json"
TRAINING_STEPS = 2_000
TRAINING_BATCH_SIZE = 32
TRAINING_SEEDS = (20_260_734, 20_260_735, 20_260_736)
TRAINING_LEARNING_RATE = 1e-3
TRAINING_WEIGHT_DECAY = 1e-4
TRAINING_MAX_GRAD_NORM = 1.0
AVG_CHECKPOINT_DIR = "outputs/checkpoints/chemvector-v0919-pair-local-oof"
SPATIAL_CHECKPOINT_DIR = "outputs/checkpoints/chemvector-v0920-pair-local-spatial-oof"


class PairJunctionVetoOOFTrainingError(RuntimeError):
    """Raised when the junction-veto OOF protocol differs."""


def implementation_sha256() -> dict[str, str]:
    return {
        "trainer": file_sha256(Path(__file__).resolve()),
        "model": file_sha256(Path(model_module.__file__).resolve()),
        "pair_local_helper": file_sha256(Path(helper.__file__).resolve()),
        "veto_helper": file_sha256(Path(veto_helper.__file__).resolve()),
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


def collate_junction_records(records: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    result = helper.collate_pair_local_records(records)
    batch, capacity = result["pair_mask"].shape
    pair_group_types = torch.full((batch, capacity, 2), -1, dtype=torch.long)
    for index, record in enumerate(records):
        pairs = torch.as_tensor(record["target_free"]["pair_indices"], dtype=torch.long)
        anchor_types = torch.as_tensor(
            record["target_aware"]["evaluation_target"]["anchor_group_types"],
            dtype=torch.long,
        )
        if len(pairs):
            types = anchor_types[pairs]
            if bool((types < 1).any()) or bool((types > 3).any()):
                raise PairJunctionVetoOOFTrainingError("pair group type is out of scope")
            pair_group_types[index, : len(pairs)] = types
    result["pair_group_types"] = pair_group_types
    return result


def train_steps(
    model: EvidenceCarrierPairJunctionVeto,
    optimizer: torch.optim.Optimizer,
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    if len(records) != 4800:
        raise ValueError("junction veto trainer requires exactly 4,800 records")
    names = (
        "loss",
        "positive_loss",
        "worst_positive_loss",
        "negative_loss",
        "hard_negative_loss",
        "junction_negative_loss",
        "worst_junction_negative_loss",
        "logit_l2_loss",
    )
    values = {name: [] for name in names}
    gradients = []
    model.train()
    for indices in base.deterministic_batch_indices(
        len(records), steps=TRAINING_STEPS, batch_size=TRAINING_BATCH_SIZE, seed=seed
    ):
        batch = helper._to_device(
            collate_junction_records([records[index] for index in indices]), device
        )
        output = veto_helper._forward(model, batch)
        losses = pair_junction_veto_loss(
            output["veto_logits"],
            output["raw_gain"],
            batch["pair_labels"],
            batch["pair_mask"],
            batch["pair_group_types"],
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


def _load_teacher(
    model_class,
    path: Path,
    device: torch.device,
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = model_class()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), file_sha256(path)


@torch.inference_mode()
def evaluate_oof_records(
    veto_model: EvidenceCarrierPairJunctionVeto,
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
        batch = helper._to_device(collate_junction_records(current), device)
        started = perf_counter()
        avg = helper._forward(avg_model, batch)
        spatial = helper._forward(spatial_model, batch)
        veto = veto_helper._forward(veto_model, batch)
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
            candidate = torch.where(remove, torch.full_like(consensus, tau - 1.0), consensus)
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
            if consensus_eval["selected_anchor_pairs"] != candidate_eval["selected_anchor_pairs"]:
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
        "consensus_metrics": aggregate_evidence_carrier_pair_evaluations(consensus_rows),
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
) -> EvidenceCarrierPairJunctionVeto:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": EvidenceCarrierPairJunctionVeto.model_type,
        "parameter_count": PAIR_JUNCTION_VETO_PARAMETER_COUNT,
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
        raise PairJunctionVetoOOFTrainingError(f"junction checkpoint differs: {mismatches}")
    model = EvidenceCarrierPairJunctionVeto()
    if checkpoint.get("model_config") != model.config_dict():
        raise PairJunctionVetoOOFTrainingError("junction checkpoint config differs")
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
            "group_type_in_forward": False,
            "rescue": "min(v0.9.19,v0.9.20)",
        },
    )
    base_index, base_metadata = base.load_oof_index(args.base_index, audit_path=args.base_audit)
    local_index, local_metadata = helper.load_local_index(args.local_index)
    frozen_implementation = implementation_sha256()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PairJunctionVetoOOFTrainingError("CUDA was requested but is unavailable")
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
        model = EvidenceCarrierPairJunctionVeto().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=TRAINING_LEARNING_RATE, weight_decay=TRAINING_WEIGHT_DECAY
        )
        metrics = train_steps(model, optimizer, train_records, seed=seed, device=device)
        if implementation_sha256() != frozen_implementation:
            raise PairJunctionVetoOOFTrainingError("implementation changed during training")
        train_audit = veto_helper.audit_train_veto(model, train_records, device=device)
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
        spatial_path = Path(args.spatial_checkpoint_dir).resolve() / f"fold{eval_fold}" / "final.pt"
        avg_model, avg_sha = _load_teacher(EvidenceCarrierPairLocalResidual, avg_path, device)
        spatial_model, spatial_sha = _load_teacher(
            EvidenceCarrierPairLocalSpatialResidual, spatial_path, device
        )
        eval_records, eval_base_meta = base.load_oof_fold(base_index, eval_fold)
        eval_crops, eval_local_meta = helper.load_local_fold(local_index, eval_fold)
        eval_records = helper.merge_pair_local_records(eval_records, eval_crops)
        evaluation = evaluate_oof_records(
            sealed, avg_model, spatial_model, eval_records, device=device
        )
        if file_sha256(checkpoint_path) != checkpoint_sha:
            raise PairJunctionVetoOOFTrainingError("checkpoint changed during evaluation")
        gate = base.oof_gate(
            evaluation["raw_metrics"],
            evaluation["candidate_metrics"],
            expected_sample_count=2400,
        )
        consensus_key = base._structural_key(evaluation["consensus_metrics"])
        candidate_key = base._structural_key(evaluation["candidate_metrics"])
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
            "strictly_better_than_consensus": candidate_key > consensus_key,
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
    consensus_metrics = aggregate_evidence_carrier_pair_evaluations(all_consensus_rows)
    candidate_metrics = aggregate_evidence_carrier_pair_evaluations(all_candidate_rows)
    aggregate_gate = base.oof_gate(raw_metrics, candidate_metrics, expected_sample_count=7200)
    all_folds_go = all(row["gate"]["status"] == "GO" for row in fold_reports)
    status = STATUS_GO if aggregate_gate["status"] == "GO" and all_folds_go else STATUS_NO_GO
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": status,
        "fold_reports": fold_reports,
        "aggregate_raw_metrics": raw_metrics,
        "aggregate_consensus_metrics": consensus_metrics,
        "aggregate_candidate_metrics": candidate_metrics,
        "aggregate_strictly_better_than_consensus": (
            base._structural_key(candidate_metrics) > base._structural_key(consensus_metrics)
        ),
        "aggregate_gate": aggregate_gate,
        "all_fold_gates_go": all_folds_go,
        "base_index_metadata": base_metadata,
        "local_index_metadata": local_metadata,
        "implementation_sha256": frozen_implementation,
        "group_type_in_forward": False,
        "validation_read": False,
        "fresh_read": False,
        "sealed_read": False,
        "internal_test_read": False,
        "mr_tadf_read": False,
    }
    _write_json_no_overwrite(report_path, report)
    summary = {
        "schema_version": "evidence-carrier-pair-junction-veto-oof-summary-0.1",
        "status": status,
        "report_path": str(report_path),
        "report_sha256": file_sha256(report_path),
        "parameter_count": PAIR_JUNCTION_VETO_PARAMETER_COUNT,
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
    parser.add_argument("--avg-checkpoint-dir", default=AVG_CHECKPOINT_DIR)
    parser.add_argument("--spatial-checkpoint-dir", default=SPATIAL_CHECKPOINT_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    print(json.dumps(train(build_arg_parser().parse_args()), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["collate_junction_records", "evaluate_oof_records", "train_steps"]
