#!/usr/bin/env python3
"""Train v0.9.20 spatial pair-local residual with the frozen v0.9.19 OOF data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

import chemvector.evidence_carrier_pair_local_spatial_residual as model_module
import chemvector.train_evidence_carrier_endpoint_oof as base
import chemvector.train_evidence_carrier_pair_local_oof as helper
from chemvector.build_evidence_carrier_pair_local_oof import file_sha256
from chemvector.evidence_carrier_pair_evaluation import (
    aggregate_evidence_carrier_pair_evaluations,
)
from chemvector.evidence_carrier_pair_local_spatial_residual import (
    PAIR_LOCAL_SPATIAL_PARAMETER_COUNT,
    EvidenceCarrierPairLocalSpatialResidual,
)


CHECKPOINT_SCHEMA_VERSION = "evidence-carrier-pair-local-spatial-oof-checkpoint-0.1"
REPORT_SCHEMA_VERSION = "evidence-carrier-pair-local-spatial-oof-report-0.1"
STATUS_GO = "FIT-OOF-PAIR-LOCAL-SPATIAL-GO"
STATUS_NO_GO = "FIT-OOF-PAIR-LOCAL-SPATIAL-NO-GO"
DEFAULT_OUTPUT_DIR = "outputs/checkpoints/chemvector-v0920-pair-local-spatial-oof"
DEFAULT_REPORT = "outputs/evaluations/chemvector-v0920-pair-local-spatial-oof/oof.json"
TRAINING_SEEDS = (20_260_728, 20_260_729, 20_260_730)


class PairLocalSpatialOOFTrainingError(RuntimeError):
    """Raised when the v0.9.20 frozen OOF contract differs."""


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


def _load_checkpoint(
    path: Path, *, device: torch.device, implementation: Mapping[str, str]
) -> EvidenceCarrierPairLocalSpatialResidual:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_type": EvidenceCarrierPairLocalSpatialResidual.model_type,
        "parameter_count": PAIR_LOCAL_SPATIAL_PARAMETER_COUNT,
        "optimizer_steps": helper.TRAINING_STEPS,
        "checkpoint_selection": "last_step_only",
        "implementation_sha256": dict(implementation),
    }
    mismatches = {
        name: (checkpoint.get(name), value)
        for name, value in expected.items()
        if checkpoint.get(name) != value
    }
    if mismatches:
        raise PairLocalSpatialOOFTrainingError(
            f"spatial checkpoint contract differs: {mismatches}"
        )
    model = EvidenceCarrierPairLocalSpatialResidual()
    if checkpoint.get("model_config") != model.config_dict():
        raise PairLocalSpatialOOFTrainingError("spatial checkpoint model config differs")
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
            "steps_per_fold": helper.TRAINING_STEPS,
            "spatial_grid": [3, 6],
        },
    )
    base_index, base_metadata = base.load_oof_index(args.base_index, audit_path=args.base_audit)
    local_index, local_metadata = helper.load_local_index(args.local_index)
    frozen_implementation = implementation_sha256()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PairLocalSpatialOOFTrainingError("CUDA was requested but is unavailable")
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
        model = EvidenceCarrierPairLocalSpatialResidual().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=helper.TRAINING_LEARNING_RATE,
            weight_decay=helper.TRAINING_WEIGHT_DECAY,
        )
        metrics = helper.train_steps(
            model, optimizer, train_records, seed=seed, device=device
        )
        if implementation_sha256() != frozen_implementation:
            raise PairLocalSpatialOOFTrainingError("implementation changed during training")
        finite = helper.audit_train_finite(model, train_records, device=device)
        checkpoint_path = output_dir / f"fold{eval_fold}" / "final.pt"
        checkpoint = {
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_type": model.model_type,
            "model": model.state_dict(),
            "model_config": model.config_dict(),
            "parameter_count": model.parameter_count(),
            "optimizer_steps": helper.TRAINING_STEPS,
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
        eval_crops, eval_local_meta = helper.load_local_fold(local_index, eval_fold)
        eval_records = helper.merge_pair_local_records(eval_records, eval_crops)
        evaluation = helper.evaluate_oof_records(sealed, eval_records, device=device)
        if file_sha256(checkpoint_path) != checkpoint_sha:
            raise PairLocalSpatialOOFTrainingError("checkpoint changed during evaluation")
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
        "schema_version": "evidence-carrier-pair-local-spatial-oof-summary-0.1",
        "status": status,
        "report_path": str(report_path),
        "report_sha256": file_sha256(report_path),
        "parameter_count": PAIR_LOCAL_SPATIAL_PARAMETER_COUNT,
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
    parser.add_argument("--local-index", default=helper.DEFAULT_LOCAL_INDEX)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    print(json.dumps(train(build_arg_parser().parse_args()), ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["train"]
