#!/usr/bin/env python3
"""Perform the one permitted aggregate-only P7 holdout evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from chemvector.components.recovery.pairwise_structural_counterfactual import (
    PairwiseStructuralCounterfactual,
    PairwiseStructuralCounterfactualConfig,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.training.atoms import train_g3_p6e9_structural_residual_arms as p6e9
from chemvector.training.atoms import train_g3_p7_pairwise_counterfactual as p7
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json


AUDIT_NAME = "chemvector-g3-p7e4-pairwise-holdout"
OUTPUT = Path("outputs/audits") / AUDIT_NAME
CALIBRATION_AUDIT = p7.OUTPUT / "audit.json"
CALIBRATION_AUDIT_SHA256 = "7b0ce7a31c398dc9d4cad1236336f16c057a27c5ed011ff52c83f412d5ebc648"
MODEL_TYPE = "chemvector-g3-p7e2-pairwise-counterfactual-patch-only-train11762"
CHECKPOINT = Path("outputs/checkpoints") / MODEL_TYPE / "step-008000.pt"
CHECKPOINT_SHA256 = "a5fb6356a442957f67177288512c1af3f5560b6e00a5d80625fa46ca65956687"
THRESHOLD = 1.6568973064422607
HOLDOUT_SHA256 = {
    p6e9.HOLDOUT_FILES[0]: "7887cde5e4fd6f21f48931f3aa3259520cd12c55e7fc56795cace314a37eb6ca",
    p6e9.HOLDOUT_FILES[1]: "3e7bb3666a1a3d24cec213cf54f068453a035516f3abbca49f85b813828344cd",
}
MINIMUM_RESCUES = 100


def fixed_threshold_metrics(records: list[dict], threshold: float) -> dict:
    executed = [row for row in records if float(row["score"]) >= threshold]
    harmful = sum(
        bool(row["prior_good"]) and not bool(row["alternative_good"])
        for row in executed
    )
    rescues = sum(
        not bool(row["prior_good"]) and bool(row["alternative_good"])
        for row in executed
    )
    return {
        "threshold": threshold,
        "executed": len(executed),
        "rescues": rescues,
        "harmful": harmful,
        "useless": len(executed) - rescues - harmful,
    }


def _load_model(device: torch.device) -> PairwiseStructuralCounterfactual:
    if file_sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError("P7 winner checkpoint SHA256 differs")
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if payload["model_type"] != MODEL_TYPE:
        raise ValueError("P7 winner model type differs")
    if float(payload["calibration_threshold"]) != THRESHOLD:
        raise ValueError("P7 winner threshold differs")
    model = PairwiseStructuralCounterfactual(
        PairwiseStructuralCounterfactualConfig(arm="patch-only")
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("P7 holdout has already been evaluated")
    if file_sha256(CALIBRATION_AUDIT) != CALIBRATION_AUDIT_SHA256:
        raise ValueError("P7 calibration audit SHA256 differs")
    calibration = json.loads(CALIBRATION_AUDIT.read_text(encoding="utf-8"))
    if (
        calibration["decision"] != "pairwise_counterfactual_calibration_go"
        or calibration["winner"] != MODEL_TYPE
    ):
        raise RuntimeError("P7 calibration did not unlock holdout")
    if any((path.stat().st_mode & 0o777) != 0 for path in p6e9.HOLDOUT_FILES):
        raise RuntimeError("P7 holdout must begin sealed")
    if not torch.cuda.is_available():
        raise RuntimeError("P7 holdout requires CUDA")
    device = torch.device("cuda")
    model = _load_model(device)
    hard = easy = None
    try:
        for path in p6e9.HOLDOUT_FILES:
            os.chmod(path, 0o400)
        for path, expected in HOLDOUT_SHA256.items():
            if file_sha256(path) != expected:
                raise ValueError(f"P7 holdout SHA256 differs: {path}")
        hard = torch.load(
            p6e9.HOLDOUT_FILES[0], map_location="cpu", weights_only=False
        )
        easy = torch.load(
            p6e9.HOLDOUT_FILES[1], map_location="cpu", weights_only=False
        )
    finally:
        for path in p6e9.HOLDOUT_FILES:
            os.chmod(path, 0)
    if hard is None or easy is None:
        raise RuntimeError("P7 holdout load did not complete")
    records = p7.evaluate_source_cache(model, hard, "hard", device)
    records += p7.evaluate_source_cache(model, easy, "easy", device)
    metrics = fixed_threshold_metrics(records, THRESHOLD)
    checks = {
        "hard_rows_738": len(hard["offset_targets"]) == 738,
        "easy_rows_2776": len(easy["offset_targets"]) == 2776,
        "zero_harmful": metrics["harmful"] == 0,
        "rescues_min100": metrics["rescues"] >= MINIMUM_RESCUES,
        "holdout_resealed_000": all(
            (path.stat().st_mode & 0o777) == 0 for path in p6e9.HOLDOUT_FILES
        ),
    }
    audit = {
        "schema_version": "g3-p7e4-pairwise-holdout-audit-0.1",
        "decision": "pairwise_counterfactual_holdout_go"
        if all(checks.values())
        else "pairwise_counterfactual_holdout_no_go",
        "checks": checks,
        "model_type": MODEL_TYPE,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "calibration_audit_sha256": CALIBRATION_AUDIT_SHA256,
        "metrics": metrics,
        "protocol": {
            "single_read": True,
            "aggregate_only": True,
            "per_sample_scores_saved": False,
            "threshold_frozen": True,
        },
        "holdout_read": True,
        "dev_confirm_external_sealed_read": False,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
