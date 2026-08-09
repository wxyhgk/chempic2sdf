#!/usr/bin/env python3
"""Train KEEP/MOVE with explicit center-field point evidence."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from chemvector.g3_structural_data import file_sha256
from chemvector.training.atoms import train_g3_p6e9_structural_residual_arms as p6e9
from chemvector.training.atoms import train_g3_p7_pairwise_counterfactual as p7
from chemvector.training.atoms import train_g3_p11e1_predicted_source_keep_move as base
from chemvector.training.atoms import train_g3_p11e3_predicted_source_keep_move_full as full


MODEL_TYPE = "chemvector-g3-p12-explicit-score-keep-move-full2928"
OUTPUT = Path("outputs/checkpoints") / MODEL_TYPE
CHECKPOINT = OUTPUT / "step-008000.pt"
STEPS = 8_000
MODEL_SEED = 2026089801
SCHEDULE_SEED = 2026089811
MINIMUM_CALIBRATION_RESCUES = 40


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P12 output already exists: {OUTPUT}")
    for path, expected in full.SHA256.items():
        if file_sha256(path) != expected:
            raise ValueError(f"P12 cache SHA256 differs: {path}")
    if (full.HOLDOUT.stat().st_mode & 0o777) != 0:
        raise RuntimeError("P12 holdout must remain sealed")
    if any((path.stat().st_mode & 0o777) != 0 for path in p6e9.HOLDOUT_FILES):
        raise RuntimeError("P12 historical holdout must remain sealed")
    if not torch.cuda.is_available():
        raise RuntimeError("P12 requires CUDA")
    device = torch.device("cuda")
    train_cache = torch.load(full.TRAIN, map_location="cpu", weights_only=False)
    calibration_cache = torch.load(full.CALIBRATION, map_location="cpu", weights_only=False)
    train_table = base.build_pair_table(train_cache, None)
    calibration_table = base.build_pair_table(calibration_cache, None)
    pools, pool_audit = base.select_stratified_pools(
        train_table, equivalent_pool=full.EQUIVALENT_POOL
    )
    if len(pools["rescue"]) != full.EXPECTED_PER_CLASS or len(pools["harmful"]) != full.EXPECTED_PER_CLASS:
        raise RuntimeError(f"P12 pool differs: {pool_audit}")
    model, training = base.train_model(
        train_cache,
        train_table,
        pools,
        steps=STEPS,
        model_seed=MODEL_SEED,
        schedule_seed=SCHEDULE_SEED,
        device=device,
        explicit_score_evidence=True,
    )
    records = base.calibration_records(
        model, calibration_cache, calibration_table, device
    )
    calibration = base.calibrate_nonnegative(records)
    OUTPUT.mkdir(parents=True)
    p7._atomic_torch_save(
        CHECKPOINT,
        {
            "checkpoint_schema_version": "g3-p12-explicit-score-keep-move-0.1",
            "model_type": MODEL_TYPE,
            "model_config": model.config_dict(),
            "model": model.state_dict(),
            "global_step": STEPS,
            "threshold": calibration["threshold"],
            "model_seed": MODEL_SEED,
            "schedule_seed": SCHEDULE_SEED,
        },
    )
    checks = {
        "explicit_score_evidence": model.config.explicit_score_evidence,
        "train_1464_each": len(pools["rescue"]) == len(pools["harmful"]) == full.EXPECTED_PER_CLASS,
        "calibration_zero_harmful": calibration["harmful"] == 0,
        "calibration_rescues_min40": calibration["rescues"] >= MINIMUM_CALIBRATION_RESCUES,
        "threshold_nonnegative": calibration["threshold"] >= 0.0,
        "holdout_mode_000": (full.HOLDOUT.stat().st_mode & 0o777) == 0,
    }
    audit = {
        "schema_version": "g3-p12-explicit-score-keep-move-audit-0.1",
        "decision": "explicit_score_keep_move_calibration_go" if all(checks.values()) else "explicit_score_keep_move_calibration_no_go",
        "checks": checks,
        "model_type": MODEL_TYPE,
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": file_sha256(CHECKPOINT),
        "parameter_count": model.parameter_count,
        "pool": pool_audit,
        "training": training,
        "calibration": calibration,
        "protocol": {
            "new_features": ["source_center_field", "alternative_center_field", "alternative_minus_source"],
            "other_model_and_data": "identical to P11-E3",
            "holdout_read": False,
            "confirm_external_sealed_read": False,
        },
        "holdout_read": False,
        "confirm_external_sealed_read": False,
    }
    p7._atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
