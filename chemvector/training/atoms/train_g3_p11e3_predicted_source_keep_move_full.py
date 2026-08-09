#!/usr/bin/env python3
"""Train full-scale explicit KEEP-vs-MOVE on real predicted sources."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from chemvector.g3_structural_data import file_sha256
from chemvector.training.atoms import train_g3_p6e9_structural_residual_arms as p6e9
from chemvector.training.atoms import train_g3_p7_pairwise_counterfactual as p7
from chemvector.training.atoms import train_g3_p11e1_predicted_source_keep_move as pilot


MODEL_TYPE = "chemvector-g3-p11e3-predicted-source-keep-move-full2928"
OUTPUT = Path("outputs/checkpoints") / MODEL_TYPE
CHECKPOINT = OUTPUT / "step-008000.pt"
ROOT = Path("outputs/audits/chemvector-g3-p11e2-predicted-source-full80k")
TRAIN = ROOT / "train.pt"
CALIBRATION = ROOT / "calibration.pt"
SHA256 = {
    TRAIN: "c57a75f98092bab25205f6c43d2d4b741daba9ce26265db13daef0f3ea7491fd",
    CALIBRATION: "1c4f12bd29c228120df2c8010747d9fa7942414aebf38a9a67d90f048fc04ce5",
}
HOLDOUT = ROOT / "holdout.pt"
EXPECTED_PER_CLASS = 1_464
EQUIVALENT_POOL = 2_048
STEPS = 8_000
MODEL_SEED = 2026089701
SCHEDULE_SEED = 2026089711
MINIMUM_CALIBRATION_RESCUES = 40


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P11-E3 output already exists: {OUTPUT}")
    for path, expected in SHA256.items():
        if file_sha256(path) != expected:
            raise ValueError(f"P11-E3 cache SHA256 differs: {path}")
    if (HOLDOUT.stat().st_mode & 0o777) != 0:
        raise RuntimeError("P11-E3 holdout must remain sealed")
    if any((path.stat().st_mode & 0o777) != 0 for path in p6e9.HOLDOUT_FILES):
        raise RuntimeError("P11-E3 historical holdout must remain sealed")
    if not torch.cuda.is_available():
        raise RuntimeError("P11-E3 requires CUDA")
    device = torch.device("cuda")
    train_cache = torch.load(TRAIN, map_location="cpu", weights_only=False)
    calibration_cache = torch.load(CALIBRATION, map_location="cpu", weights_only=False)
    train_table = pilot.build_pair_table(train_cache, None)
    calibration_table = pilot.build_pair_table(calibration_cache, None)
    pools, pool_audit = pilot.select_stratified_pools(
        train_table, equivalent_pool=EQUIVALENT_POOL
    )
    if len(pools["rescue"]) != EXPECTED_PER_CLASS or len(pools["harmful"]) != EXPECTED_PER_CLASS:
        raise RuntimeError(f"P11-E3 pool differs: {pool_audit}")
    model, training = pilot.train_model(
        train_cache,
        train_table,
        pools,
        steps=STEPS,
        model_seed=MODEL_SEED,
        schedule_seed=SCHEDULE_SEED,
        device=device,
    )
    records = pilot.calibration_records(
        model, calibration_cache, calibration_table, device
    )
    calibration = pilot.calibrate_nonnegative(records)
    OUTPUT.mkdir(parents=True)
    p7._atomic_torch_save(
        CHECKPOINT,
        {
            "checkpoint_schema_version": "g3-p11e3-predicted-source-keep-move-full-0.1",
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
        "train_1464_each": len(pools["rescue"]) == len(pools["harmful"]) == EXPECTED_PER_CLASS,
        "calibration_zero_harmful": calibration["harmful"] == 0,
        "calibration_rescues_min40": calibration["rescues"] >= MINIMUM_CALIBRATION_RESCUES,
        "threshold_nonnegative": calibration["threshold"] >= 0.0,
        "holdout_mode_000": (HOLDOUT.stat().st_mode & 0o777) == 0,
        "historical_holdout_mode_000": all((path.stat().st_mode & 0o777) == 0 for path in p6e9.HOLDOUT_FILES),
    }
    audit = {
        "schema_version": "g3-p11e3-predicted-source-keep-move-full-audit-0.1",
        "decision": "predicted_source_keep_move_full_calibration_go" if all(checks.values()) else "predicted_source_keep_move_full_calibration_no_go",
        "checks": checks,
        "model_type": MODEL_TYPE,
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": file_sha256(CHECKPOINT),
        "parameter_count": model.parameter_count,
        "pool": pool_audit,
        "training": training,
        "calibration": calibration,
        "calibration_sources": {
            "MOVE": sum(value == "MOVE" for value in calibration_cache["action"]),
            "KEEP": sum(value == "KEEP" for value in calibration_cache["action"]),
        },
        "protocol": {
            "prior": "explicit zero-offset KEEP",
            "alternative": "one frozen P6-E1 top8 MOVE",
            "peak_margin_used": False,
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
