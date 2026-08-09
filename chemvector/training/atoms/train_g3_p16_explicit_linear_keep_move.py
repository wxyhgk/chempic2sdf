#!/usr/bin/env python3
"""Fit a deterministic seven-feature linear KEEP/MOVE decision model."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from chemvector.components.recovery.explicit_keep_move_linear import (
    ExplicitKeepMoveLinearConfig,
    fit_explicit_keep_move_linear,
)
from chemvector.evaluation.audits.atoms import audit_g3_p15e0_bond_endpoint_pair_signal as p15e0
from chemvector.g3_structural_data import file_sha256
from chemvector.training.atoms import train_g3_p6e9_structural_residual_arms as p6e9
from chemvector.training.atoms import train_g3_p7_pairwise_counterfactual as p7
from chemvector.training.atoms import train_g3_p11e1_predicted_source_keep_move as base


MODEL_TYPE = "chemvector-g3-p16-explicit-linear-keep-move-fit562"
OUTPUT = Path("outputs/checkpoints") / MODEL_TYPE
CHECKPOINT = OUTPUT / "fit-newton64.pt"
SOURCE_CACHE = base.CACHE
SOURCE_CACHE_SHA256 = base.CACHE_SHA256
P15_AUDIT = p15e0.OUTPUT / "audit.json"
P15_AUDIT_SHA256 = (
    "9761aa3c5fe30c400287e845200ef690932a607790c0f7532d92a722b2b450e7"
)
EXPECTED_PER_CLASS = 281
MINIMUM_TRAINING_ACCURACY = 0.75
MINIMUM_CALIBRATION_RESCUES = 20
L2 = 0.01
NEWTON_STEPS = 64


@torch.no_grad()
def calibration_records(model, cache: dict, table: dict) -> list[dict]:
    model.eval()
    records = []
    for source_row in table["source_rows"].tolist():
        alternatives = torch.where(cache["candidate_mask"][source_row])[0]
        count = len(alternatives)
        source_rows = torch.full((count,), source_row, dtype=torch.long)
        features = p15e0.pair_baseline_features(cache, source_rows, alternatives)
        scores = model(features)
        winner = int(scores.argmax())
        target = cache["offset_targets"][source_row].float()
        prior_good = float(torch.linalg.vector_norm(target * 768.0)) <= 6.0
        alternative_good = float(
            torch.linalg.vector_norm(
                (
                    cache["candidate_offsets"][source_row, int(alternatives[winner])].float()
                    - target
                )
                * 768.0
            )
        ) <= 6.0
        records.append(
            {
                "source_row": source_row,
                "score": float(scores[winner]),
                "prior_good": prior_good,
                "alternative_good": alternative_good,
                "action": cache["action"][source_row],
            }
        )
    return records


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P16 output already exists: {OUTPUT}")
    for path, expected in (
        (SOURCE_CACHE, SOURCE_CACHE_SHA256),
        (P15_AUDIT, P15_AUDIT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P16 input SHA256 differs: {path}")
    if any((path.stat().st_mode & 0o777) != 0 for path in p6e9.HOLDOUT_FILES):
        raise RuntimeError("P16 historical holdout must remain sealed")

    cache = torch.load(SOURCE_CACHE, map_location="cpu", weights_only=False)
    train_table = base.build_pair_table(cache, "train")
    calibration_table = base.build_pair_table(cache, "calibration")
    pools, pool_audit = base.select_stratified_pools(train_table)
    if (
        len(pools["rescue"]) != EXPECTED_PER_CLASS
        or len(pools["harmful"]) != EXPECTED_PER_CLASS
    ):
        raise RuntimeError(f"P16 pool differs: {pool_audit}")
    pair_rows = torch.cat((pools["rescue"], pools["harmful"]))
    source_rows = train_table["source_row"][pair_rows]
    alternatives = train_table["alternative_index"][pair_rows]
    features = p15e0.pair_baseline_features(cache, source_rows, alternatives)
    labels = train_table["label"][pair_rows] == 1
    model, fit = fit_explicit_keep_move_linear(
        features,
        labels,
        ExplicitKeepMoveLinearConfig(l2=L2, newton_steps=NEWTON_STEPS),
    )
    records = calibration_records(model, cache, calibration_table)
    calibration = base.calibrate_nonnegative(records)

    OUTPUT.mkdir(parents=True)
    p7._atomic_torch_save(
        CHECKPOINT,
        {
            "checkpoint_schema_version": "g3-p16-explicit-linear-keep-move-0.1",
            "model_type": MODEL_TYPE,
            "model_config": model.config_dict(),
            "model": model.state_dict(),
            "fit": fit,
            "threshold": calibration["threshold"],
        },
    )
    checks = {
        "parameter_count_8": model.parameter_count == 8,
        "train_281_each": len(pools["rescue"])
        == len(pools["harmful"])
        == EXPECTED_PER_CLASS,
        "training_accuracy_min075": fit["training_accuracy"]
        >= MINIMUM_TRAINING_ACCURACY,
        "calibration_zero_harmful": calibration["harmful"] == 0,
        "calibration_rescues_min20": calibration["rescues"]
        >= MINIMUM_CALIBRATION_RESCUES,
        "threshold_nonnegative": calibration["threshold"] >= 0.0,
        "historical_holdout_mode_000": all(
            (path.stat().st_mode & 0o777) == 0 for path in p6e9.HOLDOUT_FILES
        ),
    }
    audit = {
        "schema_version": "g3-p16-explicit-linear-keep-move-audit-0.1",
        "decision": (
            "explicit_linear_keep_move_calibration_go"
            if all(checks.values())
            else "explicit_linear_keep_move_calibration_no_go"
        ),
        "checks": checks,
        "model_type": MODEL_TYPE,
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": file_sha256(CHECKPOINT),
        "parameter_count": model.parameter_count,
        "fit": fit,
        "pool": pool_audit,
        "calibration": calibration,
        "protocol": {
            "features": "fixed seven-dimensional P15 baseline",
            "solver": "deterministic damped Newton L2 logistic",
            "l2": L2,
            "maximum_newton_steps": NEWTON_STEPS,
            "feature_search": False,
            "bond_features": False,
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
