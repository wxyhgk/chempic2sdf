#!/usr/bin/env python3
"""Train linear KEEP/MOVE with the deployed per-source top8 maximum."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from chemvector.components.recovery.explicit_keep_move_linear import (
    ExplicitKeepMoveLinear,
    ExplicitKeepMoveLinearConfig,
)
from chemvector.evaluation.audits.atoms import audit_g3_p15e0_bond_endpoint_pair_signal as p15e0
from chemvector.g3_structural_data import file_sha256
from chemvector.training.atoms import train_g3_p6e9_structural_residual_arms as p6e9
from chemvector.training.atoms import train_g3_p7_pairwise_counterfactual as p7
from chemvector.training.atoms import train_g3_p11e1_predicted_source_keep_move as base


MODEL_TYPE = "chemvector-g3-p17-groupwise-linear-keep-move-train1809"
OUTPUT = Path("outputs/checkpoints") / MODEL_TYPE
CHECKPOINT = OUTPUT / "step-004000.pt"
SOURCE_CACHE = base.CACHE
SOURCE_CACHE_SHA256 = base.CACHE_SHA256
STEPS = 4_000
BATCH_RESCUE = 24
BATCH_KEEP = 32
BATCH_UNSUPPORTED = 8
LEARNING_RATE = 1e-2
L2 = 0.01
MARGIN = 1.0
SEED = 20260902
EXPECTED_TRAIN_GROUPS = {"rescue": 191, "keep": 1_584, "unsupported": 34}
MINIMUM_TRAIN_GROUP_ACCURACY = 0.75
MINIMUM_CALIBRATION_RESCUES = 20


def build_groups(cache: dict, table: dict) -> dict:
    source_rows = table["source_rows"].long()
    count = len(source_rows)
    features = torch.zeros(count, 8, 7)
    valid = cache["candidate_mask"][source_rows].clone()
    good = torch.zeros_like(valid)
    kind = torch.empty(count, dtype=torch.long)
    for local, source_row in enumerate(source_rows.tolist()):
        alternatives = torch.where(valid[local])[0]
        repeated = torch.full((len(alternatives),), source_row, dtype=torch.long)
        current = p15e0.pair_baseline_features(cache, repeated, alternatives)
        features[local, alternatives] = current
        target = cache["offset_targets"][source_row].float()
        source_good = float(torch.linalg.vector_norm(target * 768.0)) <= 6.0
        good[local] = (
            torch.linalg.vector_norm(
                (cache["candidate_offsets"][source_row].float() - target) * 768.0,
                dim=-1,
            )
            <= 6.0
        ) & valid[local]
        kind[local] = 1 if source_good else 0 if bool(good[local].any()) else 2
    return {
        "source_rows": source_rows,
        "features": features,
        "valid": valid,
        "good": good,
        "kind": kind,
    }


def groupwise_max_loss(
    scores: torch.Tensor,
    valid: torch.Tensor,
    good: torch.Tensor,
    kind: torch.Tensor,
    *,
    margin: float = MARGIN,
) -> dict[str, torch.Tensor]:
    if scores.shape != valid.shape or scores.shape != good.shape:
        raise ValueError("P17 group score/mask shapes differ")
    if valid.dtype != torch.bool or good.dtype != torch.bool:
        raise ValueError("P17 group masks must be boolean")
    if kind.shape != (len(scores),) or kind.dtype != torch.long:
        raise ValueError("P17 group kind must be a long vector")
    if bool((~valid).all(dim=1).any()) or bool((good & ~valid).any()):
        raise ValueError("P17 group masks are inconsistent")

    floor = torch.finfo(scores.dtype).min
    all_max = scores.masked_fill(~valid, floor).amax(dim=1)
    rescue = kind == 0
    keep = kind == 1
    unsupported = kind == 2
    if not bool(rescue.any() & keep.any() & unsupported.any()):
        raise ValueError("P17 training batch must contain all group kinds")
    positive_max = scores.masked_fill(~good, floor).amax(dim=1)
    bad = valid & ~good
    negative_max = scores.masked_fill(~bad, floor).amax(dim=1)
    rescue_positive = F.softplus(margin - positive_max[rescue]).mean()
    rescue_negative = F.softplus(margin + negative_max[rescue]).mean()
    rescue_ranking = F.softplus(
        margin + negative_max[rescue] - positive_max[rescue]
    ).mean()
    keep_max = F.softplus(margin + all_max[keep]).mean()
    unsupported_max = F.softplus(margin + all_max[unsupported]).mean()
    loss = (
        rescue_positive
        + rescue_negative
        + rescue_ranking
        + keep_max
        + unsupported_max
    )
    return {
        "loss": loss,
        "rescue_positive": rescue_positive,
        "rescue_negative": rescue_negative,
        "rescue_ranking": rescue_ranking,
        "keep_max": keep_max,
        "unsupported_max": unsupported_max,
    }


def schedule(groups: dict, steps: int = STEPS) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED)
    pools = {
        "rescue": torch.where(groups["kind"] == 0)[0],
        "keep": torch.where(groups["kind"] == 1)[0],
        "unsupported": torch.where(groups["kind"] == 2)[0],
    }
    sizes = {
        "rescue": BATCH_RESCUE,
        "keep": BATCH_KEEP,
        "unsupported": BATCH_UNSUPPORTED,
    }
    output = []
    for _ in range(steps):
        rows = torch.cat(
            [
                pool[torch.randint(len(pool), (sizes[name],), generator=generator)]
                for name, pool in pools.items()
            ]
        )
        output.append(rows[torch.randperm(len(rows), generator=generator)])
    return output


@torch.no_grad()
def operational_records(model, cache: dict, groups: dict) -> list[dict]:
    model.eval()
    records = []
    for local, source_row in enumerate(groups["source_rows"].tolist()):
        alternatives = torch.where(groups["valid"][local])[0]
        scores = model(groups["features"][local, alternatives])
        winner = int(scores.argmax())
        records.append(
            {
                "source_row": source_row,
                "score": float(scores[winner]),
                "prior_good": int(groups["kind"][local]) == 1,
                "alternative_good": bool(groups["good"][local, alternatives[winner]]),
                "action": cache["action"][source_row],
            }
        )
    return records


def group_accuracy(records: list[dict]) -> float:
    correct = 0
    for row in records:
        execute = row["score"] >= 0.0
        expected_execute = not row["prior_good"] and row["alternative_good"]
        correct += int(execute == expected_execute)
    return correct / len(records)


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P17 output already exists: {OUTPUT}")
    if file_sha256(SOURCE_CACHE) != SOURCE_CACHE_SHA256:
        raise ValueError("P17 source cache SHA256 differs")
    if any((path.stat().st_mode & 0o777) != 0 for path in p6e9.HOLDOUT_FILES):
        raise RuntimeError("P17 historical holdout must remain sealed")

    torch.manual_seed(SEED)
    cache = torch.load(SOURCE_CACHE, map_location="cpu", weights_only=False)
    train_groups = build_groups(cache, base.build_pair_table(cache, "train"))
    calibration_groups = build_groups(
        cache, base.build_pair_table(cache, "calibration")
    )
    observed = {
        "rescue": int((train_groups["kind"] == 0).sum()),
        "keep": int((train_groups["kind"] == 1).sum()),
        "unsupported": int((train_groups["kind"] == 2).sum()),
    }
    if observed != EXPECTED_TRAIN_GROUPS:
        raise RuntimeError(f"P17 train groups differ: {observed}")

    valid_features = train_groups["features"][train_groups["valid"]]
    mean = valid_features.mean(dim=0)
    scale = valid_features.std(dim=0, unbiased=False).clamp_min(1e-8)
    model = ExplicitKeepMoveLinear(
        ExplicitKeepMoveLinearConfig(l2=L2, newton_steps=1)
    )
    model.feature_mean.copy_(mean)
    model.feature_scale.copy_(scale)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0
    )
    sums = Counter()
    last = {}
    started = time.perf_counter()
    batches = schedule(train_groups)
    for step, rows in enumerate(batches, 1):
        optimizer.zero_grad(set_to_none=True)
        scores = model(train_groups["features"][rows].reshape(-1, 7)).reshape(-1, 8)
        losses = groupwise_max_loss(
            scores,
            train_groups["valid"][rows],
            train_groups["good"][rows],
            train_groups["kind"][rows],
        )
        regularization = L2 * model.linear.weight.square().sum()
        total = losses["loss"] + regularization
        total.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last = {name: float(value.detach()) for name, value in losses.items()} | {
            "regularization": float(regularization.detach()),
            "gradient_norm": float(gradient),
        }
        for name, value in last.items():
            sums[name] += value
        if step % 1000 == 0:
            print(json.dumps({"step": step, **last}), flush=True)

    train_records = operational_records(model, cache, train_groups)
    train_accuracy = group_accuracy(train_records)
    calibration_records = operational_records(model, cache, calibration_groups)
    calibration = base.calibrate_nonnegative(calibration_records)
    training = {
        "steps": STEPS,
        "elapsed_seconds": time.perf_counter() - started,
        "mean": {name: value / STEPS for name, value in sums.items()},
        "last": last,
        "group_accuracy_at_zero": train_accuracy,
    }

    OUTPUT.mkdir(parents=True)
    p7._atomic_torch_save(
        CHECKPOINT,
        {
            "checkpoint_schema_version": "g3-p17-groupwise-linear-keep-move-0.1",
            "model_type": MODEL_TYPE,
            "model_config": model.config_dict(),
            "model": model.state_dict(),
            "global_step": STEPS,
            "threshold": calibration["threshold"],
            "training": training,
        },
    )
    checks = {
        "parameter_count_8": model.parameter_count == 8,
        "train_groups_exact": observed == EXPECTED_TRAIN_GROUPS,
        "train_group_accuracy_min075": train_accuracy
        >= MINIMUM_TRAIN_GROUP_ACCURACY,
        "calibration_zero_harmful": calibration["harmful"] == 0,
        "calibration_rescues_min20": calibration["rescues"]
        >= MINIMUM_CALIBRATION_RESCUES,
        "threshold_nonnegative": calibration["threshold"] >= 0.0,
        "historical_holdout_mode_000": all(
            (path.stat().st_mode & 0o777) == 0 for path in p6e9.HOLDOUT_FILES
        ),
    }
    audit = {
        "schema_version": "g3-p17-groupwise-linear-keep-move-audit-0.1",
        "decision": (
            "groupwise_linear_keep_move_calibration_go"
            if all(checks.values())
            else "groupwise_linear_keep_move_calibration_no_go"
        ),
        "checks": checks,
        "model_type": MODEL_TYPE,
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": file_sha256(CHECKPOINT),
        "parameter_count": model.parameter_count,
        "train_groups": observed,
        "training": training,
        "calibration": calibration,
        "protocol": {
            "features": "same fixed seven-dimensional P16 input",
            "objective": "per-source top8 maximum risk",
            "batch_groups": {
                "rescue": BATCH_RESCUE,
                "keep": BATCH_KEEP,
                "unsupported": BATCH_UNSUPPORTED,
            },
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
