#!/usr/bin/env python3
"""Train real-source KEEP/MOVE with complete top128 competition context."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import torch

from chemvector.components.recovery.pairwise_structural_counterfactual import (
    IMAGE_SIZE,
    PAIR_RESCUE,
    PairwiseStructuralCounterfactual,
    PairwiseStructuralCounterfactualConfig,
    pairwise_counterfactual_loss,
)
from chemvector.evaluation.audits.atoms import build_g3_p14e0_top128_source_context as p14e0
from chemvector.g3_structural_data import file_sha256
from chemvector.training.atoms import train_g3_p6e9_structural_residual_arms as p6e9
from chemvector.training.atoms import train_g3_p7_pairwise_counterfactual as p7
from chemvector.training.atoms import train_g3_p11e1_predicted_source_keep_move as base


MODEL_TYPE = "chemvector-g3-p14-full-top128-context-keep-move-fit562"
OUTPUT = Path("outputs/checkpoints") / MODEL_TYPE
CHECKPOINT = OUTPUT / "step-006000.pt"
SOURCE_CACHE = base.CACHE
SOURCE_CACHE_SHA256 = base.CACHE_SHA256
CONTEXT_CACHE = p14e0.CACHE
CONTEXT_CACHE_SHA256 = (
    "282ea57292f71c450f017981502c2b31ff7f478372dc4a47b52d01696dde9e40"
)
MODEL_SEED = 2026090001
SCHEDULE_SEED = 2026090011
CAPACITY_STEPS = 1_500
FULL_STEPS = 6_000
EXPECTED_DIRECTED_PER_CLASS = 281
MINIMUM_CALIBRATION_RESCUES = 20


def _gather(cache, context, table, pair_rows, device):
    batch = base._gather(cache, table, pair_rows, device)
    source_rows = table["source_row"][pair_rows]
    batch.update(
        {
            "global_offsets": context["set_offsets"][source_rows].to(device).float(),
            "global_scores": context["set_scores"][source_rows].to(device).float(),
            "global_source_mask": context["set_source_mask"][source_rows].to(device),
        }
    )
    return batch


def _forward(model, batch):
    return model(
        batch["image"],
        batch["score"],
        batch["source"],
        batch["prior"],
        batch["alternative"],
        batch["mask"],
        batch["global_offsets"],
        batch["global_scores"],
        batch["global_source_mask"],
    )


def train_model(cache, context, table, pools, *, steps, model_seed, schedule_seed, device):
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    model = PairwiseStructuralCounterfactual(
        PairwiseStructuralCounterfactualConfig(
            arm="patch-only",
            explicit_score_evidence=True,
            local_peak_set_evidence=True,
            global_candidate_set_evidence=True,
            global_candidate_count=p14e0.SET_TOKENS,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base.LEARNING_RATE, weight_decay=base.WEIGHT_DECAY
    )
    sums = Counter()
    last = {}
    started = time.perf_counter()
    for step, rows in enumerate(base._schedule(pools, steps, schedule_seed), 1):
        batch = _gather(cache, context, table, rows, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            delta = _forward(model, batch)
            losses = pairwise_counterfactual_loss(
                delta,
                torch.zeros_like(delta),
                batch["labels"],
                batch["mask"],
            )
        losses["loss"].backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last = {
            name: float(value.detach())
            for name, value in losses.items()
            if name != "corrected_margin"
        } | {"gradient_norm": float(gradient)}
        for name, value in last.items():
            sums[name] += value
        if step % 1000 == 0 or step == steps:
            print(json.dumps({"step": step, **last}), flush=True)
    return model, {
        "steps": steps,
        "elapsed_seconds": time.perf_counter() - started,
        "mean": {name: value / steps for name, value in sums.items()},
        "last": last,
    }


@torch.no_grad()
def pair_accuracy(model, cache, context, table, rows, device):
    model.eval()
    correct = total = 0
    for start in range(0, len(rows), base.BATCH_SIZE):
        batch = _gather(
            cache, context, table, rows[start : start + base.BATCH_SIZE], device
        )
        expected = batch["labels"] == PAIR_RESCUE
        correct += int(((_forward(model, batch) > 0) == expected).sum())
        total += len(expected)
    return {"correct": correct, "total": total, "accuracy": correct / total}


@torch.no_grad()
def calibration_records(model, cache, context, table, device):
    model.eval()
    records = []
    for source_row in table["source_rows"].tolist():
        alternatives = torch.where(cache["candidate_mask"][source_row])[0]
        count = len(alternatives)
        offsets = cache["candidate_offsets"][source_row].to(device).float()
        global_offsets = context["set_offsets"][source_row : source_row + 1].to(device).float().expand(count, -1, -1)
        global_scores = context["set_scores"][source_row : source_row + 1].to(device).float().expand(count, -1)
        global_source_mask = context["set_source_mask"][source_row : source_row + 1].to(device).expand(count, -1)
        delta = model(
            cache["image_crops"][source_row : source_row + 1].to(device).float().expand(count, -1, -1, -1),
            cache["score_crops"][source_row : source_row + 1].to(device).float().expand(count, -1, -1, -1),
            torch.zeros(count, 2, device=device),
            torch.zeros(count, 2, device=device),
            offsets[alternatives.to(device)],
            torch.ones(count, dtype=torch.bool, device=device),
            global_offsets,
            global_scores,
            global_source_mask,
        ).float().cpu()
        winner = int(delta.argmax())
        target = cache["offset_targets"][source_row].float()
        prior_good = float(torch.linalg.vector_norm(target * IMAGE_SIZE)) <= 6.0
        alternative_good = float(
            torch.linalg.vector_norm(
                (
                    cache["candidate_offsets"][source_row, int(alternatives[winner])].float()
                    - target
                )
                * IMAGE_SIZE
            )
        ) <= 6.0
        records.append(
            {
                "source_row": source_row,
                "score": float(delta[winner]),
                "prior_good": prior_good,
                "alternative_good": alternative_good,
                "action": cache["action"][source_row],
            }
        )
    return records


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P14 output already exists: {OUTPUT}")
    for path, expected in (
        (SOURCE_CACHE, SOURCE_CACHE_SHA256),
        (CONTEXT_CACHE, CONTEXT_CACHE_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P14 cache SHA256 differs: {path}")
    if any((path.stat().st_mode & 0o777) != 0 for path in p6e9.HOLDOUT_FILES):
        raise RuntimeError("P14 historical holdout must remain sealed")
    if not torch.cuda.is_available():
        raise RuntimeError("P14 requires CUDA")

    device = torch.device("cuda")
    cache = torch.load(SOURCE_CACHE, map_location="cpu", weights_only=False)
    context = torch.load(CONTEXT_CACHE, map_location="cpu", weights_only=False)
    if context["sample_key"] != cache["sample_key"]:
        raise RuntimeError("P14 context and source rows differ")
    train_table = base.build_pair_table(cache, "train")
    calibration_table = base.build_pair_table(cache, "calibration")
    pools, pool_audit = base.select_stratified_pools(train_table)
    if (
        len(pools["rescue"]) != EXPECTED_DIRECTED_PER_CLASS
        or len(pools["harmful"]) != EXPECTED_DIRECTED_PER_CLASS
    ):
        raise RuntimeError(f"P14 pool differs: {pool_audit}")

    capacity_pools = {name: value[:32] for name, value in pools.items()}
    capacity_model, capacity_training = train_model(
        cache,
        context,
        train_table,
        capacity_pools,
        steps=CAPACITY_STEPS,
        model_seed=MODEL_SEED,
        schedule_seed=SCHEDULE_SEED - 1,
        device=device,
    )
    capacity_rows = torch.cat(
        (capacity_pools["rescue"], capacity_pools["harmful"])
    )
    capacity = pair_accuracy(
        capacity_model, cache, context, train_table, capacity_rows, device
    )
    if capacity["accuracy"] < 0.99:
        OUTPUT.mkdir(parents=True)
        audit = {
            "schema_version": "g3-p14-full-top128-context-capacity-audit-0.1",
            "decision": "full_top128_context_capacity_no_go",
            "capacity": capacity,
            "capacity_training": capacity_training,
            "holdout_read": False,
        }
        p7._atomic_json(OUTPUT / "audit.json", audit)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return

    del capacity_model
    model, training = train_model(
        cache,
        context,
        train_table,
        pools,
        steps=FULL_STEPS,
        model_seed=MODEL_SEED,
        schedule_seed=SCHEDULE_SEED,
        device=device,
    )
    records = calibration_records(
        model, cache, context, calibration_table, device
    )
    calibration = base.calibrate_nonnegative(records)
    OUTPUT.mkdir(parents=True)
    p7._atomic_torch_save(
        CHECKPOINT,
        {
            "checkpoint_schema_version": "g3-p14-full-top128-context-keep-move-0.1",
            "model_type": MODEL_TYPE,
            "model_config": model.config_dict(),
            "model": model.state_dict(),
            "global_step": FULL_STEPS,
            "threshold": calibration["threshold"],
            "model_seed": MODEL_SEED,
            "schedule_seed": SCHEDULE_SEED,
        },
    )
    checks = {
        "capacity_accuracy_min099": capacity["accuracy"] >= 0.99,
        "global_candidate_count_129": model.config.global_candidate_count == 129,
        "train_281_each": len(pools["rescue"])
        == len(pools["harmful"])
        == EXPECTED_DIRECTED_PER_CLASS,
        "calibration_zero_harmful": calibration["harmful"] == 0,
        "calibration_rescues_min20": calibration["rescues"]
        >= MINIMUM_CALIBRATION_RESCUES,
        "threshold_nonnegative": calibration["threshold"] >= 0.0,
        "historical_holdout_mode_000": all(
            (path.stat().st_mode & 0o777) == 0 for path in p6e9.HOLDOUT_FILES
        ),
    }
    audit = {
        "schema_version": "g3-p14-full-top128-context-keep-move-audit-0.1",
        "decision": (
            "full_top128_context_keep_move_calibration_go"
            if all(checks.values())
            else "full_top128_context_keep_move_calibration_no_go"
        ),
        "checks": checks,
        "model_type": MODEL_TYPE,
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": file_sha256(CHECKPOINT),
        "parameter_count": model.parameter_count,
        "capacity": capacity,
        "capacity_training": capacity_training,
        "pool": pool_audit,
        "training": training,
        "calibration": calibration,
        "protocol": {
            "baseline": "P13 local peak-set KEEP/MOVE",
            "new_evidence": "replayed P4-E3 top128 plus explicit source token",
            "source_outside_replayed_top128": 29,
            "full80k_context_scan": False,
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
