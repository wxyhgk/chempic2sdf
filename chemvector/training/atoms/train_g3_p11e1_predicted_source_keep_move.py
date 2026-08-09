#!/usr/bin/env python3
"""Train an explicit KEEP-vs-MOVE model on real predicted Atom sources."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from chemvector.components.recovery.pairwise_structural_counterfactual import (
    IMAGE_SIZE,
    PAIR_EQUIVALENT,
    PAIR_HARMFUL,
    PAIR_RESCUE,
    PairwiseStructuralCounterfactual,
    PairwiseStructuralCounterfactualConfig,
    pairwise_counterfactual_loss,
)
from chemvector.evaluation.audits.atoms import build_g3_p11e0_predicted_source_pilot as p11e0
from chemvector.g3_structural_data import file_sha256
from chemvector.training.atoms import train_g3_p6e9_structural_residual_arms as p6e9
from chemvector.training.atoms import train_g3_p7_pairwise_counterfactual as p7


MODEL_TYPE = "chemvector-g3-p11e1-predicted-source-keep-move-fit562"
OUTPUT = Path("outputs/checkpoints") / MODEL_TYPE
CHECKPOINT = OUTPUT / "step-006000.pt"
CACHE = p11e0.CACHE
CACHE_SHA256 = "b84e200c81c02e5f93b4a8c0e51f9d1eae5decae9b71c70742338a4437ea38ee"
MODEL_SEED = 2026089501
SCHEDULE_SEED = 2026089511
CAPACITY_STEPS = 1_500
FULL_STEPS = 6_000
BATCH_SIZE = 64
EXPECTED_DIRECTED_PER_CLASS = 281
EQUIVALENT_POOL = 512
MINIMUM_CALIBRATION_RESCUES = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4


def scaffold_split(scaffold: str) -> str:
    value = int.from_bytes(hashlib.sha256(f"p11-split:{scaffold}".encode()).digest()[:8], "big") % 5
    return "calibration" if value == 0 else "train"


def _distance_band(pixels: float) -> int:
    return 0 if pixels < 8.0 else 1 if pixels < 16.0 else 2


def build_pair_table(cache: dict, split: str | None) -> dict:
    rows = [
        index
        for index, scaffold in enumerate(cache["scaffold"])
        if split is None or scaffold_split(str(scaffold)) == split
    ]
    output: dict[str, list] = defaultdict(list)
    offsets = cache["candidate_offsets"].float()
    targets = cache["offset_targets"].float()
    masks = cache["candidate_mask"]
    distances = torch.linalg.vector_norm((offsets - targets[:, None]) * IMAGE_SIZE, dim=-1)
    source_distances = torch.linalg.vector_norm(targets * IMAGE_SIZE, dim=-1)
    for source_row in rows:
        source_good = float(source_distances[source_row]) <= 6.0
        for alternative in torch.where(masks[source_row])[0].tolist():
            alternative_good = float(distances[source_row, alternative]) <= 6.0
            if not source_good and alternative_good:
                label = PAIR_RESCUE
            elif source_good and not alternative_good:
                label = PAIR_HARMFUL
            elif source_good and alternative_good:
                label = PAIR_EQUIVALENT
            else:
                continue
            separation = float(torch.linalg.vector_norm(offsets[source_row, alternative] * IMAGE_SIZE))
            output["source_row"].append(source_row)
            output["alternative_index"].append(alternative)
            output["label"].append(label)
            output["stratum"].append(
                (
                    str(cache["target_role"][source_row]),
                    str(cache["variant"][source_row]),
                    str(cache["degree_bucket"][source_row]),
                    _distance_band(separation),
                )
            )
    return {
        "source_row": torch.tensor(output["source_row"], dtype=torch.long),
        "alternative_index": torch.tensor(output["alternative_index"], dtype=torch.long),
        "label": torch.tensor(output["label"], dtype=torch.long),
        "stratum": output["stratum"],
        "source_rows": torch.tensor(rows, dtype=torch.long),
    }


def select_stratified_pools(
    table: dict, *, equivalent_pool: int = EQUIVALENT_POOL
) -> tuple[dict[str, torch.Tensor], dict]:
    rescue: dict[tuple, list[int]] = defaultdict(list)
    harmful: dict[tuple, list[int]] = defaultdict(list)
    equivalent = []
    for index, (label, stratum) in enumerate(zip(table["label"].tolist(), table["stratum"], strict=True)):
        if label == PAIR_RESCUE:
            rescue[stratum].append(index)
        elif label == PAIR_HARMFUL:
            harmful[stratum].append(index)
        else:
            equivalent.append(index)
    selected_rescue = []
    selected_harmful = []
    strata = {}
    for stratum in sorted(rescue):
        positive = sorted(rescue[stratum], key=lambda value: hashlib.sha256(f"p11:r:{value}".encode()).hexdigest())
        negative = sorted(harmful[stratum], key=lambda value: hashlib.sha256(f"p11:h:{value}".encode()).hexdigest())
        count = min(len(positive), len(negative))
        selected_rescue.extend(positive[:count])
        selected_harmful.extend(negative[:count])
        strata[str(stratum)] = {"rescue": len(positive), "harmful": len(negative), "selected_each": count}
    equivalent.sort(key=lambda value: hashlib.sha256(f"p11:e:{value}".encode()).hexdigest())
    pools = {
        "rescue": torch.tensor(selected_rescue, dtype=torch.long),
        "harmful": torch.tensor(selected_harmful, dtype=torch.long),
        "equivalent": torch.tensor(equivalent[:equivalent_pool], dtype=torch.long),
    }
    return pools, {"strata": strata, **{name: len(value) for name, value in pools.items()}}


def _gather(cache: dict, table: dict, pair_rows: torch.Tensor, device: torch.device) -> dict:
    source_rows = table["source_row"][pair_rows]
    alternatives = table["alternative_index"][pair_rows]
    offsets = cache["candidate_offsets"][source_rows].to(device).float()
    local = torch.arange(len(pair_rows), device=device)
    return {
        "image": cache["image_crops"][source_rows].to(device).float(),
        "score": cache["score_crops"][source_rows].to(device).float(),
        "source": torch.zeros(len(pair_rows), 2, device=device),
        "prior": torch.zeros(len(pair_rows), 2, device=device),
        "alternative": offsets[local, alternatives.to(device)],
        "mask": torch.ones(len(pair_rows), dtype=torch.bool, device=device),
        "labels": table["label"][pair_rows].to(device),
    }


def _schedule(pools: dict[str, torch.Tensor], steps: int, seed: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    sizes = {"rescue": 28, "harmful": 28, "equivalent": 8}
    output = []
    for _ in range(steps):
        current = []
        for name in ("rescue", "harmful", "equivalent"):
            pool = pools[name]
            positions = torch.randint(len(pool), (sizes[name],), generator=generator)
            current.append(pool[positions])
        rows = torch.cat(current)
        output.append(rows[torch.randperm(BATCH_SIZE, generator=generator)])
    return output


def train_model(
    cache: dict,
    table: dict,
    pools: dict[str, torch.Tensor],
    *,
    steps: int,
    model_seed: int,
    schedule_seed: int,
    device: torch.device,
    explicit_score_evidence: bool = False,
) -> tuple[PairwiseStructuralCounterfactual, dict]:
    torch.manual_seed(model_seed)
    torch.cuda.manual_seed_all(model_seed)
    model = PairwiseStructuralCounterfactual(
        PairwiseStructuralCounterfactualConfig(
            arm="patch-only",
            explicit_score_evidence=explicit_score_evidence,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sums = Counter()
    started = time.perf_counter()
    last = {}
    for step, rows in enumerate(_schedule(pools, steps, schedule_seed), 1):
        batch = _gather(cache, table, rows, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            delta = model(batch["image"], batch["score"], batch["source"], batch["prior"], batch["alternative"], batch["mask"])
            losses = pairwise_counterfactual_loss(
                delta, torch.zeros_like(delta), batch["labels"], batch["mask"]
            )
        losses["loss"].backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last = {name: float(value.detach()) for name, value in losses.items() if name != "corrected_margin"} | {"gradient_norm": float(gradient)}
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
def pair_accuracy(model, cache, table, rows, device) -> dict:
    correct = total = 0
    model.eval()
    for start in range(0, len(rows), BATCH_SIZE):
        batch = _gather(cache, table, rows[start : start + BATCH_SIZE], device)
        delta = model(batch["image"], batch["score"], batch["source"], batch["prior"], batch["alternative"], batch["mask"])
        expected = batch["labels"] == PAIR_RESCUE
        correct += int(((delta > 0) == expected).sum())
        total += len(delta)
    return {"correct": correct, "total": total, "accuracy": correct / total}


@torch.no_grad()
def calibration_records(model, cache: dict, table: dict, device: torch.device) -> list[dict]:
    model.eval()
    records = []
    for source_row in table["source_rows"].tolist():
        alternatives = torch.where(cache["candidate_mask"][source_row])[0]
        count = len(alternatives)
        offsets = cache["candidate_offsets"][source_row].to(device).float()
        image = cache["image_crops"][source_row : source_row + 1].to(device).float().expand(count, -1, -1, -1)
        score = cache["score_crops"][source_row : source_row + 1].to(device).float().expand(count, -1, -1, -1)
        alternative_offsets = offsets[alternatives.to(device)]
        delta = model(
            image,
            score,
            torch.zeros(count, 2, device=device),
            torch.zeros(count, 2, device=device),
            alternative_offsets,
            torch.ones(count, dtype=torch.bool, device=device),
        ).float().cpu()
        winner = int(delta.argmax())
        target = cache["offset_targets"][source_row].float()
        source_good = float(torch.linalg.vector_norm(target * IMAGE_SIZE)) <= 6.0
        alternative_good = float(
            torch.linalg.vector_norm(
                (cache["candidate_offsets"][source_row, int(alternatives[winner])].float() - target) * IMAGE_SIZE
            )
        ) <= 6.0
        records.append(
            {
                "source_row": source_row,
                "score": float(delta[winner]),
                "prior_good": source_good,
                "alternative_good": alternative_good,
                "action": cache["action"][source_row],
            }
        )
    return records


def calibrate_nonnegative(records: list[dict]) -> dict:
    thresholds = sorted({float(row["score"]) for row in records if float(row["score"]) >= 0.0}, reverse=True)
    candidates = []
    for threshold in [math.inf, *thresholds]:
        executed = [row for row in records if row["score"] >= threshold]
        harmful = sum(row["prior_good"] and not row["alternative_good"] for row in executed)
        rescue = sum(not row["prior_good"] and row["alternative_good"] for row in executed)
        candidates.append(
            {
                "threshold": threshold,
                "executed": len(executed),
                "rescues": rescue,
                "harmful": harmful,
                "useless": len(executed) - rescue - harmful,
            }
        )
    safe = [row for row in candidates if row["harmful"] == 0]
    return max(safe, key=lambda row: (row["rescues"], -row["executed"], row["threshold"]))


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P11-E1 output already exists: {OUTPUT}")
    if file_sha256(CACHE) != CACHE_SHA256:
        raise ValueError("P11-E1 cache SHA256 differs")
    if any((path.stat().st_mode & 0o777) != 0 for path in p6e9.HOLDOUT_FILES):
        raise RuntimeError("P11-E1 historical holdout must remain sealed")
    if not torch.cuda.is_available():
        raise RuntimeError("P11-E1 requires CUDA")
    device = torch.device("cuda")
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    train_table = build_pair_table(cache, "train")
    calibration_table = build_pair_table(cache, "calibration")
    pools, pool_audit = select_stratified_pools(train_table)
    if len(pools["rescue"]) != EXPECTED_DIRECTED_PER_CLASS or len(pools["harmful"]) != EXPECTED_DIRECTED_PER_CLASS:
        raise RuntimeError(f"P11-E1 directed pool differs: {pool_audit}")
    capacity_pools = {name: value[:32] for name, value in pools.items()}
    capacity_model, capacity_training = train_model(
        cache,
        train_table,
        capacity_pools,
        steps=CAPACITY_STEPS,
        model_seed=MODEL_SEED,
        schedule_seed=SCHEDULE_SEED - 1,
        device=device,
    )
    capacity_rows = torch.cat((capacity_pools["rescue"], capacity_pools["harmful"]))
    capacity = pair_accuracy(capacity_model, cache, train_table, capacity_rows, device)
    if capacity["accuracy"] < 0.99:
        OUTPUT.mkdir(parents=True)
        audit = {
            "schema_version": "g3-p11e1-keep-move-capacity-audit-0.1",
            "decision": "predicted_source_keep_move_capacity_no_go",
            "capacity": capacity,
            "capacity_training": capacity_training,
            "pool": pool_audit,
            "historical_holdout_read": False,
        }
        p7._atomic_json(OUTPUT / "audit.json", audit)
        print(json.dumps(audit, indent=2))
        return
    del capacity_model
    model, training = train_model(
        cache,
        train_table,
        pools,
        steps=FULL_STEPS,
        model_seed=MODEL_SEED,
        schedule_seed=SCHEDULE_SEED,
        device=device,
    )
    records = calibration_records(model, cache, calibration_table, device)
    calibration = calibrate_nonnegative(records)
    OUTPUT.mkdir(parents=True)
    p7._atomic_torch_save(
        CHECKPOINT,
        {
            "checkpoint_schema_version": "g3-p11e1-predicted-source-keep-move-0.1",
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
        "train_281_each": len(pools["rescue"]) == len(pools["harmful"]) == EXPECTED_DIRECTED_PER_CLASS,
        "calibration_zero_harmful": calibration["harmful"] == 0,
        "calibration_rescues_min20": calibration["rescues"] >= MINIMUM_CALIBRATION_RESCUES,
        "threshold_nonnegative": calibration["threshold"] >= 0.0,
        "historical_holdout_mode_000": all((path.stat().st_mode & 0o777) == 0 for path in p6e9.HOLDOUT_FILES),
    }
    audit = {
        "schema_version": "g3-p11e1-predicted-source-keep-move-audit-0.1",
        "decision": "predicted_source_keep_move_calibration_go" if all(checks.values()) else "predicted_source_keep_move_calibration_no_go",
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
        "calibration_sources": Counter(cache["action"][index] for index in calibration_table["source_rows"].tolist()),
        "protocol": {
            "prior": "explicit zero-offset KEEP",
            "alternative": "one frozen P6-E1 top8 MOVE",
            "peak_margin_used": False,
            "historical_holdout_read": False,
            "confirm_external_sealed_read": False,
        },
        "historical_holdout_read": False,
        "confirm_external_sealed_read": False,
    }
    audit["calibration_sources"] = dict(audit["calibration_sources"])
    p7._atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
