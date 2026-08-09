#!/usr/bin/env python3
"""Run P7 capacity, OOF, two-arm training, and calibration selection."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch

from chemvector.components.recovery.pairwise_structural_counterfactual import (
    IMAGE_SIZE,
    PAIR_EQUIVALENT,
    PAIR_HARMFUL,
    PAIR_RESCUE,
    PAIRWISE_ARMS,
    PairwiseStructuralCounterfactual,
    PairwiseStructuralCounterfactualConfig,
    pairwise_counterfactual_loss,
)
from chemvector.components.recovery.structural_peak_residual import (
    normalized_peak_prior,
)
from chemvector.evaluation.audits.atoms import (
    build_g3_p7e0_pairwise_counterfactual_data as p7e0,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.training.atoms import train_g3_p6e9_structural_residual_arms as p6e9


CYCLE_NAME = "chemvector-g3-p7-pairwise-structural-counterfactual"
OUTPUT = Path("outputs/checkpoints") / CYCLE_NAME
CAPACITY_OUTPUT = OUTPUT / "capacity"
SEED = 20260891
MODEL_SEED = 2026089101
FOLDS = 5
BATCH_SIZE = 64
CAPACITY_DIRECTED = 256
CAPACITY_STEPS = 2_000
CAPACITY_ACCURACY = 0.99
OOF_STEPS = 3_000
FULL_STEPS = 8_000
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
EQUIVALENT_POOL = 2_048
MINIMUM_RESCUES = 100
PAIR_SHA256 = {
    p7e0.TRAIN_PAIRS: "ee39aa08f1324cd3f22ed9f082848c5e00c86795119971523320346dcad52eca",
    p7e0.CALIBRATION_PAIRS: "0f1d05da3653d49acbfc4dcf960749886d777eaedaeb82fd0da2663274103a20",
}


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def scaffold_fold(scaffold: str) -> int:
    digest = hashlib.sha256(f"p7-oof:{scaffold}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % FOLDS


def _stable_rank(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def _load_sources() -> dict[str, tuple[dict, dict]]:
    return {
        "train": (
            torch.load(p6e9.TRAIN_HARD, map_location="cpu", weights_only=False),
            torch.load(p6e9.TRAIN_EASY, map_location="cpu", weights_only=False),
        ),
        "calibration": (
            torch.load(
                p6e9.CALIBRATION_HARD, map_location="cpu", weights_only=False
            ),
            torch.load(
                p6e9.CALIBRATION_EASY, map_location="cpu", weights_only=False
            ),
        ),
    }


def _base_margin_for_pair(
    cache: dict, row: int, prior_index: int, alternative_index: int
) -> float:
    logits = cache["candidate_peak_logits"][row : row + 1].float()
    mask = cache["candidate_mask"][row : row + 1]
    prior = normalized_peak_prior(logits, mask)[0]
    return float(prior[alternative_index] - prior[prior_index])


def select_training_pairs(pairs: dict, sources: tuple[dict, dict]) -> dict:
    label = pairs["label"]
    rescue = []
    for index in torch.where(label == PAIR_RESCUE)[0].tolist():
        kind = int(pairs["source_kind"][index])
        margin = _base_margin_for_pair(
            sources[kind],
            int(pairs["source_row"][index]),
            int(pairs["prior_index"][index]),
            int(pairs["alternative_index"][index]),
        )
        if margin >= -1.95:
            rescue.append(index)
    harmful_all = torch.where(label == PAIR_HARMFUL)[0].tolist()
    harmful_all.sort(
        key=lambda index: _stable_rank(
            ":".join(
                (
                    pairs["scaffold"][index],
                    pairs["variant"][index],
                    pairs["target_role"][index],
                    str(int(pairs["source_row"][index])),
                    str(int(pairs["alternative_index"][index])),
                )
            )
        )
    )
    harmful = harmful_all[: len(rescue)]
    equivalent = torch.where(label == PAIR_EQUIVALENT)[0].tolist()
    equivalent.sort(
        key=lambda index: _stable_rank(
            f"equivalent:{pairs['scaffold'][index]}:{int(pairs['source_row'][index])}:{int(pairs['alternative_index'][index])}"
        )
    )
    equivalent = equivalent[:EQUIVALENT_POOL]
    return {
        "rescue": torch.tensor(rescue, dtype=torch.long),
        "harmful": torch.tensor(harmful, dtype=torch.long),
        "equivalent": torch.tensor(equivalent, dtype=torch.long),
    }


def _gather_sources(
    pairs: dict,
    pair_rows: torch.Tensor,
    sources: tuple[dict, dict],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    count = len(pair_rows)
    image = torch.empty(count, 1, 96, 96, dtype=torch.float32, device=device)
    score = torch.empty_like(image)
    prior_offsets = torch.empty(count, 2, dtype=torch.float32, device=device)
    alternative_offsets = torch.empty_like(prior_offsets)
    prior_margin = torch.empty(count, dtype=torch.float32, device=device)
    labels = pairs["label"][pair_rows].to(device)
    for kind in (p7e0.SOURCE_HARD, p7e0.SOURCE_EASY):
        positions = torch.where(pairs["source_kind"][pair_rows] == kind)[0]
        if not len(positions):
            continue
        selected_pairs = pair_rows[positions]
        source_rows = pairs["source_row"][selected_pairs]
        prior_indices = pairs["prior_index"][selected_pairs]
        alternative_indices = pairs["alternative_index"][selected_pairs]
        source = sources[kind]
        image[positions] = source["image_crops"][source_rows].to(device).float()
        score[positions] = source["score_crops"][source_rows].to(device).float()
        offsets = source["candidate_offsets"][source_rows].to(device).float()
        local = torch.arange(len(positions), device=device)
        prior_offsets[positions] = offsets[local, prior_indices.to(device)]
        alternative_offsets[positions] = offsets[
            local, alternative_indices.to(device)
        ]
        peak_logits = source["candidate_peak_logits"][source_rows].to(device).float()
        candidate_mask = source["candidate_mask"][source_rows].to(device)
        normalized = normalized_peak_prior(peak_logits, candidate_mask)
        prior_margin[positions] = normalized[
            local, alternative_indices.to(device)
        ] - normalized[local, prior_indices.to(device)]
    return {
        "image": image,
        "score": score,
        "source": torch.zeros_like(prior_offsets),
        "prior": prior_offsets,
        "alternative": alternative_offsets,
        "mask": torch.ones(count, dtype=torch.bool, device=device),
        "prior_margin": prior_margin,
        "labels": labels,
    }


def _schedule(
    pools: dict[str, torch.Tensor], steps: int, seed: int
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    sizes = {"rescue": 28, "harmful": 28, "equivalent": 8}
    output = []
    for _ in range(steps):
        parts = []
        for name in ("rescue", "harmful", "equivalent"):
            pool = pools[name]
            if not len(pool):
                raise ValueError(f"empty P7 training pool: {name}")
            positions = torch.randint(len(pool), (sizes[name],), generator=generator)
            parts.append(pool[positions])
        rows = torch.cat(parts)
        output.append(rows[torch.randperm(BATCH_SIZE, generator=generator)])
    return output


def _train_model(
    arm: str,
    pairs: dict,
    sources: tuple[dict, dict],
    pools: dict[str, torch.Tensor],
    *,
    steps: int,
    seed: int,
    device: torch.device,
) -> tuple[PairwiseStructuralCounterfactual, dict]:
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    model = PairwiseStructuralCounterfactual(
        PairwiseStructuralCounterfactualConfig(arm=arm)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    schedule = _schedule(pools, steps, seed)
    sums: Counter = Counter()
    last = {}
    started = time.perf_counter()
    model.train()
    for step, rows in enumerate(schedule, 1):
        batch = _gather_sources(pairs, rows, sources, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            delta = model(
                batch["image"],
                batch["score"],
                batch["source"],
                batch["prior"],
                batch["alternative"],
                batch["mask"],
            )
            losses = pairwise_counterfactual_loss(
                delta,
                batch["prior_margin"],
                batch["labels"],
                batch["mask"],
            )
        losses["loss"].backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        if not bool(torch.isfinite(gradient)):
            raise FloatingPointError("P7 gradient is not finite")
        optimizer.step()
        last = {
            name: float(value.detach())
            for name, value in losses.items()
            if name != "corrected_margin"
        } | {"gradient_norm": float(gradient.detach())}
        for name, value in last.items():
            sums[name] += value
        if step % 1000 == 0 or step == steps:
            print(json.dumps({"arm": arm, "step": step, **last}), flush=True)
    return model, {
        "steps": steps,
        "elapsed_seconds": time.perf_counter() - started,
        "mean": {name: value / steps for name, value in sums.items()},
        "last": last,
    }


@torch.no_grad()
def _pair_accuracy(
    model: PairwiseStructuralCounterfactual,
    pairs: dict,
    sources: tuple[dict, dict],
    rows: torch.Tensor,
    device: torch.device,
) -> dict:
    correct = total = 0
    model.eval()
    for start in range(0, len(rows), BATCH_SIZE):
        batch = _gather_sources(pairs, rows[start : start + BATCH_SIZE], sources, device)
        delta = model(
            batch["image"],
            batch["score"],
            batch["source"],
            batch["prior"],
            batch["alternative"],
            batch["mask"],
        )
        predicted = batch["prior_margin"] + delta
        expected = batch["labels"] == PAIR_RESCUE
        correct += int(((predicted > 0) == expected).sum())
        total += len(predicted)
    return {"correct": correct, "total": total, "accuracy": correct / total}


def _fold_pools(
    pools: dict[str, torch.Tensor], pairs: dict, held_out: int
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    train = {}
    validation = []
    for name, rows in pools.items():
        folds = torch.tensor(
            [scaffold_fold(pairs["scaffold"][index]) for index in rows.tolist()]
        )
        train[name] = rows[folds != held_out]
        if name != "equivalent":
            validation.append(rows[folds == held_out])
    return train, torch.cat(validation)


def _capacity_rows(
    pools: dict[str, torch.Tensor], pairs: dict
) -> dict[str, torch.Tensor]:
    directed = {}
    for name in ("rescue", "harmful"):
        unique = []
        seen = set()
        for index in pools[name].tolist():
            key = (int(pairs["source_kind"][index]), int(pairs["source_row"][index]))
            if key in seen:
                continue
            seen.add(key)
            unique.append(index)
            if len(unique) == CAPACITY_DIRECTED // 2:
                break
        directed[name] = torch.tensor(unique, dtype=torch.long)
    directed["equivalent"] = pools["equivalent"][: max(8, CAPACITY_DIRECTED // 8)]
    return directed


@torch.no_grad()
def evaluate_source_cache(
    model: PairwiseStructuralCounterfactual,
    cache: dict,
    source_kind: str,
    device: torch.device,
) -> list[dict]:
    model.eval()
    flat = []
    for row in range(len(cache["offset_targets"])):
        mask = cache["candidate_mask"][row]
        logits = cache["candidate_peak_logits"][row : row + 1].float()
        normalized = normalized_peak_prior(logits, mask[None])[0]
        prior_index = int(normalized.argmax())
        for alternative in torch.where(mask)[0].tolist():
            if alternative != prior_index:
                flat.append((row, prior_index, alternative, float(normalized[alternative] - normalized[prior_index])))
    best: dict[int, dict] = {}
    for start in range(0, len(flat), BATCH_SIZE):
        chunk = flat[start : start + BATCH_SIZE]
        rows = torch.tensor([value[0] for value in chunk])
        prior_indices = torch.tensor([value[1] for value in chunk])
        alternative_indices = torch.tensor([value[2] for value in chunk])
        image = cache["image_crops"][rows].to(device).float()
        score = cache["score_crops"][rows].to(device).float()
        offsets = cache["candidate_offsets"][rows].to(device).float()
        local = torch.arange(len(rows), device=device)
        prior_offsets = offsets[local, prior_indices.to(device)]
        alternative_offsets = offsets[local, alternative_indices.to(device)]
        delta = model(
            image,
            score,
            torch.zeros_like(prior_offsets),
            prior_offsets,
            alternative_offsets,
            torch.ones(len(rows), dtype=torch.bool, device=device),
        ).float().cpu()
        for position, (row, prior_index, alternative, margin) in enumerate(chunk):
            corrected = margin + float(delta[position])
            if row not in best or corrected > best[row]["score"]:
                target = cache["offset_targets"][row].float()
                candidate_offsets = cache["candidate_offsets"][row].float()
                prior_distance = float(
                    torch.linalg.vector_norm(
                        (candidate_offsets[prior_index] - target) * IMAGE_SIZE
                    )
                )
                alternative_distance = float(
                    torch.linalg.vector_norm(
                        (candidate_offsets[alternative] - target) * IMAGE_SIZE
                    )
                )
                best[row] = {
                    "source": source_kind,
                    "row": row,
                    "score": corrected,
                    "prior_good": prior_distance <= 6.0,
                    "alternative_good": alternative_distance <= 6.0,
                    "prior_index": prior_index,
                    "alternative_index": alternative,
                }
    return [best[row] for row in sorted(best)]


def calibrate_zero_harmful(records: list[dict]) -> dict:
    thresholds = sorted({float(row["score"]) for row in records}, reverse=True)
    candidates = []
    for threshold in [math.inf, *thresholds]:
        executed = [row for row in records if row["score"] >= threshold]
        harmful = sum(row["prior_good"] and not row["alternative_good"] for row in executed)
        rescue = sum(not row["prior_good"] and row["alternative_good"] for row in executed)
        useless = len(executed) - harmful - rescue
        if harmful == 0:
            candidates.append(
                {
                    "threshold": threshold,
                    "executed": len(executed),
                    "rescues": rescue,
                    "harmful": harmful,
                    "useless": useless,
                }
            )
    return max(
        candidates,
        key=lambda row: (row["rescues"], -row["executed"], row["threshold"]),
    )


def _run_arm(
    arm: str,
    pairs: dict,
    train_sources: tuple[dict, dict],
    calibration_sources: tuple[dict, dict],
    pools: dict[str, torch.Tensor],
    device: torch.device,
) -> dict:
    oof = []
    for fold in range(FOLDS):
        fold_pools, validation = _fold_pools(pools, pairs, fold)
        model, training = _train_model(
            arm,
            pairs,
            train_sources,
            fold_pools,
            steps=OOF_STEPS,
            seed=SEED + fold,
            device=device,
        )
        oof.append(
            {
                "fold": fold,
                "validation": _pair_accuracy(
                    model, pairs, train_sources, validation, device
                ),
                "training": training,
            }
        )
        del model
        torch.cuda.empty_cache()
    model, training = _train_model(
        arm,
        pairs,
        train_sources,
        pools,
        steps=FULL_STEPS,
        seed=SEED + 100,
        device=device,
    )
    records = evaluate_source_cache(
        model, calibration_sources[0], "hard", device
    ) + evaluate_source_cache(model, calibration_sources[1], "easy", device)
    winner = calibrate_zero_harmful(records)
    model_type = f"chemvector-g3-p7e2-pairwise-counterfactual-{arm}-train{2 * len(pools['rescue'])}"
    model_output = Path("outputs/checkpoints") / model_type
    checkpoint = model_output / f"step-{FULL_STEPS:06d}.pt"
    _atomic_torch_save(
        checkpoint,
        {
            "checkpoint_schema_version": "g3-p7-pairwise-counterfactual-0.1",
            "model_type": model_type,
            "model_config": model.config_dict(),
            "model": model.state_dict(),
            "global_step": FULL_STEPS,
            "calibration_threshold": winner["threshold"],
        },
    )
    result = {
        "model_type": model_type,
        "arm": arm,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "parameter_count": model.parameter_count,
        "training": training,
        "oof": oof,
        "calibration": winner,
    }
    _atomic_json(model_output / "audit.json", result)
    return result


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P7 output already exists: {OUTPUT}")
    for path, expected in PAIR_SHA256.items():
        if file_sha256(path) != expected:
            raise ValueError(f"P7 pair cache SHA256 differs: {path}")
    if any((path.stat().st_mode & 0o777) != 0 for path in p6e9.HOLDOUT_FILES):
        raise RuntimeError("P7 holdout cache is not sealed")
    if not torch.cuda.is_available():
        raise RuntimeError("P7 training requires CUDA")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    pairs = torch.load(p7e0.TRAIN_PAIRS, map_location="cpu", weights_only=False)
    sources = _load_sources()
    pools = select_training_pairs(pairs, sources["train"])
    capacity_pools = _capacity_rows(pools, pairs)
    capacity_model, capacity_training = _train_model(
        "patch-band",
        pairs,
        sources["train"],
        capacity_pools,
        steps=CAPACITY_STEPS,
        seed=SEED - 1,
        device=device,
    )
    capacity_rows = torch.cat(
        (capacity_pools["rescue"], capacity_pools["harmful"])
    )
    capacity_metrics = _pair_accuracy(
        capacity_model, pairs, sources["train"], capacity_rows, device
    )
    capacity_go = capacity_metrics["accuracy"] >= CAPACITY_ACCURACY
    capacity_audit = {
        "schema_version": "g3-p7e1-pairwise-capacity-0.1",
        "decision": "pairwise_capacity_go" if capacity_go else "pairwise_capacity_no_go",
        "metrics": capacity_metrics,
        "training": capacity_training,
        "checks": {
            "directed_256": len(capacity_rows) == CAPACITY_DIRECTED,
            "accuracy_min_099": capacity_go,
            "holdout_mode_000": all(
                (path.stat().st_mode & 0o777) == 0 for path in p6e9.HOLDOUT_FILES
            ),
        },
        "holdout_read": False,
    }
    _atomic_json(CAPACITY_OUTPUT / "audit.json", capacity_audit)
    if not all(capacity_audit["checks"].values()):
        _atomic_json(
            OUTPUT / "audit.json",
            {
                "schema_version": "g3-p7-cycle-audit-0.1",
                "decision": "stopped_p7e1_capacity_no_go",
                "capacity": capacity_audit,
                "holdout_read": False,
                "dev_confirm_external_sealed_read": False,
            },
        )
        print(json.dumps(capacity_audit, indent=2))
        return
    del capacity_model
    torch.cuda.empty_cache()
    arms = [
        _run_arm(
            arm,
            pairs,
            sources["train"],
            sources["calibration"],
            pools,
            device,
        )
        for arm in PAIRWISE_ARMS
    ]
    eligible = [
        row
        for row in arms
        if row["calibration"]["harmful"] == 0
        and row["calibration"]["rescues"] >= MINIMUM_RESCUES
    ]
    winner = max(
        eligible,
        key=lambda row: (
            row["calibration"]["rescues"],
            -row["parameter_count"],
        ),
    ) if eligible else None
    checks = {
        "capacity_go": capacity_go,
        "two_arms_completed": len(arms) == 2,
        "winner_zero_harmful": winner is not None,
        "winner_rescues_min100": winner is not None,
        "holdout_mode_000": all(
            (path.stat().st_mode & 0o777) == 0 for path in p6e9.HOLDOUT_FILES
        ),
    }
    audit = {
        "schema_version": "g3-p7-pairwise-counterfactual-calibration-0.1",
        "decision": "pairwise_counterfactual_calibration_go"
        if all(checks.values())
        else "pairwise_counterfactual_calibration_no_go",
        "checks": checks,
        "winner": winner["model_type"] if winner else None,
        "capacity": capacity_audit,
        "arms": arms,
        "training_pools": {name: len(rows) for name, rows in pools.items()},
        "protocol": {
            "folds": FOLDS,
            "oof_steps": OOF_STEPS,
            "full_steps": FULL_STEPS,
            "selection": ["zero harmful", "rescues >= 100", "max rescues", "min parameters"],
            "holdout_read": False,
        },
        "holdout_read": False,
        "dev_confirm_external_sealed_read": False,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
