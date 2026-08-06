#!/usr/bin/env python3
"""Continue P3-E16 for one unique epoch with bucket-balanced Count loss."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from chemvector.components.atoms.objects.g3_deepsets_count import (
    G3DeepSetsCount,
    G3DeepSetsCountConfig,
)
from chemvector.training.atoms import audit_g3_p3e0_latent_solver as p3e0
from chemvector.training.atoms import audit_g3_p3e7_base_expert_router as p3e7
from chemvector.training.atoms import train_g3_p2e_stride1_top128_selector_train5k as p2e
from chemvector.training.atoms import train_g3_p3e9_deepsets_count as p3e9
from chemvector.training.atoms import train_g3_p3e10_count_scale_ab as p3e10
from chemvector.training.atoms import train_g3_p3e14_style_count_scale as p3e14
from chemvector.training.atoms import train_g3_p3e16_style_count_epoch6 as p3e16
from chemvector.training.atoms import audit_g3_p3e17_p3e16_count_router as p3e17
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p3e19-bucket-balanced-count-epoch1-h64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
SOURCE_CHECKPOINT = p3e17.COUNT_CHECKPOINT
SOURCE_SHA256 = p3e17.COUNT_SHA256
FRESH_CACHE = Path(
    "outputs/audits/chemvector-g3-p3e18-freshdev8k-heldout-font-audit/candidate-cache.pt"
)
FRESH_CACHE_SHA256 = "9d5f8f4ec14e0a43c62f5db7e411b1d9e4716692d22b77d8c07a917231bef09d"
FRESH_MANIFEST_SHA256 = "aa6d0254c5b0d8911972b8377a0628c2e0e3fbc72b95af95c9d8807591f4ab0d"
TRAIN_IMAGES = p3e14.TRAIN_IMAGES
BATCH_SIZE = 64
STEPS = math.ceil(TRAIN_IMAGES / BATCH_SIZE)
EXPOSURE = TRAIN_IMAGES
FINAL_GLOBAL_STEP = p3e16.FINAL_GLOBAL_STEP + STEPS
FINAL_SAMPLES_SEEN = p3e16.FINAL_SAMPLES_SEEN + EXPOSURE
SEED = 20260819
EXPECTED_BUCKET_COUNTS = {
    "1-20": 11_756,
    "21-40": 66_020,
    "41-60": 9_836,
    "61-80": 2_372,
    "81-100": 948,
}
BASELINE_FRESH = {
    "count_accuracy": 0.362625,
    "count_mae": 1.98075,
    "1-20": 0.558,
    "21-40": 0.5255,
    "41-60": 0.2705,
    "61-80": 0.0965,
}


def count_bucket(value: int) -> str:
    for name, lower, upper in p3e16.COUNT_BUCKETS:
        if lower <= value <= upper:
            return name
    raise ValueError(f"P3-E19 Count is outside supported buckets: {value}")


def bucket_weights(records: list[dict]) -> tuple[dict[str, float], dict]:
    counts = Counter(count_bucket(int(record["label"])) for record in records)
    if dict(counts) != EXPECTED_BUCKET_COUNTS:
        raise ValueError(f"P3-E19 style Count distribution differs: {dict(counts)}")
    nonempty = len(counts)
    weights = {
        name: len(records) / (nonempty * count) for name, count in counts.items()
    }
    contribution = {name: counts[name] * weights[name] for name in counts}
    if max(contribution.values()) - min(contribution.values()) > 1e-6:
        raise RuntimeError("P3-E19 bucket contributions are not equal")
    return weights, {
        "counts": dict(sorted(counts.items())),
        "weights": dict(sorted(weights.items())),
        "weighted_contribution": dict(sorted(contribution.items())),
        "mean_sample_weight": sum(contribution.values()) / len(records),
    }


def weighted_count_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: dict[str, float],
) -> torch.Tensor:
    sample_weights = logits.new_tensor(
        [weights[count_bucket(int(value))] for value in target]
    )
    return (F.cross_entropy(logits, target, reduction="none") * sample_weights).mean()


def epoch_batches(record_count: int) -> tuple[list[list[int]], dict]:
    if record_count != TRAIN_IMAGES:
        raise ValueError("P3-E19 training record count differs")
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    order = torch.randperm(record_count, generator=generator).tolist()
    batches = [order[start : start + BATCH_SIZE] for start in range(0, record_count, BATCH_SIZE)]
    if len(batches) != STEPS or sum(map(len, batches)) != EXPOSURE:
        raise RuntimeError("P3-E19 unique epoch schedule differs")
    digest = hashlib.sha256(torch.tensor(order, dtype=torch.int64).numpy().tobytes())
    return batches, {
        "unique_images": len(set(order)),
        "steps": len(batches),
        "exposure": sum(map(len, batches)),
        "full_batches": record_count // BATCH_SIZE,
        "residual_batch_size": record_count % BATCH_SIZE,
        "order_sha256": digest.hexdigest(),
    }


def load_source(device: torch.device):
    if file_sha256(SOURCE_CHECKPOINT) != SOURCE_SHA256:
        raise ValueError("P3-E19 source checkpoint SHA256 differs")
    source = torch.load(SOURCE_CHECKPOINT, map_location=device, weights_only=False)
    if (
        source.get("model_type") != p3e16.MODEL_TYPE
        or source.get("global_step") != p3e16.FINAL_GLOBAL_STEP
        or source.get("samples_seen") != p3e16.FINAL_SAMPLES_SEEN
    ):
        raise ValueError("P3-E19 source checkpoint contract differs")
    model = G3DeepSetsCount(G3DeepSetsCountConfig(**source["model_config"])).to(device)
    model.load_state_dict(source["deepsets_count"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=p3e14.LEARNING_RATE, weight_decay=p3e14.WEIGHT_DECAY
    )
    optimizer.load_state_dict(source["optimizer"])
    return source, model, optimizer


def load_fresh_records() -> tuple[list[dict], dict]:
    if file_sha256(FRESH_CACHE) != FRESH_CACHE_SHA256:
        raise ValueError("P3-E19 fresh candidate cache SHA256 differs")
    payload = torch.load(FRESH_CACHE, map_location="cpu", weights_only=False)
    records = payload.get("records")
    if (
        payload.get("schema_version") != "g3-p3e18-candidate-cache-0.1"
        or payload.get("manifest_sha256") != FRESH_MANIFEST_SHA256
        or not isinstance(records, list)
        or len(records) != 8_000
    ):
        raise ValueError("P3-E19 fresh candidate cache contract differs")
    return records, {
        "path": str(FRESH_CACHE.resolve()),
        "sha256": FRESH_CACHE_SHA256,
        "manifest_sha256": FRESH_MANIFEST_SHA256,
        "samples": len(records),
    }


def capability_gate(fresh: dict, confirm: dict) -> tuple[str, dict[str, bool]]:
    checks = {
        "fresh_accuracy_gain5pp": (
            fresh["count_accuracy"] >= BASELINE_FRESH["count_accuracy"] + 0.05
        ),
        "fresh_mae_gain0_25": fresh["count_mae"] <= BASELINE_FRESH["count_mae"] - 0.25,
        "fresh_61_80_gain10pp": (
            fresh["by_count"]["61-80"]["count_accuracy"]
            >= BASELINE_FRESH["61-80"] + 0.10
        ),
        "fresh_1_20_regression_max5pp": (
            fresh["by_count"]["1-20"]["count_accuracy"]
            >= BASELINE_FRESH["1-20"] - 0.05
        ),
        "fresh_21_40_regression_max5pp": (
            fresh["by_count"]["21-40"]["count_accuracy"]
            >= BASELINE_FRESH["21-40"] - 0.05
        ),
        "confirm_accuracy_min45": confirm["count_accuracy"] >= 0.45,
        "confirm_mae_max1_2": confirm["count_mae"] <= 1.2,
    }
    return (
        "bucket_balanced_count_go"
        if all(checks.values())
        else "bucket_balanced_count_no_go",
        checks,
    )


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E19 output is not empty: {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("P3-E19 requires CUDA BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    style_records, evidence_audit = p3e14.load_style_evidence()
    weights, weight_audit = bucket_weights(style_records)
    batches, schedule_audit = epoch_batches(len(style_records))
    cache, cache_audit = p2e.load_cache()
    fixed_splits, selection_audit = p3e10.expanded_splits(cache)
    fixed_splits = {
        name: fixed_splits[name]
        for name in ("dev256", "confirm256", "historical_route100")
    }
    fresh_candidates, fresh_audit = load_fresh_records()
    base_model = p3e0.load_model(p3e7.BASE_CHECKPOINT, device)
    expert_model = p3e0.load_model(p3e7.EXPERT_CHECKPOINT, device)
    eval_evidence = p3e9.collect_evidence(
        (base_model, expert_model),
        {**fixed_splits, "fresh_dev8k": fresh_candidates},
        device=device,
    )
    del base_model, expert_model
    source, model, optimizer = load_source(device)
    baseline_fresh = p3e16.evaluate_bucketed(
        model, eval_evidence["fresh_dev8k"], device=device
    )
    if (
        baseline_fresh["count_accuracy"] != BASELINE_FRESH["count_accuracy"]
        or baseline_fresh["count_mae"] != BASELINE_FRESH["count_mae"]
    ):
        raise ValueError("P3-E19 fresh baseline does not reproduce P3-E18")

    audit = {
        "schema_version": "g3-p3e19-bucket-balanced-count-audit-0.1",
        "model_type": MODEL_TYPE,
        "fixed_protocol": {
            "source": "restore P3-E16 model and AdamW exactly",
            "only_change": "inverse Count-bucket-frequency weighted cross entropy",
            "unique_epochs": 1,
            "checkpoint_selection": "fixed_final_only",
            "fresh_dev_training": False,
            "fresh_dev_intermediate_selection": False,
            "sealed_test_read": False,
        },
        "schedule": schedule_audit,
        "bucket_balance": weight_audit,
        "baseline_fresh": baseline_fresh,
        "lineage": {
            "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
            "source_sha256": SOURCE_SHA256,
            "style_evidence": evidence_audit,
            "fresh_cache": fresh_audit,
            "base_checkpoint": str(p3e7.BASE_CHECKPOINT.resolve()),
            "base_sha256": p3e7.BASE_SHA256,
            "expert_checkpoint": str(p3e7.EXPERT_CHECKPOINT.resolve()),
            "expert_sha256": p3e7.EXPERT_SHA256,
            "fixed_eval_selection": selection_audit,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "audit.json", audit)

    started = time.perf_counter()
    exposure = 0
    loss_sum = gradient_sum = clip_sum = 0.0
    model.train()
    log_path = OUTPUT / "train.jsonl"
    for step, indices in enumerate(batches, start=1):
        query, candidate, global_features, target = p3e9.batch(
            style_records, indices, device
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, "bf16"):
            logits = model(query, candidate, global_features)
            loss = weighted_count_loss(logits, target, weights)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), p3e14.MAX_GRAD_NORM
        )
        optimizer.step()
        gradient = float(gradient_norm)
        loss_sum += float(loss.detach())
        gradient_sum += gradient
        clip_sum += min(1.0, p3e14.MAX_GRAD_NORM / max(gradient, 1e-12))
        exposure += len(indices)
        if step % 200 == 0 or step == STEPS:
            current = {
                "step": step,
                "global_step": p3e16.FINAL_GLOBAL_STEP + step,
                "exposure": exposure,
                "mean_weighted_loss": loss_sum / step,
                "mean_gradient_norm": gradient_sum / step,
                "mean_clip_scale": clip_sum / step,
                "elapsed_seconds": time.perf_counter() - started,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(current, separators=(",", ":")) + "\n")
            print(json.dumps(current, separators=(",", ":")), flush=True)

    evaluation_sets = {
        "style90932": style_records,
        **eval_evidence,
    }
    metrics = {
        name: p3e16.evaluate_bucketed(model, records, device=device)
        for name, records in evaluation_sets.items()
    }
    decision, checks = capability_gate(
        metrics["fresh_dev8k"], metrics["confirm256"]
    )
    protocol_checks = {
        "complete_unique_epoch": exposure == EXPOSURE,
        "fixed_steps": len(batches) == STEPS,
        "final_global_step": source["global_step"] + len(batches) == FINAL_GLOBAL_STEP,
        "final_samples_seen": source["samples_seen"] + exposure == FINAL_SAMPLES_SEEN,
    }
    if not all(protocol_checks.values()):
        decision = "bucket_balanced_count_protocol_failed"
    final = {
        "decision": decision,
        "checks": checks,
        "protocol_checks": protocol_checks,
        "baseline_fresh": baseline_fresh,
        "metrics": metrics,
        "training_seconds": time.perf_counter() - started,
    }
    payload = {
        "checkpoint_schema_version": "g3-p3e19-bucket-balanced-count-0.1",
        "model_type": MODEL_TYPE,
        "model_config": asdict(G3DeepSetsCountConfig(**source["model_config"])),
        "deepsets_count": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": FINAL_GLOBAL_STEP,
        "samples_seen": FINAL_SAMPLES_SEEN,
        "training_images": TRAIN_IMAGES,
        "balanced_exposure": exposure,
        "trainable_parameter_count": sum(value.numel() for value in model.parameters()),
        "metrics": final,
        "lineage": audit["lineage"],
    }
    checkpoint = OUTPUT / f"step-{FINAL_GLOBAL_STEP:06d}.pt"
    _atomic_torch_save(checkpoint, payload)
    _atomic_torch_save(OUTPUT / "last.pt", payload)
    final["checkpoint"] = str(checkpoint.resolve())
    final["checkpoint_sha256"] = file_sha256(checkpoint)
    _atomic_json(OUTPUT / "final_metrics.json", final)
    _atomic_json(
        OUTPUT / "status.json",
        {
            "status": "completed",
            "decision": decision,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": final["checkpoint_sha256"],
        },
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
