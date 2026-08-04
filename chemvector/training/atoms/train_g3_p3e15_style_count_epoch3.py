#!/usr/bin/env python3
"""Continue P3-E14 Count through two more complete style90k epochs."""

from __future__ import annotations

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
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p3e9_deepsets_count as p3e9
from chemvector.training.atoms import train_g3_p3e10_count_scale_ab as p3e10
from chemvector.training.atoms import train_g3_p3e14_style_count_scale as p3e14
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p3e15-deepsets-count-style90932-epoch3-h64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
SOURCE_CHECKPOINT = p3e14.OUTPUT / "step-001560.pt"
SOURCE_SHA256 = "8dd295bc1e1b976c6e851cfa566ce990a1971726c23eebd48880450c4a809687"
TRAIN_IMAGES = p3e14.TRAIN_IMAGES
CONTINUATION_EPOCHS = 2
BATCH_SIZE = 64
STEPS_PER_EPOCH = math.ceil(TRAIN_IMAGES / BATCH_SIZE)
CONTINUATION_STEPS = CONTINUATION_EPOCHS * STEPS_PER_EPOCH
CONTINUATION_EXPOSURE = CONTINUATION_EPOCHS * TRAIN_IMAGES
FINAL_GLOBAL_STEP = p3e14.EXPECTED_STEPS + CONTINUATION_STEPS
FINAL_SAMPLES_SEEN = p3e14.EXPECTED_EXPOSURE + CONTINUATION_EXPOSURE
SEED = 20260805


def continuation_batches(record_count: int) -> tuple[list[list[int]], dict]:
    if record_count != TRAIN_IMAGES:
        raise ValueError("P3-E15 continuation record count differs")
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    batches = []
    epoch_digests = []
    for _ in range(CONTINUATION_EPOCHS):
        order = torch.randperm(record_count, generator=generator).tolist()
        if len(set(order)) != record_count:
            raise RuntimeError("P3-E15 epoch does not cover every unique image")
        epoch_digests.append(
            hashlib.sha256(torch.tensor(order, dtype=torch.int64).numpy().tobytes()).hexdigest()
        )
        batches.extend(
            order[start : start + BATCH_SIZE]
            for start in range(0, record_count, BATCH_SIZE)
        )
    if len(batches) != CONTINUATION_STEPS or sum(map(len, batches)) != CONTINUATION_EXPOSURE:
        raise RuntimeError("P3-E15 continuation schedule differs")
    return batches, {
        "epochs": CONTINUATION_EPOCHS,
        "unique_images_per_epoch": record_count,
        "continuation_steps": len(batches),
        "continuation_exposure": sum(map(len, batches)),
        "full_batches_per_epoch": record_count // BATCH_SIZE,
        "residual_batch_size": record_count % BATCH_SIZE,
        "epoch_order_sha256": epoch_digests,
    }


def load_source(device: torch.device) -> tuple[dict, G3DeepSetsCount, torch.optim.AdamW]:
    if file_sha256(SOURCE_CHECKPOINT) != SOURCE_SHA256:
        raise ValueError("P3-E15 source checkpoint SHA256 differs")
    source = torch.load(SOURCE_CHECKPOINT, map_location=device, weights_only=False)
    if (
        source.get("model_type") != p3e14.MODEL_TYPE
        or source.get("global_step") != p3e14.EXPECTED_STEPS
        or source.get("samples_seen") != p3e14.EXPECTED_EXPOSURE
        or source.get("training_images") != TRAIN_IMAGES
    ):
        raise ValueError("P3-E15 source checkpoint contract differs")
    model = G3DeepSetsCount(G3DeepSetsCountConfig(**source["model_config"])).to(device)
    model.load_state_dict(source["deepsets_count"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=p3e14.LEARNING_RATE, weight_decay=p3e14.WEIGHT_DECAY
    )
    optimizer.load_state_dict(source["optimizer"])
    if any(group["lr"] != p3e14.LEARNING_RATE for group in optimizer.param_groups):
        raise ValueError("P3-E15 restored optimizer learning rate differs")
    return source, model, optimizer


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E15 output is not empty: {OUTPUT}")
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
    cache, cache_audit = p2e.load_cache()
    fixed_splits, selection_audit = p3e10.expanded_splits(cache)
    fixed_splits = {
        name: fixed_splits[name]
        for name in ("dev256", "confirm256", "historical_route100")
    }
    base_model = p3e0.load_model(p3e7.BASE_CHECKPOINT, device)
    expert_model = p3e0.load_model(p3e7.EXPERT_CHECKPOINT, device)
    eval_evidence = p3e9.collect_evidence(
        (base_model, expert_model), fixed_splits, device=device
    )
    del base_model, expert_model

    source, model, optimizer = load_source(device)
    evaluation_sets = {"style90932": style_records, **eval_evidence}
    baseline_metrics = {
        name: p3e9.evaluate(model, records, device=device)
        for name, records in evaluation_sets.items()
    }
    batches, schedule_audit = continuation_batches(len(style_records))
    trainable_parameters = sum(value.numel() for value in model.parameters())
    audit = {
        "schema_version": "g3-p3e15-style-count-epoch3-audit-0.1",
        "model_type": MODEL_TYPE,
        "fixed_protocol": {
            "continuation": "restore P3-E14 model and AdamW optimizer exactly",
            "complete_unique_epochs_added": CONTINUATION_EPOCHS,
            "total_complete_unique_epochs": 3,
            "p3e14_extra_matched_exposure": p3e14.REPEATED_EXPOSURE,
            "continuation_steps": CONTINUATION_STEPS,
            "continuation_exposure": CONTINUATION_EXPOSURE,
            "final_global_step": FINAL_GLOBAL_STEP,
            "final_samples_seen": FINAL_SAMPLES_SEEN,
            "checkpoint_selection": "fixed_final_only",
            "route100_training_or_selection": False,
            "gate": {
                "dev_count_accuracy_min": 0.5,
                "confirm_count_accuracy_min": 0.5,
                "confirm_count_mae_max": 1.0,
                "route100_count_accuracy_min": 0.4,
            },
        },
        "schedule": schedule_audit,
        "trainable_parameters": trainable_parameters,
        "baseline_metrics": baseline_metrics,
        "lineage": {
            "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
            "source_sha256": SOURCE_SHA256,
            "evidence": evidence_audit,
            "base_checkpoint": str(p3e7.BASE_CHECKPOINT.resolve()),
            "base_sha256": p3e7.BASE_SHA256,
            "expert_checkpoint": str(p3e7.EXPERT_CHECKPOINT.resolve()),
            "expert_sha256": p3e7.EXPERT_SHA256,
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
            "fixed_eval_selection": selection_audit,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "audit.json", audit)

    log_path = OUTPUT / "train.jsonl"
    started = time.perf_counter()
    continuation_exposure = 0
    loss_sum = gradient_sum = clip_sum = 0.0
    model.train()
    for continuation_step, indices in enumerate(batches, start=1):
        query, candidate, global_features, target = p3e9.batch(
            style_records, indices, device
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, "bf16"):
            loss = F.cross_entropy(model(query, candidate, global_features), target)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), p3e14.MAX_GRAD_NORM
        )
        optimizer.step()
        gradient_float = float(gradient_norm)
        loss_sum += float(loss.detach())
        gradient_sum += gradient_float
        clip_sum += min(1.0, p3e14.MAX_GRAD_NORM / max(gradient_float, 1e-12))
        continuation_exposure += len(indices)
        if continuation_step % 200 == 0 or continuation_step == CONTINUATION_STEPS:
            log = {
                "continuation_step": continuation_step,
                "global_step": p3e14.EXPECTED_STEPS + continuation_step,
                "continuation_exposure": continuation_exposure,
                "samples_seen": p3e14.EXPECTED_EXPOSURE + continuation_exposure,
                "mean_loss": loss_sum / continuation_step,
                "mean_gradient_norm": gradient_sum / continuation_step,
                "mean_clip_scale": clip_sum / continuation_step,
                "elapsed_seconds": time.perf_counter() - started,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log, separators=(",", ":")) + "\n")
            print(json.dumps(log, separators=(",", ":")), flush=True)

    metrics = {
        name: p3e9.evaluate(model, records, device=device)
        for name, records in evaluation_sets.items()
    }
    repeated = {
        name: p3e9.evaluate(model, records, device=device)
        for name, records in evaluation_sets.items()
    }
    checks = {
        "dev_count_accuracy_min50": metrics["dev256"]["count_accuracy"] >= 0.5,
        "confirm_count_accuracy_min50": metrics["confirm256"]["count_accuracy"] >= 0.5,
        "confirm_count_mae_max1": metrics["confirm256"]["count_mae"] <= 1.0,
        "route100_count_accuracy_min40": (
            metrics["historical_route100"]["count_accuracy"] >= 0.4
        ),
    }
    protocol_checks = {
        "continuation_steps": len(batches) == CONTINUATION_STEPS,
        "continuation_exposure": continuation_exposure == CONTINUATION_EXPOSURE,
        "final_global_step": source["global_step"] + len(batches) == FINAL_GLOBAL_STEP,
        "final_samples_seen": (
            source["samples_seen"] + continuation_exposure == FINAL_SAMPLES_SEEN
        ),
        "repeat_metrics": repeated == metrics,
    }
    decision = (
        "style_count_epoch3_go"
        if all(checks.values()) and all(protocol_checks.values())
        else "style_count_epoch3_no_go"
    )
    final = {
        "decision": decision,
        "checks": checks,
        "protocol_checks": protocol_checks,
        "baseline_metrics": baseline_metrics,
        "metrics": metrics,
        "training_seconds": time.perf_counter() - started,
    }
    payload = {
        "checkpoint_schema_version": "g3-p3e15-style-count-epoch3-0.1",
        "model_type": MODEL_TYPE,
        "model_config": source["model_config"],
        "deepsets_count": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": FINAL_GLOBAL_STEP,
        "samples_seen": FINAL_SAMPLES_SEEN,
        "training_images": TRAIN_IMAGES,
        "continuation_exposure": continuation_exposure,
        "trainable_parameter_count": trainable_parameters,
        "metrics": final,
        "lineage": audit["lineage"],
    }
    checkpoint = OUTPUT / f"step-{FINAL_GLOBAL_STEP:06d}.pt"
    _atomic_torch_save(checkpoint, payload)
    _atomic_torch_save(OUTPUT / "last.pt", payload)
    final["checkpoint"] = str(checkpoint.resolve())
    final["checkpoint_sha256"] = file_sha256(checkpoint)
    _atomic_json(OUTPUT / "final_metrics.json", final)
    _atomic_json(OUTPUT / "status.json", {
        "status": "completed",
        "decision": decision,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": final["checkpoint_sha256"],
    })
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
