#!/usr/bin/env python3
"""Train P3-E11 DeepSets Count with an ordered expected-count penalty."""

from __future__ import annotations

from dataclasses import asdict
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
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p3e11-ordinal-count-expanded3316-h64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
EPOCHS = 30
BATCH_SIZE = 64
EXPECTED_STEPS = EPOCHS * math.ceil(p3e10.TRAIN_IMAGES / BATCH_SIZE)
EXPECTED_EXPOSURE = EPOCHS * p3e10.TRAIN_IMAGES
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
ORDINAL_WEIGHT = 0.1
SEED = 20260713


def ordinal_count_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    ordinal_weight: float = ORDINAL_WEIGHT,
) -> dict[str, torch.Tensor]:
    if logits.ndim != 2 or logits.size(1) != 129:
        raise ValueError("P3-E11 Count logits must have 129 classes")
    if target.shape != (len(logits),) or target.dtype != torch.long:
        raise ValueError("P3-E11 Count targets have invalid shape or dtype")
    if ordinal_weight < 0.0:
        raise ValueError("P3-E11 ordinal weight must be nonnegative")
    ce = F.cross_entropy(logits, target)
    classes = torch.arange(129, device=logits.device, dtype=logits.dtype)
    expected = (logits.softmax(dim=-1) * classes).sum(dim=-1)
    ordinal = F.smooth_l1_loss(expected, target.to(expected.dtype))
    return {
        "loss": ce + ordinal_weight * ordinal,
        "cross_entropy": ce,
        "ordinal_smooth_l1": ordinal,
        "expected_count_mae": (expected - target).abs().mean(),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E11 output is not empty: {OUTPUT}")
    for path, expected in (
        (p3e7.BASE_CHECKPOINT, p3e7.BASE_SHA256),
        (p3e7.EXPERT_CHECKPOINT, p3e7.EXPERT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P3-E11 checkpoint SHA differs: {path}")
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

    cache, cache_audit = p2e.load_cache()
    splits, selection_audit = p3e10.expanded_splits(cache)
    base_model = p3e0.load_model(p3e7.BASE_CHECKPOINT, device)
    expert_model = p3e0.load_model(p3e7.EXPERT_CHECKPOINT, device)
    evidence_started = time.perf_counter()
    evidence = p3e9.collect_evidence(
        (base_model, expert_model), splits, device=device
    )
    evidence_seconds = time.perf_counter() - evidence_started
    del base_model, expert_model

    config = G3DeepSetsCountConfig(hidden_dim=64)
    model = G3DeepSetsCount(config).to(device)
    trainable_parameters = sum(value.numel() for value in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    audit = {
        "schema_version": "g3-p3e11-ordinal-count-audit-0.1",
        "model_type": MODEL_TYPE,
        "fixed_protocol": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "expected_steps": EXPECTED_STEPS,
            "expected_exposure": EXPECTED_EXPOSURE,
            "loss": "cross_entropy + 0.1 * smooth_l1(softmax expected count, target)",
            "ordinal_weight": ORDINAL_WEIGHT,
            "checkpoint_selection": "fixed_final_only",
            "route100_training_or_selection": False,
            "gate": {
                "dev_count_accuracy_min": 0.5,
                "confirm_count_accuracy_min": 0.5,
                "confirm_count_mae_max": 1.0,
                "route100_count_accuracy_min": 0.4,
            },
        },
        "model_config": asdict(config),
        "trainable_parameters": trainable_parameters,
        "lineage": {
            "base_checkpoint": str(p3e7.BASE_CHECKPOINT.resolve()),
            "base_sha256": p3e7.BASE_SHA256,
            "expert_checkpoint": str(p3e7.EXPERT_CHECKPOINT.resolve()),
            "expert_sha256": p3e7.EXPERT_SHA256,
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
            "selection": selection_audit,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "audit.json", audit)

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    train_records = evidence["train3316"]
    log_path = OUTPUT / "train.jsonl"
    step = exposure = 0
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(len(train_records), generator=generator).tolist()
        sums = {name: 0.0 for name in (
            "loss", "cross_entropy", "ordinal_smooth_l1", "expected_count_mae"
        )}
        gradient_sum = clip_sum = 0.0
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            query, candidate, global_features, target = p3e9.batch(
                train_records, indices, device
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, "bf16"):
                logits = model(query, candidate, global_features)
                losses = ordinal_count_loss(logits, target)
            losses["loss"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), MAX_GRAD_NORM
            )
            optimizer.step()
            for name in sums:
                sums[name] += float(losses[name].detach())
            gradient_float = float(gradient_norm)
            gradient_sum += gradient_float
            clip_sum += min(1.0, MAX_GRAD_NORM / max(gradient_float, 1e-12))
            step += 1
            exposure += len(indices)
        batches = math.ceil(len(order) / BATCH_SIZE)
        log = {
            "epoch": epoch,
            "step": step,
            "exposure": exposure,
            **{name: value / batches for name, value in sums.items()},
            "gradient_norm": gradient_sum / batches,
            "clip_scale": clip_sum / batches,
            "elapsed_seconds": time.perf_counter() - started,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log, separators=(",", ":")) + "\n")
        print(json.dumps(log, separators=(",", ":")), flush=True)

    metrics = {
        name: p3e9.evaluate(model, records, device=device)
        for name, records in evidence.items()
    }
    repeated = {
        name: p3e9.evaluate(model, records, device=device)
        for name, records in evidence.items()
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
        "fixed_steps": step == EXPECTED_STEPS,
        "fixed_exposure": exposure == EXPECTED_EXPOSURE,
        "repeat_metrics": repeated == metrics,
    }
    decision = (
        "ordinal_count_go"
        if all(checks.values()) and all(protocol_checks.values())
        else "ordinal_count_no_go"
    )
    final = {
        "decision": decision,
        "checks": checks,
        "protocol_checks": protocol_checks,
        "metrics": metrics,
        "evidence_seconds": evidence_seconds,
        "training_seconds": time.perf_counter() - started,
    }
    payload = {
        "checkpoint_schema_version": "g3-p3e11-ordinal-count-0.1",
        "model_type": MODEL_TYPE,
        "model_config": asdict(config),
        "deepsets_count": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_images": p3e10.TRAIN_IMAGES,
        "trainable_parameter_count": trainable_parameters,
        "ordinal_weight": ORDINAL_WEIGHT,
        "metrics": final,
        "lineage": audit["lineage"],
    }
    checkpoint = OUTPUT / f"step-{step:06d}.pt"
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
