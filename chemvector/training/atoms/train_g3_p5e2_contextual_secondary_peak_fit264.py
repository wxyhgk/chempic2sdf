#!/usr/bin/env python3
"""Train the P5-E2 full-context forced-source secondary-peak decoder."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.recovery.spatial_secondary_peak import (
    ContextualSecondaryPeakHead,
    dihedral_transform,
    secondary_peak_loss,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.training.atoms import (
    train_g3_p5e1_spatial_secondary_peak_fit264 as p5e1,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    module_state_sha256,
)


MODEL_TYPE = "chemvector-g3-p5e2-contextual-secondary-peak-fit264"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
STEPS = 3_000
BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
SEED = 20260854


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P5-E2 output already exists: {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("P5-E2 requires CUDA BF16")
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

    cache = p5e1._load_positive_cache()
    schedule = p5e1._schedule(len(cache["offset_targets"]))
    model = ContextualSecondaryPeakHead().to(device)
    initial_sha = module_state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    OUTPUT.mkdir(parents=True)
    started = time.perf_counter()
    generator = torch.Generator().manual_seed(SEED + 1)
    loss_sum = 0.0
    last = {}
    for step, rows in enumerate(schedule, start=1):
        optimizer.zero_grad(set_to_none=True)
        images = cache["image_crops"][rows].to(device).float()
        scores = cache["score_crops"][rows].to(device).float()
        offsets = cache["offset_targets"][rows].to(device).float()
        transform_ids = torch.randint(
            0, 8, (BATCH_SIZE,), generator=generator
        ).to(device)
        images, scores, offsets = dihedral_transform(
            images, scores, offsets, transform_ids
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                images,
                scores,
                cache["candidate_scores"][rows].to(device).float(),
                cache["candidate_ranks"][rows].to(device).float(),
            )
            loss = secondary_peak_loss(logits, offsets)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), MAX_GRAD_NORM
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("P5-E2 gradient norm is not finite")
        optimizer.step()
        loss_sum += float(loss.detach())
        last = {
            "step": step,
            "loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm.detach()),
        }
        if step % 250 == 0 or step == STEPS:
            print(json.dumps(last), flush=True)

    fit = p5e1._evaluate_tensors(model, cache, device=device)
    domains = {}
    streams = {}
    for split in ("calibration", "holdout"):
        manifest, actions_by_key = p5e1._action_manifest(split)
        domains[split], streams[split] = p5e1.evaluate_oracle_actions(
            model, manifest, actions_by_key, device=device
        )
    checks = {
        "calibration_recovery_min80pct": domains["calibration"]["recovery_rate_6px"]
        >= p5e1.MINIMUM_RECOVERY,
        "holdout_recovery_min80pct": domains["holdout"]["recovery_rate_6px"]
        >= p5e1.MINIMUM_RECOVERY,
        "calibration_mae_max3px": domains["calibration"]["mae_pixels"]
        <= p5e1.MAXIMUM_MAE_PIXELS,
        "holdout_mae_max3px": domains["holdout"]["mae_pixels"]
        <= p5e1.MAXIMUM_MAE_PIXELS,
    }
    decision = (
        "contextual_secondary_peak_go"
        if all(checks.values())
        else "contextual_secondary_peak_no_go"
    )
    checkpoint = OUTPUT / f"step-{STEPS:06d}.pt"
    payload = {
        "checkpoint_schema_version": "g3-p5e2-contextual-secondary-peak-0.1",
        "model_type": MODEL_TYPE,
        "config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": STEPS,
        "samples_seen": STEPS * BATCH_SIZE,
        "training_args": {
            "batch_size": BATCH_SIZE,
            "steps": STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "seed": SEED,
            "dihedral_augmentation": True,
        },
        "lineage": {
            "source_cache": str(p5e1.SOURCE_CACHE.resolve()),
            "source_cache_sha256": p5e1.SOURCE_CACHE_SHA256,
            "oracle_audit_sha256": p5e1.ORACLE_AUDIT_SHA256,
        },
        "metrics": {"decision": decision, "checks": checks, "domains": domains},
    }
    _atomic_torch_save(checkpoint, payload)
    final = {
        "schema_version": "g3-p5e2-contextual-secondary-peak-final-0.1",
        "decision": decision,
        "checks": checks,
        "model_type": MODEL_TYPE,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "parameter_count": model.parameter_count,
        "initial_state_sha256": initial_sha,
        "final_state_sha256": module_state_sha256(model),
        "training": {
            "steps": STEPS,
            "exposure": STEPS * BATCH_SIZE,
            "mean_loss": loss_sum / STEPS,
            "last": last,
        },
        "fit": fit,
        "domains": domains,
        "streams": streams,
        "schedule_sha256": hashlib.sha256(
            torch.stack(schedule).numpy().tobytes()
        ).hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        "confirm_v2_read": False,
        "dev_and_external_read": False,
    }
    _atomic_json(OUTPUT / "final_metrics.json", final)
    _atomic_json(
        OUTPUT / "audit.json",
        {
            "schema_version": "g3-p5e2-contextual-secondary-peak-audit-0.1",
            "status": "completed",
            **final,
            "protocol": {
                "fit_positive_actions": p5e1.TRAIN_POSITIVES,
                "forced_oracle_source_evaluation": True,
                "source_relative_coordinates": True,
                "full_crop_convolutional_path": True,
                "confirm_v2_read": False,
            },
        },
    )
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
