#!/usr/bin/env python3
"""Short real-action calibration from the P5-E9 generic-regularized state."""

from __future__ import annotations

from dataclasses import asdict
import copy
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.recovery.spatial_secondary_peak import (
    ContextualSecondaryPeakHead,
)
from chemvector.training.atoms import (
    train_g3_p5e1_spatial_secondary_peak_fit264 as p5e1,
)
from chemvector.training.atoms import (
    train_g3_p5e9_generic_direction_mixed as p5e9,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p5e10-real-calibrated-fit1378"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
SOURCE_CHECKPOINT = p5e9.OUTPUT / "step-015000.pt"
SOURCE_CHECKPOINT_SHA256 = (
    "b852bdc369639c8e1ceeed9e8c8b36ae4e785dae25fa24cb3666d009edd1e42e"
)
CANDIDATE_STEPS = (0, 250, 500, 1_000, 2_000)
MAXIMUM_STEPS = CANDIDATE_STEPS[-1]
LEARNING_RATE = 5e-5
SEED = 20260859


def select_candidate(metrics: dict[int, dict]) -> int:
    passed = [
        step
        for step, values in metrics.items()
        if values["recovery_rate_6px"] >= 0.80 and values["mae_pixels"] <= 3.0
    ]
    if passed:
        return min(passed)
    return min(
        metrics,
        key=lambda step: (
            -metrics[step]["recovery_rate_6px"],
            metrics[step]["mae_pixels"],
            step,
        ),
    )


def _evaluate_split(model, split: str, device: torch.device) -> tuple[dict, dict]:
    manifest, actions = p5e1._action_manifest(split)
    return p5e1.evaluate_oracle_actions(model, manifest, actions, device=device)


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"P5-E10 output already exists: {OUTPUT}")
    if file_sha256(SOURCE_CHECKPOINT) != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("P5-E10 source checkpoint SHA256 differs")
    if not torch.cuda.is_available():
        raise RuntimeError("P5-E10 requires CUDA BF16")
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

    payload = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    model = ContextualSecondaryPeakHead().to(device)
    model.load_state_dict(payload["model"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=p5e9.MIXED_LR, weight_decay=p5e9.WEIGHT_DECAY
    )
    optimizer.load_state_dict(payload["optimizer"])
    for group in optimizer.param_groups:
        group["lr"] = LEARNING_RATE
    real = p5e9._real_cache()
    schedule = p5e9._schedule(
        len(real["offset_targets"]), MAXIMUM_STEPS, p5e9.BATCH_SIZE, SEED + 10
    )
    transform_generator = torch.Generator().manual_seed(SEED + 1)
    candidate_metrics = {}
    candidate_streams = {}
    model_states = {}
    optimizer_states = {}
    started = time.perf_counter()

    def capture(step: int) -> None:
        metrics, stream = _evaluate_split(model, "calibration", device)
        candidate_metrics[step] = metrics
        candidate_streams[step] = stream
        model_states[step] = {
            name: value.detach().cpu().clone() for name, value in model.state_dict().items()
        }
        optimizer_states[step] = copy.deepcopy(optimizer.state_dict())
        print(json.dumps({"candidate_step": step, **metrics}), flush=True)

    capture(0)
    loss_sum = 0.0
    last = {}
    for step, rows in enumerate(schedule, start=1):
        transform_ids = torch.randint(
            0, 8, (p5e9.BATCH_SIZE,), generator=transform_generator
        ).to(device)
        loss, gradient = p5e9._train_step(
            model, optimizer, p5e9._batch(real, rows, device), transform_ids
        )
        loss_sum += loss
        last = {"step": step, "loss": loss, "gradient_norm": gradient}
        if step in CANDIDATE_STEPS:
            capture(step)

    selected_step = select_candidate(candidate_metrics)
    model.load_state_dict(model_states[selected_step], strict=True)
    optimizer.load_state_dict(optimizer_states[selected_step])
    calibration, calibration_stream = _evaluate_split(model, "calibration", device)
    holdout, holdout_stream = _evaluate_split(model, "holdout", device)
    if calibration != candidate_metrics[selected_step]:
        raise RuntimeError("P5-E10 selected calibration replay differs")
    checks = {
        "calibration_recovery_min80pct": calibration["recovery_rate_6px"] >= 0.80,
        "holdout_recovery_min80pct": holdout["recovery_rate_6px"] >= 0.80,
        "calibration_mae_max3px": calibration["mae_pixels"] <= 3.0,
        "holdout_mae_max3px": holdout["mae_pixels"] <= 3.0,
    }
    decision = "real_calibration_go" if all(checks.values()) else "real_calibration_no_go"
    OUTPUT.mkdir(parents=True)
    global_step = 15_000 + selected_step
    checkpoint = OUTPUT / f"step-{global_step:06d}.pt"
    _atomic_torch_save(
        checkpoint,
        {
            "checkpoint_schema_version": "g3-p5e10-real-calibration-0.1",
            "model_type": MODEL_TYPE,
            "config": asdict(model.config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "source_global_step": 15_000,
            "selected_additional_steps": selected_step,
            "metrics": {
                "decision": decision,
                "checks": checks,
                "calibration": calibration,
                "holdout": holdout,
            },
        },
    )
    final = {
        "schema_version": "g3-p5e10-real-calibration-final-0.1",
        "model_type": MODEL_TYPE,
        "decision": decision,
        "checks": checks,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "parameter_count": model.parameter_count,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "selected_additional_steps": selected_step,
        "candidate_metrics": candidate_metrics,
        "candidate_streams": candidate_streams,
        "calibration": calibration,
        "holdout": holdout,
        "streams": {
            "calibration": calibration_stream,
            "holdout": holdout_stream,
        },
        "training": {
            "maximum_steps_run": MAXIMUM_STEPS,
            "learning_rate": LEARNING_RATE,
            "mean_loss_all_candidates": loss_sum / MAXIMUM_STEPS,
            "last": last,
        },
        "selected_state_sha256": module_state_sha256(model),
        "elapsed_seconds": time.perf_counter() - started,
        "confirm_v2_read": False,
        "dev_and_external_read": False,
    }
    _atomic_json(OUTPUT / "final_metrics.json", final)
    _atomic_json(OUTPUT / "audit.json", {"status": "completed", **final})
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
