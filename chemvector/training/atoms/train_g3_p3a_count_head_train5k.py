#!/usr/bin/env python3
"""Train only the P2-W Count head on train5k and evaluate route100 once."""

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

from chemvector.components.atoms.objects.g3_atom_query_set import (
    G3AtomQuerySet,
    G3AtomQuerySetConfig,
    G3PointerUtilityExistenceModel,
)
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import (
    train_g3_p2w_fixed_identity_pointer_utility_step4800 as p2w,
)
from chemvector.training.atoms import (
    audit_g3_p2x_joint_cardinality_decode_capacity64 as p2x,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
    module_state_sha256,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p3a-count-head-train5k"
OUTPUT = Path(f"outputs/checkpoints/{MODEL_TYPE}")
SOURCE_CHECKPOINT = p2w.OUTPUT / "step-004800.pt"
SOURCE_CHECKPOINT_SHA256 = (
    "229fd1299a68aa544c0bf3ee43acabcdc1386a9df3f57dfb153113c24328b6b9"
)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 20260713
TRAINABLE_PARAMETERS = 8_513
EXPECTED_TRAIN_IMAGES = 5_000
EXPECTED_ROUTE_IMAGES = 100
EXPECTED_STEPS = math.ceil(EXPECTED_TRAIN_IMAGES / BATCH_SIZE) * EPOCHS
EXPECTED_EXPOSURE = EXPECTED_TRAIN_IMAGES * EPOCHS


@torch.no_grad()
def evaluate_count(model, records, *, device: torch.device) -> dict:
    model.eval()
    absolute_error = exact = predicted_total = target_total = 0
    distribution = Counter()
    digest = hashlib.sha256()
    for start in range(0, len(records), BATCH_SIZE):
        current = records[start : start + BATCH_SIZE]
        batch = p2h.collate(current, device)
        with _autocast_context(device, "bf16"):
            output = model(p2h.model_candidates(batch))
        predicted = output["count_logits"].argmax(dim=-1).long()
        target = batch["full_atom_mask"].sum(dim=1).long()
        errors = predicted - target
        exact += int((errors == 0).sum())
        absolute_error += int(errors.abs().sum())
        predicted_total += int(predicted.sum())
        target_total += int(target.sum())
        distribution.update(int(value) for value in errors.cpu())
        digest.update(predicted.cpu().numpy().tobytes())
    return {
        "images": len(records),
        "count_exact_images": exact,
        "count_accuracy": exact / len(records),
        "count_mae": absolute_error / len(records),
        "predicted_atoms": predicted_total,
        "target_atoms": target_total,
        "count_error_distribution": dict(sorted(distribution.items())),
        "count_predictions_sha256": digest.hexdigest(),
    }


def train_step(model, optimizer, records, *, device: torch.device) -> dict:
    batch = p2h.collate(records, device)
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with _autocast_context(device, "bf16"):
        output = model(p2h.model_candidates(batch))
        targets = batch["full_atom_mask"].sum(dim=1).long()
        loss = F.cross_entropy(output["count_logits"], targets)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.base_model.count_head.parameters(), MAX_GRAD_NORM
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError("P3-A Count gradient norm is not finite")
    optimizer.step()
    if any(
        value.grad is not None
        for name, value in model.named_parameters()
        if not name.startswith("base_model.count_head.")
    ):
        raise RuntimeError("P3-A produced gradients outside Count head")
    return {
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm.detach()),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-A output is not empty: {OUTPUT}")
    if file_sha256(SOURCE_CHECKPOINT) != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("P3-A P2-W checkpoint SHA256 differs")
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

    source = torch.load(SOURCE_CHECKPOINT, map_location=device, weights_only=False)
    base = G3AtomQuerySet(G3AtomQuerySetConfig(**source["model_config"])).to(device)
    model = G3PointerUtilityExistenceModel(base).to(device)
    model.load_state_dict(source["atom_query_set"], strict=True)
    source_state_sha256 = module_state_sha256(model)
    model.requires_grad_(False)
    model.base_model.count_head.requires_grad_(True)
    count_initial_sha256 = module_state_sha256(model.base_model.count_head)
    trainable = sum(
        value.numel() for value in model.base_model.count_head.parameters()
    )
    if trainable != TRAINABLE_PARAMETERS:
        raise RuntimeError(f"P3-A Count parameter count changed: {trainable}")
    optimizer = torch.optim.AdamW(
        model.base_model.count_head.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    cache, cache_audit = p2e.load_cache()
    train = cache["train5k"]
    route = cache["route100"]
    if len(train) != EXPECTED_TRAIN_IMAGES or len(route) != EXPECTED_ROUTE_IMAGES:
        raise RuntimeError("P3-A train/route size changed")
    route_keys = {str(record["sample_key"]) for record in route}
    train_keys = {str(record["sample_key"]) for record in train}
    if route_keys & train_keys:
        raise RuntimeError("P3-A route100 overlaps train5k")

    OUTPUT.mkdir(parents=True, exist_ok=False)
    step0_count = evaluate_count(model, route, device=device)
    step0_joint = p2x.evaluate_joint(model, route, device=device)
    _atomic_json(OUTPUT / "step0_metrics.json", {
        "route100_count": step0_count,
        "route100_joint": step0_joint,
    })
    if (
        step0_count["count_exact_images"] != 3
        or step0_count["count_mae"] != 7.66
        or step0_joint["predicted_count"]["matched_atoms"] != 2_399
    ):
        raise RuntimeError("P3-A route100 baseline changed")

    _atomic_json(OUTPUT / "audit.json", {
        "schema_version": "g3-p3a-count-head-train5k-audit-0.1",
        "scope": "Count-only train5k generalization isolation",
        "trainable_parameters": TRAINABLE_PARAMETERS,
        "query_pointer_existence_frozen": True,
        "route100_used_for_training_or_selection": False,
        "fixed_gate": {
            "route100_count_exact_min": 70,
            "route100_count_mae_max": 1.0,
            "route100_joint_matched_atoms_min_exclusive": 2_399,
        },
        "lineage": {
            "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "source_model_state_sha256": source_state_sha256,
            "count_head_initial_sha256": count_initial_sha256,
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
        },
    })

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    step = exposure = 0
    started = time.perf_counter()
    log_path = OUTPUT / "train.jsonl"
    last = {}
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(len(train), generator=generator).tolist()
        loss_sum = gradient_sum = 0.0
        batches = 0
        for start in range(0, len(order), BATCH_SIZE):
            current = [train[index] for index in order[start : start + BATCH_SIZE]]
            last = train_step(model, optimizer, current, device=device)
            step += 1
            exposure += len(current)
            batches += 1
            loss_sum += last["loss"]
            gradient_sum += last["gradient_norm"]
        record = {
            "epoch": epoch,
            "step": step,
            "exposure": exposure,
            "loss": loss_sum / batches,
            "gradient_norm": gradient_sum / batches,
            "elapsed_seconds": time.perf_counter() - started,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(json.dumps(record, separators=(",", ":")), flush=True)

    train_count = evaluate_count(model, train, device=device)
    route_count = evaluate_count(model, route, device=device)
    route_joint = p2x.evaluate_joint(model, route, device=device)
    repeated_count = evaluate_count(model, route, device=device)
    repeated_joint = p2x.evaluate_joint(model, route, device=device)
    checks = {
        "route100_count_exact_min70": route_count["count_exact_images"] >= 70,
        "route100_count_mae_max1": route_count["count_mae"] <= 1.0,
        "route100_joint_matched_atoms_improved": (
            route_joint["predicted_count"]["matched_atoms"] > 2_399
        ),
    }
    protocol_checks = {
        "fixed_steps": step == EXPECTED_STEPS,
        "fixed_exposure": exposure == EXPECTED_EXPOSURE,
        "count_head_only": all(
            value.requires_grad == name.startswith("base_model.count_head.")
            for name, value in model.named_parameters()
        ),
        "repeat_count_metrics": repeated_count == route_count,
        "repeat_joint_metrics": repeated_joint == route_joint,
        "known_count_joint_unchanged": (
            route_joint["known_count"] == step0_joint["known_count"]
        ),
    }
    decision = (
        "count_head_train5k_generalization_go"
        if all(checks.values()) and all(protocol_checks.values())
        else "count_head_train5k_generalization_no_go"
    )
    final = {
        "decision": decision,
        "checks": checks,
        "protocol_checks": protocol_checks,
        "step0_count": step0_count,
        "train5k_count": train_count,
        "route100_count": route_count,
        "route100_joint": route_joint,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload = {
        "checkpoint_schema_version": "g3-p3a-count-head-train5k-0.1",
        "model_type": MODEL_TYPE,
        "atom_query_set": model.state_dict(),
        "model_config": asdict(model.config),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_images": len(train),
        "trainable_parameter_count": TRAINABLE_PARAMETERS,
        "frozen_parameter_count": model.parameter_count() - TRAINABLE_PARAMETERS,
        "last_losses": last,
        "metrics": final,
        "lineage": {
            "source_checkpoint": str(SOURCE_CHECKPOINT.resolve()),
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
            "source_model_state_sha256": source_state_sha256,
            "count_head_initial_sha256": count_initial_sha256,
        },
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
    print(json.dumps({
        "decision": decision,
        "route100_count": route_count,
        "route100_joint": route_joint["predicted_count"],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": final["checkpoint_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
