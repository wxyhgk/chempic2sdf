#!/usr/bin/env python3
"""P3-E10 data-scale and capacity A/B for the independent DeepSets Count."""

from __future__ import annotations

import argparse
from collections import defaultdict
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
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import train_g3_p3e9_deepsets_count as p3e9
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


ARM_CONFIGS = {
    "expanded64": {
        "model_type": "chemvector-g3-p3e10-deepsets-count-expanded3316-h64",
        "hidden_dim": 64,
    },
    "expanded32": {
        "model_type": "chemvector-g3-p3e10-deepsets-count-expanded3316-h32",
        "hidden_dim": 32,
    },
}
TRAIN_IMAGES = 3_316
TRAIN_MOLECULES = 829
EPOCHS = 30
BATCH_SIZE = 64
EXPECTED_STEPS = EPOCHS * math.ceil(TRAIN_IMAGES / BATCH_SIZE)
EXPECTED_EXPOSURE = EPOCHS * TRAIN_IMAGES
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 20260713


def expanded_splits(cache: dict) -> tuple[dict[str, list[dict]], dict]:
    e9_splits, e9_audit = p3e9.split_records(cache)
    route_splits, route_audit = p3e7.select_fresh_records(cache)
    protected = {
        p2h._molecule_key(str(record["sample_key"]))
        for split in ("dev256", "confirm256")
        for record in e9_splits[split]
    }
    protected.update(
        p2h._molecule_key(str(record["sample_key"]))
        for record in route_splits["route_confirm"]
    )
    groups = defaultdict(list)
    for record in cache["train5k"]:
        if torch.equal(record["atom_mask"], record["full_atom_mask"]):
            groups[p2h._molecule_key(str(record["sample_key"]))].append(record)
    training_groups = [
        (molecule, sorted(records, key=lambda value: str(value["sample_key"])))
        for molecule, records in sorted(groups.items())
        if len(records) == 4 and molecule not in protected
    ]
    train = [record for _, records in training_groups for record in records]
    if len(training_groups) != TRAIN_MOLECULES or len(train) != TRAIN_IMAGES:
        raise RuntimeError(
            f"P3-E10 expanded split changed: {len(training_groups)}/{len(train)}"
        )
    splits = {
        "train3316": train,
        "dev256": e9_splits["dev256"],
        "confirm256": e9_splits["confirm256"],
        "historical_route100": cache["route100"],
    }
    train_molecules = {molecule for molecule, _ in training_groups}
    dev_molecules = {
        p2h._molecule_key(str(record["sample_key"]))
        for record in splits["dev256"]
    }
    confirm_molecules = {
        p2h._molecule_key(str(record["sample_key"]))
        for record in splits["confirm256"]
    }
    if train_molecules & (dev_molecules | confirm_molecules):
        raise RuntimeError("P3-E10 train overlaps dev/confirm molecules")
    digest = hashlib.sha256()
    for split, records in splits.items():
        for record in records:
            digest.update(f"{split}:{record['sample_key']}\n".encode())
    return splits, {
        "selection": "all remaining four-variant molecules; fixed P3-E9 dev/confirm",
        "training_molecules": len(train_molecules),
        "protected_molecules": len(protected),
        "images": {name: len(records) for name, records in splits.items()},
        "cross_split_molecule_overlap": 0,
        "route100_training_or_selection": False,
        "e9_sample_keys_sha256": e9_audit["sample_keys_sha256"],
        "route_sample_keys_sha256": route_audit["sample_keys_sha256"],
        "sample_keys_sha256": digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARM_CONFIGS), required=True)
    args = parser.parse_args()
    arm = ARM_CONFIGS[args.arm]
    output = Path(f"outputs/smoke/{arm['model_type']}")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"P3-E10 output is not empty: {output}")
    for path, expected in (
        (p3e7.BASE_CHECKPOINT, p3e7.BASE_SHA256),
        (p3e7.EXPERT_CHECKPOINT, p3e7.EXPERT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P3-E10 checkpoint SHA differs: {path}")
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
    splits, selection_audit = expanded_splits(cache)
    base_model = p3e0.load_model(p3e7.BASE_CHECKPOINT, device)
    expert_model = p3e0.load_model(p3e7.EXPERT_CHECKPOINT, device)
    evidence_started = time.perf_counter()
    evidence = p3e9.collect_evidence(
        (base_model, expert_model), splits, device=device
    )
    evidence_seconds = time.perf_counter() - evidence_started
    del base_model, expert_model

    config = G3DeepSetsCountConfig(hidden_dim=arm["hidden_dim"])
    model = G3DeepSetsCount(config).to(device)
    trainable_parameters = sum(value.numel() for value in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    audit = {
        "schema_version": "g3-p3e10-count-scale-ab-audit-0.1",
        "model_type": arm["model_type"],
        "arm": args.arm,
        "fixed_protocol": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "expected_steps": EXPECTED_STEPS,
            "expected_exposure": EXPECTED_EXPOSURE,
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
    output.mkdir(parents=True, exist_ok=False)
    _atomic_json(output / "audit.json", audit)

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    train_records = evidence["train3316"]
    log_path = output / "train.jsonl"
    step = exposure = 0
    started = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(len(train_records), generator=generator).tolist()
        loss_sum = gradient_sum = clip_sum = 0.0
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            query, candidate, global_features, target = p3e9.batch(
                train_records, indices, device
            )
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, "bf16"):
                logits = model(query, candidate, global_features)
                loss = F.cross_entropy(logits, target)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), MAX_GRAD_NORM
            )
            optimizer.step()
            gradient_float = float(gradient_norm)
            loss_sum += float(loss.detach())
            gradient_sum += gradient_float
            clip_sum += min(1.0, MAX_GRAD_NORM / max(gradient_float, 1e-12))
            step += 1
            exposure += len(indices)
        batches = math.ceil(len(order) / BATCH_SIZE)
        log = {
            "epoch": epoch,
            "step": step,
            "exposure": exposure,
            "loss": loss_sum / batches,
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
        "count_scale_ab_go"
        if all(checks.values()) and all(protocol_checks.values())
        else "count_scale_ab_no_go"
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
        "checkpoint_schema_version": "g3-p3e10-count-scale-ab-0.1",
        "model_type": arm["model_type"],
        "arm": args.arm,
        "model_config": asdict(config),
        "deepsets_count": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_images": TRAIN_IMAGES,
        "trainable_parameter_count": trainable_parameters,
        "metrics": final,
        "lineage": audit["lineage"],
    }
    checkpoint = output / f"step-{step:06d}.pt"
    _atomic_torch_save(checkpoint, payload)
    _atomic_torch_save(output / "last.pt", payload)
    final["checkpoint"] = str(checkpoint.resolve())
    final["checkpoint_sha256"] = file_sha256(checkpoint)
    _atomic_json(output / "final_metrics.json", final)
    _atomic_json(output / "status.json", {
        "status": "completed",
        "decision": decision,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": final["checkpoint_sha256"],
    })
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
