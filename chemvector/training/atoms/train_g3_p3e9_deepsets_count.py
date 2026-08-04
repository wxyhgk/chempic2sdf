#!/usr/bin/env python3
"""Train an independent DeepSets Count head on frozen G3 base/expert evidence."""

from __future__ import annotations

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
    g3_deepsets_count_features,
)
from chemvector.training.atoms import audit_g3_p3e0_latent_solver as p3e0
from chemvector.training.atoms import audit_g3_p3e7_base_expert_router as p3e7
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import train_g3_p3e1_latent_structured_pilot256 as p3e1
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p3e9-deepsets-count-train1760"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
TRAIN_MOLECULES = 440
DEV_MOLECULES = 64
CONFIRM_MOLECULES = 64
IMAGES_PER_MOLECULE = 4
TRAIN_IMAGES = TRAIN_MOLECULES * IMAGES_PER_MOLECULE
DEV_IMAGES = DEV_MOLECULES * IMAGES_PER_MOLECULE
CONFIRM_IMAGES = CONFIRM_MOLECULES * IMAGES_PER_MOLECULE
EPOCHS = 30
BATCH_SIZE = 64
EXPECTED_STEPS = EPOCHS * math.ceil(TRAIN_IMAGES / BATCH_SIZE)
EXPECTED_EXPOSURE = EPOCHS * TRAIN_IMAGES
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 20260713


def split_records(cache: dict) -> tuple[dict[str, list[dict]], dict]:
    route_splits, route_audit = p3e7.select_fresh_records(cache)
    domains, _ = p2h.select_capacity_domains(cache["train5k"])
    capacity = domains["easy32"] + domains["hard32"]
    capacity_keys = {str(record["sample_key"]) for record in capacity}
    eligible = [
        record for record in cache["train5k"]
        if torch.equal(record["atom_mask"], record["full_atom_mask"])
    ]
    historical_general = sorted(
        (
            record for record in eligible
            if str(record["sample_key"]) not in capacity_keys
        ),
        key=lambda record: str(record["sample_key"]),
    )[:p3e0.GENERAL_IMAGES]
    excluded = {
        p2h._molecule_key(str(record["sample_key"]))
        for record in (
            capacity
            + historical_general
            + route_splits["route_select"]
            + route_splits["route_confirm"]
        )
    }
    groups = defaultdict(list)
    for record in eligible:
        groups[p2h._molecule_key(str(record["sample_key"]))].append(record)
    available = [
        (molecule, sorted(records, key=lambda value: str(value["sample_key"])))
        for molecule, records in sorted(groups.items())
        if molecule not in excluded and len(records) == IMAGES_PER_MOLECULE
    ]
    required = TRAIN_MOLECULES + DEV_MOLECULES + CONFIRM_MOLECULES
    if len(available) < required:
        raise RuntimeError(f"P3-E9 has only {len(available)} clean molecule groups")
    selected = available[:required]
    boundaries = (TRAIN_MOLECULES, TRAIN_MOLECULES + DEV_MOLECULES)
    group_splits = {
        "train1760": selected[: boundaries[0]],
        "dev256": selected[boundaries[0] : boundaries[1]],
        "confirm256": selected[boundaries[1] :],
    }
    splits = {
        name: [record for _, records in values for record in records]
        for name, values in group_splits.items()
    }
    splits["historical_route100"] = cache["route100"]
    molecule_sets = {
        name: {molecule for molecule, _ in values}
        for name, values in group_splits.items()
    }
    if any(
        molecule_sets[left] & molecule_sets[right]
        for left, right in (("train1760", "dev256"), ("train1760", "confirm256"), ("dev256", "confirm256"))
    ):
        raise RuntimeError("P3-E9 split molecule overlap")
    digest = hashlib.sha256()
    for split, records in splits.items():
        for record in records:
            digest.update(f"{split}:{record['sample_key']}\n".encode())
    audit = {
        "selection": "all-four-variant molecule groups after excluding P3-E1/P3-E7 molecules",
        "excluded_molecules": len(excluded),
        "available_clean_molecules": len(available),
        "images": {name: len(records) for name, records in splits.items()},
        "molecules": {name: len(values) for name, values in molecule_sets.items()},
        "cross_split_molecule_overlap": 0,
        "route100_training_use": False,
        "route_audit_sample_keys_sha256": route_audit["sample_keys_sha256"],
        "sample_keys_sha256": digest.hexdigest(),
        "sample_keys": {
            name: [str(record["sample_key"]) for record in records]
            for name, records in splits.items()
        },
    }
    return splits, audit


@torch.no_grad()
def collect_evidence(models, splits, *, device) -> dict[str, list[dict]]:
    base_model, expert_model = models
    base_model.eval()
    expert_model.eval()
    result = {}
    for split, records in splits.items():
        values = []
        for start in range(0, len(records), 16):
            current = records[start : start + 16]
            batch = p2h.collate(current, device)
            with _autocast_context(device, "bf16"):
                base = base_model(p2h.model_candidates(batch))
                expert = expert_model(p2h.model_candidates(batch))
            query, candidate, global_features = g3_deepsets_count_features(
                base,
                expert,
                batch["candidate_features"],
                batch["raw_scores"],
                batch["normalized_scores"],
                batch["scale_ranks"],
            )
            labels = batch["full_atom_mask"].sum(dim=1).long()
            baseline = base["count_logits"].argmax(dim=-1).long()
            for index, record in enumerate(current):
                values.append({
                    "sample_key": str(record["sample_key"]),
                    "query": query[index].cpu(),
                    "candidate": candidate[index].cpu(),
                    "global": global_features[index].cpu(),
                    "label": int(labels[index]),
                    "baseline_count": int(baseline[index]),
                })
        result[split] = values
    return result


def batch(records: list[dict], indices: list[int], device) -> tuple[torch.Tensor, ...]:
    selected = [records[index] for index in indices]
    return (
        torch.stack([value["query"] for value in selected]).to(device),
        torch.stack([value["candidate"] for value in selected]).to(device),
        torch.stack([value["global"] for value in selected]).to(device),
        torch.tensor([value["label"] for value in selected], device=device),
    )


@torch.no_grad()
def evaluate(model, records: list[dict], *, device) -> dict:
    model.eval()
    predicted = []
    labels = []
    digest = hashlib.sha256()
    for start in range(0, len(records), BATCH_SIZE):
        indices = list(range(start, min(start + BATCH_SIZE, len(records))))
        query, candidate, global_features, target = batch(records, indices, device)
        with _autocast_context(device, "bf16"):
            logits = model(query, candidate, global_features)
        current = logits.argmax(dim=-1).long().cpu().tolist()
        predicted.extend(current)
        labels.extend(target.cpu().tolist())
        digest.update(torch.tensor(current, dtype=torch.int64).numpy().tobytes())
    errors = [prediction - target for prediction, target in zip(predicted, labels)]
    baseline_errors = [value["baseline_count"] - value["label"] for value in records]
    return {
        "images": len(records),
        "count_exact": sum(value == 0 for value in errors),
        "count_accuracy": sum(value == 0 for value in errors) / len(errors),
        "count_mae": sum(abs(value) for value in errors) / len(errors),
        "baseline_count_exact": sum(value == 0 for value in baseline_errors),
        "baseline_count_mae": sum(abs(value) for value in baseline_errors) / len(errors),
        "error_min": min(errors),
        "error_max": max(errors),
        "prediction_digest": digest.hexdigest(),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E9 output is not empty: {OUTPUT}")
    for path, expected in (
        (p3e7.BASE_CHECKPOINT, p3e7.BASE_SHA256),
        (p3e7.EXPERT_CHECKPOINT, p3e7.EXPERT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P3-E9 checkpoint SHA differs: {path}")
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
    splits, selection_audit = split_records(cache)
    base_model = p3e0.load_model(p3e7.BASE_CHECKPOINT, device)
    expert_model = p3e0.load_model(p3e7.EXPERT_CHECKPOINT, device)
    evidence_started = time.perf_counter()
    evidence = collect_evidence((base_model, expert_model), splits, device=device)
    evidence_seconds = time.perf_counter() - evidence_started
    del base_model, expert_model

    config = G3DeepSetsCountConfig()
    model = G3DeepSetsCount(config).to(device)
    trainable_parameters = sum(value.numel() for value in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    output_audit = {
        "schema_version": "g3-p3e9-deepsets-count-audit-0.1",
        "model_type": MODEL_TYPE,
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
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "audit.json", output_audit)

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    step = exposure = 0
    started = time.perf_counter()
    log_path = OUTPUT / "train.jsonl"
    train_records = evidence["train1760"]
    for epoch in range(1, EPOCHS + 1):
        order = torch.randperm(len(train_records), generator=generator).tolist()
        loss_sum = gradient_sum = clip_sum = 0.0
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            query, candidate, global_features, target = batch(
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
        name: evaluate(model, records, device=device)
        for name, records in evidence.items()
    }
    repeated = {
        name: evaluate(model, records, device=device)
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
        "deepsets_count_go"
        if all(checks.values()) and all(protocol_checks.values())
        else "deepsets_count_no_go"
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
        "checkpoint_schema_version": "g3-p3e9-deepsets-count-0.1",
        "model_type": MODEL_TYPE,
        "model_config": asdict(config),
        "deepsets_count": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_images": TRAIN_IMAGES,
        "trainable_parameter_count": trainable_parameters,
        "metrics": final,
        "lineage": output_audit["lineage"],
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
