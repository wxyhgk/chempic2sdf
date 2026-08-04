#!/usr/bin/env python3
"""Train P3-E13 local Count refinement on exact cardinality curves."""

from __future__ import annotations

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

from chemvector.components.atoms.objects.g3_cardinality_curve import (
    G3_CARDINALITY_CURVE_FEATURES,
    G3CardinalityCurveConfig,
    G3CardinalityCurveRefiner,
    g3_cardinality_curve_features,
    g3_cardinality_offset_target,
)
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
from chemvector.training.atoms import train_g3_p3e10_count_scale_ab as p3e10
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


MODEL_TYPE = "chemvector-g3-p3e13-cardinality-curve-expanded3316-h64"
OUTPUT = Path(f"outputs/smoke/{MODEL_TYPE}")
COARSE_CHECKPOINT = Path(
    "outputs/smoke/chemvector-g3-p3e10-deepsets-count-expanded3316-h64/"
    "step-001560.pt"
)
COARSE_SHA256 = "9f734bf72d1f0c432dd2d7da14212bc94b013caf12849f30c752801a1a5bb3b7"
TRAIN_IMAGES = p3e10.TRAIN_IMAGES
EPOCHS = 30
BATCH_SIZE = 64
EXPECTED_STEPS = EPOCHS * math.ceil(TRAIN_IMAGES / BATCH_SIZE)
EXPECTED_EXPOSURE = EPOCHS * TRAIN_IMAGES
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 20260713


def load_coarse_model(device: torch.device) -> G3DeepSetsCount:
    source = torch.load(COARSE_CHECKPOINT, map_location=device, weights_only=False)
    config = G3DeepSetsCountConfig(**source["model_config"])
    model = G3DeepSetsCount(config).to(device)
    model.load_state_dict(source["deepsets_count"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def collect_curve_evidence(models, splits, *, device) -> tuple[dict, str]:
    base_model, expert_model, coarse_model = models
    for model in models:
        model.eval()
    config = G3CardinalityCurveConfig()
    result = {}
    digest = hashlib.sha256()
    for split, records in splits.items():
        values = []
        for start in range(0, len(records), 16):
            current = records[start : start + 16]
            batch = p2h.collate(current, device)
            with _autocast_context(device, "bf16"):
                candidates = p2h.model_candidates(batch)
                base = base_model(candidates)
                expert = expert_model(candidates)
                query, candidate, global_features = g3_deepsets_count_features(
                    base,
                    expert,
                    batch["candidate_features"],
                    batch["raw_scores"],
                    batch["normalized_scores"],
                    batch["scale_ranks"],
                )
                coarse_logits = coarse_model(query, candidate, global_features)
            labels = batch["full_atom_mask"].sum(dim=1).long().cpu()
            for index, record in enumerate(current):
                base_scores = p3e0.edge_scores({
                    "existence_logits": base["existence_logits"][index].float(),
                    "pointer_logits": base["pointer_logits"][index].float(),
                }, "legacy_logsigmoid").cpu()
                expert_scores = p3e0.edge_scores({
                    "existence_logits": expert["existence_logits"][index].float(),
                    "pointer_logits": expert["pointer_logits"][index].float(),
                }, "legacy_logsigmoid").cpu()
                curve, counts, coarse_count = g3_cardinality_curve_features(
                    base_scores,
                    expert_scores,
                    coarse_logits[index].float().cpu(),
                    config=config,
                )
                label = int(labels[index])
                target, in_window = g3_cardinality_offset_target(
                    label, coarse_count, radius=config.radius
                )
                value = {
                    "sample_key": str(record["sample_key"]),
                    "curve": curve,
                    "counts": counts,
                    "target": target,
                    "label": label,
                    "coarse_count": coarse_count,
                    "in_window": in_window,
                }
                values.append(value)
                digest.update(f"{split}:{value['sample_key']}:".encode())
                digest.update(curve.numpy().tobytes())
                digest.update(torch.tensor(
                    (target, label, coarse_count, int(in_window)), dtype=torch.int64
                ).numpy().tobytes())
        result[split] = values
    return result, digest.hexdigest()


def batch(records: list[dict], indices: list[int], device) -> tuple[torch.Tensor, ...]:
    selected = [records[index] for index in indices]
    return (
        torch.stack([value["curve"] for value in selected]).to(device),
        torch.tensor([value["target"] for value in selected], device=device),
    )


@torch.no_grad()
def evaluate(model, records: list[dict], *, device) -> dict:
    model.eval()
    predicted = []
    labels = []
    coarse = []
    in_window = []
    digest = hashlib.sha256()
    for start in range(0, len(records), BATCH_SIZE):
        indices = list(range(start, min(start + BATCH_SIZE, len(records))))
        curve, _ = batch(records, indices, device)
        with _autocast_context(device, "bf16"):
            offset_classes = model(curve).argmax(dim=-1).long().cpu()
        for class_index, record in zip(offset_classes.tolist(), (records[i] for i in indices)):
            count = min(max(
                record["coarse_count"] + class_index - model.config.radius,
                0,
            ), model.config.max_count)
            predicted.append(count)
            labels.append(record["label"])
            coarse.append(record["coarse_count"])
            in_window.append(record["in_window"])
        digest.update(offset_classes.numpy().tobytes())
    errors = [value - target for value, target in zip(predicted, labels)]
    coarse_errors = [value - target for value, target in zip(coarse, labels)]
    window_errors = [
        error for error, covered in zip(errors, in_window) if covered
    ]
    return {
        "images": len(records),
        "count_exact": sum(value == 0 for value in errors),
        "count_accuracy": sum(value == 0 for value in errors) / len(errors),
        "count_mae": sum(abs(value) for value in errors) / len(errors),
        "coarse_count_exact": sum(value == 0 for value in coarse_errors),
        "coarse_count_mae": sum(abs(value) for value in coarse_errors) / len(errors),
        "window_oracle_exact": sum(in_window),
        "window_oracle_accuracy": sum(in_window) / len(in_window),
        "refiner_exact_within_window": sum(value == 0 for value in window_errors),
        "refiner_accuracy_within_window": (
            sum(value == 0 for value in window_errors) / len(window_errors)
            if window_errors else 0.0
        ),
        "outside_window": len(in_window) - sum(in_window),
        "error_min": min(errors),
        "error_max": max(errors),
        "prediction_digest": digest.hexdigest(),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E13 output is not empty: {OUTPUT}")
    for path, expected in (
        (p3e7.BASE_CHECKPOINT, p3e7.BASE_SHA256),
        (p3e7.EXPERT_CHECKPOINT, p3e7.EXPERT_SHA256),
        (COARSE_CHECKPOINT, COARSE_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P3-E13 checkpoint SHA differs: {path}")
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
    models = (
        p3e0.load_model(p3e7.BASE_CHECKPOINT, device),
        p3e0.load_model(p3e7.EXPERT_CHECKPOINT, device),
        load_coarse_model(device),
    )
    evidence_started = time.perf_counter()
    evidence, evidence_sha256 = collect_curve_evidence(models, splits, device=device)
    evidence_seconds = time.perf_counter() - evidence_started
    del models

    config = G3CardinalityCurveConfig()
    model = G3CardinalityCurveRefiner(config).to(device)
    trainable_parameters = sum(value.numel() for value in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    audit = {
        "schema_version": "g3-p3e13-cardinality-curve-audit-0.1",
        "model_type": MODEL_TYPE,
        "fixed_protocol": {
            "window_offsets": list(range(-config.radius, config.radius + 1)),
            "outside_window_target": "clip_to_boundary_class_and_report_separately",
            "assignment": "exact fixed-cardinality one-to-one MAP for base and expert",
            "edge_score": "logsigmoid(existence)+logsoftmax(pointer)",
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
        "curve_features": list(G3_CARDINALITY_CURVE_FEATURES),
        "trainable_parameters": trainable_parameters,
        "lineage": {
            "base_checkpoint": str(p3e7.BASE_CHECKPOINT.resolve()),
            "base_sha256": p3e7.BASE_SHA256,
            "expert_checkpoint": str(p3e7.EXPERT_CHECKPOINT.resolve()),
            "expert_sha256": p3e7.EXPERT_SHA256,
            "coarse_checkpoint": str(COARSE_CHECKPOINT.resolve()),
            "coarse_sha256": COARSE_SHA256,
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
            "selection": selection_audit,
            "evidence_sha256": evidence_sha256,
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
        loss_sum = gradient_sum = clip_sum = 0.0
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            curve, target = batch(train_records, indices, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, "bf16"):
                loss = F.cross_entropy(model(curve), target)
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
        "confirm_count_accuracy_min50": (
            metrics["confirm256"]["count_accuracy"] >= 0.5
        ),
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
        "cardinality_curve_go"
        if all(checks.values()) and all(protocol_checks.values())
        else "cardinality_curve_no_go"
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
        "checkpoint_schema_version": "g3-p3e13-cardinality-curve-0.1",
        "model_type": MODEL_TYPE,
        "model_config": asdict(config),
        "cardinality_curve_refiner": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "samples_seen": exposure,
        "training_images": TRAIN_IMAGES,
        "trainable_parameter_count": trainable_parameters,
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
