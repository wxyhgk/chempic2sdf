#!/usr/bin/env python3
"""Decompose P4-E9B failure into image trigger, pointer rank, and offset quality."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e29_bond_incident_signal as p3e29,
)
from chemvector.evaluation.evaluators.atoms import (
    evaluate_g3_p4e9_local_recovery_pilot20k as p4eval,
)
from chemvector.training.atoms import (
    train_g3_p4e9_local_recovery_pilot20k as p4train,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p5e0-action-decomposition"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
MODEL_TYPE = "chemvector-g3-p4e9b-structural-anchor-refine-pilot20k"
CHECKPOINT = Path(f"outputs/checkpoints/{MODEL_TYPE}/step-005000.pt")
CHECKPOINT_SHA256 = (
    "a564510ff69eed1da4dc2077ef68617be935881406fd15e20d84b95b400cf500"
)
ORACLE_AUDIT_SHA256 = (
    "5dc3189f7901a42edcc0d97b27cd43c8e8bb47b33573db4605d560cce0179698"
)
ORACLE_ACTIONS_SHA256 = (
    "30aff390726d2e50388e45b37bd3710be19fab8b1d08a1e5fa81bfd3a4ce5819"
)
ORACLE_IMAGES_SHA256 = (
    "b1f6db4da1735e20113a76439c761d0cad3f024e0935aef9747832afde138a9a"
)
MATCH_RADIUS_PIXELS = 6.0
IMAGE_SIZE = 768.0
TOP_K = (1, 2, 4, 8, 16)
SEED = 20260852


def _stable_ranks(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 1 or not logits.is_floating_point():
        raise ValueError("P5-E0 logits must be a floating vector")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("P5-E0 logits must be finite")
    order = torch.argsort(logits, descending=True, stable=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(order))
    return ranks


def decompose_records(records: list[dict]) -> dict:
    """Measure independent upper bounds for trigger, pointer, and offset stages."""

    if not records:
        raise ValueError("P5-E0 requires online records")
    action_counts = Counter()
    image_scores = []
    image_labels = []
    pointer_ranks = []
    offset_distances = []
    source_missing = 0
    for record in records:
        logits = record["logits"].float()
        offsets = record["offsets"].float()
        anchors = record["base_anchors"].float()
        source_ids = record["source_ids"].long()
        actions = list(record.get("oracle_actions", []))
        if logits.shape != (len(source_ids),) or offsets.shape != anchors.shape:
            raise ValueError("P5-E0 record geometry differs")
        action_counts[len(actions)] += 1
        image_scores.append(float(logits.max()))
        image_labels.append(bool(actions))
        ranks = _stable_ranks(logits)
        for action in actions:
            rows = torch.where(source_ids == int(action["base_source_id"]))[0]
            if len(rows) != 1:
                source_missing += 1
                continue
            index = int(rows[0])
            pointer_ranks.append(int(ranks[index]) + 1)
            target = torch.tensor(action["target_anchor"], dtype=torch.float32)
            addition = (anchors[index] + offsets[index]).clamp(0.0, 1.0)
            offset_distances.append(
                float(torch.linalg.vector_norm((addition - target) * IMAGE_SIZE))
            )
    labels = torch.tensor(image_labels, dtype=torch.bool)
    scores = torch.tensor(image_scores, dtype=torch.float32)
    if min(int(labels.sum()), int((~labels).sum())) == 0:
        raise ValueError("P5-E0 image trigger AUC requires both classes")
    oracle_actions = sum(action_counts[count] * count for count in action_counts)
    pointer = {
        f"top{value}": sum(rank <= value for rank in pointer_ranks)
        for value in TOP_K
    }
    pointer_offset = {
        f"top{value}_and_offset6px": sum(
            rank <= value and distance <= MATCH_RADIUS_PIXELS
            for rank, distance in zip(pointer_ranks, offset_distances)
        )
        for value in TOP_K
    }
    offset_recovered = sum(value <= MATCH_RADIUS_PIXELS for value in offset_distances)
    return {
        "images": len(records),
        "action_images": int(labels.sum()),
        "no_action_images": int((~labels).sum()),
        "oracle_actions": oracle_actions,
        "action_count_distribution": {
            str(key): value for key, value in sorted(action_counts.items())
        },
        "image_trigger_auc": p3e29.binary_auc(scores, labels),
        "image_scores": scores,
        "image_labels": labels,
        "source_missing_top128": source_missing,
        "pointer_rank": {
            **pointer,
            "median": float(np.median(pointer_ranks)) if pointer_ranks else None,
            "p90": float(np.percentile(pointer_ranks, 90)) if pointer_ranks else None,
        },
        "forced_correct_source_offset": {
            "within_6px": offset_recovered,
            "rate": offset_recovered / max(len(offset_distances), 1),
            "mae_pixels": float(np.mean(offset_distances)) if offset_distances else None,
            "p90_pixels": float(np.percentile(offset_distances, 90))
            if offset_distances
            else None,
        },
        "pointer_and_offset": pointer_offset,
    }


def calibrate_image_zero_fp(summary: dict) -> float:
    scores = summary["image_scores"]
    labels = summary["image_labels"]
    negatives = scores[~labels]
    if not len(negatives):
        raise ValueError("P5-E0 image calibration lacks no-action images")
    return math.nextafter(float(negatives.max()), math.inf)


def evaluate_image_threshold(summary: dict, threshold: float) -> dict:
    scores = summary["image_scores"].double()
    labels = summary["image_labels"]
    predicted = scores >= float(threshold)
    tp = int((predicted & labels).sum())
    fp = int((predicted & ~labels).sum())
    return {
        "threshold": float(threshold),
        "executed_images": int(predicted.sum()),
        "true_positive_images": tp,
        "false_positive_images": fp,
        "action_image_recall": tp / max(int(labels.sum()), 1),
        "zero_false_positive": fp == 0,
    }


def _public_summary(value: dict) -> dict:
    return {
        key: item
        for key, item in value.items()
        if key not in {"image_scores", "image_labels"}
    }


def _manifest(path: Path):
    records, _ = p4eval._manifest(path)
    return records


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P5-E0 output is not empty: {OUTPUT}")
    for path, expected in (
        (CHECKPOINT, CHECKPOINT_SHA256),
        (p4train.ORACLE_AUDIT, ORACLE_AUDIT_SHA256),
        (p4train.ORACLE_ACTIONS, ORACLE_ACTIONS_SHA256),
        (p4eval.ORACLE_IMAGES, ORACLE_IMAGES_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P5-E0 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P5-E0 requires CUDA BF16")
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

    model, _, _ = p4eval.load_route_checkpoint(
        "refine", CHECKPOINT_SHA256, ORACLE_AUDIT_SHA256, device
    )
    actions = p4eval._jsonl(p4train.ORACLE_ACTIONS)
    field, atom_head, vision, frozen_block, vision_sha = __import__(
        "chemvector.evaluation.audits.atoms.audit_g3_p4e6_missing_peak_sources",
        fromlist=["_load_model"],
    )._load_model(device)
    started = time.perf_counter()
    domains = {}
    stream_audits = {}
    for name, path in (
        ("calibration", p4eval.ORACLE_CALIBRATION),
        ("holdout", p4eval.ORACLE_HOLDOUT),
    ):
        manifest = _manifest(path)
        records, stream = p4eval.collect_online_records(
            manifest,
            route="refine",
            route_actions=p4eval._actions_by_sample(actions, "refine", name),
            dynamic_oracle=False,
            include_top1024=False,
            model=model,
            field=field,
            atom_head=atom_head,
            vision=vision,
            frozen_block=frozen_block,
            device=device,
        )
        domains[name] = decompose_records(records)
        stream_audits[name] = stream
    threshold = calibrate_image_zero_fp(domains["calibration"])
    trigger_threshold = {
        name: evaluate_image_threshold(value, threshold)
        for name, value in domains.items()
    }
    capability = {
        "pointer_top4_min50pct": (
            domains["holdout"]["pointer_rank"]["top4"]
            / max(domains["holdout"]["oracle_actions"], 1)
            >= 0.50
        ),
        "forced_offset6px_min80pct": (
            domains["holdout"]["forced_correct_source_offset"]["rate"] >= 0.80
        ),
        "image_trigger_zero_fp_has_nonzero_recall": (
            trigger_threshold["calibration"]["true_positive_images"] > 0
        ),
    }
    audit = {
        "schema_version": "g3-p5e0-action-decomposition-audit-0.1",
        "decision": (
            "set_conditioned_capacity_go"
            if capability["pointer_top4_min50pct"]
            and capability["forced_offset6px_min80pct"]
            else "representation_or_offset_no_go"
        ),
        "capability_checks": capability,
        "domains": {name: _public_summary(value) for name, value in domains.items()},
        "image_trigger_zero_fp": trigger_threshold,
        "protocol": {
            "image_trigger": "maximum refine logit per image",
            "pointer_rank": "stable descending candidate logit rank of Oracle source",
            "forced_offset": "model offset when Oracle source is forced",
            "match_radius_pixels": MATCH_RADIUS_PIXELS,
            "training_performed": False,
        },
        "streams": stream_audits,
        "lineage": {
            "checkpoint": str(CHECKPOINT.resolve()),
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "oracle_audit_sha256": ORACLE_AUDIT_SHA256,
            "vision_checkpoint_sha256": vision_sha,
        },
        "protocol_checks": {
            "calibration_images_3000": domains["calibration"]["images"] == 3000,
            "holdout_images_3000": domains["holdout"]["images"] == 3000,
            "source_ids_complete": all(
                value["source_missing_top128"] == 0 for value in domains.values()
            ),
            "confirm_v2_not_read": True,
            "external_and_sealed_not_read": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    if not all(audit["protocol_checks"].values()):
        audit["decision"] = "action_decomposition_protocol_failed"
    OUTPUT.mkdir(parents=True, exist_ok=False)
    from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json

    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
