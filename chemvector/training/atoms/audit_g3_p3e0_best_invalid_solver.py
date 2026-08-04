#!/usr/bin/env python3
"""Preflight the exact Hall-witness best-invalid matching oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_latent_matching import (
    g3_best_invalid_assignment,
    g3_candidate_set_covers_gt,
    g3_gt_candidate_compatibility,
    g3_hall_witnesses,
    g3_latent_positive_assignment,
)
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import audit_g3_p3e0_latent_solver as p3e0
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json


AUDIT_NAME = "chemvector-g3-p3e0-best-invalid-solver"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
MAX_COMPONENT_GT = 8
MAX_WITNESSES = 256


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def witness_statistics(records) -> dict:
    component_sizes = []
    witness_counts = []
    for record in records:
        target = record["atom_anchors"][record["full_atom_mask"]].float()
        candidates = record["candidate_anchors"].float()
        compatibility = g3_gt_candidate_compatibility(target, candidates)
        adjacency = (
            compatibility.to(torch.int16) @ compatibility.to(torch.int16).T
        ) > 0
        remaining = set(range(len(target)))
        sizes = []
        while remaining:
            stack = [min(remaining)]
            component = set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(
                    value for value in torch.where(adjacency[current])[0].tolist()
                    if value not in component
                )
            remaining -= component
            sizes.append(len(component))
        witnesses = g3_hall_witnesses(
            compatibility,
            max_component_gt=MAX_COMPONENT_GT,
            max_witnesses=MAX_WITNESSES,
        )
        component_sizes.append(max(sizes))
        witness_counts.append(len(witnesses))
    return {
        "images": len(records),
        "component_gt_p50": percentile(component_sizes, 50),
        "component_gt_p95": percentile(component_sizes, 95),
        "component_gt_max": max(component_sizes),
        "witness_count_p50": percentile(witness_counts, 50),
        "witness_count_p95": percentile(witness_counts, 95),
        "witness_count_max": max(witness_counts),
    }


def evaluate(records, outputs) -> tuple[dict, str]:
    latencies = []
    margins = []
    witness_counts = []
    unrestricted_invalid = 0
    missing_invalid = 0
    digest = hashlib.sha256()
    for record in records:
        key = str(record["sample_key"])
        target = record["atom_anchors"][record["full_atom_mask"]].float()
        candidates = record["candidate_anchors"].float()
        compatibility = g3_gt_candidate_compatibility(target, candidates)
        scores = p3e0.edge_scores(outputs[key], "legacy_logsigmoid")
        positive = g3_latent_positive_assignment(scores, compatibility)
        started = time.perf_counter()
        negative = g3_best_invalid_assignment(
            scores,
            compatibility,
            max_component_gt=MAX_COMPONENT_GT,
            max_witnesses=MAX_WITNESSES,
        )
        latencies.append((time.perf_counter() - started) * 1_000.0)
        if negative is None:
            missing_invalid += 1
            continue
        if g3_candidate_set_covers_gt(
            negative["candidate_indices"], compatibility
        ):
            raise RuntimeError("P3-E0 best-invalid audit returned valid coverage")
        unrestricted_invalid += int(negative["witness_index"] == -1)
        witness_counts.append(int(negative["witness_count"]))
        margin = (positive["score"] - negative["score"]) / len(target)
        margins.append(margin)
        digest.update(
            f"{key}:{negative['witness_index']}:"
            f"{tuple(sorted(zip(negative['query_indices'].tolist(), negative['candidate_indices'].tolist())))}:"
            f"{negative['score']:.12f}:{positive['score']:.12f}\n".encode()
        )
    return {
        "images": len(records),
        "images_without_invalid_matching": missing_invalid,
        "unrestricted_map_already_invalid": unrestricted_invalid,
        "hall_constrained_negative_images": (
            len(records) - missing_invalid - unrestricted_invalid
        ),
        "best_invalid_latency_ms_p50": percentile(latencies, 50),
        "best_invalid_latency_ms_p95": percentile(latencies, 95),
        "best_invalid_latency_ms_max": max(latencies),
        "valid_minus_invalid_margin_mean": float(np.mean(margins)),
        "valid_minus_invalid_margin_p10": percentile(margins, 10),
        "valid_minus_invalid_margin_p50": percentile(margins, 50),
        "valid_minus_invalid_margin_p90": percentile(margins, 90),
        "valid_margin_ge1_images": sum(value >= 1.0 for value in margins),
        "witness_count_max_when_used": max(witness_counts, default=0),
    }, digest.hexdigest()


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E0 best-invalid output is not empty: {OUTPUT}")
    torch.manual_seed(20260713)
    torch.cuda.manual_seed_all(20260713)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    cache, cache_audit = p2e.load_cache()
    eligible = [
        record for record in cache["train5k"]
        if torch.equal(record["atom_mask"], record["full_atom_mask"])
    ]
    domains, selection_audit = p2h.select_capacity_domains(cache["train5k"])
    capacity = domains["easy32"] + domains["hard32"]
    capacity_keys = {str(record["sample_key"]) for record in capacity}
    general = sorted(
        (
            record for record in eligible
            if str(record["sample_key"]) not in capacity_keys
        ),
        key=lambda record: str(record["sample_key"]),
    )[:p3e0.GENERAL_IMAGES]
    records = capacity + general

    path, expected_sha = p3e0.CHECKPOINTS["p2w"]
    if p3e0.file_sha256(path) != expected_sha:
        raise ValueError("P3-E0 best-invalid P2-W checkpoint SHA256 differs")
    model = p3e0.load_model(path, device)
    outputs = p3e0.collect_outputs(
        model, records, device=device, batch_size=4, precision="bf16"
    )
    first, first_digest = evaluate(records, outputs)
    repeated, repeated_digest = evaluate(records, outputs)
    compatibility = witness_statistics(eligible)
    checks = {
        "component_gt_max8": compatibility["component_gt_max"] <= MAX_COMPONENT_GT,
        "witness_count_max256": compatibility["witness_count_max"] <= MAX_WITNESSES,
        "best_invalid_p95_under100ms": first["best_invalid_latency_ms_p95"] <= 100.0,
        "repeat_digest_equal": first_digest == repeated_digest,
        "repeat_metrics_equal_except_latency": all(
            first[name] == repeated[name]
            for name in first
            if "latency" not in name
        ),
    }
    payload = {
        "schema_version": "g3-p3e0-best-invalid-solver-audit-0.1",
        "audit_name": AUDIT_NAME,
        "decision": (
            "best_invalid_solver_preflight_go"
            if all(checks.values()) else "best_invalid_solver_preflight_no_go"
        ),
        "checks": checks,
        "compatibility_train4937": compatibility,
        "sample256": first,
        "sample256_digest": first_digest,
        "lineage": {
            "checkpoint": str(path.resolve()),
            "checkpoint_sha256": expected_sha,
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
            "capacity_selection": selection_audit,
            "general_sample_keys": [str(record["sample_key"]) for record in general],
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "metrics.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
