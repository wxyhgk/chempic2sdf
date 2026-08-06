#!/usr/bin/env python3
"""Audit the frozen P3-E17 stack on all held-out-font fresh_dev images."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    ContinuousStride1AtomCenterField,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p2b_train5k_multiscale_reachability as p2b,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p2d_stride1_top128_cache as p2d,
)
from chemvector.evaluation.audits.atoms import (
    build_g3_p3e18_freshdev_manifest as manifest_protocol,
)
from chemvector.training.atoms import audit_g3_p3e0_latent_solver as p3e0
from chemvector.training.atoms import audit_g3_p3e7_base_expert_router as p3e7
from chemvector.training.atoms import audit_g3_p3e17_p3e16_count_router as p3e17
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import (
    train_g3_p1x_multiscale_peak_selector_capacity as p1x,
)
from chemvector.training.atoms import train_g3_p1z_continuous_center_field_train5k as p1z
from chemvector.training.atoms import train_g3_p3e14_style_count_scale as p3e14
from chemvector.training.atoms import train_g3_p3e16_style_count_epoch6 as p3e16
from chemvector.training.graph.scenes.g3_structural_data import load_g3_sample_manifest
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p3e18-freshdev8k-heldout-font-audit"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
CANDIDATE_CACHE = OUTPUT / "candidate-cache.pt"
MANIFEST_SHA256 = "aa6d0254c5b0d8911972b8377a0628c2e0e3fbc72b95af95c9d8807591f4ab0d"
SEED = 20260713
CANDIDATE_BATCH_SIZE = 24
MODEL_BATCH_SIZE = 16
GATES = {
    "candidate_atom_recall_min": 0.99,
    "count_accuracy_min": 0.40,
    "count_mae_max": 1.50,
    "known_router_exact_rate_min": 0.30,
    "predicted_router_exact_rate_min": 0.20,
}


def load_manifest():
    return load_g3_sample_manifest(
        manifest_protocol.MANIFEST,
        expected_sha256=MANIFEST_SHA256,
        expected_samples=manifest_protocol.EXPECTED_IMAGES,
        expected_molecules=manifest_protocol.EXPECTED_MOLECULES,
        expected_variants=manifest_protocol.EXPECTED_VARIANTS,
    )


def subset_summary(rows: list[dict], policy: dict) -> dict:
    if not rows:
        return {
            "images": 0,
            "count": {},
            "known": {},
            "predicted": {},
        }
    known_rows = [row["known"] for row in rows]
    predicted_rows = [row["predicted"] for row in rows]
    known_router = p3e7.apply_policy(known_rows, policy)
    predicted_router = p3e7.apply_policy(predicted_rows, policy)
    known_router.pop("routed_sample_keys", None)
    predicted_router.pop("routed_sample_keys", None)
    return {
        "images": len(rows),
        "count": p3e17.count_summary(predicted_rows),
        "known": {
            "models": p3e7.summarize(known_rows),
            "router": known_router,
        },
        "predicted": {
            "models": p3e7.summarize(predicted_rows),
            "router": predicted_router,
        },
    }


def stratified_metrics(
    known_rows: list[dict], predicted_rows: list[dict], policy: dict
) -> dict:
    if [row["sample_key"] for row in known_rows] != [
        row["sample_key"] for row in predicted_rows
    ]:
        raise ValueError("P3-E18 known/predicted row identity differs")
    pairs = [
        {"known": known, "predicted": predicted}
        for known, predicted in zip(known_rows, predicted_rows)
    ]
    variants = {
        variant: subset_summary(
            [row for row in pairs if row["known"]["variant"] == variant], policy
        )
        for variant in p3e16.VARIANTS
    }
    count_buckets = {}
    for name, lower, upper in p3e16.COUNT_BUCKETS:
        count_buckets[name] = subset_summary(
            [
                row
                for row in pairs
                if lower <= row["known"]["atom_count"] <= upper
            ],
            policy,
        )
    return {
        "overall": subset_summary(pairs, policy),
        "by_variant": variants,
        "by_atom_count": count_buckets,
    }


def gate(metrics: dict, candidate_audit: dict) -> tuple[str, dict[str, bool]]:
    overall = metrics["overall"]
    images = overall["images"]
    checks = {
        "candidate_atom_recall_min99": (
            candidate_audit["atom_recall"] >= GATES["candidate_atom_recall_min"]
        ),
        "count_accuracy_min40": (
            overall["count"]["count_accuracy"] >= GATES["count_accuracy_min"]
        ),
        "count_mae_max1_5": (
            overall["count"]["count_mae"] <= GATES["count_mae_max"]
        ),
        "known_router_exact_rate_min30": (
            overall["known"]["router"]["exact"] / images
            >= GATES["known_router_exact_rate_min"]
        ),
        "predicted_router_exact_rate_min20": (
            overall["predicted"]["router"]["exact"] / images
            >= GATES["predicted_router_exact_rate_min"]
        ),
    }
    return (
        "freshdev_heldout_font_go" if all(checks.values()) else "freshdev_heldout_font_no_go",
        checks,
    )


def candidate_stratified_metrics(records: list[dict]) -> dict:
    def summarize(current: list[dict]) -> dict:
        if not current:
            return {
                "images": 0,
                "matched_atoms": 0,
                "target_atoms": 0,
                "atom_recall": None,
                "exact_images": 0,
                "exact_rate": None,
            }
        matched = sum(int(record["atom_mask"].sum()) for record in current)
        targets = sum(int(record["full_atom_mask"].sum()) for record in current)
        exact = sum(
            int(record["atom_mask"].sum()) == int(record["full_atom_mask"].sum())
            for record in current
        )
        return {
            "images": len(current),
            "matched_atoms": matched,
            "target_atoms": targets,
            "atom_recall": matched / targets,
            "exact_images": exact,
            "exact_rate": exact / len(current),
        }

    return {
        "overall": summarize(records),
        "by_variant": {
            variant: summarize(
                [
                    record
                    for record in records
                    if p2h_variant(str(record["sample_key"])) == variant
                ]
            )
            for variant in p3e16.VARIANTS
        },
        "by_atom_count": {
            name: summarize(
                [
                    record
                    for record in records
                    if lower <= int(record["full_atom_mask"].sum()) <= upper
                ]
            )
            for name, lower, upper in p3e16.COUNT_BUCKETS
        },
    }


def p2h_variant(sample_key: str) -> str:
    for variant in p3e16.VARIANTS:
        if sample_key.endswith(f"_{variant}"):
            return variant
    raise ValueError(f"P3-E18 sample key has unknown variant: {sample_key}")


def compact_decomposition(value: dict) -> dict:
    compact = dict(value)
    for name in ("known_policy", "predicted_policy"):
        if name in compact:
            compact[name] = dict(compact[name])
            compact[name].pop("routed_sample_keys", None)
    return compact


def candidate_cache_is_valid(payload: dict, manifest_audit: dict) -> bool:
    records = payload.get("records")
    audit = payload.get("audit", {})
    return bool(
        payload.get("schema_version") == "g3-p3e18-candidate-cache-0.1"
        and payload.get("manifest_sha256") == MANIFEST_SHA256
        and isinstance(records, list)
        and len(records) == manifest_protocol.EXPECTED_IMAGES
        and audit.get("candidate_count") == p2d.CANDIDATES
        and payload.get("sample_keys_sha256") == manifest_audit["sample_keys_sha256"]
    )


def build_or_load_candidates(manifest, manifest_audit, device) -> tuple[list[dict], dict]:
    if CANDIDATE_CACHE.exists():
        payload = torch.load(CANDIDATE_CACHE, map_location="cpu", weights_only=False)
        if not candidate_cache_is_valid(payload, manifest_audit):
            raise ValueError("P3-E18 existing candidate cache is invalid")
        return payload["records"], {**payload["audit"], "resumed": True}

    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    records, lineage = p1x._collect_scale(
        checkpoint=p2b.STRIDE1_CHECKPOINT,
        checkpoint_sha256=p2b.STRIDE1_SHA256,
        expected_model_type=p1z.SCALE_PROTOCOLS["stride1"]["model_type"],
        field_factory=ContinuousStride1AtomCenterField,
        feature_name="stride1_features",
        maximum_objects=p2d.CANDIDATES,
        domains={"fresh_dev": manifest},
        collator=collator,
        device=device,
        batch_size=CANDIDATE_BATCH_SIZE,
        accepted_split=manifest_protocol.EXPECTED_SPLIT,
    )
    cache, formatted = p2d.format_cache(records)
    current = cache["fresh_dev"]
    reachability = formatted["reachability"]["fresh_dev"]
    audit = {
        **formatted,
        "fresh_dev": reachability,
        "lineage": lineage,
        "resumed": False,
    }
    payload = {
        "schema_version": "g3-p3e18-candidate-cache-0.1",
        "manifest_sha256": MANIFEST_SHA256,
        "sample_keys_sha256": manifest_audit["sample_keys_sha256"],
        "records": current,
        "audit": audit,
    }
    _atomic_torch_save(CANDIDATE_CACHE, payload)
    return current, audit


def main() -> None:
    metrics_path = OUTPUT / "metrics.json"
    if metrics_path.exists():
        raise RuntimeError(f"P3-E18 metrics already exist: {metrics_path}")
    if file_sha256(manifest_protocol.MANIFEST) != MANIFEST_SHA256:
        raise ValueError("P3-E18 manifest SHA256 differs")
    for path, expected in (
        (p2b.STRIDE1_CHECKPOINT, p2b.STRIDE1_SHA256),
        (p3e7.BASE_CHECKPOINT, p3e7.BASE_SHA256),
        (p3e7.EXPERT_CHECKPOINT, p3e7.EXPERT_SHA256),
        (p3e17.COUNT_CHECKPOINT, p3e17.COUNT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P3-E18 checkpoint SHA differs: {path}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("P3-E18 requires CUDA BF16")
    device = torch.device("cuda")
    manifest, manifest_audit = load_manifest()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records, candidate_audit = build_or_load_candidates(
        manifest, manifest_audit, device
    )

    base_model = p3e0.load_model(p3e7.BASE_CHECKPOINT, device)
    expert_model = p3e0.load_model(p3e7.EXPERT_CHECKPOINT, device)
    count_model = p3e14.load_count_model(p3e17.COUNT_CHECKPOINT, device)
    base_outputs, expert_outputs, counts = p3e17.collect_outputs_and_counts(
        base_model,
        expert_model,
        count_model,
        records,
        device=device,
        batch_size=MODEL_BATCH_SIZE,
    )
    known_rows = p3e7.collect_rows(records, base_outputs, expert_outputs)
    predicted_rows = p3e17.collect_rows_with_counts(
        records, base_outputs, expert_outputs, counts
    )
    metrics = stratified_metrics(known_rows, predicted_rows, p3e17.POLICY)
    decomposition = compact_decomposition(
        p3e17.decompose(known_rows, predicted_rows, p3e17.POLICY)
    )
    candidate_metrics = candidate_stratified_metrics(records)
    decision, checks = gate(metrics, candidate_audit["fresh_dev"])

    count_digest = hashlib.sha256()
    for key in sorted(counts):
        count_digest.update(f"{key}:{counts[key]}\n".encode())
    payload = {
        "schema_version": "g3-p3e18-freshdev8k-heldout-font-audit-0.1",
        "audit_name": AUDIT_NAME,
        "decision": decision,
        "checks": checks,
        "preregistered_gates": GATES,
        "candidate_reachability": candidate_audit["fresh_dev"],
        "candidate_metrics": candidate_metrics,
        "metrics": metrics,
        "decomposition": decomposition,
        "lineage": {
            "manifest": manifest_audit,
            "candidate_cache": str(CANDIDATE_CACHE.resolve()),
            "candidate_cache_file_sha256": file_sha256(CANDIDATE_CACHE),
            "candidate_protocol": candidate_audit,
            "candidate_checkpoint": str(p2b.STRIDE1_CHECKPOINT.resolve()),
            "candidate_checkpoint_sha256": p2b.STRIDE1_SHA256,
            "base_checkpoint": str(p3e7.BASE_CHECKPOINT.resolve()),
            "base_sha256": p3e7.BASE_SHA256,
            "expert_checkpoint": str(p3e7.EXPERT_CHECKPOINT.resolve()),
            "expert_sha256": p3e7.EXPERT_SHA256,
            "count_checkpoint": str(p3e17.COUNT_CHECKPOINT.resolve()),
            "count_sha256": p3e17.COUNT_SHA256,
            "router_policy": p3e17.POLICY,
            "candidate_batch_size": CANDIDATE_BATCH_SIZE,
            "model_batch_size": MODEL_BATCH_SIZE,
            "count_predictions_sha256": count_digest.hexdigest(),
            "training_or_threshold_selection": False,
            "sealed_test_read": False,
        },
        "sample_distribution": {
            "variants": dict(Counter(row["variant"] for row in known_rows)),
            "atom_count_min": min(row["atom_count"] for row in known_rows),
            "atom_count_max": max(row["atom_count"] for row in known_rows),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(metrics_path, payload)
    print(
        json.dumps(
            {
                "decision": decision,
                "checks": checks,
                "candidate_reachability": candidate_audit["fresh_dev"],
                "overall": metrics["overall"],
                "decomposition": decomposition,
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
