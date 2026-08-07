#!/usr/bin/env python3
"""Measure whether P4-E3 missed dev GT centers survive below top256."""

from __future__ import annotations

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
from chemvector.evaluation import g3_p3e21_cycle as p3_cycle
from chemvector.evaluation import g3_p4_source_union_cycle as cycle
from chemvector.evaluation.audits.atoms import (
    build_g3_p3e21_shared_cache as p3e21,
)
from chemvector.training.atoms import (
    train_g3_p1a_canonical_atom_selector as p1a,
)
from chemvector.training.atoms import (
    train_g3_p1x_multiscale_peak_selector_capacity as p1x,
)
from chemvector.training.atoms import (
    train_g3_p4e3_global_peak_rank_pilot20k as p4e3,
)
from chemvector.training.graph.scenes.g3_structural_data import (
    load_g3_sample_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p4e5-peak-budget-dev1024"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
P4E4_AUDIT = Path(
    "outputs/audits/chemvector-g3-p4e4-peak-rank-union-audit/audit.json"
)
P4E4_AUDIT_SHA256 = (
    "9746bc2c49a0822a7c735b2cf4351b4439ad85528f9b50d45e0b0eab4005df51"
)
P4E3_CHECKPOINT = p4e3.OUTPUT / f"step-{p4e3.STEPS:06d}.pt"
P4E3_CHECKPOINT_SHA256 = (
    "1c358d0c936c687fa0e40ea5e1d9364dd98a663bb762d294db76b43d5f009593"
)
BUDGETS = (128, 192, 256, 384, 512, 768, 1024)
MAXIMUM_CANDIDATES = 1024
MIN_RECALL = 0.999
BATCH_SIZE = 24
SEED = 20260845


def _empty_totals() -> dict:
    return {"images": 0, "matched": 0, "targets": 0, "exact": 0}


def _finalize(value: dict) -> dict:
    return {
        **value,
        "recall": value["matched"] / value["targets"],
        "exact_rate": value["exact"] / value["images"],
    }


def reachability(records: dict[str, dict], budget: int) -> dict:
    if budget not in BUDGETS:
        raise ValueError("P4-E5 budget is not pre-registered")
    totals = _empty_totals()
    variants = {}
    count_buckets = {}
    for sample_key, record in records.items():
        expected = record["atom_anchors"][record["atom_mask"]]
        matched = p1a.maximum_cardinality_radius_match_count(
            record["candidate_anchors"][:budget],
            expected,
            radius_pixels=6.0,
            image_size=768.0,
        )
        exact = int(matched == len(expected))
        groups = (
            totals,
            variants.setdefault(sample_key.rsplit("_", 1)[-1], _empty_totals()),
            count_buckets.setdefault(cycle.count_bucket(len(expected)), _empty_totals()),
        )
        for current in groups:
            current["images"] += 1
            current["matched"] += matched
            current["targets"] += len(expected)
            current["exact"] += exact
    return {
        **_finalize(totals),
        "by_variant": {
            key: _finalize(value) for key, value in sorted(variants.items())
        },
        "by_count_bucket": {
            key: _finalize(value) for key, value in sorted(count_buckets.items())
        },
    }


def decision_gate(metrics: dict[str, dict]) -> tuple[str, dict[str, bool], int | None]:
    passing = [budget for budget in BUDGETS if metrics[str(budget)]["recall"] >= MIN_RECALL]
    checks = {
        "top128_replays_p4e3_final": (
            metrics["128"]["matched"] == 32_195
            and metrics["128"]["exact"] == 550
        ),
        "top1024_recall_min999": metrics["1024"]["recall"] >= MIN_RECALL,
    }
    decision = (
        "tail_ranking_capacity_go"
        if all(checks.values())
        else "regional_maxima_representation_no_go"
    )
    return decision, checks, min(passing) if passing else None


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P4-E5 output is not empty: {OUTPUT}")
    for path, expected in (
        (P4E4_AUDIT, P4E4_AUDIT_SHA256),
        (P4E3_CHECKPOINT, P4E3_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P4-E5 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P4-E5 requires CUDA BF16")
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

    dev, manifest_audit = load_g3_sample_manifest(
        p3e21.MANIFEST_ROOT / "dev.jsonl",
        expected_sha256=p3e21.MANIFESTS["dev"][0],
        expected_samples=p3e21.MANIFESTS["dev"][1],
        expected_molecules=p3e21.MANIFESTS["dev"][2],
        expected_variants=p3_cycle.VARIANTS,
    )
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    started = time.perf_counter()
    collected, lineage = p1x._collect_scale(
        checkpoint=P4E3_CHECKPOINT,
        checkpoint_sha256=P4E3_CHECKPOINT_SHA256,
        expected_model_type=p4e3.MODEL_TYPE,
        field_factory=ContinuousStride1AtomCenterField,
        feature_name="stride1_features",
        maximum_objects=MAXIMUM_CANDIDATES,
        domains={"dev": dev},
        collator=collator,
        device=device,
        batch_size=BATCH_SIZE,
        accepted_split="train",
    )
    records = collected["dev"]
    metrics = {str(budget): reachability(records, budget) for budget in BUDGETS}
    decision, checks, minimum_passing_budget = decision_gate(metrics)
    peak_counts = [record["regional_maxima_count"] for record in records.values()]
    audit = {
        "schema_version": "g3-p4e5-peak-budget-dev1024-audit-0.1",
        "decision": decision,
        "checks": checks,
        "minimum_passing_budget": minimum_passing_budget,
        "protocol": {
            "budgets": list(BUDGETS),
            "maximum_candidates": MAXIMUM_CANDIDATES,
            "minimum_recall": MIN_RECALL,
            "deduplication": "complete regional-max plateau",
            "pixel_distance_deduplication": False,
            "training_performed": False,
        },
        "metrics": metrics,
        "regional_maxima": {
            "mean": float(np.mean(peak_counts)),
            "p90": float(np.percentile(peak_counts, 90)),
            "maximum": int(max(peak_counts)),
        },
        "manifest": manifest_audit,
        "lineage": lineage,
        "data_access": {
            "dev_png_read": True,
            "pilot_train_read": False,
            "confirm_v2_read": False,
            "fresh_dev_read": False,
            "uspto_read": False,
            "mr_tadf_read": False,
            "sealed_test_read": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
