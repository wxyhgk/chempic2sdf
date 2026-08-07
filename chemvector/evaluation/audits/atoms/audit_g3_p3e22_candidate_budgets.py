#!/usr/bin/env python3
"""Audit top128/192/256 candidate budgets on the frozen P3-E21 splits."""

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
from chemvector.evaluation import g3_p3e21_cycle as cycle
from chemvector.evaluation.audits.atoms import (
    audit_g3_p2b_train5k_multiscale_reachability as p2b,
)
from chemvector.evaluation.audits.atoms import (
    build_g3_p3e21_shared_cache as p3e21_cache,
)
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import (
    train_g3_p1x_multiscale_peak_selector_capacity as p1x,
)
from chemvector.training.atoms import train_g3_p1z_continuous_center_field_train5k as p1z
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p3e22-candidate-budget-audit"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
CACHE = OUTPUT / "top256-candidate-cache.pt"
BUDGETS = (128, 192, 256)
MAXIMUM_OBJECTS = max(BUDGETS)
BATCH_SIZE = 24


def budget_reachability(records: dict[str, dict[str, dict]]) -> dict:
    metrics = {}
    for split, values in records.items():
        metrics[split] = {}
        for budget in BUDGETS:
            matched_atoms = target_atoms = exact_images = 0
            for record in values.values():
                targets = record["atom_anchors"][record["atom_mask"]]
                matched = p1a.maximum_cardinality_radius_match_count(
                    record["candidate_anchors"][:budget],
                    targets,
                    radius_pixels=6.0,
                    image_size=768.0,
                )
                matched_atoms += matched
                target_atoms += len(targets)
                exact_images += int(matched == len(targets))
            metrics[split][str(budget)] = {
                "images": len(values),
                "matched_atoms": matched_atoms,
                "target_atoms": target_atoms,
                "atom_recall": matched_atoms / target_atoms,
                "exact_images": exact_images,
                "exact_rate": exact_images / len(values),
            }
    return metrics


def select_budget(metrics: dict) -> tuple[int | None, dict[str, bool]]:
    checks = {
        str(budget): all(
            metrics[split][str(budget)]["atom_recall"] >= 0.99
            for split in p3e21_cache.MANIFESTS
        )
        for budget in BUDGETS
    }
    selected = next((budget for budget in BUDGETS if checks[str(budget)]), None)
    return selected, checks


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E22 output is not empty: {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("P3-E22 requires CUDA BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(cycle.SEED)
    np.random.seed(cycle.SEED)
    torch.manual_seed(cycle.SEED)
    torch.cuda.manual_seed_all(cycle.SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    manifests, manifest_audits = p3e21_cache.load_manifests()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    started = time.perf_counter()
    records, lineage = p1x._collect_scale(
        checkpoint=p2b.STRIDE1_CHECKPOINT,
        checkpoint_sha256=p2b.STRIDE1_SHA256,
        expected_model_type=p1z.SCALE_PROTOCOLS["stride1"]["model_type"],
        field_factory=ContinuousStride1AtomCenterField,
        feature_name="stride1_features",
        maximum_objects=MAXIMUM_OBJECTS,
        domains=manifests,
        collator=collator,
        device=device,
        batch_size=BATCH_SIZE,
        accepted_split="train",
    )
    metrics = budget_reachability(records)
    selected, checks = select_budget(metrics)
    decision = "candidate_budget_go" if selected is not None else "candidate_budget_no_go"
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_torch_save(CACHE, {
        "schema_version": "g3-p3e22-top256-candidate-cache-0.1",
        "records": records,
        "manifests": manifest_audits,
        "lineage": lineage,
    })
    audit = {
        "schema_version": "g3-p3e22-candidate-budget-audit-0.1",
        "decision": decision,
        "selected_minimum_budget": selected,
        "checks": checks,
        "metrics": metrics,
        "budgets": list(BUDGETS),
        "cache": str(CACHE.resolve()),
        "cache_sha256": file_sha256(CACHE),
        "lineage": lineage,
        "fresh_dev_read": False,
        "sealed_test_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "metrics.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
