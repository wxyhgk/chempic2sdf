#!/usr/bin/env python3
"""Compare train5k and train25k stride-1 candidate sources on Micro160."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_center_field import (
    ContinuousStride1AtomCenterField,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e22_candidate_budgets as p3e22,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e29_bond_incident_signal as p3e29,
)
from chemvector.evaluation.audits.atoms import (
    build_g3_p3e21_shared_cache as p3e21_cache,
)
from chemvector.training.atoms import (
    train_g3_p1a_canonical_atom_selector as p1a,
)
from chemvector.training.atoms import (
    train_g3_p1x_multiscale_peak_selector_capacity as p1x,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
)
from chemvector.training.graph.scenes.g3_structural_data import (
    load_g3_sample_manifest,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p3e32-train25k-candidate-source-ab"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
CACHE = OUTPUT / "train25k-top256.pt"
TRAIN25K_CHECKPOINT = Path(
    "outputs/checkpoints/chemvector-g3-p2f-stride1-center-field-train25k/"
    "step-006250.pt"
)
TRAIN25K_SHA256 = (
    "9e9d7725e9e9ffecde7bcbe9fbbb095f09204a480b18acbb014d88ef5ac83668"
)
TRAIN25K_MODEL_TYPE = "chemvector-g3-p2f-stride1-center-field-train25k"
P3E31_AUDIT = Path(
    "outputs/audits/chemvector-g3-p3e31-tail-failure-sources/audit.json"
)
P3E31_AUDIT_SHA256 = (
    "bf810cd51bfc75dc93266845c3bdbb026c559319856f93900331a4b76df9f905"
)
BUDGETS = (128, 192, 256)
MIN_RECOVERED_OLD_TAIL_IMAGES = 5
MAX_REGRESSED_OLD_REACHABLE_IMAGES = 3
BATCH_SIZE = 24
SEED = 20260832


def sample_reachability(candidate: dict, target: dict, budget: int) -> tuple[int, int]:
    expected = target["atom_anchors"][target["full_atom_mask"]].float()
    matched = p1a.maximum_cardinality_radius_match_count(
        candidate["candidate_anchors"][:budget],
        expected,
        radius_pixels=6.0,
        image_size=p3e29.IMAGE_SIZE,
    )
    return matched, len(expected)


def reachability_metrics(candidates: dict[str, dict], targets: list[dict]) -> dict:
    result = {}
    for budget in BUDGETS:
        matched = atoms = exact = 0
        by_variant = defaultdict(lambda: {"images": 0, "matched": 0, "atoms": 0, "exact": 0})
        for target in targets:
            key = str(target["sample_key"])
            current, count = sample_reachability(candidates[key], target, budget)
            variant = key.rsplit("_", 1)[-1]
            matched += current
            atoms += count
            exact += int(current == count)
            value = by_variant[variant]
            value["images"] += 1
            value["matched"] += current
            value["atoms"] += count
            value["exact"] += int(current == count)
        result[str(budget)] = {
            "images": len(targets),
            "matched_atoms": matched,
            "target_atoms": atoms,
            "atom_recall": matched / atoms,
            "exact_images": exact,
            "by_variant": {
                name: {**value, "recall": value["matched"] / value["atoms"]}
                for name, value in sorted(by_variant.items())
            },
        }
    return result


def known_count_raw_metrics(candidates: dict[str, dict], targets: list[dict]) -> dict:
    matched = atoms = exact = 0
    for target in targets:
        key = str(target["sample_key"])
        expected = target["atom_anchors"][target["full_atom_mask"]].float()
        count = len(expected)
        selected = candidates[key]["candidate_anchors"][:count]
        current = p1a.maximum_cardinality_radius_match_count(
            selected,
            expected,
            radius_pixels=6.0,
            image_size=p3e29.IMAGE_SIZE,
        )
        matched += current
        atoms += count
        exact += int(current == count)
    return {
        "images": len(targets),
        "matched_atoms": matched,
        "target_atoms": atoms,
        "atom_recall": matched / atoms,
        "exact_images": exact,
    }


def compare_sources(
    old: dict[str, dict], new: dict[str, dict], targets: list[dict]
) -> dict:
    old_tail_images = recovered_images = regressed_images = 0
    old_tail_targets = recovered_targets = 0
    new_ranks = []
    cases = []
    for target in targets:
        key = str(target["sample_key"])
        expected = target["atom_anchors"][target["full_atom_mask"]].float()
        old_compatibility = torch.cdist(
            expected, old[key]["candidate_anchors"].float()
        ) <= 6.0 / p3e29.IMAGE_SIZE
        new_compatibility = torch.cdist(
            expected, new[key]["candidate_anchors"].float()
        ) <= 6.0 / p3e29.IMAGE_SIZE
        old_exact = all(bool(value) for value in old_compatibility[:, :128].any(dim=1))
        new_exact = all(bool(value) for value in new_compatibility[:, :128].any(dim=1))
        old_tail_images += int(not old_exact)
        recovered_images += int(not old_exact and new_exact)
        regressed_images += int(old_exact and not new_exact)
        for index in range(len(expected)):
            if bool(old_compatibility[index, :128].any()):
                continue
            old_tail_targets += 1
            compatible = torch.where(new_compatibility[index])[0]
            rank = int(compatible.min()) + 1 if len(compatible) else None
            recovered_targets += int(rank is not None and rank <= 128)
            if rank is not None:
                new_ranks.append(rank)
            cases.append({
                "sample_key": key,
                "target_index": index,
                "new_best_rank": rank,
                "new_top128_recovered": rank is not None and rank <= 128,
            })
    return {
        "old_tail_images": old_tail_images,
        "recovered_old_tail_images": recovered_images,
        "regressed_old_reachable_images": regressed_images,
        "old_tail_targets": old_tail_targets,
        "recovered_old_tail_targets_top128": recovered_targets,
        "new_tail_target_rank_mean": sum(new_ranks) / len(new_ranks),
        "new_tail_target_rank_median": statistics.median(new_ranks),
        "new_tail_targets_unreachable_top256": old_tail_targets - len(new_ranks),
        "cases": cases,
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E32 output is not empty: {OUTPUT}")
    if file_sha256(P3E31_AUDIT) != P3E31_AUDIT_SHA256:
        raise ValueError("P3-E32 P3-E31 audit SHA256 differs")
    if file_sha256(TRAIN25K_CHECKPOINT) != TRAIN25K_SHA256:
        raise ValueError("P3-E32 train25k checkpoint SHA256 differs")
    if not torch.cuda.is_available():
        raise RuntimeError("P3-E32 requires CUDA BF16")
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

    targets, input_audit, manifest = p3e29._load_micro160()
    selected_keys = {str(record["sample_key"]) for record in targets}
    full_manifest, _ = load_g3_sample_manifest(
        p3e21_cache.MANIFEST_ROOT / "train.jsonl",
        expected_sha256=p3e21_cache.MANIFESTS["train"][0],
        expected_samples=p3e21_cache.MANIFESTS["train"][1],
        expected_molecules=p3e21_cache.MANIFESTS["train"][2],
        expected_variants=p3e29.cycle.VARIANTS,
    )
    selected_manifest = [
        record for record in full_manifest if record.sample_key in selected_keys
    ]
    if len(selected_manifest) != len(selected_keys):
        raise RuntimeError("P3-E32 ordered Micro160 manifest is incomplete")
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    started = time.perf_counter()
    collected, lineage = p1x._collect_scale(
        checkpoint=TRAIN25K_CHECKPOINT,
        checkpoint_sha256=TRAIN25K_SHA256,
        expected_model_type=TRAIN25K_MODEL_TYPE,
        field_factory=ContinuousStride1AtomCenterField,
        feature_name="stride1_features",
        maximum_objects=256,
        domains={"micro160": selected_manifest},
        collator=collator,
        device=device,
        batch_size=BATCH_SIZE,
        accepted_split="train",
    )
    new = collected["micro160"]
    old_payload = torch.load(p3e22.CACHE, map_location="cpu", weights_only=False)
    old_all = old_payload["records"]["train"]
    old = {str(record["sample_key"]): old_all[str(record["sample_key"])] for record in targets}
    old_metrics = reachability_metrics(old, targets)
    new_metrics = reachability_metrics(new, targets)
    comparison = compare_sources(old, new, targets)
    old_raw = known_count_raw_metrics(old, targets)
    new_raw = known_count_raw_metrics(new, targets)
    checks = {
        "recover_old_tail_images_min5": (
            comparison["recovered_old_tail_images"] >= MIN_RECOVERED_OLD_TAIL_IMAGES
        ),
        "regress_old_reachable_images_max3": (
            comparison["regressed_old_reachable_images"]
            <= MAX_REGRESSED_OLD_REACHABLE_IMAGES
        ),
        "train25k_top128_recall_improves": (
            new_metrics["128"]["atom_recall"] > old_metrics["128"]["atom_recall"]
        ),
        "train25k_top192_all160_reachable": new_metrics["192"]["exact_images"] == 160,
        "tail_case_count_33": comparison["old_tail_targets"] == 33,
        "sample_count_160": len(new) == len(targets) == 160,
    }
    decision = (
        "train25k_candidate_source_go"
        if all(checks.values())
        else "train25k_candidate_source_no_go"
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_torch_save(CACHE, {
        "schema_version": "g3-p3e32-train25k-top256-cache-0.1",
        "records": new,
        "lineage": lineage,
    })
    audit = {
        "schema_version": "g3-p3e32-train25k-candidate-source-ab-audit-0.1",
        "decision": decision,
        "checks": checks,
        "old_train5k": {
            "checkpoint": str(p3e22.p2b.STRIDE1_CHECKPOINT.resolve()),
            "checkpoint_sha256": p3e22.p2b.STRIDE1_SHA256,
            "reachability": old_metrics,
            "known_count_raw": old_raw,
        },
        "new_train25k": {
            "checkpoint": str(TRAIN25K_CHECKPOINT.resolve()),
            "checkpoint_sha256": TRAIN25K_SHA256,
            "reachability": new_metrics,
            "known_count_raw": new_raw,
        },
        "comparison": comparison,
        "protocol": {
            "budgets": list(BUDGETS),
            "minimum_recovered_old_tail_images": MIN_RECOVERED_OLD_TAIL_IMAGES,
            "maximum_regressed_old_reachable_images": MAX_REGRESSED_OLD_REACHABLE_IMAGES,
            "candidate_match_radius_px": 6.0,
            "candidate_count": 256,
            "batch_size": BATCH_SIZE,
        },
        "cache": {
            "path": str(CACHE.resolve()),
            "sha256": file_sha256(CACHE),
        },
        "input": input_audit,
        "lineage": {
            "collection": lineage,
            "p3e31_audit": str(P3E31_AUDIT.resolve()),
            "p3e31_audit_sha256": P3E31_AUDIT_SHA256,
        },
        "data_access": {
            "micro160_png_read": True,
            "training_performed": False,
            "dev_read": False,
            "old_confirm_read": False,
            "fresh_dev_read": False,
            "uspto_read": False,
            "mr_tadf_read": False,
            "sealed_test_read": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
