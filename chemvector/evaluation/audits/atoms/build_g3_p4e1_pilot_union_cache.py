#!/usr/bin/env python3
"""Build source-aware top256 union caches for P4 pilot train/Micro/dev."""

from __future__ import annotations

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
from chemvector.evaluation import g3_p3e21_cycle as p3_cycle
from chemvector.evaluation import g3_p4_source_union_cycle as cycle
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e22_candidate_budgets as p3e22,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e29_bond_incident_signal as p3e29,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e32_train25k_candidate_source_ab as p3e32,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e33_candidate_source_union as p3e33,
)
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
    train_g3_p1z_continuous_center_field_train5k as p1z,
)
from chemvector.training.graph.scenes.g3_structural_data import (
    load_g3_sample_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p4e1-pilot-union-cache"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
P4E0_ROOT = Path(
    "outputs/audits/chemvector-g3-p4e0-source-union-cycle-manifests"
)
P4E0_AUDIT_SHA256 = (
    "50d46024cd07d8313590de85903c5795e9919a7f7dfa5cbbdc7beeb5e4f565e3"
)
PILOT_MANIFEST_SHA256 = (
    "22a03e2b2e37b41b7b298a1fec324fbe383e0f09c49e173a39c8d90347636daa"
)
UNION_STRATEGY = "train25k160_then_train5k"
CANDIDATES = 256
FEATURE_DIM = 72
BATCH_SIZE = 24
SEED = 20260841


def source_aware_union_record(old: dict, new: dict, sample_key: str) -> dict:
    merged = p3e33.merge_candidate_records(old, new, UNION_STRATEGY)
    old_by_id = {int(value): index for index, value in enumerate(old["source_ids"])}
    new_by_id = {int(value): index for index, value in enumerate(new["source_ids"])}
    rows = []
    for slot, source_id_value in enumerate(merged["source_ids"]):
        source_id = int(source_id_value)
        old_index = old_by_id.get(source_id)
        new_index = new_by_id.get(source_id)
        old_feature = (
            old["candidate_features"][old_index].float()
            if old_index is not None
            else torch.zeros(32)
        )
        new_feature = (
            new["candidate_features"][new_index].float()
            if new_index is not None
            else torch.zeros(32)
        )
        metadata = torch.tensor(
            (
                float(old["normalized_scores"][old_index]) if old_index is not None else 0.0,
                float(new["normalized_scores"][new_index]) if new_index is not None else 0.0,
                old_index / 255.0 if old_index is not None else 1.0,
                new_index / 255.0 if new_index is not None else 1.0,
                float(old_index is not None),
                float(new_index is not None),
                float(merged["source_codes"][slot] == 0),
                float(merged["source_codes"][slot] == 1),
            )
        )
        rows.append(torch.cat((old_feature, new_feature, metadata)))
    features = torch.stack(rows)
    if features.shape != (CANDIDATES, FEATURE_DIM):
        raise RuntimeError("P4-E1 source-aware feature shape differs")
    if not torch.equal(old["atom_mask"], new["atom_mask"]) or not torch.allclose(
        old["atom_anchors"], new["atom_anchors"], atol=0.0, rtol=0.0
    ):
        raise ValueError("P4-E1 dual-source GT tensors differ")
    return {
        "sample_key": sample_key,
        "candidate_ids": merged["source_ids"].long(),
        "candidate_anchors": merged["candidate_anchors"].float(),
        "candidate_features": features.to(torch.float16),
        "source_codes": merged["source_codes"],
        "atom_anchors": old["atom_anchors"].float(),
        "atom_mask": old["atom_mask"].bool(),
        "label": int(old["atom_mask"].sum()),
    }


def reachability(records: list[dict]) -> dict:
    matched = targets = exact = unique = 0
    by_variant = {}
    for record in records:
        expected = record["atom_anchors"][record["atom_mask"]]
        current = p1a.maximum_cardinality_radius_match_count(
            record["candidate_anchors"],
            expected,
            radius_pixels=6.0,
            image_size=768.0,
        )
        matched += current
        targets += len(expected)
        exact += int(current == len(expected))
        unique += int(len(record["candidate_ids"].unique()) == CANDIDATES)
        variant = str(record["sample_key"]).rsplit("_", 1)[-1]
        value = by_variant.setdefault(
            variant, {"images": 0, "matched": 0, "targets": 0, "exact": 0}
        )
        value["images"] += 1
        value["matched"] += current
        value["targets"] += len(expected)
        value["exact"] += int(current == len(expected))
    return {
        "images": len(records),
        "matched_atoms": matched,
        "target_atoms": targets,
        "atom_recall": matched / targets,
        "exact_images": exact,
        "unique_candidate_id_images": unique,
        "by_variant": {
            name: {**value, "recall": value["matched"] / value["targets"]}
            for name, value in sorted(by_variant.items())
        },
    }


def _load_manifests():
    pilot, pilot_audit = load_g3_sample_manifest(
        P4E0_ROOT / "pilot_train.jsonl",
        expected_sha256=PILOT_MANIFEST_SHA256,
        expected_samples=20_000,
        expected_molecules=5_000,
        expected_variants=cycle.VARIANTS,
    )
    dev, dev_audit = load_g3_sample_manifest(
        p3e21.MANIFEST_ROOT / "dev.jsonl",
        expected_sha256=p3e21.MANIFESTS["dev"][0],
        expected_samples=p3e21.MANIFESTS["dev"][1],
        expected_molecules=p3e21.MANIFESTS["dev"][2],
        expected_variants=p3_cycle.VARIANTS,
    )
    return {"pilot_train": pilot, "dev": dev}, {
        "pilot_train": pilot_audit,
        "dev": dev_audit,
    }


def _collect_source(checkpoint, sha256, model_type, domains, collator, device):
    records, lineage = p1x._collect_scale(
        checkpoint=checkpoint,
        checkpoint_sha256=sha256,
        expected_model_type=model_type,
        field_factory=ContinuousStride1AtomCenterField,
        feature_name="stride1_features",
        maximum_objects=CANDIDATES,
        domains=domains,
        collator=collator,
        device=device,
        batch_size=BATCH_SIZE,
        accepted_split="train",
    )
    return records, lineage


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P4-E1 output is not empty: {OUTPUT}")
    if file_sha256(P4E0_ROOT / "audit.json") != P4E0_AUDIT_SHA256:
        raise ValueError("P4-E1 P4-E0 audit SHA256 differs")
    if not torch.cuda.is_available():
        raise RuntimeError("P4-E1 requires CUDA BF16")
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

    domains, manifest_audit = _load_manifests()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    started = time.perf_counter()
    old, old_lineage = _collect_source(
        p3e22.p2b.STRIDE1_CHECKPOINT,
        p3e22.p2b.STRIDE1_SHA256,
        p1z.SCALE_PROTOCOLS["stride1"]["model_type"],
        domains,
        collator,
        device,
    )
    new, new_lineage = _collect_source(
        p3e32.TRAIN25K_CHECKPOINT,
        p3e32.TRAIN25K_SHA256,
        p3e32.TRAIN25K_MODEL_TYPE,
        domains,
        collator,
        device,
    )

    micro_targets, _, _ = p3e29._load_micro160()
    old_micro_all = torch.load(p3e22.CACHE, map_location="cpu", weights_only=False)[
        "records"
    ]["train"]
    new_micro_all = torch.load(p3e32.CACHE, map_location="cpu", weights_only=False)[
        "records"
    ]
    merged = {}
    for split in domains:
        keys = [record.sample_key for record in domains[split]]
        merged[split] = [
            source_aware_union_record(old[split][key], new[split][key], key)
            for key in keys
        ]
    merged["micro160"] = [
        source_aware_union_record(
            old_micro_all[str(record["sample_key"])],
            new_micro_all[str(record["sample_key"])],
            str(record["sample_key"]),
        )
        for record in micro_targets
    ]
    split_metrics = {split: reachability(records) for split, records in merged.items()}
    checks = {
        "pilot_recall_min999": split_metrics["pilot_train"]["atom_recall"] >= 0.999,
        "dev_recall_min999": split_metrics["dev"]["atom_recall"] >= 0.999,
        "micro_exact_160": split_metrics["micro160"]["exact_images"] == 160,
        "all_candidate_ids_unique": all(
            value["unique_candidate_id_images"] == value["images"]
            for value in split_metrics.values()
        ),
        "feature_dim_72": all(
            record["candidate_features"].shape == (CANDIDATES, FEATURE_DIM)
            for records in merged.values()
            for record in records
        ),
        "confirm_v2_not_read": True,
    }
    decision = "pilot_union_cache_go" if all(checks.values()) else "pilot_union_cache_no_go"
    OUTPUT.mkdir(parents=True, exist_ok=False)
    outputs = {}
    digest = hashlib.sha256()
    for split, records in merged.items():
        path = OUTPUT / f"{split}.pt"
        _atomic_torch_save(path, {
            "schema_version": "g3-p4e1-source-aware-union-cache-0.1",
            "split": split,
            "strategy": UNION_STRATEGY,
            "feature_dim": FEATURE_DIM,
            "records": records,
        })
        for record in records:
            digest.update(str(record["sample_key"]).encode())
            digest.update(record["candidate_ids"].numpy().tobytes())
            digest.update(record["candidate_features"].numpy().tobytes())
        outputs[split] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "metrics": split_metrics[split],
        }
    audit = {
        "schema_version": "g3-p4e1-pilot-union-cache-audit-0.1",
        "decision": decision,
        "checks": checks,
        "protocol": {
            "strategy": UNION_STRATEGY,
            "candidate_count": CANDIDATES,
            "feature_dim": FEATURE_DIM,
            "feature_layout": (
                "old32,new32,old/new normalized score,old/new rank,"
                "old/new availability,selected old/new one-hot"
            ),
            "deduplication": "exact shared stride1 source_id",
            "pixel_distance_deduplication": False,
            "batch_size": BATCH_SIZE,
        },
        "outputs": outputs,
        "combined_tensor_sha256": digest.hexdigest(),
        "manifests": manifest_audit,
        "lineage": {"train5k": old_lineage, "train25k": new_lineage},
        "data_access": {
            "pilot_train_png_read": True,
            "dev_png_read": True,
            "micro_cache_read": True,
            "confirm_v2_read": False,
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
