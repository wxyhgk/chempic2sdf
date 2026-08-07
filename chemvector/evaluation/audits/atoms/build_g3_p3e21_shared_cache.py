#!/usr/bin/env python3
"""Build the shared geometry/evidence cache for all P3-E21 routes."""

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
from chemvector.evaluation import g3_p3e21_cycle as protocol
from chemvector.evaluation.audits.atoms import (
    audit_g3_p2b_train5k_multiscale_reachability as p2b,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p2d_stride1_top128_cache as p2d,
)
from chemvector.training.atoms import audit_g3_p3e0_latent_solver as p3e0
from chemvector.training.atoms import audit_g3_p3e7_base_expert_router as p3e7
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import (
    train_g3_p1x_multiscale_peak_selector_capacity as p1x,
)
from chemvector.training.atoms import train_g3_p1z_continuous_center_field_train5k as p1z
from chemvector.training.atoms import train_g3_p3e9_deepsets_count as p3e9
from chemvector.training.atoms import train_g3_p3e14_style_count_scale as p3e14
from chemvector.training.atoms import audit_g3_p3e17_p3e16_count_router as p3e17
from chemvector.training.graph.scenes.g3_structural_data import load_g3_sample_manifest
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


MANIFEST_ROOT = protocol.OUTPUT / "manifests"
OUTPUT = protocol.OUTPUT / "shared-cache"
MANIFESTS = {
    "train": ("a410bdfbaff8ca2a0c6c5d3b43cd56bb4e9fe1e86dc18ed04bd2f81118b0b2a0", 3_200, 800),
    "dev": ("a33a3a364013e365777f1538af17d75b59371666c6aa36659be0c1701201f79c", 640, 160),
    "confirm": ("5fa44672546e830176c1ee1498062a69d17b9ebd041e83d212552a7eb206acfd", 640, 160),
}
CANDIDATE_BATCH_SIZE = 24
MODEL_BATCH_SIZE = 16


def stratified_reachability(records: list[dict]) -> dict:
    groups = {}

    def add(name: str, record: dict) -> None:
        value = groups.setdefault(
            name,
            {"images": 0, "matched_atoms": 0, "target_atoms": 0, "exact_images": 0},
        )
        matched = int(record["atom_mask"].sum())
        target = int(record["full_atom_mask"].sum())
        value["images"] += 1
        value["matched_atoms"] += matched
        value["target_atoms"] += target
        value["exact_images"] += int(matched == target)

    for record in records:
        count = int(record["full_atom_mask"].sum())
        add(f"count:{protocol.count_bucket(count)}", record)
        add(f"variant:{str(record['sample_key']).rsplit('_', 1)[-1]}", record)
    return {
        name: {
            **value,
            "atom_recall": value["matched_atoms"] / value["target_atoms"],
            "exact_rate": value["exact_images"] / value["images"],
        }
        for name, value in sorted(groups.items())
    }


def load_manifests():
    records = {}
    audits = {}
    for split, (sha256, images, molecules) in MANIFESTS.items():
        records[split], audits[split] = load_g3_sample_manifest(
            MANIFEST_ROOT / f"{split}.jsonl",
            expected_sha256=sha256,
            expected_samples=images,
            expected_molecules=molecules,
            expected_variants=protocol.VARIANTS,
        )
    return records, audits


def merge_shared_record(
    candidate: dict,
    evidence: dict,
    base_output: dict,
    expert_output: dict,
    count_logits: torch.Tensor,
) -> dict:
    key = str(candidate["sample_key"])
    if key != str(evidence["sample_key"]):
        raise ValueError("P3-E21 candidate/evidence sample identity differs")
    if int(evidence["label"]) != int(candidate["full_atom_mask"].sum()):
        raise ValueError("P3-E21 evidence/geometry Count differs")
    if count_logits.shape != (129,):
        raise ValueError("P3-E21 P3-E16 Count posterior width differs")
    return {
        **candidate,
        "query_features": evidence["query"].to(torch.float16),
        "count_candidate_features": evidence["candidate"].to(torch.float16),
        "global_features": evidence["global"].to(torch.float16),
        "label": int(evidence["label"]),
        "baseline_count": int(evidence["baseline_count"]),
        "base_existence_logits": base_output["existence_logits"].to(torch.float16),
        "base_pointer_logits": base_output["pointer_logits"].to(torch.float16),
        "expert_existence_logits": expert_output["existence_logits"].to(torch.float16),
        "expert_pointer_logits": expert_output["pointer_logits"].to(torch.float16),
        "p3e16_count_logits": count_logits.to(torch.float16),
    }


@torch.no_grad()
def collect_count_logits(model, evidence: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    output = {}
    model.eval()
    for start in range(0, len(evidence), MODEL_BATCH_SIZE):
        current = evidence[start : start + MODEL_BATCH_SIZE]
        indices = list(range(len(current)))
        query, candidate, global_features, _ = p3e9.batch(current, indices, device)
        with _autocast_context(device, "bf16"):
            logits = model(query, candidate, global_features)
        for index, record in enumerate(current):
            output[str(record["sample_key"])] = logits[index].float().cpu()
    return output


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E21 shared cache output is not empty: {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("P3-E21 shared cache requires CUDA BF16")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(protocol.SEED)
    np.random.seed(protocol.SEED)
    torch.manual_seed(protocol.SEED)
    torch.cuda.manual_seed_all(protocol.SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    manifests, manifest_audits = load_manifests()
    started = time.perf_counter()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    raw, candidate_lineage = p1x._collect_scale(
        checkpoint=p2b.STRIDE1_CHECKPOINT,
        checkpoint_sha256=p2b.STRIDE1_SHA256,
        expected_model_type=p1z.SCALE_PROTOCOLS["stride1"]["model_type"],
        field_factory=ContinuousStride1AtomCenterField,
        feature_name="stride1_features",
        maximum_objects=p2d.CANDIDATES,
        domains=manifests,
        collator=collator,
        device=device,
        batch_size=CANDIDATE_BATCH_SIZE,
        accepted_split="train",
    )
    cache, candidate_audit = p2d.format_cache(raw)
    del raw

    all_records = [record for split in MANIFESTS for record in cache[split]]
    base_model = p3e0.load_model(p3e7.BASE_CHECKPOINT, device)
    expert_model = p3e0.load_model(p3e7.EXPERT_CHECKPOINT, device)
    base_outputs = p3e0.collect_outputs(
        base_model, all_records, device=device, batch_size=MODEL_BATCH_SIZE, precision="bf16"
    )
    expert_outputs = p3e0.collect_outputs(
        expert_model, all_records, device=device, batch_size=MODEL_BATCH_SIZE, precision="bf16"
    )
    del base_model, expert_model

    style_evidence, evidence_audit = p3e14.load_style_evidence()
    selected_keys = {str(record["sample_key"]) for record in all_records}
    selected_evidence = [
        record for record in style_evidence if str(record["sample_key"]) in selected_keys
    ]
    if len(selected_evidence) != len(selected_keys):
        raise RuntimeError("P3-E21 selected Count evidence is incomplete")
    evidence_by_key = {str(record["sample_key"]): record for record in selected_evidence}
    count_model = p3e14.load_count_model(p3e17.COUNT_CHECKPOINT, device)
    count_logits = collect_count_logits(count_model, selected_evidence, device)
    del count_model, style_evidence, selected_evidence

    OUTPUT.mkdir(parents=True, exist_ok=False)
    split_outputs = {}
    checks = {}
    for split, records in cache.items():
        merged = [
            merge_shared_record(
                record,
                evidence_by_key[str(record["sample_key"])],
                base_outputs[str(record["sample_key"])],
                expert_outputs[str(record["sample_key"])],
                count_logits[str(record["sample_key"])],
            )
            for record in records
        ]
        payload = {
            "schema_version": "g3-p3e21-shared-cache-0.1",
            "cycle": protocol.CYCLE_NAME,
            "split": split,
            "manifest": manifest_audits[split],
            "records": merged,
        }
        path = OUTPUT / f"{split}.pt"
        _atomic_torch_save(path, payload)
        reachability = candidate_audit["reachability"][split]
        checks[f"{split}_candidate_recall_min99"] = reachability["atom_recall"] >= 0.99
        split_outputs[split] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
            "records": len(merged),
            "candidate_reachability": reachability,
            "candidate_reachability_stratified": stratified_reachability(merged),
        }
    status = "passed" if all(checks.values()) else "failed"
    audit = {
        "schema_version": "g3-p3e21-shared-cache-audit-0.1",
        "status": status,
        "checks": checks,
        "splits": split_outputs,
        "lineage": {
            "manifests": manifest_audits,
            "candidate": candidate_lineage,
            "candidate_protocol": candidate_audit,
            "style_evidence": evidence_audit,
            "base_checkpoint": str(p3e7.BASE_CHECKPOINT.resolve()),
            "base_sha256": p3e7.BASE_SHA256,
            "expert_checkpoint": str(p3e7.EXPERT_CHECKPOINT.resolve()),
            "expert_sha256": p3e7.EXPERT_SHA256,
            "count_checkpoint": str(p3e17.COUNT_CHECKPOINT.resolve()),
            "count_sha256": p3e17.COUNT_SHA256,
            "candidate_batch_size": CANDIDATE_BATCH_SIZE,
            "model_batch_size": MODEL_BATCH_SIZE,
            "fresh_dev_read": False,
            "sealed_test_read": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
