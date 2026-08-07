#!/usr/bin/env python3
"""Rebuild frozen G3 route evidence with 128 queries over top192 candidates."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_atom_query_set import (
    G3AtomQuerySet,
    G3AtomQuerySetConfig,
    G3PointerUtilityExistenceModel,
)
from chemvector.components.atoms.objects.g3_deepsets_count import (
    g3_deepsets_count_features,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p2d_stride1_top128_cache as p2d,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e22_candidate_budgets as p3e22,
)
from chemvector.training.atoms import audit_g3_p3e7_base_expert_router as p3e7
from chemvector.training.atoms import audit_g3_p3e17_p3e16_count_router as p3e17
from chemvector.training.atoms import train_g3_p3e14_style_count_scale as p3e14
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p3e23-top192-cycle-recovery"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
CANDIDATES = 192
QUERIES = 128
CLASSES = 129
BATCH_SIZE = 16
SEED = 20260823
TOP256_CACHE = p3e22.OUTPUT / "top256-candidate-cache.pt"
TOP256_CACHE_SHA256 = "c796da5a33cfc954e3f4f25d9da9d70e70296a9b76d32e0736fd6d5cd2655720"
P3E22_METRICS = p3e22.OUTPUT / "metrics.json"
P3E22_METRICS_SHA256 = "0d68899854c1097011966c8561f6fbb672e0f033c6d1bce91f5a8f24a3e03398"
OLD_CACHE_ROOT = Path(
    "outputs/audits/chemvector-g3-p3e21-five-route-sdf-cycle/shared-cache"
)
OLD_CACHE_SHA256 = {
    "train": "df28f8ec1b48cd725612ad5a776f0c19e655079b3c8201456a0088d8f8039608",
    "dev": "e5e62cac56c904eef841507fc5001d8f200c6fc3eabb9c252ef721815615dae1",
    "confirm": "3e0aa09bd2a56307f47464b8d19e3ee8df75e257a625e6f713d0a386998997ce",
}


def tensor_digest(values: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        current = value.detach().contiguous().cpu()
        digest.update(str(current.dtype).encode())
        digest.update(str(tuple(current.shape)).encode())
        digest.update(current.numpy().tobytes())
    return digest.hexdigest()


def format_top192_record(sample_key: str, source: dict) -> dict:
    full_mask = source["atom_mask"].clone()
    candidates = source["candidate_anchors"][:CANDIDATES].clone()
    representable = p2d.representable_target_mask(
        candidates, source["atom_anchors"], full_mask
    )
    candidate_ids = source["source_ids"][:CANDIDATES].clone()
    if len(torch.unique(candidate_ids)) != CANDIDATES:
        raise ValueError(f"P3-E23 candidate IDs are not unique: {sample_key}")
    return {
        "sample_key": sample_key,
        "candidate_ids": candidate_ids,
        "candidate_features": source["candidate_features"][:CANDIDATES].clone(),
        "candidate_anchors": candidates,
        "raw_scores": source["raw_scores"][:CANDIDATES].clone(),
        "normalized_scores": source["normalized_scores"][:CANDIDATES].clone(),
        "scale_ids": torch.ones(CANDIDATES, dtype=torch.long),
        "scale_ranks": torch.arange(CANDIDATES).float() / (CANDIDATES - 1),
        "atom_anchors": source["atom_anchors"].clone(),
        "atom_mask": representable,
        "full_atom_mask": full_mask,
    }


def format_top192_cache(source: dict) -> dict[str, list[dict]]:
    return {
        split: [
            format_top192_record(key, records[key]) for key in sorted(records)
        ]
        for split, records in source["records"].items()
    }


def collate(records: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    names = (
        "candidate_features",
        "candidate_anchors",
        "raw_scores",
        "normalized_scores",
        "scale_ranks",
    )
    return {
        name: torch.stack([record[name] for record in records]).to(device)
        for name in names
    }


def load_dynamic_model(path: Path, expected_sha256: str, device: torch.device):
    if file_sha256(path) != expected_sha256:
        raise ValueError(f"P3-E23 checkpoint SHA256 differs: {path}")
    source = torch.load(path, map_location=device, weights_only=False)
    original = G3AtomQuerySetConfig(**source["model_config"])
    if original.candidates != 128 or original.queries != QUERIES:
        raise ValueError("P3-E23 source query/candidate contract differs")
    config = replace(original, candidates=CANDIDATES)
    model = G3PointerUtilityExistenceModel(G3AtomQuerySet(config)).to(device)
    model.load_state_dict(source["atom_query_set"], strict=True)
    model.eval()
    return model


def validate_model_output(output: dict, batch: int) -> None:
    expected = {
        "existence_logits": (batch, QUERIES),
        "pointer_logits": (batch, QUERIES, CANDIDATES),
        "coordinates": (batch, QUERIES, 2),
        "query_hidden": (batch, QUERIES, 64),
        "count_logits": (batch, CLASSES),
    }
    for name, shape in expected.items():
        value = output[name]
        if value.shape != shape:
            raise ValueError(f"P3-E23 {name} shape differs: {tuple(value.shape)}")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"P3-E23 {name} is not finite")


@torch.no_grad()
def build_evidence(
    cache: dict[str, list[dict]], base_model, expert_model, count_model, device
) -> tuple[dict[str, list[dict]], dict]:
    outputs = {}
    repeat_checks = []
    for split, records in cache.items():
        merged = []
        for start in range(0, len(records), BATCH_SIZE):
            current = records[start : start + BATCH_SIZE]
            batch = collate(current, device)
            with _autocast_context(device, "bf16"):
                base = base_model(batch)
                expert = expert_model(batch)
                query, candidate, global_features = g3_deepsets_count_features(
                    base,
                    expert,
                    batch["candidate_features"],
                    batch["raw_scores"],
                    batch["normalized_scores"],
                    batch["scale_ranks"],
                )
                count_logits = count_model(query, candidate, global_features).float()
            validate_model_output(base, len(current))
            validate_model_output(expert, len(current))
            if query.shape != (len(current), QUERIES, 8):
                raise ValueError("P3-E23 query Count evidence shape differs")
            if candidate.shape != (len(current), CANDIDATES, 41):
                raise ValueError("P3-E23 candidate Count evidence shape differs")
            if global_features.shape != (len(current), 256):
                raise ValueError("P3-E23 global Count evidence shape differs")
            if count_logits.shape != (len(current), CLASSES):
                raise ValueError("P3-E23 P3-E16 output shape differs")
            if not all(
                bool(torch.isfinite(value).all())
                for value in (query, candidate, global_features, count_logits)
            ):
                raise FloatingPointError("P3-E23 Count evidence is not finite")
            if start == 0:
                with _autocast_context(device, "bf16"):
                    repeated_base = base_model(batch)
                    repeated_expert = expert_model(batch)
                    repeated_features = g3_deepsets_count_features(
                        repeated_base,
                        repeated_expert,
                        batch["candidate_features"],
                        batch["raw_scores"],
                        batch["normalized_scores"],
                        batch["scale_ranks"],
                    )
                    repeated_count = count_model(*repeated_features).float()
                first = tensor_digest(
                    [base["existence_logits"], base["pointer_logits"],
                     expert["existence_logits"], expert["pointer_logits"], count_logits]
                )
                second = tensor_digest(
                    [repeated_base["existence_logits"], repeated_base["pointer_logits"],
                     repeated_expert["existence_logits"], repeated_expert["pointer_logits"],
                     repeated_count]
                )
                repeat_checks.append(first == second)
            for index, record in enumerate(current):
                merged.append({
                    **record,
                    "query_features": query[index].cpu().to(torch.float16),
                    "count_candidate_features": candidate[index].cpu().to(torch.float16),
                    "global_features": global_features[index].cpu().to(torch.float16),
                    "label": int(record["full_atom_mask"].sum()),
                    "baseline_count": int(base["count_logits"][index].argmax()),
                    "base_existence_logits": base["existence_logits"][index].cpu().to(torch.float16),
                    "base_pointer_logits": base["pointer_logits"][index].cpu().to(torch.float16),
                    "expert_existence_logits": expert["existence_logits"][index].cpu().to(torch.float16),
                    "expert_pointer_logits": expert["pointer_logits"][index].cpu().to(torch.float16),
                    "p3e16_count_logits": count_logits[index].cpu().to(torch.float16),
                })
        outputs[split] = merged
    return outputs, {
        "strict_load": True,
        "base_existence_shape": [QUERIES],
        "base_pointer_shape": [QUERIES, CANDIDATES],
        "base_count_shape": [CLASSES],
        "count_candidate_evidence_shape": [CANDIDATES, 41],
        "p3e16_shape": [CLASSES],
        "repeat_runs_equal": all(repeat_checks),
    }


def reachability_metrics(records: list[dict]) -> dict:
    matched = sum(int(record["atom_mask"].sum()) for record in records)
    target = sum(int(record["full_atom_mask"].sum()) for record in records)
    exact = sum(
        int(record["atom_mask"].sum()) == int(record["full_atom_mask"].sum())
        for record in records
    )
    return {
        "images": len(records),
        "matched_atoms": matched,
        "target_atoms": target,
        "atom_recall": matched / target,
        "exact_images": exact,
        "exact_rate": exact / len(records),
    }


def count_metrics(records: list[dict]) -> dict:
    predicted = [int(record["p3e16_count_logits"].argmax()) for record in records]
    labels = [int(record["label"]) for record in records]
    return {
        "images": len(records),
        "exact_images": sum(left == right for left, right in zip(predicted, labels)),
        "accuracy": sum(left == right for left, right in zip(predicted, labels)) / len(labels),
        "mae": sum(abs(left - right) for left, right in zip(predicted, labels)) / len(labels),
        "prediction_min": min(predicted),
        "prediction_max": max(predicted),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E23 output is not empty: {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("P3-E23 requires CUDA BF16")
    for path, expected in (
        (TOP256_CACHE, TOP256_CACHE_SHA256),
        (P3E22_METRICS, P3E22_METRICS_SHA256),
        *[(OLD_CACHE_ROOT / f"{split}.pt", sha) for split, sha in OLD_CACHE_SHA256.items()],
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P3-E23 input SHA256 differs: {path}")

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
    started = time.perf_counter()

    top256 = torch.load(TOP256_CACHE, map_location="cpu", weights_only=False)
    cache = format_top192_cache(top256)
    expected = json.loads(P3E22_METRICS.read_text())["metrics"]
    reachability = {split: reachability_metrics(records) for split, records in cache.items()}
    oracle_reproduced = all(
        all(
            reachability[split][name] == expected[split]["192"][name]
            for name in ("images", "matched_atoms", "target_atoms", "exact_images")
        )
        for split in cache
    )

    base_model = load_dynamic_model(p3e7.BASE_CHECKPOINT, p3e7.BASE_SHA256, device)
    expert_model = load_dynamic_model(p3e7.EXPERT_CHECKPOINT, p3e7.EXPERT_SHA256, device)
    count_model = p3e14.load_count_model(p3e17.COUNT_CHECKPOINT, device)
    evidence, compatibility = build_evidence(
        cache, base_model, expert_model, count_model, device
    )
    del base_model, expert_model, count_model

    old = {
        split: torch.load(OLD_CACHE_ROOT / f"{split}.pt", map_location="cpu", weights_only=False)["records"]
        for split in cache
    }
    old_count = {split: count_metrics(records) for split, records in old.items()}
    new_count = {split: count_metrics(records) for split, records in evidence.items()}
    old_combined = count_metrics(old["dev"] + old["confirm"])
    new_combined = count_metrics(evidence["dev"] + evidence["confirm"])
    checks = {
        "strict_dynamic_checkpoint_load": compatibility["strict_load"],
        "dynamic_shapes_valid": compatibility["base_pointer_shape"] == [128, 192],
        "repeat_runs_equal": compatibility["repeat_runs_equal"],
        "oracle_reproduced": oracle_reproduced,
        "all_split_recall_min99": all(
            value["atom_recall"] >= 0.99 for value in reachability.values()
        ),
        "dev_count_regression_max2pp": (
            new_count["dev"]["accuracy"] >= old_count["dev"]["accuracy"] - 0.02
        ),
        "confirm_diagnostic_count_regression_max2pp": (
            new_count["confirm"]["accuracy"]
            >= old_count["confirm"]["accuracy"] - 0.02
        ),
        "dev_mae_increase_max0_10": (
            new_count["dev"]["mae"] <= old_count["dev"]["mae"] + 0.10
        ),
        "confirm_diagnostic_mae_increase_max0_10": (
            new_count["confirm"]["mae"] <= old_count["confirm"]["mae"] + 0.10
        ),
        "combined_count_regression_max1pp": (
            new_combined["accuracy"] >= old_combined["accuracy"] - 0.01
        ),
        "count_range_valid": all(
            0 <= value["prediction_min"] <= value["prediction_max"] <= 128
            for value in new_count.values()
        ),
    }
    decision = (
        "top192_cycle_recovery_go"
        if all(checks.values())
        else "top192_cycle_recovery_no_go"
    )

    OUTPUT.mkdir(parents=True, exist_ok=False)
    shard_audits = {}
    for split, records in evidence.items():
        path = OUTPUT / f"{split}.pt"
        _atomic_torch_save(path, {
            "schema_version": "g3-p3e23-top192-evidence-0.1",
            "audit": AUDIT_NAME,
            "split": split,
            "manifest": top256["manifests"][split],
            "records": records,
        })
        shard_audits[split] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    audit = {
        "schema_version": "g3-p3e23-top192-cycle-recovery-audit-0.1",
        "decision": decision,
        "checks": checks,
        "protocol": {
            "queries": QUERIES,
            "candidates": CANDIDATES,
            "count_classes": CLASSES,
            "old_confirm_role": "budget_consumed_diagnostic",
            "fresh_dev_read": False,
            "sealed_test_read": False,
        },
        "compatibility": compatibility,
        "reachability": reachability,
        "count": {
            "top128_baseline": old_count,
            "top192": new_count,
            "dev_confirm_combined_top128": old_combined,
            "dev_confirm_combined_top192": new_combined,
        },
        "inputs": {
            "top256_cache": str(TOP256_CACHE.resolve()),
            "top256_cache_sha256": TOP256_CACHE_SHA256,
            "p3e22_metrics_sha256": P3E22_METRICS_SHA256,
            "base_checkpoint_sha256": p3e7.BASE_SHA256,
            "expert_checkpoint_sha256": p3e7.EXPERT_SHA256,
            "count_checkpoint_sha256": p3e17.COUNT_SHA256,
        },
        "shards": shard_audits,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
