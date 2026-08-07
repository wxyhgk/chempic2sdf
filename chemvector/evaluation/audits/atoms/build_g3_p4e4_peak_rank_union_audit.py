#!/usr/bin/env python3
"""Audit P4-E3/train5k source unions and cache the smallest viable winner."""

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
from chemvector.training.atoms import (
    train_g3_p4e3_global_peak_rank_pilot20k as p4e3,
)
from chemvector.training.graph.scenes.g3_structural_data import (
    load_g3_sample_manifest,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p4e4-peak-rank-union-audit"
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
P4E3_CHECKPOINT_SHA256 = (
    "1c358d0c936c687fa0e40ea5e1d9364dd98a663bb762d294db76b43d5f009593"
)
P4E3_AUDIT_SHA256 = (
    "da98b8bb4a576359936a7a73b0fc65ffbcbb67080501d074aaa4e8d59db61c48"
)
P4E3_FINAL_SHA256 = (
    "bfcdbf2dbb7fd931f857cbebfad5563f916cb40418a9653faffac13e3c720cc4"
)
P4E3_CHECKPOINT = p4e3.OUTPUT / f"step-{p4e3.STEPS:06d}.pt"
PREFIXES = (128, 160, 192)
BUDGETS = (128, 192, 256)
CANDIDATES = 256
FEATURE_DIM = 72
BATCH_SIZE = 24
MIN_RECALL = 0.999
SEED = 20260844


def merge_candidate_records(old: dict, new: dict, new_prefix: int) -> dict:
    if new_prefix not in PREFIXES:
        raise ValueError("P4-E4 union prefix is not pre-registered")
    occurrences = [
        *(("new", index) for index in range(new_prefix)),
        *(("old", index) for index in range(CANDIDATES)),
        *(("new", index) for index in range(new_prefix, CANDIDATES)),
    ]
    records = {"old": old, "new": new}
    selected: list[tuple[str, int]] = []
    seen: set[int] = set()
    for source, index in occurrences:
        source_id = int(records[source]["source_ids"][index])
        if source_id in seen:
            continue
        seen.add(source_id)
        selected.append((source, index))
        if len(selected) == CANDIDATES:
            break
    if len(selected) != CANDIDATES:
        raise RuntimeError("P4-E4 union has fewer than 256 unique source IDs")
    return {
        "source_ids": torch.tensor(
            [int(records[source]["source_ids"][index]) for source, index in selected]
        ),
        "candidate_anchors": torch.stack(
            [records[source]["candidate_anchors"][index] for source, index in selected]
        ),
        "source_codes": torch.tensor(
            [1 if source == "new" else 0 for source, _ in selected],
            dtype=torch.int8,
        ),
    }


def source_aware_union_record(
    old: dict, new: dict, sample_key: str, new_prefix: int
) -> dict:
    merged = merge_candidate_records(old, new, new_prefix)
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
                float(old["normalized_scores"][old_index])
                if old_index is not None
                else 0.0,
                float(new["normalized_scores"][new_index])
                if new_index is not None
                else 0.0,
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
        raise RuntimeError("P4-E4 source-aware feature shape differs")
    if not torch.equal(old["atom_mask"], new["atom_mask"]) or not torch.allclose(
        old["atom_anchors"], new["atom_anchors"], atol=0.0, rtol=0.0
    ):
        raise ValueError("P4-E4 dual-source GT tensors differ")
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


def _empty_totals() -> dict:
    return {"images": 0, "matched": 0, "targets": 0, "exact": 0}


def _finalize_totals(value: dict) -> dict:
    return {
        **value,
        "recall": value["matched"] / value["targets"],
        "exact_rate": value["exact"] / value["images"],
    }


def reachability(records: list[dict], budget: int) -> dict:
    if budget not in BUDGETS:
        raise ValueError("P4-E4 candidate budget is not pre-registered")
    totals = _empty_totals()
    variants: dict[str, dict] = {}
    for record in records:
        expected = record["atom_anchors"][record["atom_mask"]]
        matched = p1a.maximum_cardinality_radius_match_count(
            record["candidate_anchors"][:budget],
            expected,
            radius_pixels=6.0,
            image_size=768.0,
        )
        exact = int(matched == len(expected))
        for current in (
            totals,
            variants.setdefault(
                str(record["sample_key"]).rsplit("_", 1)[-1], _empty_totals()
            ),
        ):
            current["images"] += 1
            current["matched"] += matched
            current["targets"] += len(expected)
            current["exact"] += exact
    return {
        **_finalize_totals(totals),
        "by_variant": {
            name: _finalize_totals(value)
            for name, value in sorted(variants.items())
        },
    }


def _plain_record(record: dict, sample_key: str) -> dict:
    return {
        "sample_key": sample_key,
        "candidate_anchors": record["candidate_anchors"],
        "atom_anchors": record["atom_anchors"],
        "atom_mask": record["atom_mask"],
    }


def _source_metrics(source: dict[str, dict[str, dict]]) -> dict:
    return {
        split: {
            str(budget): reachability(
                [_plain_record(record, key) for key, record in records.items()],
                budget,
            )
            for budget in BUDGETS
        }
        for split, records in source.items()
    }


def _union_metrics(
    old: dict[str, dict[str, dict]],
    new: dict[str, dict[str, dict]],
    new_prefix: int,
) -> dict:
    output = {}
    for split in old:
        if set(old[split]) != set(new[split]):
            raise ValueError(f"P4-E4 {split} source identities differ")
        records = []
        for key in old[split]:
            merged = merge_candidate_records(old[split][key], new[split][key], new_prefix)
            records.append(
                {
                    "sample_key": key,
                    "candidate_anchors": merged["candidate_anchors"],
                    "atom_anchors": old[split][key]["atom_anchors"],
                    "atom_mask": old[split][key]["atom_mask"],
                }
            )
        output[split] = {
            str(budget): reachability(records, budget) for budget in BUDGETS
        }
    return output


def choose_winner(results: dict[str, dict]) -> tuple[str, int] | None:
    feasible = []
    for prefix_name, metrics in results.items():
        for budget in BUDGETS:
            pilot = metrics["pilot_train"][str(budget)]
            dev = metrics["dev"][str(budget)]
            if pilot["recall"] >= MIN_RECALL and dev["recall"] >= MIN_RECALL:
                feasible.append(
                    (
                        budget,
                        -dev["exact"],
                        -pilot["exact"],
                        PREFIXES.index(int(prefix_name)),
                        prefix_name,
                    )
                )
                break
    if not feasible:
        return None
    winner = min(feasible)
    return winner[-1], winner[0]


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
    return p1x._collect_scale(
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


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P4-E4 output is not empty: {OUTPUT}")
    for path, expected in (
        (P4E0_ROOT / "audit.json", P4E0_AUDIT_SHA256),
        (p4e3.OUTPUT / "audit.json", P4E3_AUDIT_SHA256),
        (p4e3.OUTPUT / "final_metrics.json", P4E3_FINAL_SHA256),
        (P4E3_CHECKPOINT, P4E3_CHECKPOINT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P4-E4 input SHA256 differs: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("P4-E4 requires CUDA BF16")
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
        P4E3_CHECKPOINT,
        P4E3_CHECKPOINT_SHA256,
        p4e3.MODEL_TYPE,
        domains,
        collator,
        device,
    )
    source_metrics = {"train5k": _source_metrics(old), "p4e3": _source_metrics(new)}
    results = {
        str(prefix): _union_metrics(old, new, prefix) for prefix in PREFIXES
    }
    winner = choose_winner(results)
    checks = {
        "p4e3_dev_top128_replays_final": (
            source_metrics["p4e3"]["dev"]["128"]["matched"] == 32_195
            and source_metrics["p4e3"]["dev"]["128"]["exact"] == 550
        ),
        "one_union_budget_passes_pilot_and_dev_999": winner is not None,
        "confirm_v2_not_read": True,
        "fresh_and_external_not_read": True,
    }
    decision = "peak_rank_union_audit_go" if all(checks.values()) else "peak_rank_union_audit_no_go"

    OUTPUT.mkdir(parents=True, exist_ok=False)
    outputs = {}
    combined_digest = hashlib.sha256()
    if winner is not None:
        prefix_name, selected_budget = winner
        prefix = int(prefix_name)
        for split in domains:
            keys = [record.sample_key for record in domains[split]]
            records = [
                source_aware_union_record(old[split][key], new[split][key], key, prefix)
                for key in keys
            ]
            path = OUTPUT / f"{split}.pt"
            _atomic_torch_save(
                path,
                {
                    "schema_version": "g3-p4e4-source-aware-union-cache-0.1",
                    "split": split,
                    "new_prefix": prefix,
                    "selected_budget": selected_budget,
                    "feature_dim": FEATURE_DIM,
                    "records": records,
                },
            )
            for record in records:
                combined_digest.update(str(record["sample_key"]).encode())
                combined_digest.update(record["candidate_ids"].numpy().tobytes())
                combined_digest.update(record["candidate_features"].numpy().tobytes())
            outputs[split] = {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }

    peak_counts = {
        source_name: {
            split: {
                "mean": float(np.mean([value["regional_maxima_count"] for value in records.values()])),
                "p90": float(np.percentile([value["regional_maxima_count"] for value in records.values()], 90)),
                "maximum": int(max(value["regional_maxima_count"] for value in records.values())),
            }
            for split, records in source.items()
        }
        for source_name, source in (("train5k", old), ("p4e3", new))
    }
    audit = {
        "schema_version": "g3-p4e4-peak-rank-union-audit-0.1",
        "decision": decision,
        "checks": checks,
        "winner": (
            {"new_prefix": int(winner[0]), "selected_budget": winner[1]}
            if winner is not None
            else None
        ),
        "protocol": {
            "prefixes": list(PREFIXES),
            "budgets": list(BUDGETS),
            "minimum_pilot_and_dev_recall": MIN_RECALL,
            "winner_order": "smallest budget,dev exact,pilot exact,prefix order",
            "deduplication": "exact shared stride1 source_id",
            "pixel_distance_deduplication": False,
            "feature_dim": FEATURE_DIM,
            "batch_size": BATCH_SIZE,
        },
        "source_metrics": source_metrics,
        "union_results": results,
        "regional_maxima": peak_counts,
        "outputs": outputs,
        "combined_tensor_sha256": combined_digest.hexdigest() if outputs else None,
        "manifests": manifest_audit,
        "lineage": {"train5k": old_lineage, "p4e3": new_lineage},
        "data_access": {
            "pilot_train_png_read": True,
            "dev_png_read": True,
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
