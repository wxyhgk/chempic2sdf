#!/usr/bin/env python3
"""Audit deterministic source-id unions of train5k/train25k Atom candidates."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e22_candidate_budgets as p3e22,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e23_top192_cycle_recovery as p3e23,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e29_bond_incident_signal as p3e29,
)
from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e32_train25k_candidate_source_ab as p3e32,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _atomic_torch_save,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p3e33-candidate-source-union"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
CACHE = OUTPUT / "union-top256.pt"
P3E32_AUDIT_SHA256 = (
    "c85a6de8f4075a045cd80616f42cfede5856e13a0a39d0602f1bdee3359e45a5"
)
P3E32_CACHE_SHA256 = (
    "a93c4109ad2abf7d0d25e6ae87a4d10eb7204d69af7b6ef6e410b871d156bacb"
)
CAPACITY = 256
STRATEGIES = (
    "train25k128_then_train5k",
    "train25k160_then_train5k",
    "best_normalized_rank",
    "alternating_rank",
)
MIN_TOP128_EXACT = 145
MIN_RECOVERED_OLD_TAIL_IMAGES = 13
MAX_REGRESSED_OLD_REACHABLE_IMAGES = 1


def _ordered_occurrences(
    old: dict, new: dict, strategy: str
) -> list[tuple[str, int]]:
    if strategy == "train25k128_then_train5k":
        return [*(('train25k', i) for i in range(128)), *(('train5k', i) for i in range(256)), *(('train25k', i) for i in range(128, 256))]
    if strategy == "train25k160_then_train5k":
        return [*(('train25k', i) for i in range(160)), *(('train5k', i) for i in range(256)), *(('train25k', i) for i in range(160, 256))]
    if strategy == "alternating_rank":
        return [value for index in range(256) for value in (("train25k", index), ("train5k", index))]
    if strategy == "best_normalized_rank":
        occurrences = []
        for source, record in (("train25k", new), ("train5k", old)):
            for index, source_id in enumerate(record["source_ids"]):
                occurrences.append((index / 255.0, 0 if source == "train25k" else 1, int(source_id), source, index))
        occurrences.sort()
        return [(source, index) for _, _, _, source, index in occurrences]
    raise ValueError(f"P3-E33 unknown union strategy: {strategy}")


def merge_candidate_records(old: dict, new: dict, strategy: str) -> dict:
    if strategy not in STRATEGIES:
        raise ValueError(f"P3-E33 unknown strategy: {strategy}")
    selected = []
    seen = set()
    records = {"train5k": old, "train25k": new}
    for source, index in _ordered_occurrences(old, new, strategy):
        record = records[source]
        source_id = int(record["source_ids"][index])
        if source_id in seen:
            continue
        seen.add(source_id)
        selected.append((source, index))
        if len(selected) == CAPACITY:
            break
    if len(selected) != CAPACITY:
        raise RuntimeError("P3-E33 union has fewer than 256 unique source IDs")
    source_codes = torch.tensor(
        [1 if source == "train25k" else 0 for source, _ in selected],
        dtype=torch.int8,
    )
    return {
        "source_ids": torch.tensor(
            [int(records[source]["source_ids"][index]) for source, index in selected]
        ),
        "candidate_features": torch.stack(
            [records[source]["candidate_features"][index] for source, index in selected]
        ),
        "candidate_anchors": torch.stack(
            [records[source]["candidate_anchors"][index] for source, index in selected]
        ),
        "raw_scores": torch.tensor(
            [float(records[source]["raw_scores"][index]) for source, index in selected]
        ),
        "source_codes": source_codes,
    }


def choose_winner(results: dict[str, dict]) -> tuple[str, dict]:
    return max(
        results.items(),
        key=lambda item: (
            item[1]["reachability"]["192"]["exact_images"],
            item[1]["reachability"]["128"]["exact_images"],
            item[1]["comparison"]["recovered_old_tail_images"],
            -item[1]["comparison"]["regressed_old_reachable_images"],
            item[1]["known_count_raw"]["exact_images"],
            -STRATEGIES.index(item[0]),
        ),
    )


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E33 output is not empty: {OUTPUT}")
    if file_sha256(p3e32.OUTPUT / "audit.json") != P3E32_AUDIT_SHA256:
        raise ValueError("P3-E33 P3-E32 audit SHA256 differs")
    if file_sha256(p3e32.CACHE) != P3E32_CACHE_SHA256:
        raise ValueError("P3-E33 P3-E32 cache SHA256 differs")
    targets, input_audit, _ = p3e29._load_micro160()
    old_payload = torch.load(p3e22.CACHE, map_location="cpu", weights_only=False)
    new_payload = torch.load(p3e32.CACHE, map_location="cpu", weights_only=False)
    old_all = old_payload["records"]["train"]
    new_all = new_payload["records"]
    keys = [str(record["sample_key"]) for record in targets]
    old = {key: old_all[key] for key in keys}
    new = {key: new_all[key] for key in keys}
    unions = {}
    results = {}
    for strategy in STRATEGIES:
        union = {
            key: merge_candidate_records(old[key], new[key], strategy) for key in keys
        }
        unions[strategy] = union
        results[strategy] = {
            "reachability": p3e32.reachability_metrics(union, targets),
            "known_count_raw": p3e32.known_count_raw_metrics(union, targets),
            "comparison": p3e32.compare_sources(old, union, targets),
            "train25k_fraction": sum(
                int(record["source_codes"].sum()) for record in union.values()
            ) / (len(union) * CAPACITY),
        }
    winner_name, winner = choose_winner(results)
    checks = {
        "union_top192_all160_reachable": winner["reachability"]["192"]["exact_images"] == 160,
        "union_top128_exact_min145": winner["reachability"]["128"]["exact_images"] >= MIN_TOP128_EXACT,
        "recover_old_tail_images_min13": winner["comparison"]["recovered_old_tail_images"] >= MIN_RECOVERED_OLD_TAIL_IMAGES,
        "regress_old_reachable_images_max1": winner["comparison"]["regressed_old_reachable_images"] <= MAX_REGRESSED_OLD_REACHABLE_IMAGES,
        "all_strategies_complete": len(results) == len(STRATEGIES),
        "all_160_images_present": all(len(value) == 160 for value in unions.values()),
    }
    decision = (
        "candidate_source_union_go"
        if all(checks.values())
        else "candidate_source_union_no_go"
    )
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_torch_save(CACHE, {
        "schema_version": "g3-p3e33-candidate-source-union-cache-0.1",
        "winner": winner_name,
        "records": unions[winner_name],
        "source_code": {"0": "train5k", "1": "train25k"},
    })
    audit = {
        "schema_version": "g3-p3e33-candidate-source-union-audit-0.1",
        "decision": decision,
        "checks": checks,
        "winner": {"name": winner_name, **winner},
        "results": results,
        "protocol": {
            "strategies": list(STRATEGIES),
            "deduplication": "exact shared stride1 source_id",
            "pixel_distance_deduplication": False,
            "capacity": CAPACITY,
            "top192_exact_gate": 160,
            "top128_exact_gate": MIN_TOP128_EXACT,
            "old_tail_recovery_gate": MIN_RECOVERED_OLD_TAIL_IMAGES,
            "old_reachable_regression_gate": MAX_REGRESSED_OLD_REACHABLE_IMAGES,
        },
        "cache": {"path": str(CACHE.resolve()), "sha256": file_sha256(CACHE)},
        "input": input_audit,
        "lineage": {
            "old_cache": str(p3e22.CACHE.resolve()),
            "old_cache_sha256": p3e23.TOP256_CACHE_SHA256,
            "new_cache": str(p3e32.CACHE.resolve()),
            "new_cache_sha256": P3E32_CACHE_SHA256,
            "p3e32_audit_sha256": P3E32_AUDIT_SHA256,
        },
        "data_access": {
            "cached_prediction_only": True,
            "png_read": False,
            "training_performed": False,
            "dev_read": False,
            "old_confirm_read": False,
            "fresh_dev_read": False,
            "uspto_read": False,
            "mr_tadf_read": False,
            "sealed_test_read": False,
        },
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
