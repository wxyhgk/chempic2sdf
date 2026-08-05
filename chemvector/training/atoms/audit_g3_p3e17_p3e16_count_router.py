#!/usr/bin/env python3
"""Replay the frozen P3-E7 router with the passed P3-E16 Count model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from chemvector.components.atoms.objects.g3_base_expert_router import (
    G3_BASE_EXPERT_THRESHOLD,
)
from chemvector.components.atoms.objects.g3_deepsets_count import (
    g3_deepsets_count_features,
)
from chemvector.training.atoms import audit_g3_p3e0_latent_solver as p3e0
from chemvector.training.atoms import audit_g3_p3e7_base_expert_router as p3e7
from chemvector.training.atoms import audit_g3_p3e8_predicted_count_router as p3e8
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import train_g3_p3e14_style_count_scale as p3e14
from chemvector.training.atoms import train_g3_p3e16_style_count_epoch6 as p3e16
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p3e17-p3e16-count-router-audit"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
COUNT_CHECKPOINT = p3e16.OUTPUT / "step-008665.pt"
COUNT_SHA256 = "50b7f07a723468f1ef50b21011cc65320af9bad7c6ecfd88be3f5549444ef236"
P3E7_METRICS = p3e7.OUTPUT / "metrics.json"
P3E8_METRICS = p3e8.OUTPUT / "metrics.json"
POLICY = {
    "feature": "base_joint_score_mean",
    "direction": "le",
    "threshold": G3_BASE_EXPERT_THRESHOLD,
}


@torch.no_grad()
def collect_outputs_and_counts(
    base_model,
    expert_model,
    count_model,
    records: list[dict],
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, dict, dict[str, int]]:
    if batch_size < 1:
        raise ValueError("P3-E17 batch size must be positive")
    base_model.eval()
    expert_model.eval()
    count_model.eval()
    base_outputs = {}
    expert_outputs = {}
    counts = {}
    for start in range(0, len(records), batch_size):
        current = records[start : start + batch_size]
        batch = p2h.collate(current, device)
        candidates = p2h.model_candidates(batch)
        with _autocast_context(device, "bf16"):
            base = base_model(candidates)
            expert = expert_model(candidates)
            query, candidate, global_features = g3_deepsets_count_features(
                base,
                expert,
                batch["candidate_features"],
                batch["raw_scores"],
                batch["normalized_scores"],
                batch["scale_ranks"],
            )
            count_logits = count_model(query, candidate, global_features)
        predicted = count_logits.argmax(dim=-1).long().cpu()
        for index, record in enumerate(current):
            key = str(record["sample_key"])
            base_outputs[key] = {
                "existence_logits": base["existence_logits"][index].float().cpu(),
                "pointer_logits": base["pointer_logits"][index].float().cpu(),
                "count": int(base["count_logits"][index].argmax()),
            }
            expert_outputs[key] = {
                "existence_logits": expert["existence_logits"][index].float().cpu(),
                "pointer_logits": expert["pointer_logits"][index].float().cpu(),
                "count": int(expert["count_logits"][index].argmax()),
            }
            counts[key] = int(predicted[index])
    return base_outputs, expert_outputs, counts


def collect_rows_with_counts(records, base_outputs, expert_outputs, counts) -> list[dict]:
    rows = []
    for record in records:
        key = str(record["sample_key"])
        count = int(counts[key])
        base = p3e7.model_prediction(base_outputs[key], record, count=count)
        expert = p3e7.model_prediction(expert_outputs[key], record, count=count)
        rows.append({
            "sample_key": key,
            "variant": p2h._variant(key),
            "atom_count": int(record["full_atom_mask"].sum()),
            "count_mode": "p3e16_predicted",
            "base": base,
            "expert": expert,
            "features": p3e7.pair_features(base, expert),
        })
    return rows


def count_summary(rows: list[dict]) -> dict:
    errors = [row["base"]["decoded_count"] - row["atom_count"] for row in rows]
    return {
        "images": len(rows),
        "count_exact": sum(value == 0 for value in errors),
        "count_accuracy": sum(value == 0 for value in errors) / len(errors),
        "count_mae": sum(abs(value) for value in errors) / len(errors),
        "count_error_min": min(errors),
        "count_error_max": max(errors),
    }


def decompose(known_rows: list[dict], predicted_rows: list[dict], policy: dict) -> dict:
    if [row["sample_key"] for row in known_rows] != [
        row["sample_key"] for row in predicted_rows
    ]:
        raise ValueError("P3-E17 known/predicted row identity differs")
    known_by_key = {row["sample_key"]: row for row in known_rows}
    predicted_by_key = {row["sample_key"]: row for row in predicted_rows}
    known_policy = p3e7.apply_policy(known_rows, policy)
    predicted_policy = p3e7.apply_policy(predicted_rows, policy)
    count_correct_keys = {
        row["sample_key"]
        for row in predicted_rows
        if row["base"]["decoded_count"] == row["atom_count"]
    }

    def routed_correct(row: dict) -> bool:
        route = row["features"][policy["feature"]] <= policy["threshold"]
        return bool(row["expert"]["correct"] if route else row["base"]["correct"])

    known_correct_keys = {
        key for key, row in known_by_key.items() if routed_correct(row)
    }
    predicted_correct_keys = {
        key for key, row in predicted_by_key.items() if routed_correct(row)
    }
    route_churn = sum(
        (known_by_key[key]["features"][policy["feature"]] <= policy["threshold"])
        != (predicted_by_key[key]["features"][policy["feature"]] <= policy["threshold"])
        for key in known_by_key
    )
    conditional = [
        predicted_by_key[key] for key in sorted(count_correct_keys)
    ]
    return {
        "known_policy": known_policy,
        "predicted_policy": predicted_policy,
        "count_correct": len(count_correct_keys),
        "known_router_exact_and_count_correct": len(known_correct_keys & count_correct_keys),
        "known_router_exact_lost_to_count": len(known_correct_keys - count_correct_keys),
        "count_correct_but_known_router_wrong": len(count_correct_keys - known_correct_keys),
        "predicted_router_exact_keys_equal_intersection": (
            predicted_correct_keys == known_correct_keys & count_correct_keys
        ),
        "route_decision_churn": route_churn,
        "count_correct_conditional": {
            "images": len(conditional),
            "base_exact": sum(row["base"]["correct"] for row in conditional),
            "expert_exact": sum(row["expert"]["correct"] for row in conditional),
            "router_exact": len(predicted_correct_keys),
        },
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E17 output is not empty: {OUTPUT}")
    for path, expected in (
        (p3e7.BASE_CHECKPOINT, p3e7.BASE_SHA256),
        (p3e7.EXPERT_CHECKPOINT, p3e7.EXPERT_SHA256),
        (COUNT_CHECKPOINT, COUNT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P3-E17 checkpoint SHA differs: {path}")
    e7 = json.loads(P3E7_METRICS.read_text(encoding="utf-8"))
    e8 = json.loads(P3E8_METRICS.read_text(encoding="utf-8"))
    if e7["decision"] != "base_expert_router_audit_go" or e7["selected_policy"] != POLICY:
        raise ValueError("P3-E17 frozen P3-E7 policy differs")

    torch.manual_seed(p3e16.SEED)
    torch.cuda.manual_seed_all(p3e16.SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    cache, cache_audit = p2e.load_cache()
    splits, selection_audit = p3e7.select_fresh_records(cache)
    audit_splits = {
        "route_confirm": splits["route_confirm"],
        "historical_route100": cache["route100"],
    }
    all_records = sum(audit_splits.values(), [])
    base_model = p3e0.load_model(p3e7.BASE_CHECKPOINT, device)
    expert_model = p3e0.load_model(p3e7.EXPERT_CHECKPOINT, device)
    count_model = p3e14.load_count_model(COUNT_CHECKPOINT, device)
    base_outputs, expert_outputs, counts4 = collect_outputs_and_counts(
        base_model,
        expert_model,
        count_model,
        all_records,
        device=device,
        batch_size=4,
    )
    _, _, counts16 = collect_outputs_and_counts(
        base_model,
        expert_model,
        count_model,
        all_records,
        device=device,
        batch_size=16,
    )
    batch_count_differences = sorted(
        key for key in counts4 if counts4[key] != counts16[key]
    )
    known_rows = {
        split: p3e7.collect_rows(records, base_outputs, expert_outputs)
        for split, records in audit_splits.items()
    }
    predicted_rows = {
        split: collect_rows_with_counts(
            records, base_outputs, expert_outputs, counts4
        )
        for split, records in audit_splits.items()
    }
    known_policy = {
        split: p3e7.apply_policy(rows, POLICY) for split, rows in known_rows.items()
    }
    predicted_policy = {
        split: p3e7.apply_policy(rows, POLICY) for split, rows in predicted_rows.items()
    }
    count_metrics = {
        split: count_summary(rows) for split, rows in predicted_rows.items()
    }
    decomposition = {
        split: decompose(known_rows[split], predicted_rows[split], POLICY)
        for split in audit_splits
    }
    old_predicted = e8["predicted_count_policy_metrics"]
    checks = {
        "frozen_policy_replayed_without_selection": True,
        "known_confirm_matches_p3e7": (
            known_policy["route_confirm"]["exact"]
            == e7["route_confirm_policy_metrics"]["exact"]
        ),
        "known_route100_matches_p3e7": (
            known_policy["historical_route100"]["exact"]
            == e7["historical_route100_policy_metrics"]["exact"]
        ),
        "count_batch4_batch16_identical": not batch_count_differences,
        "predicted_exact_is_known_count_intersection": all(
            value["predicted_router_exact_keys_equal_intersection"]
            for value in decomposition.values()
        ),
        "confirm_improves_p3e8_at_least30": (
            predicted_policy["route_confirm"]["exact"]
            - old_predicted["route_confirm"]["exact"] >= 30
        ),
        "route100_improves_p3e8_at_least10": (
            predicted_policy["historical_route100"]["exact"]
            - old_predicted["historical_route100"]["exact"] >= 10
        ),
    }
    decision = (
        "p3e16_count_router_integration_go"
        if all(checks.values()) else "p3e16_count_router_integration_no_go"
    )
    digest = hashlib.sha256()
    for key in sorted(counts4):
        digest.update(f"{key}:{counts4[key]}\n".encode())
    payload = {
        "schema_version": "g3-p3e17-p3e16-count-router-audit-0.1",
        "audit_name": AUDIT_NAME,
        "decision": decision,
        "checks": checks,
        "frozen_policy": POLICY,
        "known_count_policy_metrics": known_policy,
        "predicted_count_policy_metrics": predicted_policy,
        "count_metrics": count_metrics,
        "decomposition": decomposition,
        "batch_size_count_differences": batch_count_differences,
        "rows": predicted_rows,
        "lineage": {
            "p3e7_metrics": str(P3E7_METRICS.resolve()),
            "p3e8_metrics": str(P3E8_METRICS.resolve()),
            "base_checkpoint": str(p3e7.BASE_CHECKPOINT.resolve()),
            "base_sha256": p3e7.BASE_SHA256,
            "expert_checkpoint": str(p3e7.EXPERT_CHECKPOINT.resolve()),
            "expert_sha256": p3e7.EXPERT_SHA256,
            "count_checkpoint": str(COUNT_CHECKPOINT.resolve()),
            "count_sha256": COUNT_SHA256,
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
            "selection": selection_audit,
            "formal_inference_batch_size": 4,
            "count_predictions_sha256": digest.hexdigest(),
            "route_select_read": False,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "metrics.json", payload)
    print(json.dumps({
        "decision": decision,
        "checks": checks,
        "known_count_policy_metrics": known_policy,
        "predicted_count_policy_metrics": predicted_policy,
        "count_metrics": count_metrics,
        "decomposition": decomposition,
        "batch_size_count_differences": len(batch_count_differences),
    }, indent=2))


if __name__ == "__main__":
    main()
