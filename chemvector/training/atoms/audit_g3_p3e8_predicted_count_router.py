#!/usr/bin/env python3
"""Replay the frozen P3-E7 router with the deployable P2-W predicted Count."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from chemvector.components.atoms.objects.g3_base_expert_router import (
    G3_BASE_EXPERT_THRESHOLD,
)
from chemvector.training.atoms import audit_g3_p3e0_latent_solver as p3e0
from chemvector.training.atoms import (
    audit_g3_p3e7_base_expert_router as p3e7,
)
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p3e1_latent_structured_pilot256 as p3e1
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p3e8-predicted-count-router-audit"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
P3E7_METRICS = p3e7.OUTPUT / "metrics.json"
POLICY = {
    "feature": "base_joint_score_mean",
    "direction": "le",
    "threshold": G3_BASE_EXPERT_THRESHOLD,
}


def count_summary(rows: list[dict]) -> dict:
    errors = [row["base"]["decoded_count"] - row["atom_count"] for row in rows]
    return {
        "images": len(rows),
        "count_exact": sum(value == 0 for value in errors),
        "count_mae": sum(abs(value) for value in errors) / len(errors),
        "count_error_min": min(errors),
        "count_error_max": max(errors),
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E8 output is not empty: {OUTPUT}")
    for path, expected in (
        (p3e7.BASE_CHECKPOINT, p3e7.BASE_SHA256),
        (p3e7.EXPERT_CHECKPOINT, p3e7.EXPERT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P3-E8 checkpoint SHA differs: {path}")
    e7 = json.loads(P3E7_METRICS.read_text(encoding="utf-8"))
    if e7["decision"] != "base_expert_router_audit_go":
        raise ValueError("P3-E8 requires the passed P3-E7 audit")
    if e7["selected_policy"] != POLICY:
        raise ValueError("P3-E8 frozen policy differs from P3-E7")

    torch.manual_seed(p3e1.SEED)
    torch.cuda.manual_seed_all(p3e1.SEED)
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
    base_outputs = p3e0.collect_outputs(
        base_model, all_records, device=device, batch_size=4, precision="bf16"
    )
    expert_outputs = p3e0.collect_outputs(
        expert_model, all_records, device=device, batch_size=4, precision="bf16"
    )
    rows = {
        split: p3e7.collect_rows(
            records,
            base_outputs,
            expert_outputs,
            count_mode="base_predicted",
        )
        for split, records in audit_splits.items()
    }
    predicted_summaries = {
        split: p3e7.summarize(values) for split, values in rows.items()
    }
    predicted_policy_metrics = {
        split: p3e7.apply_policy(values, POLICY) for split, values in rows.items()
    }
    count_metrics = {split: count_summary(values) for split, values in rows.items()}
    known = {
        "route_confirm": {
            "summary": e7["summaries"]["route_confirm"],
            "policy": e7["route_confirm_policy_metrics"],
        },
        "historical_route100": {
            "summary": e7["summaries"]["historical_route100"],
            "policy": e7["historical_route100_policy_metrics"],
        },
    }
    confirm_known_exact = known["route_confirm"]["policy"]["exact"]
    confirm_predicted_exact = predicted_policy_metrics["route_confirm"]["exact"]
    checks = {
        "frozen_policy_replayed_without_selection": True,
        "confirm_count_exact_below25pct": (
            count_metrics["route_confirm"]["count_exact"] < 64
        ),
        "confirm_known_to_predicted_drop_at_least50": (
            confirm_known_exact - confirm_predicted_exact >= 50
        ),
        "route100_known_to_predicted_drop_at_least20": (
            known["historical_route100"]["policy"]["exact"]
            - predicted_policy_metrics["historical_route100"]["exact"] >= 20
        ),
    }
    decision = (
        "predicted_count_is_primary_router_bottleneck"
        if all(checks.values()) else "predicted_count_bottleneck_not_confirmed"
    )
    payload = {
        "schema_version": "g3-p3e8-predicted-count-router-audit-0.1",
        "audit_name": AUDIT_NAME,
        "decision": decision,
        "checks": checks,
        "frozen_policy": POLICY,
        "known_count_reference": known,
        "predicted_count_summaries": predicted_summaries,
        "predicted_count_policy_metrics": predicted_policy_metrics,
        "count_metrics": count_metrics,
        "rows": rows,
        "lineage": {
            "p3e7_metrics": str(P3E7_METRICS.resolve()),
            "base_checkpoint": str(p3e7.BASE_CHECKPOINT.resolve()),
            "base_sha256": p3e7.BASE_SHA256,
            "expert_checkpoint": str(p3e7.EXPERT_CHECKPOINT.resolve()),
            "expert_sha256": p3e7.EXPERT_SHA256,
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
            "selection": selection_audit,
            "route_select_read": False,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "metrics.json", payload)
    print(json.dumps({
        "decision": decision,
        "checks": checks,
        "known_count_reference": known,
        "predicted_count_summaries": predicted_summaries,
        "predicted_count_policy_metrics": predicted_policy_metrics,
        "count_metrics": count_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
