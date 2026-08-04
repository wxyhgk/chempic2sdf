#!/usr/bin/env python3
"""Molecule-disjoint zero-training audit of a P2-W base and P3-E1 expert."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_latent_matching import (
    g3_unrestricted_fixed_cardinality_assignment,
)
from chemvector.training.atoms import audit_g3_p3e0_latent_solver as p3e0
from chemvector.training.atoms import (
    train_g3_p1a_canonical_atom_selector as p1a,
)
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import (
    train_g3_p2w_fixed_identity_pointer_utility_step4800 as p2w,
)
from chemvector.training.atoms import train_g3_p3e1_latent_structured_pilot256 as p3e1
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p3e7-base-expert-router-audit"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
BASE_CHECKPOINT = p2w.OUTPUT / "step-004800.pt"
BASE_SHA256 = p3e1.SOURCE_CHECKPOINT_SHA256
EXPERT_CHECKPOINT = p3e1.OUTPUT / "step-001920.pt"
EXPERT_SHA256 = "7880bc31c906c4afbbfd7265188c68921a115329aa795903ccbbac07c3529a3b"
PER_VARIANT = 128
PER_VARIANT_PER_SPLIT = 64
FEATURES = (
    "expert_minus_base_joint_score_mean",
    "expert_joint_score_mean",
    "base_joint_score_mean",
    "expert_minus_base_existence_mean",
    "expert_minus_base_pointer_logprob_mean",
    "candidate_overlap_fraction",
    "candidate_replacements",
    "expert_assignment_stable",
    "base_assignment_stable",
)


def select_fresh_records(cache: dict) -> tuple[dict[str, list[dict]], dict]:
    domains, capacity_audit = p2h.select_capacity_domains(cache["train5k"])
    capacity = domains["easy32"] + domains["hard32"]
    capacity_keys = {str(record["sample_key"]) for record in capacity}
    eligible = [
        record for record in cache["train5k"]
        if torch.equal(record["atom_mask"], record["full_atom_mask"])
    ]
    historical_general = sorted(
        (
            record for record in eligible
            if str(record["sample_key"]) not in capacity_keys
        ),
        key=lambda record: str(record["sample_key"]),
    )[:p3e0.GENERAL_IMAGES]
    excluded_molecules = {
        p2h._molecule_key(str(record["sample_key"]))
        for record in capacity + historical_general
    }
    selected_by_variant = {}
    used_molecules = set(excluded_molecules)
    for variant in p2h.EXPECTED_VARIANTS:
        selected = []
        for record in sorted(eligible, key=lambda value: str(value["sample_key"])):
            key = str(record["sample_key"])
            molecule = p2h._molecule_key(key)
            if p2h._variant(key) != variant or molecule in used_molecules:
                continue
            selected.append(record)
            used_molecules.add(molecule)
            if len(selected) == PER_VARIANT:
                break
        if len(selected) != PER_VARIANT:
            raise RuntimeError(f"P3-E7 lacks fresh records for {variant}")
        selected_by_variant[variant] = selected

    route_select = []
    route_confirm = []
    for variant in p2h.EXPECTED_VARIANTS:
        route_select.extend(selected_by_variant[variant][:PER_VARIANT_PER_SPLIT])
        route_confirm.extend(selected_by_variant[variant][PER_VARIANT_PER_SPLIT:])
    select_molecules = {
        p2h._molecule_key(str(record["sample_key"])) for record in route_select
    }
    confirm_molecules = {
        p2h._molecule_key(str(record["sample_key"])) for record in route_confirm
    }
    if select_molecules & confirm_molecules:
        raise RuntimeError("P3-E7 route splits overlap by molecule")
    digest = hashlib.sha256()
    for split_name, records in (
        ("route_select", route_select),
        ("route_confirm", route_confirm),
    ):
        for record in records:
            digest.update(f"{split_name}:{record['sample_key']}\n".encode())
    return {"route_select": route_select, "route_confirm": route_confirm}, {
        "selection": "molecule-disjoint one-image-per-molecule, 64 per variant per split",
        "capacity_selection": capacity_audit,
        "historical_training_images": len(capacity) + len(historical_general),
        "excluded_training_molecules": len(excluded_molecules),
        "route_select_images": len(route_select),
        "route_confirm_images": len(route_confirm),
        "route_select_molecules": len(select_molecules),
        "route_confirm_molecules": len(confirm_molecules),
        "cross_split_molecule_overlap": len(select_molecules & confirm_molecules),
        "training_molecule_overlap": len(
            (select_molecules | confirm_molecules) & excluded_molecules
        ),
        "variant_counts": {
            split: dict(Counter(p2h._variant(str(record["sample_key"])) for record in records))
            for split, records in (
                ("route_select", route_select),
                ("route_confirm", route_confirm),
            )
        },
        "sample_keys_sha256": digest.hexdigest(),
        "sample_keys": {
            split: [str(record["sample_key"]) for record in records]
            for split, records in (
                ("route_select", route_select),
                ("route_confirm", route_confirm),
            )
        },
    }


def model_prediction(output: dict, record: dict, *, count: int | None = None) -> dict:
    if count is None:
        count = int(record["full_atom_mask"].sum())
    scores = p3e0.edge_scores(output, "legacy_logsigmoid")
    assignment = g3_unrestricted_fixed_cardinality_assignment(scores, count)
    queries = assignment["query_indices"]
    candidates = assignment["candidate_indices"]
    pointer_logprob = output["pointer_logits"].log_softmax(dim=-1)
    selected_scores = scores[queries, candidates]
    expected = record["atom_anchors"][record["full_atom_mask"]].float()
    anchors = record["candidate_anchors"][candidates].float()
    stats = p1a.atom_sample_statistics(anchors, expected)
    perturbed = g3_unrestricted_fixed_cardinality_assignment(
        p3e0.perturb(scores), count
    )
    empty = count == 0
    return {
        "correct": bool(stats["whole_set_exact_images"]),
        "matched_atoms": int(stats["matched_atoms"]),
        "target_atoms": int(stats["target_atoms"]),
        "decoded_count": count,
        "candidate_indices": tuple(sorted(candidates.tolist())),
        "query_indices": tuple(sorted(queries.tolist())),
        "joint_score_mean": 0.0 if empty else float(selected_scores.mean()),
        "joint_score_min": 0.0 if empty else float(selected_scores.min()),
        "existence_mean": 0.0 if empty else float(output["existence_logits"][queries].mean()),
        "existence_min": 0.0 if empty else float(output["existence_logits"][queries].min()),
        "pointer_logprob_mean": (
            0.0 if empty else float(pointer_logprob[queries, candidates].mean())
        ),
        "pointer_logprob_min": (
            0.0 if empty else float(pointer_logprob[queries, candidates].min())
        ),
        "assignment_stable": float(
            tuple(sorted(perturbed["candidate_indices"].tolist()))
            == tuple(sorted(candidates.tolist()))
        ),
    }


def pair_features(base: dict, expert: dict) -> dict[str, float]:
    base_candidates = set(base["candidate_indices"])
    expert_candidates = set(expert["candidate_indices"])
    count = max(len(base_candidates), 1)
    return {
        "expert_minus_base_joint_score_mean": (
            expert["joint_score_mean"] - base["joint_score_mean"]
        ),
        "expert_joint_score_mean": expert["joint_score_mean"],
        "base_joint_score_mean": base["joint_score_mean"],
        "expert_minus_base_existence_mean": (
            expert["existence_mean"] - base["existence_mean"]
        ),
        "expert_minus_base_pointer_logprob_mean": (
            expert["pointer_logprob_mean"] - base["pointer_logprob_mean"]
        ),
        "candidate_overlap_fraction": len(base_candidates & expert_candidates) / count,
        "candidate_replacements": float(len(base_candidates - expert_candidates)),
        "expert_assignment_stable": expert["assignment_stable"],
        "base_assignment_stable": base["assignment_stable"],
    }


def collect_rows(
    records, base_outputs, expert_outputs, *, count_mode: str = "known"
) -> list[dict]:
    rows = []
    for record in records:
        key = str(record["sample_key"])
        if count_mode == "known":
            count = int(record["full_atom_mask"].sum())
        elif count_mode == "base_predicted":
            count = int(base_outputs[key]["count"])
        else:
            raise ValueError(f"Unknown P3-E7 Count mode: {count_mode}")
        base = model_prediction(base_outputs[key], record, count=count)
        expert = model_prediction(expert_outputs[key], record, count=count)
        rows.append({
            "sample_key": key,
            "variant": p2h._variant(key),
            "atom_count": int(record["full_atom_mask"].sum()),
            "count_mode": count_mode,
            "base": base,
            "expert": expert,
            "features": pair_features(base, expert),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    base_correct = sum(row["base"]["correct"] for row in rows)
    expert_correct = sum(row["expert"]["correct"] for row in rows)
    both = sum(row["base"]["correct"] and row["expert"]["correct"] for row in rows)
    base_only = sum(row["base"]["correct"] and not row["expert"]["correct"] for row in rows)
    expert_only = sum(row["expert"]["correct"] and not row["base"]["correct"] for row in rows)
    agreement = sum(
        row["base"]["candidate_indices"] == row["expert"]["candidate_indices"]
        for row in rows
    )
    return {
        "images": len(rows),
        "base_exact": base_correct,
        "expert_exact": expert_correct,
        "both_exact": both,
        "base_only_exact": base_only,
        "expert_only_exact": expert_only,
        "neither_exact": len(rows) - both - base_only - expert_only,
        "oracle_union_exact": base_correct + expert_only,
        "candidate_set_agreement": agreement,
        "base_matched_atoms": sum(row["base"]["matched_atoms"] for row in rows),
        "expert_matched_atoms": sum(row["expert"]["matched_atoms"] for row in rows),
        "target_atoms": sum(row["base"]["target_atoms"] for row in rows),
    }


def apply_policy(rows: list[dict], policy: dict | None) -> dict:
    routed = rescues = breaks = correct = 0
    routed_keys = []
    for row in rows:
        route = False
        if policy is not None:
            value = row["features"][policy["feature"]]
            route = (
                value >= policy["threshold"]
                if policy["direction"] == "ge" else value <= policy["threshold"]
            )
        routed += int(route)
        routed_keys.append(row["sample_key"]) if route else None
        base_correct = row["base"]["correct"]
        expert_correct = row["expert"]["correct"]
        rescues += int(route and expert_correct and not base_correct)
        breaks += int(route and base_correct and not expert_correct)
        correct += int(expert_correct if route else base_correct)
    return {
        "images": len(rows),
        "routed": routed,
        "rescues": rescues,
        "breaks": breaks,
        "exact": correct,
        "routed_sample_keys": routed_keys,
    }


def select_zero_break_policy(rows: list[dict]) -> tuple[dict | None, dict]:
    candidates = []
    for feature_index, feature in enumerate(FEATURES):
        thresholds = sorted({row["features"][feature] for row in rows})
        for direction_index, direction in enumerate(("ge", "le")):
            for threshold in thresholds:
                policy = {
                    "feature": feature,
                    "direction": direction,
                    "threshold": threshold,
                }
                metrics = apply_policy(rows, policy)
                if metrics["breaks"] == 0:
                    candidates.append((
                        -metrics["rescues"],
                        metrics["routed"],
                        feature_index,
                        direction_index,
                        threshold,
                        policy,
                        metrics,
                    ))
    if not candidates:
        return None, apply_policy(rows, None)
    winner = min(candidates)
    return winner[-2], winner[-1]


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E7 output is not empty: {OUTPUT}")
    for path, expected in (
        (BASE_CHECKPOINT, BASE_SHA256),
        (EXPERT_CHECKPOINT, EXPERT_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P3-E7 checkpoint SHA differs: {path}")
    torch.manual_seed(p3e1.SEED)
    torch.cuda.manual_seed_all(p3e1.SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    cache, cache_audit = p2e.load_cache()
    splits, selection_audit = select_fresh_records(cache)
    audit_splits = {
        **splits,
        "historical_route100": cache["route100"],
    }
    all_records = sum(audit_splits.values(), [])
    base_model = p3e0.load_model(BASE_CHECKPOINT, device)
    expert_model = p3e0.load_model(EXPERT_CHECKPOINT, device)
    base_outputs = p3e0.collect_outputs(
        base_model, all_records, device=device, batch_size=4, precision="bf16"
    )
    expert_outputs = p3e0.collect_outputs(
        expert_model, all_records, device=device, batch_size=4, precision="bf16"
    )
    rows = {
        split: collect_rows(records, base_outputs, expert_outputs)
        for split, records in audit_splits.items()
    }
    summaries = {split: summarize(values) for split, values in rows.items()}
    policy, select_metrics = select_zero_break_policy(rows["route_select"])
    confirm_metrics = apply_policy(rows["route_confirm"], policy)
    historical_metrics = apply_policy(rows["historical_route100"], policy)
    checks = {
        "molecule_disjoint": (
            selection_audit["cross_split_molecule_overlap"] == 0
            and selection_audit["training_molecule_overlap"] == 0
        ),
        "confirm_oracle_adds_at_least5": (
            summaries["route_confirm"]["expert_only_exact"] >= 5
        ),
        "select_policy_zero_break_rescue3": (
            select_metrics["breaks"] == 0 and select_metrics["rescues"] >= 3
        ),
        "confirm_policy_zero_break_rescue3": (
            confirm_metrics["breaks"] == 0 and confirm_metrics["rescues"] >= 3
        ),
    }
    decision = (
        "base_expert_router_audit_go"
        if all(checks.values()) else "base_expert_router_audit_no_go"
    )
    payload = {
        "schema_version": "g3-p3e7-base-expert-router-audit-0.1",
        "audit_name": AUDIT_NAME,
        "decision": decision,
        "checks": checks,
        "summaries": summaries,
        "selected_policy": policy,
        "route_select_policy_metrics": select_metrics,
        "route_confirm_policy_metrics": confirm_metrics,
        "historical_route100_policy_metrics": historical_metrics,
        "rows": rows,
        "lineage": {
            "base_checkpoint": str(BASE_CHECKPOINT.resolve()),
            "base_sha256": BASE_SHA256,
            "expert_checkpoint": str(EXPERT_CHECKPOINT.resolve()),
            "expert_sha256": EXPERT_SHA256,
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
            "selection": selection_audit,
            "route_features": list(FEATURES),
            "policy_selection": "maximize select rescues subject to zero breaks; then minimum routes",
            "historical_route100_used_for_policy": False,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "metrics.json", payload)
    print(json.dumps({
        "decision": decision,
        "checks": checks,
        "summaries": summaries,
        "selected_policy": policy,
        "route_select_policy_metrics": select_metrics,
        "route_confirm_policy_metrics": confirm_metrics,
        "historical_route100_policy_metrics": historical_metrics,
        "selection": selection_audit,
    }, indent=2))


if __name__ == "__main__":
    main()
