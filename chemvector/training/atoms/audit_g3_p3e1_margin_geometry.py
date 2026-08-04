#!/usr/bin/env python3
"""Audit whether P3-E1's fixed whole-image margin scales with the real error."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from chemvector.components.atoms.objects.g3_latent_matching import (
    g3_best_invalid_assignment,
    g3_candidate_set_gt_coverage,
    g3_gt_candidate_compatibility,
    g3_hall_witnesses,
    g3_latent_positive_assignment,
)
from chemvector.training.atoms import (
    audit_g3_p3e0_best_invalid_solver as p3e0b,
)
from chemvector.training.atoms import audit_g3_p3e0_latent_solver as p3e0
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import (
    train_g3_p2w_fixed_identity_pointer_utility_step4800 as p2w,
)
from chemvector.training.atoms import (
    train_g3_p3e1_latent_structured_pilot256 as p3e1,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p3e1-margin-geometry-audit"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
P3E1_CHECKPOINT = p3e1.OUTPUT / "step-001920.pt"
P3E1_CHECKPOINT_SHA256 = (
    "7880bc31c906c4afbbfd7265188c68921a115329aa795903ccbbac07c3529a3b"
)


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def distribution(values: list[int]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items())}


def safe_correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def evaluate(records: list[dict], outputs: dict[str, dict]) -> tuple[dict, list[dict]]:
    rows = []
    digest = hashlib.sha256()
    for record in records:
        key = str(record["sample_key"])
        target = record["atom_anchors"][record["full_atom_mask"]].float()
        candidates = record["candidate_anchors"].float()
        compatibility = g3_gt_candidate_compatibility(target, candidates)
        scores = p3e0.edge_scores(outputs[key], "legacy_logsigmoid")
        positive = g3_latent_positive_assignment(
            scores, compatibility, max_ownerships=p3e0.MAX_OWNERSHIPS
        )
        negative = g3_best_invalid_assignment(
            scores,
            compatibility,
            max_component_gt=p3e0b.MAX_COMPONENT_GT,
            max_witnesses=p3e0b.MAX_WITNESSES,
        )
        if negative is None:
            raise RuntimeError(f"P3-E1 margin audit has no invalid assignment: {key}")

        count = len(target)
        coverage = g3_candidate_set_gt_coverage(
            negative["candidate_indices"], compatibility
        )
        deficiency = count - coverage
        positive_candidates = set(positive["candidate_indices"].tolist())
        negative_candidates = set(negative["candidate_indices"].tolist())
        positive_queries = set(positive["query_indices"].tolist())
        negative_queries = set(negative["query_indices"].tolist())
        candidate_replacements = len(positive_candidates - negative_candidates)
        query_replacements = len(positive_queries - negative_queries)
        total_gap = float(positive["score"]) - float(negative["score"])
        witness_gt = 0
        witness_candidates = 0
        if int(negative["witness_index"]) >= 0:
            witness = g3_hall_witnesses(
                compatibility,
                max_component_gt=p3e0b.MAX_COMPONENT_GT,
                max_witnesses=p3e0b.MAX_WITNESSES,
            )[int(negative["witness_index"])]
            witness_gt = len(witness["gt_indices"])
            witness_candidates = int(witness["group_candidates"].sum())

        row = {
            "sample_key": key,
            "atom_count": count,
            "gt_coverage": coverage,
            "gt_deficiency": deficiency,
            "candidate_replacements": candidate_replacements,
            "query_replacements": query_replacements,
            "witness_gt": witness_gt,
            "witness_candidates": witness_candidates,
            "unrestricted_invalid": int(negative["witness_index"]) == -1,
            "valid_minus_invalid_total_gap": total_gap,
            "valid_minus_invalid_mean_gap": total_gap / count,
            "p3e1_fixed_margin_satisfied": total_gap >= count,
            "deficiency_margin_satisfied": total_gap >= deficiency,
            "replacement_margin_satisfied": total_gap >= candidate_replacements,
        }
        rows.append(row)
        digest.update((json.dumps(row, sort_keys=True) + "\n").encode())

    counts = [value["atom_count"] for value in rows]
    deficiencies = [value["gt_deficiency"] for value in rows]
    replacements = [value["candidate_replacements"] for value in rows]
    gaps = [value["valid_minus_invalid_total_gap"] for value in rows]
    mean_gaps = [value["valid_minus_invalid_mean_gap"] for value in rows]
    summary = {
        "images": len(rows),
        "atom_count_p50": percentile(counts, 50),
        "atom_count_p95": percentile(counts, 95),
        "atom_count_max": max(counts),
        "gt_deficiency_distribution": distribution(deficiencies),
        "candidate_replacement_distribution": distribution(replacements),
        "query_replacement_distribution": distribution([
            value["query_replacements"] for value in rows
        ]),
        "witness_gt_distribution": distribution([
            value["witness_gt"] for value in rows
        ]),
        "unrestricted_invalid_images": sum(
            value["unrestricted_invalid"] for value in rows
        ),
        "total_gap_p10": percentile(gaps, 10),
        "total_gap_p50": percentile(gaps, 50),
        "total_gap_p90": percentile(gaps, 90),
        "mean_gap_p50": percentile(mean_gaps, 50),
        "p3e1_fixed_margin_satisfied_images": sum(
            value["p3e1_fixed_margin_satisfied"] for value in rows
        ),
        "deficiency_margin_satisfied_images": sum(
            value["deficiency_margin_satisfied"] for value in rows
        ),
        "replacement_margin_satisfied_images": sum(
            value["replacement_margin_satisfied"] for value in rows
        ),
        "atom_count_vs_mean_gap_correlation": safe_correlation(counts, mean_gaps),
        "atom_count_vs_total_gap_correlation": safe_correlation(counts, gaps),
        "digest": digest.hexdigest(),
    }
    return summary, rows


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E1 margin audit output is not empty: {OUTPUT}")
    if file_sha256(P3E1_CHECKPOINT) != P3E1_CHECKPOINT_SHA256:
        raise ValueError("P3-E1 margin audit checkpoint SHA256 differs")
    torch.manual_seed(p3e1.SEED)
    torch.cuda.manual_seed_all(p3e1.SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    cache, cache_audit = p2e.load_cache()
    eligible = [
        record for record in cache["train5k"]
        if torch.equal(record["atom_mask"], record["full_atom_mask"])
    ]
    domains, selection_audit = p2h.select_capacity_domains(cache["train5k"])
    capacity_keys = {
        str(record["sample_key"])
        for record in domains["easy32"] + domains["hard32"]
    }
    general = sorted(
        (
            record for record in eligible
            if str(record["sample_key"]) not in capacity_keys
        ),
        key=lambda record: str(record["sample_key"]),
    )[:p3e0.GENERAL_IMAGES]
    record_domains = {
        "easy32": domains["easy32"],
        "hard32": domains["hard32"],
        "general192": general,
    }
    checkpoints = {
        "p2w_step4800": (p2w.OUTPUT / "step-004800.pt", p3e1.SOURCE_CHECKPOINT_SHA256),
        "p3e1_step1920": (P3E1_CHECKPOINT, P3E1_CHECKPOINT_SHA256),
    }
    summaries = {}
    sample_rows = {}
    all_records = domains["easy32"] + domains["hard32"] + general
    for model_name, (path, expected_sha) in checkpoints.items():
        if file_sha256(path) != expected_sha:
            raise ValueError(f"P3-E1 margin audit checkpoint differs: {model_name}")
        model = p3e0.load_model(path, device)
        outputs = p3e0.collect_outputs(
            model, all_records, device=device, batch_size=4, precision="bf16"
        )
        summaries[model_name] = {}
        sample_rows[model_name] = {}
        for domain_name, records in record_domains.items():
            summary, rows = evaluate(records, outputs)
            summaries[model_name][domain_name] = summary
            sample_rows[model_name][domain_name] = rows

    source_capacity = [
        summaries["p2w_step4800"]["easy32"],
        summaries["p2w_step4800"]["hard32"],
    ]
    checks = {
        "source_capacity_best_invalid_is_one_gt_error": all(
            summary["gt_deficiency_distribution"] == {"1": summary["images"]}
            for summary in source_capacity
        ),
        "source_capacity_deficiency_margin_satisfied63": sum(
            summary["deficiency_margin_satisfied_images"]
            for summary in source_capacity
        ) >= 63,
        "source_capacity_fixed_margin_satisfied_at_most1": sum(
            summary["p3e1_fixed_margin_satisfied_images"]
            for summary in source_capacity
        ) <= 1,
        "source_margin_mismatch_affects_at_least62_images": sum(
            summary["deficiency_margin_satisfied_images"]
            - summary["p3e1_fixed_margin_satisfied_images"]
            for summary in source_capacity
        ) >= 62,
        "trained_hard_capacity_regressed": (
            summaries["p3e1_step1920"]["hard32"]
            ["unrestricted_invalid_images"] >= 30
        ),
        "repeatable_inputs": len(general) == p3e0.GENERAL_IMAGES,
    }
    decision = (
        "fixed_whole_image_margin_misscaled"
        if all(checks.values()) else "fixed_whole_image_margin_not_disproven"
    )
    payload = {
        "schema_version": "g3-p3e1-margin-geometry-audit-0.1",
        "audit_name": AUDIT_NAME,
        "decision": decision,
        "checks": checks,
        "summaries": summaries,
        "sample_rows": sample_rows,
        "lineage": {
            "checkpoints": {
                name: {"path": str(path.resolve()), "sha256": sha}
                for name, (path, sha) in checkpoints.items()
            },
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
            "capacity_selection": selection_audit,
            "general_sample_keys": [str(record["sample_key"]) for record in general],
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "metrics.json", payload)
    print(json.dumps({
        "decision": decision,
        "checks": checks,
        "summaries": summaries,
    }, indent=2))


if __name__ == "__main__":
    main()
