#!/usr/bin/env python3
"""Zero-training audit for exact latent Atom matching and identity stability."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from chemvector.components.atoms.objects.g3_atom_query_set import (
    G3AtomQuerySet,
    G3AtomQuerySetConfig,
    G3PointerUtilityExistenceModel,
)
from chemvector.components.atoms.objects.g3_latent_matching import (
    g3_candidate_set_covers_gt,
    g3_gt_candidate_compatibility,
    g3_latent_positive_assignment,
    g3_unrestricted_fixed_cardinality_assignment,
)
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import (
    train_g3_p2w_fixed_identity_pointer_utility_step4800 as p2w,
)
from chemvector.training.atoms import train_g3_p3b_joint_atom_query_train4937 as p3b
from chemvector.training.atoms import (
    train_g3_p3d_structured_matching_replay_train4937 as p3d,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _atomic_json,
    _autocast_context,
)
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p3e0-latent-solver-audit"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
GENERAL_IMAGES = 192
PERTURBATION = 1e-4
MAX_OWNERSHIPS = 256
CHECKPOINTS = {
    "p2w": (
        p2w.OUTPUT / "step-004800.pt",
        "229fd1299a68aa544c0bf3ee43acabcdc1386a9df3f57dfb153113c24328b6b9",
    ),
    "p3b": (
        p3b.OUTPUT / "step-006175.pt",
        "6cfe359ce9ae7904665719957c8a7bb50388d3fb74e68ef03887ec419d7eb9ad",
    ),
    "p3d": (
        p3d.OUTPUT / "step-004875.pt",
        "5ee5533a15f290b4e03a990800149a2e4a4a2751d047537b0bfcef21171b8f3e",
    ),
}


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), value))


def assignment_signature(assignment: dict) -> dict:
    queries = assignment["query_indices"].cpu().tolist()
    candidates = assignment["candidate_indices"].cpu().tolist()
    return {
        "queries": tuple(sorted(queries)),
        "candidates": tuple(sorted(candidates)),
        "edges": tuple(sorted(zip(queries, candidates))),
    }


def compare_signatures(reference: dict, current: dict) -> dict:
    reference_queries = set(reference["queries"])
    current_queries = set(current["queries"])
    reference_candidates = set(reference["candidates"])
    current_candidates = set(current["candidates"])
    count = max(len(reference_queries), 1)
    return {
        "query_exact": reference["queries"] == current["queries"],
        "candidate_exact": reference["candidates"] == current["candidates"],
        "edge_exact": reference["edges"] == current["edges"],
        "query_overlap": len(reference_queries & current_queries) / count,
        "candidate_overlap": len(reference_candidates & current_candidates) / count,
    }


def load_model(path: Path, device: torch.device):
    source = torch.load(path, map_location=device, weights_only=False)
    base = G3AtomQuerySet(G3AtomQuerySetConfig(**source["model_config"])).to(device)
    model = G3PointerUtilityExistenceModel(base).to(device)
    model.load_state_dict(source["atom_query_set"], strict=True)
    model.eval()
    return model


@torch.no_grad()
def collect_outputs(model, records, *, device, batch_size: int, precision: str) -> dict:
    results = {}
    for start in range(0, len(records), batch_size):
        current = records[start : start + batch_size]
        batch = p2h.collate(current, device)
        context = (
            _autocast_context(device, "bf16")
            if precision == "bf16" else nullcontext()
        )
        with context:
            output = model(p2h.model_candidates(batch))
        for index, record in enumerate(current):
            results[str(record["sample_key"])] = {
                "existence_logits": output["existence_logits"][index].float().cpu(),
                "pointer_logits": output["pointer_logits"][index].float().cpu(),
                "count": int(output["count_logits"][index].argmax()),
            }
    return results


def edge_scores(output: dict, formula: str) -> torch.Tensor:
    pointer = output["pointer_logits"].log_softmax(dim=-1)
    if formula == "legacy_logsigmoid":
        existence = F.logsigmoid(output["existence_logits"])
    elif formula == "fixed_k_log_odds":
        existence = output["existence_logits"]
    else:
        raise ValueError(f"Unknown P3-E0 edge formula: {formula}")
    return existence[:, None] + pointer


def perturb(scores: torch.Tensor) -> torch.Tensor:
    index = torch.arange(scores.numel(), dtype=torch.float64).reshape(scores.shape)
    noise = torch.sin(index * 0.7548776662466927).float() * PERTURBATION
    return scores + noise


def evaluate_assignments(records, outputs, *, formula: str) -> tuple[dict, dict]:
    per_sample = {}
    latencies = []
    gaps = []
    valid = 0
    ownerships = []
    digest = hashlib.sha256()
    for record in records:
        key = str(record["sample_key"])
        target = record["atom_anchors"][record["full_atom_mask"]].float()
        candidates = record["candidate_anchors"].float()
        compatibility = g3_gt_candidate_compatibility(target, candidates)
        scores = edge_scores(outputs[key], formula)
        started = time.perf_counter()
        latent = g3_latent_positive_assignment(
            scores, compatibility, max_ownerships=MAX_OWNERSHIPS
        )
        latencies.append((time.perf_counter() - started) * 1_000.0)
        unrestricted = g3_unrestricted_fixed_cardinality_assignment(
            scores, len(target)
        )
        is_valid = g3_candidate_set_covers_gt(
            unrestricted["candidate_indices"], compatibility
        )
        valid += int(is_valid)
        gap = max(0.0, (unrestricted["score"] - latent["score"]) / len(target))
        gaps.append(gap)
        ownerships.append(int(latent["ownership_combinations"]))
        latent_signature = assignment_signature(latent)
        unrestricted_signature = assignment_signature(unrestricted)
        perturbed_latent = g3_latent_positive_assignment(
            perturb(scores), compatibility, max_ownerships=MAX_OWNERSHIPS
        )
        perturbed_unrestricted = g3_unrestricted_fixed_cardinality_assignment(
            perturb(scores), len(target)
        )
        per_sample[key] = {
            "latent": latent_signature,
            "unrestricted": unrestricted_signature,
            "unrestricted_valid": is_valid,
            "normalized_valid_gap": gap,
            "latent_perturbation": compare_signatures(
                latent_signature, assignment_signature(perturbed_latent)
            ),
            "unrestricted_perturbation": compare_signatures(
                unrestricted_signature, assignment_signature(perturbed_unrestricted)
            ),
        }
        digest.update(
            f"{key}:{latent_signature['edges']}:{latent['score']:.12f}\n".encode()
        )
    summary = {
        "images": len(records),
        "unrestricted_valid_images": valid,
        "unrestricted_valid_rate": valid / len(records),
        "normalized_valid_gap_mean": float(np.mean(gaps)),
        "normalized_valid_gap_p95": percentile(gaps, 95),
        "normalized_valid_gap_max": max(gaps),
        "solver_latency_ms_p50": percentile(latencies, 50),
        "solver_latency_ms_p95": percentile(latencies, 95),
        "solver_latency_ms_max": max(latencies),
        "ownership_combinations_max": max(ownerships),
        "latent_perturbation_query_exact": sum(
            value["latent_perturbation"]["query_exact"] for value in per_sample.values()
        ),
        "latent_perturbation_candidate_exact": sum(
            value["latent_perturbation"]["candidate_exact"] for value in per_sample.values()
        ),
        "unrestricted_perturbation_candidate_exact": sum(
            value["unrestricted_perturbation"]["candidate_exact"]
            for value in per_sample.values()
        ),
        "latent_digest": digest.hexdigest(),
    }
    return per_sample, summary


def compare_runs(reference: dict, current: dict) -> dict:
    latent = []
    unrestricted = []
    for key in reference:
        latent.append(compare_signatures(
            reference[key]["latent"], current[key]["latent"]
        ))
        unrestricted.append(compare_signatures(
            reference[key]["unrestricted"], current[key]["unrestricted"]
        ))
    return {
        "images": len(reference),
        "latent_query_exact": sum(value["query_exact"] for value in latent),
        "latent_candidate_exact": sum(value["candidate_exact"] for value in latent),
        "latent_edge_exact": sum(value["edge_exact"] for value in latent),
        "latent_query_overlap_mean": float(np.mean([
            value["query_overlap"] for value in latent
        ])),
        "latent_candidate_overlap_mean": float(np.mean([
            value["candidate_overlap"] for value in latent
        ])),
        "unrestricted_query_exact": sum(
            value["query_exact"] for value in unrestricted
        ),
        "unrestricted_candidate_exact": sum(
            value["candidate_exact"] for value in unrestricted
        ),
        "unrestricted_edge_exact": sum(value["edge_exact"] for value in unrestricted),
        "unrestricted_candidate_overlap_mean": float(np.mean([
            value["candidate_overlap"] for value in unrestricted
        ])),
    }


def compatibility_audit(records) -> dict:
    multi_candidates = affected = maximum_multi = maximum_ownerships = 0
    compatible_counts = []
    duplicate_coordinates = 0
    for record in records:
        target = record["atom_anchors"][record["full_atom_mask"]].float()
        candidates = record["candidate_anchors"].float()
        compatibility = g3_gt_candidate_compatibility(target, candidates)
        degree = compatibility.sum(dim=0)
        current_multi = int((degree > 1).sum())
        combinations = math.prod(int(value) for value in degree[degree > 1])
        combinations = max(combinations, 1)
        multi_candidates += current_multi
        affected += int(current_multi > 0)
        maximum_multi = max(maximum_multi, current_multi)
        maximum_ownerships = max(maximum_ownerships, combinations)
        compatible_counts.extend(int(value) for value in compatibility.sum(dim=1))
        duplicate_coordinates += len(candidates) - len(torch.unique(candidates, dim=0))
    return {
        "images": len(records),
        "candidates": len(records) * 128,
        "multi_gt_compatible_candidates": multi_candidates,
        "affected_images": affected,
        "max_multi_gt_candidates_per_image": maximum_multi,
        "max_ownership_combinations": maximum_ownerships,
        "compatible_candidates_per_gt_p50": percentile(compatible_counts, 50),
        "compatible_candidates_per_gt_p95": percentile(compatible_counts, 95),
        "compatible_candidates_per_gt_max": max(compatible_counts),
        "exact_duplicate_candidate_coordinates": duplicate_coordinates,
    }


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E0 output is not empty: {OUTPUT}")
    for name, (path, expected) in CHECKPOINTS.items():
        if file_sha256(path) != expected:
            raise ValueError(f"P3-E0 {name} checkpoint SHA256 differs")
    torch.manual_seed(20260713)
    torch.cuda.manual_seed_all(20260713)
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
    capacity = domains["easy32"] + domains["hard32"]
    capacity_keys = {str(record["sample_key"]) for record in capacity}
    general = sorted(
        (
            record for record in eligible
            if str(record["sample_key"]) not in capacity_keys
        ),
        key=lambda record: str(record["sample_key"]),
    )[:GENERAL_IMAGES]
    records = capacity + general
    if len(records) != 64 + GENERAL_IMAGES:
        raise RuntimeError("P3-E0 audit sample count changed")

    compatibility = compatibility_audit(eligible)
    models = {}
    outputs = {}
    for name, (path, _) in CHECKPOINTS.items():
        models[name] = load_model(path, device)
        outputs[name] = {
            "bf16_batch4": collect_outputs(
                models[name], records, device=device, batch_size=4, precision="bf16"
            ),
            "fp32_batch4": collect_outputs(
                models[name], records, device=device, batch_size=4, precision="fp32"
            ),
            "bf16_batch32": collect_outputs(
                models[name], records, device=device, batch_size=32, precision="bf16"
            ),
        }

    results = {}
    for model_name in CHECKPOINTS:
        results[model_name] = {}
        for formula in ("legacy_logsigmoid", "fixed_k_log_odds"):
            runs = {}
            summaries = {}
            for mode in ("bf16_batch4", "fp32_batch4", "bf16_batch32"):
                runs[mode], summaries[mode] = evaluate_assignments(
                    records, outputs[model_name][mode], formula=formula
                )
            repeated, repeated_summary = evaluate_assignments(
                records, outputs[model_name]["bf16_batch4"], formula=formula
            )
            results[model_name][formula] = {
                "runs": summaries,
                "bf16_batch4_repeat_digest_equal": (
                    summaries["bf16_batch4"]["latent_digest"]
                    == repeated_summary["latent_digest"]
                ),
                "bf16_vs_fp32": compare_runs(
                    runs["bf16_batch4"], runs["fp32_batch4"]
                ),
                "batch4_vs_batch32": compare_runs(
                    runs["bf16_batch4"], runs["bf16_batch32"]
                ),
            }
        legacy, _ = evaluate_assignments(
            records, outputs[model_name]["bf16_batch4"], formula="legacy_logsigmoid"
        )
        log_odds, _ = evaluate_assignments(
            records, outputs[model_name]["bf16_batch4"], formula="fixed_k_log_odds"
        )
        results[model_name]["formula_comparison"] = compare_runs(legacy, log_odds)

    checks = {
        "ownership_combinations_max256": (
            compatibility["max_ownership_combinations"] <= MAX_OWNERSHIPS
        ),
        "all_repeat_digests_equal": all(
            results[model][formula]["bf16_batch4_repeat_digest_equal"]
            for model in CHECKPOINTS
            for formula in ("legacy_logsigmoid", "fixed_k_log_odds")
        ),
        "solver_p95_under100ms": all(
            results[model][formula]["runs"]["bf16_batch4"][
                "solver_latency_ms_p95"
            ] <= 100.0
            for model in CHECKPOINTS
            for formula in ("legacy_logsigmoid", "fixed_k_log_odds")
        ),
    }
    payload = {
        "schema_version": "g3-p3e0-latent-solver-audit-0.1",
        "audit_name": AUDIT_NAME,
        "decision": (
            "latent_solver_preflight_go" if all(checks.values())
            else "latent_solver_preflight_no_go"
        ),
        "checks": checks,
        "scope": {
            "capacity_images": 64,
            "general_images": GENERAL_IMAGES,
            "general_selection": "first eligible non-capacity sample keys",
            "training_updates": 0,
            "perturbation": PERTURBATION,
        },
        "compatibility_train4937": compatibility,
        "results": results,
        "lineage": {
            "checkpoints": {
                name: {"path": str(path.resolve()), "sha256": expected}
                for name, (path, expected) in CHECKPOINTS.items()
            },
            "candidate_cache_sha256": p2e.CACHE_SHA256,
            "candidate_cache_tensor_sha256": cache_audit["candidate_protocol"][
                "candidate_cache_sha256"
            ],
            "capacity_selection": selection_audit,
            "sample_keys": [str(record["sample_key"]) for record in records],
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "metrics.json", payload)
    print(json.dumps({
        "decision": payload["decision"],
        "checks": checks,
        "compatibility": compatibility,
        "p2w": results["p2w"],
    }, indent=2))


if __name__ == "__main__":
    main()
