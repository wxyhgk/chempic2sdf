#!/usr/bin/env python3
"""Audit global two-endpoint Bond consistency as a top192 Atom selector signal."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from chemvector.evaluation.audits.atoms import (
    audit_g3_p3e29_bond_incident_signal as p3e29,
)
from chemvector.training.atoms import (
    train_g3_p1a_canonical_atom_selector as p1a,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p3e30-bond-pair-consistency-micro160"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
P3E29_AUDIT_SHA256 = (
    "dff767d8a61308e566c01c67a7ae96906b3bc3e2bb854337fda88be7b172fc23"
)
P3E29_EVIDENCE_SHA256 = (
    "003a6fc0384a6c5a5a16cf5be8387d10b5ee8a4cd6957792a00623b289265c29"
)
SCALE_RATIOS = (0.08, 0.16, 0.32)
ALPHAS = (0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0)
REFERENCE_BONDS = 64
MIN_TAIL_RESCUE_IMAGES = 5
MAX_REACHABLE_RECALL_REGRESSION = 0.005


def adaptive_pair_affinity(
    candidate_anchors: torch.Tensor,
    bond_scores: torch.Tensor,
    bond_endpoints: torch.Tensor,
    bond_mask: torch.Tensor,
    *,
    scale_ratio: float,
    reference_bonds: int = REFERENCE_BONDS,
) -> tuple[torch.Tensor, float]:
    """Build a symmetric candidate-pair affinity from both endpoints of each Bond."""

    if candidate_anchors.ndim != 2 or candidate_anchors.size(1) != 2:
        raise ValueError("P3-E30 candidate anchors must have shape [candidate,2]")
    if bond_endpoints.ndim != 3 or bond_endpoints.shape[1:] != (2, 2):
        raise ValueError("P3-E30 Bond endpoints must have shape [bond,2,2]")
    bonds = len(bond_endpoints)
    if bond_scores.shape != (bonds,) or bond_mask.shape != (bonds,):
        raise ValueError("P3-E30 Bond score/mask shape differs")
    if scale_ratio <= 0.0 or reference_bonds < 1:
        raise ValueError("P3-E30 adaptive scale protocol is invalid")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (candidate_anchors, bond_scores, bond_endpoints)
    ):
        raise ValueError("P3-E30 pair input contains a non-finite value")

    valid = bond_mask.bool()
    candidate_count = len(candidate_anchors)
    if not bool(valid.any()):
        return candidate_anchors.new_zeros((candidate_count, candidate_count)), 0.0
    scores = bond_scores[valid].float().clamp(0.0, 1.0)
    endpoints = bond_endpoints[valid].float()
    lengths = torch.linalg.vector_norm(endpoints[:, 0] - endpoints[:, 1], dim=-1)
    top = torch.argsort(scores, descending=True, stable=True)[: min(reference_bonds, len(scores))]
    reference_length = lengths[top].median()
    sigma = (reference_length * scale_ratio).clamp_min(torch.finfo(torch.float32).eps)

    distance0 = torch.cdist(endpoints[:, 0], candidate_anchors.float())
    distance1 = torch.cdist(endpoints[:, 1], candidate_anchors.float())
    kernel0 = torch.exp(-0.5 * (distance0 / sigma).square())
    kernel1 = torch.exp(-0.5 * (distance1 / sigma).square())
    oriented = kernel0.transpose(0, 1) @ (scores[:, None] * kernel1)
    affinity = oriented + oriented.transpose(0, 1)
    affinity.fill_diagonal_(0.0)
    normalizer = affinity.amax()
    if float(normalizer) > 0.0:
        affinity = affinity / normalizer
    return affinity, float(sigma)


def _standardize(values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    values = values.float()
    selected = values if mask is None else values[mask]
    std = selected.std(unbiased=False)
    result = torch.zeros_like(values)
    if float(std) >= 1e-8:
        result = (values - selected.mean()) / std
    return result


def greedy_pair_selection(
    raw_scores: torch.Tensor,
    affinity: torch.Tensor,
    count: int,
    *,
    alpha: float,
) -> torch.Tensor:
    if raw_scores.ndim != 1 or affinity.shape != (len(raw_scores), len(raw_scores)):
        raise ValueError("P3-E30 greedy input shape differs")
    if not 0 <= count <= len(raw_scores) or alpha < 0.0:
        raise ValueError("P3-E30 greedy Count/alpha is invalid")
    chosen = torch.zeros(len(raw_scores), dtype=torch.bool)
    raw = _standardize(raw_scores)
    selected = []
    for _ in range(count):
        available = ~chosen
        if selected:
            structural = affinity[:, selected].sum(dim=1)
        else:
            structural = affinity.sum(dim=1)
        score = raw + alpha * _standardize(structural, available)
        score[chosen] = -torch.inf
        index = int(torch.argmax(score))
        chosen[index] = True
        selected.append(index)
    return torch.tensor(selected, dtype=torch.long)


def structured_selection_metrics(
    records: list[dict],
    affinities: dict[str, torch.Tensor],
    *,
    alpha: float,
) -> dict:
    totals = defaultdict(int)
    for record in records:
        key = str(record["sample_key"])
        expected = record["atom_anchors"][record["full_atom_mask"]].float()
        count = len(expected)
        selected = greedy_pair_selection(
            record["raw_scores"], affinities[key], count, alpha=alpha
        )
        matched = p1a.maximum_cardinality_radius_match_count(
            record["candidate_anchors"][selected],
            expected,
            radius_pixels=6.0,
            image_size=p3e29.IMAGE_SIZE,
        )
        old_reachable = (
            p1a.maximum_cardinality_radius_match_count(
                record["candidate_anchors"][:128],
                expected,
                radius_pixels=6.0,
                image_size=p3e29.IMAGE_SIZE,
            )
            == count
        )
        exact = matched == count
        group = "reachable" if old_reachable else "tail"
        totals["images"] += 1
        totals["matched_atoms"] += matched
        totals["target_atoms"] += count
        totals["exact_images"] += int(exact)
        totals[f"{group}_images"] += 1
        totals[f"{group}_matched_atoms"] += matched
        totals[f"{group}_target_atoms"] += count
        totals[f"{group}_exact_images"] += int(exact)
    result = dict(totals)
    for group in ("", "reachable_", "tail_"):
        result[f"{group}recall"] = (
            result[f"{group}matched_atoms"] / result[f"{group}target_atoms"]
        )
    return result


def choose_oracle_result(results: dict[str, dict]) -> tuple[str, dict]:
    return max(
        results.items(),
        key=lambda item: (
            item[1]["tail_exact_images"],
            item[1]["exact_images"],
            item[1]["matched_atoms"],
            -float(item[0].split("alpha=")[1]),
        ),
    )


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E30 output is not empty: {OUTPUT}")
    if file_sha256(p3e29.OUTPUT / "audit.json") != P3E29_AUDIT_SHA256:
        raise ValueError("P3-E30 P3-E29 audit SHA256 differs")
    if file_sha256(p3e29.EVIDENCE_CACHE) != P3E29_EVIDENCE_SHA256:
        raise ValueError("P3-E30 P3-E29 evidence SHA256 differs")
    started = time.perf_counter()
    records, input_audit, _ = p3e29._load_micro160()
    payload = torch.load(p3e29.EVIDENCE_CACHE, map_location="cpu", weights_only=False)
    cached = {str(value["sample_key"]): value for value in payload["records"]}
    results = {}
    scales = defaultdict(list)
    for ratio in SCALE_RATIOS:
        affinities = {}
        for record in records:
            key = str(record["sample_key"])
            value = cached[key]
            affinity, scale = adaptive_pair_affinity(
                record["candidate_anchors"],
                value["bond_scores"],
                value["bond_endpoints"],
                value["bond_mask"],
                scale_ratio=ratio,
            )
            affinities[key] = affinity
            scales[f"ratio={ratio:g}"].append(scale)
        for alpha in ALPHAS:
            name = f"ratio={ratio:g},alpha={alpha:g}"
            results[name] = structured_selection_metrics(
                records, affinities, alpha=alpha
            )
    winner_name, winner = choose_oracle_result(results)
    baseline = next(value for name, value in results.items() if name.endswith("alpha=0"))
    checks = {
        "tail_rescue_min5_of27": winner["tail_exact_images"] >= MIN_TAIL_RESCUE_IMAGES,
        "tail_improves_over_raw": winner["tail_exact_images"] > baseline["tail_exact_images"],
        "reachable_recall_regression_max005": (
            winner["reachable_recall"]
            >= baseline["reachable_recall"] - MAX_REACHABLE_RECALL_REGRESSION
        ),
        "all_27_tail_images_present": winner["tail_images"] == 27,
        "all_160_images_present": winner["images"] == len(cached) == len(records) == 160,
    }
    audit = {
        "schema_version": "g3-p3e30-bond-pair-consistency-audit-0.1",
        "decision": (
            "bond_pair_consistency_signal_go"
            if all(checks.values())
            else "bond_pair_consistency_signal_no_go"
        ),
        "checks": checks,
        "oracle_result": {"name": winner_name, "metrics": winner},
        "raw_baseline": baseline,
        "results": results,
        "adaptive_scales": {
            name: {
                "mean_normalized": float(np.mean(values)),
                "minimum_normalized": float(np.min(values)),
                "maximum_normalized": float(np.max(values)),
                "mean_pixels": float(np.mean(values) * p3e29.IMAGE_SIZE),
            }
            for name, values in scales.items()
        },
        "protocol": {
            "candidate_count": 192,
            "bond_proposal_count": 192,
            "scale_ratios": list(SCALE_RATIOS),
            "alphas": list(ALPHAS),
            "reference_bonds": REFERENCE_BONDS,
            "adaptive_scale": "ratio * median length of top64 scored predicted Bonds",
            "pair_affinity": "sum_b score_b * symmetric endpoint0/endpoint1 kernels",
            "selector": "known-Count greedy raw-score plus selected-pair affinity",
            "oracle_is_signal_upper_bound_not_deployed_selector": True,
            "tail_rescue_gate": MIN_TAIL_RESCUE_IMAGES,
            "maximum_reachable_recall_regression": MAX_REACHABLE_RECALL_REGRESSION,
        },
        "input": input_audit,
        "lineage": {
            "p3e29_audit": str((p3e29.OUTPUT / "audit.json").resolve()),
            "p3e29_audit_sha256": P3E29_AUDIT_SHA256,
            "p3e29_evidence": str(p3e29.EVIDENCE_CACHE.resolve()),
            "p3e29_evidence_sha256": P3E29_EVIDENCE_SHA256,
        },
        "data_access": {
            "cached_prediction_only": True,
            "png_read": False,
            "gt_atom_read_for_audit_only": True,
            "gt_bond_read": False,
            "render_json_read": False,
            "bond_layer_read": False,
            "dev_read": False,
            "old_confirm_read": False,
            "fresh_dev_read": False,
            "uspto_read": False,
            "mr_tadf_read": False,
            "sealed_test_read": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
