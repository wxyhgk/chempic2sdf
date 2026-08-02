#!/usr/bin/env python3
"""Audit target-free clustering of continuous-oracle decoded carrier lines.

GT association is used to construct the continuous extent oracle inputs.  The
clustering function itself receives no target IDs.  Target lines and IDs are
read only after the predicted partition and medoids are frozen, for metrics.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from chemvector.audit_evidence_carrier_extent_oracle import (
    reconstruct_continuous_target_segments,
    unordered_endpoint_error_q,
)
from chemvector.audit_evidence_carrier_proposals import (
    extract_targets,
    file_sha256,
)
from chemvector.evidence_carrier_decoded_line_clustering import (
    DecodedLineClusteringConfig,
    cluster_decoded_lines,
)
from chemvector.evidence_carrier_extent_data import (
    IN_SCOPE_GROUP_KINDS,
    EvidenceCarrierExtentTargetConfig,
    EvidenceCarrierExtentWebDataset,
    build_evidence_carrier_extent_target,
)


SCHEMA_VERSION = "evidence-carrier-decoded-line-clustering-oracle-0.1"
DEFAULT_VALIDATION_SHARD = (
    "Data/sdf/visual_svg_v0_1/pilot_10k/shards/validation-000000.tar"
)
DEFAULT_OUTPUT = (
    "outputs/evaluations/"
    "chemvector-v099-p05-decoded-line-clustering-oracle-validation100/"
    "audit.json"
)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _pair_counts(
    target_ids: np.ndarray,
    cluster_labels: np.ndarray,
) -> dict[str, int | float | None]:
    if target_ids.ndim != 1 or cluster_labels.shape != target_ids.shape:
        raise ValueError("Pair labels must be aligned one-dimensional arrays")
    true_positive = false_positive = false_negative = 0
    for first in range(len(target_ids)):
        for second in range(first + 1, len(target_ids)):
            expected = int(target_ids[first]) == int(target_ids[second])
            predicted = int(cluster_labels[first]) == int(cluster_labels[second])
            true_positive += int(expected and predicted)
            false_positive += int(not expected and predicted)
            false_negative += int(expected and not predicted)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0.0
        else None
    )
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_frozen_partition(
    *,
    clusters: tuple,
    target_line_ids: np.ndarray,
    decoded_segments_pixels: np.ndarray,
    target_lines: list,
    target_groups: list,
    width: int,
    height: int,
    tolerance_q: float = 8.0,
) -> dict[str, Any]:
    """Use GT only after clusters and medoids have already been frozen."""

    target_line_ids = np.asarray(target_line_ids, dtype=np.int64)
    decoded = np.asarray(decoded_segments_pixels, dtype=np.float64)
    if target_line_ids.ndim != 1 or decoded.shape != (len(target_line_ids), 2, 2):
        raise ValueError("Frozen partition inputs are not aligned")
    cluster_labels = np.full(len(target_line_ids), -1, dtype=np.int64)
    for cluster_index, cluster in enumerate(clusters):
        for member_index in cluster.member_indices:
            if not 0 <= int(member_index) < len(cluster_labels):
                raise ValueError("Cluster member index is out of range")
            if cluster_labels[int(member_index)] >= 0:
                raise ValueError("A decoded line occurs in more than one cluster")
            cluster_labels[int(member_index)] = cluster_index
    if len(cluster_labels) and bool((cluster_labels < 0).any()):
        raise ValueError("Clustering did not partition every decoded line")

    pair_metrics = _pair_counts(target_line_ids, cluster_labels)
    predicted_sets = {
        frozenset(int(index) for index in cluster.member_indices)
        for cluster in clusters
    }
    expected_members: dict[int, set[int]] = defaultdict(set)
    for index, line_id in enumerate(target_line_ids.tolist()):
        expected_members[int(line_id)].add(index)
    expected_sets = {frozenset(values) for values in expected_members.values()}

    cross_target_clusters = 0
    cross_target_pairs = 0
    for cluster in clusters:
        labels = [int(target_line_ids[index]) for index in cluster.member_indices]
        distinct = set(labels)
        if len(distinct) > 1:
            cross_target_clusters += 1
        counts = Counter(labels)
        total_pairs = len(labels) * (len(labels) - 1) // 2
        same_pairs = sum(value * (value - 1) // 2 for value in counts.values())
        cross_target_pairs += total_pairs - same_pairs

    ordinary = {
        line_id: line
        for line_id, line in enumerate(target_lines)
        if line.group_kind in IN_SCOPE_GROUP_KINDS
    }
    representatives = np.asarray(
        [cluster.representative_segment_pixels for cluster in clusters],
        dtype=np.float64,
    ).reshape((-1, 2, 2))

    def matched_count(candidates: np.ndarray) -> int:
        matched = 0
        for line in ordinary.values():
            best = min(
                (
                    unordered_endpoint_error_q(
                        candidate,
                        line.points_q,
                        width=width,
                        height=height,
                    )
                    for candidate in candidates
                ),
                default=math.inf,
            )
            matched += int(best <= tolerance_q)
        return matched

    baseline_matched = matched_count(decoded)
    clustered_matched = matched_count(representatives)
    expected_line_count = len(ordinary)

    clusters_by_target: dict[int, set[int]] = defaultdict(set)
    for item_index, target_id in enumerate(target_line_ids.tolist()):
        clusters_by_target[int(target_id)].add(int(cluster_labels[item_index]))
    p2_total = p2_both_supported = p2_preserved = p2_merged = 0
    positive_ids = set(int(value) for value in target_line_ids.tolist())
    for group in target_groups:
        if group.kind != "parallel_2" or len(group.line_indices) != 2:
            continue
        p2_total += 1
        first, second = map(int, group.line_indices)
        if first not in positive_ids or second not in positive_ids:
            continue
        p2_both_supported += 1
        distinct = clusters_by_target[first].isdisjoint(clusters_by_target[second])
        p2_preserved += int(distinct)
        p2_merged += int(not distinct)

    partition_eligible = len(target_line_ids) > 0
    return {
        "pairs": pair_metrics,
        "cross_target_merge": {
            "clusters": cross_target_clusters,
            "pairs": cross_target_pairs,
            "sample_has_merge": cross_target_clusters > 0,
        },
        "positive_partition_eligible": partition_eligible,
        "positive_partition_exact": (
            partition_eligible and predicted_sets == expected_sets
        ),
        "q8_recall": {
            "expected": expected_line_count,
            "before_clustering_matched": baseline_matched,
            "after_clustering_matched": clustered_matched,
            "before_clustering": _ratio(baseline_matched, expected_line_count),
            "after_clustering": _ratio(clustered_matched, expected_line_count),
            "loss_lines": baseline_matched - clustered_matched,
            "loss_percentage_points": (
                100.0 * (baseline_matched - clustered_matched) / expected_line_count
                if expected_line_count
                else None
            ),
        },
        "parallel_2_distinct_preservation": {
            "total_groups": p2_total,
            "both_members_supported": p2_both_supported,
            "preserved": p2_preserved,
            "merged": p2_merged,
            "rate": _ratio(p2_preserved, p2_both_supported),
        },
    }


def audit_sample(
    sample: dict[str, Any],
    *,
    target_config: EvidenceCarrierExtentTargetConfig,
    cluster_config: DecodedLineClusteringConfig,
) -> dict[str, Any]:
    image = sample["image"]
    width, height = image.size
    target = build_evidence_carrier_extent_target(
        image,
        sample["primitives"],
        sample["layers"],
        target_config=target_config,
    )
    all_decoded = reconstruct_continuous_target_segments(
        target,
        width=width,
        height=height,
    )
    positive_indices = torch.where(target["geometry_loss_mask"])[0].tolist()
    decoded = all_decoded[positive_indices]
    scale = np.asarray((float(width - 1), float(height - 1)))
    all_anchors = (
        target["carrier_segments"].detach().cpu().numpy().astype(np.float64) * scale
    )
    anchors = all_anchors[positive_indices]
    source_ids = tuple(f"lsd-{index:06d}" for index in positive_indices)

    # Target-free boundary: this call receives geometry and stable LSD identity
    # only.  target_line_id is intentionally read below, after clusters freeze.
    clusters = cluster_decoded_lines(
        decoded,
        anchors,
        source_ids,
        width=width,
        height=height,
        config=cluster_config,
    )

    target_lines, target_groups, _ = extract_targets(
        sample["primitives"],
        sample["layers"],
        long_bond_pixels=target_config.long_bond_pixels,
    )
    target_line_ids = (
        target["target_line_id"][positive_indices].detach().cpu().numpy()
    )
    evaluation = evaluate_frozen_partition(
        clusters=clusters,
        target_line_ids=target_line_ids,
        decoded_segments_pixels=decoded,
        target_lines=target_lines,
        target_groups=target_groups,
        width=width,
        height=height,
    )
    return {
        "sample_key": str(sample["key"]),
        "positive_decoded_lines": len(decoded),
        "predicted_clusters": len(clusters),
        "reduction": len(decoded) - len(clusters),
        **evaluation,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    pair_counts = Counter()
    cross_clusters = cross_pairs = cross_samples = 0
    partition_exact = partition_eligible = partition_excluded_empty = 0
    q8_expected = q8_before = q8_after = 0
    p2_total = p2_supported = p2_preserved = p2_merged = 0
    input_lines = output_clusters = 0
    for record in records:
        pairs = record["pairs"]
        for key in ("true_positive", "false_positive", "false_negative"):
            pair_counts[key] += int(pairs[key])
        cross = record["cross_target_merge"]
        cross_clusters += int(cross["clusters"])
        cross_pairs += int(cross["pairs"])
        cross_samples += int(cross["sample_has_merge"])
        eligible = bool(record["positive_partition_eligible"])
        partition_eligible += int(eligible)
        partition_excluded_empty += int(not eligible)
        partition_exact += int(record["positive_partition_exact"])
        recall = record["q8_recall"]
        q8_expected += int(recall["expected"])
        q8_before += int(recall["before_clustering_matched"])
        q8_after += int(recall["after_clustering_matched"])
        p2 = record["parallel_2_distinct_preservation"]
        p2_total += int(p2["total_groups"])
        p2_supported += int(p2["both_members_supported"])
        p2_preserved += int(p2["preserved"])
        p2_merged += int(p2["merged"])
        input_lines += int(record["positive_decoded_lines"])
        output_clusters += int(record["predicted_clusters"])

    precision = _ratio(
        pair_counts["true_positive"],
        pair_counts["true_positive"] + pair_counts["false_positive"],
    )
    recall = _ratio(
        pair_counts["true_positive"],
        pair_counts["true_positive"] + pair_counts["false_negative"],
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0.0
        else None
    )
    return {
        "samples": len(records),
        "decoded_lines": input_lines,
        "clusters": output_clusters,
        "reduction": input_lines - output_clusters,
        "pairs": {
            **dict(pair_counts),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "cross_target_merge": {
            "clusters": cross_clusters,
            "pairs": cross_pairs,
            "samples": cross_samples,
        },
        "positive_partition": {
            "exact_samples": partition_exact,
            "eligible_samples": partition_eligible,
            "excluded_empty_positive_samples": partition_excluded_empty,
            "total_samples": len(records),
            "exact_rate": _ratio(partition_exact, partition_eligible),
        },
        "q8_recall": {
            "expected": q8_expected,
            "before_clustering_matched": q8_before,
            "after_clustering_matched": q8_after,
            "before_clustering": _ratio(q8_before, q8_expected),
            "after_clustering": _ratio(q8_after, q8_expected),
            "loss_lines": q8_before - q8_after,
            "loss_percentage_points": (
                100.0 * (q8_before - q8_after) / q8_expected
                if q8_expected
                else None
            ),
        },
        "parallel_2_distinct_preservation": {
            "total_groups": p2_total,
            "both_members_supported": p2_supported,
            "preserved": p2_preserved,
            "merged": p2_merged,
            "rate": _ratio(p2_preserved, p2_supported),
        },
    }


def gate_decision(summary: dict[str, Any]) -> dict[str, Any]:
    """Apply predeclared P0.5 feasibility gates."""

    checks = {
        "pair_f1_at_least_0_99": (
            summary["pairs"]["f1"] is not None
            and summary["pairs"]["f1"] >= 0.99
        ),
        "cross_target_merge_clusters_equal_0": (
            summary["cross_target_merge"]["clusters"] == 0
        ),
        "q8_recall_loss_at_most_0_5pp": (
            summary["q8_recall"]["loss_percentage_points"] is not None
            and summary["q8_recall"]["loss_percentage_points"] <= 0.5
        ),
        "positive_partition_exact_at_least_0_95": (
            summary["positive_partition"]["exact_rate"] is not None
            and summary["positive_partition"]["exact_rate"] >= 0.95
        ),
        "parallel_2_distinct_preservation_equal_1": (
            summary["parallel_2_distinct_preservation"]["rate"] is not None
            and summary["parallel_2_distinct_preservation"]["rate"] == 1.0
        ),
    }
    return {
        "status": "GO" if all(checks.values()) else "NO-GO",
        "checks": checks,
        "thresholds": {
            "pair_f1": 0.99,
            "cross_target_merge_clusters": 0,
            "maximum_q8_recall_loss_percentage_points": 0.5,
            "positive_partition_exact_rate": 0.95,
            "parallel_2_distinct_preservation": 1.0,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", default=DEFAULT_VALIDATION_SHARD)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")
    target_config = EvidenceCarrierExtentTargetConfig()
    cluster_config = DecodedLineClusteringConfig()
    records = [
        audit_sample(
            sample,
            target_config=target_config,
            cluster_config=cluster_config,
        )
        for sample in EvidenceCarrierExtentWebDataset(args.shard, limit=args.limit)
    ]
    if not records:
        raise RuntimeError("Decoded-line clustering audit produced no samples")
    summary = summarize_records(records)
    decision = gate_decision(summary)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "shard": str(Path(args.shard).resolve()),
        "shard_sha256": file_sha256(args.shard),
        "samples": len(records),
        "scope": "positive ordinary continuous-oracle decoded carrier lines",
        "uses_internal_test": False,
        "learned_model_used": False,
        "target_geometry_used_to_construct_decoded_oracle_lines": True,
        "target_line_id_visible_to_clustering": False,
        "gt_used_only_after_partition_and_medoids_frozen": True,
        "proposal_generator": {
            "name": "OpenCV LineSegmentDetector",
            "opencv_version": cv2.__version__,
            "refine": "LSD_REFINE_STD",
        },
        "algorithm": {
            "name": "deterministic complete-link",
            "representative": "observed-member medoid",
            "coordinate_averaging": False,
            "input_order_tie_break": (
                "canonical decoded geometry, canonical anchor geometry, "
                "stable LSD source ID"
            ),
        },
        "target_config": asdict(target_config),
        "cluster_config": asdict(cluster_config),
        "summary": summary,
        "gate": decision,
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary, "gate": decision}, indent=2))


if __name__ == "__main__":
    main()
