#!/usr/bin/env python3
"""Audit target-free radius clustering of frozen P1-A Atom candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scipy.optimize import linear_sum_assignment
import torch

from chemvector.components.atoms.g3_top128_set_ranker import extract_frozen_top128
from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.atoms import train_g3_p1d_boundary_band_set_ranker as p1d
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import (
    _autocast_context,
    load_frozen_vision_checkpoint,
)


DEFAULT_OUTPUT = Path(
    "outputs/audits/chemvector-g3-p1i-target-free-candidate-clustering"
)
THRESHOLDS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
MATCH_RADIUS_PIXELS = 6.0


def cluster_labels(distance_pixels: torch.Tensor, threshold: float) -> torch.Tensor:
    """Return deterministic connected-component labels for one distance matrix."""

    if distance_pixels.ndim != 2 or distance_pixels.shape[0] != distance_pixels.shape[1]:
        raise ValueError("P1-I candidate distance matrix must be square")
    count = len(distance_pixels)
    if threshold < 0:
        raise ValueError("P1-I clustering threshold must be non-negative")
    if threshold == 0:
        return torch.arange(count, dtype=torch.long)
    parent = list(range(count))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    rows, columns = torch.where(
        torch.triu(distance_pixels <= threshold, diagonal=1)
    )
    for left, right in zip(rows.tolist(), columns.tolist()):
        union(left, right)
    roots = [find(index) for index in range(count)]
    ordered = {root: label for label, root in enumerate(sorted(set(roots)))}
    return torch.tensor([ordered[root] for root in roots], dtype=torch.long)


def sample_cluster_statistics(
    labels: torch.Tensor,
    candidate_target_compatible: torch.Tensor,
) -> dict[str, int]:
    """Score target-free components against GT after clustering is complete."""

    if labels.ndim != 1 or candidate_target_compatible.ndim != 2:
        raise ValueError("P1-I cluster statistic inputs have invalid rank")
    if len(labels) != candidate_target_compatible.shape[0]:
        raise ValueError("P1-I labels and compatibility differ")
    component_count = int(labels.max()) + 1
    target_count = candidate_target_compatible.shape[1]
    component_compatible = torch.zeros(
        component_count, target_count, dtype=torch.bool
    )
    component_sizes = torch.bincount(labels, minlength=component_count)
    for component in range(component_count):
        component_compatible[component] = candidate_target_compatible[
            labels == component
        ].any(dim=0)
    costs = (~component_compatible.T).to(torch.int64)
    rows, columns = linear_sum_assignment(costs.numpy())
    matched = sum(
        bool(component_compatible[column, row])
        for row, column in zip(rows.tolist(), columns.tolist())
    )
    target_multiplicity = component_compatible.sum(dim=0)
    component_target_counts = component_compatible.sum(dim=1)
    return {
        "images": 1,
        "targets": target_count,
        "matched_targets": matched,
        "exact_images": int(matched == target_count),
        "components": component_count,
        "collapsed_candidates": len(labels) - component_count,
        "single_component_targets": int((target_multiplicity == 1).sum()),
        "multiple_component_targets": int((target_multiplicity >= 2).sum()),
        "zero_component_targets": int((target_multiplicity == 0).sum()),
        "ambiguous_components": int((component_target_counts >= 2).sum()),
        "pure_compatible_components": int((component_target_counts == 1).sum()),
        "largest_component": int(component_sizes.max()),
    }


def _empty_totals():
    return {
        "images": 0,
        "targets": 0,
        "matched_targets": 0,
        "exact_images": 0,
        "components": 0,
        "collapsed_candidates": 0,
        "single_component_targets": 0,
        "multiple_component_targets": 0,
        "zero_component_targets": 0,
        "ambiguous_components": 0,
        "pure_compatible_components": 0,
        "largest_component": 0,
    }


def _merge(totals, current):
    for key, value in current.items():
        if key == "largest_component":
            totals[key] = max(totals[key], value)
        else:
            totals[key] += value


def _finalize(totals):
    return {
        **totals,
        "atom_recall": totals["matched_targets"] / totals["targets"],
        "exact_rate": totals["exact_images"] / totals["images"],
        "single_component_target_rate": totals["single_component_targets"]
        / totals["targets"],
        "multiple_component_target_rate": totals["multiple_component_targets"]
        / totals["targets"],
        "mean_components_per_image": totals["components"] / totals["images"],
        "mean_collapsed_candidates_per_image": totals["collapsed_candidates"]
        / totals["images"],
    }


@torch.no_grad()
def evaluate_manifest(atom_head, vision, manifest, collator, *, device):
    totals = {threshold: _empty_totals() for threshold in THRESHOLDS}
    sample_digest = hashlib.sha256()
    for batch in p1a.iter_manifest_batches(manifest, collator, batch_size=4):
        for key in batch["keys"]:
            sample_digest.update(f"{key}\n".encode())
        pixels = batch["pixel_values"].to(device)
        targets = {
            "atom_anchors": batch["atom_anchors"].to(device),
            "atom_mask": batch["atom_mask"].to(device),
        }
        with _autocast_context(device, "bf16"):
            features = vision.forward_stroke_resolution(pixels)
            dense = atom_head(features.detach())
            candidates = extract_frozen_top128(dense)
        fixed = p1d.build_hybrid_top128_targets(candidates, targets)
        candidate_distances = (
            torch.cdist(
                candidates["candidate_anchors"].float(),
                candidates["candidate_anchors"].float(),
            )
            * 768.0
        ).cpu()
        for index in range(len(batch["keys"])):
            fixed_mask = fixed["positive_mask"][index]
            target_indices = fixed["matched_target_indices"][index, fixed_mask]
            expected = targets["atom_anchors"][index, target_indices].float()
            compatible = (
                torch.cdist(
                    candidates["candidate_anchors"][index].float(), expected
                )
                * 768.0
                <= MATCH_RADIUS_PIXELS
            ).cpu()
            for threshold in THRESHOLDS:
                labels = cluster_labels(candidate_distances[index], threshold)
                current = sample_cluster_statistics(labels, compatible)
                _merge(totals[threshold], current)
    return {
        "images": len(manifest),
        "thresholds": {
            str(threshold): _finalize(totals[threshold])
            for threshold in THRESHOLDS
        },
        "sample_keys_sha256": sample_digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P1-I requires CUDA BF16 frozen candidate replay")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError("P1-I output directory is not empty")
    device = torch.device("cuda")
    atom_head, _, p1a_payload, _ = p1d.load_p1a_boundary_ranker(device)
    vision, _, _, vision_sha = load_frozen_vision_checkpoint(
        Path(p1a_payload["lineage"]["vision"]["path"]), device
    )
    vision.requires_grad_(False).eval()
    train, train_audit = p1a.load_train_manifest()
    route, route_audit = p1a.load_route_manifest()
    collator = p1a.G3AtomCollator(maximum_atoms=128, image_size=768)
    train_result = evaluate_manifest(
        atom_head, vision, train, collator, device=device
    )
    route_result = evaluate_manifest(
        atom_head, vision, route, collator, device=device
    )
    passing = []
    for threshold in THRESHOLDS[1:]:
        train_value = train_result["thresholds"][str(threshold)]
        route_value = route_result["thresholds"][str(threshold)]
        if (
            train_value["atom_recall"] >= 0.999
            and train_value["exact_rate"] >= 0.99
            and route_value["atom_recall"] >= 0.999
            and route_value["exact_rate"] >= 0.99
            and train_value["multiple_component_target_rate"] <= 0.25
        ):
            passing.append(threshold)
    decision = (
        "target_free_radius_clustering_go"
        if passing
        else "target_free_radius_clustering_no_go"
    )
    report = {
        "schema_version": "g3-p1i-target-free-clustering-audit-0.1",
        "scope": "frozen-p1a-top128-radius-components",
        "threshold_pixels": list(THRESHOLDS),
        "match_radius_pixels": MATCH_RADIUS_PIXELS,
        "train_manifest_sha256": train_audit["sha256"],
        "route_manifest_sha256": route_audit["sha256"],
        "vision_checkpoint_sha256": vision_sha,
        "train5k": train_result,
        "route100": route_result,
        "passing_thresholds": passing,
        "selected_threshold": passing[0] if passing else None,
        "updates": 0,
        "forbidden_data_read": False,
        "decision": decision,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"decision": decision, "passing_thresholds": passing}))


if __name__ == "__main__":
    main()
