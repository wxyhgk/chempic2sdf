"""CPU feasibility audit for a line-first, endpoint-free graph traversal.

Unlike the legacy line rescue path, this route does *not* attach line
endpoints to frozen/predicted atom anchors.  It clusters the endpoints of raw
LSD segments into provisional graph nodes, then emits an unordered edge for
each segment between two clusters.  The graph can subsequently be traversed
and fused with text/atom-role proposals.  Ground-truth centers are consulted
only after proposal construction to report reachability; they never affect
clustering or edge generation.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from chemvector.components.vision.line_segment_rescue import (
    LineSegmentRescueConfig,
    detect_raster_line_segments,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUTPUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-graph-traversal-r5"


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def cluster_endpoints(points: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Single-link endpoint clustering, deterministic by endpoint order."""
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("points must have shape [N,2]")
    dsu = _DSU(len(points))
    for i in range(len(points)):
        distance = np.linalg.norm(points[i + 1 :] - points[i], axis=-1)
        for offset in np.where(distance <= radius)[0].tolist():
            dsu.union(i, i + 1 + int(offset))
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(points)):
        groups[dsu.find(i)].append(i)
    roots = sorted(groups, key=lambda root: min(groups[root]))
    labels = np.full(len(points), -1, dtype=np.int64)
    centers = []
    for label, root in enumerate(roots):
        members = groups[root]
        labels[members] = label
        centers.append(points[members].mean(axis=0))
    return labels, np.asarray(centers, dtype=np.float64)


def _nearest_mapping(centers: np.ndarray, targets: np.ndarray, radius: float) -> dict[int, int]:
    if len(centers) == 0 or len(targets) == 0:
        return {}
    distances = np.linalg.norm(centers[:, None] - targets[None, :], axis=-1)
    # A provisional line-first node is allowed to map to at most one target;
    # this is only for post-hoc scoring and does not suppress proposals.
    mapping = {}
    for row in range(len(centers)):
        col = int(distances[row].argmin())
        if float(distances[row, col]) <= radius:
            mapping[row] = col
    return mapping


def audit_sample(sample_key: str, split: str, *, cluster_radius: float, target_radius: float, max_segments: int) -> dict:
    base = DATA / "samples" / sample_key
    image = Image.open(base.with_suffix(".png")).convert("L")
    layers = json.loads(base.with_suffix(".layers.json").read_text())
    graph = json.loads(base.with_suffix(".graph.json").read_text())
    segments = detect_raster_line_segments(
        image,
        config=LineSegmentRescueConfig(maximum_candidates=max_segments),
    )
    points = segments.reshape(-1, 2) if len(segments) else np.empty((0, 2))
    labels, centers = cluster_endpoints(points, cluster_radius)
    edges = {
        tuple(sorted((int(labels[2 * i]), int(labels[2 * i + 1]))))
        for i in range(len(segments))
        if int(labels[2 * i]) != int(labels[2 * i + 1])
    }
    targets = np.asarray([atom["center"] for atom in layers["atoms"]], dtype=np.float64)
    mapping = _nearest_mapping(centers, targets, target_radius)
    mapped = {
        tuple(sorted((mapping[a], mapping[b])))
        for a, b in edges
        if a in mapping and b in mapping and mapping[a] != mapping[b]
    }
    gt_edges = {
        tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
        for bond in graph["bonds"]
    }
    true = mapped & gt_edges
    by_type = {}
    for bond_type in sorted({str(bond["chemical_type"]) for bond in graph["bonds"]}):
        typed = {
            tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
            for bond in graph["bonds"]
            if str(bond["chemical_type"]) == bond_type
        }
        by_type[bond_type] = {
            "gt": len(typed),
            "recovered": len(true & typed),
            "recall": float(len(true & typed) / len(typed)) if typed else 1.0,
        }
    return {
        "sample_key": sample_key,
        "split": split,
        "segments": int(len(segments)),
        "provisional_nodes": int(len(centers)),
        "provisional_edges": int(len(edges)),
        "mapped_nodes": int(len(mapping)),
        "gt_atoms": int(len(targets)),
        "gt_bonds": int(len(gt_edges)),
        "mapped_edges": int(len(mapped)),
        "true_edges": int(len(true)),
        "bond_recall": float(len(true) / len(gt_edges)) if gt_edges else 1.0,
        "edge_precision": float(len(true) / len(mapped)) if mapped else 1.0,
        "all_bonds_recovered": bool(true == gt_edges),
        "by_bond_type": by_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "holdout", "all"), default="holdout")
    parser.add_argument("--cluster-radius", type=float, default=12.0)
    parser.add_argument("--target-radius", type=float, default=24.0)
    parser.add_argument("--max-segments", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    metadata = {}
    for line in (DATA / "samples.jsonl").read_text().splitlines():
        row = json.loads(line)
        metadata[row["sample_key"]] = row
    rows = [row for row in metadata.values() if args.split == "all" or row["split"] == args.split]
    rows.sort(key=lambda row: row["sample_key"])
    if args.limit:
        rows = rows[: args.limit]
    samples = [audit_sample(row["sample_key"], row["split"], cluster_radius=args.cluster_radius, target_radius=args.target_radius, max_segments=args.max_segments) for row in rows]
    def mean(name: str) -> float:
        return float(np.mean([row[name] for row in samples])) if samples else 0.0
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-graph-traversal-r5-0.1",
        "read_only": True,
        "training_performed": False,
        "sealed_test_read": False,
        "proposal_contract": "LSD raster segment endpoints -> single-link provisional graph nodes -> unordered edges; no frozen bond endpoint, atom candidate, target, or nearest-neighbor edge is input",
        "split": args.split,
        "rows": len(samples),
        "config": {"cluster_radius_px": args.cluster_radius, "target_radius_px": args.target_radius, "max_segments": args.max_segments},
        "aggregate": {
            "mean_bond_recall": mean("bond_recall"),
            "mean_edge_precision": mean("edge_precision"),
            "mean_provisional_nodes": mean("provisional_nodes"),
            "mean_provisional_edges": mean("provisional_edges"),
            "mean_gt_bonds": mean("gt_bonds"),
            "images_all_bonds_recovered": int(sum(row["all_bonds_recovered"] for row in samples)),
        },
        "by_bond_type": {},
        "samples": samples,
    }
    kinds = sorted({kind for row in samples for kind in row["by_bond_type"]})
    for kind in kinds:
        gt = sum(row["by_bond_type"].get(kind, {}).get("gt", 0) for row in samples)
        recovered = sum(row["by_bond_type"].get(kind, {}).get("recovered", 0) for row in samples)
        report["by_bond_type"][kind] = {"gt": gt, "recovered": recovered, "recall": float(recovered / gt) if gt else 1.0}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT / "report.json"), **report["aggregate"], "by_bond_type": report["by_bond_type"]}, sort_keys=True))


if __name__ == "__main__":
    main()
