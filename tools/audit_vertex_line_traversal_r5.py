"""Read-only CPU audit for an endpoint-free graph traversal route.

This audit uses only the frozen r5 vertex proposal artifact and raster line
evidence.  It deliberately does not consume predicted/frozen bond endpoints:
LSD segments are attached directly to proposed atom anchors, producing an
unordered AtomPair graph.  Ground truth is read only after proposals are
written, solely for metrics; it is never used to generate candidates.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
import numpy as np

from chemvector.components.vision.line_segment_rescue import (
    LineSegmentRescueConfig,
    detect_raster_line_segments,
    attach_segments_to_atom_pairs,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
CANDIDATES = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5-scored/candidates.jsonl"
OUTPUT = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-line-traversal-r5"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _nearest(candidates: np.ndarray, targets: np.ndarray, radius: float):
    if len(candidates) == 0 or len(targets) == 0:
        return {}, np.empty((len(targets),), dtype=np.float64)
    dist = np.linalg.norm(targets[:, None] - candidates[None, :], axis=-1)
    indices = dist.argmin(axis=1)
    values = dist[np.arange(len(targets)), indices]
    return {int(i): int(j) for i, (j, d) in enumerate(zip(indices, values)) if d <= radius}, values


def _attach_topk_segments(segments: np.ndarray, anchors: np.ndarray, radius: float, angle_limit: float, topk: int | None = 4):
    """Attach each observed segment using a pair-level, top-k assignment.

    Unlike the legacy rescue helper (which independently takes nearest
    endpoint), this keeps several endpoint hypotheses and scores the complete
    unordered pair.  It is still a candidate generator only and uses no
    ground truth or predicted bond endpoint.
    """
    grouped = {}
    for segment_index, segment in enumerate(segments):
        distances = np.linalg.norm(segment[:, None, :] - anchors[None, :, :], axis=-1)
        options = []
        for side in range(2):
            values = [int(i) for i in np.argsort(distances[side]) if distances[side, i] <= radius]
            options.append(values if topk is None else values[:topk])
        if not options[0] or not options[1]:
            continue
        sd = segment[1] - segment[0]
        sl = float(np.linalg.norm(sd))
        best = None
        for left in options[0]:
            for right in options[1]:
                if left == right:
                    continue
                delta = anchors[right] - anchors[left]
                dl = float(np.linalg.norm(delta))
                if min(sl, dl) <= 0:
                    continue
                error = max(float(distances[0, left]), float(distances[1, right]))
                angle_error = math.degrees(math.acos(float(np.clip(abs(np.dot(sd, delta) / (sl * dl)), -1.0, 1.0))))
                if angle_error > angle_limit:
                    continue
                # Endpoint accuracy dominates, then alignment and stable IDs.
                key = (error, angle_error, -dl, left, right)
                if best is None or key < best[0]:
                    best = (key, tuple(sorted((left, right))), error, angle_error, sl)
        if best is not None:
            _, pair, error, angle_error, length = best
            grouped.setdefault(pair, []).append((segment_index, error, angle_error, length))
    return grouped


def _evaluate(row: dict, metadata: dict, *, endpoint_radius: float, angle: float, max_segments: int, image_kind: str) -> dict:
    key = row["sample_key"]
    sample = DATA / "samples" / key
    image = Image.open(sample.with_suffix("." + image_kind)).convert("L")
    layers = json.loads(sample.with_suffix(".layers.json").read_text())
    graph = json.loads(sample.with_suffix(".graph.json").read_text())
    candidates = np.asarray([[x["x"] * 768.0, x["y"] * 768.0] for x in row["candidates"]], dtype=np.float64)
    targets = np.asarray([a["center"] for a in layers["atoms"]], dtype=np.float64)
    owners, distances = _nearest(candidates, targets, endpoint_radius)
    segments = detect_raster_line_segments(image, config=LineSegmentRescueConfig(maximum_candidates=max_segments))
    proposal = attach_segments_to_atom_pairs(
        segments,
        candidates,
        config=LineSegmentRescueConfig(
            maximum_endpoint_distance_pixels=endpoint_radius,
            maximum_angle_degrees=angle,
            maximum_candidates=max_segments,
        ),
    )
    topk_grouped = _attach_topk_segments(segments, candidates, endpoint_radius, angle, topk=4)
    full_grouped = _attach_topk_segments(segments, candidates, endpoint_radius, angle, topk=None)
    # Diagnostic ceiling: attach each segment directly to true atom centers.
    # This is not an admissible predictor; it isolates candidate-anchor
    # assignment loss from raster segment detection loss.
    oracle_grouped = _attach_topk_segments(segments, targets, endpoint_radius, angle, topk=None)
    gt_edges = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))) for b in graph["bonds"]}
    gt_to_candidate = owners
    candidate_to_gt = {v: k for k, v in owners.items()}
    proposed_pairs = set()
    mapped_pairs = set()
    for item in proposal:
        pair = tuple(sorted(item.atom_pair))
        proposed_pairs.add(pair)
        if pair[0] in candidate_to_gt and pair[1] in candidate_to_gt:
            mapped_pairs.add(tuple(sorted((candidate_to_gt[pair[0]], candidate_to_gt[pair[1]]))))
    topk_mapped_pairs = {
        tuple(sorted((candidate_to_gt[pair[0]], candidate_to_gt[pair[1]])))
        for pair in topk_grouped
        if pair[0] in candidate_to_gt and pair[1] in candidate_to_gt
    }
    topk_true_mapped = topk_mapped_pairs & gt_edges
    full_mapped_pairs = {
        tuple(sorted((candidate_to_gt[pair[0]], candidate_to_gt[pair[1]])))
        for pair in full_grouped
        if pair[0] in candidate_to_gt and pair[1] in candidate_to_gt
    }
    full_true_mapped = full_mapped_pairs & gt_edges
    oracle_true = set(oracle_grouped) & gt_edges
    true_mapped = mapped_pairs & gt_edges
    # Upper-bound diagnostics: is the raster detector itself able to see a
    # segment ending near the two *true* atoms?  This is computed only after
    # proposal generation and never feeds candidate construction.
    segment_endpoint_reachable = {}
    for bond in graph["bonds"]:
        left, right = int(bond["begin_atom"]), int(bond["end_atom"])
        a, b = targets[left], targets[right]
        delta = b - a
        length = float(np.linalg.norm(delta))
        reachable = False
        for segment in segments:
            s0, s1 = segment
            d0 = min(float(np.linalg.norm(s0 - a) + np.linalg.norm(s1 - b)),
                     float(np.linalg.norm(s0 - b) + np.linalg.norm(s1 - a)))
            if d0 > 2.0 * endpoint_radius:
                continue
            sd = s1 - s0
            sl = float(np.linalg.norm(sd))
            if length <= 0 or sl <= 0:
                continue
            angle_error = math.degrees(math.acos(float(np.clip(abs(np.dot(sd, delta) / (sl * length)), -1.0, 1.0))))
            if angle_error <= angle:
                reachable = True
                break
        segment_endpoint_reachable[tuple(sorted((left, right)))] = reachable
    candidate_pair_reachable = {
        edge: (edge[0] in owners and edge[1] in owners)
        for edge in gt_edges
    }
    bond_type = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))): str(b["chemical_type"])
                 for b in graph["bonds"]}
    # The traversal graph's useful recall is measured against all GT edges;
    # no nearest-neighbour edge is synthesized when line evidence is absent.
    result = {
        "sample_key": key,
        "split": metadata["split"],
        "candidate_count": int(len(candidates)),
        "target_atoms": int(len(targets)),
        "candidate_atom_recall": float(len(owners) / len(targets)) if len(targets) else 1.0,
        "candidate_atom_unique": int(len(owners)),
        "segments": int(len(segments)),
        "line_pairs": int(len(proposed_pairs)),
        "mapped_pairs": int(len(mapped_pairs)),
        "true_pairs": int(len(true_mapped)),
        "gt_bonds": int(len(gt_edges)),
        "bond_recall": float(len(true_mapped) / len(gt_edges)) if gt_edges else 1.0,
        "pair_precision": float(len(true_mapped) / len(mapped_pairs)) if mapped_pairs else 1.0,
        "topk_line_pairs": int(len(topk_grouped)),
        "topk_bond_recall": float(len(topk_true_mapped) / len(gt_edges)) if gt_edges else 1.0,
        "topk_pair_precision": float(len(topk_true_mapped) / len(topk_mapped_pairs)) if topk_mapped_pairs else 1.0,
        "topk_all_bonds_recovered": bool(topk_true_mapped == gt_edges),
        "full_line_pairs": int(len(full_grouped)),
        "full_bond_recall": float(len(full_true_mapped) / len(gt_edges)) if gt_edges else 1.0,
        "full_pair_precision": float(len(full_true_mapped) / len(full_mapped_pairs)) if full_mapped_pairs else 1.0,
        "full_all_bonds_recovered": bool(full_true_mapped == gt_edges),
        "oracle_segment_bond_recall": float(len(oracle_true) / len(gt_edges)) if gt_edges else 1.0,
        "oracle_segment_pair_precision": float(len(oracle_true) / len(oracle_grouped)) if oracle_grouped else 1.0,
        "oracle_segment_all_bonds_recovered": bool(oracle_true == gt_edges),
        "all_bonds_recovered": bool(true_mapped == gt_edges),
        "unmapped_gt_bonds": int(len(gt_edges - true_mapped)),
        "gt_segment_reach": int(sum(segment_endpoint_reachable.values())),
        "gt_candidate_pair_reach": int(sum(candidate_pair_reachable.values())),
        "by_bond_type": {
            kind: {
                "gt": sum(value == kind for value in bond_type.values()),
                "segment_reach": sum(segment_endpoint_reachable.get(e, False) for e, value in bond_type.items() if value == kind),
                "candidate_pair_reach": sum(candidate_pair_reachable[e] for e, value in bond_type.items() if value == kind),
                "recovered": sum(e in true_mapped for e, value in bond_type.items() if value == kind),
            }
            for kind in sorted(set(bond_type.values()))
        },
        "proposal": [
            {
                "pair": list(item.atom_pair),
                "endpoint_error_px": item.endpoint_error_pixels,
                "angle_error_deg": item.angle_error_degrees,
                "support_segments": len(item.support_segment_indices),
                "length_px": item.maximum_support_length_pixels,
            }
            for item in proposal
        ],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "holdout", "all"), default="holdout")
    parser.add_argument("--endpoint-radius", type=float, default=16.0)
    parser.add_argument("--angle", type=float, default=12.0)
    parser.add_argument("--max-segments", type=int, default=512)
    parser.add_argument("--image-kind", choices=("png", "bond.png", "semantic.png"), default="png")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    metadata = {r["sample_key"]: r for r in _read_jsonl(DATA / "samples.jsonl")}
    rows = _read_jsonl(CANDIDATES)
    rows = [r for r in rows if args.split == "all" or metadata[r["sample_key"]]["split"] == args.split]
    rows = rows[: args.limit] if args.limit else rows
    results = [_evaluate(r, metadata[r["sample_key"]], endpoint_radius=args.endpoint_radius, angle=args.angle, max_segments=args.max_segments, image_kind=args.image_kind) for r in rows]
    def mean(name: str) -> float:
        return float(np.mean([r[name] for r in results])) if results else 0.0
    aggregate = {
        "schema_version": "chemvector-identity-aligned-vertex-line-traversal-r5-0.1",
        "read_only": True,
        "training_performed": False,
        "sealed_test_read": False,
        "split": args.split,
        "rows": len(results),
        "config": {"endpoint_radius_px": args.endpoint_radius, "angle_deg": args.angle, "max_segments": args.max_segments},
        "aggregate": {
            "mean_candidate_atom_recall": mean("candidate_atom_recall"),
            "mean_bond_recall": mean("bond_recall"),
            "mean_pair_precision": mean("pair_precision"),
            "mean_topk_bond_recall": mean("topk_bond_recall"),
            "mean_topk_pair_precision": mean("topk_pair_precision"),
            "mean_full_bond_recall": mean("full_bond_recall"),
            "mean_full_pair_precision": mean("full_pair_precision"),
            "mean_oracle_segment_bond_recall": mean("oracle_segment_bond_recall"),
            "mean_oracle_segment_pair_precision": mean("oracle_segment_pair_precision"),
            "mean_line_pairs": mean("line_pairs"),
            "mean_gt_bonds": mean("gt_bonds"),
            "mean_gt_segment_reach": mean("gt_segment_reach"),
            "mean_gt_candidate_pair_reach": mean("gt_candidate_pair_reach"),
            "image_all_bonds_recovered": sum(r["all_bonds_recovered"] for r in results),
            "images_with_any_pair": sum(r["line_pairs"] > 0 for r in results),
            "topk_image_all_bonds_recovered": sum(r["topk_all_bonds_recovered"] for r in results),
            "full_image_all_bonds_recovered": sum(r["full_all_bonds_recovered"] for r in results),
            "oracle_segment_image_all_bonds_recovered": sum(r["oracle_segment_all_bonds_recovered"] for r in results),
        },
        "by_profile": {},
        "samples": results,
    }
    groups = defaultdict(list)
    for r in results:
        groups[metadata[r["sample_key"]]["profile_id"]].append(r)
    for profile, group in sorted(groups.items()):
        aggregate["by_profile"][profile] = {
            "rows": len(group),
            "mean_candidate_atom_recall": float(np.mean([x["candidate_atom_recall"] for x in group])),
            "mean_bond_recall": float(np.mean([x["bond_recall"] for x in group])),
            "mean_pair_precision": float(np.mean([x["pair_precision"] for x in group])),
            "mean_topk_bond_recall": float(np.mean([x["topk_bond_recall"] for x in group])),
            "mean_topk_pair_precision": float(np.mean([x["topk_pair_precision"] for x in group])),
            "mean_full_bond_recall": float(np.mean([x["full_bond_recall"] for x in group])),
            "mean_full_pair_precision": float(np.mean([x["full_pair_precision"] for x in group])),
            "mean_oracle_segment_bond_recall": float(np.mean([x["oracle_segment_bond_recall"] for x in group])),
            "mean_oracle_segment_pair_precision": float(np.mean([x["oracle_segment_pair_precision"] for x in group])),
            "all_bonds_recovered": int(sum(x["all_bonds_recovered"] for x in group)),
        }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT / "report.json"), **aggregate["aggregate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
