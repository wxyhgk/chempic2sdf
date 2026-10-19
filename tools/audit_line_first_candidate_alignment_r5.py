"""Read-only line-first graph audit aligned to frozen vertex candidates.

Line segments are converted to provisional nodes/edges without target data. The
provisional nodes are then greedily matched to the frozen scored vertex
candidate field, also without targets. Ground truth is consulted only for the
final metrics. This measures whether the promising line-first route can use
the current candidate representation rather than an optimistic GT-center map.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig, detect_raster_line_segments
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
CANDIDATES = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5-scored/candidates.jsonl"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-candidate-alignment-r5-v2"
IMAGE = 768.0
TARGET_RADIUS = 6.0
AMBIGUITY_RADIUS = 20.0


def read_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def align_nodes(centers: np.ndarray, candidates: np.ndarray, radius: float):
    """Target-free one-to-one node/candidate alignment by global distance."""
    pairs = []
    if len(centers) and len(candidates):
        d = np.linalg.norm(centers[:, None] - candidates[None, :], axis=-1)
        for ni, ci in np.ndindex(d.shape):
            if d[ni, ci] <= radius:
                pairs.append((float(d[ni, ci]), ni, ci))
    node_to_candidate = {}
    used = set()
    for _, ni, ci in sorted(pairs):
        if ni not in node_to_candidate and ci not in used:
            node_to_candidate[ni] = ci
            used.add(ci)
    return node_to_candidate


def evaluate(row, candidates, *, cluster_radius, candidate_radius, max_segments):
    base = DATA / "samples" / row["sample_key"]
    image = Image.open(base.with_suffix(".png")).convert("L")
    layers = json.loads(base.with_suffix(".layers.json").read_text())
    graph = json.loads(base.with_suffix(".graph.json").read_text())
    segments = detect_raster_line_segments(image, config=LineSegmentRescueConfig(maximum_candidates=max_segments))
    points = segments.reshape(-1, 2) if len(segments) else np.empty((0, 2))
    labels, centers = cluster_endpoints(points, cluster_radius)
    provisional_edges = {tuple(sorted((int(labels[2*i]), int(labels[2*i+1])))) for i in range(len(segments)) if labels[2*i] != labels[2*i+1]}
    cand = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE] for v in candidates], dtype=np.float64)
    node_map = align_nodes(centers, cand, candidate_radius)
    candidate_edges = {tuple(sorted((node_map[a], node_map[b]))) for a,b in provisional_edges if a in node_map and b in node_map and node_map[a] != node_map[b]}
    targets = np.asarray([a["center"] for a in layers["atoms"]], dtype=np.float64)
    # Post-hoc strict candidate ownership labels; never used above.
    c2t = {}
    for ci, point in enumerate(cand):
        dist = np.linalg.norm(targets - point, axis=-1)
        close = np.where(dist <= TARGET_RADIUS)[0]
        near = np.where(dist <= AMBIGUITY_RADIUS)[0]
        if len(close) == 1 and len(near) == 1:
            c2t[ci] = int(close[0])
    mapped_edges = {tuple(sorted((c2t[a], c2t[b]))) for a,b in candidate_edges if a in c2t and b in c2t and c2t[a] != c2t[b]}
    gt_edges = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))) for b in graph["bonds"]}
    true = mapped_edges & gt_edges
    return {"sample_key": row["sample_key"], "split": row["split"], "segments": int(len(segments)), "provisional_nodes": int(len(centers)), "provisional_edges": int(len(provisional_edges)), "aligned_nodes": int(len(node_map)), "candidate_edges": int(len(candidate_edges)), "mapped_edges": int(len(mapped_edges)), "true_edges": int(len(true)), "gt_atoms": int(len(targets)), "gt_bonds": int(len(gt_edges)), "bond_recall": float(len(true)/len(gt_edges)) if gt_edges else 1.0, "edge_precision": float(len(true)/len(mapped_edges)) if mapped_edges else 1.0, "all_bonds_recovered": bool(true == gt_edges)}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("train", "holdout", "all"), default="holdout")
    p.add_argument("--cluster-radius", type=float, default=16.0)
    p.add_argument("--candidate-radius", type=float, default=16.0)
    p.add_argument("--max-segments", type=int, default=512)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    metadata = {r["sample_key"]: r for r in read_jsonl(DATA / "samples.jsonl")}
    candidate_rows = {r["sample_key"]: r["candidates"] for r in read_jsonl(CANDIDATES)}
    rows = [r for r in metadata.values() if args.split == "all" or r["split"] == args.split]
    rows.sort(key=lambda r: r["sample_key"])
    results = [evaluate(r, candidate_rows[r["sample_key"]], cluster_radius=args.cluster_radius, candidate_radius=args.candidate_radius, max_segments=args.max_segments) for r in rows]
    mean = lambda k: float(np.mean([r[k] for r in results])) if results else 0.0
    report = {"schema_version":"chemvector-identity-aligned-line-first-candidate-alignment-r5-0.2", "read_only":True, "training_performed":False, "gpu_used":False, "sealed_test_read":False, "production_checkpoint_written":False, "proposal_contract":"LSD endpoints -> provisional nodes/edges -> target-free frozen scored vertex candidate alignment; target coordinates are consulted only after proposal construction for metrics", "split":args.split, "rows":len(results), "config":{"cluster_radius_px":args.cluster_radius,"candidate_radius_px":args.candidate_radius,"target_match_radius_px":TARGET_RADIUS,"max_segments":args.max_segments}, "aggregate":{"mean_bond_recall":mean("bond_recall"),"mean_edge_precision":mean("edge_precision"),"mean_aligned_nodes":mean("aligned_nodes"),"mean_gt_bonds":mean("gt_bonds"),"images_all_bonds_recovered":int(sum(r["all_bonds_recovered"] for r in results))}, "samples":results, "gate":{"candidate_mapping_target_free":True,"training_performed":False,"sealed_test_read":False,"production_go":False}}
    output_dir = args.output or OUT
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"output":str(output_dir/"report.json"), **report["aggregate"]}, sort_keys=True))


if __name__ == "__main__": main()
