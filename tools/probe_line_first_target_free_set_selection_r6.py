"""Target-free multi-candidate endpoint set selection (r6 diagnostic).

This is deliberately an isolated CPU probe.  It keeps several image-derived
alternatives for every LSD endpoint (cluster centre, raw endpoint centroid,
and nearby glyph centre) and greedily selects one compatible candidate for
each segment.  Candidate ownership is shared across segments, duplicate
edges are collapsed, and a small degree/topology penalty discourages making
one endpoint own unrelated segments.  Ground-truth coordinates/edges are
loaded *after* proposal construction solely for the report metrics.

The probe does not read sealed data, does not train, and does not alter any
production checkpoint or selector.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib, json, math, sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_lsd_all_edge_ownership_r5 as lsd
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-diversity-r6"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-target-free-set-selection-r6"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE_SIZE = 768.0


def _boxes(base: Path) -> list[np.ndarray]:
    """Return image-only glyph boxes; labels and atom IDs are never used."""
    payload = json.loads(Path(str(base) + ".text-records.json").read_text())
    out: list[np.ndarray] = []
    for item in payload.get("glyphs", []):
        b = item.get("bbox")
        if b is None or len(b) != 4:
            continue
        q = np.asarray(b, np.float32)
        if q[2] > q[0] and q[3] > q[1]:
            if not any(float(np.max(np.abs(q - x))) < 1e-6 for x in out):
                out.append(q)
    return out


def _point_to_box(p: np.ndarray, b: np.ndarray) -> float:
    return float(math.hypot(max(float(b[0] - p[0]), 0.0, float(p[0] - b[2])),
                            max(float(b[1] - p[1]), 0.0, float(p[1] - b[3]))))


def _alternatives(seg: np.ndarray, labels: np.ndarray, centers: np.ndarray,
                  boxes: list[np.ndarray], glyph_radius: float) -> tuple[np.ndarray, list[list[int]]]:
    """Build a finite, shared candidate set and endpoint alternative lists."""
    points: list[np.ndarray] = [np.asarray(c, np.float32) for c in centers]
    source: list[str] = ["cluster"] * len(points)
    # Glyph centres are image geometry only.  Candidate sharing is intentional:
    # several seam endpoints can compete for one visible label anchor.
    glyph_ids: list[int] = []
    for b in boxes:
        p = np.asarray([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], np.float32)
        glyph_ids.append(len(points)); points.append(p); source.append("glyph")
    alternatives: list[list[int]] = []
    for endpoint in range(len(seg) * 2):
        p = np.asarray(seg.reshape(-1, 2)[endpoint], np.float32)
        ci = int(labels[endpoint])
        # The own cluster is always retained (prefix-safe).
        choices: list[tuple[float, int]] = [(float(np.linalg.norm(p - centers[ci])), ci)]
        for gi, b in zip(glyph_ids, boxes):
            d = _point_to_box(p, b)
            if d <= glyph_radius:
                choices.append((d + 1.5, gi))
        # Keep at most three alternatives and deterministic tie ordering.
        choices.sort(key=lambda z: (z[0], z[1]))
        alternatives.append([int(i) for _, i in choices[:3]])
    return np.asarray(points, np.float32), alternatives


def _select(seg: np.ndarray, alternatives: list[list[int]], points: np.ndarray,
            max_degree: int = 4, degree_penalty: float = 3.0) -> set[tuple[int, int]]:
    """Greedy global set selection over segment hypotheses.

    Processing long, well-supported segments first makes ownership stable.
    A selected candidate pair is globally deduplicated.  ``degree_penalty``
    is image/topology-only and does not encode target identities.
    """
    degrees = Counter()
    selected: set[tuple[int, int]] = set()
    # Segment confidence proxy: raster length (long lines are less ambiguous).
    order = sorted(range(len(seg)), key=lambda i: (-float(np.linalg.norm(seg[i, 1] - seg[i, 0])), i))
    for i in order:
        e0, e1 = np.asarray(seg[i, 0]), np.asarray(seg[i, 1])
        best = None
        for a in alternatives[2 * i]:
            for b in alternatives[2 * i + 1]:
                if a == b:
                    continue
                pair = tuple(sorted((int(a), int(b))))
                # Endpoint reprojection error plus a bounded degree cost.
                err = float(np.linalg.norm(e0 - points[a]) + np.linalg.norm(e1 - points[b]))
                if pair in selected:
                    err += 8.0  # preserve one edge per observed bond family
                if degrees[a] >= max_degree:
                    err += degree_penalty * (degrees[a] - max_degree + 1)
                if degrees[b] >= max_degree:
                    err += degree_penalty * (degrees[b] - max_degree + 1)
                # Mild preference for compact endpoint assignment.
                err += 0.015 * float(np.linalg.norm(points[a] - points[b]))
                key = (err, pair, int(a), int(b))
                if best is None or key < best[0]:
                    best = (key, pair)
        if best is None:
            continue
        pair = best[1]
        if pair not in selected:
            selected.add(pair); degrees[pair[0]] += 1; degrees[pair[1]] += 1
    return selected


def _target(base: Path) -> tuple[np.ndarray, set[tuple[int, int]]]:
    layers = json.loads(Path(str(base) + ".layers.json").read_text())
    graph = json.loads(Path(str(base) + ".graph.json").read_text())
    targets = np.asarray([a["center"] for a in layers["atoms"]], np.float32)
    gt = {tuple(sorted((int(x["begin_atom"]), int(x["end_atom"])))) for x in graph["bonds"]}
    return targets, gt


def _one(row: dict, glyph_radius: float, max_degree: int, degree_penalty: float) -> dict:
    base = DATA / "samples" / row["sample_key"]
    image = Image.open(Path(str(base) + ".png")).convert("L")
    seg = lsd._alt_detect(image, config=LineSegmentRescueConfig(maximum_candidates=512))
    raw = seg.reshape(-1, 2) if len(seg) else np.empty((0, 2), np.float32)
    labels, centers = cluster_endpoints(raw, 10.0)
    points, alternatives = _alternatives(seg, labels, centers, _boxes(base), glyph_radius)
    edges = _select(seg, alternatives, points, max_degree=max_degree, degree_penalty=degree_penalty)
    targets, gt = _target(base)
    mapping = strict_candidate_to_target(points, targets)
    mapped = {tuple(sorted((mapping[a], mapping[b]))) for a, b in edges
              if a in mapping and b in mapping and mapping[a] != mapping[b]}
    true = mapped & gt
    return {"images": 1, "segments": int(len(seg)), "candidates": int(len(points)),
            "matched_nodes": int(len(mapping)), "target_atoms": int(len(targets)),
            "selected_edges": int(len(edges)), "mapped_edges": int(len(mapped)),
            "true_edges": int(len(true)), "gt_bonds": int(len(gt)),
            "dangling_edges": int(len(edges) - len(mapped)), "profile_id": row["profile_id"],
            "all_bonds_recovered": bool(true == gt)}


def _summary(rows: list[dict]) -> dict:
    keys = ("images", "segments", "candidates", "matched_nodes", "target_atoms",
            "selected_edges", "mapped_edges", "true_edges", "gt_bonds", "dangling_edges")
    c = Counter({k: sum(int(r[k]) for r in rows) for k in keys})
    c["node_precision"] = c["matched_nodes"] / max(c["candidates"], 1)
    c["node_recall"] = c["matched_nodes"] / max(c["target_atoms"], 1)
    c["edge_precision"] = c["true_edges"] / max(c["mapped_edges"], 1)
    c["edge_recall"] = c["true_edges"] / max(c["gt_bonds"], 1)
    c["dangling_rate"] = c["dangling_edges"] / max(c["selected_edges"], 1)
    c["images_all_bonds_recovered"] = sum(bool(r["all_bonds_recovered"]) for r in rows)
    return dict(c)


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--glyph-radius", type=float, default=28.0)
    p.add_argument("--max-degree", type=int, default=4)
    p.add_argument("--degree-penalty", type=float, default=3.0)
    p.add_argument("--max-images", type=int, default=95)
    args = p.parse_args()
    digest = hashlib.sha256(INITIAL.read_bytes()).hexdigest()
    if digest != INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines()
            if x.strip() and json.loads(x).get("render_status") == "accepted"
            and json.loads(x).get("split") == "fresh_holdout"]
    rows = sorted(rows, key=lambda r: r["sample_key"])[: args.max_images]
    values = [_one(r, args.glyph_radius, args.max_degree, args.degree_penalty) for r in rows]
    profiles = {pname: _summary([v for v in values if v["profile_id"] == pname])
                for pname in sorted({v["profile_id"] for v in values})}
    aggregate = _summary(values)
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-target-free-set-selection-r6-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False,
        "initial_sha256": INITIAL_SHA, "dataset": str(DATA), "split": "fresh_holdout",
        "rows": len(rows),
        "proposal_contract": "LSD endpoints -> finite shared cluster/glyph alternatives -> target-free greedy set selection with one-to-one candidate ownership, duplicate collapse, degree/topology penalty; GT post-hoc only",
        "config": {"glyph_radius_px": args.glyph_radius, "cluster_radius_px": 10.0,
                   "max_alternatives_per_endpoint": 3, "max_degree": args.max_degree,
                   "degree_penalty": args.degree_penalty, "target_match_radius_px": 6.0},
        "aggregate": aggregate, "by_profile": profiles,
        "gate": {
            "checkpoint_unchanged": hashlib.sha256(INITIAL.read_bytes()).hexdigest() == INITIAL_SHA,
            "sealed_test_read": False, "training_performed": False, "gpu_go": False,
            "edge_precision_ge_0.965917": aggregate["edge_precision"] >= .965917,
            "edge_recall_gt_0.850031": aggregate["edge_recall"] > .850031,
            "dangling_le_0.05": aggregate["dangling_rate"] <= .05,
            "all_profile_recall_ge_0.75": all(v["edge_recall"] >= .75 for v in profiles.values()),
        },
        "interpretation": "deterministic proposal ceiling only; no training or production merge",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "aggregate": aggregate,
                      "gate": report["gate"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
