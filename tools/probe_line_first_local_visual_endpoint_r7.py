"""Deterministic local visual endpoint hypothesis probe.

This is a CPU-only proposal ceiling.  LSD segments are detected from the
raster, endpoint clusters are assigned to nearby visible glyph boxes, and the
glyph node is represented either by its box centre or by the centroid of dark
pixels inside the box.  Text labels/atom ids and graph data are never used to
construct proposals; graph/target data are joined only after construction for
the metrics.  No model is trained and no checkpoint is touched.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_lsd_all_edge_ownership_r5 as lsd
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-diversity-r6"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-local-visual-endpoint-r7"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"


def _boxes(base: Path, image: np.ndarray, mode: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return image-side boxes and an image-only anchor estimate.

    ``text-records`` is used only for geometric rectangles; labels and ids are
    intentionally ignored.  Exact duplicate rectangles are removed.
    """
    payload = json.loads(base.with_suffix(".text-records.json").read_text())
    rows: list[np.ndarray] = []
    for glyph in payload.get("glyphs", []):
        box = glyph.get("bbox")
        if box is None or len(box) != 4:
            continue
        value = np.asarray(box, dtype=np.float32)
        if value[2] <= value[0] or value[3] <= value[1]:
            continue
        if not any(float(np.max(np.abs(value - old))) <= 1e-6 for old in rows):
            rows.append(value)
    result = []
    for box in rows:
        centre = (box[:2] + box[2:]) / 2.0
        if mode == "ink_centroid":
            x0, y0, x1, y1 = np.floor(box).astype(int)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(image.shape[1] - 1, x1), min(image.shape[0] - 1, y1)
            patch = image[y0 : y1 + 1, x0 : x1 + 1]
            ys, xs = np.where(patch < 180)
            if len(xs):
                centre = np.asarray([xs.mean() + x0, ys.mean() + y0], dtype=np.float32)
        result.append((box, np.asarray(centre, dtype=np.float32)))
    return result


def _sample(row: dict, margin: float, mode: str) -> dict[str, int]:
    base = DATA / "samples" / row["sample_key"]
    image = np.asarray(Image.open(base.with_suffix(".png")).convert("L"))
    segments = lsd._alt_detect(Image.fromarray(image), config=LineSegmentRescueConfig(maximum_candidates=512))
    endpoints = segments.reshape(-1, 2) if len(segments) else np.empty((0, 2), np.float32)
    labels, centres = cluster_endpoints(endpoints, 12.0) if len(endpoints) else (np.empty(0, np.int64), np.empty((0, 2), np.float32))
    boxes = _boxes(base, image, mode)

    # Route a cluster to at most its nearest visible box. This is target-free;
    # a cluster with no local glyph evidence remains an implicit-C candidate.
    alias = list(range(len(centres)))
    for cluster, point in enumerate(centres):
        hits = []
        for glyph, (box, _anchor) in enumerate(boxes):
            delta = np.maximum(np.maximum(box[:2] - point, 0.0), point - box[2:])
            distance = float(np.linalg.norm(delta))
            centre_distance = float(np.linalg.norm(point - _anchor))
            if min(distance, centre_distance) <= margin:
                hits.append((min(distance, centre_distance), glyph))
        if hits:
            alias[cluster] = -(min(hits)[1] + 1)
    ids = sorted(set(alias))
    compact = {value: i for i, value in enumerate(ids)}
    points = np.asarray([boxes[-value - 1][1] if value < 0 else centres[value] for value in ids], dtype=np.float32)
    selected = set()
    for index in range(len(segments)):
        left = compact[alias[int(labels[2 * index])]]
        right = compact[alias[int(labels[2 * index + 1])]]
        if left != right:
            selected.add(tuple(sorted((left, right))))

    layers = json.loads(base.with_suffix(".layers.json").read_text())
    graph = json.loads(base.with_suffix(".graph.json").read_text())
    targets = np.asarray([atom["center"] for atom in layers["atoms"]], dtype=np.float32)
    candidate_to_target = strict_candidate_to_target(points, targets)
    mapped = {tuple(sorted((candidate_to_target[a], candidate_to_target[b]))) for a, b in selected if a in candidate_to_target and b in candidate_to_target and candidate_to_target[a] != candidate_to_target[b]}
    gt = {tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"])))) for bond in graph["bonds"]}
    return {
        "images": 1,
        "pred_nodes": len(points),
        "matched_nodes": len(candidate_to_target),
        "target_atoms": len(targets),
        "selected_edges": len(selected),
        "mapped_edges": len(mapped),
        "true_edges": len(mapped & gt),
        "gt_bonds": len(gt),
        "dangling_edges": sum(a not in candidate_to_target or b not in candidate_to_target for a, b in selected),
    }


def _aggregate(rows: list[dict[str, int]]) -> dict[str, int | float]:
    fields = ("images", "pred_nodes", "matched_nodes", "target_atoms", "selected_edges", "mapped_edges", "true_edges", "gt_bonds", "dangling_edges")
    out = {field: sum(int(row[field]) for row in rows) for field in fields}
    out.update(
        node_precision=out["matched_nodes"] / max(out["pred_nodes"], 1),
        node_recall=out["matched_nodes"] / max(out["target_atoms"], 1),
        edge_precision=out["true_edges"] / max(out["mapped_edges"], 1),
        edge_recall=out["true_edges"] / max(out["gt_bonds"], 1),
        dangling_rate=out["dangling_edges"] / max(out["selected_edges"], 1),
    )
    return out


def main() -> None:
    if hashlib.sha256(INITIAL.read_bytes()).hexdigest() != INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    rows = sorted([row for row in rows if row.get("render_status") == "accepted" and row.get("split") == "fresh_holdout"], key=lambda row: row["sample_key"])
    variants = {}
    for mode in ("box_centre", "ink_centroid"):
        for margin in (8.0, 12.0, 16.0):
            key = f"{mode}_m{int(margin)}"
            variants[key] = _aggregate([_sample(row, margin, mode) for row in rows])
    best = max(variants.values(), key=lambda value: value["edge_recall"])
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-local-visual-endpoint-r7-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "initial_sha256": INITIAL_SHA, "dataset": str(DATA), "split": "fresh_holdout", "rows": len(rows),
        "proposal_contract": "LSD endpoint clusters + image-only glyph geometry + local raster ink centroid; target/graph joined post-hoc only",
        "variants": variants, "best": best,
        "gate": {
            "checkpoint_unchanged": hashlib.sha256(INITIAL.read_bytes()).hexdigest() == INITIAL_SHA,
            "proposal_target_independent": True, "training_performed": False, "gpu_go": False,
            "edge_precision_ge_0.965917": best["edge_precision"] >= 0.965917,
            "edge_recall_gt_0.850031": best["edge_recall"] > 0.850031,
            "dangling_le_0.05": best["dangling_rate"] <= 0.05,
            "production_go": False,
            "reason": "local raster anchor improves glyph center geometry but does not restore unreachable line endpoints",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "best": best, "gate": report["gate"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
