"""Read-only target-free geometry-port probe on the r6 fresh holdout.

This probe tests whether the existing line endpoint clusters can be made less
 brittle around visible glyphs and implicit vertices without a learned model.
Text-record glyph *boxes* are used only as image-side geometry (their atom
 labels/IDs are deliberately ignored).  For a segment endpoint touching a box,
 the nearest box boundary projection (a four-side port) routes the endpoint to
 the box centre.  Endpoints not owned by a box are routed to a cluster centre
 or to deterministic direction offsets derived only from incident segments.
Ground truth is joined after proposal construction for metrics only.

No production checkpoint, selector, OCR state, or sealed test is read/written.
"""
from __future__ import annotations

from collections import Counter
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
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-geometry-ports-r6"
INITIAL = ROOT / "outputs/experiments/chemvector-selected-matching-trial-20260915/initial.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE = 768.0


def _boxes_image_only(base: Path) -> list[list[float]]:
    """Read glyph bboxes while ignoring atom IDs, text labels, and GT graph."""
    payload = json.loads(Path(str(base) + ".text-records.json").read_text())
    boxes: list[list[float]] = []
    for glyph in payload.get("glyphs", []):
        box = glyph.get("bbox")
        if box is None or len(box) != 4:
            continue
        b = [float(x) for x in box]
        if b[2] > b[0] and b[3] > b[1]:
            boxes.append(b)
    # Renderer text records normally contain one row per glyph.  Dedupe only
    # identical boxes; no identity/label information enters proposal code.
    out = []
    for b in boxes:
        if not any(np.max(np.abs(np.asarray(b) - np.asarray(x))) < 1e-6 for x in out):
            out.append(b)
    return out


def _target(base: Path):
    layers = json.loads(Path(str(base) + ".layers.json").read_text())
    graph = json.loads(Path(str(base) + ".graph.json").read_text())
    targets = np.asarray([a["center"] for a in layers["atoms"]], dtype=np.float32)
    gt = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))) for b in graph["bonds"]}
    return targets, gt


def _rect_distance(p: np.ndarray, b: list[float]) -> float:
    dx = max(float(b[0] - p[0]), 0.0, float(p[0] - b[2]))
    dy = max(float(b[1] - p[1]), 0.0, float(p[1] - b[3]))
    return math.hypot(dx, dy)


def _project_rect(p: np.ndarray, b: list[float]) -> tuple[np.ndarray, float, int]:
    """Return nearest boundary point, distance and side (0..3).

    Four side ports are represented by the nearest point on each side; the
    selected side is an orientation cue, while the routed node remains the
    glyph centre.  Ties are deterministic in top/right/bottom/left order.
    """
    x, y = float(p[0]), float(p[1]); x0, y0, x1, y1 = b
    ports = [
        (np.asarray([min(max(x, x0), x1), y0], np.float32), 0),
        (np.asarray([x1, min(max(y, y0), y1)], np.float32), 1),
        (np.asarray([min(max(x, x0), x1), y1], np.float32), 2),
        (np.asarray([x0, min(max(y, y0), y1)], np.float32), 3),
    ]
    return min(((float(np.linalg.norm(p - q)), q, s) for q, s in ports), key=lambda z: (z[0], z[2]))


def _dedupe(points: list[np.ndarray], radius: float = 0.5) -> np.ndarray:
    out: list[np.ndarray] = []
    for p in points:
        if not any(float(np.linalg.norm(p - q)) <= radius for q in out):
            out.append(np.asarray(p, np.float32))
    return np.asarray(out, np.float32) if out else np.empty((0, 2), np.float32)


def _proposals(row: dict, *, box_margin: float, offset_scales: tuple[float, ...], offset_radius: float = 24.0):
    key = row["sample_key"]
    base = DATA / "samples" / key
    image = Image.open(Path(str(base) + ".png")).convert("L")
    seg = lsd._alt_detect(image, config=LineSegmentRescueConfig(maximum_candidates=512))
    ep = seg.reshape(-1, 2) if len(seg) else np.empty((0, 2), np.float32)
    labels, centers = cluster_endpoints(ep, 10.0)
    boxes = _boxes_image_only(base)

    # Node candidates are target-free.  Keep cluster centres, glyph centres,
    # and incident-direction offsets.  Offsets are alternatives for routing;
    # they are not treated as additional edges.
    points: list[np.ndarray] = [np.asarray(x, np.float32) for x in centers]
    source: list[tuple[str, int]] = [("cluster", i) for i in range(len(centers))]
    cluster_candidates: dict[int, list[int]] = {i: [i] for i in range(len(centers))}
    for ci, c in enumerate(centers):
        dirs: list[np.ndarray] = []
        for j in range(len(seg)):
            for side in (0, 1):
                if int(labels[2 * j + side]) != ci:
                    continue
                other = np.asarray(seg[j, 1 - side], np.float32) - c
                norm = float(np.linalg.norm(other))
                if norm > 1e-3:
                    dirs.append(other / norm)
        # Use both incident directions and their perpendicular normals.  This
        # is deterministic and target-free; no GT centre or bond is consulted.
        uniq: list[np.ndarray] = []
        for d in dirs:
            for q in (d, np.asarray([-d[1], d[0]], np.float32)):
                if not any(abs(float(np.dot(q, z))) > 0.985 for z in uniq):
                    uniq.append(q)
        for d in uniq:
            for scale in offset_scales:
                p = c + float(scale) * d
                if float(np.min(p)) < -1 or float(np.max(p)) > IMAGE + 1:
                    continue
                idx = len(points); points.append(p.astype(np.float32)); source.append(("offset", ci)); cluster_candidates[ci].append(idx)

    # Glyph centre nodes.  Each box contributes four side ports conceptually;
    # ports are used to route endpoint ownership, while only centre is a node.
    glyph_nodes: list[tuple[np.ndarray, list[float]]] = []
    for b in boxes:
        center = np.asarray([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0], np.float32)
        idx = len(points); points.append(center); source.append(("glyph", len(glyph_nodes))); glyph_nodes.append((center, b))

    # Glyph indices are appended after all cluster/offset points.
    glyph_start = len(points) - len(glyph_nodes)
    # Fix closure's compact expression with explicit stable index.
    def route(e: np.ndarray, cluster: int) -> int | None:
        hits = []
        for gi, (_gc, b) in enumerate(glyph_nodes):
            _d, q, side = _project_rect(e, b)
            d = _d
            if d <= box_margin:
                hits.append((d, side, gi))
        if hits:
            return glyph_start + min(hits)[2]
        ids = cluster_candidates.get(cluster, [])
        if not ids:
            return None
        d, i = min((float(np.linalg.norm(e - points[i])), i) for i in ids)
        return i if d <= offset_radius else None

    selected: set[tuple[int, int]] = set()
    unresolved = 0
    for j in range(len(seg)):
        a = route(np.asarray(seg[j, 0], np.float32), int(labels[2 * j]))
        b = route(np.asarray(seg[j, 1], np.float32), int(labels[2 * j + 1]))
        if a is None or b is None:
            unresolved += 1
            continue
        if a == b:
            continue
        selected.add(tuple(sorted((a, b))))
    return np.asarray(points, np.float32), selected, unresolved, len(seg), len(boxes), len(centers)


def _score(row: dict, **kwargs) -> dict:
    base = DATA / "samples" / row["sample_key"]
    points, selected, unresolved, segments, boxes, clusters = _proposals(row, **kwargs)
    targets, gt = _target(base)
    c2t = strict_candidate_to_target(points, targets)
    mapped: set[tuple[int, int]] = set(); dangling = int(unresolved)
    for a, b in selected:
        if a not in c2t or b not in c2t:
            dangling += 1; continue
        ta, tb = c2t[a], c2t[b]
        if ta == tb:
            continue
        mapped.add(tuple(sorted((ta, tb))))
    true = mapped & gt
    matched_targets = set(c2t.values())
    return {
        "images": 1, "pred_nodes": int(len(points)), "matched_nodes": int(len(c2t)),
        "unique_matched_targets": int(len(matched_targets)), "target_atoms": int(len(targets)),
        "segments": int(segments), "boxes": int(boxes), "clusters": int(clusters),
        "selected_edges": int(len(selected) + unresolved), "mapped_edges": int(len(mapped)),
        "true_edges": int(len(true)), "gt_bonds": int(len(gt)), "unresolved_edges": int(dangling),
        "self_loops": 0,
    }


def _finish(rows: list[dict]) -> dict:
    fields = ("images", "pred_nodes", "matched_nodes", "unique_matched_targets", "target_atoms", "segments", "boxes", "clusters", "selected_edges", "mapped_edges", "true_edges", "gt_bonds", "unresolved_edges", "self_loops")
    c = Counter({f: sum(int(r.get(f, 0)) for r in rows) for f in fields})
    c["node_precision"] = c["matched_nodes"] / max(c["pred_nodes"], 1)
    c["node_recall"] = c["unique_matched_targets"] / max(c["target_atoms"], 1)
    c["edge_precision"] = c["true_edges"] / max(c["mapped_edges"], 1)
    c["edge_recall"] = c["true_edges"] / max(c["gt_bonds"], 1)
    c["dangling_rate"] = c["unresolved_edges"] / max(c["selected_edges"], 1)
    return dict(c)


def main() -> dict:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--box-margin", type=float, default=12.0)
    p.add_argument("--offset-scales", default="-8,-4,4,8")
    p.add_argument("--max-images", type=int, default=100)
    args = p.parse_args()
    if hashlib.sha256(INITIAL.read_bytes()).hexdigest() != INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x.strip()]
    rows = sorted([r for r in rows if r.get("render_status") == "accepted" and r.get("split") == "fresh_holdout"], key=lambda r: r["sample_key"])
    rows = rows[: args.max_images]
    scales = tuple(float(x) for x in args.offset_scales.split(",") if x.strip())
    chosen = [(_score(r, box_margin=args.box_margin, offset_scales=scales), r) for r in rows]
    agg = _finish([x[0] for x in chosen])
    profiles = {pname: _finish([x for x, r in chosen if r.get("profile_id") == pname]) for pname in sorted({r.get("profile_id") for _, r in chosen})}
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-geometry-ports-r6-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "initial_sha256": INITIAL_SHA, "dataset": str(DATA), "split": "fresh_holdout",
        "rows": len(rows),
        "proposal_contract": "target-free LSD endpoint clusters + image-only glyph bbox four-side boundary ports + incident-direction implicit offsets; GT post-hoc only",
        "config": {"box_margin_px": args.box_margin, "offset_scales_px": scales, "cluster_radius_px": 10.0, "target_match_radius_px": 6.0, "ambiguity_radius_px": 20.0},
        "aggregate": agg, "by_profile": profiles,
        "gate": {
            "checkpoint_unchanged": hashlib.sha256(INITIAL.read_bytes()).hexdigest() == INITIAL_SHA,
            "sealed_test_read": False, "training_performed": False, "gpu_go": False,
            "edge_precision_ge_0.965917": agg["edge_precision"] >= .965917,
            "edge_recall_gt_0.850031": agg["edge_recall"] > .850031,
            "dangling_le_0.05": agg["dangling_rate"] <= .05,
            "all_profile_recall_ge_0.75": all(v["edge_recall"] >= .75 for v in profiles.values()),
        },
        "interpretation": "diagnostic proposal capacity only; no training or production merge",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "aggregate": agg, "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
