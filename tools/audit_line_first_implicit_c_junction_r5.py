#!/usr/bin/env python3
"""CPU-only, target-free implicit-carbon junction refinement ceiling.

The frozen LSD endpoint clusters remain the candidate identities.  This audit
does not add a geometric offset cloud: for a cluster with at least two
incident segments whose outgoing directions are non-collinear, it computes a
regularised least-squares point constrained to the cluster centre.  Labels
are joined only after proposals are frozen and are used for diagnostics.  No
production module/checkpoint is imported for mutation, and sealed test data
is never read.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib, json, math, sys, time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from chemvector.components.vision.line_segment_rescue import (
    LineSegmentRescueConfig, detect_raster_line_segments,
)
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from tools.preflight_line_first_edge_features_r5 import DATA

INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-implicit-c-junction-r5-v2"
CLUSTER_RADIUS = 16.0
MAX_SEGMENTS = 512
ANGLE_MIN_DEG = 20.0
GUARD_RADIUS_PX = 30.0
# ``None`` is the frozen cluster-centre baseline. Numeric values are the
# regularisation strength of the incident-line least-squares refinement.
BETAS = (None, 0.1, 0.3, 1.0)
HOLDOUT_LIMIT = 50
MATCH_RADIUS = 6.0


def _segments(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = Image.open(DATA / "samples" / f"{key}.png").convert("L")
    seg = detect_raster_line_segments(image, config=LineSegmentRescueConfig(maximum_candidates=MAX_SEGMENTS))
    if not len(seg):
        return seg.astype(np.float32), np.empty(0, np.int64), np.empty((0, 2), np.float32)
    labels, centers = cluster_endpoints(seg.reshape(-1, 2), CLUSTER_RADIUS)
    return seg.astype(np.float32), labels.astype(np.int64), centers.astype(np.float32)


def refine(seg: np.ndarray, labels: np.ndarray, centers: np.ndarray, beta: float | None) -> tuple[np.ndarray, dict]:
    """Refine existing clusters only; all inputs are image/LSD evidence."""
    if not len(centers):
        return centers.copy(), {"refined": 0, "center_fallback": 0, "max_shift_px": 0.0}
    points: list[np.ndarray] = []
    refined = fallback = 0
    max_shift = 0.0
    if beta is None:
        return centers.copy(), {"refined": 0, "center_fallback": len(centers), "max_shift_px": 0.0}
    for lab, center in zip(sorted(set(labels.tolist())), centers):
        endpoint_rows = np.where(labels == lab)[0]
        incidents = []
        for flat in endpoint_rows.tolist():
            si, side = divmod(flat, 2)
            p = seg[si, side]
            other = seg[si, 1 - side]
            delta = p - other
            length = float(np.linalg.norm(delta))
            if length >= 5.0:
                incidents.append((p.astype(np.float64), delta.astype(np.float64) / length))
        max_angle = 0.0
        for i in range(len(incidents)):
            for j in range(i):
                # Directions are directed outward from the segment; retaining
                # the directed angle rejects nearly collinear seam fragments.
                dot = float(np.clip(np.dot(incidents[i][1], incidents[j][1]), -1.0, 1.0))
                max_angle = max(max_angle, math.degrees(math.acos(dot)))
        if len(incidents) < 2 or max_angle < ANGLE_MIN_DEG:
            q = center.astype(np.float64)
            fallback += 1
        else:
            # Each line contributes I-u u^T, so q is close to all incident
            # supporting lines. beta=0 is the unconstrained intersection;
            # positive beta protects against parallel/noisy LSD fragments.
            A = float(beta) * np.eye(2, dtype=np.float64)
            b = float(beta) * center.astype(np.float64)
            for p, u in incidents:
                M = np.eye(2, dtype=np.float64) - np.outer(u, u)
                A += M
                b += M @ p
            q = np.linalg.solve(A + 1e-5 * np.eye(2), b)
            if float(np.linalg.norm(q - center)) > GUARD_RADIUS_PX:
                q = center.astype(np.float64)
                fallback += 1
            else:
                refined += 1
        max_shift = max(max_shift, float(np.linalg.norm(q - center)))
        points.append(q.astype(np.float32))
    return np.asarray(points, dtype=np.float32), {
        "refined": refined, "center_fallback": fallback, "max_shift_px": max_shift,
    }


def _targets(key: str) -> tuple[np.ndarray, set[tuple[int, int]]]:
    base = DATA / "samples" / key
    layers = json.loads(base.with_suffix(".layers.json").read_text())
    graph = json.loads(base.with_suffix(".graph.json").read_text())
    atoms = np.asarray([a["center"] for a in layers["atoms"]], dtype=np.float32)
    edges = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))) for b in graph["bonds"]}
    return atoms, edges


def _mapping(points: np.ndarray, targets: np.ndarray) -> dict[int, int]:
    if not len(points) or not len(targets):
        return {}
    d = np.linalg.norm(points[:, None] - targets[None], axis=-1)
    pairs = sorted((float(d[i, j]), i, j) for i in range(len(points)) for j in range(len(targets)))
    result: dict[int, int] = {}; used: set[int] = set()
    for dist, i, j in pairs:
        if dist > MATCH_RADIUS or i in result or j in used:
            continue
        result[i] = j; used.add(j)
    return result


def _edge_pairs(labels: np.ndarray, seg: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for i in range(len(seg)):
        a, b = int(labels[2 * i]), int(labels[2 * i + 1])
        if a != b:
            pairs.add(tuple(sorted((a, b))))
    return pairs


def _one(row: dict, beta: float) -> dict:
    key = row["sample_key"]
    seg, labels, centers = _segments(key)
    points, refine_meta = refine(seg, labels, centers, beta)
    targets, gt_edges = _targets(key)
    mapping = _mapping(points, targets)
    d = np.linalg.norm(points[:, None] - targets[None], axis=-1) if len(points) else np.empty((0, len(targets)))
    atom_hit6 = int(sum(float(d[i, j]) <= 6.0 for i, j in mapping.items()))
    atom_hit10 = int(sum(float(d[i, j]) <= 10.0 for i, j in mapping.items()))
    pair_labels = _edge_pairs(labels, seg)
    label_to_cluster = {int(i): int(i) for i in range(len(centers))}
    # cluster_endpoints labels are contiguous in this implementation, but use
    # centre row indices explicitly to avoid relying on array order.
    cluster_ids = sorted(set(labels.tolist()))
    label_to_row = {lab: i for i, lab in enumerate(cluster_ids)}
    proposed_edges = set()
    for a, b in pair_labels:
        if a in label_to_row and b in label_to_row:
            proposed_edges.add(tuple(sorted((label_to_row[a], label_to_row[b]))))
    mapped_proposed = {tuple(sorted((mapping[a], mapping[b]))) for a, b in proposed_edges if a in mapping and b in mapping and mapping[a] != mapping[b]}
    edge_hit = len(mapped_proposed & gt_edges)
    return {
        "sample_key": key, "profile_id": row.get("profile_id"), "candidate_count": len(points),
        "target_atoms": len(targets), "matched_atoms": len(mapping), "atom_hit6": atom_hit6,
        "atom_hit10": atom_hit10, "proposed_edges": len(mapped_proposed), "edge_hit": edge_hit,
        "gt_bonds": len(gt_edges), "all_bonds_recovered": int(mapped_proposed >= gt_edges),
        "refine": refine_meta,
    }


def _summ(rows: list[dict]) -> dict:
    keys = ("candidate_count", "target_atoms", "matched_atoms", "atom_hit6", "atom_hit10", "proposed_edges", "edge_hit", "gt_bonds", "all_bonds_recovered")
    out = {k: int(sum(int(r[k]) for r in rows)) for k in keys}
    out.update({
        "images": len(rows),
        "candidate_per_image": out["candidate_count"] / max(len(rows), 1),
        "atom_precision_6px": out["atom_hit6"] / max(out["matched_atoms"], 1),
        "atom_recall_6px": out["atom_hit6"] / max(out["target_atoms"], 1),
        "atom_recall_10px": out["atom_hit10"] / max(out["target_atoms"], 1),
        "edge_precision": out["edge_hit"] / max(out["proposed_edges"], 1),
        "edge_recall": out["edge_hit"] / max(out["gt_bonds"], 1),
        "all_bonds_recovered_fraction": out["all_bonds_recovered"] / max(len(rows), 1),
    })
    return out


def run() -> dict:
    if file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    hold = sorted([r for r in rows if r.get("split") == "holdout" and r.get("render_status") == "accepted"], key=lambda r: r["sample_key"])[:HOLDOUT_LIMIT]
    if len(hold) != HOLDOUT_LIMIT:
        raise RuntimeError(f"expected {HOLDOUT_LIMIT} holdout rows, got {len(hold)}")
    started = time.time()
    all_results = {}
    for beta in BETAS:
        name = "center_baseline" if beta is None else f"beta_{str(beta).replace('.', '_')}"
        result = [_one(row, beta) for row in hold]
        all_results[name] = {"config": {"beta": beta, "angle_min_deg": ANGLE_MIN_DEG, "guard_radius_px": GUARD_RADIUS_PX}, "summary": _summ(result), "rows": result}
    after = file_digest(INITIAL)
    # This is a proposal ceiling, not a training receipt. In particular, no
    # labels or refined points are exported as an inference artifact.
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-implicit-c-junction-r5-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "initial_sha256": INITIAL_SHA, "initial_sha256_after": after,
        "proposal_contract": "frozen LSD segments -> existing endpoint clusters -> target-free incident-line refinement; no new offset candidates",
        "fixed_slice": {"holdout_images": HOLDOUT_LIMIT, "cluster_radius_px": CLUSTER_RADIUS, "max_segments": MAX_SEGMENTS},
        "results": all_results,
        "gate": {
            "checkpoint_unchanged": after == INITIAL_SHA, "sealed_test_read": False,
            "no_candidate_flood": all(v["summary"]["candidate_per_image"] <= 12 for v in all_results.values()),
            "cpu_probe_only": True, "training_admissible": False, "gpu_go": False, "production_go": False,
        },
        "elapsed_seconds": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "summaries": {k: v["summary"] for k, v in all_results.items()}, "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
