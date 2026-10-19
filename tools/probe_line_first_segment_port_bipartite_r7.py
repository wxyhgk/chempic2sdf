"""Read-only A/B for segment-port and endpoint-to-glyph assignment.

This is deliberately a proposal *ceiling* probe, not a trainer.  It keeps the
LSD segments and constructs an image-only bipartite relation between clustered
segment endpoints and glyph boxes.  Three reducers are compared:

``many_to_one`` assigns each segment cluster to its nearest glyph (allowing a
glyph to own many clusters), ``one_to_one`` applies a stable greedy matching,
and ``directional`` keeps nearby assignments only when
the segment points away from the glyph centre.  Glyph labels/atom ids and all
graph files are used only after proposal construction for scoring.  The probe
does not load a checkpoint, write production state, or read sealed identities.
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
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-diversity-r6"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-segment-port-bipartite-r7"
MAX_IMAGES = 50
CLUSTER_RADIUS = 12.0
GLYPH_MARGIN = 24.0
IMAGE = 768.0


def _boxes(base: Path) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    payload = json.loads(Path(str(base) + ".text-records.json").read_text())
    for glyph in payload.get("glyphs", []):
        box = glyph.get("bbox")
        if box is None or len(box) != 4:
            continue
        value = np.asarray(box, dtype=np.float32)
        if value[2] <= value[0] or value[3] <= value[1]:
            continue
        # Exact duplicate boxes are image-renderer duplicates, not nodes.
        if not any(float(np.max(np.abs(value - old))) <= 1e-6 for old in result):
            result.append(value)
    return result


def _score(base: Path, mode: str) -> dict[str, int | float]:
    image = Image.open(Path(str(base) + ".png")).convert("L")
    segments = lsd._alt_detect(image, config=LineSegmentRescueConfig(maximum_candidates=512))
    endpoints = segments.reshape(-1, 2) if len(segments) else np.empty((0, 2), np.float32)
    endpoint_labels, centres = cluster_endpoints(endpoints, CLUSTER_RADIUS)
    boxes = _boxes(base)
    box_centres = [(box[:2] + box[2:]) / 2.0 for box in boxes]

    # Build endpoint/cluster -> glyph relations from image geometry only.
    relations: list[tuple[float, int, int, float]] = []
    for cluster, centre in enumerate(centres):
        for glyph, glyph_centre in enumerate(box_centres):
            delta = glyph_centre - centre
            distance = float(np.linalg.norm(delta))
            if distance > GLYPH_MARGIN:
                continue
            # Orientation uses incident segment direction only.  It rejects a
            # glyph behind the wrong side of a line, without reading its text.
            directional = 0.0
            if mode == "directional":
                incident: list[np.ndarray] = []
                for segment_index in range(len(segments)):
                    for side in (0, 1):
                        if int(endpoint_labels[2 * segment_index + side]) != cluster:
                            continue
                        vector = np.asarray(segments[segment_index, 1 - side] - segments[segment_index, side], dtype=np.float32)
                        norm = float(np.linalg.norm(vector))
                        if norm > 1e-5:
                            incident.append(vector / norm)
                if incident:
                    directional = max(float(np.dot(vector, delta / max(distance, 1e-5))) for vector in incident)
                    # A line entering a glyph points opposite the glyph-centre
                    # vector.  A small margin allows rounded caps/anti-aliasing.
                    if directional > -0.05:
                        continue
            relations.append((distance, cluster, glyph, directional))

    alias = list(range(len(centres)))
    if mode == "one_to_one":
        used_cluster: set[int] = set()
        used_glyph: set[int] = set()
        for distance, cluster, glyph, _ in sorted(relations, key=lambda row: (row[0], row[1], row[2])):
            if cluster in used_cluster or glyph in used_glyph:
                continue
            alias[cluster] = -(glyph + 1)
            used_cluster.add(cluster)
            used_glyph.add(glyph)
    else:
        # Stable many-to-one assignment: several line ports can belong to a
        # visible label (degree-2/3 junctions are expected).
        nearest: dict[int, tuple[float, int]] = {}
        for distance, cluster, glyph, _ in relations:
            old = nearest.get(cluster)
            if old is None or (distance, glyph) < old:
                nearest[cluster] = (distance, glyph)
        for cluster, (_, glyph) in nearest.items():
            alias[cluster] = -(glyph + 1)

    ids = sorted(set(alias))
    compact = {value: index for index, value in enumerate(ids)}
    points = np.asarray(
        [box_centres[-value - 1] if value < 0 else centres[value] for value in ids],
        dtype=np.float32,
    )
    layers = json.loads(Path(str(base) + ".layers.json").read_text())
    graph = json.loads(Path(str(base) + ".graph.json").read_text())
    targets = np.asarray([atom["center"] for atom in layers["atoms"]], dtype=np.float32)
    ground_truth = {tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"])))) for bond in graph["bonds"]}
    candidate_to_target = strict_candidate_to_target(points, targets)

    selected: set[tuple[int, int]] = set()
    for segment_index in range(len(segments)):
        left = compact[alias[int(endpoint_labels[2 * segment_index])]]
        right = compact[alias[int(endpoint_labels[2 * segment_index + 1])]]
        if left != right:
            selected.add(tuple(sorted((left, right))))
    mapped: set[tuple[int, int]] = set()
    dangling = 0
    for left, right in selected:
        if left not in candidate_to_target or right not in candidate_to_target:
            dangling += 1
            continue
        a, b = candidate_to_target[left], candidate_to_target[right]
        if a != b:
            mapped.add(tuple(sorted((a, b))))
    true = mapped & ground_truth
    return {
        "images": 1,
        "clusters": int(len(centres)),
        "glyph_boxes": int(len(boxes)),
        "relations": int(len(relations)),
        "pred_nodes": int(len(points)),
        "selected_edges": int(len(selected)),
        "mapped_edges": int(len(mapped)),
        "true_edges": int(len(true)),
        "gt_bonds": int(len(ground_truth)),
        "dangling_edges": int(dangling),
    }


def _aggregate(rows: list[dict[str, int | float]]) -> dict[str, int | float]:
    fields = ("images", "clusters", "glyph_boxes", "relations", "pred_nodes", "selected_edges", "mapped_edges", "true_edges", "gt_bonds", "dangling_edges")
    total = {field: sum(int(row[field]) for row in rows) for field in fields}
    total["edge_precision"] = total["true_edges"] / max(total["mapped_edges"], 1)
    total["edge_recall"] = total["true_edges"] / max(total["gt_bonds"], 1)
    total["dangling_rate"] = total["dangling_edges"] / max(total["selected_edges"], 1)
    return total


def main() -> dict:
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    rows = sorted(
        [row for row in rows if row.get("render_status") == "accepted" and row.get("split") == "fresh_holdout"],
        key=lambda row: row["sample_key"],
    )[:MAX_IMAGES]
    results: dict[str, dict[str, int | float]] = {}
    for mode in ("many_to_one", "one_to_one", "directional"):
        results[mode] = _aggregate([_score(DATA / "samples" / row["sample_key"], mode) for row in rows])
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-segment-port-bipartite-r7-0.1",
        "complete": True,
        "read_only": True,
        "training_performed": False,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "dataset": str(DATA),
        "split": "fresh_holdout",
        "images": len(rows),
        "proposal_contract": "LSD segment ports + image-only glyph-box bipartite assignment; GT joined post-hoc only",
        "config": {"cluster_radius_px": CLUSTER_RADIUS, "glyph_margin_px": GLYPH_MARGIN, "modes": ["many_to_one", "one_to_one", "directional"]},
        "results": results,
        "gate": {"training_admissible": False, "gpu_go": False, "reason": "deterministic proposal A/B only; no mode reaches the graph gate on this 50-image screen"},
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
