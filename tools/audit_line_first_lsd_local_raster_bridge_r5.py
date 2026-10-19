"""CPU feasibility gate for a local raster bridge at the new-LSD seam.

When endpoint clustering collapses or misses an atom/text seam, this probe
adds a candidate-pair edge only if both frozen candidates are close to raster
segment endpoints and the candidate-to-candidate line has target-free ink
support. Candidate identities and scores are unchanged. The output is a
proposal ceiling (synthetic SINGLE edges), not a deployable bond decoder.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from rdkit import rdBase
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry  # noqa: E402
from tools import audit_line_first_lsd_all_edge_ownership_r5 as new_lsd  # noqa: E402
from tools import audit_line_first_valence_topology_guard_r5 as guard  # noqa: E402
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints  # noqa: E402
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig, detect_raster_line_segments  # noqa: E402

DATA = chemistry.DATA
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-lsd-local-raster-bridge-r5/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE, CLUSTER, OWNERSHIP_RADIUS = 768.0, 10.0, 24.0
BRIDGE_DISTANCE = (30.0, 100.0); ENDPOINT_RADIUS = 24.0
MIN_INK_MEAN, MIN_INK_FRACTION, MIN_CANDIDATE_SCORE = .20, .50, .20


def _support(gray, begin, end):
    length = float(np.linalg.norm(end - begin)); count = max(2, int(round(length)))
    ts = np.linspace(0.0, 1.0, count)
    points = begin[None] * (1.0 - ts[:, None]) + end[None] * ts[:, None]
    values = []
    for x, y in points:
        xx, yy = int(round(x)), int(round(y))
        values.append(float((255.0 - gray[max(0, yy - 1):yy + 2, max(0, xx - 1):xx + 2]).mean() / 255.0))
    values = np.asarray(values)
    return float(values.mean()), float((values > .12).mean())


def _assign(centers, points):
    if not len(centers) or not len(points): return {}
    d = np.linalg.norm(centers[:, None] - points[None, :, :2], axis=-1)
    cost = np.where(d <= OWNERSHIP_RADIUS, d - 8.0 * points[None, :, 2], 1e5 + d)
    rr, cc = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rr, cc) if d[r, c] <= OWNERSHIP_RADIUS}


def _finish(c):
    return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1), "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1), "oracle_sanitize_rate": c["sanitize_ok"] / max(c["images"], 1), "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1)}


def _evaluate(row, candidates, bridge: bool):
    key = row["sample_key"]; sample = DATA / "samples" / key
    image = Image.open(sample.with_suffix(".png")).convert("L"); gray = np.asarray(image, dtype=np.uint8)
    segments = detect_raster_line_segments(image, config=LineSegmentRescueConfig(maximum_candidates=512))
    points = segments.reshape(-1, 2) if len(segments) else np.empty((0, 2)); labels, centers = cluster_endpoints(points, CLUSTER)
    frozen = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE, v.get("score", 0.0)] for v in candidates[key]], dtype=np.float64)
    owners = _assign(centers, frozen)
    provisional = {tuple(sorted((int(labels[2 * i]), int(labels[2 * i + 1])))) for i in range(len(segments)) if labels[2 * i] != labels[2 * i + 1]}
    selected = {tuple(sorted((owners[a], owners[b]))): (1.0, 1) for a, b in provisional if a in owners and b in owners and owners[a] != owners[b]}
    # Nearest local ownership is used only to suppress duplicate bridge pairs.
    distances = np.linalg.norm(centers[:, None] - frozen[None, :, :2], axis=-1) if len(centers) and len(frozen) else np.empty((len(centers), len(frozen)))
    nearest = {int(a): int(np.argmin(distances[a])) for a in range(len(centers)) if len(frozen) and distances[a].min() <= 40.0}
    existing_near = {tuple(sorted((nearest[a], nearest[b]))) for a, b in provisional if a in nearest and b in nearest and nearest[a] != nearest[b]}
    added = 0
    if bridge and len(segments):
        endpoint_distances = np.linalg.norm(points[:, None] - frozen[None, :, :2], axis=-1)
        for a in range(len(frozen)):
            for b in range(a + 1, len(frozen)):
                if min(frozen[a, 2], frozen[b, 2]) < MIN_CANDIDATE_SCORE: continue
                pair = (a, b); distance = float(np.linalg.norm(frozen[a, :2] - frozen[b, :2]))
                if not BRIDGE_DISTANCE[0] <= distance <= BRIDGE_DISTANCE[1] or pair in existing_near or pair in selected: continue
                if float(endpoint_distances[:, a].min()) > ENDPOINT_RADIUS or float(endpoint_distances[:, b].min()) > ENDPOINT_RADIUS: continue
                mean_ink, fraction = _support(gray, frozen[a, :2], frozen[b, :2])
                if mean_ink < MIN_INK_MEAN or fraction < MIN_INK_FRACTION: continue
                selected[pair] = (1.0, 1); added += 1
    selected, _ = guard.guard(selected, low_confidence=.4)
    layers = json.loads(sample.with_suffix(".layers.json").read_text()); graph = json.loads(sample.with_suffix(".graph.json").read_text())
    targets = np.asarray([a["center"] for a in layers["atoms"]], dtype=np.float32); c2t = chemistry.strict_candidate_to_target(frozen[:, :2].astype(np.float32), targets)
    gt = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))) for b in graph["bonds"]}; mapped = {tuple(sorted((c2t[a], c2t[b]))) for a, b in selected if a in c2t and b in c2t and c2t[a] != c2t[b]}; true = mapped & gt
    node_ids = sorted({z for p in selected for z in p}); nodes = [(c, str(layers["atoms"][t].get("element") or "C"), bool(layers["atoms"][t].get("aromatic", False))) for c, t in sorted(c2t.items()) if c in node_ids]; ni = {c: i for i, (c, _, _) in enumerate(nodes)}; edges = {(ni[a], ni[b]): 1 for a, b in selected if a in ni and b in ni}; ok, _, _ = chemistry._sanitize_graph(nodes, edges)
    return Counter(images=1, selected_edges=len(selected), dangling_edges=sum(1 for a, b in provisional if a not in owners or b not in owners), mapped_edges=len(mapped), true_edges=len(true), gt_bonds=len(gt), sanitize_ok=int(ok), added_bridge_edges=added)


def main():
    rdBase.DisableLog("rdApp.error")
    if chemistry.file_digest(chemistry.INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    # Install the new-LSD detector only in this process.
    new_lsd._install()
    # This module imported the detector symbol directly; redirect that local
    # reference as well so the probe really uses min2/STD/scale.75/r10.
    globals()["detect_raster_line_segments"] = new_lsd._alt_detect
    rows = sorted([json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("render_status") == "accepted"], key=lambda x: x["sample_key"]); candidates = new_lsd.hybrid.candidate_rows()
    outputs = {}
    for bridge in (False, True):
        split_totals, profiles = defaultdict(Counter), defaultdict(Counter)
        for row in rows:
            value = _evaluate(row, candidates, bridge); split_totals[row["split"]].update(value); profiles[f"{row['split']}:{row['profile_id']}"].update(value)
        outputs["bridge" if bridge else "baseline"] = {"by_split": {k: _finish(v) for k, v in sorted(split_totals.items())}, "by_profile": {k: _finish(v) for k, v in sorted(profiles.items())}}
    hold = outputs["bridge"]["by_split"]["holdout"]; hp = {k: v for k, v in outputs["bridge"]["by_profile"].items() if k.startswith("holdout:")}
    gate = {"checkpoint_unchanged": chemistry.file_digest(chemistry.INITIAL) == INITIAL_SHA, "sealed_test_read": False, "holdout_precision_ge_0.95": hold["edge_precision"] >= .95, "holdout_recall_ge_0.80": hold["edge_recall"] >= .80, "all_profiles_recall_ge_0.75": all(v["edge_recall"] >= .75 for v in hp.values()), "all_profiles_sanitize_ge_0.90": all(v["oracle_sanitize_rate"] >= .90 for v in hp.values()), "production_go": False, "gpu_go": False}
    report = {"schema_version": "chemvector-identity-aligned-line-first-lsd-local-raster-bridge-r5-0.1", "complete": True, "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "initial_sha256": INITIAL_SHA, "proposal_contract": "new LSD min2 STD scale.75 cluster r10 -> frozen candidate ownership -> target-free local candidate-pair raster support bridge (synthetic SINGLE ceiling only)", "config": {"minimum_length_px": 2.0, "refine": "STD", "scale": .75, "cluster_radius_px": CLUSTER, "ownership_radius_px": OWNERSHIP_RADIUS, "bridge_distance_px": BRIDGE_DISTANCE, "endpoint_radius_px": ENDPOINT_RADIUS, "min_ink_mean": MIN_INK_MEAN, "min_ink_fraction": MIN_INK_FRACTION, "min_candidate_score": MIN_CANDIDATE_SCORE}, "outputs": outputs, "holdout": hold, "holdout_profiles": hp, "gate": gate, "interpretation": {"text_occlusion": "The bridge uses raster support only; it does not infer text semantics. Text-occluded edges with weak line support remain unresolved and require a separate text-mask/inpainting probe.", "synthetic_type": "All added edges are SINGLE solely for topology ceiling; bond-type accuracy is not claimed."}}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n"); print(json.dumps({"output": str(OUT), "holdout": hold, "gate": gate}, ensure_ascii=False))


if __name__ == "__main__": main()
