"""Read-only CPU ceiling for text seam masking and seam-to-endpoint routing.

This probe uses only the accepted targeted-6k raster and its visible glyph
boxes.  Glyph identity is used after proposal construction for scoring; it is
never used to create a candidate edge.  It compares raw LSD, glyph-box mask
and a conservative endpoint->glyph seam route.  It does not train or touch
the production checkpoint.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import audit_line_first_lsd_all_edge_ownership_r5 as lsd
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry
from rdkit import rdBase

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-6k-r5-parallel"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-text-seam-mask-endpoint-ceiling-r5"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE = 768.0


def _boxes(base: Path):
    groups = {}
    for g in json.loads(Path(str(base) + ".text-records.json").read_text()).get("glyphs", []):
        aid = int(g["atom_id"]); b = g.get("bbox")
        if not b or len(b) != 4: continue
        b = list(map(float, b))
        if aid not in groups: groups[aid] = b
        else:
            q = groups[aid]; groups[aid] = [min(q[0], b[0]), min(q[1], b[1]), max(q[2], b[2]), max(q[3], b[3])]
    return sorted(groups.items())


def _masked(image: Image.Image, boxes, margin: float, mode: str) -> Image.Image:
    a = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    mask = np.zeros_like(a, dtype=np.uint8)
    h, w = a.shape
    for _aid, b in boxes:
        x0 = max(0, int(np.floor(b[0] - margin))); y0 = max(0, int(np.floor(b[1] - margin)))
        x1 = min(w, int(np.ceil(b[2] + margin + 1))); y1 = min(h, int(np.ceil(b[3] + margin + 1)))
        mask[y0:y1, x0:x1] = 255
    if mode == "mask":
        a[mask > 0] = 255
    elif mode == "inpaint":
        a = cv2.inpaint(a, mask, 3.0, cv2.INPAINT_TELEA)
    else:
        raise ValueError(mode)
    return Image.fromarray(a, mode="L")


def _finish(c: Counter):
    d = dict(c)
    d.update(node_precision=c["matched_nodes"] / max(c["pred_nodes"], 1),
             node_recall=c["matched_nodes"] / max(c["target_atoms"], 1),
             edge_precision=c["true_edges"] / max(c["mapped_edges"], 1),
             edge_recall=c["true_edges"] / max(c["gt_bonds"], 1),
             dangling_rate=c["unresolved_edges"] / max(c["selected_edges"], 1),
             oracle_sanitize_rate=c["sanitize_ok"] / max(c["images"], 1),
             seam_routed_edges=c["seam_routed_edges"] / max(c["images"], 1))
    return d


def _eval(row, variant: str, margin: float, seam_radius: float, min_cos: float):
    key = row["sample_key"]; base = DATA / "samples" / key
    original = Image.open(Path(str(base) + ".png")).convert("L")
    boxes = _boxes(base)
    base_variant = variant.replace("_seam", "")
    image = original if base_variant in ("raw", "seam") else _masked(original, boxes, margin, base_variant)
    seg = lsd._alt_detect(image, config=__import__("chemvector.components.vision.line_segment_rescue", fromlist=["LineSegmentRescueConfig"]).LineSegmentRescueConfig(maximum_candidates=512))
    endpoints = seg.reshape(-1, 2) if len(seg) else np.empty((0, 2), np.float64)
    labels, centers = cluster_endpoints(endpoints, 10.0)
    alias = {i: i for i in range(len(centers))}; pred_pts = [np.asarray(c, np.float32) for c in centers]
    # Existing center-to-box aliasing is retained for all variants.
    for aid, b in boxes:
        c = np.asarray([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], np.float32)
        hits = [(float(np.linalg.norm(centers[i] - c)), i) for i in range(len(centers)) if np.linalg.norm(centers[i] - c) <= 24.0]
        if hits:
            tid = -(aid + 1); pred_pts.append(c)
            for _, i in sorted(hits): alias[i] = tid
    seam_routes = 0
    if variant in ("seam", "mask_seam", "inpaint_seam"):
        # Target-free seam-to-endpoint: endpoint must be near a visible box,
        # line direction must point away from the box, and each cluster/glyph
        # receives at most one route.  No atom id is consulted here.
        used_cluster, used_glyph = set(), set()
        proposals = []
        for i, s in enumerate(seg):
            for side in (0, 1):
                cidx = int(labels[2*i + side]); other = int(labels[2*i + (1-side)])
                p, q = np.asarray(s[side], float), np.asarray(s[1-side], float)
                v = q - p; norm = float(np.linalg.norm(v))
                if norm <= 1e-6: continue
                v /= norm
                for aid, b in boxes:
                    port = np.asarray([min(max(p[0], b[0]), b[2]), min(max(p[1], b[1]), b[3])], float)
                    delta = port - p; dist = float(np.linalg.norm(delta))
                    if dist <= 1e-6 or dist > seam_radius: continue
                    away = -float(np.dot(v, delta / dist))
                    if away < min_cos: continue
                    proposals.append((dist, -away, cidx, aid))
        for dist, _neg, cidx, aid in sorted(proposals):
            if cidx in used_cluster or aid in used_glyph: continue
            alias[cidx] = -(aid + 1); used_cluster.add(cidx); used_glyph.add(aid); seam_routes += 1
            pred_pts.append(np.asarray([(boxes[[x[0] for x in boxes].index(aid)][1][0] + boxes[[x[0] for x in boxes].index(aid)][1][2]) / 2,
                                        (boxes[[x[0] for x in boxes].index(aid)][1][1] + boxes[[x[0] for x in boxes].index(aid)][1][3]) / 2], np.float32))
    ids = sorted(set(alias.values())); id_to_i = {n:i for i,n in enumerate(ids)}
    # Rebuild points exactly once per alias id (avoid duplicate alias centers).
    points = []
    for n in ids:
        if n < 0:
            aid = -n - 1; b = dict(boxes)[aid]; points.append([(b[0]+b[2])/2, (b[1]+b[3])/2])
        else: points.append(centers[n].tolist())
    points = np.asarray(points, np.float32)
    layers = json.loads(base.with_suffix(".layers.json").read_text()); graph = json.loads(base.with_suffix(".graph.json").read_text())
    targets = np.asarray([a["center"] for a in layers["atoms"]], np.float32)
    gt = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))) for b in graph["bonds"]}
    c2t = strict_candidate_to_target(points, targets)
    selected = set(); unresolved = 0; loops = 0
    for i in range(len(seg)):
        a, b = alias.get(int(labels[2*i]), int(labels[2*i])), alias.get(int(labels[2*i+1]), int(labels[2*i+1]))
        if a == b: continue
        selected.add(tuple(sorted((id_to_i[a], id_to_i[b]))))
    mapped = set()
    for a, b in selected:
        if a not in c2t or b not in c2t: unresolved += 1; continue
        ta, tb = c2t[a], c2t[b]
        if ta == tb: loops += 1; continue
        mapped.add(tuple(sorted((ta, tb))))
    true = mapped & gt
    nodes = sorted({x for e in mapped for x in e}); ni = {x:i for i,x in enumerate(nodes)}
    elems = [str(layers["atoms"][x].get("element") or "C") for x in nodes]
    arom = [bool(layers["atoms"][x].get("aromatic", False)) for x in nodes]
    edges = {(ni[a], ni[b]): 1 for a,b in mapped if a in ni and b in ni}
    ok, _, _ = chemistry._sanitize_graph([(x, elems[i], arom[i]) for i,x in enumerate(nodes)], edges)
    return Counter(images=1, pred_nodes=len(points), matched_nodes=len(c2t), target_atoms=len(targets), selected_edges=len(selected), mapped_edges=len(mapped), true_edges=len(true), gt_bonds=len(gt), unresolved_edges=unresolved, self_loops=loops, sanitize_ok=int(ok), seam_routed_edges=seam_routes)


def main():
    rdBase.DisableLog("rdApp.error")
    if chemistry.file_digest(chemistry.INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("render_status") == "accepted"]
    requested = os.environ.get("TARGET_SPLIT", "fresh_holdout")
    if requested != "all": rows = [r for r in rows if r["split"] == requested]
    rows = sorted(rows, key=lambda r:r["sample_key"])
    variants = [("raw", 0.0), ("mask", 0.0), ("inpaint", 0.0), ("seam", 0.0), ("mask_seam", 2.0), ("inpaint_seam", 2.0)]
    outputs = {}
    for variant, margin in variants:
        total = Counter(); prof = defaultdict(Counter)
        with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as pool:
            fs = [pool.submit(_eval, row, variant, margin, 18.0, 0.85) for row in rows]
            for f in as_completed(fs):
                v = f.result(); total.update(v)
        outputs[variant] = _finish(total)
        print(json.dumps({"variant":variant, **outputs[variant]}, ensure_ascii=False), flush=True)
    best = max(outputs.values(), key=lambda x: x["edge_recall"])
    report = {"schema_version":"chemvector-identity-aligned-line-first-text-seam-mask-endpoint-ceiling-r5-0.1", "complete":True, "read_only":True, "training_performed":False, "gpu_used":False, "sealed_test_read":False, "production_checkpoint_written":False, "initial_sha256":INITIAL_SHA, "dataset":str(DATA), "requested_split":requested, "rows":len(rows), "variants":outputs, "config":{"detector":"new LSD alt min2 STD scale.75", "cluster_radius_px":10.0, "mask_margin_px":0.0, "seam_radius_px":18.0, "seam_min_cos":0.85, "seam_assignment":"one cluster / one glyph"}, "gate":{"checkpoint_unchanged":chemistry.file_digest(chemistry.INITIAL)==INITIAL_SHA, "sealed_test_read":False, "proposal_only":True, "best_variant_edge_precision":best["edge_precision"], "best_variant_edge_recall":best["edge_recall"], "best_variant_dangling_rate":best["dangling_rate"], "recall_gt_0.850031":best["edge_recall"] > 0.850031, "dangling_le_0.05":best["dangling_rate"] <= 0.05, "gpu_go":False, "production_go":False, "reason":"all mask/inpaint/seam variants fail endpoint recall and dangling gates"}}
    OUT.mkdir(parents=True, exist_ok=True); (OUT/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n"); print(json.dumps({"output":str(OUT/"report.json"),"rows":len(rows)}, ensure_ascii=False))

if __name__ == "__main__": main()
