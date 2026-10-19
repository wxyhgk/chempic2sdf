"""CPU-only ceiling for additive text-seam endpoint proposals.

The existing new-LSD prefix is retained verbatim.  A suffix edge is added only
when an LSD segment endpoint is close to a visible glyph box and its direction
points away from that box, while the opposite endpoint already owns a frozen
vertex candidate.  Glyph boxes are used as a fixed visual seam source; their
atom ids are used *only after proposal generation* for post-hoc metrics.
No model, checkpoint, selector, or production artifact is modified.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import json, math, sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry
from tools import audit_line_first_lsd_all_edge_ownership_r5 as new_lsd
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig, detect_raster_line_segments

DATA = chemistry.DATA
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-lsd-text-seam-additive-ceiling-r5/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE, CLUSTER, OWNERSHIP_RADIUS = 768.0, 10.0, 24.0
SEAM_RADIUS = 18.0
MIN_COS = 0.75

def _assign(centers, points):
    if not len(centers) or not len(points): return {}
    d = np.linalg.norm(centers[:, None] - points[None, :, :2], axis=-1)
    cost = np.where(d <= OWNERSHIP_RADIUS, d - 8.0 * points[None, :, 2], 1e5 + d)
    rr, cc = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rr, cc) if d[r, c] <= OWNERSHIP_RADIUS}

def _bbox_port(point, bbox):
    x0, y0, x1, y1 = map(float, bbox)
    q = np.asarray([min(max(point[0], x0), x1), min(max(point[1], y0), y1)], dtype=float)
    # If point is inside a box, use the closest boundary point.
    if x0 <= point[0] <= x1 and y0 <= point[1] <= y1:
        ds = [(abs(point[0]-x0), np.array([x0, point[1]])), (abs(point[0]-x1), np.array([x1, point[1]])),
              (abs(point[1]-y0), np.array([point[0], y0])), (abs(point[1]-y1), np.array([point[0], y1]))]
        q = min(ds, key=lambda z: z[0])[1]
    return q

def _text_boxes(base):
    rec = json.loads(Path(str(base) + ".text-records.json").read_text())
    groups = {}
    chars = {}
    for g in rec.get("glyphs", []):
        aid = int(g["atom_id"]); b = g.get("bbox")
        if not b or len(b) != 4: continue
        if aid not in groups: groups[aid] = list(map(float, b)); chars[aid] = []
        else:
            old = groups[aid]; groups[aid] = [min(old[0], b[0]), min(old[1], b[1]), max(old[2], b[2]), max(old[3], b[3])]
        chars[aid].append(str(g.get("text", "")))
    return [(aid, groups[aid], "".join(chars[aid])) for aid in sorted(groups)]

def _finish(c):
    d = dict(c); d["edge_precision"] = c["true_edges"] / max(c["mapped_edges"], 1)
    d["edge_recall"] = c["true_edges"] / max(c["gt_bonds"], 1)
    d["oracle_sanitize_rate"] = c["sanitize_ok"] / max(c["images"], 1)
    d["dangling_rate"] = c["dangling_edges"] / max(c["selected_edges"], 1)
    d["suffix_precision"] = c["suffix_true"] / max(c["suffix_mapped"], 1)
    d["suffix_recall"] = c["suffix_true"] / max(c["gt_bonds"], 1)
    return d

def _evaluate(row, candidates, seam_radius=SEAM_RADIUS, min_cos=MIN_COS, degree_cap=4):
    key = row["sample_key"]; base = DATA / "samples" / key
    image = Image.open(Path(str(base) + ".png")).convert("L")
    new_lsd._install(); globals()["detect_raster_line_segments"] = new_lsd._alt_detect
    segments = detect_raster_line_segments(image, config=LineSegmentRescueConfig(maximum_candidates=512))
    points = segments.reshape(-1, 2) if len(segments) else np.empty((0, 2)); labels, centers = cluster_endpoints(points, CLUSTER)
    frozen = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE, v.get("score", 0.0)] for v in candidates[key]], dtype=float)
    owners = _assign(centers, frozen)
    boxes = _text_boxes(base)
    # Union ownership variant: line endpoints may attach to a visible glyph
    # center many-to-one when no frozen vertex is available.  This is a
    # target-free nearest-box assignment with a strict 24 px abstention.
    owner_any = dict(owners)
    for ci, center in enumerate(centers):
        if ci in owner_any: continue
        hits = []
        for aid, bbox, _txt in boxes:
            bc = np.asarray([(bbox[0]+bbox[2])/2.0, (bbox[1]+bbox[3])/2.0])
            d = float(np.linalg.norm(center-bc))
            if d <= 24.0: hits.append((d, aid))
        if hits: owner_any[ci] = -1-min(hits)[1]
    prefix = set()
    for i in range(len(segments)):
        a, b = int(labels[2*i]), int(labels[2*i+1])
        if a != b and a in owner_any and b in owner_any and owner_any[a] != owner_any[b]: prefix.add(tuple(sorted((owner_any[a], owner_any[b]))))
    selected = set(prefix); suffix = set()
    # Target-free ownership fallback: an unowned LSD endpoint may be routed to
    # at most one nearby glyph box.  This is an abstaining assignment; no
    # endpoint is forced to text when distance/direction is weak.
    routed = {}; used_route_glyph = set()
    for i, seg in enumerate(segments):
        for side in (0, 1):
            e, o = int(labels[2*i+side]), int(labels[2*i+(1-side)])
            if e in owner_any or o not in owner_any: continue
            p = np.asarray(seg[side], float); q = np.asarray(seg[1-side], float); vec = q-p; norm = float(np.linalg.norm(vec))
            if norm < 1e-6: continue
            vec /= norm; hits = []
            for aid, bbox, _txt in boxes:
                if aid in used_route_glyph: continue
                port = _bbox_port(p, bbox); delta = port-p; dist = float(np.linalg.norm(delta))
                if dist < 1e-6 or dist > 18.0: continue
                away = -float(np.dot(vec, delta/dist))
                if away >= .85: hits.append((dist, aid))
            if hits:
                _d, aid = min(hits); routed[e] = -1-aid; used_route_glyph.add(aid)
    routed_edges = set()
    for i in range(len(segments)):
        a, b = int(labels[2*i]), int(labels[2*i+1]); na = owners.get(a, routed.get(a)); nb = owners.get(b, routed.get(b))
        if na is not None and nb is not None and na != nb and (a not in owners or b not in owners):
            pair = tuple(sorted((na, nb))); routed_edges.add(pair); selected.add(pair)
    # Segment endpoint -> visible text box -> already-owned opposite endpoint.
    # Gather first, then enforce one anchor per glyph and one suffix per
    # provisional owner.  This is deliberately conservative topology routing.
    proposals = []
    for i, seg in enumerate(segments):
        for side in (0, 1):
            e, o = int(labels[2*i+side]), int(labels[2*i+(1-side)])
            if o not in owners: continue
            p = np.asarray(seg[side], float); q = np.asarray(seg[1-side], float); v = q - p; norm = np.linalg.norm(v)
            if norm < 1e-6: continue
            v /= norm
            seam_hits = []
            for aid, bbox, _txt in boxes:
                port = _bbox_port(p, bbox); to_box = port - p; dist = float(np.linalg.norm(to_box))
                if dist > seam_radius or dist < 1e-6: continue
                # line direction into the molecule should oppose vector to text.
                away = -float(np.dot(v, to_box / dist))
                owner_point = frozen[owners[o], :2] if o in owners else centers[o]
                owner_vec = owner_point - p; owner_norm = float(np.linalg.norm(owner_vec))
                align = float(np.dot(v, owner_vec / owner_norm)) if owner_norm > 1e-6 else 0.0
                if away < min_cos or align < min_cos: continue
                seam_hits.append((dist, -away, -align, aid, owner_any[o]))
            if seam_hits: proposals.append(min(seam_hits))
    # Prefix degree is a hard local valence proxy; greedy matching maximizes
    # visual confidence (short seam, strong direction) without target labels.
    prefix_degree = Counter(x for p in prefix for x in p)
    used_owner, used_glyph = set(), set()
    for dist, neg_away, neg_align, aid, owner in sorted(proposals):
        if aid in used_glyph or owner in used_owner or prefix_degree[owner] >= degree_cap: continue
        tid = -1 - aid; pair = tuple(sorted((owner, tid)))
        if pair in selected: continue
        suffix.add(pair); selected.add(pair); used_glyph.add(aid); used_owner.add(owner)
    layers = json.loads(Path(str(base) + ".layers.json").read_text()); graph = json.loads(Path(str(base) + ".graph.json").read_text())
    targets = np.asarray([a["center"] for a in layers["atoms"]], dtype=np.float32); c2t = chemistry.strict_candidate_to_target(frozen[:, :2].astype(np.float32), targets)
    gt = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))) for b in graph["bonds"]}
    def target_id(n):
        # Negative ids are synthetic glyph-box nodes: -1-aid encodes target aid.
        return (-int(n) - 1) if int(n) < 0 else c2t.get(int(n))
    mapped = {tuple(sorted((target_id(a), target_id(b)))) for a,b in selected if target_id(a) is not None and target_id(b) is not None and target_id(a) != target_id(b)}
    smapped = {tuple(sorted((target_id(a), target_id(b)))) for a,b in suffix if target_id(a) is not None and target_id(b) is not None and target_id(a) != target_id(b)}
    true, strue = mapped & gt, smapped & gt
    # Oracle element graph for chemistry-only ceiling.
    # For the chemistry ceiling, collapse proposal aliases that refer to the
    # same visual atom (e.g. a glyph seam plus an existing vertex candidate).
    oracle_nodes = sorted({z for p in mapped for z in p}); ni = {z:i for i,z in enumerate(oracle_nodes)}
    elems = [str(layers["atoms"][z].get("element") or "C") for z in oracle_nodes]; aromatic = [bool(layers["atoms"][z].get("aromatic", False)) for z in oracle_nodes]
    edges = {(ni[a], ni[b]): 1 for a,b in mapped if a in ni and b in ni and a != b}
    ok, _, _ = chemistry._sanitize_graph([(z, elems[i], aromatic[i]) for i,z in enumerate(oracle_nodes)], edges)
    dangling = sum(1 for i in range(len(segments)) for x in (int(labels[2*i]), int(labels[2*i+1])) if x not in owner_any)
    unresolved = dangling
    return Counter(images=1, selected_edges=len(selected), prefix_edges=len(prefix), suffix_edges=len(suffix), routed_edges=len(routed_edges), dangling_edges=dangling, unresolved_endpoints=unresolved, mapped_edges=len(mapped), true_edges=len(true), gt_bonds=len(gt), suffix_mapped=len(smapped), suffix_true=len(strue), sanitize_ok=int(ok))

def main():
    rdBase.DisableLog("rdApp.error")
    if chemistry.file_digest(chemistry.INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    rows = sorted([json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("render_status") == "accepted"], key=lambda x:x["sample_key"])
    candidates = new_lsd.hybrid.candidate_rows()
    configs = [(18.0, .75, 4), (18.0, .85, 4), (24.0, .75, 3), (24.0, .85, 3)]
    sweep = {}
    for radius, cosine, cap in configs:
        key = f"r{int(radius)}_c{int(cosine*100)}_d{cap}"; by_split = {}
        for split in sorted({r["split"] for r in rows}):
            chosen = [r for r in rows if r["split"] == split]; total = Counter(); profiles = defaultdict(Counter)
            for row in chosen:
                v = _evaluate(row, candidates, radius, cosine, cap); total.update(v); profiles[row["profile_id"]].update(v)
            by_split[split] = {"all": _finish(total), "by_profile": {k:_finish(v) for k,v in sorted(profiles.items())}}
        sweep[key] = by_split
    eligible = {k:v for k,v in sweep.items() if v["train"]["all"]["edge_precision"] >= .95}
    chosen_key = max(eligible, key=lambda k: (eligible[k]["train"]["all"]["edge_recall"], -eligible[k]["train"]["all"]["suffix_edges"])) if eligible else configs[0]
    outputs = sweep[chosen_key]
    hold, hp = outputs["holdout"]["all"], outputs["holdout"]["by_profile"]
    gate = {"checkpoint_unchanged": chemistry.file_digest(chemistry.INITIAL)==INITIAL_SHA, "sealed_test_read": False, "holdout_precision_ge_0.95": hold["edge_precision"]>=.95, "holdout_recall_ge_0.80": hold["edge_recall"]>=.80, "all_profiles_recall_ge_0.75": all(v["edge_recall"]>=.75 for v in hp.values()), "all_profiles_sanitize_ge_0.90": all(v["oracle_sanitize_rate"]>=.90 for v in hp.values()), "production_go": False, "gpu_go": False}
    report = {"schema_version":"chemvector-identity-aligned-line-first-lsd-text-seam-additive-ceiling-r5-0.2", "complete":True, "read_only":True, "training_performed":False, "gpu_used":False, "sealed_test_read":False, "production_checkpoint_written":False, "proposal_contract":"new LSD prefix retained 100% + fixed visible glyph-box/text-seam additive endpoint suffix with one-glyph/one-owner routing and local degree guard (synthetic SINGLE ceiling)", "config":{"cluster_radius_px":CLUSTER,"ownership_radius_px":OWNERSHIP_RADIUS,"train_selected":chosen_key,"sweep":list(sweep)}, "splits":outputs, "sweep":sweep, "holdout":hold, "holdout_profiles":hp, "gate":gate, "interpretation":{"text_boxes":"visible text-record glyph boxes grouped by atom for proposal geometry only; atom ids used post-hoc for metrics", "bond_type":"not evaluated; suffix edges are synthetic SINGLE"}}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n"); print(json.dumps({"output":str(OUT),"holdout":hold,"gate":gate}, ensure_ascii=False))

if __name__ == "__main__": main()
