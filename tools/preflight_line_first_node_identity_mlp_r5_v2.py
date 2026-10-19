#!/usr/bin/env python3
"""Node-level CPU gate with one-to-one target-free line/candidate assignment.

The previous candidate-level probe scored every vertex proposal independently,
which counted duplicate candidates as false nodes.  This version first does
the deterministic target-free provisional-node -> scored-candidate assignment
(global nearest-distance, one candidate per line node), then learns only a
node-level keep score.  GT is read after assignment for labels/metrics.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig, detect_raster_line_segments
from chemvector.training.curriculum.checkpoints import file_digest
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from tools.shadow_line_first_edge_bond_type_r5 import DATA, INITIAL, INITIAL_SHA, MAX_SEGMENTS
from tools.audit_line_first_candidate_alignment_r5 import IMAGE, align_nodes, read_jsonl

CANDIDATES = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5-scored/candidates.jsonl"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-node-identity-mlp-r5-cpu-v3/report.json"
TRAIN_LIMIT = 160
HOLDOUT_LIMIT = 80
ALIGN_RADIUS = 16.0
TARGET_RADIUS = 6.0
AMBIGUITY_RADIUS = 20.0
STEPS = 500
SEED = 20261018


def _ink(gray: np.ndarray, p: np.ndarray) -> float:
    x, y = int(round(float(p[0]))), int(round(float(p[1]))); h, w = gray.shape; x0, x1 = max(0, x - 2), min(w, x + 3); y0, y1 = max(0, y - 2), min(h, y + 3)
    return float((255.0 - gray[y0:y1, x0:x1]).mean() / 255.0) if x1 > x0 and y1 > y0 else 0.0


def _load_candidates():
    result = {}; forbidden = {"targets", "target_indices", "matched", "element", "visible_element", "atom_id", "graph", "bonds", "ground_truth", "label"}
    for line_no, line in enumerate(CANDIDATES.read_text().splitlines(), 1):
        row = json.loads(line)
        if row.get("source") != "shadow" or set(row) & forbidden: raise ValueError(f"candidate contract failed line {line_no}")
        result[row["sample_key"]] = row["candidates"]
    if len(result) != 1170: raise ValueError("expected 1170 candidate rows")
    return result


def _node_rows(sample_key: str, values: list[dict[str, float]]):
    base = DATA / "samples" / sample_key; image = Image.open(base.with_suffix(".png")).convert("L"); gray = np.asarray(image, dtype=np.uint8)
    segments = detect_raster_line_segments(image, config=LineSegmentRescueConfig(maximum_candidates=MAX_SEGMENTS));
    if not len(segments): return np.empty((0, 14), dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
    endpoints = segments.reshape(-1, 2); labels, centers = cluster_endpoints(endpoints, ALIGN_RADIUS); groups = defaultdict(list)
    for i, label in enumerate(labels.tolist()): groups[int(label)].append(i)
    pair_set = set()
    for i in range(len(segments)):
        a, b = int(labels[2 * i]), int(labels[2 * i + 1]);
        if a != b: pair_set.add(tuple(sorted((a, b))))
    degree = defaultdict(int)
    for a, b in pair_set: degree[a] += 1; degree[b] += 1
    candidates = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE] for v in values], dtype=np.float32); scores = np.asarray([float(v["score"]) for v in values], dtype=np.float32)
    node_map = align_nodes(centers.astype(np.float64), candidates.astype(np.float64), ALIGN_RADIUS)
    rows = []; node_indices = []; candidate_indices = []
    for node, candidate_index in sorted(node_map.items()):
        center = centers[node]; member = endpoints[groups[node]]; spread = np.linalg.norm(member - center, axis=1); lengths = [float(np.linalg.norm(segments[i, 1] - segments[i, 0])) for i in range(len(segments)) if int(labels[2 * i]) == node or int(labels[2 * i + 1]) == node]
        point = candidates[candidate_index]; distance = float(np.linalg.norm(point - center)); rows.append([scores[candidate_index], float(point[0] / IMAGE), float(point[1] / IMAGE), distance / 16.0, len(member) / 16.0, float(spread.mean() / 16.0), float(spread.std() / 16.0), degree[node] / 8.0, float(np.mean(lengths) / 64.0) if lengths else 0., _ink(gray, center), _ink(gray, point), float((np.linalg.norm(candidates - point[None], axis=1) <= 16).sum() / 16.0), float((np.linalg.norm(candidates - point[None], axis=1) <= 24).sum() / 24.0), float(node / max(len(centers), 1))]); node_indices.append(node); candidate_indices.append(candidate_index)
    return np.asarray(rows, dtype=np.float32), centers, np.asarray(candidate_indices, dtype=np.int64), np.asarray(node_indices, dtype=np.int64)


def _labels(sample_key: str, values: list[dict[str, float]], candidate_indices: np.ndarray) -> np.ndarray:
    base = DATA / "samples" / sample_key; layers = json.loads(base.with_suffix(".layers.json").read_text()); targets = np.asarray([x["center"] for x in layers["atoms"]], dtype=np.float32); points = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE] for v in values], dtype=np.float32); d = np.linalg.norm(points[:, None] - targets[None], axis=-1)
    # Strict target-free assignment has already happened; labels only inspect
    # the chosen candidate after the assignment.
    out = np.zeros(len(candidate_indices), dtype=np.float32)
    for i, candidate_index in enumerate(candidate_indices.tolist()):
        close = np.where(d[candidate_index] <= TARGET_RADIUS)[0]; near = np.where(d[candidate_index] <= AMBIGUITY_RADIUS)[0]
        out[i] = float(len(close) == 1 and len(near) == 1)
    return out


class NodeMLP(nn.Module):
    def __init__(self, width):
        super().__init__(); self.net = nn.Sequential(nn.Linear(width, 40), nn.ReLU(), nn.Linear(40, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def _calibrate(scores, labels):
    order = np.argsort(-scores, kind="stable"); y = labels[order]; tp = np.cumsum(y); fp = np.cumsum(1-y); p = tp / np.maximum(tp+fp,1); r = tp / max(y.sum(),1); valid=np.where(p>=.90)[0]
    if not len(valid): return {"threshold":.5,"precision":float(p[-1]),"recall":float(r[-1]),"found":False}
    best=valid[np.where(r[valid]==r[valid].max())[0][-1]]; return {"threshold":float(scores[order[best]]),"precision":float(p[best]),"recall":float(r[best]),"found":True}


def _metrics(scores, labels, threshold):
    keep=scores>=threshold; pos=labels>.5; tp=int((keep&pos).sum()); fp=int((keep&~pos).sum()); fn=int((~keep&pos).sum()); return {"nodes":int(len(labels)),"positives":int(pos.sum()),"selected":int(keep.sum()),"tp":tp,"fp":fp,"fn":fn,"precision":tp/max(tp+fp,1),"recall":tp/max(tp+fn,1)}


def run():
    if file_digest(INITIAL)!=INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    candidates=_load_candidates(); rows=[json.loads(x) for x in (DATA/'samples.jsonl').read_text().splitlines() if x and json.loads(x).get('render_status')=='accepted']; train=sorted([r for r in rows if r['split']=='train'],key=lambda r:r['sample_key'])[:TRAIN_LIMIT]; hold=sorted([r for r in rows if r['split']=='holdout'],key=lambda r:r['sample_key'])[:HOLDOUT_LIMIT]; features={}; labels={}
    for name,chosen in (('train',train),('holdout',hold)):
        fs=[]; ys=[]
        for row in chosen:
            vals=candidates[row['sample_key']]; f,_,ci,_=_node_rows(row['sample_key'],vals); fs.append(f); ys.append(_labels(row['sample_key'],vals,ci))
        features[name]=np.concatenate(fs); labels[name]=np.concatenate(ys)
    mean,std=features['train'].mean(0),np.maximum(features['train'].std(0),1e-5); torch.manual_seed(SEED); model=NodeMLP(features['train'].shape[1]); x=torch.from_numpy((features['train']-mean)/std); y=torch.from_numpy(labels['train']); opt=torch.optim.Adam(model.parameters(),lr=2e-3); pw=torch.tensor(float((1-labels['train'].mean())/max(labels['train'].mean(),1e-4)))
    for _ in range(STEPS): loss=nn.functional.binary_cross_entropy_with_logits(model(x),y,pos_weight=pw); opt.zero_grad(); loss.backward(); opt.step()
    with torch.inference_mode(): ts=torch.sigmoid(model(torch.from_numpy((features['train']-mean)/std))).numpy(); hs=torch.sigmoid(model(torch.from_numpy((features['holdout']-mean)/std))).numpy()
    cal=_calibrate(ts,labels['train']); threshold=cal['threshold']; train_m=_metrics(ts,labels['train'],threshold); hold_m=_metrics(hs,labels['holdout'],threshold); gate={'train_threshold_calibrated':cal['found'],'holdout_precision_ge_0_90':hold_m['precision']>=.90,'holdout_recall_ge_0_85':hold_m['recall']>=.85,'checkpoint_unchanged':file_digest(INITIAL)==INITIAL_SHA,'sealed_test_read':False}; gate['node_identity_shadow_worth_attempting']=gate['train_threshold_calibrated'] and gate['holdout_precision_ge_0_90'] and gate['holdout_recall_ge_0_85'] and gate['checkpoint_unchanged'] and not gate['sealed_test_read']
    report={'schema_version':'chemvector-identity-aligned-line-first-node-identity-mlp-r5-cpu-v2-0.1','complete':True,'read_only':True,'training_performed':True,'training_scope':'isolated_cpu_node_identity_probe_one_candidate_per_line_node','optimizer_steps':STEPS,'gpu_used':False,'sealed_test_read':False,'deployment_authorized':False,'production_checkpoint_written':False,'initial_sha256':INITIAL_SHA,'feature_names':['candidate_score','candidate_x','candidate_y','node_candidate_distance','cluster_size','cluster_spread_mean','cluster_spread_std','cluster_degree','mean_segment_length','node_ink','candidate_ink','near_candidate_count16','near_candidate_count24','node_rank'],'fixed_slice':{'train_images':len(train),'holdout_images':len(hold),'assignment_radius_px':ALIGN_RADIUS,'target_radius_px':TARGET_RADIUS,'threshold_source':'train-only'},'train':train_m,'holdout':hold_m,'calibration':cal,'gate':gate}
    OUT.parent.mkdir(parents=True,exist_ok=False); OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n'); print(json.dumps({'output':str(OUT),'train':train_m,'holdout':hold_m,'calibration':cal,'gate':gate},indent=2)); return report


if __name__=='__main__': run()
