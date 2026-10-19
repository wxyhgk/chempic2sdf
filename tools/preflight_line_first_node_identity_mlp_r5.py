#!/usr/bin/env python3
"""CPU gate for a target-free line-first node-identity scorer.

Each row is a scored vertex candidate paired with local provisional LSD-node
evidence (cluster size/spread/degree/ink and candidate-to-node distance).
Features are built before any target lookup.  Unique atom ownership labels are
joined in memory afterward, a small MLP is fitted on the fixed train split,
and its threshold is calibrated on train only.  No GPU, production checkpoint,
or sealed test is touched.
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
from tools.audit_line_first_candidate_alignment_r5 import IMAGE

CANDIDATES = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5-scored/candidates.jsonl"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-node-identity-mlp-r5-cpu/report.json"
TRAIN_LIMIT = 160
HOLDOUT_LIMIT = 80
ALIGN_RADIUS = 16.0
TARGET_RADIUS = 6.0
AMBIGUITY_RADIUS = 20.0
STEPS = 500
SEED = 20261018


def _ink(gray: np.ndarray, p: np.ndarray) -> float:
    x, y = int(round(float(p[0]))), int(round(float(p[1])))
    h, w = gray.shape; x0, x1 = max(0, x - 2), min(w, x + 3); y0, y1 = max(0, y - 2), min(h, y + 3)
    return float((255.0 - gray[y0:y1, x0:x1]).mean() / 255.0) if x1 > x0 and y1 > y0 else 0.0


def _load_candidates() -> dict[str, list[dict[str, float]]]:
    result = {}; forbidden = {"targets", "target_indices", "matched", "element", "visible_element", "atom_id", "graph", "bonds", "ground_truth", "label"}
    for line_no, line in enumerate(CANDIDATES.read_text().splitlines(), 1):
        row = json.loads(line)
        if row.get("source") != "shadow" or set(row) & forbidden: raise ValueError(f"candidate contract failed line {line_no}")
        result[row["sample_key"]] = row["candidates"]
    if len(result) != 1170: raise ValueError("expected 1170 candidate rows")
    return result


def _proposal_features(sample_key: str, values: list[dict[str, float]]) -> np.ndarray:
    base = DATA / "samples" / sample_key; image = Image.open(base.with_suffix(".png")).convert("L"); gray = np.asarray(image, dtype=np.uint8)
    segments = detect_raster_line_segments(image, config=LineSegmentRescueConfig(maximum_candidates=MAX_SEGMENTS))
    if not len(values): return np.empty((0, 15), dtype=np.float32)
    points = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE] for v in values], dtype=np.float32)
    if len(segments):
        endpoints = segments.reshape(-1, 2); labels, centers = cluster_endpoints(endpoints, ALIGN_RADIUS)
        groups = defaultdict(list)
        for i, label in enumerate(labels.tolist()): groups[int(label)].append(i)
        pairs = set()
        for i in range(len(segments)):
            a, b = int(labels[2 * i]), int(labels[2 * i + 1]);
            if a != b: pairs.add(tuple(sorted((a, b))))
        degree = defaultdict(int)
        for a, b in pairs: degree[a] += 1; degree[b] += 1
        node_stats = []
        for index, center in enumerate(centers):
            member = endpoints[groups[index]]; spread = np.linalg.norm(member - center, axis=1)
            lengths = [float(np.linalg.norm(segments[i, 1] - segments[i, 0])) for i in range(len(segments)) if int(labels[2 * i]) == index or int(labels[2 * i + 1]) == index]
            node_stats.append((len(member) / 16.0, float(spread.mean() / 16.0), float(spread.std() / 16.0), degree[index] / 8.0, float(np.mean(lengths) / 64.0) if lengths else 0.0, _ink(gray, center)))
        node_points = centers
    else:
        node_points = np.empty((0, 2), dtype=np.float32); node_stats = []
    if len(node_points):
        distances = np.linalg.norm(points[:, None] - node_points[None], axis=-1); nearest = distances.argmin(axis=1); nearest_distance = distances[np.arange(len(points)), nearest]
    else:
        nearest = np.zeros(len(points), dtype=np.int64); nearest_distance = np.full(len(points), 64.0, dtype=np.float32)
    scores = np.asarray([float(v["score"]) for v in values], dtype=np.float32); order = np.argsort(-scores); ranks = np.empty(len(scores), dtype=np.float32); ranks[order] = np.arange(len(scores)); ranks = ranks / max(len(scores), 1)
    rows = []
    for i, point in enumerate(points):
        stats = node_stats[int(nearest[i])] if len(node_points) else (0., 0., 0., 0., 0., 0.)
        neighbor = np.linalg.norm(points - point[None], axis=1); neighbor[i] = 1e9
        rows.append([scores[i], float(point[0] / IMAGE), float(point[1] / IMAGE), float(nearest_distance[i] / 32.0), *stats, _ink(gray, point), float((neighbor <= 16.0).sum() / 16.0), float((neighbor <= 24.0).sum() / 24.0), float(ranks[i])])
    return np.asarray(rows, dtype=np.float32)


def _labels(sample_key: str, values: list[dict[str, float]]) -> np.ndarray:
    base = DATA / "samples" / sample_key; layers = json.loads(base.with_suffix(".layers.json").read_text()); targets = np.asarray([x["center"] for x in layers["atoms"]], dtype=np.float32)
    points = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE] for v in values], dtype=np.float32); d = np.linalg.norm(points[:, None] - targets[None], axis=-1) if len(points) else np.empty((0, len(targets)))
    pairs = [(float(d[i, j]), i, j) for i in range(len(points)) for j in range(len(targets)) if d[i, j] <= TARGET_RADIUS]
    mapping = {}; used = set()
    for _, i, j in sorted(pairs):
        if i in mapping or j in used: continue
        near = np.where(d[i] <= AMBIGUITY_RADIUS)[0]
        if len(near) != 1: continue
        mapping[i] = j; used.add(j)
    labels = np.zeros(len(points), dtype=np.float32)
    for i in mapping: labels[i] = 1.0
    return labels


class NodeMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__(); self.net = nn.Sequential(nn.Linear(width, 40), nn.ReLU(), nn.Linear(40, 1))
    def forward(self, x): return self.net(x).squeeze(-1)


def _calibrate(scores: np.ndarray, labels: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable"); y = labels[order]; tp = np.cumsum(y); fp = np.cumsum(1 - y); p = tp / np.maximum(tp + fp, 1); r = tp / max(y.sum(), 1)
    valid = np.where(p >= .90)[0]
    if not len(valid): return {"threshold": .5, "precision": float(p[-1]), "recall": float(r[-1]), "found": False}
    best = valid[np.where(r[valid] == r[valid].max())[0][-1]]; return {"threshold": float(scores[order[best]]), "precision": float(p[best]), "recall": float(r[best]), "found": True}


def _metrics(scores, labels, threshold):
    keep = scores >= threshold; positive = labels > .5; tp = int((keep & positive).sum()); fp = int((keep & ~positive).sum()); fn = int((~keep & positive).sum())
    return {"candidates": int(len(labels)), "positive_targets": int(positive.sum()), "selected": int(keep.sum()), "tp": tp, "fp": fp, "fn": fn, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def run() -> dict:
    if file_digest(INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    candidates = _load_candidates(); rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x and json.loads(x).get("render_status") == "accepted"]
    train = sorted([r for r in rows if r["split"] == "train"], key=lambda r: r["sample_key"])[:TRAIN_LIMIT]; hold = sorted([r for r in rows if r["split"] == "holdout"], key=lambda r: r["sample_key"])[:HOLDOUT_LIMIT]
    features = {}; labels = {}
    for name, chosen in (("train", train), ("holdout", hold)):
        fs=[]; ys=[]
        for row in chosen:
            vals = candidates[row["sample_key"]]; fs.append(_proposal_features(row["sample_key"], vals)); ys.append(_labels(row["sample_key"], vals))
        features[name] = np.concatenate(fs); labels[name] = np.concatenate(ys)
    mean, std = features["train"].mean(0), np.maximum(features["train"].std(0), 1e-5); torch.manual_seed(SEED); model = NodeMLP(features["train"].shape[1]); x=torch.from_numpy((features["train"]-mean)/std); y=torch.from_numpy(labels["train"]); opt=torch.optim.Adam(model.parameters(),lr=2e-3); pos_weight=torch.tensor(float((1-labels["train"].mean())/max(labels["train"].mean(),1e-4)))
    for _ in range(STEPS):
        loss=nn.functional.binary_cross_entropy_with_logits(model(x),y,pos_weight=pos_weight); opt.zero_grad(); loss.backward(); opt.step()
    with torch.inference_mode(): train_scores=torch.sigmoid(model(torch.from_numpy((features["train"]-mean)/std))).numpy(); hold_scores=torch.sigmoid(model(torch.from_numpy((features["holdout"]-mean)/std))).numpy()
    calibration=_calibrate(train_scores,labels["train"]); threshold=calibration["threshold"]; train_metrics=_metrics(train_scores,labels["train"],threshold); hold_metrics=_metrics(hold_scores,labels["holdout"],threshold)
    gate={"train_threshold_calibrated":calibration["found"],"holdout_precision_ge_0_90":hold_metrics["precision"]>=.90,"holdout_recall_ge_0_85":hold_metrics["recall"]>=.85,"checkpoint_unchanged":file_digest(INITIAL)==INITIAL_SHA,"sealed_test_read":False,"training_performed":True}; gate["node_identity_shadow_worth_attempting"]=all(gate.values())
    report={"schema_version":"chemvector-identity-aligned-line-first-node-identity-mlp-r5-cpu-0.1","complete":True,"read_only":True,"training_performed":True,"training_scope":"isolated_cpu_node_identity_probe","optimizer_steps":STEPS,"gpu_used":False,"sealed_test_read":False,"deployment_authorized":False,"production_checkpoint_written":False,"initial_sha256":INITIAL_SHA,"feature_names":["candidate_score","candidate_x","candidate_y","nearest_line_distance","cluster_size","cluster_spread_mean","cluster_spread_std","cluster_degree","mean_segment_length","node_ink","candidate_ink","near_candidate_count16","near_candidate_count24","candidate_score_rank"],"proposal_contract":"candidate rows + LSD local node evidence only; labels post-proposal","fixed_slice":{"train_images":len(train),"holdout_images":len(hold),"alignment_radius_px":ALIGN_RADIUS,"target_radius_px":TARGET_RADIUS,"threshold_source":"train-only"},"train":train_metrics,"holdout":hold_metrics,"calibration":calibration,"gate":gate}
    OUT.parent.mkdir(parents=True,exist_ok=False); OUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n"); print(json.dumps({"output":str(OUT),"train":train_metrics,"holdout":hold_metrics,"calibration":calibration,"gate":gate},indent=2)); return report


if __name__ == "__main__": run()
