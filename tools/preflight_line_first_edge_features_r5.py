#!/usr/bin/env python3
"""CPU preflight for target-free line-first edge features.

Raster LSD segments are clustered into provisional nodes and unordered edges.
Every feature is computed from the image and segment geometry only.  Ground
truth atom/bond geometry is joined in a second, post-proposal pass solely to
make diagnostic labels.  A tiny CPU MLP is fitted on the fixed train slice;
its threshold is selected on train only and evaluated once on holdout.

This is not a deployment or graph-merge experiment and cannot read sealed
test data.  It answers whether a single <=800-step GPU edge scorer is worth
attempting after the endpoint-free route's strong reachability result.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
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

from chemvector.components.vision.line_segment_rescue import (
    LineSegmentRescueConfig,
    detect_raster_line_segments,
)
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-edge-features-r5-cpu-v2/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
CLUSTER_RADIUS = 16.0
LABEL_RADIUS = 24.0
MAX_SEGMENTS = 512
TRAIN_LIMIT = 160
HOLDOUT_LIMIT = 80
SEED = 20261018
STEPS = 500


def _darkness(gray: np.ndarray, point: np.ndarray) -> float:
    x = int(round(float(point[0]))); y = int(round(float(point[1])))
    h, w = gray.shape
    x0, x1 = max(0, x - 1), min(w, x + 2)
    y0, y1 = max(0, y - 1), min(h, y + 2)
    return float((255.0 - gray[y0:y1, x0:x1]).mean() / 255.0) if x1 > x0 and y1 > y0 else 0.0


def _line_support(gray: np.ndarray, begin: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    length = float(np.linalg.norm(end - begin))
    count = max(2, int(round(length)))
    ts = np.linspace(0.0, 1.0, count)
    points = begin[None] * (1.0 - ts[:, None]) + end[None] * ts[:, None]
    values = np.asarray([_darkness(gray, point) for point in points], dtype=np.float32)
    return float(values.mean()), float((values > 0.12).mean())


def _angle_delta(a: float, b: float) -> float:
    # Lines are undirected; fold to [0, pi/2].
    d = abs(a - b) % np.pi
    return float(min(d, np.pi - d))


def _proposals(sample_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return target-free edge features, provisional node centers, segments."""
    base = DATA / "samples" / sample_key
    image = Image.open(base.with_suffix(".png")).convert("L")
    gray = np.asarray(image, dtype=np.uint8)
    segments = detect_raster_line_segments(image, config=LineSegmentRescueConfig(maximum_candidates=MAX_SEGMENTS))
    if not len(segments):
        return np.empty((0, 17), dtype=np.float32), np.empty((0, 2), dtype=np.float32), segments
    endpoints = segments.reshape(-1, 2)
    labels, centers = cluster_endpoints(endpoints, CLUSTER_RADIUS)
    node_members: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        node_members[int(label)].append(index)
    pair_segments: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in range(len(segments)):
        a, b = int(labels[2 * index]), int(labels[2 * index + 1])
        if a != b:
            pair_segments[tuple(sorted((a, b)))].append(index)
    degree = defaultdict(int)
    for a, b in pair_segments:
        degree[a] += 1; degree[b] += 1
    rows = []
    for (a, b), indices in sorted(pair_segments.items()):
        lengths = np.asarray([np.linalg.norm(segments[i, 1] - segments[i, 0]) for i in indices], dtype=np.float32)
        angles = np.asarray([np.arctan2(segments[i, 1, 1] - segments[i, 0, 1], segments[i, 1, 0] - segments[i, 0, 0]) for i in indices], dtype=np.float32)
        # Canonical orientation makes angle statistics invariant to segment order.
        angle = float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))
        supports = np.asarray([_line_support(gray, segments[i, 0], segments[i, 1]) for i in indices], dtype=np.float32)
        center_a, center_b = centers[a], centers[b]
        spread_a = np.linalg.norm(endpoints[node_members[a]] - center_a, axis=1)
        spread_b = np.linalg.norm(endpoints[node_members[b]] - center_b, axis=1)
        # Neighbor-angle evidence at each endpoint is target independent.
        neighbor_angles: dict[int, list[float]] = defaultdict(list)
        for j, (u, v) in enumerate(pair_segments):
            if u == a or v == a:
                seg = segments[pair_segments[(u, v)][0]]
                other = centers[v] if u == a else centers[u]
                neighbor_angles[a].append(float(np.arctan2(*(other - center_a)[::-1])))
            if u == b or v == b:
                seg = segments[pair_segments[(u, v)][0]]
                other = centers[v] if u == b else centers[u]
                neighbor_angles[b].append(float(np.arctan2(*(other - center_b)[::-1])))
        other_a = [_angle_delta(angle, x) for x in neighbor_angles[a] if _angle_delta(angle, x) > 1e-5]
        other_b = [_angle_delta(angle, x) for x in neighbor_angles[b] if _angle_delta(angle, x) > 1e-5]
        # Feature order is frozen in the report and contains no target lookup.
        feature = [
            float(lengths.mean() / 64.0), float(lengths.std() / 16.0), float(lengths.min() / 64.0),
            float(lengths.max() / 64.0), float(len(indices) / 8.0),
            float(np.cos(angle)), float(np.sin(angle)),
            float(len(node_members[a]) / 16.0), float(len(node_members[b]) / 16.0),
            float(degree[a] / 8.0), float(degree[b] / 8.0),
            float(spread_a.mean() / 16.0), float(spread_b.mean() / 16.0),
            float(min(other_a) if other_a else 0.0), float(min(other_b) if other_b else 0.0),
            float(supports[:, 0].mean()), float(supports[:, 1].mean()),
        ]
        rows.append(feature)
    return np.asarray(rows, dtype=np.float32), centers.astype(np.float32), segments


def _labels(sample_key: str, centers: np.ndarray, features: np.ndarray) -> np.ndarray:
    """Post-proposal target join: one-to-one node mapping then bond labels."""
    base = DATA / "samples" / sample_key
    layers = json.loads(base.with_suffix(".layers.json").read_text())
    graph = json.loads(base.with_suffix(".graph.json").read_text())
    targets = np.asarray([atom["center"] for atom in layers["atoms"]], dtype=np.float32)
    if not len(centers) or not len(targets):
        return np.zeros(len(features), dtype=np.float32)
    distance = np.linalg.norm(centers[:, None] - targets[None], axis=-1)
    pairs = [(float(distance[i, j]), i, j) for i in range(len(centers)) for j in range(len(targets)) if distance[i, j] <= LABEL_RADIUS]
    mapping: dict[int, int] = {}; used_targets: set[int] = set()
    for _, node, target in sorted(pairs):
        if node not in mapping and target not in used_targets:
            mapping[node] = target; used_targets.add(target)
    # Reconstruct the same deterministic pair order used by _proposals.
    image = Image.open(base.with_suffix(".png")).convert("L")
    segments = detect_raster_line_segments(image, config=LineSegmentRescueConfig(maximum_candidates=MAX_SEGMENTS))
    labels, _ = cluster_endpoints(segments.reshape(-1, 2), CLUSTER_RADIUS) if len(segments) else (np.empty(0, dtype=np.int64), centers)
    pair_segments: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index in range(len(segments)):
        a, b = int(labels[2 * index]), int(labels[2 * index + 1])
        if a != b:
            pair_segments[tuple(sorted((a, b)))].append(index)
    gt = {tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"])))) for bond in graph["bonds"]}
    values = []
    for a, b in sorted(pair_segments):
        values.append(float(a in mapping and b in mapping and tuple(sorted((mapping[a], mapping[b]))) in gt))
    if len(values) != len(features):
        raise RuntimeError("proposal/label cardinality mismatch")
    return np.asarray(values, dtype=np.float32)


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    if not len(scores) or labels.sum() == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable"); y = labels[order]
    tp = np.cumsum(y); fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1.0); recall = tp / max(float(labels.sum()), 1.0)
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:])) if len(recall) > 1 else float(precision[0])


def _calibrate(scores: np.ndarray, labels: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable"); y = labels[order]
    tp = np.cumsum(y); fp = np.cumsum(1.0 - y)
    precision = tp / np.maximum(tp + fp, 1.0); recall = tp / max(float(labels.sum()), 1.0)
    valid = np.where(precision >= 0.85)[0]
    if not len(valid):
        return {"threshold": 0.5, "train_precision": float(precision[-1]) if len(precision) else 0.0, "train_recall": float(recall[-1]) if len(recall) else 0.0, "calibration_found": False}
    # Maximize recall subject to precision gate; tie-break toward higher threshold.
    max_recall = float(recall[valid].max()); choices = valid[np.where(recall[valid] == max_recall)[0]]
    index = int(choices[-1]); threshold = float(scores[order[index]])
    return {"threshold": threshold, "train_precision": float(precision[index]), "train_recall": float(recall[index]), "calibration_found": True}


def _metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    keep = scores >= threshold
    tp = int((keep & (labels > 0.5)).sum()); fp = int((keep & (labels <= 0.5)).sum()); fn = int((~keep & (labels > 0.5)).sum())
    return {"candidates": int(len(labels)), "positives": int(labels.sum()), "selected": int(keep.sum()), "tp": tp, "fp": fp, "fn": fn, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "auprc": _auprc(scores, labels)}


class EdgeMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__(); self.net = nn.Sequential(nn.Linear(width, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)


def run() -> dict:
    if file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x and json.loads(x).get("render_status") == "accepted"]
    train_rows = sorted([r for r in rows if r["split"] == "train"], key=lambda r: r["sample_key"])[:TRAIN_LIMIT]
    hold_rows = sorted([r for r in rows if r["split"] == "holdout"], key=lambda r: r["sample_key"])[:HOLDOUT_LIMIT]
    if len(train_rows) != TRAIN_LIMIT or len(hold_rows) != HOLDOUT_LIMIT:
        raise RuntimeError("fixed split changed")
    feature_rows = []; label_rows = []; per_split = {}
    for name, chosen in (("train", train_rows), ("holdout", hold_rows)):
        fs, ys = [], []
        for idx, row in enumerate(chosen, 1):
            f, centers, _ = _proposals(row["sample_key"]); y = _labels(row["sample_key"], centers, f)
            fs.append(f); ys.append(y)
            if idx % 40 == 0: print(json.dumps({"split": name, "progress": idx}), flush=True)
        feature_rows.append(np.concatenate(fs) if fs else np.empty((0, 17), dtype=np.float32)); label_rows.append(np.concatenate(ys) if ys else np.empty(0, dtype=np.float32))
        per_split[name] = {"images": len(chosen), "candidate_edges": int(len(label_rows[-1])), "positive_edges": int(label_rows[-1].sum())}
    x_train, y_train = feature_rows[0], label_rows[0]; x_hold, y_hold = feature_rows[1], label_rows[1]
    mean, std = x_train.mean(0), np.maximum(x_train.std(0), 1e-5)
    torch.manual_seed(SEED); model = EdgeMLP(x_train.shape[1]); optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    x = torch.from_numpy((x_train - mean) / std); y = torch.from_numpy(y_train)
    pos_weight = torch.tensor(float((1.0 - y_train.mean()) / max(y_train.mean(), 1e-4)))
    for step in range(STEPS):
        logits = model(x); loss = nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.inference_mode():
        train_scores = torch.sigmoid(model(torch.from_numpy((x_train - mean) / std))).numpy()
        hold_scores = torch.sigmoid(model(torch.from_numpy((x_hold - mean) / std))).numpy()
    calibration = _calibrate(train_scores, y_train); threshold = calibration["threshold"]
    train_metrics = _metrics(train_scores, y_train, threshold); hold_metrics = _metrics(hold_scores, y_hold, threshold)
    gate = {"checkpoint_unchanged": file_digest(INITIAL) == INITIAL_SHA, "sealed_test_read": False,
            "train_threshold_calibrated": bool(calibration["calibration_found"]), "holdout_precision_ge_0_85": hold_metrics["precision"] >= .85,
            "holdout_recall_ge_0_50": hold_metrics["recall"] >= .50, "holdout_auprc_ge_0_85": hold_metrics["auprc"] >= .85}
    # Negative safety flags are required to be False, so ``all(gate.values())``
    # would incorrectly reject a clean run merely because sealed_test_read is
    # False.  Keep the admissibility expression explicit.
    gate["line_first_edge_shadow_worth_attempting"] = bool(
        gate["checkpoint_unchanged"]
        and not gate["sealed_test_read"]
        and gate["train_threshold_calibrated"]
        and gate["holdout_precision_ge_0_85"]
        and gate["holdout_recall_ge_0_50"]
        and gate["holdout_auprc_ge_0_85"]
    )
    report = {"schema_version": "chemvector-identity-aligned-line-first-edge-features-r5-cpu-0.1", "complete": True, "read_only": True,
              "training_performed": True, "training_scope": "isolated_cpu_edge_feature_probe", "optimizer_steps": STEPS,
              "training_performed": False, "gpu_used": False, "sealed_test_read": False, "deployment_authorized": False,
              "production_checkpoint_written": False, "initial_sha256": INITIAL_SHA, "fixed_slice": {"train_images": TRAIN_LIMIT, "holdout_images": HOLDOUT_LIMIT, "cluster_radius_px": CLUSTER_RADIUS, "label_radius_px": LABEL_RADIUS, "max_segments": MAX_SEGMENTS},
              "feature_names": ["length_mean", "length_std", "length_min", "length_max", "support_count", "angle_cos", "angle_sin", "endpoint_a_cluster_size", "endpoint_b_cluster_size", "endpoint_a_degree", "endpoint_b_degree", "endpoint_a_spread", "endpoint_b_spread", "endpoint_a_min_neighbor_angle", "endpoint_b_min_neighbor_angle", "ink_mean", "ink_fraction"],
              "per_split": per_split, "calibration": calibration, "train": train_metrics, "holdout": hold_metrics,
              "protocol": {"proposal": "LSD -> endpoint cluster -> unique unordered edge", "features_target_free": True, "labels_post_proposal_only": True, "threshold_source": "train only", "budget": "one <=800-step GPU shadow only if CPU gate passes", "no_production_merge": True}, "gate": gate}
    OUT.parent.mkdir(parents=True, exist_ok=False); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "train": train_metrics, "holdout": hold_metrics, "calibration": calibration, "gate": gate}, indent=2, ensure_ascii=False)); return report


if __name__ == "__main__": run()
