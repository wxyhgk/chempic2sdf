#!/usr/bin/env python3
"""One isolated <=800-step line-first edge/bond-type shadow.

The model sees only target-free LSD/cluster geometry and raster support
features.  Ground-truth mapping and bond order are constructed after the
proposal field is frozen and remain in memory for train/holdout supervision;
they are never written to the inference artifact.  Production weights,
graph assembly, selector and checkpoint are untouched.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from tools.preflight_line_first_edge_features_r5 import (
    DATA, CLUSTER_RADIUS, LABEL_RADIUS, MAX_SEGMENTS, TRAIN_LIMIT, HOLDOUT_LIMIT,
    _proposals,
)
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from PIL import Image

INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-edge-bond-shadow-r5"
REPORT = OUT / "report.json"
STATE = OUT / "sidecar-state.pt"
STEPS = 800
SEED = 20261018
EDGE_PRECISION_GATE = .85
EDGE_RECALL_GATE = .50
TYPE_ACCURACY_GATE = .75


def _candidate_keys(sample_key: str) -> list[tuple[int, int]]:
    image = Image.open(DATA / "samples" / f"{sample_key}.png")
    from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig, detect_raster_line_segments
    segments = detect_raster_line_segments(image, config=LineSegmentRescueConfig(maximum_candidates=MAX_SEGMENTS))
    if not len(segments):
        return []
    labels, _ = cluster_endpoints(segments.reshape(-1, 2), CLUSTER_RADIUS)
    pairs = set()
    for index in range(len(segments)):
        a, b = int(labels[2 * index]), int(labels[2 * index + 1])
        if a != b:
            pairs.add(tuple(sorted((a, b))))
    return sorted(pairs)


TYPE_NAMES = ["nonedge", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "DATIVE", "OTHER"]
TYPE_ID = {name: i for i, name in enumerate(TYPE_NAMES)}


def _typed_labels(sample_key: str, centers: np.ndarray, keys: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, int]:
    base = DATA / "samples" / sample_key
    layers = json.loads((base.with_suffix(".layers.json")).read_text())
    graph = json.loads((base.with_suffix(".graph.json")).read_text())
    targets = np.asarray([atom["center"] for atom in layers["atoms"]], dtype=np.float32)
    distance = np.linalg.norm(centers[:, None] - targets[None], axis=-1) if len(centers) else np.empty((0, len(targets)))
    pairs = [(float(distance[i, j]), i, j) for i in range(len(centers)) for j in range(len(targets)) if distance[i, j] <= LABEL_RADIUS]
    mapping: dict[int, int] = {}; used: set[int] = set()
    for _, node, target in sorted(pairs):
        if node not in mapping and target not in used:
            mapping[node] = target; used.add(target)
    type_by_edge = {}
    for bond in graph["bonds"]:
        edge = tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
        type_name = str(bond.get("chemical_type", "OTHER")).upper()
        type_by_edge[edge] = TYPE_ID.get(type_name, TYPE_ID["OTHER"])
    labels = []
    for a, b in keys:
        edge = tuple(sorted((mapping[a], mapping[b]))) if a in mapping and b in mapping else None
        labels.append(type_by_edge.get(edge, TYPE_ID["nonedge"]))
    return np.asarray(labels, dtype=np.int64), np.asarray(labels) > TYPE_ID["nonedge"], len(type_by_edge)


class EdgeBondMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(width, 48), nn.ReLU(), nn.Linear(48, 32), nn.ReLU())
        self.edge = nn.Linear(32, 1)
        self.kind = nn.Linear(32, len(TYPE_NAMES) - 1)

    def forward(self, x):
        h = self.body(x)
        return self.edge(h).squeeze(-1), self.kind(h)


def _auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable"); y = labels[order].astype(np.float64)
    if not len(y) or y.sum() <= 0: return 0.0
    tp = np.cumsum(y); fp = np.cumsum(1.0 - y); precision = tp / np.maximum(tp + fp, 1.0); recall = tp / y.sum()
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * precision))


def _calibrate(scores: np.ndarray, labels: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable"); y = labels[order].astype(np.float64)
    tp = np.cumsum(y); fp = np.cumsum(1.0 - y); p = tp / np.maximum(tp + fp, 1.0); r = tp / max(y.sum(), 1.0)
    valid = np.where(p >= EDGE_PRECISION_GATE)[0]
    if not len(valid):
        return {"threshold": .5, "precision": float(p[-1]) if len(p) else 0., "recall": float(r[-1]) if len(r) else 0., "found": False}
    best_recall = float(r[valid].max()); selected = valid[np.where(r[valid] == best_recall)[0][-1]]
    return {"threshold": float(scores[order[selected]]), "precision": float(p[selected]), "recall": float(r[selected]), "found": True}


def _metrics(scores: np.ndarray, labels: np.ndarray, type_logits: np.ndarray, type_labels: np.ndarray, threshold: float, gt_bonds: int) -> dict:
    keep = scores >= threshold; positive = labels.astype(bool)
    tp = int((keep & positive).sum()); fp = int((keep & ~positive).sum()); fn = int((~keep & positive).sum())
    typed = np.argmax(type_logits, axis=1) + 1
    selected_true = keep & positive
    type_correct = int((typed[selected_true] == type_labels[selected_true]).sum())
    return {"candidate_edges": int(len(labels)), "candidate_positive_edges": int(positive.sum()), "gt_bonds": int(gt_bonds), "selected_edges": int(keep.sum()), "tp": tp, "fp": fp, "fn": fn, "edge_precision": tp / max(tp + fp, 1), "candidate_edge_recall": tp / max(gt_bonds, 1), "selected_edge_recall": tp / max(gt_bonds, 1), "edge_auprc": _auprc(scores, labels), "conditional_bond_type_accuracy": type_correct / max(int(selected_true.sum()), 1), "typed_count": int(selected_true.sum()), "typed_correct": type_correct}


def _collect(chosen: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, dict]:
    fs=[]; ys=[]; ts=[]; gt=0; profile=defaultdict(lambda: Counter())
    for row in chosen:
        f, centers, _ = _proposals(row["sample_key"]); keys = _candidate_keys(row["sample_key"]); t, y, n = _typed_labels(row["sample_key"], centers, keys)
        if len(f) != len(t): raise RuntimeError("feature/key/label mismatch")
        fs.append(f); ts.append(t); ys.append(y.astype(np.float32)); gt += n
        profile[row["profile_id"]]["candidate_edges"] += len(t); profile[row["profile_id"]]["positive_edges"] += int(y.sum()); profile[row["profile_id"]]["gt_bonds"] += n
    return np.concatenate(fs), np.concatenate(ys), np.concatenate(ts), gt, {k: dict(v) for k,v in sorted(profile.items())}


def run() -> dict:
    if file_digest(INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x and json.loads(x).get("render_status") == "accepted"]
    train = sorted([r for r in rows if r["split"] == "train"], key=lambda r: r["sample_key"])[:TRAIN_LIMIT]
    hold = sorted([r for r in rows if r["split"] == "holdout"], key=lambda r: r["sample_key"])[:HOLDOUT_LIMIT]
    if len(train) != TRAIN_LIMIT or len(hold) != HOLDOUT_LIMIT: raise RuntimeError("fixed split changed")
    x_train, y_train, t_train, gt_train, prof_train = _collect(train); x_hold, y_hold, t_hold, gt_hold, prof_hold = _collect(hold)
    mean, std = x_train.mean(0), np.maximum(x_train.std(0), 1e-5)
    torch.manual_seed(SEED); model = EdgeBondMLP(x_train.shape[1]); device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); model.to(device)
    tx=torch.from_numpy((x_train-mean)/std).to(device); ty=torch.from_numpy(y_train).to(device); tt=torch.from_numpy(t_train).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=2e-3); pos_weight=torch.tensor(float((1-y_train.mean())/max(y_train.mean(),1e-4)),device=device); started=time.time()
    for step in range(STEPS):
        edge_logits, kind_logits = model(tx); edge_loss=nn.functional.binary_cross_entropy_with_logits(edge_logits,ty,pos_weight=pos_weight)
        positive=ty>0.5; kind_loss=nn.functional.cross_entropy(kind_logits[positive],tt[positive]-1) if bool(positive.any()) else edge_logits.sum()*0
        loss=edge_loss+0.5*kind_loss; optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.inference_mode():
        train_edge, train_kind=model(tx); hold_edge, hold_kind=model(torch.from_numpy((x_hold-mean)/std).to(device))
    train_scores=torch.sigmoid(train_edge).detach().cpu().numpy(); hold_scores=torch.sigmoid(hold_edge).detach().cpu().numpy(); train_kind=train_kind.detach().cpu().numpy(); hold_kind=hold_kind.detach().cpu().numpy()
    calibration=_calibrate(train_scores,y_train); threshold=calibration["threshold"]; train_metrics=_metrics(train_scores,y_train,train_kind,t_train,threshold,gt_train); hold_metrics=_metrics(hold_scores,y_hold,hold_kind,t_hold,threshold,gt_hold)
    OUT.mkdir(parents=True,exist_ok=False); torch.save({"model":model.state_dict(),"mean":mean,"std":std,"threshold":threshold,"feature_names": ["length_mean","length_std","length_min","length_max","support_count","angle_cos","angle_sin","endpoint_a_cluster_size","endpoint_b_cluster_size","endpoint_a_degree","endpoint_b_degree","endpoint_a_spread","endpoint_b_spread","endpoint_a_min_neighbor_angle","endpoint_b_min_neighbor_angle","ink_mean","ink_fraction"],"provenance":{"initial_sha256":INITIAL_SHA,"sealed_test_read":False,"train_images":TRAIN_LIMIT,"holdout_images":HOLDOUT_LIMIT}},STATE)
    after=file_digest(INITIAL); report={"schema_version":"chemvector-identity-aligned-line-first-edge-bond-shadow-r5-0.1","complete":True,"read_only":False,"training_performed":True,"training_scope":"isolated_line_first_edge_and_bond_type_mlp","gpu_used":device.type=="cuda","optimizer_steps":STEPS,"elapsed_seconds":time.time()-started,"sealed_test_read":False,"deployment_authorized":False,"production_checkpoint_written":False,"initial_sha256":INITIAL_SHA,"initial_checkpoint_sha256_after":after,"candidate_feature_contract":"LSD raster segments -> endpoint clusters -> unordered provisional edge; geometry/support/count features only","train":{"metrics":train_metrics,"profiles":prof_train},"holdout":{"metrics":hold_metrics,"profiles":prof_hold},"calibration":{"source":"train_only","threshold":threshold,**calibration},"state_sha256":hashlib.sha256(STATE.read_bytes()).hexdigest(),"gate":{"checkpoint_unchanged":after==INITIAL_SHA,"sealed_test_read":False,"production_go":False,"holdout_edge_precision_ge_0_85":hold_metrics["edge_precision"]>=.85,"holdout_edge_recall_ge_0_50":hold_metrics["candidate_edge_recall"]>=.50,"holdout_bond_type_accuracy_ge_0_75":hold_metrics["conditional_bond_type_accuracy"]>=.75,"training_admissible":False}}
    REPORT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n"); print(json.dumps({"output":str(REPORT),"device":str(device),"train":train_metrics,"holdout":hold_metrics,"calibration":calibration},indent=2)); return report


if __name__ == "__main__": run()
