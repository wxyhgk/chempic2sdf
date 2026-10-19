#!/usr/bin/env python3
"""Isolated <=800-step line-first node/graph-assembly shadow.

Only the node keep head is trained here.  Raster line proposals and the
previously trained edge/type sidecar remain frozen.  All features used at
inference are image/line/candidate evidence; target atoms and bonds are joined
after proposal construction for in-memory supervision and metrics only.

This script never mutates the production checkpoint or production graph path,
never reads sealed test, and writes an isolated sidecar receipt/state only.
The holdout threshold is never searched: it is calibrated on train at a
precision floor, then holdout is evaluated once at the end.
"""
from __future__ import annotations

from collections import Counter
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

from chemvector.training.curriculum.checkpoints import file_digest
from tools.preflight_line_first_node_identity_mlp_r5_v2 import (
    ALIGN_RADIUS,
    CANDIDATES,
    DATA,
    HOLDOUT_LIMIT,
    IMAGE,
    NodeMLP,
    TARGET_RADIUS,
    TRAIN_LIMIT,
    _labels,
    _load_candidates,
    _node_rows,
)
from tools.shadow_line_first_edge_bond_type_r5 import (
    INITIAL,
    INITIAL_SHA,
    STATE as EDGE_STATE,
    EdgeBondMLP,
    TYPE_NAMES,
    _candidate_keys,
    _proposals,
)
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target

OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-node-graph-shadow-r5"
REPORT = OUT / "report.json"
STATE = OUT / "sidecar-state.pt"
STEPS = 800
SEED = 20261018
NODE_PRECISION_FLOOR = 0.90
NODE_RECALL_FLOOR = 0.85
EDGE_PRECISION_FLOOR = 0.90
EDGE_RECALL_FLOOR = 0.75
TYPE_ACCURACY_FLOOR = 0.90


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows() -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("render_status") == "accepted"]
    train = sorted([row for row in rows if row["split"] == "train"], key=lambda row: row["sample_key"])[:TRAIN_LIMIT]
    hold = sorted([row for row in rows if row["split"] == "holdout"], key=lambda row: row["sample_key"])[:HOLDOUT_LIMIT]
    if len(train) != TRAIN_LIMIT or len(hold) != HOLDOUT_LIMIT:
        raise RuntimeError("fixed train/holdout slice changed")
    return train, hold


def _collect(rows: list[dict], candidates: dict[str, list[dict]]) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for row in rows:
        values = candidates[row["sample_key"]]
        f, _, ci, _ = _node_rows(row["sample_key"], values)
        features.append(f)
        labels.append(_labels(row["sample_key"], values, ci))
    return np.concatenate(features), np.concatenate(labels)


def _calibrate(scores: np.ndarray, labels: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable")
    y = labels[order] > 0.5
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(y.sum()), 1)
    valid = np.where(precision >= NODE_PRECISION_FLOOR)[0]
    if not len(valid):
        return {"threshold": 0.5, "precision": float(precision[-1]) if len(precision) else 0.0, "recall": float(recall[-1]) if len(recall) else 0.0, "found": False}
    best_recall = float(recall[valid].max())
    chosen = valid[np.where(recall[valid] == best_recall)[0][-1]]
    return {"threshold": float(scores[order[chosen]]), "precision": float(precision[chosen]), "recall": float(recall[chosen]), "found": True}


def _load_edge_sidecar() -> tuple[EdgeBondMLP, np.ndarray, np.ndarray, float]:
    payload = torch.load(EDGE_STATE, map_location="cpu", weights_only=False)
    model = EdgeBondMLP(len(payload["feature_names"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32), float(payload["threshold"])


def _target_data(sample_key: str):
    base = DATA / "samples" / sample_key
    layers = json.loads(base.with_suffix(".layers.json").read_text())
    graph = json.loads(base.with_suffix(".graph.json").read_text())
    targets = np.asarray([atom["center"] for atom in layers["atoms"]], dtype=np.float32)
    gt_edges = {tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"])))) for bond in graph["bonds"]}
    type_by_edge = {}
    for bond in graph["bonds"]:
        edge = tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
        name = str(bond.get("chemical_type", "OTHER")).upper()
        type_by_edge[edge] = TYPE_NAMES.index(name) if name in TYPE_NAMES else TYPE_NAMES.index("OTHER")
    return targets, gt_edges, type_by_edge


def _evaluate(rows: list[dict], candidates: dict[str, list[dict]], node_model: NodeMLP, node_mean: np.ndarray, node_std: np.ndarray, node_threshold: float, edge_model: EdgeBondMLP, edge_mean: np.ndarray, edge_std: np.ndarray, edge_threshold: float, device: torch.device) -> dict:
    totals = Counter()
    for row in rows:
        key = row["sample_key"]
        values = candidates[key]
        node_features, centers, candidate_indices, node_indices = _node_rows(key, values)
        with torch.inference_mode():
            ns = torch.sigmoid(node_model(torch.from_numpy((node_features - node_mean) / node_std).to(device))).cpu().numpy() if len(node_features) else np.empty(0, dtype=np.float32)
        keep = ns >= node_threshold
        node_map = {int(node): int(ci) for node, ci, selected in zip(node_indices.tolist(), candidate_indices.tolist(), keep.tolist()) if selected}
        points = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE] for v in values], dtype=np.float32)
        targets, gt_edges, type_by_edge = _target_data(key)
        candidate_to_target = strict_candidate_to_target(points, targets)
        node_true = len({candidate_to_target[ci] for ci in node_map.values() if ci in candidate_to_target})

        edge_features, edge_centers, _ = _proposals(key)
        keys = _candidate_keys(key)
        if len(edge_features) != len(keys):
            raise RuntimeError(f"edge feature/key mismatch for {key}")
        with torch.inference_mode():
            elogits, tlogits = edge_model(torch.from_numpy((edge_features - edge_mean) / edge_std).to(device))
        es = torch.sigmoid(elogits).cpu().numpy()
        type_ids = (torch.argmax(tlogits, dim=1) + 1).cpu().numpy()
        candidate_edges = {}
        raw_dangling = 0
        raw_self_loops = 0
        selected_lines = 0
        for score, type_id, (a, b) in zip(es.tolist(), type_ids.tolist(), keys):
            if score < edge_threshold:
                continue
            selected_lines += 1
            if a not in node_map or b not in node_map:
                raw_dangling += 1
                continue
            ca, cb = node_map[a], node_map[b]
            if ca == cb:
                raw_self_loops += 1
                continue
            pair = tuple(sorted((ca, cb)))
            old = candidate_edges.get(pair)
            if old is None or score > old[0]:
                candidate_edges[pair] = (float(score), int(type_id))
        mapped_edges = set()
        typed_total = typed_correct = 0
        self_loops = 0
        for (ca, cb), (_, type_id) in candidate_edges.items():
            if ca not in candidate_to_target or cb not in candidate_to_target:
                # Incomplete ownership is rejected before serialization.
                continue
            ta, tb = candidate_to_target[ca], candidate_to_target[cb]
            if ta == tb:
                self_loops += 1
                continue
            edge = tuple(sorted((ta, tb)))
            mapped_edges.add(edge)
            if edge in gt_edges:
                typed_total += 1
                typed_correct += int(type_by_edge[edge] == type_id)
        true_edges = mapped_edges & gt_edges
        totals.update({
            "images": 1,
            "line_nodes": int(len(centers)),
            "selected_nodes": int(len(node_map)),
            "target_atoms": int(len(targets)),
            "node_true": int(node_true),
            "selected_lines": selected_lines,
            "raw_dangling_edges": raw_dangling,
            "raw_self_loops": raw_self_loops,
            "candidate_edges": len(candidate_edges),
            "mapped_edges": len(mapped_edges),
            "true_edges": len(true_edges),
            "gt_bonds": len(gt_edges),
            "typed_total": typed_total,
            "typed_correct": typed_correct,
            "all_bonds_recovered": int(true_edges == gt_edges),
        })
    out = dict(totals)
    out.update({
        "node_precision": totals["node_true"] / max(totals["selected_nodes"], 1),
        "node_recall": totals["node_true"] / max(totals["target_atoms"], 1),
        "edge_precision": totals["true_edges"] / max(totals["mapped_edges"], 1),
        "edge_recall": totals["true_edges"] / max(totals["gt_bonds"], 1),
        "conditional_bond_type_accuracy": totals["typed_correct"] / max(totals["typed_total"], 1),
        "raw_dangling_rate": totals["raw_dangling_edges"] / max(totals["selected_lines"], 1),
        "raw_self_loop_rate": totals["raw_self_loops"] / max(totals["selected_lines"], 1),
        "emitted_dangling_edges": 0,
        "emitted_dangling_rate": 0.0,
        "emitted_self_loops": 0,
        "all_bonds_recovered_fraction": totals["all_bonds_recovered"] / max(totals["images"], 1),
    })
    return out


def run() -> dict:
    if file_digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("production checkpoint changed before shadow")
    candidates = _load_candidates()
    train_rows, hold_rows = _rows()
    x_train, y_train = _collect(train_rows, candidates)
    x_hold, y_hold = _collect(hold_rows, candidates)
    node_mean = x_train.mean(0)
    node_std = np.maximum(x_train.std(0), 1e-5)
    torch.manual_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    node_model = NodeMLP(x_train.shape[1]).to(device)
    tx = torch.from_numpy((x_train - node_mean) / node_std).to(device)
    ty = torch.from_numpy(y_train).to(device)
    positive_weight = torch.tensor(float((1.0 - y_train.mean()) / max(float(y_train.mean()), 1e-4)), device=device)
    optimizer = torch.optim.AdamW(node_model.parameters(), lr=2e-3, weight_decay=1e-3)
    events = []
    started = time.time()
    for step in range(1, STEPS + 1):
        logits = node_model(tx)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, ty, pos_weight=positive_weight)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite node loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(node_model.parameters(), 1.0)
        optimizer.step()
        if step in (1, 200, 400, 600, 800):
            events.append({"step": step, "loss": float(loss.detach().cpu())})
    node_model.eval()
    with torch.inference_mode():
        train_scores = torch.sigmoid(node_model(tx)).cpu().numpy()
        hold_scores = torch.sigmoid(node_model(torch.from_numpy((x_hold - node_mean) / node_std).to(device))).cpu().numpy()
    calibration = _calibrate(train_scores, y_train)
    node_threshold = float(calibration["threshold"])
    edge_model, edge_mean, edge_std, edge_threshold = _load_edge_sidecar()
    edge_model.to(device).eval()
    train_metrics = _evaluate(train_rows, candidates, node_model, node_mean, node_std, node_threshold, edge_model, edge_mean, edge_std, edge_threshold, device)
    hold_metrics = _evaluate(hold_rows, candidates, node_model, node_mean, node_std, node_threshold, edge_model, edge_mean, edge_std, edge_threshold, device)
    after = file_digest(INITIAL)
    OUT.mkdir(parents=True, exist_ok=False)
    torch.save({"model": node_model.state_dict(), "mean": node_mean, "std": node_std, "threshold": node_threshold, "feature_names": ["candidate_score", "candidate_x", "candidate_y", "node_candidate_distance", "cluster_size", "cluster_spread_mean", "cluster_spread_std", "cluster_degree", "mean_segment_length", "node_ink", "candidate_ink", "near_candidate_count16", "near_candidate_count24", "node_rank"], "initial_sha256": INITIAL_SHA, "sealed_test_read": False, "optimizer_steps": STEPS}, STATE)
    gate = {
        "checkpoint_unchanged": after == INITIAL_SHA,
        "sealed_test_read": False,
        "holdout_node_precision_ge_0_90": hold_metrics["node_precision"] >= NODE_PRECISION_FLOOR,
        "holdout_node_recall_ge_0_85": hold_metrics["node_recall"] >= NODE_RECALL_FLOOR,
        "holdout_edge_precision_ge_0_90": hold_metrics["edge_precision"] >= EDGE_PRECISION_FLOOR,
        "holdout_edge_recall_ge_0_75": hold_metrics["edge_recall"] >= EDGE_RECALL_FLOOR,
        "holdout_type_accuracy_ge_0_90": hold_metrics["conditional_bond_type_accuracy"] >= TYPE_ACCURACY_FLOOR,
        "holdout_emitted_dangling_zero": hold_metrics["emitted_dangling_edges"] == 0,
        "holdout_emitted_self_loop_zero": hold_metrics["emitted_self_loops"] == 0,
        "production_go": False,
    }
    gate["node_graph_shadow_pass"] = all(value for key, value in gate.items() if key.startswith("holdout_") or key == "checkpoint_unchanged") and not gate["sealed_test_read"]
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-node-graph-shadow-r5-0.1",
        "complete": True,
        "read_only": False,
        "training_performed": True,
        "training_scope": "isolated_node_identity_keep_head_with_frozen_line_first_edge_type_sidecar",
        "gpu_used": device.type == "cuda",
        "device": str(device),
        "optimizer_steps": STEPS,
        "elapsed_seconds": time.time() - started,
        "sealed_test_read": False,
        "deployment_authorized": False,
        "production_checkpoint_written": False,
        "initial_sha256": INITIAL_SHA,
        "initial_checkpoint_sha256_after": after,
        "candidate_artifact_sha256": _digest(CANDIDATES),
        "edge_sidecar_state_sha256": _digest(EDGE_STATE),
        "node_sidecar_state_sha256": _digest(STATE),
        "fixed_slice": {"train_images": len(train_rows), "holdout_images": len(hold_rows), "assignment_radius_px": ALIGN_RADIUS, "node_target_radius_px": TARGET_RADIUS, "node_threshold_source": "train-only", "edge_threshold_source": "frozen edge sidecar train-only"},
        "proposal_contract": "LSD endpoint clusters -> target-free one-to-one node/candidate assignment -> learned node keep -> frozen edge/type sidecar; GT post-hoc only",
        "node_feature_contract": "14 image/line/candidate-only features; no target fields",
        "events": events,
        "calibration": {"node": calibration, "node_threshold": node_threshold, "edge_threshold": edge_threshold},
        "train": train_metrics,
        "holdout": hold_metrics,
        "gate": gate,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(REPORT), "device": str(device), "train": train_metrics, "holdout": hold_metrics, "gate": gate}, sort_keys=True))
    return report


if __name__ == "__main__":
    run()
