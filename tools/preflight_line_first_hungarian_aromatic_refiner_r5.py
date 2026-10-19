#!/usr/bin/env python3
"""CPU preflight for a conservative aromatic type refiner.

The proposal field is the train-selected target-free Hungarian ownership
(radius 24 px, score regularizer lambda 8 px) followed by the frozen
line-first edge/type sidecar.  A small CPU MLP sees only frozen edge features,
edge/type logits, and candidate-graph cycle/degree evidence.  Ground-truth
atom/bond labels are joined after the complete feature field is built and are
never written to the sidecar.  The refiner is deliberately conservative: it
may demote a sidecar AROMATIC prediction to SINGLE, but never changes an
already predicted DOUBLE/TRIPLE edge or invents an edge.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import torch
from scipy.special import expit, softmax
from scipy.optimize import linear_sum_assignment
from torch import nn
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry
from tools import audit_line_first_hungarian_roundtrip_r5 as hungarian
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target
from tools.shadow_line_first_edge_bond_type_r5 import EdgeBondMLP, TYPE_NAMES, _candidate_keys, _proposals

DATA = hungarian.DATA
CANDIDATES = hungarian.CANDIDATES
INITIAL = hungarian.audit.INITIAL
INITIAL_SHA = hungarian.audit.INITIAL_SHA
EDGE_STATE = hungarian.audit.STATE
IMAGE = hungarian.IMAGE
OWNERSHIP_RADIUS = 24.0
OWNERSHIP_LAMBDA = 8.0
CLUSTER_RADIUS = 16.0
MAX_SEGMENTS = 512
TRAIN_LIMIT = 160
HOLDOUT_LIMIT = 80
STEPS = 500
SEED = 20261019
AROMATIC_PRECISION_FLOOR = 0.90
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-hungarian-aromatic-refiner-r5-cpu-v2"
REPORT = OUT / "report.json"
STATE = OUT / "sidecar-state.pt"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_rows() -> dict[str, list[dict[str, float]]]:
    values = {}
    forbidden = {"targets", "target_indices", "matched", "element", "visible_element", "atom_id", "graph", "bonds", "ground_truth", "label"}
    for line_no, line in enumerate(CANDIDATES.read_text().splitlines(), 1):
        row = json.loads(line)
        if row.get("source") != "shadow" or set(row) & forbidden:
            raise ValueError(f"candidate contract failed at line {line_no}")
        values[row["sample_key"]] = row["candidates"]
    if len(values) != 1170:
        raise ValueError(f"expected 1170 candidate rows, found {len(values)}")
    return values


def _rows() -> tuple[list[dict], list[dict], list[dict]]:
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row.get("render_status") == "accepted"]
    train = sorted([row for row in rows if row["split"] == "train"], key=lambda row: row["sample_key"])[:TRAIN_LIMIT]
    hold = sorted([row for row in rows if row["split"] == "holdout"], key=lambda row: row["sample_key"])[:HOLDOUT_LIMIT]
    full = sorted([row for row in rows if row["split"] == "holdout"], key=lambda row: row["sample_key"])
    if len(train) != TRAIN_LIMIT or len(hold) != HOLDOUT_LIMIT or len(full) != 474:
        raise RuntimeError("fixed train/holdout rows changed")
    return train, hold, full


def _assign(centers: np.ndarray, candidates: np.ndarray) -> dict[int, int]:
    if not len(centers) or not len(candidates):
        return {}
    d = np.linalg.norm(centers[:, None, :] - candidates[None, :, :2], axis=-1)
    cost = np.where(d <= OWNERSHIP_RADIUS, d - OWNERSHIP_LAMBDA * candidates[None, :, 2], 1e5 + d)
    rr, cc = linear_sum_assignment(cost)
    return {int(i): int(j) for i, j in zip(rr, cc) if d[i, j] <= OWNERSHIP_RADIUS}


def _cycle_features(selected_pairs: list[tuple[int, int]]) -> dict[tuple[int, int], np.ndarray]:
    graph = nx.Graph()
    graph.add_edges_from(selected_pairs)
    degree = dict(graph.degree())
    cycles = list(nx.cycle_basis(graph))
    flags: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0] * 7)
    for cycle in cycles:
        length = len(cycle)
        if not 3 <= length <= 8:
            continue
        slot = length - 3
        for index in range(length):
            edge = tuple(sorted((cycle[index], cycle[(index + 1) % length])))
            flags[edge][slot] = 1.0
            flags[edge][6] = 1.0
    output = {}
    for pair in selected_pairs:
        a, b = pair
        output[pair] = np.asarray(flags[pair] + [min(degree.get(a, 0), 8) / 8.0, min(degree.get(b, 0), 8) / 8.0], dtype=np.float32)
    return output


def _target_labels(key: str, points: np.ndarray, selected: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, set[tuple[int, int]], dict[tuple[int, int], int], list[tuple[int, str, bool]]]:
    base = DATA / "samples" / key
    layers = json.loads(base.with_suffix(".layers.json").read_text())
    graph = json.loads(base.with_suffix(".graph.json").read_text())
    targets = np.asarray([atom["center"] for atom in layers["atoms"]], dtype=np.float32)
    c2t = strict_candidate_to_target(points[:, :2], targets)
    gt = {tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"])))) for bond in graph["bonds"]}
    type_by = {}
    for bond in graph["bonds"]:
        edge = tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
        name = str(bond.get("chemical_type", "OTHER")).upper()
        type_by[edge] = TYPE_NAMES.index(name) if name in TYPE_NAMES else TYPE_NAMES.index("OTHER")
    labels = np.zeros(len(selected), dtype=np.float32)
    type_labels = np.zeros(len(selected), dtype=np.int64)
    mapped = np.zeros(len(selected), dtype=bool)
    node_ids = sorted({node for item in selected for node in item["pair"]})
    nodes = []
    for candidate in node_ids:
        if candidate in c2t:
            target = c2t[candidate]
            atom = layers["atoms"][target]
            nodes.append((candidate, str(atom.get("element") or "C"), bool(atom.get("aromatic", False))))
    for index, item in enumerate(selected):
        a, b = item["pair"]
        if a not in c2t or b not in c2t or c2t[a] == c2t[b]:
            continue
        edge = tuple(sorted((c2t[a], c2t[b])))
        mapped[index] = True
        if edge in gt:
            labels[index] = float(type_by.get(edge, TYPE_NAMES.index("OTHER")) == TYPE_NAMES.index("AROMATIC"))
            type_labels[index] = int(type_by.get(edge, TYPE_NAMES.index("OTHER")))
    return labels, type_labels, mapped, gt, type_by, nodes


def _extract(key: str, values: list[dict[str, float]], edge_model: EdgeBondMLP, edge_mean: np.ndarray, edge_std: np.ndarray, edge_threshold: float) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    """Build features before any target lookup; labels are joined separately."""
    edge_features, centers, _ = _proposals(key)
    points = np.asarray([[value["x"] * IMAGE, value["y"] * IMAGE, value.get("score", 0.0)] for value in values], dtype=np.float32)
    node_map = _assign(centers.astype(np.float64), points.astype(np.float64))
    keys = _candidate_keys(key)
    if len(edge_features) != len(keys):
        raise RuntimeError(f"edge/key mismatch for {key}")
    with torch.inference_mode():
        logits, type_logits = edge_model(torch.from_numpy((edge_features - edge_mean) / edge_std))
    logits_np = logits.cpu().numpy()
    probabilities = expit(logits_np)
    type_probabilities = softmax(type_logits.cpu().numpy(), axis=1)
    selected_by_pair: dict[tuple[int, int], dict] = {}
    for index, (line_a, line_b) in enumerate(keys):
        if probabilities[index] < edge_threshold or line_a not in node_map or line_b not in node_map:
            continue
        candidate_a, candidate_b = node_map[line_a], node_map[line_b]
        if candidate_a == candidate_b:
            continue
        pair = tuple(sorted((candidate_a, candidate_b)))
        old = selected_by_pair.get(pair)
        item = {"pair": pair, "edge_index": int(index), "edge_logit": float(logits_np[index]), "edge_probability": float(probabilities[index]), "type_logits": type_logits[index].cpu().numpy().astype(np.float32), "type_probabilities": type_probabilities[index].astype(np.float32), "base_type": int(np.argmax(type_logits[index].cpu().numpy()) + 1), "geometry": edge_features[index].astype(np.float32)}
        if old is None or item["edge_probability"] > old["edge_probability"]:
            selected_by_pair[pair] = item
    selected = list(selected_by_pair.values())
    cycle = _cycle_features([item["pair"] for item in selected])
    rows = []
    for item in selected:
        item["cycle_features"] = cycle[item["pair"]]
        rows.append(np.concatenate((item["geometry"], np.asarray([item["edge_logit"], item["edge_probability"]], dtype=np.float32), item["type_probabilities"], item["cycle_features"])))
    metadata = {"key": key, "points": points, "selected": selected, "node_map": node_map}
    return np.asarray(rows, dtype=np.float32), points, selected, metadata


class AromaticMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 48), nn.GELU(), nn.Linear(48, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _calibrate(scores: np.ndarray, labels: np.ndarray) -> dict:
    order = np.argsort(-scores, kind="stable")
    y = labels[order] > 0.5
    tp = np.cumsum(y); fp = np.cumsum(~y)
    precision = tp / np.maximum(tp + fp, 1); recall = tp / max(int(y.sum()), 1)
    valid = np.where(precision >= AROMATIC_PRECISION_FLOOR)[0]
    if not len(valid):
        return {"threshold": 0.5, "precision": float(precision[-1]) if len(precision) else 0.0, "recall": float(recall[-1]) if len(recall) else 0.0, "found": False}
    best = float(recall[valid].max()); chosen = valid[np.where(recall[valid] == best)[0][-1]]
    return {"threshold": float(scores[order[chosen]]), "precision": float(precision[chosen]), "recall": float(recall[chosen]), "found": True}


def _evaluate(rows: list[dict], candidate_rows: dict[str, list[dict[str, float]]], edge_model: EdgeBondMLP, edge_mean: np.ndarray, edge_std: np.ndarray, edge_threshold: float, aromatic_model: AromaticMLP, aromatic_mean: np.ndarray, aromatic_std: np.ndarray, aromatic_threshold: float, *, include_details: bool = False) -> dict:
    totals = Counter(); by_profile = defaultdict(Counter)
    for row in rows:
        features, points, selected, metadata = _extract(row["sample_key"], candidate_rows[row["sample_key"]], edge_model, edge_mean, edge_std, edge_threshold)
        with torch.inference_mode():
            scores = torch.sigmoid(aromatic_model(torch.from_numpy((features - aromatic_mean) / aromatic_std))).cpu().numpy() if len(features) else np.empty(0, dtype=np.float32)
        labels, type_labels, mapped_mask, gt, type_by, nodes = _target_labels(row["sample_key"], points, selected)
        refined = []
        baseline = []
        for item, score in zip(selected, scores):
            base_type = int(item["base_type"])
            refined_type = base_type
            if base_type == TYPE_NAMES.index("AROMATIC"):
                refined_type = TYPE_NAMES.index("AROMATIC") if score >= aromatic_threshold else TYPE_NAMES.index("SINGLE")
            baseline.append({"pair": item["pair"], "type": base_type})
            refined.append({"pair": item["pair"], "type": refined_type})
        base_mapped = {}; refined_mapped = {}
        c2t = strict_candidate_to_target(points[:, :2], np.asarray([atom["center"] for atom in json.loads((DATA / "samples" / row["sample_key"]).with_suffix(".layers.json").read_text())["atoms"]], dtype=np.float32))
        for item in baseline:
            a, b = item["pair"]
            if a in c2t and b in c2t and c2t[a] != c2t[b]:
                base_mapped[tuple(sorted((c2t[a], c2t[b])))] = item["type"]
        for item in refined:
            a, b = item["pair"]
            if a in c2t and b in c2t and c2t[a] != c2t[b]:
                refined_mapped[tuple(sorted((c2t[a], c2t[b])))] = item["type"]
        true_edges = set(base_mapped) & gt
        base_type_correct = sum(base_mapped[edge] == type_by[edge] for edge in true_edges)
        refined_type_correct = sum(refined_mapped[edge] == type_by[edge] for edge in true_edges)
        # Oracle-element RDKit sanitize potential: edge ownership is frozen;
        # only the type field is replaced by the refiner.
        target_nodes = nodes
        node_index = {candidate: index for index, (candidate, _, _) in enumerate(target_nodes)}
        base_edges = {
            (node_index[a], node_index[b]): item["type"]
            for item in baseline
            for a, b in [item["pair"]]
            if a in node_index and b in node_index
        }
        refined_edges = {
            (node_index[a], node_index[b]): item["type"]
            for item in refined
            for a, b in [item["pair"]]
            if a in node_index and b in node_index
        }
        base_ok, base_error, _ = chemistry._sanitize_graph(target_nodes, base_edges)
        refined_ok, refined_error, _ = chemistry._sanitize_graph(target_nodes, refined_edges)
        aromatic_id = TYPE_NAMES.index("AROMATIC")
        aromatic_true = sum(type_by.get(edge) == aromatic_id for edge in true_edges)
        base_aromatic_tp = sum(base_mapped.get(edge) == aromatic_id and type_by.get(edge) == aromatic_id for edge in true_edges)
        base_aromatic_fp = sum(base_mapped.get(edge) == aromatic_id and type_by.get(edge) != aromatic_id for edge in true_edges)
        base_aromatic_fn = sum(base_mapped.get(edge) != aromatic_id and type_by.get(edge) == aromatic_id for edge in true_edges)
        refined_aromatic_tp = sum(refined_mapped.get(edge) == aromatic_id and type_by.get(edge) == aromatic_id for edge in true_edges)
        refined_aromatic_fp = sum(refined_mapped.get(edge) == aromatic_id and type_by.get(edge) != aromatic_id for edge in true_edges)
        refined_aromatic_fn = sum(refined_mapped.get(edge) != aromatic_id and type_by.get(edge) == aromatic_id for edge in true_edges)
        current = Counter({"images": 1, "selected_edges": len(selected), "mapped_edges": len(base_mapped), "true_edges": len(true_edges), "gt_bonds": len(gt), "base_type_correct": base_type_correct, "refined_type_correct": refined_type_correct, "type_total": len(true_edges), "base_oracle_sanitize_ok": int(base_ok), "refined_oracle_sanitize_ok": int(refined_ok), "base_aromatic_sanitize_errors": int(base_error == "aromatic"), "refined_aromatic_sanitize_errors": int(refined_error == "aromatic"), "base_valence_errors": int(base_error == "valence"), "refined_valence_errors": int(refined_error == "valence"), "aromatic_gt": aromatic_true, "base_aromatic_tp": base_aromatic_tp, "base_aromatic_fp": base_aromatic_fp, "base_aromatic_fn": base_aromatic_fn, "refined_aromatic_tp": refined_aromatic_tp, "refined_aromatic_fp": refined_aromatic_fp, "refined_aromatic_fn": refined_aromatic_fn, "refiner_demoted": sum(item["type"] == TYPE_NAMES.index("SINGLE") and base["type"] == TYPE_NAMES.index("AROMATIC") for item, base in zip(refined, baseline))})
        totals.update(current); by_profile[row["profile_id"]].update(current)
    def finish(counter: Counter) -> dict:
        return {**dict(counter), "edge_precision": counter["true_edges"] / max(counter["mapped_edges"], 1), "edge_recall": counter["true_edges"] / max(counter["gt_bonds"], 1), "base_type_accuracy": counter["base_type_correct"] / max(counter["type_total"], 1), "refined_type_accuracy": counter["refined_type_correct"] / max(counter["type_total"], 1), "base_oracle_sanitize_rate": counter["base_oracle_sanitize_ok"] / max(counter["images"], 1), "refined_oracle_sanitize_rate": counter["refined_oracle_sanitize_ok"] / max(counter["images"], 1), "base_aromatic_precision": counter["base_aromatic_tp"] / max(counter["base_aromatic_tp"] + counter["base_aromatic_fp"], 1), "base_aromatic_recall": counter["base_aromatic_tp"] / max(counter["aromatic_gt"], 1), "refined_aromatic_precision": counter["refined_aromatic_tp"] / max(counter["refined_aromatic_tp"] + counter["refined_aromatic_fp"], 1), "refined_aromatic_recall": counter["refined_aromatic_tp"] / max(counter["aromatic_gt"], 1), "refiner_demoted_rate": counter["refiner_demoted"] / max(counter["selected_edges"], 1)}
    return {"all": finish(totals), "by_profile": {key: finish(value) for key, value in sorted(by_profile.items())}}


def run() -> dict:
    rdBase.DisableLog("rdApp.error")
    if _digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    candidate_rows = _candidate_rows()
    train_rows, hold_rows, full_rows = _rows()
    edge_payload = torch.load(EDGE_STATE, map_location="cpu", weights_only=False)
    edge_model = EdgeBondMLP(len(edge_payload["feature_names"]))
    edge_model.load_state_dict(edge_payload["model"], strict=True); edge_model.eval()
    edge_mean = np.asarray(edge_payload["mean"], dtype=np.float32); edge_std = np.asarray(edge_payload["std"], dtype=np.float32); edge_threshold = float(edge_payload["threshold"])
    train_features = []; train_labels = []
    cache = {}
    for row in train_rows + hold_rows:
        f, points, selected, metadata = _extract(row["sample_key"], candidate_rows[row["sample_key"]], edge_model, edge_mean, edge_std, edge_threshold)
        labels, _, _, _, _, _ = _target_labels(row["sample_key"], points, selected)
        cache[row["sample_key"]] = (f, points, selected, metadata)
        train_features.append(f); train_labels.append(labels)
    x_train = np.concatenate(train_features[:len(train_rows)]); y_train = np.concatenate(train_labels[:len(train_rows)])
    x_hold = np.concatenate(train_features[len(train_rows):]); y_hold = np.concatenate(train_labels[len(train_rows):])
    aromatic_mean = x_train.mean(0); aromatic_std = np.maximum(x_train.std(0), 1e-5)
    torch.manual_seed(SEED); model = AromaticMLP(x_train.shape[1]); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    device = torch.device("cpu"); xt = torch.from_numpy((x_train - aromatic_mean) / aromatic_std); yt = torch.from_numpy(y_train)
    weight = torch.tensor(float((1.0 - y_train.mean()) / max(float(y_train.mean()), 1e-4)))
    events = []
    for step in range(1, STEPS + 1):
        logits = model(xt); loss = nn.functional.binary_cross_entropy_with_logits(logits, yt, pos_weight=weight)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step in (1, 100, 250, 500): events.append({"step": step, "loss": float(loss.detach())})
    model.eval()
    with torch.inference_mode(): train_scores = torch.sigmoid(model(xt)).numpy(); hold_scores = torch.sigmoid(model(torch.from_numpy((x_hold - aromatic_mean) / aromatic_std))).numpy()
    calibration = _calibrate(train_scores, y_train); aromatic_threshold = float(calibration["threshold"])
    train_metrics = _evaluate(train_rows, candidate_rows, edge_model, edge_mean, edge_std, edge_threshold, model, aromatic_mean, aromatic_std, aromatic_threshold)
    hold_metrics = _evaluate(hold_rows, candidate_rows, edge_model, edge_mean, edge_std, edge_threshold, model, aromatic_mean, aromatic_std, aromatic_threshold)
    full_metrics = _evaluate(full_rows, candidate_rows, edge_model, edge_mean, edge_std, edge_threshold, model, aromatic_mean, aromatic_std, aromatic_threshold)
    OUT.mkdir(parents=True, exist_ok=False)
    torch.save({"model": model.state_dict(), "mean": aromatic_mean, "std": aromatic_std, "threshold": aromatic_threshold, "feature_names": [f"edge_geometry_{i}" for i in range(17)] + ["edge_logit", "edge_probability"] + [f"type_probability_{name}" for name in TYPE_NAMES[1:]] + ["cycle3", "cycle4", "cycle5", "cycle6", "cycle7", "cycle8", "cycle_any", "degree_a", "degree_b"], "initial_sha256": INITIAL_SHA, "sealed_test_read": False, "optimizer_steps": STEPS}, STATE)
    after = _digest(INITIAL)
    gate = {"checkpoint_unchanged": after == INITIAL_SHA, "sealed_test_read": False, "holdout_refined_type_accuracy_ge_0_90": hold_metrics["all"]["refined_type_accuracy"] >= .90, "holdout_refined_oracle_sanitize_ge_0_90": hold_metrics["all"]["refined_oracle_sanitize_rate"] >= .90, "holdout_edge_recall_preserved_ge_0_75": hold_metrics["all"]["edge_recall"] >= .75, "holdout_edge_precision_preserved_ge_0_90": hold_metrics["all"]["edge_precision"] >= .90, "full_refined_oracle_sanitize_ge_0_90": full_metrics["all"]["refined_oracle_sanitize_rate"] >= .90, "production_go": False}
    gate["cpu_preflight_pass"] = all(value for key, value in gate.items() if key not in {"sealed_test_read", "production_go"}) and not gate["sealed_test_read"]
    report = {"schema_version": "chemvector-identity-aligned-line-first-hungarian-aromatic-refiner-r5-cpu-v2-0.1", "complete": True, "read_only": False, "training_performed": True, "training_scope": "isolated_cpu_aromatic_type_refiner", "gpu_used": False, "optimizer_steps": STEPS, "sealed_test_read": False, "deployment_authorized": False, "production_checkpoint_written": False, "initial_sha256": INITIAL_SHA, "initial_checkpoint_sha256_after": after, "candidate_artifact_sha256": _digest(CANDIDATES), "edge_sidecar_state_sha256": _digest(EDGE_STATE), "aromatic_sidecar_state_sha256": _digest(STATE), "fixed_slice": {"train_images": len(train_rows), "holdout_images": len(hold_rows), "full_holdout_images": len(full_rows), "ownership_radius_px": OWNERSHIP_RADIUS, "ownership_score_lambda_px": OWNERSHIP_LAMBDA, "cluster_radius_px": CLUSTER_RADIUS, "edge_threshold_source": "frozen train-only edge sidecar", "aromatic_threshold_source": "train-only"}, "proposal_contract": "target-free Hungarian ownership r24/lambda8 -> frozen edge/type sidecar -> target-free cycle/degree features; labels post-hoc only", "feature_names": [f"edge_geometry_{i}" for i in range(17)] + ["edge_logit", "edge_probability"] + [f"type_probability_{name}" for name in TYPE_NAMES[1:]] + ["cycle3", "cycle4", "cycle5", "cycle6", "cycle7", "cycle8", "cycle_any", "degree_a", "degree_b"], "events": events, "calibration": calibration, "train": train_metrics, "holdout": hold_metrics, "full_holdout": full_metrics, "gate": gate}
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(REPORT), "train": train_metrics["all"], "holdout": hold_metrics["all"], "full_holdout": full_metrics["all"], "gate": gate}, sort_keys=True))
    return report


if __name__ == "__main__":
    run()
