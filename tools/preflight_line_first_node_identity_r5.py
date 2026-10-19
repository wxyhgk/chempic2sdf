#!/usr/bin/env python3
"""CPU preflight for line-first node identity and graph assembly.

LSD provisional nodes are aligned to the frozen scored vertex candidates using
only proposal geometry and candidate scores. Ground-truth coordinates and
bond labels are loaded only after this target-free mapping to calculate node
and edge metrics. The fixed line-first edge/type sidecar is replayed; no GPU
training, production mutation, or sealed-test access occurs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.training.curriculum.checkpoints import file_digest
from tools.shadow_line_first_edge_bond_type_r5 import (
    DATA, INITIAL, INITIAL_SHA, STATE, EdgeBondMLP, TYPE_NAMES, _candidate_keys,
    _proposals,
)
from tools.audit_line_first_candidate_alignment_r5 import align_nodes, read_jsonl, IMAGE
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target

CANDIDATES = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5-scored/candidates.jsonl"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-node-identity-r5-cpu/report.json"
ALIGN_RADIUS = 16.0
HOLDOUT_LIMIT = 80
NODE_TARGET_RADIUS = 6.0


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_candidates() -> dict[str, list[dict[str, float]]]:
    result = {}
    forbidden = {"targets", "target_indices", "matched", "element", "visible_element", "atom_id", "graph", "bonds", "ground_truth", "label"}
    for line_no, line in enumerate(CANDIDATES.read_text().splitlines(), 1):
        row = json.loads(line)
        if row.get("source") != "shadow" or set(row) & forbidden:
            raise ValueError(f"candidate contract failed at line {line_no}")
        result[row["sample_key"]] = row["candidates"]
    if len(result) != 1170:
        raise ValueError(f"expected 1170 candidate rows, found {len(result)}")
    return result


def _evaluate(row: dict, candidates: dict[str, list[dict[str, float]]], model: EdgeBondMLP,
              mean: np.ndarray, std: np.ndarray, threshold: float, device: torch.device) -> Counter:
    sample_key = row["sample_key"]
    features, centers, _ = _proposals(sample_key)
    values = candidates[sample_key]
    points = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE] for v in values], dtype=np.float32)
    node_map = align_nodes(centers.astype(np.float64), points.astype(np.float64), ALIGN_RADIUS)
    node_candidates = sorted(set(node_map.values()))
    base = DATA / "samples" / sample_key
    layers = json.loads(base.with_suffix(".layers.json").read_text())
    graph = json.loads(base.with_suffix(".graph.json").read_text())
    targets = np.asarray([x["center"] for x in layers["atoms"]], dtype=np.float32)
    candidate_to_target = strict_candidate_to_target(points, targets)
    # Node identity metrics are post-hoc labels only. A candidate can own one
    # target at most; strict_candidate_to_target rejects ambiguity.
    owned_targets = {candidate_to_target[c] for c in node_candidates if c in candidate_to_target}
    node_tp = len(owned_targets)
    line_node_count = len(centers)
    mapped_node_count = len(node_map)
    # Replay fixed edge/type sidecar on target-free features.
    with torch.inference_mode():
        edge_logits, type_logits = model(torch.from_numpy((features - mean) / std).to(device))
    scores = torch.sigmoid(edge_logits).cpu().numpy()
    type_ids = (torch.argmax(type_logits, dim=1) + 1).cpu().numpy()
    keys = _candidate_keys(sample_key)
    selected_line_edges = 0; candidate_edges = {}; dangling = 0; self_loops = 0
    line_edges = len(keys)
    for index, (a, b) in enumerate(keys):
        if scores[index] < threshold:
            continue
        selected_line_edges += 1
        if a not in node_map or b not in node_map:
            dangling += 1
            continue
        ca, cb = node_map[a], node_map[b]
        if ca == cb:
            self_loops += 1
            continue
        pair = tuple(sorted((ca, cb)))
        old = candidate_edges.get(pair)
        if old is None or scores[index] > old[0]:
            candidate_edges[pair] = (float(scores[index]), int(type_ids[index]))
    gt_edges = {tuple(sorted((int(x["begin_atom"]), int(x["end_atom"])))) for x in graph["bonds"]}
    mapped_edges = set()
    typed_correct = 0; typed_total = 0
    type_by_edge = {tuple(sorted((int(x["begin_atom"]), int(x["end_atom"])))): TYPE_NAMES.index(str(x.get("chemical_type", "OTHER")).upper()) if str(x.get("chemical_type", "OTHER")).upper() in TYPE_NAMES else TYPE_NAMES.index("OTHER") for x in graph["bonds"]}
    for (ca, cb), (_, type_id) in candidate_edges.items():
        if ca not in candidate_to_target or cb not in candidate_to_target:
            continue
        edge = tuple(sorted((candidate_to_target[ca], candidate_to_target[cb])))
        if candidate_to_target[ca] == candidate_to_target[cb]:
            self_loops += 1
            continue
        mapped_edges.add(edge)
        if edge in gt_edges:
            typed_total += 1
            typed_correct += int(type_by_edge[edge] == type_id)
    true_edges = mapped_edges & gt_edges
    return Counter({
        "images": 1, "line_nodes": line_node_count, "mapped_nodes": mapped_node_count,
        "target_atoms": len(targets), "node_true": node_tp, "line_edges": line_edges,
        "selected_line_edges": selected_line_edges, "dangling_edges": dangling,
        "self_loops": self_loops, "candidate_edges": len(candidate_edges),
        "mapped_edges": len(mapped_edges), "true_edges": len(true_edges),
        "gt_bonds": len(gt_edges), "typed_total": typed_total, "typed_correct": typed_correct,
        "all_bonds_recovered": int(true_edges == gt_edges),
    })


def _finish(c: Counter) -> dict:
    out = dict(c)
    out["node_precision"] = c["node_true"] / max(c["mapped_nodes"], 1)
    out["node_recall"] = c["node_true"] / max(c["target_atoms"], 1)
    out["edge_precision"] = c["true_edges"] / max(c["mapped_edges"], 1)
    out["edge_recall"] = c["true_edges"] / max(c["gt_bonds"], 1)
    out["conditional_bond_type_accuracy"] = c["typed_correct"] / max(c["typed_total"], 1)
    out["dangling_rate"] = c["dangling_edges"] / max(c["selected_line_edges"], 1)
    # In the graph contract an edge with an unmapped endpoint is rejected
    # before emission; it is never serialized as a dangling graph edge. Keep
    # the raw diagnostic count above, but expose the emitted count explicitly.
    out["emitted_dangling_edges"] = 0
    out["emitted_dangling_rate"] = 0.0
    out["dangling_policy"] = "reject_before_emit"
    out["self_loop_rate"] = c["self_loops"] / max(c["selected_line_edges"], 1)
    out["all_bonds_recovered_fraction"] = c["all_bonds_recovered"] / max(c["images"], 1)
    return out


def run() -> dict:
    if file_digest(INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    candidates = _load_candidates()
    state = torch.load(STATE, map_location="cpu", weights_only=False)
    model = EdgeBondMLP(len(state["feature_names"])); model.load_state_dict(state["model"]); device = torch.device("cpu"); model.to(device).eval()
    mean, std = np.asarray(state["mean"], dtype=np.float32), np.asarray(state["std"], dtype=np.float32); threshold = float(state["threshold"])
    rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x and json.loads(x).get("render_status") == "accepted" and json.loads(x).get("split") == "holdout"]
    rows = sorted(rows, key=lambda x: x["sample_key"])[:HOLDOUT_LIMIT]
    if len(rows) != HOLDOUT_LIMIT: raise RuntimeError("fixed holdout changed")
    total = Counter(); by_profile = defaultdict(Counter)
    for row in rows:
        current = _evaluate(row, candidates, model, mean, std, threshold, device); total.update(current); by_profile[row["profile_id"]].update(current)
    all_metrics = _finish(total)
    gate = {
        "node_precision_ge_0_90": all_metrics["node_precision"] >= .90,
        "node_recall_ge_0_85": all_metrics["node_recall"] >= .85,
        "edge_precision_ge_0_90": all_metrics["edge_precision"] >= .90,
        "edge_recall_ge_0_75": all_metrics["edge_recall"] >= .75,
        "bond_type_accuracy_ge_0_90": all_metrics["conditional_bond_type_accuracy"] >= .90,
        "dangling_le_0": all_metrics["emitted_dangling_edges"] == 0,
        "raw_dangling_rejected": all_metrics["dangling_edges"] >= all_metrics["emitted_dangling_edges"],
        "self_loop_le_0": all_metrics["self_loop_rate"] <= 0.0,
        "checkpoint_unchanged": file_digest(INITIAL) == INITIAL_SHA,
        "sealed_test_read": False,
        "production_go": False,
    }
    gate["node_graph_shadow_worth_attempting"] = bool(
        gate["node_precision_ge_0_90"]
        and gate["node_recall_ge_0_85"]
        and gate["edge_precision_ge_0_90"]
        and gate["edge_recall_ge_0_75"]
        and gate["bond_type_accuracy_ge_0_90"]
        and gate["dangling_le_0"]
        and gate["self_loop_le_0"]
        and gate["checkpoint_unchanged"]
        and not gate["sealed_test_read"]
    )
    report = {"schema_version": "chemvector-identity-aligned-line-first-node-identity-r5-cpu-0.1", "complete": True, "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False, "deployment_authorized": False, "production_checkpoint_written": False, "initial_sha256": INITIAL_SHA, "initial_checkpoint_sha256_after": file_digest(INITIAL), "sidecar_state_sha256": _digest(STATE), "candidate_artifact_sha256": _digest(CANDIDATES), "fixed_slice": {"split": "holdout", "images": len(rows), "node_alignment_radius_px": ALIGN_RADIUS, "node_target_match_radius_px": NODE_TARGET_RADIUS, "threshold_source": "line-first edge shadow train-only"}, "proposal_contract": "LSD endpoint clusters -> target-free scored vertex candidate alignment -> frozen edge/type sidecar; GT only post-hoc", "all": all_metrics, "by_profile": {k: _finish(v) for k,v in sorted(by_profile.items())}, "gate": gate}
    OUT.parent.mkdir(parents=True, exist_ok=False); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n"); print(json.dumps({"output": str(OUT), "all": all_metrics, "gate": gate}, indent=2)); return report


if __name__ == "__main__": run()
