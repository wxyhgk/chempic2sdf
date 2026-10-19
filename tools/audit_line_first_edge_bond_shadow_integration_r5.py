#!/usr/bin/env python3
"""Independent holdout graph audit for the isolated line-first sidecar.

The sidecar is loaded as an inference artifact only.  Proposals and scores
are produced without target data; target layers/graphs are joined afterward
to calculate edge recovery and bond-order metrics.  No production state is
written and no sealed test rows are read.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.checkpoints import file_digest
from tools.preflight_line_first_edge_features_r5 import DATA, CLUSTER_RADIUS, LABEL_RADIUS, MAX_SEGMENTS, _proposals
from tools.shadow_line_first_edge_bond_type_r5 import OUT as SHADOW_OUT, EdgeBondMLP, TYPE_NAMES, _candidate_keys
from tools.audit_line_first_graph_traversal_r5 import cluster_endpoints
from chemvector.components.vision.line_segment_rescue import LineSegmentRescueConfig, detect_raster_line_segments

INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-edge-bond-integration-r5/report.json"
HOLDOUT_LIMIT = 80


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _node_map(centers: np.ndarray, targets: np.ndarray) -> dict[int, int]:
    distances = np.linalg.norm(centers[:, None] - targets[None], axis=-1) if len(centers) and len(targets) else np.empty((0, 0))
    pairs = [(float(distances[i, j]), i, j) for i in range(len(centers)) for j in range(len(targets)) if distances[i, j] <= LABEL_RADIUS]
    mapping: dict[int, int] = {}; used: set[int] = set()
    for _, node, target in sorted(pairs):
        if node not in mapping and target not in used:
            mapping[node] = target; used.add(target)
    return mapping


def _evaluate(row: dict, model: EdgeBondMLP, mean: np.ndarray, std: np.ndarray, threshold: float, device: torch.device) -> Counter:
    features, centers, _ = _proposals(row["sample_key"]); keys = _candidate_keys(row["sample_key"])
    with torch.inference_mode():
        edge_logits, type_logits = model(torch.from_numpy((features - mean) / std).to(device))
    edge_scores = torch.sigmoid(edge_logits).detach().cpu().numpy(); type_pred = (torch.argmax(type_logits, dim=1) + 1).detach().cpu().numpy()
    keep = edge_scores >= threshold
    base = DATA / "samples" / row["sample_key"]
    layers = json.loads(base.with_suffix(".layers.json").read_text()); graph = json.loads(base.with_suffix(".graph.json").read_text())
    targets = np.asarray([atom["center"] for atom in layers["atoms"]], dtype=np.float32)
    mapping = _node_map(centers, targets)
    gt_types = {}; gt_edges = set()
    for bond in graph["bonds"]:
        edge = tuple(sorted((int(bond["begin_atom"]), int(bond["end_atom"]))))
        gt_edges.add(edge); gt_types[edge] = TYPE_NAMES.index(str(bond.get("chemical_type", "OTHER")).upper()) if str(bond.get("chemical_type", "OTHER")).upper() in TYPE_NAMES else TYPE_NAMES.index("OTHER")
    pred: dict[tuple[int, int], int] = {}
    for index, (a, b) in enumerate(keys):
        if not keep[index] or a not in mapping or b not in mapping or mapping[a] == mapping[b]:
            continue
        edge = tuple(sorted((mapping[a], mapping[b]))); pred.setdefault(edge, int(type_pred[index]))
    true = set(pred) & gt_edges
    type_correct = sum(pred[edge] == gt_types[edge] for edge in true)
    return Counter({"images": 1, "gt_bonds": len(gt_edges), "candidate_edges": len(keys), "mapped_nodes": len(mapping), "selected_edges": len(pred), "true_edges": len(true), "typed_true_edges": len(true), "typed_correct": type_correct, "all_bonds_recovered": int(true == gt_edges), "self_loop": 0})


def run() -> dict:
    if file_digest(INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    report_path = SHADOW_OUT / "report.json"; state_path = SHADOW_OUT / "sidecar-state.pt"
    receipt = json.loads(report_path.read_text());
    if receipt.get("initial_sha256") != INITIAL_SHA or receipt.get("sealed_test_read") is not False: raise RuntimeError("invalid sidecar provenance")
    state = torch.load(state_path, map_location="cpu", weights_only=False); feature_names = state["feature_names"]
    model = EdgeBondMLP(len(feature_names)); model.load_state_dict(state["model"]); device = torch.device("cpu"); model.to(device).eval()
    mean, std = np.asarray(state["mean"], dtype=np.float32), np.asarray(state["std"], dtype=np.float32); threshold = float(state["threshold"])
    rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x and json.loads(x).get("render_status") == "accepted" and json.loads(x).get("split") == "holdout"]
    rows = sorted(rows, key=lambda r: r["sample_key"])[:HOLDOUT_LIMIT]
    if len(rows) != HOLDOUT_LIMIT: raise RuntimeError("fixed holdout changed")
    total = Counter(); by_profile = {}
    for row in rows:
        current = _evaluate(row, model, mean, std, threshold, device)
        total.update(current); by_profile.setdefault(row["profile_id"], Counter()).update(current)
    def finish(c: Counter) -> dict:
        out = dict(c); out["edge_precision"] = c["true_edges"] / max(c["selected_edges"], 1); out["edge_recall"] = c["true_edges"] / max(c["gt_bonds"], 1); out["conditional_bond_type_accuracy"] = c["typed_correct"] / max(c["typed_true_edges"], 1); out["all_bonds_recovered_fraction"] = c["all_bonds_recovered"] / max(c["images"], 1); return out
    after = file_digest(INITIAL)
    report = {"schema_version": "chemvector-identity-aligned-line-first-edge-bond-integration-r5-0.1", "complete": True, "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False, "deployment_authorized": False, "production_checkpoint_written": False, "initial_sha256": INITIAL_SHA, "initial_checkpoint_sha256_after": after, "sidecar_report_sha256": _digest(report_path), "sidecar_state_sha256": _digest(state_path), "fixed_slice": {"split": "holdout", "images": len(rows), "node_label_radius_px": LABEL_RADIUS, "threshold_source": "sidecar train-only"}, "all": finish(total), "by_profile": {k: finish(v) for k,v in sorted(by_profile.items())}, "gate": {"checkpoint_unchanged": after == INITIAL_SHA, "sealed_test_read": False, "target_free_proposal": True, "production_go": False}}
    OUT.parent.mkdir(parents=True, exist_ok=False); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n"); print(json.dumps({"output": str(OUT), "all": report["all"], "gate": report["gate"]}, indent=2)); return report


if __name__ == "__main__": run()
