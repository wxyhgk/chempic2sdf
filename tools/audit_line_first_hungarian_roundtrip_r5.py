"""Round-trip gate for the train-selected score-distance ownership policy."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as audit  # noqa: E402

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
CANDIDATES = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5-scored/candidates.jsonl"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-hungarian-roundtrip-r5/report.json"
RADIUS = 24.0
LAMBDA = 8.0
IMAGE = 768.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assign(centers: np.ndarray, candidates: np.ndarray) -> dict[int, int]:
    if not len(centers) or not len(candidates):
        return {}
    d = np.linalg.norm(centers[:, None, :] - candidates[None, :, :2], axis=-1)
    cost = np.where(d <= RADIUS, d - LAMBDA * candidates[None, :, 2], 1e5 + d)
    rows, cols = linear_sum_assignment(cost)
    return {int(row): int(col) for row, col in zip(rows, cols) if d[row, col] <= RADIUS}


def evaluate(row, candidates, model, mean, std, threshold):
    key = row["sample_key"]
    features, centers, _ = audit._proposals(key)
    values = candidates[key]
    points = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE, v.get("score", 0.0)] for v in values], dtype=np.float32)
    node_map = assign(centers.astype(np.float64), points.astype(np.float64))
    with torch.inference_mode():
        edge_logits, type_logits = model(torch.from_numpy((features - mean) / std))
    scores = torch.sigmoid(edge_logits).cpu().numpy(); kinds = (torch.argmax(type_logits, dim=1) + 1).cpu().numpy()
    selected = {}; dangling = 0
    for index, (a, b) in enumerate(audit._candidate_keys(key)):
        if scores[index] < threshold:
            continue
        if a not in node_map or b not in node_map:
            dangling += 1; continue
        ca, cb = node_map[a], node_map[b]
        if ca == cb:
            continue
        pair = tuple(sorted((ca, cb))); old = selected.get(pair)
        if old is None or scores[index] > old[0]:
            selected[pair] = (float(scores[index]), int(kinds[index]))
    base = DATA / "samples" / key
    layers = json.loads(base.with_suffix(".layers.json").read_text()); graph = json.loads(base.with_suffix(".graph.json").read_text())
    targets = np.asarray([x["center"] for x in layers["atoms"]], dtype=np.float32)
    c2t = audit.strict_candidate_to_target(points[:, :2], targets)
    gt = {tuple(sorted((int(x["begin_atom"]), int(x["end_atom"])))) for x in graph["bonds"]}
    gt_types = {tuple(sorted((int(x["begin_atom"]), int(x["end_atom"])))): (audit.TYPE_NAMES.index(str(x.get("chemical_type", "OTHER")).upper()) if str(x.get("chemical_type", "OTHER")).upper() in audit.TYPE_NAMES else audit.TYPE_NAMES.index("OTHER")) for x in graph["bonds"]}
    mapped = {}; node_ids = sorted({x for pair in selected for x in pair})
    for (a, b), (_, kind) in selected.items():
        if a in c2t and b in c2t and c2t[a] != c2t[b]:
            mapped[tuple(sorted((c2t[a], c2t[b])))] = kind
    true = set(mapped) & gt; type_correct = sum(mapped[e] == gt_types[e] for e in true)
    target_nodes = [(candidate, str(layers["atoms"][target]["element"] or "C"), bool(layers["atoms"][target].get("aromatic", False))) for candidate, target in sorted(c2t.items()) if candidate in node_ids]
    node_index = {candidate: index for index, (candidate, _, _) in enumerate(target_nodes)}
    edges = {(node_index[a], node_index[b]): kind for (a, b), (_, kind) in selected.items() if a in node_index and b in node_index}
    oracle_ok, error, _ = audit._sanitize_graph(target_nodes, edges)
    candidate_nodes = [(candidate, "C", False) for candidate in node_ids]
    candidate_index = {candidate: index for index, candidate in enumerate(node_ids)}
    c_edges = {(candidate_index[a], candidate_index[b]): kind for (a, b), (_, kind) in selected.items()}
    candidate_ok, candidate_error, _ = audit._sanitize_graph(candidate_nodes, c_edges)
    return Counter({"images": 1, "selected_edges": len(selected), "dangling_edges": dangling, "mapped_edges": len(mapped), "true_edges": len(true), "gt_bonds": len(gt), "type_correct": type_correct, "type_total": len(true), "oracle_sanitize_ok": int(oracle_ok), "oracle_aromatic_errors": int(error == "aromatic"), "oracle_valence_errors": int(error == "valence"), "candidate_sanitize_ok": int(candidate_ok), "candidate_aromatic_errors": int(candidate_error == "aromatic"), "candidate_valence_errors": int(candidate_error == "valence"), "all_bonds_recovered": int(true == gt)})


def finish(c: Counter) -> dict:
    return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1), "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1), "bond_type_accuracy": c["type_correct"] / max(c["type_total"], 1), "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1), "oracle_sanitize_rate": c["oracle_sanitize_ok"] / max(c["images"], 1), "candidate_sanitize_rate": c["candidate_sanitize_ok"] / max(c["images"], 1), "all_bonds_recovered_fraction": c["all_bonds_recovered"] / max(c["images"], 1)}


def main():
    rdBase.DisableLog("rdApp.error")
    if digest(audit.INITIAL) != audit.INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    candidates = audit._candidate_rows()
    state = torch.load(audit.STATE, map_location="cpu", weights_only=False)
    model = audit.EdgeBondMLP(len(state["feature_names"])); model.load_state_dict(state["model"]); model.eval()
    mean, std = np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32); threshold = float(state["threshold"])
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip() and json.loads(line).get("render_status") == "accepted"]
    by_split = defaultdict(Counter); by_profile = defaultdict(Counter)
    for row in sorted(rows, key=lambda item: item["sample_key"]):
        value = evaluate(row, candidates, model, mean, std, threshold); by_split[row["split"]].update(value); by_profile[row["split"] + ":" + row["profile_id"]].update(value)
    all_metrics = {split: finish(value) for split, value in sorted(by_split.items())}
    profile_metrics = {key: finish(value) for key, value in sorted(by_profile.items())}
    hold = all_metrics["holdout"]
    gate = {"checkpoint_unchanged": digest(audit.INITIAL) == audit.INITIAL_SHA, "sealed_test_read": False, "holdout_edge_precision_ge_0_90": hold["edge_precision"] >= .90, "holdout_edge_recall_ge_0_75": hold["edge_recall"] >= .75, "holdout_type_accuracy_ge_0_90": hold["bond_type_accuracy"] >= .90, "holdout_dangling_le_0.05": hold["dangling_rate"] <= .05, "holdout_oracle_sanitize_ge_0.90": hold["oracle_sanitize_rate"] >= .90, "production_go": False}
    gate["roundtrip_shadow_worth_attempting"] = all(value for key, value in gate.items() if key not in {"sealed_test_read", "production_go"})
    report = {"schema_version": "chemvector-identity-aligned-line-first-hungarian-roundtrip-r5-0.1", "complete": True, "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "proposal_contract": "LSD -> target-free score-distance Hungarian node/candidate ownership -> frozen edge/type sidecar; GT post-hoc only", "config": {"radius_px": RADIUS, "score_lambda_px": LAMBDA, "selection_source": "train-only CPU sweep"}, "by_split": all_metrics, "by_split_profile": profile_metrics, "gate": gate, "next": "no GPU until oracle sanitize gate passes; if failed, train upstream aromatic semantics rather than ownership"}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n"); print(json.dumps(report, indent=2, ensure_ascii=False)); return report


if __name__ == "__main__": main()
