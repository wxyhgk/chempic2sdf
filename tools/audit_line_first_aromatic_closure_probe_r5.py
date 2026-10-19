"""Target-free CPU probe for conservative aromatic ring closure.

This is an evaluation-only heuristic.  It may add a candidate edge only when
two degree-one candidate nodes have compatible geometry and aromatic incident
edge evidence.  Ground truth is consulted strictly after proposal formation.
No checkpoint or production graph is modified.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as audit  # noqa: E402
from tools.audit_line_first_candidate_alignment_r5 import align_nodes  # noqa: E402
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target  # noqa: E402

SOURCE = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
TARGET = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-train-r5"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-aromatic-closure-probe-r5/report.json"
IMAGE = 768.0


def proposal(row, candidates, model, mean, std, threshold):
    key = row["sample_key"]
    features, centers, _ = audit._proposals(key)
    values = candidates[key]
    points = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE] for v in values], dtype=np.float32)
    node_map = align_nodes(centers.astype(np.float64), points.astype(np.float64), audit.ALIGN_RADIUS)
    with torch.inference_mode():
        edge_logits, type_logits = model(torch.from_numpy((features - mean) / std))
    scores = torch.sigmoid(edge_logits).cpu().numpy()
    kinds = (torch.argmax(type_logits, dim=1) + 1).cpu().numpy()
    selected = {}
    line_keys = audit._candidate_keys(key)
    for index, (a, b) in enumerate(line_keys):
        if scores[index] < threshold or a not in node_map or b not in node_map:
            continue
        ca, cb = node_map[a], node_map[b]
        if ca == cb:
            continue
        pair = tuple(sorted((ca, cb)))
        old = selected.get(pair)
        if old is None or scores[index] > old[0]:
            selected[pair] = (float(scores[index]), int(kinds[index]))
    # Conservative closure: only degree-one endpoints of already aromatic
    # strokes, with regular-bond-scale distance and a non-collinear turn.
    adjacency = defaultdict(list)
    for (a, b), (_, kind) in selected.items():
        adjacency[a].append((b, kind)); adjacency[b].append((a, kind))
    lengths = [float(np.linalg.norm(points[a] - points[b])) for a, b in selected]
    median = float(np.median(lengths)) if lengths else 0.0
    additions = []
    if median > 0:
        leaves = [node for node, links in adjacency.items() if len(links) == 1]
        for i, a in enumerate(leaves):
            for b in leaves[i + 1:]:
                pair = tuple(sorted((a, b)))
                if pair in selected or a not in adjacency or b not in adjacency:
                    continue
                if not any(kind == 4 for _, kind in adjacency[a]) or not any(kind == 4 for _, kind in adjacency[b]):
                    continue
                delta = points[b] - points[a]
                distance = float(np.linalg.norm(delta))
                if not (0.55 * median <= distance <= 1.45 * median):
                    continue
                def turn(node, other):
                    neighbour = adjacency[node][0][0]
                    first = points[neighbour] - points[node]
                    second = points[other] - points[node]
                    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
                    return float(np.degrees(np.arccos(np.clip(np.dot(first, second) / max(denom, 1e-8), -1.0, 1.0))))
                if 45.0 <= turn(a, b) <= 165.0 and 45.0 <= turn(b, a) <= 165.0:
                    additions.append((pair, (0.0, 4)))
    for pair, value in additions:
        selected[pair] = value
    return selected, points, node_map, len(additions)


def evaluate(row, candidates, model, mean, std, threshold):
    selected, points, node_map, additions = proposal(row, candidates, model, mean, std, threshold)
    key = row["sample_key"]
    base = TARGET / "samples" / key if (TARGET / "samples" / (key + ".layers.json")).exists() else SOURCE / "samples" / key
    layers = json.loads(base.with_suffix(".layers.json").read_text())
    graph = json.loads(base.with_suffix(".graph.json").read_text())
    targets = np.asarray([x["center"] for x in layers["atoms"]], dtype=np.float32)
    c2t = strict_candidate_to_target(points, targets)
    gt = {tuple(sorted((int(x["begin_atom"]), int(x["end_atom"])))) for x in graph["bonds"]}
    gt_types = {tuple(sorted((int(x["begin_atom"]), int(x["end_atom"])))): audit.TYPE_NAMES.index(str(x.get("chemical_type", "OTHER")).upper()) if str(x.get("chemical_type", "OTHER")).upper() in audit.TYPE_NAMES else audit.TYPE_NAMES.index("OTHER") for x in graph["bonds"]}
    mapped = {}
    for (a, b), (_, kind) in selected.items():
        if a in c2t and b in c2t and c2t[a] != c2t[b]:
            mapped[tuple(sorted((c2t[a], c2t[b])))] = kind
    true = set(mapped) & gt
    type_correct = sum(mapped[e] == gt_types[e] for e in true)
    nodes = [(candidate, str(layers["atoms"][target]["element"] or "C"), bool(layers["atoms"][target].get("aromatic", False))) for candidate, target in sorted(c2t.items()) if candidate in {x for pair in selected for x in pair}]
    index = {candidate: i for i, (candidate, _, _) in enumerate(nodes)}
    edges = {(index[a], index[b]): kind for (a, b), (_, kind) in selected.items() if a in index and b in index}
    ok, error, _ = audit._sanitize_graph(nodes, edges)
    return {"images": 1, "selected_edges": len(selected), "mapped_edges": len(mapped), "true_edges": len(true), "gt_bonds": len(gt), "type_correct": type_correct, "type_total": len(true), "oracle_sanitize_ok": int(ok), "aromatic_error": int(error == "aromatic"), "valence_error": int(error == "valence"), "additions": additions}


def summarize(values):
    c = Counter()
    c.update({key: sum(value.get(key, 0) for value in values) for key in ("images", "selected_edges", "mapped_edges", "true_edges", "gt_bonds", "type_correct", "type_total", "oracle_sanitize_ok", "aromatic_error", "valence_error", "additions")})
    return {"images": c["images"], "added_edges": c["additions"], "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1), "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1), "bond_type_accuracy": c["type_correct"] / max(c["type_total"], 1), "oracle_sanitize_rate": c["oracle_sanitize_ok"] / max(c["images"], 1), "aromatic_error_rate": c["aromatic_error"] / max(c["images"], 1), "valence_error_rate": c["valence_error"] / max(c["images"], 1)}


def main():
    rdBase.DisableLog("rdApp.error")
    audit.DATA = TARGET
    candidates = audit._candidate_rows()
    state = torch.load(audit.STATE, map_location="cpu", weights_only=False)
    model = audit.EdgeBondMLP(len(state["feature_names"])); model.load_state_dict(state["model"]); model.eval()
    mean, std = np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32); threshold = float(state["threshold"])
    rows = [json.loads(line) for line in (TARGET / "samples.jsonl").read_text().splitlines() if line.strip()]
    targeted = [evaluate(row, candidates, model, mean, std, threshold) for row in rows]
    # The fixed holdout is evaluated with the same target-free rule by swapping
    # the data root; no sealed identities are involved.
    audit.DATA = SOURCE
    hold_rows = [json.loads(line) for line in (SOURCE / "samples.jsonl").read_text().splitlines() if line.strip() and json.loads(line).get("split") == "holdout"]
    hold = [evaluate(row, candidates, model, mean, std, threshold) for row in hold_rows]
    report = {"schema_version": "chemvector-identity-aligned-line-first-aromatic-closure-probe-r5-0.1", "complete": True, "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "proposal_contract": "target-free candidate geometry plus predicted aromatic incident edges; GT post-hoc only", "targeted_train": summarize(targeted), "fixed_holdout": summarize(hold), "probe": {"max_added_edge_rule": "degree-one aromatic endpoints, regular-bond distance, 45-165 degree turns", "deployment_authorized": False, "training_admissible": False}}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n"); print(json.dumps(report, indent=2, ensure_ascii=False)); return report


if __name__ == "__main__":
    main()
