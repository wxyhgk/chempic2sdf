"""Target-free aromatic cycle-consistency probe for Hungarian ownership."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import networkx as nx
import numpy as np
import torch
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_hungarian_roundtrip_r5 as base  # noqa: E402

OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-aromatic-cycle-demote-r5/report.json"


def _cycle_edges(selected: dict[tuple[int, int], tuple[float, int]]) -> set[tuple[int, int]]:
    graph = nx.Graph(); graph.add_edges_from(selected)
    valid = set()
    # Enumerate simple cycles through a bounded DFS; the graph is tiny per image.
    for start in graph.nodes:
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            for nxt in graph.neighbors(node):
                if nxt == start and 5 <= len(path) <= 6:
                    valid.update(tuple(sorted((path[i], path[(i + 1) % len(path)]))) for i in range(len(path)))
                elif nxt not in path and len(path) < 6 and nxt >= start:
                    stack.append((nxt, path + [nxt]))
    return valid


def evaluate(row, candidates, model, mean, std, threshold):
    key = row["sample_key"]
    features, centers, _ = base.audit._proposals(key)
    values = candidates[key]
    points = np.asarray([[v["x"] * base.IMAGE, v["y"] * base.IMAGE, v.get("score", 0.0)] for v in values], dtype=np.float32)
    node_map = base.assign(centers.astype(np.float64), points.astype(np.float64))
    with torch.inference_mode():
        edge_logits, type_logits = model(torch.from_numpy((features - mean) / std))
    scores = torch.sigmoid(edge_logits).cpu().numpy(); kinds = (torch.argmax(type_logits, dim=1) + 1).cpu().numpy()
    selected = {}; dangling = 0
    for index, (a, b) in enumerate(base.audit._candidate_keys(key)):
        if scores[index] < threshold: continue
        if a not in node_map or b not in node_map: dangling += 1; continue
        ca, cb = node_map[a], node_map[b]
        if ca == cb: continue
        pair = tuple(sorted((ca, cb))); old = selected.get(pair)
        if old is None or scores[index] > old[0]: selected[pair] = (float(scores[index]), int(kinds[index]))
    cycle_edges = _cycle_edges(selected)
    demoted = 0
    repaired = {}
    for pair, value in selected.items():
        score, kind = value
        if kind == 4 and pair not in cycle_edges:
            repaired[pair] = (score, 1); demoted += 1
        else:
            repaired[pair] = value
    base_path = base.DATA / "samples" / key
    layers = json.loads(base_path.with_suffix(".layers.json").read_text()); graph = json.loads(base_path.with_suffix(".graph.json").read_text())
    targets = np.asarray([x["center"] for x in layers["atoms"]], dtype=np.float32)
    c2t = base.audit.strict_candidate_to_target(points[:, :2], targets)
    gt = {tuple(sorted((int(x["begin_atom"]), int(x["end_atom"])))) for x in graph["bonds"]}
    type_by = {tuple(sorted((int(x["begin_atom"]), int(x["end_atom"])))): (base.audit.TYPE_NAMES.index(str(x.get("chemical_type", "OTHER")).upper()) if str(x.get("chemical_type", "OTHER")).upper() in base.audit.TYPE_NAMES else base.audit.TYPE_NAMES.index("OTHER")) for x in graph["bonds"]}
    mapped = {}; type_correct = 0
    for (a, b), (_, kind) in repaired.items():
        if a in c2t and b in c2t and c2t[a] != c2t[b]: mapped[tuple(sorted((c2t[a], c2t[b])))] = kind
    true = set(mapped) & gt; type_correct = sum(mapped[e] == type_by[e] for e in true)
    node_ids = sorted({x for pair in repaired for x in pair})
    oracle_nodes = [(candidate, str(layers["atoms"][target]["element"] or "C"), bool(layers["atoms"][target].get("aromatic", False))) for candidate, target in sorted(c2t.items()) if candidate in node_ids]
    idx = {candidate: i for i, (candidate, _, _) in enumerate(oracle_nodes)}
    edges = {(idx[a], idx[b]): kind for (a, b), (_, kind) in repaired.items() if a in idx and b in idx}
    ok, error, _ = base.audit._sanitize_graph(oracle_nodes, edges)
    return Counter({"images": 1, "selected_edges": len(selected), "demoted_aromatic_edges": demoted, "dangling_edges": dangling, "mapped_edges": len(mapped), "true_edges": len(true), "gt_bonds": len(gt), "type_correct": type_correct, "type_total": len(true), "oracle_sanitize_ok": int(ok), "aromatic_errors": int(error == "aromatic"), "valence_errors": int(error == "valence"), "all_bonds_recovered": int(true == gt)})


def finish(c: Counter) -> dict:
    return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1), "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1), "bond_type_accuracy": c["type_correct"] / max(c["type_total"], 1), "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1), "oracle_sanitize_rate": c["oracle_sanitize_ok"] / max(c["images"], 1), "all_bonds_recovered_fraction": c["all_bonds_recovered"] / max(c["images"], 1)}


def main():
    rdBase.DisableLog("rdApp.error")
    candidates = base.audit._candidate_rows()
    state = torch.load(base.audit.STATE, map_location="cpu", weights_only=False)
    model = base.audit.EdgeBondMLP(len(state["feature_names"])); model.load_state_dict(state["model"]); model.eval()
    mean, std = np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32); threshold = float(state["threshold"])
    rows = [json.loads(line) for line in (base.DATA / "samples.jsonl").read_text().splitlines() if line.strip() and json.loads(line).get("render_status") == "accepted"]
    by = defaultdict(Counter)
    for row in sorted(rows, key=lambda item: item["sample_key"]): by[row["split"]].update(evaluate(row, candidates, model, mean, std, threshold))
    hold = finish(by["holdout"])
    report = {"schema_version": "chemvector-identity-aligned-line-first-aromatic-cycle-demote-r5-0.1", "complete": True, "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "proposal_contract": "target-free Hungarian ownership + frozen edge/type sidecar + cycle-only aromatic demotion; GT post-hoc", "config": {"ownership_radius_px": base.RADIUS, "ownership_score_lambda": base.LAMBDA, "aromatic_requires_cycle_size": [5, 6]}, "by_split": {key: finish(value) for key, value in sorted(by.items())}, "gate": {"holdout_edge_precision_ge_0_90": hold["edge_precision"] >= .90, "holdout_edge_recall_ge_0_75": hold["edge_recall"] >= .75, "holdout_oracle_sanitize_ge_0_90": hold["oracle_sanitize_rate"] >= .90, "production_go": False, "training_admissible": False}}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n"); print(json.dumps(report, indent=2, ensure_ascii=False)); return report


if __name__ == "__main__": main()
