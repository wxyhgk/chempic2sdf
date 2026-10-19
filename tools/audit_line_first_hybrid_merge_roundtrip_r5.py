"""CPU probe: Hungarian ownership with a bounded duplicate-node merge."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry  # noqa: E402
from tools import audit_line_first_valence_topology_guard_r5 as guard  # noqa: E402
from tools import shadow_line_first_edge_bond_type_r5 as shadow  # noqa: E402
from chemvector.training.curriculum.checkpoints import file_digest  # noqa: E402

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
CANDIDATES = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5-scored/candidates.jsonl"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-hybrid-merge-roundtrip-r5/report.json"
RADIUS, LAMBDA, IMAGE = 24.0, 8.0, 768.0


def candidate_rows():
    out = {}
    for line in CANDIDATES.read_text().splitlines():
        row = json.loads(line); out[row["sample_key"]] = row["candidates"]
    if len(out) != 1170: raise ValueError("candidate artifact changed")
    return out


def assign(centers, points, merge_px: float):
    if not len(centers) or not len(points): return {}, {}
    d = np.linalg.norm(centers[:, None] - points[None, :, :2], axis=-1)
    cost = np.where(d <= RADIUS, d - LAMBDA * points[None, :, 2], 1e5 + d)
    rr, cc = linear_sum_assignment(cost)
    one = {int(r): int(c) for r, c in zip(rr, cc) if d[r, c] <= RADIUS}
    nearest = {int(r): int(np.argmin(cost[r])) for r in range(len(centers))
               if float(d[r, int(np.argmin(cost[r]))]) <= RADIUS}
    if merge_px <= 0: return one, {int(r): float(d[r, c]) for r, c in one.items()}
    groups = defaultdict(list)
    for node, cand in nearest.items(): groups[cand].append(node)
    merged = dict(one)
    for cand, nodes in groups.items():
        if len(nodes) > 1 and max(float(d[node, cand]) for node in nodes) <= merge_px:
            for node in nodes: merged[node] = cand
    return merged, {int(r): float(d[r, c]) for r, c in merged.items()}


def raw(row, candidates, model, mean, std, threshold, merge_px):
    key = row["sample_key"]
    features, centers, _ = chemistry._proposals(key)
    values = candidates[key]
    points = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE, v.get("score", 0.0)] for v in values], np.float32)
    owner, _ = assign(centers.astype(np.float64), points.astype(np.float64), merge_px)
    with torch.inference_mode():
        edge_logits, type_logits = model(torch.from_numpy((features - mean) / std))
    scores = torch.sigmoid(edge_logits).numpy(); kinds = (torch.argmax(type_logits, dim=1) + 1).numpy()
    selected = {}; dangling = 0
    for index, (a, b) in enumerate(chemistry._candidate_keys(key)):
        if scores[index] < threshold: continue
        if a not in owner or b not in owner: dangling += 1; continue
        ca, cb = owner[a], owner[b]
        if ca == cb: continue
        pair = tuple(sorted((ca, cb))); old = selected.get(pair)
        if old is None or scores[index] > old[0]: selected[pair] = (float(scores[index]), int(kinds[index]))
    ring = guard._cycles(selected)
    selected = {p: (s, 1 if t == guard.TYPE_AROMATIC and p not in ring else t) for p, (s, t) in selected.items()}
    return selected, points, dangling


def finish(c: Counter) -> dict:
    return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1),
            "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1),
            "bond_type_accuracy": c["type_correct"] / max(c["type_total"], 1),
            "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1),
            "oracle_sanitize_rate": c["sanitize_ok"] / max(c["images"], 1)}


def main() -> dict:
    rdBase.DisableLog("rdApp.error")
    if file_digest(chemistry.INITIAL) != chemistry.INITIAL_SHA: raise RuntimeError("checkpoint changed")
    candidates = candidate_rows(); state = torch.load(chemistry.STATE, map_location="cpu", weights_only=False)
    model = chemistry.EdgeBondMLP(len(state["feature_names"])); model.load_state_dict(state["model"]); model.eval()
    mean, std, threshold = np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32), float(state["threshold"])
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    # Train-only selection: retain the largest recall while preserving high precision.
    sweep = {}
    for merge_px in (0.0, 12.0, 16.0, 20.0, 24.0):
        by_split, by_profile = defaultdict(Counter), defaultdict(Counter)
        for row in sorted(rows, key=lambda item: item["sample_key"]):
            selected, points, dangling = raw(row, candidates, model, mean, std, threshold, merge_px)
            selected, _ = guard.guard(selected, low_confidence=.4)
            value = guard._metrics(row, selected, points, dangling)
            by_split[row["split"]].update(value); by_profile[row["split"] + ":" + row["profile_id"]].update(value)
        sweep[str(merge_px)] = {"by_split": {k: finish(v) for k, v in sorted(by_split.items())},
                                "by_profile": {k: finish(v) for k, v in sorted(by_profile.items())}}
    eligible = {k: v for k, v in sweep.items() if v["by_split"]["train"]["edge_precision"] >= .98}
    selected = max(eligible, key=lambda k: (eligible[k]["by_split"]["train"]["edge_recall"], -float(k))) if eligible else "0.0"
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-hybrid-merge-roundtrip-r5-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "proposal_contract": "LSD -> Hungarian ownership; bounded target-free duplicate merge -> cycle demotion -> valence guard",
        "config": {"radius_px": RADIUS, "score_lambda_px": LAMBDA, "merge_candidates_px": [0, 12, 16, 20, 24], "guard_tau": .4},
        "train_selected_merge_px": float(selected), "sweep": sweep,
        "initial_sha256": chemistry.INITIAL_SHA, "initial_checkpoint_sha256_after": file_digest(chemistry.INITIAL),
        "gate": {"checkpoint_unchanged": file_digest(chemistry.INITIAL) == chemistry.INITIAL_SHA,
                 "sealed_test_read": False, "gpu_training_admissible": False, "production_go": False},
        "next": "use selected merge only if strict full-holdout gate passes; then isolate one bounded shadow",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "selected": selected, "holdout": sweep[selected]["by_split"]["holdout"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__": main()
