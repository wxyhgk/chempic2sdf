"""CPU probe for many-to-one provisional-node ownership plus duplicate dedup."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry  # noqa: E402
from tools import audit_line_first_valence_topology_guard_r5 as guard  # noqa: E402
from tools import shadow_line_first_edge_bond_type_r5 as shadow  # noqa: E402
from chemvector.training.curriculum.checkpoints import file_digest  # noqa: E402

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
CANDIDATES = ROOT / "outputs/audits/chemvector-identity-aligned-vertex-head-shadow-20261017-r5-v5-scored/candidates.jsonl"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-many-to-one-roundtrip-r5/report.json"
RADIUS = 24.0
LAMBDA = 8.0
IMAGE = 768.0


def candidate_rows():
    rows = {}
    for line in CANDIDATES.read_text().splitlines():
        row = json.loads(line); rows[row["sample_key"]] = row["candidates"]
    if len(rows) != 1170:
        raise ValueError("fixed candidate artifact changed")
    return rows


def assign_many(centers: np.ndarray, candidates: np.ndarray) -> dict[int, int]:
    if not len(centers) or not len(candidates):
        return {}
    d = np.linalg.norm(centers[:, None, :] - candidates[None, :, :2], axis=-1)
    score = candidates[:, 2][None, :]
    cost = np.where(d <= RADIUS, d - LAMBDA * score, 1e5 + d)
    return {int(i): int(np.argmin(cost[i])) for i in range(len(centers))
            if float(d[i, int(np.argmin(cost[i]))]) <= RADIUS}


def raw(row, candidates, model, mean, std, threshold):
    key = row["sample_key"]
    features, centers, _ = chemistry._proposals(key)
    values = candidates[key]
    points = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE, v.get("score", 0.0)] for v in values], np.float32)
    owner = assign_many(centers.astype(np.float64), points.astype(np.float64))
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
    cycle = {p: (s, 1 if t == guard.TYPE_AROMATIC and p not in ring else t) for p, (s, t) in selected.items()}
    return cycle, points, dangling


def finish(c: Counter) -> dict:
    return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1),
            "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1),
            "bond_type_accuracy": c["type_correct"] / max(c["type_total"], 1),
            "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1),
            "oracle_sanitize_rate": c["sanitize_ok"] / max(c["images"], 1)}


def main() -> dict:
    rdBase.DisableLog("rdApp.error")
    if file_digest(chemistry.INITIAL) != chemistry.INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    candidates = candidate_rows()
    state = torch.load(chemistry.STATE, map_location="cpu", weights_only=False)
    model = chemistry.EdgeBondMLP(len(state["feature_names"]))
    model.load_state_dict(state["model"]); model.eval()
    mean, std, threshold = np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32), float(state["threshold"])
    rows = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines() if line.strip()]
    results = {}
    for mode in ("cycle", "guard"):
        by_split, by_profile, edits = defaultdict(Counter), defaultdict(Counter), Counter()
        for row in sorted(rows, key=lambda item: item["sample_key"]):
            selected, points, dangling = raw(row, candidates, model, mean, std, threshold)
            if mode == "guard":
                selected, event = guard.guard(selected, low_confidence=.4); edits.update(event)
            metrics = guard._metrics(row, selected, points, dangling)
            by_split[row["split"]].update(metrics); by_profile[row["split"] + ":" + row["profile_id"]].update(metrics)
        results[mode] = {"by_split": {k: finish(v) for k, v in sorted(by_split.items())},
                         "by_profile": {k: finish(v) for k, v in sorted(by_profile.items())}, "edits": dict(edits)}
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-many-to-one-roundtrip-r5-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "proposal_contract": "LSD -> independent score-distance many-to-one ownership -> edge dedup -> cycle/valence guard; targets post-hoc",
        "config": {"radius_px": RADIUS, "score_lambda_px": LAMBDA, "guard_tau": .4},
        "results": results, "initial_sha256": chemistry.INITIAL_SHA,
        "initial_checkpoint_sha256_after": file_digest(chemistry.INITIAL),
        "gate": {"checkpoint_unchanged": file_digest(chemistry.INITIAL) == chemistry.INITIAL_SHA,
                 "sealed_test_read": False, "production_go": False,
                 "gpu_training_admissible": False},
        "next": "many-to-one is diagnostic only; require strict full-holdout precision/recall/sanitize before any shadow",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
