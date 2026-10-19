"""Read-only attribution audit for the frozen node + edge line-first route.

This audit never trains and never uses target labels to form predictions.  GT
atoms/bonds are joined only after the target-free node/edge proposal is built.
It reports where bonds are lost (candidate reachability, node keep, or edge
score) and compares target-free keep-all versus drop-isolated node emission.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import shadow_line_first_node_graph_r5 as shadow
from tools.audit_line_first_edge_bond_candidate_integration_r5 import strict_candidate_to_target

OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-node-edge-failure-attribution-r5"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
IMAGE = 768.0


def _target(key):
    base = shadow.DATA / "samples" / key
    layers = json.loads(base.with_suffix(".layers.json").read_text())
    graph = json.loads(base.with_suffix(".graph.json").read_text())
    pts = np.asarray([a["center"] for a in layers["atoms"]], np.float32)
    bonds = {tuple(sorted((int(b["begin_atom"]), int(b["end_atom"])))) for b in graph["bonds"]}
    return pts, bonds


def _summarize(rows):
    keys = ("images", "gt_bonds", "gt_recovered", "gt_node_drop", "gt_edge_miss", "gt_unreachable",
            "selected", "selected_mapped", "selected_tp", "selected_fp", "selected_dangling",
            "selected_self_loop", "keepall_edges", "keepall_mapped", "keepall_tp", "dropiso_edges",
            "dropiso_mapped", "dropiso_tp")
    total = Counter()
    for row in rows: total.update({k: int(row.get(k, 0)) for k in keys})
    def finish(c):
        d = dict(c)
        d.update({
            "gt_recall": c["gt_recovered"] / max(c["gt_bonds"], 1),
            "gt_node_drop_rate": c["gt_node_drop"] / max(c["gt_bonds"], 1),
            "gt_edge_miss_rate": c["gt_edge_miss"] / max(c["gt_bonds"], 1),
            "gt_unreachable_rate": c["gt_unreachable"] / max(c["gt_bonds"], 1),
            "selected_precision": c["selected_tp"] / max(c["selected_mapped"], 1),
            "selected_dangling_rate": c["selected_dangling"] / max(c["selected"], 1),
            "keepall_precision": c["keepall_tp"] / max(c["keepall_mapped"], 1),
            "keepall_recall": c["keepall_tp"] / max(c["gt_bonds"], 1),
            "dropiso_precision": c["dropiso_tp"] / max(c["dropiso_mapped"], 1),
            "dropiso_recall": c["dropiso_tp"] / max(c["gt_bonds"], 1),
        })
        return d
    return finish(total)


def _evaluate(row, candidates, node_model, node_mean, node_std, node_threshold, edge_model, edge_mean, edge_std, edge_threshold):
    key = row["sample_key"]
    vals = candidates[key]
    nf, centers, cis, nis = shadow._node_rows(key, vals)
    with torch.inference_mode():
        ns = torch.sigmoid(node_model(torch.from_numpy((nf - node_mean) / node_std))).numpy() if len(nf) else np.empty(0, np.float32)
    keep = ns >= node_threshold
    node_map = {int(n): int(ci) for n, ci, k in zip(nis.tolist(), cis.tolist(), keep.tolist()) if k}
    all_node_map = {int(n): int(ci) for n, ci in zip(nis.tolist(), cis.tolist())}
    points = np.asarray([[v["x"] * IMAGE, v["y"] * IMAGE] for v in vals], np.float32)
    targets, gt = _target(key); c2t = strict_candidate_to_target(points, targets)
    ef, _, _ = shadow._proposals(key); keys = shadow._candidate_keys(key)
    with torch.inference_mode():
        elogits, _ = edge_model(torch.from_numpy((ef - edge_mean) / edge_std))
    es = torch.sigmoid(elogits).numpy()

    # Every raw candidate pair is mapped post-hoc for attribution only.
    raw_pairs = defaultdict(list)
    for score, (a, b) in zip(es.tolist(), keys):
        if a in all_node_map and b in all_node_map:
            ca, cb = all_node_map[a], all_node_map[b]
            if ca != cb and ca in c2t and cb in c2t and c2t[ca] != c2t[cb]:
                raw_pairs[tuple(sorted((c2t[ca], c2t[cb])))].append((float(score), a, b))
    selected = []
    for score, (a, b) in zip(es.tolist(), keys):
        if score < edge_threshold: continue
        if a not in node_map or b not in node_map:
            selected.append(("dangling", float(score), a, b, None)); continue
        ca, cb = node_map[a], node_map[b]
        if ca == cb:
            selected.append(("self_loop", float(score), a, b, None)); continue
        edge = tuple(sorted((c2t[ca], c2t[cb]))) if ca in c2t and cb in c2t and c2t[ca] != c2t[cb] else None
        selected.append(("mapped", float(score), a, b, edge))
    selected_mapped = {x[4] for x in selected if x[0] == "mapped" and x[4] is not None}
    selected_tp = selected_mapped & gt
    selected_fp = selected_mapped - gt

    # GT bond attribution: candidate absent, node keep loss, or edge score loss.
    gt_recovered = gt_node_drop = gt_edge_miss = gt_unreachable = 0
    for edge in gt:
        candidates_for = raw_pairs.get(edge, [])
        if not candidates_for:
            gt_unreachable += 1; continue
        kept_pair = False
        for _score, a, b in candidates_for:
            if a in node_map and b in node_map: kept_pair = True; break
        if not kept_pair:
            gt_node_drop += 1; continue
        if edge in selected_mapped: gt_recovered += 1
        else: gt_edge_miss += 1

    # Target-free emission A/B: all candidate nodes versus dropping isolated
    # nodes (nodes with no selected edge after node/edge thresholding).
    keepall_pairs = {x[4] for x in selected if x[0] == "mapped" and x[4] is not None}
    degree = Counter()
    for _kind, _s, a, b, _e in selected:
        if _kind == "mapped": degree[a] += 1; degree[b] += 1
    dropiso_pairs = {x[4] for x in selected if x[0] == "mapped" and x[4] is not None and degree[x[2]] > 0 and degree[x[3]] > 0}
    return {
        "profile_id": row["profile_id"], "images": 1, "gt_bonds": len(gt), "gt_recovered": len(selected_tp),
        "gt_node_drop": gt_node_drop, "gt_edge_miss": gt_edge_miss, "gt_unreachable": gt_unreachable,
        "selected": len(selected), "selected_mapped": len(selected_mapped), "selected_tp": len(selected_tp),
        "selected_fp": len(selected_fp), "selected_dangling": sum(x[0] == "dangling" for x in selected),
        "selected_self_loop": sum(x[0] == "self_loop" for x in selected),
        "keepall_edges": len(keepall_pairs), "keepall_mapped": len(keepall_pairs), "keepall_tp": len(keepall_pairs & gt),
        "dropiso_edges": len(dropiso_pairs), "dropiso_mapped": len(dropiso_pairs), "dropiso_tp": len(dropiso_pairs & gt),
    }


def main():
    if shadow.file_digest(shadow.INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    candidates = shadow._load_candidates()
    npayload = torch.load(shadow.STATE, map_location="cpu", weights_only=False)
    node_model = shadow.NodeMLP(len(npayload["feature_names"])); node_model.load_state_dict(npayload["model"]); node_model.eval()
    node_mean, node_std, node_threshold = np.asarray(npayload["mean"], np.float32), np.asarray(npayload["std"], np.float32), float(npayload["threshold"])
    epayload = torch.load(shadow.EDGE_STATE, map_location="cpu", weights_only=False)
    edge_model = shadow.EdgeBondMLP(len(epayload["feature_names"])); edge_model.load_state_dict(epayload["model"]); edge_model.eval()
    edge_mean, edge_std, edge_threshold = np.asarray(epayload["mean"], np.float32), np.asarray(epayload["std"], np.float32), float(epayload["threshold"])
    rows = [json.loads(x) for x in (shadow.DATA / "samples.jsonl").read_text().splitlines() if x.strip() and json.loads(x).get("render_status") == "accepted"]
    hold = sorted([r for r in rows if r["split"] == "holdout"], key=lambda r: r["sample_key"])
    results = [_evaluate(r, candidates, node_model, node_mean, node_std, node_threshold, edge_model, edge_mean, edge_std, edge_threshold) for r in hold]
    profiles = {p: _summarize([x for x in results if x["profile_id"] == p]) for p in sorted({x["profile_id"] for x in results})}
    allm = _summarize(results)
    gate = {
        "checkpoint_unchanged": shadow.file_digest(shadow.INITIAL) == INITIAL_SHA,
        "sealed_test_read": False,
        "holdout_images_474": len(hold) == 474,
        "keepall_recall_gt_0.85": allm["keepall_recall"] > .85,
        "keepall_precision_ge_0.965917": allm["keepall_precision"] >= .965917,
        "dropiso_dangling_not_applicable": True,
        "production_go": False,
        "gpu_go": False,
    }
    report = {"schema_version": "chemvector-identity-aligned-line-first-node-edge-failure-attribution-r5-0.1", "complete": True, "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "initial_sha256": INITIAL_SHA, "proposal_contract": "frozen node-keep MLP + frozen edge/type sidecar; target-free A/B keep-all vs drop-isolated; GT post-hoc attribution", "thresholds": {"node": node_threshold, "edge": edge_threshold}, "all": allm, "by_profile": profiles, "gate": gate}
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n"); print(json.dumps({"output": str(OUT / "report.json"), "all": allm, "gate": gate}, ensure_ascii=False))


if __name__ == "__main__": main()
