#!/usr/bin/env python3
"""Diagnostic (post-hoc only) threshold sweep for the r9 sidecar.

Thresholds are *not* selected from this report; holdout labels are used only
to display the precision/recall tradeoff.  This is quarantined evidence for
designing a later train-only calibration protocol.
"""
from __future__ import annotations
from collections import Counter
import hashlib, json, os, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import rows, DirectPairResidual, SHA
from tools.shadow_direct_graph_pair_scorer_gpu_r9 import _profile_rows, STATE
from tools.audit_direct_graph_pair_scorer_oracle_sanitize_r9 import extract, DATA

OUT = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-threshold-sweep-r9"
THRESHOLDS = [round(x / 100, 2) for x in range(70, 100)]

def digest(path):
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def metrics(items, scorer, threshold):
    c = Counter()
    for item in items:
        with torch.inference_mode():
            logits = item["base_logits"] + scorer(item["features"])
        chosen = torch.where(logits.sigmoid() >= threshold)[0]
        pairs, slots, mapping = item["pairs"], item["slots"], item["mapping"]
        pred, dangling = set(), 0
        for idx in chosen.tolist():
            a, b = [int(v) for v in pairs[idx].tolist()]
            ai = int((slots == a).nonzero(as_tuple=False)[0]); bi = int((slots == b).nonzero(as_tuple=False)[0])
            if ai not in mapping or bi not in mapping: dangling += 1; continue
            edge = tuple(sorted((mapping[ai], mapping[bi])))
            if edge[0] != edge[1]: pred.add(edge)
        gt = {tuple(sorted((int(x["begin_atom"]), int(x["end_atom"])))) for x in item["graph"]["bonds"]}
        c.update(images=1, predicted_edges=len(chosen), mapped_edges=len(pred), true_edges=len(pred & gt), target_edges=len(gt), dangling_edges=dangling)
    d = dict(c); d.update(edge_precision=c["true_edges"] / max(c["mapped_edges"], 1), edge_recall=c["true_edges"] / max(c["target_edges"], 1), dangling_rate=c["dangling_edges"] / max(c["predicted_edges"], 1)); return d

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    if digest(INITIAL) != SHA: raise RuntimeError("production checkpoint changed")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required for bounded diagnostic")
    train, hold = _profile_rows(rows())
    model, config, _ = load_bundle_for_probe(INITIAL, SHA); model.to("cuda:0").eval().requires_grad_(False)
    payload = torch.load(STATE, map_location="cpu", weights_only=False); scorer = DirectPairResidual(262); scorer.load_state_dict(payload["state_dict"]); scorer.eval()
    items = [extract(row, model, config, torch.device("cuda:0")) for row in hold]
    sweep = []
    for t in THRESHOLDS:
        allm = metrics(items, scorer, t); by = {}
        for profile in sorted({r["profile_id"] for r in hold}):
            ids = [i for i, r in enumerate(hold) if r["profile_id"] == profile]; by[profile] = metrics([items[i] for i in ids], scorer, t)
        sweep.append({"threshold": t, "holdout": allm, "profiles": by, "diagnostic_holdout_labels_used": True})
    after = digest(INITIAL); report = {"schema_version":"chemvector-identity-aligned-direct-graph-pair-scorer-threshold-sweep-r9-0.1", "complete":True, "read_only":True, "training_performed":False, "gpu_used":True, "sealed_test_read":False, "production_checkpoint_written":False, "selection":"none; post-hoc holdout diagnostic only", "thresholds":THRESHOLDS, "results":sweep, "initial_checkpoint_sha256":SHA, "initial_checkpoint_sha256_after":after, "sidecar_state_sha256":digest(STATE), "gate":{"checkpoint_unchanged":after==SHA,"sealed_test_read":False,"production_go":False}}
    OUT.mkdir(parents=True); (OUT/"report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n"); print(json.dumps({"output":str(OUT/"report.json"),"best_by_precision":[x for x in sweep if x["holdout"]["edge_precision"]>=.99 and x["holdout"]["edge_recall"]>.94]}, indent=2))

if __name__ == "__main__": main()
