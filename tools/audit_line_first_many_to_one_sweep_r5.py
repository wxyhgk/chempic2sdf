#!/usr/bin/env python3
"""Train-only CPU sweep for the promising many-to-one ownership route."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import numpy as np
import torch
from rdkit import rdBase
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import audit_line_first_many_to_one_roundtrip_r5 as route
from tools import audit_line_first_valence_topology_guard_r5 as guard
from chemvector.training.curriculum.checkpoints import file_digest
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-many-to-one-sweep-r5/report.json"
DATA = route.DATA

def finish(c):
    return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1),
            "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1),
            "bond_type_accuracy": c["type_correct"] / max(c["type_total"], 1),
            "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1),
            "oracle_sanitize_rate": c["sanitize_ok"] / max(c["images"], 1)}

def main():
    rdBase.DisableLog("rdApp.error")
    if file_digest(route.chemistry.INITIAL) != route.chemistry.INITIAL_SHA: raise RuntimeError("checkpoint changed")
    candidates = route.candidate_rows(); state = torch.load(route.chemistry.STATE, map_location="cpu", weights_only=False)
    model = route.chemistry.EdgeBondMLP(len(state["feature_names"])); model.load_state_dict(state["model"]); model.eval()
    mean, std, threshold = np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32), float(state["threshold"])
    rows = sorted([json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x.strip()], key=lambda x:x["sample_key"])
    sweeps = {}
    for radius in (16., 20., 24., 28.):
        for lam in (0., 4., 8.):
            route.RADIUS, route.LAMBDA = radius, lam
            by_split, by_profile = defaultdict(Counter), defaultdict(Counter)
            for row in rows:
                selected, points, dangling = route.raw(row, candidates, model, mean, std, threshold)
                selected, _ = guard.guard(selected, low_confidence=.4)
                value = guard._metrics(row, selected, points, dangling)
                by_split[row["split"]].update(value); by_profile[f"{row['split']}:{row['profile_id']}"].update(value)
            sweeps[f"r{radius:g}_l{lam:g}"] = {"by_split": {k:finish(v) for k,v in sorted(by_split.items())}, "by_profile": {k:finish(v) for k,v in sorted(by_profile.items())}}
    baseline = json.loads((ROOT/"outputs/audits/chemvector-identity-aligned-line-first-valence-topology-guard-r5/report.json").read_text())["sweeps"]["0.4"]["by_split"]["train"]
    eligible = {k:v for k,v in sweeps.items() if v["by_split"]["train"]["edge_precision"] >= baseline["edge_precision"]}
    selected = max(eligible, key=lambda k:(eligible[k]["by_split"]["train"]["edge_recall"], eligible[k]["by_split"]["train"]["edge_precision"])) if eligible else max(sweeps, key=lambda k:sweeps[k]["by_split"]["train"]["edge_recall"])
    hold, hp = sweeps[selected]["by_split"]["holdout"], {k:v for k,v in sweeps[selected]["by_profile"].items() if k.startswith("holdout:")}
    gate={"checkpoint_unchanged":file_digest(route.chemistry.INITIAL)==route.chemistry.INITIAL_SHA,"sealed_test_read":False,"production_go":False,"selected_precision_ge_0.96592":hold["edge_precision"]>=.96592,"selected_recall_ge_0.85034":hold["edge_recall"]>=.85034,"selected_sanitize_ge_0.945":hold["oracle_sanitize_rate"]>=.945,"all_profile_recall_ge_0.75":all(v["edge_recall"]>=.75 for v in hp.values()),"all_profile_sanitize_ge_0.90":all(v["oracle_sanitize_rate"]>=.90 for v in hp.values())}
    gate["cpu_route_pass"] = all(v for k,v in gate.items() if k not in {"sealed_test_read","production_go"})
    report={"schema_version":"chemvector-identity-aligned-line-first-many-to-one-sweep-r5-0.1","complete":True,"read_only":True,"training_performed":False,"gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"selected":selected,"sweeps":sweeps,"holdout":hold,"holdout_profiles":hp,"gate":gate}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps({"output":str(OUT),"selected":selected,"holdout":hold,"gate":gate},indent=2))
if __name__ == "__main__": main()
