#!/usr/bin/env python3
"""Independent post-hoc RDKit audit for the r10 CPU coverage sidecar."""
from __future__ import annotations
from collections import Counter
import hashlib, json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.train_direct_graph_pair_scorer_cpu_r9 import DATA, SHA, DirectPairResidual, _extract, rows
from tools.audit_direct_graph_pair_scorer_chemistry_r9 import one, finish

TRIAL = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-cpu-r10-coverage"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-direct-graph-pair-scorer-chemistry-r10-coverage"
STATE = TRIAL / "scorer-state.pt"
REPORT = TRIAL / "report.json"
THRESHOLD = 0.72

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def run():
    if digest(INITIAL) != SHA: raise RuntimeError("production checkpoint changed")
    state = torch.load(STATE, map_location="cpu", weights_only=False)
    scorer = DirectPairResidual(input_dim=262); scorer.load_state_dict(state["state_dict"]); scorer.eval()
    model, config, _ = load_bundle_for_probe(INITIAL, SHA); model.eval().requires_grad_(False)
    holdout_rows = sorted((r for r in rows() if r.get("split") == "fresh_holdout"), key=lambda r: r["sample_key"])
    items = [_extract(r, model, config, need_labels=True) for r in holdout_rows]
    modes = {"baseline": (None, 0.85), "r10": (scorer, THRESHOLD)}
    all_results, profile_results = {}, {}
    for name, (active, threshold) in modes.items():
        total = Counter(); by = {}
        for item in items: total.update(one(item, active, threshold))
        for profile in sorted({i["row"]["profile_id"] for i in items}):
            c = Counter()
            for item in items:
                if item["row"]["profile_id"] == profile: c.update(one(item, active, threshold))
            by[profile] = finish(c)
        all_results[name] = finish(total); profile_results[name] = by
    after = digest(INITIAL)
    result = {"schema_version": "chemvector-identity-aligned-direct-graph-pair-scorer-chemistry-r10-coverage-0.1", "complete": True, "read_only": True, "training_performed": False, "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "dataset": str(DATA), "split": "fresh_holdout", "images": len(items), "thresholds": {"baseline": .85, "r10": THRESHOLD}, "all": all_results, "by_profile": profile_results, "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": after, "trial_report_sha256": digest(REPORT), "trial_state_sha256": digest(STATE), "gate": {"checkpoint_unchanged": after == SHA, "sealed_test_read": False, "oracle_sanitize_reported": True, "production_go": False}}
    OUT.mkdir(parents=True, exist_ok=False); (OUT / "report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "all": result["all"], "gate": result["gate"]}, indent=2, ensure_ascii=False))

if __name__ == "__main__": run()
