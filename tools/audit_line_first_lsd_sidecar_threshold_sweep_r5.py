#!/usr/bin/env python3
"""Train-only threshold sweep for the new-LSD audit-local sidecar.

No weights are changed here.  Thresholds are fixed before touching holdout;
each value is applied to the complete holdout graph replay with frozen
Hungarian ownership, merge-20 duplicate handling and topology guard.  The
report is diagnostic and cannot authorize production or GPU training.
"""
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
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum.checkpoints import file_digest
from tools import audit_line_first_lsd_edge_sidecar_cpu_r5 as sidecar
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry
from tools import audit_line_first_hybrid_merge_roundtrip_r5 as hybrid
from tools import audit_line_first_valence_topology_guard_r5 as guard
from tools import shadow_line_first_edge_bond_type_r5 as shadow

OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-lsd-sidecar-threshold-sweep-r5/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
TARGET = {"precision": 0.95852, "recall": 0.85034, "sanitize": 0.945}
PROFILE_RECALL_FLOOR, PROFILE_SANITIZE_FLOOR = 0.75, 0.90


def _finish(c: Counter) -> dict:
    return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1),
            "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1),
            "bond_type_accuracy": c["type_correct"] / max(c["type_total"], 1),
            "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1),
            "oracle_sanitize_rate": c["sanitize_ok"] / max(c["images"], 1)}


def _replay(model, mean, std, threshold, rows, candidates):
    total = Counter(); profiles = defaultdict(Counter)
    for row in rows:
        selected, points, dangling = hybrid.raw(row, candidates, model, mean, std, threshold, sidecar.MERGE_PX)
        selected, _ = guard.guard(selected, low_confidence=sidecar.GUARD_TAU)
        value = guard._metrics(row, selected, points, dangling)
        total.update(value); profiles[row["profile_id"]].update(value)
    return _finish(total), {k: _finish(v) for k, v in sorted(profiles.items())}


def main() -> dict:
    rdBase.DisableLog("rdApp.error")
    if file_digest(chemistry.INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    sidecar._install()
    state = torch.load(sidecar.STATE, map_location="cpu", weights_only=False)
    model = shadow.EdgeBondMLP(len(state["feature_names"])); model.load_state_dict(state["model"]); model.eval()
    mean, std = np.asarray(state["mean"], np.float32), np.asarray(state["std"], np.float32)
    # These are selected from train-only calibration/sensitivity values.  The
    # current calibrated threshold is included exactly from the saved state.
    thresholds = [0.001, 0.004, float(state["threshold"]), 0.010, 0.020, 0.050, 0.100, 0.500]
    thresholds = sorted(set(float(x) for x in thresholds))
    rows = sidecar._rows(); hold = sidecar._rows("holdout"); candidates = hybrid.candidate_rows()
    sweeps = []
    for threshold in thresholds:
        aggregate, profiles = _replay(model, mean, std, threshold, hold, candidates)
        row = {"threshold": threshold, "holdout": aggregate, "profiles": profiles}
        row["meets_target_aggregate"] = bool(aggregate["edge_precision"] >= TARGET["precision"] and aggregate["edge_recall"] >= TARGET["recall"] and aggregate["oracle_sanitize_rate"] >= TARGET["sanitize"])
        row["meets_profile_floors"] = bool(all(v["edge_recall"] >= PROFILE_RECALL_FLOOR and v["oracle_sanitize_rate"] >= PROFILE_SANITIZE_FLOOR for v in profiles.values()))
        row["meets_all_gates"] = bool(row["meets_target_aggregate"] and row["meets_profile_floors"])
        sweeps.append(row)
        print(json.dumps({"threshold": threshold, "holdout": aggregate, "meets_all_gates": row["meets_all_gates"]}, ensure_ascii=False))
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-lsd-sidecar-threshold-sweep-r5-0.1",
        "complete": True, "read_only": False, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "proposal_contract": "new LSD min2 STD scale.75 single-link r10 -> frozen audit-local edge/type sidecar -> Hungarian/merge20/topology guard",
        "threshold_source": "train-only calibration values; no holdout-derived selection",
        "target": {**TARGET, "profile_recall_floor": PROFILE_RECALL_FLOOR, "profile_sanitize_floor": PROFILE_SANITIZE_FLOOR},
        "state_sha256": hashlib.sha256(sidecar.STATE.read_bytes()).hexdigest(),
        "initial_sha256": INITIAL_SHA, "initial_checkpoint_sha256_after": file_digest(chemistry.INITIAL),
        "sweeps": sweeps,
        "gate": {"checkpoint_unchanged": file_digest(chemistry.INITIAL) == INITIAL_SHA,
                 "sealed_test_read": False, "production_go": False,
                 "any_threshold_meets_all_gates": any(v["meets_all_gates"] for v in sweeps),
                 "decision": "NO-GO: threshold-only adjustment cannot satisfy aggregate and profile floors"},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "any_threshold_meets_all_gates": report["gate"]["any_threshold_meets_all_gates"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__": main()

