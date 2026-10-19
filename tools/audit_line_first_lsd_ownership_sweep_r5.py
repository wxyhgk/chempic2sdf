#!/usr/bin/env python3
"""Target-free ownership sweep for the new-LSD all-edge diagnostic."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib, json, sys
from pathlib import Path

import numpy as np
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum.checkpoints import file_digest
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry
from tools import audit_line_first_hybrid_merge_roundtrip_r5 as hybrid
from tools import audit_line_first_valence_topology_guard_r5 as guard
from tools import audit_line_first_lsd_all_edge_ownership_r5 as all_edge

OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-lsd-ownership-sweep-r5/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
RADII, LAMBDAS, MERGE_PX, TAU = (24.0, 28.0, 32.0, 36.0, 40.0), (0.0, 4.0, 8.0), 20.0, 0.4


def _finish(c):
    return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1),
            "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1),
            "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1),
            "oracle_sanitize_rate": c["sanitize_ok"] / max(c["images"], 1)}


def _prepare(rows, candidates):
    values = []
    for row in rows:
        key = row["sample_key"]
        _, centers, _ = chemistry._proposals(key)
        points = np.asarray([[v["x"] * 768.0, v["y"] * 768.0, v.get("score", 0.0)] for v in candidates[key]], dtype=np.float32)
        keys = chemistry._candidate_keys(key)
        values.append((row, centers.astype(np.float64), points.astype(np.float64), keys))
    return values


def _run(values, radius, lam):
    hybrid.RADIUS, hybrid.LAMBDA = float(radius), float(lam)
    total, profiles = Counter(), defaultdict(Counter)
    for row, centers, points, keys in values:
        owner, _ = hybrid.assign(centers, points, MERGE_PX)
        selected, dangling = {}, 0
        for a, b in keys:
            if a not in owner or b not in owner:
                dangling += 1; continue
            ca, cb = owner[a], owner[b]
            if ca == cb: continue
            selected[tuple(sorted((ca, cb)))] = (1.0, 1)  # synthetic SINGLE
        selected, _ = guard.guard(selected, low_confidence=TAU)
        metric = guard._metrics(row, selected, points.astype(np.float32), dangling)
        total.update(metric); profiles[row["profile_id"]].update(metric)
    return _finish(total), {k: _finish(v) for k, v in sorted(profiles.items())}


def main():
    rdBase.DisableLog("rdApp.error")
    if file_digest(chemistry.INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    all_edge._install(); candidates = hybrid.candidate_rows()
    train_rows, hold_rows = all_edge._rows("train"), all_edge._rows("holdout")
    train_values, hold_values = _prepare(train_rows, candidates), _prepare(hold_rows, candidates)
    sweeps = []
    for radius in RADII:
        for lam in LAMBDAS:
            tr, tr_profiles = _run(train_values, radius, lam)
            ho, ho_profiles = _run(hold_values, radius, lam)
            item = {"radius_px": radius, "lambda": lam, "train": tr, "train_profiles": tr_profiles,
                    "holdout": ho, "holdout_profiles": ho_profiles}
            # Train-only admissibility: avoid selecting a precision/dangling
            # collapse while maximizing recall.
            item["train_eligible"] = bool(tr["edge_precision"] >= .90 and tr["dangling_rate"] <= .35)
            sweeps.append(item)
            print(json.dumps({"radius_px": radius, "lambda": lam, "train": tr, "holdout": ho}, ensure_ascii=False))
    eligible = [s for s in sweeps if s["train_eligible"]]
    selected = max(eligible or sweeps, key=lambda s: (s["train"]["edge_recall"], s["train"]["edge_precision"], -s["radius_px"], -s["lambda"]))
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-lsd-ownership-sweep-r5-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "proposal_contract": "new LSD min2 STD scale.75 single-link r10 -> all provisional edges -> target-free Hungarian ownership + merge20 + topology guard",
        "config": {"radii_px": RADII, "lambdas": LAMBDAS, "merge_px": MERGE_PX, "guard_tau": TAU,
                   "selection": "train-only; precision >= .90 and dangling <= .35 then recall"},
        "selected": selected, "sweeps": sweeps,
        "initial_sha256": INITIAL_SHA, "initial_checkpoint_sha256_after": file_digest(chemistry.INITIAL),
        "candidate_artifact_sha256": hashlib.sha256(hybrid.CANDIDATES.read_bytes()).hexdigest(),
        "gate": {"checkpoint_unchanged": file_digest(chemistry.INITIAL) == INITIAL_SHA,
                 "sealed_test_read": False, "production_go": False,
                 "selected_holdout_recall_ge_0_75": selected["holdout"]["edge_recall"] >= .75,
                 "selected_holdout_dangling_le_0_05": selected["holdout"]["dangling_rate"] <= .05,
                 "selected_all_profile_recall_ge_0_75": all(v["edge_recall"] >= .75 for v in selected["holdout_profiles"].values()),
                 "selected_all_profile_sanitize_ge_0_90": all(v["oracle_sanitize_rate"] >= .90 for v in selected["holdout_profiles"].values()),
                 "ownership_only_fix_sufficient": False},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "selected": selected, "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__": main()

