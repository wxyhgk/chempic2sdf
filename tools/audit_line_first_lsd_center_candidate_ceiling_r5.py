#!/usr/bin/env python3
"""Target-free ceiling using new-LSD provisional centers as candidates.

This is deliberately not a model or production path.  It measures whether the
new LSD endpoint coordinate system can close the ownership seam by itself,
without the frozen vertex candidate artifact or edge scorer.
"""
from collections import Counter, defaultdict
import hashlib, json, sys
from pathlib import Path
import numpy as np
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry
from tools import audit_line_first_valence_topology_guard_r5 as guard
from tools.audit_line_first_lsd_config_sweep_r5 import detect_segments
from chemvector.training.curriculum.checkpoints import file_digest

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-lsd-center-candidate-ceiling-r5/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"

def detect(image, *, config=None):
    return detect_segments(image, minimum_length=2.0, refine="STD", scale=0.75,
                           border_margin=2.0,
                           maximum_candidates=int(getattr(config, "maximum_candidates", 512)))

def install():
    chemistry._proposals.__globals__["detect_raster_line_segments"] = detect
    chemistry._proposals.__globals__["CLUSTER_RADIUS"] = 10.0
    chemistry._candidate_keys.__globals__["detect_raster_line_segments"] = detect
    chemistry._candidate_keys.__globals__["CLUSTER_RADIUS"] = 10.0

def finish(c):
    return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1),
            "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1),
            "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1),
            "oracle_sanitize_rate": c["sanitize_ok"] / max(c["images"], 1)}

def rows():
    return sorted([json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines()
                   if x.strip() and json.loads(x).get("render_status") == "accepted"],
                  key=lambda x: x["sample_key"])

def main():
    rdBase.DisableLog("rdApp.error")
    if file_digest(chemistry.INITIAL) != INITIAL_SHA: raise RuntimeError("checkpoint changed")
    install(); by_split = defaultdict(list)
    for row in rows(): by_split[row["split"]].append(row)
    outputs = {}
    for split, chosen in sorted(by_split.items()):
        total, profiles = Counter(), defaultdict(Counter)
        for row in chosen:
            key = row["sample_key"]; _, centers, _ = chemistry._proposals(key)
            points = np.concatenate([centers.astype(np.float32), np.ones((len(centers), 1), np.float32)], axis=1)
            selected, dangling = {}, 0
            for a, b in chemistry._candidate_keys(key):
                if a >= len(centers) or b >= len(centers): dangling += 1; continue
                if a != b: selected[tuple(sorted((int(a), int(b))))] = (1.0, 1)
            ring = guard._cycles(selected)
            selected = {p: (s, 1 if t == guard.TYPE_AROMATIC and p not in ring else t)
                        for p, (s, t) in selected.items()}
            selected, _ = guard.guard(selected, low_confidence=0.4)
            value = guard._metrics(row, selected, points, dangling)
            total.update(value); profiles[row["profile_id"]].update(value)
        outputs[split] = {"images": len(chosen), "all": finish(total),
                          "by_profile": {k: finish(v) for k, v in sorted(profiles.items())}}
    hold, hp = outputs["holdout"]["all"], outputs["holdout"]["by_profile"]
    report = {"schema_version": "chemvector-identity-aligned-line-first-lsd-center-candidate-ceiling-r5-0.1",
              "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
              "sealed_test_read": False, "production_checkpoint_written": False,
              "proposal_contract": "new LSD min2 STD scale.75 -> provisional centers as synthetic candidates -> all edges -> topology guard",
              "config": {"minimum_length_px": 2.0, "scale": 0.75, "cluster_radius_px": 10.0,
                         "candidate_source": "LSD provisional centers", "bond_type": "synthetic SINGLE"},
              "splits": outputs, "initial_sha256": INITIAL_SHA,
              "initial_checkpoint_sha256_after": file_digest(chemistry.INITIAL),
              "gate": {"checkpoint_unchanged": file_digest(chemistry.INITIAL) == INITIAL_SHA,
                       "sealed_test_read": False, "production_go": False,
                       "holdout_edge_precision_ge_0.95852": hold["edge_precision"] >= .95852,
                       "holdout_edge_recall_ge_0.85034": hold["edge_recall"] >= .85034,
                       "holdout_sanitize_ge_0.945": hold["oracle_sanitize_rate"] >= .945,
                       "all_profile_recall_ge_0.75": all(v["edge_recall"] >= .75 for v in hp.values()),
                       "all_profile_sanitize_ge_0.90": all(v["oracle_sanitize_rate"] >= .90 for v in hp.values())}}
    report["gate"]["candidate_contract_ceiling_pass"] = all(v for k, v in report["gate"].items()
                                                               if k not in {"sealed_test_read", "production_go"})
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "holdout": hold, "gate": report["gate"]}, indent=2))

if __name__ == "__main__": main()
