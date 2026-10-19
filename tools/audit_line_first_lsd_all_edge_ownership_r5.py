#!/usr/bin/env python3
"""Diagnostic: retain every new-LSD provisional edge through ownership.

No edge MLP, confidence threshold or learned bond type is used.  Every
proposal edge receives a synthetic SINGLE type and score 1.0, then fixed
Hungarian ownership, merge-20 and the target-free topology guard are applied.
The synthetic type is explicitly excluded from type conclusions; edge
reachability, dangling ownership and oracle sanitize are the purpose of this
probe.  All labels remain post-hoc metrics.
"""
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
from tools.audit_line_first_lsd_config_sweep_r5 import detect_segments

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-lsd-all-edge-ownership-r5/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"
MINIMUM_LENGTH, REFINE, SCALE, BORDER_MARGIN, CLUSTER_RADIUS = 2.0, "STD", 0.75, 2.0, 10.0
MERGE_PX, GUARD_TAU = 20.0, 0.4


def _alt_detect(image, *, config=None):
    return detect_segments(image, minimum_length=MINIMUM_LENGTH, refine=REFINE, scale=SCALE,
                           border_margin=BORDER_MARGIN,
                           maximum_candidates=int(getattr(config, "maximum_candidates", 512)))


def _install():
    import tools.preflight_line_first_edge_features_r5 as preflight
    import chemvector.components.vision.line_segment_rescue as rescue
    chemistry._proposals.__globals__["detect_raster_line_segments"] = _alt_detect
    chemistry._proposals.__globals__["CLUSTER_RADIUS"] = CLUSTER_RADIUS
    chemistry._candidate_keys.__globals__["CLUSTER_RADIUS"] = CLUSTER_RADIUS
    hybrid.shadow.detect_raster_line_segments = _alt_detect
    hybrid.shadow.CLUSTER_RADIUS = CLUSTER_RADIUS
    preflight.detect_raster_line_segments = _alt_detect
    preflight.CLUSTER_RADIUS = CLUSTER_RADIUS
    rescue.detect_raster_line_segments = _alt_detect


def _rows(split=None):
    out = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines()
           if x.strip() and json.loads(x).get("render_status") == "accepted"]
    if split is not None: out = [r for r in out if r["split"] == split]
    return sorted(out, key=lambda r: r["sample_key"])


def _finish(c: Counter):
    return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1),
            "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1),
            "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1),
            "oracle_sanitize_rate": c["sanitize_ok"] / max(c["images"], 1),
            "synthetic_type_accuracy_not_interpretable": True}


def _raw(row, candidates):
    key = row["sample_key"]
    _, centers, _ = chemistry._proposals(key)
    vals = candidates[key]
    points = np.asarray([[v["x"] * 768.0, v["y"] * 768.0, v.get("score", 0.0)] for v in vals], dtype=np.float32)
    owner, _ = hybrid.assign(centers.astype(np.float64), points.astype(np.float64), MERGE_PX)
    selected = {}; dangling = 0
    for a, b in chemistry._candidate_keys(key):
        if a not in owner or b not in owner:
            dangling += 1; continue
        ca, cb = owner[a], owner[b]
        if ca == cb: continue
        pair = tuple(sorted((ca, cb)))
        # score 1.0 retains every owned proposal; SINGLE is synthetic solely
        # to let the existing topology/sanitize metric execute.
        selected[pair] = (1.0, 1)
    return selected, points, dangling


def main():
    rdBase.DisableLog("rdApp.error")
    if file_digest(chemistry.INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    _install(); candidates = hybrid.candidate_rows(); rows = _rows()
    by_split = defaultdict(list)
    for row in rows: by_split[row["split"]].append(row)
    outputs = {}
    for split, chosen in sorted(by_split.items()):
        total = Counter(); profiles = defaultdict(Counter); dropped = Counter()
        for row in chosen:
            selected, points, dangling = _raw(row, candidates)
            guarded, edits = guard.guard(selected, low_confidence=GUARD_TAU)
            value = guard._metrics(row, guarded, points, dangling)
            total.update(value); profiles[row["profile_id"]].update(value); dropped.update(edits)
        outputs[split] = {"images": len(chosen), "all": _finish(total),
                          "by_profile": {k: _finish(v) for k, v in sorted(profiles.items())},
                          "guard_edits": dict(dropped)}
    hold = outputs["holdout"]["all"]; hp = outputs["holdout"]["by_profile"]
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-lsd-all-edge-ownership-r5-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "proposal_contract": "LSD min2 STD scale.75 -> endpoint cluster r10 -> all provisional edges -> Hungarian ownership + merge20 + topology guard",
        "config": {"minimum_length_px": MINIMUM_LENGTH, "refine": REFINE, "scale": SCALE,
                   "border_margin_px": BORDER_MARGIN, "cluster_radius_px": CLUSTER_RADIUS,
                   "ownership_radius_px": hybrid.RADIUS, "ownership_score_lambda_px": hybrid.LAMBDA,
                   "merge_px": MERGE_PX, "guard_tau": GUARD_TAU, "bond_type": "synthetic SINGLE; not interpretable"},
        "splits": outputs,
        "initial_sha256": INITIAL_SHA, "initial_checkpoint_sha256_after": file_digest(chemistry.INITIAL),
        "candidate_artifact_sha256": hashlib.sha256(hybrid.CANDIDATES.read_bytes()).hexdigest(),
        "gate": {"checkpoint_unchanged": file_digest(chemistry.INITIAL) == INITIAL_SHA,
                 "sealed_test_read": False, "production_go": False,
                 "holdout_edge_precision_ge_0_90": hold["edge_precision"] >= .90,
                 "holdout_edge_recall_ge_0_75": hold["edge_recall"] >= .75,
                 "holdout_dangling_le_0_05": hold["dangling_rate"] <= .05,
                 "holdout_oracle_sanitize_ge_0_90": hold["oracle_sanitize_rate"] >= .90,
                 "all_profile_recall_ge_0_75": all(v["edge_recall"] >= .75 for v in hp.values()),
                 "all_profile_sanitize_ge_0_90": all(v["oracle_sanitize_rate"] >= .90 for v in hp.values())},
    }
    report["gate"]["scorer_is_the_bottleneck"] = bool(hold["edge_recall"] > .85)
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "holdout": hold, "profiles": hp, "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__": main()

