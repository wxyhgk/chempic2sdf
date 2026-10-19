#!/usr/bin/env python3
"""CPU downstream round-trip for the selected low-scale LSD proposal.

This is an isolated replay only: the frozen edge/type sidecar, Hungarian
ownership (24 px, score lambda 8), duplicate merge (20 px), cycle critic and
valence/topology guard are kept fixed.  Only the upstream LSD proposal is
changed to the train-selected ``min_length=2, STD, scale=.75, single-link
radius=10`` configuration.  The production checkpoint and sealed split are
never touched.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.training.curriculum.checkpoints import file_digest
from tools import audit_line_first_hybrid_merge_roundtrip_r5 as hybrid
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry
from tools import audit_line_first_valence_topology_guard_r5 as guard
from tools import shadow_line_first_edge_bond_type_r5 as shadow
from tools.audit_line_first_lsd_config_sweep_r5 import detect_segments

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-lsd-hybrid-roundtrip-r5/report.json"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"

MINIMUM_LENGTH = 2.0
REFINE = "STD"
SCALE = 0.75
BORDER_MARGIN = 2.0
CLUSTER_RADIUS = 10.0
MERGE_PX = 20.0
GUARD_TAU = 0.4


def _alt_detect(image, *, config=None):
    # Keep the production function untouched; this adapter is process-local.
    return detect_segments(image, minimum_length=MINIMUM_LENGTH, refine=REFINE,
                           scale=SCALE, border_margin=BORDER_MARGIN,
                           maximum_candidates=int(getattr(config, "maximum_candidates", 512)))


def _install_process_local_proposal():
    """Patch only imported function references used by this read-only replay."""
    import tools.preflight_line_first_edge_features_r5 as preflight
    import chemvector.components.vision.line_segment_rescue as rescue
    shadow.detect_raster_line_segments = _alt_detect
    shadow.CLUSTER_RADIUS = CLUSTER_RADIUS
    preflight.detect_raster_line_segments = _alt_detect
    preflight.CLUSTER_RADIUS = CLUSTER_RADIUS
    # shadow._candidate_keys performs a local module import, so redirect that
    # module attribute as well.  This assignment is process-local and restored
    # automatically when the audit exits.
    rescue.detect_raster_line_segments = _alt_detect
    # ``chemistry._proposals`` and ``_candidate_keys`` are imported function
    # objects whose globals live in ``shadow``; replacing above is sufficient.


def _rows() -> list[dict]:
    out = [json.loads(line) for line in (DATA / "samples.jsonl").read_text().splitlines()
           if line.strip() and json.loads(line).get("render_status") == "accepted"]
    out.sort(key=lambda row: row["sample_key"])
    return out


def _finish(c: Counter) -> dict:
    return {
        **dict(c),
        "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1),
        "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1),
        "bond_type_accuracy": c["type_correct"] / max(c["type_total"], 1),
        "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1),
        "oracle_sanitize_rate": c["sanitize_ok"] / max(c["images"], 1),
    }


def _proposal_ceiling(rows: list[dict]) -> dict:
    """Evaluate raw endpoint-cluster reachability before sidecar selection."""
    # Reuse the sweep's target-free proposal evaluator from its own module.
    from tools.audit_line_first_lsd_config_sweep_r5 import audit_sample
    values = [audit_sample(row, minimum_length=MINIMUM_LENGTH, refine=REFINE,
                           scale=SCALE, border_margin=BORDER_MARGIN,
                           cluster_radius=CLUSTER_RADIUS, cluster_mode="single_link",
                           target_radius=24.0) for row in rows]
    return {
        "images": len(values),
        "mean_bond_recall": float(np.mean([v["bond_recall"] for v in values])) if values else 0.0,
        "mean_edge_precision": float(np.mean([v["edge_precision"] for v in values])) if values else 0.0,
        "mean_provisional_nodes": float(np.mean([v["provisional_nodes"] for v in values])) if values else 0.0,
        "mean_provisional_edges": float(np.mean([v["provisional_edges"] for v in values])) if values else 0.0,
        "images_all_bonds_recovered": int(sum(bool(v["all_bonds_recovered"]) for v in values)),
    }


def main() -> dict:
    rdBase.DisableLog("rdApp.error")
    if file_digest(chemistry.INITIAL) != INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    _install_process_local_proposal()
    candidates = hybrid.candidate_rows()
    state = torch.load(chemistry.STATE, map_location="cpu", weights_only=False)
    model = chemistry.EdgeBondMLP(len(state["feature_names"]))
    model.load_state_dict(state["model"]); model.eval()
    mean = np.asarray(state["mean"], np.float32); std = np.asarray(state["std"], np.float32)
    threshold = float(state["threshold"])
    rows = _rows(); by_split: dict[str, list[dict]] = defaultdict(list)
    for row in rows: by_split[row["split"]].append(row)
    outputs = {}
    for split, chosen in sorted(by_split.items()):
        total = Counter(); profiles: dict[str, Counter] = defaultdict(Counter)
        for row in chosen:
            selected, points, dangling = hybrid.raw(row, candidates, model, mean, std, threshold, MERGE_PX)
            selected, _ = guard.guard(selected, low_confidence=GUARD_TAU)
            value = guard._metrics(row, selected, points, dangling)
            total.update(value); profiles[row["profile_id"]].update(value)
        outputs[split] = {"images": len(chosen), "all": _finish(total),
                          "by_profile": {k: _finish(v) for k, v in sorted(profiles.items())}}
    hold = outputs["holdout"]["all"]
    profiles = outputs["holdout"]["by_profile"]
    report = {
        "schema_version": "chemvector-identity-aligned-line-first-lsd-hybrid-roundtrip-r5-0.1",
        "complete": True, "read_only": True, "training_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "proposal_contract": "LSD(min2,STD,scale.75) -> single-link endpoint cluster r10 -> fixed Hungarian ownership -> merge20 -> cycle/topology guard",
        "config": {"minimum_length_px": MINIMUM_LENGTH, "refine": REFINE, "scale": SCALE,
                   "border_margin_px": BORDER_MARGIN, "cluster_radius_px": CLUSTER_RADIUS,
                   "hungarian_radius_px": hybrid.RADIUS, "score_lambda_px": hybrid.LAMBDA,
                   "merge_candidates_px": MERGE_PX, "guard_tau": GUARD_TAU,
                   "edge_threshold_source": "frozen r5 sidecar train-only"},
        "proposal_ceiling": {split: _proposal_ceiling(rows_for_split) for split, rows_for_split in sorted(by_split.items())},
        "splits": outputs,
        "sidecar_state_sha256": hashlib.sha256(chemistry.STATE.read_bytes()).hexdigest(),
        "candidate_artifact_sha256": hashlib.sha256(hybrid.CANDIDATES.read_bytes()).hexdigest(),
        "initial_sha256": INITIAL_SHA, "initial_checkpoint_sha256_after": file_digest(chemistry.INITIAL),
        "gate": {
            "checkpoint_unchanged": file_digest(chemistry.INITIAL) == INITIAL_SHA,
            "sealed_test_read": False, "production_go": False,
            "holdout_edge_precision_ge_0_90": hold["edge_precision"] >= .90,
            "holdout_edge_recall_ge_0_75": hold["edge_recall"] >= .75,
            "holdout_bond_type_accuracy_ge_0_90": hold["bond_type_accuracy"] >= .90,
            "holdout_dangling_le_0_05": hold["dangling_rate"] <= .05,
            "holdout_oracle_sanitize_ge_0_95": hold["oracle_sanitize_rate"] >= .95,
            "all_profile_recall_ge_0_75": all(v["edge_recall"] >= .75 for v in profiles.values()),
            "all_profile_sanitize_ge_0_90": all(v["oracle_sanitize_rate"] >= .90 for v in profiles.values()),
        },
        "next": "CPU evidence only; do not launch GPU unless a separately reviewed gate authorizes it",
    }
    report["gate"]["bounded_shadow_admissible"] = all(report["gate"][k] for k in (
        "checkpoint_unchanged", "holdout_edge_precision_ge_0_90", "holdout_edge_recall_ge_0_75",
        "holdout_bond_type_accuracy_ge_0_90", "holdout_dangling_le_0_05", "holdout_oracle_sanitize_ge_0_95",
        "all_profile_recall_ge_0_75", "all_profile_sanitize_ge_0_90"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "proposal_ceiling": report["proposal_ceiling"],
                      "holdout": hold, "profiles": profiles, "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
