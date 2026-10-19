#!/usr/bin/env python3
"""Integrate the calibrated pair-presence route into the frozen graph compiler.

This is a CPU shadow only.  Atom proposals, pair candidates, vision features,
chemical/type heads, OCR, and the production checkpoint are frozen.  The only
changed value is pair-presence routing: baseline-positive pairs are retained
and calibrated residual additions are admitted by the train-only calibration
threshold.  Targets are used only for post-hoc metrics.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.components.graph.joint.direct_atom_pair_bond import enumerate_unordered_atom_pairs
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.audit_r17_selector_graph_replay import (
    ELEMENT_STATE,
    TYPE_STATE,
    AROMATIC_STATE,
    SHA,
    compile_prediction,
    digest,
    strict_identity,
    target,
)
from tools.train_direct_graph_pair_scorer_cpu_r9 import _map_atoms
from tools.train_element_context_head_cpu_r17 import ElementContextHead
from tools.train_r17_pair_incidence_known_clean_cpu import Residual
from tools.train_type_residual_cpu_r17 import TypeContextHead
from tools.train_graph_context_aromatic_cpu_r17 import AromaticContextHead

COHORT = ROOT / os.environ.get(
    "R17_PAIR_GRAPH_COHORT",
    "outputs/audits/chemvector-r17-oa13-oa33-bond-presence-manifest-v1/manifest.json",
)
PAIR_STATE = ROOT / os.environ.get(
    "R17_PAIR_GRAPH_STATE",
    "outputs/audits/chemvector-r17-oa13-oa33-bond-presence-cpu-v1/sidecar-state.pt",
)
CALIB_REPORT = ROOT / os.environ.get(
    "R17_PAIR_GRAPH_CALIB_REPORT",
    "outputs/audits/chemvector-r17-oa13-oa33-bond-presence-calibration-cpu-v4/report.json",
)
OUT = ROOT / os.environ.get(
    "R17_PAIR_GRAPH_OUT",
    "outputs/audits/chemvector-r17-oa13-oa33-pair-graph-integration-cpu-v1",
)
SOURCE = ROOT / os.environ.get(
    "R17_PAIR_GRAPH_SOURCE",
    "outputs/datasets/structure-expansion-batch-12272-20260923/samples",
)
PAIR_THRESHOLD = 0.72
MATCH_RADIUS = 16.0
SAFE_COMPILE = os.environ.get("R17_PAIR_GRAPH_SAFE_COMPILE", "0") == "1"
TARGET_PERTURB = os.environ.get("R17_PAIR_GRAPH_PERTURB_TARGET", "0") == "1"
CROSS_COHORT_CALIBRATION = os.environ.get("R17_PAIR_GRAPH_ALLOW_CROSS_COHORT_CALIBRATION", "0") == "1"


def _rows():
    manifest = json.loads(COHORT.read_text())
    rows = sorted(manifest["rows"], key=lambda x: x["sample_key"])
    return manifest, rows


def _state_digest(path: Path) -> str:
    return digest(path)


def _load_calibration():
    report = json.loads(CALIB_REPORT.read_text())
    if report.get("state_sha256") != _state_digest(PAIR_STATE):
        raise RuntimeError("calibration report/state mismatch")
    cohort_match = report.get("cohort_sha256") == _state_digest(COHORT)
    if not cohort_match and not CROSS_COHORT_CALIBRATION:
        raise RuntimeError("calibration report/cohort mismatch")
    profile = report["profile"]
    params = tuple(float(x) for x in profile["params"])
    thresholds = {str(k): (None if v is None else float(v)) for k, v in profile["thresholds"].items()}
    return report, params, thresholds, cohort_match


def _metrics(rows):
    c = Counter()
    by = defaultdict(Counter)
    for row in rows:
        p = row["profile"]
        for arm in ("baseline", "augmented"):
            s = row[f"{arm}_strict_identity"]
            for k in ("predicted_atoms", "mapped_atoms", "target_atoms", "element_correct", "predicted_edges", "unmapped_edges", "true_edges", "typed_true_edges", "target_edges"):
                c[f"{arm}_{k}"] += int(s[k]); by[p][f"{arm}_{k}"] += int(s[k])
            c[f"{arm}_sanitize"] += int(row[f"{arm}_sanitize"]); by[p][f"{arm}_sanitize"] += int(row[f"{arm}_sanitize"])
        c["images"] += 1; by[p]["images"] += 1

    def finish(x):
        n = max(int(x["images"]), 1)
        out = {"images": int(x["images"])}
        for arm in ("baseline", "augmented"):
            p = max(x[f"{arm}_predicted_edges"], 1); t = max(x[f"{arm}_target_edges"], 1); a = max(x[f"{arm}_mapped_atoms"], 1)
            out[arm] = {
                "predicted_edges": int(x[f"{arm}_predicted_edges"]),
                "true_edges": int(x[f"{arm}_true_edges"]),
                "typed_true_edges": int(x[f"{arm}_typed_true_edges"]),
                "target_edges": int(x[f"{arm}_target_edges"]),
                "edge_precision": x[f"{arm}_true_edges"] / p,
                "edge_recall": x[f"{arm}_true_edges"] / t,
                "typed_edge_precision": x[f"{arm}_typed_true_edges"] / p,
                "typed_edge_recall": x[f"{arm}_typed_true_edges"] / t,
                "element_accuracy": x[f"{arm}_element_correct"] / a,
                "sanitize_rate": x[f"{arm}_sanitize"] / n,
                "exact_typed_topology": int(x[f"{arm}_exact_typed_topology"] if f"{arm}_exact_typed_topology" in x else 0) / n,
            }
        return out

    # exact counts are stored separately because they are booleans, not sums
    for row in rows:
        p = row["profile"]
        for arm in ("baseline", "augmented"):
            c[f"{arm}_exact_typed_topology"] += int(row[f"{arm}_strict_identity"]["exact_typed_topology"])
            by[p][f"{arm}_exact_typed_topology"] += int(row[f"{arm}_strict_identity"]["exact_typed_topology"])
    return finish(c), {p: finish(v) for p, v in sorted(by.items())}


def _delta(metrics, arm):
    b, a = metrics["baseline"], metrics["augmented"]
    return {k: a[k] - b[k] for k in ("edge_precision", "edge_recall", "typed_edge_precision", "typed_edge_recall", "sanitize_rate", "exact_typed_topology")}


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != SHA:
        raise RuntimeError("production checkpoint changed")
    manifest, rows = _rows()
    report, params, thresholds, calibration_cohort_match = _load_calibration()
    if {r["molecule_id"] for r in rows if r["split"] == "train"} & {r["molecule_id"] for r in rows if r["split"] == "holdout"}:
        raise RuntimeError("cohort identity overlap")
    state = torch.load(PAIR_STATE, map_location="cpu", weights_only=False)
    if state.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("pair sidecar checkpoint binding mismatch")
    sidecar = Residual(262); sidecar.load_state_dict(state["state_dict"]); sidecar.eval()
    element_state = torch.load(ELEMENT_STATE, map_location="cpu", weights_only=False)
    type_state = torch.load(TYPE_STATE, map_location="cpu", weights_only=False)
    aromatic_state = torch.load(AROMATIC_STATE, map_location="cpu", weights_only=False)
    element_head = ElementContextHead(); element_head.load_state_dict(element_state["state_dict"]); element_head.eval()
    type_head = TypeContextHead(262); type_head.load_state_dict(type_state["state_dict"]); type_head.eval()
    aromatic_head = AromaticContextHead(526); aromatic_head.load_state_dict(aromatic_state["state_dict"]); aromatic_head.eval()
    aromatic_threshold = float(aromatic_state.get("threshold", 0.5))
    model, config, _ = load_bundle_for_probe(INITIAL, SHA); model.eval().requires_grad_(False)
    all_rows = []
    decisions = []
    with torch.inference_mode():
        for row in rows:
            key = row["sample_key"]
            atoms, centers, truth, target_types = target(key)
            if TARGET_PERTURB:
                # Keep routing/compile decisions target-free while making
                # post-hoc metric denominators empty for the sentinel run.
                atoms, centers, truth, target_types = ([{"element": "C"}] * len(centers)), centers, set(), {}
            pix = image_tensor(Image.open(SOURCE / f"{key}.png").convert("RGB"))[None]
            outputs = structure_forward(model, pix, config, -.25)
            vision, selected = outputs[0], outputs[6]
            mask = selected["atom_mask"]
            pairs, sample_indices = enumerate_unordered_atom_pairs(mask)
            direct = model.pair(vision, selected["atom_features"], selected["atom_geometry"][..., :2], selected["atom_scores"], pairs, sample_indices)
            points = selected["atom_geometry"][0, :, :2].float()
            pair_points = points[pairs]
            distance = torch.linalg.vector_norm((pair_points[:, 1] - pair_points[:, 0]) * 768.0, dim=-1)
            midpoint = pair_points.mean(1)
            pair_scores = selected["atom_scores"][0, pairs].float()
            features = torch.cat((direct["hidden"].float(), direct["presence_logits"][:, None].float(), midpoint, distance[:, None], pair_scores), -1)
            residual = sidecar(features).float()
            profile = str(row["profile_id"])
            threshold = thresholds.get(profile)
            threshold_logit = float("inf") if threshold is None else math.log(threshold / (1.0 - threshold))
            base_logits = direct["presence_logits"].float()
            adjusted = base_logits + params[0] * residual.clamp(-params[1], params[1])
            base_sel = base_logits.sigmoid() >= PAIR_THRESHOLD
            additions = (~base_sel) & (residual >= params[2]) & (adjusted >= threshold_logit)
            route_logits = base_logits.clone(); route_logits[additions] = adjusted[additions]
            nodes = torch.where(mask[0])[0].tolist()
            baseline_compiled = compile_prediction(selected, direct, mask[0], type_head, element_head, aromatic_head, aromatic_threshold, nodes, centers, atoms, target_types)
            accepted = int(additions.sum())
            if SAFE_COMPILE and baseline_compiled[0] and accepted:
                # Greedy target-free trust region: add one calibrated pair at
                # a time, retaining only additions whose frozen compiler still
                # sanitizes.  The residual score orders the deterministic
                # trials; no target fields enter this decision.
                route_logits = base_logits.clone()
                order = torch.where(additions)[0][torch.argsort(adjusted[additions], descending=True)]
                accepted = 0
                for idx in order.tolist():
                    trial = route_logits.clone(); trial[idx] = adjusted[idx]
                    trial_compiled = compile_prediction(selected, direct, mask[0], type_head, element_head, aromatic_head, aromatic_threshold, nodes, centers, atoms, target_types, presence_override=trial)
                    if trial_compiled[0]:
                        route_logits = trial; accepted += 1
            augmented_compiled = compile_prediction(selected, direct, mask[0], type_head, element_head, aromatic_head, aromatic_threshold, nodes, centers, atoms, target_types, presence_override=route_logits)
            bident = strict_identity(baseline_compiled[2], selected, centers, atoms, target_types)
            aident = strict_identity(augmented_compiled[2], selected, centers, atoms, target_types)
            all_rows.append({"split": row["split"], "sample_key": key, "profile": profile,
                             "base_pairs": int(len(pairs)), "baseline_route_edges": int(base_sel.sum()), "candidate_pair_edges": int(additions.sum()), "added_pair_edges": int(accepted),
                             "baseline_sanitize": bool(baseline_compiled[0]), "augmented_sanitize": bool(augmented_compiled[0]),
                             "baseline_compile_error": baseline_compiled[1], "augmented_compile_error": augmented_compiled[1],
                             "baseline_strict_identity": bident, "augmented_strict_identity": aident,
                             "baseline_prediction": baseline_compiled[2], "augmented_prediction": augmented_compiled[2]})
            decisions.append({"split": row["split"], "sample_key": key, "profile": profile,
                              "base_pairs": int(len(pairs)), "baseline_route_edges": int(base_sel.sum()), "candidate_pair_edges": int(additions.sum()), "added_pair_edges": int(accepted),
                              "selected_pair_indices": torch.where(base_sel | additions)[0].tolist()})
    # Add exact-topology counts before aggregate metrics.
    metrics = {}
    for split in sorted({r["split"] for r in all_rows}):
        subset = [r for r in all_rows if r["split"] == split]
        metrics[split] = _metrics(subset)
    decision_digest = hashlib.sha256(json.dumps(decisions, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    # ``fresh_holdout`` is deliberately a distinct role: these rows come from
    # the public release train split and are useful only for a cross-cohort
    # frozen shadow.  Never relabel them as a formal holdout.
    evaluation_split = "holdout" if "holdout" in metrics else "fresh_holdout"
    hold = metrics.get(evaluation_split, ({}, {}))[0]
    deltas = _delta(hold, "augmented") if hold else {}
    gate = {
        "checkpoint_unchanged": digest(INITIAL) == SHA,
        "sealed_test_read": False,
        "gpu_used": False,
        "target_free_routing": True,
        "candidate_set_frozen": True,
        "evaluation_split": evaluation_split,
        "evaluation_strict_edge_precision_drop_le_0.01": deltas.get("edge_precision", -99.0) >= -0.01,
        "evaluation_strict_typed_precision_drop_le_0.01": deltas.get("typed_edge_precision", -99.0) >= -0.01,
        "evaluation_strict_typed_recall_no_drop": deltas.get("typed_edge_recall", -99.0) >= 0.0,
        "evaluation_sanitize_no_drop": deltas.get("sanitize_rate", -99.0) >= 0.0,
        # Backward-compatible names for formal-holdout receipts.  For a
        # fresh_holdout run these are still evaluation gates, never release
        # admission gates.
        "holdout_strict_edge_precision_drop_le_0.01": deltas.get("edge_precision", -99.0) >= -0.01,
        "holdout_strict_typed_precision_drop_le_0.01": deltas.get("typed_edge_precision", -99.0) >= -0.01,
        "holdout_strict_typed_recall_no_drop": deltas.get("typed_edge_recall", -99.0) >= 0.0,
        "holdout_sanitize_no_drop": deltas.get("sanitize_rate", -99.0) >= 0.0,
        "calibration_cohort_match": calibration_cohort_match,
        "cross_cohort_calibration_explicit": CROSS_COHORT_CALIBRATION,
        "source_split_train_caveat": evaluation_split == "fresh_holdout",
        "production_go": False,
    }
    provenance = {"initial_checkpoint_sha256": SHA, "pair_state_sha256": digest(PAIR_STATE), "pair_state_report_sha256": digest(PAIR_STATE.parent / "report.json"),
                  "cohort_sha256": digest(COHORT), "calibration_report_sha256": digest(CALIB_REPORT), "cohort": str(COHORT.resolve()), "calibration_profile_scope": "metadata-only diagnostic"}
    result = {"schema_version": "chemvector-r17-pair-incidence-graph-integration-cpu-0.1", "complete": True, "read_only": True,
              "training_performed": False, "calibration_performed": False, "gpu_used": False, "sealed_test_read": False,
              "production_checkpoint_written": False, "cohort": str(COHORT.relative_to(ROOT)), "state": str(PAIR_STATE.relative_to(ROOT)),
              "calibration_report": str(CALIB_REPORT.relative_to(ROOT)), "params": params, "thresholds": thresholds, "safe_compile": SAFE_COMPILE,
              "target_perturbation": TARGET_PERTURB, "target_free_decision_digest": decision_digest,
              "evaluation_split": evaluation_split,
              "provenance": provenance, "decision_digest": decision_digest, "metrics": metrics, "holdout_deltas": deltas,
              "rows": all_rows, "gate": gate, "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": digest(INITIAL)}
    OUT.mkdir(parents=True, exist_ok=False); (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "metrics": metrics, "holdout_deltas": deltas, "gate": gate}, indent=2))


if __name__ == "__main__":
    run()
