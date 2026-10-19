#!/usr/bin/env python3
"""CPU retrain/recompute of the isolated edge/type sidecar on new LSD data.

The upstream proposal is the train-selected low-scale LSD configuration.  All
train identities are used for fitting and calibration; the complete fixed
holdout is evaluated only afterward, including six profile summaries and a
Hungarian/merge/topology graph round-trip.  This writes an audit-local sidecar
state only.  Production weights, OCR, selector and graph code are untouched.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.training.curriculum.checkpoints import file_digest
from tools import audit_line_first_graph_chemistry_roundtrip_r5 as chemistry
from tools import audit_line_first_hybrid_merge_roundtrip_r5 as hybrid
from tools import audit_line_first_valence_topology_guard_r5 as guard
from tools import shadow_line_first_edge_bond_type_r5 as shadow
from tools.audit_line_first_lsd_config_sweep_r5 import detect_segments

DATA = ROOT / "outputs/datasets/chemvector-identity-aligned-visible-text-20261017-r5-render"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-line-first-lsd-edge-sidecar-cpu-r5"
REPORT = OUT / "report.json"
STATE = OUT / "sidecar-state.pt"
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"

MINIMUM_LENGTH, REFINE, SCALE, BORDER_MARGIN, CLUSTER_RADIUS = 2.0, "STD", 0.75, 2.0, 10.0
MERGE_PX, GUARD_TAU = 20.0, 0.4
STEPS, SEED = 800, 20261007


def _alt_detect(image, *, config=None):
    return detect_segments(image, minimum_length=MINIMUM_LENGTH, refine=REFINE,
                           scale=SCALE, border_margin=BORDER_MARGIN,
                           maximum_candidates=int(getattr(config, "maximum_candidates", 512)))


def _install():
    import tools.preflight_line_first_edge_features_r5 as preflight
    import chemvector.components.vision.line_segment_rescue as rescue
    shadow.detect_raster_line_segments = _alt_detect
    shadow.CLUSTER_RADIUS = CLUSTER_RADIUS
    preflight.detect_raster_line_segments = _alt_detect
    preflight.CLUSTER_RADIUS = CLUSTER_RADIUS
    rescue.detect_raster_line_segments = _alt_detect


def _rows(split: str | None = None) -> list[dict]:
    out = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines()
           if x.strip() and json.loads(x).get("render_status") == "accepted"]
    if split is not None: out = [r for r in out if r["split"] == split]
    return sorted(out, key=lambda r: r["sample_key"])


def _collect(rows: list[dict]):
    features, labels, types, by_image = [], [], [], []
    gt_bonds = 0
    for row in rows:
        f, centres, _ = shadow._proposals(row["sample_key"])
        keys = shadow._candidate_keys(row["sample_key"])
        t, y, n = shadow._typed_labels(row["sample_key"], centres, keys)
        if len(f) != len(t):
            raise RuntimeError(f"proposal/label cardinality mismatch: {row['sample_key']} {len(f)} != {len(t)}")
        features.append(f); labels.append(y.astype(np.float32)); types.append(t)
        by_image.append({"row": row, "features": f, "labels": y.astype(np.float32), "types": t, "gt_bonds": n})
        gt_bonds += n
    return (np.concatenate(features) if features else np.empty((0, 17), np.float32),
            np.concatenate(labels) if labels else np.empty(0, np.float32),
            np.concatenate(types) if types else np.empty(0, np.int64), gt_bonds, by_image)


def _metric_dict(scores, labels, type_logits, type_labels, threshold, gt_bonds):
    return shadow._metrics(scores, labels, type_logits, type_labels, threshold, gt_bonds)


def _summarize_sidecar(model, values, mean, std, threshold):
    total = Counter(); profiles = defaultdict(Counter)
    with torch.inference_mode():
        for value in values:
            x = torch.from_numpy((value["features"] - mean) / std)
            edge, kind = model(x)
            metrics = _metric_dict(torch.sigmoid(edge).numpy(), value["labels"], kind.numpy(), value["types"], threshold, value["gt_bonds"])
            total.update(metrics); profiles[value["row"]["profile_id"]].update(metrics)
    def finish(c):
        return {**dict(c), "edge_precision": c["tp"] / max(c["tp"] + c["fp"], 1),
                "edge_recall": c["tp"] / max(c["gt_bonds"], 1),
                "bond_type_accuracy": c["typed_correct"] / max(c["typed_count"], 1)}
    return finish(total), {k: finish(v) for k, v in sorted(profiles.items())}


def _roundtrip(model, mean, std, threshold, rows):
    candidates = hybrid.candidate_rows()
    # hybrid.raw references chemistry._proposals and candidate keys whose
    # globals were patched by _install().
    total = Counter(); profiles = defaultdict(Counter)
    with torch.inference_mode():
        for row in rows:
            selected, points, dangling = hybrid.raw(row, candidates, model, mean, std, threshold, MERGE_PX)
            selected, _ = guard.guard(selected, low_confidence=GUARD_TAU)
            value = guard._metrics(row, selected, points, dangling)
            total.update(value); profiles[row["profile_id"]].update(value)
    def finish(c):
        return {**dict(c), "edge_precision": c["true_edges"] / max(c["mapped_edges"], 1),
                "edge_recall": c["true_edges"] / max(c["gt_bonds"], 1),
                "bond_type_accuracy": c["type_correct"] / max(c["type_total"], 1),
                "dangling_rate": c["dangling_edges"] / max(c["selected_edges"], 1),
                "oracle_sanitize_rate": c["sanitize_ok"] / max(c["images"], 1)}
    return finish(total), {k: finish(v) for k, v in sorted(profiles.items())}


def main() -> dict:
    rdBase.DisableLog("rdApp.error")
    if file_digest(chemistry.INITIAL) != INITIAL_SHA: raise RuntimeError("production checkpoint changed")
    _install()
    train_rows, hold_rows = _rows("train"), _rows("holdout")
    x_train, y_train, t_train, gt_train, train_values = _collect(train_rows)
    x_hold, y_hold, t_hold, gt_hold, hold_values = _collect(hold_rows)
    mean, std = x_train.mean(0), np.maximum(x_train.std(0), 1e-5)
    torch.manual_seed(SEED); model = shadow.EdgeBondMLP(x_train.shape[1]); model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    tx = torch.from_numpy((x_train - mean) / std); ty = torch.from_numpy(y_train); tt = torch.from_numpy(t_train)
    pos_weight = torch.tensor(float((1.0 - y_train.mean()) / max(y_train.mean(), 1e-4)))
    started = time.time()
    model.train()
    for _ in range(STEPS):
        edge_logits, kind_logits = model(tx)
        edge_loss = torch.nn.functional.binary_cross_entropy_with_logits(edge_logits, ty, pos_weight=pos_weight)
        positive = ty > 0.5
        kind_loss = torch.nn.functional.cross_entropy(kind_logits[positive], tt[positive] - 1) if bool(positive.any()) else edge_logits.sum() * 0
        loss = edge_loss + 0.5 * kind_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    with torch.inference_mode():
        train_edge, train_kind = model(tx); hold_edge, hold_kind = model(torch.from_numpy((x_hold - mean) / std))
    train_scores = torch.sigmoid(train_edge).numpy(); hold_scores = torch.sigmoid(hold_edge).numpy()
    calibration = shadow._calibrate(train_scores, y_train); threshold = float(calibration["threshold"])
    train_metrics = shadow._metrics(train_scores, y_train, train_kind.numpy(), t_train, threshold, gt_train)
    hold_metrics = shadow._metrics(hold_scores, y_hold, hold_kind.numpy(), t_hold, threshold, gt_hold)
    # Save only this audit-local sidecar for reproducibility.
    OUT.mkdir(parents=True, exist_ok=True)
    feature_names = ["length_mean", "length_std", "length_min", "length_max", "support_count", "angle_cos", "angle_sin", "endpoint_a_cluster_size", "endpoint_b_cluster_size", "endpoint_a_degree", "endpoint_b_degree", "endpoint_a_spread", "endpoint_b_spread", "endpoint_a_min_neighbor_angle", "endpoint_b_min_neighbor_angle", "ink_mean", "ink_fraction"]
    torch.save({"model": model.state_dict(), "mean": mean, "std": std, "threshold": threshold,
                "feature_names": feature_names, "provenance": {"initial_sha256": INITIAL_SHA,
                "sealed_test_read": False, "train_images": len(train_rows), "holdout_images": len(hold_rows),
                "proposal": "LSD min2 STD scale.75 single_link r10"}}, STATE)
    train_side, train_profiles = _summarize_sidecar(model, train_values, mean, std, threshold)
    hold_side, hold_profiles = _summarize_sidecar(model, hold_values, mean, std, threshold)
    hold_graph, hold_graph_profiles = _roundtrip(model, mean, std, threshold, hold_rows)
    train_graph, train_graph_profiles = _roundtrip(model, mean, std, threshold, train_rows)
    after = file_digest(chemistry.INITIAL)
    gate = {
        "checkpoint_unchanged": after == INITIAL_SHA, "sealed_test_read": False, "production_go": False,
        "full_holdout_sidecar_precision_ge_0_85": hold_side["edge_precision"] >= .85,
        "full_holdout_sidecar_recall_ge_0_50": hold_side["edge_recall"] >= .50,
        "full_holdout_sidecar_type_accuracy_ge_0_75": hold_side["bond_type_accuracy"] >= .75,
        "full_holdout_graph_precision_ge_0_90": hold_graph["edge_precision"] >= .90,
        "full_holdout_graph_recall_ge_0_75": hold_graph["edge_recall"] >= .75,
        "full_holdout_graph_type_accuracy_ge_0_90": hold_graph["bond_type_accuracy"] >= .90,
        "full_holdout_graph_dangling_le_0_05": hold_graph["dangling_rate"] <= .05,
        "full_holdout_graph_sanitize_ge_0_95": hold_graph["oracle_sanitize_rate"] >= .95,
        "all_profile_graph_recall_ge_0_75": all(v["edge_recall"] >= .75 for v in hold_graph_profiles.values()),
        "all_profile_graph_sanitize_ge_0_90": all(v["oracle_sanitize_rate"] >= .90 for v in hold_graph_profiles.values()),
        "all_profile_sidecar_type_accuracy_ge_0_75": all(v["bond_type_accuracy"] >= .75 for v in hold_profiles.values()),
    }
    gate["bounded_shadow_admissible"] = all(gate[k] for k in gate if k not in {"sealed_test_read", "production_go"})
    report = {"schema_version": "chemvector-identity-aligned-line-first-lsd-edge-sidecar-cpu-r5-0.1",
              "complete": True, "read_only": False, "training_performed": True,
              "training_scope": "audit-local CPU edge/type sidecar only", "optimizer_steps": STEPS,
              "elapsed_seconds": time.time() - started, "gpu_used": False, "sealed_test_read": False,
              "production_checkpoint_written": False, "initial_sha256": INITIAL_SHA,
              "initial_checkpoint_sha256_after": after,
              "proposal_contract": "LSD min2 STD scale.75 -> endpoint cluster r10 -> target-free 17-feature edge/type sidecar",
              "config": {"minimum_length_px": MINIMUM_LENGTH, "refine": REFINE, "scale": SCALE,
                         "cluster_radius_px": CLUSTER_RADIUS, "merge_px": MERGE_PX, "guard_tau": GUARD_TAU,
                         "train_images": len(train_rows), "holdout_images": len(hold_rows)},
              "calibration": {"source": "train-only", **calibration},
              "train": {"sidecar": train_side, "sidecar_by_profile": train_profiles, "graph": train_graph, "graph_by_profile": train_graph_profiles},
              "holdout": {"sidecar": hold_side, "sidecar_by_profile": hold_profiles, "graph": hold_graph, "graph_by_profile": hold_graph_profiles},
              "state_sha256": hashlib.sha256(STATE.read_bytes()).hexdigest(), "gate": gate}
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(REPORT), "holdout_sidecar": hold_side, "holdout_graph": hold_graph,
                      "profiles": hold_graph_profiles, "gate": gate}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__": main()

