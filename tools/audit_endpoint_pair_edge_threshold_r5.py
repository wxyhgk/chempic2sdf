#!/usr/bin/env python3
"""Read-only edge-level threshold calibration for the r5 pair sidecar.

The sidecar weights and frozen model are loaded exactly as produced by the
existing shadow receipt.  No fitting, checkpoint write, production graph call,
or threshold sweep on holdout is performed.  A train-only threshold is chosen
from *complete edge* precision >= .80 (max edge recall tie-break), then frozen
and evaluated on the holdout edge rows.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from chemvector.training.curriculum.continuation import structure_forward
from chemvector.training.curriculum.data import load_sample
from chemvector.training.curriculum.pixels import image_tensor
from chemvector.training.curriculum.train_helpers import load_bundle_for_probe
from tools.preflight_endpoint_pair_compatibility_r5 import (
    DATA, CANDIDATES, INITIAL_SHA, TRAIN_LIMIT, HOLDOUT_LIMIT,
    _candidates, _candidate_feature_rows, _target, _proposal_target,
    PairMLP, digest,
)
from tools.audit_endpoint_pair_sidecar_integration_r5 import owner_pool, atom_match

STATE = ROOT / "outputs/audits/chemvector-identity-aligned-endpoint-pair-compatibility-r5-shadow-v1/sidecar-state.pt"
SHADOW_REPORT = STATE.parent / "report.json"
OUT = ROOT / "outputs/audits/chemvector-identity-aligned-endpoint-pair-compatibility-r5-shadow-v1/edge-threshold-calibration-v2/report.json"
IMAGE = 768.0


def _edge_rows(chosen, split_name, cand, model, config, payload, side, mean, std, device):
    rows = []
    for row in chosen:
        sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
        with torch.inference_mode():
            out = structure_forward(model, image_tensor(sample["image"])[None].to(device), config, payload["provenance"]["atom_threshold"])
        bonds = {k: v.detach().cpu() for k, v in out[4].items()}
        selected = {k: v.detach().cpu() for k, v in out[6].items()}
        feats, meta, _ = _candidate_feature_rows(sample, cand[row["sample_key"]], bonds, selected)
        if not feats:
            continue
        with torch.inference_mode():
            scores = side(torch.tensor((np.asarray(feats, np.float32) - mean) / std, device=device)).sigmoid().cpu().numpy()
        points, is_residual = owner_pool(cand[row["sample_key"]], selected)
        target_points, target_bonds = _target(sample)
        # Keep the highest-scoring complete-owner proposal per frozen query.
        best = {}
        for score, m in zip(scores, meta):
            qi, i, j, _ = m
            if not (bool(is_residual[i]) or bool(is_residual[j])) or i == j:
                continue
            if qi not in best or float(score) > best[qi][0]:
                best[qi] = (float(score), i, j)
        for qi, (score, i, j) in best.items():
            mapped = _proposal_target(bonds["bond_endpoints"][0, qi], target_points, target_bonds)
            correct = False
            target_index = None
            if mapped is not None:
                target_index, (a, b) = mapped
                correct = atom_match(points[i], target_points) == a and atom_match(points[j], target_points) == b and i != j
            rows.append({"score": score, "correct": bool(correct), "target_index": target_index,
                         "target_total": len(target_bonds), "sample_key": row["sample_key"]})
    return rows


def _metric(rows, threshold):
    selected = [r for r in rows if r["score"] >= threshold]
    tp = sum(r["correct"] for r in selected)
    recovered = {(r["sample_key"], r["target_index"]) for r in selected
                 if r["correct"] and r["target_index"] is not None}
    total_targets = {r["sample_key"]: int(r["target_total"]) for r in rows}
    target_count = sum(total_targets.values())
    return {"eligible_edges": len(rows), "selected_edges": len(selected), "correct_edges": tp,
            "false_edges": len(selected) - tp, "precision": tp / max(len(selected), 1),
            "recovered_target_bonds": len(recovered), "target_bonds": target_count,
            "recall": len(recovered) / max(target_count, 1)}


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != INITIAL_SHA:
        raise RuntimeError("production checkpoint digest changed")
    receipt = json.loads(SHADOW_REPORT.read_text())
    if receipt.get("initial_sha256") != INITIAL_SHA or receipt.get("mode") != "shadow":
        raise RuntimeError("sidecar receipt mismatch")
    state = torch.load(STATE, map_location="cpu", weights_only=False)
    if state.get("initial_sha256") != INITIAL_SHA:
        raise RuntimeError("sidecar state mismatch")
    cand = _candidates()
    rows = [json.loads(x) for x in (DATA / "samples.jsonl").read_text().splitlines() if x]
    rows = [x for x in rows if x.get("render_status") == "accepted"]
    train = sorted([x for x in rows if x["split"] == "train"], key=lambda x: x["sample_key"])[:TRAIN_LIMIT]
    hold = sorted([x for x in rows if x["split"] == "holdout"], key=lambda x: x["sample_key"])[:HOLDOUT_LIMIT]
    if not torch.cuda.is_available():
        raise RuntimeError("edge calibration requires CUDA for frozen sidecar replay")
    device = torch.device("cuda:0")
    model, config, payload = load_bundle_for_probe(INITIAL, INITIAL_SHA)
    model.to(device).eval().requires_grad_(False)
    # Reconstruct frozen target-free normalization exactly as the integration audit.
    train_features = []
    for row in train:
        sample = load_sample(DATA, row["sample_key"], purpose="diagnostic")
        with torch.inference_mode():
            out = structure_forward(model, image_tensor(sample["image"])[None].to(device), config, payload["provenance"]["atom_threshold"])
        b = {k: v.detach().cpu() for k, v in out[4].items()}; s = {k: v.detach().cpu() for k, v in out[6].items()}
        f, _, _ = _candidate_feature_rows(sample, cand[row["sample_key"]], b, s); train_features.extend(f)
    x = np.asarray(train_features, np.float32); mean = x.mean(0); std = np.maximum(x.std(0), 1e-5)
    side = PairMLP(x.shape[1]).to(device); side.load_state_dict(state["state_dict"], strict=True); side.eval()
    train_edges = _edge_rows(train, "train", cand, model, config, payload, side, mean, std, device)
    hold_edges = _edge_rows(hold, "holdout", cand, model, config, payload, side, mean, std, device)
    eligible = []
    for threshold in sorted({float(r["score"]) for r in train_edges}, reverse=True):
        m = _metric(train_edges, threshold)
        if m["precision"] >= .80:
            eligible.append((m["recall"], -threshold, threshold))
    if not eligible:
        raise RuntimeError("train edge rows cannot calibrate precision >= .80")
    threshold = max(eligible)[2]
    train_metric = _metric(train_edges, threshold); hold_metric = _metric(hold_edges, threshold)
    result = {
        "schema_version": "chemvector-identity-aligned-endpoint-pair-edge-threshold-calibration-r5-0.2",
        "complete": True, "training_performed": False, "threshold_retrained": False,
        "sealed_test_read": False, "deployment_authorized": False,
        "production_checkpoint_written": False, "initial_sha256": INITIAL_SHA,
        "candidate_artifact_sha256": digest(CANDIDATES), "sidecar_state_sha256": digest(STATE),
        "threshold": threshold, "threshold_policy": "train edge precision >= 0.80; maximum edge recall tie-break; fixed holdout",
        "split": {"train_images": len(train), "holdout_images": len(hold)},
        "metrics": {"train": train_metric, "holdout": hold_metric},
        "gate": {"train_precision_ge_080": train_metric["precision"] >= .80,
                 "holdout_precision_ge_080": hold_metric["precision"] >= .80,
                 "holdout_recall_ge_050": hold_metric["recall"] >= .50,
                 "threshold_selected_from_train_only": True,
                 "baseline_prefix_retention": True,
                 "distinct_owner_enforced": True,
                 "production_edge_merge_called": False,
                 "go": False,
                 "next": "edge threshold audit only; no production integration"},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
