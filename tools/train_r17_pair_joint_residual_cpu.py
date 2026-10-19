#!/usr/bin/env python3
"""Bounded joint presence/type residual shadow on the r17 t2 cache.

This is the next objective probe after presence-only additions admitted
wrongly typed edges.  A tiny presence head and four-class type head are trained
jointly on frozen pair features.  Baseline edges remain untouched; sidecars
are used only for residual additions.  The result is diagnostic-only.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.training.curriculum.chemical_head_probe import INITIAL
from tools.train_r17_pair_incidence_known_clean_cpu import Residual, SHA
from tools.train_r17_pair_type_residual_cpu import (
    BASE_THRESHOLD, CONF_THRESHOLDS, MARGIN_THRESHOLDS,
    TypeResidual, digest, load_context, route_metrics, rows,
)

OUT = ROOT / os.environ.get(
    "R17_PAIR_JOINT_OUT",
    "outputs/audits/chemvector-r17-targeted-source-pair-joint-residual-cpu-20261018-t2",
)
STEPS = int(os.environ.get("R17_PAIR_JOINT_STEPS", "600"))
TYPE_WEIGHT = float(os.environ.get("R17_PAIR_JOINT_TYPE_WEIGHT", "1.0"))


def run():
    if (OUT / "report.json").exists():
        raise FileExistsError(OUT / "report.json")
    if digest(INITIAL) != SHA:
        raise RuntimeError("initial checkpoint changed")
    cache, presence, routing = load_context()
    train, hold = cache["train"], cache["hold"]
    # Presence rows: all pair rows are target-free features; labels are used
    # only for this train-only objective.
    px, py = [], []
    for item in train:
        for i in range(len(item["labels"])):
            px.append(item["features"][i].float())
            py.append(item["labels"][i].float())
    x = torch.stack(px); y = torch.stack(py)
    tx, ty = rows(train)
    type_features = tx
    presence.load_state_dict(torch.load(
        ROOT / os.environ.get(
            "R17_PAIR_JOINT_PRESENCE_STATE",
            "outputs/audits/chemvector-r17-targeted-source-pair-incidence-cpu-20261018-t2/sidecar-state.pt",
        ), map_location="cpu")["state_dict"])
    presence.train()
    type_head = TypeResidual(type_features.shape[1])
    type_head.train()
    type_base = type_features[:, -4:].clone()
    type_counts = torch.bincount(ty, minlength=4).float().clamp_min(1)
    type_weights = (type_counts.sum() / type_counts).clamp(max=8.0)
    type_weights = type_weights / type_weights.mean()
    pos_weight = ((len(y) - y.sum()) / y.sum().clamp_min(1)).clamp(1, 32)
    opt = torch.optim.AdamW(list(presence.parameters()) + list(type_head.parameters()), lr=2e-3, weight_decay=1e-4)
    events=[]
    for step in range(1, STEPS + 1):
        presence_logits = presence(x)
        type_delta = type_head(type_features)
        presence_loss = F.binary_cross_entropy_with_logits(presence_logits, y, pos_weight=pos_weight)
        type_loss = F.cross_entropy(type_base + type_delta, ty, weight=type_weights)
        loss = presence_loss + TYPE_WEIGHT * type_loss
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(presence.parameters()) + list(type_head.parameters()), 1.0)
        opt.step()
        if step in (1, 100, 300, STEPS):
            events.append({"step": step, "loss": float(loss.detach()), "presence_loss": float(presence_loss.detach()), "type_loss": float(type_loss.detach())})
    presence.eval(); type_head.eval()
    base_route = {"params": (0.0, 1.0, float("inf")), "profile_thresholds": {p: BASE_THRESHOLD for p in routing["profile_thresholds"]}}
    base_train, base_train_profiles = route_metrics(train, presence, type_head, base_route, 1.0, 1.0)
    base_hold, base_hold_profiles = route_metrics(hold, presence, type_head, base_route, 1.0, 1.0)
    best = None; trials = 0
    for conf in CONF_THRESHOLDS:
        for margin in MARGIN_THRESHOLDS:
            trials += 1
            scored, prof = route_metrics(train, presence, type_head, routing, conf, margin)
            if scored["edge_precision"] < base_train["edge_precision"] or scored["typed_edge_precision"] < base_train["typed_edge_precision"]:
                continue
            key = (scored["typed_edge_recall"], scored["edge_recall"], conf, margin)
            if best is None or key > best["key"]:
                best = {"key": key, "confidence": conf, "margin": margin, "train": scored, "train_profiles": prof}
    if best is None:
        best = {"key": None, "confidence": None, "margin": None, "train": base_train, "train_profiles": base_train_profiles, "fallback": True}
    conf = best["confidence"] if best["confidence"] is not None else 1.0
    margin = best["margin"] if best["margin"] is not None else 1.0
    hold_metrics, hold_profiles = route_metrics(hold, presence, type_head, routing, conf, margin)
    deltas = {p: {k: hold_profiles[p][k] - base_hold_profiles[p][k] for k in ("edge_precision", "edge_recall", "typed_edge_precision", "typed_edge_recall")} for p in hold_profiles}
    gate = {
        "checkpoint_unchanged": digest(INITIAL) == SHA, "sealed_test_read": False,
        "gpu_used": False, "production_checkpoint_written": False,
        "each_profile_typed_precision_drop_le_0.01": all(v["typed_edge_precision"] >= -0.01 for v in deltas.values()),
        "each_profile_edge_precision_drop_le_0.01": all(v["edge_precision"] >= -0.01 for v in deltas.values()),
        "oa13_typed_recall_gain": deltas.get("oa13", {}).get("typed_edge_recall", 0) > 0.005,
        "oa33_typed_recall_gain": deltas.get("oa33", {}).get("typed_edge_recall", 0) > 0.005,
        "production_go": False,
    }
    result = {"schema_version": "chemvector-r17-pair-joint-residual-cpu-0.1", "complete": True, "training_performed": True, "training_scope": "frozen_pair_joint_presence_type_residual_cpu", "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "steps": STEPS, "type_weight": TYPE_WEIGHT, "train_presence_rows": len(y), "train_type_rows": len(ty), "events": events, "cache_sha256": digest(ROOT / os.environ.get("R17_TYPE_RESIDUAL_CACHE", "outputs/audits/chemvector-r17-targeted-source-pair-calibration-cpu-20261018-t2/features-cache.pt")), "selection": best, "baseline_holdout": base_hold, "holdout": hold_metrics, "holdout_profiles": hold_profiles, "deltas": deltas, "gate": gate}
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": result["schema_version"], "presence_state_dict": presence.state_dict(), "type_state_dict": type_head.state_dict(), "initial_checkpoint_sha256": SHA}, OUT / "joint-state.pt")
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT / "report.json"), "selection": best, "deltas": deltas, "gate": gate}, indent=2))


if __name__ == "__main__":
    run()
