#!/usr/bin/env python3
"""Bounded CPU probe for an endpoint-conditioned joint pair/type readout.

The t2 feature cache already contains frozen pair hidden features, presence
logits, chemical logits, and unordered pair indices.  This script derives
target-free endpoint-neighborhood statistics (degree, incident score mass,
neighbor maxima/means), then fits a small shared presence/type readout on
train-only rows.  Baseline edges retain the frozen chemical type; only routed
additions use the readout type distribution.  A route threshold is selected on
train molecules and evaluated once on the independent holdout.

This is a CPU shadow artifact only: no sealed data, GPU, or production weight
is read or written.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
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
from chemvector.bond_object import BOND_TYPES
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from tools.train_r17_pair_incidence_known_clean_cpu import SHA

CACHE = ROOT / os.environ.get(
    "R17_JOINT_CACHE",
    "outputs/audits/chemvector-r17-targeted-source-pair-calibration-cpu-20261018-t2/features-cache.pt",
)
OUT = ROOT / os.environ.get(
    "R17_JOINT_OUT",
    "outputs/audits/chemvector-r17-endpoint-joint-pair-type-probe-cpu-20261018-t2",
)
STEPS = int(os.environ.get("R17_JOINT_STEPS", "300"))
BASE_THRESHOLD = 0.72
SEED = 20261018


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def endpoint_features(item: dict) -> torch.Tensor:
    """Build pair features without target fields or target identities."""
    base = item["base_logits"].float()
    pair = item["pairs"].long()
    p = base.sigmoid()
    count = int(pair.max().item()) + 1 if len(pair) else 0
    degree = torch.zeros(count); mass = torch.zeros(count); maxp = torch.zeros(count)
    for node in range(count):
        incident = (pair[:, 0] == node) | (pair[:, 1] == node)
        values = p[incident]
        degree[node] = float(incident.sum())
        mass[node] = values.sum()
        maxp[node] = values.max() if len(values) else 0.0
    left, right = pair[:, 0], pair[:, 1]
    ldeg, rdeg = degree[left], degree[right]
    lmass, rmass = mass[left], mass[right]
    lmax, rmax = maxp[left], maxp[right]
    # Remove the current edge from neighborhood mass/max to avoid copying its
    # own score into both endpoint context values.
    lmass = (lmass - p).clamp_min(0.0); rmass = (rmass - p).clamp_min(0.0)
    lmax = torch.where(lmax == p, torch.zeros_like(lmax), lmax)
    rmax = torch.where(rmax == p, torch.zeros_like(rmax), rmax)
    lmean = lmass / (ldeg - 1).clamp_min(1.0)
    rmean = rmass / (rdeg - 1).clamp_min(1.0)
    chem = item["chemical_logits"].float()
    prob = chem.softmax(-1)
    top2 = prob.topk(min(2, prob.size(1)), dim=1).values
    margin = top2[:, 0] - top2[:, 1]
    entropy = -(prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()).sum(-1)
    # Pair hidden + existing geometry + endpoint context + frozen chemical
    # evidence.  Every input is image/model output only.
    return torch.cat((item["features"].float(), torch.stack(
        (ldeg, rdeg, lmass, rmass, lmean, rmean, lmax, rmax,
         prob.max(-1).values, margin, entropy), dim=1), chem), dim=1)


class JointReadout(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.presence = nn.Linear(dim, 1)
        self.chemical = nn.Linear(dim, len(BOND_TYPES))
        nn.init.zeros_(self.presence.weight); nn.init.zeros_(self.presence.bias)
        nn.init.zeros_(self.chemical.weight); nn.init.zeros_(self.chemical.bias)

    def forward(self, x):
        return self.presence(x).squeeze(-1), self.chemical(x)


def rows(items, train: bool):
    xs, ys, ts = [], [], []
    for item in items:
        x = endpoint_features(item)
        y = (item["labels"] > .5).float()
        if train:
            pos = torch.where(y > .5)[0]; neg = torch.where(y <= .5)[0]
            if not len(pos) or not len(neg):
                continue
            # Balanced positives and hard negatives chosen by frozen presence.
            n = min(len(pos), len(neg))
            order = torch.argsort(item["base_logits"][neg], descending=True)[:n]
            take = torch.cat((pos, neg[order]))
        else:
            take = torch.arange(len(y))
        types = []
        for i in take.tolist():
            label = item["type_labels"][i]
            types.append(-1 if label not in BOND_TYPES else BOND_TYPES.index(label))
        xs.append(x[take]); ys.append(y[take]); ts.append(torch.tensor(types))
    if not xs:
        raise RuntimeError("no training rows")
    return torch.cat(xs), torch.cat(ys), torch.cat(ts)


def standardize(train_x, x):
    mean = train_x.mean(0); scale = train_x.std(0, unbiased=False).clamp_min(1e-5)
    return (train_x - mean) / scale, (x - mean) / scale, mean, scale


def evaluate(items, readout, mean, scale, threshold_delta: float):
    base_logit = math.log(BASE_THRESHOLD / (1 - BASE_THRESHOLD))
    route_logit = base_logit + threshold_delta
    total = defaultdict(int); by = defaultdict(lambda: defaultdict(int))
    for item in items:
        x = (endpoint_features(item) - mean) / scale
        residual, typed = readout(x)
        base = item["base_logits"].float(); baseline = base >= base_logit
        additions = (~baseline) & (base + residual.detach() >= route_logit)
        selected = baseline | additions
        type_prob = typed.softmax(-1)
        pred = torch.where(selected)[0]
        true = typed_true = 0
        for i in pred.tolist():
            ok = bool(item["labels"][i] > .5); true += int(ok)
            if ok and item["type_labels"][i] in BOND_TYPES:
                logits = typed[i] if bool(additions[i]) else item["chemical_logits"][i].float()
                typed_true += int(BOND_TYPES[int(logits.argmax())] == item["type_labels"][i])
        c = by[item["profile"]]
        for k, v in (("images", 1), ("predicted", len(pred)), ("true", true),
                     ("typed", typed_true), ("target", int(item["labels"].sum())),
                     ("additions", int(additions.sum()))):
            total[k] += v; c[k] += v
    def finish(c):
        pred = max(c["predicted"], 1); target = max(c["target"], 1)
        return {"images": int(c["images"]), "predicted_edges": int(c["predicted"]),
                "true_edges": int(c["true"]), "typed_true_edges": int(c["typed"]),
                "target_edges": int(c["target"]), "additions": int(c["additions"]),
                "edge_precision": c["true"] / pred, "edge_recall": c["true"] / target,
                "typed_edge_precision": c["typed"] / pred,
                "typed_edge_recall": c["typed"] / target}
    return finish(total), {p: finish(c) for p, c in sorted(by.items())}


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(INITIAL) != SHA:
        raise RuntimeError("initial checkpoint changed")
    torch.set_num_threads(1); torch.manual_seed(SEED)
    cache = torch.load(CACHE, map_location="cpu")
    train, hold = cache["train"], cache["hold"]
    tx, ty, tt = rows(train, True); hx, hy, ht = rows(hold, False)
    tx, hx, mean, scale = standardize(tx, hx)
    readout = JointReadout(tx.size(1)); opt = torch.optim.AdamW(readout.parameters(), lr=2e-3, weight_decay=1e-4)
    pos_weight = ((len(ty) - ty.sum()) / ty.sum().clamp_min(1)).clamp(1, 32)
    events = []
    for step in range(1, STEPS + 1):
        presence, chemical = readout(tx)
        loss = F.binary_cross_entropy_with_logits(presence, ty, pos_weight=pos_weight)
        positive = (ty > .5) & (tt >= 0)
        if bool(positive.any()):
            loss = loss + 0.5 * F.cross_entropy(chemical[positive], tt[positive])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0); opt.step()
        if step in (1, 100, STEPS): events.append({"step": step, "loss": float(loss.detach())})
    readout.eval()
    base_train, bp_train = evaluate(train, readout, mean, scale, float("inf"))
    base_hold, bp_hold = evaluate(hold, readout, mean, scale, float("inf"))
    best = None
    for delta in (-1.5, -1.0, -.75, -.5, -.25, 0.0, .25, .5):
        score, prof = evaluate(train, readout, mean, scale, delta)
        if score["edge_precision"] < base_train["edge_precision"] or score["typed_edge_precision"] < base_train["typed_edge_precision"]:
            continue
        key = (score["typed_edge_recall"], score["edge_recall"], -delta)
        if best is None or key > best["key"]:
            best = {"key": key, "score_delta": delta, "train": score, "train_profiles": prof}
    if best is None:
        best = {"key": None, "score_delta": float("inf"), "train": base_train, "train_profiles": bp_train, "fallback": True}
    hold_score, hold_profiles = evaluate(hold, readout, mean, scale, best["score_delta"])
    deltas = {p: {k: hold_profiles[p][k] - bp_hold[p][k]
                  for k in ("edge_precision", "edge_recall", "typed_edge_precision", "typed_edge_recall")}
              for p in hold_profiles}
    gate = {"checkpoint_unchanged": digest(INITIAL) == SHA, "sealed_test_read": False,
            "gpu_used": False, "production_checkpoint_written": False,
            "feature_target_free": True,
            "each_profile_edge_precision_drop_le_0.01": all(v["edge_precision"] >= -.01 for v in deltas.values()),
            "each_profile_typed_precision_drop_le_0.01": all(v["typed_edge_precision"] >= -.01 for v in deltas.values()),
            "oa13_typed_recall_gain": deltas.get("oa13", {}).get("typed_edge_recall", 0) > .005,
            "oa33_typed_recall_gain": deltas.get("oa33", {}).get("typed_edge_recall", 0) > .005,
            "production_go": False}
    result = {"schema_version": "chemvector-r17-endpoint-joint-pair-type-probe-cpu-0.1", "complete": True,
              "training_performed": True, "training_scope": "frozen_endpoint_neighborhood_joint_pair_type_cpu",
              "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False,
              "cache": str(CACHE.relative_to(ROOT)), "cache_sha256": digest(CACHE), "steps": STEPS,
              "input_dimension": int(tx.size(1)), "events": events, "baseline_train": base_train,
              "baseline_holdout": base_hold, "selection": best, "holdout": hold_score,
              "holdout_profiles": hold_profiles, "baseline_holdout_profiles": bp_hold, "deltas": deltas,
              "gate": gate, "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": digest(INITIAL)}
    OUT.mkdir(parents=True); torch.save({"state_dict": readout.state_dict(), "mean": mean, "scale": scale, "initial_checkpoint_sha256": SHA}, OUT / "joint-readout-state.pt"); (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "selection": best, "deltas": deltas, "gate": gate}, indent=2))


if __name__ == "__main__":
    run()
