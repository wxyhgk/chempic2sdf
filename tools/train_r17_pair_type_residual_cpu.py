#!/usr/bin/env python3
"""Bounded CPU type residual shadow on the admitted r17 t2 pair cache.

Only bond-type logits for residual *additions* are adapted.  Presence logits,
atom proposals, unordered pairs, and the production checkpoint remain frozen.
Unknown/non-edge rows are excluded from type supervision rather than treated as
a type class.  The resulting state is diagnostic-only and never production.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemvector.bond_object import BOND_TYPES
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from tools.train_r17_pair_incidence_known_clean_cpu import Residual, SHA

CACHE = ROOT / os.environ.get(
    "R17_TYPE_RESIDUAL_CACHE",
    "outputs/audits/chemvector-r17-targeted-source-pair-calibration-cpu-20261018-t2/features-cache.pt",
)
STATE = ROOT / os.environ.get(
    "R17_TYPE_RESIDUAL_STATE",
    "outputs/audits/chemvector-r17-targeted-source-pair-incidence-cpu-20261018-t2/sidecar-state.pt",
)
CALIB = ROOT / os.environ.get(
    "R17_TYPE_RESIDUAL_CALIB",
    "outputs/audits/chemvector-r17-targeted-source-pair-calibration-cpu-20261018-t2/report.json",
)
OUT = ROOT / os.environ.get(
    "R17_TYPE_RESIDUAL_OUT",
    "outputs/audits/chemvector-r17-targeted-source-pair-type-residual-cpu-20261018-t2",
)
STEPS = int(os.environ.get("R17_TYPE_RESIDUAL_STEPS", "600"))
BASE_THRESHOLD = 0.72
CONF_THRESHOLDS = (0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
MARGIN_THRESHOLDS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class TypeResidual(nn.Module):
    def __init__(self, dim: int = 266, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 4))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def load_context():
    cache = torch.load(CACHE, map_location="cpu")
    state = torch.load(STATE, map_location="cpu")
    calib = json.loads(CALIB.read_text(encoding="utf-8"))
    if state.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("presence sidecar provenance mismatch")
    head = Residual(262)
    head.load_state_dict(state["state_dict"])
    head.eval()
    params = tuple(calib["profile"]["params"])
    thresholds = dict(calib["profile"]["thresholds"])
    return cache, head, {"params": params, "profile_thresholds": thresholds}


def rows(items):
    out_x, out_y = [], []
    for item in items:
        chemical = item["chemical_logits"].float()
        features = torch.cat((item["features"].float(), chemical), -1)
        for index, label in enumerate(item["type_labels"]):
            if item["labels"][index] <= 0.5 or label is None:
                continue
            if label not in BOND_TYPES:
                continue
            out_x.append(features[index])
            out_y.append(BOND_TYPES.index(label))
    if not out_x:
        raise RuntimeError("no typed positive rows in training cache")
    return torch.stack(out_x), torch.tensor(out_y, dtype=torch.long)


def route_metrics(items, presence_head, type_head, routing, conf, margin):
    total = defaultdict(int); by = defaultdict(lambda: defaultdict(int))
    base_logit = math.log(BASE_THRESHOLD / (1 - BASE_THRESHOLD))
    alpha, clip, residual_margin = routing["params"]
    for item in items:
        p = str(item["profile"])
        threshold = float(routing["profile_thresholds"].get(p, BASE_THRESHOLD))
        threshold_logit = math.log(threshold / (1 - threshold))
        base = item["base_logits"].float()
        residual = presence_head(item["features"].float()).detach()
        adjusted = base + float(alpha) * residual.clamp(-float(clip), float(clip))
        base_sel = base >= base_logit
        additions = (~base_sel) & (residual >= float(residual_margin)) & (adjusted >= threshold_logit)
        type_features = torch.cat((item["features"].float(), item["chemical_logits"].float()), -1)
        refined = item["chemical_logits"].float() + type_head(type_features).detach()
        probs = refined.softmax(-1)
        top = torch.topk(probs, k=2, dim=-1)
        additions &= top.values[:, 0] >= conf
        additions &= (top.values[:, 0] - top.values[:, 1]) >= margin
        selected = base_sel | additions
        pred = torch.where(selected)[0].tolist(); true = typed = 0
        for i in pred:
            ok = bool(item["labels"][i] > 0.5); true += int(ok)
            if ok and item["type_labels"][i] is not None:
                # Baseline edges retain the frozen production type head.  The
                # learned type residual is evaluated only on newly admitted
                # presence additions; otherwise the baseline comparison would
                # itself be changed by the shadow head.
                logits = refined[i] if bool(additions[i]) else item["chemical_logits"][i].float()
                typed += int(BOND_TYPES[int(logits.argmax())] == item["type_labels"][i])
        target = int(item["labels"].sum())
        for bucket, value in (("images", 1), ("predicted", len(pred)), ("true", true),
                              ("typed", typed), ("target", target), ("additions", int(additions.sum()))):
            total[bucket] += value; by[p][bucket] += value
    def finish(c):
        pred=max(int(c["predicted"]),1); target=max(int(c["target"]),1)
        return {"images":int(c["images"]),"predicted_edges":int(c["predicted"]),"true_edges":int(c["true"]),"typed_true_edges":int(c["typed"]),"target_edges":int(c["target"]),"additions":int(c["additions"]),"edge_precision":c["true"]/pred,"edge_recall":c["true"]/target,"typed_edge_precision":c["typed"]/pred,"typed_edge_recall":c["typed"]/target}
    return finish(total), {p: finish(c) for p,c in sorted(by.items())}


def run():
    if (OUT / "report.json").exists(): raise FileExistsError(OUT / "report.json")
    if digest(INITIAL) != SHA: raise RuntimeError("initial checkpoint changed")
    cache, presence_head, routing = load_context()
    train, hold = cache["train"], cache["hold"]
    x, y = rows(train)
    torch.manual_seed(20261018)
    type_head = TypeResidual(x.shape[1])
    counts = torch.bincount(y, minlength=4).float().clamp_min(1)
    weights = (counts.sum() / counts).clamp(max=8.0); weights = weights / weights.mean()
    opt = torch.optim.AdamW(type_head.parameters(), lr=2e-3, weight_decay=1e-4)
    base_type_logits = x[:, -4:].clone()
    events=[]
    for step in range(1, STEPS + 1):
        logits = type_head(x)
        loss = F.cross_entropy(logits + base_type_logits, y, weight=weights)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(type_head.parameters(), 1.0); opt.step()
        if step in (1, 100, 300, STEPS): events.append({"step":step,"loss":float(loss.detach())})
    type_head.eval()
    # Baseline keeps all baseline pairs and uses frozen chemical predictions.
    base_routing = {"params": (0.0,1.0,float("inf")), "profile_thresholds": {p: BASE_THRESHOLD for p in routing["profile_thresholds"]}}
    base_train, base_train_profiles = route_metrics(train, presence_head, type_head, base_routing, 1.0, 1.0)
    base_hold, base_hold_profiles = route_metrics(hold, presence_head, type_head, base_routing, 1.0, 1.0)
    best=None; trials=0
    for conf in CONF_THRESHOLDS:
        for margin in MARGIN_THRESHOLDS:
            trials += 1
            scored, prof = route_metrics(train, presence_head, type_head, routing, conf, margin)
            if scored["edge_precision"] < base_train["edge_precision"] or scored["typed_edge_precision"] < base_train["typed_edge_precision"]:
                continue
            key=(scored["typed_edge_recall"],scored["edge_recall"],conf,margin)
            if best is None or key > best["key"]: best={"key":key,"confidence":conf,"margin":margin,"train":scored,"train_profiles":prof}
    if best is None: best={"key":None,"confidence":None,"margin":None,"train":base_train,"train_profiles":base_train_profiles,"fallback":True}
    conf=best["confidence"] if best["confidence"] is not None else 1.0; margin=best["margin"] if best["margin"] is not None else 1.0
    hold_metrics, hold_profiles = route_metrics(hold,presence_head,type_head,routing,conf,margin)
    deltas={p:{k:hold_profiles[p][k]-base_hold_profiles[p][k] for k in ("edge_precision","edge_recall","typed_edge_precision","typed_edge_recall")} for p in hold_profiles}
    gate={"checkpoint_unchanged":digest(INITIAL)==SHA,"sealed_test_read":False,"gpu_used":False,"production_checkpoint_written":False,"each_profile_typed_precision_drop_le_0.01":all(v["typed_edge_precision"]>=-0.01 for v in deltas.values()),"each_profile_edge_precision_drop_le_0.01":all(v["edge_precision"]>=-0.01 for v in deltas.values()),"oa13_typed_recall_gain":deltas.get("oa13",{}).get("typed_edge_recall",0)>0.005,"oa33_typed_recall_gain":deltas.get("oa33",{}).get("typed_edge_recall",0)>0.005,"production_go":False}
    result={"schema_version":"chemvector-r17-pair-type-residual-cpu-0.1","complete":True,"training_performed":True,"training_scope":"frozen_pair_type_residual_cpu","gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"steps":STEPS,"train_typed_rows":len(y),"events":events,"cache":str(CACHE.relative_to(ROOT)),"cache_sha256":digest(CACHE),"state_sha256":digest(STATE),"routing":routing,"selection":best,"baseline_holdout":base_hold,"holdout":hold_metrics,"holdout_profiles":hold_profiles,"deltas":deltas,"gate":gate}
    OUT.mkdir(parents=True,exist_ok=True); torch.save({"schema_version":result["schema_version"],"state_dict":type_head.state_dict(),"initial_checkpoint_sha256":SHA},OUT/"type-residual-state.pt"); (OUT/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(OUT/"report.json"),"selection":best,"deltas":deltas,"gate":gate},indent=2))

if __name__ == "__main__": run()
