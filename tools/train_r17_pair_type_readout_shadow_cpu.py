#!/usr/bin/env python3
"""Train/evaluate a bounded frozen-feature type readout on r17 t2.

Presence residual, candidate pair set, atom proposals, and production weights
are frozen.  A zero-initialized linear delta is trained on train-only true
typed edges.  On routing, baseline edges retain the frozen chemical type; the
readout is applied only to raw presence-sidecar additions.  A small train-only
presence score threshold sweep is used before the independent holdout report.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
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
from tools.train_r17_pair_incidence_known_clean_cpu import Residual, SHA

CACHE = ROOT / os.environ.get("R17_READOUT_CACHE", "outputs/audits/chemvector-r17-targeted-source-pair-calibration-cpu-20261018-t2/features-cache.pt")
PRESENCE_STATE = ROOT / os.environ.get("R17_READOUT_PRESENCE_STATE", "outputs/audits/chemvector-r17-targeted-source-pair-incidence-cpu-20261018-t2/sidecar-state.pt")
OUT = ROOT / os.environ.get("R17_READOUT_OUT", "outputs/audits/chemvector-r17-targeted-source-pair-type-readout-shadow-cpu-20261018-t2")
STEPS = int(os.environ.get("R17_READOUT_STEPS", "600"))
BASE_THRESHOLD = 0.72
ROUTE_DELTAS = (-0.50, -0.25, 0.0, 0.25, 0.50)


def digest(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class TypeReadout(nn.Module):
    def __init__(self, dim=262):
        super().__init__()
        self.out = nn.Linear(dim, 4)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(x)


def typed_rows(items):
    xs, base, ys = [], [], []
    for item in items:
        for i, label in enumerate(item["type_labels"]):
            if label is None or item["labels"][i] <= 0.5 or label not in BOND_TYPES:
                continue
            xs.append(item["features"][i].float())
            base.append(item["chemical_logits"][i].float())
            ys.append(BOND_TYPES.index(label))
    if not xs:
        raise RuntimeError("no typed train rows")
    return torch.stack(xs), torch.stack(base), torch.tensor(ys, dtype=torch.long)


def evaluate(items, presence, readout, score_delta):
    base_logit = math.log(BASE_THRESHOLD / (1 - BASE_THRESHOLD))
    threshold = base_logit + score_delta
    total = defaultdict(int); by = defaultdict(lambda: defaultdict(int))
    for item in items:
        base = item["base_logits"].float()
        residual = presence(item["features"].float()).detach()
        baseline = base >= base_logit
        additions = (~baseline) & (base + residual >= threshold)
        selected = baseline | additions
        refined = item["chemical_logits"].float() + readout(item["features"].float()).detach()
        pred = torch.where(selected)[0].tolist()
        true = typed = 0
        for i in pred:
            ok = bool(item["labels"][i] > 0.5); true += int(ok)
            if ok and item["type_labels"][i] is not None:
                logits = refined[i] if bool(additions[i]) else item["chemical_logits"][i].float()
                typed += int(BOND_TYPES[int(logits.argmax())] == item["type_labels"][i])
        c = by[item["profile"]]
        for bucket, value in (("images", 1), ("predicted", len(pred)), ("true", true), ("typed", typed), ("target", int(item["labels"].sum())), ("additions", int(additions.sum()))):
            c[bucket] += value; total[bucket] += value

    def finish(c):
        pred=max(int(c["predicted"]),1); target=max(int(c["target"]),1)
        return {"images":int(c["images"]),"predicted_edges":int(c["predicted"]),"true_edges":int(c["true"]),"typed_true_edges":int(c["typed"]),"target_edges":int(c["target"]),"additions":int(c["additions"]),"edge_precision":c["true"]/pred,"edge_recall":c["true"]/target,"typed_edge_precision":c["typed"]/pred,"typed_edge_recall":c["typed"]/target}
    return finish(total), {p: finish(c) for p,c in sorted(by.items())}


def run():
    if (OUT / "report.json").exists():
        raise FileExistsError(OUT / "report.json")
    if digest(INITIAL) != SHA:
        raise RuntimeError("initial checkpoint changed")
    cache = torch.load(CACHE, map_location="cpu")
    presence_state = torch.load(PRESENCE_STATE, map_location="cpu")
    presence = Residual(262); presence.load_state_dict(presence_state["state_dict"]); presence.eval()
    train, hold = cache["train"], cache["hold"]
    x, base, y = typed_rows(train)
    readout = TypeReadout(x.shape[1])
    counts = torch.bincount(y, minlength=4).float().clamp_min(1)
    weights = (len(y) / counts).sqrt(); weights = weights / weights.mean()
    opt = torch.optim.AdamW(readout.parameters(), lr=2e-3, weight_decay=1e-4)
    events=[]
    for step in range(1, STEPS + 1):
        loss = F.cross_entropy(base + readout(x), y, weight=weights)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(readout.parameters(), 1.0); opt.step()
        if step in (1, 100, 300, STEPS): events.append({"step":step,"loss":float(loss.detach())})
    readout.eval()
    base_train, base_train_profiles = evaluate(train, presence, readout, float("inf"))
    base_hold, base_hold_profiles = evaluate(hold, presence, readout, float("inf"))
    best = None
    for delta in ROUTE_DELTAS:
        scored, prof = evaluate(train, presence, readout, delta)
        if scored["edge_precision"] < base_train["edge_precision"] or scored["typed_edge_precision"] < base_train["typed_edge_precision"]:
            continue
        key = (scored["typed_edge_recall"], scored["edge_recall"], -delta)
        if best is None or key > best["key"]:
            best = {"key":key,"score_delta":delta,"train":scored,"train_profiles":prof}
    if best is None:
        best = {"key":None,"score_delta":float("inf"),"train":base_train,"train_profiles":base_train_profiles,"fallback":True}
    hold_metrics, hold_profiles = evaluate(hold, presence, readout, best["score_delta"])
    deltas = {p:{k:hold_profiles[p][k]-base_hold_profiles[p][k] for k in ("edge_precision","edge_recall","typed_edge_precision","typed_edge_recall")} for p in hold_profiles}
    gate={"checkpoint_unchanged":digest(INITIAL)==SHA,"sealed_test_read":False,"gpu_used":False,"production_checkpoint_written":False,"train_precision_floor":True,"each_profile_typed_precision_drop_le_0.01":all(v["typed_edge_precision"]>=-0.01 for v in deltas.values()),"each_profile_edge_precision_drop_le_0.01":all(v["edge_precision"]>=-0.01 for v in deltas.values()),"oa13_typed_recall_gain":deltas.get("oa13",{}).get("typed_edge_recall",0)>0.005,"oa33_typed_recall_gain":deltas.get("oa33",{}).get("typed_edge_recall",0)>0.005,"production_go":False}
    result={"schema_version":"chemvector-r17-pair-type-readout-shadow-cpu-0.1","complete":True,"training_performed":True,"training_scope":"frozen_pair_type_readout_cpu","gpu_used":False,"sealed_test_read":False,"production_checkpoint_written":False,"steps":STEPS,"train_typed_rows":len(y),"events":events,"cache":str(CACHE.relative_to(ROOT)),"cache_sha256":digest(CACHE),"presence_state_sha256":digest(PRESENCE_STATE),"baseline_train":base_train,"baseline_holdout":base_hold,"selection":best,"holdout":hold_metrics,"holdout_profiles":hold_profiles,"deltas":deltas,"gate":gate}
    OUT.mkdir(parents=True,exist_ok=True); torch.save({"schema_version":result["schema_version"],"state_dict":readout.state_dict(),"initial_checkpoint_sha256":SHA},OUT/"type-readout-state.pt"); (OUT/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(OUT/"report.json"),"selection":best,"deltas":deltas,"gate":gate},indent=2))


if __name__ == "__main__": run()
