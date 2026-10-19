#!/usr/bin/env python3
"""Bounded CPU probe: linear bond-type residual on the t4b cache.

This is an attribution experiment, not a deployable model.  The residual has
only 262*4+4 parameters, starts at exactly zero, and is trained on selected
true train edges.  Presence, candidate edges, chemical logits, backbone, and
the production checkpoint remain frozen.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import tools.train_r17_pair_incidence_known_clean_cpu as base
from chemvector.bond_object import BOND_TYPES

CACHE = ROOT / os.environ.get("R17_LINEAR_TYPE_CACHE", "outputs/audits/chemvector-r17-t4b-type-correction-cpu-contract3/feature-cache.pt")
PRESENCE = ROOT / os.environ.get("R17_LINEAR_TYPE_PRESENCE", "outputs/audits/chemvector-r17-t4b-presence-cpu/sidecar-state.pt")
OUT = ROOT / os.environ.get("R17_LINEAR_TYPE_OUT", "outputs/audits/chemvector-r17-linear-type-residual-cpu-20261021-v1")
THRESHOLD = float(os.environ.get("R17_LINEAR_TYPE_THRESHOLD", "0.95"))
STEPS = int(os.environ.get("R17_LINEAR_TYPE_STEPS", "300"))
SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class LinearResidual(nn.Module):
    def __init__(self, dim=262):
        super().__init__()
        self.out = nn.Linear(dim, 4)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(x)


def type_id(value):
    return BOND_TYPES.index(str(value).upper())


def entries(rows, presence):
    out = []
    for item in rows:
        with torch.inference_mode():
            z = item["base_logits"] + presence(item["features"])
            ids = torch.where(z.sigmoid() >= THRESHOLD)[0]
        out.append((item, ids))
    return out


def metrics(items, head=None):
    by = defaultdict(lambda: {"pred": 0, "true": 0, "typed": 0, "base_typed": 0, "target": 0})
    for item, ids in items:
        p = item["profile"]
        c = by[p]
        c["pred"] += len(ids); c["true"] += int(item["labels"][ids].sum()); c["target"] += len(item["target_edges"])
        if len(ids):
            logits = item["chemical_logits"][ids]
            base_pred = logits.argmax(-1)
            pred = base_pred if head is None else (logits + head(torch.cat((item["features"][ids], logits), dim=-1))).argmax(-1)
            for j, idx in enumerate(ids.tolist()):
                if item["labels"][idx] > 0.5:
                    y = type_id(item["type_labels"][idx]); c["base_typed"] += int(int(base_pred[j]) == y); c["typed"] += int(int(pred[j]) == y)
    result = {}
    for p, c in sorted(by.items()):
        result[p] = {"predicted_edges": c["pred"], "true_edges": c["true"], "target_edges": c["target"],
                     "edge_precision": c["true"] / max(c["pred"], 1), "edge_recall": c["true"] / max(c["target"], 1),
                     "typed_precision": c["typed"] / max(c["pred"], 1), "typed_recall": c["typed"] / max(c["target"], 1),
                     "baseline_typed_precision": c["base_typed"] / max(c["pred"], 1), "baseline_typed_recall": c["base_typed"] / max(c["target"], 1)}
    return result


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(base.INITIAL) != SHA:
        raise RuntimeError("production checkpoint changed")
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    if cache.get("binding", {}).get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("cache checkpoint binding mismatch")
    state = torch.load(PRESENCE, map_location="cpu", weights_only=False)
    if state.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("presence checkpoint binding mismatch")
    presence = base.Residual(262); presence.load_state_dict(state["state_dict"]); presence.eval()
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(20261021)
    train = entries(cache["train"], presence); hold = entries(cache["holdout"], presence)
    xs, ys = [], []
    for item, ids in train:
        pos = [i for i in ids.tolist() if item["labels"][i] > 0.5]
        if pos:
            xs.append(torch.cat((item["features"][pos], item["chemical_logits"][pos]), dim=-1)); ys.append(torch.tensor([type_id(item["type_labels"][i]) for i in pos]))
    x, y = torch.cat(xs), torch.cat(ys)
    head = LinearResidual(x.shape[-1])
    with torch.no_grad():
        if not torch.equal((x[:, -4:] + head(x)).argmax(-1), x[:, -4:].argmax(-1)):
            raise RuntimeError("zero residual does not preserve chemical argmax")
    counts = torch.bincount(y, minlength=4).float(); weights = (len(y) / counts.clamp_min(1)).sqrt()
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4); events = []
    for step in range(1, STEPS + 1):
        loss = F.cross_entropy(x[:, -4:] + head(x), y, weight=weights)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step()
        if step in (1, 100, STEPS): events.append({"step": step, "loss": float(loss.detach())})
    head.eval(); train_base = metrics(train); train_corrected = metrics(train, head); base_metrics = metrics(hold); corrected = metrics(hold, head)
    train_deltas = {p: {k: train_corrected[p][k] - train_base[p][k] for k in ("typed_precision", "typed_recall", "edge_precision", "edge_recall")} for p in train_base}
    deltas = {p: {k: corrected[p][k] - base_metrics[p][k] for k in ("typed_precision", "typed_recall", "edge_precision", "edge_recall")} for p in base_metrics}
    after = digest(base.INITIAL)
    report = {"schema_version": "chemvector-r17-linear-type-residual-cpu-0.1", "complete": True, "training_performed": True,
              "training_scope": "linear_262_to_4_type_residual_on_selected_true_train_edges", "parameters": sum(p.numel() for p in head.parameters()),
              "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False, "steps": STEPS,
              "train_rows": len(y), "events": events, "train_baseline": train_base, "train_corrected": train_corrected, "train_deltas": train_deltas,
              "baseline": base_metrics, "corrected": corrected, "deltas": deltas,
              "cache": str(CACHE.relative_to(ROOT)), "cache_sha256": digest(CACHE), "presence": str(PRESENCE.relative_to(ROOT)),
              "initial_checkpoint_sha256": SHA, "initial_checkpoint_sha256_after": after,
              "gate": {"checkpoint_unchanged": after == SHA, "sealed_test_read": False, "candidate_edge_set_frozen": True,
                       "each_profile_typed_precision_drop_le_0.01": all(v["typed_precision"] >= -0.01 for v in deltas.values()),
                       "all_profile_typed_recall_gain": all(v["typed_recall"] > 0 for v in deltas.values()), "production_go": False}}
    OUT.mkdir(parents=True); torch.save({"schema_version": report["schema_version"], "state_dict": head.state_dict(), "initial_checkpoint_sha256": SHA}, OUT / "sidecar-state.pt"); (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "parameters": report["parameters"], "deltas": deltas, "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    run()
