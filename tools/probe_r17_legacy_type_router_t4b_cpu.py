#!/usr/bin/env python3
"""Calibrated transfer probe for the frozen r17 context type head.

The old generic type-context head is not retrained.  A single train-only
margin gate decides whether to use its type argmax or the frozen chemical-head
argmax on an already selected edge.  This tests transfer without adding a new
MLP or changing the candidate edge set.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tools.train_r17_pair_incidence_known_clean_cpu as base
from chemvector.bond_object import BOND_TYPES
from tools.train_type_residual_cpu_r17 import TypeContextHead

CACHE = ROOT / os.environ.get(
    "R17_LEGACY_ROUTER_CACHE",
    "outputs/audits/chemvector-r17-t4b-type-correction-cpu-contract3/feature-cache.pt",
)
PRESENCE = ROOT / os.environ.get(
    "R17_LEGACY_ROUTER_PRESENCE",
    "outputs/audits/chemvector-r17-t4b-presence-cpu/sidecar-state.pt",
)
OLD_STATE = ROOT / os.environ.get(
    "R17_LEGACY_ROUTER_STATE",
    "outputs/audits/chemvector-type-residual-cpu-r17-all-m128-panel128/sidecar-state.pt",
)
OUT = ROOT / os.environ.get(
    "R17_LEGACY_ROUTER_OUT",
    "outputs/audits/chemvector-r17-legacy-type-router-t4b-cpu-20261021-v1",
)
THRESHOLD = float(os.environ.get("R17_LEGACY_ROUTER_THRESHOLD", "0.95"))
INITIAL_SHA = "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_entries(rows, presence):
    out = []
    for item in rows:
        with torch.inference_mode():
            z = item["base_logits"] + presence(item["features"])
            ids = torch.where(z.sigmoid() >= THRESHOLD)[0]
            raw = item["chemical_logits"][ids]
            chemical_pred = raw.argmax(-1)
            chemical_margin = raw.topk(2, dim=-1).values[:, 0] - raw.topk(2, dim=-1).values[:, 1]
            out_logits = old_head(item["features"][ids])
            old_pred = out_logits.argmax(-1)
            old_margin = out_logits.topk(2, dim=-1).values[:, 0] - out_logits.topk(2, dim=-1).values[:, 1]
        out.append({"item": item, "ids": ids, "chemical_pred": chemical_pred,
                    "old_pred": old_pred, "delta_margin": old_margin - chemical_margin})
    return out


def metrics(entries, delta):
    by = defaultdict(lambda: {"pred": 0, "true": 0, "base": 0, "routed": 0, "changed": 0, "rescued": 0, "regressed": 0})
    for e in entries:
        p = e["item"]["profile"]
        c = by[p]
        ids = e["ids"]
        route_old = e["delta_margin"] >= delta
        pred = torch.where(route_old, e["old_pred"], e["chemical_pred"])
        c["pred"] += int(ids.numel())
        c["true"] += int(e["item"]["labels"][ids].sum())
        c["changed"] += int((pred != e["chemical_pred"]).sum())
        for j, idx in enumerate(ids.tolist()):
            if e["item"]["labels"][idx] > 0.5:
                truth = BOND_TYPES.index(str(e["item"]["type_labels"][idx]).upper())
                base_ok = int(e["chemical_pred"][j]) == truth
                route_ok = int(pred[j]) == truth
                c["base"] += base_ok
                c["routed"] += route_ok
                c["rescued"] += int((not base_ok) and route_ok)
                c["regressed"] += int(base_ok and (not route_ok))
    result = {}
    for p, c in sorted(by.items()):
        result[p] = {
            "predicted_edges": c["pred"], "true_edges": c["true"],
            "typed_precision": c["routed"] / max(c["pred"], 1),
            "typed_recall": c["routed"] / max(c["true"], 1),
            "baseline_typed_precision": c["base"] / max(c["pred"], 1),
            "baseline_typed_recall": c["base"] / max(c["true"], 1),
            "type_decision_changes": c["changed"], "rescued": c["rescued"], "regressed": c["regressed"],
        }
    return result


def choose(train):
    baseline = metrics(train, float("inf"))
    best = None
    # A negative delta permits the legacy head to be used even when its margin
    # is slightly weaker; positive values are conservative abstention.
    grid = [round(-2.0 + 0.05 * i, 2) for i in range(81)]
    for delta in grid:
        got = metrics(train, delta)
        if any(got[p]["typed_precision"] < baseline[p]["typed_precision"] - 0.01 for p in baseline):
            continue
        key = (sum(v["typed_recall"] for v in got.values()), -abs(delta), -delta)
        if best is None or key > best["key"]:
            best = {"delta": delta, "key": key, "train": got, "baseline": baseline}
    if best is None:
        best = {"delta": float("inf"), "key": None, "train": baseline, "baseline": baseline, "fallback": True}
    return best, grid


def run():
    global old_head
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(base.INITIAL) != INITIAL_SHA:
        raise RuntimeError("production checkpoint changed")
    state = torch.load(OLD_STATE, map_location="cpu", weights_only=False)
    if state.get("initial_checkpoint_sha256") != INITIAL_SHA:
        raise RuntimeError("legacy type head checkpoint binding mismatch")
    old_head = TypeContextHead(262)
    old_head.load_state_dict(state["state_dict"])
    old_head.eval()
    pstate = torch.load(PRESENCE, map_location="cpu", weights_only=False)
    if pstate.get("initial_checkpoint_sha256") != INITIAL_SHA:
        raise RuntimeError("presence checkpoint binding mismatch")
    presence = base.Residual(262)
    presence.load_state_dict(pstate["state_dict"])
    presence.eval()
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    if cache.get("binding", {}).get("initial_checkpoint_sha256") != INITIAL_SHA:
        raise RuntimeError("feature cache checkpoint binding mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train = load_entries(cache["train"], presence)
    hold = load_entries(cache["holdout"], presence)
    selected, grid = choose(train)
    delta = float(selected["delta"])
    hold_base = metrics(hold, float("inf"))
    hold_routed = metrics(hold, delta)
    deltas = {p: {k: hold_routed[p][k] - hold_base[p][k] for k in ("typed_precision", "typed_recall")} for p in hold_base}
    after = digest(base.INITIAL)
    report = {
        "schema_version": "chemvector-r17-legacy-type-router-t4b-cpu-0.1",
        "complete": True, "training_performed": True,
        "training_scope": "train_only_margin_calibration_over_frozen_legacy_type_head",
        "gpu_used": False, "sealed_test_read": False, "production_checkpoint_written": False,
        "cache": str(CACHE.relative_to(ROOT)), "cache_sha256": digest(CACHE),
        "presence": str(PRESENCE.relative_to(ROOT)), "presence_sha256": digest(PRESENCE),
        "legacy_state": str(OLD_STATE.relative_to(ROOT)), "legacy_state_sha256": digest(OLD_STATE),
        "threshold": THRESHOLD, "delta_grid": grid, "selected": selected,
        "holdout_baseline": hold_base, "holdout_routed": hold_routed, "holdout_deltas": deltas,
        "initial_checkpoint_sha256": INITIAL_SHA, "initial_checkpoint_sha256_after": after,
        "gate": {
            "checkpoint_unchanged": after == INITIAL_SHA, "sealed_test_read": False,
            "candidate_edge_set_frozen": True,
            "each_profile_typed_precision_drop_le_0.01": all(v["typed_precision"] >= -0.01 for v in deltas.values()),
            "all_profile_typed_recall_gain": all(v["typed_recall"] > 0 for v in deltas.values()),
            "production_go": False,
        },
    }
    OUT.mkdir(parents=True)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "selected_delta": delta, "deltas": deltas, "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    run()
