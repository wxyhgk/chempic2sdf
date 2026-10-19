#!/usr/bin/env python3
"""Bounded CPU calibration probe for the frozen bond-type logits.

Only a four-class additive bias and one temperature are searched.  The frozen
candidate-edge set is produced by the already admitted presence sidecars; no
new edges are introduced and the production checkpoint is never written.

The cross-source result chooses parameters on t3e train, checks t3e holdout,
and then reports the same fixed parameters on the newer t4b holdout.  A second
diagnostic fits on t4b train only so that the amount of in-source headroom can
be compared without confusing it with the cross-source result.
"""
from __future__ import annotations

import hashlib
import json
import math
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
from tools.train_direct_graph_pair_scorer_cpu_r9 import SHA

T3E_CACHE = ROOT / os.environ.get(
    "R17_CALIB_T3E_CACHE",
    "outputs/audits/chemvector-r17-residual-density-pair-cache-20261019-t3e/features-cache.pt",
)
T3E_PRESENCE = ROOT / os.environ.get(
    "R17_CALIB_T3E_PRESENCE",
    "outputs/audits/chemvector-r17-residual-density-pair-incidence-cpu-20261019-t3e/sidecar-state.pt",
)
T4B_CACHE = ROOT / os.environ.get(
    "R17_CALIB_T4B_CACHE",
    "outputs/audits/chemvector-r17-t4b-type-correction-cpu-contract3/feature-cache.pt",
)
T4B_PRESENCE = ROOT / os.environ.get(
    "R17_CALIB_T4B_PRESENCE",
    "outputs/audits/chemvector-r17-t4b-presence-cpu/sidecar-state.pt",
)
OUT = ROOT / os.environ.get(
    "R17_CALIB_OUT", "outputs/audits/chemvector-r17-type-calibration-cpu-20261021-v1"
)
T3E_THRESHOLD = float(os.environ.get("R17_CALIB_T3E_THRESHOLD", "0.72"))
T4B_THRESHOLD = float(os.environ.get("R17_CALIB_T4B_THRESHOLD", "0.95"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_presence(path: Path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError(f"presence sidecar is not bound to frozen checkpoint: {path}")
    head = base.Residual(262)
    head.load_state_dict(state["state_dict"])
    head.eval()
    return head


def rows_from_cache(path: Path, split: str):
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("initial_checkpoint_sha256") not in (None, SHA):
        raise RuntimeError(f"cache is not bound to frozen checkpoint: {path}")
    key = split if split in cache else ("holdout" if split == "hold" else split)
    if key not in cache:
        raise KeyError((path, split, cache.keys()))
    return cache[key], cache


def selected_type_rows(rows, presence, threshold):
    """Return selected candidates and true selected type rows per image."""
    out = []
    for item in rows:
        with torch.inference_mode():
            logits = item["base_logits"] + presence(item["features"])
            selected = torch.where(logits.sigmoid() >= threshold)[0]
        out.append({"item": item, "selected": selected})
    return out


def type_index(name: str) -> int:
    value = str(name).upper()
    if value not in BOND_TYPES:
        raise ValueError(value)
    return BOND_TYPES.index(value)


def score(entries, bias, temperature, collect_loss=False, apply_profiles=None):
    by = defaultdict(lambda: {"predicted": 0, "true": 0, "typed": 0, "target": 0, "loss": 0.0, "nlogit": 0, "changed": 0})
    for entry in entries:
        item, ids = entry["item"], entry["selected"]
        profile = item["profile"]
        c = by[profile]
        c["predicted"] += int(ids.numel())
        c["true"] += int(item["labels"][ids].sum())
        c["target"] += int(len(item["target_edges"]))
        if ids.numel():
            raw = item["chemical_logits"][ids]
            active = apply_profiles is None or profile in apply_profiles
            adjusted = (raw + (bias if active else torch.zeros_like(bias))) / (temperature if active else 1.0)
            pred = adjusted.argmax(-1)
            c["changed"] += int((pred != raw.argmax(-1)).sum())
            true_idx = []
            for j, idx in enumerate(ids.tolist()):
                if item["labels"][idx] > 0.5:
                    truth = type_index(item["type_labels"][idx])
                    c["typed"] += int(int(pred[j]) == truth)
                    true_idx.append(j)
            c["nlogit"] += len(true_idx)
            if collect_loss:
                if true_idx:
                    y = torch.tensor(
                        [type_index(item["type_labels"][ids[j].item()]) for j in true_idx],
                        dtype=torch.long,
                    )
                    c["loss"] += float(torch.nn.functional.cross_entropy(adjusted[true_idx], y, reduction="sum"))
    result = {}
    total = {k: sum(c[k] for c in by.values()) for k in ("predicted", "true", "typed", "target", "loss", "nlogit", "changed")}
    for profile, c in sorted(by.items()):
        result[profile] = metrics(c)
    result["__all__"] = metrics(total)
    return result


def metrics(c):
    return {
        "predicted_edges": int(c["predicted"]),
        "true_edges": int(c["true"]),
        "target_edges": int(c["target"]),
        "edge_precision": c["true"] / max(c["predicted"], 1),
        "edge_recall": c["true"] / max(c["target"], 1),
        "typed_precision": c["typed"] / max(c["predicted"], 1),
        "typed_recall": c["typed"] / max(c["target"], 1),
        "selected_true": int(c["nlogit"]),
        "cross_entropy": c["loss"] / max(c["nlogit"], 1),
        "type_decision_changes": int(c["changed"]),
    }


def baseline(entries):
    return score(entries, torch.zeros(4), 1.0, collect_loss=True)


def candidate_grid():
    vals = (-0.30, -0.15, 0.0, 0.15, 0.30)
    temps = (0.80, 1.0, 1.20)
    for a in vals:
        for b in vals:
            for c in vals:
                for d in vals:
                    for t in temps:
                        yield torch.tensor([a, b, c, d]), t


def choose(entries):
    base_metrics = baseline(entries)
    profiles = [p for p in base_metrics if p != "__all__"]
    best = None
    tried = 0
    for bias, temp in candidate_grid():
        got = score(entries, bias, temp, collect_loss=True)
        tried += 1
        # Calibration must spend no more than one percentage point of typed
        # precision on every profile present in the selection source.
        if any(got[p]["typed_precision"] < base_metrics[p]["typed_precision"] - 0.01 for p in profiles):
            continue
        key = (got["__all__"]["typed_recall"], -got["__all__"]["cross_entropy"], -temp)
        if best is None or key > best["key"]:
            best = {"bias": bias.tolist(), "temperature": temp, "key": key, "train": got}
    return base_metrics, best, tried


def add_deltas(metrics_now, metrics_base):
    out = {}
    for p, now in metrics_now.items():
        if p == "__all__":
            continue
        out[p] = {k: now[k] - metrics_base[p][k] for k in ("typed_precision", "typed_recall", "edge_precision", "edge_recall")}
    return out


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(base.INITIAL) != SHA:
        raise RuntimeError("production checkpoint changed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    t3train, t3cache = rows_from_cache(T3E_CACHE, "train")
    t3hold, _ = rows_from_cache(T3E_CACHE, "hold")
    t4train, t4cache = rows_from_cache(T4B_CACHE, "train")
    t4hold, _ = rows_from_cache(T4B_CACHE, "holdout")
    t3train_e = selected_type_rows(t3train, load_presence(T3E_PRESENCE), T3E_THRESHOLD)
    t3hold_e = selected_type_rows(t3hold, load_presence(T3E_PRESENCE), T3E_THRESHOLD)
    t4train_e = selected_type_rows(t4train, load_presence(T4B_PRESENCE), T4B_THRESHOLD)
    t4hold_e = selected_type_rows(t4hold, load_presence(T4B_PRESENCE), T4B_THRESHOLD)

    t3_base, selected, tried = choose(t3train_e)
    if selected is None:
        raise RuntimeError("no calibration candidate satisfies the train precision floor")
    bias = torch.tensor(selected["bias"])
    temp = float(selected["temperature"])
    t3_hold_base = baseline(t3hold_e)
    t3_hold_cal = score(t3hold_e, bias, temp, collect_loss=True)
    t4_hold_base = baseline(t4hold_e)
    calibrated_profiles = sorted({p for p in t3_base if p != "__all__"})
    t4_hold_cal = score(t4hold_e, bias, temp, collect_loss=True, apply_profiles=set(calibrated_profiles))

    # Independent in-source reference: fit exactly the same tiny family on
    # t4b train, then evaluate on t4b holdout.  This is diagnostic only.
    t4_train_base, t4_selected, t4_tried = choose(t4train_e)
    if t4_selected is not None:
        t4bias = torch.tensor(t4_selected["bias"])
        t4temp = float(t4_selected["temperature"])
        t4_in_cal = score(t4hold_e, t4bias, t4temp, collect_loss=True)
    else:
        t4bias, t4temp, t4_in_cal = torch.zeros(4), 1.0, t4_hold_base

    after = digest(base.INITIAL)
    report = {
        "schema_version": "chemvector-r17-type-calibration-cpu-0.1",
        "complete": True,
        "training_performed": True,
        "training_scope": "four_class_bias_plus_temperature_on_frozen_chemical_logits",
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "thresholds": {"t3e": T3E_THRESHOLD, "t4b": T4B_THRESHOLD},
        "grid_candidates": tried,
        "selection_contract": {
            "bias_values": [-0.30, -0.15, 0.0, 0.15, 0.30],
            "temperatures": [0.80, 1.0, 1.20],
            "candidate_count": 1875,
            "selection_source": "t3e train only",
            "precision_floor": "each t3e train profile calibrated typed_precision >= baseline - 0.01",
        },
        "t3e": {
            "cache": str(T3E_CACHE.relative_to(ROOT)),
            "cache_sha256": digest(T3E_CACHE),
            "presence": str(T3E_PRESENCE.relative_to(ROOT)),
            "train_baseline": t3_base,
            "selected": selected,
            "holdout_baseline": t3_hold_base,
            "holdout_calibrated": t3_hold_cal,
            "holdout_deltas": add_deltas(t3_hold_cal, t3_hold_base),
        },
        "t4b": {
            "cache": str(T4B_CACHE.relative_to(ROOT)),
            "cache_sha256": digest(T4B_CACHE),
            "presence": str(T4B_PRESENCE.relative_to(ROOT)),
            "cross_source_holdout_baseline": t4_hold_base,
            "cross_source_holdout_calibrated": t4_hold_cal,
            "cross_source_holdout_deltas": add_deltas(t4_hold_cal, t4_hold_base),
            "cross_source_applied_profiles": calibrated_profiles,
            "in_source_train_selected": t4_selected,
            "in_source_holdout_calibrated": t4_in_cal,
            "in_source_holdout_deltas": add_deltas(t4_in_cal, t4_hold_base),
            "in_source_grid_candidates": t4_tried,
        },
        "initial_checkpoint_sha256": SHA,
        "initial_checkpoint_sha256_after": after,
        "gate": {
            "checkpoint_unchanged": after == SHA,
            "sealed_test_read": False,
            "candidate_edge_set_frozen": True,
            "cross_source_t3e_each_profile_precision_drop_le_0.01": all(v["typed_precision"] >= -0.01 for v in add_deltas(t3_hold_cal, t3_hold_base).values()),
            "cross_source_t4b_overlap_profiles_precision_drop_le_0.01": all(v["typed_precision"] >= -0.01 for p, v in add_deltas(t4_hold_cal, t4_hold_base).items() if p in {"oa13", "oa33"}),
            "production_go": False,
        },
    }
    OUT.mkdir(parents=True)
    torch.save({"schema_version": report["schema_version"], "bias": bias, "temperature": temp, "initial_checkpoint_sha256": SHA}, OUT / "calibration-state.pt")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "selected": selected, "t3e_holdout": report["t3e"]["holdout_deltas"], "t4b_cross_source": report["t4b"]["cross_source_holdout_deltas"], "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    run()
