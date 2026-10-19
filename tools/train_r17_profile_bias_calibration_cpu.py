#!/usr/bin/env python3
"""CPU-only profile-conditioned low-parameter calibration probe.

This deliberately tests a much smaller hypothesis than the rejected profile
MLP: one four-class additive bias and one temperature per rendering profile.
Parameters are selected on t4b train rows and evaluated once on the disjoint
t4b holdout. Candidate edges and the production model stay frozen.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.train_r17_type_calibration_cpu import (
    OUT as _UNUSED,
    ROOT,
    SHA,
    T4B_CACHE,
    T4B_PRESENCE,
    add_deltas,
    baseline,
    candidate_grid,
    digest,
    load_presence,
    rows_from_cache,
    score,
    selected_type_rows,
)

OUT = ROOT / os.environ.get(
    "R17_PROFILE_BIAS_OUT",
    "outputs/audits/chemvector-r17-profile-bias-calibration-cpu-20261021-v1",
)
THRESHOLD = float(os.environ.get("R17_PROFILE_BIAS_THRESHOLD", "0.95"))


def choose_profile(entries):
    chosen = {}
    for profile in sorted({x["item"]["profile"] for x in entries}):
        subset = [x for x in entries if x["item"]["profile"] == profile]
        base = baseline(subset)
        best = None
        tried = 0
        for bias, temp in candidate_grid():
            got = score(subset, bias, temp, collect_loss=True)
            tried += 1
            if got[profile]["typed_precision"] < base[profile]["typed_precision"] - 0.01:
                continue
            key = (got[profile]["typed_recall"], -got[profile]["cross_entropy"], -temp)
            if best is None or key > best["key"]:
                best = {
                    "bias": bias.tolist(),
                    "temperature": float(temp),
                    "key": key,
                    "train": got[profile],
                    "baseline": base[profile],
                    "grid_candidates": tried,
                }
        if best is None:
            best = {"bias": [0.0] * 4, "temperature": 1.0, "key": None,
                    "train": base[profile], "baseline": base[profile],
                    "grid_candidates": tried, "fallback": True}
        chosen[profile] = best
    return chosen


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    if digest(__import__("tools.train_r17_pair_incidence_known_clean_cpu", fromlist=["INITIAL"]).INITIAL) != SHA:
        raise RuntimeError("production checkpoint changed")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    train, train_cache = rows_from_cache(T4B_CACHE, "train")
    holdout, _ = rows_from_cache(T4B_CACHE, "holdout")
    presence = load_presence(T4B_PRESENCE)
    train_e = selected_type_rows(train, presence, THRESHOLD)
    hold_e = selected_type_rows(holdout, presence, THRESHOLD)
    selected = choose_profile(train_e)

    hold_base = baseline(hold_e)
    hold_cal = {}
    for profile in sorted(p for p in hold_base if p != "__all__"):
        rows = [x for x in hold_e if x["item"]["profile"] == profile]
        p = selected[profile]
        hold_cal.update(score(rows, torch.tensor(p["bias"]), p["temperature"], collect_loss=True))
    deltas = add_deltas(hold_cal, hold_base)
    after = digest(__import__("tools.train_r17_pair_incidence_known_clean_cpu", fromlist=["INITIAL"]).INITIAL)
    report = {
        "schema_version": "chemvector-r17-profile-bias-calibration-cpu-0.1",
        "complete": True,
        "training_performed": True,
        "training_scope": "four_class_bias_plus_temperature_per_render_profile",
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "threshold": THRESHOLD,
        "cache": str(T4B_CACHE.relative_to(ROOT)),
        "cache_sha256": digest(T4B_CACHE),
        "presence": str(T4B_PRESENCE.relative_to(ROOT)),
        "grid": {"bias_values": [-0.30, -0.15, 0.0, 0.15, 0.30], "temperatures": [0.80, 1.0, 1.20], "candidates_per_profile": 1875},
        "selected": selected,
        "holdout_baseline": hold_base,
        "holdout_calibrated": hold_cal,
        "holdout_deltas": deltas,
        "initial_checkpoint_sha256": SHA,
        "initial_checkpoint_sha256_after": after,
        "gate": {
            "checkpoint_unchanged": after == SHA,
            "sealed_test_read": False,
            "candidate_edge_set_frozen": True,
            "each_profile_typed_precision_drop_le_0.01": all(v["typed_precision"] >= -0.01 for v in deltas.values()),
            "each_profile_edge_metrics_unchanged": all(v["edge_precision"] == 0.0 and v["edge_recall"] == 0.0 for v in deltas.values()),
            "all_profile_typed_recall_gain": all(v["typed_recall"] > 0 for v in deltas.values()),
            "production_go": False,
        },
    }
    OUT.mkdir(parents=True)
    torch.save({"schema_version": report["schema_version"], "selected": selected, "initial_checkpoint_sha256": SHA}, OUT / "calibration-state.pt")
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(OUT / "report.json"), "deltas": deltas, "gate": report["gate"]}, indent=2))


if __name__ == "__main__":
    run()
