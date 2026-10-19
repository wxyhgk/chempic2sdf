#!/usr/bin/env python3
"""CPU-only type-aware acceptance shadow for the r17 pair sidecar.

The t2 presence residual found real missing edges but admitted many edges whose
frozen chemical head predicted the wrong bond type.  This read-only audit keeps
the residual and frozen pair set fixed, and fits only two target-free gates on
the train split: chemical top-1 confidence and top-1/top-2 margin.  Baseline
pairs are always retained; the gates apply only to residual additions.  No
weights or production checkpoint are changed.
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
from chemvector.bond_object import BOND_TYPES
from chemvector.training.curriculum.chemical_head_probe import INITIAL
from tools.train_r17_pair_incidence_known_clean_cpu import Residual, SHA

CACHE = ROOT / os.environ.get(
    "R17_TYPE_GATE_CACHE",
    "outputs/audits/chemvector-r17-targeted-source-pair-calibration-cpu-20261018-t2/features-cache.pt",
)
STATE = ROOT / os.environ.get(
    "R17_TYPE_GATE_STATE",
    "outputs/audits/chemvector-r17-targeted-source-pair-incidence-cpu-20261018-t2/sidecar-state.pt",
)
CALIB = ROOT / os.environ.get(
    "R17_TYPE_GATE_CALIB",
    "outputs/audits/chemvector-r17-targeted-source-pair-calibration-cpu-20261018-t2/report.json",
)
OUT = ROOT / os.environ.get(
    "R17_TYPE_GATE_OUT",
    "outputs/audits/chemvector-r17-targeted-source-pair-type-aware-gate-cpu-20261018-t2",
)
BASE_THRESHOLD = 0.72
CONF_THRESHOLDS = (0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
MARGIN_THRESHOLDS = (0.0, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metrics(items, head, thresholds, conf_threshold, margin_threshold):
    total = defaultdict(int)
    by = defaultdict(lambda: defaultdict(int))
    base_logit = math.log(BASE_THRESHOLD / (1.0 - BASE_THRESHOLD))
    for item in items:
        profile = str(item["profile"])
        alpha, clip, residual_margin = thresholds["params"]
        threshold = thresholds["profile_thresholds"].get(profile, BASE_THRESHOLD)
        threshold_logit = math.log(threshold / (1.0 - threshold))
        base = item["base_logits"]
        residual = head(item["features"]).detach()
        adjusted = base + float(alpha) * residual.clamp(-float(clip), float(clip))
        chemical = item["chemical_logits"]
        probs = chemical.softmax(-1)
        top2 = torch.topk(probs, k=2, dim=-1).values
        confidence = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
        base_selected = base >= base_logit
        additions = (
            (~base_selected)
            & (residual >= float(residual_margin))
            & (adjusted >= threshold_logit)
            & (confidence >= conf_threshold)
            & (margin >= margin_threshold)
        )
        selected = base_selected | additions
        true = typed = 0
        indices = torch.where(selected)[0].tolist()
        for i in indices:
            is_true = bool(item["labels"][i] > 0.5)
            true += int(is_true)
            if is_true and item["type_labels"][i] is not None:
                pred = BOND_TYPES[int(chemical[i].argmax())]
                typed += int(pred == item["type_labels"][i])
        target = int(item["labels"].sum())
        c = by[profile]
        for bucket, value in (("images", 1), ("predicted", len(indices)), ("true", true),
                              ("typed", typed), ("target", target),
                              ("additions", int(additions.sum()))):
            total[bucket] += value
            c[bucket] += value

    def finish(c):
        predicted = max(int(c["predicted"]), 1)
        target = max(int(c["target"]), 1)
        return {
            "images": int(c["images"]), "predicted_edges": int(c["predicted"]),
            "true_edges": int(c["true"]), "typed_true_edges": int(c["typed"]),
            "target_edges": int(c["target"]), "additions": int(c["additions"]),
            "edge_precision": c["true"] / predicted,
            "edge_recall": c["true"] / target,
            "typed_edge_precision": c["typed"] / predicted,
            "typed_edge_recall": c["typed"] / target,
        }
    return finish(total), {p: finish(c) for p, c in sorted(by.items())}


def run():
    if (OUT / "report.json").exists():
        raise FileExistsError(OUT / "report.json")
    if digest(INITIAL) != SHA:
        raise RuntimeError("initial checkpoint changed")
    cache = torch.load(CACHE, map_location="cpu")
    state = torch.load(STATE, map_location="cpu")
    if state.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("sidecar checkpoint provenance mismatch")
    calib = json.loads(CALIB.read_text(encoding="utf-8"))
    if calib.get("cohort_sha256") != cache.get("provenance", {}).get("manifest_sha256"):
        raise RuntimeError("calibration/cache cohort provenance mismatch")
    head = Residual(262)
    head.load_state_dict(state["state_dict"])
    head.eval()
    profile = calib["profile"]
    thresholds = {"params": tuple(profile["params"]), "profile_thresholds": profile["thresholds"]}
    train = cache["train"]
    hold = cache["hold"]
    base_train, base_train_profiles = metrics(
        train, head, {"params": (0.0, 1.0, float("inf")),
                      "profile_thresholds": {p: BASE_THRESHOLD for p in profile["thresholds"]}},
        1.0, 1.0,
    )
    base_hold, base_hold_profiles = metrics(
        hold, head, {"params": (0.0, 1.0, float("inf")),
                     "profile_thresholds": {p: BASE_THRESHOLD for p in profile["thresholds"]}},
        1.0, 1.0,
    )
    best = None
    trials = 0
    for conf in CONF_THRESHOLDS:
        for margin in MARGIN_THRESHOLDS:
            trials += 1
            scored, scored_profiles = metrics(train, head, thresholds, conf, margin)
            if scored["edge_precision"] < base_train["edge_precision"]:
                continue
            if scored["typed_edge_precision"] < base_train["typed_edge_precision"]:
                continue
            key = (scored["typed_edge_recall"], scored["edge_recall"], conf, margin)
            if best is None or key > best["key"]:
                best = {"key": key, "confidence": conf, "margin": margin,
                        "train": scored, "train_profiles": scored_profiles}
    if best is None:
        best = {"key": None, "confidence": None, "margin": None,
                "train": base_train, "train_profiles": base_train_profiles,
                "fallback": True}
    scored_hold, scored_hold_profiles = metrics(
        hold, head, thresholds, best["confidence"] if best["confidence"] is not None else 1.0,
        best["margin"] if best["margin"] is not None else 1.0,
    )
    deltas = {
        p: {k: scored_hold_profiles[p][k] - base_hold_profiles[p][k]
            for k in ("edge_precision", "edge_recall", "typed_edge_precision", "typed_edge_recall")}
        for p in scored_hold_profiles
    }
    gate = {
        "checkpoint_unchanged": digest(INITIAL) == SHA,
        "sealed_test_read": False, "gpu_used": False,
        "production_checkpoint_written": False, "edge_set_frozen": True,
        "each_profile_typed_precision_drop_le_0.01": all(v["typed_edge_precision"] >= -0.01 for v in deltas.values()),
        "each_profile_edge_precision_drop_le_0.01": all(v["edge_precision"] >= -0.01 for v in deltas.values()),
        "oa13_typed_recall_gain": deltas.get("oa13", {}).get("typed_edge_recall", 0.0) > 0.005,
        "oa33_typed_recall_gain": deltas.get("oa33", {}).get("typed_edge_recall", 0.0) > 0.005,
        "production_go": False,
    }
    result = {
        "schema_version": "chemvector-r17-pair-type-aware-gate-cpu-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "calibration_performed": False, "gpu_used": False,
        "sealed_test_read": False, "production_checkpoint_written": False,
        "cache": str(CACHE.relative_to(ROOT)), "cache_sha256": digest(CACHE),
        "state": str(STATE.relative_to(ROOT)), "state_sha256": digest(STATE),
        "calibration": str(CALIB.relative_to(ROOT)), "calibration_sha256": digest(CALIB),
        "trials": trials, "thresholds": thresholds,
        "baseline_train": base_train, "baseline_holdout": base_hold,
        "selection": best, "holdout": scored_hold,
        "holdout_profiles": scored_hold_profiles, "deltas": deltas, "gate": gate,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT / "report.json"), "selection": best,
                      "deltas": deltas, "gate": gate}, indent=2))


if __name__ == "__main__":
    run()
