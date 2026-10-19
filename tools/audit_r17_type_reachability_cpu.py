#!/usr/bin/env python3
"""Read-only attribution of frozen chemical-head reachability on r17 t2.

This audit does not train or alter a checkpoint.  It uses the existing t2
feature cache and presence sidecar to inspect true edges that are outside the
frozen presence baseline, then reports their bond-type distribution and the
frozen chemical head's top-1/margin.  It is deliberately independent of the
sealed test set and is only a diagnostic for deciding whether a chemical-head
CPU probe has any reachable target.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from chemvector.bond_object import BOND_TYPES
from tools.train_r17_pair_incidence_known_clean_cpu import Residual, SHA, digest

CACHE = ROOT / os.environ.get(
    "R17_TYPE_REACHABILITY_CACHE",
    "outputs/audits/chemvector-r17-targeted-source-pair-calibration-cpu-20261018-t2/features-cache.pt",
)
STATE = ROOT / os.environ.get(
    "R17_TYPE_REACHABILITY_STATE",
    "outputs/audits/chemvector-r17-targeted-source-pair-incidence-cpu-20261018-t2/sidecar-state.pt",
)
CALIB = ROOT / os.environ.get(
    "R17_TYPE_REACHABILITY_CALIB",
    "outputs/audits/chemvector-r17-targeted-source-pair-calibration-cpu-20261018-t2/report.json",
)
OUT = ROOT / os.environ.get(
    "R17_TYPE_REACHABILITY_OUT",
    "outputs/audits/chemvector-r17-targeted-source-pair-type-reachability-cpu-20261018-t2",
)
BASE_THRESHOLD = 0.72


def _q(values):
    if not values:
        return {"n": 0, "min": None, "p50": None, "p90": None, "max": None, "mean": None}
    x = sorted(float(v) for v in values)
    def at(q):
        return x[min(len(x) - 1, int(q * (len(x) - 1)))]
    return {"n": len(x), "min": x[0], "p50": at(.5), "p90": at(.9), "max": x[-1], "mean": sum(x) / len(x)}


def _finish(bucket):
    typed = bucket["typed"]
    n = bucket["n"]
    return {
        "true_nonbaseline_edges": n,
        "typed_true_nonbaseline_edges": typed,
        "type_top1_correct": bucket["correct"],
        "type_top1_accuracy": bucket["correct"] / max(typed, 1),
        "type_distribution": dict(sorted(bucket["types"].items())),
        "top1_distribution": dict(sorted(bucket["top1"].items())),
        "top1_probability": _q(bucket["prob"]),
        "top1_margin": _q(bucket["margin"]),
        "high_confidence_ge_0.60": sum(v >= .60 for v in bucket["prob"]),
        "high_margin_ge_0.10": sum(v >= .10 for v in bucket["margin"]),
    }


def run():
    if not CACHE.exists() or not STATE.exists() or not CALIB.exists():
        raise FileNotFoundError("cache/state/calibration report missing")
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    state = torch.load(STATE, map_location="cpu", weights_only=False)
    report = json.loads(CALIB.read_text())
    if state.get("initial_checkpoint_sha256") != SHA:
        raise RuntimeError("sidecar was not trained from frozen checkpoint")
    presence = Residual(262)
    presence.load_state_dict(state["state_dict"])
    presence.eval()
    alpha, clip, residual_margin = (float(v) for v in report["profile"]["params"])
    thresholds = {str(k): float(v) for k, v in report["profile"]["thresholds"].items() if v is not None}
    base_logit = math.log(BASE_THRESHOLD / (1.0 - BASE_THRESHOLD))
    by_split = {}
    total_typed = 0
    total_candidate = 0
    total_routed = 0
    for split in ("train", "holdout"):
        rows = cache["train"] if split == "train" else cache["hold"]
        by_profile = {}
        for item in rows:
            p = str(item["profile"])
            threshold = thresholds.get(p, BASE_THRESHOLD)
            threshold_logit = math.log(threshold / (1.0 - threshold))
            base = item["base_logits"].float()
            residual = presence(item["features"].float()).detach()
            adjusted = base + alpha * residual.clamp(-clip, clip)
            base_sel = base >= base_logit
            additions = (~base_sel) & (residual >= residual_margin) & (adjusted >= threshold_logit)
            logits = item["chemical_logits"].float()
            probs = logits.softmax(-1)
            top_prob, top_idx = probs.max(-1)
            top2 = torch.topk(probs, k=2, dim=-1).values
            margin = top2[:, 0] - top2[:, 1]
            b = by_profile.setdefault(p, {"candidate": _new_bucket(), "routed": _new_bucket(), "baseline": _new_bucket()})
            for i in torch.where(item["labels"] > .5)[0].tolist():
                label = item["type_labels"][i]
                if label not in BOND_TYPES:
                    continue
                is_base = bool(base_sel[i])
                is_routed = bool(additions[i])
                target = b["baseline"] if is_base else b["candidate"]
                _add(target, label, int(top_idx[i]), float(top_prob[i]), float(margin[i]))
                if is_routed:
                    _add(b["routed"], label, int(top_idx[i]), float(top_prob[i]), float(margin[i]))
        by_split[split] = {p: {k: _finish(v) for k, v in buckets.items()} for p, buckets in sorted(by_profile.items())}
        total_candidate += sum(v["candidate"]["true_nonbaseline_edges"] for v in by_split[split].values())
        total_routed += sum(v["routed"]["true_nonbaseline_edges"] for v in by_split[split].values())
        total_typed += sum(v["candidate"]["typed_true_nonbaseline_edges"] for v in by_split[split].values())

    # A type-head-only probe cannot create a pair that the frozen presence
    # route never admits.  It is only plausible when there are routed true
    # additions with materially wrong/low-margin frozen type predictions.
    routed_rows = [v["routed"] for s in by_split.values() for v in s.values()]
    routed_typed = sum(v["typed_true_nonbaseline_edges"] for v in routed_rows)
    routed_correct = sum(v["type_top1_correct"] for v in routed_rows)
    candidate_rows = [v["candidate"] for s in by_split.values() for v in s.values()]
    candidate_typed = sum(v["typed_true_nonbaseline_edges"] for v in candidate_rows)
    candidate_correct = sum(v["type_top1_correct"] for v in candidate_rows)
    verdict = {
        "routed_true_additions": total_routed,
        "routed_type_top1_accuracy": routed_correct / max(routed_typed, 1),
        "candidate_true_nonbaseline_edges": total_candidate,
        "candidate_type_top1_accuracy": candidate_correct / max(candidate_typed, 1),
        # A type-only head can repair labels only for already-routed pairs;
        # it cannot recover the many true pairs rejected by presence routing.
        "standalone_type_head_only_probe_worthwhile": False,
        "oracle_candidate_union_type_capacity_exists": bool(total_candidate > 0 and candidate_correct < candidate_typed),
        "routed_type_error_count": routed_typed - routed_correct,
        "reason": (
            "Candidate-union analysis shows type errors, but standalone type-only training cannot recover true pairs rejected by frozen presence routing."
            if total_candidate > 0
            else "No nonbaseline typed candidates were found."
        ),
    }
    result = {
        "schema_version": "chemvector-r17-type-reachability-cpu-0.1",
        "complete": True,
        "read_only": True,
        "gpu_used": False,
        "sealed_test_read": False,
        "production_checkpoint_written": False,
        "cache": str(CACHE.relative_to(ROOT)),
        "cache_sha256": digest(CACHE),
        "state": str(STATE.relative_to(ROOT)),
        "state_sha256": digest(STATE),
        "calibration": str(CALIB.relative_to(ROOT)),
        "calibration_sha256": digest(CALIB),
        "base_threshold": BASE_THRESHOLD,
        "profile_params": [alpha, clip, residual_margin],
        "profile_thresholds": thresholds,
        "splits": by_split,
        "verdict": verdict,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(report_path), "verdict": verdict}, indent=2))


def _new_bucket():
    return {"n": 0, "typed": 0, "correct": 0, "types": Counter(), "top1": Counter(), "prob": [], "margin": []}


def _add(bucket, label, top_idx, prob, margin):
    bucket["n"] += 1
    bucket["typed"] += 1
    bucket["correct"] += int(BOND_TYPES[top_idx] == label)
    bucket["types"][label] += 1
    bucket["top1"][BOND_TYPES[top_idx]] += 1
    bucket["prob"].append(prob)
    bucket["margin"].append(margin)


if __name__ == "__main__":
    run()
