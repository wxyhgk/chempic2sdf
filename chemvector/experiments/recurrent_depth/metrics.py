"""Geometric atom-set scoring; never call it molecular graph correctness."""

import numpy as np

from .data import assign


def score_records(records, logits):
    details = []
    for row, prediction in zip(records, logits, strict=True):
        selected = (prediction >= 0) & row["mask"].numpy()
        matched = len(assign(row["anchors"].numpy(), selected, row["targets"].numpy()))
        count = len(row["targets"])
        fp, fn = int(selected.sum()) - matched, count - matched
        oracle = len(assign(row["anchors"].numpy(), row["mask"].numpy(), row["targets"].numpy()))
        details.append({"key": row["key"], "tp": matched, "fp": fp, "fn": fn, "targets": count,
                        "exact": fp == 0 and fn == 0, "candidate_oracle": oracle})
    tp, fp, fn = (sum(r[k] for r in details) for k in ("tp", "fp", "fn"))
    return {"images": len(details), "exact": float(np.mean([r["exact"] for r in details])),
            "tp": tp, "fp": fp, "fn": fn, "fp_per_image": fp / len(details),
            "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
            "candidate_oracle_recall": sum(r["candidate_oracle"] for r in details) / max(tp + fn, 1)}, details


def decide(trials, source, gate):
    arms = {}
    for name in ("tied1", "tied4", "untied4"):
        selected = [r["metrics"] for r in trials if r["arm"] == name and r["evaluation_loops"] == r["trained_loops"]]
        arms[name] = {key: float(np.mean([r[key] for r in selected])) for key in ("exact", "fp_per_image", "recall")}
    by_seed = {(r["seed"], r["arm"]): r["metrics"] for r in trials if r["evaluation_loops"] == r["trained_loops"]}
    positive = sum(by_seed[s, "tied4"]["exact"] > by_seed[s, "tied1"]["exact"] for s, a in by_seed if a == "tied4")
    checks = {
        "same_parameter_gain": arms["tied4"]["exact"] - arms["tied1"]["exact"] >= gate["min_exact_gain_vs_tied1"],
        "same_unrolled_compute_control": arms["tied4"]["exact"] >= arms["untied4"]["exact"] - gate["max_exact_drop_vs_untied4"],
        "false_positive_budget": arms["tied4"]["fp_per_image"] <= arms["tied1"]["fp_per_image"] + gate["max_fp_increase_per_image"],
        "seed_consistency": positive >= gate["min_positive_seeds"],
        "frozen_source_guard": arms["tied4"]["exact"] >= source["exact"] - gate["max_exact_drop_vs_source"],
    }
    return {"means": arms, "checks": checks, "passed": all(checks.values()),
            "meaning": "candidate_selection_pilot_only_not_graph_or_architecture_proof", "promotion_authorized": False}
