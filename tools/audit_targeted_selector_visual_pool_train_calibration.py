"""CPU-only, train-calibrated threshold diagnostic for the failed shadow.

The shadow is terminal and is not retrained.  This audit derives exactly one
threshold from train decisions (maximize train strict recall subject to a
predeclared 0.85 conservative precision) and evaluates that threshold once on
the fixed holdout.  It does not read sealed data or write a checkpoint.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHADOW = ROOT / "outputs/audits/chemvector-targeted-selector-visual-pool-shadow-20261006-r1/report.json"
OUTPUT = ROOT / "outputs/audits/chemvector-targeted-selector-visual-pool-train-calibration-20261006-r1/report.json"
TARGET_PRECISION = 0.85


def stats(rows: list[dict], threshold: float) -> dict:
    selected = [r for r in rows if r.get("existence_logit", float("-inf")) >= threshold]
    recovered = sum(r["status"] == "strict_positive" for r in selected)
    false = sum(r["status"] == "confirmed_negative" for r in selected)
    unknown = sum(r["status"] == "strict_positive" and r.get("winner_slot") is None for r in selected)
    available = sum(r["status"] == "strict_positive" for r in rows)
    return {"threshold": threshold, "selected": len(selected), "strict_recovered": recovered,
            "confirmed_false_selected": false, "unknown_selected": unknown,
            "conservative_precision": recovered / max(len(selected), 1),
            "strict_recall": recovered / max(available, 1)}


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    shadow = json.loads(SHADOW.read_text(encoding="utf-8"))
    if shadow.get("sealed_test_read") is not False or shadow.get("production_checkpoint_written"):
        raise ValueError("Shadow report is not safe for diagnostic calibration")
    checkpoints = shadow.get("checkpoints", [])
    if not checkpoints or checkpoints[-1].get("step") != 100:
        raise ValueError("Expected completed terminal 100-step shadow")
    final = checkpoints[-1]
    train = [r for r in final["train"]["decisions"] if r["status"] in {"strict_positive", "confirmed_negative"} and "existence_logit" in r]
    holdout = [r for r in final["holdout"]["decisions"] if r["status"] in {"strict_positive", "confirmed_negative"} and "existence_logit" in r]
    # One deterministic train-only calibration rule.  Candidate thresholds are
    # the observed train scores; ties use the highest threshold with maximal
    # recall, making the rule fully reproducible and holdout-blind.
    # Sweep once in descending score order (O(n log n), not a quadratic
    # recomputation for every observed threshold).
    ordered = sorted(train, key=lambda r: float(r["existence_logit"]), reverse=True)
    available = sum(r["status"] == "strict_positive" for r in train)
    feasible = []
    recovered = false = 0
    index = 0
    while index < len(ordered):
        threshold = float(ordered[index]["existence_logit"])
        while index < len(ordered) and float(ordered[index]["existence_logit"]) == threshold:
            if ordered[index]["status"] == "strict_positive":
                recovered += 1
            else:
                false += 1
            index += 1
        selected = recovered + false
        precision = recovered / max(selected, 1)
        if precision >= TARGET_PRECISION:
            feasible.append({"threshold": threshold, "selected": selected,
                             "strict_recovered": recovered,
                             "confirmed_false_selected": false,
                             "unknown_selected": 0,
                             "conservative_precision": precision,
                             "strict_recall": recovered / max(available, 1)})
    chosen = max(feasible, key=lambda item: (item["strict_recall"], -item["threshold"])) if feasible else None
    holdout_result = stats(holdout, chosen["threshold"]) if chosen is not None else None
    result = {
        "schema_version": "chemvector-targeted-selector-visual-pool-train-calibration-0.1",
        "complete": True, "read_only": True, "training_performed": False,
        "model_loaded": False, "gpu_used": False, "optimizer_steps": 0,
        "sealed_test_read": False, "shadow_terminal_no_retry": True,
        "source_shadow_sha256": __import__("hashlib").sha256(SHADOW.read_bytes()).hexdigest(),
        "calibration": {"target_conservative_precision": TARGET_PRECISION,
                         "rule": "maximize_train_strict_recall_subject_to_train_precision_target",
                         "train_groups_used": len(train), "chosen": chosen,
                         "feasible_thresholds": len(feasible)},
        "holdout": holdout_result,
        "gate": {"train_target_met": chosen is not None,
                 "holdout_target_met": bool(holdout_result and holdout_result["conservative_precision"] >= TARGET_PRECISION),
                 "holdout_evaluated_once": holdout_result is not None,
                 "next": "redesign the existence objective; calibration cannot rescue this head"},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run()
