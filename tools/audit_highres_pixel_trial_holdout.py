"""Audit seen/holdout reachability for the high-resolution pixel trial."""

from __future__ import annotations

import json
from collections import defaultdict
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get(
    "CHEMVECTOR_HIGHRES_TRIAL_REPORT",
    str(ROOT / "outputs/audits/chemvector-atom-highres-pixel-trial-20261006-r3/report.json"),
)).resolve()
OUTPUT = Path(os.environ.get(
    "CHEMVECTOR_HIGHRES_HOLDOUT_OUTPUT",
    str(SOURCE.parent / "holdout-audit.json"),
)).resolve()


def _summary(rows: list[dict]) -> dict:
    peaks = sum(int(row["highres_rescue_peak_count"]) for row in rows)
    matched = sum(int(row["highres_hungarian_matched_count"]) for row in rows)
    recovered = sum(len(row["highres_newly_reachable_gap_ids"]) for row in rows)
    targets = sum(int(row["targets"]) for row in rows)
    baseline = sum(int(row["raw_matched"]) for row in rows)
    return {
        "images": len(rows),
        "targets": targets,
        "baseline_raw_matched": baseline,
        "rescue_peaks": peaks,
        "rescue_hungarian_matched": matched,
        "rescue_precision_6px": matched / max(peaks, 1),
        "newly_reachable_gap_targets": recovered,
    }


def _union_summary(rows: list[dict], k: str) -> dict:
    values = [row["union_primary_plus_rescue"][k] for row in rows]
    return {
        "images": len(rows),
        "candidate_count": sum(int(value["candidate_count"]) for value in values),
        "matched_count": sum(int(value["matched_count"]) for value in values),
        "newly_matched_gap_targets": sum(len(value["newly_matched_gap_ids"]) for value in values),
    }


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    report = json.loads(SOURCE.read_text())
    if report["initial_sha256"] != "09d1661198bde9b80dc2a4398a9a6d7d856098a2af1cc4e05425fe5e3c88dbe1":
        raise ValueError("Initial checkpoint contract changed")
    train_keys = {event["sample_key"] for event in report["events"]}
    rows = report["results"]
    if len(rows) != 1200 or len(train_keys) != report["train_images"]:
        raise ValueError("Trial report does not contain the expected 1200-row/120-image split")
    seen = [row for row in rows if row["sample_key"] in train_keys]
    holdout = [row for row in rows if row["sample_key"] not in train_keys]
    if len(seen) != report["train_images"] or len(holdout) != 1200 - report["train_images"]:
        raise ValueError("Seen/holdout partition is not disjoint and complete")
    by_style = defaultdict(lambda: {"seen": [], "holdout": []})
    for row in seen:
        by_style[row["profile_id"]]["seen"].append(row)
    for row in holdout:
        by_style[row["profile_id"]]["holdout"].append(row)
    result = {
        "schema_version": "chemvector-atom-highres-pixel-trial-holdout-audit-0.1",
        "source_report": str(SOURCE.relative_to(ROOT)),
        "training_performed": False,
        "sealed_test_read": False,
        "seen": _summary(seen),
        "holdout": _summary(holdout),
        "union_seen": {k: _union_summary(seen, k) for k in ("8", "16", "32")},
        "union_holdout": {k: _union_summary(holdout, k) for k in ("8", "16", "32")},
        "by_style": {
            style: {split: _summary(values) for split, values in sorted(parts.items())}
            for style, parts in sorted(by_style.items())
        },
        "gate": {
            "holdout_gap_recovery_positive": _summary(holdout)["newly_reachable_gap_targets"] > 0,
            "holdout_rescue_precision_at_least_0_25": _summary(holdout)["rescue_precision_6px"] >= 0.25,
            "union_holdout_k16_gap_recovery_positive": _union_summary(holdout, "16")["newly_matched_gap_targets"] > 0,
            "pass": (
                _summary(holdout)["newly_reachable_gap_targets"] > 0
                and _summary(holdout)["rescue_precision_6px"] >= 0.25
                and _union_summary(holdout, "16")["newly_matched_gap_targets"] > 0
            ),
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["gate"], ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
