#!/usr/bin/env python3
"""Summarize predicted-vertex count errors and multiple-bond false positives."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from chemvector.experiment_loop.state import atomic_write_json


def summarize_records(records: list[dict]) -> dict:
    count_errors = Counter(
        int(row["predicted_vertices"]) - int(row["expected_vertices"])
        for row in records
    )
    unmatched = [item for row in records for item in row.get("unmatched_vertices", [])]
    within_bond_3px = [
        item
        for item in unmatched
        if item.get("nearest_bond_distance_pixels") is not None
        and float(item["nearest_bond_distance_pixels"]) <= 3.0
    ]
    near_multiple = [item for item in unmatched if item.get("near_multiple_bond")]
    missing = [item for row in records for item in row.get("missing_vertices", [])]
    missing_with_candidate = [
        item for item in missing if item.get("candidate_available") is True
    ]
    missing_without_candidate = [
        item for item in missing if item.get("candidate_available") is False
    ]
    samples = len(records)
    under = sum(count for error, count in count_errors.items() if error < 0)
    over = sum(count for error, count in count_errors.items() if error > 0)
    exact = count_errors.get(0, 0)
    return {
        "samples": samples,
        "count": {
            "exact": exact / max(samples, 1),
            "under_rate": under / max(samples, 1),
            "over_rate": over / max(samples, 1),
            "distribution": {
                str(error): count for error, count in sorted(count_errors.items())
            },
        },
        "unmatched_vertices": {
            "count": len(unmatched),
            "within_bond_3px_count": len(within_bond_3px),
            "images_with_within_bond_3px": sum(
                any(
                    item.get("nearest_bond_distance_pixels") is not None
                    and float(item["nearest_bond_distance_pixels"]) <= 3.0
                    for item in row.get("unmatched_vertices", [])
                )
                for row in records
            ),
            "near_multiple_bond_count": len(near_multiple),
            "near_multiple_bond_rate": len(near_multiple) / max(len(unmatched), 1),
            "images_with_near_multiple_bond": sum(
                any(item.get("near_multiple_bond") for item in row.get("unmatched_vertices", []))
                for row in records
            ),
            "nearest_multiple_bond_types": dict(
                sorted(
                    Counter(
                        item.get("nearest_bond_display_type") for item in near_multiple
                    ).items()
                )
            ),
        },
        "missing_vertices": {
            "count": len(missing),
            "visible_text": sum(bool(item.get("visible_text")) for item in missing),
            "implicit": sum(not bool(item.get("visible_text")) for item in missing),
            "with_candidate": len(missing_with_candidate),
            "without_candidate": len(missing_without_candidate),
            "coverage_unknown": len(missing)
            - len(missing_with_candidate)
            - len(missing_without_candidate),
            "images_with_missing_with_candidate": sum(
                any(
                    item.get("candidate_available") is True
                    for item in row.get("missing_vertices", [])
                )
                for row in records
            ),
            "images_with_missing_without_candidate": sum(
                any(
                    item.get("candidate_available") is False
                    for item in row.get("missing_vertices", [])
                )
                for row in records
            ),
        },
        "recommended_hypothesis": (
            "global_count_and_set_selection"
            if under > over
            else "duplicate_suppression_and_set_selection"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--next-task")
    args = parser.parse_args()
    records = [
        json.loads(line)
        for line in Path(args.records).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    summary = summarize_records(records)
    atomic_write_json(Path(args.output), summary)
    if args.next_task:
        atomic_write_json(
            Path(args.next_task),
            {
                "action": "implement_hypothesis",
                "hypothesis": summary["recommended_hypothesis"],
                "evidence": {
                    "count_exact": summary["count"]["exact"],
                    "count_under_rate": summary["count"]["under_rate"],
                    "near_multiple_bond_false_positive_rate": summary[
                        "unmatched_vertices"
                    ]["near_multiple_bond_rate"],
                },
                "stable_baseline_unchanged": "v0.7.10",
            },
        )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
