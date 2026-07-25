#!/usr/bin/env python3
"""Compare detector-score and selector-score Junction projection evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BOOLEAN_METRICS = (
    "junction_count_match",
    "junction_exact_6px",
    "bond_count_match",
    "topology_exact",
    "sdf_valid",
    "nonstereo_structure_match",
    "chemical_match",
)
SUMMARY_METRICS = (
    "junction_count_accuracy",
    "junction_exact_6px",
    "bond_count_accuracy",
    "topology_exact_rate",
    "edge_precision",
    "edge_recall",
    "edge_f1",
    "sdf_valid_rate",
    "nonstereo_structure_match_rate",
    "chemical_match_rate",
)


def load_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def paired_flips(detector: list[dict], selector: list[dict]) -> dict:
    detector_by_key = {row["key"]: row for row in detector}
    selector_by_key = {row["key"]: row for row in selector}
    if list(detector_by_key) != list(selector_by_key):
        raise ValueError("Projection A/B record keys or order differ")
    flips = {}
    for metric in BOOLEAN_METRICS:
        improved = []
        regressed = []
        for key, detector_row in detector_by_key.items():
            selector_row = selector_by_key[key]
            before = bool(detector_row[metric])
            after = bool(selector_row[metric])
            if not before and after:
                improved.append(key)
            elif before and not after:
                regressed.append(key)
        flips[metric] = {
            "zero_to_one": len(improved),
            "one_to_zero": len(regressed),
            "zero_to_one_keys": improved,
            "one_to_zero_keys": regressed,
        }
    return flips


def edge_totals(records: list[dict]) -> dict:
    return {
        "true_positives": sum(row["topology_true_positives"] for row in records),
        "false_positives": sum(row["topology_false_positives"] for row in records),
        "false_negatives": sum(row["topology_false_negatives"] for row in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector-dir", required=True)
    parser.add_argument("--selector-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-topology-delta", type=float, default=0.03)
    parser.add_argument(
        "--maximum-edge-recall-regression", type=float, default=0.005
    )
    args = parser.parse_args()

    detector_dir = Path(args.detector_dir)
    selector_dir = Path(args.selector_dir)
    detector_summary = json.loads(
        (detector_dir / "summary.json").read_text(encoding="utf-8")
    )
    selector_summary = json.loads(
        (selector_dir / "summary.json").read_text(encoding="utf-8")
    )
    if detector_summary["sample_keys_sha256"] != selector_summary["sample_keys_sha256"]:
        raise ValueError("Projection A/B sample hashes differ")
    detector_records = load_records(detector_dir / "records.jsonl")
    selector_records = load_records(selector_dir / "records.jsonl")
    if len(detector_records) != len(selector_records):
        raise ValueError("Projection A/B sample counts differ")

    deltas = {
        metric: float(selector_summary[metric]) - float(detector_summary[metric])
        for metric in SUMMARY_METRICS
    }
    topology_go = deltas["topology_exact_rate"] >= args.minimum_topology_delta
    recall_safe = (
        deltas["edge_recall"] >= -args.maximum_edge_recall_regression
    )
    result = {
        "schema_version": "junction-projection-ab-0.1",
        "samples": len(detector_records),
        "sample_keys_sha256": detector_summary["sample_keys_sha256"],
        "detector_dir": str(detector_dir),
        "selector_dir": str(selector_dir),
        "detector": {metric: detector_summary[metric] for metric in SUMMARY_METRICS},
        "selector": {metric: selector_summary[metric] for metric in SUMMARY_METRICS},
        "selector_minus_detector": deltas,
        "paired_flips": paired_flips(detector_records, selector_records),
        "edge_totals": {
            "detector": edge_totals(detector_records),
            "selector": edge_totals(selector_records),
        },
        "gate": {
            "minimum_topology_delta": args.minimum_topology_delta,
            "maximum_edge_recall_regression": args.maximum_edge_recall_regression,
            "topology_go": topology_go,
            "edge_recall_safe": recall_safe,
            "decision": "GO" if topology_go and recall_safe else "NO-GO",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
