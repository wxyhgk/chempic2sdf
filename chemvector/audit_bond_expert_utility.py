#!/usr/bin/env python3
"""Compare two BondObject audits and emit per-image expert-utility labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    ("topology_exact", "predicted"),
    ("chemical_exact", "predicted"),
    ("compiled_sdf_exact", None),
)


def load_records(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    records = {str(row["key"]): row for row in rows}
    if len(records) != len(rows):
        raise ValueError(f"Duplicate keys in {path}")
    return records


def metric_value(record: dict, metric: str, section: str | None) -> bool:
    source = record if section is None else record[section]
    return bool(source[metric])


def compare_records(base: dict[str, dict], expert: dict[str, dict]) -> tuple[dict, list[dict]]:
    if set(base) != set(expert):
        raise ValueError("Base and expert audit keys differ")
    labels = []
    summary = {"samples": len(base), "metrics": {}}
    for metric, section in METRICS:
        base_exact = {
            key for key, row in base.items() if metric_value(row, metric, section)
        }
        expert_exact = {
            key for key, row in expert.items() if metric_value(row, metric, section)
        }
        summary["metrics"][metric] = {
            "base_exact": len(base_exact),
            "expert_exact": len(expert_exact),
            "oracle_union_exact": len(base_exact | expert_exact),
            "intersection_exact": len(base_exact & expert_exact),
            "expert_rescue": len(expert_exact - base_exact),
            "expert_break": len(base_exact - expert_exact),
            "base_exact_rate": len(base_exact) / max(len(base), 1),
            "expert_exact_rate": len(expert_exact) / max(len(base), 1),
            "oracle_union_exact_rate": len(base_exact | expert_exact)
            / max(len(base), 1),
        }
    for key in sorted(base):
        row = {"key": key}
        for metric, section in METRICS:
            base_value = metric_value(base[key], metric, section)
            expert_value = metric_value(expert[key], metric, section)
            row[f"base_{metric}"] = base_value
            row[f"expert_{metric}"] = expert_value
            row[f"{metric}_utility"] = (
                "expert_rescue"
                if expert_value and not base_value
                else "expert_break"
                if base_value and not expert_value
                else "both_exact"
                if base_value
                else "both_wrong"
            )
        labels.append(row)
    return summary, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-records", required=True)
    parser.add_argument("--expert-records", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base_path = Path(args.base_records)
    expert_path = Path(args.expert_records)
    summary, labels = compare_records(
        load_records(base_path), load_records(expert_path)
    )
    summary["protocol"] = {
        "base_records": str(base_path),
        "expert_records": str(expert_path),
        "uses_uspto_for_training": False,
        "uses_mr_tadf_for_training": False,
        "oracle_uses_ground_truth": True,
        "deployable": False,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "labels.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
