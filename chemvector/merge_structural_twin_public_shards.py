#!/usr/bin/env python3
"""Merge disjoint fixed-order v0.10.16 public regression shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chemvector.audit_structural_twin_end_to_end import _write_json
from chemvector.benchmark_structural_twin_public import _summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-samples", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Merged public output is not empty: {output}")
    configs = []
    records = []
    for value in args.shards:
        root = Path(value)
        report = json.loads((root / "report.json").read_text(encoding="utf-8"))
        configs.append(report["config"])
        records.extend(
            json.loads(line)
            for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    configs.sort(key=lambda row: row["start"])
    expected_start = 0
    manifest_hashes = {row["manifest_sha256"] for row in configs}
    for config in configs:
        if config["start"] != expected_start:
            raise ValueError("Public shard ranges are not contiguous from zero")
        if not config.get("conditional_baseline"):
            raise ValueError("Public shard does not use frozen conditional baseline")
        expected_start += config["samples"]
    if len(manifest_hashes) != 1:
        raise ValueError("Public shards reference different manifests")
    if expected_start != args.expected_samples or len(records) != args.expected_samples:
        raise ValueError("Merged public sample coverage differs")
    if len({row["key"] for row in records}) != len(records):
        raise ValueError("Merged public records contain duplicate keys")
    records.sort(key=lambda row: row["key"])
    summary = _summarize(records)
    gate = {
        "passed": summary["sdf_valid_breaks"] == 0
        and summary["nonstereo_exact_breaks"] == 0
        and summary["chemical_exact_breaks"] == 0
        and summary["no_action_ocr_breaks"] == 0
        and summary["no_action_graph_breaks"] == 0
        and summary["candidate_sdf_valid"] >= summary["baseline_sdf_valid"]
        and summary["candidate_nonstereo_exact"]
        >= summary["baseline_nonstereo_exact"]
        and summary["candidate_chemical_exact"]
        >= summary["baseline_chemical_exact"],
        "require_action": True,
        "scope": "unseen public USPTO clean 1k fixed-order regression",
    }
    merged_config = {
        "schema_version": "structural-twin-public-ab-merged-0.1",
        "manifest_sha256": next(iter(manifest_hashes)),
        "samples": args.expected_samples,
        "shard_ranges": [
            [row["start"], row["start"] + row["samples"]] for row in configs
        ],
        "conditional_baseline": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "config.json", merged_config)
    (output / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    report = {"config": merged_config, "summary": summary, "gate": gate}
    _write_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
