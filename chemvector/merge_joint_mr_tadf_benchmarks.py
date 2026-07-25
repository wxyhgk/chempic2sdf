#!/usr/bin/env python3
"""Merge completed MR-TADF joint benchmark segments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from chemvector.benchmark_mr_tadf import ensure_benchmark_config, sha256_file
from chemvector.benchmark_joint_mr_tadf import _summary


ARTIFACT_DIRS = (
    "input",
    "expected_sdf",
    "expected_render",
    "scene",
    "sdf",
    "render",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-samples", type=int, default=1100)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    for name in ARTIFACT_DIRS:
        (output / name).mkdir(parents=True, exist_ok=True)
    records = []
    seen = set()
    segment_summaries = []
    shared_config = None
    for value in args.segments:
        segment = Path(value)
        segment_summary = json.loads((segment / "summary.json").read_text())
        segment_summaries.append(segment_summary)
        config_path = segment / "benchmark_config.json"
        if config_path.is_file():
            segment_config = json.loads(config_path.read_text())
            core_config = {
                key: item
                for key, item in segment_config.items()
                if key not in {"start_index", "samples"}
            }
            if shared_config is None:
                shared_config = core_config
            elif core_config != shared_config:
                raise ValueError("Segment benchmark configurations differ")
        for line in (segment / "records.jsonl").read_text().splitlines():
            record = json.loads(line)
            key = record["molecule_id"]
            if key in seen:
                raise ValueError(f"Duplicate molecule across segments: {key}")
            seen.add(key)
            records.append(record)
        for name in ARTIFACT_DIRS:
            source = segment / name
            for path in source.iterdir():
                target = output / name / path.name
                if target.exists():
                    raise ValueError(f"Duplicate artifact across segments: {target}")
                shutil.copy2(path, target)
    if len(records) != args.expected_samples:
        raise ValueError(
            f"Merged samples {len(records)} differ from expected {args.expected_samples}"
        )
    records.sort(key=lambda item: item["molecule_id"])
    (output / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )
    summary = _summary(records)
    summary.update(
        {
            "protocol": "v0.9.63-joint-object-oracle-atom-chemistry",
            "merged_segments": [str(Path(value)) for value in args.segments],
            "segment_summaries": segment_summaries,
            "oracle_atom_chemistry": True,
        }
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    manifest = Path("Data/benchmarks/mr_tadf_v1/manifest.csv")
    primary_checkpoint = Path(
        "outputs/checkpoints/chemvector-v0962-representable-hard-replay-step1500/best.pt"
    )
    fallback_checkpoint = Path(
        "outputs/checkpoints/chemvector-v0949-cached-proposal-object-set-train8974-step3000/best.pt"
    )
    merged_config = shared_config or {
        "benchmark_manifest": str(manifest),
        "benchmark_manifest_sha256": sha256_file(manifest),
        "benchmark_type": "MR-TADF v1 main structure benchmark",
        "model_version": "v0.9.63",
        "inference_protocol": "joint object MILP with Oracle atom chemistry",
        "checkpoint": str(primary_checkpoint),
        "checkpoint_sha256": sha256_file(primary_checkpoint),
        "fallback_checkpoint": str(fallback_checkpoint),
        "fallback_checkpoint_sha256": sha256_file(fallback_checkpoint),
        "maximum_atoms": 128,
        "maximum_bonds": 192,
        "atom_threshold": -0.25,
        "bond_threshold": -0.5,
        "endpoint_topk": 2,
        "incidence_weight": 0.5,
        "oracle_atom_chemistry": True,
        "primary_comparison": "SDF attributed graph isomorphism",
    }
    merged_config.update(
        {
            "samples": len(records),
            "merged_segments": [str(Path(value)) for value in args.segments],
        }
    )
    ensure_benchmark_config(
        output / "benchmark_config.json",
        merged_config,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
