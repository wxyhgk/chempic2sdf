#!/usr/bin/env python3
"""Merge AtomPair-group primary predictions with valid baseline fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from chemvector.benchmark_joint_mr_tadf import _summary
from chemvector.benchmark_mr_tadf import ensure_benchmark_config


ARTIFACT_DIRS = ("input", "expected_sdf", "expected_render", "scene", "sdf", "render")


def load_records(run: Path) -> dict[str, dict]:
    return {
        record["molecule_id"]: record
        for line in (run / "records.jsonl").read_text().splitlines()
        if (record := json.loads(line))
    }


def select_record(group: dict, baseline: dict) -> tuple[dict, str]:
    if group.get("sdf_valid") or not baseline.get("sdf_valid"):
        selected = dict(group)
        source = "atom_pair_group_primary"
    else:
        selected = dict(baseline)
        source = "baseline_invalid_fallback"
    selected["atom_pair_decode_path"] = source
    selected["seconds"] = float(group.get("seconds", 0.0)) + (
        float(baseline.get("seconds", 0.0))
        if source == "baseline_invalid_fallback"
        else 0.0
    )
    return selected, source


def copy_single(source_dir: Path, target_dir: Path, molecule_id: str) -> None:
    candidates = list(source_dir.glob(f"{molecule_id}.*"))
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one artifact in {source_dir} for {molecule_id}, found {len(candidates)}"
        )
    shutil.copy2(candidates[0], target_dir / candidates[0].name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-run", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-samples", type=int, default=1100)
    parser.add_argument("--model-version", default="v0.9.83")
    parser.add_argument(
        "--protocol",
        default="AtomPair group primary; baseline fallback only when primary SDF is invalid",
    )
    args = parser.parse_args()
    group_run = Path(args.group_run)
    baseline_run = Path(args.baseline_run)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    for name in ARTIFACT_DIRS:
        (output / name).mkdir(parents=True, exist_ok=True)
    group_records = load_records(group_run)
    baseline_records = load_records(baseline_run)
    if set(group_records) != set(baseline_records):
        raise ValueError("Group and baseline benchmark molecule ids differ")
    if len(group_records) != args.expected_samples:
        raise ValueError("Benchmark sample count differs from expected")

    records = []
    for molecule_id in sorted(group_records):
        selected, source_name = select_record(
            group_records[molecule_id], baseline_records[molecule_id]
        )
        source = group_run if source_name == "atom_pair_group_primary" else baseline_run
        for name in ("input", "expected_sdf", "expected_render"):
            copy_single(group_run / name, output / name, molecule_id)
        scene = source / "scene" / f"{molecule_id}.json"
        if scene.is_file():
            shutil.copy2(scene, output / "scene" / scene.name)
        source_sdf = source / selected["prediction_sdf_file"]
        target_sdf = output / "sdf" / source_sdf.name
        shutil.copy2(source_sdf, target_sdf)
        selected["prediction_sdf_file"] = str(target_sdf.relative_to(output))
        source_render = source / selected["prediction_render_file"]
        target_render = output / "render" / f"{molecule_id}.png"
        shutil.copy2(source_render, target_render)
        selected["prediction_render_file"] = str(target_render.relative_to(output))
        records.append(selected)
    (output / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )
    summary = _summary(records)
    paths = {
        name: sum(record["atom_pair_decode_path"] == name for record in records)
        for name in ("atom_pair_group_primary", "baseline_invalid_fallback")
    }
    summary.update(
        {
            "protocol": args.protocol,
            "group_run": str(group_run),
            "baseline_run": str(baseline_run),
            "oracle_atom_chemistry": True,
            "atom_pair_decode_paths": paths,
        }
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    group_config = json.loads((group_run / "benchmark_config.json").read_text())
    baseline_config = json.loads((baseline_run / "benchmark_config.json").read_text())
    config = {
        **group_config,
        "model_version": args.model_version,
        "inference_protocol": args.protocol,
        "baseline_checkpoint": baseline_config["checkpoint"],
        "group_run": str(group_run),
        "baseline_run": str(baseline_run),
        "samples": len(records),
    }
    ensure_benchmark_config(output / "benchmark_config.json", config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
