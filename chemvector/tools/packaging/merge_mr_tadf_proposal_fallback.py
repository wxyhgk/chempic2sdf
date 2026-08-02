#!/usr/bin/env python3
"""Build an invalid-only Bond proposal fallback MR-TADF benchmark run."""

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


def select_record(primary: dict, fallback: dict) -> tuple[dict, str]:
    if primary.get("sdf_valid"):
        selected = dict(primary)
        source = "single_scale_primary"
    else:
        selected = dict(fallback)
        source = "multiscale_invalid_fallback"
        selected["decode_path"] = f"multiscale_{selected.get('decode_path', 'invalid')}"
        selected["seconds"] = float(primary.get("seconds", 0.0)) + float(
            fallback.get("seconds", 0.0)
        )
    selected["proposal_decode_path"] = source
    return selected, source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-run", required=True)
    parser.add_argument("--fallback-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-samples", type=int, default=1100)
    args = parser.parse_args()
    primary_run = Path(args.primary_run)
    fallback_run = Path(args.fallback_run)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {output}")
    for name in ARTIFACT_DIRS:
        (output / name).mkdir(parents=True, exist_ok=True)
    primary_records = load_records(primary_run)
    fallback_records = load_records(fallback_run)
    if set(primary_records) != set(fallback_records):
        raise ValueError("Primary and fallback benchmark molecule ids differ")
    if len(primary_records) != args.expected_samples:
        raise ValueError("Benchmark sample count differs from expected")
    records = []
    for molecule_id in sorted(primary_records):
        selected, source_name = select_record(
            primary_records[molecule_id], fallback_records[molecule_id]
        )
        source = primary_run if source_name == "single_scale_primary" else fallback_run
        for name in ("input", "expected_sdf", "expected_render"):
            candidates = list((primary_run / name).glob(f"{molecule_id}.*"))
            if len(candidates) != 1:
                raise ValueError(f"Expected one {name} artifact for {molecule_id}")
            shutil.copy2(candidates[0], output / name / candidates[0].name)
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
    summary.update(
        {
            "protocol": "v0.9.65-single-scale-primary-multiscale-invalid-fallback",
            "primary_run": str(primary_run),
            "fallback_run": str(fallback_run),
            "oracle_atom_chemistry": True,
            "proposal_decode_paths": {
                name: sum(record["proposal_decode_path"] == name for record in records)
                for name in ("single_scale_primary", "multiscale_invalid_fallback")
            },
        }
    )
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    primary_config = json.loads((primary_run / "benchmark_config.json").read_text())
    fallback_config = json.loads((fallback_run / "benchmark_config.json").read_text())
    config = {
        **primary_config,
        "model_version": "v0.9.65",
        "inference_protocol": "single-scale primary; multiscale Bond proposal on invalid only",
        "fallback_proposal_bond_checkpoint": fallback_config["bond_checkpoint"],
        "primary_run": str(primary_run),
        "fallback_run": str(fallback_run),
        "samples": len(records),
    }
    config.pop("merged_segments", None)
    ensure_benchmark_config(output / "benchmark_config.json", config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
