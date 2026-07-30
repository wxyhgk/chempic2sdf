#!/usr/bin/env python3
"""Benchmark v0.10.16 against v0.10.10 on fixed public USPTO images."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

from rdkit import Chem, RDLogger

from chemvector.audit_structural_twin_end_to_end import (
    _command,
    _metrics,
    _read_mol,
    _run,
    _sha256_file,
    _write_json,
)
from chemvector.benchmark_public_ocsr import sdf_graph_match


DEFAULT_MANIFEST = (
    "Data/benchmarks/public_ocsr_v1/prepared/uspto_clean_1k/manifest.csv"
)
DEFAULT_OUTPUT = "outputs/evaluations/uspto-clean-100-v01016-structural-twin-ab"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--conditional-baseline", action="store_true")
    return parser.parse_args()


def _summarize(records: list[dict]) -> dict:
    action = [row for row in records if row["candidate"]["suppression"]["executed"]]
    no_action = [row for row in records if not row["candidate"]["suppression"]["executed"]]
    return {
        "samples": len(records),
        "actions_executed": len(action),
        "action_keys": [row["key"] for row in action],
        "baseline_inference_runs": sum(
            row["baseline_source"] == "executed_v01010" for row in records
        ),
        "baseline_no_action_identities": sum(
            row["baseline_source"] == "candidate_no_action_identity"
            for row in records
        ),
        "baseline_sdf_valid": sum(row["baseline"]["sdf_valid"] for row in records),
        "candidate_sdf_valid": sum(row["candidate"]["sdf_valid"] for row in records),
        "sdf_valid_rescues": sum(
            not row["baseline"]["sdf_valid"] and row["candidate"]["sdf_valid"]
            for row in records
        ),
        "sdf_valid_breaks": sum(
            row["baseline"]["sdf_valid"] and not row["candidate"]["sdf_valid"]
            for row in records
        ),
        "baseline_nonstereo_exact": sum(
            row["baseline"]["nonstereo_exact"] for row in records
        ),
        "candidate_nonstereo_exact": sum(
            row["candidate"]["nonstereo_exact"] for row in records
        ),
        "nonstereo_exact_rescues": sum(
            not row["baseline"]["nonstereo_exact"]
            and row["candidate"]["nonstereo_exact"]
            for row in records
        ),
        "nonstereo_exact_breaks": sum(
            row["baseline"]["nonstereo_exact"]
            and not row["candidate"]["nonstereo_exact"]
            for row in records
        ),
        "baseline_chemical_exact": sum(
            row["baseline"]["chemical_exact"] for row in records
        ),
        "candidate_chemical_exact": sum(
            row["candidate"]["chemical_exact"] for row in records
        ),
        "chemical_exact_rescues": sum(
            not row["baseline"]["chemical_exact"]
            and row["candidate"]["chemical_exact"]
            for row in records
        ),
        "chemical_exact_breaks": sum(
            row["baseline"]["chemical_exact"]
            and not row["candidate"]["chemical_exact"]
            for row in records
        ),
        "no_action_samples": len(no_action),
        "no_action_ocr_breaks": sum(
            row["baseline"]["ocr_labels"] != row["candidate"]["ocr_labels"]
            for row in no_action
        ),
        "no_action_graph_breaks": sum(
            not row["baseline_candidate_graph_equal"] for row in no_action
        ),
    }


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.start < 0:
        raise ValueError("samples must be positive and start must be non-negative")
    manifest = Path(args.manifest)
    root = manifest.parent
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "ready"]
    rows = rows[args.start : args.start + args.samples]
    if len(rows) != args.samples:
        raise ValueError("Public manifest contains fewer ready rows than requested")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Public benchmark output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": "structural-twin-public-ab-0.1",
        "model_tag": "v0.10.16",
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "selection": (
            f"ready rows [{args.start}, {args.start + args.samples}) "
            "in official manifest order"
        ),
        "start": args.start,
        "samples": args.samples,
        "device": args.device,
        "conditional_baseline": args.conditional_baseline,
        "conditional_baseline_evidence": (
            "USPTO clean fixed first100: 97 no-action outputs had zero OCR/graph differences"
            if args.conditional_baseline
            else None
        ),
    }
    _write_json(output / "config.json", config)
    records = []
    RDLogger.DisableLog("rdApp.*")
    for index, row in enumerate(rows):
        key = row["molecule_id"]
        image = root / row["image_file"]
        truth = root / row["sdf_file"]
        if _sha256_file(image) != row["image_sha256"]:
            raise ValueError(f"Public image hash differs for {key}")
        if _sha256_file(truth) != row["sdf_sha256"]:
            raise ValueError(f"Public SDF hash differs for {key}")
        expected = _read_mol(truth)
        if expected is None:
            raise ValueError(f"Public ground-truth SDF is invalid for {key}")
        baseline_dir = output / "baseline" / key
        candidate_dir = output / "candidate" / key
        candidate_run = _run(_command(image, candidate_dir, args.device, candidate=True))
        if candidate_run["returncode"]:
            raise RuntimeError(
                f"Public candidate inference failed for {key}: {candidate_run}"
            )
        candidate, candidate_mol = _metrics(candidate_dir, expected)
        if args.conditional_baseline and not candidate["suppression"]["executed"]:
            baseline = copy.deepcopy(candidate)
            baseline["suppression"] = {"enabled": False, "executed": False}
            baseline_mol = candidate_mol
            baseline_run = {
                "returncode": 0,
                "seconds": 0.0,
                "stdout": "candidate no-action identity",
                "stderr": "",
            }
            baseline_source = "candidate_no_action_identity"
        else:
            baseline_run = _run(
                _command(image, baseline_dir, args.device, candidate=False)
            )
            if baseline_run["returncode"]:
                raise RuntimeError(
                    f"Public baseline inference failed for {key}: {baseline_run}"
                )
            baseline, baseline_mol = _metrics(baseline_dir, expected)
            baseline_source = "executed_v01010"
        graph_equal = bool(
            baseline_mol is not None
            and candidate_mol is not None
            and sdf_graph_match(baseline_mol, candidate_mol, use_chirality=True)
        )
        if baseline_mol is None and candidate_mol is None:
            graph_equal = (
                baseline["vertices"] == candidate["vertices"]
                and baseline["selected_bonds"] == candidate["selected_bonds"]
            )
        record = {
            "key": key,
            "dataset": row["dataset"],
            "subset": row["subset"],
            "baseline": baseline,
            "candidate": candidate,
            "baseline_candidate_graph_equal": graph_equal,
            "baseline_source": baseline_source,
            "baseline_seconds": baseline_run["seconds"],
            "candidate_seconds": candidate_run["seconds"],
        }
        records.append(record)
        with (output / "records.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if (index + 1) % 5 == 0 or candidate["suppression"]["executed"]:
            print(
                json.dumps(
                    {
                        "completed": index + 1,
                        "key": key,
                        "executed": candidate["suppression"]["executed"],
                        "baseline_valid": baseline["sdf_valid"],
                        "candidate_valid": candidate["sdf_valid"],
                    }
                ),
                flush=True,
            )
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
        "require_action": False,
        "scope": "unseen public USPTO clean fixed-order regression",
    }
    report = {"config": config, "summary": summary, "gate": gate}
    _write_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
