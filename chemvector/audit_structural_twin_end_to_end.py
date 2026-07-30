#!/usr/bin/env python3
"""Run a fixed v0.10.10/v0.10.16 end-to-end SDF regression pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

from rdkit import Chem, RDLogger

from chemvector.benchmark_public_ocsr import sdf_graph_match


DATA_ROOT = Path("Data/sdf/chemvector_safe_suppression_v0_1/route_select")
DEFAULT_ROUTE_SELECT_REPORT = (
    "outputs/audits/chemvector-v01015-structural-twin-route-select/recovery_report.json"
)
ROUTE_SELECT_REPORT_SHA256 = (
    "8973b527ad211f5501e68679f7169f45bc059bdc3ee6382e4fe50d4b8672b7c7"
)
DEFAULT_OUTPUT = "outputs/audits/chemvector-v01016-end-to-end-pilot40"
PYTHON = Path(".venv-train/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-select-report", default=DEFAULT_ROUTE_SELECT_REPORT)
    parser.add_argument("--samples-per-bucket", type=int, default=20)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _load_rows() -> dict[str, dict]:
    with (DATA_ROOT / "samples.csv").open(newline="", encoding="utf-8") as handle:
        return {row["sample_key"]: row for row in csv.DictReader(handle)}


def _select_samples(report: dict, rows: dict[str, dict], count: int) -> list[dict]:
    action_cases = {
        row["key"]: row for row in report["metrics"]["all"]["cases"]
    }
    selected = []
    used_molecules = set()
    for key in sorted(action_cases):
        source = rows[key]
        molecule_id = source["molecule_id"]
        if molecule_id in used_molecules:
            continue
        selected.append({"bucket": "action", **source})
        used_molecules.add(molecule_id)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("Not enough molecule-distinct action cases")
    control_count = 0
    for key in sorted(rows):
        source = rows[key]
        molecule_id = source["molecule_id"]
        if key in action_cases or molecule_id in used_molecules:
            continue
        selected.append({"bucket": "control", **source})
        used_molecules.add(molecule_id)
        control_count += 1
        if control_count == count:
            break
    if control_count != count:
        raise ValueError("Not enough molecule-distinct control cases")
    return selected


def _extract_member(
    archives: dict[str, tarfile.TarFile], row: dict, suffix: str
) -> bytes:
    shard = row["output_shard"]
    archive = archives.get(shard)
    if archive is None:
        archive = tarfile.open(DATA_ROOT / shard, "r")
        archives[shard] = archive
    member = archive.extractfile(f"{row['sample_key']}{suffix}")
    if member is None:
        raise FileNotFoundError(f"{row['sample_key']}{suffix}")
    return member.read()


def _command(image: Path, output: Path, device: str, *, candidate: bool) -> list[str]:
    command = [
        str(PYTHON),
        "-m",
        "chemvector.infer_direct_atom_pair_bond",
        "--image",
        str(image),
        "--output-dir",
        str(output),
        "--device",
        device,
        "--model-version",
        (
            "v0.10.16-structural-twin-direct-aromatic-ocr"
            if candidate
            else "v0.10.10-direct-aromatic-ocr"
        ),
        "--checkpoint",
        "outputs/checkpoints/chemvector-v0103-direct-atom-pair-style10k/selected.pt",
        "--role-checkpoint",
        "outputs/checkpoints/chemvector-v0105-direct-atom-role-style10k/selected.pt",
        "--bbox-checkpoint",
        "outputs/checkpoints/chemvector-v0108-direct-atom-text-bbox-style10k/selected.pt",
        "--ocr-checkpoint",
        "outputs/checkpoints/chemvector-v0109-correct-label-ocr-style10k/selected.pt",
        "--enable-ocr",
        "--aromatic-ring-fusion",
        "--aromatic-mean-threshold",
        "0.40",
        "--aromatic-minimum-threshold",
        "0.05",
    ]
    if candidate:
        command.extend(
            (
                "--structural-twin-suppression",
                "--action-ranking-checkpoint",
                "outputs/checkpoints/chemvector-v01013-safe-atom-action-50k/samples-030000.pt",
            )
        )
    return command


def _run(command: list[str]) -> dict:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    return {
        "returncode": completed.returncode,
        "seconds": time.perf_counter() - started,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def _read_mol(path: Path) -> Chem.Mol | None:
    return Chem.MolFromMolFile(
        str(path), sanitize=True, removeHs=False, strictParsing=False
    )


def _metrics(output: Path, expected: Chem.Mol) -> tuple[dict, Chem.Mol | None]:
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    predicted = _read_mol(output / "prediction-repaired.sdf")
    return (
        {
            "sdf_valid": bool(metrics["sdf_valid"] and predicted is not None),
            "sdf_readable": predicted is not None,
            "decode_path": metrics["diagnostics"]["decode_path"],
            "nonstereo_exact": bool(
                predicted is not None
                and sdf_graph_match(expected, predicted, use_chirality=False)
            ),
            "chemical_exact": bool(
                predicted is not None
                and sdf_graph_match(expected, predicted, use_chirality=True)
            ),
            "vertices": metrics["vertices"],
            "selected_bonds": metrics["selected_bonds"],
            "formula": metrics["formula"],
            "ocr_labels": metrics["diagnostics"]["ocr"]["labels"],
            "suppression": metrics["diagnostics"]["structural_twin_suppression"],
        },
        predicted,
    )


def _summary(records: list[dict]) -> dict:
    action = [row for row in records if row["bucket"] == "action"]
    control = [row for row in records if row["bucket"] == "control"]
    return {
        "samples": len(records),
        "action_samples": len(action),
        "control_samples": len(control),
        "candidate_actions_executed": sum(
            bool(row["candidate"]["suppression"]["executed"]) for row in records
        ),
        "action_protocol_misses": sum(
            not row["candidate"]["suppression"]["executed"] for row in action
        ),
        "control_protocol_breaks": sum(
            bool(row["candidate"]["suppression"]["executed"]) for row in control
        ),
        "baseline_sdf_valid": sum(row["baseline"]["sdf_valid"] for row in records),
        "candidate_sdf_valid": sum(row["candidate"]["sdf_valid"] for row in records),
        "baseline_sdf_readable": sum(
            row["baseline"]["sdf_readable"] for row in records
        ),
        "candidate_sdf_readable": sum(
            row["candidate"]["sdf_readable"] for row in records
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
        "chemical_exact_breaks": sum(
            row["baseline"]["chemical_exact"]
            and not row["candidate"]["chemical_exact"]
            for row in records
        ),
        "control_ocr_breaks": sum(
            row["baseline"]["ocr_labels"] != row["candidate"]["ocr_labels"]
            for row in control
        ),
        "control_graph_breaks": sum(
            not row["baseline_candidate_graph_equal"] for row in control
        ),
    }


def main() -> None:
    args = parse_args()
    if args.samples_per_bucket < 1:
        raise ValueError("samples-per-bucket must be positive")
    report_path = Path(args.route_select_report)
    if _sha256_file(report_path) != ROUTE_SELECT_REPORT_SHA256:
        raise ValueError("Route-select recovery report differs")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = _load_rows()
    selected = _select_samples(report, rows, args.samples_per_bucket)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.reuse_existing:
        raise RuntimeError(f"End-to-end pilot output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": "structural-twin-end-to-end-pilot-0.1",
        "route_select_report": str(report_path),
        "route_select_report_sha256": ROUTE_SELECT_REPORT_SHA256,
        "samples_per_bucket": args.samples_per_bucket,
        "selection": "lexical sample_key, molecule-distinct, action then control",
        "device": args.device,
    }
    _write_json(output / "config.json", config)
    input_dir = output / "inputs"
    input_dir.mkdir(exist_ok=args.reuse_existing)
    archives: dict[str, tarfile.TarFile] = {}
    records = []
    (output / "records.jsonl").write_text("", encoding="utf-8")
    RDLogger.DisableLog("rdApp.*")
    try:
        for index, row in enumerate(selected):
            key = row["sample_key"]
            image_bytes = _extract_member(archives, row, ".png")
            sdf_bytes = _extract_member(archives, row, ".sdf")
            if _sha256_bytes(image_bytes) != row["png_sha256"]:
                raise ValueError(f"PNG hash differs for {key}")
            if _sha256_bytes(sdf_bytes) != row["sdf_sha256"]:
                raise ValueError(f"SDF hash differs for {key}")
            image_path = input_dir / f"{key}.png"
            truth_path = input_dir / f"{key}.sdf"
            image_path.write_bytes(image_bytes)
            truth_path.write_bytes(sdf_bytes)
            expected = _read_mol(truth_path)
            if expected is None:
                raise ValueError(f"Ground-truth SDF is invalid for {key}")
            baseline_dir = output / "baseline" / key
            candidate_dir = output / "candidate" / key
            if args.reuse_existing:
                baseline_run = {
                    "returncode": 0,
                    "seconds": 0.0,
                    "stdout": "reused",
                    "stderr": "",
                }
                candidate_run = dict(baseline_run)
            else:
                baseline_run = _run(
                    _command(image_path, baseline_dir, args.device, candidate=False)
                )
                candidate_run = _run(
                    _command(image_path, candidate_dir, args.device, candidate=True)
                )
            if baseline_run["returncode"] or candidate_run["returncode"]:
                raise RuntimeError(
                    f"Inference command failed for {key}: baseline={baseline_run} candidate={candidate_run}"
                )
            baseline, baseline_mol = _metrics(baseline_dir, expected)
            candidate, candidate_mol = _metrics(candidate_dir, expected)
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
                "molecule_id": row["molecule_id"],
                "bucket": row["bucket"],
                "baseline": baseline,
                "candidate": candidate,
                "baseline_candidate_graph_equal": graph_equal,
                "baseline_seconds": baseline_run["seconds"],
                "candidate_seconds": candidate_run["seconds"],
            }
            records.append(record)
            with (output / "records.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {
                        "completed": index + 1,
                        "key": key,
                        "bucket": row["bucket"],
                        "executed": candidate["suppression"]["executed"],
                        "baseline_valid": baseline["sdf_valid"],
                        "candidate_valid": candidate["sdf_valid"],
                    }
                ),
                flush=True,
            )
    finally:
        for archive in archives.values():
            archive.close()
    summary = _summary(records)
    gate = {
        "passed": summary["action_protocol_misses"] == 0
        and summary["control_protocol_breaks"] == 0
        and summary["sdf_valid_breaks"] == 0
        and summary["nonstereo_exact_breaks"] == 0
        and summary["chemical_exact_breaks"] == 0
        and summary["control_ocr_breaks"] == 0
        and summary["control_graph_breaks"] == 0
        and summary["candidate_sdf_valid"] >= summary["baseline_sdf_valid"]
        and summary["candidate_nonstereo_exact"]
        >= summary["baseline_nonstereo_exact"]
        and summary["candidate_chemical_exact"]
        >= summary["baseline_chemical_exact"],
        "scope": "engineering pilot only; not a new generalization claim",
    }
    result = {"config": config, "summary": summary, "gate": gate}
    _write_json(output / "report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
