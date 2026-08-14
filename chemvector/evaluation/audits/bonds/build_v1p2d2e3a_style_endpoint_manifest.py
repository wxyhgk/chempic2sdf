#!/usr/bin/env python3
"""Build a ChemScene-safe, scaffold-disjoint style endpoint manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any

from Data.sdf.chemscene_v0_2 import build_chemscene_v0_2
from chemvector.g3_structural_data import file_sha256
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json


AUDIT_NAME = "chemvector-v1p2d2e3a-style-endpoint-safe-manifest"
DEFAULT_OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
DATA_ROOT = Path("Data/sdf/chemvector_joint_v1/style_train_100k")
SOURCE_COMMITMENT = (
    "2082372d253fa309715becfd5cba9e5cdb78eb8d84a5ece4ff1eb73bd67390eb"
)
SAMPLES_CSV_SHA256 = (
    "f38a8d10a9ff558d646d2a8340354347022e905f95ff9ea73f29fc7088bfa26a"
)
SPLIT_SEED = 20260823
CALIBRATION_SCAFFOLDS = 750
HOLDOUT_SCAFFOLDS = 750
EXPECTED_SAMPLES = 100_000
EXPECTED_MOLECULES = 25_000
EXPECTED_VARIANTS = frozenset({"coordgen", "original", "rdkit", "stress"})
JSON_SUFFIXES = (".chemmol.json", ".render.json", ".layers.json")


def _rank_scaffold(scaffold: str) -> tuple[str, str]:
    return (
        hashlib.sha256(f"{SPLIT_SEED}:{scaffold}".encode()).hexdigest(),
        scaffold,
    )


def assign_scaffold_splits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scaffolds = sorted({str(row["scaffold"]) for row in rows}, key=_rank_scaffold)
    if len(scaffolds) < CALIBRATION_SCAFFOLDS + HOLDOUT_SCAFFOLDS + 1:
        raise ValueError("P2-D2-E3A has too few scaffolds for isolated splits")
    calibration = set(scaffolds[:CALIBRATION_SCAFFOLDS])
    holdout = set(
        scaffolds[
            CALIBRATION_SCAFFOLDS : CALIBRATION_SCAFFOLDS + HOLDOUT_SCAFFOLDS
        ]
    )
    assigned = []
    for source in rows:
        row = dict(source)
        scaffold = str(row["scaffold"])
        row["split"] = (
            "calibration"
            if scaffold in calibration
            else "holdout"
            if scaffold in holdout
            else "fit"
        )
        assigned.append(row)
    return assigned


def _sample_key(member_name: str) -> str | None:
    for suffix in JSON_SUFFIXES:
        if member_name.endswith(suffix):
            return member_name[: -len(suffix)]
    if member_name.endswith(".png") and not member_name.endswith(
        (".text.png", ".bond.png", ".semantic.png")
    ):
        return member_name[:-4]
    return None


def preflight_scenes(
    shard_paths: list[Path], expected_keys: set[str]
) -> tuple[list[str], dict[str, dict[str, str]]]:
    physical_order = []
    payloads: dict[str, dict[str, dict]] = defaultdict(dict)
    failures: dict[str, dict[str, str]] = {}
    inspected = set()
    for shard in shard_paths:
        with tarfile.open(shard, "r") as archive:
            for member in archive:
                key = _sample_key(member.name)
                if key is None or key not in expected_keys:
                    continue
                if member.name == f"{key}.png":
                    physical_order.append(key)
                    continue
                suffix = next(
                    (value for value in JSON_SUFFIXES if member.name.endswith(value)),
                    None,
                )
                if suffix is None:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"P2-D2-E3A cannot extract {member.name}")
                payloads[key][suffix] = json.loads(handle.read())
                if len(payloads[key]) != len(JSON_SUFFIXES):
                    continue
                current = payloads.pop(key)
                inspected.add(key)
                try:
                    build_chemscene_v0_2(
                        current[".chemmol.json"],
                        current[".render.json"],
                        current[".layers.json"],
                    )
                except Exception as error:
                    failures[key] = {
                        "exception": type(error).__name__,
                        "message": str(error),
                    }
    if payloads:
        raise ValueError("P2-D2-E3A has incomplete JSON payload groups")
    if inspected != expected_keys:
        raise ValueError(
            f"P2-D2-E3A inspected {len(inspected)} samples, expected {len(expected_keys)}"
        )
    if len(physical_order) != len(expected_keys) or set(physical_order) != expected_keys:
        raise ValueError("P2-D2-E3A physical tar order differs from samples.csv")
    return physical_order, failures


def _split_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = Counter(str(row["split"]) for row in rows)
    molecules: dict[str, set[str]] = defaultdict(set)
    scaffolds: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row["split"])
        molecules[split].add(str(row["molecule_id"]))
        scaffolds[split].add(str(row["scaffold"]))
    names = ("fit", "calibration", "holdout")
    if any(
        scaffolds[names[first]] & scaffolds[names[second]]
        for first, second in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("P2-D2-E3A scaffold split overlap")
    if any(
        molecules[names[first]] & molecules[names[second]]
        for first, second in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("P2-D2-E3A molecule split overlap")
    return {
        "samples": dict(sorted(split_counts.items())),
        "molecules": {name: len(molecules[name]) for name in names},
        "scaffolds": {name: len(scaffolds[name]) for name in names},
        "cross_split_molecule_overlap": 0,
        "cross_split_scaffold_overlap": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P2-D2-E3A output is not empty: {args.output_dir}")
    commitment = json.loads(
        (DATA_ROOT / "commitment.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (DATA_ROOT / "validation.json").read_text(encoding="utf-8")
    )
    if commitment.get("commitment_sha256") != SOURCE_COMMITMENT:
        raise ValueError("P2-D2-E3A style commitment differs")
    if file_sha256(DATA_ROOT / "samples.csv") != SAMPLES_CSV_SHA256:
        raise ValueError("P2-D2-E3A samples.csv SHA256 differs")
    if validation.get("status") != "passed":
        raise ValueError("P2-D2-E3A source validation did not pass")
    with (DATA_ROOT / "samples.csv").open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != EXPECTED_SAMPLES:
        raise ValueError("P2-D2-E3A source sample count differs")
    by_molecule: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        by_molecule[str(row["molecule_id"])].append(row)
    if len(by_molecule) != EXPECTED_MOLECULES:
        raise ValueError("P2-D2-E3A source molecule count differs")
    if any(
        len(group) != 4
        or {str(row["variant"]) for row in group} != EXPECTED_VARIANTS
        for group in by_molecule.values()
    ):
        raise ValueError("P2-D2-E3A source molecule variants differ")
    shard_paths = [
        DATA_ROOT / item["path"]
        for item in commitment["tars"]
        if str(item["path"]).startswith("shards/train-")
    ]
    if len(shard_paths) != 20:
        raise ValueError("P2-D2-E3A source shard count differs")
    rows_by_key = {str(row["sample_key"]): row for row in source_rows}
    physical_order, failures = preflight_scenes(shard_paths, set(rows_by_key))
    rejected_molecules = {
        str(rows_by_key[key]["molecule_id"]) for key in failures
    }
    safe_rows = [
        rows_by_key[key]
        for key in physical_order
        if str(rows_by_key[key]["molecule_id"]) not in rejected_molecules
    ]
    assigned = assign_scaffold_splits(safe_rows)
    split_audit = _split_audit(assigned)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps(
                {
                    "sample_key": row["sample_key"],
                    "molecule_id": row["molecule_id"],
                    "inchikey": row["inchikey"],
                    "scaffold": row["scaffold"],
                    "split": row["split"],
                    "variant": row["variant"],
                    "source_shard": row["output_shard"],
                    "font_family_id": row["font_family_id"],
                    "font_bucket": row["font_bucket"],
                    "line_width_bucket": row["line_width_bucket"],
                    "color_mode": row["color_mode"],
                    "stress_operation": row["stress_operation"] or None,
                },
                sort_keys=True,
            )
            + "\n"
            for row in assigned
        ),
        encoding="utf-8",
    )
    rejection_payload = {
        "schema_version": "chemvector-v1p2d2e3a-rejections-0.1",
        "failed_samples": failures,
        "rejected_molecules": sorted(rejected_molecules),
        "policy": "reject all four variants when any molecule variant fails ChemScene v0.2",
        "generated_png_files": 0,
    }
    _atomic_json(args.output_dir / "rejections.json", rejection_payload)
    audit = {
        "schema_version": "chemvector-v1p2d2e3a-style-endpoint-manifest-0.1",
        "audit_name": AUDIT_NAME,
        "status": "passed",
        "source_commitment": SOURCE_COMMITMENT,
        "source_samples_csv_sha256": SAMPLES_CSV_SHA256,
        "source_samples": len(source_rows),
        "source_molecules": len(by_molecule),
        "inspected_samples": len(physical_order),
        "failed_samples": len(failures),
        "rejected_molecules": len(rejected_molecules),
        "safe_samples": len(assigned),
        "safe_molecules": len({str(row["molecule_id"]) for row in assigned}),
        "split_seed": SPLIT_SEED,
        "split": split_audit,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "rejections_sha256": file_sha256(args.output_dir / "rejections.json"),
        "generated_png_files": 0,
        "fresh_dev_samples_read": 0,
        "sealed_test_samples_read": 0,
    }
    _atomic_json(args.output_dir / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
