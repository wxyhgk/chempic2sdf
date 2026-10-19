"""Deterministic molecule-grouped selection for the external OCR R0 audit."""

from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Any

import chemvector.external_ocr_r0_contracts as contract


def build_fresh_selection() -> dict[str, Any]:
    samples_csv = contract.FRESH_SAMPLES_PATH
    commitment_path = contract.FRESH_COMMITMENT_PATH
    contract.validate_frozen_file(
        commitment_path,
        contract.FRESH_COMMITMENT_FILE_SHA256,
        "fresh commitment",
    )
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    if commitment.get("commitment_sha256") != contract.FRESH_COMMITMENT_SHA256:
        raise ValueError("Fresh data commitment differs from the frozen R0 protocol")
    expected_samples_sha = next(
        item["sha256"]
        for item in commitment["samples_csv"]
        if item["path"] == "samples.csv"
    )
    if (
        expected_samples_sha != contract.FRESH_SAMPLES_SHA256
        or contract.sha256_file(samples_csv) != contract.FRESH_SAMPLES_SHA256
    ):
        raise ValueError("Fresh samples.csv SHA256 differs from its commitment")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with samples_csv.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "fresh_dev":
                raise ValueError("R0 selection source contains a non-fresh_dev row")
            grouped[row["molecule_id"]].append(row)
    if len(grouped) != 2_000:
        raise ValueError(f"Fresh selection expected 2,000 molecule groups, found {len(grouped)}")
    expected_variants = {"original", "rdkit", "coordgen", "stress"}
    for molecule_id, rows in grouped.items():
        if len(rows) != 4 or {row["variant"] for row in rows} != expected_variants:
            raise ValueError(f"Molecule is not a complete four-variant group: {molecule_id}")

    ranked = sorted(
        grouped,
        key=lambda molecule_id: (
            contract.sha256_bytes(
                f"external-ocr-r0:{molecule_id}".encode("utf-8")
            ),
            molecule_id,
        ),
    )
    selected_molecules = ranked[: contract.SELECTION_MOLECULES]
    selected_rows = sorted(
        (row for molecule_id in selected_molecules for row in grouped[molecule_id]),
        key=lambda row: row["sample_key"],
    )
    selection = {
        "schema_version": contract.SELECTION_SCHEMA_VERSION,
        "audit_id": "external-ocr-r0",
        "split_id": "chemvector_joint_v1/fresh_eval_12k/fresh_dev",
        "selection_method": "smallest_sha256(external-ocr-r0:molecule_id)",
        "molecule_grouped": True,
        "molecules": len(selected_molecules),
        "samples": len(selected_rows),
        "molecule_ids": selected_molecules,
        "sample_keys": [row["sample_key"] for row in selected_rows],
        "sample_metadata": {
            row["sample_key"]: {
                "molecule_id": row["molecule_id"],
                "variant": row["variant"],
                "font_family_group": row["font_family_group"],
                "png_sha256": row["png_sha256"],
            }
            for row in selected_rows
        },
        "inputs": {
            "commitment": {
                "path": str(commitment_path),
                "sha256": contract.sha256_file(commitment_path),
            },
            "samples_csv": {
                "path": str(samples_csv),
                "sha256": contract.sha256_file(samples_csv),
            },
            "commitment_id": contract.FRESH_COMMITMENT_SHA256,
        },
        "forbidden_sources": ["sealed_test", "MR-TADF", "USPTO"],
    }
    if (
        selection["molecules"] != contract.SELECTION_MOLECULES
        or selection["samples"] != contract.SELECTION_MOLECULES * 4
    ):
        raise ValueError("Frozen R0 selection cardinality differs")
    return selection


def validate_selection(selection: dict[str, Any]) -> None:
    if selection.get("schema_version") != contract.SELECTION_SCHEMA_VERSION:
        raise ValueError("Selection manifest schema differs")
    if not selection.get("molecule_grouped"):
        raise ValueError("R0 selection must be molecule-grouped")
    if selection.get("molecules") != contract.SELECTION_MOLECULES:
        raise ValueError("R0 selection must contain exactly 500 molecules")
    if selection.get("samples") != contract.SELECTION_MOLECULES * 4:
        raise ValueError("R0 selection must contain exactly 2,000 samples")
    if selection.get("samples") != len(selection.get("sample_keys", [])):
        raise ValueError("Selection sample count differs")
    if len(set(selection["sample_keys"])) != len(selection["sample_keys"]):
        raise ValueError("Selection contains duplicate sample keys")
    if selection["inputs"].get("commitment_id") != contract.FRESH_COMMITMENT_SHA256:
        raise ValueError("Selection data commitment differs")
    if selection != build_fresh_selection():
        raise ValueError("Selection manifest differs from the deterministic frozen selection")


def write_selection(output: str | Path) -> dict[str, Any]:
    selection = build_fresh_selection()
    contract.write_new_json(output, selection)
    return selection


def validate_fresh_shards(shards: list[str]) -> None:
    commitment = json.loads(
        contract.FRESH_COMMITMENT_PATH.read_text(encoding="utf-8")
    )
    expected = {
        str(
            Path("Data/sdf/chemvector_joint_v1/fresh_eval_12k") / item["path"]
        ): item["sha256"]
        for item in commitment["tars"]
        if item["path"].startswith("shards/fresh_dev-")
    }
    if set(shards) != set(expected):
        raise ValueError("Runtime fresh_dev shard paths differ from the commitment")
    for path in shards:
        contract.validate_frozen_file(Path(path), expected[path], "fresh_dev shard")
