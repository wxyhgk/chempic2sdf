#!/usr/bin/env python3
"""Freeze molecule-disjoint manifests for ChemTraverse capacity pilots."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Any


def _bytes(records: list[dict[str, Any]]) -> bytes:
    return b"".join(
        (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for record in records
    )


VARIANTS = frozenset({"original", "rdkit", "coordgen", "stress"})


def complete_molecule_groups(
    records: list[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], int]:
    grouped: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for record in records:
        grouped.setdefault(record["molecule_id"], []).append(record)
    groups = [
        group
        for group in grouped.values()
        if len(group) == 4 and {record["variant"] for record in group} == VARIANTS
    ]
    return groups, len(grouped) - len(groups)


def build_splits(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups, _ = complete_molecule_groups(records)
    if len(groups) < 1_375:
        raise ValueError("Capacity source has fewer than 1,375 complete molecules")
    return {
        "overfit_32": [record for group in groups[:8] for record in group],
        "train_5k": [record for group in groups[:1_250] for record in group],
        "dev_500": [record for group in groups[1_250:1_375] for record in group],
        "dev_route_select_100": [
            record for group in groups[1_250:1_275] for record in group
        ],
        "dev_route_confirm_400": [
            record for group in groups[1_275:1_375] for record in group
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    source_bytes = args.source_manifest.read_bytes()
    records = [
        json.loads(line)
        for line in source_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    complete_groups, skipped_incomplete = complete_molecule_groups(records)
    splits = build_splits(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "schema_version": "chemtraverse-capacity-manifests-0.1",
        "source_manifest": str(args.source_manifest.resolve()),
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "selection": "first complete molecule groups in source manifest order",
        "complete_source_molecules": len(complete_groups),
        "skipped_incomplete_molecules": skipped_incomplete,
        "splits": {},
    }
    molecule_sets = {}
    for name, split_records in splits.items():
        payload = _bytes(split_records)
        filename = f"{name}.jsonl"
        (args.output_dir / filename).write_bytes(payload)
        molecules = {record["molecule_id"] for record in split_records}
        molecule_sets[name] = molecules
        summary["splits"][name] = {
            "path": filename,
            "samples": len(split_records),
            "molecules": len(molecules),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if not molecule_sets["overfit_32"] <= molecule_sets["train_5k"]:
        raise ValueError("overfit_32 must be a subset of train_5k")
    if molecule_sets["train_5k"] & molecule_sets["dev_500"]:
        raise ValueError("train_5k and dev_500 molecules overlap")
    if (
        molecule_sets["dev_route_select_100"]
        & molecule_sets["dev_route_confirm_400"]
    ):
        raise ValueError("Route-select and route-confirm molecules overlap")
    if (
        molecule_sets["dev_route_select_100"]
        | molecule_sets["dev_route_confirm_400"]
    ) != molecule_sets["dev_500"]:
        raise ValueError("Route-select and route-confirm do not partition dev_500")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
