#!/usr/bin/env python3
"""Audit stride-4 endpoint target occupancy over the safe style manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import tarfile
from typing import Any

from Data.sdf.chemscene_v0_2 import build_chemscene_v0_2
from chemvector.evaluation.audits.bonds.build_v1p2d2e3a_style_endpoint_manifest import (
    DATA_ROOT,
    JSON_SUFFIXES,
    _sample_key,
)
from chemvector.g3_structural_data import file_sha256
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json


AUDIT_NAME = "chemvector-v1p2d2e3b-style-endpoint-slot-capacity"
DEFAULT_OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
MANIFEST = Path(
    "outputs/audits/chemvector-v1p2d2e3a-style-endpoint-safe-manifest/manifest.jsonl"
)
MANIFEST_SHA256 = (
    "ee41d57275d8c48ebe9461b69eeef5bbb90abaa10de9319b746f8274024686e6"
)
EXPECTED_SAMPLES = 99_336
GRID_SIZE = 192
CURRENT_SLOTS = 3


def endpoint_cell_occupancy(scene: dict[str, Any]) -> Counter[tuple[int, int]]:
    unique_ports = {
        (int(port[0]), int(port[1]))
        for bond in scene["bond_objects"]
        for port in bond["ports"]
    }
    return Counter(
        (
            min(GRID_SIZE - 1, max(0, int(x / 1023.0 * GRID_SIZE))),
            min(GRID_SIZE - 1, max(0, int(y / 1023.0 * GRID_SIZE))),
        )
        for x, y in unique_ports
    )


def route_decision(maximum_occupancy: int) -> str:
    if maximum_occupancy <= CURRENT_SLOTS:
        return "current_slots_sufficient"
    if maximum_occupancy == CURRENT_SLOTS + 1:
        return "increase_slots_to_4"
    return "dynamic_cell_set_required"


def _update(totals: Counter[str], occupancy: Counter[tuple[int, int]]) -> None:
    maximum = max(occupancy.values(), default=0)
    totals["samples"] += 1
    totals["ports"] += sum(occupancy.values())
    totals["occupied_cells"] += len(occupancy)
    totals["collision_excess"] += sum(value - 1 for value in occupancy.values())
    totals["collision_images"] += int(maximum > 1)
    totals["over_capacity_images"] += int(maximum > CURRENT_SLOTS)
    totals[f"maximum_{maximum}_images"] += 1
    totals["maximum_occupancy"] = max(totals["maximum_occupancy"], maximum)


def _summary(totals: Counter[str]) -> dict[str, Any]:
    samples = max(totals["samples"], 1)
    histogram = {
        key.removeprefix("maximum_").removesuffix("_images"): value
        for key, value in sorted(totals.items())
        if key.startswith("maximum_") and key.endswith("_images")
    }
    return {
        "samples": totals["samples"],
        "unique_ports": totals["ports"],
        "occupied_cells": totals["occupied_cells"],
        "collision_excess": totals["collision_excess"],
        "collision_images": totals["collision_images"],
        "collision_image_rate": totals["collision_images"] / samples,
        "over_capacity_images": totals["over_capacity_images"],
        "over_capacity_rate": totals["over_capacity_images"] / samples,
        "maximum_occupancy": totals["maximum_occupancy"],
        "maximum_occupancy_histogram": histogram,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"P2-D2-E3B output is not empty: {args.output_dir}")
    if file_sha256(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("P2-D2-E3B safe manifest SHA256 differs")
    rows = [
        json.loads(line)
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != EXPECTED_SAMPLES:
        raise ValueError("P2-D2-E3B safe manifest sample count differs")
    by_key = {str(row["sample_key"]): row for row in rows}
    shard_names = list(dict.fromkeys(str(row["source_shard"]) for row in rows))
    totals: Counter[str] = Counter()
    splits: dict[str, Counter[str]] = defaultdict(Counter)
    variants: dict[str, Counter[str]] = defaultdict(Counter)
    operations: dict[str, Counter[str]] = defaultdict(Counter)
    payloads: dict[str, dict[str, dict]] = defaultdict(dict)
    inspected = set()
    over_capacity = []
    for shard_name in shard_names:
        with tarfile.open(DATA_ROOT / shard_name, "r") as archive:
            for member in archive:
                key = _sample_key(member.name)
                if key is None or key not in by_key:
                    continue
                suffix = next(
                    (value for value in JSON_SUFFIXES if member.name.endswith(value)),
                    None,
                )
                if suffix is None:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"P2-D2-E3B cannot extract {member.name}")
                payloads[key][suffix] = json.loads(handle.read())
                if len(payloads[key]) != len(JSON_SUFFIXES):
                    continue
                current = payloads.pop(key)
                scene = build_chemscene_v0_2(
                    current[".chemmol.json"],
                    current[".render.json"],
                    current[".layers.json"],
                )
                occupancy = endpoint_cell_occupancy(scene)
                row = by_key[key]
                _update(totals, occupancy)
                _update(splits[str(row["split"])], occupancy)
                _update(variants[str(row["variant"])], occupancy)
                if row["stress_operation"] is not None:
                    _update(operations[str(row["stress_operation"])], occupancy)
                maximum = max(occupancy.values(), default=0)
                if maximum > CURRENT_SLOTS:
                    over_capacity.append(
                        {
                            "sample_key": key,
                            "molecule_id": row["molecule_id"],
                            "split": row["split"],
                            "variant": row["variant"],
                            "stress_operation": row["stress_operation"],
                            "maximum_occupancy": maximum,
                            "over_capacity_cells": [
                                {"cell": list(cell), "ports": count}
                                for cell, count in sorted(occupancy.items())
                                if count > CURRENT_SLOTS
                            ],
                        }
                    )
                inspected.add(key)
    if payloads or inspected != set(by_key):
        raise ValueError("P2-D2-E3B did not inspect every safe sample exactly once")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = args.output_dir / "over_capacity.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in over_capacity),
        encoding="utf-8",
    )
    overall = _summary(totals)
    audit = {
        "schema_version": "chemvector-v1p2d2e3b-slot-capacity-0.1",
        "audit_name": AUDIT_NAME,
        "status": "passed",
        "decision": route_decision(int(overall["maximum_occupancy"])),
        "manifest_sha256": MANIFEST_SHA256,
        "grid_size": GRID_SIZE,
        "current_slots": CURRENT_SLOTS,
        "summary": overall,
        "by_split": {name: _summary(value) for name, value in sorted(splits.items())},
        "by_variant": {
            name: _summary(value) for name, value in sorted(variants.items())
        },
        "by_stress_operation": {
            name: _summary(value) for name, value in sorted(operations.items())
        },
        "over_capacity_rows": len(over_capacity),
        "over_capacity_rows_sha256": file_sha256(rows_path),
        "generated_png_files": 0,
        "fresh_dev_samples_read": 0,
        "sealed_test_samples_read": 0,
    }
    _atomic_json(args.output_dir / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
