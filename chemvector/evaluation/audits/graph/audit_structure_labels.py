#!/usr/bin/env python3
"""Audit Render+ChemMol structural labels against semantic layer ground truth."""

from __future__ import annotations

import argparse
from itertools import islice
import json
import math
from pathlib import Path

from chemvector.data import ChemVectorIterableDataset, synthesize_structure_labels
from chemvector.train import expand_shards


def compare_structure_labels(sample: dict, tolerance_pixels: float) -> dict:
    expected = sample["layers"]
    actual = synthesize_structure_labels(sample)
    errors = []
    maximum_atom_delta = 0.0
    maximum_bond_delta = 0.0

    expected_atoms = {int(atom["atom_id"]): atom for atom in expected["atoms"]}
    actual_atoms = {int(atom["atom_id"]): atom for atom in actual["atoms"]}
    if set(expected_atoms) != set(actual_atoms):
        errors.append("atom_ids")
    for atom_id in sorted(set(expected_atoms) & set(actual_atoms)):
        left = expected_atoms[atom_id]
        right = actual_atoms[atom_id]
        for field in ("element", "formal_charge", "aromatic", "implicit_h_count"):
            if left[field] != right[field]:
                errors.append(f"atom_{atom_id}_{field}")
        delta = math.dist(left["center"], right["center"])
        maximum_atom_delta = max(maximum_atom_delta, delta)
        if delta > tolerance_pixels:
            errors.append(f"atom_{atom_id}_center")

    expected_bonds = {int(bond["bond_id"]): bond for bond in expected["bonds"]}
    actual_bonds = {int(bond["bond_id"]): bond for bond in actual["bonds"]}
    if set(expected_bonds) != set(actual_bonds):
        errors.append("bond_ids")
    for bond_id in sorted(set(expected_bonds) & set(actual_bonds)):
        left = expected_bonds[bond_id]
        right = actual_bonds[bond_id]
        for field in (
            "begin_atom",
            "end_atom",
            "chemical_type",
            "display_type",
            "direction",
        ):
            if left[field] != right[field]:
                errors.append(f"bond_{bond_id}_{field}")
        delta = max(
            math.dist(left["begin"], right["begin"]),
            math.dist(left["end"], right["end"]),
        )
        maximum_bond_delta = max(maximum_bond_delta, delta)
        if delta > tolerance_pixels:
            errors.append(f"bond_{bond_id}_endpoints")

    return {
        "key": sample["key"],
        "atoms": len(expected_atoms),
        "bonds": len(expected_bonds),
        "maximum_atom_delta_pixels": maximum_atom_delta,
        "maximum_bond_delta_pixels": maximum_bond_delta,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--tolerance-pixels", type=float, default=0.01)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset = iter(
        ChemVectorIterableDataset(expand_shards(args.shards), sequence_version="0.4")
    )
    records = islice(dataset, args.samples) if args.samples is not None else dataset
    samples = atoms = bonds = 0
    maximum_atom_delta = maximum_bond_delta = 0.0
    failures = []
    for sample in records:
        if "layers" not in sample:
            raise ValueError(f"Audit sample lacks semantic layers: {sample['key']}")
        result = compare_structure_labels(sample, args.tolerance_pixels)
        samples += 1
        atoms += result["atoms"]
        bonds += result["bonds"]
        maximum_atom_delta = max(
            maximum_atom_delta, result["maximum_atom_delta_pixels"]
        )
        maximum_bond_delta = max(
            maximum_bond_delta, result["maximum_bond_delta_pixels"]
        )
        if result["errors"] and len(failures) < 100:
            failures.append(result)
        if samples % 1000 == 0:
            print(f"audited={samples}", flush=True)

    summary = {
        "schema_version": "chemvector-structure-label-audit-0.1",
        "status": "passed" if not failures else "failed",
        "compared_fields": [
            "atom_id",
            "atom_element",
            "atom_formal_charge",
            "atom_aromatic",
            "atom_implicit_h_count",
            "atom_center",
            "bond_id",
            "bond_begin_atom",
            "bond_end_atom",
            "bond_endpoints",
            "bond_chemical_type",
            "bond_display_type",
            "bond_direction",
        ],
        "excluded_fields": [
            "atom_visible_text",
            "atom_text",
            "atom_bbox",
            "bond_bbox",
            "text_layer",
            "bond_layer",
            "semantic_layer",
        ],
        "samples": samples,
        "atoms": atoms,
        "bonds": bonds,
        "tolerance_pixels": args.tolerance_pixels,
        "maximum_atom_delta_pixels": maximum_atom_delta,
        "maximum_bond_delta_pixels": maximum_bond_delta,
        "failures": failures,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
