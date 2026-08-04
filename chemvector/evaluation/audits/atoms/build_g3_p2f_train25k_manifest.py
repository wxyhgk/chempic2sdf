#!/usr/bin/env python3
"""Build a route-scaffold-disjoint 25k-image manifest for P2-F."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tarfile

from chemvector.training.atoms import train_g3_p1a_canonical_atom_selector as p1a
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256


DATA_ROOT = Path("Data/sdf/chemvector_joint_v1/style_train_100k")
OUTPUT = Path("outputs/audits/chemvector-g3-p2f-train25k-manifest")
MANIFEST = OUTPUT / "manifest.jsonl"
MOLECULES = 6_250
IMAGES = 25_000


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P2-F manifest output is not empty: {OUTPUT}")
    route, route_audit = p1a.load_route_manifest()
    train5k, train5k_audit = p1a.load_train_manifest()
    route_molecules = {record.molecule_id for record in route}
    with (DATA_ROOT / "samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    route_scaffolds = {
        row["scaffold"] for row in rows if row["molecule_id"] in route_molecules
    }
    grouped = {}
    order = []
    for row in rows:
        molecule = row["molecule_id"]
        if molecule not in grouped:
            grouped[molecule] = []
            order.append(molecule)
        grouped[molecule].append(row)
    selected = []
    selected_molecules = []
    for molecule in order:
        current = grouped[molecule]
        if molecule in route_molecules or current[0]["scaffold"] in route_scaffolds:
            continue
        if len(current) != 4:
            raise RuntimeError(f"P2-F molecule does not have four layouts: {molecule}")
        selected.extend(current)
        selected_molecules.append(molecule)
        if len(selected_molecules) == MOLECULES:
            break
    if len(selected) != IMAGES:
        raise RuntimeError("P2-F could not select exactly 25,000 images")
    selected_by_key = {row["sample_key"]: row for row in selected}
    physical_order = []
    for shard in dict.fromkeys(row["output_shard"] for row in selected):
        with tarfile.open(DATA_ROOT / shard) as archive:
            for member in archive:
                if not member.name.endswith(".png"):
                    continue
                key = member.name[:-4]
                if key in selected_by_key:
                    physical_order.append(key)
    if len(physical_order) != IMAGES or len(set(physical_order)) != IMAGES:
        raise RuntimeError("P2-F could not recover unique physical tar order")
    selected = [selected_by_key[key] for key in physical_order]
    selected_keys = {row["sample_key"] for row in selected}
    train5k_keys = {record.sample_key for record in train5k}
    lines = []
    for row in selected:
        lines.append(json.dumps({
            "inchikey": row["inchikey"],
            "molecule_id": row["molecule_id"],
            "sample_key": row["sample_key"],
            "shard": str((DATA_ROOT / row["output_shard"]).resolve()),
            "variant": row["variant"],
        }, sort_keys=True))
    OUTPUT.mkdir(parents=True, exist_ok=False)
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    key_sha = hashlib.sha256(
        "".join(f"{row['sample_key']}\n" for row in selected).encode()
    ).hexdigest()
    audit = {
        "schema_version": "g3-p2f-train25k-manifest-0.1",
        "status": "passed",
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": file_sha256(MANIFEST),
        "sample_keys_sha256": key_sha,
        "samples": len(selected),
        "molecules": len(selected_molecules),
        "variants": sorted({row["variant"] for row in selected}),
        "train5k_is_subset": train5k_keys <= selected_keys,
        "train5k_sample_overlap": len(train5k_keys & selected_keys),
        "route_molecule_overlap": len(set(selected_molecules) & route_molecules),
        "route_scaffold_overlap": len(
            {row["scaffold"] for row in selected} & route_scaffolds
        ),
        "route_manifest": route_audit,
        "train5k_manifest": train5k_audit,
        "source_commitment": json.loads(
            (DATA_ROOT / "commitment.json").read_text(encoding="utf-8")
        ),
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
