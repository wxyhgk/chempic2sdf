#!/usr/bin/env python3
"""Build the scaffold-disjoint style corpus manifest for P3-E14 Count."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tarfile

from chemvector.training.atoms import audit_g3_p3e7_base_expert_router as p3e7
from chemvector.training.atoms import (
    train_g3_p2e_stride1_top128_selector_train5k as p2e,
)
from chemvector.training.atoms import train_g3_p3e10_count_scale_ab as p3e10
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256


DATA_ROOT = Path("Data/sdf/chemvector_joint_v1/style_train_100k")
SOURCE_COMMITMENT = "2082372d253fa309715becfd5cba9e5cdb78eb8d84a5ece4ff1eb73bd67390eb"
OUTPUT = Path("outputs/audits/chemvector-g3-p3e14-style90932-manifest")
MANIFEST = OUTPUT / "manifest.jsonl"
EXPECTED_MOLECULES = 22_733
EXPECTED_IMAGES = 90_932
EXPECTED_VARIANTS = {"coordgen", "original", "rdkit", "stress"}


def protected_identifiers(cache: dict, rows_by_key: dict[str, dict]) -> tuple[set, set, dict]:
    count_splits, count_audit = p3e10.expanded_splits(cache)
    route_splits, route_audit = p3e7.select_fresh_records(cache)
    protected_records = {
        "dev256": count_splits["dev256"],
        "confirm256": count_splits["confirm256"],
        "historical_route100": count_splits["historical_route100"],
        "route_select": route_splits["route_select"],
        "route_confirm": route_splits["route_confirm"],
    }
    protected_rows = {}
    for split, records in protected_records.items():
        current = []
        for record in records:
            key = str(record["sample_key"])
            if key not in rows_by_key:
                raise RuntimeError(f"P3-E14 protected sample is absent from style corpus: {key}")
            current.append(rows_by_key[key])
        protected_rows[split] = current
    molecules = {
        row["molecule_id"] for values in protected_rows.values() for row in values
    }
    scaffolds = {
        row["scaffold"] for values in protected_rows.values() for row in values
    }
    return molecules, scaffolds, {
        "sets": {
            split: {
                "images": len(values),
                "molecules": len({row["molecule_id"] for row in values}),
                "scaffolds": len({row["scaffold"] for row in values}),
            }
            for split, values in protected_rows.items()
        },
        "protected_molecules": len(molecules),
        "protected_scaffolds": len(scaffolds),
        "p3e10_split_sha256": count_audit["sample_keys_sha256"],
        "p3e7_split_sha256": route_audit["sample_keys_sha256"],
    }


def select_rows(
    rows: list[dict], protected_molecules: set[str], protected_scaffolds: set[str]
) -> list[dict]:
    selected = [
        row for row in rows
        if row["molecule_id"] not in protected_molecules
        and row["scaffold"] not in protected_scaffolds
    ]
    groups: dict[str, list[dict]] = {}
    for row in selected:
        groups.setdefault(row["molecule_id"], []).append(row)
    if any(
        len(values) != 4 or {row["variant"] for row in values} != EXPECTED_VARIANTS
        for values in groups.values()
    ):
        raise RuntimeError("P3-E14 selected molecule lacks the four fixed variants")
    return selected


def physical_order(rows: list[dict]) -> list[dict]:
    by_key = {row["sample_key"]: row for row in rows}
    ordered = []
    for shard in dict.fromkeys(row["output_shard"] for row in rows):
        with tarfile.open(DATA_ROOT / shard) as archive:
            for member in archive:
                if member.name.endswith(".png") and member.name[:-4] in by_key:
                    ordered.append(by_key[member.name[:-4]])
    if len(ordered) != len(rows) or len({row["sample_key"] for row in ordered}) != len(rows):
        raise RuntimeError("P3-E14 failed to recover unique physical tar order")
    return ordered


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E14 manifest output is not empty: {OUTPUT}")
    commitment = json.loads((DATA_ROOT / "commitment.json").read_text(encoding="utf-8"))
    validation = json.loads((DATA_ROOT / "validation.json").read_text(encoding="utf-8"))
    if commitment.get("commitment_sha256") != SOURCE_COMMITMENT:
        raise ValueError("P3-E14 style corpus commitment differs")
    if validation.get("status") != "passed":
        raise ValueError("P3-E14 style corpus validation did not pass")
    with (DATA_ROOT / "samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows_by_key = {row["sample_key"]: row for row in rows}
    if len(rows) != 100_000 or len(rows_by_key) != len(rows):
        raise RuntimeError("P3-E14 style source count differs")
    cache, cache_audit = p2e.load_cache()
    protected_molecules, protected_scaffolds, protected_audit = protected_identifiers(
        cache, rows_by_key
    )
    selected = physical_order(select_rows(rows, protected_molecules, protected_scaffolds))
    molecules = {row["molecule_id"] for row in selected}
    scaffolds = {row["scaffold"] for row in selected}
    if len(selected) != EXPECTED_IMAGES or len(molecules) != EXPECTED_MOLECULES:
        raise RuntimeError(
            f"P3-E14 selected count changed: {len(selected)}/{len(molecules)}"
        )
    lines = [
        json.dumps({
            "inchikey": row["inchikey"],
            "molecule_id": row["molecule_id"],
            "sample_key": row["sample_key"],
            "scaffold": row["scaffold"],
            "shard": str((DATA_ROOT / row["output_shard"]).resolve()),
            "variant": row["variant"],
        }, sort_keys=True)
        for row in selected
    ]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sample_digest = hashlib.sha256(
        "".join(f"{row['sample_key']}\n" for row in selected).encode()
    ).hexdigest()
    audit = {
        "schema_version": "g3-p3e14-style-count-manifest-0.1",
        "status": "passed",
        "selection": "exclude protected molecules and every protected scaffold",
        "samples": len(selected),
        "molecules": len(molecules),
        "scaffolds": len(scaffolds),
        "variants": sorted({row["variant"] for row in selected}),
        "shards": len({row["output_shard"] for row in selected}),
        "protected": protected_audit,
        "protected_molecule_overlap": len(molecules & protected_molecules),
        "protected_scaffold_overlap": len(scaffolds & protected_scaffolds),
        "sample_keys_sha256": sample_digest,
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": file_sha256(MANIFEST),
        "source_commitment": commitment,
        "candidate_cache_lineage": {
            "path": str(p2e.CACHE.resolve()),
            "sha256": p2e.CACHE_SHA256,
            "tensor_sha256": cache_audit["candidate_protocol"]["candidate_cache_sha256"],
        },
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
