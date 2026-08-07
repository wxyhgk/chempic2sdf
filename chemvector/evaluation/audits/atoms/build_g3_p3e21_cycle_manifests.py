#!/usr/bin/env python3
"""Build balanced molecule/scaffold-disjoint manifests for the P3-E21 cycle."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from chemvector.evaluation import g3_p3e21_cycle as protocol
from chemvector.evaluation.audits.atoms import (
    build_g3_p3e14_style_count_manifest as style_manifest,
)
from chemvector.training.atoms import train_g3_p2h_atom_query_set_capacity64 as p2h
from chemvector.training.atoms import train_g3_p3e14_style_count_scale as p3e14
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256


DATA_ROOT = Path("Data/sdf/chemvector_joint_v1/style_train_100k")
OUTPUT = protocol.OUTPUT / "manifests"
SOURCE_MANIFEST_SHA256 = "5a234b4984d045db635c0196ce481653574ff39897892db0330ab021a063dda1"


def allocate_scaffold_disjoint(
    groups: list[dict],
    *,
    split_counts: dict[str, int] = protocol.SPLIT_MOLECULES_PER_BUCKET,
) -> tuple[dict[str, list[dict]], dict]:
    by_bucket = defaultdict(list)
    for group in groups:
        by_bucket[group["bucket"]].append(group)
    scaffold_owner = {}
    used_molecules = set()
    selected = {split: [] for split in split_counts}
    for split in ("confirm", "dev", "train"):
        required = split_counts[split]
        for bucket, _, _ in protocol.COUNT_BUCKETS:
            accepted = []
            ordered = sorted(
                by_bucket[bucket],
                key=lambda value: (
                    hashlib.sha256(
                        f"{protocol.SEED}:{value['molecule_id']}".encode()
                    ).hexdigest(),
                    value["molecule_id"],
                ),
            )
            for group in ordered:
                molecule = group["molecule_id"]
                scaffold = group["scaffold"]
                owner = scaffold_owner.get(scaffold)
                if molecule in used_molecules or owner not in (None, split):
                    continue
                accepted.append(group)
                used_molecules.add(molecule)
                scaffold_owner[scaffold] = split
                if len(accepted) == required:
                    break
            if len(accepted) != required:
                raise RuntimeError(
                    f"P3-E21 cannot allocate {split}/{bucket}: {len(accepted)}"
                )
            selected[split].extend(accepted)
    scaffold_sets = {
        split: {group["scaffold"] for group in values}
        for split, values in selected.items()
    }
    overlaps = {
        f"{left}_{right}": len(scaffold_sets[left] & scaffold_sets[right])
        for left, right in (("train", "dev"), ("train", "confirm"), ("dev", "confirm"))
    }
    if any(overlaps.values()):
        raise RuntimeError("P3-E21 allocated cross-split scaffold overlap")
    return selected, {
        "scaffold_owner_count": len(scaffold_owner),
        "cross_split_scaffold_overlap": overlaps,
    }


def build_groups(evidence: list[dict], rows_by_key: dict[str, dict]) -> list[dict]:
    grouped = defaultdict(list)
    for record in evidence:
        grouped[p2h._molecule_key(str(record["sample_key"]))].append(record)
    groups = []
    for molecule, records in grouped.items():
        labels = {int(record["label"]) for record in records}
        variants = {p2h._variant(str(record["sample_key"])) for record in records}
        rows = [rows_by_key[str(record["sample_key"])] for record in records]
        scaffolds = {row["scaffold"] for row in rows}
        molecule_ids = {row["molecule_id"] for row in rows}
        inchikeys = {row["inchikey"] for row in rows}
        if (
            len(records) != 4
            or variants != protocol.VARIANTS
            or len(labels) != 1
            or len(scaffolds) != 1
            or len(molecule_ids) != 1
            or len(inchikeys) != 1
        ):
            raise ValueError(f"P3-E21 source molecule group differs: {molecule}")
        groups.append({
            "molecule_key": molecule,
            "molecule_id": next(iter(molecule_ids)),
            "inchikey": next(iter(inchikeys)),
            "scaffold": next(iter(scaffolds)),
            "count": next(iter(labels)),
            "bucket": protocol.count_bucket(next(iter(labels))),
            "sample_keys": {str(record["sample_key"]) for record in records},
        })
    return groups


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E21 manifest output is not empty: {OUTPUT}")
    if file_sha256(style_manifest.MANIFEST) != SOURCE_MANIFEST_SHA256:
        raise ValueError("P3-E21 source manifest SHA256 differs")
    evidence, evidence_audit = p3e14.load_style_evidence()
    with (DATA_ROOT / "samples.csv").open(newline="", encoding="utf-8") as handle:
        rows_by_key = {row["sample_key"]: row for row in csv.DictReader(handle)}
    source_lines = [
        json.loads(line)
        for line in style_manifest.MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_by_key = {row["sample_key"]: row for row in source_lines}
    groups = build_groups(evidence, rows_by_key)
    selected, allocation_audit = allocate_scaffold_disjoint(groups)
    split_by_key = {
        key: split
        for split, values in selected.items()
        for group in values
        for key in group["sample_keys"]
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for split in protocol.SPLIT_MOLECULES_PER_BUCKET:
        ordered = [row for row in source_lines if split_by_key.get(row["sample_key"]) == split]
        expected_images = (
            protocol.SPLIT_MOLECULES_PER_BUCKET[split]
            * len(protocol.COUNT_BUCKETS)
            * len(protocol.VARIANTS)
        )
        if len(ordered) != expected_images:
            raise RuntimeError(f"P3-E21 {split} image Count differs: {len(ordered)}")
        path = OUTPUT / f"{split}.jsonl"
        path.write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in ordered) + "\n",
            encoding="utf-8",
        )
        manifests[split] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "images": len(ordered),
            "molecules": len(selected[split]),
            "scaffolds": len({group["scaffold"] for group in selected[split]}),
            "bucket_molecules": dict(
                sorted(Counter(group["bucket"] for group in selected[split]).items())
            ),
            "sample_keys_sha256": hashlib.sha256(
                "".join(f"{row['sample_key']}\n" for row in ordered).encode()
            ).hexdigest(),
        }
    audit = {
        "schema_version": "g3-p3e21-cycle-manifests-0.1",
        "status": "passed",
        "cycle": protocol.CYCLE_NAME,
        "seed": protocol.SEED,
        "selection": "Count-balanced; molecule, InChIKey, and scaffold disjoint",
        "manifests": manifests,
        "allocation": allocation_audit,
        "source_manifest": str(style_manifest.MANIFEST.resolve()),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_evidence": evidence_audit,
        "fresh_dev_read": False,
        "sealed_test_read": False,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
