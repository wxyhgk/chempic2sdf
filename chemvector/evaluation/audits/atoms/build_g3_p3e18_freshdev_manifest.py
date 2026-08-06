#!/usr/bin/env python3
"""Build the sealed-test-free fresh_dev manifest for the P3-E18 audit."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import tarfile

from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256


DATA_ROOT = Path("Data/sdf/chemvector_joint_v1/fresh_eval_12k")
SOURCE_COMMITMENT = (
    "13a456afb2c475a73fc4dca82df5fe62d4a31277d6b584637d9cd92736024512"
)
SAMPLES_CSV_SHA256 = (
    "b46571e188620e8e117e294648ae1a5a936cc8e0958c88da6ad830af0e9d0619"
)
OUTPUT = Path("outputs/audits/chemvector-g3-p3e18-freshdev8k-manifest")
MANIFEST = OUTPUT / "manifest.jsonl"
EXPECTED_IMAGES = 8_000
EXPECTED_MOLECULES = 2_000
EXPECTED_VARIANTS = frozenset({"coordgen", "original", "rdkit", "stress"})
EXPECTED_SPLIT = "fresh_dev"
EXPECTED_FONT_GROUP = "heldout"


def validate_fresh_rows(rows: list[dict]) -> dict:
    if len(rows) != EXPECTED_IMAGES:
        raise ValueError(f"P3-E18 fresh_dev image count differs: {len(rows)}")
    keys = [row["sample_key"] for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("P3-E18 fresh_dev contains duplicate sample keys")
    if {row["split"] for row in rows} != {EXPECTED_SPLIT}:
        raise ValueError("P3-E18 source includes a non-fresh_dev split")
    if {row["font_family_group"] for row in rows} != {EXPECTED_FONT_GROUP}:
        raise ValueError("P3-E18 source includes a non-held-out font group")
    if {row["variant"] for row in rows} != EXPECTED_VARIANTS:
        raise ValueError("P3-E18 source variants differ")
    groups = defaultdict(list)
    for row in rows:
        groups[row["molecule_id"]].append(row)
    if len(groups) != EXPECTED_MOLECULES:
        raise ValueError(f"P3-E18 fresh_dev molecule count differs: {len(groups)}")
    if any(
        len(group) != len(EXPECTED_VARIANTS)
        or {row["variant"] for row in group} != EXPECTED_VARIANTS
        for group in groups.values()
    ):
        raise ValueError("P3-E18 molecule does not contain exactly four variants")
    return {
        "images": len(rows),
        "molecules": len(groups),
        "variants": dict(sorted(Counter(row["variant"] for row in rows).items())),
        "font_family_groups": dict(
            sorted(Counter(row["font_family_group"] for row in rows).items())
        ),
    }


def physical_order(rows: list[dict]) -> list[dict]:
    by_key = {row["sample_key"]: row for row in rows}
    shard_names = list(dict.fromkeys(row["output_shard"] for row in rows))
    if not shard_names or any(not name.startswith("shards/fresh_dev-") for name in shard_names):
        raise ValueError("P3-E18 source references an unexpected shard path")
    ordered = []
    for shard_name in shard_names:
        with tarfile.open(DATA_ROOT / shard_name) as archive:
            for member in archive:
                if member.name.endswith(".png") and member.name[:-4] in by_key:
                    ordered.append(by_key[member.name[:-4]])
    if len(ordered) != len(rows) or len({row["sample_key"] for row in ordered}) != len(rows):
        raise RuntimeError("P3-E18 failed to recover unique physical tar order")
    return ordered


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P3-E18 manifest output is not empty: {OUTPUT}")
    commitment = json.loads((DATA_ROOT / "commitment.json").read_text(encoding="utf-8"))
    validation = json.loads((DATA_ROOT / "validation.json").read_text(encoding="utf-8"))
    if commitment.get("commitment_sha256") != SOURCE_COMMITMENT:
        raise ValueError("P3-E18 fresh_eval commitment differs")
    if file_sha256(DATA_ROOT / "samples.csv") != SAMPLES_CSV_SHA256:
        raise ValueError("P3-E18 fresh_dev samples.csv SHA256 differs")
    if validation.get("status") != "passed":
        raise ValueError("P3-E18 fresh_eval validation did not pass")
    with (DATA_ROOT / "samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    source_audit = validate_fresh_rows(rows)
    ordered = physical_order(rows)
    lines = [
        json.dumps(
            {
                "inchikey": row["inchikey"],
                "molecule_id": row["molecule_id"],
                "sample_key": row["sample_key"],
                "scaffold": row["scaffold"],
                "shard": str((DATA_ROOT / row["output_shard"]).resolve()),
                "variant": row["variant"],
            },
            sort_keys=True,
        )
        for row in ordered
    ]
    OUTPUT.mkdir(parents=True, exist_ok=False)
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sample_keys_sha256 = hashlib.sha256(
        "".join(f"{row['sample_key']}\n" for row in ordered).encode()
    ).hexdigest()
    audit = {
        "schema_version": "g3-p3e18-freshdev-manifest-0.1",
        "status": "passed",
        "selection": "all fresh_dev rows in physical tar order",
        **source_audit,
        "shards": [
            str((DATA_ROOT / name).resolve())
            for name in dict.fromkeys(row["output_shard"] for row in ordered)
        ],
        "sample_keys_sha256": sample_keys_sha256,
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": file_sha256(MANIFEST),
        "source_samples_csv_sha256": SAMPLES_CSV_SHA256,
        "source_commitment": commitment,
        "sealed_test_read": False,
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
