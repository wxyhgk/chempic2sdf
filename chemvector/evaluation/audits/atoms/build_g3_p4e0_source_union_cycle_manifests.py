#!/usr/bin/env python3
"""Build leakage-free pilot/full/confirm_v2 manifests for the G3 P4 cycle."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import tarfile

from chemvector.evaluation import g3_p4_source_union_cycle as cycle
from chemvector.evaluation.audits.atoms import (
    build_g3_p3e21_shared_cache as p3e21,
)
from chemvector.training.graph.scenes.train_g3_hierarchical_graph import _atomic_json
from chemvector.g3_structural_data import file_sha256


AUDIT_NAME = "chemvector-g3-p4e0-source-union-cycle-manifests"
OUTPUT = Path(f"outputs/audits/{AUDIT_NAME}")
DATA_ROOT = Path("Data/sdf/chemvector_joint_v1/style_train_100k")
STYLE_COMMITMENT = (
    "2082372d253fa309715becfd5cba9e5cdb78eb8d84a5ece4ff1eb73bd67390eb"
)
SAMPLES_CSV_SHA256 = (
    "f38a8d10a9ff558d646d2a8340354347022e905f95ff9ea73f29fc7088bfa26a"
)
MOLECULES_CSV_SHA256 = (
    "4c8de7aaf8ddbaad47199a9436057b5d545a1bd76e827daa2d7e3673e21c0685"
)


def _rank(seed: int, split: str, molecule_id: str) -> str:
    return hashlib.sha256(f"{seed}:{split}:{molecule_id}".encode()).hexdigest()


def allocate_cycle_molecules(
    molecules: list[dict],
    *,
    blocked_molecules: set[str],
    blocked_scaffolds: set[str],
) -> tuple[dict[str, list[dict]], dict]:
    eligible = [
        row
        for row in molecules
        if row["molecule_id"] not in blocked_molecules
        and row["scaffold"] not in blocked_scaffolds
        and 1 <= int(row["heavy_atoms"]) <= 100
    ]
    by_bucket = defaultdict(list)
    for row in eligible:
        by_bucket[cycle.count_bucket(int(row["heavy_atoms"]))].append(row)

    confirm = []
    confirm_scaffolds = set()
    for bucket, required in cycle.CONFIRM_V2_BUCKET_MOLECULES.items():
        if required == 0:
            continue
        ordered = sorted(
            by_bucket[bucket],
            key=lambda row: (
                _rank(cycle.SEED, "confirm_v2", row["molecule_id"]),
                row["molecule_id"],
            ),
        )
        selected = []
        for row in ordered:
            if row["scaffold"] in confirm_scaffolds:
                continue
            selected.append(row)
            confirm_scaffolds.add(row["scaffold"])
            if len(selected) == required:
                break
        if len(selected) != required:
            raise RuntimeError(
                f"P4-E0 cannot allocate confirm_v2/{bucket}: {len(selected)}"
            )
        confirm.extend(selected)

    confirm_molecules = {row["molecule_id"] for row in confirm}
    train_eligible = [
        row
        for row in eligible
        if row["molecule_id"] not in confirm_molecules
        and row["scaffold"] not in confirm_scaffolds
    ]
    full_train = sorted(
        train_eligible,
        key=lambda row: (
            _rank(cycle.SEED, "full_train", row["molecule_id"]),
            row["molecule_id"],
        ),
    )[: cycle.FULL_TRAIN_MOLECULES]
    if len(full_train) != cycle.FULL_TRAIN_MOLECULES:
        raise RuntimeError("P4-E0 cannot allocate full_train")
    pilot_train = sorted(
        full_train,
        key=lambda row: (
            _rank(cycle.SEED, "pilot_train", row["molecule_id"]),
            row["molecule_id"],
        ),
    )[: cycle.PILOT_TRAIN_MOLECULES]
    return {
        "pilot_train": pilot_train,
        "full_train": full_train,
        "confirm_v2": confirm,
    }, {
        "source_molecules": len(molecules),
        "blocked_molecules": len(blocked_molecules),
        "blocked_scaffolds": len(blocked_scaffolds),
        "eligible_molecules": len(eligible),
        "eligible_scaffolds": len({row["scaffold"] for row in eligible}),
        "confirm_scaffolds": len(confirm_scaffolds),
        "train_eligible_after_confirm": len(train_eligible),
    }


def physical_order(rows: list[dict]) -> list[dict]:
    by_key = {row["sample_key"]: row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("P4-E0 selected duplicate sample keys")
    ordered = []
    shards = list(dict.fromkeys(row["output_shard"] for row in rows))
    for shard in shards:
        with tarfile.open(DATA_ROOT / shard) as archive:
            for member in archive:
                if member.name.endswith(".png") and member.name[:-4] in by_key:
                    ordered.append(by_key[member.name[:-4]])
    if len(ordered) != len(rows) or len({row["sample_key"] for row in ordered}) != len(rows):
        raise RuntimeError("P4-E0 failed to recover physical tar order")
    return ordered


def _load_blocked() -> tuple[set[str], set[str], dict]:
    molecules = set()
    scaffolds = set()
    audits = {}
    for split, (sha256, images, split_molecules) in p3e21.MANIFESTS.items():
        path = p3e21.MANIFEST_ROOT / f"{split}.jsonl"
        if file_sha256(path) != sha256:
            raise ValueError(f"P4-E0 P3-E21 {split} manifest SHA256 differs")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        if len(rows) != images or len({row["molecule_id"] for row in rows}) != split_molecules:
            raise ValueError(f"P4-E0 P3-E21 {split} identity count differs")
        molecules.update(row["molecule_id"] for row in rows)
        scaffolds.update(row["scaffold"] for row in rows)
        audits[split] = {
            "path": str(path.resolve()),
            "sha256": sha256,
            "images": images,
            "molecules": split_molecules,
        }
    return molecules, scaffolds, audits


def _manifest_line(row: dict) -> str:
    return json.dumps(
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


def main() -> None:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"P4-E0 output is not empty: {OUTPUT}")
    for path, expected in (
        (DATA_ROOT / "samples.csv", SAMPLES_CSV_SHA256),
        (DATA_ROOT / "molecules.csv", MOLECULES_CSV_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"P4-E0 source SHA256 differs: {path}")
    commitment = json.loads((DATA_ROOT / "commitment.json").read_text())
    if commitment.get("commitment_sha256") != STYLE_COMMITMENT:
        raise ValueError("P4-E0 style commitment differs")
    blocked_molecules, blocked_scaffolds, blocked_audit = _load_blocked()
    with (DATA_ROOT / "molecules.csv").open(newline="", encoding="utf-8") as handle:
        molecules = list(csv.DictReader(handle))
    with (DATA_ROOT / "samples.csv").open(newline="", encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    allocated, allocation_audit = allocate_cycle_molecules(
        molecules,
        blocked_molecules=blocked_molecules,
        blocked_scaffolds=blocked_scaffolds,
    )
    sample_by_molecule = defaultdict(list)
    for row in samples:
        sample_by_molecule[row["molecule_id"]].append(row)

    OUTPUT.mkdir(parents=True, exist_ok=False)
    manifests = {}
    split_molecule_sets = {}
    split_scaffold_sets = {}
    for split, selected_molecules in allocated.items():
        molecule_ids = {row["molecule_id"] for row in selected_molecules}
        rows = [row for molecule in molecule_ids for row in sample_by_molecule[molecule]]
        expected_images = len(molecule_ids) * len(cycle.VARIANTS)
        if (
            len(rows) != expected_images
            or {row["variant"] for row in rows} != cycle.VARIANTS
            or any(len(sample_by_molecule[molecule]) != 4 for molecule in molecule_ids)
        ):
            raise RuntimeError(f"P4-E0 {split} four-variant contract differs")
        ordered = physical_order(rows)
        path = OUTPUT / f"{split}.jsonl"
        path.write_text(
            "\n".join(_manifest_line(row) for row in ordered) + "\n",
            encoding="utf-8",
        )
        split_molecule_sets[split] = molecule_ids
        split_scaffold_sets[split] = {
            row["scaffold"] for row in selected_molecules
        }
        manifests[split] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "images": len(ordered),
            "molecules": len(molecule_ids),
            "scaffolds": len(split_scaffold_sets[split]),
            "variants": dict(sorted(Counter(row["variant"] for row in ordered).items())),
            "stress_operations": dict(
                sorted(Counter(row["stress_operation"] or "none" for row in ordered).items())
            ),
            "count_buckets": dict(
                sorted(
                    Counter(
                        cycle.count_bucket(int(row["heavy_atoms"]))
                        for row in selected_molecules
                    ).items()
                )
            ),
            "sample_keys_sha256": hashlib.sha256(
                "".join(f"{row['sample_key']}\n" for row in ordered).encode()
            ).hexdigest(),
        }

    checks = {
        "pilot_subset_full": split_molecule_sets["pilot_train"] <= split_molecule_sets["full_train"],
        "train_confirm_molecule_disjoint": not (
            split_molecule_sets["full_train"] & split_molecule_sets["confirm_v2"]
        ),
        "train_confirm_scaffold_disjoint": not (
            split_scaffold_sets["full_train"] & split_scaffold_sets["confirm_v2"]
        ),
        "all_cycle_molecules_blocked_disjoint": all(
            not (values & blocked_molecules) for values in split_molecule_sets.values()
        ),
        "all_cycle_scaffolds_blocked_disjoint": all(
            not (values & blocked_scaffolds) for values in split_scaffold_sets.values()
        ),
        "pilot_images_20000": manifests["pilot_train"]["images"] == 20_000,
        "full_images_80000": manifests["full_train"]["images"] == 80_000,
        "confirm_v2_images_700": manifests["confirm_v2"]["images"] == 700,
    }
    audit = {
        "schema_version": "g3-p4e0-source-union-cycle-manifests-audit-0.1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "cycle": cycle.CYCLE_NAME,
        "seed": cycle.SEED,
        "selection": {
            "blocked": "all P3-E21 train/dev/confirm molecule and scaffold identities",
            "confirm_v2": "Count-stratified, unique scaffold, SHA-ranked",
            "full_train": "confirm/P3-E21-disjoint, SHA-ranked",
            "pilot_train": "deterministic molecule subset of full_train",
        },
        "manifests": manifests,
        "allocation": allocation_audit,
        "blocked_inputs": blocked_audit,
        "source": {
            "root": str(DATA_ROOT.resolve()),
            "commitment_sha256": STYLE_COMMITMENT,
            "samples_csv_sha256": SAMPLES_CSV_SHA256,
            "molecules_csv_sha256": MOLECULES_CSV_SHA256,
        },
        "data_access": {
            "sealed_test_read": False,
            "fresh_dev_read": False,
            "uspto_read": False,
            "mr_tadf_read": False,
        },
    }
    _atomic_json(OUTPUT / "audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
