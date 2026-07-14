#!/usr/bin/env python3
"""Audit MR-TADF SDF files before any image rendering is allowed."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


APPROVED_FIELDS = (
    "molecule_id",
    "sdf_file",
    "inchikey",
    "canonical_isomeric_smiles",
    "scaffold",
    "atoms",
    "heavy_atoms",
    "bonds",
    "rings",
    "molecular_weight",
    "elements",
    "maximum_bond_length",
    "median_bond_length",
    "maximum_to_median_bond_ratio",
)

REJECTED_FIELDS = (
    "molecule_id",
    "sdf_file",
    "reason",
    "detail",
    "inchikey",
)


def load_training_identity(manifest: Path | None) -> tuple[set[str], set[str]]:
    if manifest is None:
        return set(), set()
    inchikeys: set[str] = set()
    scaffolds: set[str] = set()
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("inchikey"):
                inchikeys.add(row["inchikey"])
            if row.get("scaffold"):
                scaffolds.add(row["scaffold"])
    return inchikeys, scaffolds


def record_count(path: Path) -> int:
    separators = sum(
        line.strip() == "$$$$"
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
    )
    return separators if separators else 1


def bond_length_statistics(mol: Chem.Mol) -> tuple[float, float, float]:
    conformer = mol.GetConformer()
    lengths = []
    for bond in mol.GetBonds():
        begin = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
        end = conformer.GetAtomPosition(bond.GetEndAtomIdx())
        lengths.append(
            math.dist(
                (begin.x, begin.y, begin.z),
                (end.x, end.y, end.z),
            )
        )
    lengths.sort()
    maximum = max(lengths, default=0.0)
    median = lengths[len(lengths) // 2] if lengths else 0.0
    ratio = maximum / median if median else 0.0
    return maximum, median, ratio


def reject(
    path: Path,
    reason: str,
    detail: str = "",
    *,
    inchikey: str = "",
) -> dict[str, str]:
    return {
        "molecule_id": path.stem,
        "sdf_file": path.name,
        "reason": reason,
        "detail": detail,
        "inchikey": inchikey,
    }


def inspect_sdf(
    path: Path,
    training_inchikeys: set[str],
    training_scaffolds: set[str],
) -> tuple[dict[str, str | int | float] | None, dict[str, str] | None]:
    if record_count(path) != 1:
        return None, reject(path, "multiple_records")
    try:
        mol = Chem.MolFromMolFile(
            str(path),
            sanitize=False,
            removeHs=False,
            strictParsing=False,
        )
    except Exception as error:
        return None, reject(path, "read_exception", str(error))
    if mol is None:
        return None, reject(path, "unreadable")
    if mol.GetNumAtoms() == 0:
        return None, reject(path, "zero_atoms")
    try:
        Chem.SanitizeMol(mol)
    except Exception as error:
        return None, reject(path, "sanitize_failed", str(error))
    if len(Chem.GetMolFrags(mol)) != 1:
        return None, reject(path, "multiple_components")
    if mol.GetNumConformers() != 1:
        return None, reject(path, "missing_or_multiple_conformers")

    conformer = mol.GetConformer()
    coordinates = [
        conformer.GetAtomPosition(index) for index in range(mol.GetNumAtoms())
    ]
    maximum_abs_z = max(abs(point.z) for point in coordinates)
    x_span = max(point.x for point in coordinates) - min(point.x for point in coordinates)
    y_span = max(point.y for point in coordinates) - min(point.y for point in coordinates)
    if maximum_abs_z > 1e-4 or x_span <= 1e-5 or y_span <= 1e-5:
        return None, reject(
            path,
            "not_strictly_2d",
            f"max_abs_z={maximum_abs_z:.6g},x_span={x_span:.6g},y_span={y_span:.6g}",
        )

    inchikey = Chem.MolToInchiKey(mol)
    if not inchikey:
        return None, reject(path, "empty_inchikey")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=True,
    )
    if inchikey in training_inchikeys:
        return None, reject(path, "training_structure_overlap", inchikey, inchikey=inchikey)
    if scaffold and scaffold in training_scaffolds:
        return None, reject(path, "training_scaffold_overlap", scaffold, inchikey=inchikey)

    elements = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    maximum, median, ratio = bond_length_statistics(mol)
    approved: dict[str, str | int | float] = {
        "molecule_id": path.stem,
        "sdf_file": path.name,
        "inchikey": inchikey,
        "canonical_isomeric_smiles": Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        ),
        "scaffold": scaffold,
        "atoms": mol.GetNumAtoms(),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "bonds": mol.GetNumBonds(),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "molecular_weight": round(Descriptors.MolWt(mol), 6),
        "elements": ";".join(
            f"{element}:{count}" for element, count in sorted(elements.items())
        ),
        "maximum_bond_length": round(maximum, 6),
        "median_bond_length": round(median, 6),
        "maximum_to_median_bond_ratio": round(ratio, 6),
    }
    return approved, None


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="Data/MR-TADF_SDF")
    parser.add_argument(
        "--training-manifest",
        default="Data/sdf/webdataset_full_v0/molecules.csv",
    )
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    RDLogger.DisableLog("rdApp.*")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "audit"
    training_manifest = Path(args.training_manifest) if args.training_manifest else None
    training_inchikeys, training_scaffolds = load_training_identity(training_manifest)

    # Only top-level SDF files are candidates. Hidden notebook checkpoints and
    # any later generated directories are never eligible for rendering.
    candidates = sorted(input_dir.glob("*.sdf"))
    approved = []
    rejected = []
    for path in candidates:
        row, failure = inspect_sdf(path, training_inchikeys, training_scaffolds)
        if row is not None:
            approved.append(row)
        if failure is not None:
            rejected.append(failure)

    by_inchikey: dict[str, list[dict]] = defaultdict(list)
    for row in approved:
        by_inchikey[str(row["inchikey"])].append(row)
    unique_approved = []
    for rows in by_inchikey.values():
        rows.sort(key=lambda row: str(row["sdf_file"]))
        unique_approved.append(rows[0])
        for duplicate in rows[1:]:
            rejected.append(
                reject(
                    input_dir / str(duplicate["sdf_file"]),
                    "duplicate_inchikey",
                    f"kept={rows[0]['sdf_file']}",
                    inchikey=str(duplicate["inchikey"]),
                )
            )
    approved = sorted(unique_approved, key=lambda row: str(row["sdf_file"]))
    rejected.sort(key=lambda row: (row["sdf_file"], row["reason"]))

    reason_counts = Counter(row["reason"] for row in rejected)
    nested_sdf_files = sorted(set(input_dir.rglob("*.sdf")) - set(candidates))
    candidate_digest = hashlib.sha256(
        "\n".join(path.name for path in candidates).encode("utf-8")
    ).hexdigest()
    summary = {
        "schema_version": "mr-tadf-audit-0.1",
        "input_dir": str(input_dir),
        "training_manifest": str(training_manifest) if training_manifest else None,
        "candidate_files": len(candidates),
        "approved_unique_structures": len(approved),
        "rejected_files": len(rejected),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "training_structure_overlaps": reason_counts["training_structure_overlap"],
        "training_scaffold_overlaps": reason_counts["training_scaffold_overlap"],
        "nested_sdf_files_ignored": len(nested_sdf_files),
        "approved_long_bond_ratio_at_least_1_75": sum(
            float(row["maximum_to_median_bond_ratio"]) >= 1.75
            for row in approved
        ),
        "candidate_filename_sha256": candidate_digest,
        "rendering_policy": "Only rows in approved.csv may be rendered.",
        "png_files_generated": 0,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "approved.csv", APPROVED_FIELDS, approved)
    write_csv(output_dir / "rejected.csv", REJECTED_FIELDS, rejected)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
