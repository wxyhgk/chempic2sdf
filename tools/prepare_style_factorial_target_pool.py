"""Prepare a deterministic, identity-disjoint style-factorial train pool.

Only committed CSV/JSON metadata is read.  No raster, sealed payload, model,
or optimizer is touched.  The output is a manifest proposal; training remains
unauthorized until a frozen-model reachability audit admits it.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data/sdf/chemvector_joint_v1/style_train_100k"
SAMPLES = DATA / "samples.csv"
MOLECULES = DATA / "molecules.csv"
COMMITMENT = DATA / "commitment.json"
CURRENT = ROOT / "outputs/datasets/chemvector-targeted-selector-visual-pool-release-20261006-r1/manifest.json"
V02_ROWS = ROOT / "outputs/audits/curriculum-v02-release-preparation-20260919/sample-split-proposal.json"
OUTPUT = ROOT / "outputs/audits/chemvector-style-factorial-target-pool-20261007-r1"
TRAIN_MOLECULES = 1600
DEV_MOLECULES = 400
SEED = "chemvector-style-factorial-20261007-r1:"
CELLS = (("N", "SINGLE"), ("N", "DOUBLE"), ("N", "TRIPLE"), ("N", "AROMATIC"),
         ("O", "SINGLE"), ("O", "DOUBLE"), ("O", "TRIPLE"), ("O", "AROMATIC"))
VARIANTS = ("original", "rdkit", "coordgen", "stress")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cells(smiles: str) -> set[tuple[str, str]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    found: set[tuple[str, str]] = set()
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in {"N", "O"}:
            continue
        for bond in atom.GetBonds():
            kind = "AROMATIC" if bond.GetIsAromatic() else str(bond.GetBondType()).upper()
            if kind in {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}:
                found.add((atom.GetSymbol(), kind))
    return found


def rank(molecule_id: str, cell: tuple[str, str]) -> str:
    return hashlib.sha256((SEED + "/".join(cell) + ":" + molecule_id).encode()).hexdigest()


def run() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    commitment = json.loads(COMMITMENT.read_text(encoding="utf-8"))
    if commitment.get("dataset") != "style_train_100k" or commitment.get("split_counts", {}).get("train") != commitment.get("samples"):
        raise ValueError("source commitment is not train-only style_train_100k")
    samples = list(csv.DictReader(SAMPLES.open(encoding="utf-8")))
    molecules = list(csv.DictReader(MOLECULES.open(encoding="utf-8")))
    by_molecule: dict[str, dict] = {r["molecule_id"]: r for r in molecules}
    variants: dict[str, set[str]] = defaultdict(set)
    for row in samples:
        variants[row["molecule_id"]].add(row["variant"])
    eligible = []
    for molecule_id, row in by_molecule.items():
        if variants[molecule_id] != set(VARIANTS):
            continue
        molecule_cells = cells(row["canonical_isomeric_smiles"])
        eligible.append({"molecule_id": molecule_id, "inchikey": row["inchikey"],
                         "scaffold": row["scaffold"], "smiles": row["canonical_isomeric_smiles"],
                         "cells": sorted([list(c) for c in molecule_cells]),
                         "cell_set": molecule_cells})
    current = json.loads(CURRENT.read_text(encoding="utf-8"))["samples"]
    old_ids = {str(r.get("molecule_id")) for r in current}
    old_inchikeys = {str(r.get("inchikey")) for r in current}
    old_scaffolds = {str(r.get("scaffold")) for r in current}
    v02 = json.loads(V02_ROWS.read_text(encoding="utf-8"))
    old_ids |= {str(r.get("molecule_id")) for r in v02}
    old_inchikeys |= {str(r.get("inchikey")) for r in v02}
    old_scaffolds |= {str(r.get("scaffold")) for r in v02}
    eligible_before_identity_filter = list(eligible)
    overlap_counts = {
        "molecule_id": len({r["molecule_id"] for r in eligible_before_identity_filter} & old_ids),
        "inchikey": len({r["inchikey"] for r in eligible_before_identity_filter} & old_inchikeys),
        "scaffold": len({r["scaffold"] for r in eligible_before_identity_filter} & old_scaffolds),
    }
    eligible = [r for r in eligible_before_identity_filter if r["molecule_id"] not in old_ids
                and r["inchikey"] not in old_inchikeys and r["scaffold"] not in old_scaffolds]
    selected: dict[str, list[dict]] = {"train": [], "dev": []}
    used: set[str] = set()
    # Reserve a balanced quota for every N/O-bond cell, then deterministic fill.
    for split, target in (("train", TRAIN_MOLECULES), ("dev", DEV_MOLECULES)):
        chosen: dict[str, dict] = {}
        quota = 120 if split == "train" else 30
        for cell in CELLS:
            options = sorted((r for r in eligible if cell in r["cell_set"] and r["molecule_id"] not in used and r["molecule_id"] not in chosen),
                             key=lambda r: rank(r["molecule_id"], cell))
            for row in options[:quota]:
                chosen[row["molecule_id"]] = row
        pool = sorted((r for r in eligible if r["molecule_id"] not in used and r["molecule_id"] not in chosen),
                      key=lambda r: rank(r["molecule_id"], (split, "fill")))
        for row in pool:
            if len(chosen) >= target:
                break
            chosen[row["molecule_id"]] = row
        if len(chosen) != target:
            raise ValueError(f"unable to select {target} {split} molecules; got {len(chosen)}")
        selected[split] = list(chosen.values())
        used |= set(chosen)
    train_ids, dev_ids = ({r["molecule_id"] for r in selected["train"]}, {r["molecule_id"] for r in selected["dev"]})
    if train_ids & dev_ids:
        raise ValueError("identity overlap")
    sample_by_mol = defaultdict(list)
    for row in samples:
        sample_by_mol[row["molecule_id"]].append(row)
    rows_out: dict[str, list[dict]] = {"train": [], "dev": []}
    cell_counts = {split: Counter() for split in selected}
    for split, group in selected.items():
        for molecule in sorted(group, key=lambda r: r["molecule_id"]):
            for cell in molecule["cell_set"]:
                cell_counts[split]["/".join(cell)] += 1
            for row in sorted(sample_by_mol[molecule["molecule_id"]], key=lambda r: r["variant"]):
                rows_out[split].append({"sample_key": row["sample_key"], "molecule_id": row["molecule_id"],
                    "inchikey": row["inchikey"], "scaffold": row["scaffold"], "variant": row["variant"],
                    "profile": {"font_family_id": row["font_family_id"], "font_family_group": row["font_family_group"],
                                "font_bucket": row["font_bucket"], "line_width_bucket": row["line_width_bucket"],
                                "rotation_bucket": row["rotation_bucket"], "multiple_bond_offset_bucket": row["multiple_bond_offset_bucket"],
                                "color_mode": row["color_mode"]}, "output_shard": row["output_shard"]})
    report = {
        "schema_version": "chemvector-style-factorial-target-pool-0.1", "complete": True,
        "read_only": True, "training_performed": False, "training_authorized": False,
        "gpu_used": False, "sealed_test_read": False, "deployment_authorized": False,
        "source": {"dataset": "style_train_100k", "commitment_sha256": digest(COMMITMENT),
                   "samples_sha256": digest(SAMPLES), "molecules_sha256": digest(MOLECULES)},
        "source_counts": {"samples": len(samples), "molecules": len(by_molecule), "eligible_after_identity_exclusion": len(eligible)},
        "excluded_identity_overlap": overlap_counts,
        "split": {"train_molecules": len(train_ids), "dev_molecules": len(dev_ids),
                   "train_samples": len(rows_out["train"]), "dev_samples": len(rows_out["dev"]),
                   "molecule_overlap": len(train_ids & dev_ids),
                   "cell_counts": {k: dict(v) for k, v in cell_counts.items()}},
        "gate": {"four_variants_per_molecule": True, "identity_disjoint": not bool(train_ids & dev_ids),
                 "old_identity_disjoint": True, "training_admissible": False,
                 "next": "frozen initial endpoint reachability replay; no training until raw/selected coverage is measured"},
    }
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "train-manifest.json").write_text(json.dumps({"schema_version": report["schema_version"], "split": "train", "rows": rows_out["train"]}, indent=2, ensure_ascii=False) + "\n")
    (OUTPUT / "dev-manifest.json").write_text(json.dumps({"schema_version": report["schema_version"], "split": "dev", "rows": rows_out["dev"]}, indent=2, ensure_ascii=False) + "\n")
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), "split": report["split"], "gate": report["gate"]}, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run()
