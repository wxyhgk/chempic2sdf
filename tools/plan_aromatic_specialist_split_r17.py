#!/usr/bin/env python3
"""Plan a scaffold-disjoint aromatic specialist slice (metadata only).

The source is the existing quarantine structure-expansion batch.  This tool
reads molecule/sample metadata only; it never opens PNG/SVG/SDF payloads,
loads a model, accesses sealed evaluation data, or authorizes training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from rdkit import Chem


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DEFAULT = ROOT / "outputs/datasets/structure-expansion-batch-12272-20260923"
R6_DEFAULT = ROOT / "outputs/datasets/chemvector-identity-aligned-line-first-targeted-diversity-r6"
V02_DEFAULT = ROOT / "outputs/datasets/curriculum-v0.2-expanded-20260919"
PROFILES = ("oa00", "oa11", "oa20", "oa33")
SPLITS = ("fresh_holdout", "dev", "train")
SIZES = {"fresh_holdout": 128, "dev": 64, "train": 256}
QUOTAS = {
    "fresh_holdout": {
        "aromatic_C": 32, "aromatic_N": 32, "aromatic_S": 16,
        "fused": 32, "heteroaromatic": 32, "control_SINGLE": 32,
        "control_DOUBLE": 24, "control_TRIPLE": 16,
    },
    "dev": {
        "aromatic_C": 16, "aromatic_N": 16, "aromatic_S": 8,
        "fused": 16, "heteroaromatic": 16, "control_SINGLE": 16,
        "control_DOUBLE": 12, "control_TRIPLE": 8,
    },
    "train": {
        "aromatic_C": 64, "aromatic_N": 64, "aromatic_S": 32,
        "fused": 64, "heteroaromatic": 64, "control_SINGLE": 64,
        "control_DOUBLE": 48, "control_TRIPLE": 32,
    },
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def nonstereo_connectivity(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid candidate SMILES: {smiles}")
    Chem.RemoveStereochemistry(mol)
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def group_key(row: dict) -> str:
    scaffold = str(row.get("scaffold") or "").strip()
    if scaffold:
        return "scaffold:" + scaffold
    return "acyclic:" + nonstereo_connectivity(row["canonical_smiles"])


def feature_set(smiles: str) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid candidate SMILES: {smiles}")
    aromatic = [atom for atom in mol.GetAtoms() if atom.GetIsAromatic()]
    features = set()
    for symbol in ("C", "N", "S"):
        if any(atom.GetSymbol() == symbol for atom in aromatic):
            features.add("aromatic_" + symbol)
    rings = mol.GetRingInfo().AtomRings()
    if any(len(set(left) & set(right)) >= 2
           for index, left in enumerate(rings) for right in rings[index + 1:]):
        features.add("fused")
    if any(atom.GetSymbol() != "C" for atom in aromatic):
        features.add("heteroaromatic")
    if not aromatic:
        bond_types = {str(bond.GetBondType()).upper() for bond in mol.GetBonds()}
        for bond_type in ("SINGLE", "DOUBLE", "TRIPLE"):
            if bond_type in bond_types:
                features.add("control_" + bond_type)
    return features


def stable(*values: str) -> str:
    return hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()


def old_rows(path: Path, name: str) -> list[dict]:
    if name == "r6":
        return [json.loads(line) for line in (path / "samples.jsonl").read_text().splitlines() if line.strip()]
    return json.loads((path / "molecules.json").read_text())


def choose(candidates: list[dict], split: str, used: set[str]) -> list[dict]:
    available = [row for row in candidates if row["group_key"] not in used]
    chosen: list[dict] = []
    quota = QUOTAS[split]

    def count(feature: str) -> int:
        return sum(feature in row["features"] for row in chosen)

    # Satisfy the rarest quota first.  A candidate that covers several still
    # deficient features is preferred; the hash is the sole tie-break.
    order = sorted(quota, key=lambda feature: sum(feature in row["features"] for row in available))
    for feature in order:
        while count(feature) < quota[feature]:
            options = [row for row in available if row not in chosen and feature in row["features"]]
            if not options:
                raise ValueError(f"Cannot satisfy {split} quota {feature}")
            def score(row: dict):
                gain = sum(count(other) < quota[other] and other in row["features"] for other in quota)
                return (gain, stable(split, feature, row["group_key"]))
            chosen.append(max(options, key=score))

    while len(chosen) < SIZES[split]:
        options = [row for row in available if row not in chosen]
        if not options:
            raise ValueError(f"Insufficient candidates for {split}")
        def score(row: dict):
            deficit_gain = sum(count(feature) < quota[feature] and feature in row["features"] for feature in quota)
            return (deficit_gain, stable(split, "fill", row["group_key"]))
        chosen.append(max(options, key=score))
    return chosen


def make_plan(source: Path, output: Path, r6: Path, v02: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    molecules = json.loads((source / "molecules.json").read_text())
    samples = json.loads((source / "samples.json").read_text())
    if any(row.get("split") != "quarantine" or row.get("training_authorized") is not False for row in molecules):
        raise ValueError("Source is not quarantine-only")
    by_molecule = {row["molecule_id"]: row for row in molecules}
    if len(by_molecule) != len(molecules):
        raise ValueError("Duplicate source molecule ID")
    by_samples: dict[tuple[str, str], dict] = {}
    for row in samples:
        key = (row["molecule_id"], row["profile_id"])
        if key in by_samples:
            raise ValueError("Duplicate source sample/profile")
        by_samples[key] = row

    blocked_smiles, blocked_connections = set(), set()
    for path, name in ((r6, "r6"), (v02, "v02")):
        for row in old_rows(path, name):
            smiles = row.get("canonical_smiles")
            if smiles:
                blocked_smiles.add(smiles)
                blocked_connections.add(nonstereo_connectivity(smiles))

    groups: dict[str, dict] = {}
    for molecule in molecules:
        smiles = molecule["canonical_smiles"]
        if smiles in blocked_smiles or nonstereo_connectivity(smiles) in blocked_connections:
            continue
        key = group_key(molecule)
        features = feature_set(smiles)
        candidate = {"group_key": key, "molecule_id": molecule["molecule_id"],
                     "canonical_smiles": smiles, "features": sorted(features),
                     "heavy_atoms": int(molecule.get("heavy_atoms", 0))}
        old = groups.get(key)
        if old is None or stable(candidate["molecule_id"]) < stable(old["molecule_id"]):
            groups[key] = candidate
    candidates = sorted(groups.values(), key=lambda row: stable("candidate", row["group_key"]))
    selected: dict[str, list[dict]] = {}
    used: set[str] = set()
    for split in SPLITS:
        selected[split] = choose(candidates, split, used)
        used.update(row["group_key"] for row in selected[split])

    rows = []
    for split in SPLITS:
        for molecule in selected[split]:
            for profile in PROFILES:
                sample = by_samples.get((molecule["molecule_id"], profile))
                if sample is None:
                    raise ValueError(f"Missing profile {profile} for {molecule['molecule_id']}")
                rows.append({"sample_key": sample["sample_key"], "molecule_id": molecule["molecule_id"],
                             "canonical_smiles": molecule["canonical_smiles"], "group_key": molecule["group_key"],
                             "profile_id": profile, "split": split, "features": molecule["features"],
                             "heavy_atoms": molecule["heavy_atoms"]})
    split_groups = {split: {row["group_key"] for row in selected[split]} for split in SPLITS}
    overlaps = {f"{left}:{right}": len(split_groups[left] & split_groups[right])
                for index, left in enumerate(SPLITS) for right in SPLITS[index + 1:]}
    counts = {split: {feature: sum(feature in row["features"] for row in selected[split])
                      for feature in QUOTAS[split]} for split in SPLITS}
    report = {
        "schema_version": "chemvector-aromatic-specialist-plan-r17-0.1",
        "status": "quarantine_plan_only", "training_authorized": False,
        "training_performed": False, "gpu_used": False, "model_forwards": 0,
        "optimizer_steps": 0, "sealed_test_read": False, "production_checkpoint_written": False,
        "source": {"path": str(source.relative_to(ROOT)), "molecule_count": len(molecules),
                    "sample_count": len(samples), "candidate_groups_after_block": len(candidates),
                    "blocked_canonical_count": len(blocked_smiles), "blocked_connectivity_count": len(blocked_connections),
                    "sha256": {str(path.relative_to(ROOT)): digest(path) for path in (
                        source / "molecules.json", source / "samples.json", source / "summary.json")}},
        "selection": {"profiles": list(PROFILES), "sizes": SIZES, "quotas": QUOTAS,
                       "groups_selected": len(used), "images_selected": len(rows)},
        "counts": counts, "identity_overlaps": overlaps,
        "gate": {"metadata_only": True, "source_quarantine": True,
                 "identity_disjoint": all(value == 0 for value in overlaps.values()),
                 "all_profiles_present": len(rows) == sum(SIZES.values()) * len(PROFILES),
                 "training_admissible": False, "next": "external identity admission then frozen CPU sidecar"},
        "rows": sorted(rows, key=lambda row: (row["split"], row["sample_key"])),
    }
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (output / "rows.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                                       for row in report["rows"]))
    print(json.dumps({key: report[key] for key in ("selection", "counts", "identity_overlaps", "gate")},
                     ensure_ascii=False, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r6", type=Path, default=R6_DEFAULT)
    parser.add_argument("--v02", type=Path, default=V02_DEFAULT)
    args = parser.parse_args()
    make_plan(args.source, args.output, args.r6, args.v02)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
