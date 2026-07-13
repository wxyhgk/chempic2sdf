#!/usr/bin/env python3
"""Filter and reproducibly sample the downloaded 2D SDF datasets."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator

from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdMolDescriptors


ALLOWED_ELEMENTS = {"B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I", "Se"}
ALLOWED_BONDS = {
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
}
ELEMENT_TARGET_FRACTIONS = {
    "B": 0.06,
    "Se": 0.04,
    "Si": 0.04,
    "P": 0.06,
    "S": 0.15,
    "halogen": 0.20,
    "base": 0.45,
}


@dataclass(slots=True)
class Candidate:
    molecule: Chem.Mol
    source: str
    source_id: str
    smiles: str
    element_group: str
    size_bin: str
    feature_bin: str
    heavy_atoms: int
    rings: int
    chiral_centers: int
    formal_charge: int
    complexity_score: float


class Reservoir:
    def __init__(self, limit: int, rng: random.Random) -> None:
        self.limit = limit
        self.rng = rng
        self.seen = 0
        self.items: list[Candidate] = []

    def add(self, item: Candidate) -> None:
        self.seen += 1
        if len(self.items) < self.limit:
            self.items.append(item)
            return
        position = self.rng.randrange(self.seen)
        if position < self.limit:
            self.items[position] = item


def open_sdf(path: Path) -> tuple[BinaryIO, object | None]:
    if path.suffix == ".gz":
        return gzip.open(path, "rb"), None
    if path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        members = [name for name in archive.namelist() if name.lower().endswith((".sdf", ".sd"))]
        if len(members) != 1:
            archive.close()
            raise ValueError(f"Expected exactly one SDF in {path}, found {members}")
        return archive.open(members[0], "r"), archive
    return path.open("rb"), None


def iter_molecules(path: Path) -> Iterator[Chem.Mol | None]:
    stream, owner = open_sdf(path)
    try:
        supplier = Chem.ForwardSDMolSupplier(
            stream,
            sanitize=True,
            removeHs=False,
            strictParsing=False,
        )
        yield from supplier
    finally:
        stream.close()
        if owner is not None:
            owner.close()


def size_bin(heavy_atoms: int) -> str:
    if heavy_atoms < 20:
        return "10-19"
    if heavy_atoms < 40:
        return "20-39"
    if heavy_atoms < 60:
        return "40-59"
    return "60-100"


def element_group(elements: set[str]) -> str:
    for symbol in ("Se", "B", "Si", "P", "S"):
        if symbol in elements:
            return symbol
    if elements & {"F", "Cl", "Br", "I"}:
        return "halogen"
    return "base"


def feature_bin(charge: int, chiral_centers: int, rings: int) -> str:
    labels = []
    if charge:
        labels.append("charged")
    if chiral_centers:
        labels.append("stereo")
    if rings >= 4:
        labels.append("polycyclic")
    return "+".join(labels) if labels else "standard"


def source_id(mol: Chem.Mol, source: str, index: int) -> str:
    preferred = {
        "chembl": ("chembl_id", "CHEMBL_ID"),
        "coconut": ("coconut_id", "COCONUT_ID", "identifier"),
        "chebi": ("ChEBI ID", "CHEBI_ID", "chebi_id"),
    }
    for key in preferred[source]:
        if mol.HasProp(key):
            return mol.GetProp(key).strip()
    if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
        return mol.GetProp("_Name").strip()
    return f"{source}:{index}"


def analyze(mol: Chem.Mol, source: str, index: int) -> tuple[Candidate | None, str | None]:
    if len(Chem.GetMolFrags(mol)) != 1:
        return None, "multi_component"
    if Chem.GetMolSubstanceGroups(mol):
        return None, "substance_group"

    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    if not elements <= ALLOWED_ELEMENTS:
        if "H" in elements:
            return None, "explicit_hydrogen"
        return None, "unsupported_element"
    if any(atom.GetIsotope() for atom in mol.GetAtoms()):
        return None, "isotope"
    if any(atom.HasQuery() for atom in mol.GetAtoms()):
        return None, "query_atom"
    if any(bond.HasQuery() or bond.GetBondType() not in ALLOWED_BONDS for bond in mol.GetBonds()):
        return None, "unsupported_bond"

    heavy_atoms = mol.GetNumHeavyAtoms()
    if not 10 <= heavy_atoms <= 100:
        return None, "heavy_atom_range"

    try:
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None, "canonicalization"
    if not smiles:
        return None, "canonicalization"

    rings = mol.GetRingInfo().NumRings()
    chiral_centers = sum(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()
    )
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    if source == "coconut":
        bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        macrocycles = sum(1 for ring in mol.GetRingInfo().AtomRings() if len(ring) >= 8)
    else:
        bridgeheads = 0
        spiro = 0
        macrocycles = 0
    complexity = (
        rings * 2
        + chiral_centers * 3
        + bridgeheads * 3
        + spiro * 3
        + macrocycles * 4
        + max(0, heavy_atoms - 25) / 10
    )

    return Candidate(
        molecule=mol,
        source=source,
        source_id=source_id(mol, source, index),
        smiles=smiles,
        element_group=element_group(elements),
        size_bin=size_bin(heavy_atoms),
        feature_bin=feature_bin(charge, chiral_centers, rings),
        heavy_atoms=heavy_atoms,
        rings=rings,
        chiral_centers=chiral_centers,
        formal_charge=charge,
        complexity_score=round(complexity, 3),
    ), None


def scan_chembl(path: Path, target: int, rng: random.Random) -> tuple[list[Candidate], dict]:
    target_by_group = {
        group: round(target * fraction) for group, fraction in ELEMENT_TARGET_FRACTIONS.items()
    }
    target_by_group["base"] += target - sum(target_by_group.values())
    reservoirs = {
        group: Reservoir(max(quota * 2, 2_000), rng)
        for group, quota in target_by_group.items()
    }
    rejected: Counter[str] = Counter()
    eligible = Counter()
    seen_smiles: set[str] = set()
    total = 0

    for total, mol in enumerate(iter_molecules(path), 1):
        if total % 100_000 == 0:
            print(f"ChEMBL: scanned {total:,}, eligible {sum(eligible.values()):,}", flush=True)
        if mol is None:
            rejected["parse_or_sanitize"] += 1
            continue
        item, reason = analyze(mol, "chembl", total)
        if item is None:
            rejected[reason or "unknown"] += 1
            continue
        if item.smiles in seen_smiles:
            rejected["duplicate_source_structure"] += 1
            continue
        seen_smiles.add(item.smiles)
        eligible[item.element_group] += 1
        reservoirs[item.element_group].add(item)

    selected: list[Candidate] = []
    leftovers: list[Candidate] = []
    for group, reservoir in reservoirs.items():
        rng.shuffle(reservoir.items)
        quota = min(target_by_group[group], len(reservoir.items))
        selected.extend(reservoir.items[:quota])
        leftovers.extend(reservoir.items[quota:])
    if len(selected) < target:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target - len(selected)])
    rng.shuffle(selected)
    selected = selected[:target]
    return selected, {
        "scanned": total,
        "eligible_by_element_group": dict(eligible),
        "rejected": dict(rejected),
        "selected": len(selected),
        "selected_by_element_group": dict(Counter(item.element_group for item in selected)),
    }


def coconut_tier(item: Candidate) -> str | None:
    if item.complexity_score >= 18:
        return "high"
    if item.complexity_score >= 9:
        return "medium"
    return None


def balanced_take(items: Iterable[Candidate], target: int, rng: random.Random) -> list[Candidate]:
    buckets: dict[tuple[str, str, str], list[Candidate]] = defaultdict(list)
    for item in items:
        buckets[(item.element_group, item.size_bin, item.feature_bin)].append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    result: list[Candidate] = []
    active = list(buckets)
    rng.shuffle(active)
    while active and len(result) < target:
        next_active = []
        for key in active:
            bucket = buckets[key]
            if bucket:
                result.append(bucket.pop())
                if len(result) == target:
                    break
            if bucket:
                next_active.append(key)
        active = next_active
        rng.shuffle(active)
    return result


def scan_coconut(
    path: Path,
    target: int,
    excluded_smiles: set[str],
    rng: random.Random,
) -> tuple[list[Candidate], dict]:
    reservoirs = {
        "high": Reservoir(target * 2, rng),
        "medium": Reservoir(target * 2, rng),
    }
    rejected: Counter[str] = Counter()
    eligible = Counter()
    seen_smiles: set[str] = set()
    total = 0

    for total, mol in enumerate(iter_molecules(path), 1):
        if total % 100_000 == 0:
            print(f"COCONUT: scanned {total:,}, complex eligible {sum(eligible.values()):,}", flush=True)
        if mol is None:
            rejected["parse_or_sanitize"] += 1
            continue
        item, reason = analyze(mol, "coconut", total)
        if item is None:
            rejected[reason or "unknown"] += 1
            continue
        if item.smiles in excluded_smiles:
            rejected["duplicate_training_structure"] += 1
            continue
        if item.smiles in seen_smiles:
            rejected["duplicate_source_structure"] += 1
            continue
        seen_smiles.add(item.smiles)
        tier = coconut_tier(item)
        if tier is None:
            rejected["not_complex"] += 1
            continue
        eligible[tier] += 1
        reservoirs[tier].add(item)

    high_target = round(target * 0.6)
    high = balanced_take(reservoirs["high"].items, high_target, rng)
    medium = balanced_take(reservoirs["medium"].items, target - len(high), rng)
    selected = high + medium
    if len(selected) < target:
        used = {item.smiles for item in selected}
        leftovers = [
            item
            for reservoir in reservoirs.values()
            for item in reservoir.items
            if item.smiles not in used
        ]
        selected.extend(balanced_take(leftovers, target - len(selected), rng))
    rng.shuffle(selected)
    return selected[:target], {
        "scanned": total,
        "eligible_by_complexity_tier": dict(eligible),
        "rejected": dict(rejected),
        "selected": min(len(selected), target),
        "selected_by_complexity_tier": dict(
            Counter(coconut_tier(item) for item in selected[:target])
        ),
    }


def chebi_category(item: Candidate) -> str:
    if item.element_group in {"B", "Se", "Si", "P", "S"}:
        return f"element_{item.element_group}"
    if item.formal_charge:
        return "charged"
    if item.chiral_centers:
        return "stereo"
    if item.rings >= 4:
        return "polycyclic"
    return "other_in_scope"


def scan_chebi(
    path: Path,
    target: int,
    excluded_smiles: set[str],
    rng: random.Random,
) -> tuple[list[Candidate], dict]:
    reservoirs: dict[str, Reservoir] = {}
    rejected: Counter[str] = Counter()
    eligible = Counter()
    seen_smiles: set[str] = set()
    total = 0

    for total, mol in enumerate(iter_molecules(path), 1):
        if mol is None:
            rejected["parse_or_sanitize"] += 1
            continue
        item, reason = analyze(mol, "chebi", total)
        if item is None:
            rejected[reason or "unknown"] += 1
            continue
        if item.smiles in excluded_smiles:
            rejected["duplicate_training_structure"] += 1
            continue
        if item.smiles in seen_smiles:
            rejected["duplicate_source_structure"] += 1
            continue
        seen_smiles.add(item.smiles)
        category = chebi_category(item)
        eligible[category] += 1
        reservoirs.setdefault(category, Reservoir(target, rng)).add(item)

    selected = balanced_take(
        (item for reservoir in reservoirs.values() for item in reservoir.items),
        target,
        rng,
    )
    rng.shuffle(selected)
    return selected, {
        "scanned": total,
        "eligible_by_category": dict(eligible),
        "rejected": dict(rejected),
        "selected": len(selected),
        "selected_by_category": dict(Counter(chebi_category(item) for item in selected)),
    }


def write_dataset(path: Path, manifest_path: Path, items: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as sdf_stream, gzip.open(
        manifest_path, "wt", encoding="utf-8", newline=""
    ) as manifest_stream:
        writer = Chem.SDWriter(sdf_stream)
        fields = [
            "source",
            "source_id",
            "canonical_smiles",
            "element_group",
            "size_bin",
            "feature_bin",
            "heavy_atoms",
            "rings",
            "chiral_centers",
            "formal_charge",
            "complexity_score",
        ]
        manifest = csv.DictWriter(manifest_stream, fieldnames=fields)
        manifest.writeheader()
        for item in items:
            mol = item.molecule
            mol.SetProp("dataset_source", item.source)
            mol.SetProp("dataset_source_id", item.source_id)
            mol.SetProp("canonical_smiles", item.smiles)
            writer.write(mol)
            manifest.writerow(
                {
                    "source": item.source,
                    "source_id": item.source_id,
                    "canonical_smiles": item.smiles,
                    "element_group": item.element_group,
                    "size_bin": item.size_bin,
                    "feature_bin": item.feature_bin,
                    "heavy_atoms": item.heavy_atoms,
                    "rings": item.rings,
                    "chiral_centers": item.chiral_centers,
                    "formal_charge": item.formal_charge,
                    "complexity_score": item.complexity_score,
                }
            )
        writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("prepared"))
    parser.add_argument("--chembl-size", type=int, default=150_000)
    parser.add_argument("--coconut-size", type=int, default=30_000)
    parser.add_argument("--chebi-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260713)
    return parser.parse_args()


def size_label(value: int) -> str:
    return f"{value // 1000}k" if value % 1000 == 0 else str(value)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    started = time.time()

    chembl, chembl_stats = scan_chembl(
        args.raw_dir / "chembl_37.sdf.gz", args.chembl_size, rng
    )
    training_smiles = {item.smiles for item in chembl}
    write_dataset(
        args.output_dir / "train" / f"chembl_{size_label(args.chembl_size)}.sdf.gz",
        args.output_dir / "train" / f"chembl_{size_label(args.chembl_size)}.manifest.csv.gz",
        chembl,
    )
    del chembl

    coconut, coconut_stats = scan_coconut(
        args.raw_dir / "coconut_sdf_2d_lite-07-2026.zip",
        args.coconut_size,
        training_smiles,
        rng,
    )
    training_smiles.update(item.smiles for item in coconut)
    write_dataset(
        args.output_dir / "train" / f"coconut_complex_{size_label(args.coconut_size)}.sdf.gz",
        args.output_dir / "train" / f"coconut_complex_{size_label(args.coconut_size)}.manifest.csv.gz",
        coconut,
    )
    del coconut

    chebi, chebi_stats = scan_chebi(
        args.raw_dir / "chebi_3_stars.sdf.gz",
        args.chebi_size,
        training_smiles,
        rng,
    )
    write_dataset(
        args.output_dir / "test" / f"chebi_specialty_{size_label(args.chebi_size)}.sdf.gz",
        args.output_dir / "test" / f"chebi_specialty_{size_label(args.chebi_size)}.manifest.csv.gz",
        chebi,
    )

    summary = {
        "created_at_unix": int(time.time()),
        "elapsed_seconds": round(time.time() - started, 2),
        "rdkit_version": rdBase.rdkitVersion,
        "seed": args.seed,
        "filters": {
            "single_connected_component": True,
            "allowed_elements": sorted(ALLOWED_ELEMENTS),
            "heavy_atom_range": [10, 100],
            "allowed_bond_types": ["single", "double", "triple", "aromatic"],
            "exclude_isotopes": True,
            "exclude_explicit_hydrogen_atoms": True,
            "exclude_query_atoms_and_bonds": True,
            "exclude_substance_groups": True,
            "rdkit_sanitize_required": True,
        },
        "chembl": chembl_stats,
        "coconut": coconut_stats,
        "chebi": chebi_stats,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
