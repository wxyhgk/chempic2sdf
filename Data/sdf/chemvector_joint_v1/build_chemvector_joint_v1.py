#!/usr/bin/env python3
"""Build ChemVector Joint v1 style-train and fresh evaluation datasets."""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import heapq
import html
import io
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import stat
import sys
import tarfile
import time
from typing import Any, Iterable, Mapping, Sequence

import fastjsonschema
import numpy as np
from PIL import Image
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D


HERE = Path(__file__).resolve().parent
SDF_ROOT = HERE.parent
PROJECT_ROOT = SDF_ROOT.parents[1]
STYLE_ROOT = HERE / "style_train_100k"
FRESH_ROOT = HERE / "fresh_eval_12k"
FULL_ROOT = SDF_ROOT / "webdataset_full_v0"
RAW_CHEMBL = SDF_ROOT / "extracted/raw/chembl_37.sdf"
SEED = 20260728
GIT_REVISION = "14bf3702ff30baf7209ef34f20734b63f28996fc"
VARIANTS = ("original", "rdkit", "coordgen", "stress")
STRESS_OPERATIONS = ("long_bond", "crowded_subgraph", "near_nonbonded", "crossed_bonds")
STYLE_MOLECULES = 25_000
STYLE_SAMPLES = 100_000
FRESH_DEV_MOLECULES = 2_000
SEALED_MOLECULES = 1_000
SHARD_SIZE = 5_000
WIDTH = HEIGHT = 768
ALLOWED_ELEMENTS = {"B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I", "Se"}
SIZE_BUCKETS = (("8-20", 8, 20), ("21-40", 21, 40), ("41-60", 41, 60), ("61-80", 61, 80))
FRESH_LABEL_REPLACEMENTS = {
    "CHEMBL3623248": "CHEMBL540483",
    "CHEMBL4225941": "CHEMBL503233",
    "CHEMBL6092103": "CHEMBL504097",
    "CHEMBL2385807": "CHEMBL498957",
    "CHEMBL6198954": "CHEMBL501517",
    "CHEMBL3822931": "CHEMBL501945",
    "CHEMBL4745445": "CHEMBL499990",
    "CHEMBL2097032": "CHEMBL503304",
    "CHEMBL5401851": "CHEMBL507824",
}
JOINT_MEMBERS = (
    ".png", ".text.png", ".bond.png", ".semantic.png",
    ".chemmol.json", ".render.json", ".layers.json", ".sdf", ".meta.json",
)
SAMPLE_FIELDS = (
    "sample_key", "molecule_id", "inchikey", "scaffold", "split", "variant",
    "source", "source_id", "output_shard", "font_family_id", "font_family_group",
    "font_size", "font_bucket", "line_width_bucket", "bond_line_width",
    "rotation_bucket", "rotation_degrees", "multiple_bond_offset_bucket",
    "multiple_bond_offset", "color_mode", "stress_operation", "png_sha256",
    "chemmol_sha256", "render_sha256", "layers_sha256", "sdf_sha256", "meta_sha256",
)
MOLECULE_FIELDS = (
    "molecule_id", "source", "source_id", "inchikey", "scaffold",
    "canonical_isomeric_smiles", "split", "heavy_atoms", "rings", "fused_rings",
    "bridged_atoms", "chiral_centers", "element_bucket", "size_bucket",
    "source_structure", "image_count",
)
REJECTED_FIELDS = ("source", "source_id", "stage", "error_code", "error_message", "replacement")


if str(SDF_ROOT) not in sys.path:
    sys.path.insert(0, str(SDF_ROOT))
import build_chemvision_layers as layer_builder
import build_webdataset_sample as molecule_pipeline


@dataclass(slots=True)
class StyleSpec:
    font_family_group: str
    color_mode: str
    line_width_bucket: str
    rotation_bucket: str
    multiple_bond_offset_bucket: str
    multiple_bond_offset: float


@dataclass(slots=True)
class FreshCandidate:
    rank: int
    source_id: str
    molecule_id: str
    inchikey: str
    scaffold: str
    smiles: str
    size_bucket: str
    heavy_atoms: int
    rings: int
    fused_rings: int
    bridged_atoms: int
    chiral_centers: int
    element_bucket: str
    molblock: str


STYLE_BY_SAMPLE: dict[str, StyleSpec] = {}
SDF_BY_MOLECULE: dict[str, bytes] = {}
IDENTITY_BY_MOLECULE: dict[str, dict[str, str]] = {}


def stable_int(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(values)):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json_bytes(value))
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def size_bucket(heavy_atoms: int) -> str:
    for name, low, high in SIZE_BUCKETS:
        if low <= heavy_atoms <= high:
            return name
    raise ValueError(f"heavy atom count outside fresh range: {heavy_atoms}")


def ring_features(mol: Chem.Mol) -> tuple[int, int, int]:
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    fused = sum(1 for index, ring in enumerate(rings) if any(len(ring & other) >= 2 for other in rings[index + 1 :]))
    ring_memberships = Counter(atom for ring in rings for atom in ring)
    bridged = sum(count >= 3 for count in ring_memberships.values())
    return len(rings), fused, bridged


def element_bucket(mol: Chem.Mol) -> str:
    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    found = [symbol for symbol in ("B", "Se", "Si", "P", "S", "N") if symbol in elements]
    return "+".join(found) if found else "base"


def valid_fresh_molecule(mol: Chem.Mol | None) -> bool:
    if mol is None:
        return False
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    heavy = mol.GetNumHeavyAtoms()
    if not 8 <= heavy <= 80:
        return False
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ALLOWED_ELEMENTS:
            return False
        if atom.GetIsotope() or atom.GetNumRadicalElectrons() or atom.HasQuery():
            return False
    return not any(bond.HasQuery() for bond in mol.GetBonds())


def benchmark_identity_index() -> tuple[set[str], set[str], dict[str, Any]]:
    """Load benchmark identities separately so style-train can exclude them."""

    inchikeys: set[str] = set()
    scaffolds: set[str] = set()
    audit: dict[str, Any] = {}

    mr_rows = load_csv(PROJECT_ROOT / "Data/benchmarks/mr_tadf_v1/manifest.csv")
    mr_valid = 0
    for row in mr_rows:
        key = row.get("inchikey", "")
        smiles = row.get("canonical_isomeric_smiles", "")
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if key:
            inchikeys.add(key)
        if mol is not None:
            scaffolds.add(molecule_pipeline.scaffold_smiles(mol, smiles))
            mr_valid += 1
    audit["mr_tadf_v1"] = {"records": len(mr_rows), "indexed": mr_valid}

    public_root = PROJECT_ROOT / "Data/benchmarks/public_ocsr_v1"
    parsed = failures = 0
    for subset in ("uspto_10k", "uspto_10k_l", "uspto_10k_abb"):
        for row in load_csv(public_root / f"manifests/{subset}.csv"):
            source = row.get("source_structure_file", "")
            if not source:
                failures += 1
                continue
            try:
                mol = Chem.MolFromMolFile(
                    str(public_root / source), sanitize=True, removeHs=False,
                    strictParsing=False,
                )
                if mol is None:
                    failures += 1
                    continue
                smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                inchikeys.add(Chem.MolToInchiKey(mol))
                scaffolds.add(molecule_pipeline.scaffold_smiles(mol, smiles))
                parsed += 1
            except Exception:
                failures += 1
    audit["uspto_30k"] = {"records": 30_000, "indexed": parsed, "unresolved": failures}
    return inchikeys, scaffolds, audit


def build_overlap_index(force: bool = False) -> dict[str, Any]:
    cache = HERE / "work/old_overlap_index.json"
    if cache.is_file() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))

    molecule_ids: set[str] = set()
    inchikeys: set[str] = set()
    scaffolds: set[str] = set()
    audit: dict[str, Any] = {}

    full_rows = load_csv(FULL_ROOT / "molecules.csv")
    for row in full_rows:
        molecule_ids.add(row["molecule_id"])
        inchikeys.add(row["inchikey"])
        scaffolds.add(row["scaffold"])
    audit["webdataset_full_v0"] = {"records": len(full_rows)}

    synthetic_rows = load_csv(SDF_ROOT / "molgrapher_synthetic_v0_1/pilot_10k/samples.csv")
    for row in synthetic_rows:
        molecule_ids.add(row.get("molecule_id", ""))
        if row.get("inchikey"):
            inchikeys.add(row["inchikey"])
        if row.get("scaffold"):
            scaffolds.add(row["scaffold"])
    audit["molgrapher_synthetic_v0_1"] = {"records": len(synthetic_rows)}

    benchmark_keys, benchmark_scaffolds, benchmark_audit = benchmark_identity_index()
    inchikeys.update(benchmark_keys)
    scaffolds.update(benchmark_scaffolds)
    audit.update(benchmark_audit)

    value = {
        "schema_version": "chemvector-joint-overlap-index-0.1",
        "generated_at_unix": int(time.time()),
        "molecule_ids": sorted(molecule_ids),
        "inchikeys": sorted(inchikeys),
        "scaffolds": sorted(scaffolds),
        "benchmark_inchikeys": sorted(benchmark_keys),
        "benchmark_scaffolds": sorted(benchmark_scaffolds),
        "audit": audit,
    }
    write_json(cache, value)
    return value


def select_style_plan(overlap: Mapping[str, Any], force: bool = False) -> list[dict[str, str]]:
    plan_path = HERE / "work/style_molecules.plan.csv"
    if plan_path.is_file() and not force:
        return load_csv(plan_path)
    benchmark_keys = set(overlap["benchmark_inchikeys"])
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    with (FULL_ROOT / "molecules.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != "train" or "supplement" in row["shard"]:
                continue
            if row["inchikey"] in benchmark_keys:
                continue
            by_source[row["source"]].append(row)
    targets = {"chembl": 21_250, "coconut": 3_750}
    selected: list[dict[str, str]] = []
    for source, target in targets.items():
        ordered = sorted(
            by_source[source],
            key=lambda row: (stable_int(SEED, "style-molecule", row["molecule_id"]), row["molecule_id"]),
        )
        if len(ordered) < target:
            raise RuntimeError(f"not enough {source} train molecules: {len(ordered)} < {target}")
        selected.extend(ordered[:target])
    selected.sort(key=lambda row: stable_int(SEED, "style-plan-order", row["molecule_id"]))
    for index, row in enumerate(selected):
        row["joint_index"] = str(index)
    fields = ["joint_index", "molecule_id", "source", "source_id", "inchikey", "scaffold", "stratum", "canonical_isomeric_smiles", "shard", "structure_ref"]
    write_csv(plan_path, selected, fields)
    return selected


def candidate_from_mol(mol: Chem.Mol, index: int, old: Mapping[str, set[str]]) -> FreshCandidate | None:
    if not valid_fresh_molecule(mol):
        return None
    source_id = molecule_pipeline.get_source_id(mol, "chembl", index)
    molecule_id = f"chembl:{source_id}"
    if molecule_id in old["molecule_ids"]:
        return None
    try:
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        inchikey = Chem.MolToInchiKey(mol)
        scaffold = molecule_pipeline.scaffold_smiles(mol, smiles)
    except Exception:
        return None
    if inchikey in old["inchikeys"] or scaffold in old["scaffolds"]:
        return None
    rings, fused, bridged = ring_features(mol)
    heavy = mol.GetNumHeavyAtoms()
    return FreshCandidate(
        rank=stable_int(SEED, "fresh-raw-chembl", molecule_id, inchikey, scaffold),
        source_id=source_id,
        molecule_id=molecule_id,
        inchikey=inchikey,
        scaffold=scaffold,
        smiles=smiles,
        size_bucket=size_bucket(heavy),
        heavy_atoms=heavy,
        rings=rings,
        fused_rings=fused,
        bridged_atoms=bridged,
        chiral_centers=len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        element_bucket=element_bucket(mol),
        molblock=Chem.MolToMolBlock(mol, includeStereo=True, kekulize=True, forceV3000=True),
    )


def select_fresh_plan(overlap: Mapping[str, Any], force: bool = False) -> list[dict[str, str]]:
    plan_path = HERE / "work/fresh_molecules.plan.csv"
    structure_root = HERE / "work/fresh_structures"
    if plan_path.is_file() and structure_root.is_dir() and not force:
        cached = load_csv(plan_path)
        repaired = repair_fresh_sdf_roundtrips(cached)
        repaired += repair_fresh_label_roundtrips(cached, overlap)
        if repaired:
            write_csv(plan_path, cached, tuple(cached[0]))
        return cached
    if force and structure_root.exists():
        shutil.rmtree(structure_root)
    structure_root.mkdir(parents=True, exist_ok=True)

    old = {
        "molecule_ids": set(overlap["molecule_ids"]),
        "inchikeys": set(overlap["inchikeys"]),
        "scaffolds": set(overlap["scaffolds"]),
    }
    heap_capacity = 10_000
    heaps: dict[str, list[tuple[int, int, FreshCandidate]]] = {name: [] for name, _, _ in SIZE_BUCKETS}
    counters = Counter()
    supplier = Chem.SDMolSupplier(str(RAW_CHEMBL), sanitize=True, removeHs=False, strictParsing=False)
    started = time.perf_counter()
    for index, mol in enumerate(supplier, 1):
        counters["records"] += 1
        candidate = candidate_from_mol(mol, index, old) if mol is not None else None
        if candidate is None:
            counters["rejected"] += 1
        else:
            counters["eligible"] += 1
            heap = heaps[candidate.size_bucket]
            item = (-candidate.rank, index, candidate)
            if len(heap) < heap_capacity:
                heapq.heappush(heap, item)
            elif candidate.rank < -heap[0][0]:
                heapq.heapreplace(heap, item)
        if index % 250_000 == 0:
            print(
                f"Scanned ChEMBL {index:,}: eligible={counters['eligible']:,}, "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    selected_by_bucket: dict[str, list[FreshCandidate]] = {}
    used_scaffolds: set[str] = set()
    used_inchikeys: set[str] = set()
    for bucket, _, _ in SIZE_BUCKETS:
        ordered = sorted((item[2] for item in heaps[bucket]), key=lambda item: (item.rank, item.molecule_id))
        chosen: list[FreshCandidate] = []
        for candidate in ordered:
            if candidate.scaffold in used_scaffolds or candidate.inchikey in used_inchikeys:
                continue
            chosen.append(candidate)
            used_scaffolds.add(candidate.scaffold)
            used_inchikeys.add(candidate.inchikey)
            if len(chosen) == 750:
                break
        if len(chosen) != 750:
            raise RuntimeError(f"fresh bucket {bucket} has only {len(chosen)} unique candidates")
        selected_by_bucket[bucket] = chosen

    rows: list[dict[str, str]] = []
    split_indices = Counter()
    for bucket, _, _ in SIZE_BUCKETS:
        for group_index in range(250):
            group = selected_by_bucket[bucket][group_index * 3 : group_index * 3 + 3]
            assignments = ((group[0], "fresh_dev"), (group[1], "fresh_dev"), (group[2], "sealed_test"))
            for candidate, split in assignments:
                split_index = split_indices[split]
                split_indices[split] += 1
                mol = Chem.MolFromMolBlock(
                    candidate.molblock, sanitize=True, removeHs=False, strictParsing=False
                )
                if mol is None:
                    raise RuntimeError(f"selected fresh candidate cannot roundtrip: {candidate.molecule_id}")
                record = molecule_pipeline.canonicalize_record("chembl", mol, candidate.smiles, split_index)
                if record.molecule_id != candidate.molecule_id or record.inchikey != candidate.inchikey:
                    raise RuntimeError(f"fresh canonical identity changed: {candidate.molecule_id}")
                record.split = split
                filename = f"{split_index:06d}_{candidate.source_id}.sdf"
                sdf_payload = (
                    Chem.MolToMolBlock(record.mol, includeStereo=True, kekulize=True, forceV3000=True)
                    + "$$$$\n"
                ).encode("utf-8")
                (structure_root / filename).write_bytes(sdf_payload)
                row = {
                    "selection_index": str(split_index),
                    "split": split,
                    "source": "chembl",
                    "source_id": candidate.source_id,
                    "molecule_id": candidate.molecule_id,
                    "inchikey": candidate.inchikey,
                    "scaffold": candidate.scaffold,
                    "canonical_isomeric_smiles": record.smiles,
                    "size_bucket": candidate.size_bucket,
                    "heavy_atoms": str(candidate.heavy_atoms),
                    "rings": str(candidate.rings),
                    "fused_rings": str(candidate.fused_rings),
                    "bridged_atoms": str(candidate.bridged_atoms),
                    "chiral_centers": str(candidate.chiral_centers),
                    "element_bucket": candidate.element_bucket,
                    "structure_file": f"work/fresh_structures/{filename}",
                    "sdf_sha256": sha256_bytes(sdf_payload),
                }
                rows.append(row)
    fields = (
        "selection_index", "split", "source", "source_id", "molecule_id", "inchikey",
        "scaffold", "canonical_isomeric_smiles", "size_bucket", "heavy_atoms", "rings",
        "fused_rings", "bridged_atoms", "chiral_centers", "element_bucket",
        "structure_file", "sdf_sha256",
    )
    rows.sort(key=lambda row: (row["split"], int(row["selection_index"])))
    write_csv(plan_path, rows, fields)
    repaired = repair_fresh_sdf_roundtrips(rows)
    repaired += repair_fresh_label_roundtrips(rows, overlap)
    if repaired:
        write_csv(plan_path, rows, fields)
    write_json(
        HERE / "work/fresh_selection_audit.json",
        {
            "schema_version": "chemvector-joint-fresh-selection-0.1",
            "seed": SEED,
            "source": str(RAW_CHEMBL.relative_to(PROJECT_ROOT)),
            "source_sha256": sha256_file(RAW_CHEMBL),
            "scan_counts": dict(counters),
            "split_counts": dict(split_indices),
            "size_bucket_counts": dict(Counter(row["size_bucket"] for row in rows)),
            "old_index_audit": overlap["audit"],
            "old_molecule_overlap": len({row["molecule_id"] for row in rows} & old["molecule_ids"]),
            "old_inchikey_overlap": len({row["inchikey"] for row in rows} & old["inchikeys"]),
            "old_scaffold_overlap": len({row["scaffold"] for row in rows} & old["scaffolds"]),
        },
    )
    return rows


def repair_fresh_sdf_roundtrips(rows: Sequence[dict[str, str]]) -> int:
    repaired = 0
    for row in rows:
        path = HERE / row["structure_file"]
        payload = path.read_bytes()
        mol = Chem.MolFromMolBlock(
            payload.decode("utf-8").split("$$$$", 1)[0],
            sanitize=True, removeHs=False, strictParsing=False,
        )
        actual = Chem.MolToInchiKey(mol) if mol is not None else ""
        source_id = mol.GetProp("_Name").strip() if mol is not None and mol.HasProp("_Name") else ""
        if actual == row["inchikey"] and source_id == row["source_id"]:
            continue
        replacement = mol if actual == row["inchikey"] else Chem.MolFromSmiles(row["canonical_isomeric_smiles"])
        if replacement is None:
            raise RuntimeError(f"cannot repair fresh SDF: {row['molecule_id']}")
        replacement.SetProp("_Name", row["source_id"])
        if replacement.GetNumConformers() == 0:
            molecule_pipeline.rdDepictor.Compute2DCoords(replacement, canonOrient=True, clearConfs=True)
        payload = (
            Chem.MolToMolBlock(
                replacement, includeStereo=True, kekulize=True, forceV3000=True
            )
            + "$$$$\n"
        ).encode("utf-8")
        roundtrip = Chem.MolFromMolBlock(
            payload.decode("utf-8").split("$$$$", 1)[0],
            sanitize=True, removeHs=False, strictParsing=False,
        )
        if roundtrip is None or Chem.MolToInchiKey(roundtrip) != row["inchikey"]:
            raise RuntimeError(f"fresh SDF repair failed: {row['molecule_id']}")
        if not roundtrip.HasProp("_Name") or roundtrip.GetProp("_Name").strip() != row["source_id"]:
            raise RuntimeError(f"fresh SDF source ID repair failed: {row['molecule_id']}")
        path.write_bytes(payload)
        row["sdf_sha256"] = sha256_bytes(payload)
        repaired += 1
    if repaired:
        print(f"Repaired {repaired} fresh SDF roundtrip representation(s)", flush=True)
    return repaired


def repair_fresh_label_roundtrips(
    rows: Sequence[dict[str, str]], overlap: Mapping[str, Any]
) -> int:
    failed: dict[str, dict[str, str]] = {}
    for row in rows:
        path = HERE / row["structure_file"]
        mol = Chem.MolFromMolBlock(
            path.read_text(encoding="utf-8").split("$$$$", 1)[0],
            sanitize=True, removeHs=False, strictParsing=False,
        )
        if mol is None:
            raise RuntimeError(f"cannot parse fresh SDF for label audit: {row['molecule_id']}")
        mol.SetProp("_Name", row["source_id"])
        record = molecule_pipeline.canonicalize_record(
            "chembl", mol, row["canonical_isomeric_smiles"], int(row["selection_index"])
        )
        reconstructed = molecule_pipeline.mol_from_chemmol(molecule_pipeline.build_chemmol(record))
        Chem.SanitizeMol(reconstructed)
        if Chem.MolToInchiKey(reconstructed) != row["inchikey"]:
            failed[row["source_id"]] = row
    if not failed:
        return 0
    if set(failed) - set(FRESH_LABEL_REPLACEMENTS):
        raise RuntimeError(f"unplanned fresh label failures: {sorted(set(failed) - set(FRESH_LABEL_REPLACEMENTS))}")

    wanted = {FRESH_LABEL_REPLACEMENTS[source_id] for source_id in failed}
    old = {
        "molecule_ids": set(overlap["molecule_ids"]),
        "inchikeys": set(overlap["inchikeys"]),
        "scaffolds": set(overlap["scaffolds"]),
    }
    candidates: dict[str, FreshCandidate] = {}
    supplier = Chem.SDMolSupplier(str(RAW_CHEMBL), sanitize=True, removeHs=False, strictParsing=False)
    for index, mol in enumerate(supplier, 1):
        if mol is None:
            continue
        source_id = molecule_pipeline.get_source_id(mol, "chembl", index)
        if source_id not in wanted:
            continue
        candidate = candidate_from_mol(mol, index, old)
        if candidate is None:
            raise RuntimeError(f"fresh replacement is no longer eligible: {source_id}")
        candidates[source_id] = candidate
        if set(candidates) == wanted:
            break
    if set(candidates) != wanted:
        raise RuntimeError(f"fresh replacements missing from ChEMBL: {sorted(wanted - set(candidates))}")

    failed_ids = {row["molecule_id"] for row in failed.values()}
    used_ids = {row["molecule_id"] for row in rows if row["molecule_id"] not in failed_ids}
    used_keys = {row["inchikey"] for row in rows if row["molecule_id"] not in failed_ids}
    used_scaffolds = {row["scaffold"] for row in rows if row["molecule_id"] not in failed_ids}
    replacement_rows = []
    for failed_source_id in sorted(failed):
        row = failed[failed_source_id]
        replacement_source_id = FRESH_LABEL_REPLACEMENTS[failed_source_id]
        candidate = candidates[replacement_source_id]
        if candidate.size_bucket != row["size_bucket"]:
            raise RuntimeError(f"fresh replacement size bucket mismatch: {failed_source_id}")
        if (
            candidate.molecule_id in used_ids
            or candidate.inchikey in used_keys
            or candidate.scaffold in used_scaffolds
        ):
            raise RuntimeError(f"fresh replacement overlap: {replacement_source_id}")
        source_mol = Chem.MolFromMolBlock(
            candidate.molblock, sanitize=True, removeHs=False, strictParsing=False
        )
        if source_mol is None:
            raise RuntimeError(f"cannot parse fresh replacement: {replacement_source_id}")
        source_mol.SetProp("_Name", replacement_source_id)
        record = molecule_pipeline.canonicalize_record(
            "chembl", source_mol, candidate.smiles, int(row["selection_index"])
        )
        reconstructed = molecule_pipeline.mol_from_chemmol(molecule_pipeline.build_chemmol(record))
        Chem.SanitizeMol(reconstructed)
        if Chem.MolToInchiKey(reconstructed) != record.inchikey:
            raise RuntimeError(f"fresh replacement label roundtrip failed: {replacement_source_id}")
        record.mol.SetProp("_Name", replacement_source_id)
        payload = (
            Chem.MolToMolBlock(record.mol, includeStereo=True, kekulize=True, forceV3000=True)
            + "$$$$\n"
        ).encode("utf-8")
        roundtrip = Chem.MolFromMolBlock(
            payload.decode("utf-8").split("$$$$", 1)[0],
            sanitize=True, removeHs=False, strictParsing=False,
        )
        if roundtrip is None or Chem.MolToInchiKey(roundtrip) != record.inchikey:
            raise RuntimeError(f"fresh replacement SDF roundtrip failed: {replacement_source_id}")
        old_path = HERE / row["structure_file"]
        filename = f"{int(row['selection_index']):06d}_{replacement_source_id}.sdf"
        new_path = HERE / "work/fresh_structures" / filename
        new_path.write_bytes(payload)
        if old_path != new_path:
            old_path.unlink()
        rings, fused, bridged = ring_features(record.mol)
        row.update(
            {
                "source_id": replacement_source_id,
                "molecule_id": candidate.molecule_id,
                "inchikey": record.inchikey,
                "scaffold": record.scaffold,
                "canonical_isomeric_smiles": record.smiles,
                "size_bucket": candidate.size_bucket,
                "heavy_atoms": str(record.mol.GetNumHeavyAtoms()),
                "rings": str(rings),
                "fused_rings": str(fused),
                "bridged_atoms": str(bridged),
                "chiral_centers": str(len(Chem.FindMolChiralCenters(record.mol, includeUnassigned=True))),
                "element_bucket": element_bucket(record.mol),
                "structure_file": f"work/fresh_structures/{filename}",
                "sdf_sha256": sha256_bytes(payload),
            }
        )
        used_ids.add(candidate.molecule_id)
        used_keys.add(record.inchikey)
        used_scaffolds.add(record.scaffold)
        replacement_rows.append(
            {
                "failed_source_id": failed_source_id,
                "replacement_source_id": replacement_source_id,
                "split": row["split"],
                "selection_index": row["selection_index"],
                "size_bucket": row["size_bucket"],
                "reason": "ChemMol roundtrip InChIKey mismatch from coordinate-derived stereochemistry",
            }
        )
    write_csv(
        HERE / "work/fresh_label_replacements.csv",
        replacement_rows,
        ("failed_source_id", "replacement_source_id", "split", "selection_index", "size_bucket", "reason"),
    )
    print(f"Replaced {len(replacement_rows)} fresh molecule(s) that cannot roundtrip through ChemMol", flush=True)
    return len(replacement_rows)


def plan(force: bool = False) -> dict[str, Any]:
    RDLogger.DisableLog("rdApp.*")
    if rdBase.rdkitVersion != "2026.03.3":
        raise RuntimeError(f"Expected RDKit 2026.03.3, got {rdBase.rdkitVersion}")
    overlap = build_overlap_index(force=force)
    style_rows = select_style_plan(overlap, force=force)
    fresh_rows = select_fresh_plan(overlap, force=force)
    result = {
        "status": "passed",
        "style_molecules": len(style_rows),
        "style_source_counts": dict(Counter(row["source"] for row in style_rows)),
        "fresh_molecules": len(fresh_rows),
        "fresh_split_counts": dict(Counter(row["split"] for row in fresh_rows)),
        "fresh_size_bucket_counts": {
            f"{split}:{bucket}": count
            for (split, bucket), count in sorted(
                Counter((row["split"], row["size_bucket"]) for row in fresh_rows).items()
            )
        },
    }
    write_json(HERE / "work/plan_validation.json", result)
    return result


def train_font_specs() -> list[dict[str, Any]]:
    return [dict(item, split="style_train", font_family_group="train_seen") for item in layer_builder.FONT_SPECS_V077]


def fresh_dev_font_specs() -> list[dict[str, Any]]:
    return [
        {
            "font_family_id": "stix-regular",
            "font_family": "STIX",
            "font_style": "Regular",
            "source_file": "/usr/share/fonts/opentype/stix-word/STIX-Regular.otf",
            "output_file": "fonts/STIX-Regular.otf",
            "license": "SIL Open Font License 1.1",
            "license_url": "https://www.stixfonts.org/",
            "split": "fresh_dev",
            "font_family_group": "heldout",
        },
        {
            "font_family_id": "p052-roman",
            "font_family": "P052",
            "font_style": "Roman",
            "source_file": "/usr/share/fonts/opentype/urw-base35/P052-Roman.otf",
            "output_file": "fonts/P052-Roman.otf",
            "license": "AGPL-3 with Font exception",
            "license_url": "https://github.com/ArtifexSoftware/urw-base35-fonts",
            "split": "fresh_dev",
            "font_family_group": "heldout",
        },
    ]


def sealed_font_specs() -> list[dict[str, Any]]:
    return [
        {
            "font_family_id": "latin-modern-sans",
            "font_family": "Latin Modern Sans",
            "font_style": "Regular",
            "source_file": "/usr/share/texmf/fonts/opentype/public/lm/lmsans10-regular.otf",
            "output_file": "fonts/LatinModernSans-Regular.otf",
            "license": "GUST Font License 1.0",
            "license_url": "https://www.gust.org.pl/projects/e-foundry/licenses",
            "split": "sealed_test",
            "font_family_group": "heldout",
        },
        {
            "font_family_id": "latin-modern-roman",
            "font_family": "Latin Modern Roman",
            "font_style": "Regular",
            "source_file": "/usr/share/texmf/fonts/opentype/public/lm/lmroman10-regular.otf",
            "output_file": "fonts/LatinModernRoman-Regular.otf",
            "license": "GUST Font License 1.0",
            "license_url": "https://www.gust.org.pl/projects/e-foundry/licenses",
            "split": "sealed_test",
            "font_family_group": "heldout",
        },
    ]


def copy_fonts(output: Path, font_specs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    (output / "fonts").mkdir(parents=True, exist_ok=True)
    rows = []
    for item in font_specs:
        source = Path(str(item["source_file"]))
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output / str(item["output_file"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append(
            {
                "font_family_id": item["font_family_id"],
                "font_family": item["font_family"],
                "font_style": item["font_style"],
                "font_file": item["output_file"],
                "font_file_sha256": sha256_file(source),
                "license": item["license"],
                "license_url": item["license_url"],
                "split": item["split"],
                "font_family_group": item["font_family_group"],
            }
        )
    return rows


def exact_sequence(count: int, weighted: Sequence[tuple[str, float]], salt: str) -> list[str]:
    result: list[str] = []
    assigned = 0
    for index, (name, fraction) in enumerate(weighted):
        amount = count - assigned if index == len(weighted) - 1 else round(count * fraction)
        result.extend([name] * amount)
        assigned += amount
    random.Random(stable_int(SEED, "style-sequence", salt, count)).shuffle(result)
    if len(result) != count:
        raise RuntimeError(f"style sequence {salt} has {len(result)} values, expected {count}")
    return result


def balanced_font_sequence(count: int, font_ids: Sequence[str], salt: str) -> list[str]:
    values = [font_ids[index % len(font_ids)] for index in range(count)]
    random.Random(stable_int(SEED, "font-sequence", salt)).shuffle(values)
    return values


def build_style_tasks(
    molecule_rows: Sequence[Mapping[str, str]],
    sample_rows: Mapping[str, Mapping[str, str]],
    font_specs: Sequence[Mapping[str, Any]],
    split: str,
) -> list[layer_builder.RenderTask]:
    source_by_molecule_variant = {
        (row["molecule_id"], row["variant"]): (key, row)
        for key, row in sample_rows.items()
    }
    ordered_pairs = [
        (row["molecule_id"], int(row.get("joint_index", row.get("selection_index", "0"))), variant)
        for row in molecule_rows for variant in VARIANTS
    ]
    ordered_pairs.sort(key=lambda item: (stable_int(SEED, split, "task-order", item[0], item[2]), item[0], item[2]))
    count = len(ordered_pairs)
    font_ids = [str(item["font_family_id"]) for item in font_specs]
    font_map = {str(item["font_family_id"]): item for item in font_specs}
    fonts = balanced_font_sequence(count, font_ids, split)
    buckets = exact_sequence(count, (("small", .25), ("normal", .45), ("large", .25), ("extreme", .05)), split + "-font-bucket")
    line_buckets = exact_sequence(count, (("thin", 1 / 3), ("normal", 1 / 3), ("thick", 1 / 3)), split + "-line")
    rotation_buckets = exact_sequence(count, (("negative", 1 / 3), ("near_zero", 1 / 3), ("positive", 1 / 3)), split + "-rotation")
    offset_buckets = exact_sequence(count, (("tight", .25), ("normal", .25), ("wide", .25), ("extra_wide", .25)), split + "-offset")
    colors = exact_sequence(count, (("black_white", .70), ("limited_color", .15), ("grayscale", .15)), split + "-color")
    repeat_pairs = {
        pair for pair in sorted(ordered_pairs, key=lambda item: stable_int(SEED, split, "repeat", item[0], item[2]))[:1000]
    }
    bucket_ranges = {"small": (8, 12), "normal": (13, 19), "large": (20, 28), "extreme": (29, 36)}
    line_ranges = {"thin": (1.4, 1.7), "normal": (1.8, 2.2), "thick": (2.3, 2.6)}
    rotation_ranges = {"negative": (-12.0, -3.0), "near_zero": (-2.9, 2.9), "positive": (3.0, 12.0)}
    offsets = {"tight": 0.10, "normal": 0.14, "wide": 0.18, "extra_wide": 0.22}

    tasks: list[layer_builder.RenderTask] = []
    for number, pair in enumerate(ordered_pairs):
        molecule_id, molecule_index, variant = pair
        source_key, source_row = source_by_molecule_variant[(molecule_id, variant)]
        font = font_map[fonts[number]]
        font_low, font_high = bucket_ranges[buckets[number]]
        font_size = font_low + stable_int(SEED, split, "font-size", molecule_id, variant) % (font_high - font_low + 1)
        line_low, line_high = line_ranges[line_buckets[number]]
        line_rng = random.Random(stable_int(SEED, split, "line-width", molecule_id, variant))
        rotation_low, rotation_high = rotation_ranges[rotation_buckets[number]]
        rotation_rng = random.Random(stable_int(SEED, split, "rotation", molecule_id, variant))
        if split == "train":
            sample_key = f"train_{molecule_index:06d}_{variant}"
        else:
            sample_key = f"{split}_{molecule_index:06d}_{variant}"
        task = layer_builder.RenderTask(
            sample_key=sample_key,
            molecule_id=molecule_id,
            split=split,
            variant=variant,
            source_sample_key=source_key,
            source_shard=str(source_row["shard"]),
            output_shard="",
            font_family_id=str(font["font_family_id"]),
            font_source_file=str(font["source_file"]),
            font_file_sha256=sha256_file(str(font["source_file"])),
            font_size=int(font_size),
            font_bucket=buckets[number],
            bond_line_width=round(line_rng.uniform(line_low, line_high), 6),
            rotation_degrees=round(rotation_rng.uniform(rotation_low, rotation_high), 6),
            repeat_check=pair in repeat_pairs,
        )
        style = StyleSpec(
            font_family_group=str(font["font_family_group"]),
            color_mode=colors[number],
            line_width_bucket=line_buckets[number],
            rotation_bucket=rotation_buckets[number],
            multiple_bond_offset_bucket=offset_buckets[number],
            multiple_bond_offset=offsets[offset_buckets[number]],
        )
        STYLE_BY_SAMPLE[sample_key] = style
        source_render = layer_builder.SOURCE_ASSETS.get(source_key)
        if source_render is not None:
            source_render[2]["render_parameters"]["multiple_bond_offset"] = style.multiple_bond_offset
        tasks.append(task)
    for index, task in enumerate(sorted(tasks, key=lambda item: item.sample_key)):
        task.output_shard = f"shards/{split}-{index // SHARD_SIZE:06d}.tar"
    return tasks


def draw_semantic_svg_joint(
    task: layer_builder.RenderTask,
    chemmol: dict,
    source_render: dict,
) -> tuple[bytes, Chem.Mol, list, dict]:
    style = STYLE_BY_SAMPLE[task.sample_key]
    mol = layer_builder.restore_source_geometry(chemmol, source_render)
    source_params = source_render["render_parameters"]
    kekulize = bool(source_params.get("kekulize", True))
    try:
        prepared = rdMolDraw2D.PrepareMolForDrawing(
            Chem.Mol(mol), kekulize=kekulize, addChiralHs=False, wedgeBonds=True
        )
    except Exception:
        kekulize = False
        prepared = rdMolDraw2D.PrepareMolForDrawing(
            Chem.Mol(mol), kekulize=False, addChiralHs=False, wedgeBonds=True
        )
    drawer = rdMolDraw2D.MolDraw2DSVG(WIDTH, HEIGHT)
    options = drawer.drawOptions()
    options.rotate = task.rotation_degrees
    options.bondLineWidth = task.bond_line_width
    options.fixedFontSize = task.font_size
    options.fontFile = task.font_source_file
    options.padding = 0.08
    options.multipleBondOffset = style.multiple_bond_offset
    options.addStereoAnnotation = False
    options.includeMetadata = False
    options.flagCloseContactsDist = -1
    options.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
    if style.color_mode in {"black_white", "grayscale"}:
        options.useBWAtomPalette()
    drawer.DrawMolecule(prepared)
    atom_points = [drawer.GetDrawCoords(index) for index in range(prepared.GetNumAtoms())]
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    if style.color_mode == "grayscale":
        svg = svg.replace("#000000", "#5A5A5A")
    params = {
        "layout": source_params["layout"],
        "rotation_degrees": task.rotation_degrees,
        "bond_line_width": task.bond_line_width,
        "font_size": task.font_size,
        "padding": 0.08,
        "multiple_bond_offset": style.multiple_bond_offset,
        "color_mode": style.color_mode,
        "kekulize": kekulize,
        "stress_operation": source_params.get("stress_operation"),
        "font_family_id": task.font_family_id,
        "font_file_sha256": task.font_file_sha256,
        "renderer_id": "rdkit-svg-semantic-joint-1.0",
    }
    if "stress_details" in source_params:
        params["stress_details"] = source_params["stress_details"]
    return svg.encode("utf-8"), prepared, atom_points, params


def render_joint(task: layer_builder.RenderTask, chemmol: dict, source_render: dict) -> tuple[dict[str, bytes], dict]:
    generated, metrics = layer_builder.render_and_layers(task, chemmol, source_render)
    render = json.loads(generated[".render.json"])
    layers = json.loads(generated[".layers.json"])
    style = STYLE_BY_SAMPLE[task.sample_key]
    render["render_parameters"]["color_mode"] = style.color_mode
    render["render_parameters"]["multiple_bond_offset"] = style.multiple_bond_offset
    render["render_parameters"]["font_family_group"] = style.font_family_group
    render["render_parameters"]["renderer_id"] = "rdkit-svg-semantic-joint-1.0"
    layers["schema_version"] = "chemvision-layers-0.2"
    layers["source"]["dataset"] = "webdataset_full_v0" if task.split == "train" else "chembl_37_raw"
    layers["render_parameters"]["color_mode"] = style.color_mode
    layers["render_parameters"]["multiple_bond_offset"] = style.multiple_bond_offset
    layers["render_parameters"]["font_family_group"] = style.font_family_group
    layers["render_parameters"]["renderer_id"] = "rdkit-svg-semantic-joint-1.0"
    generated[".render.json"] = json_bytes(render)
    generated[".layers.json"] = json_bytes(layers)
    return generated, metrics


def assemble_joint_sample(task: layer_builder.RenderTask) -> tuple[dict[str, bytes], dict]:
    chemmol_bytes, chemmol, source_render = layer_builder.SOURCE_ASSETS[task.source_sample_key]
    generated, metrics = render_joint(task, chemmol, source_render)
    generated[".chemmol.json"] = chemmol_bytes
    generated[".sdf"] = SDF_BY_MOLECULE[task.molecule_id]
    style = STYLE_BY_SAMPLE[task.sample_key]
    identity = IDENTITY_BY_MOLECULE[task.molecule_id]
    hashes = {name.removeprefix("."): sha256_bytes(value) for name, value in generated.items()}
    meta = {
        "schema_version": "chemvector-joint-meta-1.0",
        "sample_key": task.sample_key,
        "molecule_id": task.molecule_id,
        "inchikey": identity["inchikey"],
        "scaffold": identity["scaffold"],
        "split": task.split,
        "variant": task.variant,
        "source_sample_key": task.source_sample_key,
        "source_shard": task.source_shard,
        "output_shard": task.output_shard,
        "seed": SEED,
        "rdkit_version": rdBase.rdkitVersion,
        "git_revision": GIT_REVISION,
        "renderer_id": "rdkit-svg-semantic-joint-1.0",
        "style": {
            **asdict(style),
            "font_family_id": task.font_family_id,
            "font_size": task.font_size,
            "font_bucket": task.font_bucket,
            "bond_line_width": task.bond_line_width,
            "rotation_degrees": task.rotation_degrees,
        },
        "sha256": hashes,
    }
    generated[".meta.json"] = json_bytes(meta)
    if task.repeat_check:
        repeated, _ = render_joint(task, chemmol, source_render)
        repeated[".chemmol.json"] = chemmol_bytes
        repeated[".sdf"] = SDF_BY_MOLECULE[task.molecule_id]
        repeated_hashes = {name.removeprefix("."): sha256_bytes(value) for name, value in repeated.items()}
        repeated_meta = dict(meta)
        repeated_meta["sha256"] = repeated_hashes
        repeated[".meta.json"] = json_bytes(repeated_meta)
        if {name: sha256_bytes(value) for name, value in generated.items()} != {
            name: sha256_bytes(value) for name, value in repeated.items()
        }:
            raise ValueError("deterministic repeat generation hash mismatch")
    render = json.loads(generated[".render.json"])
    row = {
        "sample_key": task.sample_key,
        "molecule_id": task.molecule_id,
        "inchikey": identity["inchikey"],
        "scaffold": identity["scaffold"],
        "split": task.split,
        "variant": task.variant,
        "source": identity["source"],
        "source_id": identity["source_id"],
        "output_shard": task.output_shard,
        "font_family_id": task.font_family_id,
        "font_family_group": style.font_family_group,
        "font_size": task.font_size,
        "font_bucket": task.font_bucket,
        "line_width_bucket": style.line_width_bucket,
        "bond_line_width": task.bond_line_width,
        "rotation_bucket": style.rotation_bucket,
        "rotation_degrees": task.rotation_degrees,
        "multiple_bond_offset_bucket": style.multiple_bond_offset_bucket,
        "multiple_bond_offset": style.multiple_bond_offset,
        "color_mode": style.color_mode,
        "stress_operation": render["render_parameters"].get("stress_operation") or "",
        "png_sha256": sha256_bytes(generated[".png"]),
        "chemmol_sha256": sha256_bytes(generated[".chemmol.json"]),
        "render_sha256": sha256_bytes(generated[".render.json"]),
        "layers_sha256": sha256_bytes(generated[".layers.json"]),
        "sdf_sha256": sha256_bytes(generated[".sdf"]),
        "meta_sha256": sha256_bytes(generated[".meta.json"]),
        "foreground_iou": metrics["foreground_iou"],
        "unclassified_fraction": metrics["unclassified_fraction"],
    }
    return generated, row


def tar_add_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    info.mtime = 0
    info.mode = 0o640
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(value))


def render_chunk_joint(output_root: str, shard: str, chunk_index: int, tasks: list[layer_builder.RenderTask]) -> dict[str, Any]:
    output = Path(output_root)
    chunk_path = output / f"work/chunks/{Path(shard).stem}-{chunk_index:03d}.tar"
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    rejected = []
    started = time.perf_counter()
    with tarfile.open(chunk_path, "w") as archive:
        for task in sorted(tasks, key=lambda item: item.sample_key):
            try:
                members, row = assemble_joint_sample(task)
                for suffix in JOINT_MEMBERS:
                    tar_add_bytes(archive, task.sample_key + suffix, members[suffix])
                rows.append(row)
            except Exception as error:
                rejected.append(
                    {
                        "source": IDENTITY_BY_MOLECULE.get(task.molecule_id, {}).get("source", ""),
                        "source_id": IDENTITY_BY_MOLECULE.get(task.molecule_id, {}).get("source_id", ""),
                        "stage": "semantic_svg_render",
                        "error_code": type(error).__name__,
                        "error_message": str(error),
                        "replacement": "0",
                    }
                )
    return {
        "shard": shard,
        "chunk_index": chunk_index,
        "chunk_path": str(chunk_path.relative_to(output)),
        "rows": rows,
        "rejected": rejected,
        "elapsed_seconds": time.perf_counter() - started,
        "bytes": chunk_path.stat().st_size,
    }


def init_joint_worker(assets: dict, styles: dict, sdfs: dict, identities: dict) -> None:
    layer_builder.SOURCE_ASSETS = assets
    layer_builder.LAYER_VALIDATOR = None
    layer_builder.RENDER_VALIDATOR = None
    layer_builder.draw_semantic_svg = draw_semantic_svg_joint
    global STYLE_BY_SAMPLE, SDF_BY_MOLECULE, IDENTITY_BY_MOLECULE
    STYLE_BY_SAMPLE = styles
    SDF_BY_MOLECULE = sdfs
    IDENTITY_BY_MOLECULE = identities
    RDLogger.DisableLog("rdApp.*")


def merge_chunks(output: Path, results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_shard: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        by_shard[str(result["shard"])].append(result)
    shards = []
    for shard, chunks in sorted(by_shard.items()):
        target = output / shard
        target.parent.mkdir(parents=True, exist_ok=True)
        members = 0
        with tarfile.open(target, "w") as destination:
            for chunk in sorted(chunks, key=lambda item: int(item["chunk_index"])):
                with tarfile.open(output / str(chunk["chunk_path"])) as source:
                    for member in source:
                        extracted = source.extractfile(member)
                        if extracted is None:
                            raise RuntimeError(f"cannot extract {member.name}")
                        tar_add_bytes(destination, member.name, extracted.read())
                        members += 1
        shards.append({"path": shard, "bytes": target.stat().st_size, "members": members, "sha256": sha256_file(target)})
    shutil.rmtree(output / "work")
    return shards


def render_tasks(
    output: Path,
    tasks: Sequence[layer_builder.RenderTask],
    assets: dict,
    sdfs: dict[str, bytes],
    identities: dict[str, dict[str, str]],
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    output.mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(parents=True, exist_ok=True)
    by_shard: dict[str, list[layer_builder.RenderTask]] = defaultdict(list)
    for task in tasks:
        by_shard[task.output_shard].append(task)
    jobs = []
    for shard, shard_tasks in sorted(by_shard.items()):
        ordered = sorted(shard_tasks, key=lambda item: item.sample_key)
        for chunk_index, start in enumerate(range(0, len(ordered), 250)):
            jobs.append((shard, chunk_index, ordered[start : start + 250]))

    global SDF_BY_MOLECULE, IDENTITY_BY_MOLECULE
    SDF_BY_MOLECULE = sdfs
    IDENTITY_BY_MOLECULE = identities
    layer_builder.SOURCE_ASSETS = assets
    layer_builder.draw_semantic_svg = draw_semantic_svg_joint
    results = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)),
        mp_context=context,
        initializer=init_joint_worker,
        initargs=(assets, STYLE_BY_SAMPLE, sdfs, identities),
    ) as executor:
        futures = {
            executor.submit(render_chunk_joint, str(output), shard, chunk_index, chunk_tasks): (shard, chunk_index)
            for shard, chunk_index, chunk_tasks in jobs
        }
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if completed % 10 == 0 or completed == len(futures):
                rendered = sum(len(item["rows"]) for item in results)
                rejected = sum(len(item["rejected"]) for item in results)
                print(
                    f"Rendered chunks {completed}/{len(futures)}: samples={rendered:,}, rejected={rejected}",
                    flush=True,
                )
    shard_rows = merge_chunks(output, results)
    rows = sorted(
        (row for result in results for row in result["rows"]),
        key=lambda row: row["sample_key"],
    )
    rejected = sorted(
        (row for result in results for row in result["rejected"]),
        key=lambda row: (row["source_id"], row["stage"]),
    )
    return rows, rejected, shard_rows


def style_molecule_manifest(plan_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    result = []
    for row in plan_rows:
        mol = Chem.MolFromSmiles(row["canonical_isomeric_smiles"])
        rings, fused, bridged = ring_features(mol) if mol is not None else (0, 0, 0)
        result.append(
            {
                "molecule_id": row["molecule_id"],
                "source": row["source"],
                "source_id": row["source_id"],
                "inchikey": row["inchikey"],
                "scaffold": row["scaffold"],
                "canonical_isomeric_smiles": row["canonical_isomeric_smiles"],
                "split": "train",
                "heavy_atoms": mol.GetNumHeavyAtoms() if mol is not None else "",
                "rings": rings,
                "fused_rings": fused,
                "bridged_atoms": bridged,
                "chiral_centers": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)) if mol is not None else "",
                "element_bucket": element_bucket(mol) if mol is not None else "",
                "size_bucket": molecule_pipeline.molecule_stratum(mol).split(":")[1] if mol is not None else "",
                "source_structure": row["structure_ref"],
                "image_count": 4,
            }
        )
    return result


def fresh_molecule_manifest(plan_rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "molecule_id": row["molecule_id"],
            "source": row["source"],
            "source_id": row["source_id"],
            "inchikey": row["inchikey"],
            "scaffold": row["scaffold"],
            "canonical_isomeric_smiles": row["canonical_isomeric_smiles"],
            "split": row["split"],
            "heavy_atoms": row["heavy_atoms"],
            "rings": row["rings"],
            "fused_rings": row["fused_rings"],
            "bridged_atoms": row["bridged_atoms"],
            "chiral_centers": row["chiral_centers"],
            "element_bucket": row["element_bucket"],
            "size_bucket": row["size_bucket"],
            "source_structure": row["structure_file"],
            "image_count": 4,
        }
        for row in plan_rows
    ]


def prepare_style_assets(
    plan_rows: Sequence[Mapping[str, str]],
) -> tuple[dict, dict[str, bytes], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    molecule_ids = {row["molecule_id"] for row in plan_rows}
    sample_rows = layer_builder.load_sample_rows(FULL_ROOT, molecule_ids)
    assets = layer_builder.load_source_assets(FULL_ROOT, sample_rows)
    sdfs = {}
    identities = {}
    for row in plan_rows:
        path = FULL_ROOT / row["structure_ref"]
        payload = path.read_bytes()
        sdfs[row["molecule_id"]] = payload
        identities[row["molecule_id"]] = {
            "source": row["source"],
            "source_id": row["source_id"],
            "inchikey": row["inchikey"],
            "scaffold": row["scaffold"],
        }
    return assets, sdfs, identities, sample_rows


def prepare_fresh_assets(
    plan_rows: Sequence[Mapping[str, str]],
) -> tuple[dict, dict[str, bytes], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    molecule_pipeline.SEED = SEED
    assets = {}
    sdfs = {}
    identities = {}
    sample_rows = {}
    for row_index, row in enumerate(plan_rows):
        sdf_path = HERE / row["structure_file"]
        sdf_payload = sdf_path.read_bytes()
        mol = Chem.MolFromMolBlock(
            sdf_payload.decode("utf-8").split("$$$$", 1)[0],
            sanitize=True, removeHs=False, strictParsing=False,
        )
        if mol is None:
            raise RuntimeError(f"cannot parse selected fresh SDF: {row['molecule_id']}")
        mol.SetProp("_Name", row["source_id"])
        record = molecule_pipeline.canonicalize_record("chembl", mol, row["canonical_isomeric_smiles"], row_index)
        if record.molecule_id != row["molecule_id"] or record.inchikey != row["inchikey"]:
            raise RuntimeError(f"fresh asset identity mismatch: {row['molecule_id']}")
        record.split = row["split"]
        chemmol = molecule_pipeline.build_chemmol(record)
        chemmol["structure_ref"] = f"sample:{row['molecule_id']}.sdf"
        chemmol_payload = json_bytes(chemmol)
        for variant_index, variant in enumerate(VARIANTS):
            source_key = f"{row['split']}_{int(row['selection_index']):06d}_{variant}"
            _, source_render = molecule_pipeline.build_render(record, variant, variant_index)
            assets[source_key] = (chemmol_payload, copy.deepcopy(chemmol), source_render)
            sample_rows[source_key] = {
                "molecule_id": row["molecule_id"],
                "variant": variant,
                "shard": "raw/chembl_37.sdf",
            }
        sdfs[row["molecule_id"]] = sdf_payload
        identities[row["molecule_id"]] = {
            "source": "chembl",
            "source_id": row["source_id"],
            "inchikey": row["inchikey"],
            "scaffold": row["scaffold"],
        }
    return assets, sdfs, identities, sample_rows


def output_exists_and_passed(output: Path) -> bool:
    validation = output / "validation.json"
    if not validation.is_file():
        return False
    try:
        return json.loads(validation.read_text(encoding="utf-8")).get("status") == "passed"
    except Exception:
        return False


def reset_output(output: Path) -> None:
    if output.exists():
        sealed = output / "sealed_test"
        if sealed.exists():
            sealed.chmod(0o700)
            for path in sealed.rglob("*"):
                path.chmod(0o700 if path.is_dir() else 0o600)
        shutil.rmtree(output)
    output.mkdir(parents=True)


def build_preview(output: Path, sample_rows: Sequence[Mapping[str, Any]], count: int = 500) -> None:
    preview_root = output / "previews"
    if preview_root.exists():
        shutil.rmtree(preview_root)
    preview_root.mkdir(parents=True)
    selected = sorted(sample_rows, key=lambda row: stable_int(SEED, "preview", row["sample_key"]))[:count]
    wanted_by_shard: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        wanted_by_shard[str(row["output_shard"])].add(str(row["sample_key"]) + ".png")
    payloads = {}
    for shard, wanted in wanted_by_shard.items():
        with tarfile.open(output / shard) as archive:
            for member in archive:
                if member.name in wanted:
                    extracted = archive.extractfile(member)
                    if extracted is not None:
                        payloads[member.name] = extracted.read()
    cards = []
    for row in selected:
        key = str(row["sample_key"])
        with Image.open(io.BytesIO(payloads[key + ".png"])) as image:
            preview = image.convert("RGB")
            preview.thumbnail((256, 256), Image.Resampling.LANCZOS)
            filename = key + ".png"
            preview.save(preview_root / filename, format="PNG", compress_level=1)
        cards.append(
            f"<figure><img src='{html.escape(filename)}'><figcaption>{html.escape(key)}<br>"
            f"{html.escape(str(row['variant']))} / {html.escape(str(row['font_family_id']))}<br>"
            f"{html.escape(str(row['color_mode']))}</figcaption></figure>"
        )
    page = """<!doctype html><html><head><meta charset='utf-8'><title>ChemVector Joint v1 preview</title>
<style>body{font-family:sans-serif;margin:20px;color:#222}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}figure{margin:0;border:1px solid #bbb;padding:8px}img{width:256px;height:256px;object-fit:contain;background:white}figcaption{font-size:12px}</style></head><body><h1>ChemVector Joint v1 preview</h1><div class='grid'>""" + "".join(cards) + "</div></body></html>\n"
    (preview_root / "index.html").write_text(page, encoding="utf-8")


def write_common_manifests(
    output: Path,
    sample_rows: Sequence[Mapping[str, Any]],
    molecule_rows: Sequence[Mapping[str, Any]],
    font_rows: Sequence[Mapping[str, Any]],
    rejected_rows: Sequence[Mapping[str, Any]],
) -> None:
    clean_samples = [{field: row.get(field, "") for field in SAMPLE_FIELDS} for row in sample_rows]
    write_csv(output / "samples.csv", clean_samples, SAMPLE_FIELDS)
    write_csv(output / "molecules.csv", molecule_rows, MOLECULE_FIELDS)
    write_csv(
        output / "fonts.csv", font_rows,
        ("font_family_id", "font_family", "font_style", "font_file", "font_file_sha256", "license", "license_url", "split", "font_family_group"),
    )
    write_csv(output / "rejected.csv", rejected_rows, REJECTED_FIELDS)


def build_style(workers: int, force: bool = False) -> dict[str, Any]:
    if output_exists_and_passed(STYLE_ROOT) and not force:
        return json.loads((STYLE_ROOT / "summary.json").read_text(encoding="utf-8"))
    plan_rows = load_csv(HERE / "work/style_molecules.plan.csv")
    if len(plan_rows) != STYLE_MOLECULES:
        raise RuntimeError("style plan missing; run --stage plan")
    reset_output(STYLE_ROOT)
    started = time.perf_counter()
    print("Loading style-train labels and SDF assets...", flush=True)
    assets, sdfs, identities, sample_rows = prepare_style_assets(plan_rows)
    global STYLE_BY_SAMPLE
    STYLE_BY_SAMPLE = {}
    layer_builder.SOURCE_ASSETS = assets
    fonts = train_font_specs()
    font_rows = copy_fonts(STYLE_ROOT, fonts)
    tasks = build_style_tasks(plan_rows, sample_rows, fonts, "train")
    print(f"Rendering {len(tasks):,} style-train samples...", flush=True)
    rendered, rejected, shards = render_tasks(STYLE_ROOT, tasks, assets, sdfs, identities, workers)
    molecules = style_molecule_manifest(plan_rows)
    write_common_manifests(STYLE_ROOT, rendered, molecules, font_rows, rejected)
    if len(rendered) == STYLE_SAMPLES:
        build_preview(STYLE_ROOT, rendered)
    summary = dataset_summary("style_train_100k", rendered, molecules, font_rows, rejected, shards, time.perf_counter() - started)
    write_json(STYLE_ROOT / "summary.json", summary)
    return summary


def build_fresh(workers: int, force: bool = False) -> dict[str, Any]:
    if output_exists_and_passed(FRESH_ROOT) and not force:
        return json.loads((FRESH_ROOT / "summary.json").read_text(encoding="utf-8"))
    plan_rows = load_csv(HERE / "work/fresh_molecules.plan.csv")
    if len(plan_rows) != FRESH_DEV_MOLECULES + SEALED_MOLECULES:
        raise RuntimeError("fresh plan missing; run --stage plan")
    reset_output(FRESH_ROOT)
    started = time.perf_counter()
    print("Building fresh ChEMBL labels and source layouts...", flush=True)
    assets, sdfs, identities, sample_rows = prepare_fresh_assets(plan_rows)
    global STYLE_BY_SAMPLE
    STYLE_BY_SAMPLE = {}
    layer_builder.SOURCE_ASSETS = assets
    dev_plan = [row for row in plan_rows if row["split"] == "fresh_dev"]
    sealed_plan = [row for row in plan_rows if row["split"] == "sealed_test"]
    dev_fonts = fresh_dev_font_specs()
    sealed_fonts = sealed_font_specs()
    font_rows = copy_fonts(FRESH_ROOT, [*dev_fonts, *sealed_fonts])
    dev_tasks = build_style_tasks(dev_plan, sample_rows, dev_fonts, "fresh_dev")
    sealed_tasks = build_style_tasks(sealed_plan, sample_rows, sealed_fonts, "sealed_test")
    for task in sealed_tasks:
        task.output_shard = f"sealed_test/{Path(task.output_shard).name}"
    print(f"Rendering {len(dev_tasks) + len(sealed_tasks):,} fresh evaluation samples...", flush=True)
    rendered, rejected, shards = render_tasks(
        FRESH_ROOT, [*dev_tasks, *sealed_tasks], assets, sdfs, identities, workers
    )
    molecules = fresh_molecule_manifest(plan_rows)
    dev_rows = [row for row in rendered if row["split"] == "fresh_dev"]
    sealed_rows = [row for row in rendered if row["split"] == "sealed_test"]
    write_common_manifests(FRESH_ROOT, dev_rows, [row for row in molecules if row["split"] == "fresh_dev"], font_rows, rejected)
    sealed_root = FRESH_ROOT / "sealed_test"
    sealed_root.mkdir(exist_ok=True)
    write_csv(sealed_root / "samples.csv", sealed_rows, SAMPLE_FIELDS)
    write_csv(sealed_root / "molecules.csv", [row for row in molecules if row["split"] == "sealed_test"], MOLECULE_FIELDS)
    if len(dev_rows) == FRESH_DEV_MOLECULES * 4:
        build_preview(FRESH_ROOT, dev_rows)
    summary = dataset_summary("fresh_eval_12k", rendered, molecules, font_rows, rejected, shards, time.perf_counter() - started)
    write_json(FRESH_ROOT / "summary.json", summary)
    return summary


def dataset_summary(
    dataset_name: str,
    samples: Sequence[Mapping[str, Any]],
    molecules: Sequence[Mapping[str, Any]],
    fonts: Sequence[Mapping[str, Any]],
    rejected: Sequence[Mapping[str, Any]],
    shards: Sequence[Mapping[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    actual_shards = []
    root = STYLE_ROOT if dataset_name == "style_train_100k" else FRESH_ROOT
    selection_replacements = (
        len(load_csv(HERE / "work/fresh_label_replacements.csv"))
        if dataset_name == "fresh_eval_12k"
        else 0
    )
    for path in sorted([*(root / "shards").glob("*.tar"), *(root / "sealed_test").glob("*.tar")]):
        actual_shards.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": "chemvector-joint-summary-1.0",
        "dataset": dataset_name,
        "status": "generated",
        "seed": SEED,
        "git_revision": GIT_REVISION,
        "rdkit_version": rdBase.rdkitVersion,
        "renderer_id": "rdkit-svg-semantic-joint-1.0",
        "molecules": len(molecules),
        "samples": len(samples),
        "rejected": len(rejected),
        "automatic_replacements": selection_replacements,
        "split_counts": dict(Counter(str(row["split"]) for row in samples)),
        "source_counts": dict(Counter(str(row["source"]) for row in molecules)),
        "variant_counts": dict(Counter(str(row["variant"]) for row in samples)),
        "color_mode_counts": dict(Counter(str(row["color_mode"]) for row in samples)),
        "font_family_counts": dict(Counter(str(row["font_family_id"]) for row in samples)),
        "font_family_group_counts": dict(Counter(str(row["font_family_group"]) for row in samples)),
        "font_bucket_counts": dict(Counter(str(row["font_bucket"]) for row in samples)),
        "line_width_bucket_counts": dict(Counter(str(row["line_width_bucket"]) for row in samples)),
        "rotation_bucket_counts": dict(Counter(str(row["rotation_bucket"]) for row in samples)),
        "multiple_bond_offset_bucket_counts": dict(Counter(str(row["multiple_bond_offset_bucket"]) for row in samples)),
        "stress_operation_counts": dict(Counter(str(row["stress_operation"]) for row in samples if row["stress_operation"])),
        "size_bucket_counts": dict(Counter(str(row["size_bucket"]) for row in molecules)),
        "element_bucket_counts": dict(Counter(str(row["element_bucket"]) for row in molecules)),
        "font_files": len(fonts),
        "shards": actual_shards,
        "bytes": sum(item["bytes"] for item in actual_shards),
        "elapsed_seconds": round(elapsed_seconds, 3),
    }


def all_dataset_rows(output: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    samples = load_csv(output / "samples.csv")
    molecules = load_csv(output / "molecules.csv")
    if output == FRESH_ROOT:
        samples.extend(load_csv(output / "sealed_test/samples.csv"))
        molecules.extend(load_csv(output / "sealed_test/molecules.csv"))
    return samples, molecules


def audit_tar_package(
    output: Path,
    sample_rows: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    row_by_key = {row["sample_key"]: row for row in sample_rows}
    errors: list[str] = []
    tar_samples = 0
    tar_members = 0
    iou_min = 1.0
    unclassified_max = 0.0
    chemically_validated: set[str] = set()
    style_schema = fastjsonschema.compile(
        json.loads((HERE / "schemas/chemvector-joint-style-1.0.schema.json").read_text(encoding="utf-8"))
    )
    tar_paths = sorted([*(output / "shards").glob("*.tar"), *(output / "sealed_test").glob("*.tar")])
    for shard_index, shard in enumerate(tar_paths, 1):
        shard_relative = shard.relative_to(output).as_posix()
        grouped: dict[str, dict[str, bytes]] = defaultdict(dict)
        with tarfile.open(shard) as archive:
            for member in archive:
                if not member.isfile():
                    continue
                tar_members += 1
                suffix = next(
                    (value for value in sorted(JOINT_MEMBERS, key=len, reverse=True) if member.name.endswith(value)),
                    None,
                )
                if suffix is None:
                    errors.append(f"unexpected_member:{shard.name}:{member.name}")
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    errors.append(f"unreadable_member:{shard.name}:{member.name}")
                    continue
                grouped[member.name[: -len(suffix)]][suffix] = extracted.read()
        for key, members in grouped.items():
            tar_samples += 1
            if set(members) != set(JOINT_MEMBERS):
                errors.append(f"incomplete_sample:{key}:{sorted(members)}")
                continue
            row = row_by_key.get(key)
            if row is None:
                errors.append(f"unmanifested_sample:{key}")
                continue
            try:
                with Image.open(io.BytesIO(members[".png"])) as image:
                    if image.size != (WIDTH, HEIGHT) or image.mode != "RGB":
                        errors.append(f"png_format:{key}:{image.size}:{image.mode}")
                    combined = np.asarray(image).copy().min(axis=2) < 255
                with Image.open(io.BytesIO(members[".text.png"])) as image:
                    text_mask = np.asarray(image.convert("L")).copy() > 0
                with Image.open(io.BytesIO(members[".bond.png"])) as image:
                    bond_mask = np.asarray(image.convert("L")).copy() > 0
                with Image.open(io.BytesIO(members[".semantic.png"])) as image:
                    semantic = np.asarray(image.convert("L")).copy()
                expected_semantic = np.zeros_like(semantic)
                expected_semantic[bond_mask] = 1
                expected_semantic[text_mask] = 2
                expected_semantic[text_mask & bond_mask] = 3
                if not np.array_equal(semantic, expected_semantic):
                    errors.append(f"semantic_mismatch:{key}")
                classified = text_mask | bond_mask
                union = int((combined | classified).sum())
                iou = int((combined & classified).sum()) / max(union, 1)
                unclassified = int((combined & ~classified).sum()) / max(int(combined.sum()), 1)
                iou_min = min(iou_min, iou)
                unclassified_max = max(unclassified_max, unclassified)
                if iou < 0.999 or unclassified > 0.001:
                    errors.append(f"foreground_iou:{key}:{iou:.8f}:{unclassified:.8f}")

                chemmol = json.loads(members[".chemmol.json"])
                render = json.loads(members[".render.json"])
                layers = json.loads(members[".layers.json"])
                meta = json.loads(members[".meta.json"])
                if row["output_shard"] != shard_relative or meta.get("output_shard") != shard_relative:
                    errors.append(
                        f"output_shard_mismatch:{key}:{shard_relative}:"
                        f"{row['output_shard']}:{meta.get('output_shard')}"
                    )
                if not (
                    chemmol["molecule_id"] == render["molecule_id"] == layers["molecule_id"] == row["molecule_id"]
                ):
                    errors.append(f"molecule_id_mismatch:{key}")
                style_schema(meta["style"])
                if layers.get("schema_version") != "chemvision-layers-0.2":
                    errors.append(f"layers_schema:{key}")
                if layers["render_parameters"]["color_mode"] != row["color_mode"]:
                    errors.append(f"layers_color:{key}")
                if render["render_parameters"]["multiple_bond_offset"] != float(row["multiple_bond_offset"]):
                    errors.append(f"render_offset:{key}")
                expected_hashes = {
                    ".png": row["png_sha256"],
                    ".chemmol.json": row["chemmol_sha256"],
                    ".render.json": row["render_sha256"],
                    ".layers.json": row["layers_sha256"],
                    ".sdf": row["sdf_sha256"],
                    ".meta.json": row["meta_sha256"],
                }
                for suffix, expected in expected_hashes.items():
                    if sha256_bytes(members[suffix]) != expected:
                        errors.append(f"manifest_hash:{key}{suffix}")
                for name, expected in meta["sha256"].items():
                    suffix = "." + name
                    if suffix not in members or sha256_bytes(members[suffix]) != expected:
                        errors.append(f"meta_hash:{key}:{name}")
                if row["molecule_id"] not in chemically_validated:
                    reconstructed = molecule_pipeline.mol_from_chemmol(chemmol)
                    Chem.SanitizeMol(reconstructed)
                    sdf_mol = Chem.MolFromMolBlock(
                        members[".sdf"].decode("utf-8").split("$$$$", 1)[0],
                        sanitize=True, removeHs=False, strictParsing=False,
                    )
                    if sdf_mol is None:
                        errors.append(f"sdf_parse:{key}")
                    else:
                        chemmol_key = Chem.MolToInchiKey(reconstructed)
                        sdf_key = Chem.MolToInchiKey(sdf_mol)
                        if chemmol_key != row["inchikey"] or sdf_key != row["inchikey"]:
                            errors.append(f"chemical_roundtrip:{key}:{chemmol_key}:{sdf_key}")
                    chemically_validated.add(row["molecule_id"])
            except Exception as error:
                errors.append(f"sample_validation:{key}:{type(error).__name__}:{error}")
        if shard_index % 5 == 0 or shard_index == len(tar_paths):
            print(
                f"Validated tar shards {shard_index}/{len(tar_paths)}: samples={tar_samples:,}, errors={len(errors)}",
                flush=True,
            )
    return {
        "tar_samples": tar_samples,
        "tar_members": tar_members,
        "shards": len(tar_paths),
        "foreground_iou_min": iou_min,
        "unclassified_foreground_fraction_max": unclassified_max,
        "chemical_roundtrip_molecules": len(chemically_validated),
        "errors": errors,
    }


def style_layout_audit(samples: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    fields = (
        "font_family_id", "font_bucket", "line_width_bucket",
        "rotation_bucket", "multiple_bond_offset_bucket", "color_mode",
    )
    missing: dict[str, list[str]] = {}
    for field in fields:
        by_value: dict[str, set[str]] = defaultdict(set)
        for row in samples:
            by_value[row[field]].add(row["variant"])
        bad = [value for value, variants in by_value.items() if variants != set(VARIANTS)]
        if bad:
            missing[field] = bad
    return {"status": "passed" if not missing else "failed", "missing_layouts": missing}


def validate_dataset(output: Path, overlap: Mapping[str, Any]) -> dict[str, Any]:
    samples, molecules = all_dataset_rows(output)
    rejected = load_csv(output / "rejected.csv")
    audit = audit_tar_package(output, samples)
    errors = list(audit.pop("errors"))
    variants: dict[str, set[str]] = defaultdict(set)
    for row in samples:
        variants[row["molecule_id"]].add(row["variant"])
    incomplete = [molecule_id for molecule_id, values in variants.items() if values != set(VARIANTS)]
    if incomplete or len(variants) != len(molecules):
        errors.append(f"incomplete_variants:{len(incomplete)}:{len(variants)}:{len(molecules)}")
    unresolved_rejected = [row for row in rejected if row.get("replacement") != "1"]
    selection_replacements = (
        len(load_csv(HERE / "work/fresh_label_replacements.csv"))
        if output == FRESH_ROOT
        else 0
    )
    if unresolved_rejected:
        errors.append(f"unresolved_rejected:{len(unresolved_rejected)}")
    if output == STYLE_ROOT:
        if len(samples) != STYLE_SAMPLES or len(molecules) != STYLE_MOLECULES:
            errors.append(f"style_counts:{len(samples)}:{len(molecules)}")
        if set(row["split"] for row in samples) != {"train"}:
            errors.append("style_nontrain_split")
        if Counter(row["source"] for row in molecules) != Counter({"chembl": 21_250, "coconut": 3_750}):
            errors.append(f"style_source_ratio:{dict(Counter(row['source'] for row in molecules))}")
        if sum(row["color_mode"] == "black_white" for row in samples) < 60_000:
            errors.append("style_black_white_below_60_percent")
        layout_audit = style_layout_audit(samples)
        if layout_audit["status"] != "passed":
            errors.append(f"style_layout_binding:{layout_audit['missing_layouts']}")
        benchmark_keys = set(overlap["benchmark_inchikeys"])
        benchmark_overlap = len({row["inchikey"] for row in molecules} & benchmark_keys)
        if benchmark_overlap:
            errors.append(f"style_benchmark_overlap:{benchmark_overlap}")
        old_scaffold_overlap = None
        cross_split_scaffold_overlap = 0
        font_overlap = 0
    else:
        if len(samples) != 12_000 or len(molecules) != 3_000:
            errors.append(f"fresh_counts:{len(samples)}:{len(molecules)}")
        split_counts = Counter(row["split"] for row in samples)
        if split_counts != Counter({"fresh_dev": 8_000, "sealed_test": 4_000}):
            errors.append(f"fresh_split_counts:{dict(split_counts)}")
        old_ids = set(overlap["molecule_ids"])
        old_keys = set(overlap["inchikeys"])
        old_scaffolds = set(overlap["scaffolds"])
        old_molecule_overlap = len({row["molecule_id"] for row in molecules} & old_ids)
        old_inchikey_overlap = len({row["inchikey"] for row in molecules} & old_keys)
        old_scaffold_overlap = len({row["scaffold"] for row in molecules} & old_scaffolds)
        if old_molecule_overlap or old_inchikey_overlap or old_scaffold_overlap:
            errors.append(f"old_overlap:{old_molecule_overlap}:{old_inchikey_overlap}:{old_scaffold_overlap}")
        by_split = defaultdict(lambda: {"molecule": set(), "inchikey": set(), "scaffold": set(), "font": set()})
        for row in molecules:
            by_split[row["split"]]["molecule"].add(row["molecule_id"])
            by_split[row["split"]]["inchikey"].add(row["inchikey"])
            by_split[row["split"]]["scaffold"].add(row["scaffold"])
        for row in samples:
            by_split[row["split"]]["font"].add(row["font_family_id"])
        cross = {
            name: len(by_split["fresh_dev"][name] & by_split["sealed_test"][name])
            for name in ("molecule", "inchikey", "scaffold", "font")
        }
        if any(cross.values()):
            errors.append(f"fresh_cross_split_overlap:{cross}")
        cross_split_scaffold_overlap = cross["scaffold"]
        font_overlap = cross["font"]
        layout_audit = style_layout_audit(samples)
        benchmark_overlap = None
    expected_repeat = 1000
    if not (output / "previews/index.html").is_file():
        errors.append("missing_previews")
    return {
        "schema_version": "chemvector-joint-validation-1.0",
        "status": "passed" if not errors else "failed",
        "dataset": output.name,
        "samples": len(samples),
        "molecules": len(molecules),
        "rejected": len(rejected),
        "automatic_replacements": len(rejected) - len(unresolved_rejected) + selection_replacements,
        "deterministic_repeat_samples": expected_repeat,
        "old_dataset_scaffold_overlap": old_scaffold_overlap,
        "cross_split_scaffold_overlap": cross_split_scaffold_overlap,
        "font_family_overlap": font_overlap,
        "benchmark_inchikey_overlap": benchmark_overlap,
        "style_layout_audit": layout_audit,
        **audit,
        "errors": errors[:200],
    }


def build_commitment(output: Path) -> dict[str, Any]:
    samples, molecules = all_dataset_rows(output)
    tar_paths = sorted([*(output / "shards").glob("*.tar"), *(output / "sealed_test").glob("*.tar")])
    tar_rows = [
        {
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in tar_paths
    ]
    samples_files = [output / "samples.csv"]
    if (output / "sealed_test/samples.csv").is_file():
        samples_files.append(output / "sealed_test/samples.csv")
    font_rows = load_csv(output / "fonts.csv")
    commitment = {
        "schema_version": "chemvector-joint-commitment-1.0",
        "dataset": output.name,
        "seed": SEED,
        "git_revision": GIT_REVISION,
        "rdkit_version": rdBase.rdkitVersion,
        "samples": len(samples),
        "molecules": len(molecules),
        "split_counts": dict(Counter(row["split"] for row in samples)),
        "samples_csv": [
            {"path": path.relative_to(output).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in samples_files
        ],
        "tars": tar_rows,
        "molecule_id_set_sha256": set_sha256(row["molecule_id"] for row in molecules),
        "inchikey_set_sha256": set_sha256(row["inchikey"] for row in molecules),
        "scaffold_set_sha256": set_sha256(row["scaffold"] for row in molecules),
        "renderer_style_schema": {
            "path": "schemas/chemvector-joint-style-1.0.schema.json",
            "sha256": sha256_file(HERE / "schemas/chemvector-joint-style-1.0.schema.json"),
        },
        "font_family_groups_sha256": sha256_bytes(
            json_bytes(sorted((row["font_family_id"], row["font_family_group"], row["split"]) for row in font_rows))
        ),
        "sealed_access": "locked_until_checkpoint_gate" if output == FRESH_ROOT else "not_applicable",
    }
    commitment["commitment_sha256"] = sha256_bytes(json_bytes(commitment))
    write_json(output / "commitment.json", commitment)
    return commitment


def replacement_candidates(
    source: str,
    used_ids: set[str],
    benchmark_keys: set[str],
) -> list[dict[str, str]]:
    candidates = []
    with (FULL_ROOT / "molecules.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["split"] == "train"
                and row["source"] == source
                and "supplement" not in row["shard"]
                and row["molecule_id"] not in used_ids
                and row["inchikey"] not in benchmark_keys
            ):
                candidates.append(row)
    return sorted(
        candidates,
        key=lambda row: (
            stable_int(SEED, "style-replacement", source, row["molecule_id"]),
            row["molecule_id"],
        ),
    )


def patch_tar_samples(
    tar_path: Path,
    replacement_members: Mapping[str, Mapping[str, bytes]],
) -> None:
    temporary = tar_path.with_suffix(".patched.tar")
    keys = set(replacement_members)
    with tarfile.open(tar_path) as source, tarfile.open(temporary, "w") as destination:
        for member in source:
            key = next((candidate for candidate in keys if member.name.startswith(candidate + ".")), None)
            if key is not None:
                continue
            extracted = source.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"cannot read existing tar member: {member.name}")
            tar_add_bytes(destination, member.name, extracted.read())
        for key in sorted(keys):
            for suffix in JOINT_MEMBERS:
                tar_add_bytes(destination, key + suffix, replacement_members[key][suffix])
    temporary.replace(tar_path)


def repair_style_rejections(workers: int = 1) -> dict[str, Any]:
    rejected = load_csv(STYLE_ROOT / "rejected.csv")
    unresolved = [row for row in rejected if row.get("replacement") != "1"]
    if not unresolved:
        return {"status": "nothing_to_repair", "replacements": 0}
    failed_ids = {f"{row['source']}:{row['source_id']}" for row in unresolved}
    if len(failed_ids) != len(unresolved):
        raise RuntimeError("multiple rejected variants for one molecule require manual audit")
    plan_rows = load_csv(HERE / "work/style_molecules.plan.csv")
    plan_by_id = {row["molecule_id"]: row for row in plan_rows}
    if not failed_ids.issubset(plan_by_id):
        raise RuntimeError(f"rejected molecule absent from plan: {failed_ids - set(plan_by_id)}")
    initial_backup = HERE / "work/style_molecules.initial.csv"
    if not initial_backup.exists():
        shutil.copyfile(HERE / "work/style_molecules.plan.csv", initial_backup)

    all_sample_rows = layer_builder.load_sample_rows(FULL_ROOT, set(plan_by_id))
    global STYLE_BY_SAMPLE, SDF_BY_MOLECULE, IDENTITY_BY_MOLECULE
    STYLE_BY_SAMPLE = {}
    layer_builder.SOURCE_ASSETS = {}
    original_tasks = build_style_tasks(plan_rows, all_sample_rows, train_font_specs(), "train")
    templates = {
        (task.molecule_id, task.variant): task
        for task in original_tasks if task.molecule_id in failed_ids
    }
    if len(templates) != len(failed_ids) * 4:
        raise RuntimeError("failed molecule task templates incomplete")

    overlap = build_overlap_index(force=False)
    benchmark_keys = set(overlap["benchmark_inchikeys"])
    used_ids = set(plan_by_id)
    generated_by_shard: dict[str, dict[str, dict[str, bytes]]] = defaultdict(dict)
    new_rows = []
    new_molecule_rows = []
    replacement_records = []
    new_plan_by_failed = {}
    layer_builder.draw_semantic_svg = draw_semantic_svg_joint

    for failed_id in sorted(failed_ids):
        failed_plan = plan_by_id[failed_id]
        source = failed_plan["source"]
        chosen = None
        chosen_rows = None
        chosen_members = None
        chosen_molecule_row = None
        last_error = ""
        for candidate in replacement_candidates(source, used_ids, benchmark_keys):
            candidate = dict(candidate)
            candidate["joint_index"] = failed_plan["joint_index"]
            try:
                assets, sdfs, identities, candidate_samples = prepare_style_assets([candidate])
                layer_builder.SOURCE_ASSETS = assets
                SDF_BY_MOLECULE = sdfs
                IDENTITY_BY_MOLECULE = identities
                task_rows = []
                member_rows = {}
                for variant in VARIANTS:
                    template = templates[(failed_id, variant)]
                    source_key, source_row = next(
                        (key, row)
                        for key, row in candidate_samples.items()
                        if row["variant"] == variant
                    )
                    style = STYLE_BY_SAMPLE[template.sample_key]
                    assets[source_key][2]["render_parameters"]["multiple_bond_offset"] = style.multiple_bond_offset
                    task = layer_builder.RenderTask(
                        sample_key=template.sample_key,
                        molecule_id=candidate["molecule_id"],
                        split="train",
                        variant=variant,
                        source_sample_key=source_key,
                        source_shard=source_row["shard"],
                        output_shard=template.output_shard,
                        font_family_id=template.font_family_id,
                        font_source_file=template.font_source_file,
                        font_file_sha256=template.font_file_sha256,
                        font_size=template.font_size,
                        font_bucket=template.font_bucket,
                        bond_line_width=template.bond_line_width,
                        rotation_degrees=template.rotation_degrees,
                        repeat_check=True,
                    )
                    members, row = assemble_joint_sample(task)
                    task_rows.append(row)
                    member_rows[task.sample_key] = members
                molecule_row = style_molecule_manifest([candidate])[0]
                chosen = candidate
                chosen_rows = task_rows
                chosen_members = member_rows
                chosen_molecule_row = molecule_row
                break
            except Exception as error:
                last_error = f"{type(error).__name__}: {error}"
                continue
        if chosen is None or chosen_rows is None or chosen_members is None or chosen_molecule_row is None:
            raise RuntimeError(f"cannot replace {failed_id}: {last_error}")
        used_ids.add(chosen["molecule_id"])
        new_plan_by_failed[failed_id] = chosen
        new_rows.extend(chosen_rows)
        new_molecule_rows.append(chosen_molecule_row)
        for row in chosen_rows:
            generated_by_shard[row["output_shard"]][row["sample_key"]] = chosen_members[row["sample_key"]]
        replacement_records.append(
            {
                "failed_molecule_id": failed_id,
                "failed_source": source,
                "replacement_molecule_id": chosen["molecule_id"],
                "replacement_source_id": chosen["source_id"],
                "joint_index": failed_plan["joint_index"],
                "reason": next(row["error_message"] for row in unresolved if f"{row['source']}:{row['source_id']}" == failed_id),
            }
        )
        print(f"Replacement selected: {failed_id} -> {chosen['molecule_id']}", flush=True)

    for shard, member_rows in generated_by_shard.items():
        patch_tar_samples(STYLE_ROOT / shard, member_rows)
        print(f"Patched {shard} with {len(member_rows)} replacement samples", flush=True)

    existing_samples = load_csv(STYLE_ROOT / "samples.csv")
    existing_samples = [row for row in existing_samples if row["molecule_id"] not in failed_ids]
    existing_samples.extend({field: str(row.get(field, "")) for field in SAMPLE_FIELDS} for row in new_rows)
    existing_samples.sort(key=lambda row: row["sample_key"])
    write_csv(STYLE_ROOT / "samples.csv", existing_samples, SAMPLE_FIELDS)
    existing_molecules = load_csv(STYLE_ROOT / "molecules.csv")
    existing_molecules = [row for row in existing_molecules if row["molecule_id"] not in failed_ids]
    existing_molecules.extend({field: str(row.get(field, "")) for field in MOLECULE_FIELDS} for row in new_molecule_rows)
    existing_molecules.sort(key=lambda row: row["molecule_id"])
    write_csv(STYLE_ROOT / "molecules.csv", existing_molecules, MOLECULE_FIELDS)
    for row in rejected:
        if f"{row['source']}:{row['source_id']}" in failed_ids:
            row["replacement"] = "1"
    write_csv(STYLE_ROOT / "rejected.csv", rejected, REJECTED_FIELDS)

    updated_plan = []
    for row in plan_rows:
        replacement = new_plan_by_failed.get(row["molecule_id"])
        updated_plan.append(replacement if replacement is not None else row)
    updated_plan.sort(key=lambda row: int(row["joint_index"]))
    write_csv(HERE / "work/style_molecules.plan.csv", updated_plan, tuple(updated_plan[0]))
    write_csv(
        HERE / "work/style_replacements.csv",
        replacement_records,
        ("failed_molecule_id", "failed_source", "replacement_molecule_id", "replacement_source_id", "joint_index", "reason"),
    )
    font_rows = load_csv(STYLE_ROOT / "fonts.csv")
    summary = dataset_summary(
        "style_train_100k", existing_samples, existing_molecules, font_rows, rejected, [], 0.0
    )
    summary["initial_rejected"] = len(rejected)
    summary["automatic_replacements"] = len(replacement_records)
    write_json(STYLE_ROOT / "summary.json", summary)
    build_preview(STYLE_ROOT, existing_samples)
    return {"status": "repaired", "replacements": len(replacement_records), "records": replacement_records}


def lock_sealed_test() -> None:
    sealed = FRESH_ROOT / "sealed_test"
    if not sealed.is_dir():
        raise RuntimeError("sealed_test directory missing")
    for path in sealed.rglob("*"):
        path.chmod(0o000 if path.is_file() else 0o000)
    sealed.chmod(0o000)


def write_completion_marker() -> None:
    style_validation_path = STYLE_ROOT / "validation.json"
    fresh_validation_path = FRESH_ROOT / "validation.json"
    if not style_validation_path.is_file() or not fresh_validation_path.is_file():
        return
    style_validation = json.loads(style_validation_path.read_text(encoding="utf-8"))
    fresh_validation = json.loads(fresh_validation_path.read_text(encoding="utf-8"))
    if style_validation.get("status") != "passed" or fresh_validation.get("status") != "passed":
        return
    write_json(
        HERE / "validation.json",
        {
            "status": "passed",
            "style_train_samples": style_validation["samples"],
            "fresh_dev_samples": 8_000,
            "sealed_test_samples": 4_000,
            "old_dataset_scaffold_overlap": fresh_validation["old_dataset_scaffold_overlap"],
            "cross_split_scaffold_overlap": fresh_validation["cross_split_scaffold_overlap"],
        },
    )


def finalize_dataset(output: Path, overlap: Mapping[str, Any]) -> dict[str, Any]:
    print(f"Validating {output.name} tar members and chemical roundtrips...", flush=True)
    validation = validate_dataset(output, overlap)
    write_json(output / "validation.json", validation)
    if validation["status"] != "passed":
        raise RuntimeError(f"{output.name} validation failed: {validation['errors'][:10]}")
    commitment = build_commitment(output)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    summary["status"] = "passed"
    summary["validation_status"] = "passed"
    summary["commitment_sha256"] = commitment["commitment_sha256"]
    write_json(output / "summary.json", summary)
    if output == FRESH_ROOT:
        write_completion_marker()
        lock_sealed_test()
    return validation


def run_stage(stage: str, workers: int, force: bool) -> dict[str, Any]:
    RDLogger.DisableLog("rdApp.*")
    if rdBase.rdkitVersion != "2026.03.3":
        raise RuntimeError(f"Expected RDKit 2026.03.3, got {rdBase.rdkitVersion}")
    if stage == "plan":
        return plan(force=force)
    if not (HERE / "work/plan_validation.json").is_file():
        plan(force=False)
    overlap = build_overlap_index(force=False)
    if stage == "style":
        summary = build_style(workers, force=force)
        validation = finalize_dataset(STYLE_ROOT, overlap)
        return {"summary": summary, "validation": validation}
    if stage == "repair-style":
        repair = repair_style_rejections(workers)
        validation = finalize_dataset(STYLE_ROOT, overlap)
        return {"repair": repair, "validation": validation}
    if stage == "fresh":
        summary = build_fresh(workers, force=force)
        validation = finalize_dataset(FRESH_ROOT, overlap)
        return {"summary": summary, "validation": validation}
    if stage == "validate":
        results = {}
        for output in (STYLE_ROOT, FRESH_ROOT):
            if output.exists():
                if output == FRESH_ROOT and (output / "sealed_test").exists():
                    (output / "sealed_test").chmod(0o700)
                    for path in (output / "sealed_test").rglob("*"):
                        path.chmod(0o600)
                results[output.name] = finalize_dataset(output, overlap)
        return results
    plan_result = plan(force=force)
    style_summary = build_style(workers, force=force)
    style_validation = finalize_dataset(STYLE_ROOT, overlap)
    fresh_summary = build_fresh(workers, force=force)
    fresh_validation = finalize_dataset(FRESH_ROOT, overlap)
    return {
        "plan": plan_result,
        "style_summary": style_summary,
        "style_validation": style_validation,
        "fresh_summary": fresh_summary,
        "fresh_validation": fresh_validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("plan", "style", "repair-style", "fresh", "validate", "all"), default="all")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    result = run_stage(args.stage, args.workers, args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
