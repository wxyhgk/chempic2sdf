#!/usr/bin/env python3
"""Build and validate a 1000-molecule ChemMol WebDataset sample."""

from __future__ import annotations

import csv
import hashlib
import heapq
import io
import json
import math
import random
import re
import shutil
import tarfile
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2
import fastjsonschema
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import rdCoordGen, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Geometry import Point2D

try:
    from .chemmol_sequence import serialize_chemmolseq as serialize_chemmolseq_v2
except ImportError:
    from chemmol_sequence import serialize_chemmolseq as serialize_chemmolseq_v2


SEED = 20260713
WIDTH = 768
HEIGHT = 768
QUANTIZATION_MAX = 1023
OUTPUT = Path("webdataset_sample_1k")
SOURCE_SPECS = (
    ("chembl", Path("extracted/prepared/train/chembl_150k.sdf"), 830, 25),
    ("coconut", Path("extracted/prepared/train/coconut_complex_30k.sdf"), 170, 15),
)
SPLIT_FRACTIONS = {"train": 0.90, "validation": 0.05, "internal_test": 0.05}
VARIANTS = ("original", "rdkit", "coordgen", "stress")
STRESS_OPERATIONS = ("long_bond", "crowded_subgraph", "near_nonbonded", "crossed_bonds")


@dataclass(slots=True)
class MoleculeRecord:
    source: str
    source_id: str
    molecule_id: str
    mol: Chem.Mol
    source_atom_indices: list[int]
    smiles: str
    inchikey: str
    scaffold: str
    stratum: str
    split: str = ""


def stable_int(*parts: object) -> int:
    value = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def enum_name(value: object) -> str:
    text = str(value)
    return text.rsplit(".", 1)[-1]


def clean_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")


def get_source_id(mol: Chem.Mol, source: str, index: int) -> str:
    keys = {
        "chembl": ("dataset_source_id", "chembl_id", "_Name"),
        "coconut": ("dataset_source_id", "coconut_id", "identifier", "_Name"),
    }
    for key in keys[source]:
        if mol.HasProp(key) and mol.GetProp(key).strip():
            return mol.GetProp(key).strip()
    return f"{source}_{index}"


def element_group(mol: Chem.Mol) -> str:
    elements = {atom.GetSymbol() for atom in mol.GetAtoms()}
    for symbol in ("Se", "B", "Si", "P", "S"):
        if symbol in elements:
            return symbol
    if elements & {"F", "Cl", "Br", "I"}:
        return "halogen"
    return "base"


def molecule_stratum(mol: Chem.Mol) -> str:
    heavy = mol.GetNumHeavyAtoms()
    size = "small" if heavy < 25 else "medium" if heavy < 50 else "large"
    rings = mol.GetRingInfo().NumRings()
    stereo = any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms())
    complexity = "complex" if rings >= 4 or stereo else "standard"
    return f"{element_group(mol)}:{size}:{complexity}"


def scaffold_smiles(mol: Chem.Mol, smiles: str) -> str:
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=True)
    return scaffold if scaffold else f"ACYCLIC:{smiles}"


def select_source(path: Path, source: str, count: int) -> list[tuple[Chem.Mol, str]]:
    heap: list[tuple[int, int, Chem.Mol, str]] = []
    supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False, strictParsing=False)
    for index, mol in enumerate(supplier, 1):
        if mol is None:
            continue
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        priority = stable_int(SEED, source, smiles)
        item = (-priority, index, Chem.Mol(mol), smiles)
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, item)
    selected = [(item[2], item[3]) for item in heap]
    selected.sort(key=lambda item: stable_int(SEED, source, item[1]))
    if len(selected) != count:
        raise RuntimeError(f"Selected {len(selected)} of requested {count} from {source}")
    return selected


def canonicalize_record(source: str, mol: Chem.Mol, smiles: str, index: int) -> MoleculeRecord:
    source_id = get_source_id(mol, source, index)
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True, includeChirality=True, includeIsotopes=True))
    order = sorted(range(mol.GetNumAtoms()), key=lambda atom_index: (ranks[atom_index], atom_index))
    canonical = Chem.RenumberAtoms(mol, order)
    canonical_block = Chem.MolToMolBlock(
        canonical, includeStereo=True, kekulize=True, forceV3000=True
    )
    canonical = Chem.MolFromMolBlock(
        canonical_block, sanitize=True, removeHs=False, strictParsing=False
    )
    if canonical is None:
        raise ValueError(f"Canonical MolBlock roundtrip failed for {source}:{source_id}")
    Chem.AssignStereochemistry(canonical, cleanIt=True, force=True)
    canonical_smiles = Chem.MolToSmiles(canonical, canonical=True, isomericSmiles=True)
    inchikey = Chem.MolToInchiKey(canonical)
    molecule_id = f"{source}:{source_id}"
    return MoleculeRecord(
        source=source,
        source_id=source_id,
        molecule_id=molecule_id,
        mol=canonical,
        source_atom_indices=order,
        smiles=canonical_smiles,
        inchikey=inchikey,
        scaffold=scaffold_smiles(canonical, canonical_smiles),
        stratum=molecule_stratum(canonical),
    )


def assign_scaffold_splits(records: list[MoleculeRecord]) -> dict[str, int]:
    groups: dict[str, list[MoleculeRecord]] = defaultdict(list)
    total_strata = Counter(record.stratum for record in records)
    for record in records:
        groups[record.scaffold].append(record)

    targets = {split: round(len(records) * fraction) for split, fraction in SPLIT_FRACTIONS.items()}
    targets["train"] += len(records) - sum(targets.values())
    counts = Counter()
    split_strata: dict[str, Counter] = {split: Counter() for split in SPLIT_FRACTIONS}
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), stable_int(SEED, "scaffold", item[0])),
    )

    for scaffold, group in ordered_groups:
        group_strata = Counter(record.stratum for record in group)
        scores = {}
        for split, fraction in SPLIT_FRACTIONS.items():
            global_ratio = (counts[split] + len(group)) / max(targets[split], 1)
            stratum_ratios = []
            for stratum, amount in group_strata.items():
                expected = max(total_strata[stratum] * fraction, 1.0)
                stratum_ratios.append((split_strata[split][stratum] + amount) / expected)
            scores[split] = global_ratio + 0.25 * sum(stratum_ratios) / max(len(stratum_ratios), 1)
        split = min(scores, key=lambda value: (scores[value], stable_int(SEED, scaffold, value)))
        for record in group:
            record.split = split
        counts[split] += len(group)
        split_strata[split].update(group_strata)
    return dict(counts)


def build_chemmol(record: MoleculeRecord) -> dict:
    mol = record.mol
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    atoms = []
    for atom in mol.GetAtoms():
        atoms.append(
            {
                "id": atom.GetIdx(),
                "source_atom_index": record.source_atom_indices[atom.GetIdx()],
                "element": atom.GetSymbol(),
                "atomic_number": atom.GetAtomicNum(),
                "formal_charge": atom.GetFormalCharge(),
                "isotope": atom.GetIsotope(),
                "aromatic": atom.GetIsAromatic(),
                "chiral_tag": enum_name(atom.GetChiralTag()),
                "cip_code": atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
                "explicit_h_count": atom.GetNumExplicitHs(),
                "implicit_h_count": atom.GetNumImplicitHs(),
                "no_implicit": atom.GetNoImplicit(),
                "radical_electrons": atom.GetNumRadicalElectrons(),
            }
        )
    bonds = []
    for bond in mol.GetBonds():
        stereo = enum_name(bond.GetStereo())
        ez = "E" if stereo == "STEREOE" else "Z" if stereo == "STEREOZ" else None
        bonds.append(
            {
                "id": bond.GetIdx(),
                "begin_atom": bond.GetBeginAtomIdx(),
                "end_atom": bond.GetEndAtomIdx(),
                "type": enum_name(bond.GetBondType()),
                "aromatic": bond.GetIsAromatic(),
                "conjugated": bond.GetIsConjugated(),
                "stereo": stereo,
                "stereo_atoms": list(bond.GetStereoAtoms()),
                "ez": ez,
            }
        )
    return {
        "schema_version": "chemmol-0.1",
        "molecule_id": record.molecule_id,
        "structure_ref": f"structures/{clean_identifier(record.source)}_{clean_identifier(record.source_id)}.sdf",
        "canonical_isomeric_smiles": record.smiles,
        "inchikey": record.inchikey,
        "atoms": atoms,
        "bonds": bonds,
    }


def mol_from_chemmol(data: dict) -> Chem.Mol:
    rw = Chem.RWMol()
    for atom_data in data["atoms"]:
        atom = Chem.Atom(atom_data["atomic_number"])
        atom.SetFormalCharge(atom_data["formal_charge"])
        atom.SetIsotope(atom_data["isotope"])
        atom.SetIsAromatic(atom_data["aromatic"])
        atom.SetChiralTag(getattr(Chem.ChiralType, atom_data["chiral_tag"]))
        atom.SetNumExplicitHs(atom_data["explicit_h_count"])
        atom.SetNoImplicit(atom_data["no_implicit"])
        atom.SetNumRadicalElectrons(atom_data["radical_electrons"])
        rw.AddAtom(atom)
    for bond_data in data["bonds"]:
        rw.AddBond(
            bond_data["begin_atom"],
            bond_data["end_atom"],
            getattr(Chem.BondType, bond_data["type"]),
        )
        bond = rw.GetBondBetweenAtoms(bond_data["begin_atom"], bond_data["end_atom"])
        bond.SetIsAromatic(bond_data["aromatic"])
        bond.SetIsConjugated(bond_data["conjugated"])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    for bond_data in data["bonds"]:
        bond = mol.GetBondWithIdx(bond_data["id"])
        stereo_atoms = bond_data["stereo_atoms"]
        if len(stereo_atoms) == 2:
            bond.SetStereoAtoms(stereo_atoms[0], stereo_atoms[1])
        bond.SetStereo(getattr(Chem.BondStereo, bond_data["stereo"]))
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    return mol


def valid_original_coordinates(mol: Chem.Mol) -> bool:
    if mol.GetNumConformers() == 0:
        return False
    conformer = mol.GetConformer()
    points = np.array([[conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
    return bool(
        np.isfinite(points).all()
        and np.ptp(points[:, 0]) > 1e-3
        and np.ptp(points[:, 1]) > 1e-3
        and safe_bond_lengths(mol)
    )


def safe_bond_lengths(mol: Chem.Mol, minimum_ratio: float = 0.08) -> bool:
    if mol.GetNumBonds() == 0 or mol.GetNumConformers() == 0:
        return True
    conformer = mol.GetConformer()
    lengths = []
    for bond in mol.GetBonds():
        begin = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
        end = conformer.GetAtomPosition(bond.GetEndAtomIdx())
        lengths.append(math.hypot(begin.x - end.x, begin.y - end.y))
    median = float(np.median(lengths))
    return median > 1e-6 and min(lengths) >= median * minimum_ratio


def median_bond_length(mol: Chem.Mol) -> float:
    conformer = mol.GetConformer()
    lengths = []
    for bond in mol.GetBonds():
        a = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
        b = conformer.GetAtomPosition(bond.GetEndAtomIdx())
        lengths.append(math.hypot(a.x - b.x, a.y - b.y))
    return float(np.median(lengths)) if lengths else 1.5


def component_without_bond(mol: Chem.Mol, start: int, blocked: tuple[int, int]) -> set[int]:
    result = set()
    queue = deque([start])
    while queue:
        atom_index = queue.popleft()
        if atom_index in result:
            continue
        result.add(atom_index)
        for neighbor in mol.GetAtomWithIdx(atom_index).GetNeighbors():
            edge = (atom_index, neighbor.GetIdx())
            if edge == blocked or edge == blocked[::-1]:
                continue
            if neighbor.GetIdx() not in result:
                queue.append(neighbor.GetIdx())
    return result


def stress_layout(mol: Chem.Mol, operation: str, rng: random.Random) -> dict:
    conformer = mol.GetConformer()
    median = max(median_bond_length(mol), 0.2)
    details: dict[str, object] = {"operation": operation}

    if operation == "long_bond":
        candidates = [bond for bond in mol.GetBonds() if not bond.IsInRing()] or list(mol.GetBonds())
        bond = rng.choice(candidates)
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        side = component_without_bond(mol, end, (begin, end))
        a, b = conformer.GetAtomPosition(begin), conformer.GetAtomPosition(end)
        dx, dy = b.x - a.x, b.y - a.y
        length = max(math.hypot(dx, dy), 1e-6)
        factor = rng.uniform(1.5, 4.0)
        shift_x, shift_y = dx / length * median * (factor - 1), dy / length * median * (factor - 1)
        for atom_index in side:
            point = conformer.GetAtomPosition(atom_index)
            conformer.SetAtomPosition(atom_index, (point.x + shift_x, point.y + shift_y, 0.0))
        details.update({"bond_id": bond.GetIdx(), "factor": round(factor, 6)})

    elif operation == "crowded_subgraph":
        center = rng.randrange(mol.GetNumAtoms())
        selected = {center}
        frontier = {center}
        for _ in range(2):
            frontier = {
                neighbor.GetIdx()
                for atom_index in frontier
                for neighbor in mol.GetAtomWithIdx(atom_index).GetNeighbors()
            } - selected
            selected.update(frontier)
        points = [conformer.GetAtomPosition(i) for i in selected]
        cx = sum(point.x for point in points) / len(points)
        cy = sum(point.y for point in points) / len(points)
        factor = rng.uniform(0.42, 0.68)
        for atom_index in selected:
            point = conformer.GetAtomPosition(atom_index)
            conformer.SetAtomPosition(atom_index, (cx + (point.x - cx) * factor, cy + (point.y - cy) * factor, 0.0))
        details.update({"center_atom": center, "atom_ids": sorted(selected), "factor": round(factor, 6)})

    elif operation == "near_nonbonded":
        terminals = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetDegree() == 1]
        source = rng.choice(terminals or list(range(mol.GetNumAtoms())))
        nonneighbors = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetIdx() != source and mol.GetBondBetweenAtoms(source, atom.GetIdx()) is None
        ]
        target = rng.choice(nonneighbors)
        target_point = conformer.GetAtomPosition(target)
        angle = rng.uniform(0, math.tau)
        distance = median * rng.uniform(0.18, 0.42)
        conformer.SetAtomPosition(
            source,
            (target_point.x + math.cos(angle) * distance, target_point.y + math.sin(angle) * distance, 0.0),
        )
        details.update({"source_atom": source, "target_atom": target, "distance_factor": round(distance / median, 6)})

    else:
        pairs = [
            (first, second)
            for i, first in enumerate(mol.GetBonds())
            for second in list(mol.GetBonds())[i + 1 :]
            if len({first.GetBeginAtomIdx(), first.GetEndAtomIdx(), second.GetBeginAtomIdx(), second.GetEndAtomIdx()}) == 4
        ]
        if not pairs:
            return stress_layout(mol, "long_bond", rng)
        first, second = rng.choice(pairs)
        atom_ids = [first.GetBeginAtomIdx(), first.GetEndAtomIdx(), second.GetBeginAtomIdx(), second.GetEndAtomIdx()]
        points = [conformer.GetAtomPosition(i) for i in atom_ids]
        cx = sum(point.x for point in points) / 4
        cy = sum(point.y for point in points) / 4
        half = median * rng.uniform(0.75, 1.15)
        positions = ((cx - half, cy), (cx + half, cy), (cx, cy - half), (cx, cy + half))
        for atom_index, (x, y) in zip(atom_ids, positions):
            conformer.SetAtomPosition(atom_index, (x, y, 0.0))
        details.update({"bond_ids": [first.GetIdx(), second.GetIdx()]})
    return details


def center_coordinates(mol: Chem.Mol) -> None:
    conformer = mol.GetConformer()
    points = [conformer.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    center_x = (min(point.x for point in points) + max(point.x for point in points)) / 2
    center_y = (min(point.y for point in points) + max(point.y for point in points)) / 2
    for atom_index, point in enumerate(points):
        conformer.SetAtomPosition(atom_index, (point.x - center_x, point.y - center_y, 0.0))


def layout_molecule(record: MoleculeRecord, variant: str, sample_index: int) -> tuple[Chem.Mol, dict]:
    mol = Chem.Mol(record.mol)
    rng = random.Random(stable_int(SEED, record.molecule_id, variant))
    stress = None
    if variant == "original":
        if not valid_original_coordinates(mol):
            rdDepictor.Compute2DCoords(mol, canonOrient=True, clearConfs=True)
            layout = "rdkit_fallback"
        else:
            layout = "original"
    elif variant == "rdkit":
        rdDepictor.Compute2DCoords(mol, canonOrient=True, clearConfs=True)
        layout = "rdkit"
    elif variant == "coordgen":
        mol.RemoveAllConformers()
        rdCoordGen.AddCoords(mol)
        layout = "coordgen"
    else:
        rdDepictor.Compute2DCoords(mol, canonOrient=True, clearConfs=True)
        operation = STRESS_OPERATIONS[
            stable_int(SEED, "stress-operation", record.molecule_id) % len(STRESS_OPERATIONS)
        ]
        stress = stress_layout(mol, operation, rng)
        if not safe_bond_lengths(mol):
            rdDepictor.Compute2DCoords(mol, canonOrient=True, clearConfs=True)
            stress = stress_layout(mol, "long_bond", rng)
            stress["fallback_from"] = operation
        layout = "stress"
    center_coordinates(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol, {"layout": layout, "stress": stress}


def rendering_parameters(record: MoleculeRecord, variant: str) -> dict:
    rng = random.Random(stable_int(SEED, "render", record.molecule_id, variant))
    bucket = stable_int(SEED, "color", record.molecule_id, variant) % 100
    color_mode = "black_white" if bucket < 85 else "scan_grayscale" if bucket < 95 else "limited_color"
    return {
        "rotation_degrees": round(rng.uniform(-12.0, 12.0), 6),
        "bond_line_width": round(rng.uniform(1.4, 2.6), 6),
        "font_size": rng.randint(10 if variant == "stress" else 13, 16 if variant == "stress" else 19),
        "padding": round(rng.uniform(0.04, 0.12), 6),
        "multiple_bond_offset": round(rng.uniform(0.08 if variant == "stress" else 0.12, 0.27 if variant == "stress" else 0.20), 6),
        "color_mode": color_mode,
        "kekulize": bool(stable_int(SEED, "kekule", record.molecule_id, variant) % 2),
    }


def prepare_and_draw(mol: Chem.Mol, params: dict, include_png: bool = True) -> tuple[bytes | None, Chem.Mol, list, list]:
    try:
        prepared = rdMolDraw2D.PrepareMolForDrawing(
            Chem.Mol(mol), kekulize=params["kekulize"], addChiralHs=False, wedgeBonds=True
        )
    except Exception:
        params["kekulize"] = False
        prepared = rdMolDraw2D.PrepareMolForDrawing(
            Chem.Mol(mol), kekulize=False, addChiralHs=False, wedgeBonds=True
        )
    drawer = (
        rdMolDraw2D.MolDraw2DCairo(WIDTH, HEIGHT)
        if include_png
        else rdMolDraw2D.MolDraw2DSVG(WIDTH, HEIGHT)
    )
    options = drawer.drawOptions()
    options.rotate = params["rotation_degrees"]
    options.bondLineWidth = params["bond_line_width"]
    options.fixedFontSize = params["font_size"]
    options.padding = params["padding"]
    options.multipleBondOffset = params["multiple_bond_offset"]
    options.addStereoAnnotation = False
    options.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
    if params["color_mode"] != "limited_color":
        options.useBWAtomPalette()
    if "scale_bounds" not in params:
        conformer = prepared.GetConformer()
        coordinates = [conformer.GetAtomPosition(i) for i in range(prepared.GetNumAtoms())]
        radius = max(math.hypot(point.x, point.y) for point in coordinates)
        extent = max(radius * (1.0 + params["padding"]), 0.5) + 0.35
        params["scale_bounds"] = [-round(extent, 6), -round(extent, 6), round(extent, 6), round(extent, 6)]
    min_x, min_y, max_x, max_y = params["scale_bounds"]
    drawer.SetScale(WIDTH, HEIGHT, Point2D(min_x, min_y), Point2D(max_x, max_y))
    drawer.DrawMolecule(prepared)
    atom_points = [drawer.GetDrawCoords(i) for i in range(prepared.GetNumAtoms())]
    bond_rows = []
    for bond in prepared.GetBonds():
        begin = atom_points[bond.GetBeginAtomIdx()]
        end = atom_points[bond.GetEndAtomIdx()]
        bond_rows.append(
            {
                "bond_id": bond.GetIdx(),
                "begin_image_x": round(begin.x, 2),
                "begin_image_y": round(begin.y, 2),
                "end_image_x": round(end.x, 2),
                "end_image_y": round(end.y, 2),
                "display_type": enum_name(bond.GetBondType()),
                "direction": enum_name(bond.GetBondDir()),
            }
        )
    if include_png:
        drawer.FinishDrawing()
        png = drawer.GetDrawingText()
    else:
        png = None
    return png, prepared, atom_points, bond_rows


def degrade_scan(png: bytes, seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    gray = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("OpenCV failed to decode rendered PNG")
    blurred = cv2.GaussianBlur(gray, (3, 3), sigmaX=0.35)
    values = (blurred.astype(np.float32) - 127.5) * 0.92 + 127.5
    values += rng.normal(0, 2.2, values.shape)
    rgb = cv2.cvtColor(np.clip(values, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    ok, encoded = cv2.imencode(
        ".png", rgb, [cv2.IMWRITE_PNG_COMPRESSION, 1]
    )
    if not ok:
        raise ValueError("OpenCV failed to encode degraded PNG")
    return encoded.tobytes()


def quantize(value: float, maximum: int) -> int:
    return max(0, min(QUANTIZATION_MAX, round(value / max(maximum - 1, 1) * QUANTIZATION_MAX)))


def build_render(record: MoleculeRecord, variant: str, sample_index: int) -> tuple[bytes, dict]:
    layout_mol, layout_info = layout_molecule(record, variant, sample_index)
    params = rendering_parameters(record, variant)
    png, prepared, atom_points, bond_rows = prepare_and_draw(layout_mol, params)
    if not any(0 <= point.x < WIDTH and 0 <= point.y < HEIGHT for point in atom_points):
        raise ValueError(f"Render outside canvas for {record.molecule_id} {variant}")
    if params["color_mode"] == "scan_grayscale":
        png = degrade_scan(png, stable_int(SEED, "scan", record.molecule_id, variant))
    conformer = prepared.GetConformer()
    atoms = []
    for atom_index, draw_point in enumerate(atom_points):
        point = conformer.GetAtomPosition(atom_index)
        atoms.append(
            {
                "atom_id": atom_index,
                "molecule_x": round(point.x, 6),
                "molecule_y": round(point.y, 6),
                "image_x": round(draw_point.x, 2),
                "image_y": round(draw_point.y, 2),
                "quantized_x": quantize(draw_point.x, WIDTH),
                "quantized_y": quantize(draw_point.y, HEIGHT),
            }
        )
    render_params = {
        "layout": layout_info["layout"],
        **params,
        "stress_operation": layout_info["stress"]["operation"] if layout_info["stress"] else None,
    }
    if layout_info["stress"]:
        render_params["stress_details"] = layout_info["stress"]
    render = {
        "schema_version": "render-0.1",
        "molecule_id": record.molecule_id,
        "variant": variant,
        "width": WIDTH,
        "height": HEIGHT,
        "atoms": atoms,
        "bonds": bond_rows,
        "render_parameters": render_params,
    }
    return png, render


def serialize_chemmolseq(chemmol: dict, render: dict) -> str:
    return serialize_chemmolseq_v2(chemmol, render)


def validate_chemical_roundtrip(chemmol: dict) -> None:
    mol = mol_from_chemmol(chemmol)
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    inchikey = Chem.MolToInchiKey(mol)
    expected_smiles = chemmol["canonical_isomeric_smiles"]
    normalized_actual = Chem.MolToSmiles(
        Chem.MolFromSmiles(smiles), canonical=True, isomericSmiles=True
    )
    normalized_expected = Chem.MolToSmiles(
        Chem.MolFromSmiles(expected_smiles), canonical=True, isomericSmiles=True
    )
    if normalized_actual != normalized_expected or inchikey != chemmol["inchikey"]:
        raise ValueError(
            f"Chemical roundtrip mismatch for {chemmol['molecule_id']}: "
            f"actual={smiles} / {inchikey}, expected={expected_smiles} / {chemmol['inchikey']}"
        )


def validate_geometry_roundtrip(chemmol: dict, render: dict) -> float:
    mol = mol_from_chemmol(chemmol)
    for bond_data in render["bonds"]:
        mol.GetBondWithIdx(bond_data["bond_id"]).SetBondDir(
            getattr(Chem.BondDir, bond_data["direction"])
        )
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for atom in render["atoms"]:
        conformer.SetAtomPosition(atom["atom_id"], (atom["molecule_x"], atom["molecule_y"], 0.0))
    mol.RemoveAllConformers()
    mol.AddConformer(conformer, assignId=True)
    params = dict(render["render_parameters"])
    _, _, points, bonds = prepare_and_draw(mol, params, include_png=False)
    max_delta = 0.0
    for expected, actual in zip(render["atoms"], points):
        delta = math.hypot(expected["image_x"] - actual.x, expected["image_y"] - actual.y)
        max_delta = max(max_delta, delta)
    expected_directions = [bond["direction"] for bond in render["bonds"]]
    actual_directions = [bond["direction"] for bond in bonds]
    if expected_directions != actual_directions or max_delta > 0.15:
        raise ValueError(
            f"Geometry roundtrip mismatch for {chemmol['molecule_id']} "
            f"{render['variant']}: delta={max_delta}, "
            f"directions_equal={expected_directions == actual_directions}"
        )
    return max_delta


def tar_add_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(value)
    info.mtime = 0
    archive.addfile(info, io.BytesIO(value))


def write_structure(record: MoleculeRecord) -> str:
    relative = f"structures/{clean_identifier(record.source)}_{clean_identifier(record.source_id)}.sdf"
    path = OUTPUT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    mol = Chem.Mol(record.mol)
    if not valid_original_coordinates(mol):
        rdDepictor.Compute2DCoords(mol, canonOrient=True, clearConfs=True)
    mol.SetProp("molecule_id", record.molecule_id)
    mol.SetProp("canonical_isomeric_smiles", record.smiles)
    mol.SetProp("inchikey", record.inchikey)
    writer = Chem.SDWriter(str(path))
    writer.write(mol)
    writer.close()
    return relative


def build_records() -> list[MoleculeRecord]:
    records = []
    for source, path, target, reserve in SOURCE_SPECS:
        selected = select_source(path, source, target + reserve)
        for index, (mol, smiles) in enumerate(selected, 1):
            try:
                records.append(canonicalize_record(source, mol, smiles, index))
            except Exception as error:
                print(f"candidate rejected during canonicalization: {source}:{index}: {error}", flush=True)
    records.sort(key=lambda record: stable_int(SEED, "record", record.molecule_id))
    return records


def main() -> None:
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    started = time.perf_counter()
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    (OUTPUT / "shards").mkdir(parents=True)
    shutil.copytree(
        "schemas",
        OUTPUT / "schemas",
        ignore=shutil.ignore_patterns(".ipynb_checkpoints", "__pycache__", "*.pyc"),
    )

    chemmol_schema = json.loads(Path("schemas/chemmol-0.1.schema.json").read_text())
    render_schema = json.loads(Path("schemas/render-0.1.schema.json").read_text())
    chemmol_validator = fastjsonschema.compile(chemmol_schema)
    render_validator = fastjsonschema.compile(render_schema)

    candidates = build_records()
    source_targets = {source: target for source, _, target, _ in SOURCE_SPECS}
    accepted_counts = Counter()
    records: list[MoleculeRecord] = []
    assets: dict[str, tuple[dict, list[tuple[str, bytes, dict, str]]]] = {}
    rejected_rows = []
    validation = {
        "chemical_roundtrip_passed": 0,
        "geometry_roundtrip_passed": 0,
        "max_geometry_delta_pixels": 0.0,
    }

    for candidate_index, record in enumerate(candidates):
        if accepted_counts[record.source] >= source_targets[record.source]:
            continue
        try:
            chemmol = build_chemmol(record)
            chemmol_validator(chemmol)
            validate_chemical_roundtrip(chemmol)
            rendered = []
            molecule_max_delta = 0.0
            for variant in VARIANTS:
                png, render = build_render(record, variant, candidate_index)
                render_validator(render)
                delta = validate_geometry_roundtrip(chemmol, render)
                molecule_max_delta = max(molecule_max_delta, delta)
                rendered.append((variant, png, render, serialize_chemmolseq(chemmol, render)))
        except Exception as error:
            rejected_rows.append(
                {
                    "molecule_id": record.molecule_id,
                    "source": record.source,
                    "stage": "prevalidation",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        records.append(record)
        assets[record.molecule_id] = (chemmol, rendered)
        accepted_counts[record.source] += 1
        validation["chemical_roundtrip_passed"] += 1
        validation["geometry_roundtrip_passed"] += len(VARIANTS)
        validation["max_geometry_delta_pixels"] = max(
            validation["max_geometry_delta_pixels"], molecule_max_delta
        )
        if len(records) % 100 == 0:
            print(f"prevalidated {len(records)}/1000 molecules", flush=True)

    if accepted_counts != Counter(source_targets):
        raise RuntimeError(
            f"Insufficient validated candidates: accepted={dict(accepted_counts)}, "
            f"targets={source_targets}, rejected={len(rejected_rows)}"
        )

    records.sort(key=lambda record: stable_int(SEED, "accepted", record.molecule_id))
    split_counts = assign_scaffold_splits(records)
    molecule_manifest_rows = []
    sample_manifest_rows = []

    by_split: dict[str, list[MoleculeRecord]] = defaultdict(list)
    for record in records:
        by_split[record.split].append(record)

    for split in SPLIT_FRACTIONS:
        shard_path = OUTPUT / "shards" / f"{split}-000000.tar"
        with tarfile.open(shard_path, "w") as archive:
            for split_index, record in enumerate(by_split[split]):
                structure_ref = write_structure(record)
                chemmol, rendered = assets[record.molecule_id]
                chemmol["structure_ref"] = structure_ref
                molecule_manifest_rows.append(
                    {
                        "molecule_id": record.molecule_id,
                        "source": record.source,
                        "source_id": record.source_id,
                        "split": split,
                        "scaffold": record.scaffold,
                        "stratum": record.stratum,
                        "canonical_isomeric_smiles": record.smiles,
                        "inchikey": record.inchikey,
                        "structure_ref": structure_ref,
                    }
                )
                for variant, png, render, seq in rendered:
                    sample_key = f"{split}_{split_index:06d}_{variant}"
                    meta = {
                        "schema_version": "sample-meta-0.1",
                        "sample_key": sample_key,
                        "molecule_id": record.molecule_id,
                        "source": record.source,
                        "source_id": record.source_id,
                        "split": split,
                        "variant": variant,
                        "structure_ref": structure_ref,
                        "rdkit_version": rdBase.rdkitVersion,
                        "seed": SEED,
                        "chemmolseq_sha256": hashlib.sha256(seq.encode("utf-8")).hexdigest(),
                    }
                    tar_add_bytes(archive, f"{sample_key}.png", png)
                    tar_add_bytes(archive, f"{sample_key}.chemmol.json", json_bytes(chemmol))
                    tar_add_bytes(archive, f"{sample_key}.render.json", json_bytes(render))
                    tar_add_bytes(archive, f"{sample_key}.meta.json", json_bytes(meta))
                    tar_add_bytes(archive, f"{sample_key}.chemmolseq.txt", seq.encode("utf-8"))
                    sample_manifest_rows.append(
                        {
                            "sample_key": sample_key,
                            "molecule_id": record.molecule_id,
                            "split": split,
                            "variant": variant,
                            "shard": str(shard_path.relative_to(OUTPUT)),
                            "color_mode": render["render_parameters"]["color_mode"],
                            "layout": render["render_parameters"]["layout"],
                            "stress_operation": render["render_parameters"]["stress_operation"] or "",
                        }
                    )
                if (split_index + 1) % 100 == 0:
                    print(f"{split}: built {split_index + 1}/{len(by_split[split])} molecules", flush=True)

    if rejected_rows:
        with (OUTPUT / "rejected.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rejected_rows[0].keys())
            writer.writeheader()
            writer.writerows(rejected_rows)

    for filename, rows in (
        ("molecules.csv", molecule_manifest_rows),
        ("samples.csv", sample_manifest_rows),
    ):
        with (OUTPUT / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "schema_versions": {
            "chemmol": "chemmol-0.1",
            "render": "render-0.1",
            "sample_meta": "sample-meta-0.1",
        },
        "seed": SEED,
        "rdkit_version": rdBase.rdkitVersion,
        "molecules": len(records),
        "samples": len(sample_manifest_rows),
        "split_counts": split_counts,
        "source_counts": dict(Counter(record.source for record in records)),
        "variant_counts": dict(Counter(row["variant"] for row in sample_manifest_rows)),
        "color_mode_counts": dict(Counter(row["color_mode"] for row in sample_manifest_rows)),
        "stress_operation_counts": dict(
            Counter(row["stress_operation"] for row in sample_manifest_rows if row["stress_operation"])
        ),
        "rejected_candidates": len(rejected_rows),
        "validation": validation,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
