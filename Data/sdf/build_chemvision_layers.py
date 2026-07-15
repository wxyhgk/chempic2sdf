#!/usr/bin/env python3
"""Build the ChemVision Layers v0.1 10k pilot from webdataset_full_v0."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import html
import io
import json
import math
import multiprocessing as mp
import random
import re
import shutil
import tarfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cairosvg
import fastjsonschema
import jsonschema
import numpy as np
from PIL import Image
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem.Draw import rdMolDraw2D
from svgpathtools import parse_path

from build_webdataset_sample import enum_name, mol_from_chemmol, quantize, stable_int


SEED = 20260715
WIDTH = 768
HEIGHT = 768
VARIANTS = ("original", "rdkit", "coordgen", "stress")
SPLIT_TARGETS = {"train": 2250, "validation": 125, "internal_test": 125}
FONT_BUCKET_COUNTS = {"small": 2500, "normal": 4500, "large": 2500, "extreme": 500}
FONT_BUCKET_RANGES = {
    "small": (8, 12),
    "normal": (13, 19),
    "large": (20, 28),
    "extreme": (29, 36),
}
FEATURE_TARGETS = {
    "B": 500,
    "N": 500,
    "F": 250,
    "halogen": 250,
    "rare": 250,
    "implicit_C": 2500,
}
SPLIT_FEATURE_TARGETS = {
    "train": {"B": 450, "N": 450, "F": 225, "halogen": 225, "rare": 225, "implicit_C": 2250},
    "validation": {"B": 25, "N": 25, "F": 13, "halogen": 13, "rare": 13, "implicit_C": 125},
    "internal_test": {"B": 25, "N": 25, "F": 12, "halogen": 12, "rare": 12, "implicit_C": 125},
}
FONT_SPECS = (
    {
        "font_family_id": "dejavu-sans",
        "font_family": "DejaVu Sans",
        "font_style": "Book",
        "source_file": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "output_file": "fonts/DejaVuSans.ttf",
        "license": "Bitstream Vera License / DejaVu public-domain additions",
        "license_url": "https://dejavu-fonts.github.io/License.html",
        "split": "train",
    },
    {
        "font_family_id": "liberation-sans",
        "font_family": "Liberation Sans",
        "font_style": "Regular",
        "source_file": "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "output_file": "fonts/LiberationSans-Regular.ttf",
        "license": "SIL Open Font License 1.1",
        "license_url": "https://openfontlicense.org/open-font-license-official-text/",
        "split": "train",
    },
    {
        "font_family_id": "gnu-free-sans",
        "font_family": "FreeSans",
        "font_style": "Regular",
        "source_file": "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "output_file": "fonts/FreeSans.ttf",
        "license": "GPL-3+ with Special Font Exception",
        "license_url": "https://www.gnu.org/software/freefont/license.html",
        "split": "train",
    },
    {
        "font_family_id": "dejavu-serif",
        "font_family": "DejaVu Serif",
        "font_style": "Book",
        "source_file": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "output_file": "fonts/DejaVuSerif.ttf",
        "license": "Bitstream Vera License / DejaVu public-domain additions",
        "license_url": "https://dejavu-fonts.github.io/License.html",
        "split": "validation",
    },
    {
        "font_family_id": "liberation-serif",
        "font_family": "Liberation Serif",
        "font_style": "Regular",
        "source_file": "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        "output_file": "fonts/LiberationSerif-Regular.ttf",
        "license": "SIL Open Font License 1.1",
        "license_url": "https://openfontlicense.org/open-font-license-official-text/",
        "split": "internal_test",
    },
)
FONT_SPECS_V077 = (
    {
        "font_family_id": "dejavu-sans",
        "font_family": "DejaVu Sans",
        "font_style": "Book",
        "source_file": "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "output_file": "fonts/DejaVuSans.ttf",
        "license": "Bitstream Vera License / DejaVu public-domain additions",
        "license_url": "https://dejavu-fonts.github.io/License.html",
        "split": "train",
    },
    {
        "font_family_id": "dejavu-sans-bold",
        "font_family": "DejaVu Sans",
        "font_style": "Bold",
        "source_file": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "output_file": "fonts/DejaVuSans-Bold.ttf",
        "license": "Bitstream Vera License / DejaVu public-domain additions",
        "license_url": "https://dejavu-fonts.github.io/License.html",
        "split": "train",
    },
    {
        "font_family_id": "dejavu-sans-condensed",
        "font_family": "DejaVu Sans Condensed",
        "font_style": "Book",
        "source_file": "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
        "output_file": "fonts/DejaVuSansCondensed.ttf",
        "license": "Bitstream Vera License / DejaVu public-domain additions",
        "license_url": "https://dejavu-fonts.github.io/License.html",
        "split": "train",
    },
    {
        "font_family_id": "dejavu-serif",
        "font_family": "DejaVu Serif",
        "font_style": "Book",
        "source_file": "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "output_file": "fonts/DejaVuSerif.ttf",
        "license": "Bitstream Vera License / DejaVu public-domain additions",
        "license_url": "https://dejavu-fonts.github.io/License.html",
        "split": "train",
    },
    {
        "font_family_id": "dejavu-serif-bold",
        "font_family": "DejaVu Serif",
        "font_style": "Bold",
        "source_file": "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
        "output_file": "fonts/DejaVuSerif-Bold.ttf",
        "license": "Bitstream Vera License / DejaVu public-domain additions",
        "license_url": "https://dejavu-fonts.github.io/License.html",
        "split": "train",
    },
    {
        "font_family_id": "dejavu-serif-condensed",
        "font_family": "DejaVu Serif Condensed",
        "font_style": "Book",
        "source_file": "/usr/share/fonts/truetype/dejavu/DejaVuSerifCondensed.ttf",
        "output_file": "fonts/DejaVuSerifCondensed.ttf",
        "license": "Bitstream Vera License / DejaVu public-domain additions",
        "license_url": "https://dejavu-fonts.github.io/License.html",
        "split": "train",
    },
    {
        "font_family_id": "liberation-sans",
        "font_family": "Liberation Sans",
        "font_style": "Regular",
        "source_file": "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "output_file": "fonts/LiberationSans-Regular.ttf",
        "license": "SIL Open Font License 1.1",
        "license_url": "https://openfontlicense.org/open-font-license-official-text/",
        "split": "train",
    },
    {
        "font_family_id": "liberation-sans-narrow",
        "font_family": "Liberation Sans Narrow",
        "font_style": "Regular",
        "source_file": "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf",
        "output_file": "fonts/LiberationSansNarrow-Regular.ttf",
        "license": "SIL Open Font License 1.1",
        "license_url": "https://openfontlicense.org/open-font-license-official-text/",
        "split": "train",
    },
    {
        "font_family_id": "liberation-serif",
        "font_family": "Liberation Serif",
        "font_style": "Regular",
        "source_file": "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
        "output_file": "fonts/LiberationSerif-Regular.ttf",
        "license": "SIL Open Font License 1.1",
        "license_url": "https://openfontlicense.org/open-font-license-official-text/",
        "split": "train",
    },
    {
        "font_family_id": "gnu-free-sans",
        "font_family": "FreeSans",
        "font_style": "Regular",
        "source_file": "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "output_file": "fonts/FreeSans.ttf",
        "license": "GPL-3+ with Special Font Exception",
        "license_url": "https://www.gnu.org/software/freefont/license.html",
        "split": "train",
    },
    {
        "font_family_id": "gnu-free-serif",
        "font_family": "FreeSerif",
        "font_style": "Regular",
        "source_file": "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        "output_file": "fonts/FreeSerif.ttf",
        "license": "GPL-3+ with Special Font Exception",
        "license_url": "https://www.gnu.org/software/freefont/license.html",
        "split": "train",
    },
    {
        "font_family_id": "yrsa",
        "font_family": "Yrsa",
        "font_style": "Regular",
        "source_file": "/usr/share/fonts/truetype/fonts-yrsa-rasa/Yrsa-Regular.ttf",
        "output_file": "fonts/Yrsa-Regular.ttf",
        "license": "SIL Open Font License 1.1",
        "license_url": "https://scripts.sil.org/OFL",
        "split": "train",
    },
    {
        "font_family_id": "rasa",
        "font_family": "Rasa",
        "font_style": "Regular",
        "source_file": "/usr/share/fonts/truetype/fonts-yrsa-rasa/Rasa-Regular.ttf",
        "output_file": "fonts/Rasa-Regular.ttf",
        "license": "SIL Open Font License 1.1",
        "license_url": "https://scripts.sil.org/OFL",
        "split": "validation",
    },
    {
        "font_family_id": "nimbus-roman",
        "font_family": "Nimbus Roman",
        "font_style": "Regular",
        "source_file": "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf",
        "output_file": "fonts/NimbusRoman-Regular.otf",
        "license": "AGPL-3 with Font exception",
        "license_url": "https://github.com/ArtifexSoftware/urw-base35-fonts",
        "split": "validation",
    },
    {
        "font_family_id": "urw-bookman",
        "font_family": "URW Bookman",
        "font_style": "Light",
        "source_file": "/usr/share/fonts/opentype/urw-base35/URWBookman-Light.otf",
        "output_file": "fonts/URWBookman-Light.otf",
        "license": "AGPL-3 with Font exception",
        "license_url": "https://github.com/ArtifexSoftware/urw-base35-fonts",
        "split": "internal_test",
    },
    {
        "font_family_id": "ubuntu",
        "font_family": "Ubuntu",
        "font_style": "Regular",
        "source_file": "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
        "output_file": "fonts/Ubuntu-R.ttf",
        "license": "Ubuntu Font Licence 1.0",
        "license_url": "https://ubuntu.com/legal/font-licence",
        "split": "internal_test",
    },
)
FONT_PROFILES = {
    "v01-original": FONT_SPECS,
    "v077-font-expanded": FONT_SPECS_V077,
}
EXPECTED_MEMBERS = (
    ".png",
    ".text.png",
    ".bond.png",
    ".semantic.png",
    ".layers.json",
    ".chemmol.json",
    ".render.json",
    ".meta.json",
)
SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


@dataclass(slots=True)
class MoleculeCandidate:
    row: dict[str, str]
    features: Counter


@dataclass(slots=True)
class RenderTask:
    sample_key: str
    molecule_id: str
    split: str
    variant: str
    source_sample_key: str
    source_shard: str
    output_shard: str
    font_family_id: str
    font_source_file: str
    font_file_sha256: str
    font_size: int
    font_bucket: str
    bond_line_width: float
    rotation_degrees: float
    repeat_check: bool


SOURCE_ASSETS: dict[str, tuple[bytes, dict, dict]] = {}
OUTPUT_ROOT = Path()
LAYER_VALIDATOR = None
RENDER_VALIDATOR = None


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def explicit_atom_text(atom: dict[str, Any]) -> str:
    element = str(atom["element"])
    isotope = int(atom.get("isotope", 0))
    charge = int(atom.get("formal_charge", 0))
    radicals = int(atom.get("radical_electrons", 0))
    explicit_h = int(atom.get("explicit_h_count", 0))
    implicit_h = int(atom.get("implicit_h_count", 0))
    hydrogens = explicit_h + implicit_h
    h_suffix = "" if hydrogens == 0 else "H" if hydrogens == 1 else f"H{hydrogens}"
    if charge:
        sign = "+" if charge > 0 else "-"
        charge_suffix = sign if abs(charge) == 1 else f"{abs(charge)}{sign}"
    else:
        charge_suffix = ""
    return (
        (str(isotope) if isotope else "")
        + element
        + h_suffix
        + charge_suffix
        + "." * radicals
    )


def atom_display_text(atom: dict[str, Any]) -> str:
    if (
        str(atom["element"]) == "C"
        and int(atom.get("isotope", 0)) == 0
        and int(atom.get("formal_charge", 0)) == 0
        and int(atom.get("radical_electrons", 0)) == 0
    ):
        return ""
    return explicit_atom_text(atom)


def feature_counts(mol: Chem.Mol) -> Counter:
    counts = Counter(atom.GetSymbol() for atom in mol.GetAtoms())
    implicit_c = sum(
        1
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C"
        and atom.GetIsotope() == 0
        and atom.GetFormalCharge() == 0
        and atom.GetNumRadicalElectrons() == 0
        and atom.GetNumExplicitHs() == 0
        and not atom.GetNoImplicit()
    )
    return Counter(
        {
            "B": counts["B"],
            "N": counts["N"],
            "F": counts["F"],
            "halogen": counts["Cl"] + counts["Br"] + counts["I"],
            "rare": counts["P"] + counts["S"] + counts["Si"] + counts["Se"],
            "implicit_C": implicit_c,
        }
    )


def select_molecules(source_root: Path) -> list[MoleculeCandidate]:
    by_split: dict[str, list[MoleculeCandidate]] = defaultdict(list)
    with (source_root / "molecules.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            mol = Chem.MolFromSmiles(row["canonical_isomeric_smiles"])
            if mol is None:
                continue
            by_split[row["split"]].append(MoleculeCandidate(row=row, features=feature_counts(mol)))

    selected: list[MoleculeCandidate] = []
    weights = {"B": 100.0, "F": 25.0, "halogen": 18.0, "rare": 18.0, "N": 2.0, "implicit_C": 0.05}
    for split, target_count in SPLIT_TARGETS.items():
        candidates = by_split[split]
        quotas = Counter(SPLIT_FEATURE_TARGETS[split])
        chosen: list[MoleculeCandidate] = []
        chosen_ids: set[str] = set()
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -sum(weights[name] * candidate.features[name] for name in weights),
                stable_int(SEED, "coverage", candidate.row["molecule_id"]),
            ),
        )
        totals = Counter()
        for candidate in ranked:
            if all(totals[name] >= amount for name, amount in quotas.items()):
                break
            useful = any(
                totals[name] < amount and candidate.features[name] > 0
                for name, amount in quotas.items()
            )
            if not useful:
                continue
            chosen.append(candidate)
            chosen_ids.add(candidate.row["molecule_id"])
            totals.update(candidate.features)
        missing = {name: max(0, amount - totals[name]) for name, amount in quotas.items()}
        if any(missing.values()):
            raise RuntimeError(f"Cannot meet selection quotas for {split}: {missing}")
        fillers = sorted(
            (candidate for candidate in candidates if candidate.row["molecule_id"] not in chosen_ids),
            key=lambda candidate: stable_int(SEED, "fill", candidate.row["molecule_id"]),
        )
        chosen.extend(fillers[: target_count - len(chosen)])
        if len(chosen) != target_count:
            raise RuntimeError(f"Selected {len(chosen)} molecules for {split}, expected {target_count}")
        selected.extend(chosen)

    totals = Counter()
    for candidate in selected:
        totals.update(candidate.features)
    missing = {name: max(0, amount - totals[name]) for name, amount in FEATURE_TARGETS.items()}
    if any(missing.values()):
        raise RuntimeError(f"Global selection quotas not met: {missing}")
    return selected


def load_sample_rows(source_root: Path, molecule_ids: set[str]) -> dict[str, dict[str, str]]:
    rows = {}
    variants_by_molecule: dict[str, set[str]] = defaultdict(set)
    with (source_root / "samples.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["molecule_id"] not in molecule_ids:
                continue
            rows[row["sample_key"]] = row
            variants_by_molecule[row["molecule_id"]].add(row["variant"])
    bad = [molecule_id for molecule_id in molecule_ids if variants_by_molecule[molecule_id] != set(VARIANTS)]
    if bad:
        raise RuntimeError(f"Selected molecule lacks four variants: {bad[0]}")
    return rows


def load_source_assets(source_root: Path, sample_rows: dict[str, dict[str, str]]) -> dict[str, tuple[bytes, dict, dict]]:
    names_by_shard: dict[str, set[str]] = defaultdict(set)
    for key, row in sample_rows.items():
        names_by_shard[row["shard"]].update({f"{key}.chemmol.json", f"{key}.render.json"})
    raw: dict[str, dict[str, bytes]] = defaultdict(dict)
    for shard_index, (shard, wanted) in enumerate(sorted(names_by_shard.items()), 1):
        with tarfile.open(source_root / shard, "r") as archive:
            for member in archive:
                if member.name not in wanted:
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"Cannot read {member.name} from {shard}")
                key = member.name.split(".", 1)[0]
                raw[key][member.name[len(key) + 1 :]] = extracted.read()
        if shard_index % 25 == 0 or shard_index == len(names_by_shard):
            print(f"Loaded source shards: {shard_index}/{len(names_by_shard)}", flush=True)
    assets = {}
    for key in sample_rows:
        payload = raw.get(key, {})
        if set(payload) != {"chemmol.json", "render.json"}:
            raise RuntimeError(f"Missing source labels for {key}: {sorted(payload)}")
        chemmol_bytes = payload["chemmol.json"]
        assets[key] = (
            chemmol_bytes,
            json.loads(chemmol_bytes),
            json.loads(payload["render.json"]),
        )
    return assets


def assign_output_shards(tasks: list[RenderTask]) -> None:
    by_split: dict[str, list[RenderTask]] = defaultdict(list)
    for task in tasks:
        by_split[task.split].append(task)
    for split, split_tasks in by_split.items():
        split_tasks.sort(key=lambda task: task.sample_key)
        shard_size = 3000 if split == "train" else 500
        for index, task in enumerate(split_tasks):
            task.output_shard = f"shards/{split}-{index // shard_size:06d}.tar"


def build_tasks(selected: list[MoleculeCandidate], sample_rows: dict[str, dict[str, str]]) -> list[RenderTask]:
    font_by_id = {item["font_family_id"]: item for item in FONT_SPECS}
    allowed_fonts: dict[str, list[str]] = defaultdict(list)
    for item in FONT_SPECS:
        allowed_fonts[item["split"]].append(item["font_family_id"])
    source_by_molecule_variant = {
        (row["molecule_id"], row["variant"]): (key, row)
        for key, row in sample_rows.items()
    }
    ordered_pairs = []
    for candidate in selected:
        for variant in VARIANTS:
            ordered_pairs.append((candidate.row["molecule_id"], candidate.row["split"], variant))
    ordered_pairs.sort(key=lambda item: stable_int(SEED, "sample-order", *item))

    bucket_sequence = []
    for bucket, count in FONT_BUCKET_COUNTS.items():
        bucket_sequence.extend([bucket] * count)
    if len(bucket_sequence) != len(ordered_pairs):
        raise RuntimeError("Font bucket totals do not match sample count")
    repeat_keys = {
        pair
        for pair in sorted(ordered_pairs, key=lambda item: stable_int(SEED, "repeat", *item))[:1000]
    }

    tasks = []
    for number, ((molecule_id, split, variant), bucket) in enumerate(zip(ordered_pairs, bucket_sequence), 1):
        source_key, source_row = source_by_molecule_variant[(molecule_id, variant)]
        font_ids = allowed_fonts[split]
        font_id = font_ids[stable_int(SEED, "font", molecule_id, variant) % len(font_ids)]
        font = font_by_id[font_id]
        low, high = FONT_BUCKET_RANGES[bucket]
        font_size = low + stable_int(SEED, "font-size", molecule_id, variant) % (high - low + 1)
        width_rng = random.Random(stable_int(SEED, "line-width", molecule_id, variant))
        rotation_rng = random.Random(stable_int(SEED, "rotation", molecule_id, variant))
        tasks.append(
            RenderTask(
                sample_key=f"{number:09d}",
                molecule_id=molecule_id,
                split=split,
                variant=variant,
                source_sample_key=source_key,
                source_shard=source_row["shard"],
                output_shard="",
                font_family_id=font_id,
                font_source_file=font["source_file"],
                font_file_sha256=file_sha256(font["source_file"]),
                font_size=font_size,
                font_bucket=bucket,
                bond_line_width=round(width_rng.uniform(1.4, 2.6), 6),
                rotation_degrees=round(rotation_rng.uniform(-12.0, 12.0), 6),
                repeat_check=(molecule_id, split, variant) in repeat_keys,
            )
        )
    assign_output_shards(tasks)
    return tasks


def restore_source_geometry(chemmol: dict, source_render: dict) -> Chem.Mol:
    mol = mol_from_chemmol(chemmol)
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for atom in source_render["atoms"]:
        conformer.SetAtomPosition(
            atom["atom_id"],
            (atom["molecule_x"], atom["molecule_y"], 0.0),
        )
    mol.RemoveAllConformers()
    mol.AddConformer(conformer, assignId=True)
    for bond in source_render["bonds"]:
        mol.GetBondWithIdx(bond["bond_id"]).SetBondDir(getattr(Chem.BondDir, bond["direction"]))
    return mol


def draw_semantic_svg(task: RenderTask, chemmol: dict, source_render: dict) -> tuple[bytes, Chem.Mol, list, dict]:
    mol = restore_source_geometry(chemmol, source_render)
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
    options.multipleBondOffset = float(source_params.get("multiple_bond_offset", 0.16))
    options.addStereoAnnotation = False
    options.includeMetadata = False
    options.flagCloseContactsDist = -1
    options.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
    options.useBWAtomPalette()
    drawer.DrawMolecule(prepared)
    atom_points = [drawer.GetDrawCoords(index) for index in range(prepared.GetNumAtoms())]
    drawer.FinishDrawing()
    params = {
        "layout": source_params["layout"],
        "rotation_degrees": task.rotation_degrees,
        "bond_line_width": task.bond_line_width,
        "font_size": task.font_size,
        "padding": 0.08,
        "multiple_bond_offset": options.multipleBondOffset,
        "color_mode": "black_white",
        "kekulize": kekulize,
        "stress_operation": source_params.get("stress_operation"),
        "font_family_id": task.font_family_id,
        "font_file_sha256": task.font_file_sha256,
        "renderer_id": "rdkit-svg-semantic-0.1",
    }
    if "stress_details" in source_params:
        params["stress_details"] = source_params["stress_details"]
    return drawer.GetDrawingText().encode("utf-8"), prepared, atom_points, params


def local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def element_classes(element: ET.Element) -> set[str]:
    return set(element.attrib.get("class", "").split())


def class_index(classes: set[str], prefix: str) -> int | None:
    for value in classes:
        if value.startswith(prefix):
            return int(value[len(prefix) :])
    return None


def path_bbox(element: ET.Element, expansion: float = 0.0) -> tuple[float, float, float, float] | None:
    if local_name(element) != "path" or not element.attrib.get("d"):
        return None
    try:
        xmin, xmax, ymin, ymax = parse_path(element.attrib["d"]).bbox()
    except Exception:
        return None
    return (
        max(0.0, xmin - expansion),
        max(0.0, ymin - expansion),
        min(float(WIDTH), xmax + expansion),
        min(float(HEIGHT), ymax + expansion),
    )


def merge_bbox(current, value):
    if value is None:
        return current
    if current is None:
        return list(value)
    return [
        min(current[0], value[0]),
        min(current[1], value[1]),
        max(current[2], value[2]),
        max(current[3], value[3]),
    ]


def split_svg(svg: bytes, line_width: float) -> tuple[bytes, bytes, dict[int, list[float]], dict[int, list[float]], int]:
    root = ET.fromstring(svg)
    text_bbox: dict[int, list[float]] = {}
    bond_bbox: dict[int, list[float]] = {}
    classless_foreground = 0
    drawable = {"path", "line", "polyline", "polygon", "circle", "ellipse"}
    for element in list(root):
        name = local_name(element)
        classes = element_classes(element)
        bond_id = class_index(classes, "bond-")
        atom_id = class_index(classes, "atom-")
        if bond_id is not None:
            bond_bbox[bond_id] = merge_bbox(
                bond_bbox.get(bond_id),
                path_bbox(element, line_width / 2.0 + 1.5),
            )
        elif atom_id is not None:
            text_bbox[atom_id] = merge_bbox(
                text_bbox.get(atom_id),
                path_bbox(element, 1.5),
            )
        elif name in drawable:
            classless_foreground += 1

    def filtered(layer: str) -> bytes:
        clone = copy.deepcopy(root)
        for element in list(clone):
            name = local_name(element)
            classes = element_classes(element)
            has_bond = class_index(classes, "bond-") is not None
            has_atom = class_index(classes, "atom-") is not None
            if name == "rect":
                clone.remove(element)
            elif layer == "text" and not (has_atom and not has_bond):
                if name not in {"defs", "style"}:
                    clone.remove(element)
            elif layer == "bond" and not (has_bond or (name in drawable and not has_atom)):
                if name not in {"defs", "style"}:
                    clone.remove(element)
        return ET.tostring(clone, encoding="utf-8", xml_declaration=True)

    return filtered("text"), filtered("bond"), text_bbox, bond_bbox, classless_foreground


def pil_png(image: Image.Image, mode: str) -> bytes:
    output = io.BytesIO()
    image.convert(mode).save(output, format="PNG", compress_level=1, optimize=False)
    return output.getvalue()


def rasterize_combined(svg: bytes) -> tuple[bytes, np.ndarray]:
    png = cairosvg.svg2png(bytestring=svg, output_width=WIDTH, output_height=HEIGHT)
    with Image.open(io.BytesIO(png)) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb).copy()
        return pil_png(rgb, "RGB"), array


def rasterize_alpha(svg: bytes) -> tuple[bytes, np.ndarray]:
    png = cairosvg.svg2png(bytestring=svg, output_width=WIDTH, output_height=HEIGHT)
    with Image.open(io.BytesIO(png)) as image:
        rgba = image.convert("RGBA")
        alpha = np.asarray(rgba.getchannel("A")).copy()
        return pil_png(Image.fromarray(alpha, mode="L"), "L"), alpha


def render_and_layers(task: RenderTask, chemmol: dict, source_render: dict) -> tuple[dict[str, bytes], dict]:
    svg, prepared, points, render_params = draw_semantic_svg(task, chemmol, source_render)
    text_svg, bond_svg, text_bboxes, bond_bboxes, classless_count = split_svg(
        svg, task.bond_line_width
    )
    combined_png, combined_rgb = rasterize_combined(svg)
    text_png, text_alpha = rasterize_alpha(text_svg)
    bond_png, bond_alpha = rasterize_alpha(bond_svg)
    text_pixels = text_alpha > 0
    bond_pixels = bond_alpha > 0
    semantic = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    semantic[bond_pixels] = 1
    semantic[text_pixels] = 2
    semantic[text_pixels & bond_pixels] = 3
    semantic_png = pil_png(Image.fromarray(semantic, mode="L"), "L")
    combined_foreground = combined_rgb.min(axis=2) < 255
    classified = text_pixels | bond_pixels
    intersection = int((combined_foreground & classified).sum())
    union = int((combined_foreground | classified).sum())
    foreground_iou = intersection / max(union, 1)
    unclassified = int((combined_foreground & ~classified).sum())
    unclassified_fraction = unclassified / max(int(combined_foreground.sum()), 1)
    if foreground_iou < 0.999 or unclassified_fraction > 0.001:
        raise ValueError(
            f"Layer alignment failed: iou={foreground_iou:.8f}, "
            f"unclassified_fraction={unclassified_fraction:.8f}"
        )

    atoms = []
    for atom_data, point in zip(chemmol["atoms"], points):
        atom_id = atom_data["id"]
        text = atom_display_text(atom_data)
        bbox = text_bboxes.get(atom_id)
        if bbox is None:
            text = ""
        elif not text:
            text = explicit_atom_text(atom_data)
        if bbox is not None:
            x0, y0, x1, y1 = (int(math.floor(bbox[0])), int(math.floor(bbox[1])), int(math.ceil(bbox[2])), int(math.ceil(bbox[3])))
            if not text_pixels[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)].any():
                raise ValueError(f"Visible atom {atom_id} bbox has no text pixels")
            bbox = [round(value, 2) for value in bbox]
        atoms.append(
            {
                "atom_id": atom_id,
                "element": atom_data["element"],
                "formal_charge": atom_data["formal_charge"],
                "aromatic": atom_data["aromatic"],
                "implicit_h_count": atom_data["implicit_h_count"],
                "center": [round(point.x, 2), round(point.y, 2)],
                "visible_text": bool(text),
                "text": text,
                "bbox": bbox,
            }
        )

    chemical_bonds = {bond["id"]: bond for bond in chemmol["bonds"]}
    bonds = []
    render_bonds = []
    for bond in prepared.GetBonds():
        bond_id = bond.GetIdx()
        chemical = chemical_bonds[bond_id]
        if {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} != {
            chemical["begin_atom"],
            chemical["end_atom"],
        }:
            raise ValueError(f"Prepared bond endpoints changed for bond {bond_id}")
        bbox = bond_bboxes.get(bond_id)
        if bbox is None:
            raise ValueError(f"Bond {bond_id} has no semantic SVG path")
        begin = points[chemical["begin_atom"]]
        end = points[chemical["end_atom"]]
        display_type = enum_name(bond.GetBondType())
        direction = enum_name(bond.GetBondDir())
        bonds.append(
            {
                "bond_id": bond_id,
                "begin_atom": chemical["begin_atom"],
                "end_atom": chemical["end_atom"],
                "begin": [round(begin.x, 2), round(begin.y, 2)],
                "end": [round(end.x, 2), round(end.y, 2)],
                "chemical_type": chemical["type"],
                "display_type": display_type,
                "direction": direction,
                "bbox": [round(value, 2) for value in bbox],
            }
        )
        render_bonds.append(
            {
                "bond_id": bond_id,
                "begin_image_x": round(begin.x, 2),
                "begin_image_y": round(begin.y, 2),
                "end_image_x": round(end.x, 2),
                "end_image_y": round(end.y, 2),
                "display_type": display_type,
                "direction": direction,
            }
        )

    conformer = prepared.GetConformer()
    render_atoms = []
    for atom_id, point in enumerate(points):
        molecule_point = conformer.GetAtomPosition(atom_id)
        render_atoms.append(
            {
                "atom_id": atom_id,
                "molecule_x": round(molecule_point.x, 6),
                "molecule_y": round(molecule_point.y, 6),
                "image_x": round(point.x, 2),
                "image_y": round(point.y, 2),
                "quantized_x": quantize(point.x, WIDTH),
                "quantized_y": quantize(point.y, HEIGHT),
            }
        )
    render = {
        "schema_version": "render-0.1",
        "molecule_id": task.molecule_id,
        "variant": task.variant,
        "width": WIDTH,
        "height": HEIGHT,
        "atoms": render_atoms,
        "bonds": render_bonds,
        "render_parameters": render_params,
    }
    layers = {
        "schema_version": "chemvision-layers-0.1",
        "sample_key": task.sample_key,
        "source": {
            "dataset": "webdataset_full_v0",
            "source_sample_key": task.source_sample_key,
            "source_shard": task.source_shard,
        },
        "molecule_id": task.molecule_id,
        "split": task.split,
        "variant": task.variant,
        "width": WIDTH,
        "height": HEIGHT,
        "mask_encoding": {"background": 0, "bond": 1, "text": 2, "overlap": 3},
        "atoms": atoms,
        "bonds": bonds,
        "render_parameters": {
            "renderer_id": "rdkit-svg-semantic-0.1",
            "rdkit_version": rdBase.rdkitVersion,
            "font_family_id": task.font_family_id,
            "font_file_sha256": task.font_file_sha256,
            "font_size": task.font_size,
            "bond_line_width": task.bond_line_width,
            "rotation_degrees": task.rotation_degrees,
            "layout": render_params["layout"],
            "stress_operation": render_params.get("stress_operation"),
            "color_mode": "black_white",
        },
    }
    if LAYER_VALIDATOR is not None:
        LAYER_VALIDATOR(layers)
    if RENDER_VALIDATOR is not None:
        RENDER_VALIDATOR(render)
    metrics = {
        "foreground_iou": foreground_iou,
        "unclassified_fraction": unclassified_fraction,
        "classless_svg_paths": classless_count,
    }
    return {
        ".png": combined_png,
        ".text.png": text_png,
        ".bond.png": bond_png,
        ".semantic.png": semantic_png,
        ".layers.json": json_bytes(layers),
        ".render.json": json_bytes(render),
    }, metrics


def hard_negative_counts(chemmol: dict, layers: dict, source_render: dict) -> Counter:
    counts = Counter()
    adjacency = defaultdict(list)
    bond_types = {}
    for bond in chemmol["bonds"]:
        adjacency[bond["begin_atom"]].append(bond["end_atom"])
        adjacency[bond["end_atom"]].append(bond["begin_atom"])
        bond_types[frozenset((bond["begin_atom"], bond["end_atom"]))] = bond["type"]
    visible_points = [atom["center"] for atom in layers["atoms"] if atom["visible_text"]]
    stress_details = source_render.get("render_parameters", {}).get("stress_details") or {}
    long_endpoints = set()
    if stress_details.get("operation") == "long_bond" and "bond_id" in stress_details:
        bond = chemmol["bonds"][stress_details["bond_id"]]
        long_endpoints = {bond["begin_atom"], bond["end_atom"]}
    for atom in layers["atoms"]:
        if atom["element"] != "C" or atom["visible_text"]:
            continue
        atom_id = atom["atom_id"]
        degree = len(adjacency[atom_id])
        counts["total"] += 1
        if degree == 2:
            counts["two_bond_endpoint"] += 1
        if degree == 3:
            counts["three_way_vertex"] += 1
        if degree >= 4:
            counts["four_way_vertex"] += 1
        if atom["aromatic"]:
            counts["aromatic_vertex"] += 1
        if any(bond_types[frozenset((atom_id, other))] == "DOUBLE" for other in adjacency[atom_id]):
            counts["near_double_bond"] += 1
        if atom_id in long_endpoints:
            counts["long_bond_endpoint"] += 1
        if visible_points and min(
            math.hypot(atom["center"][0] - point[0], atom["center"][1] - point[1])
            for point in visible_points
        ) <= 48.0:
            counts["near_text"] += 1
    return counts


def assemble_sample(task: RenderTask) -> tuple[dict[str, bytes], dict, Counter]:
    chemmol_bytes, chemmol, source_render = SOURCE_ASSETS[task.source_sample_key]
    generated, metrics = render_and_layers(task, chemmol, source_render)
    render = json.loads(generated[".render.json"])
    layers = json.loads(generated[".layers.json"])
    generated[".chemmol.json"] = chemmol_bytes
    hashes = {name.removeprefix("."): sha256_bytes(value) for name, value in generated.items()}
    meta = {
        "schema_version": "chemvision-meta-0.1",
        "sample_key": task.sample_key,
        "source_sample_key": task.source_sample_key,
        "source_shard": task.source_shard,
        "output_shard": task.output_shard,
        "molecule_id": task.molecule_id,
        "split": task.split,
        "variant": task.variant,
        "rdkit_version": rdBase.rdkitVersion,
        "seed": SEED,
        "renderer_id": "rdkit-svg-semantic-0.1",
        "sha256": hashes,
    }
    generated[".meta.json"] = json_bytes(meta)
    if task.repeat_check:
        repeated, _ = render_and_layers(task, chemmol, source_render)
        repeated[".chemmol.json"] = chemmol_bytes
        repeated_hashes = {name.removeprefix("."): sha256_bytes(value) for name, value in repeated.items()}
        repeated_meta = dict(meta)
        repeated_meta["sha256"] = repeated_hashes
        repeated[".meta.json"] = json_bytes(repeated_meta)
        if {name: sha256_bytes(value) for name, value in generated.items()} != {
            name: sha256_bytes(value) for name, value in repeated.items()
        }:
            raise ValueError("Deterministic repeat generation hash mismatch")
    row = {
        "sample_key": task.sample_key,
        "molecule_id": task.molecule_id,
        "split": task.split,
        "variant": task.variant,
        "source_sample_key": task.source_sample_key,
        "source_shard": task.source_shard,
        "output_shard": task.output_shard,
        "font_family_id": task.font_family_id,
        "font_size": task.font_size,
        "font_bucket": task.font_bucket,
        "bond_line_width": task.bond_line_width,
        "rotation_degrees": task.rotation_degrees,
        "stress_operation": render["render_parameters"].get("stress_operation") or "",
        "combined_sha256": sha256_bytes(generated[".png"]),
        "text_sha256": sha256_bytes(generated[".text.png"]),
        "bond_sha256": sha256_bytes(generated[".bond.png"]),
        "semantic_sha256": sha256_bytes(generated[".semantic.png"]),
        "layers_sha256": sha256_bytes(generated[".layers.json"]),
        "foreground_iou": metrics["foreground_iou"],
        "unclassified_fraction": metrics["unclassified_fraction"],
        "classless_svg_paths": metrics["classless_svg_paths"],
        "visible_atom_count": sum(atom["visible_text"] for atom in layers["atoms"]),
        "visible_elements": Counter(atom["element"] for atom in layers["atoms"] if atom["visible_text"]),
        "hard_negatives": hard_negative_counts(chemmol, layers, source_render),
    }
    return generated, row, row["hard_negatives"]


def tar_add_bytes(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(value)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(value))


def render_chunk(shard_name: str, chunk_index: int, tasks: list[RenderTask]) -> dict:
    chunk_name = f"{Path(shard_name).stem}-{chunk_index:03d}.tar"
    chunk_relative = f"work/chunks/{chunk_name}"
    output_path = OUTPUT_ROOT / chunk_relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    rejected = []
    started = time.perf_counter()
    with tarfile.open(output_path, "w") as archive:
        for index, task in enumerate(sorted(tasks, key=lambda item: item.sample_key), 1):
            try:
                members, row, _ = assemble_sample(task)
                for suffix in EXPECTED_MEMBERS:
                    tar_add_bytes(archive, f"{task.sample_key}{suffix}", members[suffix])
                rows.append(row)
            except Exception as error:
                rejected.append(
                    {
                        "source_sample_key": task.source_sample_key,
                        "molecule_id": task.molecule_id,
                        "split": task.split,
                        "variant": task.variant,
                        "stage": "svg_render",
                        "error_code": type(error).__name__,
                        "error_message": str(error),
                    }
                )
            if index % 100 == 0:
                print(
                    f"{shard_name} chunk {chunk_index}: {index}/{len(tasks)}",
                    flush=True,
                )
    return {
        "shard": shard_name,
        "chunk": chunk_index,
        "chunk_relative": chunk_relative,
        "rows": rows,
        "rejected": rejected,
        "elapsed_seconds": time.perf_counter() - started,
        "bytes": output_path.stat().st_size,
    }


def merge_chunks(output: Path, results: list[dict]) -> list[dict]:
    by_shard: dict[str, list[dict]] = defaultdict(list)
    for result in results:
        by_shard[result["shard"]].append(result)
    shard_results = []
    for shard, chunks in sorted(by_shard.items()):
        final_path = output / shard
        final_path.parent.mkdir(parents=True, exist_ok=True)
        member_count = 0
        with tarfile.open(final_path, "w") as destination:
            for chunk in sorted(chunks, key=lambda item: item["chunk"]):
                with tarfile.open(output / chunk["chunk_relative"], "r") as source:
                    for member in source:
                        extracted = source.extractfile(member)
                        if extracted is None:
                            raise RuntimeError(
                                f"Cannot merge {member.name} from {chunk['chunk_relative']}"
                            )
                        tar_add_bytes(destination, member.name, extracted.read())
                        member_count += 1
        shard_results.append(
            {
                "shard": shard,
                "bytes": final_path.stat().st_size,
                "members": member_count,
            }
        )
    shutil.rmtree(output / "work")
    return shard_results


def init_worker(output_root: str, assets, layer_schema, render_schema) -> None:
    global OUTPUT_ROOT, SOURCE_ASSETS, LAYER_VALIDATOR, RENDER_VALIDATOR
    OUTPUT_ROOT = Path(output_root)
    SOURCE_ASSETS = assets
    LAYER_VALIDATOR = jsonschema.Draft202012Validator(layer_schema).validate
    RENDER_VALIDATOR = fastjsonschema.compile(render_schema)
    RDLogger.DisableLog("rdApp.*")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_fonts(output: Path) -> list[dict]:
    (output / "fonts").mkdir(parents=True, exist_ok=True)
    rows = []
    for item in FONT_SPECS:
        source = Path(item["source_file"])
        if not source.is_file():
            raise FileNotFoundError(source)
        target = output / item["output_file"]
        shutil.copy2(source, target)
        rows.append(
            {
                "font_family_id": item["font_family_id"],
                "font_family": item["font_family"],
                "font_style": item["font_style"],
                "font_file": item["output_file"],
                "font_file_sha256": file_sha256(source),
                "license": item["license"],
                "license_url": item["license_url"],
                "split": item["split"],
            }
        )
    shutil.copy2("/usr/share/doc/fonts-dejavu-core/copyright", output / "fonts/LICENSE-DejaVu.txt")
    shutil.copy2("/usr/share/doc/fonts-liberation2/copyright", output / "fonts/LICENSE-Liberation.txt")
    shutil.copy2("/usr/share/doc/fonts-freefont-ttf/copyright", output / "fonts/LICENSE-FreeFont.txt")
    if any("fonts-yrsa-rasa" in item["source_file"] for item in FONT_SPECS):
        shutil.copy2(
            "/usr/share/doc/fonts-yrsa-rasa/copyright",
            output / "fonts/LICENSE-Yrsa-Rasa.txt",
        )
    if any("urw-base35" in item["source_file"] for item in FONT_SPECS):
        shutil.copy2(
            "/usr/share/doc/fonts-urw-base35/copyright",
            output / "fonts/LICENSE-URW-Base35.txt",
        )
    if any("/ubuntu/" in item["source_file"] for item in FONT_SPECS):
        shutil.copy2(
            "/usr/share/doc/fonts-ubuntu/copyright",
            output / "fonts/LICENSE-Ubuntu.txt",
        )
    return rows


def build_preview(output: Path, sample_rows: list[dict]) -> None:
    preview_dir = output / "previews"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True)
    chosen = sorted(sample_rows, key=lambda row: stable_int(SEED, "preview", row["sample_key"]))[:500]
    wanted_by_shard: dict[str, set[str]] = defaultdict(set)
    for row in chosen:
        for suffix in (".png", ".text.png", ".bond.png", ".semantic.png"):
            wanted_by_shard[row["output_shard"]].add(row["sample_key"] + suffix)
    images = {}
    for shard, wanted in wanted_by_shard.items():
        with tarfile.open(output / shard, "r") as archive:
            for member in archive:
                if member.name not in wanted:
                    continue
                extracted = archive.extractfile(member)
                if extracted is not None:
                    images[member.name] = extracted.read()
    html_rows = []
    for row in chosen:
        key = row["sample_key"]
        layer_files = []
        for label, suffix in (
            ("combined", ".png"),
            ("text", ".text.png"),
            ("bond", ".bond.png"),
            ("semantic", ".semantic.png"),
        ):
            payload = images[key + suffix]
            with Image.open(io.BytesIO(payload)) as image:
                if suffix == ".semantic.png":
                    index = np.asarray(image.convert("L"))
                    rgb = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
                    rgb[index == 1] = (70, 70, 70)
                    rgb[index == 2] = (220, 35, 35)
                    rgb[index == 3] = (125, 45, 180)
                    preview = Image.fromarray(rgb)
                elif suffix in {".text.png", ".bond.png"}:
                    alpha = np.asarray(image.convert("L"))
                    preview = Image.fromarray(255 - alpha, mode="L").convert("RGB")
                else:
                    preview = image.convert("RGB")
                preview.thumbnail((192, 192), Image.Resampling.LANCZOS)
                filename = f"{key}.{label}.png"
                preview.save(preview_dir / filename, format="PNG", compress_level=1)
                layer_files.append((label, filename))
        cells = "".join(
            f"<td><img src='{html.escape(filename)}' alt='{html.escape(label)}'><br>{html.escape(label)}</td>"
            for label, filename in layer_files
        )
        description = (
            f"{key}<br>{html.escape(row['split'])} / {html.escape(row['variant'])}<br>"
            f"{html.escape(row['font_family_id'])} {row['font_size']}px"
        )
        html_rows.append(f"<tr><th>{description}</th>{cells}</tr>")
    document = """<!doctype html><html><head><meta charset='utf-8'><title>ChemVision Layers pilot</title>
<style>body{font-family:sans-serif;margin:20px;color:#222}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:6px;text-align:center;vertical-align:middle}th{font-weight:400;text-align:left}img{width:192px;height:192px;object-fit:contain;background:white}</style>
</head><body><h1>ChemVision Layers v0.1 pilot preview</h1><table>""" + "".join(html_rows) + "</table></body></html>\n"
    (preview_dir / "index.html").write_text(document, encoding="utf-8")


def validate_package(
    output: Path,
    sample_rows: list[dict],
    molecule_rows: list[dict],
    font_rows: list[dict],
    rejected_rows: list[dict],
    visible_elements: Counter,
    hard_negatives: Counter,
) -> dict:
    errors = []
    if len(sample_rows) != 10000:
        errors.append(f"sample_count={len(sample_rows)}")
    if len(molecule_rows) != 2500:
        errors.append(f"molecule_count={len(molecule_rows)}")
    if rejected_rows:
        errors.append(f"rejected={len(rejected_rows)}")
    counts_by_split = Counter(row["split"] for row in molecule_rows)
    if counts_by_split != Counter(SPLIT_TARGETS):
        errors.append(f"split_counts={dict(counts_by_split)}")
    sample_split_counts = Counter(row["split"] for row in sample_rows)
    expected_sample_splits = Counter({key: value * 4 for key, value in SPLIT_TARGETS.items()})
    if sample_split_counts != expected_sample_splits:
        errors.append(f"sample_split_counts={dict(sample_split_counts)}")
    bucket_counts = Counter(row["font_bucket"] for row in sample_rows)
    if bucket_counts != Counter(FONT_BUCKET_COUNTS):
        errors.append(f"font_bucket_counts={dict(bucket_counts)}")
    variant_counts = Counter(row["variant"] for row in sample_rows)
    if variant_counts != Counter({variant: 2500 for variant in VARIANTS}):
        errors.append(f"variant_counts={dict(variant_counts)}")
    variants_by_molecule: dict[str, set[str]] = defaultdict(set)
    for row in sample_rows:
        variants_by_molecule[row["molecule_id"]].add(row["variant"])
    incomplete_molecules = [
        molecule_id
        for molecule_id, variants in variants_by_molecule.items()
        if variants != set(VARIANTS)
    ]
    if incomplete_molecules or len(variants_by_molecule) != len(molecule_rows):
        errors.append(f"incomplete_molecule_variants={incomplete_molecules[:1]}")
    visible_requirements = {
        "B": 2000,
        "N": 2000,
        "F": 1000,
        "halogen": 1000,
        "rare": 1000,
    }
    visible_actual = {
        "B": visible_elements["B"],
        "N": visible_elements["N"],
        "F": visible_elements["F"],
        "halogen": visible_elements["Cl"] + visible_elements["Br"] + visible_elements["I"],
        "rare": visible_elements["P"] + visible_elements["S"] + visible_elements["Si"] + visible_elements["Se"],
    }
    for name, minimum in visible_requirements.items():
        if visible_actual[name] < minimum:
            errors.append(f"visible_{name}={visible_actual[name]}<{minimum}")
    required_hard_negative_categories = {
        "two_bond_endpoint",
        "three_way_vertex",
        "four_way_vertex",
        "aromatic_vertex",
        "near_double_bond",
        "long_bond_endpoint",
        "near_text",
    }
    if hard_negatives["total"] < 10000:
        errors.append(f"implicit_carbon_hard_negatives={hard_negatives['total']}")
    missing_hard_negative_categories = sorted(
        name for name in required_hard_negative_categories if hard_negatives[name] == 0
    )
    if missing_hard_negative_categories:
        errors.append(f"missing_hard_negative_categories={missing_hard_negative_categories}")
    scaffold_splits: dict[str, set[str]] = defaultdict(set)
    for row in molecule_rows:
        scaffold_splits[row["scaffold_id"]].add(row["split"])
    scaffold_overlap = sum(1 for splits in scaffold_splits.values() if len(splits) > 1)
    if scaffold_overlap:
        errors.append(f"scaffold_overlap={scaffold_overlap}")
    font_sets: dict[str, set[str]] = defaultdict(set)
    for row in font_rows:
        font_sets[row["split"]].add(row["font_family_id"])
        if file_sha256(output / row["font_file"]) != row["font_file_sha256"]:
            errors.append(f"font_hash={row['font_family_id']}")
    font_overlap = 0
    split_names = list(SPLIT_TARGETS)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            font_overlap += len(font_sets[left] & font_sets[right])
    if font_overlap:
        errors.append(f"font_family_overlap={font_overlap}")
    if not (output / "previews/index.html").is_file():
        errors.append("missing_preview_index")

    row_by_key = {row["sample_key"]: row for row in sample_rows}
    if len(row_by_key) != len(sample_rows):
        errors.append("duplicate_sample_keys")
    tar_samples = 0
    tar_members = 0
    iou_min = 1.0
    unclassified_max = 0.0
    chemically_validated: set[str] = set()
    for shard in sorted((output / "shards").glob("*.tar")):
        grouped: dict[str, dict[str, bytes]] = defaultdict(dict)
        with tarfile.open(shard, "r") as archive:
            for member in archive:
                tar_members += 1
                suffix = next((value for value in sorted(EXPECTED_MEMBERS, key=len, reverse=True) if member.name.endswith(value)), None)
                if suffix is None:
                    errors.append(f"unexpected_member={member.name}")
                    continue
                key = member.name[: -len(suffix)]
                extracted = archive.extractfile(member)
                if extracted is None:
                    errors.append(f"unreadable_member={member.name}")
                    continue
                grouped[key][suffix] = extracted.read()
        for key, members in grouped.items():
            tar_samples += 1
            if set(members) != set(EXPECTED_MEMBERS):
                errors.append(f"incomplete_sample={key}")
                continue
            row = row_by_key.get(key)
            if row is None:
                errors.append(f"unmanifested_sample={key}")
                continue
            with Image.open(io.BytesIO(members[".png"])) as combined_image:
                if combined_image.size != (WIDTH, HEIGHT) or combined_image.mode != "RGB":
                    errors.append(f"combined_format={key}")
                combined = np.asarray(combined_image).copy().min(axis=2) < 255
            with Image.open(io.BytesIO(members[".text.png"])) as image:
                if image.size != (WIDTH, HEIGHT) or image.mode != "L":
                    errors.append(f"text_format={key}")
                text_mask = np.asarray(image).copy() > 0
            with Image.open(io.BytesIO(members[".bond.png"])) as image:
                if image.size != (WIDTH, HEIGHT) or image.mode != "L":
                    errors.append(f"bond_format={key}")
                bond_mask = np.asarray(image).copy() > 0
            with Image.open(io.BytesIO(members[".semantic.png"])) as image:
                semantic = np.asarray(image.convert("L")).copy()
                if image.size != (WIDTH, HEIGHT) or image.mode != "L" or not set(np.unique(semantic)).issubset({0, 1, 2, 3}):
                    errors.append(f"semantic_format={key}")
            expected_semantic = np.zeros_like(semantic)
            expected_semantic[bond_mask] = 1
            expected_semantic[text_mask] = 2
            expected_semantic[text_mask & bond_mask] = 3
            if not np.array_equal(semantic, expected_semantic):
                errors.append(f"semantic_mismatch={key}")
            classified = text_mask | bond_mask
            union = int((combined | classified).sum())
            iou = int((combined & classified).sum()) / max(union, 1)
            unclassified = int((combined & ~classified).sum()) / max(int(combined.sum()), 1)
            iou_min = min(iou_min, iou)
            unclassified_max = max(unclassified_max, unclassified)
            if iou < 0.999 or unclassified > 0.001:
                errors.append(f"layer_iou={key}:{iou:.8f}:{unclassified:.8f}")
            layers = json.loads(members[".layers.json"])
            LAYER_VALIDATOR(layers)
            render = json.loads(members[".render.json"])
            RENDER_VALIDATOR(render)
            chemmol = json.loads(members[".chemmol.json"])
            if layers["molecule_id"] != chemmol["molecule_id"] or render["molecule_id"] != chemmol["molecule_id"]:
                errors.append(f"molecule_id_mismatch={key}")
            if chemmol["molecule_id"] not in chemically_validated:
                try:
                    mol = mol_from_chemmol(chemmol)
                    Chem.SanitizeMol(mol)
                    if Chem.MolToInchiKey(mol) != chemmol["inchikey"]:
                        errors.append(f"chemical_roundtrip={chemmol['molecule_id']}")
                except Exception as error:
                    errors.append(
                        f"chemical_roundtrip={chemmol['molecule_id']}:{type(error).__name__}"
                    )
                chemically_validated.add(chemmol["molecule_id"])
            for atom in layers["atoms"]:
                if atom["visible_text"]:
                    if not atom["text"] or atom["bbox"] is None:
                        errors.append(f"visible_text_label={key}:{atom['atom_id']}")
                elif atom["text"] or atom["bbox"] is not None:
                    errors.append(f"implicit_text_label={key}:{atom['atom_id']}")
            chemical_bonds = {bond["id"]: bond for bond in chemmol["bonds"]}
            render_bonds = {bond["bond_id"]: bond for bond in render["bonds"]}
            for bond in layers["bonds"]:
                chemical = chemical_bonds[bond["bond_id"]]
                rendered = render_bonds[bond["bond_id"]]
                if bond["begin_atom"] != chemical["begin_atom"] or bond["end_atom"] != chemical["end_atom"]:
                    errors.append(f"bond_endpoint={key}:{bond['bond_id']}")
                if bond["chemical_type"] != chemical["type"] or bond["display_type"] != rendered["display_type"]:
                    errors.append(f"bond_type={key}:{bond['bond_id']}")
            expected_hashes = {
                ".png": row["combined_sha256"],
                ".text.png": row["text_sha256"],
                ".bond.png": row["bond_sha256"],
                ".semantic.png": row["semantic_sha256"],
                ".layers.json": row["layers_sha256"],
            }
            for suffix, expected in expected_hashes.items():
                if sha256_bytes(members[suffix]) != expected:
                    errors.append(f"manifest_hash={key}{suffix}")
            meta = json.loads(members[".meta.json"])
            for name, expected in meta["sha256"].items():
                suffix = "." + name
                if suffix not in members or sha256_bytes(members[suffix]) != expected:
                    errors.append(f"meta_hash={key}:{name}")

    if tar_samples != 10000 or tar_members != 80000:
        errors.append(f"tar_totals={tar_samples}:{tar_members}")
    status = "passed" if not errors else "failed"
    return {
        "schema_version": "chemvision-validation-0.1",
        "status": status,
        "samples": len(sample_rows),
        "molecules": len(molecule_rows),
        "rejected": len(rejected_rows),
        "foreground_iou_min": iou_min,
        "unclassified_foreground_fraction_max": unclassified_max,
        "scaffold_overlap": scaffold_overlap,
        "font_family_overlap": font_overlap,
        "deterministic_repeat_samples": 1000,
        "chemical_roundtrip_molecules": len(chemically_validated),
        "visible_element_instances": dict(visible_elements),
        "implicit_carbon_hard_negatives": dict(hard_negatives),
        "tar_members": tar_members,
        "errors": errors[:100],
    }


def main() -> None:
    global FONT_SPECS
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="webdataset_full_v0")
    parser.add_argument("--output", default="chemvision_layers_v0_1/pilot_10k")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument(
        "--font-profile",
        choices=tuple(FONT_PROFILES),
        default="v01-original",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    FONT_SPECS = FONT_PROFILES[args.font_profile]
    if rdBase.rdkitVersion != "2026.03.3":
        raise RuntimeError(f"Expected RDKit 2026.03.3, got {rdBase.rdkitVersion}")
    source_root = Path(args.source).resolve()
    output = Path(args.output).resolve()
    generated_paths = [
        output / "shards",
        output / "previews",
        output / "fonts",
        output / "work",
        output / "samples.csv",
        output / "molecules.csv",
        output / "fonts.csv",
        output / "rejected.csv",
        output / "summary.json",
        output / "validation.json",
    ]
    existing = [path for path in generated_paths if path.exists() and not (path.is_dir() and not any(path.iterdir()))]
    if existing and not args.overwrite:
        raise RuntimeError(f"Generated output already exists; pass --overwrite: {existing[0]}")
    if args.overwrite:
        for path in generated_paths:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    (output / "shards").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    RDLogger.DisableLog("rdApp.*")

    schema_root = Path(__file__).resolve().parent
    layer_schema = json.loads((schema_root / "chemvision_layers_v0_1/schemas/chemvision-layers-0.1.schema.json").read_text())
    render_schema = json.loads((schema_root / "schemas/render-0.1.schema.json").read_text())
    global LAYER_VALIDATOR, RENDER_VALIDATOR, SOURCE_ASSETS, OUTPUT_ROOT
    LAYER_VALIDATOR = jsonschema.Draft202012Validator(layer_schema).validate
    RENDER_VALIDATOR = fastjsonschema.compile(render_schema)
    OUTPUT_ROOT = output

    print("Selecting 2,500 molecules", flush=True)
    selected = select_molecules(source_root)
    selected_ids = {candidate.row["molecule_id"] for candidate in selected}
    sample_rows_source = load_sample_rows(source_root, selected_ids)
    tasks = build_tasks(selected, sample_rows_source)
    print(f"Loading labels for {len(tasks)} source samples", flush=True)
    SOURCE_ASSETS = load_source_assets(source_root, sample_rows_source)
    font_rows = copy_fonts(output)

    tasks_by_shard: dict[str, list[RenderTask]] = defaultdict(list)
    for task in tasks:
        tasks_by_shard[task.output_shard].append(task)
    chunk_jobs = []
    for shard, shard_tasks in sorted(tasks_by_shard.items()):
        ordered = sorted(shard_tasks, key=lambda item: item.sample_key)
        for chunk_index, start in enumerate(range(0, len(ordered), 250)):
            chunk_jobs.append((shard, chunk_index, ordered[start : start + 250]))
    results = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(chunk_jobs)),
        mp_context=context,
        initializer=init_worker,
        initargs=(str(output), SOURCE_ASSETS, layer_schema, render_schema),
    ) as executor:
        futures = {
            executor.submit(render_chunk, shard, chunk_index, chunk_tasks): (
                shard,
                chunk_index,
            )
            for shard, chunk_index, chunk_tasks in chunk_jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"Completed {result['shard']} chunk {result['chunk']}: "
                f"{len(result['rows'])} samples, "
                f"{len(result['rejected'])} rejected, {result['elapsed_seconds']:.1f}s",
                flush=True,
            )
    shard_results = merge_chunks(output, results)

    rendered_rows = sorted(
        (row for result in results for row in result["rows"]),
        key=lambda row: row["sample_key"],
    )
    rejected_rows = sorted(
        (row for result in results for row in result["rejected"]),
        key=lambda row: (row["source_sample_key"], row["variant"]),
    )
    clean_sample_rows = [
        {key: value for key, value in row.items() if key not in {"foreground_iou", "unclassified_fraction", "classless_svg_paths", "visible_atom_count", "visible_elements", "hard_negatives"}}
        for row in rendered_rows
    ]
    write_csv(
        output / "samples.csv",
        clean_sample_rows,
        [
            "sample_key", "molecule_id", "split", "variant", "source_sample_key", "source_shard",
            "output_shard", "font_family_id", "font_size", "font_bucket", "bond_line_width",
            "rotation_degrees", "stress_operation", "combined_sha256", "text_sha256", "bond_sha256",
            "semantic_sha256", "layers_sha256",
        ],
    )
    candidate_by_id = {candidate.row["molecule_id"]: candidate for candidate in selected}
    source_asset_by_molecule = {}
    for task in tasks:
        source_asset_by_molecule.setdefault(task.molecule_id, SOURCE_ASSETS[task.source_sample_key][1])
    visible_by_molecule = {}
    for row in rendered_rows:
        visible_by_molecule.setdefault(row["molecule_id"], row["visible_atom_count"])
    molecule_rows = []
    for molecule_id in sorted(selected_ids, key=lambda value: stable_int(SEED, "molecule-manifest", value)):
        candidate = candidate_by_id[molecule_id]
        chemmol = source_asset_by_molecule[molecule_id]
        molecule_rows.append(
            {
                "molecule_id": molecule_id,
                "source": candidate.row["source"],
                "source_structure": candidate.row["structure_ref"],
                "scaffold_id": candidate.row["scaffold"],
                "split": candidate.row["split"],
                "image_count": 4,
                "visible_atom_count": visible_by_molecule.get(
                    molecule_id,
                    sum(bool(atom_display_text(atom)) for atom in chemmol["atoms"]),
                ),
                "bond_count": len(chemmol["bonds"]),
            }
        )
    write_csv(
        output / "molecules.csv",
        molecule_rows,
        ["molecule_id", "source", "source_structure", "scaffold_id", "split", "image_count", "visible_atom_count", "bond_count"],
    )
    write_csv(
        output / "fonts.csv",
        font_rows,
        ["font_family_id", "font_family", "font_style", "font_file", "font_file_sha256", "license", "license_url", "split"],
    )
    write_csv(
        output / "rejected.csv",
        rejected_rows,
        ["source_sample_key", "molecule_id", "split", "variant", "stage", "error_code", "error_message"],
    )
    if len(rendered_rows) == 10000:
        build_preview(output, rendered_rows)

    visible_elements = Counter()
    hard_negatives = Counter()
    for row in rendered_rows:
        visible_elements.update(row["visible_elements"])
        hard_negatives.update(row["hard_negatives"])
    summary = {
        "schema_version": "chemvision-summary-0.1",
        "seed": SEED,
        "font_profile": args.font_profile,
        "rdkit_version": rdBase.rdkitVersion,
        "renderer_id": "rdkit-svg-semantic-0.1",
        "molecules": len(molecule_rows),
        "samples": len(rendered_rows),
        "rejected": len(rejected_rows),
        "source_counts": dict(Counter(row["source"] for row in molecule_rows)),
        "molecule_split_counts": dict(Counter(row["split"] for row in molecule_rows)),
        "sample_split_counts": dict(Counter(row["split"] for row in rendered_rows)),
        "variant_counts": dict(Counter(row["variant"] for row in rendered_rows)),
        "font_bucket_counts": dict(Counter(row["font_bucket"] for row in rendered_rows)),
        "font_size_counts": {str(key): value for key, value in sorted(Counter(row["font_size"] for row in rendered_rows).items())},
        "font_family_counts": dict(Counter(row["font_family_id"] for row in rendered_rows)),
        "layout_counts": dict(Counter(row["variant"] for row in rendered_rows)),
        "stress_operation_counts": dict(Counter(row["stress_operation"] for row in rendered_rows if row["stress_operation"])),
        "visible_element_instances": dict(visible_elements),
        "implicit_carbon_hard_negatives": dict(hard_negatives),
        "bond_line_width_min": min((row["bond_line_width"] for row in rendered_rows), default=None),
        "bond_line_width_max": max((row["bond_line_width"] for row in rendered_rows), default=None),
        "rotation_degrees_min": min((row["rotation_degrees"] for row in rendered_rows), default=None),
        "rotation_degrees_max": max((row["rotation_degrees"] for row in rendered_rows), default=None),
        "foreground_iou_min_during_build": min((row["foreground_iou"] for row in rendered_rows), default=0.0),
        "unclassified_fraction_max_during_build": max((row["unclassified_fraction"] for row in rendered_rows), default=1.0),
        "classless_svg_path_count": sum(row["classless_svg_paths"] for row in rendered_rows),
        "deterministic_repeat_samples": sum(task.repeat_check for task in tasks),
        "shards": len(shard_results),
        "bytes": sum(result["bytes"] for result in shard_results),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_bytes(json_bytes(summary))
    validation = validate_package(
        output,
        clean_sample_rows,
        molecule_rows,
        font_rows,
        rejected_rows,
        visible_elements,
        hard_negatives,
    )
    (output / "validation.json").write_bytes(json_bytes(validation))
    print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
    if validation["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
