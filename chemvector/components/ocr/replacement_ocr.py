#!/usr/bin/env python3
"""Render full molecule images with balanced OCR label replacements."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import glob
import hashlib
import io
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D
from torch.utils.data import IterableDataset

from Data.sdf.chemmol_webdataset import ChemMolWebDataset
from chemvector.ocr import OCRTextCodec, atom_display_text


PERIODIC_ELEMENT_LABELS = tuple(
    Chem.GetPeriodicTable().GetElementSymbol(atomic_number)
    for atomic_number in range(1, 119)
)

COMMON_ELEMENT_LABELS = (
    "H",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Si",
    "P",
    "S",
    "Cl",
    "Se",
    "Br",
    "I",
)

COMMON_ELEMENT_WEIGHTS = (
    2.0,  # H
    10.0,  # B
    1.0,  # C
    6.0,  # N
    5.0,  # O
    3.0,  # F
    2.0,  # Si
    1.0,  # P
    2.0,  # S
    3.0,  # Cl
    2.0,  # Se
    3.0,  # Br
    2.0,  # I
)

CHEMICAL_DISPLAY_LABELS = (
    "H",
    "H2",
    "H3",
    "NH",
    "NH2",
    "NH3",
    "N+",
    "N-",
    "NH+",
    "NH2+",
    "O-",
    "O+",
    "OH",
    "S+",
    "S-",
    "SH",
    "P+",
    "B-",
    "BH",
    "BH2",
    "2H",
    "3H",
    "13C",
    "14C",
    "15N",
    "18O",
)

PLACEHOLDER_LABELS = (
    "A",
    "E",
    "M",
    "Q",
    "X",
    "Y",
    "Z",
    "R",
    "R1",
    "R2",
    "R3",
    "R'",
    "R''",
    "R(n)",
    "*",
    "?",
    "[M]",
)

CHEMICAL_ABBREVIATIONS = (
    "Ac",
    "Ar",
    "Bn",
    "Boc",
    "Bz",
    "Cbz",
    "Cy",
    "Et",
    "Fmoc",
    "Het",
    "iPr",
    "Me",
    "Ms",
    "nBu",
    "nPr",
    "Ph",
    "Piv",
    "PMB",
    "Pr",
    "tBu",
    "TBDMS",
    "Tf",
    "THP",
    "TMS",
    "Ts",
)

LABEL_GROUPS = {
    "common_element": COMMON_ELEMENT_LABELS,
    "periodic_element": PERIODIC_ELEMENT_LABELS,
    "chemical_display": CHEMICAL_DISPLAY_LABELS,
    "placeholder": PLACEHOLDER_LABELS,
    "abbreviation": CHEMICAL_ABBREVIATIONS,
}


@dataclass(frozen=True, slots=True)
class ReplacementOCRConfig:
    replacement_image_probability: float = 0.35
    minimum_replacements: int = 1
    maximum_replacements: int = 3
    common_element_probability: float = 0.50
    periodic_element_probability: float = 0.10
    chemical_display_probability: float = 0.15
    placeholder_probability: float = 0.10
    abbreviation_probability: float = 0.15
    black_white: bool = True
    maximum_original_blank_anchors: int | None = None
    maximum_original_visible_anchors: int | None = None
    priority_original_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.minimum_replacements < 1:
            raise ValueError("minimum_replacements must be positive")
        if self.maximum_replacements < self.minimum_replacements:
            raise ValueError("maximum_replacements must not be smaller than minimum")
        if not 0.0 <= self.replacement_image_probability <= 1.0:
            raise ValueError("replacement_image_probability must be in [0, 1]")
        anchor_limits = (
            self.maximum_original_blank_anchors,
            self.maximum_original_visible_anchors,
        )
        if any(value is not None and value < 0 for value in anchor_limits):
            raise ValueError("original anchor limits must be nonnegative")
        probabilities = (
            self.common_element_probability,
            self.periodic_element_probability,
            self.chemical_display_probability,
            self.placeholder_probability,
            self.abbreviation_probability,
        )
        if any(value < 0 for value in probabilities):
            raise ValueError("replacement probabilities must be nonnegative")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-9):
            raise ValueError("replacement probabilities must sum to one")


def stable_seed(seed: int, key: str) -> int:
    digest = hashlib.blake2b(
        f"{seed}:{key}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def sample_replacement_label(
    rng: random.Random,
    config: ReplacementOCRConfig,
) -> tuple[str, str]:
    group = rng.choices(
        tuple(LABEL_GROUPS),
        weights=(
            config.common_element_probability,
            config.periodic_element_probability,
            config.chemical_display_probability,
            config.placeholder_probability,
            config.abbreviation_probability,
        ),
        k=1,
    )[0]
    if group == "common_element":
        return rng.choices(
            COMMON_ELEMENT_LABELS,
            weights=COMMON_ELEMENT_WEIGHTS,
            k=1,
        )[0], group
    return rng.choice(LABEL_GROUPS[group]), group


def choose_replacements(
    sample: dict,
    *,
    rng: random.Random,
    config: ReplacementOCRConfig,
) -> dict[int, tuple[str, str]]:
    atoms = sample["chemmol"]["atoms"]
    if rng.random() >= config.replacement_image_probability:
        return {}
    count = min(
        len(atoms),
        rng.randint(config.minimum_replacements, config.maximum_replacements),
    )
    selected = rng.sample(range(len(atoms)), count)
    replacements = {}
    for atom_id in selected:
        original = atom_display_text(atoms[atom_id])
        for _ in range(20):
            label, group = sample_replacement_label(rng, config)
            if label != original:
                break
        replacements[atom_id] = (label, group)
    return replacements


def mol_from_chemmol(chemmol: dict, render: dict) -> Chem.Mol:
    rw_mol = Chem.RWMol()
    for row in chemmol["atoms"]:
        atom = Chem.Atom(int(row["atomic_number"]))
        atom.SetFormalCharge(int(row["formal_charge"]))
        atom.SetIsotope(int(row["isotope"]))
        atom.SetIsAromatic(bool(row["aromatic"]))
        atom.SetChiralTag(getattr(Chem.ChiralType, row["chiral_tag"]))
        atom.SetNumExplicitHs(int(row["explicit_h_count"]))
        atom.SetNoImplicit(bool(row["no_implicit"]))
        atom.SetNumRadicalElectrons(int(row["radical_electrons"]))
        rw_mol.AddAtom(atom)
    for row in chemmol["bonds"]:
        rw_mol.AddBond(
            int(row["begin_atom"]),
            int(row["end_atom"]),
            getattr(Chem.BondType, row["type"]),
        )
        bond = rw_mol.GetBondBetweenAtoms(
            int(row["begin_atom"]), int(row["end_atom"])
        )
        bond.SetIsAromatic(bool(row["aromatic"]))
        bond.SetIsConjugated(bool(row["conjugated"]))
    mol = rw_mol.GetMol()
    Chem.SanitizeMol(mol)
    for row in chemmol["bonds"]:
        bond = mol.GetBondWithIdx(int(row["id"]))
        stereo_atoms = row.get("stereo_atoms", [])
        if len(stereo_atoms) == 2:
            bond.SetStereoAtoms(int(stereo_atoms[0]), int(stereo_atoms[1]))
        bond.SetStereo(getattr(Chem.BondStereo, row["stereo"]))
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.Set3D(False)
    for row in render["atoms"]:
        conformer.SetAtomPosition(
            int(row["atom_id"]),
            (float(row["molecule_x"]), float(row["molecule_y"]), 0.0),
        )
    mol.RemoveAllConformers()
    mol.AddConformer(conformer, assignId=True)
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    return mol


def render_replaced_molecule(
    sample: dict,
    replacements: dict[int, tuple[str, str]],
    *,
    black_white: bool = True,
) -> tuple[Image.Image, list[tuple[float, float]]]:
    render = sample["render"]
    params = render["render_parameters"]
    mol = mol_from_chemmol(sample["chemmol"], render)
    try:
        prepared = rdMolDraw2D.PrepareMolForDrawing(
            Chem.Mol(mol),
            kekulize=bool(params["kekulize"]),
            addChiralHs=False,
            wedgeBonds=True,
        )
    except Exception:
        prepared = rdMolDraw2D.PrepareMolForDrawing(
            Chem.Mol(mol),
            kekulize=False,
            addChiralHs=False,
            wedgeBonds=True,
        )
    width = int(render["width"])
    height = int(render["height"])
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.rotate = float(params["rotation_degrees"])
    options.bondLineWidth = float(params["bond_line_width"])
    options.fixedFontSize = int(params["font_size"])
    options.padding = float(params["padding"])
    options.multipleBondOffset = float(params["multiple_bond_offset"])
    options.addStereoAnnotation = False
    options.setBackgroundColour((1.0, 1.0, 1.0, 1.0))
    if black_white:
        options.useBWAtomPalette()
    for atom_id, (label, _) in replacements.items():
        options.atomLabels[int(atom_id)] = label
    min_x, min_y, max_x, max_y = params["scale_bounds"]
    drawer.SetScale(
        width,
        height,
        Point2D(float(min_x), float(min_y)),
        Point2D(float(max_x), float(max_y)),
    )
    drawer.DrawMolecule(prepared)
    anchors = [
        (drawer.GetDrawCoords(atom_id).x / width, drawer.GetDrawCoords(atom_id).y / height)
        for atom_id in range(prepared.GetNumAtoms())
    ]
    drawer.FinishDrawing()
    image = Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")
    image.load()
    return image, anchors


def build_replacement_sample(
    sample: dict,
    *,
    seed: int,
    config: ReplacementOCRConfig,
) -> dict:
    rng = random.Random(stable_seed(seed, sample["key"]))
    replacements = choose_replacements(sample, rng=rng, config=config)
    image, anchors = render_replaced_molecule(
        sample,
        replacements,
        black_white=config.black_white,
    )
    texts = []
    categories = []
    for atom_id, atom in enumerate(sample["chemmol"]["atoms"]):
        replacement = replacements.get(atom_id)
        if replacement is None:
            text = atom_display_text(atom)
            category = "original_visible" if text else "original_blank"
        else:
            text, category = replacement
        texts.append(text)
        categories.append(category)
    selected_atom_ids = list(range(len(anchors)))
    if (
        config.maximum_original_blank_anchors is not None
        or config.maximum_original_visible_anchors is not None
    ):
        replacement_ids = set(replacements)
        priority_original = [
            atom_id
            for atom_id, text in enumerate(texts)
            if atom_id not in replacement_ids
            and text in config.priority_original_labels
        ]
        original_blank = [
            atom_id
            for atom_id, category in enumerate(categories)
            if atom_id not in replacement_ids and category == "original_blank"
        ]
        original_visible = [
            atom_id
            for atom_id, category in enumerate(categories)
            if atom_id not in replacement_ids
            and atom_id not in priority_original
            and category == "original_visible"
        ]
        blank_limit = config.maximum_original_blank_anchors
        visible_limit = config.maximum_original_visible_anchors
        selected_atom_ids = sorted(
            replacement_ids
            | set(priority_original)
            | set(
                rng.sample(
                    original_blank,
                    min(len(original_blank), blank_limit),
                )
                if blank_limit is not None
                else original_blank
            )
            | set(
                rng.sample(
                    original_visible,
                    min(len(original_visible), visible_limit),
                )
                if visible_limit is not None
                else original_visible
            )
        )
        if not selected_atom_ids:
            selected_atom_ids = [rng.randrange(len(anchors))]
    return {
        "key": sample["key"],
        "image": image,
        "anchors": [anchors[atom_id] for atom_id in selected_atom_ids],
        "texts": [texts[atom_id] for atom_id in selected_atom_ids],
        "categories": [categories[atom_id] for atom_id in selected_atom_ids],
        "atom_ids": selected_atom_ids,
        "replacements": {
            str(atom_id): {"text": value[0], "category": value[1]}
            for atom_id, value in replacements.items()
        },
    }


class MoleculeOCRReplacementDataset(IterableDataset):
    def __init__(
        self,
        shards: str | list[str],
        *,
        seed: int = 20260715,
        config: ReplacementOCRConfig | None = None,
        sample_limit: int | None = None,
        shardshuffle: int | bool = False,
        sample_shuffle_buffer: int = 0,
    ) -> None:
        super().__init__()
        if sample_limit is not None and sample_limit < 1:
            raise ValueError("sample_limit must be positive")
        shard_patterns = [shards] if isinstance(shards, str) else shards
        expanded_shards = []
        for pattern in shard_patterns:
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"No OCR training shards matched: {pattern}")
            expanded_shards.extend(matches)
        self.shards = expanded_shards
        self.seed = seed
        self.config = config or ReplacementOCRConfig()
        self.sample_limit = sample_limit
        self.shardshuffle = shardshuffle
        self.sample_shuffle_buffer = sample_shuffle_buffer

    def __iter__(self):
        dataset = ChemMolWebDataset(
            self.shards,
            validate_legacy_hash=False,
            shardshuffle=self.shardshuffle,
            sample_shuffle_buffer=self.sample_shuffle_buffer,
            sequence_version="0.4",
        )
        for index, sample in enumerate(dataset):
            if self.sample_limit is not None and index >= self.sample_limit:
                break
            yield build_replacement_sample(
                sample,
                seed=self.seed,
                config=self.config,
            )


def collate_replacement_ocr(
    samples: list[dict],
    codec: OCRTextCodec,
) -> dict[str, torch.Tensor | list[str]]:
    maximum_atoms = max(len(sample["anchors"]) for sample in samples)
    encoded = [[codec.encode(text) for text in sample["texts"]] for sample in samples]
    maximum_length = max(len(item) for group in encoded for item in group)
    token_ids = torch.full(
        (len(samples), maximum_atoms, maximum_length),
        codec.PAD_TOKEN_ID,
        dtype=torch.long,
    )
    anchors = torch.zeros((len(samples), maximum_atoms, 2), dtype=torch.float32)
    anchor_mask = torch.zeros((len(samples), maximum_atoms), dtype=torch.bool)
    texts = []
    categories = []
    pixel_values = []
    for sample_index, sample in enumerate(samples):
        array = np.asarray(sample["image"], dtype=np.float32) / 255.0
        pixel_values.append(torch.from_numpy(array).permute(2, 0, 1))
        count = len(sample["anchors"])
        anchors[sample_index, :count] = torch.tensor(sample["anchors"])
        anchor_mask[sample_index, :count] = True
        for atom_index, item in enumerate(encoded[sample_index]):
            token_ids[sample_index, atom_index, : len(item)] = torch.tensor(item)
        texts.extend(sample["texts"])
        categories.extend(sample["categories"])
    pixels = torch.stack(pixel_values)
    return {
        "pixel_values": (pixels - 0.5) / 0.5,
        "ocr_anchor_points": anchors,
        "ocr_anchor_mask": anchor_mask,
        "ocr_token_ids": token_ids,
        "texts": texts,
        "categories": categories,
        "keys": [sample["key"] for sample in samples],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shards",
        nargs="+",
        default=["Data/sdf/webdataset_full_v0/shards/train-000000.tar"],
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/diagnostics/replacement-ocr-preview",
    )
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260715)
    args = parser.parse_args()

    dataset = MoleculeOCRReplacementDataset(
        args.shards,
        seed=args.seed,
        sample_limit=args.samples,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(dataset):
        sample["image"].save(output_dir / f"{index:04d}-{sample['key']}.png")
        (output_dir / f"{index:04d}-{sample['key']}.json").write_text(
            json.dumps(
                {
                    "key": sample["key"],
                    "texts": sample["texts"],
                    "categories": sample["categories"],
                    "replacements": sample["replacements"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
