"""Tests for OCR-only label replacement on full molecule images."""

from __future__ import annotations

import random
import unittest

import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor

from chemvector.ocr import OCRTextCodec
from chemvector.replacement_ocr import (
    PERIODIC_ELEMENT_LABELS,
    ReplacementOCRConfig,
    build_replacement_sample,
    collate_replacement_ocr,
    render_replaced_molecule,
    sample_replacement_label,
)


def molecule_sample() -> dict:
    mol = Chem.MolFromSmiles("CCN")
    rdDepictor.Compute2DCoords(mol)
    conformer = mol.GetConformer()
    atoms = []
    render_atoms = []
    for atom in mol.GetAtoms():
        atoms.append(
            {
                "id": atom.GetIdx(),
                "atomic_number": atom.GetAtomicNum(),
                "element": atom.GetSymbol(),
                "formal_charge": atom.GetFormalCharge(),
                "isotope": atom.GetIsotope(),
                "aromatic": atom.GetIsAromatic(),
                "chiral_tag": str(atom.GetChiralTag()),
                "explicit_h_count": atom.GetNumExplicitHs(),
                "implicit_h_count": atom.GetNumImplicitHs(),
                "no_implicit": atom.GetNoImplicit(),
                "radical_electrons": atom.GetNumRadicalElectrons(),
            }
        )
        point = conformer.GetAtomPosition(atom.GetIdx())
        render_atoms.append(
            {
                "atom_id": atom.GetIdx(),
                "molecule_x": point.x,
                "molecule_y": point.y,
            }
        )
    bonds = []
    for bond in mol.GetBonds():
        bonds.append(
            {
                "id": bond.GetIdx(),
                "begin_atom": bond.GetBeginAtomIdx(),
                "end_atom": bond.GetEndAtomIdx(),
                "type": str(bond.GetBondType()),
                "aromatic": bond.GetIsAromatic(),
                "conjugated": bond.GetIsConjugated(),
                "stereo": str(bond.GetStereo()),
                "stereo_atoms": list(bond.GetStereoAtoms()),
            }
        )
    return {
        "key": "unit-test-molecule",
        "chemmol": {"atoms": atoms, "bonds": bonds},
        "render": {
            "width": 768,
            "height": 768,
            "atoms": render_atoms,
            "render_parameters": {
                "rotation_degrees": 0.0,
                "bond_line_width": 2.0,
                "font_size": 16,
                "padding": 0.08,
                "multiple_bond_offset": 0.15,
                "kekulize": True,
                "scale_bounds": [-3.0, -3.0, 3.0, 3.0],
            },
        },
    }


class ReplacementOCRTests(unittest.TestCase):
    def test_periodic_table_contains_all_current_elements(self) -> None:
        self.assertEqual(len(PERIODIC_ELEMENT_LABELS), 118)
        self.assertEqual(PERIODIC_ELEMENT_LABELS[0], "H")
        self.assertEqual(PERIODIC_ELEMENT_LABELS[-1], "Og")

    def test_all_label_groups_are_sampled_as_plain_strings(self) -> None:
        config = ReplacementOCRConfig()
        sampled = {
            sample_replacement_label(random.Random(seed), config)[1]
            for seed in range(500)
        }
        self.assertEqual(
            sampled,
            {
                "common_element",
                "periodic_element",
                "chemical_display",
                "placeholder",
                "abbreviation",
            },
        )
        codec = OCRTextCodec()
        self.assertEqual(codec.decode(codec.encode("Ph")), "Ph")

    def test_full_molecule_render_keeps_arbitrary_label_literal(self) -> None:
        sample = molecule_sample()
        image, anchors = render_replaced_molecule(
            sample,
            {0: ("Ph", "abbreviation"), 1: ("R1", "placeholder")},
        )
        self.assertEqual(image.size, (768, 768))
        self.assertEqual(len(anchors), 3)
        self.assertLess(image.convert("L").getextrema()[0], 255)

    def test_replacement_sample_and_collator_are_deterministic(self) -> None:
        config = ReplacementOCRConfig(
            replacement_image_probability=1.0,
            minimum_replacements=2,
            maximum_replacements=2,
        )
        first = build_replacement_sample(molecule_sample(), seed=7, config=config)
        second = build_replacement_sample(molecule_sample(), seed=7, config=config)
        self.assertEqual(first["replacements"], second["replacements"])
        self.assertEqual(first["image"].tobytes(), second["image"].tobytes())

        batch = collate_replacement_ocr([first, second], OCRTextCodec())
        self.assertEqual(batch["pixel_values"].shape, (2, 3, 768, 768))
        self.assertEqual(batch["ocr_anchor_points"].shape, (2, 3, 2))
        self.assertTrue(torch.all(batch["ocr_anchor_mask"]))
        self.assertEqual(len(batch["texts"]), 6)

    def test_anchor_sampling_keeps_every_replacement(self) -> None:
        sample = build_replacement_sample(
            molecule_sample(),
            seed=13,
            config=ReplacementOCRConfig(
                replacement_image_probability=1.0,
                minimum_replacements=2,
                maximum_replacements=2,
                maximum_original_blank_anchors=0,
                maximum_original_visible_anchors=0,
            ),
        )
        replacement_ids = {int(value) for value in sample["replacements"]}
        self.assertEqual(set(sample["atom_ids"]), replacement_ids)
        self.assertEqual(len(sample["anchors"]), 2)

    def test_anchor_sampling_keeps_priority_original_labels(self) -> None:
        sample = build_replacement_sample(
            molecule_sample(),
            seed=17,
            config=ReplacementOCRConfig(
                replacement_image_probability=0.0,
                maximum_original_blank_anchors=0,
                maximum_original_visible_anchors=0,
                priority_original_labels=("NH2",),
            ),
        )
        self.assertEqual(sample["texts"], ["NH2"])
        self.assertEqual(sample["atom_ids"], [2])


if __name__ == "__main__":
    unittest.main()
