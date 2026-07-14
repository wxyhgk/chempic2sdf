"""Tests for ChemSVG v0.4 visual-primitive generation grammar."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from rdkit import Chem

from Data.sdf.chemsvg_sequence_v4 import serialize_chemsvg_v4
from Data.sdf.chemsvg_tokenizer_v4 import ChemSVGTokenizerV4
from Data.sdf.test_chemmol_sequence import make_annotations
from chemvector.generation_v4 import (
    ChemSVGGrammarV4,
    generate_chemsvg_v4,
    write_sdf_v4,
)


class GenerationV4Tests(unittest.TestCase):
    def test_ground_truth_sequences_are_accepted(self) -> None:
        tokenizer = ChemSVGTokenizerV4()
        for smiles in (
            "CCO",
            "c1cc[nH]c1",
            "F/C=C/F",
            "N[C@@H](C)C(=O)O",
        ):
            with self.subTest(smiles=smiles):
                sequence = serialize_chemsvg_v4(*make_annotations(smiles))
                grammar = ChemSVGGrammarV4(tokenizer)
                for token_id in tokenizer.encode(sequence):
                    self.assertIn(token_id, grammar.allowed_token_ids())
                    grammar.consume(token_id)
                self.assertEqual(grammar.phase, "done")

    def test_stroke_endpoints_are_selected_from_visual_vertices(self) -> None:
        tokenizer = ChemSVGTokenizerV4()
        sequence = serialize_chemsvg_v4(*make_annotations("CCO"))
        token_ids = tokenizer.encode(sequence)
        grammar = ChemSVGGrammarV4(tokenizer)
        stroke_id = tokenizer.token_to_id[tokenizer.STROKE]
        stroke_index = token_ids.index(stroke_id)
        for token_id in token_ids[: stroke_index + 1]:
            grammar.consume(token_id)
        allowed_x = {
            int(tokenizer.id_to_token[token_id][2:])
            for token_id in grammar.allowed_token_ids()
        }
        self.assertEqual(allowed_x, {vertex.x for vertex in grammar.vertices})

    def test_sdf_write_and_read_preserves_stereochemistry(self) -> None:
        chemmol, render = make_annotations("N[C@@H](C)C(=O)O")
        sequence = serialize_chemsvg_v4(chemmol, render)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "molecule.sdf"
            write_sdf_v4(sequence, path)
            mol = next(item for item in Chem.SDMolSupplier(str(path)) if item is not None)
            self.assertEqual(Chem.MolToInchiKey(mol), chemmol["inchikey"])

    def test_beam_width_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "beam_width must be positive"):
            generate_chemsvg_v4(
                None,
                torch.zeros(1, 3, 32, 32),
                ChemSVGTokenizerV4(),
                beam_width=0,
            )


if __name__ == "__main__":
    unittest.main()
