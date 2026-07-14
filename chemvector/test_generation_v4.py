"""Tests for ChemSVG v0.4 visual-primitive generation grammar."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
from rdkit import Chem, rdBase

from Data.sdf.chemmol_sequence import ChemMolSequenceError
from Data.sdf.chemsvg_sequence_v4 import serialize_chemsvg_v4
from Data.sdf.chemsvg_sequence_v4 import StrokeRecordV4, VertexRecordV4
from Data.sdf.chemsvg_tokenizer_v4 import ChemSVGTokenizerV4
from Data.sdf.test_chemmol_sequence import make_annotations
from chemvector.generation_v4 import (
    ChemSVGGrammarV4,
    _completed_result_v4,
    _fuse_ocr_vertex_prefix,
    _vertex_prefix_ids,
    generate_detected_chemsvg_v4,
    generate_chemsvg_v4,
    write_sdf_v4,
)


class GenerationV4Tests(unittest.TestCase):
    @staticmethod
    def _vertex(x: int, element: str = "C") -> VertexRecordV4:
        return VertexRecordV4(
            x=x,
            y=100,
            element=element,
            formal_charge=0,
            isotope=0,
            aromatic=False,
            explicit_h_count=0,
            no_implicit=False,
            radical_electrons=0,
            canonical_chiral_tag="CHI_UNSPECIFIED",
        )

    def test_invalid_completed_sequence_is_reported_without_raising(self) -> None:
        class RejectingTokenizer:
            @staticmethod
            def decode(input_ids: tuple[int, ...]) -> str:
                raise ChemMolSequenceError("invalid stereo references")

        result = _completed_result_v4((1, 2, 3), RejectingTokenizer())
        self.assertTrue(result.completed)
        self.assertFalse(result.chemically_valid)
        self.assertEqual(result.error, "invalid stereo references")

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

    def test_visual_first_can_finish_before_chemical_sanitization(self) -> None:
        tokenizer = ChemSVGTokenizerV4()
        stroke = StrokeRecordV4(
            begin_x=100,
            begin_y=100,
            end_x=200,
            end_y=100,
            display_type="DOUBLE",
            bond_type="DOUBLE",
            direction="NONE",
            stereo="STEREONONE",
        )
        visual = ChemSVGGrammarV4(tokenizer, visual_first=True)
        visual.vertices = [self._vertex(100, "F"), self._vertex(200)]
        visual.strokes = [stroke]
        visual.stroke_started = True
        visual.phase = "record"
        self.assertTrue(visual._can_finish_molecule())

        chemical = ChemSVGGrammarV4(tokenizer)
        chemical.vertices = list(visual.vertices)
        chemical.strokes = [stroke]
        chemical.stroke_started = True
        chemical.phase = "record"
        with rdBase.BlockLogs():
            self.assertFalse(chemical._can_finish_molecule())

    def test_stroke_scorer_pairs_limit_endpoints_without_distance_rules(self) -> None:
        tokenizer = ChemSVGTokenizerV4()
        vertices = [self._vertex(100), self._vertex(200), self._vertex(900)]
        allowed_pair = tuple(sorted(((100, 100), (900, 100))))
        grammar = ChemSVGGrammarV4(
            tokenizer,
            visual_first=True,
            allowed_stroke_pairs={allowed_pair},
        )
        grammar.vertices = vertices
        self.assertEqual(
            set(grammar._begin_coordinates()),
            {(100, 100), (900, 100)},
        )

    def test_ocr_fusion_rewrites_vertex_prefix_tokens(self) -> None:
        tokenizer = ChemSVGTokenizerV4()
        sequence = serialize_chemsvg_v4(*make_annotations("CCO"))
        token_ids = tokenizer.encode(sequence)
        stroke_index = token_ids.index(tokenizer.token_to_id[tokenizer.STROKE])
        prefix = tuple(token_ids[:stroke_index])
        grammar = ChemSVGGrammarV4(tokenizer, visual_first=True)
        for token_id in prefix:
            grammar.consume(token_id)

        labels = tuple("N" if index == 0 else "" for index in range(len(grammar.vertices)))
        fused = _fuse_ocr_vertex_prefix(prefix, grammar, labels, tokenizer)

        self.assertEqual(grammar.vertices[0].element, "N")
        first_vertex = fused.index(tokenizer.token_to_id[tokenizer.VERTEX])
        self.assertEqual(
            fused[first_vertex + 3],
            tokenizer.token_to_id["el:N"],
        )

    def test_detected_vertex_prefix_is_accepted_by_visual_grammar(self) -> None:
        tokenizer = ChemSVGTokenizerV4()
        vertices = [self._vertex(100), self._vertex(900, "N")]
        prefix = _vertex_prefix_ids(tokenizer, vertices)
        grammar = ChemSVGGrammarV4(tokenizer, visual_first=True)
        for token_id in prefix:
            grammar.consume(token_id)
        self.assertEqual(grammar.phase, "record")
        self.assertEqual(grammar.vertices, vertices)

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

    def test_stroke_threshold_must_be_a_probability(self) -> None:
        with self.assertRaisesRegex(ValueError, "stroke_threshold"):
            generate_chemsvg_v4(
                None,
                torch.zeros(1, 3, 32, 32),
                ChemSVGTokenizerV4(),
                stroke_threshold=1.1,
            )

    def test_detected_ocr_radius_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "ocr_roi_radius"):
            generate_detected_chemsvg_v4(
                None,
                torch.zeros(1, 3, 32, 32),
                ChemSVGTokenizerV4(),
                ocr_roi_radius=0,
            )


if __name__ == "__main__":
    unittest.main()
