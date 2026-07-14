"""Tests for ChemMolSeq v0.3 constrained generation grammar."""

from __future__ import annotations

import unittest

import torch

from Data.sdf.chemmol_sequence_v3 import serialize_chemmolseq_v3
from Data.sdf.chemmol_tokenizer_v3 import ChemMolTokenizerV3
from Data.sdf.test_chemmol_sequence import make_annotations
from chemvector.generation_v3 import ChemMolGrammarV3, generate_chemmolseq_v3


class GenerationV3Tests(unittest.TestCase):
    def consume_tokens(
        self,
        grammar: ChemMolGrammarV3,
        tokenizer: ChemMolTokenizerV3,
        tokens: tuple[str, ...],
    ) -> None:
        for token in tokens:
            token_id = tokenizer.token_to_id[token]
            self.assertIn(token_id, grammar.allowed_token_ids(), token)
            grammar.consume(token_id)

    def consume_atom(
        self,
        grammar: ChemMolGrammarV3,
        tokenizer: ChemMolTokenizerV3,
        element: str,
        *,
        aromatic: bool = False,
    ) -> None:
        self.consume_tokens(
            grammar,
            tokenizer,
            (
                tokenizer.ATOM,
                f"el:{element}",
                "n:0",
                "n:0",
                f"n:{int(aromatic)}",
                "n:0",
                "n:0",
                "n:0",
                "chi:CHI_UNSPECIFIED",
                tokenizer.END_RECORD,
            ),
        )

    def consume_bond(
        self,
        grammar: ChemMolGrammarV3,
        tokenizer: ChemMolTokenizerV3,
        begin_atom: int,
        end_atom: int,
        bond_type: str,
    ) -> None:
        self.consume_tokens(
            grammar,
            tokenizer,
            (
                tokenizer.BOND,
                f"n:{begin_atom}",
                f"n:{end_atom}",
                f"bt:{bond_type}",
                "st:STEREONONE",
                "n:-1",
                "n:-1",
                tokenizer.END_RECORD,
            ),
        )

    def test_ground_truth_sequences_are_accepted(self) -> None:
        tokenizer = ChemMolTokenizerV3()
        for smiles in ("CCO", "c1cc[nH]c1", "F/C=C/F", "N[C@@H](C)C(=O)O"):
            with self.subTest(smiles=smiles):
                sequence = serialize_chemmolseq_v3(*make_annotations(smiles))
                grammar = ChemMolGrammarV3(tokenizer)
                for token_id in tokenizer.encode(sequence):
                    self.assertIn(token_id, grammar.allowed_token_ids())
                    grammar.consume(token_id)
                self.assertEqual(grammar.phase, "done")

    def test_bond_types_respect_remaining_valence(self) -> None:
        tokenizer = ChemMolTokenizerV3()
        grammar = ChemMolGrammarV3(tokenizer)
        self.consume_tokens(grammar, tokenizer, (tokenizer.BOS, tokenizer.MOL))
        self.consume_atom(grammar, tokenizer, "C")
        for _ in range(4):
            self.consume_atom(grammar, tokenizer, "F")
        for end_atom in range(1, 4):
            self.consume_bond(grammar, tokenizer, 0, end_atom, "SINGLE")

        self.consume_tokens(grammar, tokenizer, (tokenizer.BOND, "n:0", "n:4"))
        allowed = set(grammar.allowed_token_ids())
        self.assertIn(tokenizer.token_to_id["bt:SINGLE"], allowed)
        self.assertNotIn(tokenizer.token_to_id["bt:DOUBLE"], allowed)
        self.assertNotIn(tokenizer.token_to_id["bt:TRIPLE"], allowed)
        self.assertNotIn(tokenizer.token_to_id["bt:AROMATIC"], allowed)

    def test_aromatic_bond_requires_aromatic_endpoints(self) -> None:
        tokenizer = ChemMolTokenizerV3()
        grammar = ChemMolGrammarV3(tokenizer)
        self.consume_tokens(grammar, tokenizer, (tokenizer.BOS, tokenizer.MOL))
        self.consume_atom(grammar, tokenizer, "C")
        self.consume_atom(grammar, tokenizer, "C")
        self.consume_tokens(grammar, tokenizer, (tokenizer.BOND, "n:0", "n:1"))
        self.assertNotIn(
            tokenizer.token_to_id["bt:AROMATIC"],
            grammar.allowed_token_ids(),
        )

    def test_coordinates_require_a_sanitizable_aromatic_graph(self) -> None:
        tokenizer = ChemMolTokenizerV3()
        sequence = serialize_chemmolseq_v3(*make_annotations("c1ccccc1"))
        token_ids = tokenizer.encode(sequence)
        bond_token_id = tokenizer.token_to_id[tokenizer.BOND]
        final_bond_index = max(
            index for index, token_id in enumerate(token_ids) if token_id == bond_token_id
        )
        grammar = ChemMolGrammarV3(tokenizer)
        for token_id in token_ids[:final_bond_index]:
            grammar.consume(token_id)
        self.assertNotIn(
            tokenizer.token_to_id[tokenizer.COORDS],
            grammar.allowed_token_ids(),
        )

        coords_index = token_ids.index(tokenizer.token_to_id[tokenizer.COORDS])
        for token_id in token_ids[final_bond_index:coords_index]:
            grammar.consume(token_id)
        self.assertIn(
            tokenizer.token_to_id[tokenizer.COORDS],
            grammar.allowed_token_ids(),
        )

    def test_beam_width_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "beam_width must be positive"):
            generate_chemmolseq_v3(
                None,
                torch.zeros(1, 3, 32, 32),
                ChemMolTokenizerV3(),
                beam_width=0,
            )


if __name__ == "__main__":
    unittest.main()
