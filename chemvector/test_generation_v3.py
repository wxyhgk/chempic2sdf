"""Tests for ChemMolSeq v0.3 constrained generation grammar."""

from __future__ import annotations

import unittest

from Data.sdf.chemmol_sequence_v3 import serialize_chemmolseq_v3
from Data.sdf.chemmol_tokenizer_v3 import ChemMolTokenizerV3
from Data.sdf.test_chemmol_sequence import make_annotations
from chemvector.generation_v3 import ChemMolGrammarV3


class GenerationV3Tests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
