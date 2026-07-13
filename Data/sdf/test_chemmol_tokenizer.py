#!/usr/bin/env python3
"""Unit tests for the ChemMolSeq grammar tokenizer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from .chemmol_sequence import serialize_chemmolseq
    from .chemmol_tokenizer import ChemMolTokenizer
    from .test_chemmol_sequence import make_annotations
except ImportError:
    from chemmol_sequence import serialize_chemmolseq
    from chemmol_tokenizer import ChemMolTokenizer
    from test_chemmol_sequence import make_annotations


class ChemMolTokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = ChemMolTokenizer()

    def test_vocabulary_is_compact(self) -> None:
        self.assertGreater(self.tokenizer.vocab_size, 2000)
        self.assertLess(self.tokenizer.vocab_size, 2500)

    def test_encode_decode_is_lossless(self) -> None:
        for smiles in ("c1cc[nH]c1", "F/C=C/F", "[CH3]", "N[C@@H](C)C(=O)O"):
            with self.subTest(smiles=smiles):
                chemmol, render = make_annotations(smiles)
                sequence = serialize_chemmolseq(chemmol, render)
                decoded = self.tokenizer.decode(self.tokenizer.encode(sequence))
                self.assertEqual(decoded, sequence)

    def test_batch_padding(self) -> None:
        short = serialize_chemmolseq(*make_annotations("CC"))
        long = serialize_chemmolseq(*make_annotations("c1ccccc1"))
        batch = self.tokenizer.encode_batch([short, long], pad_to_multiple_of=8)
        self.assertEqual(len(batch["input_ids"]), 2)
        self.assertEqual(len(batch["input_ids"][0]), len(batch["input_ids"][1]))
        self.assertEqual(len(batch["input_ids"][0]) % 8, 0)
        self.assertEqual(sum(batch["attention_mask"][0]), len(self.tokenizer.encode(short)))

    def test_max_length_is_a_cap_not_fixed_padding(self) -> None:
        sequence = serialize_chemmolseq(*make_annotations("CCO"))
        encoded_length = len(self.tokenizer.encode(sequence))
        batch = self.tokenizer.encode_batch([sequence], max_length=3072, pad_to_multiple_of=8)
        self.assertGreaterEqual(len(batch["input_ids"][0]), encoded_length)
        self.assertLess(len(batch["input_ids"][0]), 3072)

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            self.tokenizer.save(path)
            loaded = ChemMolTokenizer.load(path)
            self.assertEqual(loaded.vocabulary, self.tokenizer.vocabulary)


if __name__ == "__main__":
    unittest.main()
