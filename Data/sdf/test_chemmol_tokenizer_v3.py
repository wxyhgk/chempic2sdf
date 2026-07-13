#!/usr/bin/env python3
"""Unit tests for the ChemMolSeq v0.3 tokenizer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .chemmol_sequence_v3 import serialize_chemmolseq_v3
from .chemmol_tokenizer_v3 import ChemMolTokenizerV3
from .test_chemmol_sequence import make_annotations


class ChemMolTokenizerV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = ChemMolTokenizerV3()

    def test_vocabulary_is_smaller_than_v02(self) -> None:
        self.assertEqual(self.tokenizer.vocab_size, 2126)

    def test_encode_decode_is_lossless(self) -> None:
        for smiles in ("c1cc[nH]c1", "F/C=C/F", "[CH3]", "N[C@@H](C)C(=O)O"):
            with self.subTest(smiles=smiles):
                sequence = serialize_chemmolseq_v3(*make_annotations(smiles))
                self.assertEqual(
                    self.tokenizer.decode(self.tokenizer.encode(sequence)),
                    sequence,
                )

    def test_dynamic_batch_padding(self) -> None:
        short = serialize_chemmolseq_v3(*make_annotations("CC"))
        long = serialize_chemmolseq_v3(*make_annotations("c1ccccc1"))
        batch = self.tokenizer.encode_batch([short, long], pad_to_multiple_of=8)
        self.assertEqual(len(batch["input_ids"][0]), len(batch["input_ids"][1]))
        self.assertEqual(len(batch["input_ids"][0]) % 8, 0)

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            self.tokenizer.save(path)
            loaded = ChemMolTokenizerV3.load(path)
            self.assertEqual(loaded.vocabulary, self.tokenizer.vocabulary)


if __name__ == "__main__":
    unittest.main()
