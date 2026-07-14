#!/usr/bin/env python3
"""Unit tests for the ChemSVG v0.4 tokenizer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from .chemsvg_sequence_v4 import serialize_chemsvg_v4
from .chemsvg_tokenizer_v4 import ChemSVGTokenizerV4
from .test_chemmol_sequence import make_annotations


class ChemSVGTokenizerV4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = ChemSVGTokenizerV4()

    def test_encode_decode_is_lossless(self) -> None:
        for smiles in ("CCO", "c1cc[nH]c1", "F/C=C/F", "N[C@@H](C)C(=O)O"):
            with self.subTest(smiles=smiles):
                sequence = serialize_chemsvg_v4(*make_annotations(smiles))
                self.assertEqual(
                    self.tokenizer.decode(self.tokenizer.encode(sequence)),
                    sequence,
                )

    def test_dynamic_batch_padding(self) -> None:
        short = serialize_chemsvg_v4(*make_annotations("CC"))
        long = serialize_chemsvg_v4(*make_annotations("c1ccccc1"))
        batch = self.tokenizer.encode_batch([short, long], pad_to_multiple_of=8)
        self.assertEqual(len(batch["input_ids"][0]), len(batch["input_ids"][1]))
        self.assertEqual(len(batch["input_ids"][0]) % 8, 0)

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            self.tokenizer.save(path)
            loaded = ChemSVGTokenizerV4.load(path)
            self.assertEqual(loaded.vocabulary, self.tokenizer.vocabulary)


if __name__ == "__main__":
    unittest.main()
