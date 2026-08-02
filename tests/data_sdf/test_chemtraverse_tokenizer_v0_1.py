"""Tests for the ChemTraverse v0.1 tokenizer."""

from __future__ import annotations

import tempfile
from pathlib import Path

from Data.sdf.chemtraverse_tokenizer_v0_1 import ChemTraverseTokenizerV01
from Data.sdf.chemtraverse_v0_1 import serialize_chemtraverse_v0_1
from tests.data_sdf.test_chemtraverse_v0_1 import chiral_scene, ring_scene


def test_encode_decode_preserves_cycle_stereo_and_utf8() -> None:
    tokenizer = ChemTraverseTokenizerV01()
    scenes = [ring_scene(), chiral_scene()]
    scenes[0]["molecule_id"] = "分子-A1"
    for scene in scenes:
        sequence = serialize_chemtraverse_v0_1(scene)
        assert tokenizer.decode(tokenizer.encode(sequence)) == sequence


def test_dynamic_padding_and_save_load() -> None:
    tokenizer = ChemTraverseTokenizerV01()
    sequences = [
        serialize_chemtraverse_v0_1(ring_scene()),
        serialize_chemtraverse_v0_1(chiral_scene()),
    ]
    batch = tokenizer.encode_batch(sequences, pad_to_multiple_of=8)
    assert len(batch["input_ids"][0]) == len(batch["input_ids"][1])
    assert len(batch["input_ids"][0]) % 8 == 0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tokenizer.json"
        tokenizer.save(path)
        assert ChemTraverseTokenizerV01.load(path).vocabulary == tokenizer.vocabulary
