"""Tests for incremental decoding and ChemMol grammar constraints."""

from __future__ import annotations

import unittest

import torch

from Data.sdf.chemmol_tokenizer import ChemMolTokenizer
from chemvector.generation import ChemMolGrammar
from chemvector.model import ChemVectorConfig, ChemVectorModel


class GenerationTests(unittest.TestCase):
    def test_ground_truth_sequence_is_accepted_by_grammar(self) -> None:
        tokenizer = ChemMolTokenizer()
        sequence = """<mol v=0.2>
<a 0 C 0 0 0 0 0 0 CHI_UNSPECIFIED 100 200>
<a 1 O 0 0 0 0 0 0 CHI_UNSPECIFIED 300 200>
<b 0 0 1 SINGLE SINGLE STEREONONE -1 -1 NONE>
<mol_end>
"""
        grammar = ChemMolGrammar(tokenizer)
        for token_id in tokenizer.encode(sequence):
            self.assertIn(token_id, grammar.allowed_token_ids())
            grammar.consume(token_id)
        self.assertEqual(grammar.phase, "done")

    def test_incremental_logits_match_full_decoder(self) -> None:
        torch.manual_seed(7)
        config = ChemVectorConfig(
            vocab_size=31,
            d_model=32,
            decoder_layers=2,
            decoder_heads=4,
            decoder_ffn_dim=64,
            dropout=0.0,
            max_target_length=16,
            vision_dims=(8, 16),
            vision_depths=(1, 1),
        )
        model = ChemVectorModel(config).eval()
        memory = torch.randn(1, 5, config.d_model)
        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        full_logits = model.decode(input_ids, memory)
        caches = None
        step_logits = []
        for position in range(input_ids.size(1)):
            logits, caches = model.decode_step(
                input_ids[:, position : position + 1],
                position,
                memory,
                caches,
            )
            step_logits.append(logits)
        incremental_logits = torch.cat(step_logits, dim=1)
        torch.testing.assert_close(incremental_logits, full_logits, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
