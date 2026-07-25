import unittest

import torch

from chemvector.bond_set_selector import (
    BondSetSelector,
    BondSetSelectorConfig,
    bond_set_selector_loss,
    decode_bond_set,
    decode_bond_set_threshold,
    structured_unique_bond_counts,
)
from chemvector.node_conditioned_bond import (
    NodeConditionedBondConfig,
    NodeConditionedBondObject,
)
from chemvector.pretrain_bond_set_selector import read_line_band_features


class BondSetSelectorTest(unittest.TestCase):
    def test_line_band_features_preserve_candidate_batches(self) -> None:
        reader = NodeConditionedBondObject(
            NodeConditionedBondConfig(
                line_samples=16, line_band=9, hidden_dim=32
            )
        )
        pixels = torch.ones((2, 3, 64, 64))
        object_batches = [
            [
                {"endpoints": torch.tensor(((0.1, 0.1), (0.5, 0.5)))},
                {"endpoints": torch.tensor(((0.5, 0.5), (0.9, 0.1)))},
            ],
            [
                {"endpoints": torch.tensor(((0.2, 0.8), (0.8, 0.8)))},
            ],
        ]
        features = read_line_band_features(reader, pixels, object_batches)
        self.assertEqual([tuple(item.shape) for item in features], [(2, 46), (1, 46)])

    def test_forward_loss_and_structured_decode(self) -> None:
        config = BondSetSelectorConfig(
            input_dim=6,
            d_model=16,
            layers=1,
            heads=4,
            ffn_dim=32,
            maximum_bonds=4,
        )
        model = BondSetSelector(config)
        features = torch.randn(1, 4, 6)
        mask = torch.tensor(((True, True, True, True),))
        display = torch.zeros((1, 4), dtype=torch.long)
        chemical = torch.zeros((1, 4), dtype=torch.long)
        output = model(features, mask, display, chemical)
        targets = torch.tensor(((True, True, False, False),))
        losses = bond_set_selector_loss(
            output,
            mask,
            targets,
            torch.tensor((2,)),
            display,
            chemical,
        )
        self.assertTrue(torch.isfinite(losses["loss"]))

        output["count_logits"].fill_(-10.0)
        output["count_logits"][0, 2] = 10.0
        output["keep_logits"] = torch.tensor(((4.0, 3.0, 2.0, 1.0),))
        pairs = torch.tensor((((0, 1), (1, 0), (1, 2), (2, 2)),))
        self.assertEqual(decode_bond_set(output, pairs, mask), [[0, 2]])
        override = torch.tensor((1,))
        self.assertEqual(decode_bond_set(output, pairs, mask, override), [[0]])
        self.assertEqual(
            decode_bond_set_threshold(
                output["keep_logits"], pairs, mask, threshold=0.8
            ),
            [[0, 2]],
        )

    def test_structured_count_deduplicates_and_rejects_self_loops(self) -> None:
        logits = torch.logit(torch.tensor(((0.9, 0.8, 0.7, 0.1),)))
        pairs = torch.tensor((((0, 1), (1, 0), (2, 2), (1, 2)),))
        mask = torch.ones((1, 4), dtype=torch.bool)
        counts = structured_unique_bond_counts(
            logits, pairs, mask, threshold=0.25
        )
        self.assertEqual(counts.tolist(), [1])


if __name__ == "__main__":
    unittest.main()
