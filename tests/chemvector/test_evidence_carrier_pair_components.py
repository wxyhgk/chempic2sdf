import unittest

import torch

from chemvector.evidence_carrier_pair_components import (
    apply_multi_component_label_oracle,
    pair_component_metadata,
)


class PairComponentsTest(unittest.TestCase):
    def test_component_metadata_is_canonical(self) -> None:
        pairs = torch.tensor([[5, 6], [0, 1], [1, 2], [6, 7], [0, 2]])
        result = pair_component_metadata(pairs)
        self.assertEqual(result["component_ids"].tolist(), [1, 0, 0, 1, 0])
        self.assertEqual(result["component_anchor_counts"].tolist(), [3, 3, 3, 3, 3])
        self.assertEqual(result["component_edge_counts"].tolist(), [2, 3, 3, 2, 3])
        self.assertTrue(bool(result["multi_anchor_component"].all()))

    def test_oracle_leaves_isolated_pair_unchanged(self) -> None:
        pairs = torch.tensor([[0, 1], [2, 3], [3, 4]])
        scores = torch.tensor([0.25, 0.5, 0.75])
        labels = torch.tensor([0, 0, 1])
        result = apply_multi_component_label_oracle(
            scores, pairs, labels, threshold=0.0
        )
        torch.testing.assert_close(result, torch.tensor([0.25, -1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
