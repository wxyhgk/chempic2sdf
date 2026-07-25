import unittest

import torch

from chemvector.edge_evidence import score_full_adjacency_edges
from chemvector.node_conditioned_bond import (
    NodeConditionedBondConfig,
    NodeConditionedBondObject,
)


class EdgeEvidenceTests(unittest.TestCase):
    def test_chunked_scores_match_single_teacher_call(self) -> None:
        torch.manual_seed(7)
        teacher = NodeConditionedBondObject(
            NodeConditionedBondConfig(
                line_samples=16,
                line_band=9,
                line_band_radius_pixels=4.0,
                hidden_dim=16,
            )
        ).eval()
        pixels = torch.rand(2, 3, 64, 64).mul(2.0).sub(1.0)
        points = torch.rand(2, 5, 2)
        mask = torch.tensor(
            [[True, True, True, True, True], [True, True, True, False, False]]
        )
        pairs = torch.triu_indices(5, 5, offset=1).transpose(0, 1)
        expected = score_full_adjacency_edges(
            teacher, pixels, points, mask, pairs, chunk_size=len(pairs)
        )
        actual = score_full_adjacency_edges(
            teacher, pixels, points, mask, pairs, chunk_size=3
        )
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    def test_invalid_chunk_size_is_rejected(self) -> None:
        teacher = NodeConditionedBondObject()
        with self.assertRaises(ValueError):
            score_full_adjacency_edges(
                teacher,
                torch.zeros(1, 3, 32, 32),
                torch.zeros(1, 2, 2),
                torch.ones(1, 2, dtype=torch.bool),
                torch.tensor([[0, 1]]),
                chunk_size=0,
            )


if __name__ == "__main__":
    unittest.main()
