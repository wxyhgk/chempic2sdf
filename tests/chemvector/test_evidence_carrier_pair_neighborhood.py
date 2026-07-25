import unittest

import numpy as np
import torch

from chemvector.evidence_carrier_pair_neighborhood import (
    PairNeighborhoodConfig,
    build_pair_neighborhood,
)


class PairNeighborhoodTest(unittest.TestCase):
    def test_candidate_and_endpoint_order_are_invariant(self) -> None:
        segments = np.asarray(
            [
                [[0.0, 10.0], [20.0, 10.0]],
                [[0.0, 13.0], [20.0, 13.0]],
                [[0.0, 7.0], [10.0, 7.0]],
                [[10.0, 16.0], [20.0, 16.0]],
            ]
        )
        original = build_pair_neighborhood(segments, [[0, 1]])
        transformed = build_pair_neighborhood(
            segments[::-1, ::-1].copy(), [[2, 3]]
        )
        torch.testing.assert_close(
            original["neighbor_features"], transformed["neighbor_features"]
        )
        torch.testing.assert_close(original["neighbor_mask"], transformed["neighbor_mask"])

    def test_left_and_right_partial_support_are_preserved(self) -> None:
        segments = np.asarray(
            [
                [[0.0, 10.0], [20.0, 10.0]],
                [[0.0, 13.0], [20.0, 13.0]],
                [[0.0, 7.0], [10.0, 7.0]],
                [[10.0, 16.0], [20.0, 16.0]],
            ]
        )
        result = build_pair_neighborhood(segments, [[0, 1]])
        active = result["neighbor_features"][0, result["neighbor_mask"][0]]
        self.assertEqual(len(active), 2)
        self.assertTrue(bool(((active[:, 3] > active[:, 4]).any())))
        self.assertTrue(bool(((active[:, 4] > active[:, 3]).any())))

    def test_limits_neighbors_and_rejects_invalid_config(self) -> None:
        segments = np.asarray(
            [[[0.0, y], [20.0, y]] for y in range(20)], dtype=np.float64
        )
        result = build_pair_neighborhood(
            segments, [[8, 9]], config=PairNeighborhoodConfig(maximum_neighbors=4)
        )
        self.assertEqual(int(result["neighbor_mask"].sum()), 4)
        with self.assertRaisesRegex(ValueError, "maximum_neighbors"):
            PairNeighborhoodConfig(maximum_neighbors=0)


if __name__ == "__main__":
    unittest.main()
