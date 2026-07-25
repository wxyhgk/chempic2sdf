import unittest

import torch

from chemvector.build_predicted_junction_relation_data import (
    match_predicted_junctions,
)


class PredictedJunctionRelationDataTest(unittest.TestCase):
    def test_matching_is_one_to_one_and_radius_bounded(self) -> None:
        predicted = torch.tensor(
            [[0.10, 0.10], [0.101, 0.10], [0.80, 0.80]], dtype=torch.float32
        )
        targets = {4: (0.10, 0.10), 7: (0.50, 0.50)}

        mapping, distances = match_predicted_junctions(
            predicted,
            targets,
            image_size=100,
            maximum_distance_pixels=5.0,
        )

        self.assertEqual(len(mapping), 1)
        self.assertEqual(set(mapping.values()), {4})
        self.assertLessEqual(max(distances), 5.0)


if __name__ == "__main__":
    unittest.main()
