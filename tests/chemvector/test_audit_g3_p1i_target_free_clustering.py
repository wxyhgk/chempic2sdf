import unittest

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p1i_target_free_clustering import (
    cluster_labels,
    sample_cluster_statistics,
)


class G3P1ITargetFreeClusteringTests(unittest.TestCase):
    def test_connected_components_include_transitive_neighbors(self) -> None:
        distances = torch.tensor(
            [[0.0, 1.0, 2.0, 8.0], [1.0, 0.0, 1.0, 7.0],
             [2.0, 1.0, 0.0, 6.0], [8.0, 7.0, 6.0, 0.0]]
        )

        labels = cluster_labels(distances, 1.0)

        self.assertEqual(labels.tolist(), [0, 0, 0, 1])
        self.assertEqual(cluster_labels(distances, 0.0).tolist(), [0, 1, 2, 3])

    def test_component_matching_preserves_one_to_one_targets(self) -> None:
        labels = torch.tensor([0, 0, 1, 2])
        compatible = torch.tensor(
            [[True, False], [True, False], [False, True], [False, False]]
        )

        result = sample_cluster_statistics(labels, compatible)

        self.assertEqual(result["matched_targets"], 2)
        self.assertEqual(result["exact_images"], 1)
        self.assertEqual(result["single_component_targets"], 2)
        self.assertEqual(result["collapsed_candidates"], 1)

    def test_component_shared_by_two_targets_cannot_count_twice(self) -> None:
        labels = torch.tensor([0, 0])
        compatible = torch.tensor([[True, False], [False, True]])

        result = sample_cluster_statistics(labels, compatible)

        self.assertEqual(result["matched_targets"], 1)
        self.assertEqual(result["exact_images"], 0)
        self.assertEqual(result["ambiguous_components"], 1)


if __name__ == "__main__":
    unittest.main()
