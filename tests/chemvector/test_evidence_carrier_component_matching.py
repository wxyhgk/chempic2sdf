import unittest

import torch

from chemvector.evidence_carrier_component_matching import (
    apply_component_matching_selection,
    component_structured_loss,
    enumerate_component_matchings,
    optimal_component_matchings,
)


class ComponentMatchingTest(unittest.TestCase):
    def test_enumerates_triangle_matchings(self) -> None:
        masks = enumerate_component_matchings([[0, 1], [1, 2], [0, 2]])
        self.assertEqual(masks.shape, (4, 3))
        self.assertEqual(masks.sum(dim=1).tolist(), [0, 1, 1, 1])

    def test_optimal_matchings_accept_equivalent_positive_edges(self) -> None:
        masks = enumerate_component_matchings([[0, 1], [1, 2], [0, 2]])
        optimal = optimal_component_matchings(masks, [1, 1, 0])
        self.assertEqual(int(optimal.sum()), 2)
        energies = torch.zeros(3, requires_grad=True)
        loss = component_structured_loss(energies, masks, optimal)
        self.assertAlmostEqual(float(loss.detach()), 0.693147, places=5)
        loss.backward()
        self.assertIsNotNone(energies.grad)

    def test_selection_only_deletes_active_edges(self) -> None:
        masks = enumerate_component_matchings([[0, 1], [1, 2]])
        baseline = torch.tensor([2.0, 1.5, -0.25])
        residual = torch.tensor([-4.0, 2.0, 100.0])
        result = apply_component_matching_selection(
            baseline,
            residual,
            [{"active_pair_indices": torch.tensor([0, 1]), "matching_masks": masks}],
            threshold=0.0,
        )
        torch.testing.assert_close(result, torch.tensor([-1.0, 1.5, -0.25]))


if __name__ == "__main__":
    unittest.main()
