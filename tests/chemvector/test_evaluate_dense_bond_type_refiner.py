import unittest

import torch

from chemvector.evaluate_dense_bond_type_refiner import (
    apply_visible_multiple_overrides,
    apply_type_predictions,
    pack_predicted_bond_endpoints,
    reverse_packed_bond_endpoints,
)


class DenseBondTypeRefinerTest(unittest.TestCase):
    def test_pack_uses_one_independent_pair_per_bond(self) -> None:
        ranked = [
            [
                {
                    "endpoints": torch.tensor(((0.1, 0.2), (0.3, 0.4))),
                    "display": 0,
                    "chemical": 0,
                },
                {
                    "endpoints": torch.tensor(((0.5, 0.6), (0.7, 0.8))),
                    "display": 0,
                    "chemical": 0,
                },
            ]
        ]
        points, pair_indices, pair_mask = pack_predicted_bond_endpoints(ranked, [2])
        self.assertEqual(points.shape, (1, 4, 2))
        self.assertTrue(torch.equal(pair_indices[0], torch.tensor(((0, 1), (2, 3)))))
        self.assertTrue(pair_mask.all())

    def test_reverse_swaps_each_bond_without_mixing_bonds(self) -> None:
        points = torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]]
        )
        reversed_points = reverse_packed_bond_endpoints(points, [2])
        expected = torch.tensor(
            [[[0.3, 0.4], [0.1, 0.2], [0.7, 0.8], [0.5, 0.6]]]
        )
        self.assertTrue(torch.equal(reversed_points, expected))

    def test_apply_replaces_only_requested_type_fields(self) -> None:
        ranked = [
            [
                {
                    "score": 0.8,
                    "endpoints": torch.zeros(2, 2),
                    "display": 0,
                    "chemical": 0,
                    "direction": 2,
                },
                {
                    "score": 0.2,
                    "endpoints": torch.ones(2, 2),
                    "display": 1,
                    "chemical": 1,
                    "direction": 1,
                },
            ]
        ]
        refined = apply_type_predictions(
            ranked,
            [1],
            torch.tensor([[0.0, 3.0, 1.0, 2.0]]),
            torch.tensor([[0.0, 1.0, 4.0, 2.0]]),
        )
        self.assertEqual(refined[0][0]["display"], 1)
        self.assertEqual(refined[0][0]["chemical"], 2)
        self.assertEqual(refined[0][0]["direction"], 2)
        self.assertEqual(refined[0][1]["display"], 1)
        self.assertEqual(ranked[0][0]["display"], 0)

    def test_visible_multiple_override_preserves_chemical_and_aromatic(self) -> None:
        ranked = [
            [
                {"display": 0, "chemical": 3},
                {"display": 3, "chemical": 3},
                {"display": 0, "chemical": 0},
            ]
        ]
        logits = torch.tensor(
            [
                [0.0, 4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
        refined = apply_visible_multiple_overrides(ranked, [3], logits, 0.8)
        self.assertEqual(refined[0][0], {"display": 1, "chemical": 3})
        self.assertEqual(refined[0][1], {"display": 3, "chemical": 3})
        self.assertEqual(refined[0][2], {"display": 0, "chemical": 0})


if __name__ == "__main__":
    unittest.main()
