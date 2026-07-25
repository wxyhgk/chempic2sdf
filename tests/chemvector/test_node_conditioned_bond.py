import unittest

import torch

from chemvector.node_conditioned_bond import (
    NodeConditionedBondConfig,
    NodeConditionedBondObject,
    enumerate_all_pairs,
    node_conditioned_bond_loss,
)
from chemvector.evaluate_node_conditioned_bond import (
    match_vertices,
    point_to_segment_distance,
    unmatched_vertex_diagnostics,
)
from chemvector.pretrain_node_conditioned_bond import KnownEndpointBondCollator


class NodeConditionedBondTests(unittest.TestCase):
    def test_endpoint_jitter_is_bounded_and_leaves_padding_fixed(self) -> None:
        collator = KnownEndpointBondCollator(
            image_size=100,
            endpoint_jitter_sigma_pixels=2.0,
            endpoint_jitter_max_pixels=3.0,
            endpoint_jitter_probability=1.0,
            seed=7,
        )
        points = torch.tensor([[[0.5, 0.5], [0.2, 0.2], [0.0, 0.0]]])
        mask = torch.tensor([[True, True, False]])
        jittered = collator._jitter_points(points, mask)
        distances = torch.linalg.vector_norm(jittered - points, dim=-1) * 100
        self.assertLessEqual(float(distances[0, :2].max()), 3.0001)
        self.assertEqual(float(distances[0, 2]), 0.0)

    def test_endpoint_jitter_defaults_to_identity(self) -> None:
        collator = KnownEndpointBondCollator()
        points = torch.rand((1, 3, 2))
        mask = torch.ones((1, 3), dtype=torch.bool)
        self.assertTrue(torch.equal(collator._jitter_points(points, mask), points))

    def test_enumerate_all_pairs_keeps_long_pairs(self) -> None:
        mask = torch.tensor([[True, True, True, True]])
        pairs, pair_mask = enumerate_all_pairs(mask)
        self.assertEqual(pair_mask.sum().item(), 6)
        self.assertIn([0, 3], pairs[0, pair_mask[0]].tolist())

    def test_wide_line_band_preserves_parallel_strokes(self) -> None:
        config = NodeConditionedBondConfig(
            line_samples=32, line_band=13, line_band_radius_pixels=6.0
        )
        model = NodeConditionedBondObject(config)
        pixels = torch.ones((1, 3, 64, 64))
        pixels[:, :, 30, 8:56] = -1.0
        pixels[:, :, 34, 8:56] = -1.0
        points = torch.tensor([[[8 / 64, 32 / 64], [56 / 64, 32 / 64]]])
        pairs = torch.tensor([[[0, 1]]])
        mask = torch.ones((1, 1), dtype=torch.bool)
        bands, geometry = model.extract_line_bands(pixels, points, pairs, mask)
        profile = bands[0, 0].mean(dim=-1)
        self.assertEqual(bands.shape, (1, 1, 13, 32))
        self.assertEqual(geometry.shape, (1, 5))
        self.assertGreaterEqual(int((profile >= 0.49).sum()), 4)

    def test_node_conditioned_bond_forward_and_loss(self) -> None:
        model = NodeConditionedBondObject(
            NodeConditionedBondConfig(line_samples=16, line_band=9, hidden_dim=32)
        )
        pixels = torch.ones((1, 3, 64, 64))
        points = torch.tensor([[[0.1, 0.1], [0.5, 0.5], [0.9, 0.1]]])
        point_mask = torch.ones((1, 3), dtype=torch.bool)
        pairs, pair_mask = enumerate_all_pairs(point_mask)
        output = model(pixels, points, pairs, pair_mask)
        edge = torch.tensor([1.0, 0.0, 1.0])
        display = torch.tensor([0, 0, 1])
        chemical = torch.tensor([0, 0, 1])
        direction = torch.tensor([0, 0, 0])
        losses = node_conditioned_bond_loss(
            output, edge, display, chemical, direction
        )
        self.assertEqual(output["edge_logits"].shape, (3,))
        self.assertEqual(output["hidden"].shape, (3, 32))
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_vertex_matching_rejects_distant_predictions(self) -> None:
        predicted = torch.tensor([[0.1, 0.1], [0.9, 0.9]])
        expected = torch.tensor([[0.1, 0.1], [0.5, 0.5]])
        self.assertEqual(match_vertices(predicted, expected), {0: 0})

    def test_segment_distance_uses_finite_endpoints(self) -> None:
        point = torch.tensor([2.0, 1.0])
        begin = torch.tensor([0.0, 0.0])
        end = torch.tensor([1.0, 0.0])
        self.assertAlmostEqual(point_to_segment_distance(point, begin, end), 2**0.5)

    def test_unmatched_vertex_marks_near_double_bond(self) -> None:
        sample = {
            "layers": {
                "width": 100,
                "height": 100,
                "atoms": [
                    {"atom_id": 0, "center": [10.0, 50.0]},
                    {"atom_id": 1, "center": [90.0, 50.0]},
                ],
                "bonds": [
                    {
                        "bond_id": 0,
                        "begin": [10.0, 50.0],
                        "end": [90.0, 50.0],
                        "display_type": "DOUBLE",
                    }
                ],
            }
        }
        rows = unmatched_vertex_diagnostics(
            torch.tensor([[0.5, 0.55]]), {}, sample
        )
        self.assertTrue(rows[0]["near_multiple_bond"])
        self.assertAlmostEqual(rows[0]["nearest_bond_distance_pixels"], 5.0)


if __name__ == "__main__":
    unittest.main()
