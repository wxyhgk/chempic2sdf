"""Tests for joint proposal node-edge reasoning."""

from __future__ import annotations

import unittest

import torch
from PIL import Image

from chemvector.proposal_graph import (
    ProposalGraphCollator,
    ProposalGraphConfig,
    ProposalGraphRefiner,
    decode_proposal_graph,
    proposal_graph_loss,
    proposal_graph_targets,
)


class ProposalGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ProposalGraphConfig(
            input_channels=8,
            node_dim=8,
            edge_dim=4,
            maximum_proposals=4,
            line_samples=8,
            line_band=3,
            incident_topk=2,
            matching_radius_pixels=8.0,
        )

    def test_targets_map_nodes_and_edges_one_to_one(self) -> None:
        points = torch.tensor([[[0.2, 0.5], [0.8, 0.5], [0.5, 0.1], [0.0, 0.0]]])
        proposal_mask = torch.tensor([[True, True, True, False]])
        pairs = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
        anchors = torch.tensor([[[0.2, 0.5], [0.8, 0.5]]])
        anchor_mask = torch.tensor([[True, True]])
        adjacency = torch.zeros(1, 2, 2, dtype=torch.bool)
        adjacency[0, 0, 1] = adjacency[0, 1, 0] = True

        nodes, edges, assignments = proposal_graph_targets(
            points,
            proposal_mask,
            pairs,
            anchors,
            anchor_mask,
            adjacency,
            matching_radius_pixels=8.0,
            image_size=64,
        )

        self.assertEqual(nodes.tolist(), [[1.0, 1.0, 0.0, 0.0]])
        self.assertEqual(edges[0, 0].item(), 1.0)
        self.assertEqual(assignments[0, :3].tolist(), [0, 1, -1])

    def test_forward_loss_and_decode_are_finite(self) -> None:
        model = ProposalGraphRefiner(self.config)
        features = torch.randn(1, 8, 8, 8)
        pixels = torch.ones(1, 3, 32, 32)
        pixels[:, :, 16, 5:27] = -1.0
        heatmap = torch.full((1, 1, 8, 8), -20.0)
        heatmap[0, 0, 4, 1] = 10.0
        heatmap[0, 0, 4, 6] = 9.0
        heatmap[0, 0, 1, 4] = 8.0
        dense = {
            "vertex_heatmap_logits": heatmap,
            "vertex_offset_logits": torch.zeros(1, 2, 8, 8),
            "vertex_aromatic_logits": torch.zeros(1, 1, 8, 8),
        }
        output = model(features, pixels, dense)
        anchors = torch.tensor([[[1.5 / 8, 4.5 / 8], [6.5 / 8, 4.5 / 8]]])
        anchor_mask = torch.tensor([[True, True]])
        adjacency = torch.zeros(1, 2, 2, dtype=torch.bool)
        adjacency[0, 0, 1] = adjacency[0, 1, 0] = True

        losses = proposal_graph_loss(
            output,
            anchors,
            anchor_mask,
            adjacency,
            matching_radius_pixels=8.0,
            image_size=32,
        )
        losses["loss"].backward()
        decoded = decode_proposal_graph(output, threshold=0.1)

        self.assertEqual(output["node_logits"].shape, (1, 4))
        self.assertEqual(output["edge_logits"].shape, (1, 6))
        self.assertEqual(output["incident_edge_support"].shape, (1, 4, 2))
        self.assertEqual(output["local_node_logits"].shape, (1, 4))
        self.assertEqual(output["rescue_node_logits"].shape, (1, 4))
        self.assertIn("proposal_id", decoded[0][0])
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertGreaterEqual(len(decoded[0]), 1)

    def test_global_joint_graph_backpropagates_node_loss_into_edges(self) -> None:
        config = ProposalGraphConfig(
            input_channels=8,
            node_dim=8,
            edge_dim=4,
            maximum_proposals=4,
            line_samples=8,
            line_band=3,
            incident_topk=2,
            matching_radius_pixels=8.0,
            global_layers=1,
            global_heads=2,
            global_ffn_dim=16,
            joint_node_edge_gradients=True,
        )
        model = ProposalGraphRefiner(config)
        torch.nn.init.normal_(model.node_rescue_classifier[-1].weight)
        features = torch.randn(1, 8, 8, 8)
        pixels = torch.ones(1, 3, 32, 32)
        pixels[:, :, 16, 5:27] = -1.0
        heatmap = torch.full((1, 1, 8, 8), -20.0)
        heatmap[0, 0, 4, 1] = 10.0
        heatmap[0, 0, 4, 6] = 9.0
        dense = {
            "vertex_heatmap_logits": heatmap,
            "vertex_offset_logits": torch.zeros(1, 2, 8, 8),
            "vertex_aromatic_logits": torch.zeros(1, 1, 8, 8),
        }

        output = model(features, pixels, dense)
        output["node_logits"].sum().backward()

        self.assertIsNotNone(model.edge_classifier.weight.grad)
        self.assertGreater(float(model.edge_classifier.weight.grad.abs().sum()), 0.0)

    def test_global_heads_must_divide_node_dimension(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            ProposalGraphConfig(
                input_channels=8,
                node_dim=10,
                edge_dim=4,
                maximum_proposals=4,
                line_samples=8,
                line_band=3,
                incident_topk=2,
                global_heads=4,
            )

    def test_collator_builds_symmetric_adjacency(self) -> None:
        sample = {
            "key": "sample",
            "image": Image.new("RGB", (32, 32), "white"),
            "render": {
                "width": 32,
                "height": 32,
                "atoms": [
                    {"atom_id": 0, "image_x": 8, "image_y": 16},
                    {"atom_id": 1, "image_x": 24, "image_y": 16},
                ],
            },
            "chemmol": {
                "atoms": [
                    {"element": "C", "aromatic": False},
                    {"element": "O", "aromatic": False},
                ],
                "bonds": [{"begin_atom": 0, "end_atom": 1}],
            },
        }

        batch = ProposalGraphCollator(image_size=32)([sample])

        self.assertEqual(batch["adjacency"].shape, (1, 2, 2))
        self.assertTrue(batch["adjacency"][0, 0, 1])
        self.assertTrue(batch["adjacency"][0, 1, 0])
        self.assertFalse(batch["adjacency"][0, 0, 0])

    def test_loss_without_positive_edge_is_finite(self) -> None:
        model = ProposalGraphRefiner(self.config)
        features = torch.randn(1, 8, 8, 8)
        pixels = torch.ones(1, 3, 32, 32)
        heatmap = torch.full((1, 1, 8, 8), -20.0)
        heatmap[0, 0, 4, 1] = 10.0
        dense = {
            "vertex_heatmap_logits": heatmap,
            "vertex_offset_logits": torch.zeros(1, 2, 8, 8),
            "vertex_aromatic_logits": torch.zeros(1, 1, 8, 8),
        }
        output = model(features, pixels, dense)
        anchors = torch.tensor([[[1.5 / 8, 4.5 / 8]]])
        mask = torch.tensor([[True]])
        adjacency = torch.zeros(1, 1, 1, dtype=torch.bool)

        losses = proposal_graph_loss(
            output,
            anchors,
            mask,
            adjacency,
            matching_radius_pixels=8.0,
            image_size=32,
        )

        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertEqual(losses["edge_targets"].sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
