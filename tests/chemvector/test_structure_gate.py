"""Tests for conservative structure expert routing."""

from __future__ import annotations

import unittest

import torch

from chemvector.model import ChemVectorConfig, ChemVectorModel
from chemvector.structure_gate import (
    StructureStyleGate,
    extract_structure_overlay,
    load_packed_structure_expert,
    summarize_gate_scores,
)


class StructureStyleGateTests(unittest.TestCase):
    def test_gate_emits_one_logit_per_image(self) -> None:
        gate = StructureStyleGate(16, hidden_dim=8)
        self.assertEqual(gate(torch.randn(3, 16, 24, 24)).shape, (3,))

    def test_summary_reports_conservative_false_routes(self) -> None:
        summary = summarize_gate_scores(
            torch.tensor((0.95, 0.80)),
            torch.tensor((0.10, 0.92)),
            thresholds=(0.9,),
        )
        result = summary["thresholds"]["0.9000"]
        self.assertEqual(result["expert_recall"], 0.5)
        self.assertEqual(result["base_false_route_rate"], 0.5)

    def test_overlay_rejects_non_structure_changes(self) -> None:
        with self.assertRaisesRegex(ValueError, "Non-structure tensor changed"):
            extract_structure_overlay(
                {"vision.weight": torch.tensor((1.0,))},
                {"vision.weight": torch.tensor((2.0,))},
            )

    def test_packed_expert_matches_full_expert_heads(self) -> None:
        config = ChemVectorConfig(
            vocab_size=64,
            d_model=64,
            decoder_layers=1,
            decoder_heads=4,
            decoder_ffn_dim=128,
            max_target_length=32,
            vision_dims=(16, 32, 48, 64),
            vision_depths=(1, 1, 1, 1),
            stroke_enabled=True,
            stroke_feature_dim=16,
            relation_enabled=True,
            vertex_enabled=True,
            vertex_feature_dim=16,
        )
        base = ChemVectorModel(config).eval()
        expert = ChemVectorModel(config).eval()
        expert.load_state_dict(base.state_dict())
        with torch.no_grad():
            expert.vertex_head.heatmap.bias.add_(1.0)
            expert.relation_head.edge.bias.add_(2.0)
        overlay = extract_structure_overlay(base.state_dict(), expert.state_dict())
        view = load_packed_structure_expert(
            base,
            {"expert_structure_tensors": overlay},
            torch.device("cpu"),
        )
        features = torch.randn(2, 16, 12, 12)
        self.assertTrue(
            torch.equal(
                view.vertex_head(features)["vertex_heatmap_logits"],
                expert.vertex_head(features)["vertex_heatmap_logits"],
            )
        )
        pair_features = torch.randn(5, 16)
        self.assertTrue(
            torch.equal(
                view.relation_head(pair_features)["edge_logits"],
                expert.relation_head(pair_features)["edge_logits"],
            )
        )


if __name__ == "__main__":
    unittest.main()
