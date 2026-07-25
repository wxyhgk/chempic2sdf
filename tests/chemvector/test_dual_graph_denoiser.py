import unittest

import torch

from chemvector.dual_graph_denoiser import (
    DualGraphDenoiserConfig,
    GroundedDualGraphDenoiser,
    straight_through_node_features,
)
from chemvector.structured_selector import (
    GroundedStructuredSelector,
    StructuredSelectorConfig,
)


class DualGraphDenoiserTests(unittest.TestCase):
    def test_feature_and_pixel_patch_modes_preserve_zero_initialized_output(self) -> None:
        for mode in ("feature", "pixel"):
            model = self._model(patch_mode=mode)
            hidden = torch.randn(1, 4, 8)
            base_logits = torch.zeros(1, 4)
            points = torch.rand(1, 4, 2)
            mask = torch.ones(1, 4, dtype=torch.bool)
            pairs = torch.triu_indices(4, 4, offset=1).transpose(0, 1)
            edges = torch.randn(1, len(pairs))
            visual = torch.randn(1, 4, 12)
            pixel = torch.randint(0, 256, (1, 4, 1, 8, 8), dtype=torch.uint8)
            auxiliary = torch.randn(1, 4, 4)
            expected = model.base_selector(
                hidden, base_logits, points, mask, edges, pairs, auxiliary, visual,
                torch.tensor([[0, 1, 0, 1]])
            )
            actual = model(
                hidden,
                base_logits,
                points,
                mask,
                edges,
                pairs,
                auxiliary,
                visual,
                torch.tensor([[0, 1, 0, 1]]),
                candidate_pixel_patches=pixel if mode == "pixel" else None,
            )
            self.assertTrue(torch.allclose(actual, expected))

    def test_semantic_pixel_context_is_zero_initialized_and_reports_logits(self) -> None:
        model = self._model(patch_mode="semantic_pixel")
        hidden = torch.randn(1, 4, 8)
        base_logits = torch.zeros(1, 4)
        points = torch.rand(1, 4, 2)
        mask = torch.ones(1, 4, dtype=torch.bool)
        pairs = torch.triu_indices(4, 4, offset=1).transpose(0, 1)
        edges = torch.randn(1, len(pairs))
        auxiliary = torch.randn(1, 4, 4)
        roles = torch.tensor([[0, 1, 0, 1]])
        patches = torch.randint(0, 256, (1, 4, 1, 8, 8), dtype=torch.uint8)
        expected = model.base_selector(
            hidden, base_logits, points, mask, edges, pairs, auxiliary, None, roles
        )
        output = model.forward_with_edges(
            hidden,
            base_logits,
            points,
            mask,
            edges,
            pairs,
            auxiliary,
            None,
            roles,
            candidate_pixel_patches=patches,
        )
        self.assertTrue(torch.allclose(output["node_logits"], expected))
        self.assertEqual(output["pass_through_logits"].shape, (1, 4))

        output["node_logits"].sum().backward()
        self.assertIsNone(model.pass_through_context.classifier.weight.grad)
        self.assertIsNotNone(model.pass_through_context.context_projection.weight.grad)

    def test_straight_through_support_marks_collinear_middle_candidate(self) -> None:
        points = torch.tensor([[[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]]])
        pairs = torch.tensor([[0, 1], [0, 2], [1, 2]])
        probabilities = torch.tensor([[0.95, 0.01, 0.95]])
        auxiliary = torch.zeros(1, 3, 4)
        auxiliary[0, 0, 3] = 1.0
        auxiliary[0, 2, 1] = 1.0
        features = straight_through_node_features(
            points,
            torch.ones(1, 3, dtype=torch.bool),
            probabilities,
            pairs,
            auxiliary,
            topk=3,
        )
        self.assertGreater(float(features[0, 1, 0]), 0.8)
        self.assertGreater(float(features[0, 1, 1]), 0.8)
        self.assertLess(float(features[0, 0, 0]), 0.1)

    def test_joint_refiner_updates_all_valid_edges(self) -> None:
        model = self._model(refine_edges=True, edge_auxiliary_dim=4)
        pairs = torch.triu_indices(4, 4, offset=1).transpose(0, 1)
        output = model.forward_with_edges(
            torch.randn(1, 4, 8),
            torch.zeros(1, 4),
            torch.rand(1, 4, 2),
            torch.ones(1, 4, dtype=torch.bool),
            torch.randn(1, len(pairs)),
            pairs,
            torch.randn(1, 4, 4),
            None,
            torch.tensor([[0, 1, 0, 1]]),
            torch.rand(1, len(pairs), 4),
        )
        self.assertEqual(output["edge_logits"].shape, (1, len(pairs)))
        output["edge_logits"].sum().backward()
        gradients = [
            layer.edge_residual.weight.grad for layer in model.message_layers
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))

    def _model(
        self,
        *,
        refine_edges: bool = False,
        edge_auxiliary_dim: int = 0,
        straight_through_topk: int = 0,
        patch_mode: str = "none",
    ) -> GroundedDualGraphDenoiser:
        base = GroundedStructuredSelector(
            StructuredSelectorConfig(
                candidate_dim=8,
                auxiliary_dim=4,
                visual_dim=12 if patch_mode == "feature" else 0,
                d_model=8,
                heads=2,
                ffn_dim=16,
                layers=1,
                maximum_candidates=4,
                incident_topk=2,
                role_count=2,
            )
        )
        return GroundedDualGraphDenoiser(
            base,
            DualGraphDenoiserConfig(
                d_model=8,
                edge_dim=4,
                layers=2,
                refine_edges=refine_edges,
                edge_auxiliary_dim=edge_auxiliary_dim,
                straight_through_topk=straight_through_topk,
                patch_mode=patch_mode,
                patch_size=8,
            ),
        )

    def test_zero_residual_preserves_deployed_base_selector(self) -> None:
        model = self._model()
        hidden = torch.randn(1, 4, 8)
        base_logits = torch.tensor([[2.0, 1.0, -1.0, -30.0]])
        points = torch.rand(1, 4, 2)
        mask = torch.tensor([[True, True, True, False]])
        pairs = torch.triu_indices(4, 4, offset=1).transpose(0, 1)
        edges = torch.randn(1, len(pairs))
        auxiliary = torch.randn(1, 4, 4)
        roles = torch.tensor([[0, 1, 0, 0]])

        expected = model.base_selector(
            hidden, base_logits, points, mask, edges, pairs, auxiliary, None, roles
        )
        actual = model(
            hidden, base_logits, points, mask, edges, pairs, auxiliary, None, roles
        )

        self.assertTrue(torch.equal(actual, expected))

    def test_full_adjacency_residual_receives_gradients(self) -> None:
        model = self._model()
        hidden = torch.randn(1, 4, 8)
        base_logits = torch.zeros(1, 4)
        points = torch.rand(1, 4, 2)
        mask = torch.ones(1, 4, dtype=torch.bool)
        pairs = torch.triu_indices(4, 4, offset=1).transpose(0, 1)
        edges = torch.randn(1, len(pairs))
        auxiliary = torch.randn(1, 4, 4)
        roles = torch.tensor([[0, 1, 0, 1]])
        output = model(
            hidden, base_logits, points, mask, edges, pairs, auxiliary, None, roles
        )
        output.sum().backward()

        self.assertIsNotNone(model.message_layers[0].edge_encoder[1].weight.grad)
        self.assertEqual(len(pairs), 6)


if __name__ == "__main__":
    unittest.main()
