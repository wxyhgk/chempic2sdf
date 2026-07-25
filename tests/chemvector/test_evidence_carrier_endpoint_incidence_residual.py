from __future__ import annotations

import unittest

import torch

from chemvector.evidence_carrier_endpoint_incidence_residual import (
    ENDPOINT_RESIDUAL_PARAMETER_COUNT,
    EvidenceCarrierEndpointIncidenceResidual,
    endpoint_incidence_context,
    endpoint_safe_residual_loss,
)
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
)


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.zeros((1, 3, 21), dtype=torch.float32)
    tokens[0, :, 0] = torch.tensor((1.0, 2.0, 3.0))
    tokens[0, :, -1] = torch.tensor((1.0, 0.5, -0.5))
    indices = torch.tensor((((0, 1), (0, 2), (3, 4)),), dtype=torch.long)
    mask = torch.ones((1, 3), dtype=torch.bool)
    global_features = torch.zeros((1, 4), dtype=torch.float32)
    return tokens, indices, mask, global_features


class EndpointIncidenceResidualTest(unittest.TestCase):
    def test_parameter_count_and_zero_residual_baseline(self) -> None:
        model = EvidenceCarrierEndpointIncidenceResidual()
        tokens, indices, mask, global_features = _inputs()
        output = model(tokens, indices, mask, global_features)

        self.assertEqual(model.parameter_count(), ENDPOINT_RESIDUAL_PARAMETER_COUNT)
        torch.testing.assert_close(output["residual"], torch.zeros_like(output["residual"]))
        torch.testing.assert_close(
            output["margin"],
            tokens[..., -1] - SAMPLE_THRESHOLD_BASELINE_TAU,
        )

    def test_pair_order_is_equivariant(self) -> None:
        torch.manual_seed(7)
        model = EvidenceCarrierEndpointIncidenceResidual()
        with torch.no_grad():
            model.residual_head[-1].weight.fill_(0.1)
        tokens, indices, mask, global_features = _inputs()
        order = torch.tensor((2, 0, 1))
        first = model(tokens, indices, mask, global_features)["residual"]
        second = model(
            tokens[:, order], indices[:, order], mask[:, order], global_features
        )["residual"]
        torch.testing.assert_close(second, first[:, order])

    def test_anchor_renumbering_does_not_change_output(self) -> None:
        torch.manual_seed(11)
        model = EvidenceCarrierEndpointIncidenceResidual()
        with torch.no_grad():
            model.residual_head[-1].weight.fill_(0.1)
        tokens, indices, mask, global_features = _inputs()
        mapping = torch.tensor((4, 2, 0, 3, 1))
        remapped = mapping[indices]
        first = model(tokens, indices, mask, global_features)["residual"]
        second = model(tokens, remapped, mask, global_features)["residual"]
        torch.testing.assert_close(first, second)

    def test_endpoint_context_distinguishes_incidence_graphs(self) -> None:
        encoded = torch.zeros((1, 3, 12), dtype=torch.float32)
        encoded[0, :, 0] = torch.tensor((1.0, 2.0, 4.0))
        mask = torch.ones((1, 3), dtype=torch.bool)
        first_indices = torch.tensor((((0, 1), (0, 2), (3, 4)),))
        second_indices = torch.tensor((((0, 1), (2, 3), (2, 4)),))
        first = endpoint_incidence_context(encoded, first_indices, mask)
        second = endpoint_incidence_context(encoded, second_indices, mask)
        self.assertFalse(torch.equal(first, second))

    def test_loss_penalizes_worst_negative_and_upward_residual(self) -> None:
        margins = torch.tensor(((0.5, 0.4, -1.0),), requires_grad=True)
        residuals = torch.tensor(((0.0, 0.3, -0.2),), requires_grad=True)
        indices = torch.tensor((((0, 1), (0, 2), (3, 4)),))
        labels = torch.tensor(((1, 0, 0),))
        mask = torch.ones((1, 3), dtype=torch.bool)
        result = endpoint_safe_residual_loss(
            margins, residuals, indices, labels, mask
        )
        self.assertGreater(float(result["worst_negative_loss"].detach()), 0.0)
        self.assertGreater(float(result["negative_upward_loss"].detach()), 0.0)
        result["loss"].backward()
        self.assertTrue(torch.isfinite(margins.grad).all())
        self.assertTrue(torch.isfinite(residuals.grad).all())

    def test_training_updates_encoder_through_incidence_pooling(self) -> None:
        torch.manual_seed(23)
        model = EvidenceCarrierEndpointIncidenceResidual()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        tokens, indices, mask, global_features = _inputs()
        labels = torch.tensor(((1, 0, 0),))
        initial = model.pair_encoder[0].weight.detach().clone()
        for _ in range(3):
            output = model(tokens, indices, mask, global_features)
            loss = endpoint_safe_residual_loss(
                output["margin"],
                output["residual"],
                indices,
                labels,
                mask,
            )["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        self.assertFalse(torch.equal(initial, model.pair_encoder[0].weight))

    def test_loss_has_dead_zone_for_safe_pairs(self) -> None:
        margins = torch.tensor(((0.5, -0.5),))
        residuals = torch.tensor(((0.0, 0.0),))
        indices = torch.tensor((((0, 1), (2, 3)),))
        labels = torch.tensor(((1, 0),))
        mask = torch.ones((1, 2), dtype=torch.bool)
        result = endpoint_safe_residual_loss(
            margins, residuals, indices, labels, mask
        )
        self.assertEqual(float(result["positive_loss"]), 0.0)
        self.assertEqual(float(result["negative_loss"]), 0.0)
        self.assertEqual(float(result["worst_negative_loss"]), 0.0)


if __name__ == "__main__":
    unittest.main()
