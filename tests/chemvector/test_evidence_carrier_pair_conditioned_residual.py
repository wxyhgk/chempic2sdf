from __future__ import annotations

import inspect
import math
import unittest

import torch
from torch.nn import functional as F

from chemvector.evidence_carrier_pair_conditioned_residual import (
    PAIR_RESIDUAL_PARAMETER_COUNT,
    EvidenceCarrierPairConditionedResidual,
    pair_conditioned_residual_loss,
)
from chemvector.evidence_carrier_sample_threshold_predictor import (
    SAMPLE_THRESHOLD_BASELINE_TAU,
    SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM,
    SAMPLE_THRESHOLD_PAIR_TOKEN_DIM,
)


class PairConditionedResidualModelTest(unittest.TestCase):
    def test_parameter_count_and_zero_initialized_output_are_exact_baseline(self) -> None:
        torch.manual_seed(1)
        model = EvidenceCarrierPairConditionedResidual()
        final = model.residual_head[-1]

        self.assertEqual(model.parameter_count(), PAIR_RESIDUAL_PARAMETER_COUNT)
        self.assertEqual(PAIR_RESIDUAL_PARAMETER_COUNT, 2_841)
        torch.testing.assert_close(final.weight, torch.zeros_like(final.weight))
        torch.testing.assert_close(final.bias, torch.zeros_like(final.bias))

        pair_tokens = torch.randn(2, 4, SAMPLE_THRESHOLD_PAIR_TOKEN_DIM)
        pair_mask = torch.tensor(
            ((True, True, False, False), (True, True, True, True))
        )
        global_features = torch.rand(
            2, SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM
        )
        output = model(pair_tokens, pair_mask, global_features)
        raw_gain = pair_tokens[..., -1]

        torch.testing.assert_close(
            output["residual"], torch.zeros_like(output["residual"])
        )
        torch.testing.assert_close(output["corrected_gain"], raw_gain)
        expected_margin = raw_gain - SAMPLE_THRESHOLD_BASELINE_TAU
        expected_margin = torch.where(
            pair_mask, expected_margin, torch.zeros_like(expected_margin)
        )
        torch.testing.assert_close(output["margin"], expected_margin)

    def test_pair_permutation_is_equivariant_with_nonzero_residual_head(self) -> None:
        torch.manual_seed(2)
        model = EvidenceCarrierPairConditionedResidual()
        final = model.residual_head[-1]
        with torch.no_grad():
            final.weight.copy_(torch.randn_like(final.weight))
            final.bias.copy_(torch.randn_like(final.bias))

        pair_tokens = torch.randn(2, 5, SAMPLE_THRESHOLD_PAIR_TOKEN_DIM)
        pair_mask = torch.tensor(
            ((True, True, True, False, False), (True, True, True, True, True))
        )
        global_features = torch.rand(
            2, SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM
        )
        permutation = torch.tensor((3, 1, 4, 0, 2))
        inverse = torch.argsort(permutation)

        original = model(pair_tokens, pair_mask, global_features)
        permuted = model(
            pair_tokens[:, permutation],
            pair_mask[:, permutation],
            global_features,
        )

        for name in ("residual", "corrected_gain", "margin"):
            torch.testing.assert_close(
                permuted[name][:, inverse], original[name], atol=1e-6, rtol=1e-6
            )
        for name in ("image_context", "pooled_pair_mean", "pooled_pair_max"):
            torch.testing.assert_close(
                permuted[name], original[name], atol=1e-6, rtol=1e-6
            )

    def test_empty_pair_set_has_zero_pooling_and_stable_shapes(self) -> None:
        model = EvidenceCarrierPairConditionedResidual()
        pair_tokens = torch.empty(3, 0, SAMPLE_THRESHOLD_PAIR_TOKEN_DIM)
        pair_mask = torch.empty(3, 0, dtype=torch.bool)
        global_features = torch.tensor(
            ((0.0, 0.0, 0.0, 0.0),) * 3, dtype=torch.float32
        )

        output = model(pair_tokens, pair_mask, global_features)

        self.assertEqual(output["residual"].shape, (3, 0))
        self.assertEqual(output["corrected_gain"].shape, (3, 0))
        self.assertEqual(output["margin"].shape, (3, 0))
        self.assertEqual(output["image_context"].shape, (3, 28))
        torch.testing.assert_close(
            output["pooled_pair_mean"], torch.zeros(3, 12)
        )
        torch.testing.assert_close(
            output["pooled_pair_max"], torch.zeros(3, 12)
        )
        torch.testing.assert_close(output["image_context"], torch.zeros(3, 28))

    def test_forward_contract_has_no_target_aware_argument(self) -> None:
        parameters = tuple(
            inspect.signature(EvidenceCarrierPairConditionedResidual.forward)
            .parameters
        )
        self.assertEqual(
            parameters,
            ("self", "pair_tokens", "pair_mask", "global_count_features"),
        )

        model = EvidenceCarrierPairConditionedResidual()
        with self.assertRaises(TypeError):
            model(
                torch.empty(1, 0, SAMPLE_THRESHOLD_PAIR_TOKEN_DIM),
                torch.empty(1, 0, dtype=torch.bool),
                torch.zeros(1, SAMPLE_THRESHOLD_GLOBAL_COUNT_FEATURE_DIM),
                pair_labels=torch.empty(1, 0, dtype=torch.long),
            )


class PairConditionedResidualLossTest(unittest.TestCase):
    def test_fixed_loss_matches_manual_per_image_terms(self) -> None:
        margins = torch.tensor(((2.0, -1.0, 0.5),), dtype=torch.float32)
        residuals = torch.tensor(((0.2, -0.4, 0.6),), dtype=torch.float32)
        pair_indices = torch.tensor((((0, 1), (0, 2), (3, 4)),))
        pair_labels = torch.tensor(((1, 0, 0),))
        pair_mask = torch.ones(1, 3, dtype=torch.bool)

        result = pair_conditioned_residual_loss(
            margins,
            residuals,
            pair_indices,
            pair_labels,
            pair_mask,
        )

        positive = F.softplus(torch.tensor(-2.0))
        negative = torch.stack(
            (F.softplus(torch.tensor(-1.0)), F.softplus(torch.tensor(0.5)))
        ).mean()
        balanced = torch.stack((positive, negative)).mean()
        ranking = F.softplus(torch.tensor(1.0 - 1.0 - 2.0))
        classification_and_ranking = balanced + 0.5 * ranking
        hard = negative
        residual_l2 = torch.tensor((0.2**2 + 0.4**2 + 0.6**2) / 3.0)
        expected = classification_and_ranking + 0.5 * hard + 0.01 * residual_l2

        torch.testing.assert_close(result["balanced_bce"], balanced)
        torch.testing.assert_close(result["ranking_loss"], ranking)
        torch.testing.assert_close(
            result["classification_and_ranking_loss"],
            classification_and_ranking,
        )
        torch.testing.assert_close(result["hard_negative_loss"], hard)
        torch.testing.assert_close(result["residual_l2_loss"], residual_l2)
        torch.testing.assert_close(result["loss"], expected)
        self.assertEqual(result["images_with_hard_negatives"], 1)
        self.assertEqual(result["ranking_pair_count"], 1)

    def test_empty_active_set_returns_differentiable_zero(self) -> None:
        margins = torch.empty(2, 0, requires_grad=True)
        residuals = torch.empty(2, 0, requires_grad=True)
        pair_indices = torch.empty(2, 0, 2, dtype=torch.long)
        pair_labels = torch.empty(2, 0, dtype=torch.long)
        pair_mask = torch.empty(2, 0, dtype=torch.bool)

        result = pair_conditioned_residual_loss(
            margins,
            residuals,
            pair_indices,
            pair_labels,
            pair_mask,
        )

        self.assertTrue(
            math.isclose(float(result["loss"].detach()), 0.0, abs_tol=0.0)
        )
        result["loss"].backward()
        self.assertIsNotNone(margins.grad)
        self.assertIsNotNone(residuals.grad)

    def test_residual_l2_is_reduced_per_image_before_batch_mean(self) -> None:
        margins = torch.zeros((2, 3), dtype=torch.float32)
        residuals = torch.tensor(
            ((1.0, 3.0, 99.0), (2.0, 4.0, 6.0)), dtype=torch.float32
        )
        pair_indices = torch.tensor(
            (
                ((0, 1), (0, 2), (-1, -1)),
                ((0, 1), (0, 2), (1, 2)),
            ),
            dtype=torch.long,
        )
        pair_labels = torch.tensor(((1, 0, -1), (1, 0, 0)), dtype=torch.long)
        pair_mask = torch.tensor(((True, True, False), (True, True, True)))

        result = pair_conditioned_residual_loss(
            margins,
            residuals,
            pair_indices,
            pair_labels,
            pair_mask,
        )

        first = torch.tensor((1.0**2 + 3.0**2) / 2.0)
        second = torch.tensor((2.0**2 + 4.0**2 + 6.0**2) / 3.0)
        torch.testing.assert_close(
            result["residual_l2_loss"], torch.stack((first, second)).mean()
        )
        self.assertEqual(result["images_with_residuals"], 2)


if __name__ == "__main__":
    unittest.main()
