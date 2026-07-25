from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from chemvector.evidence_carrier_pair_scorer import (
    PAIR_SCORER_PARAMETER_COUNT,
    EvidenceCarrierPairScorer,
    evidence_carrier_pair_scorer_loss,
    transform_evidence_carrier_pair_features,
)


def features() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 3.0, 40.0, 1.0, 0.95, 1.0, 0.70, 0.95],
            [2.0, 6.0, 20.0, 0.8, 0.70, 0.6, 0.45, 0.30],
            [0.2, 2.8, 80.0, 1.0, 0.98, 1.0, 0.75, 0.98],
            [3.0, 7.0, 12.0, 0.5, 0.40, 0.3, 0.20, 0.10],
        ],
        dtype=torch.float32,
    )


def fitted_model() -> EvidenceCarrierPairScorer:
    torch.manual_seed(20260722)
    model = EvidenceCarrierPairScorer()
    values = features().unsqueeze(0)
    model.fit_feature_normalization(
        values,
        torch.ones((1, len(values[0])), dtype=torch.bool),
        source_partition="train-fit",
    )
    return model


class EvidenceCarrierPairScorerTest(unittest.TestCase):
    def test_parameter_count_and_fixed_target_free_transform(self) -> None:
        model = EvidenceCarrierPairScorer()
        self.assertEqual(model.parameter_count(), PAIR_SCORER_PARAMETER_COUNT)
        self.assertEqual(PAIR_SCORER_PARAMETER_COUNT, 833)

        value = features()[:1].unsqueeze(0)
        transformed = transform_evidence_carrier_pair_features(
            value,
            torch.ones((1, 1), dtype=torch.bool),
        )
        self.assertAlmostEqual(float(transformed[0, 0, 0]), 1.0 / 8.0)
        self.assertAlmostEqual(float(transformed[0, 0, 1]), 3.0 / 8.0)
        self.assertGreater(float(transformed[0, 0, 2]), 0.0)
        self.assertLess(float(transformed[0, 0, 2]), 1.0)
        torch.testing.assert_close(transformed[0, 0, 3:], value[0, 0, 3:])

    def test_normalization_is_fit_only_single_use_and_ignores_padding(self) -> None:
        model = EvidenceCarrierPairScorer()
        value = features().unsqueeze(0)
        mask = torch.tensor([[True, True, True, False]])
        value[0, 3] = float("nan")
        report = model.fit_feature_normalization(
            value,
            mask,
            source_partition="train-fit",
        )
        self.assertEqual(report["sample_count"], 3)
        self.assertTrue(bool(model.normalization_fitted))
        self.assertEqual(int(model.normalization_sample_count), 3)
        with self.assertRaisesRegex(RuntimeError, "already fitted"):
            model.fit_feature_normalization(
                value,
                mask,
                source_partition="train-fit",
            )
        with self.assertRaisesRegex(ValueError, "train-fit"):
            EvidenceCarrierPairScorer().fit_feature_normalization(
                value,
                mask,
                source_partition="validation",
            )

    def test_forward_rejects_unfitted_normalization(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not been fitted"):
            EvidenceCarrierPairScorer()(
                features().unsqueeze(0),
                torch.ones((1, 4), dtype=torch.bool),
            )

    def test_pair_permutation_is_equivariant_and_padding_is_inert(self) -> None:
        model = fitted_model().eval()
        value = features().unsqueeze(0)
        mask = torch.tensor([[True, True, True, False]])
        changed = value.clone()
        changed[0, 3] = torch.tensor(
            [float("nan"), -1000.0, -1.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        )
        permutation = torch.tensor([2, 0, 3, 1])
        with torch.no_grad():
            baseline = model(value, mask)
            padded_changed = model(changed, mask)
            permuted = model(value[:, permutation], mask[:, permutation])
        torch.testing.assert_close(
            baseline["pair_gains"][mask],
            padded_changed["pair_gains"][mask],
        )
        torch.testing.assert_close(
            permuted["pair_gains"],
            baseline["pair_gains"][:, permutation],
        )
        self.assertEqual(float(baseline["pair_gains"][0, 3]), 0.0)

    def test_state_dict_roundtrip_preserves_normalization_and_gains(self) -> None:
        model = fitted_model().eval()
        clone = EvidenceCarrierPairScorer().eval()
        clone.load_state_dict(model.state_dict())
        value = features().unsqueeze(0)
        mask = torch.tensor([[True, True, False, True]])
        with torch.no_grad():
            expected = model(value, mask)["pair_gains"]
            actual = clone(value, mask)["pair_gains"]
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
        torch.testing.assert_close(clone.feature_mean, model.feature_mean)
        torch.testing.assert_close(clone.feature_scale, model.feature_scale)
        self.assertEqual(
            int(clone.normalization_sample_count),
            int(model.normalization_sample_count),
        )

    def test_balanced_bce_handles_empty_classes_and_all_padding(self) -> None:
        indices = torch.tensor([[[0, 1], [1, 2]]])
        mask = torch.ones((1, 2), dtype=torch.bool)
        positive = evidence_carrier_pair_scorer_loss(
            torch.tensor([[0.0, 1.0]], requires_grad=True),
            indices,
            torch.ones((1, 2), dtype=torch.long),
            mask,
        )
        expected_positive = F.softplus(-torch.tensor([0.0, 1.0])).mean()
        torch.testing.assert_close(positive["balanced_bce"], expected_positive)
        self.assertEqual(positive["negative_pair_count"], 0)
        self.assertEqual(positive["ranking_pair_count"], 0)

        negative = evidence_carrier_pair_scorer_loss(
            torch.tensor([[0.0, -1.0]], requires_grad=True),
            indices,
            torch.zeros((1, 2), dtype=torch.long),
            mask,
        )
        expected_negative = F.softplus(torch.tensor([0.0, -1.0])).mean()
        torch.testing.assert_close(negative["balanced_bce"], expected_negative)
        self.assertEqual(negative["positive_pair_count"], 0)

        gains = torch.randn((2, 3), requires_grad=True)
        empty = evidence_carrier_pair_scorer_loss(
            gains,
            torch.full((2, 3, 2), -1, dtype=torch.long),
            torch.full((2, 3), -1, dtype=torch.long),
            torch.zeros((2, 3), dtype=torch.bool),
        )
        self.assertEqual(float(empty["loss"].detach()), 0.0)
        self.assertEqual(empty["images_with_pairs"], 0)
        empty["loss"].backward()
        torch.testing.assert_close(gains.grad, torch.zeros_like(gains))

    def test_ranking_uses_only_negative_pairs_sharing_an_anchor(self) -> None:
        gains = torch.tensor([[0.0, 1.0, 100.0]], requires_grad=True)
        indices = torch.tensor([[[0, 1], [0, 2], [3, 4]]])
        labels = torch.tensor([[1, 0, 0]])
        mask = torch.ones((1, 3), dtype=torch.bool)
        result = evidence_carrier_pair_scorer_loss(
            gains,
            indices,
            labels,
            mask,
            classification_weight=0.0,
            ranking_weight=1.0,
            ranking_margin=1.0,
        )
        torch.testing.assert_close(result["ranking_loss"], F.softplus(torch.tensor(2.0)))
        self.assertEqual(result["ranking_pair_count"], 1)
        result["loss"].backward()
        self.assertEqual(float(gains.grad[0, 2]), 0.0)

    def test_loss_is_invariant_to_pair_order_and_endpoint_exchange(self) -> None:
        gains = torch.tensor([[0.5, -0.2, 0.1]])
        indices = torch.tensor([[[0, 1], [0, 2], [2, 3]]])
        labels = torch.tensor([[1, 0, 1]])
        mask = torch.ones((1, 3), dtype=torch.bool)
        baseline = evidence_carrier_pair_scorer_loss(
            gains,
            indices,
            labels,
            mask,
        )
        permutation = torch.tensor([2, 0, 1])
        changed_indices = indices[:, permutation].flip(-1)
        changed = evidence_carrier_pair_scorer_loss(
            gains[:, permutation],
            changed_indices,
            labels[:, permutation],
            mask[:, permutation],
        )
        for name in ("loss", "balanced_bce", "ranking_loss"):
            torch.testing.assert_close(changed[name], baseline[name])
        self.assertEqual(changed["ranking_pair_count"], baseline["ranking_pair_count"])


if __name__ == "__main__":
    unittest.main()
