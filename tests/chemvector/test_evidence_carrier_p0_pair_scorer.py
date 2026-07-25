from __future__ import annotations

from unittest.mock import patch
import unittest

import torch
from torch import nn

from chemvector.evidence_carrier_p0_pair_scorer import (
    P0_PAIR_CONSISTENCY_FEATURE_DIM,
    P0_PAIR_SCORER_PARAMETER_COUNT,
    EvidenceCarrierP0PairScorer,
    extract_frozen_p0_anchor_evidence,
    p0_decoded_pair_consistency_features,
)


def pair_features() -> torch.Tensor:
    return torch.tensor(
        [[
            [1.0, 3.0, 40.0, 1.0, 0.95, 1.0, 0.7, 0.95],
            [2.0, 6.0, 20.0, 0.8, 0.70, 0.6, 0.4, 0.3],
            [0.2, 2.8, 80.0, 1.0, 0.98, 1.0, 0.8, 0.99],
        ]],
        dtype=torch.float32,
    )


def fitted_model() -> EvidenceCarrierP0PairScorer:
    torch.manual_seed(20260722)
    model = EvidenceCarrierP0PairScorer()
    mask = torch.ones((1, 3), dtype=torch.bool)
    model.fit_feature_normalization(
        pair_features(),
        torch.tensor(
            [[
                [0.0] * P0_PAIR_CONSISTENCY_FEATURE_DIM,
                [1.0] * P0_PAIR_CONSISTENCY_FEATURE_DIM,
                [2.0] * P0_PAIR_CONSISTENCY_FEATURE_DIM,
            ]],
            dtype=torch.float32,
        ),
        mask,
        source_partition="train-fit",
    )
    return model


class EvidenceCarrierP0PairScorerTest(unittest.TestCase):
    def test_decoded_consistency_is_symmetric_and_zero_for_identical_lines(self) -> None:
        decoded = torch.tensor(
            [[
                [[10.0, 50.0], [90.0, 50.0]],
                [[10.0, 50.0], [90.0, 50.0]],
                [[10.0, 55.0], [90.0, 55.0]],
            ]]
        )
        pairs = torch.tensor([[[0, 1], [0, 2]]])
        centerlines = torch.tensor(
            [[
                [[10.0, 50.0], [90.0, 50.0]],
                [[10.0, 52.5], [90.0, 52.5]],
            ]]
        )
        mask = torch.ones((1, 2), dtype=torch.bool)
        size = torch.tensor([[100.0, 100.0]])

        forward = p0_decoded_pair_consistency_features(
            decoded, pairs, centerlines, mask, size
        )
        reversed_members = p0_decoded_pair_consistency_features(
            decoded, pairs.flip(-1), centerlines, mask, size
        )

        torch.testing.assert_close(forward, reversed_members, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            forward[0, 0],
            torch.zeros(P0_PAIR_CONSISTENCY_FEATURE_DIM),
            atol=1e-6,
            rtol=0.0,
        )
        self.assertGreater(float(forward[0, 1, 1]), 0.0)
        self.assertGreater(float(forward[0, 1, 3]), 0.0)

    def test_model_is_symmetric_and_pair_permutation_equivariant(self) -> None:
        model = fitted_model().eval()
        generator = torch.Generator().manual_seed(7)
        anchors = torch.randn((1, 4, 128), generator=generator)
        pairs = torch.tensor([[[0, 1], [0, 2], [2, 3]]])
        consistency = torch.tensor(
            [[
                [0.0] * P0_PAIR_CONSISTENCY_FEATURE_DIM,
                [1.0] * P0_PAIR_CONSISTENCY_FEATURE_DIM,
                [2.0] * P0_PAIR_CONSISTENCY_FEATURE_DIM,
            ]]
        )
        mask = torch.ones((1, 3), dtype=torch.bool)
        permutation = torch.tensor([2, 0, 1])
        with torch.no_grad():
            baseline = model(
                anchors, pairs, pair_features(), consistency, mask
            )["pair_gains"]
            reversed_members = model(
                anchors, pairs.flip(-1), pair_features(), consistency, mask
            )["pair_gains"]
            permuted = model(
                anchors,
                pairs[:, permutation],
                pair_features()[:, permutation],
                consistency[:, permutation],
                mask[:, permutation],
            )["pair_gains"]

        torch.testing.assert_close(baseline, reversed_members, atol=0.0, rtol=0.0)
        torch.testing.assert_close(permuted, baseline[:, permutation])

    def test_default_parameter_contract_and_state_roundtrip(self) -> None:
        model = fitted_model().eval()
        self.assertEqual(model.parameter_count(), P0_PAIR_SCORER_PARAMETER_COUNT)
        self.assertEqual(P0_PAIR_SCORER_PARAMETER_COUNT, 11873)
        clone = EvidenceCarrierP0PairScorer().eval()
        clone.load_state_dict(model.state_dict())
        torch.testing.assert_close(clone.descriptor_mean, model.descriptor_mean)
        torch.testing.assert_close(clone.descriptor_scale, model.descriptor_scale)
        self.assertTrue(bool(clone.normalization_fitted))

    def test_padding_cannot_change_valid_pair_gain(self) -> None:
        model = fitted_model().eval()
        anchors = torch.zeros((1, 3, 128))
        pairs = torch.tensor([[[0, 1], [1, 2], [-1, -1]]])
        mask = torch.tensor([[True, True, False]])
        consistency = torch.zeros((1, 3, P0_PAIR_CONSISTENCY_FEATURE_DIM))
        changed_features = pair_features().clone()
        changed_features[0, 2] = float("nan")
        changed_consistency = consistency.clone()
        changed_consistency[0, 2] = float("nan")
        with torch.no_grad():
            baseline = model(
                anchors, pairs, pair_features(), consistency, mask
            )["pair_gains"]
            changed = model(
                anchors, pairs, changed_features, changed_consistency, mask
            )["pair_gains"]
        torch.testing.assert_close(baseline, changed)
        self.assertEqual(float(changed[0, 2]), 0.0)

    def test_frozen_p0_extractor_rejects_trainable_parameters(self) -> None:
        class FakeP0(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(1.0))

            def forward(self, pixel_values, anchors, mask, **kwargs):
                return {
                    "carrier_features": torch.zeros((*mask.shape, 128)),
                    "marker": anchors,
                }

        model = FakeP0()
        pixels = torch.zeros((1, 3, 10, 10))
        anchors = torch.tensor([[[[0.1, 0.5], [0.9, 0.5]]]])
        mask = torch.ones((1, 1), dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "must be frozen"):
            extract_frozen_p0_anchor_evidence(model, pixels, anchors, mask)

        model.requires_grad_(False)
        with patch(
            "chemvector.evidence_carrier_p0_pair_scorer."
            "decode_evidence_carrier_extent",
            return_value={"refined_segments_pixels": anchors * 9.0},
        ):
            result = extract_frozen_p0_anchor_evidence(
                model, pixels, anchors, mask
            )
        self.assertEqual(result["anchor_features"].shape, (1, 1, 128))
        self.assertEqual(result["decoded_segments_pixels"].shape, (1, 1, 2, 2))


if __name__ == "__main__":
    unittest.main()
