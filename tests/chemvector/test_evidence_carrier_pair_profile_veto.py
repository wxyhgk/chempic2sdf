import unittest

import torch

from chemvector.evidence_carrier_pair_profile_veto import (
    PAIR_PROFILE_RAW_DIM,
    PAIR_PROFILE_VETO_PARAMETER_COUNT,
    EvidenceCarrierPairProfileVeto,
    pair_local_profile_features,
)


class PairProfileVetoTest(unittest.TestCase):
    def test_profile_shape_and_one_pixel_gap_sensitivity(self) -> None:
        solid = torch.zeros(1, 1, 2, 24, 48)
        solid[..., 0, 8:16, :] = 1.0
        gap = solid.clone()
        gap[..., 0, 12, :] = 0.0
        first = pair_local_profile_features(solid)
        second = pair_local_profile_features(gap)
        self.assertEqual(first.shape, (1, 1, PAIR_PROFILE_RAW_DIM))
        self.assertGreater(float((first - second).abs().sum()), 1.0)

    def test_left_and_right_junction_positions_remain_distinct(self) -> None:
        left = torch.zeros(1, 1, 2, 24, 48)
        right = torch.zeros_like(left)
        left[..., 0, 4:20, :8] = 1.0
        right[..., 0, 4:20, -8:] = 1.0
        self.assertFalse(
            torch.equal(pair_local_profile_features(left), pair_local_profile_features(right))
        )

    def test_model_parameter_and_initial_veto_contract(self) -> None:
        model = EvidenceCarrierPairProfileVeto().eval()
        self.assertEqual(model.parameter_count(), PAIR_PROFILE_VETO_PARAMETER_COUNT)
        tokens = torch.zeros(1, 2, 21)
        tokens[..., -1] = 2.0
        crops = torch.rand(1, 2, 2, 24, 48)
        pairs = torch.tensor([[[0, 1], [1, 2]]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        output = model(tokens, crops, pairs, mask, torch.zeros(1, 4))
        self.assertFalse(bool(output["veto_mask"].any()))
        torch.testing.assert_close(output["corrected_gain"], tokens[..., -1])


if __name__ == "__main__":
    unittest.main()
