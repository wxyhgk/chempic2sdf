import unittest

import torch

from chemvector.evidence_carrier_pair_local_spatial_residual import (
    PAIR_LOCAL_SPATIAL_PARAMETER_COUNT,
    EvidenceCarrierPairLocalSpatialResidual,
)


class PairLocalSpatialResidualTest(unittest.TestCase):
    def test_parameter_and_zero_residual_contract(self) -> None:
        model = EvidenceCarrierPairLocalSpatialResidual()
        self.assertEqual(model.parameter_count(), PAIR_LOCAL_SPATIAL_PARAMETER_COUNT)
        tokens = torch.randn(1, 2, 21)
        crops = torch.rand(1, 2, 2, 24, 48)
        pairs = torch.tensor([[[0, 1], [1, 2]]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        counts = torch.zeros(1, 4)
        output = model(tokens, crops, pairs, mask, counts)
        torch.testing.assert_close(output["residual"], torch.zeros(1, 2))

    def test_spatial_encoder_distinguishes_vertical_position(self) -> None:
        torch.manual_seed(11)
        model = EvidenceCarrierPairLocalSpatialResidual().eval()
        top = torch.zeros(1, 2, 24, 48)
        bottom = torch.zeros_like(top)
        top[:, 0, 3:6, 8:40] = 1.0
        bottom[:, 0, 18:21, 8:40] = 1.0
        with torch.no_grad():
            top_feature = model.local_encoder(top)
            bottom_feature = model.local_encoder(bottom)
        self.assertGreater(float((top_feature - bottom_feature).abs().sum()), 1e-4)


if __name__ == "__main__":
    unittest.main()
