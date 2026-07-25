import unittest

import torch

from chemvector.evidence_carrier_pair_neighborhood_veto import (
    PAIR_NEIGHBOR_CONTEXT_DIM,
    PAIR_NEIGHBORHOOD_VETO_PARAMETER_COUNT,
    EvidenceCarrierPairNeighborhoodVeto,
)


class PairNeighborhoodVetoTest(unittest.TestCase):
    def test_model_parameter_shape_and_initial_veto_contract(self) -> None:
        model = EvidenceCarrierPairNeighborhoodVeto().eval()
        self.assertEqual(model.parameter_count(), PAIR_NEIGHBORHOOD_VETO_PARAMETER_COUNT)
        tokens = torch.zeros(1, 2, 21)
        tokens[..., -1] = 2.0
        crops = torch.rand(1, 2, 2, 24, 48)
        pairs = torch.tensor([[[0, 1], [1, 2]]])
        pair_mask = torch.ones(1, 2, dtype=torch.bool)
        neighbors = torch.zeros(1, 2, 12, 8)
        neighbor_mask = torch.zeros(1, 2, 12, dtype=torch.bool)
        neighbors[0, 0, 0, 3] = 1.0
        neighbor_mask[0, 0, 0] = True
        output = model(
            tokens,
            crops,
            pairs,
            pair_mask,
            torch.zeros(1, 4),
            neighbors,
            neighbor_mask,
        )
        self.assertEqual(
            output["neighborhood_context"].shape,
            (1, 2, PAIR_NEIGHBOR_CONTEXT_DIM),
        )
        self.assertEqual(output["neighbor_counts"].tolist(), [[1, 0]])
        self.assertFalse(bool(output["veto_mask"].any()))
        torch.testing.assert_close(output["corrected_gain"], tokens[..., -1])

    def test_rejects_neighbor_on_inactive_pair(self) -> None:
        model = EvidenceCarrierPairNeighborhoodVeto().eval()
        tokens = torch.zeros(1, 1, 21)
        crops = torch.zeros(1, 1, 2, 24, 48)
        pairs = torch.full((1, 1, 2), -1, dtype=torch.long)
        pair_mask = torch.zeros(1, 1, dtype=torch.bool)
        neighbors = torch.zeros(1, 1, 12, 8)
        neighbor_mask = torch.ones(1, 1, 12, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "inactive pairs"):
            model(
                tokens,
                crops,
                pairs,
                pair_mask,
                torch.zeros(1, 4),
                neighbors,
                neighbor_mask,
            )


if __name__ == "__main__":
    unittest.main()
