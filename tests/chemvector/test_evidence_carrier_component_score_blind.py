import unittest

import torch

from chemvector.evidence_carrier_component_score_blind import (
    COMPONENT_SCORE_BLIND_PARAMETER_COUNT,
    EvidenceCarrierComponentScoreBlind,
)


class ComponentScoreBlindTest(unittest.TestCase):
    def test_residual_is_invariant_to_all_score_inputs(self) -> None:
        torch.manual_seed(7)
        model = EvidenceCarrierComponentScoreBlind().eval()
        self.assertEqual(model.parameter_count(), COMPONENT_SCORE_BLIND_PARAMETER_COUNT)
        tokens = torch.rand(1, 2, 21)
        crops = torch.rand(1, 2, 2, 24, 48)
        pairs = torch.tensor([[[0, 1], [1, 2]]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        counts = torch.full((1, 2), 3.0)
        first = model(
            tokens,
            crops,
            pairs,
            mask,
            torch.zeros(1, 4),
            torch.tensor([[0.8, 1.2]]),
            counts,
            torch.full((1, 2), 2.0),
        )
        changed_tokens = tokens.clone()
        changed_tokens[..., -1] = torch.tensor([[-100.0, 100.0]])
        second = model(
            changed_tokens,
            crops,
            pairs,
            mask,
            torch.zeros(1, 4),
            torch.tensor([[100.0, -100.0]]),
            counts,
            torch.full((1, 2), 2.0),
        )
        torch.testing.assert_close(first["edge_residual"], second["edge_residual"])
        self.assertFalse(torch.equal(first["edge_energy"], second["edge_energy"]))


if __name__ == "__main__":
    unittest.main()
