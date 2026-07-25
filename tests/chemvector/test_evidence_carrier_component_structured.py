import unittest

import torch

from chemvector.evidence_carrier_component_structured import (
    COMPONENT_STRUCTURED_PARAMETER_COUNT,
    EvidenceCarrierComponentStructured,
    shared_anchor_pair_context,
)


class ComponentStructuredTest(unittest.TestCase):
    def test_shared_anchor_context_reaches_competing_edges(self) -> None:
        hidden = torch.zeros(1, 3, 64)
        hidden[0, 0, 0] = 1.0
        pairs = torch.tensor([[[0, 1], [1, 2], [3, 4]]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        context = shared_anchor_pair_context(hidden, pairs, mask)
        self.assertGreater(float(context[0, 1].abs().sum()), 0.0)
        self.assertEqual(float(context[0, 2].abs().sum()), 0.0)

    def test_initial_energy_equals_frozen_baseline(self) -> None:
        model = EvidenceCarrierComponentStructured().eval()
        self.assertEqual(model.parameter_count(), COMPONENT_STRUCTURED_PARAMETER_COUNT)
        tokens = torch.zeros(1, 2, 21)
        crops = torch.rand(1, 2, 2, 24, 48)
        pairs = torch.tensor([[[0, 1], [1, 2]]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        baseline = torch.tensor([[2.0, 1.5]])
        output = model(
            tokens,
            crops,
            pairs,
            mask,
            torch.zeros(1, 4),
            baseline,
            torch.full((1, 2), 3.0),
            torch.full((1, 2), 2.0),
        )
        torch.testing.assert_close(output["edge_residual"], torch.zeros_like(baseline))
        torch.testing.assert_close(
            output["edge_energy"], baseline - float(model.tau0)
        )


if __name__ == "__main__":
    unittest.main()
