import unittest

import torch

from chemvector.evidence_carrier_pair_local_veto import (
    PAIR_LOCAL_VETO_PARAMETER_COUNT,
    EvidenceCarrierPairLocalVeto,
    pair_local_veto_loss,
)


class PairLocalVetoTest(unittest.TestCase):
    def _inputs(self):
        tokens = torch.zeros(1, 3, 21)
        tokens[0, :, -1] = torch.tensor([2.0, 1.0, -1.0])
        crops = torch.rand(1, 3, 2, 24, 48)
        pairs = torch.tensor([[[0, 1], [1, 2], [2, 3]]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        counts = torch.zeros(1, 4)
        return tokens, crops, pairs, mask, counts

    def test_initial_state_never_changes_scores(self) -> None:
        model = EvidenceCarrierPairLocalVeto().eval()
        self.assertEqual(model.parameter_count(), PAIR_LOCAL_VETO_PARAMETER_COUNT)
        inputs = self._inputs()
        output = model(*inputs)
        self.assertFalse(bool(output["veto_mask"].any()))
        torch.testing.assert_close(output["corrected_gain"], inputs[0][..., -1])

    def test_positive_logit_only_removes_active_candidate(self) -> None:
        model = EvidenceCarrierPairLocalVeto().eval()
        with torch.no_grad():
            model.residual_head[-1].bias.fill_(2.0)
        tokens, crops, pairs, mask, counts = self._inputs()
        output = model(tokens, crops, pairs, mask, counts)
        self.assertEqual(output["veto_mask"].tolist(), [[True, True, False]])
        self.assertLess(float(output["corrected_gain"][0, 0]), float(model.tau0))
        self.assertEqual(float(output["corrected_gain"][0, 2]), -1.0)
        self.assertTrue(bool((output["corrected_gain"] <= tokens[..., -1]).all()))

    def test_loss_penalizes_false_veto_and_missed_hard_negative(self) -> None:
        labels = torch.tensor([[1, 0, 0]])
        gains = torch.tensor([[2.0, 2.0, -1.0]])
        mask = torch.ones(1, 3, dtype=torch.bool)
        unsafe = pair_local_veto_loss(
            torch.tensor([[2.0, -2.0, -2.0]]), gains, labels, mask, tau0=0.7
        )
        safe = pair_local_veto_loss(
            torch.tensor([[-2.0, 2.0, 2.0]]), gains, labels, mask, tau0=0.7
        )
        self.assertGreater(float(unsafe["loss"]), float(safe["loss"]))
        self.assertEqual(unsafe["hard_negative_pair_count"], 1)


if __name__ == "__main__":
    unittest.main()
