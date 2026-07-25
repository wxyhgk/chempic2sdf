import unittest

import torch

from chemvector.evidence_carrier_pair_junction_veto import (
    EvidenceCarrierPairJunctionVeto,
    junction_negative_mask,
    pair_junction_veto_loss,
)


class PairJunctionVetoTest(unittest.TestCase):
    def test_junction_mask_selects_negative_with_single_group(self) -> None:
        types = torch.tensor([[[1, 3], [2, 2], [1, 1], [2, 3]]])
        labels = torch.tensor([[0, 0, 1, 0]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        self.assertEqual(
            junction_negative_mask(types, labels, mask).tolist(),
            [[True, False, False, False]],
        )

    def test_loss_adds_junction_emphasis(self) -> None:
        logits = torch.tensor([[-2.0, -2.0]])
        gains = torch.tensor([[2.0, 2.0]])
        labels = torch.tensor([[0, 0]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        types = torch.tensor([[[1, 3], [2, 2]]])
        loss = pair_junction_veto_loss(
            logits, gains, labels, mask, types, tau0=0.7
        )
        self.assertEqual(loss["junction_negative_pair_count"], 1)
        self.assertGreater(float(loss["junction_negative_loss"]), 0.0)
        self.assertGreater(float(loss["worst_junction_negative_loss"]), 0.0)

    def test_model_does_not_accept_group_type_input(self) -> None:
        model = EvidenceCarrierPairJunctionVeto()
        self.assertFalse(model.config_dict()["group_type_in_forward"])


if __name__ == "__main__":
    unittest.main()
