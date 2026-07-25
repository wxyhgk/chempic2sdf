import unittest

import torch

from chemvector.evidence_carrier_pair_local_residual import (
    PAIR_LOCAL_RESIDUAL_PARAMETER_COUNT,
    EvidenceCarrierPairLocalResidual,
)


class PairLocalResidualTest(unittest.TestCase):
    def _inputs(self):
        torch.manual_seed(7)
        tokens = torch.randn(2, 4, 21)
        crops = torch.rand(2, 4, 2, 24, 48)
        pairs = torch.tensor(
            [[[0, 1], [1, 2], [2, 3], [-1, -1]], [[0, 2], [0, 1], [-1, -1], [-1, -1]]]
        )
        mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
        counts = torch.rand(2, 4)
        return tokens, crops, pairs, mask, counts

    def test_parameter_contract_and_zero_initialized_residual(self) -> None:
        model = EvidenceCarrierPairLocalResidual()
        self.assertEqual(model.parameter_count(), PAIR_LOCAL_RESIDUAL_PARAMETER_COUNT)
        output = model(*self._inputs())
        mask = self._inputs()[3]
        self.assertEqual(output["endpoint_context"].shape, (2, 4, 66))
        self.assertEqual(output["image_context"].shape, (2, 36))
        torch.testing.assert_close(output["residual"], torch.zeros_like(output["residual"]))
        tokens = self._inputs()[0]
        torch.testing.assert_close(output["corrected_gain"][mask], tokens[..., -1][mask])

    def test_pair_order_permutation_is_equivariant(self) -> None:
        model = EvidenceCarrierPairLocalResidual().eval()
        tokens, crops, pairs, mask, counts = self._inputs()
        permutation = torch.tensor([2, 0, 1, 3])
        original = model(tokens, crops, pairs, mask, counts)
        changed = model(
            tokens[:, permutation],
            crops[:, permutation],
            pairs[:, permutation],
            mask[:, permutation],
            counts,
        )
        torch.testing.assert_close(
            changed["residual"], original["residual"][:, permutation]
        )
        torch.testing.assert_close(
            changed["endpoint_context"], original["endpoint_context"][:, permutation]
        )

    def test_local_raster_receives_gradient_after_output_layer_moves(self) -> None:
        model = EvidenceCarrierPairLocalResidual()
        with torch.no_grad():
            model.residual_head[-1].weight.fill_(0.1)
        tokens, crops, pairs, mask, counts = self._inputs()
        crops.requires_grad_(True)
        model(tokens, crops, pairs, mask, counts)["residual"].sum().backward()
        self.assertIsNotNone(crops.grad)
        self.assertGreater(float(crops.grad.abs().sum()), 0.0)

    def test_rejects_invalid_crop_range(self) -> None:
        model = EvidenceCarrierPairLocalResidual()
        tokens, crops, pairs, mask, counts = self._inputs()
        crops[0, 0, 0, 0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, r"in \[0,1\]"):
            model(tokens, crops, pairs, mask, counts)


if __name__ == "__main__":
    unittest.main()
