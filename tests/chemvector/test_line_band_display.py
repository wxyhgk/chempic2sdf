import unittest

import torch

from chemvector.line_band_display import (
    LineBandDisplayConfig,
    LineBandDisplayRefiner,
)


class LineBandDisplayRefinerTest(unittest.TestCase):
    def test_forward_emits_one_classification_per_confirmed_pair(self) -> None:
        model = LineBandDisplayRefiner(
            LineBandDisplayConfig(
                line_samples=32,
                line_band=17,
                line_band_radius_pixels=8.0,
                hidden_dim=16,
            )
        )
        output = model(
            torch.ones(1, 3, 64, 64),
            torch.tensor([[[0.1, 0.1], [0.9, 0.9], [0.2, 0.8], [0.8, 0.2]]]),
            torch.tensor([[[0, 1], [2, 3]]]),
            torch.tensor([[True, True]]),
        )
        self.assertEqual(output["display_logits"].shape, (2, 4))
        self.assertEqual(output["line_bands"].shape, (2, 1, 17, 32))


if __name__ == "__main__":
    unittest.main()
