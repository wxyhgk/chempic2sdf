import unittest

import torch

from chemvector.endpoint_evidence import endpoint_ray_features


class EndpointEvidenceTests(unittest.TestCase):
    def test_pass_through_midpoint_differs_from_true_line_end(self) -> None:
        pixels = torch.ones(1, 3, 64, 64)
        pixels[:, :, 31:34, 16:49] = -1.0
        points = torch.tensor([[[0.5, 0.5], [0.75, 0.5]]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        features = endpoint_ray_features(
            pixels, points, mask, torch.tensor([[0, 1]])
        )[0, 0]
        begin_in, begin_out, end_in, end_out = features.tolist()
        self.assertGreater(begin_in, 0.9)
        self.assertGreater(begin_out, 0.9)
        self.assertGreater(end_in, 0.9)
        self.assertLess(end_out, 0.1)

    def test_invalid_pairs_are_zeroed(self) -> None:
        features = endpoint_ray_features(
            torch.ones(1, 3, 32, 32),
            torch.rand(1, 3, 2),
            torch.tensor([[True, True, False]]),
            torch.tensor([[0, 1], [0, 2]]),
        )
        self.assertTrue(torch.equal(features[0, 1], torch.zeros(4)))


if __name__ == "__main__":
    unittest.main()
