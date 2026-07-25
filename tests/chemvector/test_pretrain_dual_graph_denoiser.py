import unittest

import torch

from chemvector.pretrain_dual_graph_denoiser import (
    best_deployed_threshold,
    pass_through_auxiliary_loss,
    validation_key,
)


class DualGraphDenoiserTrainingTests(unittest.TestCase):
    def test_pass_through_auxiliary_ignores_unlabeled_background(self) -> None:
        logits = torch.tensor([[0.0, 0.0, 20.0]], requires_grad=True)
        loss = pass_through_auxiliary_loss(
            logits,
            torch.tensor([[False, True, False]]),
            torch.tensor([[0, -1, -1]]),
            torch.ones(1, 3, dtype=torch.bool),
        )
        loss.backward()
        self.assertGreater(float(logits.grad[0, 0]), 0.0)
        self.assertLess(float(logits.grad[0, 1]), 0.0)
        self.assertEqual(float(logits.grad[0, 2]), 0.0)

    def test_validation_uses_best_deployed_threshold(self) -> None:
        result = {
            "real_thresholds": {
                "0.50": {"exact_rate": 0.4, "count_exact_rate": 0.5, "f1": 0.9},
                "0.70": {"exact_rate": 0.6, "count_exact_rate": 0.7, "f1": 0.8},
            }
        }
        self.assertEqual(best_deployed_threshold(result)["threshold"], 0.7)
        self.assertEqual(validation_key(result), (0.6, 0.7, 0.8))


if __name__ == "__main__":
    unittest.main()
