import unittest

import torch

from chemvector.evaluate_semantic_bond_existence import threshold_metrics


class SemanticBondExistenceEvaluationTest(unittest.TestCase):
    def test_threshold_metrics_uses_whole_sample_exact(self) -> None:
        rows = [
            (torch.tensor([0.9, 0.1]), torch.tensor([True, False])),
            (torch.tensor([0.8, 0.7]), torch.tensor([True, False])),
        ]

        metrics = threshold_metrics(rows, 0.5)

        self.assertEqual(metrics["true_positive"], 2)
        self.assertEqual(metrics["false_positive"], 1)
        self.assertEqual(metrics["false_negative"], 0)
        self.assertEqual(metrics["sample_set_exact"], 0.5)


if __name__ == "__main__":
    unittest.main()
