"""Tests for proposal graph training metrics."""

from __future__ import annotations

import unittest

import torch

from chemvector.pretrain_proposal_graph import (
    balanced_validation_key,
    edge_totals,
    summarize_edges,
    update_edge_metrics,
)


class ProposalGraphTrainingTests(unittest.TestCase):
    def test_balanced_validation_key_penalizes_retention_regression(self) -> None:
        def metrics(vertex_exact: float, count_exact: float, f1: float) -> dict:
            return {
                "best_6px": {
                    "vertex_exact_rate": vertex_exact,
                    "count_exact_rate": count_exact,
                    "f1": f1,
                },
                "best_edges": {"f1": f1},
            }

        balanced = balanced_validation_key(
            metrics(0.2, 0.3, 0.8),
            metrics(0.2, 0.3, 0.8),
        )
        forgotten = balanced_validation_key(
            metrics(0.4, 0.5, 0.9),
            metrics(0.0, 0.1, 0.5),
        )

        self.assertGreater(balanced, forgotten)

    def test_edge_metrics_respect_pair_mask_and_threshold(self) -> None:
        totals = edge_totals()
        update_edge_metrics(
            totals,
            torch.tensor([[2.0, -2.0, 2.0]]),
            torch.tensor([[1.0, 1.0, 0.0]]),
            torch.tensor([[True, True, False]]),
            0.5,
        )

        metrics = summarize_edges(totals)
        self.assertEqual(metrics["true_positives"], 1)
        self.assertEqual(metrics["predictions"], 1)
        self.assertEqual(metrics["targets"], 2)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
