"""Tests for the P1-X multiscale peak set ranker."""

from __future__ import annotations

import unittest

import torch

from chemvector.components.atoms.g3_multiscale_peak_set_ranker import (
    G3_MULTISCALE_CANDIDATES,
    G3MultiscalePeakSetRanker,
    build_dynamic_multiscale_targets,
    select_known_count_multiscale,
)


def candidates(batch: int = 1) -> dict[str, torch.Tensor]:
    return {
        "candidate_ids": torch.arange(G3_MULTISCALE_CANDIDATES)[None].expand(batch, -1).clone(),
        "candidate_features": torch.randn(batch, G3_MULTISCALE_CANDIDATES, 32),
        "candidate_anchors": torch.rand(batch, G3_MULTISCALE_CANDIDATES, 2),
        "raw_scores": torch.rand(batch, G3_MULTISCALE_CANDIDATES),
        "normalized_scores": torch.randn(batch, G3_MULTISCALE_CANDIDATES),
        "scale_ids": torch.tensor([[0] * 96 + [1] * 32]).expand(batch, -1).clone(),
        "scale_ranks": torch.tensor(
            [index / 95 for index in range(96)]
            + [index / 31 for index in range(32)]
        )[None].expand(batch, -1).clone(),
    }


def permute(values: dict[str, torch.Tensor], order: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: value[:, order] for name, value in values.items()}


class G3MultiscalePeakSetRankerTests(unittest.TestCase):
    def test_zero_start_and_permutation_equivariance(self) -> None:
        torch.manual_seed(11)
        values = candidates()
        ranker = G3MultiscalePeakSetRanker().eval()
        initial = ranker(values)
        torch.testing.assert_close(
            initial["ranking_logits"], values["normalized_scores"], rtol=0, atol=0
        )
        with torch.no_grad():
            ranker.residual_head.weight.normal_(std=0.1)
        order = torch.randperm(128)
        inverse = order.argsort()
        direct = ranker(values)["ranking_logits"]
        shuffled = ranker(permute(values, order))["ranking_logits"]
        torch.testing.assert_close(
            shuffled[:, inverse], direct, rtol=1e-5, atol=1e-5
        )

    def test_dynamic_match_uses_one_candidate_per_gt_and_keeps_cardinality(self) -> None:
        values = candidates()
        values["candidate_anchors"][0, :5] = torch.tensor(
            [
                [0.100, 0.100],
                [0.101, 0.100],
                [0.200, 0.200],
                [0.201, 0.200],
                [0.800, 0.800],
            ]
        )
        values["normalized_scores"].fill_(-10.0)
        values["normalized_scores"][0, 1] = 3.0
        values["normalized_scores"][0, 3] = 2.0
        ranked = {**values, "ranking_logits": values["normalized_scores"]}
        targets = torch.tensor([[[0.100, 0.100], [0.200, 0.200]]])
        mask = torch.tensor([[True, True]])

        dynamic = build_dynamic_multiscale_targets(ranked, targets, mask)

        self.assertEqual(int(dynamic["positive_mask"].sum()), 2)
        self.assertTrue(bool(dynamic["positive_mask"][0, 1]))
        self.assertTrue(bool(dynamic["positive_mask"][0, 3]))
        self.assertEqual(
            sorted(dynamic["matched_target_indices"][dynamic["positive_mask"]].tolist()),
            [0, 1],
        )
        self.assertGreaterEqual(int(dynamic["alternative_candidates"][0]), 2)

    def test_dynamic_match_rejects_uncovered_gt(self) -> None:
        values = candidates()
        values["candidate_anchors"].fill_(0.1)
        ranked = {**values, "ranking_logits": values["normalized_scores"]}
        with self.assertRaisesRegex(RuntimeError, "cannot cover every GT"):
            build_dynamic_multiscale_targets(
                ranked,
                torch.tensor([[[0.9, 0.9]]]),
                torch.tensor([[True]]),
            )

    def test_known_count_selection_uses_scores_then_stable_identity(self) -> None:
        values = candidates()
        values["normalized_scores"].zero_()
        values["normalized_scores"][0, 9] = 2.0
        ranked = {**values, "ranking_logits": values["normalized_scores"]}

        selected = select_known_count_multiscale(ranked, torch.tensor([3]))

        self.assertEqual(int(selected["selected_mask"].sum()), 3)
        self.assertTrue(bool(selected["selected_mask"][0, 9]))
        self.assertTrue(bool(selected["selected_mask"][0, 0]))
        self.assertTrue(bool(selected["selected_mask"][0, 1]))


if __name__ == "__main__":
    unittest.main()
