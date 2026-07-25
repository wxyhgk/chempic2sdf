"""Tests for dual-branch candidate coverage metrics."""

import unittest

import torch

from chemvector.evaluate_dual_vertex_candidates import (
    candidate_matches,
    merge_candidate_pools,
    new_totals,
    select_candidates,
    summarize_totals,
    update_totals,
)


class DualVertexCandidateTests(unittest.TestCase):
    def test_merge_uses_one_text_first_total_capacity(self) -> None:
        text = [{"score": 0.9}, {"score": 0.8}]
        implicit = [{"score": 0.95}, {"score": 0.7}, {"score": 0.6}]
        selected_text, selected_implicit, merged = merge_candidate_pools(
            text, implicit, maximum=3
        )
        self.assertEqual(len(selected_text), 2)
        self.assertEqual(len(selected_implicit), 1)
        self.assertEqual(len(merged), 3)
        self.assertEqual(selected_implicit[0]["score"], 0.95)

    def test_candidate_oracle_ignores_extra_candidates(self) -> None:
        candidates = [
            {"x": 0.1, "y": 0.1, "score": 0.9},
            {"x": 0.5, "y": 0.5, "score": 0.8},
            {"x": 0.9, "y": 0.9, "score": 0.7},
        ]
        targets = torch.tensor([[0.1, 0.1], [0.5, 0.5]])
        self.assertEqual(candidate_matches(candidates, targets, radius_pixels=6.0), 2)

    def test_threshold_and_limit_are_applied_after_score_sort(self) -> None:
        candidates = [
            {"x": 0.0, "y": 0.0, "score": 0.2},
            {"x": 0.0, "y": 0.0, "score": 0.9},
            {"x": 0.0, "y": 0.0, "score": 0.7},
        ]
        selected = select_candidates(candidates, threshold=0.3, maximum=1)
        self.assertEqual([item["score"] for item in selected], [0.9])

    def test_totals_separate_candidate_oracle_and_deployed_exact(self) -> None:
        candidates = [
            {"x": 0.1, "y": 0.1, "score": 0.9},
            {"x": 0.5, "y": 0.5, "score": 0.4},
            {"x": 0.9, "y": 0.9, "score": 0.8},
        ]
        targets = torch.tensor([[0.1, 0.1], [0.5, 0.5]])
        totals = new_totals()
        update_totals(
            totals,
            candidates,
            [candidates[0], candidates[2]],
            targets,
            radius_pixels=6.0,
        )
        summary = summarize_totals(totals)
        self.assertEqual(summary["candidate_oracle_exact"], 1.0)
        self.assertEqual(summary["known_count_exact"], 0.0)
        self.assertEqual(summary["deployed_exact"], 0.0)
        self.assertEqual(summary["candidate_missing_vertices"], 0)
        self.assertEqual(summary["average_candidate_pairs"], 3.0)


if __name__ == "__main__":
    unittest.main()
