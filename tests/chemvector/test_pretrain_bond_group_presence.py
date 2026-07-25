import unittest

import torch

from chemvector.pretrain_bond_group_presence import (
    _selected_target_indices,
    _topk_valid,
    capacity_checks,
    generalization_checks,
    generalization_checkpoint_eligible,
)


class BondGroupPresenceTrainingTests(unittest.TestCase):
    def test_topk_only_uses_valid_candidates(self) -> None:
        selected = _topk_valid(
            torch.tensor((0.1, 9.0, 0.8, 0.7)),
            torch.tensor((True, False, True, True)),
            2,
        )
        self.assertEqual(selected.tolist(), [2, 3])

    def test_selected_geometry_matching_is_unordered_and_one_to_one(self) -> None:
        endpoints = torch.tensor(
            (
                ((0.8, 0.2), (0.2, 0.2)),
                ((0.2, 0.5), (0.8, 0.5)),
                ((0.2, 0.8), (0.8, 0.8)),
            )
        )
        expected = torch.tensor(
            (
                ((0.2, 0.2), (0.8, 0.2)),
                ((0.2, 0.5), (0.8, 0.5)),
            )
        )
        matched = _selected_target_indices(
            endpoints,
            torch.tensor((0, 1, 2)),
            expected,
            tolerance_pixels=6.0,
        )
        self.assertEqual(matched, {0, 1})

    def test_capacity_checks_require_both_group_metrics(self) -> None:
        metrics = {
            "representable_group_recall": 0.99,
            "representable_exactly_one_rate": 0.98,
        }
        self.assertTrue(capacity_checks(metrics, True)["go"])
        metrics["representable_exactly_one_rate"] = 0.96
        self.assertFalse(capacity_checks(metrics, True)["go"])

    def test_generalization_checks_use_relative_ordinary_retention(self) -> None:
        initial = {
            "collision_holdout": {"candidate_ceiling_recall": 0.44},
            "ordinary_validation": {
                "candidate_ceiling_recall": 0.97,
                "known_gt_count_geometry_recall": 0.95,
                "known_gt_count_image_exact": 0.50,
            },
        }
        best = {
            "collision_holdout": {
                "candidate_ceiling_recall": 0.44,
                "collision_candidate_recall": 0.44,
                "collision_known_count_recall": 0.25,
            },
            "ordinary_validation": {
                "candidate_ceiling_recall": 0.97,
                "known_gt_count_geometry_recall": 0.945,
                "known_gt_count_image_exact": 0.495,
            },
        }
        self.assertTrue(generalization_checks(initial, best, True)["go"])
        self.assertTrue(generalization_checkpoint_eligible(initial, best))
        best["ordinary_validation"]["known_gt_count_geometry_recall"] = 0.944
        self.assertFalse(generalization_checks(initial, best, True)["go"])
        self.assertFalse(generalization_checkpoint_eligible(initial, best))


if __name__ == "__main__":
    unittest.main()
