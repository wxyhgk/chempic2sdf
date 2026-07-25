import unittest

import torch

from chemvector.equivalence_bond_selector import (
    attach_equivalence_labels,
    build_equivalence_labels,
    equivalence_bond_selector_loss,
    equivalence_selection_metrics,
    normalized_logmeanexp,
)


def base_example() -> dict:
    return {
        "key": "sample",
        "mask": torch.tensor((True, True, True, True, True, False)),
        "pairs": torch.tensor(
            (
                (0, 1),
                (1, 0),
                (1, 2),
                (2, 3),
                (3, 3),
                (3, 4),
            )
        ),
        "keep_targets": torch.tensor((True, False, True, False, False, False)),
        "expected_pairs": [(0, 1), (1, 2), (3, 4)],
        "expected_display": [1, 2, 3],
        "expected_chemical": [2, 3, 4],
        "count_target": 3,
    }


class EquivalenceBondSelectorTest(unittest.TestCase):
    def test_labels_group_equivalent_pairs_and_count_only_representable_bonds(self) -> None:
        labels = build_equivalence_labels(base_example())

        self.assertEqual(labels["group_ids"].tolist(), [0, 0, 1, 2, 3, -1])
        self.assertEqual(
            labels["group_mask"].tolist(),
            [True, True, True, True, False, False],
        )
        self.assertEqual(
            labels["group_targets"].tolist(),
            [True, True, False, False, False, False],
        )
        self.assertEqual(labels["canonical_targets"].sum().item(), 2)
        self.assertEqual(labels["representable_count"], 2)
        self.assertEqual(labels["gt_count"], 3)
        self.assertEqual(labels["group_pairs"][:4].tolist(), [[0, 1], [1, 2], [2, 3], [3, 3]])

    def test_attaching_labels_preserves_the_deployable_feature_tensor(self) -> None:
        features = torch.randn(6, 7)
        junction = {
            "key": "sample",
            "mask": torch.ones(6, dtype=torch.bool),
            "features": features,
        }
        base = base_example()
        base["mask"] = torch.ones(6, dtype=torch.bool)
        base["keep_targets"][-1] = True

        attached = attach_equivalence_labels(junction, base)

        self.assertIs(attached["features"], features)
        self.assertIs(junction["features"], features)
        self.assertNotIn("group_ids", junction)

    def test_normalized_logmeanexp_does_not_reward_identical_duplicates(self) -> None:
        single = normalized_logmeanexp(torch.tensor((2.75,)), temperature=0.7)
        repeated = normalized_logmeanexp(
            torch.full((9,), 2.75), temperature=0.7
        )

        self.assertTrue(torch.allclose(single, repeated, atol=1e-6, rtol=0.0))

    def test_loss_is_finite_and_nonwinner_is_not_an_ordinary_negative(self) -> None:
        mask = torch.tensor(((True, True, True),))
        group_ids = torch.tensor(((0, 0, 1),))
        group_mask = torch.tensor(((True, True, False),))
        group_targets = torch.tensor(((True, False, False),))
        canonical_targets = torch.tensor(((True, False, False),))
        representable_counts = torch.tensor((1,))
        group_display_targets = torch.zeros((1, 3), dtype=torch.long)
        group_chemical_targets = torch.zeros((1, 3), dtype=torch.long)

        def negative_only_loss(nonwinner_logit: float) -> tuple[torch.Tensor, torch.Tensor]:
            logits = torch.tensor(
                ((0.4, nonwinner_logit, -0.3),), requires_grad=True
            )
            losses = equivalence_bond_selector_loss(
                {"keep_logits": logits},
                mask,
                group_ids,
                group_mask,
                group_targets,
                canonical_targets,
                representable_counts,
                group_display_targets,
                group_chemical_targets,
                positive_weight=0.0,
                negative_weight=1.0,
                exactly_one_weight=0.0,
                representative_weight=0.0,
                cardinality_weight=0.0,
                margin_weight=0.0,
            )
            self.assertTrue(torch.isfinite(losses["loss"]))
            losses["loss"].backward()
            return losses["loss"].detach(), logits.grad.detach()

        low_loss, low_gradient = negative_only_loss(-8.0)
        high_loss, high_gradient = negative_only_loss(8.0)

        self.assertTrue(torch.allclose(low_loss, high_loss, atol=1e-7, rtol=0.0))
        self.assertEqual(low_gradient[0, 1].item(), 0.0)
        self.assertEqual(high_gradient[0, 1].item(), 0.0)
        self.assertGreater(low_gradient[0, 2].item(), 0.0)
        self.assertGreater(high_gradient[0, 2].item(), 0.0)

    def test_candidate_cardinality_penalizes_duplicate_probability_mass(self) -> None:
        common = {
            "mask": torch.tensor(((True, True),)),
            "group_ids": torch.tensor(((0, 0),)),
            "group_mask": torch.tensor(((True, False),)),
            "group_targets": torch.tensor(((True, False),)),
            "canonical_targets": torch.tensor(((True, False),)),
            "representable_counts": torch.tensor((1,)),
            "group_display_targets": torch.zeros((1, 2), dtype=torch.long),
            "group_chemical_targets": torch.zeros((1, 2), dtype=torch.long),
        }

        def cardinality(logits: tuple[float, float], mode: str) -> float:
            losses = equivalence_bond_selector_loss(
                {"keep_logits": torch.tensor((logits,))},
                **common,
                positive_weight=0.0,
                negative_weight=0.0,
                exactly_one_weight=0.0,
                representative_weight=0.0,
                cardinality_weight=1.0,
                margin_weight=0.0,
                cardinality_mode=mode,
            )
            return float(losses["cardinality_loss"])

        candidate_single = cardinality((8.0, -8.0), "candidate")
        candidate_duplicate = cardinality((8.0, 8.0), "candidate")
        group_single = cardinality((8.0, -8.0), "group")
        group_duplicate = cardinality((8.0, 8.0), "group")

        self.assertGreater(candidate_duplicate, candidate_single + 0.4)
        self.assertLess(abs(group_duplicate - group_single), 1e-3)

    def test_group_selection_metrics_count_duplicates_and_negative_groups(self) -> None:
        logits = torch.tensor(
            (
                (2.0, 2.0, 2.0, 2.0, -2.0),
                (2.0, -2.0, -2.0, -2.0, -2.0),
            )
        )
        mask = torch.tensor(
            (
                (True, True, True, True, False),
                (True, True, False, True, False),
            )
        )
        group_ids = torch.tensor(((0, 0, 1, 2, -1), (0, 0, -1, 1, -1)))
        group_mask = torch.tensor(((True, True, True), (True, True, False)))
        group_targets = torch.tensor(((True, True, False), (True, False, False)))

        metrics = equivalence_selection_metrics(
            logits,
            mask,
            group_ids,
            group_mask,
            group_targets,
            threshold=0.5,
        )

        self.assertEqual(metrics["samples"], 2)
        self.assertEqual(metrics["positive_groups"], 3)
        self.assertAlmostEqual(metrics["group_precision"], 0.75)
        self.assertAlmostEqual(metrics["group_recall"], 1.0)
        self.assertAlmostEqual(metrics["exactly_one_rate"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["miss_rate"], 0.0)
        self.assertAlmostEqual(metrics["multi_selected_group_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["group_set_exact_rate"], 0.5)
        self.assertAlmostEqual(metrics["strict_group_selection_exact_rate"], 0.5)
        self.assertAlmostEqual(metrics["same_group_extra_candidates_per_image"], 0.5)
        self.assertAlmostEqual(metrics["selected_negative_groups_per_image"], 0.5)
        self.assertAlmostEqual(metrics["selected_negative_candidates_per_image"], 0.5)


if __name__ == "__main__":
    unittest.main()
