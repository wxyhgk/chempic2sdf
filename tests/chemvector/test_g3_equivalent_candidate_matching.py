import unittest

import torch

from chemvector.components.atoms.g3_equivalent_candidate_matching import (
    build_score_matched_equivalent_targets,
)


def example():
    candidate_ids = torch.arange(128)[None]
    anchors = torch.full((1, 128, 2), 0.9)
    anchors[0, :4] = torch.tensor(
        [[0.100, 0.100], [0.101, 0.100], [0.500, 0.500], [0.501, 0.500]]
    )
    logits = torch.full((1, 128), -10.0)
    logits[0, :4] = torch.tensor([0.0, 2.0, 0.0, 1.0])
    fixed = torch.zeros(1, 128, dtype=torch.bool)
    fixed[0, [0, 2]] = True
    matched = torch.full((1, 128), -1, dtype=torch.long)
    matched[0, [0, 2]] = torch.tensor([0, 1])
    targets = torch.tensor([[[0.100, 0.100], [0.500, 0.500]]])
    return (
        {
            "candidate_ids": candidate_ids,
            "candidate_anchors": anchors,
            "ranking_logits": logits,
        },
        {"positive_mask": fixed, "matched_target_indices": matched},
        targets,
    )


class G3EquivalentCandidateMatchingTests(unittest.TestCase):
    def test_score_preferred_equivalents_replace_fixed_identities(self) -> None:
        ranked, hybrid, targets = example()

        result = build_score_matched_equivalent_targets(ranked, hybrid, targets)

        selected = torch.where(result["positive_mask"][0])[0].tolist()
        self.assertEqual(selected, [1, 3])
        self.assertEqual(int(result["identity_changes"][0]), 2)
        self.assertEqual(int(result["alternative_candidates"][0]), 2)
        self.assertEqual(int(result["representable_counts"][0]), 2)
        self.assertEqual(
            result["matched_target_indices"][0, selected].tolist(), [0, 1]
        )

    def test_assignment_is_one_to_one_when_groups_overlap(self) -> None:
        ranked, hybrid, targets = example()
        targets[0] = torch.tensor([[0.100, 0.100], [0.106, 0.100]])
        ranked["candidate_anchors"][0, :4] = torch.tensor(
            [[0.100, 0.100], [0.103, 0.100], [0.106, 0.100], [0.700, 0.700]]
        )
        ranked["ranking_logits"][0, :4] = torch.tensor([0.0, 3.0, 2.0, -1.0])

        result = build_score_matched_equivalent_targets(ranked, hybrid, targets)

        selected = torch.where(result["positive_mask"][0])[0]
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(selected.unique()), 2)
        assigned = result["matched_target_indices"][0, selected]
        self.assertEqual(set(assigned.tolist()), {0, 1})

    def test_exact_score_tie_is_broken_by_candidate_identity(self) -> None:
        ranked, hybrid, targets = example()
        ranked["ranking_logits"][0, :4] = 1.0

        first = build_score_matched_equivalent_targets(ranked, hybrid, targets)
        second = build_score_matched_equivalent_targets(ranked, hybrid, targets)

        torch.testing.assert_close(first["positive_mask"], second["positive_mask"])
        self.assertEqual(torch.where(first["positive_mask"][0])[0].tolist(), [0, 2])

    def test_invalid_fixed_identity_mapping_is_rejected(self) -> None:
        ranked, hybrid, targets = example()
        hybrid["matched_target_indices"][0, 2] = 0

        with self.assertRaisesRegex(ValueError, "one-to-one"):
            build_score_matched_equivalent_targets(ranked, hybrid, targets)


if __name__ == "__main__":
    unittest.main()
