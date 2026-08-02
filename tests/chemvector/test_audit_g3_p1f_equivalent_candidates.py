import unittest

import torch

from chemvector.evaluation.audits.atoms.audit_g3_p1f_equivalent_candidates import (
    sample_equivalence_statistics,
)


class G3P1FEquivalentCandidateTests(unittest.TestCase):
    def test_geometry_equivalent_selection_rescues_fixed_identity(self) -> None:
        candidate_ids = torch.tensor(
            [
                [
                    10,
                    11,
                    20,
                    21,
                    *[
                        value
                        for value in range(128)
                        if value not in {10, 11, 20, 21}
                    ],
                ]
            ]
        )
        anchors = torch.full((1, 128, 2), 0.9)
        anchors[0, :4] = torch.tensor(
            [[0.100, 0.100], [0.101, 0.100], [0.500, 0.500], [0.501, 0.500]]
        )
        logits = torch.full((1, 128), -10.0)
        logits[0, :4] = torch.tensor([0.0, 2.0, 0.0, 1.0])
        ranked = {
            "candidate_ids": candidate_ids,
            "candidate_anchors": anchors,
            "candidate_features": torch.zeros(1, 128, 384),
            "base_logits": logits.clone(),
            "ranking_logits": logits,
            "grid_height": torch.tensor([8]),
            "grid_width": torch.tensor([8]),
        }
        positive = torch.zeros(1, 128, dtype=torch.bool)
        positive[0, [0, 2]] = True
        matched = torch.full((1, 128), -1, dtype=torch.long)
        matched[0, [0, 2]] = torch.tensor([0, 1])
        hybrid = {
            "positive_mask": positive,
            "matched_target_indices": matched,
        }
        targets = torch.tensor([[0.100, 0.100], [0.500, 0.500]])

        result = sample_equivalence_statistics(ranked, hybrid, targets, 0)

        self.assertEqual(result["fixed_identity_exact_images"], 0)
        self.assertEqual(result["geometry_equivalent_exact_images"], 1)
        self.assertEqual(result["identity_artifact_rescue_images"], 1)
        self.assertEqual(result["fixed_negative_selected_near_gt_atoms"], 2)


if __name__ == "__main__":
    unittest.main()
